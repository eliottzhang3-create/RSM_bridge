from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"
SCRIPTS = RSMOL / "scripts"
COMMON = SCRIPTS / "evaluate_qwen2_audio_instruct_common.py"
MMAU = SCRIPTS / "evaluate_mmau_test_mini_qwen2_audio_instruct.py"
MMAR = SCRIPTS / "evaluate_mmar_qwen2_audio_instruct.py"


def _load_common():
    spec = importlib.util.spec_from_file_location("qwen2_audio_eval_common_test", COMMON)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Qwen2-Audio common evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    eos_token_id = 99
    pad_token_id = None

    def __call__(self, text, *, truncation, max_length=None, **_kwargs):
        ids = list(range(1, len(str(text).split()) + 1))
        if truncation and max_length is not None:
            ids = ids[: int(max_length)]
        return {"input_ids": ids}

    def decode(self, ids, **_kwargs):
        return " ".join(f"token{int(value)}" for value in ids)


class _FakeProcessor:
    tokenizer = _FakeTokenizer()


class Qwen2AudioEvalStaticTests(unittest.TestCase):
    def test_common_module_is_dependency_light_and_prompt_contract_is_exact(self) -> None:
        module = _load_common()
        conversation = module.build_audio_analysis_conversation("question a) one b) two")
        self.assertEqual(len(conversation), 1)
        self.assertEqual(conversation[0]["role"], "user")
        self.assertEqual([part["type"] for part in conversation[0]["content"]], ["audio", "text"])
        self.assertNotIn("system", json.dumps(conversation).lower())

        effective, audit = module.prepare_prompt_text(
            _FakeProcessor(),
            "one two three four",
            max_prompt_tokens=3,
            truncate=True,
        )
        self.assertEqual(effective, "token1 token2 token3")
        self.assertEqual(audit["prompt_original_token_count"], 4)
        self.assertEqual(audit["prompt_token_count"], 3)
        self.assertTrue(audit["prompt_truncated"])

    def test_artifact_audit_requires_the_exact_five_shard_inventory(self) -> None:
        module = _load_common()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {
                "config.json": {
                    "model_type": "qwen2_audio",
                    "architectures": ["Qwen2AudioForConditionalGeneration"],
                },
                "generation_config.json": {},
                "preprocessor_config.json": {},
                "tokenizer.json": {},
                "tokenizer_config.json": {},
                "vocab.json": {},
            }
            for name, payload in payloads.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            (root / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
            shards = [f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)]
            index = {"weight_map": {f"tensor.{i}": name for i, name in enumerate(shards)}}
            (root / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            for name in shards:
                (root / name).write_bytes(b"x")

            report = module.audit_model_artifact(root)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(len(report["shards"]), 5)
            self.assertEqual(report["total_shard_bytes"], 5)
            self.assertTrue(report["local_files_only"])
            self.assertFalse(report["trust_remote_code"])

    def test_model_runtime_is_local_single_gpu_bf16(self) -> None:
        source = COMMON.read_text(encoding="utf-8")
        for marker in (
            "Qwen2AudioForConditionalGeneration",
            "AutoProcessor.from_pretrained",
            "local_files_only=True",
            "trust_remote_code=False",
            "torch_dtype=torch.bfloat16",
            'device_map={"": 0}',
            'parameter_devices != {str(device)}',
            'floating_dtypes != {str(torch.bfloat16)}',
            "audios=[audio]",
            "add_generation_prompt=True",
            "use_cache=False",
            "pad_token_id = sorted(eos_ids)[0]",
        ):
            self.assertIn(marker, source)

    def test_mmau_reuses_author_protocol_and_shares_raw_prediction(self) -> None:
        source = MMAU.read_text(encoding="utf-8")
        for marker in (
            "official.build_mellow_author_reply_prompt",
            "official.decode_mellow_author_reply_audio",
            "official.mellow_author_reply_audio_segment",
            "prefer_official_audio_file=True",
            "qwen.generate_top_p_argmax",
            "top_p=0.8",
            "max_new_tokens=max_new_tokens",
            "return str(value)",
            '"prediction_text_shared_without_preparse": True',
            "official.write_mellow_author_reply_evaluation",
        ):
            self.assertIn(marker, source)
        self.assertNotIn("strip_choice", source)

    def test_mmar_reuses_official_protocol_and_shares_raw_prediction(self) -> None:
        source = MMAR.read_text(encoding="utf-8")
        for marker in (
            "qwen.generate_greedy",
            "max_prompt_tokens=MAX_PROMPT_TOKENS",
            "max_new_tokens=max_new_tokens",
            "return str(value)",
            'args.dtype != "bf16"',
            "official.write_choice_label_prefix_evaluation",
            '"prediction_text_shared_without_preparse": True',
            'get("prediction_key", "answer_prediction")',
        ):
            self.assertIn(marker, source)
        self.assertNotIn("strip_choice", source)

    def test_existing_common_runners_keep_their_old_default_record_fields(self) -> None:
        mmau_common = (SCRIPTS / "evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.py").read_text(
            encoding="utf-8"
        )
        mmar_common = (SCRIPTS / "evaluate_mmar_5_10x2_5_mesh_mellow.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('{"audio2_reused": True}', mmau_common)
        self.assertIn('{"audio2_reused": True, "single_audio_slot": True}', mmar_common)
        self.assertIn("**record_static_fields", mmau_common)
        self.assertIn("**record_static_fields", mmar_common)

    def test_remote_entrypoints_use_one_5090_and_shared_smoke_full_output(self) -> None:
        wrappers = {
            "mmau": RSMOL / "run_mmau_test_mini_qwen2_audio_instruct_5090.sh",
            "mmar": RSMOL / "run_mmar_qwen2_audio_instruct_5090.sh",
        }
        for name, path in wrappers.items():
            with self.subTest(name=name):
                source = path.read_text(encoding="utf-8")
                self.assertIn("vc submit", source)
                self.assertIn("-p pdgpu-5090", source)
                self.assertIn("-g 1 -n 1", source)
                self.assertIn("--dtype bf16", source)
                self.assertIn("--run-official-evaluation", source)
                self.assertIn("OUTPUT_DIR", source)
                self.assertIn("MODE", source)

        preflight = (RSMOL / "run_qwen2_audio_instruct_eval_preflight.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("audit_qwen2_audio_instruct_artifact.py", preflight)
        self.assertIn("RSMOL_QWEN2_AUDIO_MODEL_PATH", preflight)
        audit_source = (SCRIPTS / "audit_qwen2_audio_instruct_artifact.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("load_processor=True", audit_source)
        common_source = COMMON.read_text(encoding="utf-8")
        self.assertIn("Qwen2AudioForConditionalGeneration.__name__", common_source)
        self.assertIn('report["transformers_version"]', common_source)


if __name__ == "__main__":
    unittest.main()
