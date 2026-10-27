"""Dependency-light checks for the isolated CED-Small benchmark evaluation."""
from __future__ import annotations

import ast
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code/RSmol"
SCRIPTS = RSMOL / "scripts"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CedBenchmarkEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_module("ced_benchmark_runner", SCRIPTS / "evaluate_mizar_s1_ced_benchmark.py")
        cls.runtime = cls.runner.runtime_adapter
        cls.route = load_module("ced_eval_contract", cls.runtime.ROUTE_ROOT / "contract.py")
        cls.original = load_module("htsat_eval_protocol", SCRIPTS / "mizar_s1_eval_runtime.py")

    def fixture(self):
        contract = self.route.semantic_contract({
            "rows": 1477492, "source_rows": self.route.SOURCE_COUNTS,
            "rows_canonical_sha256": "manifest", "mapping_sha256": "mapping",
        }, "text-model", "ced-small")
        saved = {
            "schema_version": 1, "route_variant": self.route.ROUTE, "mode": "formal",
            "state_dict": {"test_parameter": [1]}, "contract": contract,
            "contract_sha256": self.route.digest(contract), "total_step": 17313,
            "num_epochs": 3, "loss_reduction": self.runtime.TRAINING_LOSS_REDUCTION,
        }
        runtime = {"contract": contract, "contract_sha256": saved["contract_sha256"],
                   "route": self.route.ROUTE, "mode": "formal"}
        return contract, saved, runtime

    def resign(self, saved, runtime):
        saved["contract_sha256"] = self.route.digest(saved["contract"])
        runtime["contract_sha256"] = saved["contract_sha256"]

    def test_generation_settings_match_reference_except_architecture(self):
        original, ced = dict(self.original.PROTOCOL), dict(self.runtime.PROTOCOL)
        differences = {key for key in original.keys() | ced.keys()
                       if original.get(key) != ced.get(key)}
        self.assertEqual(differences, {"version", "prefix_layout", "prefix_tokens", "audio_tokens",
                                       "answer_start_position", "first_answer_logit_position"})
        self.assertEqual(ced["prefix_layout"], [126, 1, 256])
        self.assertEqual(ced["prefix_tokens"], 383)
        self.assertEqual(ced["answer_start_position"], 383)
        self.assertEqual(ced["first_answer_logit_position"], 382)
        self.assertEqual(ced["max_new_tokens"], 300)

    def test_dataset_scorer_resume_and_audio_functions_are_exact_copies(self):
        def functions(path):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

        original = functions(SCRIPTS / "evaluate_mizar_s1_benchmark.py")
        ced = functions(SCRIPTS / "evaluate_mizar_s1_ced_benchmark.py")
        self.assertEqual(set(original), set(ced))
        for name in original.keys() - {"parse_args", "generate"}:
            with self.subTest(function=name):
                self.assertEqual(original[name], ced[name])

    def test_defaults_point_to_completed_ced_training_run(self):
        for benchmark, folder in (("mmau", "mmau_test_mini_step17313"), ("mmar", "mmar_step17313")):
            args = self.runner.parse_args(["--benchmark", benchmark])
            self.assertEqual(args.expected_step, 17313)
            self.assertEqual(args.expected_geometry, (8, 8, 4))
            self.assertEqual(args.checkpoint, self.runtime.DEFAULT_CHECKPOINT.resolve())
            self.assertEqual(args.runtime_config, self.runtime.DEFAULT_RUNTIME.resolve())
            self.assertEqual(args.route_root, self.runtime.ROUTE_ROOT.resolve())
            self.assertEqual(args.output_dir, (self.runtime.DEFAULT_RUN.parent / "eval" / folder).resolve())
        mmau = self.runner.parse_args(["--benchmark", "mmau"])
        mmar = self.runner.parse_args(["--benchmark", "mmar"])
        self.assertEqual(mmau.metadata_json.name, "mmau-test-mini.json")
        self.assertEqual(mmau.parquet.name, "test_mini.parquet")
        self.assertEqual(mmau.parquet_batch_size, 8)
        self.assertEqual(mmar.metadata_json.name, "MMAR-meta.json")
        self.assertEqual(mmar.audio_root.name, "mmar-audio")

    def test_checkpoint_validates_and_rejects_wrong_route_step_and_geometry(self):
        contract, saved, runtime = self.fixture()
        self.assertIs(self.runtime.validate_checkpoint(saved, runtime, self.route, 17313), contract)
        with self.assertRaisesRegex(ValueError, "expected training geometry"):
            self.runtime.validate_checkpoint(saved, runtime, self.route, 17313, (8, 4, 1))
        with self.assertRaisesRegex(ValueError, "step / epoch horizon"):
            self.runtime.validate_checkpoint(saved, runtime, self.route, 17314)
        saved["route_variant"] = "htsat_mizar_s1_fixed20_gbs256_v1"
        with self.assertRaisesRegex(ValueError, "complete formal CED"):
            self.runtime.validate_checkpoint(saved, runtime, self.route, 17313)

    def test_source_runtime_and_loss_identity_remain_strict(self):
        _, saved, runtime = self.fixture()
        with patch.object(self.route, "code_digest", return_value="another-source"):
            with self.assertRaisesRegex(ValueError, "training source differs"):
                self.runtime.validate_checkpoint(saved, runtime, self.route, 17313)
        with self.assertRaisesRegex(ValueError, "runtime.json does not belong"):
            self.runtime.validate_checkpoint(saved, {**runtime, "contract_sha256": "other"}, self.route, 17313)
        saved["loss_reduction"] = "another-reduction"
        with self.assertRaisesRegex(ValueError, "training loss contract mismatch"):
            self.runtime.validate_checkpoint(saved, runtime, self.route, 17313)

    def test_training_audio_and_mapper_contracts_are_enforced(self):
        for section, key, value, message in (
            (None, "prefix", [129, 129, 1, 256], "training contract mismatch"),
            (None, "attention_mask", "padding_mask", "training contract mismatch"),
            ("audio_encoder", "name", "HTSAT", "CEDSmall audio encoder"),
            ("audio_encoder", "hidden_cache", True, "online hidden-feature"),
            ("mapper", "resample", True, "mapper contract"),
            ("mapper", "name", "mlp", "mapper contract"),
            ("mapper", "output_dim", 384, "mapper contract"),
        ):
            with self.subTest(section=section, key=key):
                contract, saved, runtime = self.fixture()
                (contract if section is None else contract[section])[key] = value
                self.resign(saved, runtime)
                with self.assertRaisesRegex(ValueError, message):
                    self.runtime.validate_checkpoint(saved, runtime, self.route, 17313)
        contract, saved, runtime = self.fixture()
        del contract["attention_mask"]
        self.resign(saved, runtime)
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            self.runtime.validate_checkpoint(saved, runtime, self.route, 17313)

    def test_asset_audit_checks_all_six_local_ced_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for name in self.runtime.CED_ASSET_FILES:
                path = root / name
                path.write_bytes(name.encode())
                entries.append({"name": name, "bytes": path.stat().st_size,
                                "sha256": self.runtime.sha256_file(path)})
            contract = {"text_model_dir": str(root), "ced_model_dir": str(root),
                        "assets": {"text": [entries[0]], "ced": entries}}
            audit = self.runtime.audit_assets(contract)
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["verified_files"], 7)
            (root / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "component changed"):
                self.runtime.audit_assets(contract)

    def test_generation_preserves_two_halves_and_uses_training_tokenizer(self):
        class Tensor:
            shape = (1, 383, 576)

            def numel(self):
                return 320000

            def unsqueeze(self, _):
                return self

            def to(self, _):
                return self

            def eq(self, _):
                return self

            def sum(self):
                return 11

        first, second, prefix = Tensor(), Tensor(), Tensor()
        model = Mock()
        model.generate_prefix_inference.return_value = (prefix, {}, None)
        tokenizer = Mock(pad_token_id=17)
        tokenizer.encode.return_value = [1, 2]
        text_encoder = Mock()
        text_encoder.tokenize.return_value = {"input_ids": Tensor(), "attention_mask": Tensor()}
        fake_torch = types.SimpleNamespace(inference_mode=nullcontext)
        decoded = {"generated_text": " raw <|endoftext|> ", "generated_token_count": 2}
        with (patch.dict(sys.modules, {"torch": fake_torch}),
              patch.object(self.runner, "_fixed20_pair", return_value=(first, second)),
              patch.object(self.runtime, "greedy_decode", return_value=decoded) as decode):
            result = self.runner.generate((model, tokenizer, "cuda:0", text_encoder,
                                           None, None, None), {"prompt": "A Question?"})
        text_encoder.tokenize.assert_called_once_with("a question?", 256)
        batch = model.generate_prefix_inference.call_args.args[0]
        self.assertIs(batch["audio1"], first)
        self.assertIs(batch["audio2"], second)
        decode.assert_called_once_with(model, tokenizer, prefix)
        self.assertEqual(result["generated_text"], " raw <|endoftext|> ")
        self.assertEqual(result["prefix_shape"], [1, 383, 576])
        self.assertEqual(result["answer_start_position"], 383)
        self.assertEqual(result["first_answer_logit_position"], 382)
        self.assertFalse(result["audio2_reused"])

    def decode_fixture(self, ids, finite=True):
        class Tensor:
            def __init__(self, length):
                self.shape, self.device = (1, length, 576), "cuda:0"

        class Logits:
            def __init__(self, token):
                self.token = token

            def __getitem__(self, _):
                return self

        embedding = Mock(return_value=Tensor(1))
        lm = Mock(side_effect=[types.SimpleNamespace(logits=Logits(token)) for token in ids])
        lm.get_input_embeddings.return_value = embedding
        model = types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=lm))
        tokenizer = Mock(eos_token_id=0)
        tokenizer.decode.return_value = "  raw <|endoftext|>  "
        fake_torch = types.SimpleNamespace(
            isfinite=lambda _: types.SimpleNamespace(all=lambda: finite),
            argmax=lambda logits, dim: types.SimpleNamespace(item=lambda: logits.token),
            tensor=lambda *args, **kwargs: object(), long="long",
            cat=lambda tensors, dim: Tensor(sum(t.shape[1] for t in tensors)))
        return model, tokenizer, Tensor(383), fake_torch

    def test_greedy_stops_on_first_eos_and_recomputes_without_cache(self):
        model, tokenizer, prefix, torch = self.decode_fixture([42, 0, 99])
        with patch.dict(sys.modules, {"torch": torch}):
            result = self.runtime.greedy_decode(model, tokenizer, prefix)
        self.assertEqual(result["generated_token_ids"], [42, 0])
        self.assertEqual(result["stop_reason"], "eos_token")
        self.assertEqual(result["generated_text"], "  raw <|endoftext|>  ")
        tokenizer.decode.assert_called_once_with([42, 0], skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)
        self.assertEqual([call.kwargs["inputs_embeds"].shape[1] for call in
                          model.caption_decoder.lm.call_args_list], [383, 384])
        for call in model.caption_decoder.lm.call_args_list:
            self.assertIsNone(call.kwargs["attention_mask"])
            self.assertIsNone(call.kwargs["labels"])
            self.assertFalse(call.kwargs["use_cache"])

    def test_greedy_enforces_300_horizon_prefix_and_finite_logits(self):
        model, tokenizer, prefix, torch = self.decode_fixture([42] * 300)
        with patch.dict(sys.modules, {"torch": torch}):
            result = self.runtime.greedy_decode(model, tokenizer, prefix)
            self.assertEqual(result["generated_token_count"], 300)
            self.assertEqual(result["stop_reason"], "max_new_tokens")
            with self.assertRaisesRegex(ValueError, "exactly max_new_tokens=300"):
                self.runtime.greedy_decode(model, tokenizer, prefix, max_new_tokens=32)
            prefix.shape = (1, 515, 576)
            with self.assertRaisesRegex(ValueError, "prefix shape"):
                self.runtime.greedy_decode(model, tokenizer, prefix)
        model, tokenizer, prefix, torch = self.decode_fixture([42], finite=False)
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                self.runtime.greedy_decode(model, tokenizer, prefix)

    def test_materialized_predictions_keep_raw_eos_and_official_order(self):
        records = [{"id": "a", "answer": "one"}, {"id": "b", "answer": "two"}]
        text = "  B) two <|endoftext|>  "
        state = [{"id": "b", "status": "generated", "generated_text": text}]
        output = self.runner._materialize(state, records, "answer_prediction", "mmar")
        self.assertEqual([row["id"] for row in output], ["a", "b"])
        self.assertEqual(output[1]["answer_prediction"], text)
        self.assertEqual(records, [{"id": "a", "answer": "one"}, {"id": "b", "answer": "two"}])

    def test_submission_defaults_and_overrides_use_isolated_runner(self):
        bash = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
        if not Path(bash).is_file():
            self.skipTest("Bash is unavailable")

        def bash_path(path):
            value = Path(path).resolve().as_posix()
            return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "bin"
            binary.mkdir()
            stub = binary / "vc"
            stub.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$RSM_TEST_CAPTURE"\n',
                            encoding="utf-8", newline="\n")
            stub.chmod(0o755)
            for benchmark in ("mmau", "mmar"):
                submit = root / f"run_mizar_s1_ced_small_{benchmark}_eval_5090.sh"
                submit.write_bytes((RSMOL / submit.name).read_bytes().replace(b"\r\n", b"\n"))
                for override in (False, True):
                    with self.subTest(benchmark=benchmark, override=override):
                        capture = root / "args"
                        env = {k: v for k, v in os.environ.items() if not k.startswith("RSMOL_")}
                        env.update(RSM_TEST_CAPTURE=bash_path(capture), PATH=f"{bash_path(binary)}:/usr/bin:/bin")
                        if override:
                            env["RSMOL_MIZAR_CED_RUN_ROOT"] = "/another run/ced"
                        subprocess.run([bash, bash_path(submit)], env=env,
                                       capture_output=True, text=True, check=True)
                        submission = capture.read_text(encoding="utf-8").splitlines()
                        for flag, value in (("-p", "pdgpu-5090"), ("-c", "8"), ("-m", "32G"),
                                            ("-g", "1"), ("-n", "1")):
                            self.assertEqual(submission[submission.index(flag) + 1], value)
                        self.assertLessEqual(len(submission[submission.index("-j") + 1]), 60)
                        command = shlex.split(submission[submission.index("--cmd") + 1])
                        self.assertEqual(command[:2], ["bash", "scripts/evaluate_mizar_s1_ced_benchmark.sh"])
                        options = dict(zip(command[2::2], command[3::2]))
                        run = "/another run/ced" if override else self.runtime.DEFAULT_RUN.parent.as_posix()
                        self.assertEqual(options["--checkpoint"], run + "/formal/checkpoints/model--step-17313.ckpt")
                        self.assertEqual(options["--runtime-config"], run + "/formal/runtime.json")
                        self.assertTrue(options["--route-root"].endswith(self.runtime.ROUTE_ROOT.name))
                        folder = "mmau_test_mini_step17313" if benchmark == "mmau" else "mmar_step17313"
                        self.assertEqual(options["--output-dir"], run + "/eval/" + folder)
                        self.assertEqual(options["--expected-step"], "17313")
                        self.assertEqual(options["--expected-geometry"], "8,8,4")
                        self.assertEqual(options["--benchmark"], benchmark)
                        if benchmark == "mmau":
                            self.assertEqual(options["--parquet-batch-size"], "8")


if __name__ == "__main__":
    unittest.main()
