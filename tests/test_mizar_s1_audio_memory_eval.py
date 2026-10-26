"""CPU-only checks for the isolated audio-memory benchmark route."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"
EVAL = RSMOL / "mizar_s1_audio_memory_eval"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AudioMemoryEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_module("isolated_audio_memory_runner", EVAL / "evaluate_mizar_s1_benchmark.py")
        cls.runtime = cls.runner.runtime_adapter
        cls.route = load_module("isolated_audio_memory_contract", cls.runtime.ROUTE_ROOT / "contract.py")
        cls.original = load_module("original_mizar_runtime_for_comparison", RSMOL / "scripts/mizar_s1_eval_runtime.py")

    def checkpoint_fixture(self):
        contract = self.route.semantic_contract({
            "rows": 1477492, "source_rows": self.route.SOURCE_COUNTS,
            "rows_canonical_sha256": "manifest", "mapping_sha256": "mapping",
        }, "text-model", "htsat")
        digest = self.route.digest(contract)
        saved = {
            "schema_version": 1, "route_variant": self.route.ROUTE, "mode": "formal",
            "state_dict": {"test_parameter": [1]}, "contract": contract,
            "contract_sha256": digest, "total_step": 5771, "num_epochs": 1,
            "loss_reduction": self.runtime.TRAINING_LOSS_REDUCTIONS[self.route.ROUTE],
        }
        run = {"contract": contract, "contract_sha256": digest,
               "route": self.route.ROUTE, "mode": "formal"}
        return contract, saved, run

    def test_protocol_and_default_checkpoint(self):
        self.assertEqual(self.runtime.PROTOCOL, self.original.PROTOCOL)
        self.assertEqual(self.runtime.AUDIO_MEMORY_CONTRACT,
                         self.checkpoint_fixture()[0]["audio_memory"])
        for benchmark, folder in (("mmau", "mmau_test_mini_step5771"),
                                  ("mmar", "mmar_step5771")):
            args = self.runner.parse_args(["--benchmark", benchmark])
            self.assertEqual(args.expected_step, 5771)
            self.assertEqual(args.expected_geometry, (8, 8, 4))
            self.assertEqual(args.output_dir,
                             (self.runtime.DEFAULT_RUN.parent / "eval" / folder).resolve())
            self.assertEqual(args.checkpoint, self.runtime.DEFAULT_CHECKPOINT.resolve())

    def test_checkpoint_requires_audio_memory_architecture(self):
        contract, saved, run = self.checkpoint_fixture()
        self.assertIs(self.runtime.validate_checkpoint(saved, run, self.route, 5771), contract)
        with self.assertRaisesRegex(ValueError, "step / epoch horizon"):
            self.runtime.validate_checkpoint(saved, run, self.route, 17313)
        contract["audio_memory"]["cross_attention"]["rounds"] = 1
        saved["contract_sha256"] = self.route.digest(contract)
        run["contract_sha256"] = saved["contract_sha256"]
        with self.assertRaisesRegex(ValueError, "audio-memory architecture"):
            self.runtime.validate_checkpoint(saved, run, self.route, 5771)

    def test_generation_passes_both_stage2_halves(self):
        class FakeTensor:
            def __init__(self, shape=(), count=0):
                self.shape, self.count = shape, count

            def numel(self):
                return self.count

            def unsqueeze(self, _):
                return self

            def to(self, _):
                return self

            def eq(self, _):
                return self

            def sum(self):
                return 0

        first, second = FakeTensor(count=320000), FakeTensor(count=320000)
        stage2_first, stage2_second = object(), object()
        prefix = FakeTensor(shape=(1, 515, 576))
        model = Mock()
        model.generate_prefix_inference.return_value = (
            prefix, {"stage2": stage2_first}, {"stage2": stage2_second})
        tokenizer = Mock(pad_token_id=17)
        tokenizer.encode.return_value = [1, 2]
        text_encoder = Mock()
        text_encoder.tokenize.return_value = {"input_ids": FakeTensor(),
                                             "attention_mask": FakeTensor()}
        fake_torch = types.ModuleType("torch")
        fake_torch.inference_mode = nullcontext
        decoded = {"generated_text": "answer", "generated_token_count": 1,
                   "audio_memory_audit": {"status": "PASS", "forward_count": 1}}
        with (patch.dict(sys.modules, {"torch": fake_torch}),
              patch.object(self.runner, "_fixed20_pair", return_value=(first, second)),
              patch.object(self.runtime, "greedy_decode", return_value=decoded) as decode):
            result = self.runner.generate((model, tokenizer, "cuda:0", text_encoder,
                                           None, None, None), {"prompt": "Question"})
        self.assertIs(decode.call_args.kwargs["audio_stage2_1"], stage2_first)
        self.assertIs(decode.call_args.kwargs["audio_stage2_2"], stage2_second)
        self.assertFalse(result["audio2_reused"])

    def test_forward_audit_requires_one_shared_audio_kv(self):
        class Tensor:
            def __init__(self, shape):
                self.shape = shape

        mesh = types.SimpleNamespace(audio_injection_enabled=True)
        model = types.SimpleNamespace(caption_decoder=types.SimpleNamespace(
            lm=types.SimpleNamespace(model=mesh)))
        first, second = Tensor((1, 1024, 192)), Tensor((1, 1024, 192))
        audit = self.runtime.AudioForwardAudit(model, first, second)
        audit.forward_count = 1  # The first-forward finite/norm probe needs real torch.
        query = Tensor((1, 515, 576))
        kv = (Tensor((1, 9, 2048, 64)), Tensor((1, 9, 2048, 64)))
        audit.begin_forward(515)
        audit._memory(None, (first, second), Tensor((1, 2048, 576)))
        audit._cross(None, (query, kv, 0), query)
        audit._perceive(0, (query,), Tensor((1, 515, 5)))
        audit._cross(None, (query, kv, 1), query)
        audit._perceive(1, (query,), Tensor((1, 515, 5)))
        audit.finish_forward()
        self.assertEqual(audit.report()["event_counts"],
                         {"memory": 1, "cross_0": 1, "perceive_0": 1,
                          "cross_1": 1, "perceive_1": 1})

        audit.begin_forward(515)
        audit._cross(None, (query, kv, 0), query)
        other_kv = (Tensor((1, 9, 2048, 64)), Tensor((1, 9, 2048, 64)))
        with self.assertRaisesRegex(ValueError, "did not share one audio KV"):
            audit._cross(None, (query, other_kv, 1), query)

    def test_wrappers_use_isolated_runner_and_same_gpu_request(self):
        for benchmark in ("mmau", "mmar"):
            wrapper = (RSMOL / f"run_mizar_s1_audio_memory_{benchmark}_eval_5090.sh").read_text(encoding="utf-8")
            self.assertIn(f"--benchmark {benchmark}", wrapper)
            self.assertIn("-p pdgpu-5090", wrapper)
            self.assertIn("-c 8 -m 32G -g 1 -n 1", wrapper)
            self.assertIn("mizar_s1_audio_memory_eval/evaluate_mizar_s1_benchmark.sh", wrapper)
            self.assertIn("model--step-${EXPECTED_STEP}.ckpt", wrapper)
        inner = (EVAL / "evaluate_mizar_s1_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn("mizar_s1_audio_memory_eval/evaluate_mizar_s1_benchmark.py", inner)
        for helper in ("audio_memory_mmau_reference.py", "audio_memory_mmar_reference.py"):
            self.assertIn('raise SystemExit("Use evaluate_mizar_s1_benchmark.py',
                          (EVAL / helper).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
