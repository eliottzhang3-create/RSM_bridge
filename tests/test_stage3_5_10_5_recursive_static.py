"""Dependency-light contracts for the isolated recursive 5-10-5 evaluator."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3_5_10_5_recursive.py"
RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3_5_10_5_recursive.sh"
SUBMIT = ROOT / "code" / "RSmol" / "run_stage3_eval_5_10_5_recursive_5090.sh"
LINEAR_EVALUATOR = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3_5_10_5.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("stage3_5_10_5_recursive_static", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage3FiveTenFiveRecursiveStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_evaluator()
        cls.source = EVALUATOR.read_text(encoding="utf-8")
        cls.runtime = RUNTIME.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def _valid_config(self, vocab_size=3):
        return {
            "model_type": "llama",
            "vocab_size": vocab_size,
            "architectures": ["RecursiveLlamaForCausalLM"],
            "num_hidden_layers": 30,
            "recursive_layer_count": 20,
            "recursive_loops": 2,
            "recursive_loops_scope": "middle_only",
            "recursive_prefix_layer_count": 5,
            "recursive_middle_layer_count": 10,
            "recursive_suffix_layer_count": 5,
            "recursive_source_layer_indices_0based": list(self.module.SOURCE_MAPPING_0BASED),
            "recursive_source_layer_indices_1based": list(self.module.SOURCE_MAPPING_1BASED),
            "logical_to_physical": list(self.module.LOGICAL_TO_PHYSICAL),
            "recursive_logical_to_physical": list(self.module.LOGICAL_TO_PHYSICAL),
            "logical_to_physical_schedule": list(self.module.LOGICAL_TO_PHYSICAL),
            "recursive_logical_to_physical_schedule": list(self.module.LOGICAL_TO_PHYSICAL),
        }

    def test_exact_recursive_contract(self):
        self.assertEqual(self.module.ARCHITECTURE_CONTRACT, "logical_30_physical_20_5_10_5_loops_2")
        self.assertEqual(self.module.MODEL_LABEL, "recursive_5_10_5_middle_loop2")
        self.assertEqual((self.module.LOGICAL_LAYER_COUNT, self.module.PHYSICAL_LAYER_COUNT), (30, 20))
        self.assertEqual((self.module.RECURSIVE_LOOPS, self.module.LOOPS_SCOPE), (2, "middle_only"))
        self.assertEqual(
            self.module.SOURCE_MAPPING_0BASED,
            (0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29),
        )
        self.assertEqual(
            self.module.LOGICAL_TO_PHYSICAL,
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
             5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
        )

    def test_artifact_contract_and_vocab_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "config.json").write_text(json.dumps(self._valid_config()), encoding="utf-8")
            (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.json").write_text(
                '{"model": {"vocab": {"a": 0, "b": 1, "c": 2}}}', encoding="utf-8"
            )
            (model / "model.safetensors").write_bytes(b"placeholder")
            result = self.module.inspect_model_artifacts_5_10_5_recursive(model)
            self.assertEqual(result["model_label"] if "model_label" in result else result["label"], self.module.MODEL_LABEL)
            self.assertTrue(all(result["recursive_audit"]["contract_checks"].values()))
            self.assertTrue(result["vocab_compatible"])

    def test_artifact_rejects_wrong_loop_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            config = self._valid_config()
            config["recursive_loops"] = 1
            (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.json").write_text(
                '{"model": {"vocab": {"a": 0, "b": 1, "c": 2}}}', encoding="utf-8"
            )
            (model / "model.safetensors").write_bytes(b"placeholder")
            with self.assertRaisesRegex(ValueError, "Invalid SmolLM2-5-10-5 checkpoint contract"):
                self.module.inspect_model_artifacts_5_10_5_recursive(model)

    def test_isolated_registration_and_official_protocol(self):
        self.assertIn('"code.RSmol.recursive_model_5_10_5"', self.source)
        self.assertNotIn("from code.RSmol.recursive_model import", self.source)
        self.assertNotIn("recursive_model_5_10_5_linear", self.source)
        for marker in (
            "num_fewshot=5 if task in {\"mmlu\", \"gsm8k\"} else None",
            "TaskManager(include_path=str(overlay_dir))",
            "local_files_only=True",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "lm_eval_results.json",
            "log_samples.json",
            "summary.json",
            "summary.csv",
            "audit_report.json",
            "ensure_external_output",
            "do_sample=False",
        ):
            self.assertIn(marker, self.source)

    def test_runtime_and_submission_isolated_namespace(self):
        for marker in (
            "RSMOL_STAGE3_5_10_5_RECURSIVE_MODEL",
            "RSMOL_STAGE3_5_10_5_RECURSIVE_BENCHMARK_ROOT",
            "RSMOL_STAGE3_5_10_5_RECURSIVE_OUTPUT_DIR",
            "RSMOL_STAGE3_5_10_5_RECURSIVE_DEVICE",
            "RSMOL_STAGE3_5_10_5_RECURSIVE_BATCH_SIZE",
            "RSMOL_STAGE3_5_10_5_RECURSIVE_SEED",
            "RSMOL_STAGE3_5_10_5_RECURSIVE_TASKS",
            "RSMOL_STAGE3_5_10_5_RECURSIVE_CACHE_ROOT",
            "RSMOL_STAGE3_5_10_5_RECURSIVE_LOG_ROOT",
            "RSMOL_STAGE3_5_10_5_RECURSIVE_SMOKE",
            "RSMOL_STAGE3_5_10_5_RECURSIVE_LIMIT",
            "RSMOL_STAGE3_5_10_5_RECURSIVE_NO_LOG_SAMPLES",
            "RSMOL_STAGE3_5_10_5_RECURSIVE_VALIDATION_ONLY",
            "evaluate_stage3_5_10_5_recursive.py",
        ):
            self.assertIn(marker, self.runtime + self.submit)
        self.assertIn("-p pdgpu-5090", self.submit)
        self.assertIn("-c 8 -m 32G -g 1 -n 1", self.submit)
        self.assertNotIn("evaluate_stage3.py", self.submit)
        self.assertNotIn("evaluate_stage3_5_10_5.py", self.submit)

    def test_linear_evaluator_is_not_modified_by_this_contract(self):
        self.assertTrue(LINEAR_EVALUATOR.is_file())
        self.assertNotIn("recursive_5_10_5_middle_loop2", LINEAR_EVALUATOR.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
