"""Dependency-light contracts for the isolated 5-10-5 Stage 3 evaluator."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3_5_10_5.py"
RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3_5_10_5.sh"
SUBMIT = ROOT / "code" / "RSmol" / "run_stage3_eval_5_10_5_5090.sh"
ORIGINAL = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("stage3_5_10_5_static", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage3FiveTenFiveStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_evaluator()
        cls.source = EVALUATOR.read_text(encoding="utf-8")
        cls.runtime = RUNTIME.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")
        cls.original = ORIGINAL.read_text(encoding="utf-8")

    def test_exact_architecture_contract(self):
        self.assertEqual(self.module.ARCHITECTURE_CONTRACT, "logical_30_physical_20_5_10_5_loops_2")
        self.assertEqual(self.module.LOGICAL_LAYER_COUNT, 30)
        self.assertEqual(self.module.PHYSICAL_LAYER_COUNT, 20)
        self.assertEqual(self.module.RECURSIVE_LOOPS, 2)
        self.assertEqual(self.module.LOOPS_SCOPE, "middle_only")
        self.assertEqual(self.module.PREFIX_LAYER_COUNT, 5)
        self.assertEqual(self.module.MIDDLE_LAYER_COUNT, 10)
        self.assertEqual(self.module.SUFFIX_LAYER_COUNT, 5)
        self.assertEqual(
            self.module.SOURCE_MAPPING_0BASED,
            (0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29),
        )
        self.assertEqual(
            self.module.LOGICAL_TO_PHYSICAL,
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
             5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
        )

    def test_artifact_pre_audit_accepts_only_5_10_5(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            config = {
                "model_type": "llama",
                "vocab_size": 3,
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
            (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.json").write_text(
                '{"model": {"vocab": {"a": 0, "b": 1, "c": 2}}}', encoding="utf-8"
            )
            (model / "model.safetensors").write_bytes(b"placeholder")
            info = self.module.inspect_model_artifacts_5_10_5(model)
            self.assertEqual(info["label"], "recursive_5_10_5")
            self.assertEqual(info["architecture_contract"], self.module.ARCHITECTURE_CONTRACT)
            self.assertEqual(info["config_vocab_size"], 3)
            self.assertEqual(info["tokenizer_vocab_size"], 3)
            self.assertTrue(info["vocab_compatible"])
            self.assertTrue(all(info["recursive_audit"]["contract_checks"].values()))

    def test_artifact_pre_audit_rejects_vocab_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            config = {
                "model_type": "llama",
                "vocab_size": 4,
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
            (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.json").write_text(
                '{"model": {"vocab": {"a": 0, "b": 1, "c": 2}}}', encoding="utf-8"
            )
            (model / "model.safetensors").write_bytes(b"placeholder")
            with self.assertRaisesRegex(ValueError, "vocab mismatch"):
                self.module.inspect_model_artifacts_5_10_5(model)

    def test_entrypoint_isolated_and_protocol_is_official(self):
        self.assertIn('"code.RSmol.recursive_model_5_10_5"', self.source)
        self.assertNotIn("from code.RSmol.recursive_model import", self.source)
        self.assertNotIn("reference_model_path", self.source.split("def parse_args", 1)[1])
        for marker in (
            "STAGE3_TASKS", "hellaswag", "mmlu", "gsm8k", "arc_easy", "arc_challenge",
            "num_fewshot=5 if task == \"mmlu\" else None",
            "TaskManager(include_path=str(overlay_dir))", "local_files_only=True",
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
            "lm_eval_results.json", "log_samples.json", "summary.json", "summary.csv",
            "audit_report.json", "run_config.json", "model_label", "architecture_contract",
            "ensure_external_output", "no device_map", "do_sample",
        ):
            self.assertIn(marker, self.source)

    def test_default_paths_and_remote_wrapper(self):
        self.assertEqual(
            self.module.DEFAULT_MODEL.as_posix(),
            "/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5",
        )
        self.assertEqual(
            self.module.DEFAULT_BENCHMARK_ROOT.as_posix(),
            "/hpc_stor03/sjtu_home/jinwei.zhang/data/eval_datasets",
        )
        for marker in (
            "RSMOL_STAGE3_5_10_5_MODEL", "RSMOL_STAGE3_5_10_5_BENCHMARK_ROOT",
            "RSMOL_STAGE3_5_10_5_OUTPUT_DIR", "RSMOL_STAGE3_5_10_5_TASKS",
            "RSMOL_STAGE3_5_10_5_LOG_ROOT", "bash scripts/evaluate_stage3_5_10_5.sh",
            "external", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
        ):
            self.assertIn(marker, self.runtime + self.submit)
        self.assertIn("-p pdgpu-5090", self.submit)
        self.assertIn("-c 8 -m 32G -g 1 -n 1", self.submit)
        self.assertNotIn("evaluate_stage3.py", self.submit)

    def test_original_entrypoint_text_is_unchanged_by_new_files(self):
        self.assertNotIn("5-10-5", self.original)


if __name__ == "__main__":
    unittest.main()
