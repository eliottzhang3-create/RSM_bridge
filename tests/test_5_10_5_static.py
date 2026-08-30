"""Dependency-free contracts for the SmolLM2 5-10-5 pipeline."""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "code" / "RSmol" / "recursive_model_5_10_5.py"
CONVERTER = ROOT / "code" / "RSmol" / "scripts" / "convert_stepwise_5_10_5.py"
SMOKE = ROOT / "code" / "RSmol" / "scripts" / "smoke_recursive_5_10_5.py"
STAGE4 = ROOT / "code" / "RSmol" / "scripts" / "train_stage4_5_10_5_ddp.py"
RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "train_stage4_5_10_5_ddp.sh"
SUBMIT = ROOT / "code" / "RSmol" / "run_stage4_5_10_5_5090.sh"
CONVERT_RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "convert_stepwise_5_10_5.sh"
CONVERT_SUBMIT = ROOT / "code" / "RSmol" / "run_convert_stepwise_5_10_5_5090.sh"


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_5_10_5_static", CONVERTER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FiveTenFiveStaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_text = MODEL.read_text(encoding="utf-8")
        cls.converter_text = CONVERTER.read_text(encoding="utf-8")
        cls.smoke_text = SMOKE.read_text(encoding="utf-8")
        cls.stage4_text = STAGE4.read_text(encoding="utf-8")
        cls.runtime_text = RUNTIME.read_text(encoding="utf-8")
        cls.submit_text = SUBMIT.read_text(encoding="utf-8")
        cls.convert_runtime_text = CONVERT_RUNTIME.read_text(encoding="utf-8")
        cls.convert_submit_text = CONVERT_SUBMIT.read_text(encoding="utf-8")
        cls.converter = load_converter()

    def test_stage4_parser_rejects_unsupported_gates(self):
        spec = importlib.util.spec_from_file_location("stage4_5_10_5_static", STAGE4)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for gate in ("B", "C"):
            with self.assertRaisesRegex(ValueError, "unsupported"):
                module._parse_args(["--gate", gate, "--dry-run"])
        self.assertEqual(module._parse_args(["--gate", "D", "--dry-run"]).max_optimizer_steps, 10)
        self.assertEqual(module._parse_args(["--gate", "FORMAL", "--dry-run"]).max_optimizer_steps, 9244)

    def test_mapping_and_schedule_are_exact(self):
        self.assertEqual(
            self.converter.SOURCE_LAYER_INDICES_0BASED,
            (0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29),
        )
        self.assertEqual(
            self.converter.LOGICAL_TO_PHYSICAL,
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
             5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
        )
        self.assertEqual(len(self.converter.LOGICAL_TO_PHYSICAL), 30)
        with self.assertRaises(ValueError):
            self.converter.build_target_config(type("Config", (), {"num_hidden_layers": 29})())

    def test_model_contains_explicit_schedule_and_logical_slots(self):
        for marker in (
            "LOGICAL_LAYER_COUNT = 30", "PHYSICAL_LAYER_COUNT = 20",
            "MIDDLE_LAYER_COUNT = 10", "LOGICAL_TO_PHYSICAL", "LogicalSlotCacheView",
            "logical_slot=logical_index", "for logical_index, physical_index in enumerate(self.logical_to_physical)",
            "return DynamicCache()", "cache_implementation='dynamic'",
        ):
            self.assertIn(marker, self.model_text)
        self.assertIn("if schedule != build_5_10_5_schedule()", self.model_text)

    def test_converter_metadata_and_external_atomic_output_contract(self):
        for marker in (
            "AutoConfig.from_pretrained", "raw_layers != 30", "actual_layers != 30",
            "source_layers_module", "len(source_layers_module) != 30", "recursive_layer_count = 20",
            "recursive_prefix_layer_count = 5", "recursive_middle_layer_count = 10",
            "recursive_suffix_layer_count = 5", "target.architectures = [\"RecursiveLlamaForCausalLM\"]",
            "logical_to_physical", "tempfile.mkdtemp", "staging.replace(output)",
            "allow-overwrite", "FORBIDDEN_CHECKOUT",
        ):
            self.assertIn(marker, self.converter_text)
        self.assertEqual(self.converter.DEFAULT_OUTPUT_DIR.as_posix(), "/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5")

    def test_stage1_audits_full_forward_backward_cache_and_warning(self):
        for marker in (
            "schedule_trace_audit", "backward_trace_audit", "LOGICAL_TO_PHYSICAL",
            "expected_slots = logical_layers", "precreated_cache", "cache_slot_state",
            "cache_no_cache_tokens_equal", "generation_warning", "[generation-warning]", "RSMOL_5_10_5_SAMPLE_PROMPTS",
        ):
            self.assertIn(marker, self.smoke_text)

    def test_stage4_gate_restrictions_and_formal_contract(self):
        for marker in (
            "MODEL_ARCHITECTURE_CONTRACT = \"logical_30_physical_20_5_10_5_loops_2\"",
            "LOGICAL_TO_PHYSICAL", "if config.gate in {\"B\", \"C\"}",
            "Stage 4 5-10-5 Gate", "DEFAULT_MAX_OPTIMIZER_STEPS = 10",
            "DEFAULT_FORMAL_OPTIMIZER_STEPS = 9244", "data_cursors_by_rank",
            "ParquetFile.iter_batches", "columns=[\"text\"]", "token_weighted_gradient_scale",
            "checkpoint_contract", "architecture_contract = MODEL_ARCHITECTURE_CONTRACT",
            "betas=DEFAULT_ADAMW_BETAS", "amsgrad=DEFAULT_ADAMW_AMSGRAD",
        ):
            self.assertIn(marker, self.stage4_text)
        self.assertIn("train_stage4_5_10_5_ddp.py", self.runtime_text)
        self.assertIn("train_stage4_5_10_5_ddp.sh", self.submit_text)
        self.assertIn("RSMOL_5_10_5_MODEL_DIR", self.runtime_text)
        self.assertIn("RSMOL_5_10_5_MAX_OPTIMIZER_STEPS", self.submit_text)

    def test_conversion_shell_propagates_external_source_and_destination(self):
        self.assertIn("convert_stepwise_5_10_5.py", self.convert_runtime_text)
        self.assertIn("RSMOL_5_10_5_SOURCE_CHECKPOINT", self.convert_runtime_text)
        self.assertIn("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5", self.convert_runtime_text)
        self.assertIn("convert_stepwise_5_10_5.sh", self.convert_submit_text)


if __name__ == "__main__":
    unittest.main()
