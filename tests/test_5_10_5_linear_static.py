"""Dependency-light contracts for the isolated 5-10-5 linear pipeline."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "code" / "RSmol" / "recursive_model_5_10_5_linear.py"
CONVERTER = ROOT / "code" / "RSmol" / "scripts" / "convert_stepwise_5_10_5_linear.py"
SMOKE = ROOT / "code" / "RSmol" / "scripts" / "smoke_5_10_5_linear.py"
STAGE4 = ROOT / "code" / "RSmol" / "scripts" / "train_stage4_5_10_5_linear_ddp.py"
CONVERT_RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "convert_stepwise_5_10_5_linear.sh"
CONVERT_SUBMIT = ROOT / "code" / "RSmol" / "run_convert_stepwise_5_10_5_linear_3090.sh"
SMOKE_RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "smoke_5_10_5_linear.sh"
SMOKE_SUBMIT = ROOT / "code" / "RSmol" / "run_smoke_5_10_5_linear_3090.sh"
STAGE4_RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "train_stage4_5_10_5_linear_ddp.sh"
STAGE4_SUBMIT = ROOT / "code" / "RSmol" / "run_stage4_5_10_5_linear_3090.sh"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FiveTenFiveLinearStaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_text = MODEL.read_text(encoding="utf-8")
        cls.converter_text = CONVERTER.read_text(encoding="utf-8")
        cls.smoke_text = SMOKE.read_text(encoding="utf-8")
        cls.stage4_text = STAGE4.read_text(encoding="utf-8")
        cls.convert_runtime_text = CONVERT_RUNTIME.read_text(encoding="utf-8")
        cls.convert_submit_text = CONVERT_SUBMIT.read_text(encoding="utf-8")
        cls.smoke_runtime_text = SMOKE_RUNTIME.read_text(encoding="utf-8")
        cls.smoke_submit_text = SMOKE_SUBMIT.read_text(encoding="utf-8")
        cls.stage4_runtime_text = STAGE4_RUNTIME.read_text(encoding="utf-8")
        cls.stage4_submit_text = STAGE4_SUBMIT.read_text(encoding="utf-8")
        cls.converter = load_module(CONVERTER, "stage3_5_10_5_linear_converter_static")
        cls.stage4 = load_module(STAGE4, "stage4_5_10_5_linear_static")

    def test_exact_source_mapping_and_linear_schedule(self):
        expected_source = (0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29)
        self.assertEqual(self.converter.SOURCE_LAYER_INDICES_0BASED, expected_source)
        self.assertEqual(self.converter.LOGICAL_TO_PHYSICAL, tuple(range(20)))
        self.assertEqual(self.converter.build_source_mapping(30), expected_source)
        with self.assertRaises(ValueError):
            self.converter.build_source_mapping(20)

    def test_target_config_contract(self):
        source = SimpleNamespace(num_hidden_layers=30)
        target = self.converter.build_target_config(source)
        self.assertEqual(target.num_hidden_layers, 20)
        self.assertEqual(target.recursive_source_layer_count, 30)
        self.assertEqual(target.recursive_layer_count, 20)
        self.assertEqual(target.recursive_loops, 1)
        self.assertEqual(target.recursive_loops_scope, "none")
        self.assertEqual(target.logical_to_physical, list(range(20)))
        self.assertEqual(target.architectures, ["SmolLM2_5_10_5LinearForCausalLM"])

    def test_no_recursive_second_pass_and_independent_model_contract(self):
        for marker in (
            "LOGICAL_LAYER_COUNT = 20",
            "PHYSICAL_LAYER_COUNT = 20",
            "RECURSIVE_LOOPS = 1",
            "LOGICAL_TO_PHYSICAL = tuple(range(PHYSICAL_LAYER_COUNT))",
            "for logical_index, physical_index in enumerate(self.logical_to_physical)",
            "SmolLM2_5_10_5LinearForCausalLM",
            "source_layer_count != 30",
            "requires recursive_loops_scope='none'",
            "all_physical_layers_independent",
        ):
            self.assertIn(marker, self.model_text)
        self.assertNotIn("middle_only", self.model_text)
        self.assertNotIn("schedule[15:25]", self.model_text)

    def test_converter_reads_source_depth_and_rejects_unsafe_output(self):
        for marker in (
            "AutoConfig.from_pretrained",
            "raw_layers != 30",
            "actual_layers != 30",
            "len(source_layers_module) != 30",
            "recursive_source_layer_count = 30",
            "target.num_hidden_layers = 20",
            "recursive_loops = 1",
            "recursive_loops_scope = \"none\"",
            "tempfile.mkdtemp",
            "staging.replace(output)",
            "FORBIDDEN_CHECKOUT",
        ):
            self.assertIn(marker, self.converter_text)
        self.assertEqual(
            self.converter.DEFAULT_OUTPUT_DIR.as_posix(),
            "/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5-linear",
        )

    def test_stage1_and_stage4_audit_contracts(self):
        for marker in (
            "schedule_trace_audit",
            "backward_trace_audit",
            "expected_slots = logical_layers",
            "precreated_cache",
            "cache_no_cache_tokens_equal",
            "generation_warning",
            "source_layer_count",
        ):
            self.assertIn(marker, self.smoke_text)
        for marker in (
            "MODEL_ARCHITECTURE_CONTRACT = \"logical_20_physical_20_5_10_5_linear_loops_1\"",
            "MAPPING_POLICY = \"explicit_5_10_5_linear_source_layers\"",
            "LOGICAL_LAYER_COUNT = 20",
            "LOGICAL_TO_PHYSICAL = tuple(range(PHYSICAL_LAYER_COUNT))",
            "recursive_model_5_10_5_linear",
            "DEFAULT_FORMAL_OPTIMIZER_STEPS = 9244",
            "DEFAULT_FORMAL_WARMUP_STEPS",
            "betas=DEFAULT_ADAMW_BETAS",
            "amsgrad=DEFAULT_ADAMW_AMSGRAD",
            "ParquetFile.iter_batches",
            "columns=[\"text\"]",
            "token_weighted_gradient_scale",
            "checkpoint_contract",
            "decoder_layer_storage_unique",
        ):
            self.assertIn(marker, self.stage4_text)
        self.assertNotIn("middle_only", self.stage4_text)
        self.assertNotIn("loops=2", self.stage4_text)

    def test_shells_are_isolated_and_use_3090_eight_gpu_submit(self):
        self.assertIn("convert_stepwise_5_10_5_linear.py", self.convert_runtime_text)
        self.assertIn("RSMOL_5_10_5_LINEAR_SOURCE_CHECKPOINT", self.convert_runtime_text)
        self.assertIn("SmolLM2-5-10-5-linear", self.convert_runtime_text)
        self.assertIn("smoke_5_10_5_linear.py", self.smoke_runtime_text)
        self.assertIn("RSMOL_5_10_5_LINEAR_MODEL_DIR", self.smoke_submit_text)
        self.assertIn("train_stage4_5_10_5_linear_ddp.py", self.stage4_runtime_text)
        self.assertIn("RSMOL_5_10_5_LINEAR_AUDIT_REPORT", self.stage4_submit_text)
        self.assertIn('QUEUE="${RSMOL_5_10_5_LINEAR_QUEUE:-pdgpu-3090}"', self.stage4_submit_text)
        self.assertIn("-g 8", self.stage4_submit_text)
        self.assertIn("-m 256G", self.stage4_submit_text)
        self.assertIn("train_stage4_5_10_5_linear_ddp.sh", self.stage4_submit_text)
        self.assertNotIn("RSMOL_5_10_5_MODEL_DIR", self.stage4_submit_text)

    def test_stage4_parser_formal_defaults_and_gate_restriction(self):
        config = self.stage4._parse_args(["--gate", "FORMAL", "--dry-run"])
        self.assertEqual(config.max_optimizer_steps, 9244)
        self.assertEqual(config.warmup_steps, 463)
        self.assertEqual(config.scheduler_total_steps, 9244)
        self.assertEqual(config.save_every, 500)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.stage4._parse_args(["--gate", "B", "--dry-run"])
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.stage4._parse_args(["--gate", "C", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
