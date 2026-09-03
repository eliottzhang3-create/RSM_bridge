"""Dependency-light contracts for the isolated dynamic 5-10xr-5 variant."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "code" / "RSmol" / "recursive_model_5_10xr_5.py"
CONVERTER = ROOT / "code" / "RSmol" / "scripts" / "convert_stepwise_5_10xr_5.py"
SMOKE = ROOT / "code" / "RSmol" / "scripts" / "smoke_recursive_5_10xr_5.py"
STAGE4 = ROOT / "code" / "RSmol" / "scripts" / "train_stage4_5_10xr_5_ddp.py"
STAGE4_RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "train_stage4_5_10xr_5_ddp.sh"
STAGE4_SUBMIT = ROOT / "code" / "RSmol" / "run_stage4_5_10xr_5_3090.sh"
SMOKE_RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "smoke_recursive_5_10xr_5.sh"
SMOKE_SUBMIT = ROOT / "code" / "RSmol" / "run_smoke_recursive_5_10xr_5_3090.sh"
CONVERT_RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "convert_stepwise_5_10xr_5.sh"
CONVERT_SUBMIT = ROOT / "code" / "RSmol" / "run_convert_stepwise_5_10xr_5_3090.sh"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DynamicVariantStaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_text = MODEL.read_text(encoding="utf-8")
        cls.converter_text = CONVERTER.read_text(encoding="utf-8")
        cls.smoke_text = SMOKE.read_text(encoding="utf-8")
        cls.stage4_text = STAGE4.read_text(encoding="utf-8")
        cls.stage4_runtime_text = STAGE4_RUNTIME.read_text(encoding="utf-8")
        cls.stage4_submit_text = STAGE4_SUBMIT.read_text(encoding="utf-8")
        cls.smoke_runtime_text = SMOKE_RUNTIME.read_text(encoding="utf-8")
        cls.smoke_submit_text = SMOKE_SUBMIT.read_text(encoding="utf-8")
        cls.convert_runtime_text = CONVERT_RUNTIME.read_text(encoding="utf-8")
        cls.convert_submit_text = CONVERT_SUBMIT.read_text(encoding="utf-8")
        cls.converter = load_module(CONVERTER, "convert_5_10xr_5_static")
        cls.stage4 = load_module(STAGE4, "stage4_5_10xr_5_static")

    def test_dynamic_schedule_all_r_values(self):
        for r, logical in ((4, 50), (5, 60), (6, 70), (7, 80)):
            schedule = self.converter.SOURCE_LAYER_INDICES_0BASED
            self.assertEqual(len(schedule), 20)
            self.assertEqual(len(self.stage4.LOGICAL_TO_PHYSICAL), 80)
            model_text = self.model_text
            self.assertIn("def build_5_10xr_5_schedule", model_text)
            self.assertIn("middle_loop_count must be in [4, 7]", model_text)
            self.assertEqual(5 + 10 * r + 5, logical)

    def test_sampling_probabilities_and_step_determinism(self):
        self.assertEqual(self.stage4.SAMPLING_WEIGHTS, {4: 16, 5: 25, 6: 36, 7: 49})
        self.assertEqual(self.stage4.SAMPLING_WEIGHT_TOTAL, 126)
        self.assertEqual(
            {r: self.stage4.SAMPLING_WEIGHTS[r] / self.stage4.SAMPLING_WEIGHT_TOTAL for r in (4, 5, 6, 7)},
            {4: 16 / 126, 5: 25 / 126, 6: 36 / 126, 7: 49 / 126},
        )
        values = [self.stage4.sample_middle_loop_count(17, step) for step in range(512)]
        self.assertEqual(values[:32], [self.stage4.sample_middle_loop_count(17, step) for step in range(32)])
        self.assertTrue(all(value in (4, 5, 6, 7) for value in values))
        before = random.getstate()
        self.stage4.sample_middle_loop_count(17, 999)
        self.assertEqual(random.getstate(), before)

    def test_exact_checkpoint_sampling_contract(self):
        contract = {
            "sampling_policy": "increasing_power_weight",
            "sampling_alpha": 2,
            "sampling_weights": {"4": 16, "5": 25, "6": 36, "7": 49},
            "sampling_weight_total": 126,
            "sampler_key": self.stage4.SAMPLER_KEY,
        }
        self.stage4._validate_exact_sampling_contract(contract, label="unit")
        for key, bad_value in (
            ("sampling_weights", {"4": 16, "5": 25, "6": 36, "7": 48}),
            ("sampling_weight_total", 125),
            ("sampler_key", "legacy"),
        ):
            broken = dict(contract)
            broken[key] = bad_value
            with self.assertRaises(ValueError):
                self.stage4._validate_exact_sampling_contract(broken, label="unit")

    def test_audit_counts_and_gate_a_hook_order_are_static_contracts(self):
        self.assertIn("expected_entries = r * MIDDLE_LAYER_COUNT", self.smoke_text)
        self.assertIn("expected_entries = int(middle_loop_count) * MIDDLE_LAYER_COUNT", self.stage4_text)
        self.assertIn("each_middle_loop_has_exactly_ten_physical_calls", self.stage4_text)
        gate_a = self.stage4_text[self.stage4_text.index("def _synthetic_gate_a"):]
        preaudit = gate_a.index("all_r_backward_audits = _all_r_backward_audit(")
        hooks = gate_a.index("sequence, handles = _register_forward_trace(model)")
        self.assertLess(preaudit, hooks)
        self.assertIn("len(sequence) == expected_trace_length", self.stage4_text)
        self.assertIn("len(backward_sequence) == expected_trace_length", self.stage4_text)
        self.assertIn("all(chunk == forward_expected for chunk in forward_chunks)", self.stage4_text)
        self.assertIn("all(chunk == backward_expected for chunk in backward_chunks)", self.stage4_text)

    def test_model_selective_bptt_contract(self):
        for marker in (
            "torch.func import functional_call", "detached_parameters", "parameter_grad_enabled",
            "middle_loop_count", "parameter_gradient_enabled_middle_loops",
            "backward_traversed_middle_loops", "_bind_cache_middle_loop_count",
            "cache is already bound to a different middle_loop_count",
            "DEFAULT_INFERENCE_MIDDLE_LOOPS = 7", "PARAMETER_GRADIENT_TAIL_LOOPS = 4",
            "Invalid 5-10xr-5 sampling metadata", "SAMPLER_KEY",
        ):
            self.assertIn(marker, self.model_text)
        self.assertNotIn("no_grad():", self.model_text)

    def test_converter_metadata_and_isolation(self):
        for marker in (
            "AutoConfig.from_pretrained", "raw_layers != 30", "actual_layers != 30",
            "source_layers_module", "len(source_layers_module) != 30", "target.num_hidden_layers = 80",
            "recursive_source_num_hidden_layers = 30", "recursive_source_layer_count = 30",
            "recursive_min_middle_loops", "recursive_max_middle_loops",
            "recursive_parameter_gradient_tail_loops", "sampling_weights",
            "target.architectures = [\"RecursiveLlama5_10xr_5ForCausalLM\"]",
            "tempfile.mkdtemp", "staging.replace(output)", "allow-overwrite", "FORBIDDEN_CHECKOUT",
        ):
            self.assertIn(marker, self.converter_text)
        self.assertEqual(self.converter.DEFAULT_OUTPUT_DIR.as_posix(), "/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10xr-5")
        self.assertNotIn("recursive_model_5_10_5", self.converter_text)

    def test_stage1_and_stage4_audit_contracts(self):
        for marker in (
            "r_values_audited", "forward_trace_audit", "backward_trace_audit", "cache_r_mismatch_rejected",
            "save_reload_audit", "default_inference_r", "fixed_parameter_gradient_tail_loops",
            "explicit_r4_fixed", "all_supported_r_fixed", "all_supported_r",
            "early_parameter_gradient_edges_absent",
            "early_hidden_gradient_norms", "exact_parameter_gradient_tail",
            "expected_entries = r * MIDDLE_LAYER_COUNT", "middle loop {loop} must contain physical layers 5..14 exactly once",
            "parameter_identity_and_requires_grad_restored",
            "suffix_layers_with_grad", "suffix_all_receive_finite_nonzero_grad",
            "recursive_sampling_policy", "recursive_sampling_weight_total", "recursive_sampler_key",
        ):
            self.assertIn(marker, self.smoke_text)
        self.assertIn("prepare_inputs_for_generation", self.model_text)
        for marker in (
            "MODEL_ARCHITECTURE_CONTRACT = \"logical_50_80_physical_20_5_10xr_5_dynamic_r4_7_tail4\"",
            "sample_middle_loop_count", "broadcast_middle_loop_count", "optimizer_step",
            "all_rank_r_equal", "parameter_gradient_enabled_middle_loops", "sampling_contract",
            "DEFAULT_FORMAL_OPTIMIZER_STEPS = 4500", "DEFAULT_FORMAL_WARMUP_STEPS",
            "FORMAL requires scheduler_total_steps=4500", "warmup_steps=225",
            "data_cursors_by_rank", "checkpoint_contract", "fixed_parameter_gradient_tail_loops",
            "r_sequence_matches_uninterrupted", "sampled_r_expected_from_step_key",
            "selective_middle_gradient_audit", "early_parameter_gradient_edges_absent",
            "all_r_backward_audits", "for middle_loop_count in range(MIN_MIDDLE_LOOPS, MAX_MIDDLE_LOOPS + 1)",
            "Install the sampled-r trace hooks only after the all-r pre-audit",
            "all_microbatches_same_r_trace", "expected_sampled_r_trace_total_length",
            "prefix_layers_with_grad", "prefix_all_receive_finite_nonzero_grad",
            "suffix_layers_with_grad", "suffix_all_receive_finite_nonzero_grad",
            "sampler seed mismatch", "sampled_r_expected_from_step_key",
            "\"sampling_weight_total\": SAMPLING_WEIGHT_TOTAL",
            "_validate_exact_sampling_contract", "Resume checkpoint_complete.json",
            "sha256", "random.Random(local_seed).randrange(SAMPLING_WEIGHT_TOTAL)",
        ):
            self.assertIn(marker, self.stage4_text)

    def test_wrappers_use_only_new_namespace(self):
        for text in (self.stage4_runtime_text, self.stage4_submit_text, self.smoke_runtime_text, self.smoke_submit_text, self.convert_runtime_text, self.convert_submit_text):
            self.assertIn("RSMOL_5_10XR_5", text)
            self.assertNotIn("RSMOL_5_10_5", text)
        self.assertIn("pdgpu-3090", self.stage4_submit_text)
        self.assertIn("-c 32 -m 256G -g 8", self.stage4_submit_text)
        self.assertIn("-c 8 -m 32G -g 1", self.smoke_submit_text)
        self.assertIn("4500", self.stage4_runtime_text)
        self.assertIn("225", self.stage4_runtime_text)

    def test_formal_sample_contract(self):
        self.assertEqual(4500 * 8 * 16 * 8, 4_608_000)
        self.assertEqual(4500 * 16 * 8, 576_000)
        self.assertEqual(4500 * 16, 72_000)
        self.assertEqual(self.stage4.DEFAULT_FORMAL_WARMUP_STEPS, 225)
        self.assertEqual(self.stage4.formal_save_steps(4500, 500)[-1], 4500)


if __name__ == "__main__":
    unittest.main()
