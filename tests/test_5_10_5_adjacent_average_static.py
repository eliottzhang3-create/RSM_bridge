"""Static and lightweight tensor contracts for adjacent-average 5-10-5."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "code" / "RSmol" / "recursive_model_5_10_5_adjacent_average.py"
CONVERTER = ROOT / "code" / "RSmol" / "scripts" / "convert_stepwise_5_10_5_adjacent_average.py"
RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "convert_stepwise_5_10_5_adjacent_average.sh"
SUBMIT = ROOT / "code" / "RSmol" / "run_convert_stepwise_5_10_5_adjacent_average_3090.sh"
LEGACY_MODEL = ROOT / "code" / "RSmol" / "recursive_model_5_10_5.py"
LEGACY_CONVERTER = ROOT / "code" / "RSmol" / "scripts" / "convert_stepwise_5_10_5.py"


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_5_10_5_adjavg_static", CONVERTER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AdjacentAverageStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL.read_text(encoding="utf-8")
        cls.converter_text = CONVERTER.read_text(encoding="utf-8")
        cls.runtime = RUNTIME.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")
        cls.legacy_model = LEGACY_MODEL.read_text(encoding="utf-8")
        cls.legacy_converter = LEGACY_CONVERTER.read_text(encoding="utf-8")
        cls.converter = load_converter()

    def test_exact_pairing_and_full_source_coverage(self):
        self.assertEqual(
            self.converter.MIDDLE_SOURCE_LAYER_PAIRS_0BASED,
            ((5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
             (15, 16), (17, 18), (19, 20), (21, 22), (23, 24)),
        )
        self.assertEqual(self.converter.source_layer_coverage(), tuple(range(30)))
        self.assertEqual(
            self.converter.LOGICAL_TO_PHYSICAL,
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
             5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
        )

    def test_target_config_records_average_not_legacy_mapping(self):
        config = type("Config", (), {"num_hidden_layers": 30})()
        target = self.converter.build_target_config(config)
        self.assertEqual(target.recursive_initialization_policy, self.converter.INITIALIZATION_POLICY)
        self.assertEqual(target.recursive_average_accumulator_dtype, "float32")
        self.assertEqual(target.recursive_middle_source_layer_pairs_0based[0], [5, 6])
        self.assertEqual(target.recursive_source_layer_coverage_0based, list(range(30)))
        self.assertIsNone(target.recursive_source_layer_indices_0based)
        self.assertIsNone(target.recursive_source_mapping_0based)

    def test_converter_rejects_recursive_source_metadata(self):
        self.assertIn("recursive_source_fields", self.converter_text)
        self.assertIn("original 30-physical-layer SmolLM2", self.converter_text)

    def test_fp32_average_and_nonfloating_guard(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable in the local static environment")

        class Tiny(torch.nn.Module):
            def __init__(self, value: float, marker: int = 7):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([value], dtype=torch.bfloat16))
                self.register_buffer("marker", torch.tensor([marker], dtype=torch.int64))

        left = Tiny(1.0)
        right = Tiny(3.0)
        target = Tiny(0.0)
        audit = self.converter.average_modules_checked(left, right, target, "tiny")
        self.assertEqual(float(target.weight.float().item()), 2.0)
        self.assertEqual(int(target.marker.item()), 7)
        self.assertEqual(audit["floating_tensor_count"], 1)
        with self.assertRaisesRegex(ValueError, "non-floating pair values differ"):
            self.converter.average_modules_checked(Tiny(1.0, 7), Tiny(3.0, 8), Tiny(0.0), "bad")

    def test_isolated_files_and_submission_resources(self):
        for marker in (
            "adjacent_layer_parameter_average_fp32_v1",
            "average_modules_checked",
            "adjacent_average_conversion_metadata.json",
            "tempfile.mkdtemp",
            "staging.replace(output)",
        ):
            self.assertIn(marker, self.converter_text + self.model)
        self.assertIn("-p pdgpu-3090", self.submit)
        self.assertIn("-c 8 -m 32G -g 1 -n 1", self.submit)
        self.assertIn("RSMOL_5_10_5_ADJAVG_SOURCE_CHECKPOINT", self.runtime + self.submit)
        self.assertNotIn("adjacent_layer_parameter_average_fp32_v1", self.legacy_model)
        self.assertNotIn("adjacent_layer_parameter_average_fp32_v1", self.legacy_converter)


if __name__ == "__main__":
    unittest.main()
