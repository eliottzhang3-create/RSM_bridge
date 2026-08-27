"""Dependency-free contract checks for Stage 1 conversion safeguards.

These tests intentionally do not import Torch or Transformers.  They keep the
production mapping and forbidden-output policy reviewable on login/development
machines where the remote runtime is unavailable.
"""

from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RECURSIVE_SOURCE = ROOT / "code" / "RSmol" / "recursive_model.py"
CONVERTER_SOURCE = ROOT / "code" / "RSmol" / "scripts" / "convert_stepwise.py"
SMOKE_SOURCE = ROOT / "code" / "RSmol" / "scripts" / "smoke_recursive.py"


def load_converter_without_optional_dependencies():
    spec = importlib.util.spec_from_file_location("stepwise_converter_static", CONVERTER_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load converter source: {CONVERTER_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StepwiseStaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recursive_text = RECURSIVE_SOURCE.read_text(encoding="utf-8")
        cls.converter_text = CONVERTER_SOURCE.read_text(encoding="utf-8")
        cls.smoke_text = SMOKE_SOURCE.read_text(encoding="utf-8")
        cls.converter = load_converter_without_optional_dependencies()

    def test_production_mapping_is_literal_and_policy_is_stable(self) -> None:
        self.assertIn('MAPPING_POLICY = "explicit_1based_odd_plus_last"', self.recursive_text)
        self.assertIn("PROJECT_SOURCE_LAYERS = 30", self.recursive_text)
        self.assertIn("mapping = tuple(range(0, 27, 2)) + (29,)", self.recursive_text)
        self.assertIn("production_mapping = build_stepwise_mapping(PROJECT_SOURCE_LAYERS)", self.converter_text)
        self.assertIn("mapping_policy = MAPPING_POLICY", self.converter_text)
        self.assertIn("expected_production_mapping = list(build_stepwise_mapping(PROJECT_SOURCE_LAYERS))", self.smoke_text)

    def test_logical_and_physical_depth_contract_is_explicit(self) -> None:
        self.assertIn("target_config.num_hidden_layers = logical_layer_count", self.converter_text)
        self.assertIn("target_config.recursive_layer_count = physical_layer_count", self.converter_text)
        self.assertIn("expected_slots = logical_layers", self.smoke_text)
        self.assertIn("logical_layer_count", self.recursive_text)
        self.assertIn("logical_cache_slot_count", self.recursive_text)
        self.assertIn('"source_logical_layers": source_layers', self.converter_text)
        self.assertIn('"target_logical_layers": int(target_config.num_hidden_layers)', self.converter_text)
        self.assertIn('"target_physical_unique_layers": len(mapping)', self.converter_text)
        self.assertNotIn("cache_config.num_hidden_layers = int(config.num_hidden_layers) * DEFAULT_LOOPS", self.recursive_text)

        source = SimpleNamespace(num_hidden_layers=4)
        target = self.converter.build_target_config(
            source,
            (0, 3),
            loops=2,
            mapping_policy="explicit_fixture",
        )
        self.assertEqual(target.num_hidden_layers, 4)
        self.assertEqual(target.recursive_layer_count, 2)
        self.assertEqual(target.recursive_loops, 2)
        with self.assertRaisesRegex(ValueError, "depth mismatch"):
            self.converter.build_target_config(
                SimpleNamespace(num_hidden_layers=2),
                (0, 1),
                loops=2,
                mapping_policy="explicit_fixture",
            )

    def test_forbidden_checkout_root_and_children_are_rejected(self) -> None:
        forbidden = self.converter.FORBIDDEN_CHECKOUT
        with self.assertRaises(ValueError):
            self.converter.reject_forbidden_output(forbidden)
        with self.assertRaises(ValueError):
            self.converter.reject_forbidden_output(forbidden / "converted-model")
        with self.assertRaises(ValueError):
            self.converter.convert(
                argparse.Namespace(
                    source_checkpoint=forbidden / "input-checkpoint",
                    output_dir=forbidden / "converted-model",
                    seed=0,
                    allow_overwrite=False,
                    source_layer_indices=None,
                )
            )

    def test_forbidden_checkout_siblings_are_allowed(self) -> None:
        forbidden = self.converter.FORBIDDEN_CHECKOUT
        self.converter.reject_forbidden_output(forbidden.parent / "RSLAM-sibling")

    def test_output_directory_is_explicit_and_overwrite_is_opt_in(self) -> None:
        self.assertIn('parser.add_argument("--output-dir", type=Path, required=True)', self.converter_text)
        self.assertIn('parser.add_argument("--allow-overwrite", action="store_true")', self.converter_text)
        self.assertIn("if output.exists() and not args.allow_overwrite", self.converter_text)


if __name__ == "__main__":
    unittest.main()
