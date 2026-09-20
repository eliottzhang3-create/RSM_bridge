"""Dependency-light checks for the Mellow-v0 eval-only conversion route."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "code" / "RSmol" / "scripts"
CONVERTER = SCRIPT_DIR / "convert_mellow_v0_to_audio_smollm2_eval.py"
EVALUATOR = SCRIPT_DIR / "evaluate_mmau_test_mini_audio_smollm2.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("mellow_eval_artifact_static", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MellowV0AudioSmolLM2ConversionStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.converter = CONVERTER.read_text(encoding="utf-8")
        cls.evaluator_text = EVALUATOR.read_text(encoding="utf-8")
        cls.evaluator = load_evaluator()

    def test_converter_is_cpu_only_and_eval_only(self) -> None:
        self.assertTrue(CONVERTER.is_file())
        self.assertIn('map_location="cpu"', self.converter)
        self.assertNotIn("torch.cuda", self.converter)
        self.assertNotIn("vc submit", self.converter)
        self.assertIn('"training_state_included": False', self.converter)
        self.assertIn('"included": False', self.converter)
        self.assertNotIn('torch.save(groups["htsat"]', self.converter)

    def test_exact_mellow_weight_groups_are_converted(self) -> None:
        for marker in (
            '"c2l": "audio_encoder.base.c2l."',
            '"bridge": "audio_encoder.projection."',
            '"text": "caption_decoder.lm."',
            '"layer_norm.weight": "norm.weight"',
            '"layer_norm.bias": "norm.bias"',
            "load_state_dict(dict(source), strict=True)",
            "text_model.load_state_dict(dict(source), strict=True)",
        ):
            self.assertIn(marker, self.converter)

    def test_artifact_has_a_distinct_non_training_contract(self) -> None:
        contract = "mellow_v0_to_audio_smollm2_compact_eval_v1"
        config = "mellow_audio_smollm2_eval_config.json"
        self.assertIn(contract, self.converter)
        self.assertIn(contract, self.evaluator_text)
        self.assertIn(config, self.converter)
        self.assertIn(config, self.evaluator_text)
        self.assertIn("_audit_eval_only_artifact", self.evaluator_text)
        self.assertIn("_audit_partition_checkpoint", self.evaluator_text)

    def test_existing_evaluator_selects_exactly_one_artifact_kind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eval_config = root / self.evaluator.EVAL_ONLY_CONFIG_FILENAME
            partition_config = root / self.evaluator.PARTITION_CONFIG_FILENAME
            eval_config.write_text("{}", encoding="utf-8")
            kind, selected = self.evaluator._artifact_config_path(root)
            self.assertEqual(kind, "eval_only_model")
            self.assertEqual(selected, eval_config)
            partition_config.write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                self.evaluator._artifact_config_path(root)

    def test_compact_existing_mmau_backend_is_unchanged(self) -> None:
        for marker in (
            "DEFAULT_AUDIO_PREFIX_TOKENS = 130",
            "compact_single_audio_prefix=True",
            "skip_second_prefix=True",
            "run_model_generation=_run_model_generation",
        ):
            self.assertIn(marker, self.evaluator_text)


if __name__ == "__main__":
    unittest.main()
