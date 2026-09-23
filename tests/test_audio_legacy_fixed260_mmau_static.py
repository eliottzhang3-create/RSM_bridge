"""Dependency-light contracts for the original fixed-260 MMAU evaluator."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"
SCRIPTS = RSMOL / "scripts"
EVALUATOR = SCRIPTS / "evaluate_mmau_test_mini_audio_5_10x2_5_mesh_mellow_legacy_fixed260.py"
RUNTIME = SCRIPTS / "evaluate_mmau_test_mini_audio_5_10x2_5_mesh_mellow_legacy_fixed260.sh"
SUBMIT = RSMOL / "run_mmau_test_mini_audio_5_10x2_5_mesh_mellow_legacy_fixed260_5090.sh"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("legacy_fixed260_mmau_static", EVALUATOR)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


class LegacyFixed260MMAUStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()
        cls.source = EVALUATOR.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_exact_historical_checkpoint_and_schedule_are_locked(self):
        normalized = str(self.module.DEFAULT_CHECKPOINT).replace("\\", "/")
        self.assertTrue(
            normalized.endswith(
                "audio_5_10x2_5_mesh_mellow/"
                "formal_restart_save500_20260910_105248/checkpoint-011343"
            )
        )
        self.assertEqual(self.module.EXPECTED_FINAL_STEP, 11_343)
        self.assertEqual(self.module.EXPECTED_EPOCHS, 3)
        self.assertEqual(self.module.EXPECTED_STEPS_PER_EPOCH, 3_781)
        self.assertEqual(self.module.EXPECTED_FINAL_EPOCH_INDEX, 2)
        self.assertEqual(self.module.EXPECTED_FINAL_BATCH_IN_EPOCH, 15_124)
        self.assertEqual(self.module.EXPECTED_WARMUP_STEPS, 568)

    def test_legacy_schema_is_not_mislabeled_as_shared_store(self):
        for marker in (
            '"min_lr": 0.0',
            '"save_every": 500',
            '"batch_in_epoch": EXPECTED_FINAL_BATCH_IN_EPOCH',
            '"contract",',
            '"compact_single_audio_prefix",',
            '"answer_termination",',
            '"historical_schema"',
            '"historical pre-answer-EOS checkpoint"',
        ):
            self.assertIn(marker, self.source)
        self.assertIn(self.module.HISTORICAL_TRAINER_COMMIT, self.source)
        self.assertNotIn("node_shared_unique_store_fullshuffle", self.source)

    def test_current_fixed260_runtime_and_verbatim_predictions_are_reused(self):
        for marker in (
            "run_model_generation=fixed260._run_model_generation",
            "prepare_prediction=fixed260.prepare_model_output_for_official_scorer",
            "prediction_format=fixed260.PREDICTION_FORMAT",
            "audio_prefix_tokens=260",
            "audit_checkpoint=_audit_checkpoint",
        ):
            self.assertIn(marker, self.source)
        self.assertEqual(
            self.module.fixed260.prepare_model_output_for_official_scorer("a) answer"),
            "a) answer",
        )

    def test_parser_defaults_to_full_official_checkpoint(self):
        args = self.module.parse_args(["--output-dir", "/tmp/legacy-fixed260-mmau"])
        self.assertEqual(args.mode, "full")
        self.assertEqual(args.checkpoint, Path(self.module.DEFAULT_CHECKPOINT))
        self.assertEqual(args.max_prompt_tokens, 129)
        self.assertEqual(args.max_new_tokens, 32)

    def test_submission_is_full_official_evaluation(self):
        for marker in (
            "--mode full",
            "--run-official-evaluation",
            "formal_restart_save500_20260910_105248/checkpoint-011343",
            "pdgpu-5090",
            "-g 1",
            RUNTIME.name,
        ):
            self.assertIn(marker, self.submit)
        self.assertTrue(RUNTIME.is_file())


if __name__ == "__main__":
    unittest.main()
