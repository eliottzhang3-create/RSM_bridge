"""Dependency-light contracts for checkpoint-037810 MMAU/MMAR evaluation."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"
SCRIPTS = RSMOL / "scripts"
MMAU = SCRIPTS / "evaluate_mmau_test_mini_audio_5_10x2_5_mesh_mellow_shared_store.py"
MMAR = SCRIPTS / "evaluate_mmar_audio_5_10x2_5_mesh_mellow_shared_store.py"
MMAU_RUNTIME = SCRIPTS / "evaluate_mmau_test_mini_audio_5_10x2_5_mesh_mellow_shared_store.sh"
MMAR_RUNTIME = SCRIPTS / "evaluate_mmar_audio_5_10x2_5_mesh_mellow_shared_store.sh"
MMAU_SUBMIT = RSMOL / "run_mmau_test_mini_audio_5_10x2_5_mesh_mellow_shared_store_4090.sh"
MMAR_SUBMIT = RSMOL / "run_mmar_audio_5_10x2_5_mesh_mellow_shared_store_5090.sh"


def load_module(name: str, path: Path):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


class SharedStoreEvaluationStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mmau = load_module("shared_store_mmau_static", MMAU)
        cls.mmar = load_module("shared_store_mmar_static", MMAR)
        cls.mmau_source = MMAU.read_text(encoding="utf-8")
        cls.mmar_source = MMAR.read_text(encoding="utf-8")
        cls.mmau_submit = MMAU_SUBMIT.read_text(encoding="utf-8")
        cls.mmar_submit = MMAR_SUBMIT.read_text(encoding="utf-8")

    def test_exact_completed_checkpoint_and_schedule_are_locked(self):
        normalized = str(self.mmau.DEFAULT_CHECKPOINT).replace("\\", "/")
        self.assertTrue(normalized.endswith("formal_10epochs_20260923/checkpoint-037810"))
        self.assertEqual(self.mmau.EXPECTED_FINAL_STEP, 37_810)
        self.assertEqual(self.mmau.EXPECTED_EPOCHS, 10)
        self.assertEqual(self.mmau.EXPECTED_STEPS_PER_EPOCH, 3_781)
        self.assertEqual(self.mmau.EXPECTED_WARMUP_STEPS, 1_891)
        self.assertEqual(self.mmau.EXPECTED_PREFIX_TOKENS, {"single": 260, "dual": 260})

    def test_predictions_are_passed_verbatim_without_leading_label_removal(self):
        for value in ("a) dog", " B) music", "d) answer", "free form"):
            self.assertEqual(self.mmau.prepare_model_output_for_official_scorer(value), value)
        self.assertEqual(
            self.mmau.PREDICTION_FORMAT,
            "mellow_author_reply_raw_generation_choice_label_scoring_v1",
        )
        for source in (self.mmau_source, self.mmar_source):
            self.assertIn("prepare_prediction=", source)
            self.assertNotIn('re.sub(r"^\\s*[a-d]\\)\\s*"', source)

    def test_shared_store_loader_and_fixed260_prefix_are_explicit(self):
        for marker in (
            "def _audit_checkpoint",
            "def _load_runtime_model",
            "def _build_fixed260_reused_audio1_prefix",
            "compact_single_audio_prefix=False",
            "audio2_prefix_materialized",
            "audio2_reused",
            "htsat_audio1_embedding_reused_for_slot2",
            "bridge_invocations_for_reused_embedding",
            "prefix_token_count",
        ):
            self.assertIn(marker, self.mmau_source)
        self.assertIn("model.encode_audio(", self.mmau_source)
        self.assertIn("skip_second_prefix=False", self.mmau_source)
        self.assertIn("generate_audio_checkpoint_reasonaqa import _greedy_decode", self.mmau_source)
        self.assertNotIn("official._run_model_generation(", self.mmau_source)
        self.assertNotIn("_validate_checkpoint_contract(args)", self.mmau_source)

    def test_mmau_uses_mellow_author_reply_protocol_without_changing_slot_contract(self):
        for marker in (
            "MMAU_PROTOCOL_CONTRACT",
            "build_mellow_author_reply_prompt",
            "decode_mellow_author_reply_audio",
            "mellow_author_reply_audio_segment",
            "top_p=0.8",
            "temperature=1.0",
            "mellow_wrapper_decode_then_split_stop_token",
            "write_mellow_author_reply_evaluation",
            '"mmau_v051525_evaluation"',
            '"comparable": bool(',
        ):
            self.assertIn(marker, self.mmau_source)
        self.assertIn("htsat_audio1_embedding_reused_for_slot2", self.mmau_source)
        self.assertIn("--max-new-tokens 300", self.mmau_submit)
        self.assertIn("--dtype fp32", self.mmau_submit)
        self.assertIn("mellow_author_reply_protocol_v1", self.mmau_submit)
        self.assertIn(
            "run_model_generation=_run_mmau_author_reply_generation",
            self.mmau_source,
        )

    def test_mmar_keeps_the_legacy_generation_path(self):
        self.assertIn("run_model_generation=shared._run_model_generation", self.mmar_source)
        self.assertNotIn("_run_mmau_author_reply_generation", self.mmar_source)
        self.assertIn("autocast_enabled=True", self.mmau_source)
        self.assertIn("autocast_enabled=False", self.mmau_source)

    def test_mmar_uses_context_safe_fixed260_prompt_budget(self):
        self.assertEqual(self.mmar.MMAR_MAX_NEW_TOKENS, 32)
        self.assertEqual(self.mmar.FIXED260_MAX_PROMPT_TOKENS, 476)
        args = self.mmar.parse_args(["--output-dir", "/tmp/shared-store-mmar"])
        self.assertEqual(args.mode, "full")
        self.assertEqual(args.max_prompt_tokens, 476)
        self.assertEqual(args.checkpoint, Path(self.mmau.DEFAULT_CHECKPOINT))

    def test_common_pipelines_separate_inference_and_official_score(self):
        mmau_common = (SCRIPTS / "evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.py").read_text(encoding="utf-8")
        mmar_common = (SCRIPTS / "evaluate_mmar_5_10x2_5_mesh_mellow.py").read_text(encoding="utf-8")
        for source in (mmau_common, mmar_common):
            self.assertIn('"inference_coverage"', source)
            self.assertIn('"comparable_official_score"', source)
            self.assertIn("official_scorer_not_requested", source)
        self.assertIn('"prompt_length_audit"', mmau_common)
        self.assertIn('"prompt_exceeds_max_tokens"', mmau_common)

    def test_submission_wrappers_are_full_official_runs(self):
        for submit, runtime in (
            (self.mmau_submit, MMAU_RUNTIME.name),
            (self.mmar_submit, MMAR_RUNTIME.name),
        ):
            self.assertIn("--mode full", submit)
            self.assertIn("--run-official-evaluation", submit)
            self.assertIn("checkpoint-037810", submit)
            self.assertIn("-c 8 -m 32G -g 1", submit)
            self.assertIn(runtime, submit)
        self.assertIn("pdgpu-4090", self.mmau_submit)
        self.assertIn("pdgpu-5090", self.mmar_submit)
        self.assertIn("--max-prompt-tokens 476", self.mmar_submit)
        self.assertTrue(MMAU_RUNTIME.is_file())
        self.assertTrue(MMAR_RUNTIME.is_file())


if __name__ == "__main__":
    unittest.main()
