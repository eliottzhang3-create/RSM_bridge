"""Dependency-light contracts for native Mellow-v0 MMAU evaluation."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"
SCRIPTS = RSMOL / "scripts"
PREFLIGHT = SCRIPTS / "audit_mellow_v0_artifact.py"
EVALUATOR = SCRIPTS / "evaluate_mmau_test_mini_mellow_v0.py"
PREFLIGHT_SH = RSMOL / "run_mellow_v0_artifact_preflight.sh"
SMOKE_SH = RSMOL / "run_mmau_test_mini_mellow_v0_smoke_4090.sh"
FULL_SH = RSMOL / "run_mmau_test_mini_mellow_v0_full_4090.sh"


def load_evaluator():
    scripts_text = str(SCRIPTS)
    if scripts_text not in sys.path:
        sys.path.insert(0, scripts_text)
    spec = importlib.util.spec_from_file_location("native_mellow_v0_mmau", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeMellowV0MMAUStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.preflight = PREFLIGHT.read_text(encoding="utf-8")
        cls.evaluator_text = EVALUATOR.read_text(encoding="utf-8")
        cls.evaluator = load_evaluator()
        cls.smoke = SMOKE_SH.read_text(encoding="utf-8")
        cls.full = FULL_SH.read_text(encoding="utf-8")

    def test_preflight_strictly_loads_complete_native_checkpoint_on_cpu(self) -> None:
        for marker in (
            '"htsat": "audio_encoder.base.htsat."',
            '"c2l": "audio_encoder.base.c2l."',
            '"projection": "audio_encoder.projection."',
            '"text_decoder": "caption_decoder.lm."',
            'map_location="cpu"',
            'load_state_dict(state, strict=True)',
            '"runtime_state_shapes_match": True',
            '"runtime_state_dtypes_match": True',
            '"example_audio_check": "NOT_REQUESTED"',
            "decoder.py independently hardcodes id 0",
            '"separator_token_id": 0',
        ):
            self.assertIn(marker, self.preflight)
        self.assertNotIn("resource/1.wav", self.preflight)
        self.assertNotIn("resource/2.wav", self.preflight)
        self.assertNotIn("torch.cuda", self.preflight)

    def test_evaluator_uses_native_two_slot_prefix_with_separate_encoding(self) -> None:
        self.assertEqual(self.evaluator.MELLOW_PREFIX_TOKENS, 389)
        self.assertEqual(self.evaluator.MELLOW_AUDIO_TOKENS_PER_SLOT, 129)
        for marker in (
            '"audio1": audio',
            '"audio2": audio',
            "model.generate_prefix_inference",
            '"audio2_reused": False',
            '"audio2_encoded_separately": True',
            '"native_audio_encoder_invocations": 2',
            '"compact_single_audio_prefix_used": False',
        ):
            self.assertIn(marker, self.evaluator_text)

    def test_generation_is_greedy_without_top_p_and_prediction_is_verbatim(self) -> None:
        prepare = self.evaluator.prepare_model_output_for_official_scorer
        for value in ("a) answer", "  D) untouched  ", "free text"):
            self.assertEqual(prepare(value), value)
        common = Path(self.evaluator.official.__file__).read_text(encoding="utf-8")
        for marker in (
            "torch.argmax",
            "use_cache=False",
            '"do_sample": False',
            '"top_p": None',
            '"temperature": 0.0',
            "generated_text = str(generation.get",
        ):
            self.assertIn(marker, self.evaluator_text + "\n" + common)
        self.assertNotIn("cumulative_probs", self.evaluator_text)
        self.assertNotIn("sorted_indices_to_remove", self.evaluator_text)

    def test_preflight_report_is_a_required_fresh_artifact_gate(self) -> None:
        for marker in (
            "_load_and_validate_preflight",
            'report.get("status") != "PASS"',
            'report.get("strict_state_dict_load") is not True',
            "mellow_checkpoint_sha256",
            "current_source_hashes",
            "preflight report is stale",
        ):
            self.assertIn(marker, self.evaluator_text)

    def test_full_requires_persistent_completed_first_five_smoke_gate(self) -> None:
        for marker in (
            'SMOKE_GATE_FILENAME = "mellow_v0_smoke_gate.json"',
            "_require_completed_smoke(args)",
            '"expected_rows": official.SMOKE_ROWS',
            '"terminal_records": official.SMOKE_ROWS',
            '"official_evaluation_status": "NOT_REQUESTED"',
            "_smoke_gate_payload(args, report)",
        ):
            self.assertIn(marker, self.evaluator_text)

    def test_smoke_and_full_share_output_and_respect_scheduler_limits(self) -> None:
        default_output = (
            "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0/"
            "mmau_test_mini_matched_protocol_audio1x2_verbatim_v1"
        )
        for wrapper in (self.smoke, self.full):
            self.assertIn(default_output, wrapper)
            self.assertIn("-p pdgpu-4090", wrapper)
            self.assertIn("-c 8 -m 32G -g 1", wrapper)
            self.assertIn("--preflight-report", wrapper)
            self.assertIn("--max-prompt-tokens 129", wrapper)
            self.assertIn("--max-new-tokens 32", wrapper)
        self.assertIn("--mode smoke", self.smoke)
        self.assertNotIn("--run-official-evaluation", self.smoke)
        self.assertIn("--mode full", self.full)
        self.assertIn("--run-official-evaluation", self.full)
        self.assertIn("scripts/audit_mellow_v0_artifact.sh", PREFLIGHT_SH.read_text(encoding="utf-8"))

    def test_existing_complete_denominator_and_smoke_resume_backend_is_reused(self) -> None:
        combined = self.evaluator_text + "\n" + Path(
            self.evaluator.official.__file__
        ).read_text(encoding="utf-8")
        for marker in (
            "official.run(",
            "SMOKE_ROWS",
            "EXPECTED_FULL_ROWS",
            "prompt_exceeds_max_tokens",
            "skipped_rows_scored_as_incorrect",
            "prepare_prediction=prepare_model_output_for_official_scorer",
        ):
            self.assertIn(marker, combined)


if __name__ == "__main__":
    unittest.main()
