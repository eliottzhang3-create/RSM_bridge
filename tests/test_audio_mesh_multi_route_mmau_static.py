"""Static contracts for the shared x2/x4/x5 zero-slot MMAU evaluator."""
from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"
SCRIPTS = RSMOL / "scripts"
EVALUATOR = SCRIPTS / "evaluate_mmau_test_mini_audio_mesh_shared_store.py"
RUNTIME = SCRIPTS / "evaluate_mmau_test_mini_audio_mesh_shared_store.sh"
SUBMIT = RSMOL / "run_mmau_test_mini_audio_mesh_shared_store_3090.sh"
SMOKE = RSMOL / "run_mmau_test_mini_audio_mesh_shared_store_smoke_3090.sh"
NINE_SLOT_SUBMIT = RSMOL / "run_mmau_test_mini_audio_mesh_9slot_3090.sh"
DECODER = SCRIPTS / "generate_audio_checkpoint_reasonaqa.py"


class AudioMeshMultiRouteMMAUStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = EVALUATOR.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")
        cls.smoke = SMOKE.read_text(encoding="utf-8")
        cls.nine_slot_submit = NINE_SLOT_SUBMIT.read_text(encoding="utf-8")
        cls.decoder = DECODER.read_text(encoding="utf-8")

    def test_all_three_route_contracts_are_explicit(self) -> None:
        for route in ("x2_7slot", "x4", "x5_7slot"):
            self.assertIn(f'"{route}"', self.source)
        for filename in (
            "audio_mesh_x2_7slot_fixed260_zero_slot_config.json",
            "audio_mesh_x4_fixed260_zero_slot_config.json",
            "audio_mesh_x5_7slot_fixed260_zero_slot_config.json",
        ):
            self.assertIn(filename, self.source)

    def test_x5_9slot_route_contract_and_submission_entrypoint_are_explicit(self) -> None:
        for marker in (
            '"x5_9slot"',
            "audio_5_10x5_5_mesh_9slot_mellow_shared_store_configurable_epochs",
            "recursive_model_5_10x5_5_mesh_9slot",
            "audio_mesh_x5_9slot_fixed260_zero_slot_config.json",
            "AudioMeshX5NineSlotZeroModel",
            "memory_slots=9",
            "formal_3epochs_20261006_054353274259875-20/checkpoint-011343",
        ):
            self.assertIn(marker, self.source)
        self.assertTrue(NINE_SLOT_SUBMIT.is_file())
        for marker in (
            'export RSMOL_MMAU_ROUTE="x5_9slot"',
            'export RSMOL_MMAU_MODE="full"',
            "mmau_audio_mesh_zero_slot",
        ):
            self.assertIn(marker, self.nine_slot_submit)
        self.assertIn("pdgpu-3090", self.submit)

    def test_inference_contract_is_fp32_runtime_zero_and_fixed260(self) -> None:
        for marker in (
            "autocast_enabled=False",
            "runtime_zero_second_slot",
            "runtime_zero_waveform_on_gpu",
            "torch.zeros((int(audio1.shape[0]),)",
            '"prefix_token_count": 260',
            '"inference_dtype": "float32"',
            '"prompt_answer_layout": "prompt_and_answer_concatenated_before_batch_right_padding"',
        ):
            self.assertIn(marker, self.source)
        self.assertIn("same_real_mask = torch.zeros", self.source)

    def test_both_scoring_paths_are_retained(self) -> None:
        for marker in (
            "write_mellow_author_reply_evaluation",
            "prediction.split(')')[0].lower()",
            '"mmau_v051525_evaluation"',
        ):
            self.assertIn(marker, self.source)
        self.assertIn("--run-official-evaluation", self.submit)

    def test_decoder_accepts_route_specific_trace_without_breaking_default(self) -> None:
        self.assertIn("expected_trace: Sequence[Mapping[str, int]] | None = None", self.decoder)
        self.assertIn("expected_trace = list(expected_trace)", self.decoder)

    def test_submission_uses_one_gpu_3090_and_smoke_is_single(self) -> None:
        for marker in ("pdgpu-3090", "-c 8 -m 32G -g 1 -n 1", "--dtype fp32"):
            self.assertIn(marker, self.submit)
        self.assertIn('RSMOL_MMAU_MODE="smoke"', self.smoke)
        self.assertIn('RSMOL_MMAU_ROUTE:-x5_7slot', self.smoke)
        self.assertTrue(RUNTIME.is_file())


if __name__ == "__main__":
    unittest.main()
