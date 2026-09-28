from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"


class AudioX4SharedStoreStaticTest(unittest.TestCase):
    def read(self, relative: str) -> str:
        return (RSMOL / relative).read_text(encoding="utf-8")

    def test_isolated_package_and_contracts(self) -> None:
        init_text = self.read(
            "audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs/__init__.py"
        )
        model = self.read(
            "audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs/model.py"
        )
        data = self.read(
            "audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs/data.py"
        )
        self.assertIn("runtime_zero_second_slot", init_text)
        self.assertIn("logical_50_physical_20_5_10x4_5_mesh", model)
        self.assertIn("AudioMeshSilenceSlotModel", model)
        self.assertIn("silence_second_slot_mask", data)
        self.assertIn("same_real_audio_mask", data)

    def test_zero_waveform_implementation_is_inherited(self) -> None:
        silence_model = self.read("audio_5_10x2_5_mesh_mellow_silence_slot/model.py")
        self.assertIn("torch.zeros_like(audio1[:1])", silence_model)
        self.assertIn("silence_embedding.expand", silence_model)
        self.assertIn("projected_second = self.bridge(second)", silence_model)

    def test_trainer_uses_x4_and_fresh_text_initialization(self) -> None:
        trainer = self.read("scripts/train_audio_shared_store_5_10x4_5_mesh_mellow_ddp.py")
        self.assertIn("recursive_model_5_10x4_5_mesh", trainer)
        self.assertIn("checkpoint-003081", trainer)
        self.assertIn("_validate_x4_text_checkpoint", trainer)
        self.assertIn('"training_state_loaded": False', trainer)
        self.assertIn("trace_matches_5_10x4_5", trainer)
        self.assertIn("bridge_has_finite_nonzero_gradients", trainer)
        self.assertIn("c2l_has_finite_nonzero_gradients", trainer)
        self.assertIn("htsat_backbone_has_no_gradients", trainer)
        self.assertIn("args.micro_batch_size != 8", trainer)
        self.assertIn("args.gradient_accumulation_steps != 4", trainer)
        self.assertIn("default=500", trainer)
        self.assertIn("default=4", trainer)

    def test_submission_isolated_on_3090(self) -> None:
        for mode in ("smoke20", "resume2", "formal"):
            wrapper = self.read(
                f"run_audio_shared_store_{mode}_configurable_epochs_5_10x4_5_mesh_mellow_3090.sh"
            )
            self.assertIn("-p pdgpu-3090", wrapper)
            self.assertIn("5_10x4_5_mesh_mellow", wrapper)
            self.assertNotIn("pdgpu-5090", wrapper)
        stage = self.read(
            "scripts/stage_audio_shared_store_configurable_epochs_5_10x4_5_mesh_mellow.sh"
        )
        self.assertIn("/dev/shm/rsmol_audio_x4_zero_slot_", stage)
        self.assertIn(
            "train_audio_shared_store_configurable_epochs_5_10x4_5_mesh_mellow_ddp.py",
            stage,
        )


if __name__ == "__main__":
    unittest.main()

