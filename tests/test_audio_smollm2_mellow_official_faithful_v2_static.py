"""Static contracts for the isolated Mellow official-faithful-v2 route."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"
SCRIPTS = RSMOL / "scripts"
PACKAGE = (
    RSMOL
    / "audio_smollm2_135m_mellow_official_faithful_v2_shared_store_configurable_epochs"
)
TRAIN = (
    SCRIPTS
    / "train_audio_smollm2_shared_store_mellow_official_faithful_v2_135m_ddp.py"
)
ENTRY = (
    SCRIPTS
    / "train_audio_smollm2_shared_store_configurable_epochs_135m_mellow_official_faithful_v2_ddp.py"
)
STAGE = (
    SCRIPTS
    / "stage_audio_smollm2_shared_store_configurable_epochs_135m_mellow_official_faithful_v2.sh"
)
SMOKE = (
    SCRIPTS
    / "train_audio_smollm2_shared_store_smoke20_configurable_epochs_135m_mellow_official_faithful_v2_ddp.sh"
)
REFERENCE = (
    SCRIPTS
    / "train_audio_smollm2_shared_store_reference22_configurable_epochs_135m_mellow_official_faithful_v2_ddp.sh"
)
RESUME = (
    SCRIPTS
    / "train_audio_smollm2_shared_store_resume2_configurable_epochs_135m_mellow_official_faithful_v2_ddp.sh"
)
FORMAL = (
    SCRIPTS
    / "train_audio_smollm2_shared_store_formal_configurable_epochs_135m_mellow_official_faithful_v2_ddp.sh"
)
SUBMITS = tuple(
    RSMOL
    / f"run_audio_smollm2_shared_store_{mode}_configurable_epochs_135m_mellow_official_faithful_v2_3090.sh"
    for mode in ("smoke20", "reference22", "resume2", "formal")
)
OLD_PACKAGE = RSMOL / "audio_smollm2_135m_mellow_shared_store_configurable_epochs"
OLD_TRAIN = SCRIPTS / "train_audio_smollm2_shared_store_mellow_faithful_135m_ddp.py"


class MellowOfficialFaithfulV2StaticTest(unittest.TestCase):
    def test_files_exist_and_python_parses(self) -> None:
        python_files = (
            PACKAGE / "__init__.py",
            PACKAGE / "data.py",
            PACKAGE / "model.py",
            PACKAGE / "sampler.py",
            PACKAGE / "mellow_templates.py",
            PACKAGE / "grad_norm_tracker.py",
            TRAIN,
            ENTRY,
        )
        for path in (
            *python_files,
            PACKAGE / "README.md",
            STAGE,
            SMOKE,
            REFERENCE,
            RESUME,
            FORMAL,
            *SUBMITS,
        ):
            self.assertTrue(path.is_file(), path)
        for path in python_files:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_new_contract_and_old_route_are_distinct(self) -> None:
        new_contract = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
        old_contract = (OLD_PACKAGE / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("htsat_train_separator0_official_gradnorm", new_contract)
        self.assertNotEqual(new_contract, old_contract)
        self.assertNotIn("official_faithful_v2", old_contract)
        self.assertNotIn("official_faithful_v2", OLD_TRAIN.read_text(encoding="utf-8"))

    def test_model_keeps_frozen_htsat_in_train_mode_and_separator_zero(self) -> None:
        model = (PACKAGE / "model.py").read_text(encoding="utf-8")
        for marker in (
            "OFFICIAL_SEPARATOR_TOKEN_ID = 0",
            "nn.Module.train(self, mode)",
            "parameter.requires_grad_(False)",
            "parameter.requires_grad_(True)",
            "self.htsat_wrapper.named_buffers()",
            "load_htsat_buffer_state",
            "last_separator_pair_ids",
            "separator_equals_pad",
        ):
            self.assertIn(marker, model)
        self.assertNotIn("self.htsat_wrapper.eval()", model)
        self.assertNotIn("self.htsat_backbone.eval()", model)
        resolve_start = model.index("def _resolve_separator")
        resolve_end = model.index("def train", resolve_start)
        self.assertNotIn('convert_tokens_to_ids("!")', model[resolve_start:resolve_end])

    def test_official_grad_norm_tracker_contract(self) -> None:
        tracker = (PACKAGE / "grad_norm_tracker.py").read_text(encoding="utf-8")
        for marker in (
            "c8204d8eb99b4384fd7a76ad57995731e0c0c2bf",
            "overdrive_factor: float = 2.5",
            "momentum: float = 0.995",
            "l2_limit = running_l2 * self.overdrive_factor",
            "max_limit = running_max * self.overdrive_factor",
            "min_scale = min(min_scale, l2_limit / l2_norm)",
            "parameter.grad.mul_(min_scale)",
            "return math.sqrt(total_l2_squared), min_scale",
            "self.running_norm = restored",
        ):
            self.assertIn(marker, tracker)
        state_dict = tracker[tracker.index("def state_dict"):tracker.index("def load_state_dict")]
        self.assertNotIn("history", state_dict)

    def test_trainer_uses_tracker_scaler_and_rank_buffers(self) -> None:
        trainer = TRAIN.read_text(encoding="utf-8")
        for marker in (
            "GradNormTracker(",
            "torch.cuda.amp.GradScaler(enabled=False)",
            "grad_scaler.scale(output.loss / args.gradient_accumulation_steps).backward()",
            "grad_scaler.unscale_(optimizer)",
            "grad_norm_tracker.track_and_clip_(",
            "grad_norm_tracker_reference_audit()",
            "GradNormTracker state_dict resume reference failed",
            "list(owner.named_parameters())",
            "grad_scaler.step(optimizer)",
            "grad_scaler.update()",
            'temporary / "htsat_buffers_by_rank.pt"',
            '"grad_norm_tracker_state"',
            "model.load_htsat_buffer_state(rank_buffer_state)",
            "broadcast_epoch_state(owner, optimizer, world)",
            'if value.device.type == "cuda"',
            "dist.broadcast_object_list(payload, src=0)",
            "htsat_buffer_change_audit",
            "last_separator_pair_ids",
            "OFFICIAL_SEPARATOR_TOKEN_ID",
        ):
            self.assertIn(marker, trainer)
        self.assertNotIn("clip_grad_norm_", trainer)
        self.assertNotIn(
            "audio_smollm2_135m_mellow_shared_store_configurable_epochs import",
            trainer,
        )
        self.assertIn('"fixed_clip_grad_norm": None', trainer)

    def test_checkpoint_and_formal_gate_require_v2_resume(self) -> None:
        trainer = TRAIN.read_text(encoding="utf-8")
        for marker in (
            '"htsat_buffers_by_rank.pt"',
            '"htsat_buffer_inventory"',
            '"grad_norm_tracker_contract"',
            '"grad_scaler_state": {}',
            "formal requires --smoke20-report and --smoke-resume-report",
            "formal gate rejects resume2 status/cursor",
            "resume_state_restore_audit",
            "checkpoint22",
        ):
            self.assertIn(marker, trainer)
        formal = FORMAL.read_text(encoding="utf-8")
        self.assertIn("--smoke20-report", formal)
        self.assertIn("--smoke-resume-report", formal)

    def test_batch_geometry_storage_and_submission_queue(self) -> None:
        trainer = TRAIN.read_text(encoding="utf-8")
        for marker in (
            "QUALIFICATION_MICRO_BATCH = 4",
            "QUALIFICATION_GRAD_ACCUM = 1",
            "QUALIFICATION_GLOBAL_BATCH = 32",
            "FORMAL_MICRO_BATCH = 8",
            "FORMAL_GRAD_ACCUM = 4",
            "FORMAL_GLOBAL_BATCH = 256",
            "CANONICAL_SAVE_EVERY_STEPS = 5_000",
            "CANONICAL_CHECKPOINT_RETENTION = 4",
        ):
            self.assertIn(marker, trainer)
        stage = STAGE.read_text(encoding="utf-8")
        self.assertIn("/dev/shm/rsmol_smollm2_mellow_official_faithful_v2_", stage)
        self.assertIn(
            "train_audio_smollm2_shared_store_configurable_epochs_135m_mellow_official_faithful_v2_ddp.py",
            stage,
        )
        for path in SUBMITS:
            text = path.read_text(encoding="utf-8")
            self.assertIn("-p pdgpu-3090", text)
            self.assertNotIn("pdgpu-5090", text)
            self.assertIn("official_faithful_v2", text)
        for path in (SMOKE, REFERENCE, RESUME):
            text = path.read_text(encoding="utf-8")
            self.assertIn("--micro-batch-size 4", text)
            self.assertIn("--gradient-accumulation-steps 1", text)
        formal = FORMAL.read_text(encoding="utf-8")
        self.assertIn("--micro-batch-size 8", formal)
        self.assertIn("--gradient-accumulation-steps 4", formal)


if __name__ == "__main__":
    unittest.main()
