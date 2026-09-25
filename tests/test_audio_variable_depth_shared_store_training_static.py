"""Static contracts for variable-depth Audio MeSH phases 5--8."""
from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"
SCRIPTS = RSMOL / "scripts"
TRAINER = SCRIPTS / "train_audio_shared_store_5_10x2to10_5_mesh_mellow_ddp.py"
STAGE = SCRIPTS / "stage_audio_shared_store_5_10x2to10_5_mesh_mellow.sh"
SMOKE = SCRIPTS / "train_audio_shared_store_5_10x2to10_5_mesh_mellow_smoke20_ddp.sh"
RESUME = SCRIPTS / "train_audio_shared_store_5_10x2to10_5_mesh_mellow_resume2_ddp.sh"
FORMAL = SCRIPTS / "train_audio_shared_store_5_10x2to10_5_mesh_mellow_formal_ddp.sh"
SUBMITS = (
    RSMOL / "run_audio_shared_store_5_10x2to10_5_mesh_mellow_smoke20_5090.sh",
    RSMOL / "run_audio_shared_store_5_10x2to10_5_mesh_mellow_resume2_5090.sh",
    RSMOL / "run_audio_shared_store_5_10x2to10_5_mesh_mellow_formal_5090.sh",
)
FIXED_TRAINER = SCRIPTS / "train_audio_shared_store_5_10x2_5_mesh_mellow_ddp.py"


class VariableDepthSharedStoreTrainingStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trainer = TRAINER.read_text(encoding="utf-8")

    def test_phase5_isolated_trainer_is_parseable(self) -> None:
        for path in (TRAINER, STAGE, SMOKE, RESUME, FORMAL, *SUBMITS):
            self.assertTrue(path.is_file(), path)
        ast.parse(self.trainer)
        fixed = FIXED_TRAINER.read_text(encoding="utf-8")
        self.assertNotIn("2to10", fixed)
        self.assertIn("FORMAL_EPOCHS = 3", fixed)

    def test_phase5_depth_sampling_and_ddp_contract(self) -> None:
        for marker in (
            "SynchronizedDepthSampler(seed=args.seed)",
            "recursive_depth = depth_sampler.sample(device)",
            "recursive_depth=recursive_depth",
            "find_unused_parameters=True",
            '"ddp_sync_each_micro_step": True',
            '"depth_sampler": depth_sampler.state_dict()',
            '"depth_sampler_rank_summaries": depth_summaries',
            "resume checkpoint lacks complete depth-sampler rank coverage",
        ):
            self.assertIn(marker, self.trainer)
        self.assertNotIn("ddp.no_sync()", self.trainer)

    def test_phase6_phase7_phase8_gates(self) -> None:
        for marker in (
            "SMOKE_FIRST_STOP = 20", "SMOKE_TOTAL_STEPS = 22",
            '"phase6_smoke20"', '"phase7_exact_resume2"',
            '"phase8_formal_7epochs"',
            "formal training requires --smoke20-report and --smoke-resume-report",
            "smoke20 depth-sampler draw cursor is not exactly 80",
            "resume2 depth-sampler draw cursor is not exactly 88",
            "resume_verified_two_steps",
        ):
            self.assertIn(marker, self.trainer)
        self.assertIn("--resume-from", RESUME.read_text(encoding="utf-8"))
        formal = FORMAL.read_text(encoding="utf-8")
        self.assertIn("--smoke20-report", formal)
        self.assertIn("--smoke-resume-report", formal)

    def test_fixed_seven_epoch_shape_and_real_step_warmup(self) -> None:
        global_batch = 8 * 8 * 4
        steps_per_epoch = 968_059 // global_batch
        total_steps = steps_per_epoch * 7
        self.assertEqual(global_batch, 256)
        self.assertEqual(steps_per_epoch, 3_781)
        self.assertEqual(total_steps, 26_467)
        self.assertEqual(math.ceil(total_steps * 0.05), 1_324)
        for shell in (SMOKE, RESUME, FORMAL):
            text = shell.read_text(encoding="utf-8")
            self.assertIn("--epochs 7", text)
            self.assertIn("--max-lr 1e-3", text)
            self.assertIn("--min-lr 1e-4", text)
        self.assertIn('default_warmup = math.ceil(shape["total_steps"] * 0.05)', self.trainer)

    def test_shared_store_staging_and_submission_topology(self) -> None:
        stage = STAGE.read_text(encoding="utf-8")
        for marker in (
            '/dev/shm/rsmol_variable_depth_train_${RUN_ID}',
            'SHM_MARGIN_KIB=$((10 * 1024 * 1024))',
            'cp -a "$SOURCE_STORE"/. "$STAGED_STORE"/',
            'rm -rf -- "$STAGED_STORE"',
            "train_audio_shared_store_5_10x2to10_5_mesh_mellow_ddp.py",
        ):
            self.assertIn(marker, stage)
        for submit in SUBMITS:
            text = submit.read_text(encoding="utf-8")
            for marker in ("vc submit", "-p pdgpu-5090", "-c 32 -m 256G -g 8 -n 1"):
                self.assertIn(marker, text)

    def test_checkpoint_contains_complete_resume_state(self) -> None:
        for marker in (
            '"optimizer": optimizer.state_dict()',
            '"scheduler": scheduler.state_dict()',
            '"rng_states_by_rank"',
            '"cursor": cursor',
            '"checkpoint_complete.json"',
            "base._restore_rng_state",
        ):
            self.assertIn(marker, self.trainer)


if __name__ == "__main__":
    unittest.main()
