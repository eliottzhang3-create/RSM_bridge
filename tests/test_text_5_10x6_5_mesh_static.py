from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "code/RSmol/recursive_model_5_10x6_5_mesh.py"
TRAINER = ROOT / "code/RSmol/scripts/train_stage4_5_10x6_5_mesh_ddp.py"
CONVERTER = ROOT / "code/RSmol/scripts/convert_stepwise_5_10x6_5_mesh.py"
AUDIT = ROOT / "code/RSmol/scripts/audit_stage1_5_10x6_5_mesh.py"
TRAIN_SHELL = ROOT / "code/RSmol/scripts/train_stage4_5_10x6_5_mesh_ddp.sh"
SHELL_SCRIPTS = (
    ROOT / "code/RSmol/scripts/convert_stepwise_5_10x6_5_mesh.sh",
    ROOT / "code/RSmol/scripts/audit_stage1_5_10x6_5_mesh.sh",
    TRAIN_SHELL,
    ROOT / "code/RSmol/run_audit_stage1_5_10x6_5_mesh_3090.sh",
    ROOT / "code/RSmol/run_stage4_5_10x6_5_mesh_smoke_3090.sh",
    ROOT / "code/RSmol/run_stage4_5_10x6_5_mesh_resume_3090.sh",
    ROOT / "code/RSmol/run_stage4_5_10x6_5_mesh_formal_3090.sh",
)


class TextMeshX6StaticTest(unittest.TestCase):
    def test_python_sources_parse(self) -> None:
        for path in (MODEL, TRAINER, CONVERTER, AUDIT):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_shell_scripts_are_strict_unix_scripts(self) -> None:
        invalid_double_brace = bytes((36, 123, 123))
        for path in SHELL_SCRIPTS:
            raw = path.read_bytes()
            self.assertTrue(raw.startswith(b"#!"), path)
            self.assertIn(b"set -euo pipefail", raw, path)
            self.assertNotIn(b"\r", raw, path)
            self.assertNotIn(invalid_double_brace, raw, path)

    def test_model_contract(self) -> None:
        source = MODEL.read_text(encoding="utf-8")
        for expected in (
            "LOGICAL_LAYER_COUNT = 70",
            "PHYSICAL_LAYER_COUNT = 20",
            "RECURSIVE_LOOPS = 6",
            "MEMORY_SLOT_COUNT = 7",
            "ROUTER_COUNT = 7",
            "ROUTER_PARAMETER_COUNT = 14",
            '"recursive_loops": 6, "memory_slots": 7, "router_count": 14',
            'MODEL_ARCHITECTURE_CONTRACT = "logical_70_physical_20_5_10x6_5_mesh"',
            "tuple(range(5, 15)) * RECURSIVE_LOOPS",
            "range(15, 20), 65",
            "!= (70, 20, 6)",
        ):
            self.assertIn(expected, source)
        self.assertIn('{"0", "1", "2", "3", "4", "5"}', source)
        self.assertNotIn("logical_50_physical_20_5_10x6_5_mesh", source)

    def test_converter_contract(self) -> None:
        source = CONVERTER.read_text(encoding="utf-8")
        for expected in (
            "target.num_hidden_layers = 70",
            "target.recursive_loops = 6",
            'target.mesh_architecture_contract = "logical_70_physical_20_5_10x6_5"',
            "truncated_normal_std_sqrt_2_over_7d_clamped_3std_bias_zero",
            '"logical_layer_count": 70',
            '"loops": 6',
            "RecursiveLlama5_10x6_5MeshForCausalLM",
        ):
            self.assertIn(expected, source)
        self.assertIn("SmolLM2-5-10-5", source)
        self.assertIn("SmolLM2-5-10x6-5-mesh", source)

    def test_stage1_audits_all_six_transitions(self) -> None:
        source = AUDIT.read_text(encoding="utf-8")
        self.assertIn("list(range(70))", source)
        self.assertIn("for loop in range(6)", source)
        self.assertIn("fourteen_router_outputs", source)
        self.assertIn("seven_write_history_entries", source)
        self.assertIn("fourteen_router_gradients", source)
        for suffix in range(6):
            self.assertIn(f'"write_{suffix}"', source)
            self.assertIn(f'"read_{suffix}"', source)

    def test_training_matches_successful_x4_geometry(self) -> None:
        source = TRAINER.read_text(encoding="utf-8")
        for expected in (
            "DEFAULT_WORLD_SIZE = 8",
            "DEFAULT_MICRO_BATCH_SIZE = 4",
            "DEFAULT_GRADIENT_ACCUMULATION_STEPS = 32",
            "DEFAULT_CONTEXT_LENGTH = 1024",
            "DEFAULT_STEPS_PER_EPOCH = 9_244",
            "DEFAULT_FORMAL_OPTIMIZER_STEPS = 3_081",
            "DEFAULT_FORMAL_WARMUP_STEPS = 155",
            "DEFAULT_MAX_LR = 1e-3",
            "DEFAULT_MIN_LR = 1e-4",
            "DistributedParquetStream",
            "resume_row_offset = self.row_offset",
            "batch = batch.slice(resume_row_offset - batch_start)",
            "resume cursor policy mismatch",
            "torch.bfloat16",
            "broadcast_buffers=False",
            "checkpoint_retention=DEFAULT_CHECKPOINT_RETENTION",
            "ROUTER_PARAMETER_COUNT",
            "expected_router_parameter_tensors = ROUTER_PARAMETER_COUNT * 2",
            '"write_4", "read_4"',
            '"write_5", "read_5"',
            "fourteen router statistics were not produced",
            "discard_accumulation_window_retry_same_optimizer_step_with_next_data",
            "nonfinite_window_skipped",
            "nonfinite_windows_skipped",
            "torch.cuda.get_rng_state",
            "torch.cuda.set_rng_state",
        ):
            self.assertIn(expected, source)
        self.assertNotIn("requires 20 router tensors", source)
        self.assertIn("recursive_model_5_10x6_5_mesh", source)
        self.assertNotIn("stage_text_shared_store_5_10x6_5_mesh.sh", source)
        shell = TRAIN_SHELL.read_text(encoding="utf-8")
        for expected in (
            "MICRO=4",
            "GA=32",
            "MAX_STEPS=3081",
            "SCHEDULER=3081",
            "WARMUP=155",
            "STEPS_PER_EPOCH=9244",
            "--save-every 500",
            "RSMOL_5_10X6_5_MESH_",
        ):
            self.assertIn(expected, shell)

    def test_submission_wrappers_are_isolated_and_use_3090(self) -> None:
        for path in SHELL_SCRIPTS[3:]:
            source = path.read_text(encoding="utf-8")
            self.assertIn("pdgpu-3090", source)
            self.assertIn("5_10x6_5", source)
            self.assertIn("RSMOL_5_10X6_5_MESH", source)
            self.assertNotIn("RSMOL_5_10X4_5_MESH", source)
        resume = SHELL_SCRIPTS[5].read_text(encoding="utf-8")
        self.assertIn("checkpoint_complete.json", resume)
        self.assertIn("training_state.pt", resume)
        formal = SHELL_SCRIPTS[6].read_text(encoding="utf-8")
        for expected in (
            "RSMOL_5_10X6_5_MESH_RESUME_FROM",
            "checkpoint_complete.json",
            "checkpoint_manifest.json",
            "training_state.pt",
            "formal resume output path must be new and absent",
            "RSMOL_5_10X6_5_MESH_STAGE4_GATE=FORMAL",
            'RESUME_ENV=" RSMOL_5_10X6_5_MESH_RESUME_FROM=$RESUME_Q"',
        ):
            self.assertIn(expected, formal)
        self.assertIn("does not read or require either report", formal)
        self.assertNotIn("run_stage4_5_10x6_5_mesh_smoke_3090.sh", formal)
        self.assertNotIn("run_stage4_5_10x6_5_mesh_resume_3090.sh", formal)


if __name__ == "__main__":
    unittest.main()
