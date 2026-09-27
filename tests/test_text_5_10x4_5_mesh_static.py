from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "code/RSmol/recursive_model_5_10x4_5_mesh.py"
TRAINER = ROOT / "code/RSmol/scripts/train_stage4_5_10x4_5_mesh_ddp.py"
CONVERTER = ROOT / "code/RSmol/scripts/convert_stepwise_5_10x4_5_mesh.py"
AUDIT = ROOT / "code/RSmol/scripts/audit_stage1_5_10x4_5_mesh.py"
SHELL_SCRIPTS = (
    ROOT / "code/RSmol/run_audit_stage1_5_10x4_5_mesh_4090.sh",
    ROOT / "code/RSmol/run_stage4_5_10x4_5_mesh_smoke_4090.sh",
    ROOT / "code/RSmol/run_stage4_5_10x4_5_mesh_resume_4090.sh",
    ROOT / "code/RSmol/run_stage4_5_10x4_5_mesh_formal_4090.sh",
    ROOT / "code/RSmol/scripts/audit_stage1_5_10x4_5_mesh.sh",
    ROOT / "code/RSmol/scripts/convert_stepwise_5_10x4_5_mesh.sh",
    ROOT / "code/RSmol/scripts/stage_text_shared_store_5_10x4_5_mesh.sh",
    ROOT / "code/RSmol/scripts/train_stage4_5_10x4_5_mesh_ddp.sh",
)


class TextMeshX4StaticTest(unittest.TestCase):
    def test_python_sources_parse(self) -> None:
        for path in (MODEL, TRAINER, CONVERTER, AUDIT):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_shell_scripts_use_unix_line_endings(self) -> None:
        for path in SHELL_SCRIPTS:
            raw = path.read_bytes()
            self.assertNotIn(b"\r", raw, f"{path} contains CR/CRLF bytes that break Bash continuations")
            self.assertTrue(raw.startswith(b"#!"), f"{path} lacks a shebang")
            self.assertIn(b"set -euo pipefail", raw, f"{path} lacks strict Bash mode")

    def test_model_contract_is_isolated_x4(self) -> None:
        source = MODEL.read_text(encoding="utf-8")
        self.assertIn("LOGICAL_LAYER_COUNT = 50", source)
        self.assertIn("RECURSIVE_LOOPS = 4", source)
        self.assertIn("MEMORY_SLOT_COUNT = 7", source)
        self.assertIn("ROUTER_COUNT = 5", source)
        self.assertIn("ROUTER_PARAMETER_COUNT = 10", source)
        self.assertIn("tuple(range(5, 15)) * RECURSIVE_LOOPS", source)
        self.assertIn("range(15, 20), 45", source)

    def test_formal_training_contract(self) -> None:
        source = TRAINER.read_text(encoding="utf-8")
        self.assertIn("DEFAULT_STEPS_PER_EPOCH = 9_244", source)
        self.assertIn("DEFAULT_FORMAL_EPOCHS = 2", source)
        self.assertIn("DEFAULT_FORMAL_WARMUP_STEPS = 925", source)
        self.assertIn("DEFAULT_MAX_LR = 1e-3", source)
        self.assertIn("DEFAULT_MIN_LR = 1e-4", source)
        self.assertIn('relative_to(Path("/dev/shm"))', source)
        self.assertIn("reset_for_epoch", source)
        self.assertIn("rows_to_skip", source)
        self.assertIn("checkpoint_retention=DEFAULT_CHECKPOINT_RETENTION", source)
        self.assertIn("def _runtime_setup", source)
        self.assertIn("def _init_process_group", source)
        self.assertLess(source.index("model.to(device)"), source.index("_init_process_group(rank=rank"))
        self.assertIn("RSMOL_5_10X4_5_MESH_LOG_INTERVAL_STEPS", source)

        stage = (ROOT / "code/RSmol/scripts/stage_text_shared_store_5_10x4_5_mesh.sh").read_text(encoding="utf-8")
        self.assertIn("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", stage)
        self.assertIn('RSMOL_5_10X4_5_MESH_LOG_INTERVAL_STEPS:-1', stage)

    def test_converter_and_audit_match_x4(self) -> None:
        converter = CONVERTER.read_text(encoding="utf-8")
        audit = AUDIT.read_text(encoding="utf-8")
        self.assertIn("target.num_hidden_layers = 50", converter)
        self.assertIn("target.recursive_loops = 4", converter)
        self.assertIn('"logical_layer_count": 50', converter)
        self.assertIn('"loops": 4', converter)
        self.assertIn("truncated_normal_std_sqrt_2_over_7d_clamped_3std_bias_zero", converter)
        self.assertNotIn("truncated_normal_std_sqrt_2_over_5d_clamped_3std_bias_zero", converter)
        self.assertIn("list(range(50))", audit)
        self.assertIn("for loop in range(4)", audit)
        self.assertIn("ten_router_outputs", audit)
        self.assertIn("five_write_history_entries", audit)


if __name__ == "__main__":
    unittest.main()
