from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "code/RSmol/recursive_model_5_10x4_5_mesh.py"
TRAINER = ROOT / "code/RSmol/scripts/train_stage4_5_10x4_5_mesh_ddp.py"
CONVERTER = ROOT / "code/RSmol/scripts/convert_stepwise_5_10x4_5_mesh.py"
AUDIT = ROOT / "code/RSmol/scripts/audit_stage1_5_10x4_5_mesh.py"
DDP_PREFLIGHT = ROOT / "code/RSmol/scripts/audit_ddp_preflight_5_10x4_5_mesh.py"
NCCL_TRANSPORT = ROOT / "code/RSmol/scripts/audit_nccl_transport_5_10x4_5_mesh.py"
SHELL_SCRIPTS = (
    ROOT / "code/RSmol/run_audit_stage1_5_10x4_5_mesh_4090.sh",
    ROOT / "code/RSmol/run_stage4_5_10x4_5_mesh_smoke_4090.sh",
    ROOT / "code/RSmol/run_stage4_5_10x4_5_mesh_resume_4090.sh",
    ROOT / "code/RSmol/run_stage4_5_10x4_5_mesh_formal_4090.sh",
    ROOT / "code/RSmol/scripts/audit_stage1_5_10x4_5_mesh.sh",
    ROOT / "code/RSmol/scripts/convert_stepwise_5_10x4_5_mesh.sh",
    ROOT / "code/RSmol/scripts/stage_text_shared_store_5_10x4_5_mesh.sh",
    ROOT / "code/RSmol/scripts/train_stage4_5_10x4_5_mesh_ddp.sh",
    ROOT / "code/RSmol/scripts/audit_ddp_preflight_5_10x4_5_mesh.sh",
    ROOT / "code/RSmol/run_audit_ddp_preflight_5_10x4_5_mesh_4090.sh",
    ROOT / "code/RSmol/scripts/audit_nccl_transport_5_10x4_5_mesh.sh",
    ROOT / "code/RSmol/run_audit_nccl_transport_5_10x4_5_mesh_4090.sh",
    ROOT / "code/RSmol/scripts/audit_post_stage_nccl_5_10x4_5_mesh.sh",
    ROOT / "code/RSmol/run_audit_post_stage_nccl_5_10x4_5_mesh_4090.sh",
)


class TextMeshX4StaticTest(unittest.TestCase):
    def test_python_sources_parse(self) -> None:
        for path in (MODEL, TRAINER, CONVERTER, AUDIT, DDP_PREFLIGHT, NCCL_TRANSPORT):
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
        run_training = source[source.index("def run_training"):]
        self.assertLess(
            run_training.index('phase="model_loaded_cpu"'),
            run_training.index("_init_process_group(rank=rank"),
        )
        self.assertLess(
            run_training.index("_init_process_group(rank=rank"),
            run_training.index('phase="nccl_warmup_start"'),
        )
        self.assertLess(
            run_training.index('phase="nccl_warmup_pass"'),
            run_training.index("\n        model.to(device)"),
        )
        self.assertIn("NCCL communicator creation is lazy", source)
        self.assertIn("warmup_work.wait()", source)
        self.assertIn("RSMOL_5_10X4_5_MESH_LOG_INTERVAL_STEPS", source)
        startup_diagnostics = source[
            source.index("def _startup_diagnostics"):source.index("def _validate_router_stats")
        ]
        self.assertNotIn("all_gather_object", startup_diagnostics)
        self.assertIn('print(f"[startup][rank={rank}] phase={phase}"', source)
        self.assertIn('phase="process_group_init_start"', source)
        self.assertIn('phase="process_group_initialized"', source)
        self.assertIn('phase="ddp_init_start"', source)

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

    def test_ddp_preflight_is_standard_ab_comparison(self) -> None:
        source = DDP_PREFLIGHT.read_text(encoding="utf-8")
        self.assertIn('choices=("x2", "x4")', source)
        self.assertIn("output_loading_info=True", source)
        self.assertIn("parameter_fingerprint", source)
        self.assertIn("dist.all_reduce", source)
        self.assertIn("dist.broadcast", source)
        self.assertIn("async_op=True", source)
        self.assertIn("work.wait()", source)
        self.assertIn("torch.cuda.synchronize(device)", source)
        self.assertIn("device_uuid", source)
        self.assertLess(source.index('name="parameter_contract_ready"'), source.index('name="process_group_init_start"'))
        self.assertLess(source.index('name="process_group_init_start"'), source.index('name="pre_model_nccl_warmup_start"'))
        self.assertLess(source.index('name="pre_model_nccl_warmup_pass"'), source.index("\n        model.to(device)"))
        self.assertIn("ddp = DDP(", source)
        self.assertIn('"init_sync": True', source)
        self.assertNotIn("init_sync=False", source)
        shell = (ROOT / "code/RSmol/scripts/audit_ddp_preflight_5_10x4_5_mesh.sh").read_text(encoding="utf-8")
        self.assertNotIn("NCCL_P2P_DISABLE", shell)
        self.assertNotIn("NCCL_CUMEM_ENABLE", shell)
        wrapper = (ROOT / "code/RSmol/run_audit_ddp_preflight_5_10x4_5_mesh_4090.sh").read_text(encoding="utf-8")
        self.assertIn("pdgpu-4090", wrapper)

    def test_post_stage_nccl_audit_preserves_isolation(self) -> None:
        stage = (ROOT / "code/RSmol/scripts/audit_post_stage_nccl_5_10x4_5_mesh.sh").read_text(encoding="utf-8")
        self.assertIn("/dev/shm/rsmol_text_5_10x4_5_post_stage_audit_", stage)
        self.assertIn("audit_text_parquet_store_5_10x4_5_mesh.py", stage)
        self.assertIn("shm_before_stage.txt", stage)
        self.assertIn("shm_after_stage.txt", stage)
        self.assertIn("audit_nccl_transport_5_10x4_5_mesh.sh" , stage)
        self.assertIn("baseline", stage)
        self.assertNotIn("train_stage4_5_10x4_5_mesh_ddp.py", stage)
        wrapper = (ROOT / "code/RSmol/run_audit_post_stage_nccl_5_10x4_5_mesh_4090.sh").read_text(encoding="utf-8")
        self.assertIn("pdgpu-4090", wrapper)


if __name__ == "__main__":
    unittest.main()
