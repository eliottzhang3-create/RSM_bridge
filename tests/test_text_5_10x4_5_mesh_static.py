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
    ROOT / "code/RSmol/run_stage4_5_10x4_5_mesh_smoke_3090.sh",
    ROOT / "code/RSmol/run_stage4_5_10x4_5_mesh_resume_3090.sh",
    ROOT / "code/RSmol/run_stage4_5_10x4_5_mesh_formal_3090.sh",
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
    ROOT / "code/RSmol/scripts/audit_same_allocation_text_5_10x4_5_mesh.sh",
    ROOT / "code/RSmol/run_audit_same_allocation_text_5_10x4_5_mesh_3090.sh",
)


class TextMeshX4StaticTest(unittest.TestCase):
    def test_python_sources_parse(self) -> None:
        for path in (MODEL, TRAINER, CONVERTER, AUDIT, DDP_PREFLIGHT, NCCL_TRANSPORT):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_shell_scripts_use_unix_line_endings(self) -> None:
        for path in SHELL_SCRIPTS:
            raw = path.read_bytes()
            self.assertNotIn(b"\r", raw, f"{path} contains CR/CRLF bytes that break Bash continuations")
            self.assertNotIn(b"${{", raw, f"{path} contains an invalid double-brace Bash substitution")
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
        self.assertIn("DEFAULT_FORMAL_EPOCHS = 1", source)
        self.assertIn("DEFAULT_FORMAL_OPTIMIZER_STEPS = 3_081", source)
        self.assertIn("DEFAULT_FORMAL_WARMUP_STEPS = 155", source)
        self.assertIn("DEFAULT_MICRO_BATCH_SIZE = 4", source)
        self.assertIn("DEFAULT_GRADIENT_ACCUMULATION_STEPS = 32", source)
        self.assertIn("DEFAULT_MAX_LR = 1e-3", source)
        self.assertIn("DEFAULT_MIN_LR = 1e-4", source)
        self.assertIn("checkpoint_retention=DEFAULT_CHECKPOINT_RETENTION", source)
        self.assertIn("def _dist_setup", source)
        self.assertNotIn("def _init_process_group", source)
        self.assertNotIn("persistent_data_source", source)
        self.assertNotIn("stage_report", source)
        run_training = source[source.index("def run_training"):]
        self.assertLess(
            run_training.index("_dist_setup(config)"),
            run_training.index("\n        model.to(device)"),
        )
        self.assertNotIn('phase="nccl_warmup_start"', run_training)
        self.assertNotIn("warmup_work", run_training)
        startup_diagnostics = source[
            source.index("def _startup_diagnostics"):source.index("def _validate_router_stats")
        ]
        self.assertIn("all_gather_object", startup_diagnostics)
        self.assertIn("DistributedParquetStream", source)
        train_shell = (ROOT / "code/RSmol/scripts/train_stage4_5_10x4_5_mesh_ddp.sh").read_text(encoding="utf-8")
        self.assertIn('--data-dir "$DATA"', train_shell)
        self.assertIn("MAX_STEPS=3081", train_shell)
        self.assertIn("SCHEDULER=3081", train_shell)
        self.assertIn("WARMUP=155", train_shell)
        self.assertIn("STEPS_PER_EPOCH=9244", train_shell)
        self.assertIn("EPOCHS=1", train_shell)
        self.assertIn("MICRO=4", train_shell)
        self.assertIn("GA=32", train_shell)
        self.assertIn("RSMOL_5_10X4_5_MESH_LOG_INTERVAL_STEPS", train_shell)
        self.assertIn("_validate_resume_contract", source)
        self.assertIn("resume checkpoint manifest differs", source)
        self.assertNotIn("stage_text_shared_store_5_10x4_5_mesh.sh", train_shell)
        staged_diagnostic = (ROOT / "code/RSmol/scripts/stage_text_shared_store_5_10x4_5_mesh.sh").read_text(encoding="utf-8")
        self.assertIn("MAX_STEPS=3081", staged_diagnostic)
        self.assertIn("--scheduler-total-steps 3081", staged_diagnostic)
        self.assertNotIn("--persistent-data-source", staged_diagnostic)
        self.assertNotIn("--stage-report", staged_diagnostic)

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

    def test_same_allocation_audit_is_fail_fast_and_uses_3090(self) -> None:
        audit = (ROOT / "code/RSmol/scripts/audit_same_allocation_text_5_10x4_5_mesh.sh").read_text(encoding="utf-8")
        expected_order = (
            "01_model_free_nccl",
            "02_historical_x2_smoke",
            "04_copy_to_shared_memory",
            "06_post_stage_nccl",
            "07_x4_smoke",
        )
        positions = [audit.index(value) for value in expected_order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("timeout --signal=TERM", audit)
        self.assertIn("train_stage4_5_10x2_5_mesh_ddp.sh", audit)
        self.assertIn("train_stage4_5_10x4_5_mesh_ddp.py", audit)
        self.assertIn("audit_nccl_transport_5_10x4_5_mesh.sh", audit)
        self.assertIn("nvidia-smi topo -m", audit)
        self.assertIn("/dev/shm/rsmol_text_5_10x4_5_same_allocation_", audit)
        wrapper = (ROOT / "code/RSmol/run_audit_same_allocation_text_5_10x4_5_mesh_3090.sh").read_text(encoding="utf-8")
        self.assertIn("pdgpu-3090", wrapper)
        self.assertNotIn("pdgpu-4090", wrapper)

    def test_production_submission_wrappers_use_validated_3090_queue(self) -> None:
        wrappers = {
            "smoke": ROOT / "code/RSmol/run_stage4_5_10x4_5_mesh_smoke_3090.sh",
            "resume": ROOT / "code/RSmol/run_stage4_5_10x4_5_mesh_resume_3090.sh",
            "formal": ROOT / "code/RSmol/run_stage4_5_10x4_5_mesh_formal_3090.sh",
        }
        for mode, path in wrappers.items():
            source = path.read_text(encoding="utf-8")
            self.assertIn("pdgpu-3090", source)
            self.assertNotIn("pdgpu-4090", source)
            self.assertIn("-c 32 -m 256G -g 8 -n 1", source)
            self.assertIn("train_stage4_5_10x4_5_mesh_ddp.sh", source)
            self.assertNotIn("stage_text_shared_store_5_10x4_5_mesh.sh", source)
        resume = wrappers["resume"].read_text(encoding="utf-8")
        self.assertIn("checkpoint_complete.json", resume)
        self.assertIn("training_state.pt", resume)
        self.assertIn("RSMOL_5_10X4_5_MESH_RESUME_FROM", resume)
        formal = wrappers["formal"].read_text(encoding="utf-8")
        self.assertIn("formal_third_epoch_3081steps_20260927_3090_v1", formal)


if __name__ == "__main__":
    unittest.main()
