from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "code/RSmol/recursive_model_5_10x2_5_mesh_7slot.py"
CONVERTER = ROOT / "code/RSmol/scripts/convert_stepwise_5_10x2_5_mesh_7slot.py"
TRAINER = ROOT / "code/RSmol/scripts/train_stage4_5_10x2_5_mesh_7slot_ddp.py"
TRAIN_SHELL = ROOT / "code/RSmol/scripts/train_stage4_5_10x2_5_mesh_7slot_ddp.sh"
SHELL_SCRIPTS = (
    ROOT / "code/RSmol/scripts/convert_stepwise_5_10x2_5_mesh_7slot.sh",
    TRAIN_SHELL,
    ROOT / "code/RSmol/run_stage4_5_10x2_5_mesh_7slot_smoke_3090.sh",
    ROOT / "code/RSmol/run_stage4_5_10x2_5_mesh_7slot_resume_3090.sh",
    ROOT / "code/RSmol/run_stage4_5_10x2_5_mesh_7slot_formal_3090.sh",
)


class TextMeshX2SevenSlotStaticTest(unittest.TestCase):
    def test_python_sources_parse(self) -> None:
        for path in (MODEL, CONVERTER, TRAINER):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_shell_scripts_are_strict_unix_scripts(self) -> None:
        for path in SHELL_SCRIPTS:
            raw = path.read_bytes()
            self.assertTrue(raw.startswith(b"#!"), path)
            self.assertIn(b"set -euo pipefail", raw, path)
            self.assertNotIn(b"\r", raw, path)
            self.assertNotIn(bytes((36, 123, 123)), raw, path)

    def test_model_is_x2_with_seven_slots_and_three_router_groups(self) -> None:
        source = MODEL.read_text(encoding="utf-8")
        for expected in (
            "LOGICAL_LAYER_COUNT = 30",
            "PHYSICAL_LAYER_COUNT = 20",
            "RECURSIVE_LOOPS = 2",
            "MEMORY_SLOT_COUNT = 7",
            "ROUTER_COUNT = 3",
            "ROUTER_PARAMETER_COUNT = 6",
            'MODEL_ARCHITECTURE_CONTRACT = "logical_30_physical_20_5_10x2_5_mesh_7slot_3router"',
            '"recursive_loops": 2, "memory_slots": 7, "router_count": 6',
            "LOGICAL_TO_PHYSICAL[15:25] != tuple(range(5, 15))",
        ):
            self.assertIn(expected, source)
        self.assertNotIn("MEMORY_SLOT_COUNT = 5", source)

    def test_converter_contract_and_router_initialization(self) -> None:
        source = CONVERTER.read_text(encoding="utf-8")
        for expected in (
            "recursive_model_5_10x2_5_mesh_7slot",
            "target.num_hidden_layers = 30",
            "target.recursive_loops = 2",
            '"logical_layer_count": 30',
            '"memory_slots": MEMORY_SLOT_COUNT, "router_groups": 3',
            "truncated_normal_std_sqrt_2_over_7d_clamped_3std_bias_zero",
            "SmolLM2-5-10x2-5-mesh-7slot",
        ):
            self.assertIn(expected, source)

    def test_training_contract_and_isolated_paths(self) -> None:
        source = TRAINER.read_text(encoding="utf-8")
        for expected in (
            "DEFAULT_WORLD_SIZE = 8",
            "DEFAULT_MICRO_BATCH_SIZE = 4",
            "DEFAULT_GRADIENT_ACCUMULATION_STEPS = 32",
            "DEFAULT_CONTEXT_LENGTH = 1024",
            "DEFAULT_STEPS_PER_EPOCH = 9_244",
            "DEFAULT_FORMAL_OPTIMIZER_STEPS = 3_081",
            "DEFAULT_FORMAL_WARMUP_STEPS = 155",
            "router_parameters_in_optimizer = len(router_names) == 12",
            'expected_names = {"write_pre", "read_pre", "write_0", "read_0", "write_1", "read_1"}',
            "recursive_model_5_10x2_5_mesh_7slot",
            "DistributedParquetStream",
            "torch.bfloat16",
            "broadcast_buffers=False",
            "_validate_resume_contract",
        ):
            self.assertIn(expected, source)
        self.assertNotIn("RSMOL_5_10X4_5_MESH", source)
        self.assertNotIn("x4 router contract", source)
        shell = TRAIN_SHELL.read_text(encoding="utf-8")
        for expected in (
            "RSMOL_5_10X2_5_MESH_7SLOT_",
            "MICRO=4",
            "GA=32",
            "MAX_STEPS=3081",
            "SCHEDULER=3081",
            "WARMUP=155",
            "STEPS_PER_EPOCH=9244",
            "EPOCHS=1",
        ):
            self.assertIn(expected, shell)
        self.assertNotIn("RSMOL_5_10X4_5_MESH", shell)

    def test_submission_wrappers_use_3090_and_isolated_contract(self) -> None:
        for path in SHELL_SCRIPTS[2:]:
            source = path.read_text(encoding="utf-8")
            self.assertIn("pdgpu-3090", source)
            self.assertIn("-c 32 -m 256G -g 8 -n 1", source)
            self.assertIn("5_10x2_5_mesh_7slot", source)
            self.assertNotIn("RSMOL_5_10X4_5_MESH", source)
        resume = SHELL_SCRIPTS[3].read_text(encoding="utf-8")
        self.assertIn("checkpoint_complete.json", resume)
        self.assertIn("training_state.pt", resume)
        self.assertIn("RSMOL_5_10X2_5_MESH_7SLOT_RESUME_FROM", resume)


if __name__ == "__main__":
    unittest.main()
