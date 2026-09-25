"""Dependency-light contracts for the isolated T=2..10 Audio MeSH phases 1--4."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"
MODEL = RSMOL / "recursive_model_5_10x2to10_5_mesh.py"
PACKAGE = RSMOL / "audio_5_10x2to10_5_mesh_mellow_shared_store"
CONVERTER = RSMOL / "scripts" / "convert_audio_shared_store_5_10x2to10_5_mesh.py"
AUDIT = RSMOL / "scripts" / "audit_audio_5_10x2to10_5_mesh_structure.py"
PREFLIGHT = RSMOL / "scripts" / "preflight_audio_5_10x2to10_5_mesh_t10_activation.py"
PREFLIGHT_SHELL = RSMOL / "scripts" / "preflight_audio_5_10x2to10_5_mesh_t10_activation.sh"
SUBMIT = RSMOL / "run_audio_5_10x2to10_5_mesh_t10_preflight_5090.sh"
FIXED_MODEL = RSMOL / "recursive_model_5_10x2_5_mesh.py"


class VariableDepthAudioMeshStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = MODEL.read_text(encoding="utf-8")
        cls.package_model = (PACKAGE / "model.py").read_text(encoding="utf-8")
        cls.depth_sampling = (PACKAGE / "depth_sampling.py").read_text(encoding="utf-8")
        cls.converter = CONVERTER.read_text(encoding="utf-8")
        cls.audit = AUDIT.read_text(encoding="utf-8")
        cls.preflight = PREFLIGHT.read_text(encoding="utf-8")

    def test_new_route_is_isolated_and_python_is_parseable(self) -> None:
        paths = [
            MODEL, PACKAGE / "__init__.py", PACKAGE / "model.py", PACKAGE / "data.py",
            PACKAGE / "depth_sampling.py", CONVERTER, AUDIT, PREFLIGHT,
            PREFLIGHT_SHELL, SUBMIT,
        ]
        for path in paths:
            self.assertTrue(path.is_file(), path)
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"))
        self.assertIn("for loop in range(2):", FIXED_MODEL.read_text(encoding="utf-8"))
        self.assertNotIn("2to10", FIXED_MODEL.read_text(encoding="utf-8"))

    def test_depth_and_router_contract(self) -> None:
        for marker in (
            "MIN_RECURSIVE_DEPTH = 2", "MAX_RECURSIVE_DEPTH = 10",
            "MIN_LOGICAL_LAYER_COUNT = 30", "MAX_LOGICAL_LAYER_COUNT = 110",
            "self.pre_write", "self.pre_read", "self.loop1_write", "self.loop1_read",
            "self.refine_write", "self.refine_read", "self.out_read",
            "suffix_start = PREFIX_LAYER_COUNT + depth * MIDDLE_LAYER_COUNT",
        ):
            self.assertIn(marker, self.model)
        self.assertIn("if loop_index == depth - 1:", self.model)
        self.assertIn('read_router, read_name = self.out_read, "out_read"', self.model)
        self.assertIn('read_router, read_name = self.refine_read, "refine_read"', self.model)

    def test_audio_and_sampler_keep_one_depth_per_microstep(self) -> None:
        self.assertIn("def forward(self, *, recursive_depth: int", self.package_model)
        self.assertIn("self.mesh_model.set_recursive_depth(depth)", self.package_model)
        self.assertIn("dist.broadcast(depth, src=0)", self.depth_sampling)
        self.assertIn("torch.Generator(device=\"cpu\")", self.depth_sampling)
        self.assertIn("generator_state", self.depth_sampling)
        init_text = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("DEFAULT_EPOCHS = 7", init_text)
        self.assertIn("DEFAULT_WARMUP_RATIO = 0.05", init_text)
        self.assertIn("RECURSIVE_DEPTHS = tuple(range(2, 11))", init_text)

    def test_checkpoint_migration_is_explicit_and_t2_exact(self) -> None:
        for marker in (
            '"model.write_routers.0.": "model.pre_write."',
            '"model.read_routers.0.": "model.pre_read."',
            '"model.write_routers.1.": "model.loop1_write."',
            '"model.read_routers.1.": "model.loop1_read."',
            '"model.write_routers.2.": "model.refine_write."',
            '"model.read_routers.2.": "model.out_read."',
            'source_key = f"model.read_routers.1.{suffix}"',
            'destination = f"model.refine_read.{suffix}"',
            "recursive_depth=2", "T=2 migration parity failed",
            '"not_copied": ["optimizer", "scheduler", "training cursor", "rank RNG"]',
        ):
            self.assertIn(marker, self.converter)

    def test_structure_and_gradient_audit_is_automatic(self) -> None:
        self.assertIn("for depth in range(2, 11):", self.audit)
        self.assertIn("for depth in (2, 3, 6, 10)", self.audit)
        self.assertIn('"refine_read": max(0, depth - 2)', self.audit)
        self.assertIn("all_loop_boundaries_have_gradients", self.audit)

    def test_t10_preflight_uses_exact_training_pressure_and_safe_resources(self) -> None:
        for marker in (
            '"recursive_depth": 10', "recursive_depth=10", "torch.bfloat16",
            "torch_dtype=torch.float32",
            "AdamW", "gradient_accumulation_steps", "find_unused_parameters=True",
            "torch.cuda.reset_peak_memory_stats", "peak_reserved_ratio",
            "torch.cuda.OutOfMemoryError", "(batch_size, 379)", "320000",
            '"sequence_length": 639',
        ):
            self.assertIn(marker, self.preflight)
        submit = SUBMIT.read_text(encoding="utf-8")
        for marker in ("-p pdgpu-5090", "-c 8 -m 32G -g 1 -n 1", "vc submit"):
            self.assertIn(marker, submit)
        shell = PREFLIGHT_SHELL.read_text(encoding="utf-8")
        self.assertIn('USER_CONDA_BASE="${USER_CONDA_BASE:-/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3}"', shell)


if __name__ == "__main__":
    unittest.main()
