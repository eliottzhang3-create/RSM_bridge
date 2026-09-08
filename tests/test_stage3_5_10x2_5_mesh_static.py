"""Dependency-light static contracts for isolated MeSH Stage 3 evaluation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3_5_10x2_5_mesh.py"
RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3_5_10x2_5_mesh.sh"
SUBMIT = ROOT / "code" / "RSmol" / "run_stage3_eval_5_10x2_5_mesh_5090.sh"
MODEL = ROOT / "code" / "RSmol" / "recursive_model_5_10x2_5_mesh.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("stage3_5_10x2_5_mesh_static", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MeshStage3StaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_evaluator()
        cls.source = EVALUATOR.read_text(encoding="utf-8")
        cls.runtime = RUNTIME.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")
        cls.model_source = MODEL.read_text(encoding="utf-8")

    def _config(self):
        return {
            "model_type": "llama",
            "architectures": ["RecursiveLlama5_10x2_5MeshForCausalLM"],
            "vocab_size": 3,
            "num_hidden_layers": 30,
            "recursive_layer_count": 20,
            "recursive_loops": 2,
            "recursive_loops_scope": "middle_only",
            "recursive_prefix_layer_count": 5,
            "recursive_middle_layer_count": 10,
            "recursive_suffix_layer_count": 5,
            "mesh_memory_slots": 5,
            "mesh_router_count": 6,
            "mesh_transition_query": "prefix_output",
            "recursive_mapping_policy": "explicit_5_10_5_source_layers_mesh",
            "logical_to_physical": list(self.module.LOGICAL_TO_PHYSICAL),
            "logical_to_physical_schedule": list(self.module.LOGICAL_TO_PHYSICAL),
            "recursive_logical_to_physical": list(self.module.LOGICAL_TO_PHYSICAL),
            "recursive_logical_to_physical_schedule": list(self.module.LOGICAL_TO_PHYSICAL),
            "recursive_source_layer_indices_0based": list(self.module.SOURCE_MAPPING_0BASED),
            "recursive_source_layer_indices_1based": [i + 1 for i in self.module.SOURCE_MAPPING_0BASED],
        }

    def test_isolated_files_and_real_mesh_module(self):
        self.assertTrue(EVALUATOR.is_file())
        self.assertTrue(RUNTIME.is_file())
        self.assertTrue(SUBMIT.is_file())
        self.assertIn("recursive_model_5_10x2_5_mesh.py", self.source)
        self.assertIn('"code.RSmol.recursive_model_5_10x2_5_mesh"', self.source)
        for forbidden in ("recursive_model_5_10_5", "recursive_model_5_10xpoisson_parcae", "recursive_model_5_10xr_5"):
            self.assertNotIn(forbidden, self.source)
        self.assertNotIn("evaluate_stage3_5_10_5", self.source)

    def test_mesh_contract_constants_and_schedule(self):
        self.assertEqual(self.module.LOGICAL_LAYER_COUNT, 30)
        self.assertEqual(self.module.PHYSICAL_LAYER_COUNT, 20)
        self.assertEqual(self.module.RECURSIVE_LOOPS, 2)
        self.assertEqual(self.module.MEMORY_SLOT_COUNT, 5)
        self.assertEqual(self.module.ROUTER_COUNT, 6)
        self.assertEqual(self.module.TRANSITION_QUERY, "prefix_output")
        self.assertEqual(tuple(self.module.LOGICAL_TO_PHYSICAL), (tuple(range(5)) + tuple(range(5, 15)) + tuple(range(5, 15)) + tuple(range(15, 20))))
        self.assertEqual(len(self.module.SOURCE_MAPPING_0BASED), 20)
        self.assertEqual(self.module.ARCHITECTURE_CONTRACT, "logical_30_physical_20_5_10x2_5_mesh")

    def test_artifact_contract_and_optional_metadata_warning(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary).resolve()
            (model / "config.json").write_text(json.dumps(self._config()), encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"placeholder")
            (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.json").write_text('{"model":{"vocab":{"a":0,"b":1,"c":2}}}', encoding="utf-8")
            report = self.module.inspect_model_artifacts_5_10x2_5_mesh(model)
            self.assertEqual(report["recursive_audit"]["logical_layer_count"], 30)
            self.assertEqual(report["recursive_audit"]["physical_layer_count"], 20)
            self.assertTrue(report["warnings"])
            self.assertIn("metadata audit is limited", report["warnings"][0])

    def test_wrong_schedule_is_hard_failure(self):
        config = self._config()
        config["logical_to_physical"] = list(range(30))
        with self.assertRaisesRegex(ValueError, "logical_to_physical"):
            self.module._strict_config_contract(config)

    def test_router_collapse_is_warning_only_and_report_contract_is_present(self):
        self.assertIn("router collapse/slot imbalance diagnostic (non-fatal)", self.source)
        # The backbone must be resolved before router_modules is accessed.
        # This catches the runtime UnboundLocalError that otherwise appears
        # only after loading a real CUDA checkpoint.
        runtime_source = self.source.split("def _runtime_audit(", 1)[1].split(
            "def recursive_runtime_audit_5_10x2_5_mesh", 1
        )[0]
        backbone_pos = runtime_source.index('recursive_model = getattr(model, "model", model)')
        router_pos = runtime_source.index("router_modules = list(getattr(recursive_model", backbone_pos)
        self.assertLess(backbone_pos, router_pos)
        self.assertIn('"checks": checks', self.source)
        self.assertIn('"warnings": warnings', self.source)
        self.assertIn('"hard_failures": hard_failures', self.source)
        self.assertIn('"traceback": traceback_text', self.source)
        self.assertIn('"runtime_audit": runtime_audit', self.source)
        self.assertIn('"benchmark_outputs": benchmark_outputs', self.source)

    def test_runtime_and_submit_environment_contract(self):
        for marker in (
            "RSMOL_STAGE3_5_10X2_5_MESH_MODEL",
            "RSMOL_STAGE3_5_10X2_5_MESH_BENCHMARK_ROOT",
            "RSMOL_STAGE3_5_10X2_5_MESH_OUTPUT_DIR",
            "RSMOL_STAGE3_5_10X2_5_MESH_REPORT_PATH",
            "RSMOL_STAGE3_5_10X2_5_MESH_DEVICE",
            "RSMOL_STAGE3_5_10X2_5_MESH_TASKS",
            "RSMOL_STAGE3_5_10X2_5_MESH_MAX_NEW_TOKENS",
            "evaluate_stage3_5_10x2_5_mesh.py",
        ):
            self.assertIn(marker, self.runtime + self.submit)
        self.assertIn("vc submit", self.submit)
        self.assertIn("-p pdgpu-5090", self.submit)
        self.assertIn("bash scripts/evaluate_stage3_5_10x2_5_mesh.sh", self.submit)
        self.assertNotIn("evaluate_stage3.py", self.submit)

    def test_defaults_and_flags(self):
        self.assertEqual(tuple(self.module.STAGE3_TASKS), ("hellaswag", "mmlu", "gsm8k", "arc_easy", "arc_challenge"))
        self.assertEqual(self.module.DEFAULT_MODEL.name, "checkpoint-009244")
        config = self.module.parse_args(["--output-dir", "/tmp/mesh-eval", "--validation-only", "--smoke", "--tasks", "mmlu"])
        self.assertEqual(config.tasks, ("mmlu",))
        self.assertEqual(config.limit, 2)
        self.assertEqual(config.dtype, "bfloat16")
        self.assertEqual(config.max_new_tokens, 2)


if __name__ == "__main__":
    unittest.main()
