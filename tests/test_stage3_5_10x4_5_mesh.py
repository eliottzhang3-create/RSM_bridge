"""CPU-only checks for x4 architecture adaptation and x2 evaluation parity."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code/RSmol/scripts"
EVALUATOR = SCRIPTS / "evaluate_stage3_5_10x4_5_mesh.py"
REFERENCE = SCRIPTS / "evaluate_stage3_5_10x2_5_mesh.py"


def load_evaluator(path):
    spec = importlib.util.spec_from_file_location("_test_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MeshX4Stage3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator(EVALUATOR)
        cls.reference = load_evaluator(REFERENCE)

    def converted_config(self):
        # Exercise the actual conversion function so serialized router counts
        # and schedules cannot silently disagree with the new evaluator.
        tree = ast.parse((SCRIPTS / "convert_stepwise_5_10x4_5_mesh.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_target_config")
        scope = dict(copy=copy, LOGICAL_TO_PHYSICAL=self.evaluator.LOGICAL_TO_PHYSICAL,
                     SOURCE_LAYER_INDICES_0BASED=self.evaluator.SOURCE_MAPPING_0BASED,
                     MAPPING_POLICY=self.evaluator.MAPPING_POLICY, MEMORY_SLOT_COUNT=7, ROUTER_PARAMETER_COUNT=10)
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[])), "converter", "exec"), scope)
        config = scope["_target_config"](types.SimpleNamespace(model_type="llama", vocab_size=3), {"kind": "clean_source"}, Path("source"))
        return vars(config)

    def test_real_conversion_config_passes_and_other_mesh_variants_fail(self):
        config = self.converted_config()
        audit = self.evaluator._strict_config_contract(config)
        self.assertEqual((audit["logical_layer_count"], audit["memory_slots"], audit["router_count"]), (50, 7, 10))
        self.assertEqual(tuple(audit["logical_to_physical"]), tuple(range(5)) + tuple(range(5, 15)) * 4 + tuple(range(15, 20)))
        for key, value in (("num_hidden_layers", 30), ("recursive_loops", 2), ("mesh_memory_slots", 8), ("mesh_router_count", 6), ("logical_to_physical", list(range(50)))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.evaluator._strict_config_contract({**config, key: value})

    def test_defaults_match_x2_except_model_and_output_paths(self):
        config = self.evaluator.parse_args([])
        reference = self.reference.parse_args([])
        self.assertEqual(str(config.model_path).replace("\\", "/"),
                         "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x4_5_mesh/formal_2epochs_18488steps_20261007_004959_3090_v1/checkpoint-018488")
        for name in ("tasks", "benchmark_root", "device", "dtype", "batch_size", "seed", "limit", "max_new_tokens", "log_samples", "validation_only", "smoke"):
            self.assertEqual(getattr(config, name), getattr(reference, name), name)
        self.assertEqual(config.dtype, "bfloat16")
        self.assertEqual(config.batch_size, 1)
        for name in ("output_dir", "cache_dir", "log_root"):
            self.assertNotEqual(getattr(config, name), getattr(reference, name))

    def test_benchmark_execution_and_cli_are_unchanged_from_x2(self):
        trees = [ast.parse(path.read_text(encoding="utf-8")) for path in (EVALUATOR, REFERENCE)]
        for name in ("_run_single_task", "parse_args", "run_evaluation"):
            functions = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name) for tree in trees]
            # Only architecture-specific function names differ in orchestration.
            actual = ast.dump(functions[0], include_attributes=False).replace("5_10x4_5_mesh", "5_10x2_5_mesh")
            self.assertEqual(actual, ast.dump(functions[1], include_attributes=False), name)

    def test_artifact_metadata_uses_x4_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "config.json").write_text(json.dumps(self.converted_config()), encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"placeholder")
            (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.json").write_text('{"model":{"vocab":{"a":0,"b":1,"c":2}}}', encoding="utf-8")
            report = self.evaluator.inspect_model_artifacts_5_10x4_5_mesh(model)
            self.assertEqual(report["recursive_audit"]["logical_layer_count"], 50)
            metadata = {"architecture_contract": self.evaluator.ARCHITECTURE_CONTRACT, "memory_slots": 7, "router_parameter_count": 10}
            metadata_path = model / "mesh_checkpoint_metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertFalse(self.evaluator.inspect_model_artifacts_5_10x4_5_mesh(model)["warnings"])
            metadata_path.write_text(json.dumps({**metadata, "memory_slots": 5}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata contract mismatch"):
                self.evaluator.inspect_model_artifacts_5_10x4_5_mesh(model)

    def test_all_four_loop_router_queries_are_required(self):
        names = {"write_pre", "read_pre"} | {f"{kind}_{loop}" for kind in ("write", "read") for loop in range(4)}
        query = types.SimpleNamespace(shape=(1, 4, 576))
        backbone = types.SimpleNamespace(
            last_memory_shape=(1, 7, 4, 576), last_prefix_output=query,
            last_router_queries={name: query for name in names},
            last_router_weights={name: object() for name in names},
        )
        trace = [{"logical_index": i, "physical_index": p} for i, p in enumerate(self.evaluator.LOGICAL_TO_PHYSICAL)]
        audit = self.evaluator._audit_transition_and_memory(backbone, logical_trace=trace, expected_shape=[1, 7, 4, 576])
        self.assertEqual((audit["logical_steps"], len(audit["router_query_names"])), (50, 10))
        del backbone.last_router_queries["read_3"]
        with self.assertRaisesRegex(RuntimeError, "all 10 transition routers"):
            self.evaluator._audit_transition_and_memory(backbone, logical_trace=trace, expected_shape=[1, 7, 4, 576])

    def test_cache_audit_requires_all_50_logical_slots(self):
        tensor = types.SimpleNamespace(shape=(1, 3, 4, 64))
        class Cache:
            layers = [types.SimpleNamespace(keys=tensor, values=tensor) for _ in range(50)]
            def __len__(self):
                return len(self.layers)
            def get_seq_length(self, layer_idx):
                return 4
        cache = Cache()
        self.assertEqual(self.evaluator._cache_slot_audit(cache, expected_length=4)["logical_slot_count"], 50)
        cache.layers = cache.layers[:30]
        with self.assertRaisesRegex(RuntimeError, "capacity 30 < 50"):
            self.evaluator._cache_slot_audit(cache, expected_length=4)

    def test_lm_eval_receives_matching_tasks_fewshot_and_seeds(self):
        lm_eval = types.ModuleType("lm_eval")
        evaluator = types.ModuleType("lm_eval.evaluator")
        tasks = types.ModuleType("lm_eval.tasks")
        evaluator.simple_evaluate = Mock(return_value={"results": {}})
        tasks.TaskManager = Mock(return_value="task-manager")
        lm_eval.evaluator = evaluator
        lm_eval.tasks = tasks
        mesh = types.SimpleNamespace(register_auto_class=Mock())
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"lm_eval": lm_eval, "lm_eval.evaluator": evaluator, "lm_eval.tasks": tasks}), patch.object(self.evaluator, "_load_mesh", return_value=mesh), patch.object(self.evaluator, "set_offline_environment"), patch.object(self.evaluator, "write_json"):
            config = self.evaluator.parse_args([])
            for task in config.tasks:
                self.evaluator._run_single_task(config, task, Path(temporary), Path(temporary), Path(temporary) / "stderr.log")
                kwargs = evaluator.simple_evaluate.call_args.kwargs
                self.assertEqual(kwargs["tasks"], [task])
                self.assertEqual(kwargs["num_fewshot"], 5 if task in {"mmlu", "gsm8k"} else None)
                self.assertEqual((kwargs["model"], kwargs["device"], kwargs["batch_size"], kwargs["limit"]), ("hf", "cuda:0", 1, None))
                self.assertIn("dtype=bfloat16,local_files_only=True", kwargs["model_args"])
                self.assertIn("checkpoint-018488", kwargs["model_args"])
                for seed in ("random_seed", "numpy_random_seed", "torch_random_seed", "fewshot_random_seed"):
                    self.assertEqual(kwargs[seed], 0)
            self.assertEqual(mesh.register_auto_class.call_count, 5)

    def test_preflight_failure_still_writes_audit_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.evaluator.parse_args(["--output-dir", str(root / "result"), "--log-root", str(root / "logs"), "--cache-dir", str(root / "cache"), "--validation-only"])
            with patch.object(self.evaluator, "inspect_model_artifacts_5_10x4_5_mesh", side_effect=ValueError("wrong checkpoint")), patch.object(self.evaluator, "set_offline_environment"), self.assertRaisesRegex(ValueError, "wrong checkpoint"):
                self.evaluator.run_evaluation(config)
            report = json.loads((config.output_dir / "audit_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["formal_eval_executed"])
            self.assertIn("wrong checkpoint", report["traceback"])


if __name__ == "__main__":
    unittest.main()
