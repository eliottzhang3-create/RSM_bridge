"""Exercise x4 loading, recipe, trace and resume guards without GPU dependencies."""
import ast
import copy
import io
import json
import math
from pathlib import Path, PurePosixPath
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs128_3epochs_recursive_mesh_5_10x4_5_ced_small"
sys.path.insert(0, str(ROUTE))
import audit_reference as audit
import contract
import write_config


def function(path, name, scope):
    audit.execute([audit.find_function(audit.tree(path), name)], scope)
    return scope[name]


def checkpoint_config():
    return dict(hidden_size=576, num_hidden_layers=50, recursive_layer_count=20,
                recursive_loops=4, mesh_memory_slots=7, mesh_router_count=10,
                mesh_architecture_contract=contract.TEXT_MODEL_CONTRACT,
                model_architecture_contract=contract.TEXT_MODEL_ARCHITECTURE,
                logical_to_physical=list(contract.TEXT_MODEL_SCHEDULE),
                vocab_size=49152, eos_token_id=0)


class BehaviorTests(unittest.TestCase):
    def test_entire_parent_recipe_update_loop_and_original_sources(self):
        result = audit.audit()
        self.assertEqual(result["status"], "PASS")
        self.assertGreaterEqual(result["protected_original_files"], 53)
        self.assertEqual(len(result["recipe"]["changed_fields"]), 7)
        self.assertIn("audio_encoder", result["recipe"]["identical_fields"])
        self.assertIn("mapper", result["recipe"]["identical_fields"])

    def test_x4_checkpoint_contract_rejects_x2_wrong_schedule_and_missing_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"synthetic")
            (root / "tokenizer.json").write_text("{}")
            original = checkpoint_config()
            contract.write_json(root / "config.json", original)
            result = contract.recursive_checkpoint_contract(root)
            self.assertEqual((result["config"]["logical_layers"], result["config"]["memory_slots"],
                              result["config"]["routers"]), (50, 7, 10))
            for field, value in (("num_hidden_layers", 30), ("recursive_loops", 2),
                                 ("mesh_memory_slots", 5), ("mesh_router_count", 6),
                                 ("logical_to_physical", list(range(50))),
                                 ("mesh_architecture_contract", "other")):
                with self.subTest(field=field):
                    contract.write_json(root / "config.json", dict(original, **{field: value}))
                    with self.assertRaises(ValueError):
                        contract.recursive_checkpoint_contract(root)
            contract.write_json(root / "config.json", original)
            (root / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "no recursive model weight"):
                contract.recursive_checkpoint_contract(root)

    def test_strict_x4_loader_executes_actual_rejection_paths(self):
        config = SimpleNamespace(**checkpoint_config())
        shape = SimpleNamespace(weight=SimpleNamespace(shape=(49152, 576)))
        model = SimpleNamespace(config=config, model=SimpleNamespace(ablate_after_two_loops=False),
                                parameters=lambda: [], get_input_embeddings=lambda: shape,
                                get_output_embeddings=lambda: shape)
        report = dict(logical_layer_count=50, physical_layer_count=20, recursive_loops=4,
                      logical_cache_slot_count=50, memory_slots=7, router_count=10,
                      schedule=list(contract.TEXT_MODEL_SCHEDULE), router_objects_independent=True,
                      memory_persistent=False)
        loader = Mock(return_value=(model, {}))
        register = Mock()
        scope = dict(Path=Path, Any=object, torch=SimpleNamespace(float32="fp32"),
                     TEXT_MODEL_ARCHITECTURE=contract.TEXT_MODEL_ARCHITECTURE,
                     TEXT_MODEL_CONTRACT=contract.TEXT_MODEL_CONTRACT,
                     TEXT_MODEL_SCHEDULE=contract.TEXT_MODEL_SCHEDULE,
                     _AUTO_CLASS_REGISTERED=False,
                     _import_recursive_model=lambda: (SimpleNamespace(from_pretrained=loader), lambda _: report, register))
        source = ROUTE / "models/recursive_text.py"
        function(source, "validate_recursive_config", scope)
        load = function(source, "load_recursive_text_model", scope)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text("{}")
            self.assertIs(load(directory), model)
            load(directory)
            register.assert_called_once()
            self.assertEqual(loader.call_args.kwargs, dict(local_files_only=True,
                                                          torch_dtype="fp32", output_loading_info=True))
            for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
                loader.return_value = (model, {field: ["bad"]})
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "did not load completely"):
                    load(directory)
            loader.return_value = (model, {})
            for field, bad in (("memory_slots", 5), ("router_count", 6), ("recursive_loops", 2),
                               ("router_objects_independent", False), ("memory_persistent", True)):
                previous = report[field]
                report[field] = bad
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load(directory)
                report[field] = previous
            model.model.ablate_after_two_loops = True
            with self.assertRaisesRegex(ValueError, "all four loops"):
                load(directory)

    def test_actual_forward_trace_audit_rejects_partial_execution_or_shared_routers(self):
        inspect = function(ROUTE / "train.py", "recursive_execution_audit",
                           {"TEXT_MODEL_SCHEDULE": contract.TEXT_MODEL_SCHEDULE})
        core = SimpleNamespace(last_forward_trace=[dict(logical_index=i, physical_index=p)
                               for i, p in enumerate(contract.TEXT_MODEL_SCHEDULE)],
                               layers=list(range(20)), recursive_loops=4, memory_slots=7,
                               write_routers=[object() for _ in range(5)],
                               read_routers=[object() for _ in range(5)], ablate_after_two_loops=False)
        model = SimpleNamespace(caption_decoder=SimpleNamespace(lm=SimpleNamespace(model=core)))
        original = copy.copy(core.__dict__)
        self.assertTrue(inspect(model)["passed"])
        for key, bad in (("last_forward_trace", core.last_forward_trace[:30]),
                         ("memory_slots", 5), ("recursive_loops", 2),
                         ("ablate_after_two_loops", True),
                         ("read_routers", core.write_routers)):
            with self.subTest(key=key):
                setattr(core, key, bad)
                self.assertFalse(inspect(model)["passed"])
                setattr(core, key, original[key])

    def test_gbs128_config_checkpoint_steps_and_complete_horizon_lr(self):
        geo = contract.geometry(sum(contract.SOURCE_COUNTS.values()))
        self.assertEqual((geo["total_steps"], geo["warmup_steps"]), (34626, 1731))
        for step in range(34627):
            expected = (max(.1, (step + 1) / 1731) if step < 1731 else
                        .1 + .9 * .5 * (1 + math.cos(math.pi * (step - 1731) / (34626 - 1731))))
            self.assertAlmostEqual(contract.lr_factor(step, 34626), expected, places=15)
        prepared = dict(mode="formal", rows=1477492, source_rows=contract.SOURCE_COUNTS,
                        rows_canonical_sha256="rows", mapping_sha256="mapping")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, output = root / "prepared.json", root / "runtime.json"
            contract.write_json(report, prepared)
            args = ["write_config.py", "--prepared", str(report), "--output", str(output),
                    "--mode", "formal", "--run-dir", str(root), "--text-model-dir", str(root),
                    "--ced-model-dir", str(root)]
            with patch.object(sys, "argv", args), patch.object(write_config, "StagingGate", return_value=SimpleNamespace(info=None)), \
                 patch.object(write_config, "recursive_checkpoint_contract", return_value={"fixture": True}), \
                 patch.object(write_config, "asset_identity", return_value={"fixture": True}), \
                 patch.object(write_config.importlib.metadata, "version", return_value="fixture"), \
                 patch("sys.stdout", new=io.StringIO()):
                write_config.main()
            result = json.loads(output.read_text())
        self.assertEqual(result["checkpoint_steps"], [5771, 11542, 17313, 23084, 28855, 34626])
        self.assertEqual(result["max_optimizer_steps"], 34626)
        self.assertEqual(result["contract_sha256"], contract.digest(result["contract"]))

    def test_completion_check_executes_dynamic_steps_and_rejects_old_step(self):
        source = (ROUTE / "run.sh").read_text(encoding="utf-8")
        check = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            for status, steps, accepted in (("PASS", 34626, True), ("PASS", 17313, False),
                                             ("FAIL", 34626, False)):
                contract.write_json(path, dict(status=status, optimizer_steps=steps))
                with self.subTest(status=status, steps=steps), patch.object(sys, "argv", ["-", str(path), "formal"]), \
                     patch("sys.stdout", new=io.StringIO()):
                    if accepted:
                        exec(compile(check, "completion", "exec"), {})
                    else:
                        with self.assertRaises(SystemExit):
                            exec(compile(check, "completion", "exec"), {})

    def test_resume_guard_requires_this_route_same_code_data_and_recipe(self):
        train = audit.find_function(audit.tree(ROUTE / "train.py"), "train")
        branch = next(n for n in train.body if isinstance(n, ast.If) and ast.unparse(n.test) == "config['resume_checkpoint']")
        guard = next(n for n in branch.body if isinstance(n, ast.If))
        saved = dict(schema_version=1, route_variant=contract.ROUTE, mode="formal", contract_sha256="identity")
        for field, bad in ((None, None), ("route_variant", "ced_small_mizar_s1_fixed20_gbs256_recursive_mesh_v1"),
                           ("route_variant", "htsat_mizar_s1_fixed20_gbs128_recursive_mesh_5_10x4_5_v1"),
                           ("schema_version", 0), ("mode", "smoke"), ("contract_sha256", "different-assets-code-data")):
            scope = dict(saved=saved if field is None else dict(saved, **{field: bad}),
                         ROUTE=contract.ROUTE, config={"contract_sha256": "identity"})
            with self.subTest(field=field, bad=bad):
                if field is None:
                    audit.execute([guard], scope)
                else:
                    with self.assertRaisesRegex(ValueError, "full formal checkpoint"):
                        audit.execute([guard], scope)

    def test_staging_priority_and_cursor_follow_16_samples_per_rank_update(self):
        from array import array
        plan = function(ROUTE / "stage_data.py", "epoch_priorities",
                        dict(array=array, data_order=contract.data_order, WORLD_SIZE=8, RANK_UPDATE_SAMPLES=16))
        count = 1049
        permutation = list(range(count))
        random.Random(3).shuffle(permutation)
        order = contract.data_order(count)
        for start in (0, 3):
            priorities, first = plan(count, start_batch=start, permutation=permutation)
            for rank in range(8):
                offset = rank * 131
                self.assertEqual(first[str(rank)], [order[i] for i in permutation[offset:offset + 128]])
                for cursor in range(128):
                    self.assertEqual(priorities[order[permutation[offset + cursor]]],
                                     8 if cursor < start * 16 else cursor // 16)

    def test_staging_guard_accepts_only_exact_new_direct_tmpfs_prefix(self):
        class PathView(PurePosixPath):
            def resolve(self): return self
            def is_symlink(self): return False
        validate = function(ROUTE / "stage_data.py", "validate_stage_root", {"Path": PathView})
        good = "/dev/shm/mizar_s1_gbs128_recursive_mesh_5_10x4_5_ced_small_formal_123"
        self.assertEqual(str(validate(good)), good)
        for path in ("/dev/shm/mizar_s1_gbs256_old", "/dev/shm/unrelated", good + "/nested", "/tmp/" + good.rsplit("/", 1)[1]):
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate(path)


if __name__ == "__main__":
    unittest.main()
