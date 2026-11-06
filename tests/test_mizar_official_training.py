"""Real file/protocol/recipe checks; tensor checks live in a separate optional suite."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_official_final_s1_gbs128_5_20x2_5"
REF = ROUTE.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8")
sys.path.insert(0, str(ROUTE))
import contract as c
import stage_data as stage
from data.audio_paths import AudioPaths
from data.staging import StagingGate, wait_for_start


def node(path, name):
    return next(x for x in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name == name)


def function(path, name, globals_dict):
    tree = ast.Module(body=[node(path, name)], type_ignores=[])
    namespace = dict(globals_dict)
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[name]


def prepared_fixture(directory):
    root = Path(directory) / "staged"
    root.mkdir()
    (root / "audio").mkdir()
    original = Path(directory) / "original.wav"
    original.write_bytes(b"raw recording bytes; protocol test does not decode audio")
    state = {"token": "owned-test", "producer_pid": os.getpid(), "epoch": 0,
             "start_batch": 0, "ready_through_batch": 0, "heartbeat": time.time(),
             "phase": "COPYING", "copied_files": 0, "copied_bytes": 0}
    c.write_json(root / ".owner.json", {"token": state["token"], "root": str(root)})
    c.write_json(root / "STAGING.json", state)
    prepared = {"status": "TRAINING_READY", "read_policy": "shared_until_complete",
                "stage_root": str(root), "async_staging": {k: state[k] for k in
                    ("token", "producer_pid", "epoch", "start_batch")},
                "planned_files": 1, "planned_bytes": original.stat().st_size,
                "staging_epoch_steps": 10, "initial_steps": 0, "precopy_seconds": 0., "phase_seconds": {}}
    prepared["async_staging"]["status_path"] = str(root / "STAGING.json")
    row = {"s1_raw_paths": ["audio/recording.wav"], "s1_original_paths": [str(original)]}
    info = {"source": str(original), "bytes": original.stat().st_size, "mtime_ns": original.stat().st_mtime_ns}
    return root, original, state, prepared, row, info


class RecipeTests(unittest.TestCase):
    def test_geometry_schedule_saves(self):
        g = c.geometry(1477492)
        self.assertEqual((g["steps_per_epoch"], g["total_steps"], g["warmup_steps"],
                          g["dropped_rows_per_epoch"]), (11542, 34626, 1731, 116))
        self.assertEqual((c.BATCH_SIZE, c.RANK_UPDATE_SAMPLES, c.GLOBAL_BATCH_SIZE), (4, 16, 128))
        self.assertEqual(c.TEXT_MODEL_SCHEDULE, tuple([*range(25), *range(5, 25), *range(25, 30)]))
        self.assertEqual(c.checkpoint_steps(), [5771, 11542, 17313, 23084, 28855, 34626])
        self.assertEqual(sum(c.SOURCE_COUNTS.values()), 1477492)

    def test_all_scheduler_steps_match_original_function(self):
        reference = function(REF / "contract.py", "lr_factor", {"math": math})
        for step in range(34627):
            self.assertEqual(c.lr_factor(step, 34626), reference(step, 34626))
        self.assertEqual(c.lr_factor(0, 34626), .1)
        self.assertEqual(c.lr_factor(1730, 34626), 1.)
        self.assertEqual(c.lr_factor(34626, 34626), .1)

    def test_lr_roles_include_recurrent_norms_and_outer_boundaries(self):
        classify = function(ROUTE / "optim.py", "parameter_role", {"model_parameter_role": c.model_parameter_role})
        for layer in range(30):
            for suffix in ("self_attn.q_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight"):
                name = f"caption_decoder.lm.model.layers.{layer}.{suffix}"
                role, lr = c.model_parameter_role(name)
                self.assertEqual(lr, 6e-4 if 5 <= layer < 25 else 3e-4)
                self.assertEqual(classify(name), role)
        for loop in (0, 1):
            for layer in (5, 24):
                for matrix in ("A", "B"):
                    name = f"caption_decoder.lm.model.layers.{layer}.mlp.down_proj.lora_{matrix}.{loop}"
                    self.assertEqual(classify(name), f"loop{loop+1}_lora_{matrix}")
                    self.assertEqual(c.model_parameter_role(name)[1], 6e-4)
        for name in ("audio_encoder.base.mapper.audio_type", "caption_decoder.lm.model.embed_tokens.weight",
                     "caption_decoder.lm.model.norm.weight", "caption_decoder.lm.lm_head.weight"):
            self.assertEqual(c.model_parameter_role(name)[1], 3e-4)
        self.assertEqual(c.model_parameter_role("audio_encoder.base.ced.blocks.0.weight"), ("frozen_ced", None))
        for name in ("anything.weight", "caption_decoder.lm.model.layers.25.mlp.up_proj.lora_A.0",
                     "caption_decoder.lm.model.layers.5.mlp.up_proj.lora_B.2"):
            with self.assertRaises(ValueError):
                classify(name)

    def test_sampler_priority_uses_sixteen_samples_per_update_and_resume(self):
        rows = 1024
        permutation = list(range(rows))
        random.Random(88).shuffle(permutation)
        priorities, first = stage.epoch_priorities(rows, epoch=1, start_batch=2, permutation=permutation)
        order = c.data_order(rows)
        for rank in range(8):
            shard = [order[i] for i in permutation[rank * 128:(rank+1)*128]]
            self.assertEqual(first[str(rank)], shard)
            for cursor, row in enumerate(shard):
                self.assertEqual(priorities[row], cursor // 16 if cursor >= 32 else 8)

    def test_originals_unchanged_and_copied_data_math_match(self):
        guard = json.loads((ROUTE / "source_guard.json").read_text())
        for relative, expected in guard.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), expected, relative)
        for directory in ("data", "utils"):
            for file in (REF / directory).glob("*.py"):
                self.assertEqual((ROUTE / directory / file.name).read_text(), file.read_text(), file.name)
        for path, name in (("train.py", "answer_loss"), ("optim.py", "GradNormTracker"),
                           ("contract.py", "raw_references"), ("contract.py", "data_order"),
                           ("stage_data.py", "copy_background"), ("stage_data.py", "copy_one")):
            self.assertEqual(ast.dump(node(ROUTE / path, name)), ast.dump(node(REF / path, name)), name)

    def test_contract_import_does_not_import_tensor_libraries(self):
        # This test suite itself imports only stdlib + stdlib project code.
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_gpu_loader_never_calls_cpu_dependencies(self):
        tree = ast.parse((ROUTE / "runtime.py").read_text())
        calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertNotIn("dependencies", calls)
        self.assertNotIn("reload_verified", calls)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", (ROUTE / "runtime.py").read_text())

    def test_all_route_local_import_targets_are_present(self):
        for path in ROUTE.rglob("*.py"):
            for item in ast.walk(ast.parse(path.read_text())):
                if isinstance(item, ast.ImportFrom) and item.module and item.module.split(".")[0] in {
                        "utils", "data", "contract", "optim", "runtime", "stage_data"}:
                    target = ROUTE.joinpath(*item.module.split("."))
                    self.assertTrue(target.with_suffix(".py").is_file() or target.is_dir(), (path, item.module))

    def test_collective_switch_stays_before_fetch_and_outside_microbatches(self):
        train = node(ROUTE / "train.py", "train")
        update = next(n for n in ast.walk(train) if isinstance(n, ast.For) and ast.unparse(n.target) == "update_index")
        gate = next(i for i, n in enumerate(update.body) if "staging.wait_for_batch" in ast.unparse(n))
        switch = next(i for i, n in enumerate(update.body) if isinstance(n, ast.If) and "dataset.audio_paths.local" in ast.unparse(n.test))
        micro = next(i for i, n in enumerate(update.body) if isinstance(n, ast.For) and ast.unparse(n.target) == "micro_index")
        self.assertLess(gate, switch)
        self.assertLess(switch, micro)
        switch_text = ast.unparse(update.body[switch])
        self.assertIn("dist.all_reduce(ready, op=dist.ReduceOp.MIN)", switch_text)
        self.assertIn("dataset.audio_paths.use_local()", switch_text)
        micro_text = ast.unparse(update.body[micro])
        self.assertIn("ddp.no_sync()", micro_text)
        self.assertIn("scaler.scale(loss / ACCUMULATION_STEPS).backward()", micro_text)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        prepared = {"rows": 1477492, "source_rows": c.SOURCE_COUNTS, "read_policy": "shared_until_complete",
                    "rows_canonical_sha256": "rows", "mapping_sha256": "mapping"}
        contract = c.semantic_contract(prepared, ROUTE / "fixture_model", {"fixture": True})
        self.config = {"contract": contract, "contract_sha256": c.digest(contract)}
        self.saved = {"schema_version": 1, "route_variant": c.ROUTE, "mode": "formal", "num_epochs": 3,
                      "contract": contract, "contract_sha256": c.digest(contract), "state_dict": {},
                      "optimizer": {"state": {0: {"step": 11542}}}, "scheduler": {"last_epoch": 11542},
                      "grad_scaler": {"scale": 65536}, "grad_norm_tracker": {"x": (1., 10.)},
                      "random_state_by_rank": [{k: [] for k in ("python", "numpy", "torch", "cuda")} for _ in range(8)],
                      "total_step": 11542, "epoch": 1, "next_optimizer_step": 0,
                      "loss_reduction": "mean_of_rank_local_microbatch_token_means_then_DDP_mean",
                      "batch_geometry": {"world_size": 8, "microbatch_size": 4,
                                         "gradient_accumulation_steps": 4, "global_batch_size": 128}}

    def test_epoch_boundary_cursor_is_accepted(self):
        self.assertEqual(c.validate_resume_payload(self.saved, self.config), (11542, 1, 0))

    def test_wrong_route_partial_state_cursor_rng_and_recipe_are_rejected(self):
        variants = [("route_variant", "old_route"), ("mode", "smoke"), ("epoch", 0),
                    ("next_optimizer_step", 1), ("total_step", 34626), ("random_state_by_rank", [{}]*7),
                    ("scheduler", {"last_epoch": 11}), ("grad_norm_tracker", {}), ("grad_scaler", {}),
                    ("contract_sha256", "changed"), ("optimizer", {"state": {0: {"step": 5}}}),
                    ("loss_reduction", "global_token_mean")]
        for key, value in variants:
            with self.subTest(field=key), self.assertRaises(ValueError):
                saved = copy.deepcopy(self.saved)
                saved[key] = value
                c.validate_resume_payload(saved, self.config)

    def test_model_only_cannot_resume(self):
        with self.assertRaises(ValueError):
            c.validate_resume_payload({"state_dict": {}}, self.config)


class StagingTests(unittest.TestCase):
    def test_training_reads_shared_while_real_background_copy_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, original, state, prepared, row, info = prepared_fixture(tmp)
            started, release = threading.Event(), threading.Event()
            errors = []
            real_copy = stage.copy_one
            def blocked_copy(*args):
                started.set()
                if not release.wait(5):
                    raise TimeoutError("test release missing")
                return real_copy(*args)
            def produce():
                try:
                    stage.copy_background(root, prepared, [(row["s1_raw_paths"][0], info)],
                                          Path(tmp)/"report.json", state, workers=1)
                except Exception as e:
                    errors.append(e)
            with patch.object(stage, "copy_one", blocked_copy):
                worker = threading.Thread(target=produce)
                worker.start()
                try:
                    self.assertTrue(started.wait(3))
                    published = json.loads((Path(tmp)/"report.json").read_text())
                    self.assertEqual(published["status"], "TRAINING_READY")
                    self.assertEqual(published["initial_steps"], 0)
                    gate = StagingGate(published, timeout=.01)
                    gate.wait_for_batch(2, 9999)  # Must never wait for a watermark under this policy.
                    self.assertFalse(gate.complete)
                    paths = AudioPaths(root, "shared_until_complete")
                    self.assertEqual(paths.resolve(row), [original])
                    self.assertFalse((root / row["s1_raw_paths"][0]).exists())
                finally:
                    release.set()
                    worker.join(5)
            self.assertFalse(errors, errors)
            self.assertFalse(worker.is_alive())
            gate.wait_for_batch(2, 9999)
            self.assertTrue(gate.complete)
            paths.use_local()  # DDP selection order is checked separately.
            local = paths.resolve(row)[0]
            self.assertEqual(local.read_bytes(), original.read_bytes())
            self.assertEqual(paths.read_files, {"shared": 1, "local": 1})
            self.assertFalse(list(root.rglob("*.partial")))

    def test_complete_counts_identity_failure_heartbeat_and_dead_producer(self):
        mutations = ({"phase": "COMPLETE", "copied_files": 0}, {"token": "other-job"},
                     {"phase": "FAILED", "error": "copy failed"}, {"heartbeat": time.time()-200})
        for change in mutations:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root, _, state, prepared, _, _ = prepared_fixture(tmp)
                c.write_json(root / "STAGING.json", dict(state, **change))
                with self.assertRaises(RuntimeError):
                    StagingGate(prepared).wait_for_batch(0, 0)
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, prepared, _, _ = prepared_fixture(tmp)
            with patch("data.staging.process_alive", return_value=False), self.assertRaises(RuntimeError):
                StagingGate(prepared).wait_for_batch(0, 0)

    def test_resume_waits_for_this_allocation_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, prepared, _, _ = prepared_fixture(tmp)
            report = Path(tmp)/"report.json"
            stale = copy.deepcopy(prepared)
            stale["stage_root"] = str(Path(tmp)/"previous-allocation")
            c.write_json(report, stale)
            released = threading.Event()
            def publish():
                time.sleep(.2)
                c.write_json(report, prepared)
                released.set()
            worker = threading.Thread(target=publish)
            worker.start()
            try:
                wait_for_start(report, root, os.getpid(), poll=.01, timeout=2.)
                self.assertTrue(released.is_set())
            finally:
                worker.join(3)

    def test_copy_detects_changed_source_and_never_publishes_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, original, _, _, row, info = prepared_fixture(tmp)
            original.write_bytes(b"changed length")
            with self.assertRaises(IOError):
                stage.copy_one(root, (row["s1_raw_paths"][0], info))
            self.assertFalse((root / row["s1_raw_paths"][0]).exists())
            self.assertFalse(list(root.rglob("*.partial")))

    def test_local_path_escape_and_partial_and_unowned_cleanup_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, _, row, _ = prepared_fixture(tmp)
            paths = AudioPaths(root)
            for path in ("../escape.wav", "audio/unfinished.partial"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    paths.resolve(dict(row, s1_raw_paths=[path]))
            with self.assertRaises(ValueError):
                stage.cleanup(root, "wrong")
            self.assertTrue(root.is_dir())


if __name__ == "__main__":
    unittest.main()
