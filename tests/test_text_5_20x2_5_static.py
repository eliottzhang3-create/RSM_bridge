"""Pure-text route contracts, inherited algorithm parity, and CPU preflight."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import itertools
import json
import math
import random
import sys
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "code/RSmol"
SCRIPTS = BASE / "scripts"
SCHEDULE = tuple(range(25)) + tuple(range(5, 30))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = load_file("text_52025_train_test", SCRIPTS / "train_stage4_5_20x2_5_ddp.py")
CONVERT = load_file("text_52025_convert_test", SCRIPTS / "convert_stepwise_5_20x2_5.py")
PREPARE = load_file("text_52025_prepare_test", SCRIPTS / "prepare_text_5_20x2_5.py")


def model_helpers():
    tree = ast.parse((BASE / "recursive_model_5_20x2_5.py").read_text(encoding="utf-8"))
    names = {"build_5_20x2_5_schedule", "build_5_20x2_5_source_mapping", "logical_slot_for_execution", "LogicalSlotCacheView"}
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             or isinstance(n, ast.ImportFrom) and n.module == "__future__"
             or isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    namespace = {"inspect": inspect}
    exec(compile(ast.Module(nodes, type_ignores=[]), "model_helpers", "exec"), namespace)
    return types.SimpleNamespace(**namespace)


def state_fixture():
    config = TRAIN._parse_args(["--gate", "FORMAL"])
    return {"configuration": asdict(config), "optimizer_step": 500,
            "checkpoint_contract": {"text_route": TRAIN.TEXT_ROUTE, "architecture": TRAIN.MODEL_ARCHITECTURE_CONTRACT},
            "scheduler_config": TRAIN.scheduler_metadata(total_steps=18488, warmup_steps=925)}


def audit_fixture(tmp):
    data = tmp / "data"
    data.mkdir()
    token = tmp / "tokenizer"
    token.mkdir()
    (token / "tokenizer.json").write_text("{}", encoding="utf-8")
    shards = []
    for name in TRAIN.EXPECTED_PARQUET_NAMES:
        path = data / name
        path.write_bytes(b"test parquet identity")
        shards.append({"name": name, "path": str(path), "bytes": path.stat().st_size, "num_rows": 200000})
    report = {"status": "PASS", "gate": "B", "cpu_only": True,
              "configuration": {"data_dir": str(data), "tokenizer_path": str(token), "context_length": 1024},
              "dataset_audit": {"shards": shards, "total_rows": 85 * 200000, "content_audit": True},
              "manifest": {"world_size": 1, "seed": 999, "rank_shards": {"0": ["wrong"]}}}
    path = tmp / "audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return data, token, path, report


class TextRouteStaticTest(unittest.TestCase):
    def test_original_files_unchanged(self):
        guard = json.loads((BASE / "text_5_20x2_5/source_guard.json").read_text(encoding="utf-8"))
        self.assertEqual(len(guard), 10)
        for name, digest in guard.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_actual_model_schedule_mapping_and_cache_slots(self):
        helpers = model_helpers()
        self.assertEqual(helpers.build_5_20x2_5_schedule(), SCHEDULE)
        self.assertEqual(helpers.build_5_20x2_5_source_mapping(30), tuple(range(30)))
        for logical, physical in enumerate(SCHEDULE):
            self.assertEqual(helpers.logical_slot_for_execution(logical, physical), logical)
        for kwargs in ({"logical_layer_count": 30}, {"physical_layer_count": 20}):
            with self.assertRaises(ValueError):
                helpers.build_5_20x2_5_schedule(**kwargs)
        with self.assertRaises(ValueError):
            helpers.build_5_20x2_5_source_mapping(50)
        with self.assertRaises(ValueError):
            helpers.logical_slot_for_execution(25, 25)
        cache = types.SimpleNamespace(update=lambda k, v, slot, cache_kwargs=None: (slot, cache_kwargs),
                                      get_seq_length=lambda slot: slot + 100)
        for logical, physical in enumerate(SCHEDULE):
            view = helpers.LogicalSlotCacheView(cache, physical_index=physical, logical_slot=logical)
            self.assertEqual(view.update(None, None, physical, {"test": 1}), (logical, {"test": 1}))
            self.assertEqual(view.get_seq_length(), logical + 100)
        with self.assertRaises(ValueError):
            view.update(None, None, 0)

    def test_config_preserves_source_except_explicit_geometry(self):
        source = types.SimpleNamespace(num_hidden_layers=30, hidden_size=576, intermediate_size=1536,
                                       num_attention_heads=9, num_key_value_heads=3, vocab_size=49152,
                                       rope_theta=100000.0, tie_word_embeddings=True, rms_norm_eps=1e-5,
                                       max_position_embeddings=8192, pad_token_id=2,
                                       rope_scaling={"rope_type": "default"})
        before = copy.deepcopy(vars(source))
        target = CONVERT.build_target_config(source)
        self.assertEqual(vars(source), before)
        self.assertEqual((target.num_hidden_layers, target.recursive_layer_count, target.recursive_middle_layer_count), (50, 30, 20))
        for key, value in before.items():
            if key != "num_hidden_layers":
                self.assertEqual(getattr(target, key), value)
        target.rope_scaling["changed"] = True
        self.assertNotIn("changed", source.rope_scaling)
        self.assertEqual(target.logical_to_physical, list(SCHEDULE))
        self.assertEqual(target.recursive_source_layer_indices_0based, list(range(30)))
        source.recursive_layer_count = 30
        with self.assertRaises(ValueError):
            CONVERT.build_target_config(source)

    def test_formal_geometry_budget_and_saves(self):
        config = TRAIN._parse_args(["--gate", "FORMAL"])
        self.assertEqual((config.world_size, config.micro_batch_size, config.gradient_accumulation_steps), (8, 4, 16))
        self.assertEqual((config.max_optimizer_steps, config.scheduler_total_steps, config.warmup_steps), (18488, 18488, 925))
        self.assertEqual(TRAIN.DEFAULT_GLOBAL_EFFECTIVE_BATCH, 512)
        self.assertEqual(TRAIN.DEFAULT_FORMAL_GLOBAL_SAMPLES, 9244 * 1024)
        self.assertEqual(TRAIN.DEFAULT_FORMAL_SAMPLES_PER_RANK, 1183232)
        self.assertEqual(TRAIN.formal_save_steps()[-3:], [17500, 18000, 18488])
        self.assertEqual(len(TRAIN.formal_save_steps()), 37)

    def test_formal_rejects_conflicting_recipe(self):
        for flag, value in (("--micro-batch-size", "8"), ("--gradient-accumulation-steps", "8"),
                            ("--max-optimizer-steps", "9244"), ("--warmup-steps", "463"),
                            ("--scheduler-total-steps", "9244"), ("--world-size", "4"),
                            ("--context-length", "512"), ("--checkpoint-retention", "4"),
                            ("--save-every", "1000"), ("--max-grad-norm", "2"),
                            ("--seed", "1"), ("--record-buffer-size", "2048")):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                TRAIN._parse_args(["--gate", "FORMAL", flag, value])
        with self.assertRaises(ValueError):
            TRAIN._parse_args(["--gate", "FORMAL", "--learning-rate", "0.001", "--max-lr", "0.001", "--min-lr", "0.0001"])
        for gate in ("B", "C"):
            with self.assertRaises(ValueError):
                TRAIN._parse_args(["--gate", gate])
        self.assertEqual(TRAIN._parse_args(["--gate", "D"]).max_optimizer_steps, 10)

    def test_resume_accepts_same_route_and_rejects_cross_route_or_recipe(self):
        state = state_fixture()
        self.assertEqual(TRAIN.validate_formal_resume_state(state)["optimizer_step"], 500)
        for section, key, value in (("checkpoint_contract", "architecture", "old"),
                                    ("checkpoint_contract", "text_route", "mesh"),
                                    ("configuration", "micro_batch_size", 8),
                                    ("configuration", "gradient_accumulation_steps", 8),
                                    ("configuration", "gate", "D"),
                                    ("configuration", "optimizer_betas", [0.9, 0.999]),
                                    ("scheduler_config", "warmup_steps", 463),
                                    ("scheduler_config", "total_steps_for_schedule", 9244),
                                    ("scheduler_config", "max_lr", 0.001)):
            wrong = copy.deepcopy(state)
            wrong[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                TRAIN.validate_formal_resume_state(wrong)
        for step in (-1, 18489):
            with self.assertRaises(ValueError):
                TRAIN.validate_formal_resume_state(dict(state, optimizer_step=step))

    def test_all_scheduler_positions(self):
        actual = [TRAIN.cosine_warmup_factor(step, 18488, 925) * 8e-4 for step in range(18489)]
        expected = [8e-5 + (8e-4 - 8e-5) * step / 925 if step < 925 else
                    8e-5 + (8e-4 - 8e-5) * (1 + math.cos(math.pi * (step - 925) / (18488 - 925))) / 2
                    for step in range(18489)]
        self.assertLess(max(abs(a - b) for a, b in zip(actual, expected)), 1e-15)
        self.assertEqual(actual[0], 8e-5)
        self.assertAlmostEqual(actual[925], 8e-4)
        self.assertAlmostEqual(actual[-1], 8e-5)

    def test_preserved_data_loss_optimizer_and_update_algorithms(self):
        old = ast.parse((SCRIPTS / "train_stage4_5_10_5_ddp.py").read_text(encoding="utf-8"))
        new = ast.parse((SCRIPTS / "train_stage4_5_20x2_5_ddp.py").read_text(encoding="utf-8"))
        before = {n.name: n for n in old.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        after = {n.name: n for n in new.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in ("assign_shards", "_iter_shard_texts", "_tokenize_ids", "collate_dynamic_padding",
                     "_bf16_autocast", "_causal_loss_sum", "token_weighted_gradient_scale",
                     "_training_window", "_load_scheduler", "cosine_warmup_factor", "compute_warmup_steps",
                     "_ddp_wrap", "optimizer_parameter_audit", "_gather_rank_cursors"):
            with self.subTest(function=name):
                self.assertEqual(ast.dump(before[name]), ast.dump(after[name]))
        def update_tail(node):
            pos = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                       and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "_ddp_wrap")
            return ast.dump(ast.Module(node.body[pos:], type_ignores=[]))
        self.assertEqual(update_tail(before["run_training"]), update_tail(after["run_training"]))
        # Stream cursor / drop policy is inherited; its sole runtime change is micro8 -> micro4.
        a = ast.get_source_segment((SCRIPTS / "train_stage4_5_10_5_ddp.py").read_text(encoding="utf-8"), before["DistributedParquetStream"])
        b = ast.get_source_segment((SCRIPTS / "train_stage4_5_20x2_5_ddp.py").read_text(encoding="utf-8"), after["DistributedParquetStream"])
        self.assertEqual(ast.dump(ast.parse(a.replace("exactly eight", "exactly four").replace("micro_batch_size != 8", "micro_batch_size != 4").replace("micro_batch_size=8", "micro_batch_size=4"))), ast.dump(ast.parse(b)))

    def test_model_forward_attention_and_cache_algorithms_preserved(self):
        old_text = (BASE / "recursive_model_5_10_5.py").read_text(encoding="utf-8").replace("5_10_5", "5_20x2_5")
        new_text = (BASE / "recursive_model_5_20x2_5.py").read_text(encoding="utf-8")
        old = {n.name: n for n in ast.parse(old_text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        new = {n.name: n for n in ast.parse(new_text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in ("_causal_mask", "_call_decoder_layer", "_cache_seq_length", "make_dynamic_cache"):
            self.assertEqual(ast.dump(old[name]), ast.dump(new[name]))
        def method(node, name):
            return next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
        self.assertEqual(ast.dump(method(old["RecursiveLlama5_20x2_5Model"], "forward")),
                         ast.dump(method(new["RecursiveLlama5_20x2_5Model"], "forward")))
        for name in ("update", "get_seq_length", "get_usable_length"):
            self.assertEqual(ast.dump(method(old["LogicalSlotCacheView"], name)),
                             ast.dump(method(new["LogicalSlotCacheView"], name)))

    def test_stream_runs_micro4_cursor_and_preserves_shard_drop_policy(self):
        parquet = types.ModuleType("pyarrow.parquet")
        reads = []
        class FakeParquet:
            def __init__(self, path):
                reads.append(path)
            def iter_batches(self, **kwargs):
                self_kwargs = kwargs
                assert self_kwargs["columns"] == ["text"] and self_kwargs["use_threads"] is False
                yield types.SimpleNamespace(column=lambda _: types.SimpleNamespace(to_pylist=lambda: [str(i) for i in range(11)] + [None, " "]))
        parquet.ParquetFile = FakeParquet
        arrow = types.ModuleType("pyarrow")
        arrow.parquet = parquet
        tokenizer = lambda text, **kwargs: {"input_ids": [int(text) + 3, 2]}
        with patch.dict(sys.modules, {"pyarrow": arrow, "pyarrow.parquet": parquet}), \
             patch.object(TRAIN, "collate_dynamic_padding", side_effect=lambda rows, **kw: copy.deepcopy(rows)):
            stream = TRAIN.DistributedParquetStream([Path("rank0-first"), Path("rank0-second")], tokenizer,
                rank=0, micro_batch_size=4, context_length=1024, pad_token_id=0,
                seed=0, record_buffer_size=4096, device=None)
            batches = list(itertools.islice(iter(stream), 3))
            self.assertEqual([len(batch) for batch in batches], [4, 4, 4])
            self.assertEqual(reads, [Path("rank0-first"), Path("rank0-second")])
            cursor = stream.cursor()
            self.assertEqual((cursor["samples_seen"], cursor["microbatches_seen"], cursor["shard_index"]), (12, 3, 1))
            self.assertFalse(cursor["bounded_shuffle_bitwise_exact"])
            self.assertIn("bounded_shuffle_rng_state", cursor)

    def test_deterministic_shards_and_token_weighting(self):
        files = [Path(name) for name in TRAIN.EXPECTED_PARQUET_NAMES]
        manifest = TRAIN.assign_shards(files, world_size=8, seed=0)
        self.assertEqual(list(manifest["rank_shard_counts"].values()), [11] * 5 + [10] * 3)
        expected = files.copy()
        random.Random(0).shuffle(expected)
        self.assertEqual(manifest["shards_shuffled"], list(map(str, expected)))
        # Unequal token populations: the DDP mean of locally scaled sums is the token mean.
        sums, tokens = [5, 20, 3, 11, 8, 50, 15, 1], [2, 8, 1, 4, 3, 9, 5, 1]
        scale = TRAIN.token_weighted_gradient_scale(world_size=8, global_window_tokens=sum(tokens))
        self.assertAlmostEqual(sum(s * scale for s in sums) / 8, sum(sums) / sum(tokens))

    def test_cpu_audit_compatibility_rebuilds_assignment_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            data, token, path, report = audit_fixture(Path(directory))
            result = TRAIN.validate_cpu_audit(path, data_dir=data, tokenizer_path=token)
            self.assertEqual(list(result["manifest"]["rank_shard_counts"].values()), [11] * 5 + [10] * 3)
            for mutate in (lambda r: r.update(status="FAIL"),
                           lambda r: r["configuration"].update(context_length=512),
                           lambda r: r["dataset_audit"]["shards"][0].update(bytes=0),
                           lambda r: r["dataset_audit"].update(content_audit=False)):
                changed = copy.deepcopy(report)
                mutate(changed)
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    TRAIN.validate_cpu_audit(path, data_dir=data, tokenizer_path=token)

    def test_existing_conversion_output_rejected_before_tensor_import(self):
        with tempfile.TemporaryDirectory() as directory:
            args = CONVERT.parse_args(["--output-dir", directory])
            with self.assertRaises(FileExistsError):
                CONVERT.convert(args)
        with self.assertRaises(ValueError):
            CONVERT.reject_forbidden_output(ROOT / "bad-model")

    def test_preflight_reuses_report_without_model_or_tensor_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            data, token, path, report = audit_fixture(tmp)
            config = vars(CONVERT.build_target_config(types.SimpleNamespace(num_hidden_layers=30)))
            (token / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (token / "recursive_conversion_metadata.json").write_text(json.dumps({"status": "PASS", "text_route": TRAIN.TEXT_ROUTE}), encoding="utf-8")
            with patch.object(PREPARE.subprocess, "run", side_effect=AssertionError("Audit should be reused")):
                self.assertEqual(PREPARE.main(["--model-dir", str(token), "--data-dir", str(data),
                                              "--output-dir", str(tmp / "new-formal"), "--audit-report", str(path)]), 0)


if __name__ == "__main__":
    unittest.main()
