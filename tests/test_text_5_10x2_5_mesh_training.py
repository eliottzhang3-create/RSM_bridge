"""Exercise scheduling, epoch boundaries and resume without local GPU packages."""

from __future__ import annotations

import argparse
import ast
import math
import os
import random
import sys
import types
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch


TRAINER = Path(__file__).resolve().parents[1] / "code/RSmol/scripts/train_stage4_5_10x2_5_mesh_ddp.py"


def trainer_namespace():
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    names = {"Stage4Config", "_env", "_parse_args", "DistributedParquetStream", "_cosine_lr", "_validate_resume_contract", "_configure_fp32", "_assert_fp32_model"}
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and (target.id.startswith("DEFAULT_") or target.id in {"MODEL_ARCHITECTURE_CONTRACT", "PRECISION_CONTRACT", "DATA_ROOT_DEFAULT", "OUTPUT_ROOT_DEFAULT"}) for target in node.targets):
            nodes.append(node)
    module = types.ModuleType("_text_mesh_x2_contract")
    module.__dict__.update(argparse=argparse, math=math, os=os, random=random, Path=Path, dataclass=dataclass)
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(TRAINER), "exec"), module.__dict__)
    return module


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def long(self):
        return self

    def clone(self):
        return FakeTensor(self.values)

    def ne(self, value):
        return FakeTensor([item != value for item in self.values])

    def bool(self):
        return self

    def __eq__(self, value):
        return [item == value for item in self.values]

    def __setitem__(self, indices, value):
        for index, selected in enumerate(indices):
            if selected:
                self.values[index] = value


class FakeBatch:
    def __init__(self, values):
        self.values = values
        self.num_rows = len(values)

    def column(self, name):
        assert name == "text"
        return types.SimpleNamespace(to_pylist=lambda: self.values)


class TextMeshX2TrainingTest(unittest.TestCase):
    def setUp(self):
        self.trainer = trainer_namespace()
        self.paths = [Path(f"shard-{index}.parquet") for index in range(32)]
        self.rows = {path: [f"{path.stem}-row-{index}" for index in range(5)] for path in self.paths}

    def new_stream(self, rank=0):
        def tokenizer(texts, **kwargs):
            return {"input_ids": FakeTensor(texts), "attention_mask": FakeTensor([1] * len(texts))}

        return self.trainer.DistributedParquetStream(self.paths, tokenizer, rank=rank, world_size=8, batch_size=2, context_length=1024, pad_token_id=-1, seed=7)

    def parquet_modules(self):
        rows = self.rows

        class ParquetFile:
            def __init__(self, path):
                self.path = path

            def iter_batches(self, *, batch_size, columns, use_threads):
                values = rows[self.path]
                for start in range(0, len(values), batch_size):
                    yield FakeBatch(values[start:start + batch_size])

        pyarrow = types.ModuleType("pyarrow")
        parquet = types.ModuleType("pyarrow.parquet")
        parquet.ParquetFile = ParquetFile
        pyarrow.parquet = parquet
        return patch.dict(sys.modules, {"pyarrow": pyarrow, "pyarrow.parquet": parquet})

    def formal_config(self, *args):
        with patch.dict(os.environ, {}, clear=True):
            return self.trainer._parse_args(["--gate", "FORMAL", *args])

    def test_formal_schedule_uses_one_epoch_and_five_percent_warmup(self):
        config = self.formal_config()
        self.assertEqual((config.precision, config.steps_per_epoch, config.epochs, config.max_optimizer_steps, config.scheduler_total_steps, config.warmup_steps), ("fp32", 9244, 1, 9244, 9244, 463))
        self.assertEqual(config.warmup_steps, math.ceil(0.05 * config.scheduler_total_steps))
        self.assertEqual((config.max_lr, config.min_lr), (8e-4, 2e-5))
        lr = [self.trainer._cosine_lr(step, config) for step in range(config.scheduler_total_steps)]
        self.assertAlmostEqual(lr[0], 8e-4 / 463)
        self.assertEqual(lr[462], 8e-4)
        self.assertTrue(all(left >= right for left, right in zip(lr[463:], lr[464:])))
        self.assertAlmostEqual(lr[-1], config.min_lr, delta=1e-10)
        for invalid in (["--epochs", "2"], ["--steps-per-epoch", "9245"], ["--max-optimizer-steps", "18488"], ["--scheduler-total-steps", "18488"], ["--warmup-steps", "925"], ["--max-lr", "1e-3"], ["--min-lr", "8e-5"]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.formal_config(*invalid)

    def test_second_epoch_is_deterministic_and_keeps_rank_assignment(self):
        all_assigned = []
        for rank in range(8):
            stream = self.new_stream(rank)
            original = list(stream.local_paths)
            self.assertEqual(original, self.paths[rank::8])
            stream.reset_epoch(1)
            self.assertCountEqual(stream.local_paths, original)
            same = self.new_stream(rank)
            same.reset_epoch(1)
            self.assertEqual(stream.local_paths, same.local_paths)
            all_assigned.extend(stream.local_paths)
        self.assertCountEqual(all_assigned, self.paths)
        stream = self.new_stream()
        original = list(stream.local_paths)
        stream.reset_epoch(1)
        self.assertNotEqual(stream.local_paths, original)

    def test_resume_at_each_batch_boundary_has_no_repeated_or_missing_rows(self):
        with self.parquet_modules():
            for epoch in (0, 1):
                stream = self.new_stream()
                stream.reset_epoch(epoch)
                expected = [value for path in stream.local_paths for value in self.rows[path]]
                consumed = []
                for batch in stream:
                    consumed.extend(batch["input_ids"].values)
                    cursor = stream.cursor()
                    resumed = self.new_stream()
                    resumed.restore_cursor(cursor)
                    self.assertEqual(resumed.cursor(), cursor)
                    remaining = [value for next_batch in resumed for value in next_batch["input_ids"].values]
                    self.assertEqual(consumed + remaining, expected)

    def test_production_loop_stops_at_9244_without_starting_second_epoch(self):
        tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_training")
        loop = next(node for node in run.body if isinstance(node, ast.Try)).body
        loop = next(node for node in loop if isinstance(node, ast.While))
        condition = compile(ast.fix_missing_locations(ast.Expression(body=loop.test)), str(TRAINER), "eval")
        transition = compile(ast.fix_missing_locations(ast.Module(body=[loop.body[0]], type_ignores=[])), str(TRAINER), "exec")
        stream = self.new_stream()
        stream.row_offset = 4
        stream.microbatches_seen = 100
        original_iterator = iter(stream)
        scope = {"config": self.formal_config(), "stream_obj": stream, "stream": original_iterator, "rank": 1}
        for step in (0, 5000, 9243):
            scope["optimizer_step"] = step
            self.assertTrue(eval(condition, scope))
            exec(transition, scope)
            self.assertIs(scope["stream"], original_iterator)
            self.assertEqual((stream.epoch, stream.row_offset), (0, 4))
        # Resuming the final checkpoint must not consume another batch.
        resumed = self.new_stream()
        resumed.restore_cursor(stream.cursor())
        scope.update(stream_obj=resumed, stream=iter(resumed), optimizer_step=9244)
        self.assertFalse(eval(condition, scope))
        self.assertEqual((resumed.epoch, resumed.row_offset, resumed.microbatches_seen), (0, 4, 100))

    def test_resume_rejects_changed_shards_and_invalid_row_offsets(self):
        stream = self.new_stream()
        cursor = stream.cursor()
        for change in ({"rank": 1}, {"policy": "old_bf16_cursor"}, {"shard_order": []}, {"row_offset": -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.new_stream().restore_cursor({**cursor, **change})
        with self.parquet_modules():
            for row_offset in (1, 6):
                resumed = self.new_stream()
                resumed.restore_cursor({**cursor, "row_offset": row_offset})
                with self.assertRaises(ValueError):
                    list(resumed)

    def test_resume_rejects_old_precision_or_schedule(self):
        config = self.formal_config()
        state = {"configuration": asdict(config), "optimizer_step": 9244, "scheduler": {"type": "cosine_warmup", "step": 9244, "total_steps": 9244, "warmup_steps": 463, "max_lr": 8e-4, "min_lr": 2e-5}}
        self.trainer._validate_resume_contract(state, config)
        for key, value in (("precision", "bf16"), ("epochs", 2), ("scheduler_total_steps", 18488), ("warmup_steps", 925), ("seed", 8)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "contract mismatch"):
                self.trainer._validate_resume_contract({**state, "configuration": {**state["configuration"], key: value}}, config)
        with self.assertRaises(ValueError):
            self.trainer._validate_resume_contract({**state, "scheduler": {**state["scheduler"], "step": 0}}, config)

    def test_fp32_configuration_disables_tf32_and_rejects_bf16_parameters(self):
        observed = {}
        self.trainer.torch = types.SimpleNamespace(
            float32="float32",
            set_default_dtype=lambda value: observed.update(dtype=value),
            set_float32_matmul_precision=lambda value: observed.update(matmul=value),
            backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)), cudnn=types.SimpleNamespace(allow_tf32=True)),
        )
        self.trainer._configure_fp32()
        self.assertEqual(observed, {"dtype": "float32", "matmul": "highest"})
        self.assertFalse(self.trainer.torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(self.trainer.torch.backends.cudnn.allow_tf32)
        parameter = types.SimpleNamespace(dtype="float32", is_floating_point=lambda: True)
        model = types.SimpleNamespace(named_parameters=lambda: [("weight", parameter)])
        self.trainer._assert_fp32_model(model)
        parameter.dtype = "bfloat16"
        with self.assertRaisesRegex(ValueError, "FP32 training requires"):
            self.trainer._assert_fp32_model(model)


if __name__ == "__main__":
    unittest.main()
