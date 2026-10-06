from __future__ import annotations

import ast
import random
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


TRAINER = Path(__file__).resolve().parents[1] / "code/RSmol/scripts/train_stage4_5_10x4_5_mesh_ddp.py"


def stream_class():
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "DistributedParquetStream")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    namespace = {"random": random}
    exec(compile(ast.fix_missing_locations(module), str(TRAINER), "exec"), namespace)
    return namespace["DistributedParquetStream"]


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


class FakeColumn:
    def __init__(self, values):
        self.values = values

    def to_pylist(self):
        return self.values


class FakeBatch:
    def __init__(self, values):
        self.values = values
        self.num_rows = len(values)

    def column(self, name):
        assert name == "text"
        return FakeColumn(self.values)


class TextMeshX4StreamTest(unittest.TestCase):
    def setUp(self):
        self.stream_type = stream_class()
        self.paths = [Path(f"shard-{index}.parquet") for index in range(32)]
        self.rows = {path: [f"{path.stem}-row-{index}" for index in range(5)] for path in self.paths}

    def new_stream(self, rank=0):
        def tokenizer(texts, **kwargs):
            return {"input_ids": FakeTensor(texts), "attention_mask": FakeTensor([1] * len(texts))}

        return self.stream_type(self.paths, tokenizer, rank=rank, world_size=8, batch_size=2, context_length=1024, pad_token_id=-1, seed=7)

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

    def test_second_epoch_shuffles_only_local_shards(self):
        stream = self.new_stream()
        original = list(stream.local_paths)
        self.assertEqual(original, [self.paths[0], self.paths[8], self.paths[16], self.paths[24]])
        stream.reset_epoch(1)
        self.assertCountEqual(stream.local_paths, original)
        self.assertNotEqual(stream.local_paths, original)
        same = self.new_stream()
        same.reset_epoch(1)
        self.assertEqual(stream.local_paths, same.local_paths)

    def test_checkpoint_cursor_resumes_at_next_row_in_second_epoch(self):
        with self.parquet_modules():
            stream = self.new_stream()
            stream.reset_epoch(1)
            iterator = iter(stream)
            first = next(iterator)["input_ids"].values
            cursor = stream.cursor()
            resumed = self.new_stream()
            resumed.restore_cursor(cursor)
            self.assertEqual(resumed.cursor(), cursor)
            rest = [value for batch in resumed for value in batch["input_ids"].values]
            expected = [value for path in stream.local_paths for value in self.rows[path]]
            self.assertEqual(first + rest, expected)


if __name__ == "__main__":
    unittest.main()
