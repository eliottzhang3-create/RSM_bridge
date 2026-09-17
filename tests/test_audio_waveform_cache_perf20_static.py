"""Dependency-light contracts for waveform-cache PERF20 input and sampling."""
from __future__ import annotations

import ast
import random
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "code" / "RSmol" / "audio_5_10x2_5_mesh_mellow" / "data.py"
TRAIN = ROOT / "code" / "RSmol" / "scripts" / "train_audio_5_10x2_5_mesh_mellow_ddp.py"
INNER = ROOT / "code" / "RSmol" / "scripts" / "train_audio_perf20_5_10x2_5_mesh_mellow_ddp.sh"


class _SamplerStub:
    @classmethod
    def __class_getitem__(cls, item):
        return cls


class AudioWaveformCachePerf20StaticTest(unittest.TestCase):
    def test_cache_contract_and_mmap_reader_are_explicit(self) -> None:
        text = DATA.read_text(encoding="utf-8")
        ast.parse(text)
        for marker in (
            "class WaveformShardCache",
            'WAVEFORM_CACHE_FORMAT = "raw_fixed_waveform_shards_v1"',
            'dtype="<f4"',
            'mode="c"',
            '"BUILDING"',
            '"status": "PASS"',
            '"sample_rate": 32000',
            '"samples_per_audio": 320000',
            '"bytes_per_audio": 1280000',
            "audio path is absent from waveform cache",
            "self.waveform_cache.load_location(first_location)",
            "self.waveform_cache.load_location(second_location)",
        ):
            self.assertIn(marker, text)

    def test_shard_aware_sampler_assigns_disjoint_shards_to_ranks(self) -> None:
        tree = ast.parse(DATA.read_text(encoding="utf-8"))
        class_node = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ShardAwareDistributedBatchSampler"
        )
        namespace = {"Sampler": _SamplerStub, "random": random, "Any": object}
        exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(DATA), "exec"), namespace)
        cls = namespace["ShardAwareDistributedBatchSampler"]

        samplers = []
        for rank in range(2):
            sampler = cls.__new__(cls)
            sampler.batch_size = 2
            sampler.num_replicas = 2
            sampler.rank = rank
            sampler.seed = 17
            sampler.epoch = 0
            sampler.groups = {
                0: list(range(0, 8)),
                1: list(range(8, 16)),
                2: list(range(16, 24)),
                3: list(range(24, 32)),
            }
            sampler.shards_per_rank = 2
            samplers.append(sampler)
        rank_batches = [list(iter(sampler)) for sampler in samplers]
        self.assertEqual([len(rows) for rows in rank_batches], [8, 8])
        global_batches = rank_batches[0] + rank_batches[1]
        self.assertEqual(sorted(index for batch in global_batches for index in batch), list(range(32)))
        self.assertTrue(all(len({index // 8 for index in batch}) == 1 for batch in global_batches))
        rank_shards = [
            {index // 8 for batch in batches for index in batch}
            for batches in rank_batches
        ]
        self.assertTrue(rank_shards[0].isdisjoint(rank_shards[1]))
        self.assertEqual([len(shards) for shards in rank_shards], [2, 2])
        samplers[0].set_epoch(1)
        self.assertNotEqual(rank_batches[0], list(iter(samplers[0])))

    def test_waveform_shard_experiment_is_retired_from_perf20(self) -> None:
        train = TRAIN.read_text(encoding="utf-8")
        inner = INNER.read_text(encoding="utf-8")
        for marker in (
            '"--waveform-cache-dir"',
            "the 64-shard mmap experiment is abandoned",
            '"retired_waveform_shard_experiment": True',
            'PERF20_INPUT_MODES = ("online", "warm_online", "waveform_preload", "full_preload", "shared_waveform_store", "store_rank_ram_preload", "store_rank_ram_prefetch")',
        ):
            self.assertIn(marker, train)
        self.assertIn('"shards_per_rank"', DATA.read_text(encoding="utf-8"))
        self.assertNotIn("ShardAwareDistributedBatchSampler", train)
        self.assertNotIn("--waveform-cache-dir", inner)
        self.assertNotIn("perf20_waveform_shards", inner)


if __name__ == "__main__":
    unittest.main()
