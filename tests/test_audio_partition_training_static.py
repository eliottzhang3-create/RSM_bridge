"""Dependency-light audits for the partition trainer (no local torch required)."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/RSmol/scripts/train_audio_partitioned_5_10x2_5_mesh_mellow_ddp.py"
MODEL = ROOT / "code/RSmol/audio_5_10x2_5_mesh_mellow/model.py"
DATA = ROOT / "code/RSmol/audio_5_10x2_5_mesh_mellow/data.py"


def functions(*names):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "math": math,
        "random": random,
        "hashlib": hashlib,
        "json": json,
        "Path": Path,
        "CONTRACT": "component_partitions6_rank_ram_compact_audio_answer_eos_v2",
    }
    module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[]))
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return [namespace[name] for name in names]


def data_functions(*names):
    tree = ast.parse(DATA.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(DATA), "exec"), namespace)
    return [namespace[name] for name in names]


class FakeDataset:
    def __init__(self, singles: int, duals: int):
        self.singles, self.duals = singles, duals

    def __len__(self):
        return self.singles + self.duals

    def audio_structure(self, index):
        return index < self.singles, False


class PartitionTrainingContracts(unittest.TestCase):
    def test_quota_and_p0_position(self):
        quota, order = functions("_quotas", "_order")
        counts = [278680, 137874, 137873, 137874, 137874, 137884]
        for epoch, row in enumerate(quota(counts, 11, 256)):
            self.assertEqual(sum(row), sum(counts) // 256)
            self.assertEqual(row[0], 278680 // 256)
            self.assertTrue(all(538 <= number <= 540 for number in row[1:]))
            arrangement = order(7, epoch)
            self.assertEqual(sorted(arrangement), list(range(6)))
            self.assertNotEqual(arrangement[-1], 0)

    def test_p0_global_shuffle_no_repeat(self):
        (plan,) = functions("_plan")
        dataset = FakeDataset(520, 510)
        chunks, audit = plan(dataset, seed=7, epoch=0, pid=0, steps=4, global_batch=256)
        flattened = [row for chunk in chunks for row in chunk]
        self.assertEqual(len(flattened), 1024)
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(audit["repeated_rows"], 0)
        self.assertEqual(audit["dropped_rows"], 6)
        self.assertTrue(any(len({index < 520 for index in chunk}) == 2 for chunk in chunks))
        again, repeated_audit = plan(dataset, seed=7, epoch=0, pid=0, steps=4, global_batch=256)
        self.assertEqual(chunks, again)
        self.assertEqual(audit["plan_sha256"], repeated_audit["plan_sha256"])

    def test_small_partition_repeat_stays_in_partition(self):
        (plan,) = functions("_plan")
        dataset = FakeDataset(260, 260)
        chunks, audit = plan(dataset, seed=0, epoch=0, pid=3, steps=3, global_batch=256)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(audit["repeated_rows"], 248)
        for chunk in chunks:
            self.assertEqual(len(chunk), 256)
            self.assertTrue(all(0 <= index < 520 for index in chunk))

    def test_prefix_and_checkpoint_isolation(self):
        text = SCRIPT.read_text(encoding="utf-8")
        model = MODEL.read_text(encoding="utf-8")
        data = DATA.read_text(encoding="utf-8")
        for marker in ("dist.init_process_group", "timeout=timedelta", "cache._get_audio_id", "cache.items", "malloc_trim", "_formal_smoke_gate", "rng_states_by_rank", "plan_hash", "dist.barrier", "_mesh_runtime_gradient_audit"):
            self.assertIn(marker, text)
        self.assertIn("AUDIO_SINGLE_PREFIX_TOKENS = AUDIO_TOKENS_PER_CLIP + 1", model)
        self.assertIn("compact_single_audio_prefix: bool = False", model)
        self.assertIn("single_audio_slot_mask", model)
        self.assertIn("audio2_reused", data)
        self.assertIn("def audio_structure", data)
        self.assertIn("def close(self)", data)
        self.assertIn('convert_tokens_to_ids("<|endoftext|>")', data)
        self.assertIn("max_length=max_answer_tokens - 1", data)
        for filename in ("run_audio_partition_smoke20_5_10x2_5_mesh_mellow_5090.sh", "run_audio_partition_formal_5_10x2_5_mesh_mellow_5090.sh"):
            self.assertTrue((ROOT / "code/RSmol" / filename).is_file())

    def test_formal_gate_requires_matching_resume_and_releases(self):
        (gate,) = functions("_formal_smoke_gate")
        inventory = {"root": "/partition", "report_sha256": "abc"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-000020"
            def segment(steps):
                return {"segment": {"steps": steps}, "steps": [{}] * steps, "release": [{"passed": True}] * 8, "cgroup_release": {"anon_drop_bytes": 10, "required_bytes": 8}}
            initial = {"status": "PASS", "mode": "smoke", "training_contract": "component_partitions6_rank_ram_compact_audio_answer_eos_v2", "hard_failures": [], "inventory": inventory, "seed": 3, "first_step_gradient_audit": {"trace_matches_5_10_10_5": True}, "start_cursor": {"segment": 0, "segment_step": 0, "global_step": 0}, "end_cursor": {"segment": 2, "segment_step": 0, "global_step": 20}, "checkpoints": [str(checkpoint)], "segments": [segment(10), segment(10)]}
            resumed = {"status": "PASS", "mode": "smoke", "training_contract": "component_partitions6_rank_ram_compact_audio_answer_eos_v2", "hard_failures": [], "inventory": inventory, "seed": 3, "first_step_gradient_audit": {"trace_matches_5_10_10_5": True}, "start_cursor": initial["end_cursor"], "end_cursor": {"segment": 3, "segment_step": 0, "global_step": 22}, "resume_checkpoint": str(checkpoint.resolve()), "resume_verified_two_steps": True, "segments": [segment(2)]}
            first, second = root / "first.json", root / "second.json"
            first.write_text(json.dumps(initial), encoding="utf-8")
            second.write_text(json.dumps(resumed), encoding="utf-8")
            args = SimpleNamespace(smoke20_report=first, smoke_resume_report=second, seed=3)
            self.assertEqual(gate(args, inventory)["checkpoint20"], str(checkpoint.resolve()))
            resumed["resume_checkpoint"] = "wrong"
            second.write_text(json.dumps(resumed), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                gate(args, inventory)

    def test_answer_termination_reserves_budget_and_deduplicates_tail(self):
        (append_eos,) = data_functions("_append_terminal_endoftext")
        class Tokenizer:
            eos_token_id = 7
            @staticmethod
            def convert_tokens_to_ids(token):
                return 7 if token == "<|endoftext|>" else 99
        rows = append_eos([[1, 2], [3, 7], [4, 7, 7], []], Tokenizer(), max_answer_tokens=3)
        self.assertEqual(rows, [[1, 2, 7], [3, 7], [4, 7], [7]])
        self.assertTrue(all(len(row) <= 3 and row[-1] == 7 for row in rows))


if __name__ == "__main__":
    unittest.main()
