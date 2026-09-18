"""Dependency-light contracts for the SmolLM2 partition-v2 comparison route."""
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
SCRIPT = ROOT / "code/RSmol/scripts/train_audio_partitioned_smollm2_135m_mellow_ddp.py"
MESH_SCRIPT = ROOT / "code/RSmol/scripts/train_audio_partitioned_5_10x2_5_mesh_mellow_ddp.py"
MODEL = ROOT / "code/RSmol/audio_smollm2_135m_mellow/model.py"
DATA = ROOT / "code/RSmol/audio_5_10x2_5_mesh_mellow/data.py"
CONTRACT = "smollm2_component_partitions6_rank_ram_compact_audio_answer_eos_v2"


def selected_functions(path: Path, *names: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "math": math,
        "random": random,
        "hashlib": hashlib,
        "json": json,
        "Path": Path,
        "CONTRACT": CONTRACT,
        "CONFIG_FILENAME": "audio_smollm2_partition_config.json",
        "ORIGINAL_SMOLLM2_CONTRACT": "original_smollm2_135m_standard_llama_30_layers_hidden576_audio_mellow",
        "MAPPER_CONTRACT": "mellow_c2l_527x768__concat_cls_frames__projection_768x576x576_biasfree_dropout0.5__cls_preserving_avgpool8",
        "SMOKE_SEGMENTS": ((2, 10), (0, 10), (1, 2)),
    }
    module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return [namespace[name] for name in names]


class FakeDataset:
    def __init__(self, rows: int):
        self.rows = rows

    def __len__(self):
        return self.rows


class SmolLM2PartitionContracts(unittest.TestCase):
    def test_schedule_and_plan_are_identical_to_mesh_partition_route(self):
        smol_quota, smol_order, smol_plan = selected_functions(SCRIPT, "_quotas", "_order", "_plan")
        mesh_quota, mesh_order, mesh_plan = selected_functions(MESH_SCRIPT, "_quotas", "_order", "_plan")
        counts = [278680, 137874, 137873, 137874, 137874, 137884]
        self.assertEqual(smol_quota(counts, 10, 256), mesh_quota(counts, 10, 256))
        for epoch in range(10):
            self.assertEqual(smol_order(0, epoch), mesh_order(0, epoch))
            self.assertNotEqual(smol_order(0, epoch)[-1], 0)
        smol_chunks, smol_audit = smol_plan(FakeDataset(1030), seed=7, epoch=2, pid=0, steps=4, global_batch=256)
        mesh_chunks, mesh_audit = mesh_plan(FakeDataset(1030), seed=7, epoch=2, pid=0, steps=4, global_batch=256)
        self.assertEqual(smol_chunks, mesh_chunks)
        self.assertEqual(smol_audit, mesh_audit)

    def test_partition_runtime_contract_is_complete_and_architecture_isolated(self):
        text = SCRIPT.read_text(encoding="utf-8")
        for marker in (
            CONTRACT,
            "SMOKE_SEGMENTS = ((2, 10), (0, 10), (1, 2))",
            "cache._get_audio_id",
            "materialize_from_cached_metadata",
            "malloc_trim",
            "release_min_fraction",
            "cgroup_release",
            "rng_states_by_rank",
            "plan_hash",
            "_formal_smoke_gate",
            "standard_30_layer_smollm2",
            "all_decoder_layers_have_finite_gradient",
            "answer_eos_contract",
            "resume_parameter_change_audit",
            '"optimizer_betas": [0.9, 0.95]',
            '"gradient_clip_norm": 0.5',
            '"effective_global_batch_size"',
            '"periodic_validation": False',
            "partition baseline requires epochs=10, max_lr=1e-3, min_lr=0, save_every=500, retention=4, seed=0",
            "refusing nonempty output directory",
            "--tokenizer-path cannot be combined with --resume-from",
            "resume output-dir must be separate from the source checkpoint and its parent",
            "broadcast_buffers=False",
            "find_unused_parameters=False",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("RecursiveLlamaForCausalLM", text)
        self.assertNotIn("write_routers", text)
        self.assertNotIn("read_routers", text)

    def test_compact_prefix_and_eos_are_explicit(self):
        model = MODEL.read_text(encoding="utf-8")
        data = DATA.read_text(encoding="utf-8")
        for marker in (
            "AUDIO_SINGLE_PREFIX_TOKENS = AUDIO_TOKENS_PER_CLIP + 1",
            "AUDIO_DUAL_PREFIX_TOKENS = AUDIO_PREFIX_TOKENS",
            "compact_single_audio_prefix",
            "single_audio_slot_mask",
            "skip_second_prefix",
            "last_prefix_lengths",
        ):
            self.assertIn(marker, model)
        self.assertIn('convert_tokens_to_ids("<|endoftext|>")', data)
        self.assertIn("max_length=max_answer_tokens - 1", data)

    def test_formal_gate_accepts_only_matching_baseline_v2_reports(self):
        (gate,) = selected_functions(SCRIPT, "_formal_smoke_gate")
        inventory = {"root": "/partition", "report_sha256": "abc"}

        def audit():
            return {
                "standard_30_layer_smollm2": True,
                "all_decoder_layers_have_finite_gradient": True,
                "embedding_has_finite_gradient": True,
                "lm_head_has_finite_gradient": True,
                "all_bridge_gradients_finite": True,
                "all_c2l_gradients_finite": True,
                "htsat_frozen_and_gradient_free": True,
                "has_router_parameters": False,
                "compact_prefix_contract": True,
                "answer_eos_contract": True,
                "training_mode_contract": True,
            }

        schedule = [
            {"epoch": 0, "position": 0, "partition_id": 2, "steps": 10},
            {"epoch": 0, "position": 1, "partition_id": 0, "steps": 10},
            {"epoch": 0, "position": 2, "partition_id": 1, "steps": 2},
        ]

        def segment(schedule_row):
            return {"segment": schedule_row, "steps": [{}] * schedule_row["steps"], "release": [{"passed": True}] * 8, "cgroup_release": {"anon_drop_bytes": 10, "required_bytes": 8}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-000020"
            checkpoint22 = root / "checkpoint-000022"
            for checkpoint_path, step in ((checkpoint, 20), (checkpoint22, 22)):
                checkpoint_path.mkdir()
                (checkpoint_path / "checkpoint_complete.json").write_text(json.dumps({"status": "complete", "global_step": step, "contract": CONTRACT}), encoding="utf-8")
                (checkpoint_path / "audio_smollm2_partition_config.json").write_text(json.dumps({
                    "contract": CONTRACT,
                    "mode": "smoke",
                    "architecture_contract": "original_smollm2_135m_standard_llama_30_layers_hidden576_audio_mellow",
                    "mapper_contract": "mellow_c2l_527x768__concat_cls_frames__projection_768x576x576_biasfree_dropout0.5__cls_preserving_avgpool8",
                    "compact_single_audio_prefix": True,
                    "prefix_tokens": {"single": 130, "dual": 260},
                    "answer_termination": {"token": "<|endoftext|>", "included_in_max_answer_tokens": True, "supervised": True},
                }), encoding="utf-8")
            initial = {"status": "PASS", "mode": "smoke", "training_contract": CONTRACT, "hard_failures": [], "inventory": inventory, "schedule": schedule, "seed": 0, "first_step_gradient_audit": audit(), "start_cursor": {"segment": 0, "segment_step": 0, "global_step": 0}, "end_cursor": {"segment": 2, "segment_step": 0, "global_step": 20}, "checkpoints": [str(checkpoint)], "segments": [segment(schedule[0]), segment(schedule[1])]}
            change = {"all_groups_changed": True}
            change.update({group: {"changed": True, "finite": True, "exact_equal": False, "max_abs_delta": 1e-6} for group in ("text", "bridge", "c2l")})
            resumed = {"status": "PASS", "mode": "smoke", "training_contract": CONTRACT, "hard_failures": [], "inventory": inventory, "schedule": schedule, "seed": 0, "first_step_gradient_audit": audit(), "start_cursor": initial["end_cursor"], "end_cursor": {"segment": 3, "segment_step": 0, "global_step": 22}, "resume_checkpoint": str(checkpoint.resolve()), "resume_verified_two_steps": True, "resume_parameter_change_audit": change, "checkpoints": [str(checkpoint22)], "segments": [segment(schedule[2])]}
            first, second = root / "first.json", root / "second.json"
            first.write_text(json.dumps(initial), encoding="utf-8")
            second.write_text(json.dumps(resumed), encoding="utf-8")
            args = SimpleNamespace(smoke20_report=first, smoke_resume_report=second, seed=0)
            self.assertEqual(gate(args, inventory)["checkpoint20"], str(checkpoint.resolve()))
            resumed["training_contract"] = "component_partitions6_rank_ram_compact_audio_answer_eos_v2"
            second.write_text(json.dumps(resumed), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                gate(args, inventory)

    def test_submission_wrappers_use_3090_queue_and_exact_resources(self):
        for name in (
            "run_audio_smollm2_135m_mellow_smoke20_3090.sh",
            "run_audio_smollm2_135m_mellow_resume2_3090.sh",
            "run_audio_smollm2_135m_mellow_formal_3090.sh",
        ):
            text = (ROOT / "code/RSmol" / name).read_text(encoding="utf-8")
            self.assertIn("vc submit -p pdgpu-3090", text)
            self.assertIn("-c 32 -m 256G -g 8 -n 1", text)
            self.assertNotIn("pdgpu-5090", text)


if __name__ == "__main__":
    unittest.main()
