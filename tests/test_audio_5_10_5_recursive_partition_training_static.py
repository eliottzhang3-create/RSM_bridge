"""Dependency-light contracts for fixed-recursive partition-v2 training."""
from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/RSmol/scripts/train_audio_partitioned_5_10_5_recursive_mellow_ddp.py"
SHARED_DATA = ROOT / "code/RSmol/audio_5_10x2_5_mesh_mellow/data.py"
MODEL = ROOT / "code/RSmol/audio_5_10_5_recursive_mellow/model.py"
CONTRACT = "recursive_5_10_5_component_partitions6_rank_ram_compact_audio_answer_eos_v2"
ARCHITECTURE = "logical_30_physical_20_5_10_5_loops_2_no_mesh_audio_mellow"
MAPPER = "mellow_c2l_527x768__concat_cls_frames__projection_768x576x576_biasfree_dropout0.5__cls_preserving_avgpool8"
SCHEDULE = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)


def selected_functions(*names: str):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "Path": Path,
        "json": json,
        "CONTRACT": CONTRACT,
        "CONFIG_FILENAME": "audio_recursive_5_10_5_partition_config.json",
        "RECURSIVE_AUDIO_CONTRACT": ARCHITECTURE,
        "MAPPER_CONTRACT": MAPPER,
        "SMOKE_SEGMENTS": ((2, 10), (0, 10), (1, 2)),
        "LOGICAL_LAYER_COUNT": 30,
        "PHYSICAL_LAYER_COUNT": 20,
        "RECURSIVE_LOOPS": 2,
        "PREFIX_LAYER_COUNT": 5,
        "MIDDLE_LAYER_COUNT": 10,
        "SUFFIX_LAYER_COUNT": 5,
        "LOGICAL_TO_PHYSICAL": SCHEDULE,
        "SOURCE_LAYER_INDICES_0BASED": (0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29),
    }
    module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[]))
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return [namespace[name] for name in names]


class RecursivePartitionContracts(unittest.TestCase):
    def test_runtime_contract_reuses_exact_partition_algorithms(self):
        text = SCRIPT.read_text(encoding="utf-8")
        for marker in (
            "partition_common._inventory",
            "partition_common._quotas",
            "partition_common._order",
            "partition_common._plan",
            "partition_common._RankLocalStoreWaveforms",
            "partition_common._memory",
            "partition_common._trim",
            "SMOKE_SEGMENTS = ((2, 10), (0, 10), (1, 2))",
            "broadcast_buffers=False",
            "find_unused_parameters=False",
            "rng_states_by_rank",
            "plan_hash",
        ):
            self.assertIn(marker, text)

    def test_architecture_and_training_contract_are_fail_closed(self):
        text = SCRIPT.read_text(encoding="utf-8")
        for marker in (
            CONTRACT,
            "RecursiveLlamaForCausalLM.from_pretrained",
            "validate_recursive_5_10_5",
            "exact_fixed_recursive_5_10x2_5",
            "both_middle_loops_have_finite_gradients",
            "all_decoder_layers_have_finite_gradient",
            "compact_prefix_contract",
            "answer_eos_contract",
            '"optimizer_betas": [0.9, 0.95]',
            '"gradient_clip_norm": 0.5',
            '"periodic_validation": False',
            "checkpoint_audio_state_hashes",
            "optimizer_parameter_names",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("RecursiveLlamaForCausalLM.from_pretrained(args.model_path, trust_remote_code=True", text)

    def test_compact_prefix_and_supervised_eos_are_inherited_explicitly(self):
        model = MODEL.read_text(encoding="utf-8")
        data = SHARED_DATA.read_text(encoding="utf-8")
        for marker in (
            "AUDIO_SINGLE_PREFIX_TOKENS",
            "AUDIO_DUAL_PREFIX_TOKENS",
            "AudioSmolLM2Model",
        ):
            self.assertIn(marker, model)
        self.assertIn('convert_tokens_to_ids("<|endoftext|>")', data)
        self.assertIn("max_length=max_answer_tokens - 1", data)

    def test_formal_gate_accepts_only_exact_recursive_20_plus_2(self):
        metadata, gate = selected_functions("_recursive_metadata", "_formal_smoke_gate")
        inventory = {"root": "/partition", "report_sha256": "abc"}
        schedule = [
            {"epoch": 0, "position": 0, "partition_id": 2, "steps": 10},
            {"epoch": 0, "position": 1, "partition_id": 0, "steps": 10},
            {"epoch": 0, "position": 2, "partition_id": 1, "steps": 2},
        ]

        def audit():
            return {
                "exact_fixed_recursive_5_10x2_5": True,
                "forward_trace_matches_exact_5_10x2_5": True,
                "both_middle_loops_have_finite_gradients": True,
                "all_decoder_layers_have_finite_gradient": True,
                "embedding_has_finite_gradient": True,
                "lm_head_has_finite_gradient": True,
                "all_bridge_gradients_finite": True,
                "all_c2l_gradients_finite": True,
                "htsat_frozen_and_gradient_free": True,
                "no_mesh_router_or_memory_parameters": True,
                "compact_prefix_contract": True,
                "answer_eos_contract": True,
                "training_mode_contract": True,
            }

        def segment(row):
            return {"segment": row, "steps": [{}] * row["steps"], "release": [{"passed": True}] * 8, "cgroup_release": {"anon_drop_bytes": 10, "required_bytes": 8}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = [root / "checkpoint-000020", root / "checkpoint-000022"]
            change = {group: {"changed": True, "finite": True, "exact_equal": False, "max_abs_delta": 1e-6} for group in ("text", "bridge", "c2l")}
            change["all_groups_changed"] = True
            for checkpoint, step in zip(checkpoints, (20, 22)):
                checkpoint.mkdir()
                (checkpoint / "checkpoint_complete.json").write_text(json.dumps({"status": "complete", "global_step": step, "contract": CONTRACT}), encoding="utf-8")
                (checkpoint / "audio_recursive_5_10_5_partition_config.json").write_text(json.dumps({
                    "contract": CONTRACT,
                    "mode": "smoke",
                    "architecture_contract": ARCHITECTURE,
                    "mapper_contract": MAPPER,
                    "recursive_text_contract": metadata(),
                    "compact_single_audio_prefix": True,
                    "prefix_tokens": {"single": 130, "dual": 260},
                    "answer_termination": {"token": "<|endoftext|>", "included_in_max_answer_tokens": True, "supervised": True},
                    "inventory": inventory,
                    "schedule": schedule,
                    "epochs": 10,
                    "world_size": 8,
                    "micro_batch_size": 8,
                    "gradient_accumulation_steps": 4,
                    "effective_global_batch_size": 256,
                    "seed": 0,
                    "total_steps": 22,
                    "warmup_steps": 2,
                    "resume_from": str(checkpoints[0].resolve()) if step == 22 else None,
                    "resume_parameter_change_audit": change if step == 22 else None,
                }), encoding="utf-8")
            initial = {"status": "PASS", "mode": "smoke", "training_contract": CONTRACT, "hard_failures": [], "inventory": inventory, "schedule": schedule, "seed": 0, "start_cursor": {"segment": 0, "segment_step": 0, "global_step": 0}, "end_cursor": {"segment": 2, "segment_step": 0, "global_step": 20}, "first_step_gradient_audit": audit(), "segments": [segment(schedule[0]), segment(schedule[1])], "checkpoints": [str(checkpoints[0])]}
            resumed = {"status": "PASS", "mode": "smoke", "training_contract": CONTRACT, "hard_failures": [], "inventory": inventory, "schedule": schedule, "seed": 0, "start_cursor": initial["end_cursor"], "end_cursor": {"segment": 3, "segment_step": 0, "global_step": 22}, "first_step_gradient_audit": audit(), "segments": [segment(schedule[2])], "checkpoints": [str(checkpoints[1])], "resume_checkpoint": str(checkpoints[0].resolve()), "resume_verified_two_steps": True, "resume_parameter_change_audit": change}
            first, second = root / "first.json", root / "second.json"
            first.write_text(json.dumps(initial), encoding="utf-8")
            second.write_text(json.dumps(resumed), encoding="utf-8")
            args = SimpleNamespace(smoke20_report=first, smoke_resume_report=second, seed=0)
            self.assertEqual(gate(args, inventory)["checkpoint20"], str(checkpoints[0].resolve()))
            resumed["first_step_gradient_audit"]["both_middle_loops_have_finite_gradients"] = False
            second.write_text(json.dumps(resumed), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                gate(args, inventory)

    def test_all_submit_wrappers_use_5090(self):
        for name in (
            "run_audio_5_10_5_recursive_mellow_smoke20_5090.sh",
            "run_audio_5_10_5_recursive_mellow_resume2_5090.sh",
            "run_audio_5_10_5_recursive_mellow_formal_5090.sh",
        ):
            text = (ROOT / "code/RSmol" / name).read_text(encoding="utf-8")
            self.assertIn("vc submit -p pdgpu-5090", text)
            self.assertIn("-c 32 -m 256G -g 8 -n 1", text)
            self.assertNotIn("pdgpu-3090", text)


if __name__ == "__main__":
    unittest.main()
