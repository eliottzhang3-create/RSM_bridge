"""Dependency-light contracts for the isolated fixed260 silence-slot route."""
from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "code/RSmol/audio_5_10x2_5_mesh_mellow_silence_slot"
MODEL = PACKAGE / "model.py"
DATA = PACKAGE / "data.py"
TRAINER = ROOT / "code/RSmol/scripts/train_audio_partitioned_5_10x2_5_mesh_mellow_silence_slot_ddp.py"
MAIN_MODEL = ROOT / "code/RSmol/audio_5_10x2_5_mesh_mellow/model.py"
MAIN_TRAINER = ROOT / "code/RSmol/scripts/train_audio_partitioned_5_10x2_5_mesh_mellow_ddp.py"
CONTRACT = "component_partitions6_rank_ram_fixed260_runtime_silence_second_slot_answer_eos_v2"
CONFIG_FILENAME = "audio_mesh_fixed260_silence_slot_config.json"
ARCHITECTURE = "logical_30_physical_20_5_10x2_5_mesh_audio_mellow_fixed260_runtime_silence_second_slot"
MAPPER = "mellow_c2l_527x768__concat_cls_frames__projection_768x576x576_biasfree_dropout0.5__cls_preserving_avgpool8"
ANSWER = {"token": "<|endoftext|>", "included_in_max_answer_tokens": True, "supervised": True}
PREFIX = {"single": 260, "dual": 260}
SILENCE = {
    "single_audio_second_slot": "one exact-zero waveform created on GPU per containing microbatch",
    "silence_persisted_or_loaded": False,
    "silence_h2d_transfer": False,
    "silence_shape": [1, 1, 320000],
    "explicit_identical_dual_audio": "reuse audio1 encoder embedding; retain second slot",
    "distinct_dual_audio": "encode real audio2",
    "fixed_prefix_tokens_per_row": 260,
}


def selected_functions(*names: str):
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "Any": object,
        "Path": Path,
        "json": json,
        "CONTRACT": CONTRACT,
        "CONFIG_FILENAME": CONFIG_FILENAME,
        "SILENCE_SLOT_ARCHITECTURE_CONTRACT": ARCHITECTURE,
        "MAPPER_CONTRACT": MAPPER,
        "ANSWER_TERMINATION_CONTRACT": ANSWER,
        "FIXED_PREFIX_TOKEN_CONTRACT": PREFIX,
        "SILENCE_SLOT_CONTRACT": SILENCE,
    }
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return [namespace[name] for name in names]


class SilenceSlotPartitionContracts(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in (PACKAGE / "__init__.py", DATA, MODEL, TRAINER):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_runtime_silence_is_model_only_and_one_per_microbatch(self):
        model = MODEL.read_text(encoding="utf-8")
        data = DATA.read_text(encoding="utf-8")
        self.assertIn("torch.zeros_like(audio1[:1])", model)
        self.assertIn('"runtime_silence_waveforms_created": 1 if silence_count else 0', model)
        self.assertIn("silence_embedding.expand(batch, -1, -1)", model)
        self.assertIn("projected_second = self.bridge(second)", model)
        self.assertNotIn("torch.zeros", data)
        self.assertNotIn("silence.wav", data)
        self.assertIn("silence_second_slot_mask", data)
        self.assertIn("same_real_audio_mask", data)

    def test_three_second_slot_classes_are_disjoint(self):
        model = MODEL.read_text(encoding="utf-8")
        for marker in (
            "distinct_cpu = ~(silence_cpu | same_cpu)",
            "silence_cpu & same_cpu",
            "second = first",
            "second.index_copy",
            "same_real_audio_reused_first_embedding",
            "second_encoder_input_batch_size",
        ):
            self.assertIn(marker, model)

    def test_fixed260_prefix_and_answer_alignment(self):
        model = MODEL.read_text(encoding="utf-8")
        self.assertIn("FIXED_PREFIX_TOKEN_CONTRACT", model)
        self.assertIn("(audio_prefix1, separator, audio_prefix2, separator, text_embeds)", model)
        self.assertIn("prefix_length=prefix_length", model)
        self.assertIn("text_attention_mask.to(device=inputs_embeds.device)", model)
        self.assertIn("fixed silence-slot prefix must be 260 tokens", model)
        self.assertNotIn("if not bool(single_audio_slot_mask_cpu[row])", model)

    def test_checkpoint_and_smoke_gate_are_route_isolated(self):
        trainer = TRAINER.read_text(encoding="utf-8")
        expected_contract = (
            "component_partitions6_rank_ram_fixed260_runtime_silence_"
            "second_slot_answer_eos_v2"
        )
        for marker in (
            expected_contract,
            'CONFIG_FILENAME = "audio_mesh_fixed260_silence_slot_config.json"',
            '"compact_single_audio_prefix": False',
            '"prefix_tokens": FIXED_PREFIX_TOKEN_CONTRACT',
            '"silence_slot": SILENCE_SLOT_CONTRACT',
            'parser.add_argument("--min-lr", type=float, default=1e-4)',
            '"min_lr": 1e-4',
            "formal gate rejects",
            "_validate_resume_artifact",
            "rng_states_by_rank",
            "plan_hash",
            "partition_common._trim()",
            "cache._get_audio_id",
            "runtime_silence_encoder_calls_on_rank",
            "runtime_silence_rows_on_rank0",
            "global_distinct_real_dual_audio_rows",
            "_audio_runtime_gradient_audit",
            "all_bridge_gradients_finite",
            "all_c2l_gradients_finite",
            "htsat_frozen_and_gradient_free",
            "answer_labels_start_after_fixed260_prefix",
            "max_cuda_memory_allocated_bytes",
        ):
            self.assertIn(marker, trainer)
        self.assertNotIn(expected_contract, MAIN_TRAINER.read_text(encoding="utf-8"))
        self.assertNotIn("runtime_silence", MAIN_MODEL.read_text(encoding="utf-8"))

    def test_launchers_are_complete_and_use_5090(self):
        names = (
            "run_audio_partition_smoke20_5_10x2_5_mesh_mellow_silence_slot_5090.sh",
            "run_audio_partition_resume2_5_10x2_5_mesh_mellow_silence_slot_5090.sh",
            "run_audio_partition_formal_5_10x2_5_mesh_mellow_silence_slot_5090.sh",
        )
        for name in names:
            text = (ROOT / "code/RSmol" / name).read_text(encoding="utf-8")
            self.assertIn("vc submit -p pdgpu-5090", text)
            self.assertIn("-c 32 -m 256G -g 8 -n 1", text)
            self.assertIn("silence_slot", text)
        for phase in ("smoke20", "resume2", "formal"):
            path = ROOT / "code/RSmol/scripts" / f"train_audio_partition_{phase}_5_10x2_5_mesh_mellow_silence_slot_ddp.sh"
            self.assertTrue(path.is_file())
            text = path.read_text(encoding="utf-8")
            self.assertIn("--epochs 10", text)
            self.assertIn("--micro-batch-size 8 --gradient-accumulation-steps 4", text)

    def test_formal_gate_accepts_only_own_covered_20_plus_2_reports(self):
        validate, gate = selected_functions("_validate_resume_artifact", "_formal_smoke_gate")
        inventory = {"root": "/partition", "partitions": [0, 1, 2]}

        def segment(steps: int):
            return {
                "segment": {"steps": steps},
                "steps": [{}] * steps,
                "release": [{"passed": True}] * 8,
                "cgroup_release": {"anon_drop_bytes": 9, "required_bytes": 8},
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-000020"
            for relative in ("mesh_model/config.json", "tokenizer/tokenizer_config.json"):
                target = checkpoint / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("{}", encoding="utf-8")
            for name in ("audio_bridge.pt", "training_state.pt"):
                (checkpoint / name).write_bytes(b"audit")
            config = {
                "contract": CONTRACT,
                "architecture_contract": ARCHITECTURE,
                "mapper_contract": MAPPER,
                "compact_single_audio_prefix": False,
                "prefix_tokens": PREFIX,
                "answer_termination": ANSWER,
                "silence_slot": SILENCE,
            }
            (checkpoint / CONFIG_FILENAME).write_text(json.dumps(config), encoding="utf-8")
            (checkpoint / "checkpoint_complete.json").write_text(
                json.dumps({"status": "complete", "contract": CONTRACT, "global_step": 20}),
                encoding="utf-8",
            )
            self.assertEqual(validate(checkpoint)["contract"], CONTRACT)
            initial = {
                "status": "PASS", "mode": "smoke", "training_contract": CONTRACT,
                "hard_failures": [], "inventory": inventory, "seed": 0,
                "start_cursor": {"segment": 0, "segment_step": 0, "global_step": 0},
                "end_cursor": {"segment": 2, "segment_step": 0, "global_step": 20},
                "prefix_contract": PREFIX, "silence_slot_contract": SILENCE,
                "first_step_gradient_audit": {
                    "trace_matches_5_10_10_5": True,
                    "fixed260_silence_slot_contract": True,
                    "all_bridge_gradients_finite": True,
                    "all_c2l_gradients_finite": True,
                    "htsat_frozen_and_gradient_free": True,
                    "fixed260_prefix_observed": True,
                    "answer_labels_start_after_fixed260_prefix": True,
                },
                "slot_audit_totals": {
                    "global_single_audio_rows": 2000, "global_dual_audio_rows": 3120,
                    "global_same_real_dual_audio_rows": 20, "global_distinct_real_dual_audio_rows": 3100,
                    "runtime_silence_encoder_calls_on_rank0": 70, "runtime_silence_rows_on_rank0": 250,
                },
                "segments": [segment(10), segment(10)], "checkpoints": [str(checkpoint)],
            }
            resumed = {
                "status": "PASS", "mode": "smoke", "training_contract": CONTRACT,
                "hard_failures": [], "inventory": inventory, "seed": 0,
                "start_cursor": initial["end_cursor"],
                "end_cursor": {"segment": 3, "segment_step": 0, "global_step": 22},
                "prefix_contract": PREFIX, "silence_slot_contract": SILENCE,
                "first_step_gradient_audit": dict(initial["first_step_gradient_audit"]),
                "slot_audit_totals": {
                    "global_single_audio_rows": 200, "global_dual_audio_rows": 312,
                    "global_same_real_dual_audio_rows": 2, "global_distinct_real_dual_audio_rows": 310,
                    "runtime_silence_encoder_calls_on_rank0": 7, "runtime_silence_rows_on_rank0": 25,
                },
                "segments": [segment(2)], "resume_checkpoint": str(checkpoint.resolve()),
                "resume_verified_two_steps": True,
            }
            first_path, resumed_path = root / "smoke.json", root / "resume.json"
            first_path.write_text(json.dumps(initial), encoding="utf-8")
            resumed_path.write_text(json.dumps(resumed), encoding="utf-8")
            args = SimpleNamespace(smoke20_report=first_path, smoke_resume_report=resumed_path, seed=0)
            self.assertEqual(gate(args, inventory)["checkpoint20"], str(checkpoint.resolve()))
            resumed["training_contract"] = "component_partitions6_rank_ram_compact_audio_answer_eos_v2"
            resumed_path.write_text(json.dumps(resumed), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                gate(args, inventory)


if __name__ == "__main__":
    unittest.main()
