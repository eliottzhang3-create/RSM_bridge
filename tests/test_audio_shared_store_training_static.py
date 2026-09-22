"""Dependency-light contracts for isolated node-shared Audio MeSH training."""
from __future__ import annotations

import ast
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"
PACKAGE = RSMOL / "audio_5_10x2_5_mesh_mellow_shared_store"
TRAIN = RSMOL / "scripts" / "train_audio_shared_store_5_10x2_5_mesh_mellow_ddp.py"
STAGE = RSMOL / "scripts" / "stage_audio_shared_store_5_10x2_5_mesh_mellow.sh"
SMOKE = RSMOL / "scripts" / "train_audio_shared_store_smoke20_5_10x2_5_mesh_mellow_ddp.sh"
RESUME = RSMOL / "scripts" / "train_audio_shared_store_resume2_5_10x2_5_mesh_mellow_ddp.sh"
FORMAL = RSMOL / "scripts" / "train_audio_shared_store_formal_5_10x2_5_mesh_mellow_ddp.sh"
SUBMITS = (
    RSMOL / "run_audio_shared_store_smoke20_5_10x2_5_mesh_mellow_5090.sh",
    RSMOL / "run_audio_shared_store_resume2_5_10x2_5_mesh_mellow_5090.sh",
    RSMOL / "run_audio_shared_store_formal_5_10x2_5_mesh_mellow_5090.sh",
)
CONTRACT = "node_shared_unique_store_fullshuffle_compact_audio_answer_eos_v2"


class AudioSharedStoreTrainingStaticTest(unittest.TestCase):
    def test_isolated_files_exist_and_python_parses(self) -> None:
        for path in (PACKAGE / "__init__.py", PACKAGE / "data.py", PACKAGE / "model.py",
                     PACKAGE / "README.md", TRAIN, STAGE, SMOKE, RESUME, FORMAL, *SUBMITS):
            self.assertTrue(path.is_file(), path)
        ast.parse(TRAIN.read_text(encoding="utf-8"))
        self.assertIn(CONTRACT, (PACKAGE / "__init__.py").read_text(encoding="utf-8"))

    def test_route_does_not_import_or_mutate_partition_training(self) -> None:
        combined = "\n".join(path.read_text(encoding="utf-8") for path in (
            PACKAGE / "data.py", PACKAGE / "model.py", TRAIN, STAGE, SMOKE, RESUME, FORMAL,
        ))
        self.assertNotIn("train_audio_partitioned_5_10x2_5_mesh_mellow_ddp", combined)
        self.assertNotIn("component_partitions6", combined)
        self.assertNotIn("partition_schedule", combined)
        self.assertIn("import train_audio_5_10x2_5_mesh_mellow_ddp as base", combined)

    def test_complete_store_is_staged_once_and_strictly_audited(self) -> None:
        stage = STAGE.read_text(encoding="utf-8")
        train = TRAIN.read_text(encoding="utf-8")
        for marker in (
            "rsmol_reasonaqa_train_unique_waveforms_32k_10s_f32_v3",
            'STAGED_STORE="/dev/shm/rsmol_shared_train_${RUN_ID}"',
            'SHM_MARGIN_KIB=$((10 * 1024 * 1024))',
            'cp -a "$SOURCE_STORE"/. "$STAGED_STORE"/',
            'cp "$SOURCE_MANIFEST" "$STAGED_MANIFEST"',
            "staged waveform byte-size mismatch",
            "staged manifest/index SHA256 mismatch",
            '"$STAGED_STORE" == /dev/shm/rsmol_shared_train_*',
            'rm -rf -- "$STAGED_STORE"',
        ):
            self.assertIn(marker, stage)
        for marker in (
            'Path("/dev/shm") not in staged.parents',
            '"status": "PASS"',
            '"format": "manifest_unique_fixed_waveform_store_v1"',
            'if (root / "BUILDING").exists()',
            "source/staged metadata SHA256 audit failed",
            '"one complete immutable store staged in node-shared /dev/shm; no rank-local full-store clone"',
            "DDP ranks do not see one shared staged waveform inode",
            '"excluded_from_optimizer_step_timing": True',
        ):
            self.assertIn(marker, train)

    def test_full_manifest_shuffle_and_fixed_topology(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        for marker in (
            "DistributedSampler(", "shuffle=True", "drop_last=True",
            "sampler.set_epoch(epoch)", '"entire_manifest_each_epoch": True',
            "world != 8", "args.micro_batch_size != 8",
            "args.gradient_accumulation_steps != 4", "args.num_workers != 0",
            '"global_batch_size": global_batch',
        ):
            self.assertIn(marker, text)
        self.assertNotIn("ShardAwareDistributedBatchSampler", text)
        self.assertNotIn("RankLocal", text)

    def test_complete_manifest_step_budget_is_37810(self) -> None:
        tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                    and item.name == "_training_shape")
        namespace = {
            "argparse": argparse,
            "SMOKE_TOTAL_STEPS": 22,
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAIN), "exec"), namespace)
        formal_args = SimpleNamespace(
            mode="formal", epochs=10, world_size=8,
            micro_batch_size=8, gradient_accumulation_steps=4,
        )
        formal = namespace["_training_shape"](formal_args, 968_059)
        self.assertEqual(formal["global_batch_size"], 256)
        self.assertEqual(formal["steps_per_epoch"], 3_781)
        self.assertEqual(formal["total_steps"], 37_810)
        self.assertEqual(formal["dropped_rows_per_epoch"], 123)
        smoke_args = SimpleNamespace(
            mode="smoke", epochs=None, world_size=8,
            micro_batch_size=8, gradient_accumulation_steps=4,
        )
        self.assertEqual(namespace["_training_shape"](smoke_args, 968_059)["total_steps"], 22)

    def test_model_and_loss_contract_is_unchanged(self) -> None:
        train = TRAIN.read_text(encoding="utf-8")
        model = (PACKAGE / "model.py").read_text(encoding="utf-8")
        data = (PACKAGE / "data.py").read_text(encoding="utf-8")
        for marker in (
            "AUDIO_SINGLE_PREFIX_TOKENS", "AUDIO_DUAL_PREFIX_TOKENS",
            "AUDIO_TOKENS_PER_CLIP", "MESH_HIDDEN_SIZE", "AudioMeshModel",
        ):
            self.assertIn(marker, model)
        self.assertIn("ReasonAQADataset", data)
        self.assertIn("collate_reasonaqa", data)
        for marker in (
            '"token": "<|endoftext|>"', '"supervised": True',
            "_answer_label_audit", "answer-only label mask",
            "_mesh_runtime_gradient_audit(owner, require_router_stats=False)",
            '"trace_matches_5_10_10_5"',
        ):
            self.assertIn(marker, train)

    def test_smoke_resume_and_epoch_boundary_are_exact(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        for marker in (
            "SMOKE_FIRST_STOP = 20", "SMOKE_TOTAL_STEPS = 22",
            '{"epoch": 0, "batch_in_epoch": 80, "global_step": 20}',
            'cursor["global_step"] in {20, 22}',
            'f"checkpoint-{cursor[\'global_step\']:06d}"',
            "rng_states_by_rank", "_restore_rng_state",
            '"global_step": cursor["global_step"]',
            'int(config.get("batch_in_epoch", -1)) != cursor["batch_in_epoch"]',
            "Construct DDP before restoring RNG",
            "epoch_completed = False", "epoch_completed = True",
            "if epoch_completed:",
        ):
            self.assertIn(marker, text)
        self.assertLess(text.index("ddp = DDP("), text.index("cursor = _resume("))
        self.assertIn("resume2 requires --resume-from", RESUME.read_text(encoding="utf-8"))

    def test_formal_gate_accepts_only_matching_20_plus_2_reports(self) -> None:
        tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in {"_json", "_formal_gate"}]
        namespace = {
            "Any": Any,
            "Path": Path,
            "json": json,
            "TRAINING_CONTRACT": CONTRACT,
        }
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(TRAIN), "exec"), namespace)
        inventory = {
            "manifest_sha256": "manifest", "index_sha256": "index",
            "waveform_sha256": "waveform", "total_waveform_bytes": 123,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "smoke" / "checkpoint-000020"
            checkpoint.mkdir(parents=True)
            common = {
                "status": "PASS", "mode": "smoke", "training_contract": CONTRACT,
                "hard_failures": [], "store_inventory": inventory,
                "first_step_gradient_audit": {"trace_matches_5_10_10_5": True},
                "answer_only_label_audit": {"passed": True},
            }
            first = {**common, "start_global_step": 0, "end_global_step": 20,
                     "checkpoints": [str(checkpoint)]}
            resumed = {**common, "start_global_step": 20, "end_global_step": 22,
                       "resume_checkpoint": str(checkpoint.resolve()),
                       "resume_verified_two_steps": True}
            first_path, resumed_path = root / "first.json", root / "resumed.json"
            first_path.write_text(json.dumps(first), encoding="utf-8")
            resumed_path.write_text(json.dumps(resumed), encoding="utf-8")
            args = SimpleNamespace(mode="formal", smoke20_report=first_path,
                                   smoke_resume_report=resumed_path)
            result = namespace["_formal_gate"](args, inventory)
            self.assertEqual(result["checkpoint20"], str(checkpoint.resolve()))
            resumed["start_global_step"] = 19
            resumed_path.write_text(json.dumps(resumed), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                namespace["_formal_gate"](args, inventory)

    def test_formal_defaults_and_5090_submission_are_isolated(self) -> None:
        formal = FORMAL.read_text(encoding="utf-8")
        for marker in ("--epochs 10", "--max-lr 1e-3", "--min-lr 0",
                       "--save-every 500", "--checkpoint-retention 4"):
            self.assertIn(marker, formal)
        self.assertIn("--smoke20-report", TRAIN.read_text(encoding="utf-8"))
        self.assertIn("--smoke-resume-report", TRAIN.read_text(encoding="utf-8"))
        for submit in SUBMITS:
            text = submit.read_text(encoding="utf-8")
            for marker in ("vc submit", "-p pdgpu-5090", "-c 32", "-m 256G", "-g 8", "-n 1"):
                self.assertIn(marker, text)
            self.assertIn("shared-store", text)
        train = TRAIN.read_text(encoding="utf-8")
        self.assertIn('"mesh_model_path": str(args.model_path.resolve())', train)
        self.assertIn('"mapper_initialization": "random_c2l_and_xavier_projection"', train)
        for path in (SMOKE, RESUME, FORMAL):
            self.assertIn("audio_5_10x2_5_mesh_mellow_shared_store", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
