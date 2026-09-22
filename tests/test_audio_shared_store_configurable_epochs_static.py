"""Contracts for the isolated configurable-epoch shared-store training line."""
from __future__ import annotations

import ast
import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"
SCRIPTS = RSMOL / "scripts"
BASE_TRAIN = SCRIPTS / "train_audio_shared_store_5_10x2_5_mesh_mellow_ddp.py"
ENTRY = SCRIPTS / "train_audio_shared_store_configurable_epochs_5_10x2_5_mesh_mellow_ddp.py"
STAGE = SCRIPTS / "stage_audio_shared_store_configurable_epochs_5_10x2_5_mesh_mellow.sh"
SMOKE = SCRIPTS / "train_audio_shared_store_smoke20_configurable_epochs_5_10x2_5_mesh_mellow_ddp.sh"
RESUME = SCRIPTS / "train_audio_shared_store_resume2_configurable_epochs_5_10x2_5_mesh_mellow_ddp.sh"
FORMAL = SCRIPTS / "train_audio_shared_store_formal_configurable_epochs_5_10x2_5_mesh_mellow_ddp.sh"
SUBMITS = (
    RSMOL / "run_audio_shared_store_smoke20_configurable_epochs_5_10x2_5_mesh_mellow_5090.sh",
    RSMOL / "run_audio_shared_store_resume2_configurable_epochs_5_10x2_5_mesh_mellow_5090.sh",
    RSMOL / "run_audio_shared_store_formal_configurable_epochs_5_10x2_5_mesh_mellow_5090.sh",
)
CONTRACT = "node_shared_unique_store_fullshuffle_fixed260_audio_reuse_answer_eos_v2"
AUDIO_SLOT_SEMANTICS = "fixed260_second_slot_reuses_audio1_htsat_embedding_then_runs_bridge_separately"


def _base_functions(*names: str) -> dict[str, Any]:
    tree = ast.parse(BASE_TRAIN.read_text(encoding="utf-8"))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "argparse": argparse,
        "json": json,
        "math": math,
        "TRAINING_CONTRACT": CONTRACT,
        "FORMAL_EPOCHS": 10,
        "CANONICAL_MAX_LR": 1e-3,
        "CANONICAL_MIN_LR": 1e-4,
        "AUDIO_SLOT_SEMANTICS": AUDIO_SLOT_SEMANTICS,
        "PREFIX_TOKENS": {"single": 260, "dual": 260},
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(BASE_TRAIN), "exec"), namespace)
    return namespace


class ConfigurableEpochSharedStoreStaticTest(unittest.TestCase):
    def test_isolated_files_exist_and_original_line_is_unchanged(self) -> None:
        for path in (ENTRY, STAGE, SMOKE, RESUME, FORMAL, *SUBMITS):
            self.assertTrue(path.is_file(), path)
        ast.parse(ENTRY.read_text(encoding="utf-8"))

        original_smoke = (SCRIPTS / "train_audio_shared_store_smoke20_5_10x2_5_mesh_mellow_ddp.sh").read_text(encoding="utf-8")
        original_resume = (SCRIPTS / "train_audio_shared_store_resume2_5_10x2_5_mesh_mellow_ddp.sh").read_text(encoding="utf-8")
        original_formal = (SCRIPTS / "train_audio_shared_store_formal_5_10x2_5_mesh_mellow_ddp.sh").read_text(encoding="utf-8")
        self.assertIn("--epochs 3", original_smoke)
        self.assertIn("--epochs 3", original_resume)
        self.assertIn("--epochs 3", original_formal)
        self.assertIn("FORMAL_EPOCHS = 3", BASE_TRAIN.read_text(encoding="utf-8"))

    def test_entry_requires_positive_explicit_epochs_and_reuses_audited_trainer(self) -> None:
        text = ENTRY.read_text(encoding="utf-8")
        for marker in (
            "import train_audio_shared_store_5_10x2_5_mesh_mellow_ddp as shared_store",
            "if args.epochs is None:",
            "if args.epochs <= 0:",
            "shared_store.FORMAL_EPOCHS = args.epochs",
            "shared_store.run(args)",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("FORMAL_EPOCHS = 10", text)

    def test_ten_epoch_shape_and_warmup_use_real_optimizer_steps(self) -> None:
        namespace = _base_functions("_training_shape")
        args = SimpleNamespace(
            epochs=10, world_size=8, micro_batch_size=8,
            gradient_accumulation_steps=4,
        )
        shape = namespace["_training_shape"](args, 968_059)
        self.assertEqual(shape["global_batch_size"], 256)
        self.assertEqual(shape["steps_per_epoch"], 3_781)
        self.assertEqual(shape["dropped_rows_per_epoch"], 123)
        self.assertEqual(shape["total_steps"], 37_810)
        self.assertEqual(math.ceil(shape["total_steps"] * 0.05), 1_891)

        train = BASE_TRAIN.read_text(encoding="utf-8")
        self.assertIn('default_warmup = math.ceil(shape["total_steps"] * 0.05)', train)
        self.assertIn("if args.warmup_steps != default_warmup:", train)
        self.assertIn("warmup_steps must equal ceil(total_steps * 0.05)", train)

    def test_shell_entries_require_epochs_and_do_not_hardcode_three(self) -> None:
        for path in (SMOKE, RESUME, FORMAL):
            text = path.read_text(encoding="utf-8")
            self.assertIn("--epochs", text)
            self.assertIn("positive integer", text)
            self.assertNotIn("--epochs 3", text)
            self.assertIn("audio_5_10x2_5_mesh_mellow_shared_store_configurable_epochs", text)
            self.assertIn("stage_audio_shared_store_configurable_epochs_5_10x2_5_mesh_mellow.sh", text)
        self.assertIn("--resume-from", RESUME.read_text(encoding="utf-8"))

        stage = STAGE.read_text(encoding="utf-8")
        self.assertIn("train_audio_shared_store_configurable_epochs_5_10x2_5_mesh_mellow_ddp.py", stage)
        for marker in (
            'STAGED_STORE="/dev/shm/rsmol_shared_train_${RUN_ID}"',
            'SHM_MARGIN_KIB=$((10 * 1024 * 1024))',
            'cp -a "$SOURCE_STORE"/. "$STAGED_STORE"/',
            'rm -rf -- "$STAGED_STORE"',
        ):
            self.assertIn(marker, stage)

    def test_ten_epoch_formal_gate_rejects_stale_three_epoch_reports(self) -> None:
        namespace = _base_functions("_json", "_formal_gate")
        inventory = {
            "manifest_sha256": "manifest",
            "index_sha256": "index",
            "waveform_sha256": "waveform",
            "total_waveform_bytes": 123,
        }
        shape = {
            "global_batch_size": 256,
            "steps_per_epoch": 3_781,
            "microbatches_per_epoch": 15_124,
            "dropped_rows_per_epoch": 123,
            "total_steps": 37_810,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "smoke" / "checkpoint-000020"
            checkpoint.mkdir(parents=True)
            common = {
                "status": "PASS",
                "mode": "smoke",
                "training_contract": CONTRACT,
                "hard_failures": [],
                "store_inventory": inventory,
                "epochs": 10,
                "max_lr": 1e-3,
                "min_lr": 1e-4,
                "compact_single_audio_prefix": False,
                "prefix_tokens": {"single": 260, "dual": 260},
                "single_audio_slot_semantics": AUDIO_SLOT_SEMANTICS,
                "shape": shape,
                "warmup_steps": 1_891,
                "warmup_default_ceil_5_percent": 1_891,
                "sampler": {"seed": 0},
                "initialization": {"fresh_source": str(root.resolve())},
                "first_step_gradient_audit": {"trace_matches_5_10_10_5": True},
                "answer_only_label_audit": {"passed": True, "terminal_eos_supervised": True},
            }
            first = {
                **common,
                "start_global_step": 0,
                "end_global_step": 20,
                "checkpoints": [str(checkpoint)],
            }
            resumed = {
                **common,
                "start_global_step": 20,
                "end_global_step": 22,
                "resume_checkpoint": str(checkpoint.resolve()),
                "resume_verified_two_steps": True,
            }
            first_path = root / "first.json"
            resumed_path = root / "resumed.json"
            first_path.write_text(json.dumps(first), encoding="utf-8")
            resumed_path.write_text(json.dumps(resumed), encoding="utf-8")
            args = SimpleNamespace(
                mode="formal",
                smoke20_report=first_path,
                smoke_resume_report=resumed_path,
                warmup_steps=1_891,
                seed=0,
                model_path=root,
            )
            result = namespace["_formal_gate"](args, inventory, shape)
            self.assertEqual(result["checkpoint20"], str(checkpoint.resolve()))

            for key, stale in (
                ("epochs", 3),
                ("shape", {**shape, "total_steps": 11_343}),
                ("warmup_steps", 568),
                ("warmup_default_ceil_5_percent", 568),
            ):
                with self.subTest(key=key):
                    resumed_path.write_text(json.dumps({**resumed, key: stale}), encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        namespace["_formal_gate"](args, inventory, shape)

    def test_submission_topology_is_unchanged(self) -> None:
        for submit in SUBMITS:
            text = submit.read_text(encoding="utf-8")
            for marker in (
                "vc submit", "-p pdgpu-5090", "-c 32", "-m 256G",
                "-g 8", "-n 1", "configurable_epochs",
            ):
                self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()
