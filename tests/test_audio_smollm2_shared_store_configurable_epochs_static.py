"""Static contracts for the isolated SmolLM2 shared-store training line."""
from __future__ import annotations

import argparse
import ast
import math
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"
SCRIPTS = RSMOL / "scripts"
PACKAGE = RSMOL / "audio_smollm2_135m_mellow_shared_store"
BASE_MODEL = RSMOL / "audio_smollm2_135m_mellow" / "model.py"
TRAIN = SCRIPTS / "train_audio_smollm2_shared_store_135m_mellow_ddp.py"
ENTRY = SCRIPTS / "train_audio_smollm2_shared_store_configurable_epochs_135m_mellow_ddp.py"
STAGE = SCRIPTS / "stage_audio_smollm2_shared_store_configurable_epochs_135m_mellow.sh"
SMOKE = SCRIPTS / "train_audio_smollm2_shared_store_smoke20_configurable_epochs_135m_mellow_ddp.sh"
RESUME = SCRIPTS / "train_audio_smollm2_shared_store_resume2_configurable_epochs_135m_mellow_ddp.sh"
FORMAL = SCRIPTS / "train_audio_smollm2_shared_store_formal_configurable_epochs_135m_mellow_ddp.sh"
SUBMITS = (
    RSMOL / "run_audio_smollm2_shared_store_smoke20_configurable_epochs_135m_mellow_3090.sh",
    RSMOL / "run_audio_smollm2_shared_store_resume2_configurable_epochs_135m_mellow_3090.sh",
    RSMOL / "run_audio_smollm2_shared_store_formal_configurable_epochs_135m_mellow_3090.sh",
)
CONTRACT = (
    "smollm2_node_shared_unique_store_fullshuffle_"
    "fixed260_audio_reuse_answer_eos_v2"
)


class SmolLM2SharedStoreConfigurableEpochsStaticTest(unittest.TestCase):
    def test_isolated_surface_exists_and_parses(self) -> None:
        paths = (
            PACKAGE / "__init__.py",
            PACKAGE / "data.py",
            PACKAGE / "model.py",
            PACKAGE / "README.md",
            TRAIN,
            ENTRY,
            STAGE,
            SMOKE,
            RESUME,
            FORMAL,
            *SUBMITS,
        )
        for path in paths:
            self.assertTrue(path.is_file(), path)
        ast.parse(TRAIN.read_text(encoding="utf-8"))
        ast.parse(ENTRY.read_text(encoding="utf-8"))
        self.assertIn(CONTRACT, (PACKAGE / "__init__.py").read_text(encoding="utf-8"))

    def test_route_uses_original_smollm2_without_mesh_or_recursion(self) -> None:
        train = TRAIN.read_text(encoding="utf-8")
        package = "\n".join(
            (PACKAGE / name).read_text(encoding="utf-8")
            for name in ("__init__.py", "data.py", "model.py")
        )
        for marker in (
            "import train_audio_smollm2_135m_mellow_ddp as baseline",
            "ORIGINAL_SMOLLM2_CONTRACT",
            '"standard_30_layer_smollm2"',
            '"has_router_parameters"',
            '"architecture_audit": "standard 30-layer SmolLM2',
        ):
            self.assertIn(marker, train)
        self.assertIn("audio_smollm2_135m_mellow.model", package)
        for forbidden in (
            "train_audio_partitioned_smollm2_135m_mellow_ddp",
            "AudioMeshModel",
            "mesh_checkpoint",
            "component_partitions6",
            "recursive",
        ):
            self.assertNotIn(forbidden, train)

    def test_thirty_epoch_shape_and_real_five_percent_warmup(self) -> None:
        tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
        node = next(
            item for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "_training_shape"
        )
        namespace = {"argparse": argparse, "FORMAL_EPOCHS": 30}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAIN), "exec"), namespace)
        args = SimpleNamespace(
            epochs=30,
            world_size=8,
            micro_batch_size=8,
            gradient_accumulation_steps=4,
        )
        shape = namespace["_training_shape"](args, 968_059)
        self.assertEqual(shape["global_batch_size"], 256)
        self.assertEqual(shape["steps_per_epoch"], 3_781)
        self.assertEqual(shape["dropped_rows_per_epoch"], 123)
        self.assertEqual(shape["total_steps"], 113_430)
        self.assertEqual(math.ceil(shape["total_steps"] * 0.05), 5_672)
        text = TRAIN.read_text(encoding="utf-8")
        self.assertIn('default_warmup = math.ceil(shape["total_steps"] * 0.05)', text)
        self.assertIn("if args.warmup_steps != default_warmup:", text)

    def test_configurable_entry_and_shells_require_explicit_epochs(self) -> None:
        entry = ENTRY.read_text(encoding="utf-8")
        for marker in (
            "if args.epochs is None:",
            "if args.epochs <= 0:",
            "shared_store.FORMAL_EPOCHS = args.epochs",
            "shared_store.run(args)",
        ):
            self.assertIn(marker, entry)
        for path in (SMOKE, RESUME, FORMAL):
            text = path.read_text(encoding="utf-8")
            self.assertIn("--epochs", text)
            self.assertIn("positive integer", text)
            self.assertIn("audio_smollm2_135m_mellow_shared_store_configurable_epochs", text)
            self.assertNotIn("--epochs 10", text)
            self.assertNotIn("--epochs 30", text)
        self.assertIn("--resume-from", RESUME.read_text(encoding="utf-8"))

    def test_single_audio_embedding_reuse_and_two_bridge_calls(self) -> None:
        tree = ast.parse(BASE_MODEL.read_text(encoding="utf-8"))
        cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AudioSmolLM2Model"
        )
        node = next(
            node for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "encode_audio"
        )
        namespace = {
            "torch": SimpleNamespace(Tensor=object),
            "record_function": lambda _: nullcontext(),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(BASE_MODEL), "exec"), namespace)
        wave1 = SimpleNamespace(ndim=3, data_ptr=lambda: 1)
        embedding, slot1, slot2 = object(), object(), object()
        owner = SimpleNamespace(
            _waveform_embedding=Mock(return_value=embedding),
            bridge=Mock(side_effect=[slot1, slot2]),
        )
        result = namespace["encode_audio"](owner, wave1, None, None)
        owner._waveform_embedding.assert_called_once_with(wave1)
        self.assertEqual(owner.bridge.call_args_list, [call(embedding), call(embedding)])
        self.assertEqual(result, (slot1, slot2))

        wave2 = SimpleNamespace(ndim=3, data_ptr=lambda: 2)
        first, second = object(), object()
        owner = SimpleNamespace(
            _waveform_embedding=Mock(side_effect=[first, second]),
            bridge=Mock(side_effect=[slot1, slot2]),
        )
        namespace["encode_audio"](owner, wave1, wave2, None)
        self.assertEqual(owner._waveform_embedding.call_args_list, [call(wave1), call(wave2)])
        self.assertEqual(owner.bridge.call_args_list, [call(first), call(second)])

    def test_training_audits_fixed260_reuse_eos_and_resume(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        for marker in (
            "SMOKE_FIRST_STOP = 20",
            "SMOKE_TOTAL_STEPS = 22",
            "FORMAL_EPOCHS = 30",
            'args.compact_single_audio_prefix = False',
            'single_slot_mask & ~reused_mask',
            'model.last_audio_tokens_per_clip != (AUDIO_TOKENS_PER_CLIP, AUDIO_TOKENS_PER_CLIP)',
            '"terminal_eos_supervised": True',
            '"two_bridge_invocations_contract": True',
            "def _dataset_audio_slot_audit(",
            '"dataset_audio_slot_audit": dataset_audio_slot_audit',
            '"dual_audio_contract": "preserve both explicit waveform slots"',
            "resume_parameter_change_audit",
            "baseline._validate_parameter_change_audit",
            'expected_mode="smoke"',
            "Construct DDP before restoring RNG",
        ):
            self.assertIn(marker, text)

    def test_store_staging_and_3090_submission_contract(self) -> None:
        stage = STAGE.read_text(encoding="utf-8")
        for marker in (
            'STAGED_STORE="/dev/shm/rsmol_smollm2_shared_train_${RUN_ID}"',
            'RSMOL_SMOLLM2_SHARED_STORE_RUN_ID',
            'SHM_MARGIN_KIB=$((10 * 1024 * 1024))',
            'cp -a "$SOURCE_STORE"/. "$STAGED_STORE"/',
            "staged manifest/index SHA256 mismatch",
            '"$STAGED_STORE" == /dev/shm/rsmol_smollm2_shared_train_*',
            'rm -rf -- "$STAGED_STORE"',
            "train_audio_smollm2_shared_store_configurable_epochs_135m_mellow_ddp.py",
        ):
            self.assertIn(marker, stage)
        for submit in SUBMITS:
            text = submit.read_text(encoding="utf-8")
            for marker in (
                "vc submit",
                "-p pdgpu-3090",
                "-c 32",
                "-m 256G",
                "-g 8",
                "-n 1",
            ):
                self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()
