"""Dependency-light contracts for fixed 5-10-5 formal audio training."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "code" / "RSmol" / "audio_5_10_5_recursive_mellow"
MODEL = PKG / "model.py"
DATA = PKG / "data.py"
TRAIN = ROOT / "code" / "RSmol" / "scripts" / "train_audio_5_10_5_recursive_mellow_ddp.py"
INNER = ROOT / "code" / "RSmol" / "scripts" / "train_audio_5_10_5_recursive_mellow_formal_ddp.sh"
SUBMIT = ROOT / "code" / "RSmol" / "run_audio_5_10_5_recursive_mellow_formal_5090.sh"
BASE_MODEL = ROOT / "code" / "RSmol" / "audio_smollm2_135m_mellow" / "model.py"
BASE_TRAIN = ROOT / "code" / "RSmol" / "scripts" / "train_audio_smollm2_135m_mellow_ddp.py"


class RecursiveAudioStaticTest(unittest.TestCase):
    def test_files_parse_and_exist(self) -> None:
        for path in (PKG / "__init__.py", PKG / "README.md", MODEL, DATA, TRAIN, INNER, SUBMIT):
            self.assertTrue(path.is_file(), path)
        for path in (MODEL, DATA, TRAIN):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_recursive_architecture_is_exact_and_router_free(self) -> None:
        text = MODEL.read_text(encoding="utf-8")
        for marker in (
            "RecursiveLlamaForCausalLM",
            "LOGICAL_LAYER_COUNT",
            "PHYSICAL_LAYER_COUNT",
            "LOGICAL_TO_PHYSICAL",
            "recursive_loops_scope",
            "middle_only",
            "forward_trace_matches_exact_5_10x2_5",
            "prefix_middle_suffix_invocation_counts_valid",
            "no_mesh_router_or_memory_parameters",
            "RECURSIVE_HIDDEN_SIZE = 576",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("write_routers", text)
        self.assertNotIn("read_routers", text)

    def test_audio_contract_is_shared_exactly(self) -> None:
        data = DATA.read_text(encoding="utf-8")
        model = MODEL.read_text(encoding="utf-8")
        self.assertIn("audio_5_10x2_5_mesh_mellow.data", data)
        self.assertIn("AudioSmolLM2Model", model)
        for marker in ("AUDIO_TOKENS_PER_CLIP", "AUDIO_PREFIX_TOKENS", "MAPPER_CONTRACT", "_load_mellow_wrapper"):
            self.assertIn(marker, model)

    def test_trainer_is_formal_only_and_uses_text_checkpoint_as_model_path(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        for marker in (
            'core.GATE_CHOICES = ("FORMAL",)',
            'core.DEFAULT_GATE = "FORMAL"',
            'core.MODEL_PATH_OPTIONS = ("--model-path", "--recursive-checkpoint")',
            "checkpoint-step-009244",
            "RecursiveLlamaForCausalLM.from_pretrained",
            "register_auto_class",
            "audio_recursive_5_10_5_config.json",
            "audio_5_10_5_recursive_mellow_composite_v1",
            '"dataset_rows": 968059',
            '"formal_steps": 11343',
            '"warmup_steps": 568',
            "fixed 5-10-5 FORMAL refuses a non-empty output directory",
            "core._validate_route_training_plan = _validate_recursive_training_plan",
            "core._validate_route_checkpoint_config = _validate_recursive_checkpoint_config",
            "audio_initialization_hashes",
            "run_start_audio_state_hashes",
            "DDP audio initialization hashes differ across ranks",
            "core._after_ddp_initialization = _after_recursive_ddp_initialization",
        ):
            self.assertIn(marker, text)
        self.assertNotIn('core.GATE_CHOICES = ("STAGE7"', text)

    def test_formal_shell_contract(self) -> None:
        inner = INNER.read_text(encoding="utf-8")
        submit = SUBMIT.read_text(encoding="utf-8")
        for marker in (
            "--gate FORMAL",
            "--micro-batch-size 8",
            "--gradient-accumulation-steps 4",
            "--epochs 3",
            "--max-lr 1e-3",
            "--min-lr 0",
            "--save-every 500",
            "--checkpoint-retention 4",
            "stage1_with_clotho_aqa_v2_drop12",
        ):
            self.assertIn(marker, inner)
        self.assertNotIn("--max-steps", inner)
        self.assertIn("--nproc_per_node=8", inner)
        self.assertIn("vc submit", submit)
        self.assertIn("-g 8", submit)

    def test_reuse_seams_preserve_original_baseline_defaults(self) -> None:
        base_model = BASE_MODEL.read_text(encoding="utf-8")
        base_train = BASE_TRAIN.read_text(encoding="utf-8")
        self.assertIn("validate_text_model = staticmethod(validate_original_smollm2)", base_model)
        for marker in (
            'GATE_CHOICES = ("STAGE7", "FORMAL")',
            'MODEL_PATH_OPTIONS = ("--model-path", "--smollm2-model")',
            'ARTIFACT_CONTRACT = "audio_smollm2_135m_mellow_composite_v1"',
            'CONFIG_FILENAME = "audio_smollm2_config.json"',
            "def _load_text_backbone",
        ):
            self.assertIn(marker, base_train)


if __name__ == "__main__":
    unittest.main()
