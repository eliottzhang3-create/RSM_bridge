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
PARTITION_TRAIN = ROOT / "code" / "RSmol" / "scripts" / "train_audio_partitioned_5_10_5_recursive_mellow_ddp.py"
INNER = tuple(ROOT / "code" / "RSmol" / "scripts" / name for name in (
    "train_audio_5_10_5_recursive_mellow_smoke20_ddp.sh",
    "train_audio_5_10_5_recursive_mellow_resume2_ddp.sh",
    "train_audio_5_10_5_recursive_mellow_formal_ddp.sh",
))
SUBMIT = tuple(ROOT / "code" / "RSmol" / name for name in (
    "run_audio_5_10_5_recursive_mellow_smoke20_5090.sh",
    "run_audio_5_10_5_recursive_mellow_resume2_5090.sh",
    "run_audio_5_10_5_recursive_mellow_formal_5090.sh",
))
BASE_MODEL = ROOT / "code" / "RSmol" / "audio_smollm2_135m_mellow" / "model.py"
BASE_TRAIN = ROOT / "code" / "RSmol" / "scripts" / "train_audio_smollm2_135m_mellow_ddp.py"


class RecursiveAudioStaticTest(unittest.TestCase):
    def test_files_parse_and_exist(self) -> None:
        for path in (PKG / "__init__.py", PKG / "README.md", MODEL, DATA, TRAIN, PARTITION_TRAIN, *INNER, *SUBMIT):
            self.assertTrue(path.is_file(), path)
        for path in (MODEL, DATA, TRAIN, PARTITION_TRAIN):
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
            "both_middle_loops_have_finite_gradients",
            "middle_loop_input_finite_gradients",
            "middle_loop_output_finite_gradients",
            "no_mesh_router_or_memory_parameters",
            "RECURSIVE_HIDDEN_SIZE = 576",
            "AUDIO_SINGLE_PREFIX_TOKENS",
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

    def test_partition_trainer_uses_exact_recursive_source_and_v2_contract(self) -> None:
        text = PARTITION_TRAIN.read_text(encoding="utf-8")
        for marker in (
            'choices=("smoke", "formal")',
            "checkpoint-step-009244",
            "RecursiveLlamaForCausalLM.from_pretrained",
            "register_auto_class",
            "audio_recursive_5_10_5_partition_config.json",
            "recursive_5_10_5_component_partitions6_rank_ram_compact_audio_answer_eos_v2",
            "formal fixed-recursive step budget must equal 37,810",
            "refusing nonempty output directory",
            "_formal_smoke_gate",
            "audio_initialization_hashes",
            "checkpoint_audio_state_hashes",
            "DDP audio initialization hashes differ across ranks",
            "exact_fixed_recursive_5_10x2_5",
            "resume_parameter_change_audit",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("write_routers", text)
        self.assertNotIn("read_routers", text)

    def test_formal_shell_contract(self) -> None:
        for inner in INNER:
            text = inner.read_text(encoding="utf-8")
            self.assertIn("train_audio_partitioned_5_10_5_recursive_mellow_ddp.py", text)
            self.assertIn("--epochs 10", text)
            self.assertIn("--micro-batch-size 8", text)
            self.assertIn("--gradient-accumulation-steps 4", text)
            self.assertIn("--nproc_per_node=8", text)
        self.assertIn("--mode smoke", INNER[0].read_text(encoding="utf-8"))
        self.assertIn("--expected-resume-step 20", INNER[1].read_text(encoding="utf-8"))
        self.assertIn("--mode formal", INNER[2].read_text(encoding="utf-8"))
        for submit in SUBMIT:
            text = submit.read_text(encoding="utf-8")
            self.assertIn("vc submit -p pdgpu-5090", text)
            self.assertIn("-c 32 -m 256G -g 8 -n 1", text)

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

    def test_readme_documents_new_gate_and_historical_contract(self) -> None:
        text = (PKG / "README.md").read_text(encoding="utf-8")
        for marker in (
            "SMOKE_OUT=",
            "RESUME_OUT=",
            "FORMAL_OUT=",
            "checkpoint-000020",
            "checkpoint-000022",
            "pdgpu-5090",
            "37,810",
            "not a valid",
        ):
            self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()
