"""Dependency-light contract checks for Stage 2 training validation."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "code" / "RSmol" / "scripts" / "train_stage2_single_gpu.py"
RUNTIME_SHELL = ROOT / "code" / "RSmol" / "scripts" / "train_stage2_single_gpu.sh"
SUBMIT_SHELL = ROOT / "code" / "RSmol" / "run_stage2_single_gpu_training_5090.sh"


def load_stage2_module():
    spec = importlib.util.spec_from_file_location("stage2_training_static", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Stage 2 source: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage2StaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.runtime = RUNTIME_SHELL.read_text(encoding="utf-8")
        cls.submit = SUBMIT_SHELL.read_text(encoding="utf-8")
        cls.stage2 = load_stage2_module()

    def test_exact_85_shard_manifest(self) -> None:
        names = self.stage2.expected_parquet_names()
        self.assertEqual(len(names), 85)
        self.assertEqual(names[0], "train-00000-of-00085.parquet")
        self.assertEqual(names[-1], "train-00084-of-00085.parquet")
        self.assertIn("exactly the 85 parquet shards", self.source)
        self.assertIn("ParquetFile.iter_batches", self.source)
        self.assertIn("audit_parquet_shards", self.source)
        self.assertIn('required in ("text", "source")', self.source)
        self.assertIn('"row_groups"', self.source)
        self.assertIn('"num_rows"', self.source)
        self.assertIn('"total_rows"', self.source)
        self.assertIn('"footer_only": True', self.source)

    def test_training_defaults_and_single_gpu_contract(self) -> None:
        self.assertIn("micro_batch_size: int = 8", self.source)
        self.assertIn("gradient_accumulation_steps: int = 16", self.source)
        self.assertIn("learning_rate: float = 2e-4", self.source)
        self.assertIn("context_length: int = 1024", self.source)
        self.assertIn("warmup_steps: int = 2", self.source)
        self.assertIn("torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16)", self.source)
        self.assertIn("torch.float32", self.source)
        self.assertIn('config.use_cache = False', self.source)
        self.assertIn('torch.optim.AdamW', self.source)
        self.assertIn('--micro-batch-size', self.runtime)
        self.assertIn('--gradient-accumulation-steps', self.runtime)
        self.assertIn('-g 1', self.submit)

    def test_label_padding_and_causal_shift_contract(self) -> None:
        self.assertIn('labels.masked_fill_(attention_mask == 0, -100)', self.source)
        self.assertIn('labels[:, 1:] != -100', self.source)
        self.assertIn('shift_logits = logits[..., :-1, :]', self.source)
        self.assertIn('shift_labels = labels[..., 1:].contiguous()', self.source)
        self.assertIn('add_special_tokens=False', self.source)
        self.assertIn('No source/prefix mask', self.source)
        self.assertIn('Non-padding tokens must remain supervised', self.source)
        self.assertNotIn('labels[attention_mask == 1] = -100', self.source)
        self.assertNotIn('input_ids[:, :-1]', self.source)
        self.assertIn('def _ensure_padding_token', self.source)
        self.assertIn('tokenizer.pad_token = tokenizer.eos_token', self.source)
        self.assertIn('synthetic_pad_token', self.source)

    def test_audit_and_checkpoint_contracts_are_explicit(self) -> None:
        required = (
            'def strict_toy_batch_audit',
            'register_full_backward_hook',
            'Backward must traverse the second recursive loop and then the first loop',
            'shared_parameter_identity_unique',
            'physical_layer_gradient_norms',
            'physical_layer_shared_parameter_identity',
            'saved_gradients',
            'config_state',
            'torch_random_state',
            'audit_model_training_state_restored',
            'audit_config_state_restored',
            'audit_rng_state_restored',
            'optimizer_matches_model_exactly_once',
            'optimizer_step_exactly_once_per_window',
            'representative_parameter_changed',
            'checkpoint_reload_continuation_check',
            'continuation_step_succeeded',
            'ensure_external_output',
        )
        for marker in required:
            self.assertIn(marker, self.source)
        self.assertIn('checkpoint-step-{optimizer_step:06d}', self.source)
        self.assertIn('training_state.pt', self.source)

    def test_external_output_policy_rejects_checkout(self) -> None:
        with self.assertRaises(ValueError):
            self.stage2.ensure_external_output(ROOT / "outputs" / "bad-checkpoint")
        with tempfile.TemporaryDirectory() as temporary:
            path = self.stage2.ensure_external_output(Path(temporary) / "safe-output")
            self.assertTrue(str(path).endswith("safe-output"))

    def test_runtime_wrapper_has_no_direct_login_node_execution(self) -> None:
        self.assertIn('vc submit', self.submit)
        self.assertIn('bash scripts/train_stage2_single_gpu.sh', self.submit)
        self.assertIn('conda activate', self.runtime)
        self.assertIn('python -u code/RSmol/scripts/train_stage2_single_gpu.py', self.runtime)
        self.assertIn('/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset', self.runtime)


if __name__ == "__main__":
    unittest.main()
