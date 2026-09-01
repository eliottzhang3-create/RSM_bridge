from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "code" / "RSmol" / "scripts" / "audit_smol_block.py"
RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "audit_smol_block_5090.sh"
SUBMIT = ROOT / "code" / "RSmol" / "run_audit_smol_block_5090.sh"


class SmolBlockAuditStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = PYTHON.read_text(encoding="utf-8")
        cls.runtime = RUNTIME.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_isolated_entrypoints_and_original_default(self) -> None:
        self.assertTrue(PYTHON.is_file())
        self.assertTrue(RUNTIME.is_file())
        self.assertTrue(SUBMIT.is_file())
        self.assertIn("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2", self.source)
        self.assertIn("audit_smol_block.py", self.runtime)
        self.assertNotIn("recursive_model", self.source.lower())
        self.assertNotIn("linear evaluator", self.source.lower())

    def test_gpu_submission_contract(self) -> None:
        self.assertIn("pdgpu-5090", self.submit)
        self.assertIn("-g", self.submit)
        self.assertIn('RSMOL_SMOL_BLOCK_GPUS:-1', self.submit)
        self.assertIn("docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1", self.submit)

    def test_local_checkpoint_and_offline_loading(self) -> None:
        self.assertIn("local_files_only=True", self.source)
        self.assertIn("AutoConfig.from_pretrained", self.source)
        self.assertIn("AutoModelForCausalLM.from_pretrained", self.source)
        self.assertIn("low_cpu_mem_usage=True", self.source)
        self.assertIn('"config.json"', self.source)
        self.assertIn('"model.safetensors"', self.source)

    def test_config_and_actual_layer_introspection(self) -> None:
        for field in (
            "model_type",
            "architectures",
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rms_norm_eps",
            "rope_theta",
            "max_position_embeddings",
            "vocab_size",
            "torch_dtype",
            "attn_implementation",
        ):
            self.assertIn(field, self.source)
        self.assertIn("len(model.model.layers)", self.source)
        self.assertIn("all_layer_signatures_identical", self.source)
        self.assertIn("parameter_inventory", self.source)
        self.assertIn("shared_parameter_groups", self.source)

    def test_norm_attention_mlp_and_residual_audit(self) -> None:
        for text in (
            "input_layernorm",
            "post_attention_layernorm",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "RMSNorm",
            "RoPE",
            "causal_mask",
            "attention_residual_reconstruction",
            "mlp_residual_reconstruction",
            "norm_non_identity",
            "ablation_results",
        ):
            self.assertIn(text, self.source)
        self.assertIn("register_forward_pre_hook", self.source)
        self.assertIn("register_forward_hook", self.source)
        self.assertIn("Full model decoder trace mismatch", self.source)

    def test_external_output_and_fresh_directory(self) -> None:
        self.assertIn("require_external_output", self.source)
        self.assertIn("non-empty output directory", self.source)
        self.assertIn("/code/RSmol", self.source)
        self.assertIn("smol_block_audit.json", self.source)
        self.assertIn("smol_block_audit.md", self.source)

    def test_shell_parameter_forwarding_and_logs(self) -> None:
        for name in (
            "RSMOL_SMOL_BLOCK_MODEL_PATH",
            "RSMOL_SMOL_BLOCK_DEVICE",
            "RSMOL_SMOL_BLOCK_DTYPE",
            "RSMOL_SMOL_BLOCK_LAYER_INDEX",
            "RSMOL_SMOL_BLOCK_SEED",
            "RSMOL_SMOL_BLOCK_SEQ_LEN",
            "RSMOL_SMOL_BLOCK_OUTPUT_DIR",
            "RSMOL_SMOL_BLOCK_LOG_ROOT",
        ):
            self.assertIn(name, self.runtime)
            self.assertIn(name, self.submit)
        self.assertIn("smol_block_audit_runtime.log", self.runtime)
        self.assertIn("JOB=1:1", self.submit)


if __name__ == "__main__":
    unittest.main()
