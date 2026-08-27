"""Runtime Stage 2 checks; skipped on dependency-light workstations."""

from __future__ import annotations

import unittest


class Stage2RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover - expected local limitation
            raise unittest.SkipTest(f"Stage 2 runtime dependencies unavailable: {exc}")
        cls.torch = torch
        cls.transformers = transformers
        from code.RSmol.recursive_model import RecursiveLlamaForCausalLM

        config = transformers.LlamaConfig(
            vocab_size=31,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            _attn_implementation="eager",
        )
        config.recursive_loops = 2
        config.recursive_layer_count = 2
        config.recursive_mapping_policy = "explicit_fixture"
        config.recursive_source_layer_indices_0based = [0, 3]
        config.recursive_source_layer_indices_1based = [1, 4]
        cls.model = RecursiveLlamaForCausalLM(config)

    def test_dynamic_padding_masks_padding_only(self) -> None:
        from code.RSmol.scripts.train_stage2_single_gpu import collate_dynamic_padding

        batch = collate_dynamic_padding([[3, 4, 5], [6, 7]], pad_token_id=0)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 3))
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(batch["labels"].tolist(), [[3, 4, 5], [6, 7, -100]])

    def test_strict_toy_audit(self) -> None:
        from code.RSmol.scripts.train_stage2_single_gpu import strict_toy_batch_audit

        self.model.config.use_cache = True
        self.model.eval()
        result = strict_toy_batch_audit(self.model, device=self.torch.device("cpu"))
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["backward_second_loop_then_first"])
        self.assertTrue(result["shared_gradient_accumulated"])
        self.assertEqual(result["optimizer_step_calls"], 1)
        self.assertGreater(result["representative_update_norm"], 0.0)
        self.assertEqual(result["audited_physical_layers"], [0, 1])
        self.assertEqual(set(result["physical_layer_gradient_norms"]), {"0", "1"})
        self.assertTrue(all(result["physical_layer_gradient_finite_nonzero"].values()))
        self.assertTrue(all(result["physical_layer_shared_parameter_identity"].values()))
        self.assertTrue(result["audit_model_training_state_restored"])
        self.assertTrue(result["audit_config_state_restored"])
        self.assertTrue(result["audit_rng_state_restored"])
        self.assertTrue(self.model.config.use_cache)
        self.assertFalse(self.model.training)


if __name__ == "__main__":
    unittest.main()
