"""Optional real CPU MeSH checks; run in mellow_c8204d8 without large assets."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HAS_TENSORS = (importlib.util.find_spec("torch") is not None
               and importlib.util.find_spec("transformers") is not None)


@unittest.skipUnless(HAS_TENSORS, "requires torch and transformers in mellow_c8204d8")
class MeshCedEvaluationTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LlamaConfig

        rsmol = ROOT / "code/RSmol"
        sys.path.insert(0, str(rsmol))
        from recursive_model_5_10x2_5_mesh import RecursiveLlamaForCausalLM

        spec = importlib.util.spec_from_file_location(
            "mesh_eval_tensor_runtime", rsmol / "scripts/mizar_s1_ced_recursive_mesh_eval_runtime.py")
        cls.runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runtime)
        cls.torch, cls.config_class, cls.model_class = torch, LlamaConfig, RecursiveLlamaForCausalLM
        torch.set_num_threads(1)

    def make_model(self):
        self.torch.manual_seed(1234)
        config = self.config_class(
            vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=30,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
            pad_token_id=17, eos_token_id=0, use_cache=False, attention_dropout=0.)
        config.recursive_layer_count, config.recursive_loops = 20, 2
        config.logical_to_physical = list(self.runtime.TEXT_MODEL_SCHEDULE)
        return self.model_class(config).float().eval().requires_grad_(False)

    def test_trace_hook_preserves_real_logits_and_parameter_values(self):
        torch = self.torch
        model = self.make_model()
        inputs = torch.randn(1, 8, 32)
        state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        with torch.inference_mode():
            expected = model(inputs_embeds=inputs, attention_mask=None, labels=None, use_cache=False)
            audit = self.runtime.install_mesh_trace_audit(model)
            actual = model(inputs_embeds=inputs, attention_mask=None, labels=None, use_cache=False)
        self.assertTrue(torch.equal(actual.logits, expected.logits))
        self.assertIsNone(actual.past_key_values)
        self.assertEqual(audit["checked_forwards"], 1)
        self.assertEqual(audit["status"], "PASS")
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, state[name]), name)
        self.assertFalse(model.model.audit_mode)
        self.assertFalse(model.model.gradient_audit_mode)
        self.assertFalse(model.model.routing_stats_mode)

    def test_each_full_forward_starts_with_fresh_five_slot_memory(self):
        torch = self.torch
        model = self.make_model()
        backbone = model.model
        original_write = backbone._write
        first_writes, write_count = [], 0

        def record_write(memory, *args):
            nonlocal write_count
            if write_count % 3 == 0:
                first_writes.append(memory.clone())
            write_count += 1
            return original_write(memory, *args)

        inputs = [torch.randn(1, length, 32) for length in (8, 9)]
        audit = self.runtime.install_mesh_trace_audit(model)
        with patch.object(backbone, "_write", side_effect=record_write), torch.inference_mode():
            for value in inputs:
                output = model(inputs_embeds=value, attention_mask=None, labels=None, use_cache=False)
                self.assertIsNone(output.past_key_values)
        self.assertEqual(write_count, 6)
        self.assertEqual(len(first_writes), 2)
        for memory, value in zip(first_writes, inputs):
            self.assertEqual(tuple(memory.shape), (1, 5, value.shape[1], 32))
            self.assertTrue(torch.equal(memory[:, 0], value))
            self.assertEqual(int(memory[:, 1:].count_nonzero()), 0)
        self.assertEqual(audit["checked_forwards"], 2)
        self.assertEqual(audit["last_sequence_tokens"], 9)


if __name__ == "__main__":
    unittest.main()
