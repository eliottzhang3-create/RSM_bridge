"""Optional actual CPU tensor checks; no downloads, model assets or CUDA."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/RSmol"))
from mizar_official_eval import runtime

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "requires actual torch CPU tensors")
class OfficialTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def make_lm(self):
        torch = self.torch
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = torch.nn.Parameter(torch.tensor(0.001))
            def forward(self, hidden): return hidden + self.offset
        class LM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([Layer() for _ in range(30)])
                self.embed = torch.nn.Embedding(4, 576)
            def get_input_embeddings(self): return self.embed
            def forward(self, inputs_embeds, attention_mask=None, labels=None, use_cache=False):
                hidden = inputs_embeds
                for layer in self.model.layers:
                    hidden = layer(hidden)
                logits = torch.zeros(1, hidden.shape[1], 4)
                logits[..., 0] = hidden.mean(-1).abs() + 1
                return NS(logits=logits, past_key_values=None)
        return LM().float().eval().requires_grad_(False)

    def test_read_only_hooks_preserve_real_logits_and_weights(self):
        torch = self.torch
        model = self.make_lm()
        inputs = torch.randn(1, 383, 576)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.inference_mode():
            before = model(inputs_embeds=inputs, attention_mask=None, use_cache=False).logits
            audit = runtime.install_forward_audit(model)
            after = model(inputs_embeds=inputs, attention_mask=None, use_cache=False).logits
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(audit["last_layers"], list(range(30)))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state[name]), name)

    def test_same_greedy_stops_on_raw_eos_and_records_actual_layers(self):
        torch = self.torch
        lm = self.make_lm()
        audit = runtime.install_forward_audit(lm)
        tokenizer = NS(eos_token_id=0, decode=lambda ids, **kwargs: "<|endoftext|>")
        model = NS(caption_decoder=NS(lm=lm))
        with torch.inference_mode():
            output = runtime.greedy_decode(model, tokenizer, torch.zeros(1, 383, 576))
        self.assertEqual(output["generated_token_ids"], [0])
        self.assertEqual(output["generated_text"], "<|endoftext|>")
        self.assertEqual(audit["checked_forwards"], 1)

    def test_actual_hooks_reject_skipped_or_reordered_layers(self):
        torch = self.torch
        for reorder in (False, True):
            model = self.make_lm()
            runtime.install_forward_audit(model)
            if reorder:
                layers = model.model.layers
                layers[1], layers[2] = layers[2], layers[1]
            else:
                model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:-1])
            with self.assertRaisesRegex(RuntimeError, "layer sequence"), torch.inference_mode():
                model(inputs_embeds=torch.zeros(1, 383, 576), use_cache=False)


if __name__ == "__main__":
    unittest.main()
