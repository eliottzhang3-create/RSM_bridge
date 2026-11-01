"""Real small CPU MeSH tensors, never downloads weights or runs CUDA."""
from __future__ import annotations

import csv
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/RSmol/scripts"))
import calibrate_mesh_text_router_k as cal

HAS_TENSORS = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


@unittest.skipUnless(HAS_TENSORS, "real CPU tensor checks require torch and transformers")
class RouterTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LlamaConfig
        sys.path.insert(0, str(ROOT / "code/RSmol"))
        from recursive_model_5_10x2_5_mesh import RecursiveLlamaForCausalLM
        cls.torch, cls.config_class, cls.model_class = torch, LlamaConfig, RecursiveLlamaForCausalLM
        torch.set_num_threads(1)

    def model(self, eos=31):
        cfg = self.config_class(vocab_size=32, hidden_size=12, intermediate_size=24,
            num_hidden_layers=30, num_attention_heads=3, num_key_value_heads=1,
            max_position_embeddings=1024, eos_token_id=eos, pad_token_id=17,
            use_cache=False, attention_dropout=0.)
        cfg.recursive_layer_count, cfg.recursive_loops = 20, 2
        cfg.logical_to_physical = list(cal.SCHEDULE)
        return self.model_class(cfg).float().eval()

    def tokenizer(self, eos=31, length=2):
        torch = self.torch
        class Tokenizer:
            eos_token_id = eos
            def __call__(self, _text, **kwargs):
                assert kwargs == {"add_special_tokens": True, "truncation": False, "return_tensors": "pt"}
                return {"input_ids": torch.tensor([[2] * length])}
            def decode(self, ids, **_kwargs): return " ".join(str(i) for i in ids)
        return Tokenizer()

    def test_read_only_hooks_preserve_logits_all_weights_and_remove_handles(self):
        t, lm = self.torch, self.model()
        ids = t.tensor([[1, 2, 3]])
        state = {name: value.clone() for name, value in lm.state_dict().items()}
        with t.inference_mode():
            original = lm(input_ids=ids, use_cache=False).logits
            with cal.RouterRecorder(lm) as recorder:
                recorder.begin(3, 0)
                observed = lm(input_ids=ids, use_cache=False).logits
                rows = recorder.finish(ids[0].tolist(), 2, "p", 0, "generation", self.tokenizer())
            self.assertTrue(t.equal(original, observed))
        self.assertEqual(len(rows), 18)
        for name, value in lm.state_dict().items(): self.assertTrue(t.equal(value, state[name]), name)
        self.assertFalse(recorder.handles)
        self.assertTrue(all(not module._forward_hooks for module in lm.modules()))

    def test_fixed100_generation_final_token_observation_and_no_duplicate_positions(self):
        t, lm = self.torch, self.model()
        # All output logits tie at zero: deterministic argmax=0, EOS=31 never selected.
        with t.no_grad(): lm.lm_head.weight.zero_()
        stream, rows = io.StringIO(), []
        writer = csv.DictWriter(stream, fieldnames=cal.CSV_FIELDS); writer.writeheader()
        with t.inference_mode(), cal.RouterRecorder(lm) as recorder:
            result = cal.generate_one(lm, self.tokenizer(), {"id": "p", "prompt": "text"}, recorder, writer, rows)
        self.assertEqual(result["generated_token_ids"], [0] * 100)
        self.assertEqual(result["forward_count"], 101)
        self.assertEqual(result["stop_reason"], "max_new_tokens")
        self.assertEqual(len(rows), 102 * 6)
        self.assertEqual(len({(r["sequence_position"], r["router"]) for r in rows}), len(rows))
        final = [row for row in rows if row["sequence_position"] == 101]
        self.assertEqual(len(final), 6)
        self.assertTrue(all(row["forward_kind"] == "post_generation_observation" for row in final))

    def test_eos_stops_at_one_but_still_records_the_eos_input(self):
        t, lm = self.torch, self.model(eos=0)
        with t.no_grad(): lm.lm_head.weight.zero_()
        rows, writer = [], csv.DictWriter(io.StringIO(), fieldnames=cal.CSV_FIELDS)
        with t.inference_mode(), cal.RouterRecorder(lm) as recorder:
            result = cal.generate_one(lm, self.tokenizer(eos=0), {"id": "p", "prompt": "text"}, recorder, writer, rows)
        self.assertEqual(result["generated_token_ids"], [0])
        self.assertEqual(result["stop_reason"], "eos_token")
        self.assertEqual(result["forward_count"], 2)
        self.assertEqual(len(rows), 3 * 6)

    def test_context_overflow_and_eos_mismatch_fail_before_forward(self):
        t, lm = self.torch, self.model()
        rows, writer = [], csv.DictWriter(io.StringIO(), fieldnames=cal.CSV_FIELDS)
        with patch.object(lm, "forward", side_effect=AssertionError("must not forward")):
            with self.assertRaisesRegex(ValueError, "no truncation"):
                cal.generate_one(lm, self.tokenizer(length=925), {"id": "p", "prompt": "x"}, None, writer, rows)
            with self.assertRaisesRegex(ValueError, "EOS mismatch"):
                cal.generate_one(lm, self.tokenizer(eos=0), {"id": "p", "prompt": "x"}, None, writer, rows)

    def test_measured_logit_formula_matches_real_fp32_expanded_linear(self):
        t = self.torch
        router = t.nn.Linear(12, 5)
        query = t.randn(2, 10, 12)
        expanded = t.nn.Linear(12, 7)
        k = 7.3
        with t.no_grad():
            expanded.weight[:5].copy_(router.weight)
            expanded.bias[:5].copy_(router.bias)
            expanded.weight[5:].zero_(); expanded.bias[5:].fill_(-k)
            original = router(query)
            actual = t.softmax(expanded(query), -1)[..., 5:].sum(-1)
        expected = t.tensor([[cal.new_mass(k, cal.logsumexp(z)) for z in sample] for sample in original.tolist()])
        self.assertTrue(t.allclose(expected, actual, rtol=1e-6, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
