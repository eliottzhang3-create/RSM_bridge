"""Optional actual CPU autograd tests, using small dimensions and no CED assets."""
from __future__ import annotations

import copy
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/RSmol"))
AVAILABLE = all(importlib.util.find_spec(x) for x in ("torch", "transformers"))


@unittest.skipUnless(AVAILABLE, "requires real torch + transformers; no downloads or mocked tensor PASS")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from mizar_official_recursive_5_20x2_5 import contract, model, loop_lora, tensor_runtime
        cls.torch, cls.LlamaConfig, cls.Reference = torch, LlamaConfig, LlamaForCausalLM
        cls.contract, cls.module, cls.lora = contract, model, loop_lora
        cls.runtime = tensor_runtime
        torch.set_num_threads(1)

    def lm(self, enabled=True):
        cfg = self.LlamaConfig(vocab_size=32, hidden_size=24, intermediate_size=48,
                              num_hidden_layers=30, num_attention_heads=3, num_key_value_heads=1,
                              pad_token_id=17, eos_token_id=0, tie_word_embeddings=True,
                              use_cache=False, attention_dropout=0., max_position_embeddings=1024)
        value = cfg.to_dict()
        # build_text_config intentionally accepts production geometry only.
        value.update(num_hidden_layers=50, recursive_layer_count=30, recursive_loops=2,
                     recursive_prefix_layer_count=5, recursive_middle_layer_count=20,
                     recursive_suffix_layer_count=5, logical_to_physical=list(self.contract.SCHEDULE),
                     recursive_loops_scope="middle_only", recursive_source_layer_indices_0based=list(range(30)))
        cfg = self.runtime.config_from_dict(value)
        lm = self.module.MizarRecursiveForCausalLM(cfg).float()
        if enabled: lm.model.enable_new_modules()
        return lm

    def inputs(self):
        return self.torch.randn(1, 130, 24), tuple(self.torch.randn(1, 126, 24) for _ in range(2))

    def test_real_transformers_constructor_preserves_canonical_config_exactly(self):
        import json
        value = self.lm(enabled=False).config.to_dict()
        value.pop("attn_implementation", None)
        value["custom_preserved"] = {"sentinel": 42}
        before = copy.deepcopy(value)
        runtime = self.runtime.config_from_dict(value)
        self.assertEqual(value, before)
        self.assertEqual(json.loads(json.dumps(value)), before)
        self.assertEqual(runtime._attn_implementation, "eager")

    def test_real_runtime_config_nested_values_do_not_alias_canonical_dict(self):
        value = self.lm(enabled=False).config.to_dict()
        value["custom_preserved"] = {"sentinel": 42}
        before = copy.deepcopy(value)
        runtime = self.runtime.config_from_dict(value)
        runtime.custom_preserved["sentinel"] = -1
        runtime.logical_to_physical[0] = 29
        self.assertEqual(value, before)

    def test_strict_load_before_extensions_and_preserve_all_inherited_objects(self):
        torch = self.torch
        lm = self.lm(enabled=False)
        inherited = {n: p.clone() for n, p in lm.state_dict().items()}
        objects = dict(lm.named_parameters())
        rng = torch.get_rng_state().clone()
        lm.model.enable_new_modules()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for n, p in objects.items():
            self.assertIs(dict(lm.named_parameters())[n], p)
            self.assertTrue(torch.equal(p, inherited[n]))
        self.assertEqual(len(lm.model.layers), 30)
        self.assertEqual(len({id(x) for x in lm.model.layers}), 30)
        self.assertIs(lm.get_input_embeddings().weight, lm.get_output_embeddings().weight)
        self.assertEqual(self.lora.validate_loop_lora(lm.model)["matrix_count"], 560)
        with self.assertRaises(RuntimeError): lm.load_state_dict(inherited, strict=True)
        with self.assertRaises(ValueError): lm.model.enable_new_modules()

    def test_real_single_pass_matches_strict_standard_thirty_layers(self):
        torch = self.torch
        lm = self.lm(enabled=False).eval()
        cfg = copy.deepcopy(lm.config)
        cfg.num_hidden_layers = 30
        standard = self.Reference(cfg).float().eval()
        standard.load_state_dict(lm.state_dict(), strict=True)
        lm.model.enable_new_modules()
        x, pair = self.inputs()
        with torch.no_grad():
            actual = lm.lm_head(lm.model._execute(x, single_pass_audit=True).last_hidden_state)
            expected = standard(inputs_embeds=x, use_cache=False).logits
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            recursive = lm(inputs_embeds=x, audio_information=pair).logits
        self.assertFalse(torch.allclose(recursive, expected, rtol=1e-5, atol=1e-5))

    def test_full_fifty_layer_trace_and_independent_loop_calls(self):
        torch = self.torch
        lm = self.lm()
        layers, loops, handles = [], [], []
        for i, item in enumerate(lm.model.layers):
            handles.append(item.register_forward_pre_hook(lambda m, a, i=i: layers.append(i)))
        for item in lm.modules():
            if isinstance(item, self.lora.LoopLoRALinear):
                handles.append(item.register_forward_pre_hook(lambda m, a: loops.append(m.loop_index)))
        x, pair = self.inputs()
        try: output = lm(inputs_embeds=x, audio_information=pair)
        finally:
            for handle in handles: handle.remove()
        self.assertEqual(layers, list(range(25)) + list(range(5, 25)) + list(range(25, 30)))
        self.assertEqual(loops, [0] * 140 + [1] * 140)
        self.assertEqual(lm.model.last_input_projection_calls, 1)
        self.assertEqual(lm.model.last_serial_injection_trace, self.contract.SERIAL_TRACE)
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(all(m.loop_index is None for m in lm.modules() if isinstance(m, self.lora.LoopLoRALinear)))

    def test_nonzero_branches_backward_both_lora_loops_and_shared_base(self):
        torch = self.torch
        lm = self.lm()
        with torch.no_grad():
            for item in lm.modules():
                if isinstance(item, self.lora.LoopLoRALinear):
                    for b in item.lora_B: b.normal_(0, .01)
            for gate in (lm.model.audio_gate, *lm.model.audio_gates.values()): gate.bias.fill_(.1)
        x, pair = self.inputs()
        x.requires_grad_()
        for value in pair: value.requires_grad_()
        lm(inputs_embeds=x, audio_information=pair).logits.square().mean().backward()
        for name, parameter in lm.named_parameters():
            if ".lora_" in name or name.endswith("layers.5.self_attn.q_proj.weight"):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertTrue(all(v.grad is not None and v.grad.abs().sum() > 0 for v in pair))
        self.assertIsNotNone(x.grad)

    def test_serial_gate_reads_post_input_hidden_and_only_audio_positions_change(self):
        torch = self.torch
        lm = self.lm().eval()
        core = lm.model
        x, pair = self.inputs()
        seen = []
        with torch.no_grad():
            core.audio_linear.weight.copy_(torch.eye(24))
            core.audio_gate.bias.fill_(.3)
            for gate in core.audio_gates.values(): gate.weight.fill_(.001); gate.bias.fill_(.1)
        handles = [core.audio_gates[str(i)].register_forward_pre_hook(lambda m, a: seen.append(a[0].clone()))
                   for i in (8, 4)]
        # Isolate the real serial injection code by replacing decoder computation
        # with identity; the separate full-forward test executes actual attention.
        try:
            with torch.no_grad(), patch.object(self.module, "_call_decoder_layer", side_effect=lambda layer, hidden, **kw: (hidden,)):
                result = core._execute(x, pair).last_hidden_state
        finally:
            for handle in handles: handle.remove()
        audio = x[:, :126]
        expected_seen = []
        for ced in (8, 4):
            audio = audio + torch.tanh(core.audio_gate(audio)) * core.audio_linear(x[:, :126])
            expected_seen.append(audio.clone())
            audio = audio + torch.tanh(core.audio_gates[str(ced)](audio)) * pair[1 if ced == 8 else 0]
        for a, b in zip(seen, expected_seen): torch.testing.assert_close(a, b, rtol=0, atol=0)
        expected = core.norm(torch.cat((audio, x[:, 126:]), dim=1))
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(result[:, 126:], core.norm(x[:, 126:]), rtol=0, atol=0)

    def test_scope_is_reset_after_decoder_exception(self):
        lm = self.lm()
        layer = lm.model.layers[5]
        def fail(*a, **kw): raise RuntimeError("deliberate decoder failure")
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            self.lora.call_with_loop_lora(layer, 1, fail)
        self.assertTrue(all(m.loop_index is None for m in self.lora.projections(layer)))
        item = layer.self_attn.q_proj
        item.lora_A[1] = item.lora_A[0]
        with self.assertRaises(ValueError): self.lora.validate_loop_lora(lm.model)

    def test_zero_gates_zero_B_initialize_with_staged_gradient_unlock(self):
        torch = self.torch
        lm = self.lm()
        x, pair = self.inputs()
        lm(inputs_embeds=x, audio_information=pair).logits.square().mean().backward()
        for item in lm.modules():
            if isinstance(item, self.lora.LoopLoRALinear):
                for a, b in zip(item.lora_A, item.lora_B):
                    self.assertEqual(torch.count_nonzero(a.grad).item(), 0)
                    self.assertGreater(b.grad.abs().sum().item(), 0)
        self.assertEqual(torch.count_nonzero(lm.model.audio_linear.weight.grad).item(), 0)
        self.assertGreater(lm.model.audio_gate.weight.grad.abs().sum().item(), 0)

    def test_exact_tensor_and_logits_save_reload_for_all_thirty_layers_and_new_branches(self):
        torch = self.torch
        lm = self.lm().eval()
        with torch.no_grad():
            for m in lm.modules():
                if isinstance(m, self.lora.LoopLoRALinear):
                    for b in m.lora_B: b.normal_(0, .01)
            lm.model.audio_gate.bias.fill_(.1)
        stream = io.BytesIO()
        torch.save(lm.state_dict(), stream)
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
        restored = self.lm().eval()
        restored.load_state_dict(state, strict=True)
        for name, value in restored.state_dict().items(): self.assertTrue(torch.equal(value, state[name]), name)
        x, pair = self.inputs()
        with torch.no_grad():
            torch.testing.assert_close(lm(inputs_embeds=x, audio_information=pair).logits,
                                       restored(inputs_embeds=x, audio_information=pair).logits, rtol=0, atol=0)

    def test_cached_masked_missing_audio_and_single_pass_training_rejected(self):
        torch = self.torch
        lm = self.lm()
        x, pair = self.inputs()
        for kw in ({"use_cache": True}, {"attention_mask": torch.ones(1, 130)}, {"past_key_values": object()}):
            with self.assertRaises(ValueError): lm(inputs_embeds=x, audio_information=pair, **kw)
        with self.assertRaises(ValueError): lm(inputs_embeds=x, audio_information=None)
        with self.assertRaises(ValueError): lm.model._execute(x, single_pass_audit=True)
        lm.model.gradient_checkpointing = True
        with self.assertRaises(ValueError): lm(inputs_embeds=x, audio_information=pair)


if __name__ == "__main__":
    unittest.main()
