"""Real CPU models with synthetic inputs; no CUDA, assets or downloads.

Use the actual 20-layer/30-execution implementations with hidden size 32 and a
small vocabulary. Only adapter width and decode horizon are reduced in these
tests; production constants remain 576 and 300. Run in mellow_c8204d8.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code/RSmol/scripts"), str(ROOT / "code/RSmol")]
import mizar_s1_ced_recursive_eval_runtime as adapter

HAS_TENSORS = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_TENSORS, "requires torch and transformers in mellow_c8204d8")
class RecursiveCedTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LlamaConfig
        cls.torch, cls.config_class = torch, LlamaConfig
        torch.set_num_threads(1)
        cls.backends = {}
        for variant, spec in adapter.ROUTE_SPECS.items():
            path = adapter.ROOT / "recursive_model_5_10_5.py" if variant == "recursive_5_10_5" else (
                spec["route_root"] / (spec["implementation"].replace(".", "/") + ".py"))
            cls.backends[variant] = load_module("tensor_backend_" + variant, path)

    def model(self, variant):
        torch = self.torch
        torch.manual_seed(1234)
        config = self.config_class(vocab_size=32, hidden_size=32, intermediate_size=48,
            num_hidden_layers=30, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=1024, pad_token_id=17, eos_token_id=0,
            use_cache=False, attention_dropout=0.)
        config.recursive_layer_count, config.recursive_loops = 20, 2
        config.logical_to_physical = list(adapter.TEXT_MODEL_SCHEDULE)
        lm = self.backends[variant].RecursiveLlamaForCausalLM(config).float()
        spec = adapter.variant_spec(variant)
        if spec["layer_order"]:
            lm.model.enable_layer_injection()
            with torch.no_grad():
                lm.model.audio_gates["4"].weight.fill_(.02)
                lm.model.audio_gates["8"].weight.fill_(-.01)
                lm.model.audio_gates["4"].bias.fill_(.15)
                lm.model.audio_gates["8"].bias.fill_(.25)
        elif spec["reinjection"]:
            lm.model.enable_audio_reinjection()
            with torch.no_grad():
                lm.model.audio_gate.weight.fill_(.01)
                lm.model.audio_gate.bias.fill_(.2)
        return lm.eval().requires_grad_(False)

    def inputs(self, variant):
        torch = self.torch
        prefix = torch.randn(1, 383, 32)
        features = (torch.randn(1, 126, 32), torch.randn(1, 126, 32) + 2) if adapter.variant_spec(variant)["layer_order"] else None
        return prefix, features

    def call(self, lm, prefix, features):
        return lm(inputs_embeds=prefix, attention_mask=None, labels=None, use_cache=False,
                  **({"audio_information": features} if features is not None else {}))

    def test_trace_hooks_preserve_real_logits_and_all_state_for_four_routes(self):
        torch = self.torch
        for variant in adapter.ROUTE_SPECS:
            with self.subTest(variant=variant), torch.inference_mode():
                lm = self.model(variant)
                prefix, features = self.inputs(variant)
                state = {k: v.clone() for k, v in lm.state_dict().items()}
                expected = self.call(lm, prefix, features)
                audit = adapter.install_recursive_trace_audit(lm, eval_variant=variant)
                actual = self.call(lm, prefix, features)
                self.assertTrue(torch.equal(actual.logits, expected.logits))
                self.assertIsNone(actual.past_key_values)
                self.assertEqual(audit["checked_forwards"], 1)
                self.assertEqual(audit["last_physical_schedule"], list(adapter.TEXT_MODEL_SCHEDULE))
                for key, value in lm.state_dict().items(): self.assertTrue(torch.equal(value, state[key]), key)

    def test_adapter_logits_and_tokens_match_direct_model_calls_at_each_generation_step(self):
        torch = self.torch
        for variant in adapter.ROUTE_SPECS:
            with self.subTest(variant=variant), torch.inference_mode():
                lm = self.model(variant)
                prefix, features = self.inputs(variant)
                expected_logits, expected_ids, sequence = [], [], prefix
                for _ in range(3):
                    logits = self.call(lm, sequence, features).logits[:, -1, :].clone()
                    expected_logits.append(logits)
                    token = int(logits.argmax(-1).item())
                    expected_ids.append(token)
                    sequence = torch.cat((sequence, lm.get_input_embeddings()(torch.tensor([[token]]))), dim=1)
                captured = []
                def record(_m, _i, output): captured.append(output.logits[:, -1, :].clone())
                handle = lm.register_forward_hook(record)
                audit = adapter.install_recursive_trace_audit(lm, eval_variant=variant)
                model = SimpleNamespace(caption_decoder=SimpleNamespace(lm=lm))
                tokenizer = SimpleNamespace(eos_token_id=-1, decode=lambda *_a, **_k: " raw <|endoftext|> ")
                try:
                    with patch.object(adapter, "HIDDEN_SIZE", 32), patch.object(adapter, "MAX_NEW_TOKENS", 3):
                        result = adapter.greedy_decode(model, tokenizer, prefix, max_new_tokens=3,
                            eval_variant=variant, audio_information=features)
                finally:
                    handle.remove()
                self.assertEqual(result["generated_token_ids"], expected_ids)
                self.assertEqual(len(captured), 3)
                self.assertEqual(audit["checked_forwards"], 3)
                for actual, expected in zip(captured, expected_logits): self.assertTrue(torch.equal(actual, expected))

    def test_nonzero_layer_gates_use_original_A4_A8_pair_in_correct_order(self):
        torch = self.torch
        for variant in ("recursive_5_10_5_layer4_8_inject", "recursive_5_10_5_layer8_4_inject"):
            with self.subTest(variant=variant), torch.inference_mode():
                lm = self.model(variant)
                prefix, features = self.inputs(variant)
                original, seen = lm.model.inject_audio, []
                def observe(hidden, audio, layer, index):
                    seen.append((index, layer, audio))
                    return original(hidden, audio, layer, index)
                with patch.object(lm.model, "inject_audio", side_effect=observe):
                    actual = self.call(lm, prefix, features).logits
                order = adapter.variant_spec(variant)["layer_order"]
                self.assertEqual([(i, k) for i, k, _ in seen], list(zip((5, 15), order)))
                for (_, layer, value) in seen: self.assertIs(value, features[0 if layer == 4 else 1])
                swapped = self.call(lm, prefix, tuple(reversed(features))).logits
                self.assertFalse(torch.equal(actual[:, 382], swapped[:, 382]))

    def test_audio_reinjection_projects_original_prefix_once_and_shares_gate_twice(self):
        torch = self.torch
        variant = "recursive_5_10_5_audio_reinject"
        with torch.inference_mode():
            lm = self.model(variant)
            prefix, _ = self.inputs(variant)
            projections, gates = [], []
            def project(_m, inputs): projections.append(inputs[0].clone())
            def gate(_m, inputs): gates.append(inputs[0].shape)
            handles = [lm.model.audio_linear.register_forward_pre_hook(project), lm.model.audio_gate.register_forward_pre_hook(gate)]
            try:
                self.call(lm, prefix, None)
                extended = torch.cat((prefix, torch.randn(1, 1, 32)), dim=1)
                self.call(lm, extended, None)
            finally:
                for handle in handles: handle.remove()
            self.assertEqual(len(projections), 2)
            self.assertEqual(len(gates), 4)
            for value in projections: self.assertTrue(torch.equal(value, prefix[:, :126]))
            self.assertEqual(lm.model.last_reinject_trace, (5, 15))

    def test_different_samples_do_not_reuse_features_or_mutate_input_prefix(self):
        torch = self.torch
        for variant in adapter.ROUTE_SPECS:
            with self.subTest(variant=variant), torch.inference_mode():
                lm = self.model(variant)
                first, features1 = self.inputs(variant)
                second, features2 = self.inputs(variant)
                original = first.clone()
                a = self.call(lm, first, features1).logits.clone()
                b = self.call(lm, second, features2).logits.clone()
                c = self.call(lm, first, features1).logits.clone()
                self.assertTrue(torch.equal(first, original))
                self.assertTrue(torch.equal(a, c))
                self.assertFalse(torch.equal(a, b))

    def test_strict_state_roundtrip_keeps_trained_gate_values_and_missing_keys_fail(self):
        torch = self.torch
        for variant in adapter.ROUTE_SPECS:
            with self.subTest(variant=variant), torch.inference_mode():
                lm = self.model(variant)
                prefix, features = self.inputs(variant)
                state = copy.deepcopy(lm.state_dict())
                restored = self.model(variant)
                restored.load_state_dict(state, strict=True)
                self.assertTrue(torch.equal(self.call(lm, prefix, features).logits, self.call(restored, prefix, features).logits))
                state.pop(next(iter(state)))
                with self.assertRaises(RuntimeError): restored.load_state_dict(state, strict=True)

    def test_training_decoder_first_answer_logits_match_full_prefix_evaluation(self):
        torch = self.torch
        for variant, spec in adapter.ROUTE_SPECS.items():
            with self.subTest(variant=variant), torch.inference_mode():
                contract = load_module("tensor_contract_" + variant, spec["route_root"] / "contract.py")
                with patch.dict(sys.modules, {"contract": contract,
                    "models.recursive_text": SimpleNamespace(load_recursive_text_model=None)}):
                    decoder_module = load_module("tensor_decoder_" + variant, spec["route_root"] / "models/decoder.py")
                decoder = decoder_module.DecoderModel.__new__(decoder_module.DecoderModel)
                torch.nn.Module.__init__(decoder)
                decoder.lm, decoder.lm_embedding_size = self.model(variant), 32
                audio = torch.randn(1, 126, 32)
                prompt = {"input_ids": torch.randint(1, 16, (1, 256))}
                answer = {"input_ids": torch.randint(1, 16, (1, 300))}
                _, features = self.inputs(variant)
                training = decoder(audio, None, prompt, answer,
                    **({"audio_information": features} if features is not None else {}))
                prefix = decoder.generate_prefix_inference(audio, None, prompt)
                evaluated = self.call(decoder.lm, prefix, features)
                torch.testing.assert_close(training.logits[:, 382], evaluated.logits[:, -1], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
