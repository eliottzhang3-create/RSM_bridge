"""Real CPU tensor checks for all seven new evaluation routes; never mock tensors."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/RSmol"))
sys.path.insert(0, str(ROOT / "code/RSmol/scripts"))
import mizar_s1_ced_multilayer_eval_runtime as adapter

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))
NEW = {name: spec for name, spec in adapter.ROUTE_SPECS.items() if spec.get("extension")}


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(AVAILABLE, "requires real torch + transformers; no mocked tensor PASS")
class ExtensionTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)
        helper = load("extension_tensor_helpers", "tests/test_mizar_s1_ced_multilayer_eval_tensor.py")
        helper.LEGACY_SPECS = {name: spec for name, spec in NEW.items() if not spec.get("official")}
        helper.MultilayerTensorTests.setUpClass()
        cls.helper = helper.MultilayerTensorTests()
        official = load("official_extension_tensor_helpers", "tests/test_mizar_official_recursive_tensor.py")
        official.TensorTests.setUpClass()
        cls.official = official.TensorTests()

    def fixture(self, variant):
        t, spec = self.torch, NEW[variant]
        if spec.get("official"):
            lm, width, namespace = self.official.lm(), 24, {}
            with t.no_grad():
                for gate in (lm.model.audio_gate, *lm.model.audio_gates.values()):
                    gate.weight.fill_(.01); gate.bias.fill_(.2)
                for name, value in lm.named_parameters():
                    if ".lora_B." in name: value.normal_(0., .03)
        else:
            namespace = self.helper.namespaces[variant]
            with patch.dict(sys.modules, namespace): lm = self.helper.model(variant)
            width = 36
        prefix = t.randn(1, 383, width)
        context = ({"audio_information": (t.randn(1, 126, width), t.randn(1, 126, width) + 2)}
                   if spec["context_kind"] == "pair" else {})
        return lm.eval(), prefix, context, namespace

    def test_logits_tokens_and_actual_execution_match_direct_training_forward_for_all_seven(self):
        t = self.torch
        for variant, spec in NEW.items():
            lm, prefix, context, namespace = self.fixture(variant)
            width = prefix.shape[-1]
            with patch.dict(sys.modules, namespace), patch.object(adapter, "HIDDEN_SIZE", width), t.inference_mode():
                audit = adapter.install_recursive_trace_audit(lm, eval_variant=variant)
                trained = {name: value.clone() for name, value in lm.state_dict().items()}
                lm.load_state_dict(trained, strict=True)
                missing = dict(trained)
                missing.pop(next(name for name in missing if ".lora_B." in name) if spec.get("loop_lora")
                            else next(name for name in missing if "write_routers" in name))
                with self.assertRaises(RuntimeError): lm.load_state_dict(missing, strict=True)
                lm.load_state_dict(trained, strict=True)
                current, expected = prefix, []
                for _ in range(4):
                    result = lm(inputs_embeds=current, attention_mask=None, use_cache=False, **context)
                    token = int(result.logits[:, -1].argmax(-1))
                    expected.append(token)
                    if token == 0: break
                    current = t.cat((current, lm.get_input_embeddings()(t.tensor([[token]]))), 1)
                model = types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=lm))
                tokenizer = types.SimpleNamespace(eos_token_id=0, decode=lambda ids, **kw: str(ids))
                with patch.object(adapter, "MAX_NEW_TOKENS", 4):
                    actual = adapter.greedy_decode(model, tokenizer, prefix, 4, eval_variant=variant, context=context)
                self.assertEqual(actual["generated_token_ids"], expected)
                self.assertEqual(audit["last_physical_schedule"], list(spec["schedule"]))
                self.assertEqual(audit["checked_forwards"], 2 * len(expected))
                if spec.get("loop_lora"):
                    self.assertEqual(audit["loop_lora_projection_calls"], 280 if spec.get("official") else 140)
                    # Both loops' learned B matrices survive strict loading.
                    self.assertTrue(any(t.count_nonzero(v) for n, v in trained.items() if ".lora_B." in n))
                first = lm(inputs_embeds=prefix, attention_mask=None, use_cache=False, **context).logits.clone()
                if context:
                    changed_context = {"audio_information": tuple(item + 2 for item in context["audio_information"])}
                    changed = lm(inputs_embeds=prefix, attention_mask=None, use_cache=False, **changed_context).logits
                    self.assertFalse(t.equal(first, changed))
                lm(inputs_embeds=prefix + 1, attention_mask=None, use_cache=False, **context)
                repeated = lm(inputs_embeds=prefix, attention_mask=None, use_cache=False, **context).logits
                t.testing.assert_close(first, repeated, rtol=0, atol=0)

    def test_full_sequence_memory_seed_queries_write_formula_and_generated_length(self):
        t = self.torch
        for variant, spec in NEW.items():
            if not spec["mesh"]: continue
            lm, prefix, _, namespace = self.fixture(variant)
            core = lm.model
            core.audit_mode = True
            with patch.dict(sys.modules, namespace), t.inference_mode():
                for length in (383, 384):
                    inputs = prefix if length == 383 else t.cat((prefix, prefix[:, :1]), 1)
                    lm(inputs_embeds=inputs, attention_mask=None, use_cache=False)
                    self.assertEqual(core.last_initial_memory.shape, (1, spec["memory_slots"], length, 36))
                    t.testing.assert_close(core.last_initial_memory[:, 0], inputs, rtol=0, atol=0)
                    self.assertEqual(int(t.count_nonzero(core.last_initial_memory[:, 1:])), 0)
                    memory = core.last_initial_memory.clone()
                    for index, suffix in enumerate(("pre", "0", "1")):
                        query = core.last_prefix_output if index == 0 else core.last_core_inputs[index - 1]
                        value = core.last_prefix_output if index == 0 else core.last_core_outputs[index - 1]
                        for action in ("write", "read"):
                            t.testing.assert_close(core.last_router_queries[f"{action}_{suffix}"], query, rtol=0, atol=0)
                        weights = core.last_router_weights[f"write_{suffix}"]
                        memory += value.unsqueeze(1) * weights.transpose(1, 2).unsqueeze(-1)
                        t.testing.assert_close(core.last_memory_write_history[index], memory, rtol=0, atol=0)

    def test_real_lora_family_parallel_formula_and_audio_text_boundary(self):
        t = self.torch
        from torch.nn import functional as F
        for variant, spec in NEW.items():
            if not spec.get("family_lora"): continue
            lm, _, _, _ = self.fixture(variant)
            for target in ("self_attn.q_proj", "mlp.up_proj"):
                module = lm.model.layers[5].get_submodule(target)
                value = t.randn(1, 130, module.in_features)
                def delta(branch, region):
                    return F.linear(F.linear(region, module.lora_A[branch][1]), module.lora_B[branch][1])
                with t.inference_mode():
                    expected = F.linear(value, module.weight, module.bias)
                    if module.mode in ("shared", "shared_modality"):
                        expected = expected + module.scaling * delta("shared", value)
                    if module.mode in ("modality", "shared_modality"):
                        private = t.cat((delta("audio", value[:, :126]), delta("text", value[:, 126:])), 1)
                        expected = expected + module.scaling * private
                    module.loop_index = 1
                    try: actual = module(value)
                    finally: module.loop_index = None
                    t.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
