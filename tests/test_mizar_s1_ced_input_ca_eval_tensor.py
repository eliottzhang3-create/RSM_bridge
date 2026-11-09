"""Real training-model logits, nonzero gates and strict state checks on CPU."""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code/RSmol/scripts"), str(ROOT / "code/RSmol")]
import mizar_s1_ced_multilayer_eval_runtime as adapter

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))
NEW = {name: spec for name, spec in adapter.ROUTE_SPECS.items() if spec.get("input_ca_extension")}
CA, INPUT = tuple(NEW)


@unittest.skipUnless(AVAILABLE, "requires real torch + transformers; never substitute mocked tensor PASS")
class InputCaTensorTests(unittest.TestCase):
    def fixture(self, variant):
        import torch as t
        from transformers import LlamaConfig
        t.set_num_threads(1)
        t.manual_seed(1234)
        spec, width = NEW[variant], 36 if variant == CA else 24
        cfg = LlamaConfig(vocab_size=32, hidden_size=width, intermediate_size=48,
                          num_hidden_layers=len(spec["schedule"]), num_attention_heads=9 if variant == CA else 3,
                          num_key_value_heads=3 if variant == CA else 1, max_position_embeddings=1024,
                          pad_token_id=17, eos_token_id=0, use_cache=False, attention_dropout=0., tie_word_embeddings=True)
        cfg.recursive_layer_count, cfg.recursive_loops = (20 if variant == CA else 30), 2
        cfg.recursive_prefix_layer_count = cfg.recursive_suffix_layer_count = 5
        cfg.recursive_middle_layer_count = 10 if variant == CA else 20
        cfg.logical_to_physical = list(spec["schedule"])
        cfg.recursive_loops_scope = "middle_only"
        cfg.recursive_source_layer_indices_0based = list(range(30))
        cfg._attn_implementation = "eager"
        namespace = {}
        if variant == CA:
            with patch.dict(sys.modules):
                for name in list(sys.modules):
                    if name == "models" or name.startswith("models.") or name == "contract":
                        del sys.modules[name]
                package = types.ModuleType("models")
                package.__path__ = [str(spec["route_root"] / "models")]
                sys.modules["models"] = package
                implementation = importlib.import_module(spec["implementation"])
                lm = implementation.RecursiveLlamaForCausalLM(cfg).float()
                lm.model.enable_audio_cross_attention()
                namespace = {name: module for name, module in sys.modules.items()
                             if name == "models" or name.startswith("models.")}
            gates = (lm.model.audio_crossattn.input_gate, lm.model.audio_crossattn.gate2)
        else:
            module = importlib.import_module(spec["implementation"])
            lm = module.MizarInputOnlyRecursiveForCausalLM(cfg).float()
            lm.model.enable_new_modules()
            gates = (lm.model.audio_gate,)
        with t.no_grad():
            for index, gate in enumerate(gates):
                gate.weight.fill_(.01)
                gate.bias.fill_(.2 if index == 0 else -.15)
        prefix = t.randn(1,383,width)
        context = ({"audio_grids": t.randn(1,3,126,1536),
                    "prompt_mask": t.tensor([[1]*180+[0]*76])} if variant == CA else {})
        return t, lm.eval(), prefix, context, namespace

    def test_adapter_logits_tokens_and_observed_execution_equal_direct_training_model(self):
        for variant, spec in NEW.items():
            t, lm, prefix, context, namespace = self.fixture(variant)
            with patch.dict(sys.modules, namespace), t.inference_mode():
                observed = []
                lm.register_forward_hook(lambda _m, _i, out: observed.append(out.logits[:, -1].clone()))
                current, expected = prefix, []
                for _ in range(4):
                    result = lm(inputs_embeds=current, attention_mask=None, labels=None, use_cache=False, **context)
                    token = int(result.logits[:, -1].argmax(-1))
                    expected.append(token)
                    if token == 0: break
                    current = t.cat((current, lm.get_input_embeddings()(t.tensor([[token]]))), 1)
                direct = observed[:]
                observed.clear()
                with patch.object(adapter, "HIDDEN_SIZE", prefix.shape[-1]), patch.object(adapter, "MAX_NEW_TOKENS", 4):
                    audit = adapter.install_recursive_trace_audit(lm, eval_variant=variant)
                    actual = adapter.greedy_decode(types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=lm)),
                        types.SimpleNamespace(eos_token_id=0, decode=lambda ids, **kw: str(ids)),
                        prefix, 4, eval_variant=variant, context=context)
                self.assertEqual(actual["generated_token_ids"], expected)
                self.assertEqual(audit["checked_forwards"], len(expected))
                self.assertEqual(audit["last_physical_schedule"], list(spec["schedule"]))
                for left,right in zip(direct,observed): t.testing.assert_close(left,right,rtol=0,atol=0)
                before = lm(inputs_embeds=prefix,use_cache=False,**context).logits.clone()
                lm(inputs_embeds=prefix+1,use_cache=False,**context)
                after = lm(inputs_embeds=prefix,use_cache=False,**context).logits
                t.testing.assert_close(before,after,rtol=0,atol=0)

    def test_learned_nonzero_branches_survive_strict_reload_missing_keys_and_wrong_depth_fail(self):
        for variant in NEW:
            t,lm,prefix,context,namespace = self.fixture(variant)
            with patch.dict(sys.modules,namespace), t.inference_mode():
                state = {key:value.clone() for key,value in lm.state_dict().items()}
                key = next(key for key in state if "input_gate.bias" in key or "audio_gate.bias" in key)
                self.assertGreater(int(t.count_nonzero(state[key])),0)
                expected = lm(inputs_embeds=prefix,use_cache=False,**context).logits.clone()
                broken = dict(state)
                broken.pop(key)
                with self.assertRaises(RuntimeError): lm.load_state_dict(broken,strict=True)
                lm.load_state_dict(state,strict=True)
                t.testing.assert_close(expected,lm(inputs_embeds=prefix,use_cache=False,**context).logits,rtol=0,atol=0)
                if variant == CA:
                    changed = {**context,"audio_grids":context["audio_grids"]+t.randn_like(context["audio_grids"])}
                    self.assertFalse(t.equal(expected,lm(inputs_embeds=prefix,use_cache=False,**changed).logits))
                    with self.assertRaises(ValueError):
                        lm(inputs_embeds=prefix,use_cache=False,**{**context,"audio_grids":t.randn(1,4,126,1536)})
                else:
                    from mizar_official_recursive_5_20x2_5_input_only.tensor_runtime import assert_input_only
                    assert_input_only(lm)
                    lm.model.loop_lora_enabled = False
                    with self.assertRaises(ValueError): assert_input_only(lm)

    def test_ca_gates_read_raw_then_post_input_hidden_and_leave_separator_text_unchanged(self):
        t,lm,prefix,context,namespace = self.fixture(CA)
        branch = lm.model.audio_crossattn
        with patch.dict(sys.modules,namespace), t.inference_mode():
            hidden = prefix.clone()
            bank = branch.project_audio(context["audio_grids"],hidden)
            residual = branch.project_input(prefix)
            raw = hidden[:,:126]
            injected = raw+t.tanh(branch.input_gate(raw))*residual
            depth = t.softmax(branch.gate1(injected),-1)
            mixed = (bank*depth.permute(0,2,1).unsqueeze(-1)).sum(1)
            delta,_ = branch.cross_attention(branch.query_norm(injected),mixed,mixed,need_weights=False)
            expected = injected+t.tanh(branch.gate2(injected))*delta
            calls = {}
            for name in ("input_gate","gate1","query_norm","gate2"):
                getattr(branch,name).register_forward_pre_hook(lambda _m,args,name=name: calls.update({name:args[0].clone()}))
            actual = branch(hidden,bank,residual,0)
            t.testing.assert_close(actual[:,:126],expected,rtol=0,atol=0)
            t.testing.assert_close(actual[:,126:],hidden[:,126:],rtol=0,atol=0)
            t.testing.assert_close(calls["input_gate"],raw,rtol=0,atol=0)
            for name in ("gate1","query_norm","gate2"): t.testing.assert_close(calls[name],injected,rtol=0,atol=0)

    def test_input_only_residual_uses_original_input_and_updates_audio_at_both_entrances(self):
        t,lm,prefix,context,namespace = self.fixture(INPUT)
        before, after = {}, {}
        core = lm.model
        with t.inference_mode():
            residual = core.audio_linear(prefix[:,:126])
            def hidden(out): return out[0] if isinstance(out,tuple) else out
            core.layers[4].register_forward_hook(lambda _m,_i,out: before.update({5:hidden(out).clone()}))
            visits = []
            def middle_end(_m,_i,out):
                visits.append(1)
                if len(visits)==1: before[25] = hidden(out).clone()
            core.layers[24].register_forward_hook(middle_end)
            def middle_start(_m,args): after[len(after)] = args[0].clone()
            core.layers[5].register_forward_pre_hook(middle_start)
            lm(inputs_embeds=prefix,use_cache=False)
            for index,boundary in enumerate((5,25)):
                current = before[boundary]
                expected = current[:,:126]+t.tanh(core.audio_gate(current[:,:126]))*residual
                t.testing.assert_close(after[index][:,:126],expected,rtol=0,atol=0)
                t.testing.assert_close(after[index][:,126:],current[:,126:],rtol=0,atol=0)
            self.assertEqual(core.last_input_injection_trace,(5,25))


if __name__ == "__main__": unittest.main()
