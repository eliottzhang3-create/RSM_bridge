"""Real CPU training models with synthetic inputs, no weights/downloads/CUDA.

The 20 physical layers, 30 executions, 383 prefix positions, four CED grids
and query32 geometry stay real. Most tests reduce only LM width/vocabulary and
decode horizon. The structure audit also uses production width 576.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/RSmol/scripts"))
import mizar_s1_ced_multilayer_eval_runtime as adapter

HAS_TENSORS = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_TENSORS, "requires torch and transformers; never substitute mocks for tensors")
class MultilayerTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LlamaConfig
        cls.torch, cls.config_class = torch, LlamaConfig
        torch.set_num_threads(1)
        cls.modules = {}
        cls.namespaces = {}
        for variant, spec in adapter.ROUTE_SPECS.items():
            # Isolate the real same-named training packages for each route.
            with patch.dict(sys.modules):
                for name in list(sys.modules):
                    if name == "models" or name.startswith("models.") or name == "contract":
                        del sys.modules[name]
                package = types.ModuleType("models")
                package.__path__ = [str(spec["route_root"] / "models")]
                sys.modules["models"] = package
                contract = load_module("contract", spec["route_root"] / "contract.py")
                implementation = load_module(spec["implementation"],
                    spec["route_root"] / (spec["implementation"].replace(".", "/") + ".py"))
                loader = load_module("models.recursive_text", spec["route_root"] / "models/recursive_text.py")
                decoder = load_module("models.decoder", spec["route_root"] / "models/decoder.py")
                # Four-depth listen imports its branch lazily when it is enabled.
                if spec["kind"] == "listen36912":
                    load_module("models.audio_listen", spec["route_root"] / "models/audio_listen.py")
                cls.modules[variant] = (contract, implementation, loader, decoder)
                cls.namespaces[variant] = {name: module for name, module in sys.modules.items()
                    if name == "contract" or name == "models" or name.startswith("models.")}

    def model(self, variant, width=36):
        t = self.torch
        t.manual_seed(1234)
        cfg = self.config_class(vocab_size=32, hidden_size=width, intermediate_size=48,
            num_hidden_layers=30, num_attention_heads=9, num_key_value_heads=3,
            max_position_embeddings=1024, pad_token_id=17, eos_token_id=0,
            use_cache=False, attention_dropout=0.)
        cfg.recursive_layer_count, cfg.recursive_loops = 20, 2
        cfg.logical_to_physical = list(adapter.TEXT_MODEL_SCHEDULE)
        spec = adapter.variant_spec(variant)
        if spec["mesh"]:
            cfg.mesh_memory_slots, cfg.mesh_router_count = 5, 6
            cfg.mesh_architecture_contract = spec["text_contract"]
            cfg.model_architecture_contract = spec["text_architecture"]
        else:
            cfg.recursive_prefix_layer_count, cfg.recursive_middle_layer_count, cfg.recursive_suffix_layer_count = 5, 10, 5
            cfg.recursive_loops_scope = "middle_only"
            cfg.recursive_source_layer_indices_0based = list(self.modules[variant][0].TEXT_MODEL_SOURCE_MAPPING)
        lm = self.modules[variant][1].RecursiveLlamaForCausalLM(cfg).float()
        core, kind = lm.model, spec["kind"]
        if kind in ("crossattn", "query32"):
            core.enable_audio_cross_attention()
            gate_modules = [core.audio_crossattn.gate2]
        elif kind == "inject48":
            core.enable_layer_injection()
            gate_modules = list(core.audio_gates.values())
        elif kind == "listen48":
            core.enable_audio_listen()
            gate_modules = list(core.audio_gates.values())
        else:
            with patch.dict(sys.modules, self.namespaces[variant]):
                core.enable_audio_listen()
            gate_modules = list(core.audio_listen.amplitudes)
        with t.no_grad():
            for i, gate in enumerate(gate_modules):
                gate.weight.fill_(.01)
                gate.bias.fill_(.2 if i == 0 else -.15)
        return lm.eval()

    def inputs(self, variant, width=36):
        t = self.torch
        prefix = t.randn(1, 383, width)
        if adapter.variant_spec(variant)["kind"] in ("inject48", "listen48"):
            context = {"audio_information": (t.randn(1, 126, width), t.randn(1, 126, width) + 2)}
        else:
            context = {"audio_grids": t.randn(1, 4, 126, 1536),
                       "prompt_mask": t.tensor([[1] * 180 + [0] * 76])}
        return prefix, context

    @staticmethod
    def call(lm, prefix, context):
        return lm(inputs_embeds=prefix, attention_mask=None, labels=None, use_cache=False, **context)

    def test_hooks_preserve_real_logits_state_and_transient_memory_for_all_five(self):
        t = self.torch
        for variant in adapter.ROUTE_SPECS:
            with self.subTest(variant=variant), t.inference_mode():
                lm, (prefix, context) = self.model(variant), self.inputs(variant)
                state = copy.deepcopy(lm.state_dict())
                expected = self.call(lm, prefix, context)
                with patch.object(adapter, "HIDDEN_SIZE", 36):
                    audit = adapter.install_recursive_trace_audit(lm, eval_variant=variant)
                    actual = self.call(lm, prefix, context)
                self.assertTrue(t.equal(expected.logits, actual.logits))
                self.assertIsNone(actual.past_key_values)
                self.assertEqual(audit["checked_forwards"], 1)
                self.assertEqual(audit["last_physical_schedule"], list(adapter.TEXT_MODEL_SCHEDULE))
                for key, value in lm.state_dict().items(): self.assertTrue(t.equal(value, state[key]), key)
                self.assertFalse(any("memory_slot" in name or name.endswith(".memory") for name in state))

    def test_adapter_matches_direct_model_logits_and_tokens_at_every_decode_step(self):
        t = self.torch
        for variant in adapter.ROUTE_SPECS:
            with self.subTest(variant=variant), t.inference_mode():
                lm, (prefix, context) = self.model(variant), self.inputs(variant)
                expected_logits, expected_ids, sequence = [], [], prefix
                for _ in range(3):
                    logits = self.call(lm, sequence, context).logits[:, -1].clone()
                    expected_logits.append(logits)
                    token = int(logits.argmax(-1).item())
                    expected_ids.append(token)
                    sequence = t.cat((sequence, lm.get_input_embeddings()(t.tensor([[token]]))), dim=1)
                captured = []
                handle = lm.register_forward_hook(lambda _m, _i, out: captured.append(out.logits[:, -1].clone()))
                model = types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=lm))
                tokenizer = types.SimpleNamespace(eos_token_id=-1, decode=lambda *_a, **_k: " raw EOS ")
                try:
                    with patch.object(adapter, "HIDDEN_SIZE", 36), patch.object(adapter, "MAX_NEW_TOKENS", 3):
                        audit = adapter.install_recursive_trace_audit(lm, eval_variant=variant)
                        result = adapter.greedy_decode(model, tokenizer, prefix, 3, eval_variant=variant, context=context)
                finally:
                    handle.remove()
                self.assertEqual(result["generated_token_ids"], expected_ids)
                self.assertEqual(audit["checked_forwards"], 3)
                for actual, expected in zip(captured, expected_logits): self.assertTrue(t.equal(actual, expected))

    def test_training_decoder_and_prefix_evaluation_first_answer_logits_align(self):
        t = self.torch
        for variant in adapter.ROUTE_SPECS:
            with self.subTest(variant=variant), t.inference_mode():
                decoder_class = self.modules[variant][3].DecoderModel
                decoder = decoder_class.__new__(decoder_class)
                t.nn.Module.__init__(decoder)
                decoder.lm, decoder.lm_embedding_size = self.model(variant), 36
                audio = t.randn(1, 126, 36)
                prompt = {"input_ids": t.randint(1, 16, (1, 256)),
                          "attention_mask": t.tensor([[1] * 180 + [0] * 76])}
                answer = {"input_ids": t.randint(1, 16, (1, 300))}
                _, context = self.inputs(variant)
                kwargs = {"audio_information": context["audio_information"]} if "audio_information" in context else {"audio_grids": context["audio_grids"]}
                train_output = decoder(audio, None, prompt, answer, **kwargs)
                prefix = decoder.generate_prefix_inference(audio, None, prompt)
                context = {**context, **({"prompt_mask": prompt["attention_mask"]} if "prompt_mask" in context else {})}
                evaluated = self.call(decoder.lm, prefix, context)
                t.testing.assert_close(train_output.logits[:, 382], evaluated.logits[:, -1], rtol=1e-5, atol=1e-6)

    def test_strict_roundtrip_preserves_learned_branches_and_missing_keys_fail(self):
        t = self.torch
        for variant in adapter.ROUTE_SPECS:
            with self.subTest(variant=variant), t.inference_mode():
                lm, (prefix, context) = self.model(variant), self.inputs(variant)
                restored = self.model(variant)
                state = copy.deepcopy(lm.state_dict())
                restored.load_state_dict(state, strict=True)
                self.assertTrue(t.equal(self.call(lm, prefix, context).logits, self.call(restored, prefix, context).logits))
                key = next(key for key in state if any(part in key for part in ("audio_crossattn", "audio_gates", "audio_listen")))
                state.pop(key)
                with self.assertRaises(RuntimeError): restored.load_state_dict(state, strict=True)

    def test_samples_and_prompt_masks_never_leak_across_full_forwards(self):
        t = self.torch
        for variant in adapter.ROUTE_SPECS:
            with self.subTest(variant=variant), t.inference_mode():
                lm, (prefix, context) = self.model(variant), self.inputs(variant)
                original = prefix.clone()
                before = self.call(lm, prefix, context).logits.clone()
                changed = copy.deepcopy(context)
                if "audio_information" in changed:
                    changed["audio_information"] = tuple(reversed(changed["audio_information"]))
                else:
                    changed["audio_grids"] = changed["audio_grids"] * -2
                after = self.call(lm, prefix, changed).logits.clone()
                repeated = self.call(lm, prefix, context).logits.clone()
                self.assertTrue(t.equal(prefix, original))
                self.assertTrue(t.equal(before, repeated))
                self.assertFalse(t.equal(before[:, 382], after[:, 382]))

    def test_query32_compresses_each_depth_once_per_forward_and_keeps_prefix383(self):
        t = self.torch
        variant = "recursive_5_10_5_layer36912_query32_crossattn"
        with t.inference_mode():
            lm, (prefix, context) = self.model(variant), self.inputs(variant)
            branch = lm.model.audio_crossattn
            sources = []
            handle = branch.compression_attention.register_forward_pre_hook(lambda _m, values: sources.append(tuple(v.clone() for v in values)))
            try:
                self.call(lm, prefix, context)
                self.call(lm, t.cat((prefix, t.randn(1, 1, 36)), dim=1), context)
            finally:
                handle.remove()
            self.assertEqual(len(sources), 2)
            self.assertEqual(tuple(prefix.shape), (1, 383, 36))
            for q, k, v in sources:
                self.assertEqual(tuple(q.shape), (4, 32, 36))
                self.assertEqual(tuple(k.shape), (4, 126, 36))
                self.assertTrue(t.equal(q[0], q[3]))
                self.assertTrue(t.equal(k, v))
                self.assertFalse(t.equal(k[0], k[1]))

    def test_mesh_actual_writes_listens_reads_and_injections_have_training_order(self):
        t = self.torch
        for variant, spec in adapter.ROUTE_SPECS.items():
            if not spec["mesh"]: continue
            with self.subTest(variant=variant), t.inference_mode():
                lm, (prefix, context) = self.model(variant), self.inputs(variant)
                core, events = lm.model, []
                write, read = core._write, core._read
                def observed_write(*args): events.append("write"); return write(*args)
                def observed_read(*args): events.append("read"); return read(*args)
                if spec["kind"] == "listen36912":
                    handle = core.audio_listen.register_forward_pre_hook(lambda *_: events.append("listen"))
                else:
                    handles = [gate.register_forward_pre_hook(lambda *_: events.append("inject" if spec["kind"] == "inject48" else "listen")) for gate in core.audio_gates.values()]
                    handle = types.SimpleNamespace(remove=lambda: [h.remove() for h in handles])
                try:
                    # Test-only observation of real methods; production uses read-only module hooks.
                    with patch.object(core, "_write", side_effect=observed_write), patch.object(core, "_read", side_effect=observed_read):
                        self.call(lm, prefix, context)
                finally:
                    handle.remove()
                boundary = ["write", "read", "inject"] if spec["kind"] == "inject48" else ["write", "listen", "read"]
                self.assertEqual(events, boundary * 2 + ["write", "read"])

    def test_context_rejects_wrong_features_masks_dtype_device_and_nonfinite_values(self):
        t = self.torch
        for variant in adapter.ROUTE_SPECS:
            prefix, context = self.inputs(variant)
            with patch.object(adapter, "HIDDEN_SIZE", 36):
                self.assertIs(adapter.validate_context(variant, context, prefix), context)
                bad = copy.deepcopy(context)
                if "audio_information" in bad:
                    bad["audio_information"] = (bad["audio_information"][0].double(), bad["audio_information"][1])
                else:
                    bad["prompt_mask"].zero_()
                with self.assertRaises(ValueError): adapter.validate_context(variant, bad, prefix)
                bad = copy.deepcopy(context)
                value = bad["audio_information"][0] if "audio_information" in bad else bad["audio_grids"]
                value.flatten()[0] = float("nan")
                with self.assertRaises(ValueError): adapter.validate_context(variant, bad, prefix)

    def test_production_width_structure_audit_uses_actual_training_validators(self):
        t = self.torch
        for variant, spec in adapter.ROUTE_SPECS.items():
            contract, implementation, loader, _ = self.modules[variant]
            lm = self.model(variant, width=576)
            base = t.nn.Module()
            if spec["kind"] in ("inject48", "listen48"):
                base.layer_mlps = t.nn.ModuleDict({str(k): t.nn.Sequential(t.nn.LayerNorm(1536),
                    t.nn.Linear(1536, 576), t.nn.GELU(), t.nn.Linear(576, 576), t.nn.LayerNorm(576)) for k in (4, 8)})
            model = types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=lm,
                is_recursive_mesh=spec["mesh"], is_recursive_5_10_5=not spec["mesh"],
                text_model_contract=spec["text_contract"]), audio_encoder=types.SimpleNamespace(base=base))
            identity = {"synthetic_text": True}
            with patch.dict(sys.modules, self.namespaces[variant]), patch.object(contract, "recursive_checkpoint_contract", return_value=identity):
                recipe = {"text_model_dir": "synthetic", "recursive_text_checkpoint": identity,
                          "recursive_model_source_sha256": contract.recursive_model_source_sha256()}
                audit = adapter.audit_recursive_route(contract, recipe, model, eval_variant=variant)
                self.assertEqual(audit["new_parameter_count"], spec["new_parameters"])
                lm.model.layers[1] = lm.model.layers[0]
                with self.assertRaises(ValueError): adapter.audit_recursive_route(contract, recipe, model, eval_variant=variant)


if __name__ == "__main__":
    unittest.main()
