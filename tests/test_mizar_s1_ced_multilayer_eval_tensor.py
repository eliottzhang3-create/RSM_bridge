"""Real CPU training models with synthetic inputs, no weights/downloads/CUDA.

The 20 physical layers, 30/50 executions, 383 prefix positions, four CED grids
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
LORA_VARIANTS = tuple(variant for variant, spec in adapter.ROUTE_SPECS.items() if spec.get("loop_lora"))


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
                implementation = load_module(spec["implementation"], adapter.implementation_path(spec))
                loader = load_module("models.recursive_text", spec["route_root"] / "models/recursive_text.py")
                decoder = load_module("models.decoder", spec["route_root"] / "models/decoder.py")
                # Four-depth listen imports its branch lazily when it is enabled.
                if spec["kind"] == "listen36912":
                    load_module("models.audio_listen", spec["route_root"] / "models/audio_listen.py")
                cls.modules[variant] = (contract, implementation, loader, decoder)
                cls.namespaces[variant] = {name: module for name, module in sys.modules.items()
                    if name == "contract" or name == "models" or name.startswith("models.")
                    or name == spec["implementation"]}

    def model(self, variant, width=36):
        t = self.torch
        t.manual_seed(1234)
        spec = adapter.variant_spec(variant)
        cfg = self.config_class(vocab_size=32, hidden_size=width,
            intermediate_size=1536 if spec.get("loop_lora") and width == 576 else 48,
            num_hidden_layers=len(spec["schedule"]), num_attention_heads=9, num_key_value_heads=3,
            max_position_embeddings=1024, pad_token_id=17, eos_token_id=0,
            use_cache=False, attention_dropout=0.)
        cfg.recursive_layer_count, cfg.recursive_loops = 20, spec["loops"]
        cfg.logical_to_physical = list(spec["schedule"])
        if spec["mesh"]:
            cfg.mesh_memory_slots, cfg.mesh_router_count = spec["memory_slots"], 2 * (spec["loops"] + 1)
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
        elif kind in ("inject48", "inject84", "serial"):
            core.enable_layer_injection()
            gate_modules = list(core.audio_gates.values())
            if kind == "serial":
                core.enable_audio_reinjection()
                gate_modules.append(core.audio_gate)
        elif kind == "mesh_x4":
            gate_modules = []
        elif spec["context_kind"] in ("pair", "seeds"):
            core.enable_audio_listen()
            gate_modules = list(core.audio_gates.values()) if spec["interaction_layers"] else []
        else:
            with patch.dict(sys.modules, self.namespaces[variant]):
                core.enable_audio_listen()
            gate_modules = list(core.audio_listen.amplitudes)
        with t.no_grad():
            for i, gate in enumerate(gate_modules):
                gate.weight.fill_(.01)
                gate.bias.fill_(.2 if i == 0 else -.15)
            if spec["split_memory"]:
                # Simulate learned rows that have departed from split3 initialization.
                for router in list(core.audio_listen_write_routers) + list(core.audio_listen_read_routers):
                    router.weight[5].add_(.03)
                    router.bias[6].add_(.4)
        if spec.get("loop_lora"):
            core.enable_loop_lora()
            with t.no_grad():
                for physical in range(5, 15):
                    for target in adapter.LOOP_LORA_TARGETS:
                        projection = core.layers[physical].get_submodule(target)
                        groups = (projection.lora_B.values() if spec.get("modality_lora") else (projection.lora_B,))
                        for group in groups:
                            for loop, matrix in enumerate(group):
                                matrix.normal_(0., .03 * (loop + 1))
        return lm.eval()

    def inputs(self, variant, width=36):
        t = self.torch
        prefix = t.randn(1, 383, width)
        kind = adapter.variant_spec(variant)["context_kind"]
        if kind == "pair":
            context = {"audio_information": (t.randn(1, 126, width), t.randn(1, 126, width) + 2)}
        elif kind == "seeds":
            context = {"audio_memory_seeds": tuple(t.randn(1, 126, width) + i for i in range(3))}
            prefix[:, :126] = sum(context["audio_memory_seeds"])
        elif kind == "none":
            context = {}
        else:
            context = {"audio_grids": t.randn(1, 4, 126, 1536),
                       "prompt_mask": t.tensor([[1] * 180 + [0] * 76])}
        return prefix, context

    @staticmethod
    def call(lm, prefix, context):
        return lm(inputs_embeds=prefix, attention_mask=None, labels=None, use_cache=False, **context)

    def decoder(self, variant):
        cls = self.modules[variant][3].DecoderModel
        decoder = cls.__new__(cls)
        self.torch.nn.Module.__init__(decoder)
        decoder.lm, decoder.lm_embedding_size = self.model(variant), 36
        if adapter.variant_spec(variant)["kind"] == "prompt_mix":
            decoder._init_prompt_mix_mlp()
        return decoder.eval()

    def test_hooks_preserve_real_logits_state_and_transient_memory_for_all_routes(self):
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
                self.assertEqual(audit["last_physical_schedule"], list(adapter.variant_spec(variant)["schedule"]))
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
                kind = adapter.variant_spec(variant)["kind"]
                if kind == "prompt_mix": decoder._init_prompt_mix_mlp()
                audio = t.randn(1, 126, 36)
                prompt = {"input_ids": t.randint(1, 16, (1, 256)),
                          "attention_mask": t.tensor([[1] * 180 + [0] * 76])}
                answer = {"input_ids": t.randint(1, 16, (1, 300))}
                _, context = self.inputs(variant)
                kwargs = ({"audio_information": context["audio_information"]} if "audio_information" in context else
                          {"audio_grids": context["audio_grids"]} if "audio_grids" in context else
                          {"audio_information": (t.randn_like(audio), t.randn_like(audio))} if kind == "prompt_mix" else {})
                train_output = decoder(audio, None, prompt, answer, **kwargs)
                if kind == "prompt_mix":
                    prefix, seeds = decoder.generate_prefix_inference(audio, None, prompt, **kwargs)
                    context = {"audio_memory_seeds": seeds}
                else:
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
                key = next(key for key in state if any(part in key for part in ("audio_crossattn", "audio_gates", "audio_listen", "write_routers")))
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
                elif "audio_memory_seeds" in changed:
                    changed["audio_memory_seeds"] = tuple(reversed(changed["audio_memory_seeds"]))
                else:
                    if changed: changed["audio_grids"] = changed["audio_grids"] * -2
                after = self.call(lm, prefix if context else prefix * -2, changed).logits.clone()
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

    def test_prompt_mix_uses_tokenizer_mask_once_and_reuses_exact_weighted_seeds(self):
        t = self.torch
        variant = "recursive_mesh_layer4_8_audio7slot_split3_prompt_mix"
        with t.inference_mode():
            decoder = self.decoder(variant)
            # A valid literal ! and EOS participate even though padding uses the same ! id.
            decoder.lm.get_input_embeddings().weight[17].fill_(.7)
            prompt = {"input_ids": t.tensor([[17, 0, 3] + [17] * 253]),
                      "attention_mask": t.tensor([[1, 1, 1] + [0] * 253])}
            audio, pair = t.randn(1, 126, 36), (t.randn(1, 126, 36), t.randn(1, 126, 36))
            calls, forwarded = [], []
            handle = decoder.prompt_mix_mlp.register_forward_hook(lambda *_: calls.append(1))
            capture = decoder.lm.register_forward_pre_hook(
                lambda _m, _i, kw: forwarded.append(kw["audio_memory_seeds"]), with_kwargs=True)
            try:
                prefix, seeds = decoder.generate_prefix_inference(audio, None, prompt, audio_information=pair)
                context = adapter.sample_context(variant, {}, seeds, prompt)
                model = types.SimpleNamespace(caption_decoder=decoder)
                tokenizer = types.SimpleNamespace(eos_token_id=-1, decode=lambda *_a, **_k: " raw ")
                with patch.object(adapter, "HIDDEN_SIZE", 36), patch.object(adapter, "MAX_NEW_TOKENS", 3):
                    audit = adapter.install_recursive_trace_audit(decoder.lm, eval_variant=variant)
                    adapter.greedy_decode(model, tokenizer, prefix, 3, eval_variant=variant, context=context)
                self.assertEqual(len(calls), 1)
                self.assertEqual(audit["checked_forwards"], 3)
                self.assertTrue(all(value is seeds for value in forwarded))
            finally:
                handle.remove(); capture.remove()
            embedding = decoder.lm.get_input_embeddings()(prompt["input_ids"])
            alpha = t.softmax(decoder.prompt_mix_mlp(embedding[:, :3].mean(1)), dim=-1)
            for i, value in enumerate((audio, *pair)):
                t.testing.assert_close(seeds[i], alpha[:, i, None, None] * value, rtol=1e-5, atol=1e-6)
            t.testing.assert_close(prefix[:, :126], seeds[0] + seeds[1] + seeds[2], rtol=0, atol=0)
            changed = {key: value.clone() for key, value in prompt.items()}
            changed["input_ids"][:, 3:] = 8
            _, padded_seeds = decoder.generate_prefix_inference(audio, None, changed, audio_information=pair)
            for before, after in zip(seeds, padded_seeds): self.assertTrue(t.equal(before, after))
            changed["attention_mask"][:, :2] = 0
            _, excluded = decoder.generate_prefix_inference(audio, None, changed, audio_information=pair)
            self.assertFalse(t.equal(seeds[0], excluded[0]))

    def test_prompt_mix_strict_state_restores_trained_mlp_and_rejects_missing_weights(self):
        t = self.torch
        variant = "recursive_mesh_layer4_8_audio7slot_split3_prompt_mix"
        with t.inference_mode():
            decoder, restored = self.decoder(variant), self.decoder(variant)
            decoder.prompt_mix_mlp[2].weight.add_(.17)
            state = copy.deepcopy(decoder.state_dict())
            restored.load_state_dict(state, strict=True)
            for key, value in restored.state_dict().items(): self.assertTrue(t.equal(value, state[key]), key)
            state.pop("prompt_mix_mlp.2.weight")
            with self.assertRaises(RuntimeError): restored.load_state_dict(state, strict=True)

    def test_serial_nonzero_input_residual_precedes_layer_gate_and_uses_raw_prefix_projection(self):
        self.assert_serial_residual("recursive_5_10_5_layer8_4_inject_audio_reinject")

    def test_loop_lora_retains_serial_input_then_ced8_and_ced4_residuals(self):
        self.assert_serial_residual("recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8")

    def test_loop_lora_retains_serial_input_then_ced4_and_ced8_residuals(self):
        self.assert_serial_residual("recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8", order=(4, 8))

    def test_rank16_and_modality_routes_retain_serial_post_input_ced_gates(self):
        for variant in LORA_VARIANTS:
            if adapter.variant_spec(variant)["new_parameters"] == 6034179:
                with self.subTest(variant=variant): self.assert_serial_residual(variant)

    def assert_serial_residual(self, variant, order=(8, 4)):
        t = self.torch
        with t.inference_mode():
            lm, (prefix, context) = self.model(variant), self.inputs(variant)
            core, projected, gate_inputs = lm.model, [], []
            handles = [core.audio_linear.register_forward_pre_hook(lambda _m, values: projected.append(values[0].clone()))]
            for gate in (core.audio_gate, core.audio_gates["8"], core.audio_gates["4"]):
                handles.append(gate.register_forward_pre_hook(lambda _m, values: gate_inputs.append(values[0].clone())))
            try:
                with patch.object(adapter, "HIDDEN_SIZE", 36):
                    audit = adapter.install_recursive_trace_audit(lm, eval_variant=variant)
                    self.call(lm, prefix, context)
            finally:
                for handle in handles: handle.remove()
            expected_events = [event for boundary, layer in zip((5, 15), order)
                               for event in ([boundary, "input"], [boundary, f"layer{layer}"])]
            self.assertEqual(audit["last_branch_events"], expected_events)
            self.assertEqual(len(projected), 1)
            t.testing.assert_close(projected[0], prefix[:, :126], rtol=0, atol=0)
            information = core.audio_linear(prefix[:, :126])
            for index in (0, 2):
                before = gate_inputs[index]
                expected = before + t.tanh(core.audio_gate(before)) * information
                t.testing.assert_close(gate_inputs[index + 1], expected, rtol=0, atol=0)
                self.assertFalse(t.equal(gate_inputs[index + 1], before))

    def test_loop_lora_strict_restore_covers_shared_base_and_both_active_adapter_sets(self):
        for variant in LORA_VARIANTS:
            with self.subTest(variant=variant): self.assert_loop_lora_strict_restore(variant)

    def assert_loop_lora_strict_restore(self, variant):
        t = self.torch
        with t.inference_mode():
            lm, (prefix, context) = self.model(variant), self.inputs(variant)
            projection = lm.model.layers[5].get_submodule("self_attn.q_proj")
            projection.weight.add_(.013)
            modality = adapter.variant_spec(variant).get("modality_lora")
            a_groups = list(projection.lora_A.values()) if modality else [projection.lora_A]
            b_groups = list(projection.lora_B.values()) if modality else [projection.lora_B]
            for a, b in zip(a_groups, b_groups):
                a[0].add_(.021)
                b[1].add_(.034)
            state = copy.deepcopy(lm.state_dict())
            matrices = [p for name, p in lm.named_parameters() if ".lora_A." in name or ".lora_B." in name]
            count = adapter.variant_spec(variant)["loop_lora_config"]["matrix_count"]
            self.assertEqual(len(matrices), count)
            self.assertEqual(len({p.untyped_storage().data_ptr() for p in matrices}), count)
            expected = self.call(lm, prefix, context).logits.clone()
            restored = self.model(variant)
            restored.load_state_dict(state, strict=True)
            self.assertTrue(t.equal(self.call(restored, prefix, context).logits, expected))
            for key, value in restored.state_dict().items(): self.assertTrue(t.equal(value, state[key]), key)
            suffixes = ["weight"] + [f"lora_{matrix}.{region}{loop}" for matrix in ("A", "B")
                for region in (("audio.", "text.") if modality else ("",)) for loop in (0, 1)]
            for suffix in suffixes:
                incomplete = dict(state)
                incomplete.pop("model.layers.5.self_attn.q_proj." + suffix)
                with self.subTest(suffix=suffix), self.assertRaises(RuntimeError):
                    restored.load_state_dict(incomplete, strict=True)
            for physical in range(5, 15):
                for target in adapter.LOOP_LORA_TARGETS:
                    module = lm.model.layers[physical].get_submodule(target)
                    for group in (module.lora_B.values() if modality else (module.lora_B,)):
                        group[0].zero_(); group[1].zero_()
            self.assertFalse(t.equal(self.call(lm, prefix, context).logits[:, 382], expected[:, 382]))
            lm.load_state_dict(state, strict=True)
            for physical in range(5, 15):
                for target in adapter.LOOP_LORA_TARGETS:
                    module = lm.model.layers[physical].get_submodule(target)
                    for bank in (module.lora_A, module.lora_B):
                        for matrices in (bank.values() if modality else (bank,)):
                            first, second = matrices[0].clone(), matrices[1].clone()
                            matrices[0].copy_(second); matrices[1].copy_(first)
            self.assertFalse(t.equal(self.call(lm, prefix, context).logits[:, 382], expected[:, 382]))

    def test_loop_lora_hooks_after_eval_freeze_reject_wrong_loop_and_reset_after_exception(self):
        for variant in LORA_VARIANTS:
            with self.subTest(variant=variant): self.assert_loop_lora_hooks_after_freeze(variant)

    def assert_loop_lora_hooks_after_freeze(self, variant):
        t = self.torch
        with t.inference_mode():
            lm, (prefix, context) = self.model(variant), self.inputs(variant)
            lm.requires_grad_(False)
            with patch.object(adapter, "HIDDEN_SIZE", 36):
                audit = adapter.install_recursive_trace_audit(lm, eval_variant=variant)
                expected = self.call(lm, prefix, context).logits.clone()
                self.call(lm, prefix * -2, context)
                self.assertTrue(t.equal(self.call(lm, prefix, context).logits, expected))
                self.assertEqual(audit["loop_lora_projection_calls"], 140)
                self.assertEqual(audit["loop_lora_calls_per_loop"], [70, 70])
                projection = lm.model.layers[5].get_submodule("self_attn.q_proj")
                handle = projection.register_forward_pre_hook(lambda module, _inputs: setattr(module, "loop_index", 1), prepend=True)
                try:
                    with self.assertRaisesRegex(RuntimeError, "wrong physical layer or loop"):
                        self.call(lm, prefix, context)
                finally:
                    handle.remove()
                for physical in range(5, 15):
                    for target in adapter.LOOP_LORA_TARGETS:
                        self.assertIsNone(lm.model.layers[physical].get_submodule(target).loop_index)
                self.assertTrue(t.equal(self.call(lm, prefix, context).logits, expected))
                self.assertTrue(audit["loop_lora_indices_reset"])

    def test_mesh_loop_lora_preserves_five_slots_router_queries_and_post_read_ced4_then_ced8(self):
        t = self.torch
        variant = "recursive_mesh_layer4_8_inject_loop_lora_r8"
        with t.inference_mode():
            lm, (prefix, context) = self.model(variant), self.inputs(variant)
            core, gate_inputs = lm.model, {}
            self.assertFalse(hasattr(core, "audio_linear"))
            self.assertFalse(hasattr(core, "audio_gate"))
            self.assertEqual(len(core.write_routers) + len(core.read_routers), 6)
            core.audit_mode = True
            def capture(layer):
                return lambda _m, values: gate_inputs.setdefault(layer, []).append(values[0].clone())
            handles = [core.audio_gates[str(layer)].register_forward_pre_hook(capture(layer)) for layer in (4, 8)]
            try:
                expected_logits = self.call(lm, prefix, context).logits.clone()
            finally:
                for handle in handles: handle.remove()
            memory = core.last_initial_memory.clone()
            self.assertEqual(tuple(memory.shape), (1, 5, 383, 36))
            t.testing.assert_close(memory[:, 0], prefix, rtol=0, atol=0)
            self.assertEqual(int(t.count_nonzero(memory[:, 1:])), 0)
            self.assertEqual(len(core.last_router_queries), 6)
            self.assertEqual(len(core.last_memory_write_history), 3)
            self.assertEqual(core.last_injection_trace, ((5, 4), (15, 8)))
            self.assertEqual(list(core.last_loop_lora_trace),
                             [(i, i if i < 15 else i - 10, 0 if i < 15 else 1) for i in range(5, 25)])
            for stage, suffix in enumerate(("pre", "0", "1")):
                query = core.last_prefix_output if stage == 0 else core.last_core_inputs[stage - 1]
                value = core.last_prefix_output if stage == 0 else core.last_core_outputs[stage - 1]
                for action in ("write", "read"):
                    t.testing.assert_close(core.last_router_queries[f"{action}_{suffix}"], query, rtol=0, atol=0)
                memory += value.unsqueeze(1) * core.last_router_weights[f"write_{suffix}"].transpose(1, 2).unsqueeze(-1)
                t.testing.assert_close(memory, core.last_memory_write_history[stage], rtol=0, atol=0)
                read = (memory * core.last_router_weights[f"read_{suffix}"].transpose(1, 2).unsqueeze(-1)).sum(1)
                if stage < 2:
                    layer = (4, 8)[stage]
                    self.assertEqual(len(gate_inputs[layer]), 1)
                    t.testing.assert_close(gate_inputs[layer][0], read[:, :126], rtol=0, atol=0)
                    audio = read[:, :126] + t.tanh(core.audio_gates[str(layer)](read[:, :126])) * context["audio_information"][stage]
                    combined = t.cat((audio, read[:, 126:]), dim=1)
                    t.testing.assert_close(core.last_core_inputs[stage], combined, rtol=0, atol=0)
                    t.testing.assert_close(core.last_core_inputs[stage][:, 126:], read[:, 126:], rtol=0, atol=0)
            self.call(lm, prefix * -2, context)
            self.assertTrue(t.equal(self.call(lm, prefix, context).logits, expected_logits))
            core.audit_mode = False
            with patch.object(adapter, "HIDDEN_SIZE", 36):
                audit = adapter.install_recursive_trace_audit(lm, eval_variant=variant)
                self.assertTrue(t.equal(self.call(lm, prefix, context).logits, expected_logits))
                extended = t.cat((prefix, t.randn(1, 1, 36)), dim=1)
                self.call(lm, extended, context)
                self.assertEqual(audit["last_sequence_tokens"], 384)
                self.assertEqual(audit["loop_lora_calls_per_loop"], [70, 70])

    def test_textmesh_memory_values_queries_final_audio_bypass_and_generated_text_growth(self):
        t = self.torch
        for variant in LORA_VARIANTS:
            if not adapter.variant_spec(variant).get("text_only_memory"): continue
            with self.subTest(variant=variant), t.inference_mode():
                lm, (prefix, context) = self.model(variant), self.inputs(variant)
                core, suffix_inputs = lm.model, []
                core.audit_mode = True
                handle = core.layers[15].register_forward_pre_hook(lambda _m, values: suffix_inputs.append(values[0].clone()))
                try:
                    expected = self.call(lm, prefix, context).logits.clone()
                finally:
                    handle.remove()
                memory = core.last_initial_memory.clone()
                self.assertEqual(tuple(memory.shape), (1, 5, 257, 36))
                t.testing.assert_close(memory[:, 0], prefix[:, 126:], rtol=0, atol=0)
                self.assertEqual(int(t.count_nonzero(memory[:, 1:])), 0)
                for stage, suffix in enumerate(("pre", "0", "1")):
                    query = core.last_prefix_output if stage == 0 else core.last_core_inputs[stage - 1]
                    value = core.last_prefix_output if stage == 0 else core.last_core_outputs[stage - 1]
                    for action in ("write", "read"):
                        t.testing.assert_close(core.last_router_queries[f"{action}_{suffix}"], query[:, 126:], rtol=0, atol=0)
                    memory += value[:, 126:].unsqueeze(1) * core.last_router_weights[f"write_{suffix}"].transpose(1, 2).unsqueeze(-1)
                    t.testing.assert_close(memory, core.last_memory_write_history[stage], rtol=0, atol=0)
                    read = (memory * core.last_router_weights[f"read_{suffix}"].transpose(1, 2).unsqueeze(-1)).sum(1)
                    target = core.last_core_inputs[stage] if stage < 2 else suffix_inputs[0]
                    t.testing.assert_close(target[:, 126:], read, rtol=0, atol=0)
                t.testing.assert_close(suffix_inputs[0][:, :126], core.last_core_outputs[1][:, :126], rtol=0, atol=0)
                core.audit_mode = False
                lm.requires_grad_(False)
                with patch.object(adapter, "HIDDEN_SIZE", 36):
                    audit = adapter.install_recursive_trace_audit(lm, eval_variant=variant)
                    extended = t.cat((prefix, t.randn(1, 2, 36)), dim=1)
                    self.call(lm, extended, context)
                    self.assertEqual(audit["last_text_memory_shape"], [1, 5, 259, 36])
                    if adapter.variant_spec(variant)["modality_lora"]:
                        self.assertEqual(audit["modality_lora_execution"]["text_tokens"], 259)
                    self.assertTrue(t.equal(self.call(lm, prefix, context).logits, expected))
                    self.assertEqual(audit["last_text_memory_shape"], [1, 5, 257, 36])

    def test_modality_lora_projection_boundary_includes_separator_padding_and_generated_tokens(self):
        t = self.torch
        variant = "recursive_textmesh_layer8_4_inject_audio_reinject_loop_modality_lora_r8"
        lm = self.model(variant)
        with t.inference_mode():
            for physical in range(5, 15):
                for target in adapter.LOOP_LORA_TARGETS:
                    module = lm.model.layers[physical].get_submodule(target)
                    inputs = t.randn(1, 385, module.in_features)
                    base = t.nn.functional.linear(inputs, module.weight, module.bias)
                    state = {m: [b.clone() for b in module.lora_B[m]] for m in ("audio", "text")}
                    try:
                        for loop in (0, 1):
                            for modality, region, other in (("audio", slice(None, 126), slice(126, None)),
                                                            ("text", slice(126, None), slice(None, 126))):
                                for m in ("audio", "text"):
                                    for b in module.lora_B[m]: b.zero_()
                                module.lora_B[modality][loop].copy_(state[modality][loop])
                                module.loop_index = loop
                                actual = module(inputs)
                                delta = t.nn.functional.linear(t.nn.functional.linear(inputs[:, region],
                                    module.lora_A[modality][loop]), module.lora_B[modality][loop])
                                t.testing.assert_close(actual[:, region], base[:, region] + 2 * delta, rtol=0, atol=0)
                                t.testing.assert_close(actual[:, other], base[:, other], rtol=0, atol=0)
                                self.assertFalse(t.equal(actual[:, region], base[:, region]))
                                if modality == "text":
                                    for position in (126, 382, 383, 384):
                                        self.assertFalse(t.equal(actual[:, position], base[:, position]))
                    finally:
                        module.loop_index = None
                        for m in ("audio", "text"):
                            for b, saved in zip(module.lora_B[m], state[m]): b.copy_(saved)

    def test_rank8_rank16_and_modality_checkpoints_cannot_cross_load(self):
        nonmesh16 = "recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r16"
        pairs = (("recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8", nonmesh16),
                 ("recursive_textmesh_layer8_4_inject_audio_reinject_loop_lora_r16",
                  "recursive_textmesh_layer8_4_inject_audio_reinject_loop_modality_lora_r8"))
        for first, second in pairs:
            a, b = self.model(first), self.model(second)
            with self.subTest(pair=(first, second)), self.assertRaises(RuntimeError):
                a.load_state_dict(b.state_dict(), strict=True)
            with self.subTest(pair=(second, first)), self.assertRaises(RuntimeError):
                b.load_state_dict(a.state_dict(), strict=True)

    def test_four_loop_mesh_uses_full_sequence_seven_slots_five_transitions_and_fresh_memory(self):
        t = self.torch
        variant = "recursive_mesh_5_10x4_5"
        with t.inference_mode():
            lm, (prefix, context) = self.model(variant), self.inputs(variant)
            core = lm.model
            core.audit_mode = True
            output = self.call(lm, prefix, context).logits.clone()
            memory = core.last_initial_memory.clone()
            self.assertEqual(tuple(memory.shape), (1, 7, 383, 36))
            t.testing.assert_close(memory[:, 0], prefix, rtol=0, atol=0)
            self.assertEqual(int(t.count_nonzero(memory[:, 1:])), 0)
            self.assertEqual(len(core.last_core_inputs), 4)
            self.assertEqual(len(core.last_router_queries), 10)
            self.assertEqual(len(core.last_memory_write_history), 5)
            for stage, suffix in enumerate(("pre", "0", "1", "2", "3")):
                query = core.last_prefix_output if stage == 0 else core.last_core_inputs[stage - 1]
                value = core.last_prefix_output if stage == 0 else core.last_core_outputs[stage - 1]
                for action in ("write", "read"):
                    t.testing.assert_close(core.last_router_queries[f"{action}_{suffix}"], query, rtol=0, atol=0)
                memory += value.unsqueeze(1) * core.last_router_weights[f"write_{suffix}"].transpose(1, 2).unsqueeze(-1)
                t.testing.assert_close(memory, core.last_memory_write_history[stage], rtol=0, atol=0)
            self.call(lm, prefix * -2, {})
            self.assertTrue(t.equal(self.call(lm, prefix, {}).logits, output))
            core.audit_mode = False
            with patch.object(adapter, "HIDDEN_SIZE", 36):
                adapter.install_recursive_trace_audit(lm, eval_variant=variant)
                core.ablate_after_two_loops = True
                with self.assertRaisesRegex(RuntimeError, "four trained MeSH loops"):
                    self.call(lm, prefix, {})

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
                    gates = core.audio_gates.values() if spec["interaction_layers"] else []
                    handles = [gate.register_forward_pre_hook(lambda *_: events.append("listen" if spec["kind"] == "listen48" else "inject")) for gate in gates]
                    handle = types.SimpleNamespace(remove=lambda: [h.remove() for h in handles])
                try:
                    # Test-only observation of real methods; production uses read-only module hooks.
                    with patch.object(core, "_write", side_effect=observed_write), patch.object(core, "_read", side_effect=observed_read):
                        self.call(lm, prefix, context)
                finally:
                    handle.remove()
                if spec["split_memory"]:
                    boundary = ["write", "write", "read", "read"]
                    extra = ["inject"] if spec["interaction_layers"] else []
                    self.assertEqual(events, (boundary + extra) * 2 + boundary)
                elif spec["kind"] == "mesh_x4":
                    self.assertEqual(events, ["write", "read"] * 5)
                else:
                    boundary = ["write", "read", "inject"] if spec["kind"] in ("inject48", "inject84", "serial") else ["write", "listen", "read"]
                    self.assertEqual(events, boundary * 2 + ["write", "read"])

    def test_context_rejects_wrong_features_masks_dtype_device_and_nonfinite_values(self):
        t = self.torch
        for variant in adapter.ROUTE_SPECS:
            prefix, context = self.inputs(variant)
            with patch.object(adapter, "HIDDEN_SIZE", 36):
                self.assertIs(adapter.validate_context(variant, context, prefix), context)
                bad = copy.deepcopy(context)
                key = "audio_information" if "audio_information" in bad else "audio_memory_seeds"
                if key in bad:
                    bad[key] = (bad[key][0].double(), *bad[key][1:])
                elif not bad:
                    bad["audio_information"] = (t.randn(1, 126, 36),) * 2
                else:
                    bad["prompt_mask"].zero_()
                with self.assertRaises(ValueError): adapter.validate_context(variant, bad, prefix)
                if not context: continue
                bad = copy.deepcopy(context)
                value = bad[key][0] if key in bad else bad["audio_grids"]
                value.flatten()[0] = float("nan")
                with self.assertRaises(ValueError): adapter.validate_context(variant, bad, prefix)

    def test_audio7_actual_seeds_router_queries_writes_and_post_read_injections(self):
        t = self.torch
        for variant, spec in adapter.ROUTE_SPECS.items():
            if not spec["split_memory"]: continue
            with self.subTest(variant=variant), t.inference_mode():
                lm, (prefix, context) = self.model(variant), self.inputs(variant)
                core = lm.model
                # Test-only diagnostics observe the actual training forward.
                core.audit_mode = True
                self.call(lm, prefix, context)
                audio, text = core.last_initial_memory
                self.assertEqual(tuple(audio.shape), (1, 7, 126, 36))
                self.assertEqual(tuple(text.shape), (1, 5, 257, 36))
                seeds = context.get("audio_memory_seeds", (prefix[:, :126], *context.get("audio_information", ())))
                t.testing.assert_close(audio[:, 0], seeds[0], rtol=0, atol=0)
                t.testing.assert_close(text[:, 0], prefix[:, 126:], rtol=0, atol=0)
                self.assertEqual(int(t.count_nonzero(audio[:, 1:5])), 0)
                self.assertEqual(int(t.count_nonzero(text[:, 1:])), 0)
                for slot, feature in zip((5, 6), seeds[1:]):
                    t.testing.assert_close(audio[:, slot], feature, rtol=0, atol=0)
                memory = [audio.clone(), text.clone()]
                for stage, suffix in enumerate(("pre", "0", "1")):
                    query = core.last_prefix_output if stage == 0 else core.last_core_inputs[stage - 1]
                    value = core.last_prefix_output if stage == 0 else core.last_core_outputs[stage - 1]
                    read_hidden = []
                    for region_index, (stem, region) in enumerate((("audio_", slice(None, 126)), ("", slice(126, None)))):
                        for action in ("write", "read"):
                            t.testing.assert_close(core.last_router_queries[f"{stem}{action}_{suffix}"], query[:, region], rtol=0, atol=0)
                        weights = core.last_router_weights[f"{stem}write_{suffix}"]
                        memory[region_index] += value[:, region].unsqueeze(1) * weights.transpose(1, 2).unsqueeze(-1)
                        t.testing.assert_close(core.last_memory_write_history[stage][region_index], memory[region_index], rtol=0, atol=0)
                        read_weights = core.last_router_weights[f"{stem}read_{suffix}"]
                        read_hidden.append((memory[region_index] * read_weights.transpose(1, 2).unsqueeze(-1)).sum(1))
                    expected = t.cat(read_hidden, dim=1)
                    if stage < 2:
                        if spec["interaction_layers"]:
                            layer = spec["interaction_layers"][stage]
                            gate = t.tanh(core.audio_gates[str(layer)](expected[:, :126]))
                            residual = gate * context["audio_information"][stage]
                            t.testing.assert_close(core.last_injection_audit[stage]["read_hidden"], expected, rtol=0, atol=0)
                            expected = t.cat((expected[:, :126] + residual, expected[:, 126:]), dim=1)
                        t.testing.assert_close(core.last_core_inputs[stage], expected, rtol=0, atol=0)
                self.assertEqual(len(core.last_router_queries), 12)
                if spec["interaction_layers"]:
                    self.assertEqual(core.last_injection_trace, ((5, 4), (15, 8)))
                else:
                    self.assertFalse(hasattr(core, "audio_gates"))

    def test_production_width_structure_audit_uses_actual_training_validators(self):
        t = self.torch
        for variant, spec in adapter.ROUTE_SPECS.items():
            contract, implementation, loader, _ = self.modules[variant]
            lm = self.model(variant, width=576)
            base = t.nn.Module()
            if spec["context_kind"] in ("pair", "seeds"):
                base.layer_mlps = t.nn.ModuleDict({str(k): t.nn.Sequential(t.nn.LayerNorm(1536),
                    t.nn.Linear(1536, 576), t.nn.GELU(), t.nn.Linear(576, 576), t.nn.LayerNorm(576)) for k in (4, 8)})
            model = types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=lm,
                is_recursive_mesh=spec["mesh"], is_recursive_5_10_5=not spec["mesh"],
                text_model_contract=spec["text_contract"]), audio_encoder=types.SimpleNamespace(base=base))
            if spec["kind"] == "prompt_mix":
                model.caption_decoder.prompt_mix_mlp = t.nn.Sequential(t.nn.Linear(576, 576), t.nn.GELU(), t.nn.Linear(576, 3))
            identity = {"synthetic_text": True}
            with patch.dict(sys.modules, self.namespaces[variant]), patch.object(contract, "recursive_checkpoint_contract", return_value=identity):
                recipe = {"text_model_dir": "synthetic", "recursive_text_checkpoint": identity,
                          "recursive_model_source_sha256": contract.recursive_model_source_sha256()}
                audit = adapter.audit_recursive_route(contract, recipe, model, eval_variant=variant)
                self.assertEqual(audit["new_parameter_count"], spec["new_parameters"])
                if spec["split_memory"]:
                    self.assertEqual((audit["audio_memory_slots"], audit["text_memory_slots"], audit["routers"]), (7, 5, 12))
                    with self.assertRaises(ValueError): loader.validate_audio_listen(lm, fresh=True)
                lm.model.layers[1] = lm.model.layers[0]
                with self.assertRaises(ValueError): adapter.audit_recursive_route(contract, recipe, model, eval_variant=variant)


if __name__ == "__main__":
    unittest.main()
