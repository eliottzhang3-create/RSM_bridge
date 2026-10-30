"""Real CPU tensor/autograd tests, synthetic weights only, no downloads or CUDA.

Checkpoint tests mock eight-rank collectives/CUDA RNG; they do not qualify a
real eight-GPU resume. Missing dependencies produce explicit skips.
"""
import ast
import copy
import importlib.util
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer36912_listen"
REQUIRED = ("torch", "torchaudio", "transformers", "soundfile", "numpy")
HAS_TENSORS = all(importlib.util.find_spec(name) is not None for name in REQUIRED)


@unittest.skipUnless(HAS_TENSORS, "requires torch/torchaudio/transformers/soundfile/numpy")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(ROUTE), str(ROOT / "code/RSmol")]
        import numpy as np
        import torch
        import contract
        import train
        from transformers import LlamaConfig
        from models.audio_listen import MultilayerAudioListen
        from models.recursive_mesh_listen import RecursiveLlamaForCausalLM
        from recursive_model_5_10x2_5_mesh import RecursiveLlamaForCausalLM as OriginalLM
        from models.ced import CEDSmallWrapper, CEDTemporalMapper
        from models.decoder import DecoderModel
        from models import recursive_text
        from optim import build_optimizer, GradNormTracker
        cls.torch, cls.np, cls.contract, cls.train = torch, np, contract, train
        cls.config_class, cls.lm_class, cls.original_class = LlamaConfig, RecursiveLlamaForCausalLM, OriginalLM
        cls.adapter_class, cls.wrapper_class, cls.mapper_class = MultilayerAudioListen, CEDSmallWrapper, CEDTemporalMapper
        cls.decoder_class, cls.loader = DecoderModel, recursive_text
        cls.build_optimizer, cls.tracker_class = staticmethod(build_optimizer), GradNormTracker
        torch.set_num_threads(1)

    def lm(self, enable=True):
        config = self.config_class(vocab_size=32, hidden_size=24, intermediate_size=32,
            num_hidden_layers=30, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=1024, eos_token_id=0, pad_token_id=17,
            use_cache=False, attention_dropout=0.)
        config.recursive_layer_count, config.recursive_loops = 20, 2
        model = self.lm_class(config).float()
        if enable:
            model.model.enable_audio_listen()
        return model

    def context(self, length=389):
        t = self.torch
        return t.randn(1, length, 24), t.randn(1, 4, 126, 1536), t.cat((t.ones(1, 180), t.zeros(1, 76)), 1).long()

    def toy_wrapper(self):
        t, nn = self.torch, self.torch.nn
        class Block(nn.Module):
            def __init__(self, index):
                super().__init__()
                self.offset = nn.Parameter(t.tensor(float(index)), requires_grad=False)
            def forward(self, x):
                return x + self.offset
        class CED(nn.Module):
            maximal_allowed_length = 1012
            def __init__(self):
                super().__init__()
                self.init_bn = nn.BatchNorm2d(64)
                self.blocks = nn.Sequential(*(Block(i) for i in range(1, 13)))
                self.norm = nn.LayerNorm(384)
            def forward_features(self, x):
                self.calls = getattr(self, "calls", 0) + 1
                self.base = t.arange(x.shape[0]*252*384, dtype=x.dtype).reshape(x.shape[0], 252, 384)/1024
                return self.norm(self.blocks(self.base))
        wrapper = self.wrapper_class.__new__(self.wrapper_class)
        nn.Module.__init__(wrapper)
        wrapper.ced = CED().eval()
        for p in wrapper.ced.parameters():
            p.requires_grad = False
        wrapper.mapper = self.mapper_class(384, 4, 576, output_tokens=126, resample=False)
        wrapper.mapper_type = "freq_merge_126"
        wrapper._waveform_to_mel = lambda wave: t.zeros(wave.shape[0], 64, 2001)
        return wrapper

    def fixture(self):
        t, nn = self.torch, self.torch.nn
        t.manual_seed(1234)
        decoder = self.decoder_class.__new__(self.decoder_class)
        nn.Module.__init__(decoder)
        decoder.lm, decoder.lm_embedding_size = self.lm(), 24
        model = nn.Module()
        model.audio_encoder, model.caption_decoder = nn.Module(), decoder
        model.audio_encoder.base = nn.Module()
        model.audio_encoder.base.mapper = self.mapper_class(384, 4, 24, output_tokens=126, resample=False)
        targets = t.randint(1, 16, (1, 300))
        targets[:, 199], targets[:, 200:] = 0, 17
        _, grids, mask = self.context()
        return model.train(), dict(grid=t.randn(1, 126, 4, 384), grids=grids,
            prompt={"input_ids": t.ones(1, 256, dtype=t.long), "attention_mask": mask},
            answer={"input_ids": targets, "attention_mask": targets.ne(17).long()})

    def loss(self, model, inputs):
        audio = model.audio_encoder.base.mapper(inputs["grid"])
        output = model.caption_decoder(audio, None, inputs["prompt"], inputs["answer"], audio_grids=inputs["grids"])
        self.assertEqual(tuple(output.logits.shape), (1, 683, 32))
        self.assertIsNone(output.past_key_values)
        return self.train.answer_loss(output, {"answer": inputs["answer"]}, 17)

    def test_production_parameter_inventory_rng_and_normalization(self):
        t = self.torch
        before = t.get_rng_state().clone()
        adapter = self.adapter_class()
        self.assertTrue(t.equal(before, t.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 3552788)
        self.assertEqual(len(list(adapter.parameters())), 20)
        self.assertEqual(len({id(m.weight) for m in adapter.projections}), 4)
        norms = list(adapter.audio_input_norms) + list(adapter.audio_output_norms) + [adapter.text_norm, adapter.query_norm]
        self.assertEqual(len({id(n) for n in norms}), 10)
        for norm in norms:
            self.assertEqual(norm.eps, 1e-5)
            self.assertFalse(norm.elementwise_affine)
            self.assertEqual(list(norm.parameters()), [])
        self.assertFalse(hasattr(adapter, "gamma"))
        for gate in adapter.gates:
            self.assertEqual(gate.weight.count_nonzero().item(), 0)
            self.assertEqual(gate.bias.count_nonzero().item(), 0)
            t.testing.assert_close(t.sigmoid(gate(t.randn(2, 576))), t.full((2, 4), .5), rtol=0, atol=0)
        self.assertEqual(len({id(m.weight) for m in adapter.amplitudes}), 2)
        for amplitude in adapter.amplitudes:
            self.assertEqual(tuple(amplitude.weight.shape), (1, 576))
            self.assertEqual(tuple(amplitude.bias.shape), (1,))
            self.assertEqual(amplitude.weight.count_nonzero().item(), 0)
            self.assertEqual(amplitude.bias.count_nonzero().item(), 0)
            t.testing.assert_close(t.tanh(amplitude(t.randn(2, 126, 576))), t.zeros(2, 126, 1), rtol=0, atol=0)
        for router in adapter.listen_routers:
            self.assertGreater(router.weight.count_nonzero().item(), 0)
            self.assertEqual(router.bias.count_nonzero().item(), 0)
            self.assertLessEqual(router.weight.abs().max().item(), 3*(2/(5*576))**.5 + 1e-7)

    def test_v1_projection_and_router_initial_weights_are_preserved(self):
        t, nn = self.torch, self.torch.nn
        t.manual_seed(721)
        adapter = self.adapter_class()
        # Recreate the v1 initialization sequence, before its scalar amplitudes.
        with t.random.fork_rng(devices=[]):
            projections = [nn.Linear(1536, 576) for _ in range(4)]
            gates = [nn.Linear(576, 4) for _ in range(2)]
            routers = [nn.Linear(576, 5) for _ in range(2)]
            for projection in projections:
                nn.init.xavier_uniform_(projection.weight)
                nn.init.zeros_(projection.bias)
            for gate in gates:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
            with t.no_grad():
                for router in routers:
                    std = math.sqrt(2.0 / (5 * 576))
                    values = t.empty_like(router.weight)
                    values.normal_(0.0, std).clamp_(-3.0 * std, 3.0 * std)
                    router.weight.copy_(values)
                    router.bias.zero_()
            for old, new in zip(projections + gates + routers,
                                list(adapter.projections) + list(adapter.gates) + list(adapter.listen_routers)):
                t.testing.assert_close(new.weight, old.weight, rtol=0, atol=0)
                t.testing.assert_close(new.bias, old.bias, rtol=0, atol=0)

    def test_real_parameter_inventory_loader_rejects_scalar_or_shared_gates(self):
        t, nn = self.torch, self.torch.nn
        adapter = self.adapter_class()
        core = SimpleNamespace(audio_listen=adapter, layers=[None]*20,
            write_routers=[nn.Linear(576, 5) for _ in range(3)],
            read_routers=[nn.Linear(576, 5) for _ in range(3)])
        self.assertEqual(self.loader.validate_audio_listen(SimpleNamespace(model=core))["actual_parameters"], 3552788)
        original = adapter.amplitudes
        adapter.amplitudes = nn.ParameterList([nn.Parameter(t.zeros(1)) for _ in range(2)])
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.loader.validate_audio_listen(SimpleNamespace(model=core))
        adapter.amplitudes = nn.ModuleList([original[0], original[0]])
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.loader.validate_audio_listen(SimpleNamespace(model=core))

    def test_single_ced_forward_tap_values_temporal_order_and_hook_cleanup(self):
        t = self.torch
        wrapper = self.toy_wrapper()
        final, layers = wrapper._forward_hidden(t.zeros(2, 64, 2001))
        self.assertEqual(wrapper.ced.calls, 1)
        for value, offset in zip(layers, (6, 21, 45, 78)):
            t.testing.assert_close(value, (wrapper.ced.base + offset).reshape(2, 504, 384), rtol=0, atol=0)
        t.testing.assert_close(final, wrapper.ced.norm(wrapper.ced.base+78).reshape(2, 504, 384), rtol=0, atol=0)
        raw = layers[1].reshape(2, 2, 4, 63, 384)
        grid = wrapper._hidden_to_temporal_grid(layers[1])
        t.testing.assert_close(grid[:, 63+19, 2], raw[:, 1, 2, 19], rtol=0, atol=0)
        self.assertTrue(all(not b._forward_hooks for b in wrapper.ced.blocks))
        with patch.object(wrapper.ced.blocks[7], "forward", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                wrapper._forward_hidden(t.zeros(1, 64, 2001))
        self.assertTrue(all(not b._forward_hooks for b in wrapper.ced.blocks))

    def test_ced_frozen_and_four_projections_outside_no_grad(self):
        t = self.torch
        wrapper = self.toy_wrapper().train()
        output = wrapper(t.zeros(1, 640000))
        self.assertEqual(tuple(output["layer_grids"].shape), (1, 4, 126, 1536))
        self.assertFalse(output["layer_grids"].requires_grad)
        adapter = self.adapter_class()
        hidden = t.randn(1, 383, 576)
        bank = adapter.project_audio(output["layer_grids"], hidden, t.ones(1, 256))
        ((bank*t.randn_like(bank)).mean()+output["embedding"].square().mean()).backward()
        self.assertTrue(all(p.grad is not None and t.isfinite(p.grad).all() and p.grad.ne(0).any()
                            for p in adapter.projections.parameters()))
        self.assertTrue(all(p.grad is None for p in wrapper.ced.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.ne(0).any() for p in wrapper.mapper.parameters()))
        self.assertFalse(wrapper.ced.training)

    def test_zero_token_gates_logits_loss_and_original_gradients_equal_mother(self):
        t = self.torch
        t.manual_seed(7)
        model = self.lm(enable=False)
        original = self.original_class(model.config)
        original.load_state_dict(model.state_dict(), strict=True)
        rng = t.get_rng_state().clone()
        model.model.enable_audio_listen()
        self.assertTrue(t.equal(rng, t.get_rng_state()))
        model.eval(); original.eval()
        x, grids, mask = self.context()
        y = model(inputs_embeds=x, use_cache=False, audio_grids=grids, prompt_mask=mask).logits
        z = original(inputs_embeds=x, use_cache=False).logits
        t.testing.assert_close(y, z, rtol=0, atol=0)
        targets = t.arange(6).view(1, 6)
        loss = t.nn.functional.cross_entropy(y[:, -6:].transpose(1, 2), targets)
        old_loss = t.nn.functional.cross_entropy(z[:, -6:].transpose(1, 2), targets)
        t.testing.assert_close(loss, old_loss, rtol=0, atol=0)
        loss.backward(); old_loss.backward()
        for name, parameter in original.named_parameters():
            other = dict(model.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            t.testing.assert_close(other.grad, parameter.grad, rtol=0, atol=0, msg=name)

    def test_schedule_two_writes_original_router_queries_and_single_projection(self):
        t = self.torch
        model = self.lm().eval()
        adapter, core = model.model.audio_listen, model.model
        with t.no_grad():
            for amplitude in adapter.amplitudes:
                amplitude.bias.fill_(.4)
        calls, projections, rounds = [], [], []
        amplitude_inputs = []
        handles = [m.register_forward_pre_hook(lambda _, args, i=i: calls.append(i)) for i, m in enumerate(core.layers)]
        handles += [m.register_forward_hook(lambda *args, i=i: projections.append(i)) for i, m in enumerate(adapter.projections)]
        handles += [m.register_forward_pre_hook(lambda _, args, i=i: amplitude_inputs.append((i, args[0].detach().cpu().clone())))
                    for i, m in enumerate(adapter.amplitudes)]
        normal_listen = adapter.forward
        def listen(memory, boundary, bank, mask, index, reads):
            rounds.append((index, id(bank)))
            out = normal_listen(memory, boundary, bank, mask, index, reads)
            t.testing.assert_close(out[:, :, 126:], memory[:, :, 126:], rtol=0, atol=0)
            return out
        try:
            x, grids, mask = self.context()
            core.audit_mode = True
            with patch.object(adapter, "forward", side_effect=listen), t.no_grad():
                model(inputs_embeds=x, audio_grids=grids, prompt_mask=mask, use_cache=False)
            self.assertEqual(calls, list(self.contract.TEXT_MODEL_SCHEDULE))
            self.assertEqual(projections, [0, 1, 2, 3])
            self.assertEqual([r for r, _ in rounds], [0, 1])
            self.assertEqual(rounds[0][1], rounds[1][1])
            self.assertEqual([i for i, _ in amplitude_inputs], [0, 1])
            t.testing.assert_close(amplitude_inputs[0][1], core.last_prefix_output[:, :126], rtol=0, atol=0)
            t.testing.assert_close(amplitude_inputs[1][1], core.last_core_outputs[0][:, :126], rtol=0, atol=0)
            for i in range(2):
                for kind in ("write", "read"):
                    t.testing.assert_close(core.last_router_queries[f"{kind}_{i}"], core.last_core_inputs[i], rtol=0, atol=0)
        finally:
            for h in handles:
                h.remove()

    def test_answer_perturbation_does_not_change_gates_or_earlier_answer_logits(self):
        t = self.torch
        model = self.lm().eval()
        adapter = model.model.audio_listen
        with t.no_grad():
            for amplitude in adapter.amplitudes:
                amplitude.weight.normal_(0, .1)
                amplitude.bias.fill_(.5)
            for gate in adapter.gates:
                gate.weight.normal_(0, .1)
        x, grids, mask = self.context()
        gates = []
        handles = [m.register_forward_hook(lambda _, args, output: gates.append(output.detach().clone()))
                   for m in list(adapter.gates) + list(adapter.amplitudes)]
        try:
            with t.no_grad():
                a = model(inputs_embeds=x, audio_grids=grids, prompt_mask=mask, use_cache=False).logits
                changed = x.clone(); changed[:, 386:] += t.randn_like(changed[:, 386:])*10
                b = model(inputs_embeds=changed, audio_grids=grids, prompt_mask=mask, use_cache=False).logits
            t.testing.assert_close(a[:, :386], b[:, :386], rtol=0, atol=0)
            for first, second in zip(gates[:4], gates[4:]):
                t.testing.assert_close(first, second, rtol=0, atol=0)
        finally:
            for h in handles:
                h.remove()

    def test_gate_pool_masks_pad_and_only_reads_prompt(self):
        t = self.torch
        adapter = self.adapter_class(width=24)
        x, _, mask = self.context()
        expected = adapter.pool_prompt(x, mask)
        changed = x.clone()
        changed[:, :127] += t.randn_like(changed[:, :127])*100
        changed[:, 127+180:] += t.randn_like(changed[:, 127+180:])*100
        t.testing.assert_close(adapter.pool_prompt(changed, mask), expected, rtol=0, atol=0)

    def test_extreme_write_has_no_fixed_scale_and_reports_token_gate_statistics(self):
        t = self.torch
        adapter = self.adapter_class(width=24)
        x, grids, mask = self.context()
        bank = adapter.project_audio(grids, x, mask)
        bank = bank[:, :1].expand(-1, 4, -1, -1)
        memory = t.zeros(1, 5, 389, 24)
        reads = t.zeros(1, 389, 5); reads[:, :, 0] = 1
        adapter.collect_diagnostics = True
        with t.no_grad():
            for gate, router in zip(adapter.gates, adapter.listen_routers):
                gate.bias.fill_(30); router.weight.zero_(); router.bias.fill_(-30); router.bias[0] = 30
            for sign in (-1, 1):
                adapter.last_diagnostics = []
                for i in range(2):
                    adapter.amplitudes[i].bias.fill_(sign*30)
                    delta = adapter(memory, x, bank, mask, i, reads)
                    t.testing.assert_close(delta[:, 0, :126], sign*4*bank[:, 0], rtol=1e-6, atol=1e-6)
                    self.assertLessEqual(delta.square().mean(-1).sqrt().max().item(), 4.00001)
                    self.assertGreater(delta.square().mean(-1).sqrt().max().item(), 3.99)
                    self.assertEqual(delta[:, :, 126:].count_nonzero().item(), 0)
                for stats in adapter.diagnostics():
                    self.assertLessEqual(stats["direct_read_token_rms_max"], 4.00001)
                    self.assertGreater(stats["direct_read_token_rms_max"], 3.99)
                    self.assertEqual(stats["depth_gate_saturation_fraction"], 1.)
                    self.assertEqual(stats["amplitude_mean"], float(sign))
                    self.assertEqual(stats["amplitude_min"], float(sign))
                    self.assertEqual(stats["amplitude_max"], float(sign))
                    self.assertEqual(stats["amplitude_saturation_fraction"], 1.)
                    self.assertNotIn("gamma", stats)
                    self.assertNotIn("beta", stats)

    def test_nonuniform_raw_audio_token_gates_formula_and_diagnostics(self):
        t = self.torch
        adapter = self.adapter_class(width=24)
        x, grids, mask = self.context()
        bank = adapter.project_audio(grids, x, mask)
        memory = t.zeros(1, 5, 389, 24)
        reads = t.full((1, 389, 5), .2)
        adapter.collect_diagnostics = True
        with t.no_grad():
            for r, layer in enumerate(adapter.amplitudes):
                layer.weight[0, 0] = r + 1
                layer.bias[0] = .25
                amplitude = t.tanh((r+1)*x[:, :126, :1]+.25)
                slot_probs = t.softmax(adapter.listen_routers[r](adapter.query_norm(x[:, :126])), -1)
                aggregate = bank.sum(1)*.5
                expected = amplitude.unsqueeze(1)*aggregate.unsqueeze(1)*slot_probs.transpose(1, 2).unsqueeze(-1)
                actual = adapter(memory, x, bank, mask, r, reads)
                t.testing.assert_close(actual[:, :, :126], expected, rtol=0, atol=0)
                self.assertEqual(actual[:, :, 126:].count_nonzero().item(), 0)
                stats = adapter.diagnostics()[-1]
                for name, value in (("mean", amplitude.mean()), ("min", amplitude.min()), ("max", amplitude.max()),
                                    ("abs_mean", amplitude.abs().mean()),
                                    ("saturation_fraction", (amplitude.abs() > .99).float().mean())):
                    self.assertEqual(stats["amplitude_"+name], value.item())

    def test_true_autograd_zero_start_and_unlock_after_first_optimizer_update(self):
        t = self.torch
        model, inputs = self.fixture()
        optimizer, scheduler, _ = self.build_optimizer(model, 17313)
        self.loss(model, inputs).backward()
        first = self.train.listen_gradient_audit(model)
        self.assertTrue(all(v["present"] and v["finite"] for v in first.values()))
        self.assertTrue(first["amplitude_0"]["nonzero"] and first["amplitude_1"]["nonzero"])
        for layer in model.caption_decoder.lm.model.audio_listen.amplitudes:
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
            self.assertGreater(layer.bias.grad.abs().sum().item(), 0)
        self.assertTrue(all(not v["nonzero"] for k, v in first.items() if not k.startswith("amplitude")))
        self.tracker_class().track_and_clip_(model.named_parameters())
        optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        self.loss(model, inputs).backward()
        second = self.train.listen_gradient_audit(model)
        self.assertTrue(all(v["present"] and v["finite"] and v["nonzero"] for v in second.values()), second)
        adapter = model.caption_decoder.lm.model.audio_listen
        adapter.capture_gradient_diagnostics(2)
        self.assertEqual(len(adapter.last_gradient_diagnostics["modules"]), 10)
        self.assertTrue(all(adapter.last_gradient_diagnostics["modules"][f"amplitude_{i}"]["present"] for i in range(2)))
        self.assertTrue(all(v["finite"] and v["l2"] > 0 for v in adapter.last_gradient_diagnostics["modules"].values()))

    def test_original_parameter_optimizer_groups_keep_lr_and_decay(self):
        model, _ = self.fixture()
        _, _, groups = self.build_optimizer(model, 17313)
        self.assertEqual({g["role"] for g in groups}, {"map", "default", "listen_projections", "listen_gates", "listen_routers", "listen_amplitudes"})
        for group in groups:
            self.assertEqual(group["base_lr"], 3e-4 if group["role"] == "default" else 6e-4)
            params = dict(model.named_parameters())
            for name in group["names"]:
                self.assertEqual(group["weight_decay"], 0. if params[name].ndim <= 1 or name.endswith(".bias") else .01)

    def test_listen_rejects_cache_attention_mask_and_invalid_context(self):
        t = self.torch
        model = self.lm()
        x, grids, mask = self.context()
        for kwargs in ({"use_cache": True}, {"use_cache": False, "attention_mask": t.ones(1, 389)},
                       {"use_cache": False, "past_key_values": object()}):
            with self.assertRaises(ValueError):
                model(inputs_embeds=x, audio_grids=grids, prompt_mask=mask, **kwargs)
        for bad in (None, t.zeros_like(mask), mask[:, :255], mask.float()*.5):
            with self.assertRaises(ValueError):
                model(inputs_embeds=x, audio_grids=grids, prompt_mask=bad, use_cache=False)

    def test_text_loader_rejects_every_partial_loading_category(self):
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            factory = SimpleNamespace(from_pretrained=lambda *a, **kw: (object(), {key: ["fixture"]}))
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory)/"config.json").write_text("{}")
                with patch.object(self.loader, "_import_recursive_model", return_value=(factory, None, lambda: None)), \
                     patch.object(self.loader, "_AUTO_CLASS_REGISTERED", False):
                    with self.assertRaisesRegex(ValueError, "did not load strictly"):
                        self.loader.load_recursive_text_model(directory)

    def test_full_checkpoint_cpu_continuation_matches_uninterrupted_and_final_reload(self):
        t, np, train = self.torch, self.np, self.train
        random.seed(10); np.random.seed(10)
        model, inputs = self.fixture()
        opt, sched, _ = self.build_optimizer(model, 17313)
        scaler, tracker = t.cuda.amp.GradScaler(enabled=False), self.tracker_class()
        state = {"deadline": 32, "seen": {}, "complete": False}
        model.caption_decoder.lm.model.audio_listen.gradient_audit_state = state
        prepared = dict(rows=1477492, rows_canonical_sha256="r", mapping_sha256="m", source_rows={})
        recipe = self.contract.semantic_contract(prepared, "text", "ced")
        config = {"mode": "formal", "contract": recipe, "contract_sha256": self.contract.digest(recipe)}
        def cpu_rng(device):
            return dict(python=random.getstate(), numpy=np.random.get_state(), torch=t.get_rng_state(), cuda=t.get_rng_state())
        def gather(dest, value):
            dest[:] = [copy.deepcopy(value) for _ in range(8)]
        def require(value, *args):
            if not value:
                raise AssertionError(args)
        def update(candidate, optimizer, scheduler, amp, norm, index, folder):
            random.random(); np.random.random()  # Include each saved generator in continuation evidence.
            amp.scale(self.loss(candidate, inputs)).backward()
            amp.unscale_(optimizer)
            train.audit_listen_update(candidate, candidate.caption_decoder.lm.model.audio_listen.gradient_audit_state,
                                      index, folder, 0, t.device("cpu"))
            norm.track_and_clip_(candidate.named_parameters())
            amp.step(optimizer); amp.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        with tempfile.TemporaryDirectory() as directory, patch.object(train, "rng_state", side_effect=cpu_rng), \
             patch.object(train.dist, "all_gather_object", side_effect=gather), patch.object(train.dist, "get_rank", return_value=0), \
             patch.object(train.dist, "barrier"), patch.object(train, "collective_require", side_effect=require):
            folder = Path(directory)
            for index in (1, 2, 3):
                update(model, opt, sched, scaler, tracker, index, folder)
            path = folder/"checkpoint.ckpt"
            train.save_checkpoint(path, model, opt, sched, scaler, tracker, config, 3, 0, 3, t.device("cpu"))
            saved = t.load(path, map_location="cpu", weights_only=False)
            self.contract.validate_resume_payload(saved, config, recipe["geometry"])
            update(model, opt, sched, scaler, tracker, 4, folder)
            expected_rng = cpu_rng(None)
            restored, _ = self.fixture()
            other, other_sched, _ = self.build_optimizer(restored, 17313)
            other_scaler, other_tracker = t.cuda.amp.GradScaler(enabled=False), self.tracker_class()
            restored.load_state_dict(saved["state_dict"], strict=True)
            other.load_state_dict(saved["optimizer"]); other_sched.load_state_dict(saved["scheduler"])
            other_scaler.load_state_dict(saved["grad_scaler"]); other_tracker.load_state_dict(saved["grad_norm_tracker"])
            restored.caption_decoder.lm.model.audio_listen.gradient_audit_state = copy.deepcopy(saved["listen_gradient_state"])
            rng = saved["random_state_by_rank"][0]
            random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); t.set_rng_state(rng["torch"])
            update(restored, other, other_sched, other_scaler, other_tracker, 4, folder)
            for name, value in model.state_dict().items():
                t.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0, msg=name)
            for a, b in zip(opt.state.values(), other.state.values()):
                for key in a:
                    t.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            self.assertEqual(sched.state_dict(), other_sched.state_dict())
            self.assertEqual(tracker.state_dict(), other_tracker.state_dict())
            self.assertEqual(scaler.state_dict(), other_scaler.state_dict())
            self.assertEqual(random.getstate(), expected_rng["python"])
            np.testing.assert_equal(np.random.get_state(), expected_rng["numpy"])
            self.assertTrue(t.equal(t.get_rng_state(), expected_rng["torch"]))
            # Exercise the production final-reload function with actual four-step
            # state. Completed-run cursor handling is separately tested without
            # pretending this CPU fixture trained all 17,313 formal updates.
            train.save_checkpoint(path, restored, other, other_sched, other_scaler, other_tracker,
                                  config, 4, 0, 4, t.device("cpu"))
            self.assertEqual(train.reload_audit(path, restored, config, 4)["status"], "PASS")
            damaged = t.load(path, weights_only=False); del damaged["listen_gradient_state"]
            t.save(damaged, path)
            with self.assertRaises(ValueError):
                train.reload_audit(path, restored, config, 4)


# Preserve each inherited check as an explicit skip when ML dependencies are
# absent; do not claim those fixtures executed on the basis of a wrapper test.
if HAS_TENSORS:
    sys.path[:0] = [str(ROUTE), str(ROOT / "code/RSmol")]
    from runtime_checks import RuntimeChecks as InheritedRuntimeChecks
else:
    class InheritedRuntimeChecks(unittest.TestCase):
        pass
    source = ast.parse((ROUTE / "runtime_checks.py").read_text(encoding="utf-8"))
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "RuntimeChecks")
    for method in cls.body:
        if isinstance(method, ast.FunctionDef) and method.name.startswith("test_"):
            setattr(InheritedRuntimeChecks, method.name, unittest.skip("requires real CPU tensor dependencies")(lambda self: None))

if __name__ == "__main__":
    unittest.main()
