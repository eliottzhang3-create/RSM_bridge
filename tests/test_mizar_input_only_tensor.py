"""Optional real CPU autograd/AdamW/resume tests with small dimensions, no assets."""
import copy
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_official_final_s1_gbs128_5_20x2_5_input_only"
sys.path.insert(0, str(ROUTE))
REQUIRED = ("torch", "transformers", "torchaudio", "numpy", "soundfile")
AVAILABLE = all(importlib.util.find_spec(x) for x in REQUIRED)


@unittest.skipUnless(AVAILABLE, "real Torch/Transformers/audio dependencies unavailable; no mocked tensor PASS")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import contract
        import optim
        import train
        from transformers import LlamaConfig
        from mizar_official_recursive_5_20x2_5_input_only.model import MizarInputOnlyRecursiveForCausalLM
        from mizar_official_recursive_5_20x2_5_input_only.tensor_runtime import config_from_dict
        cls.torch, cls.c, cls.optim, cls.train = torch, contract, optim, train
        cls.config_from_dict = staticmethod(config_from_dict)
        cls.LlamaConfig, cls.LM = LlamaConfig, MizarInputOnlyRecursiveForCausalLM
        torch.set_num_threads(1)

    def model(self):
        torch = self.torch
        cfg = self.LlamaConfig(vocab_size=32, hidden_size=24, intermediate_size=48,
                              num_hidden_layers=30, num_attention_heads=3, num_key_value_heads=1,
                              pad_token_id=17, eos_token_id=0, tie_word_embeddings=True,
                              use_cache=False, attention_dropout=0., max_position_embeddings=1024)
        value = cfg.to_dict()
        value.update(num_hidden_layers=50, recursive_layer_count=30, recursive_loops=2,
                     recursive_prefix_layer_count=5, recursive_middle_layer_count=20,
                     recursive_suffix_layer_count=5, logical_to_physical=list(self.c.TEXT_MODEL_SCHEDULE),
                     recursive_loops_scope="middle_only", recursive_source_layer_indices_0based=list(range(30)))
        model = torch.nn.Module()
        model.caption_decoder = torch.nn.Module()
        model.caption_decoder.lm = self.LM(self.config_from_dict(value)).float()
        model.caption_decoder.lm.model.enable_new_modules()
        model.audio_encoder = torch.nn.Module()
        model.audio_encoder.base = torch.nn.Module()
        base = model.audio_encoder.base
        base.mapper = torch.nn.Sequential(torch.nn.Linear(24, 24), torch.nn.Dropout(.05))
        base.ced = torch.nn.Linear(24, 24)
        for p in base.ced.parameters():
            p.requires_grad = False
        return model

    def loss(self, model):
        torch = self.torch
        base = model.audio_encoder.base
        features = torch.randn(1, 126, 24)
        audio = base.mapper(features)
        x = torch.cat((audio, torch.randn(1, 4, 24)), dim=1)
        return model.caption_decoder.lm(inputs_embeds=x).logits.square().mean()

    def update(self, model, optimizer, scheduler, tracker):
        optimizer.zero_grad(set_to_none=True)
        self.loss(model).backward()
        tracker.track_and_clip_(model.named_parameters())
        optimizer.step()
        scheduler.step()

    def test_optimizer_covers_each_unique_parameter_and_correct_norm_lr_decay(self):
        torch = self.torch
        model = self.model()
        optimizer, scheduler, audit = self.optim.build_optimizer(model, 34626)
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        actual = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(expected))
        for group in optimizer.param_groups:
            for name, p in zip(group["param_names"], group["params"]):
                self.assertEqual(group["initial_lr"], self.c.model_parameter_role(name)[1])
                self.assertEqual(group["weight_decay"], 0. if p.ndim <= 1 or name.endswith(".bias") else .01)
            self.assertAlmostEqual(group["lr"], group["initial_lr"] * .1)
        self.assertEqual(sum(g["numel"] for g in audit), sum(p.numel() for p in model.parameters() if p.requires_grad))
        self.assertIs(model.caption_decoder.lm.lm_head.weight, model.caption_decoder.lm.model.embed_tokens.weight)
        self.assertEqual(scheduler.last_epoch, 0)

    def test_zero_branch_gradients_unlock_after_real_first_adamw_update(self):
        model = self.model()
        optimizer, scheduler, _ = self.optim.build_optimizer(model, 34626)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            self.loss(model).backward()
            nonzero = {role: False for role in self.train.TRAINABLE_ROLES}
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    self.assertIsNone(p.grad)
                    continue
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(self.torch.isfinite(p.grad).all(), name)
                nonzero[self.optim.parameter_role(name)] |= bool(p.grad.ne(0).any())
            for role in ("map", "recurrent_base", "outer_lm", "other_inherited_lm", "reinject_gate",
                         ):
                self.assertTrue(nonzero[role], role)
            for role in ("reinject_linear",):
                self.assertEqual(nonzero[role], step == 1, role)
            optimizer.step()
            scheduler.step()
        self.assertEqual(model.caption_decoder.lm.model.last_layer_trace, self.c.TEXT_MODEL_SCHEDULE)

    def test_exact_adamw_scheduler_gradnorm_and_cpu_rng_resume(self):
        torch = self.torch
        torch.manual_seed(1234)
        model = self.model()
        optimizer, scheduler, _ = self.optim.build_optimizer(model, 34626)
        tracker = self.optim.GradNormTracker()
        self.update(model, optimizer, scheduler, tracker)
        stream = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "grad_norm_tracker": tracker.state_dict(),
                    "rng": torch.get_rng_state()}, stream)
        self.update(model, optimizer, scheduler, tracker)
        expected = {n: p.clone() for n, p in model.state_dict().items()}
        expected_opt = copy.deepcopy(optimizer.state_dict())
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=False)
        restored = self.model()
        opt, sched, _ = self.optim.build_optimizer(restored, 34626)
        self.optim.validate_optimizer_state(saved, opt, 1, 34626)
        restored.load_state_dict(saved["model"], strict=True)
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        clip = self.optim.GradNormTracker()
        clip.load_state_dict(saved["grad_norm_tracker"])
        torch.set_rng_state(saved["rng"])
        self.update(restored, opt, sched, clip)
        for name, p in restored.state_dict().items():
            self.assertTrue(torch.equal(p, expected[name]), name)
        for index, entry in opt.state_dict()["state"].items():
            for key, value in entry.items():
                self.assertTrue(torch.equal(value, expected_opt["state"][index][key]), key)
        self.assertEqual(sched.state_dict(), scheduler.state_dict())
        self.assertEqual(clip.state_dict(), tracker.state_dict())

    def test_resume_rejects_wrong_named_groups_lr_and_moment_shape(self):
        model = self.model()
        optimizer, scheduler, _ = self.optim.build_optimizer(model, 34626)
        tracker = self.optim.GradNormTracker()
        self.update(model, optimizer, scheduler, tracker)
        saved = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "grad_norm_tracker": tracker.state_dict()}
        fresh, _, _ = self.optim.build_optimizer(model, 34626)
        for kind in ("name", "lr", "shape", "norm"):
            bad = copy.deepcopy(saved)
            if kind == "name":
                bad["optimizer"]["param_groups"][0]["param_names"][0] = "wrong.weight"
            elif kind == "lr":
                bad["optimizer"]["param_groups"][0]["lr"] = 6e-4
            elif kind == "norm":
                bad["grad_norm_tracker"].pop(next(iter(bad["grad_norm_tracker"])))
            else:
                next(iter(bad["optimizer"]["state"].values()))["exp_avg"] = self.torch.zeros(1)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.optim.validate_optimizer_state(bad, fresh, 1, 34626)

    def test_answer_only_shift_mask_and_microbatch_mean_are_preserved(self):
        torch = self.torch
        losses, counts = [], []
        for count, bias in ((1, -2.), (9, 2.)):
            logits = torch.zeros(1, 683, 24)
            logits[:, :, 5] = bias
            logits.requires_grad_()
            targets = torch.full((1, 300), 17, dtype=torch.long)
            targets[:, :count] = 5
            mask = torch.zeros_like(targets)
            mask[:, :count] = 1
            # A non-PAD token outside the tokenizer mask must not contribute.
            targets[0, 200] = 6
            loss = self.train.answer_loss(SimpleNamespace(logits=logits),
                                         {"answer": {"input_ids": targets, "attention_mask": mask}}, 17)
            expected = torch.nn.functional.cross_entropy(logits[:, 382:382+count].reshape(-1, 24),
                                                         targets[:, :count].reshape(-1))
            torch.testing.assert_close(loss, expected)
            loss.backward()
            self.assertEqual(logits.grad[:, :382].abs().sum().item(), 0)
            self.assertEqual(logits.grad[:, 382+count:].abs().sum().item(), 0)
            losses.append(loss.detach())
            counts.append(count)
        micro_mean = torch.stack(losses).mean()
        global_token_mean = sum(loss*n for loss, n in zip(losses, counts))/sum(counts)
        self.assertGreater(abs(float(micro_mean-global_token_mean)), .1)
        with self.assertRaises(ValueError):
            self.train.answer_loss(SimpleNamespace(logits=torch.zeros(1, 683, 24)),
                {"answer": {"input_ids": torch.full((1, 300), 17), "attention_mask": torch.zeros(1, 300)}}, 17)


@unittest.skipUnless(all(importlib.util.find_spec(x) for x in ("torch", "transformers")),
                     "requires real Torch/Transformers; no production assets or downloads")
class ModelTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LlamaConfig
        from mizar_official_recursive_5_20x2_5_input_only import model, tensor_runtime, contract
        cls.torch, cls.cfg, cls.mod, cls.runtime, cls.contract = torch, LlamaConfig, model, tensor_runtime, contract
        torch.set_num_threads(1)

    def config(self):
        cfg = self.cfg(vocab_size=32, hidden_size=24, intermediate_size=48,
                       num_hidden_layers=50, num_attention_heads=3, num_key_value_heads=1,
                       pad_token_id=17, eos_token_id=0, tie_word_embeddings=True,
                       use_cache=False, attention_dropout=0., max_position_embeddings=1024)
        cfg.recursive_layer_count = 30
        cfg.recursive_loops = 2
        cfg.recursive_prefix_layer_count = cfg.recursive_suffix_layer_count = 5
        cfg.recursive_middle_layer_count = 20
        cfg.logical_to_physical = list(self.contract.SCHEDULE)
        cfg.recursive_loops_scope = "middle_only"
        cfg.recursive_source_layer_indices_0based = list(range(30))
        return self.runtime.config_from_dict(cfg.to_dict())

    def lm(self):
        lm = self.mod.MizarInputOnlyRecursiveForCausalLM(self.config()).float()
        lm.model.enable_new_modules()
        return lm

    def test_same_seed_all_inherited_and_retained_input_tensors_equal_parent(self):
        torch = self.torch
        from mizar_official_recursive_5_20x2_5.model import MizarRecursiveForCausalLM
        torch.manual_seed(0)
        parent = MizarRecursiveForCausalLM(self.config()).float()
        parent.model.enable_new_modules()
        torch.manual_seed(0)
        simple = self.lm()
        for name, value in simple.state_dict().items():
            self.assertTrue(torch.equal(value, parent.state_dict()[name]), name)
        self.runtime.assert_input_only(simple)
        self.assertFalse(any("lora" in n or "audio_gates" in n for n in simple.state_dict()))
        self.assertIs(simple.lm_head.weight, simple.model.embed_tokens.weight)

    def test_only_four_new_tensors_and_registration_does_not_advance_rng(self):
        torch = self.torch
        lm = self.mod.MizarInputOnlyRecursiveForCausalLM(self.config()).float()
        inherited = {n: p.clone() for n, p in lm.state_dict().items()}
        before = torch.get_rng_state().clone()
        lm.model.enable_new_modules()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        new = set(lm.state_dict()) - set(inherited)
        self.assertEqual(new, {f"model.{m}.{p}" for m in ("audio_linear", "audio_gate") for p in ("weight", "bias")})
        for name, value in inherited.items():
            self.assertTrue(torch.equal(lm.state_dict()[name], value), name)
        self.assertEqual(sum(lm.state_dict()[n].numel() for n in new), 24*24+24+24+1)
        with self.assertRaises(ValueError):
            lm.model.enable_new_modules()
        with self.assertRaises(RuntimeError):
            lm.load_state_dict(inherited, strict=True)

    def test_real_fifty_calls_two_input_injections_one_projection(self):
        lm = self.lm()
        trace, gates, projections = [], [], []
        handles = [layer.register_forward_pre_hook(lambda m, a, i=i: trace.append(i))
                   for i, layer in enumerate(lm.model.layers)]
        handles.append(lm.model.audio_gate.register_forward_pre_hook(lambda m, a: gates.append(a[0].shape)))
        handles.append(lm.model.audio_linear.register_forward_pre_hook(lambda m, a: projections.append(a[0].shape)))
        try:
            out = lm(inputs_embeds=self.torch.randn(1, 130, 24))
            out.logits.square().mean().backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(trace, list(self.contract.SCHEDULE))
        self.assertEqual(gates, [(1, 126, 24)]*2)
        self.assertEqual(projections, [(1, 126, 24)])
        self.assertEqual(lm.model.last_input_injection_trace, (5, 25))
        self.assertTrue(self.torch.isfinite(out.logits).all())
        self.assertTrue(all(p.grad is not None for p in lm.parameters()))

    def test_residual_formula_reads_current_hidden_and_only_directly_changes_audio(self):
        torch = self.torch
        lm = self.lm().eval()
        core = lm.model
        with torch.no_grad():
            core.audio_linear.weight.copy_(torch.eye(24))
            core.audio_gate.weight.fill_(.001)
            core.audio_gate.bias.fill_(.2)
        x = torch.randn(1, 130, 24, requires_grad=True)
        original = x[:, :126]
        gate_inputs = []
        handle = core.audio_gate.register_forward_pre_hook(lambda m, a: gate_inputs.append(a[0].detach().clone()))
        try:
            with patch.object(self.mod, "_call_decoder_layer", side_effect=lambda layer, hidden, **kw: (hidden,)):
                actual = core._execute(x).last_hidden_state
        finally:
            handle.remove()
        audio = original
        projected = core.audio_linear(original)
        for gate_input in gate_inputs:
            torch.testing.assert_close(gate_input, audio, rtol=0, atol=0)
            audio = audio + torch.tanh(core.audio_gate(audio))*projected
        expected = core.norm(torch.cat((audio, x[:, 126:]), dim=1))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[:, 126:], core.norm(x[:, 126:]), rtol=0, atol=0)
        # A non-constant objective demonstrates task gradient through both
        # residual uses into the original input (and hence the trained mapper).
        weights = torch.linspace(.1, 2., 24).reshape(1, 1, 24)
        actual[:, :126].mul(weights).sum().backward()
        self.assertGreater(x.grad[:, :126].abs().sum().item(), 0)
        self.assertGreater(core.audio_linear.weight.grad.abs().sum().item(), 0)

    def test_single_pass_alignment_and_recursive_is_distinct(self):
        torch = self.torch
        from transformers import LlamaForCausalLM
        lm = self.lm().eval()
        cfg = copy.deepcopy(lm.config)
        cfg.num_hidden_layers = 30
        standard = LlamaForCausalLM(cfg).float().eval()
        inherited = {n:v for n,v in lm.state_dict().items() if not n.startswith(("model.audio_linear.", "model.audio_gate."))}
        standard.load_state_dict(inherited, strict=True)
        x = torch.randn(1, 130, 24)
        with torch.no_grad():
            expected = standard(inputs_embeds=x, use_cache=False).logits
            actual = lm.lm_head(lm.model._execute(x, single_pass_audit=True).last_hidden_state)
            recursive = lm(inputs_embeds=x).logits
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        self.assertFalse(torch.allclose(recursive, expected, rtol=1e-5, atol=1e-5))

    def test_strict_tensor_and_logits_reload(self):
        torch = self.torch
        lm = self.lm().eval()
        with torch.no_grad():
            lm.model.audio_gate.bias.fill_(.1)
        x = torch.randn(1, 130, 24)
        stream = io.BytesIO()
        torch.save(lm.state_dict(), stream)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
        restored = self.lm().eval()
        restored.load_state_dict(saved, strict=True)
        with torch.no_grad():
            torch.testing.assert_close(lm(inputs_embeds=x).logits, restored(inputs_embeds=x).logits, rtol=0, atol=0)
        for name, value in restored.state_dict().items():
            self.assertTrue(torch.equal(value, saved[name]), name)

    def test_removed_module_and_invalid_forward_arguments_are_rejected(self):
        lm = self.lm()
        x = self.torch.randn(1, 130, 24)
        for kwargs in ({"audio_information": ()}, {"attention_mask": self.torch.ones(1, 130)}, {"use_cache": True}):
            with self.assertRaises(ValueError):
                lm(inputs_embeds=x, **kwargs)
        lm.model.layer_mlps = self.torch.nn.ModuleDict()
        with self.assertRaises(ValueError):
            self.runtime.assert_input_only(lm)


if __name__ == "__main__":
    unittest.main()


