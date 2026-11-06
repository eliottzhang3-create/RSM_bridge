"""Optional real CPU autograd/AdamW/resume tests with small dimensions, no assets."""
import copy
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_official_final_s1_gbs128_5_20x2_5"
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
        from mizar_official_recursive_5_20x2_5.model import MizarRecursiveForCausalLM
        from mizar_official_recursive_5_20x2_5.tensor_runtime import config_from_dict
        cls.torch, cls.c, cls.optim, cls.train = torch, contract, optim, train
        cls.config_from_dict = staticmethod(config_from_dict)
        cls.LlamaConfig, cls.LM = LlamaConfig, MizarRecursiveForCausalLM
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
        base.layer_mlps = torch.nn.ModuleDict({str(i): torch.nn.Linear(24, 24) for i in (4, 8)})
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
        pair = tuple(base.layer_mlps[str(i)](features) for i in (4, 8))
        return model.caption_decoder.lm(inputs_embeds=x, audio_information=pair).logits.square().mean()

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
                         "layer4_gate", "layer8_gate", "loop1_lora_B", "loop2_lora_B"):
                self.assertTrue(nonzero[role], role)
            for role in ("reinject_linear", "layer4_mlp", "layer8_mlp", "loop1_lora_A", "loop2_lora_A"):
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


if __name__ == "__main__":
    unittest.main()
