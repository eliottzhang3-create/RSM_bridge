"""Real CPU tensor coverage; no downloads, pretrained assets, CUDA or submission."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs128_3epochs_recursive_mesh_5_10x4_5_ced_small"
DEPENDENCIES = ("torch", "torchaudio", "transformers", "soundfile")
HAS_TENSORS = all(importlib.util.find_spec(name) is not None for name in DEPENDENCIES)

if HAS_TENSORS:
    sys.path[:0] = [str(ROUTE), str(ROOT / "code/RSmol")]
    import torch
    from runtime_checks import RuntimeChecks
    torch.set_num_threads(1)
else:
    class RuntimeChecks(unittest.TestCase):
        pass

    def missing(self):
        self.skipTest("requires torch/torchaudio/transformers/soundfile")

    source = ast.parse((ROUTE / "runtime_checks.py").read_text(encoding="utf-8"))
    for node in ast.walk(source):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            setattr(RuntimeChecks, node.name, missing)


@unittest.skipUnless(HAS_TENSORS, "requires torch/torchaudio/transformers/soundfile")
class X4TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import contract
        import train
        from transformers import LlamaConfig
        from recursive_model_5_10x4_5_mesh import RecursiveLlamaForCausalLM
        from models.decoder import DecoderModel
        from models.ced import CEDTemporalMapper
        from optim import build_optimizer, GradNormTracker
        cls.contract, cls.train = contract, train
        cls.Config, cls.LM = LlamaConfig, RecursiveLlamaForCausalLM
        cls.Decoder, cls.Mapper = DecoderModel, CEDTemporalMapper
        cls.optimizer = staticmethod(build_optimizer)
        cls.Tracker = GradNormTracker

    def lm(self, hidden=32):
        config = self.Config(vocab_size=32, hidden_size=hidden, intermediate_size=48,
                             num_hidden_layers=50, num_attention_heads=4 if hidden == 32 else 9,
                             num_key_value_heads=2 if hidden == 32 else 3,
                             max_position_embeddings=1024, eos_token_id=0, pad_token_id=17,
                             use_cache=False, attention_dropout=0., tie_word_embeddings=False)
        config.recursive_layer_count, config.recursive_loops = 20, 4
        config.mesh_memory_slots, config.mesh_router_count = 7, 10
        config.mesh_architecture_contract = self.contract.TEXT_MODEL_CONTRACT
        config.model_architecture_contract = self.contract.TEXT_MODEL_ARCHITECTURE
        config.logical_to_physical = list(self.contract.TEXT_MODEL_SCHEDULE)
        return self.LM(config).float()

    def fixture(self):
        torch.manual_seed(9)
        decoder = self.Decoder.__new__(self.Decoder)
        torch.nn.Module.__init__(decoder)
        decoder.lm, decoder.lm_embedding_size = self.lm(), 32
        model = torch.nn.Module()
        model.audio_encoder = torch.nn.Module()
        model.audio_encoder.base = torch.nn.Module()
        model.audio_encoder.base.mapper = self.Mapper(384, 4, 32, output_tokens=126, resample=False)
        model.caption_decoder = decoder
        targets = torch.randint(1, 16, (1, 300))
        targets[:, 199], targets[:, 200:] = 0, 17
        inputs = dict(grid=torch.randn(1, 126, 4, 384),
                      prompt={"input_ids": torch.ones(1, 256, dtype=torch.long)},
                      answer={"input_ids": targets, "attention_mask": targets.ne(17).long()})
        return model.train(), inputs

    def loss(self, model, inputs):
        audio = model.audio_encoder.base.mapper(inputs["grid"])
        out = model.caption_decoder(audio, None, inputs["prompt"], inputs["answer"])
        self.assertEqual(tuple(out.logits.shape), (1, 683, 32))
        self.assertIsNone(out.past_key_values)
        return self.train.answer_loss(out, {"answer": inputs["answer"]}, 17)

    def test_actual_ced_prefix_x4_forward_gradients_and_lr_groups(self):
        model, inputs = self.fixture()
        calls = []
        handles = [layer.register_forward_pre_hook(lambda _, args, i=i: calls.append(i))
                   for i, layer in enumerate(model.caption_decoder.lm.model.layers)]
        try:
            self.loss(model, inputs).backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, list(self.contract.TEXT_MODEL_SCHEDULE))
        result = self.train.first_gradient_audit(model)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["recursive_execution"]["routers"], 10)
        _, _, groups = self.optimizer(model, 34626)
        for group in groups:
            self.assertEqual(group["base_lr"], 6e-4 if group["role"] == "map" else 3e-4)

    def test_memory_has_seven_slots_and_is_rebuilt_each_forward(self):
        lm = self.lm().eval()
        core = lm.model
        core.audit_mode = True
        x = torch.randn(1, 7, 32)
        with torch.no_grad():
            first = lm(inputs_embeds=x, use_cache=False).logits
            self.assertEqual(core.last_memory_shape, (1, 7, 7, 32))
            torch.testing.assert_close(core.last_initial_memory[:, 0], x, rtol=0, atol=0)
            self.assertFalse(core.last_initial_memory[:, 1:].ne(0).any())
            self.assertEqual(len(core.last_memory_write_history), 5)
            lm(inputs_embeds=x * 3, use_cache=False)
            repeated = lm(inputs_embeds=x, use_cache=False).logits
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        self.assertFalse(any("memory" in name for name in lm.state_dict()))

    def test_all_four_recurrent_boundaries_receive_gradients(self):
        lm = self.lm()
        lm.model.gradient_audit_mode = True
        lm(inputs_embeds=torch.randn(1, 8, 32), use_cache=False).logits.square().mean().backward()
        self.assertEqual(len(lm.model.last_core_input_refs), 4)
        self.assertEqual(len(lm.model.last_core_output_refs), 4)
        for tensor in lm.model.last_core_input_refs + lm.model.last_core_output_refs:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertTrue(tensor.grad.ne(0).any())

    def test_real_strict_x4_checkpoint_preserves_all_parameters_and_routers(self):
        from models.recursive_text import load_recursive_text_model
        original = self.lm(hidden=576)
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            restored = load_recursive_text_model(directory)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(len(restored.model.write_routers), 5)
        self.assertEqual(len(restored.model.read_routers), 5)
        self.assertTrue(all(p.dtype == torch.float32 for p in restored.parameters()))

    def test_full_checkpoint_resume_matches_next_optimizer_update(self):
        model, inputs = self.fixture()
        optimizer, scheduler, _ = self.optimizer(model, 34626)
        tracker = self.Tracker()

        def update(m, opt, sched, track):
            opt.zero_grad(set_to_none=True)
            self.loss(m, inputs).backward()
            track.track_and_clip_(m.named_parameters())
            opt.step()
            sched.step()

        update(model, optimizer, scheduler, tracker)
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model--step-1.ckpt"
            cfg = dict(mode="formal", contract={}, contract_sha256="fixture")

            def gather(output, value):
                output[:] = [copy.deepcopy(value) for _ in range(8)]

            with patch.object(self.train, "rng_state", return_value={"torch": rng}), \
                 patch.object(self.train.dist, "all_gather_object", side_effect=gather), \
                 patch.object(self.train.dist, "get_rank", return_value=0), patch.object(self.train.dist, "barrier"):
                self.train.save_checkpoint(path, model, optimizer, scheduler,
                    SimpleNamespace(state_dict=lambda: {"synthetic_cpu_scaler": True}), tracker, cfg, 1, 0, 1, "cpu")
            self.assertEqual(self.train.reload_audit(path, model, cfg, 1)["status"], "PASS")
            saved = torch.load(path, weights_only=False)
        self.assertEqual(saved["batch_geometry"], dict(world_size=8, microbatch_size=4,
                         gradient_accumulation_steps=4, global_batch_size=128))
        self.assertEqual(saved["route_variant"], self.contract.ROUTE)
        self.assertEqual(saved["grad_scaler"], {"synthetic_cpu_scaler": True})
        restored = copy.deepcopy(model)
        restored.load_state_dict(saved["state_dict"], strict=True)
        opt2, sched2, _ = self.optimizer(restored, 34626)
        opt2.load_state_dict(saved["optimizer"])
        sched2.load_state_dict(saved["scheduler"])
        tracker2 = self.Tracker()
        tracker2.load_state_dict(saved["grad_norm_tracker"])
        torch.set_rng_state(rng)
        update(model, optimizer, scheduler, tracker)
        torch.set_rng_state(saved["random_state_by_rank"][0]["torch"])
        update(restored, opt2, sched2, tracker2)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(tracker.state_dict(), tracker2.state_dict())
        self.assertEqual(scheduler.state_dict(), sched2.state_dict())


if __name__ == "__main__":
    unittest.main()
