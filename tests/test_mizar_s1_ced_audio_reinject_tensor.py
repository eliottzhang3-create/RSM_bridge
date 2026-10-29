"""Optional CPU tensor checks in mellow_c8204d8; no remote weights or CUDA required."""
import ast
import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine"
ROUTE = FAMILY / "mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_audio_reinject"
HAS_TENSORS = all(importlib.util.find_spec(name) is not None
                  for name in ("torch", "torchaudio", "transformers", "soundfile"))


@unittest.skipUnless(HAS_TENSORS, "requires torch/torchaudio/transformers/soundfile in mellow_c8204d8")
class NonMeshCEDTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(ROUTE), str(ROOT / "code/RSmol")]
        import torch
        from transformers import LlamaConfig
        from models.ced import CEDTemporalMapper
        from models.decoder import DecoderModel
        from optim import build_optimizer, GradNormTracker
        from models.recursive_reinject import RecursiveLlamaForCausalLM
        import train
        cls.torch, cls.config_class, cls.model_class = torch, LlamaConfig, RecursiveLlamaForCausalLM
        cls.mapper_class, cls.decoder_class = CEDTemporalMapper, DecoderModel
        cls.build_optimizer = staticmethod(build_optimizer)
        cls.tracker_class, cls.train = GradNormTracker, train
        torch.set_num_threads(1)

    def make_model_and_inputs(self):
        torch, nn = self.torch, self.torch.nn
        torch.manual_seed(1234)
        config = self.config_class(vocab_size=32, hidden_size=32, intermediate_size=48,
            num_hidden_layers=30, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=1024, pad_token_id=17, eos_token_id=0,
            use_cache=False, attention_dropout=0.)
        config.recursive_layer_count, config.recursive_loops = 20, 2
        decoder = self.decoder_class.__new__(self.decoder_class)
        nn.Module.__init__(decoder)
        decoder.lm, decoder.lm_embedding_size = self.model_class(config).float(), 32
        decoder.lm.model.enable_audio_reinjection()
        model = nn.Module()
        model.audio_encoder, model.caption_decoder = nn.Module(), decoder
        model.audio_encoder.base = nn.Module()
        model.audio_encoder.base.mapper = self.mapper_class(384, 4, 32, output_tokens=126, resample=False)
        targets = torch.randint(1, 16, (1, 300))
        targets[:, 199], targets[:, 200:] = 0, 17
        inputs = dict(grid=torch.randn(1, 126, 4, 384),
                      prompt={"input_ids": torch.ones(1, 256, dtype=torch.long)},
                      answer={"input_ids": targets, "attention_mask": targets.ne(17).long()})
        return model.train(), inputs

    def loss(self, model, inputs):
        audio = model.audio_encoder.base.mapper(inputs["grid"])
        output = model.caption_decoder(audio, None, inputs["prompt"], inputs["answer"])
        self.assertIsNone(output.past_key_values)
        self.assertEqual(tuple(output.logits.shape), (1, 683, 32))
        return self.train.answer_loss(output, {"answer": inputs["answer"]}, 17)

    def test_real_nonmesh_schedule_full_prefix_and_mapper_lm_backward(self):
        torch = self.torch
        model, inputs = self.make_model_and_inputs()
        calls = []
        handles = [layer.register_forward_pre_hook(lambda _, args, index=i: calls.append(index))
                   for i, layer in enumerate(model.caption_decoder.lm.model.layers)]
        try:
            self.loss(model, inputs).backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, list(range(15)) + list(range(5, 15)) + list(range(15, 20)))
        self.assertEqual(len({id(layer) for layer in model.caption_decoder.lm.model.layers}), 20)
        for name, parameter in model.named_parameters():
            self.assertFalse(any(part.startswith(("router", "mesh_", "memory_slot")) for part in name.split(".")))
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertTrue(any(p.grad.ne(0).any() for p in model.audio_encoder.base.mapper.parameters()))
        core = model.caption_decoder.lm.model
        self.assertEqual(core.last_reinject_trace, (5, 15))
        self.assertTrue(all(p.grad.eq(0).all() for p in core.audio_linear.parameters()))
        self.assertTrue(any(p.grad.ne(0).any() for p in core.audio_gate.parameters()))
        self.assertTrue(self.train.first_gradient_audit(model)["passed"])

    def test_zero_gate_preserves_actual_backbone_outputs_shared_gradients_and_rng(self):
        from recursive_model_5_10_5 import RecursiveLlamaForCausalLM as Original
        torch = self.torch
        model, _ = self.make_model_and_inputs()
        candidate = model.caption_decoder.lm
        reference = Original(copy.deepcopy(candidate.config)).float()
        original_state = {k: v for k,v in candidate.state_dict().items()
                          if not k.startswith(("model.audio_linear.", "model.audio_gate."))}
        reference.load_state_dict(original_state, strict=True)
        fresh = self.model_class(copy.deepcopy(candidate.config)).float()
        state = torch.get_rng_state().clone()
        fresh.model.enable_audio_reinjection()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        with self.assertRaisesRegex(RuntimeError, "already registered"):
            fresh.model.enable_audio_reinjection()
        reference.eval()
        candidate.eval()
        x = torch.randn(1,130,32,requires_grad=True)
        other = x.detach().clone().requires_grad_()
        expected = reference(inputs_embeds=x,use_cache=False).logits
        actual = candidate(inputs_embeds=other,use_cache=False).logits
        self.assertTrue(torch.equal(expected,actual))
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(x.grad,other.grad,rtol=1e-6,atol=1e-7)
        ref_params = dict(reference.named_parameters())
        for name,param in candidate.named_parameters():
            if name in ref_params:
                torch.testing.assert_close(param.grad,ref_params[name].grad,rtol=1e-6,atol=1e-7)

    def test_actual_pretrained_loader_preserves_every_backbone_tensor_and_adds_only_four(self):
        from recursive_model_5_10_5 import RecursiveLlamaForCausalLM as Original
        from models.recursive_text import load_recursive_text_model, validate_audio_reinjection
        from contract import TEXT_MODEL_SCHEDULE, TEXT_MODEL_SOURCE_MAPPING
        torch = self.torch
        # Keep the real route's hidden size and layer geometry, with a tiny
        # synthetic vocabulary/MLP; this needs no downloaded checkpoint.
        config = self.config_class(vocab_size=32, hidden_size=576, intermediate_size=8,
            num_hidden_layers=30, num_attention_heads=4, num_key_value_heads=1,
            max_position_embeddings=1024, pad_token_id=17, eos_token_id=0,
            use_cache=False, attention_dropout=0.)
        config.recursive_layer_count, config.recursive_loops = 20, 2
        config.recursive_prefix_layer_count = config.recursive_suffix_layer_count = 5
        config.recursive_middle_layer_count, config.recursive_loops_scope = 10, "middle_only"
        config.logical_to_physical = list(TEXT_MODEL_SCHEDULE)
        config.recursive_source_layer_indices_0based = list(TEXT_MODEL_SOURCE_MAPPING)
        reference = Original(config).float()
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            candidate = load_recursive_text_model(directory)
        original_state, actual = reference.state_dict(), candidate.state_dict()
        self.assertEqual(set(actual) - set(original_state), {
            "model.audio_linear.weight", "model.audio_linear.bias",
            "model.audio_gate.weight", "model.audio_gate.bias"})
        for name, value in original_state.items():
            self.assertTrue(torch.equal(value, actual[name]), name)
        self.assertEqual(validate_audio_reinjection(candidate)["parameter_count"], 332929)
        self.assertTrue(all(p.eq(0).all() for p in candidate.model.audio_gate.parameters()))
        self.assertTrue(candidate.model.audio_linear.bias.eq(0).all())
        self.assertTrue(candidate.model.audio_linear.weight.ne(0).any())
        self.assertFalse(candidate.config.use_cache)

    def test_open_gate_receives_real_mapper_and_linear_gradients(self):
        torch = self.torch
        model,inputs = self.make_model_and_inputs()
        optimizer,scheduler,audit = self.build_optimizer(model,200)
        self.loss(model,inputs).backward()
        first = self.train.first_gradient_audit(model)
        self.assertTrue(first["passed"])
        self.assertTrue(first["gate_is_zero"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        self.loss(model,inputs).backward()
        opened = self.train.first_gradient_audit(model)
        self.assertTrue(opened["passed"])
        self.assertFalse(opened["gate_is_zero"])
        self.assertTrue(opened["nonzero_by_role"]["reinject_linear"])
        names = {name for row in audit if row["role"].startswith("reinject") for name in row["names"]}
        self.assertEqual(len(names),4)
        self.assertTrue(all(row["base_lr"] == 6e-4 for row in audit if row["role"].startswith("reinject")))

    def test_actual_injections_only_touch_audio_and_share_A_and_gate(self):
        torch = self.torch
        model,_ = self.make_model_and_inputs()
        core = model.caption_decoder.lm.model
        core.audio_gate.weight.data.fill_(.01)
        core.audio_gate.bias.data.fill_(.1)
        h = torch.randn(1,130,32,requires_grad=True)
        original = h.detach().clone()
        a = core.audio_linear(torch.randn(1,126,32,requires_grad=True))
        result = core.inject_audio(h,a,5)
        self.assertTrue(torch.equal(h,original))
        self.assertTrue(torch.equal(result[:,126:],h[:,126:]))
        torch.testing.assert_close(result[:,:126]-h[:,:126],torch.tanh(core.audio_gate(h[:,:126]))*a)
        counts = dict(linear=0,gate=0)
        gate_inputs = []
        def linear_hook(*args):
            counts["linear"] += 1
        def gate_hook(_,args):
            counts["gate"] += 1
            gate_inputs.append(args[0].detach().clone())
        handles = [core.audio_linear.register_forward_pre_hook(linear_hook),core.audio_gate.register_forward_pre_hook(gate_hook)]
        core.collect_reinject_stats = True
        try:
            model.caption_decoder.lm(inputs_embeds=h,use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(counts,dict(linear=1,gate=2))
        self.assertEqual(core.last_reinject_trace,(5,15))
        self.assertFalse(torch.equal(*gate_inputs))
        self.assertEqual([x["logical_index"] for x in core.reinjection_diagnostics()],[5,15])
        with self.assertRaisesRegex(ValueError,"use_cache=False"):
            model.caption_decoder.lm(inputs_embeds=h,use_cache=True)
        with self.assertRaisesRegex(ValueError,"126-token"):
            model.caption_decoder.lm(inputs_embeds=h[:,:125],use_cache=False)

    def test_full_checkpoint_cpu_roundtrip_preserves_next_stochastic_update(self):
        torch = self.torch
        model, inputs = self.make_model_and_inputs()
        optimizer, scheduler, _ = self.build_optimizer(model, 200)
        tracker = self.tracker_class()

        def update(candidate, opt, sched, grad_tracker):
            self.loss(candidate, inputs).backward()
            grad_tracker.track_and_clip_(candidate.named_parameters())
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()

        update(model, optimizer, scheduler, tracker)
        state = {"cpu": torch.get_rng_state()}
        scaler = SimpleNamespace(state_dict=lambda: {"synthetic_cpu_scaler": True})
        config = dict(mode="formal", contract={"synthetic_cpu_test": True}, contract_sha256="fixture")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model--step-1.ckpt"
            def gather(output, local):
                output[:] = [copy.deepcopy(local) for _ in range(8)]
            with patch.object(self.train, "rng_state", return_value=state), \
                    patch.object(self.train.dist, "all_gather_object", side_effect=gather), \
                    patch.object(self.train.dist, "get_rank", return_value=0), \
                    patch.object(self.train.dist, "barrier"):
                self.train.save_checkpoint(path, model, optimizer, scheduler, scaler, tracker,
                                           config, 1, 0, 1, torch.device("cpu"))
            saved = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(saved["route_variant"], "ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_audio_reinject_v1")
        self.assertEqual(len(saved["random_state_by_rank"]), 8)
        self.assertEqual((saved["total_step"], saved["epoch"], saved["next_optimizer_step"]), (1, 0, 1))
        self.assertIn("grad_scaler", saved)
        self.assertEqual(len([k for k in saved["state_dict"] if ".audio_linear." in k or ".audio_gate." in k]),4)
        update(model, optimizer, scheduler, tracker)
        restored, _ = self.make_model_and_inputs()
        opt, sched, _ = self.build_optimizer(restored, 200)
        restored.load_state_dict(saved["state_dict"], strict=True)
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        other_tracker = self.tracker_class()
        other_tracker.load_state_dict(saved["grad_norm_tracker"])
        torch.set_rng_state(saved["random_state_by_rank"][0]["cpu"])
        update(restored, opt, sched, other_tracker)
        for (name, a), (_, b) in zip(model.named_parameters(), restored.named_parameters()):
            self.assertTrue(torch.equal(a, b), name)
        self.assertEqual(scheduler.state_dict(), sched.state_dict())
        self.assertEqual(tracker.state_dict(), other_tracker.state_dict())
        for key,value in optimizer.state_dict()["state"].items():
            for item,tensor in value.items():
                self.assertTrue(torch.equal(tensor,opt.state_dict()["state"][key][item]))
        missing = dict(saved["state_dict"])
        missing.pop("caption_decoder.lm.model.audio_gate.weight")
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(missing,strict=True)

    def test_three_real_mappers_share_initial_weights_before_different_text_initializers(self):
        torch, nn = self.torch, self.torch.nn
        class ToyCED(nn.Module):
            def __init__(self, encoder_config, d_out):
                super().__init__()
                self.encoder = nn.Linear(3, 3)
                self.mapper = NonMeshCEDTensorTests.mapper_class(384, 4, d_out, output_tokens=126, resample=False)
        states = []
        for index, route in enumerate(("mizar_stage1_fixed20_gbs256_3epochs_ced_small",
                "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small", ROUTE.name)):
            tree = ast.parse((FAMILY / route / "models/mellow.py").read_text(encoding="utf-8"))
            scope = dict(torch=torch, nn=nn, get_audio_encoder=lambda _: (ToyCED, 384),
                         get_decoder=lambda _: lambda *args: nn.Linear(5 + index, 5 + index))
            nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "actual-mellow-construction", "exec"), scope)
            torch.manual_seed(1234)
            model = scope["Mellow"]("CEDSmall", 384, "synthetic-text", 383, False, 576, True, True, "synthetic-ced")
            states.append({name: value.detach().clone() for name, value in model.audio_encoder.base.mapper.state_dict().items()})
        for name in states[0]:
            self.assertTrue(torch.equal(states[0][name], states[1][name]), name)
            self.assertTrue(torch.equal(states[0][name], states[2][name]), name)


if __name__ == "__main__":
    unittest.main()
