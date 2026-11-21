"""Real tiny-model CPU checks. Production geometry is checked by the stdlib suite.

Only fixture dimensions/JSON provenance validation are patched for small models;
the explicit class, safetensors loading/equality, recurrence, loss and optimizer
execute normally. No remote weights, CUDA, downloads or training submission.
"""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced_small"
sys.path.insert(0, str(ROUTE))
try:
    import torch
    import torch.nn as nn
    from transformers import LlamaConfig
    import contract as C
    from models import recursive_text as L
    import recursive_model_mizar_text_7_14x2_7_mesh as M
    from mizar_text_7_14x2_7_mesh.contract import build_target_config, validate_model_config
    from safetensors.torch import save_file
    from optim import build_optimizer, GradNormTracker
    import train as T
    AVAILABLE, MISSING = True, ""
except ImportError as error:
    AVAILABLE, MISSING = False, str(error)


@unittest.skipUnless(AVAILABLE, "Real tensor dependencies unavailable: " + MISSING)
class TensorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(12)
        config = LlamaConfig(num_hidden_layers=30, hidden_size=16, intermediate_size=32,
                             num_attention_heads=4, num_key_value_heads=2, vocab_size=43,
                             tie_word_embeddings=True, max_position_embeddings=1024,
                             pad_token_id=0, bos_token_id=0, eos_token_id=0, _attn_implementation="eager")
        self.model = M.MizarText7_14x2_7MeshForCausalLM(build_target_config(config)).float()
        self.ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])

    def test_explicit_loader_inherits_trained_routers_and_never_loads_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with torch.no_grad():
                for index, router in enumerate(list(self.model.model.write_routers) + list(self.model.model.read_routers)):
                    router.weight.fill_(.01 * (index + 1)); router.bias.fill_(.001 * (index + 1))
            self.model.save_pretrained(path, safe_serialization=True)
            (path / "training_state.pt").write_bytes(b"unreadable pickle; weights only")
            with patch.object(L, "recursive_checkpoint_contract", return_value={"source_checkpoint_step": 9244}), \
                 patch.object(L, "validate_recursive_config", side_effect=lambda config: validate_model_config(config.to_dict(), production=False)), \
                 patch.object(L, "validate_parameter_audit"), \
                 patch.object(torch, "load", side_effect=AssertionError("must not deserialize text state")):
                restored = L.load_recursive_text_model(path)
            for key, tensor in self.model.state_dict().items():
                self.assertTrue(torch.equal(tensor, restored.state_dict()[key]), key)
            self.assertIs(restored.model.embed_tokens.weight, restored.lm_head.weight)
            audit = restored.audio_initialization_audit
            self.assertEqual(audit["tensor_inheritance"]["router_tensors"], 12)
            self.assertFalse(audit["restore_text_training_state"])
            self.assertEqual(audit["audio_optimizer_start_step"], 0)

    def test_tensor_audit_rejects_changed_router_missing_and_extra_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.model.save_pretrained(path, safe_serialization=True)
            L.audit_checkpoint_tensors(self.model, path)
            with torch.no_grad(): self.model.model.write_routers[0].weight.add_(1)
            with self.assertRaisesRegex(ValueError, "inheritance"): L.audit_checkpoint_tensors(self.model, path)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            del state["model.read_routers.1.bias"]
            save_file(state, str(path / "model.safetensors"))
            with self.assertRaisesRegex(ValueError, "missing inherited"): L.audit_checkpoint_tensors(self.model, path)
            state["extra.key"] = torch.ones(1)
            save_file(state, str(path / "model.safetensors"))
            with self.assertRaisesRegex(ValueError, "unexpected"): L.audit_checkpoint_tensors(self.model, path)

    def test_actual_42_calls_memory_and_both_loop_gradients(self):
        wrapper = SimpleNamespace(caption_decoder=SimpleNamespace(lm=self.model))
        self.model.model.audit_mode = True
        self.model.model.gradient_audit_mode = True
        output = self.model(self.ids, use_cache=False)
        with patch.multiple(T, PREFIX_TOKENS=5, ANSWER_TOKENS=6):
            forward = T.recursive_forward_audit(wrapper, 1, require_memory=True)
            self.assertEqual(forward["memory_shape"], [1, 5, 11, 16])
            self.assertEqual(forward["logical_calls"], 42)
            loss = torch.nn.functional.cross_entropy(output.logits[:, :-1].reshape(-1, 43), self.ids[:, 1:].reshape(-1))
            loss.backward()
            backward = T.recursive_backward_audit(wrapper, 1)
        self.assertTrue(backward["passed"])
        self.assertEqual(len(backward["boundary_gradients"]), 4)
        self.assertEqual(self.model.model.last_core_input_refs, [])
        self.assertTrue(T.router_gradient_audit(wrapper)["passed"])

    def test_actual_683_positions_and_answer_only_shift(self):
        ids = torch.randint(1, 43, (1, 683))
        self.model.model.audit_mode = True
        with torch.no_grad(): output = self.model(ids, use_cache=False)
        wrapper = SimpleNamespace(caption_decoder=SimpleNamespace(lm=self.model))
        self.assertEqual(T.recursive_forward_audit(wrapper, 1, require_memory=True)["memory_shape"], [1, 5, 683, 16])
        targets = torch.randint(1, 43, (1, 300)); targets[:, 20:] = 17; targets[:, 19] = 0
        mask = torch.zeros_like(targets); mask[:, :20] = 1
        logits = torch.randn(1, 683, 43, requires_grad=True)
        inputs = {"answer": {"input_ids": targets, "attention_mask": mask}}
        loss = T.answer_loss(SimpleNamespace(logits=logits), inputs, 17)
        expected = torch.nn.functional.cross_entropy(logits[:, 382:402].reshape(-1, 43), targets[:, :20].reshape(-1))
        torch.testing.assert_close(loss, expected)
        gradient, = torch.autograd.grad(loss, logits)
        self.assertEqual(int(gradient[:, :382].count_nonzero()), 0)
        self.assertGreater(int(gradient[:, 401].count_nonzero()), 0)
        self.assertEqual(int(gradient[:, 402:].count_nonzero()), 0)

    def test_new_optimizer_and_full_state_roundtrip_continue_identically(self):
        class Wrapper(nn.Module):
            def __init__(self, lm):
                super().__init__()
                self.caption_decoder = nn.Module(); self.caption_decoder.lm = lm
                self.audio_encoder = nn.Module(); self.audio_encoder.base = nn.Module()
                self.audio_encoder.base.mapper = nn.Linear(16, 16)
        model = Wrapper(self.model)
        optimizer, scheduler, _ = build_optimizer(model, 34626)
        tracker = GradNormTracker()
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.assertEqual(len(optimizer.state), 0)
        def update(candidate, opt, sched, clip):
            opt.zero_grad(set_to_none=True)
            loss = candidate.caption_decoder.lm(self.ids, use_cache=False).logits.square().mean()
            loss = loss + candidate.audio_encoder.base.mapper(torch.ones(1, 16)).square().mean()
            loss.backward(); clip.track_and_clip_(candidate.named_parameters()); opt.step(); sched.step()
        update(model, optimizer, scheduler, tracker)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step1.ckpt"
            config = {"mode": "formal", "contract": {}, "contract_sha256": "fixture"}
            def gather(states, state):
                for i in range(8): states[i] = copy.deepcopy(state)
            with patch.object(T.dist, "all_gather_object", side_effect=gather), \
                 patch.object(T.dist, "get_rank", return_value=0), patch.object(T.dist, "barrier"), \
                 patch.object(T, "rng_state", return_value={"torch": torch.get_rng_state()}):
                T.save_checkpoint(path, model, optimizer, scheduler, scaler, tracker, config, 1, 0, 1, torch.device("cpu"))
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(len(saved["random_state_by_rank"]), 8)
            self.assertEqual(saved["batch_geometry"]["microbatch_size"], 4)
            self.assertEqual(saved["route_variant"], C.ROUTE)
            T.reload_audit(path, model, config, 1)
            restored = copy.deepcopy(model)
            other_opt, other_sched, _ = build_optimizer(restored, 34626)
            other_tracker = GradNormTracker()
            restored.load_state_dict(saved["state_dict"], strict=True)
            other_opt.load_state_dict(saved["optimizer"]); other_sched.load_state_dict(saved["scheduler"])
            other_tracker.load_state_dict(saved["grad_norm_tracker"])
            scaler.load_state_dict(saved["grad_scaler"])
            torch.set_rng_state(saved["random_state_by_rank"][0]["torch"])
            update(model, optimizer, scheduler, tracker)
            update(restored, other_opt, other_sched, other_tracker)
            for key, tensor in model.state_dict().items(): self.assertTrue(torch.equal(tensor, restored.state_dict()[key]), key)
            self.assertEqual(scheduler.state_dict(), other_sched.state_dict())
            self.assertEqual(tracker.state_dict(), other_tracker.state_dict())

    def make_audio_wrapper(self):
        model = nn.Module()
        model.caption_decoder = nn.Module(); model.caption_decoder.lm = copy.deepcopy(self.model)
        model.audio_encoder = nn.Module(); model.audio_encoder.base = nn.Module()
        model.audio_encoder.base.mapper = nn.Linear(16, 16)
        model.audio_encoder.base.ced = nn.Linear(4, 4).requires_grad_(False).eval()
        return model

    def test_audio_weights_only_inherit_projector_lm_routers_then_two_new_updates(self):
        import audio_initialization as I
        from test_mizar_s1_text71427_audio_reinit import source_fixture
        model = self.make_audio_wrapper()
        with torch.no_grad():
            model.audio_encoder.base.mapper.weight.fill_(.123)
            model.caption_decoder.lm.model.write_routers[0].weight.add_(.05)
        trained = copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path, header = source_fixture(Path(directory))
            header.update(state_dict=trained, optimizer={"stale_lr": .9}, scheduler={"last_epoch": 34626},
                          random_state_by_rank=["old rng"], epoch=3, next_optimizer_step=0)
            torch.save(header, path)
            source = I.inspect_audio_source(path, fingerprint=True)
            fresh = self.make_audio_wrapper()
            fresh.audio_encoder.base.ced.load_state_dict(model.audio_encoder.base.ced.state_dict())
            audit = I.initialize_audio_weights(fresh, source)
            self.assertTrue(audit["all_model_tensors_equal"])
            self.assertEqual(audit["trained_router_tensors"], 12)
            self.assertFalse(audit["restored_training_state"])
            for key, value in trained.items(): self.assertTrue(torch.equal(value, fresh.state_dict()[key]), key)
            optimizer, scheduler, groups = build_optimizer(fresh, 34626)
            self.assertEqual(len(optimizer.state), 0)
            self.assertEqual(scheduler.last_epoch, 0)
            self.assertEqual({group["base_lr"] for group in groups}, {2e-4})
            for group in optimizer.param_groups: self.assertAlmostEqual(group["lr"], 2e-5)
            tracker = GradNormTracker()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                embeddings = fresh.caption_decoder.lm.get_input_embeddings()(self.ids)
                mapped = fresh.audio_encoder.base.mapper(torch.ones_like(embeddings))
                output = fresh.caption_decoder.lm(inputs_embeds=embeddings + mapped, use_cache=False)
                loss = torch.nn.functional.cross_entropy(output.logits[:, :-1].reshape(-1, 43), self.ids[:, 1:].reshape(-1))
                loss.backward(); tracker.track_and_clip_(fresh.named_parameters()); optimizer.step(); scheduler.step()
            self.assertEqual(scheduler.last_epoch, 2)
            for prefix in ("audio_encoder.base.mapper.", "caption_decoder.lm.model.write_routers.", "caption_decoder.lm.model.layers."):
                self.assertTrue(any(not torch.equal(value, fresh.state_dict()[key]) for key, value in trained.items() if key.startswith(prefix)))
            unchanged = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(unchanged["total_step"], 34626)
            for key, value in trained.items(): self.assertTrue(torch.equal(value, unchanged["state_dict"][key]))

    def test_audio_weights_reject_missing_projector_dtype_nonfinite_and_tied_conflict(self):
        import audio_initialization as I
        from test_mizar_s1_text71427_audio_reinit import source_fixture
        model = self.make_audio_wrapper()
        with tempfile.TemporaryDirectory() as directory:
            path, header = source_fixture(Path(directory))
            source = I.inspect_audio_source(path)
            for case in ("missing_projector", "dtype", "nonfinite", "tied"):
                bad = dict(header, state_dict={key: tensor.clone() for key, tensor in model.state_dict().items()})
                state = bad["state_dict"]
                if case == "missing_projector": del state["audio_encoder.base.mapper.weight"]
                elif case == "dtype": state["audio_encoder.base.mapper.weight"] = state["audio_encoder.base.mapper.weight"].double()
                elif case == "nonfinite": state["audio_encoder.base.mapper.bias"][0] = float("nan")
                else: state["caption_decoder.lm.lm_head.weight"].add_(1)
                with self.subTest(case=case), self.assertRaises(ValueError): I.inherit_model_state(model, bad, source)


if __name__ == "__main__": unittest.main()
