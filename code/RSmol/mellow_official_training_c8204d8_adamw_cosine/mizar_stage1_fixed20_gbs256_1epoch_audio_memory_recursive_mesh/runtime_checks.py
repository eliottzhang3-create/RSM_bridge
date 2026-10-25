"""Small CPU contract checks inside the existing GPU-job environment (no downloads)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import soundfile as sf

from contract import (ACCUMULATION_STEPS, BATCH_SIZE, GLOBAL_BATCH_SIZE,
                      RANK_UPDATE_SAMPLES, SEED, WORLD_SIZE, data_order, geometry, lr_factor)
from data.audiotext_dataset import text_pair
from data.fixed20 import HALF_SAMPLES, load_fixed20, load_mono, self_test
from data.sampler import CustomDistributedSampler
from models.decoder import DecoderModel, downsample
from optim import GradNormTracker, build_optimizer
from stage_data import epoch_priorities, first_optimizer_rows
from train import answer_loss
from audio_memory import AudioAttention, AudioMemoryBuilder, temporal_encoding, ungroup_stage2
from optim import configure_trainability


class RuntimeChecks(unittest.TestCase):
    def test_gbs256_optimizer_geometry(self):
        actual = geometry(1477492)
        self.assertEqual((WORLD_SIZE, BATCH_SIZE, ACCUMULATION_STEPS, GLOBAL_BATCH_SIZE), (8, 8, 4, 256))
        self.assertEqual(actual["steps_per_epoch"], 5771)
        self.assertEqual(actual["total_steps"], 5771)
        self.assertEqual(actual["warmup_steps"], 289)
        self.assertEqual(actual["dropped_rows_per_epoch"], 116)

    def test_fixed20_single_order_and_right_padding(self):
        self_test()
        length = HALF_SAMPLES * 3
        long = torch.arange(length, dtype=torch.float32)
        with patch("data.fixed20.load_mono", return_value=long):
            a, b = load_fixed20("one recording")
        start = HALF_SAMPLES // 2
        self.assertTrue(torch.equal(torch.cat((a, b)), long[start:start + 2 * HALF_SAMPLES]))
        short = torch.ones(HALF_SAMPLES // 2)
        with patch("data.fixed20.load_mono", return_value=short):
            a, b = load_fixed20("short recording")
        self.assertEqual(int(a.count_nonzero()), len(short))
        self.assertEqual(int(b.count_nonzero()), 0)

    def test_fixed20_true_pair_is_independent(self):
        first = torch.ones(HALF_SAMPLES // 2)
        second = torch.full((HALF_SAMPLES * 2,), 2.)
        with patch("data.fixed20.load_mono", side_effect=[first, second]):
            a, b = load_fixed20("raw1", "raw2")
        self.assertEqual(int(a.count_nonzero()), len(first))
        self.assertTrue(torch.equal(b, torch.full((HALF_SAMPLES,), 2.)))

    def test_decode_averages_every_channel_then_resamples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "four_channels.wav"
            samples = torch.tensor([1., -1., 0., 0.]).repeat(8000, 1)
            sf.write(path, samples.numpy(), 16000, subtype="FLOAT")
            mono = load_mono(path)
        self.assertEqual(mono.shape, (16000,))
        self.assertEqual(int(mono.count_nonzero()), 0)

    def test_sampler_staging_and_consumed_cursor_agree(self):
        count = 2073  # eight full optimizer updates plus an uneven per-rank tail
        order = data_order(count)
        planned = first_optimizer_rows(count)
        priorities, _ = epoch_priorities(count)
        permutation = torch.randperm(count, generator=torch.Generator().manual_seed(SEED)).tolist()
        all_indices = []
        for rank in range(WORLD_SIZE):
            sampler = CustomDistributedSampler(range(count), WORLD_SIZE, rank, seed=SEED, drop_last=True)
            indices = list(sampler)
            self.assertEqual([order[i] for i in indices[:8 * RANK_UPDATE_SAMPLES]], planned[str(rank)])
            self.assertEqual(indices, permutation[rank * 259:(rank + 1) * 259])
            for cursor, dataset_index in enumerate(indices[:8 * RANK_UPDATE_SAMPLES]):
                self.assertEqual(priorities[order[dataset_index]], cursor // RANK_UPDATE_SAMPLES)
            sampler.start_index = 3 * RANK_UPDATE_SAMPLES
            self.assertEqual(list(sampler), indices[3 * RANK_UPDATE_SAMPLES:])
            self.assertEqual(len(sampler), 259 - 3 * RANK_UPDATE_SAMPLES)
            all_indices.extend(indices)
        self.assertEqual(len(set(all_indices)), 2072)
        priorities, _ = epoch_priorities(count, epoch=1, start_batch=7)
        for rank in range(WORLD_SIZE):
            sampler = CustomDistributedSampler(range(count), WORLD_SIZE, rank, seed=SEED, drop_last=True)
            sampler.set_epoch(1)
            for cursor, dataset_index in enumerate(list(sampler)[:8 * RANK_UPDATE_SAMPLES]):
                expected = 8 if cursor < 7 * RANK_UPDATE_SAMPLES else cursor // RANK_UPDATE_SAMPLES
                self.assertEqual(priorities[order[dataset_index]], expected)

    def test_accumulated_gradient_is_mean_of_microbatch_means(self):
        model = nn.Linear(3, 1, bias=False)
        batches = [(torch.full((BATCH_SIZE, 3), float(i + 1)),
                    torch.full((BATCH_SIZE, 1), float(2 * i - 1)))
                   for i in range(ACCUMULATION_STEPS)]
        for x, y in batches:
            nn.functional.mse_loss(model(x), y).div(ACCUMULATION_STEPS).backward()
        accumulated = model.weight.grad.detach().clone()
        model.zero_grad(set_to_none=True)
        mean_loss = sum(nn.functional.mse_loss(model(x), y) for x, y in batches) / ACCUMULATION_STEPS
        mean_loss.backward()
        self.assertTrue(torch.allclose(accumulated, model.weight.grad, rtol=1e-6, atol=1e-6))

    def test_text_metadata_fallback_and_cot_preservation(self):
        _, caption = text_pair({"input": "caption first audio", "answer": "Fallback Answer."})
        self.assertEqual(caption, "Fallback Answer.")
        _, caption = text_pair({"input": "caption both audios", "answer": "fallback",
                                "metadata": {"caption1": "DOG.", "caption2": "RAIN."}})
        self.assertEqual(caption, "The audio 1 is dog. The audio 2 is rain.")
        row = {"input": "Question A", "answer": "<think>ABC</think><answer>D</answer>",
               "prompt_contract": "ke_omni_r_cot_v1"}
        self.assertEqual(text_pair(row), (row["input"], row["answer"]))
        self.assertEqual(text_pair({"input": "ABC", "answer": "DEF"}), ("abc", "def"))

    def test_one_separator_padding_visibility_and_next_token_ce(self):
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(32, 6)

            def get_input_embeddings(self):
                return self.model.embed_tokens

            def forward(self, **kwargs):
                self.seen = kwargs
                return SimpleNamespace(logits=torch.zeros(2, 815, 32))

        decoder = DecoderModel.__new__(DecoderModel)
        nn.Module.__init__(decoder)
        decoder.text_decoder = "recursive-mesh-fixture"
        decoder.text_model_contract = "logical_30_physical_20_5_10x2_5"
        decoder.is_recursive_mesh = True
        decoder.lora = False
        decoder.lm = TinyLM()
        decoder._embedding = decoder.lm.model.embed_tokens
        decoder.lm_embedding_size = 6
        decoder.separator_token_id = 0
        a, b = torch.randn(2, 1025, 6), torch.randn(2, 1025, 6)
        prompt = {"input_ids": torch.ones(2, 256, dtype=torch.long),
                  "attention_mask": torch.zeros(2, 256, dtype=torch.long)}
        pad_id, eos_id = 17, 0
        targets = (torch.arange(300).repeat(2, 1) % 15 + 1).long()
        targets[:, 199] = eos_id
        targets[:, 200:] = pad_id
        answer = {"input_ids": targets,
                  "attention_mask": targets.ne(pad_id).long()}
        decoder(a, b, prompt, answer)
        seen = decoder.lm.seen
        embedding = seen["inputs_embeds"]
        self.assertEqual(embedding.shape, (2, 815, 6))
        self.assertIsNone(seen["attention_mask"])
        self.assertIsNone(seen["labels"])
        self.assertTrue(torch.equal(embedding[:, :129], downsample(a)))
        self.assertTrue(torch.equal(embedding[:, 129:258], downsample(b)))
        self.assertTrue(torch.equal(embedding[:, 258], decoder.lm.model.embed_tokens.weight[0].expand(2, -1)))
        # Only positions 514..813 predict the 300 answer targets. This detects
        # a one-position shift and including PAD targets in the loss.
        logits = torch.zeros(2, 815, 32)
        for batch in range(2):
            for index in range(200):
                logits[batch, 514 + index, targets[batch, index]] = 30.
        loss = answer_loss(SimpleNamespace(logits=logits), {"answer": answer}, pad_id)
        self.assertLess(loss.item(), 1e-6)
        eos_wrong = logits.clone()
        eos_wrong[:, 514 + 199] = 0
        eos_wrong[:, 514 + 199, 1] = 30.
        self.assertGreater(answer_loss(SimpleNamespace(logits=eos_wrong), {"answer": answer}, pad_id).item(), .1)
        pad_noise = logits.clone()
        pad_noise[:, 514 + 200:514 + 300, 1] = 30.
        self.assertLess(answer_loss(SimpleNamespace(logits=pad_noise), {"answer": answer}, pad_id).item(), 1e-6)
        logits[:, 514:714] = 0
        self.assertGreater(answer_loss(SimpleNamespace(logits=logits), {"answer": answer}, pad_id).item(), 2.)

    def test_gradnorm_ema_common_scale_and_restore(self):
        a, b = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(1))
        a.grad, b.grad = torch.tensor([10., 0.]), torch.tensor([100.])
        tracker = GradNormTracker()
        norm, scale = tracker.track_and_clip_([("a", a), ("b", b)])
        self.assertAlmostEqual(norm, (10**2 + 100**2)**.5, places=4)
        self.assertAlmostEqual(scale, .025)
        self.assertTrue(torch.allclose(a.grad, torch.tensor([.25, 0.])))
        self.assertTrue(torch.allclose(b.grad, torch.tensor([2.5])))
        self.assertAlmostEqual(tracker.running_norm["b"][0], .995 + .005 * 6.25)
        restored = GradNormTracker()
        restored.load_state_dict(tracker.state_dict())
        self.assertEqual(restored.state_dict(), tracker.state_dict())

    def test_optimizer_groups_scheduler_step_order_and_full_resume(self):
        def make_model():
            model = nn.Module()
            model.audio_encoder = nn.Module()
            model.audio_encoder.base = nn.Module()
            model.audio_encoder.base.c2l = nn.Linear(3, 3)
            model.audio_encoder.projection = nn.Sequential(nn.Linear(3, 3), nn.LayerNorm(3))
            model.caption_decoder = nn.Module()
            model.caption_decoder.lm = nn.Module()
            model.caption_decoder.lm.model = nn.Module()
            mesh = model.caption_decoder.lm.model
            mesh.layers = nn.ModuleList([nn.Linear(3, 3) for _ in range(20)])
            mesh.embed_tokens = nn.Embedding(32, 3)
            mesh.norm = nn.LayerNorm(3)
            mesh.read_routers = nn.ModuleList([nn.Linear(3, 5) for _ in range(3)])
            mesh.write_routers = nn.ModuleList([nn.Linear(3, 5) for _ in range(3)])
            mesh.perceive_routers = nn.ModuleList([nn.Linear(3, 5) for _ in range(2)])
            mesh.audio_memory_builder = nn.Linear(3, 3)
            model.caption_decoder.lm.lm_head = nn.Linear(3, 32)
            configure_trainability(model)
            self.assertFalse(any(p.requires_grad for p in model.audio_encoder.parameters()))
            self.assertTrue(all(p.requires_grad == (5 <= i < 15)
                                for i, layer in enumerate(mesh.layers) for p in layer.parameters()))
            self.assertFalse(any(p.requires_grad for p in mesh.embed_tokens.parameters()))
            self.assertFalse(any(p.requires_grad for p in mesh.norm.parameters()))
            self.assertFalse(any(p.requires_grad for p in model.caption_decoder.lm.lm_head.parameters()))
            return model

        model = make_model()
        optimizer, scheduler, audit = build_optimizer(model, 200)
        self.assertEqual({x["base_lr"] for x in audit}, {5e-5})
        self.assertEqual({x["min_lr"] for x in audit}, {5e-6})
        self.assertEqual({x["weight_decay"] for x in audit}, {0., .01})
        for group, description in zip(optimizer.param_groups, audit):
            self.assertAlmostEqual(group["lr"], description["base_lr"] * .1)
        for _ in range(3):
            sum(p.square().sum() for p in model.parameters() if p.requires_grad).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        restored = make_model()
        other_optimizer, other_scheduler, _ = build_optimizer(restored, 200)
        restored.load_state_dict(model.state_dict())
        other_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        other_scheduler.load_state_dict(scheduler.state_dict())
        from audits import assert_state_exact
        assert_state_exact(optimizer.state_dict(), other_optimizer.state_dict(), "optimizer")
        assert_state_exact(scheduler.state_dict(), other_scheduler.state_dict(), "scheduler")
        with self.assertRaisesRegex(ValueError, "serialized state differs"):
            bad = copy.deepcopy(other_optimizer.state_dict())
            next(iter(bad["state"].values()))["exp_avg"].add_(1.)
            assert_state_exact(optimizer.state_dict(), bad, "optimizer")
        for candidate, opt, sched in ((model, optimizer, scheduler), (restored, other_optimizer, other_scheduler)):
            sum(p.square().sum() for p in candidate.parameters() if p.requires_grad).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
        for a, b in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(scheduler.last_epoch, 4)
        for group in optimizer.param_groups:
            self.assertAlmostEqual(group["lr"], group["initial_lr"] * lr_factor(4, 200))

    def test_real_stage2_ungroup_and_both_chronological_halves(self):
        packed = torch.arange(1024, dtype=torch.float32).view(1, 1024, 1).expand(1, 1024, 192)
        grid = ungroup_stage2(packed)
        self.assertEqual(tuple(grid.shape), (1, 8, 128, 192))
        for frequency in range(8):
            for time in range(128):
                expected = ((time // 32) * 8 + frequency) * 32 + time % 32
                self.assertEqual(grid[0, frequency, time, 0].item(), expected)
        builder = AudioMemoryBuilder()
        builder.reset_parameters()
        observed = []
        handle = builder.stage2_norm.register_forward_pre_hook(lambda _m, args: observed.append(args[0].clone()))
        memory = builder(packed, packed + 10000)
        handle.remove()
        self.assertEqual(tuple(memory.shape), (1, 2048, 576))
        self.assertTrue(torch.equal(observed[0][:, :, :128], grid))
        self.assertTrue(torch.equal(observed[0][:, :, 128:], grid + 10000))
        before = memory.detach().reshape(1, 8, 256, 576)
        with torch.no_grad():
            builder.frequency_embedding[3].fill_(1.)
        after = builder(packed, packed + 10000).detach().reshape(1, 8, 256, 576)
        delta = after - before
        self.assertTrue(torch.allclose(delta[:, 3], torch.ones_like(delta[:, 3]), atol=1e-6))
        self.assertEqual(int(delta[:, :3].count_nonzero()) + int(delta[:, 4:].count_nonzero()), 0)
        time = temporal_encoding(device="cpu", dtype=torch.float32).reshape(256, 576)
        self.assertAlmostEqual(time[0, 0].item(), __import__("math").sin(.5), places=6)
        self.assertAlmostEqual(time[255, 0].item(), __import__("math").sin(255.5), places=6)

    def test_htsat_stage2_tap_keeps_complete_final_embedding(self):
        from models.htsat import HTSATWrapper
        htsat = HTSATWrapper().eval()
        stage_calls = [0, 0]
        def count(index):
            def hook(*_args):
                stage_calls[index] += 1
            return hook
        handles = [htsat.htsat.layers[i].register_forward_hook(count(i - 2)) for i in (2, 3)]
        waveform = torch.randn(1, HALF_SAMPLES)
        try:
            with torch.no_grad():
                tapped = htsat(waveform, capture_stage2=True)
                original = htsat(waveform, capture_stage2=False)
            self.assertEqual(tuple(tapped["stage2"].shape), (1, 1024, 192))
            self.assertEqual(tuple(tapped["embedding"].shape), (1, 1025, 768))
            self.assertEqual(stage_calls, [2, 2])
            self.assertTrue(torch.equal(tapped["embedding"], original["embedding"]))
        finally:
            for handle in handles:
                handle.remove()

    def test_real_mesh_zero_parity_queries_freeze_and_boundary_gradients(self):
        """Exercise the actual 20-layer implementation with a small FFN/vocab."""
        import importlib.util
        from transformers import LlamaConfig
        from recursive_model_audio_memory import RecursiveLlamaForCausalLM
        from audits import assert_query_semantics, frozen_digest, gradient_audit
        from initialization import NEW_PREFIXES, SOURCE_ROUTE, load_trained_initialization

        config = LlamaConfig(hidden_size=576, intermediate_size=64, num_hidden_layers=30,
                             num_attention_heads=9, num_key_value_heads=3, vocab_size=32,
                             recursive_layer_count=20, recursive_loops=2, use_cache=False,
                             tie_word_embeddings=False, attention_dropout=0.)
        # An independent execution of the unchanged legacy model is the parity
        # reference; toggling the new model alone would miss a legacy regression.
        legacy_path = Path(__file__).resolve().parents[2] / "recursive_model_5_10x2_5_mesh.py"
        spec = importlib.util.spec_from_file_location("legacy_mesh_runtime_reference", legacy_path)
        legacy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy_module)
        legacy = legacy_module.RecursiveLlamaForCausalLM(copy.deepcopy(config)).eval()
        wrapper = nn.Module()
        wrapper.audio_encoder = nn.Linear(2, 2)
        wrapper.caption_decoder = nn.Module()
        wrapper.caption_decoder.lm = RecursiveLlamaForCausalLM(copy.deepcopy(config))
        source = {"caption_decoder.lm." + name: value for name, value in legacy.state_dict().items()}
        source.update({"audio_encoder." + name: value for name, value in wrapper.audio_encoder.state_dict().items()})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.ckpt"
            torch.save({"route_variant": SOURCE_ROUTE, "total_step": 17313, "state_dict": source}, path)
            audit = load_trained_initialization(wrapper, path)
            self.assertTrue(audit["legacy_tensors_exact"])
            self.assertTrue(all(name.startswith(NEW_PREFIXES) for name in audit["new_parameter_tensors"]))
            broken = dict(source)
            broken.pop("caption_decoder.lm.model.layers.5.mlp.down_proj.weight")
            torch.save({"route_variant": SOURCE_ROUTE, "total_step": 17313, "state_dict": broken}, path)
            with self.assertRaisesRegex(ValueError, "legacy initialization mismatch"):
                load_trained_initialization(wrapper, path)
        configure_trainability(wrapper)
        wrapper.eval()
        lm = wrapper.caption_decoder.lm
        mesh = lm.model
        mesh.audit_mode = True
        ids = torch.tensor([[2, 3, 4]])
        first, second = torch.randn(1, 1024, 192), torch.randn(1, 1024, 192)
        with torch.no_grad():
            expected = legacy(input_ids=ids, use_cache=False).logits
            actual = lm(input_ids=ids, audio_stage2_1=first, audio_stage2_2=second, use_cache=False).logits
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(assert_query_semantics(mesh)["memory_recurrence"])
        del legacy
        frozen_before = frozen_digest(wrapper)
        # Open both output projections to exercise every new upstream path.
        with torch.no_grad():
            for output in mesh.audio_attention.o_projs:
                output.weight.normal_(0., .01)
        mesh.gradient_audit_mode = True
        result = lm(input_ids=ids, audio_stage2_1=first, audio_stage2_2=second, use_cache=False)
        result.logits.square().mean().backward()
        self.assertTrue(assert_query_semantics(mesh)["perceive1_H0"])
        self.assertTrue(gradient_audit(wrapper)["all_trainable_grads_present_finite"])
        self.assertGreater(mesh.audio_memory_builder.frequency_embedding.grad.norm().item(), 0.)
        for router in mesh.perceive_routers:
            self.assertGreater(router.weight.grad.norm().item(), 0.)
        self.assertEqual(frozen_digest(wrapper), frozen_before)

    def test_zero_output_projections_open_gradient_path_after_first_update(self):
        torch.manual_seed(1234)
        attention = AudioAttention()
        attention.reset_parameters()
        self.assertEqual((attention.heads, attention.head_dim), (9, 64))
        self.assertIsNot(attention.q_projs[0].weight, attention.q_projs[1].weight)
        self.assertIsNot(attention.o_projs[0].weight, attention.o_projs[1].weight)
        query = torch.randn(1, 4, 576)
        memory = torch.randn(1, 16, 576)
        kv = attention.prepare_kv(memory)
        outputs = [attention(query, kv, i) for i in range(2)]
        self.assertTrue(all(x.count_nonzero() == 0 for x in outputs))
        sum(x.sum() for x in outputs).backward()
        for proj in attention.o_projs:
            self.assertGreater(proj.weight.grad.abs().sum().item(), 0.)
        self.assertEqual(attention.k_proj.weight.grad.count_nonzero().item(), 0)
        with torch.no_grad():
            for proj in attention.o_projs:
                proj.weight.normal_(0., .01)
        attention.zero_grad(set_to_none=True)
        kv = attention.prepare_kv(memory)
        sum(attention(query, kv, i).square().mean() for i in range(2)).backward()
        for proj in list(attention.q_projs) + [attention.k_proj, attention.v_proj]:
            self.assertGreater(proj.weight.grad.abs().sum().item(), 0.)


if __name__ == "__main__":
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps({"runtime_checks": "PASS", "checks": result.testsRun,
                      "torch": torch.__version__, "cuda": torch.version.cuda,
                      "cuda_devices": torch.cuda.device_count()}), flush=True)
