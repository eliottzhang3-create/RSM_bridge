"""Small CPU contract checks inside the existing GPU-job environment (no downloads)."""
import copy
from contextlib import nullcontext
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
                      CED_ASSET_FILES, CED_EXPECTED_CONFIG, RANK_UPDATE_SAMPLES,
                      SEED, WORLD_SIZE, data_order, geometry, lr_factor, write_json)
from data.audiotext_dataset import text_pair
from data.fixed20 import HALF_SAMPLES, load_fixed20, load_mono, self_test
from data.sampler import CustomDistributedSampler
from models.decoder import DecoderModel
from models.ced import CEDSmallWrapper, CEDTemporalMapper
from models.mellow import Mellow
import torchaudio
from optim import GradNormTracker, build_optimizer
from stage_data import epoch_priorities, first_optimizer_rows
from train import answer_loss


class RuntimeChecks(unittest.TestCase):
    def test_loader_is_offline_strict_and_keeps_only_frozen_encoder(self):
        cfg = dict(CED_EXPECTED_CONFIG, model_type="ced",
                   id2label={str(i): f"synthetic_test_label_{i}" for i in range(527)}, auto_map={
            "AutoConfig": "configuration_ced.CedConfig",
            "AutoModelForAudioClassification": "modeling_ced.CedForAudioClassification"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in CED_ASSET_FILES:
                (root / name).write_bytes(b"isolated-synthetic-fixture")
            write_json(root / "config.json", cfg)
            encoder = self.make_wrapper().ced
            encoder.config = SimpleNamespace(**cfg, to_dict=lambda: cfg)
            loaded = nn.Module()
            loaded.encoder = encoder
            for failure in (None, "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
                with self.subTest(failure=failure), patch(
                        "models.ced.local_ced_config", return_value=nullcontext(encoder.config)), patch(
                        "models.ced.AutoModelForAudioClassification.from_pretrained",
                        return_value=(loaded, {} if failure is None else {failure: ["synthetic mismatch"]})) as loader:
                    if failure:
                        with self.assertRaisesRegex(ValueError, "did not load completely"):
                            CEDSmallWrapper({"pretrained_audioencoder_path": str(root)})
                    else:
                        wrapper = CEDSmallWrapper({"pretrained_audioencoder_path": str(root)})
                        self.assertTrue(all(not p.requires_grad for p in wrapper.ced.parameters()))
                        self.assertTrue(all(p.requires_grad for p in wrapper.mapper.parameters()))
                        self.assertFalse(wrapper.ced.training)
                    kwargs = loader.call_args.kwargs
                    self.assertTrue(kwargs["local_files_only"] and kwargs["use_safetensors"])
                    self.assertTrue(kwargs["trust_remote_code"] and kwargs["output_loading_info"])
                    self.assertEqual(kwargs["torch_dtype"], torch.float32)
                    self.assertIs(kwargs["config"], encoder.config)

    def test_mellow_rejoins_both_halves_before_encoding(self):
        class CaptureAudio(nn.Module):
            def forward(self, waveform):
                self.seen = waveform.clone()
                return torch.ones(2, 126, 576), None, {"fixture": True}

        class CaptureDecoder(nn.Module):
            def forward(self, audio, second, prompt, answer):
                self.second = second
                return audio

            def generate_prefix_inference(self, audio, second, prompt):
                self.second = second
                return audio

        model = Mellow.__new__(Mellow)
        nn.Module.__init__(model)
        model.audio_encoder, model.caption_decoder = CaptureAudio(), CaptureDecoder()
        inputs = {"audio1": torch.ones(2, 320000), "audio2": torch.full((2, 320000), 2.),
                  "input": {}, "answer": {}}
        for inference in (False, True):
            if inference:
                model.generate_prefix_inference(inputs)
            else:
                model(inputs)
            waveform = model.audio_encoder.seen
            self.assertEqual(tuple(waveform.shape), (2, 640000))
            self.assertTrue(torch.equal(waveform[:, :320000], inputs["audio1"]))
            self.assertTrue(torch.equal(waveform[:, 320000:], inputs["audio2"]))
            self.assertIsNone(model.caption_decoder.second)

    def make_wrapper(self):
        class ToyCED(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
                self.init_bn = nn.BatchNorm2d(64)
                for p in self.init_bn.parameters():
                    p.requires_grad = False
                self.dropout = nn.Dropout(.5)
                self.maximal_allowed_length = 1012

            def forward_features(self, x):
                self.seen_segments = x.detach().clone()
                return self.dropout(x.mean(dim=(1, 2, 3)).view(-1, 1, 1).expand(-1, 252, 384)) * self.weight

        wrapper = CEDSmallWrapper.__new__(CEDSmallWrapper)
        nn.Module.__init__(wrapper)
        wrapper.ced = ToyCED()
        wrapper.resample = torchaudio.transforms.Resample(32000, 16000)
        wrapper.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, win_length=512, center=True, n_fft=512,
            f_min=0., f_max=8000., hop_length=160, n_mels=64)
        wrapper.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=120)
        wrapper.mapper = CEDTemporalMapper(384, 4, 576, output_tokens=126, resample=False, dropout=.05)
        wrapper.mapper_type = "freq_merge_126"
        return wrapper.train()

    def test_ced_temporal_grid_preserves_chunk_frequency_time_order(self):
        wrapper = self.make_wrapper()
        hidden = torch.arange(504 * 384).reshape(1, 504, 384)
        grid = wrapper._hidden_to_temporal_grid(hidden)
        self.assertEqual(tuple(grid.shape), (1, 126, 4, 384))
        for t in range(126):
            for f in range(4):
                token = (t // 63) * 252 + f * 63 + t % 63
                self.assertTrue(torch.equal(grid[0, t, f], hidden[0, token]))
        with self.assertRaises(ValueError):
            wrapper._hidden_to_temporal_grid(hidden[:, :-1])

    def test_online_db_matches_individual_cache_extraction(self):
        wrapper = self.make_wrapper()
        tone = torch.sin(torch.arange(640000) * (.1 * torch.pi))
        waveforms = torch.stack((tone * 1000., tone * .0001))
        with torch.no_grad():
            batch = wrapper._waveform_to_mel(waveforms)
            individual = torch.cat([wrapper._waveform_to_mel(x[None]) for x in waveforms])
            raw_mel = wrapper.mel_spectrogram(wrapper.resample(waveforms))
            shared_cutoff = wrapper.amplitude_to_db(raw_mel)
        torch.testing.assert_close(batch, individual, rtol=1e-5, atol=1e-4)
        self.assertFalse(torch.allclose(batch[1], shared_cutoff[1]))

    def test_ced_frozen_eval_and_mapper_gradient_with_whole20_splits(self):
        wrapper = self.make_wrapper()
        before = {name: tensor.clone() for name, tensor in wrapper.ced.state_dict().items()}
        self.assertTrue(wrapper.mapper.training)
        self.assertTrue(all(not m.training for m in wrapper.ced.modules()))
        embedding = wrapper(torch.randn(2, 640000))["embedding"]
        self.assertEqual(tuple(embedding.shape), (2, 126, 576))
        self.assertEqual(tuple(wrapper.ced.seen_segments.shape), (4, 1, 64, 1012))
        # Segment two has 989 real frames, then 23 zeros after the frozen BN.
        self.assertEqual(wrapper.ced.seen_segments[1::2, ..., 989:].count_nonzero().item(), 0)
        embedding.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in wrapper.ced.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in wrapper.mapper.parameters()))
        self.assertTrue(any(p.grad.ne(0).any() for p in wrapper.mapper.parameters()))
        self.assertTrue(all(torch.equal(before[name], tensor) for name, tensor in wrapper.ced.state_dict().items()))

    def test_gbs256_optimizer_geometry(self):
        actual = geometry(1477492)
        self.assertEqual((WORLD_SIZE, BATCH_SIZE, ACCUMULATION_STEPS, GLOBAL_BATCH_SIZE), (8, 8, 4, 256))
        self.assertEqual(actual["steps_per_epoch"], 5771)
        self.assertEqual(actual["total_steps"], 17313)
        self.assertEqual(actual["warmup_steps"], 866)
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
                return SimpleNamespace(logits=torch.zeros(2, 683, 32))

        decoder = DecoderModel.__new__(DecoderModel)
        nn.Module.__init__(decoder)
        decoder.text_decoder, decoder.lora, decoder.lm = "smollm2", False, TinyLM()
        decoder.lm_embedding_size = 6
        a, b = torch.randn(2, 126, 6), None
        prompt = {"input_ids": torch.ones(2, 256, dtype=torch.long),
                  "attention_mask": torch.zeros(2, 256, dtype=torch.long)}
        pad_id, eos_id = 17, 0
        targets = (torch.arange(300).repeat(2, 1) % 15 + 1).long()
        targets[:, 199] = eos_id
        targets[:, 200:] = pad_id
        answer = {"input_ids": targets, "attention_mask": targets.ne(pad_id).long()}
        decoder(a, b, prompt, answer)
        seen = decoder.lm.seen
        embedding = seen["inputs_embeds"]
        self.assertEqual(embedding.shape, (2, 683, 6))
        self.assertIsNone(seen["attention_mask"])
        self.assertIsNone(seen["labels"])
        self.assertTrue(torch.equal(embedding[:, :126], a))
        self.assertTrue(torch.equal(embedding[:, 126], decoder.lm.model.embed_tokens.weight[0].expand(2, -1)))
        # Only positions 382..681 predict the 300 answer targets. This detects
        # a one-position shift and including PAD targets in the loss.
        logits = torch.zeros(2, 683, 32)
        for batch in range(2):
            for index in range(200):
                logits[batch, 382 + index, targets[batch, index]] = 30.
        loss = answer_loss(SimpleNamespace(logits=logits), {"answer": answer}, pad_id)
        self.assertLess(loss.item(), 1e-6)
        eos_wrong = logits.clone()
        eos_wrong[:, 382 + 199] = 0
        eos_wrong[:, 382 + 199, 1] = 30.
        self.assertGreater(answer_loss(SimpleNamespace(logits=eos_wrong), {"answer": answer}, pad_id).item(), .1)
        pad_noise = logits.clone()
        pad_noise[:, 382 + 200:382 + 300, 1] = 30.
        self.assertLess(answer_loss(SimpleNamespace(logits=pad_noise), {"answer": answer}, pad_id).item(), 1e-6)
        logits[:, 382:582] = 0
        self.assertGreater(answer_loss(SimpleNamespace(logits=logits), {"answer": answer}, pad_id).item(), 2.)

    def test_recursive_loss_matches_ced_baseline_with_tokenizer_masks(self):
        logits = torch.randn(2, 683, 32, requires_grad=True)
        targets = torch.randint(1, 16, (2, 300))
        targets[:, 120] = 17  # literal ! in real text is ignored by both routes
        targets[:, 199] = 0   # answer EOS remains a valid target
        targets[:, 200:] = 17
        mask = torch.ones_like(targets)
        mask[:, 200:] = 0
        inputs = {"answer": {"input_ids": targets, "attention_mask": mask}}
        recursive_loss = answer_loss(SimpleNamespace(logits=logits), inputs, 17)
        predictions = logits[:, 382:-1]
        baseline_loss = nn.functional.cross_entropy(
            predictions.reshape(-1, 32), targets.flatten(), ignore_index=17)
        torch.testing.assert_close(recursive_loss, baseline_loss)
        recursive_grad, = torch.autograd.grad(recursive_loss, logits, retain_graph=True)
        baseline_grad, = torch.autograd.grad(baseline_loss, logits)
        torch.testing.assert_close(recursive_grad, baseline_grad)

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
            model.audio_encoder.base.mapper = nn.Sequential(nn.Linear(3, 3), nn.LayerNorm(3))
            model.caption_decoder = nn.Module()
            model.caption_decoder.lm = nn.Linear(3, 3)
            model.caption_decoder.lm.model = nn.Module()
            model.caption_decoder.lm.model.audio_linear = nn.Linear(3, 3)
            model.caption_decoder.lm.model.audio_gate = nn.Linear(3, 1)
            return model

        model = make_model()
        optimizer, scheduler, audit = build_optimizer(model, 200)
        self.assertEqual({x["base_lr"] for x in audit}, {3e-4, 6e-4})
        self.assertEqual({x["weight_decay"] for x in audit}, {0., .01})
        for group, description in zip(optimizer.param_groups, audit):
            self.assertAlmostEqual(group["lr"], description["base_lr"] * .1)
        for _ in range(3):
            sum(p.square().sum() for p in model.parameters()).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        restored = make_model()
        other_optimizer, other_scheduler, _ = build_optimizer(restored, 200)
        restored.load_state_dict(model.state_dict())
        other_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        other_scheduler.load_state_dict(scheduler.state_dict())
        for candidate, opt, sched in ((model, optimizer, scheduler), (restored, other_optimizer, other_scheduler)):
            sum(p.square().sum() for p in candidate.parameters()).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
        for a, b in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(scheduler.last_epoch, 4)
        for group in optimizer.param_groups:
            self.assertAlmostEqual(group["lr"], group["initial_lr"] * lr_factor(4, 200))


if __name__ == "__main__":
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps({"runtime_checks": "PASS", "checks": result.testsRun,
                      "torch": torch.__version__, "cuda": torch.version.cuda,
                      "cuda_devices": torch.cuda.device_count()}), flush=True)
