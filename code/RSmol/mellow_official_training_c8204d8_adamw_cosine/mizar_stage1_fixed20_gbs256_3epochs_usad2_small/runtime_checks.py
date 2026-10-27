"""CPU tensor checks for the USAD2-Small route; no model weights are loaded."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

from contract import (ACCUMULATION_STEPS, ANSWER_TOKENS, AUDIO_TOKENS,
                      BATCH_SIZE, GLOBAL_BATCH_SIZE, PREFIX_TOKENS,
                      RANK_UPDATE_SAMPLES, SEED, WORLD_SIZE, data_order,
                      geometry, lr_factor)
from data.audiotext_dataset import text_pair
from data.fixed20 import HALF_SAMPLES, load_fixed20, load_mono, self_test
from data.sampler import CustomDistributedSampler
from models.decoder import DecoderModel
from models.mellow import AudioEncoder, Projection, temporal_average_groups
from models.usad2 import USAD2Wrapper
from optim import GradNormTracker, build_optimizer
from stage_data import epoch_priorities, first_optimizer_rows
from train import answer_loss


class RuntimeChecks(unittest.TestCase):
    def test_formal_gbs256_geometry_and_sequence_contract(self):
        actual = geometry(1477492)
        self.assertEqual((WORLD_SIZE, BATCH_SIZE, ACCUMULATION_STEPS, GLOBAL_BATCH_SIZE), (8, 8, 4, 256))
        self.assertEqual((actual["steps_per_epoch"], actual["total_steps"], actual["warmup_steps"],
                          actual["dropped_rows_per_epoch"]), (5771, 17313, 866, 116))
        self.assertEqual((AUDIO_TOKENS, PREFIX_TOKENS, PREFIX_TOKENS + ANSWER_TOKENS), (251, 508, 808))

    def test_fixed20_cropping_padding_and_true_pair_policy(self):
        self_test()
        long = torch.arange(HALF_SAMPLES * 3, dtype=torch.float32)
        with patch("data.fixed20.load_mono", return_value=long):
            whole = load_fixed20("long single")
        start = (len(long) - HALF_SAMPLES * 2) // 2
        self.assertTrue(torch.equal(whole, long[start:start + HALF_SAMPLES * 2]))
        short = torch.ones(HALF_SAMPLES // 2)
        with patch("data.fixed20.load_mono", return_value=short):
            whole = load_fixed20("short single")
        self.assertEqual(whole.shape, (HALF_SAMPLES * 2,))
        self.assertEqual(int(whole.count_nonzero()), len(short))
        first = torch.ones(HALF_SAMPLES // 2)
        second = torch.full((HALF_SAMPLES * 2,), 2.)
        with patch("data.fixed20.load_mono", side_effect=[first, second]):
            whole = load_fixed20("first", "second")
        self.assertEqual(int(whole[:HALF_SAMPLES].count_nonzero()), len(first))
        self.assertTrue(torch.equal(whole[HALF_SAMPLES:], torch.full((HALF_SAMPLES,), 2.)))

    def test_decode_averages_channels_then_resamples_to_32k(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "four_channels.wav"
            samples = torch.tensor([1., -1., 0., 0.]).repeat(8000, 1)
            sf.write(path, samples.numpy(), 16000, subtype="FLOAT")
            mono = load_mono(path)
        self.assertEqual(mono.shape, (16000,))
        self.assertEqual(int(mono.count_nonzero()), 0)

    def test_sampler_staging_and_first_updates_match_mother(self):
        count = 2073
        order = data_order(count)
        planned = first_optimizer_rows(count)
        priorities, _ = epoch_priorities(count)
        permutation = torch.randperm(count, generator=torch.Generator().manual_seed(SEED)).tolist()
        for rank in range(WORLD_SIZE):
            sampler = CustomDistributedSampler(range(count), WORLD_SIZE, rank, seed=SEED, drop_last=True)
            indices = list(sampler)
            self.assertEqual([order[i] for i in indices[:8 * RANK_UPDATE_SAMPLES]], planned[str(rank)])
            self.assertEqual(indices, permutation[rank * 259:(rank + 1) * 259])
            for cursor, dataset_index in enumerate(indices[:8 * RANK_UPDATE_SAMPLES]):
                self.assertEqual(priorities[order[dataset_index]], cursor // RANK_UPDATE_SAMPLES)
            sampler.start_index = 3 * RANK_UPDATE_SAMPLES
            self.assertEqual(list(sampler), indices[3 * RANK_UPDATE_SAMPLES:])

    def test_partial_four_frame_group_uses_only_valid_frames(self):
        frames = torch.arange(999, dtype=torch.float32).reshape(1, 999, 1)
        pooled = temporal_average_groups(frames, torch.tensor([999]))
        self.assertEqual(pooled.shape, (1, 250, 1))
        self.assertEqual(pooled[0, 0, 0].item(), 1.5)
        self.assertEqual(pooled[0, 248, 0].item(), 993.5)
        self.assertEqual(pooled[0, 249, 0].item(), 997.)
        padded = temporal_average_groups(torch.tensor([[[2.], [4.], [100.], [100.]]]),
                                         torch.tensor([2]))
        self.assertEqual(padded.item(), 3.)

    def test_encoder_freeze_projection_gradient_and_token_count(self):
        class FrozenFake(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(()), requires_grad=False)
                self.register_buffer("features", torch.arange(999, dtype=torch.float32)
                                     .reshape(1, 999, 1).expand(1, 999, 384).clone())

            def forward(self, waveform):
                return {"embedding": self.features.expand(waveform.shape[0], -1, -1) * self.scale,
                        "lengths": torch.full((waveform.shape[0],), 999, dtype=torch.long,
                                               device=waveform.device)}

        encoder = AudioEncoder.__new__(AudioEncoder)
        nn.Module.__init__(encoder)
        encoder.base = FrozenFake()
        encoder.projection = Projection(384, 576, p=0.)
        encoder.train()
        self.assertFalse(encoder.base.training)
        output, _, audit = encoder(torch.zeros(1, 640000))
        self.assertEqual(output.shape, (1, 251, 576))
        self.assertEqual(audit["encoder_frames"], 999)
        output.square().mean().backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in encoder.projection.parameters()))
        self.assertTrue(all(p.grad is None for p in encoder.base.parameters()))

    def test_wrapper_resamples_and_encodes_one_complete_20_second_waveform(self):
        class RecordingCore(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, *, wavs, wav_lengths):
                self.calls += 1
                self.waveform = wavs.detach().clone()
                self.input_lengths = wav_lengths.detach().clone()
                self.grad_enabled = torch.is_grad_enabled()
                return {"x": torch.zeros(wavs.shape[0], 999, 384, device=wavs.device),
                        "x_lengths": torch.full((wavs.shape[0],), 999,
                                                dtype=torch.long, device=wavs.device)}

        wrapper = USAD2Wrapper.__new__(USAD2Wrapper)
        nn.Module.__init__(wrapper)
        wrapper.usad2 = RecordingCore()
        wrapper.train()
        self.assertFalse(wrapper.usad2.training)
        waveform = torch.cat((torch.ones(2, 320000), torch.full((2, 320000), 2.)), dim=1)
        with patch("models.usad2.torchaudio.functional.resample",
                   wraps=torchaudio.functional.resample) as resampler:
            output = wrapper(waveform)
        self.assertEqual(resampler.call_count, 1)
        self.assertIs(resampler.call_args.args[0], waveform)
        self.assertEqual(resampler.call_args.kwargs, {"orig_freq": 32000, "new_freq": 16000})
        self.assertEqual(wrapper.usad2.calls, 1)
        self.assertEqual(wrapper.usad2.waveform.shape, (2, 320000))
        self.assertEqual(wrapper.usad2.input_lengths.tolist(), [320000, 320000])
        self.assertFalse(wrapper.usad2.grad_enabled)
        self.assertEqual(output["embedding"].shape, (2, 999, 384))
        self.assertTrue(torch.allclose(wrapper.usad2.waveform[:, 1000:150000],
                                       torch.ones(2, 149000), atol=.001))
        self.assertTrue(torch.allclose(wrapper.usad2.waveform[:, 170000:310000],
                                       torch.full((2, 140000), 2.), atol=.001))

    def test_global_summary_is_projected_and_temporal_pooling_follows_projection(self):
        frames = torch.arange(999, dtype=torch.float32).reshape(1, 999, 1).expand(1, 999, 384)

        class FakeFeatures(nn.Module):
            def forward(self, waveform):
                return {"embedding": frames, "lengths": torch.tensor([999])}

        class RecordingNonlinearProjection(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = SimpleNamespace(in_features=384)

            def forward(self, features):
                self.seen = features.detach().clone()
                return features[:, :, :1].square().expand(-1, -1, 576)

        encoder = AudioEncoder.__new__(AudioEncoder)
        nn.Module.__init__(encoder)
        encoder.base, encoder.projection = FakeFeatures(), RecordingNonlinearProjection()
        output, _, _ = encoder(torch.zeros(1, 640000))
        self.assertEqual(encoder.projection.seen.shape, (1, 1000, 384))
        self.assertTrue(torch.equal(encoder.projection.seen[:, 1:], frames))
        self.assertEqual(encoder.projection.seen[0, 0, 0].item(), 499.)
        self.assertEqual(output[0, 0, 0].item(), 499. ** 2)
        self.assertEqual(output[0, 1, 0].item(), 3.5)
        self.assertAlmostEqual(output[0, -1, 0].item(),
                               (996. ** 2 + 997. ** 2 + 998. ** 2) / 3, delta=.05)

    def test_single_audio_separator_and_answer_only_next_token_ce(self):
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(32, 6)

            def forward(self, **kwargs):
                self.seen = kwargs
                return SimpleNamespace(logits=torch.zeros(2, 808, 32))

        decoder = DecoderModel.__new__(DecoderModel)
        nn.Module.__init__(decoder)
        decoder.text_decoder, decoder.lora, decoder.lm = "smollm2", False, TinyLM()
        audio = torch.randn(2, 251, 6)
        prompt = {"input_ids": torch.ones(2, 256, dtype=torch.long),
                  "attention_mask": torch.zeros(2, 256, dtype=torch.long)}
        pad_id, eos_id = 17, 0
        targets = (torch.arange(300).repeat(2, 1) % 15 + 1).long()
        targets[:, 199] = eos_id
        targets[:, 200:] = pad_id
        answer = {"input_ids": targets}
        decoder(audio, prompt, answer)
        seen = decoder.lm.seen
        embedding = seen["inputs_embeds"]
        self.assertEqual(embedding.shape, (2, 808, 6))
        self.assertIsNone(seen["attention_mask"])
        self.assertIsNone(seen["labels"])
        self.assertTrue(torch.equal(embedding[:, :251], audio))
        self.assertTrue(torch.equal(embedding[:, 251], decoder.lm.model.embed_tokens.weight[0].expand(2, -1)))
        self.assertTrue(torch.equal(embedding[:, 252:508],
                                    decoder.lm.model.embed_tokens(prompt["input_ids"])))
        logits = torch.zeros(2, 808, 32)
        for batch in range(2):
            for index in range(200):
                logits[batch, 507 + index, targets[batch, index]] = 30.
        loss = answer_loss(SimpleNamespace(logits=logits), {"answer": answer}, pad_id)
        self.assertLess(loss.item(), 1e-6)
        eos_wrong = logits.clone()
        eos_wrong[:, 507 + 199] = 0
        eos_wrong[:, 507 + 199, 1] = 30.
        self.assertGreater(answer_loss(SimpleNamespace(logits=eos_wrong), {"answer": answer}, pad_id).item(), .1)
        pad_noise = logits.clone()
        pad_noise[:, 507 + 200:507 + 300, 1] = 30.
        self.assertLess(answer_loss(SimpleNamespace(logits=pad_noise), {"answer": answer}, pad_id).item(), 1e-6)

    def test_gradnorm_and_optimizer_roles_remain_mizar(self):
        a, b = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(1))
        a.grad, b.grad = torch.tensor([10., 0.]), torch.tensor([100.])
        tracker = GradNormTracker()
        norm, scale = tracker.track_and_clip_([("a", a), ("b", b)])
        self.assertAlmostEqual(norm, (10**2 + 100**2)**.5, places=4)
        self.assertAlmostEqual(scale, .025)
        self.assertTrue(torch.allclose(a.grad, torch.tensor([.25, 0.])))
        self.assertTrue(torch.allclose(b.grad, torch.tensor([2.5])))
        restored = GradNormTracker()
        restored.load_state_dict(tracker.state_dict())
        self.assertEqual(restored.state_dict(), tracker.state_dict())

        def make_model():
            model = nn.Module()
            model.audio_encoder = nn.Module()
            model.audio_encoder.base = nn.Module()
            model.audio_encoder.base.usad2 = nn.Linear(3, 3)
            for parameter in model.audio_encoder.base.usad2.parameters():
                parameter.requires_grad_(False)
            model.audio_encoder.projection = nn.Sequential(nn.Linear(3, 3), nn.LayerNorm(3))
            model.caption_decoder = nn.Module()
            model.caption_decoder.lm = nn.Linear(3, 3)
            return model

        model = make_model()
        optimizer, scheduler, audit = build_optimizer(model, 200)
        self.assertEqual({row["base_lr"] for row in audit}, {3e-4, 6e-4})
        self.assertEqual({row["weight_decay"] for row in audit}, {0., .01})
        for _ in range(3):
            sum(p.square().sum() for p in model.parameters() if p.requires_grad).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        restored_model = make_model()
        other_optimizer, other_scheduler, _ = build_optimizer(restored_model, 200)
        restored_model.load_state_dict(model.state_dict())
        other_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        other_scheduler.load_state_dict(scheduler.state_dict())
        for candidate, opt, sched in ((model, optimizer, scheduler),
                                       (restored_model, other_optimizer, other_scheduler)):
            sum(p.square().sum() for p in candidate.parameters() if p.requires_grad).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
        for left, right in zip(model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(scheduler.last_epoch, 4)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"],
                               optimizer.param_groups[0]["initial_lr"] * lr_factor(4, 200))

    def test_text_metadata_contract(self):
        self.assertEqual(text_pair({"input": "Question A", "answer": "Keep CASE"}),
                         ("question a", "keep case"))
        row = {"input": "Question A", "answer": "<think>A</think><answer>B</answer>",
               "prompt_contract": "ke_omni_r_cot_v1"}
        self.assertEqual(text_pair(row), (row["input"], row["answer"]))


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps({"runtime_checks": "PASS", "checks": result.testsRun,
                      "torch": torch.__version__, "cuda": torch.version.cuda,
                      "cuda_visible": torch.cuda.is_available()}), flush=True)
