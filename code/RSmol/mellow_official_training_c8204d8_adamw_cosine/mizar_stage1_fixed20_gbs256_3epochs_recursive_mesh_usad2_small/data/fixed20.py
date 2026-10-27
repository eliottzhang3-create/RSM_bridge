"""Mizar fixed20 waveform policy, decoded online as one complete waveform."""
import torch
import torchaudio
from utils.audio_io import load_audio

SAMPLE_RATE = 32000
HALF_SAMPLES = 10 * SAMPLE_RATE


def fixed_len(waveform, target_len):
    if waveform.numel() > target_len:
        start = (waveform.numel() - target_len) // 2
        return waveform[start:start + target_len].contiguous()
    if waveform.numel() < target_len:
        return torch.nn.functional.pad(waveform, (0, target_len - waveform.numel()))
    return waveform


def load_mono(path):
    waveform, sample_rate = load_audio(path, channels_first=True)
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE)
    waveform = waveform[0].contiguous()
    if not torch.isfinite(waveform).all():
        raise ValueError(f"nonfinite decoded audio: {path}")
    return waveform


def load_fixed20(path1, path2=""):
    first = load_mono(path1)
    if path2:
        whole = torch.cat((fixed_len(first, HALF_SAMPLES),
                           fixed_len(load_mono(path2), HALF_SAMPLES)))
    else:
        whole = fixed_len(first, HALF_SAMPLES * 2)
    if whole.shape != (2 * HALF_SAMPLES,):
        raise RuntimeError(f"fixed20 waveform has shape {tuple(whole.shape)}")
    return whole.contiguous()


def self_test():
    # A single recording is cropped/padded to 20s; each explicit pair member
    # is independently cropped/padded to 10s before one complete waveform is formed.
    long = torch.arange(27, dtype=torch.float32)
    assert torch.equal(fixed_len(long, 20), long[3:23])
    short = fixed_len(torch.arange(7, dtype=torch.float32), 20)
    assert torch.equal(short[:7], torch.arange(7, dtype=torch.float32))
    assert torch.count_nonzero(short[7:]) == 0
    pair = torch.cat((fixed_len(torch.ones(3), 10), fixed_len(torch.full((12,), 2.), 10)))
    assert torch.equal(pair[:10], torch.tensor([1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]))
    assert torch.equal(pair[10:], torch.full((10,), 2.))
    return {"center_crop_or_pad_fixed20": True, "right_padding": True,
            "independent_pair_crop_pad": True, "duplicate_single_audio": False}
