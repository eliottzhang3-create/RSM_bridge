"""Audio decoding helpers independent of torchaudio's TorchCodec backend."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import torch


AudioPath = Union[str, Path]


def load_audio(path: AudioPath, *, channels_first: bool = True):
    """Decode audio as float32 with torchaudio-compatible tensor layout."""
    samples, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    if sample_rate <= 0:
        raise ValueError(f"invalid sample rate {sample_rate} for {path}")
    if samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError(f"decoded empty audio file: {path}")
    if channels_first:
        samples = samples.T
    waveform = torch.from_numpy(np.ascontiguousarray(samples))
    return waveform, int(sample_rate)
