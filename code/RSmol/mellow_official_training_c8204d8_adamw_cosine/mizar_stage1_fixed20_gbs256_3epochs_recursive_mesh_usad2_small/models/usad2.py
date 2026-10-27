"""Explicit local loader and frozen waveform frontend for USAD2-Small."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
from torch import nn
import torchaudio

from contract import USAD2_DIM, USAD2_FRAMES, validate_usad2_assets


SAMPLE_RATE_IN = 32000
SAMPLE_RATE_USAD2 = 16000
WAVEFORM_SAMPLES_20S = 20 * SAMPLE_RATE_IN
USAD2_SAMPLES_20S = 20 * SAMPLE_RATE_USAD2
USAD2_FRAMES_20S = USAD2_FRAMES


def load_local_usad2(model_dir: str | Path):
    """Load the shipped Python package directly; never use HF dynamic modules."""
    root = Path(model_dir).expanduser().resolve()
    validate_usad2_assets(root)

    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    package = importlib.import_module("USAD2")
    if Path(package.__file__).resolve().parent != root:
        raise ImportError(f"USAD2 package resolved outside the requested asset directory: {package.__file__}")
    config_module = importlib.import_module("USAD2.configuration_usad2")
    modeling_module = importlib.import_module("USAD2.modeling_usad2")
    for module in (config_module, modeling_module):
        if Path(module.__file__).resolve().parent != root:
            raise ImportError(f"USAD2 module resolved outside the requested asset directory: {module.__file__}")
    Usad2Config = config_module.Usad2Config
    Usad2Model = modeling_module.Usad2Model

    config = Usad2Config.from_pretrained(str(root), local_files_only=True)
    if (config.encoder_dim != USAD2_DIM or config.sample_rate != SAMPLE_RATE_USAD2
            or config.num_layers != 12 or config.num_attention_heads != 6):
        raise ValueError("USAD2-Small config does not match the expected 384-dim/12-layer/16-kHz contract")
    model, loading = Usad2Model.from_pretrained(
        str(root), config=config, local_files_only=True, use_safetensors=True,
        output_loading_info=True, ignore_mismatched_sizes=False)
    problems = {key: loading.get(key, []) for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if loading.get(key)}
    if problems:
        raise RuntimeError(f"USAD2 safetensors load was not exact: {problems}")
    model.float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class USAD2Wrapper(nn.Module):
    """One 20-second 32-kHz waveform in, 999 frozen 384-d features out."""
    def __init__(self, model_dir: str | Path):
        super().__init__()
        self.usad2 = load_local_usad2(model_dir)

    def train(self, mode: bool = True):
        super().train(mode)
        self.usad2.eval()
        return self

    def forward(self, waveform_32k: torch.Tensor):
        if waveform_32k.ndim != 2 or waveform_32k.shape[1] != WAVEFORM_SAMPLES_20S:
            raise ValueError(f"USAD2 route expects [B,{WAVEFORM_SAMPLES_20S}] 20-second 32-kHz audio, "
                             f"got {tuple(waveform_32k.shape)}")
        if waveform_32k.dtype != torch.float32:
            raise TypeError(f"USAD2 input must be FP32, got {waveform_32k.dtype}")
        waveform_16k = torchaudio.functional.resample(
            waveform_32k, orig_freq=SAMPLE_RATE_IN, new_freq=SAMPLE_RATE_USAD2)
        if waveform_16k.shape[1] != USAD2_SAMPLES_20S:
            raise RuntimeError(f"20-second resample produced {waveform_16k.shape[1]} samples, "
                               f"expected {USAD2_SAMPLES_20S}")
        lengths = torch.full((waveform_16k.shape[0],), USAD2_SAMPLES_20S,
                             dtype=torch.long, device=waveform_16k.device)
        with torch.no_grad():
            result = self.usad2(wavs=waveform_16k, wav_lengths=lengths)
        hidden, output_lengths = result["x"], result["x_lengths"]
        if hidden.shape != (waveform_16k.shape[0], USAD2_FRAMES_20S, USAD2_DIM):
            raise RuntimeError(f"USAD2 20-second output must be [B,{USAD2_FRAMES_20S},{USAD2_DIM}], "
                               f"got {tuple(hidden.shape)}")
        if (hidden.dtype != torch.float32 or hidden.device != waveform_16k.device
                or output_lengths.shape != (waveform_16k.shape[0],)
                or output_lengths.dtype != torch.long or output_lengths.device != hidden.device):
            raise RuntimeError("USAD2 must return FP32 features and int64 lengths [B] on the input device")
        if not torch.equal(output_lengths, torch.full_like(output_lengths, USAD2_FRAMES_20S)):
            raise RuntimeError(f"USAD2 20-second output lengths differ from 999: {output_lengths.tolist()}")
        return {"embedding": hidden, "lengths": output_lengths}
