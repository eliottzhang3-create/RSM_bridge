"""Mellow-style USAD2-Small frontend and trainable projection."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from contract import AUDIO_GROUP_SIZE, AUDIO_TOKENS
from models.audio import get_audio_encoder
from models.decoder import get_decoder


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.linear1)
        init_layer(self.linear2)
        init_bn(self.layer_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


def temporal_average_groups(frames: torch.Tensor, lengths: torch.Tensor,
                           group_size: int = AUDIO_GROUP_SIZE) -> torch.Tensor:
    """Non-overlapping mean pooling; a partial final group uses its valid frames."""
    if frames.ndim != 3 or lengths.ndim != 1 or lengths.shape[0] != frames.shape[0]:
        raise ValueError("expected frames [B,T,D] and lengths [B]")
    if group_size <= 0 or bool((lengths <= 0).any().item()) or bool((lengths > frames.shape[1]).any().item()):
        raise ValueError("invalid temporal pooling group size or frame lengths")
    groups = math.ceil(frames.shape[1] / group_size)
    padded_steps = groups * group_size
    pad = padded_steps - frames.shape[1]
    valid = (torch.arange(frames.shape[1], device=frames.device)[None, :] < lengths[:, None])
    values = frames * valid.unsqueeze(-1).to(frames.dtype)
    mask = valid.to(frames.dtype)
    if pad:
        values = F.pad(values, (0, 0, 0, pad))
        mask = F.pad(mask, (0, pad))
    values = values.reshape(frames.shape[0], groups, group_size, frames.shape[2]).sum(dim=2)
    counts = mask.reshape(frames.shape[0], groups, group_size).sum(dim=2).clamp_min_(1)
    return values / counts.unsqueeze(-1)


class AudioEncoder(nn.Module):
    def __init__(self, audioenc_name: str, d_in: int, d_out: int,
                 use_pretrained_audioencoder: bool, freeze_audio_encoder_weights: bool,
                 pretrained_audioencoder_path: str):
        super().__init__()
        audio_encoder, encoder_dim = get_audio_encoder(audioenc_name)
        if not use_pretrained_audioencoder or not freeze_audio_encoder_weights:
            raise ValueError("USAD2-Small must load its local pretrained weights and remain frozen")
        if d_in != encoder_dim or d_out != 576:
            raise ValueError(f"USAD2-Small mapper dimensions must be 384->576, got {d_in}->{d_out}")
        self.base = audio_encoder(pretrained_audioencoder_path)
        self.projection = Projection(encoder_dim, d_out)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, waveform_32k: torch.Tensor):
        # Only frozen feature extraction is under no_grad; projection stays trainable.
        with torch.no_grad():
            encoded = self.base(waveform_32k)
            frames, lengths = encoded["embedding"], encoded["lengths"]
        if frames.shape[-1] != self.projection.linear1.in_features:
            raise RuntimeError(f"unexpected USAD2 hidden width {frames.shape[-1]}")
        valid = (torch.arange(frames.shape[1], device=frames.device)[None, :] < lengths[:, None])
        denominator = lengths.to(frames.dtype).unsqueeze(-1).clamp_min_(1)
        global_token = (frames * valid.unsqueeze(-1).to(frames.dtype)).sum(dim=1, keepdim=True) / denominator.unsqueeze(1)
        # USAD2 has no CLS token. Use the valid-frame mean as one global summary,
        # prepend it to all frame states, then apply the same trainable Mellow map.
        projected = self.projection(torch.cat((global_token, frames), dim=1))
        global_projected = projected[:, :1]
        pooled_frames = temporal_average_groups(projected[:, 1:], lengths)
        output = torch.cat((global_projected, pooled_frames), dim=1).contiguous()
        if output.shape[1] != AUDIO_TOKENS:
            raise RuntimeError(f"mapper must produce {AUDIO_TOKENS} tokens, got {output.shape[1]}")
        audit = {"encoder_frames": int(frames.shape[1]), "encoder_dim": int(frames.shape[2]),
                 "audio_tokens": int(output.shape[1]), "mapper_group_size": AUDIO_GROUP_SIZE}
        return output, None, audit


class Mellow(nn.Module):
    def __init__(self, audioenc_name: str, d_in: int, text_decoder: str,
                 prefix_length: int, freeze_text_decoder_weights: bool, d_out: int,
                 use_pretrained_audioencoder: bool, freeze_audio_encoder_weights: bool,
                 pretrained_audioencoder_path: str | None = None):
        super().__init__()
        self.audio_encoder = AudioEncoder(
            audioenc_name, d_in, d_out, use_pretrained_audioencoder,
            freeze_audio_encoder_weights, pretrained_audioencoder_path)
        self.caption_decoder = get_decoder("Decoder")(
            text_decoder, prefix_length, freeze_text_decoder_weights)

    def forward(self, input_dict):
        audio, _, _ = self.audio_encoder(input_dict["audio"])
        return self.caption_decoder(audio, input_dict["input"], input_dict["answer"])

    def generate_prefix_inference(self, input_dict):
        audio, _, audit = self.audio_encoder(input_dict["audio"])
        prefix = self.caption_decoder.generate_prefix_inference(audio, input_dict["input"])
        return prefix, audit
