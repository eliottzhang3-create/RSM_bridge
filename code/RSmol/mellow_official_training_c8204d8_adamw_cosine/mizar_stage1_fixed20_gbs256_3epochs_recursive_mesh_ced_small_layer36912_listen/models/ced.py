"""Mizar S1 CED-Small + freq_merge_126, computed online in FP32.

The mapper classes below retain the official Mizar definitions and initialization.
Only the fixed20 waveform wrapper and per-sample batched dB handling are adapted.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModelForAudioClassification

from contract import AUDIO_TOKENS, validate_ced_assets, validate_ced_config
from models.ced_offline import local_ced_config


class CEDFrequencyMerger(nn.Module):
    def __init__(self, hidden_dim: int, freq_bins: int, d_out: int, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim * freq_bins),
            nn.Linear(hidden_dim * freq_bins, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time_steps, freq_bins, hidden_dim = x.shape
        return self.net(x.reshape(bsz, time_steps, freq_bins * hidden_dim))


class CEDTemporalMapper(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        freq_bins: int,
        d_out: int,
        output_tokens: int = 258,
        resample: bool = True,
        max_temporal_tokens: int = 512,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.output_tokens = int(output_tokens)
        self.resample = bool(resample)
        self.freq_merger = CEDFrequencyMerger(hidden_dim, freq_bins, d_out, dropout=dropout)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_out),
            nn.LayerNorm(d_out),
        )
        self.pos_in = nn.Parameter(torch.empty(1, max_temporal_tokens, d_out))
        self.pos_out = nn.Parameter(torch.empty(1, max(output_tokens, max_temporal_tokens), d_out))
        self.audio_type = nn.Parameter(torch.empty(1, 1, d_out))
        self.refine_ln = nn.LayerNorm(d_out)
        self.refine_mlp = nn.Sequential(
            nn.Linear(d_out, d_out * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out * 2, d_out),
        )
        nn.init.normal_(self.pos_in, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_out, mean=0.0, std=0.02)
        nn.init.normal_(self.audio_type, mean=0.0, std=0.02)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        bsz, time_steps, _freq_bins, _hidden_dim = grid.shape
        h = self.freq_merger(grid)
        global_h = grid.mean(dim=(1, 2))
        h = h + self.global_proj(global_h).unsqueeze(1)
        if time_steps > self.pos_in.shape[1]:
            raise ValueError(f"CED temporal length {time_steps} exceeds max positional length {self.pos_in.shape[1]}")
        h = h + self.pos_in[:, :time_steps, :].to(device=h.device, dtype=h.dtype)

        if self.resample and time_steps != self.output_tokens:
            h = F.interpolate(
                h.transpose(1, 2),
                size=self.output_tokens,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).contiguous()

        out_tokens = h.shape[1]
        if out_tokens > self.pos_out.shape[1]:
            raise ValueError(f"CED output length {out_tokens} exceeds max positional length {self.pos_out.shape[1]}")
        h = h + self.pos_out[:, :out_tokens, :].to(device=h.device, dtype=h.dtype)
        h = h + self.audio_type.to(device=h.device, dtype=h.dtype)
        h = h + self.refine_mlp(self.refine_ln(h))
        return h


class CEDSmallWrapper(nn.Module):
    single_audio_mode = True

    def __init__(self, encoder_config=None, d_out=576):
        super().__init__()
        encoder_config = dict(encoder_config or {})
        if d_out != 576:
            raise ValueError("CED mapper requires the unchanged SmolLM2 hidden size 576")
        model_dir = Path(encoder_config["pretrained_audioencoder_path"]).resolve()
        validate_ced_assets(model_dir)
        with local_ced_config(model_dir) as config:
            model, loading = AutoModelForAudioClassification.from_pretrained(
                str(model_dir), config=config, trust_remote_code=True, local_files_only=True,
                torch_dtype=torch.float32, use_safetensors=True, output_loading_info=True,
            )
        problems = {key: loading[key] for key in
                    ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
        if problems:
            raise ValueError(f"CED pretrained weights did not load completely: {problems}")
        self.ced = model.encoder
        if not hasattr(self.ced, "blocks") or len(self.ced.blocks) != 12:
            raise ValueError("four-depth CED extraction requires exactly 12 encoder blocks")
        self.config = self.ced.config
        self.geometry = validate_ced_config(self.config.to_dict())
        self.hidden_dim, self.freq_bins, self.time_bins_per_split = 384, 4, 63
        if self.ced.maximal_allowed_length != self.config.target_length:
            raise ValueError("CED implementation/config split-length mismatch")
        self.resample = torchaudio.transforms.Resample(32000, 16000)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, win_length=int(self.config.win_size), center=bool(self.config.center),
            n_fft=int(self.config.n_fft), f_min=float(self.config.f_min), f_max=float(self.config.f_max),
            hop_length=int(self.config.hop_size), n_mels=int(self.config.n_mels),
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=120)
        self.mapper = CEDTemporalMapper(384, 4, d_out, output_tokens=AUDIO_TOKENS,
                                        resample=False, dropout=0.05)
        self.mapper_type = "freq_merge_126"
        for param in self.ced.parameters():
            param.requires_grad = False
        self.ced.eval()
        self.last_audio_layout = None

    def train(self, mode=True):
        super().train(mode)
        # The official cache builder extracts frozen features in eval mode.
        # Mapper dropout and gradients still follow the requested training mode.
        self.ced.eval()
        return self

    def _waveform_to_mel(self, waveform):
        if waveform.dim() != 2 or waveform.shape[1] != 640000:
            raise ValueError(f"CED expects whole fixed20 [B,640000] at 32kHz, got {tuple(waveform.shape)}")
        waveform = self.resample(waveform.float())
        mel = self.mel_spectrogram(waveform)
        # [B,1,mel,time] gives independent top_db cutoffs; [B,mel,time]
        # would interpret B as channels and share a cutoff across recordings.
        return self.amplitude_to_db(mel.unsqueeze(1)).squeeze(1)

    def _prepare_ced_input(self, input_values):
        x = torch.unsqueeze(input_values, 1)
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.ced.init_bn(x)
        return torch.permute(x, (0, 2, 1, 3))

    def _forward_hidden(self, mel):
        if mel.shape[1:] != (64, 2001):
            raise ValueError(f"unexpected CED fixed20 log-mel shape: {tuple(mel.shape)}")
        x = self._prepare_ced_input(mel)
        splits = list(x.split(self.ced.maximal_allowed_length, -1))
        if len(splits) != 2:
            raise ValueError("fixed20 CED must produce exactly two spectrogram segments")
        if splits[-1].shape[-1] < self.ced.maximal_allowed_length:
            pad = x.new_zeros(*x.shape[:-1], self.ced.maximal_allowed_length)
            pad[..., :splits[-1].shape[-1]] = splits[-1]
            splits[-1] = pad
        stacked = torch.stack(splits, dim=1)
        bsz, n_splits = stacked.shape[:2]
        flat = stacked.reshape(bsz * n_splits, *stacked.shape[2:])
        # Observe the original forward, including its final norm, without replacing
        # vendor code or modifying any CED block output. Hooks never return tensors.
        selected = (2, 5, 8, 11)
        captured = {}
        handles = []
        def capture(index):
            def hook(_module, _inputs, output):
                if index in captured:
                    raise RuntimeError(f"CED block {index + 1} executed more than once")
                if not isinstance(output, torch.Tensor) or output.shape != (bsz * 2, 252, 384):
                    raise ValueError(f"unexpected CED block {index + 1} output geometry")
                captured[index] = output
            return hook
        try:
            for index in selected:
                handles.append(self.ced.blocks[index].register_forward_hook(capture(index)))
            hidden = self.ced.forward_features(flat)
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(selected):
            raise RuntimeError("CED did not execute all selected blocks")
        block_hidden = [captured[index].reshape(bsz, 504, 384) for index in selected]
        if hidden.shape != (bsz * 2, 252, 384):
            raise ValueError(f"CED forward_features must return [2B,252,384], got {tuple(hidden.shape)}")
        return hidden.reshape(bsz, 504, 384), block_hidden

    def _hidden_to_temporal_grid(self, hidden):
        if hidden.ndim != 3 or hidden.shape[1:] != (504, 384):
            raise ValueError(f"CED hidden must be [B,504,384], got {tuple(hidden.shape)}")
        # Preserve [split,frequency,time] before moving frequency behind time.
        # Directly reshaping concatenated hidden to [B,4,126,384] is incorrect.
        grid = hidden.reshape(hidden.shape[0], 2, 4, 63, 384)
        return grid.permute(0, 1, 3, 2, 4).reshape(hidden.shape[0], 126, 4, 384)

    def forward(self, waveform):
        with torch.no_grad():
            mel = self._waveform_to_mel(waveform)
            hidden, block_hidden = self._forward_hidden(mel)
            grid = self._hidden_to_temporal_grid(hidden)
            layer_grids = torch.stack([
                self._hidden_to_temporal_grid(value).reshape(waveform.shape[0], 126, 1536)
                for value in block_hidden], dim=1)
        embedding = self.mapper(grid)
        if embedding.shape != (waveform.shape[0], AUDIO_TOKENS, 576):
            raise ValueError(f"unexpected CED mapper output: {tuple(embedding.shape)}")
        self.last_audio_layout = {"waveform_samples": 640000, "mel_frames": int(mel.shape[-1]),
                                  "hidden": list(hidden.shape), "grid": list(grid.shape),
                                  "embedding": list(embedding.shape), "dtype": str(embedding.dtype),
                                  "mapper": self.mapper_type, "encoder_eval": not self.ced.training,
                                  "tap_layers": [3, 6, 9, 12], "tap_norm": "pre_final_norm",
                                  "layer_grids": list(layer_grids.shape)}
        return {"embedding": embedding, "layer_grids": layer_grids,
                "clipwise_output": embedding.new_zeros(embedding.shape[0], 527)}
