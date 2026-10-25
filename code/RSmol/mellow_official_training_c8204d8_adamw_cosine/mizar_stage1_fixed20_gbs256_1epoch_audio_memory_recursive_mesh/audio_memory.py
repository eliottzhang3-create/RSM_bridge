"""Stage-2 audio memory and two single-layer attention reads into MeSH."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def ungroup_stage2(tokens: torch.Tensor) -> torch.Tensor:
    """Undo HTSAT's freq_ratio=4 image packing: [B,1024,192] -> [B,8,128,192].

    The packed 32x32 height consists of four consecutive 8-row frequency
    bands. Those bands represent consecutive time chunks, not 32 independent
    frequency bins. This is the same inverse packing as HTSAT.forward_features.
    """
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (1024, 192):
        raise ValueError(f"expected HTSAT stage2 [B,1024,192], got {tuple(tokens.shape)}")
    batch = tokens.shape[0]
    image = tokens.reshape(batch, 32, 32, 192).permute(0, 3, 1, 2)
    image = image.reshape(batch, 192, 4, 8, 32)
    grid = image.permute(0, 1, 3, 2, 4).reshape(batch, 192, 8, 128)
    return grid.permute(0, 2, 3, 1).contiguous()


def temporal_encoding(*, device, dtype, width=576) -> torch.Tensor:
    """20 s / 256 bins, centered at 0.0390625 ... 19.9609375 s.

    As in the earlier route, sinusoids use time/bin_seconds as their position
    ID; IDs therefore run from 0.5 through 255.5, repeated for all 8 bands.
    """
    time_seconds = (torch.arange(256, device=device, dtype=torch.float32) + 0.5) * (20.0 / 256)
    time_ids = time_seconds / (20.0 / 256)
    scales = torch.exp(torch.arange(width // 2, device=device, dtype=torch.float32)
                       * (-math.log(10000.0) / max(width // 2 - 1, 1)))
    angles = time_ids[:, None] * scales[None, :]
    encoding = torch.stack((angles.sin(), angles.cos()), dim=-1).reshape(256, width)
    return encoding.to(dtype=dtype).view(1, 1, 256, width)


class AudioMemoryBuilder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage2_norm = nn.LayerNorm(192)
        self.stage2_proj = nn.Linear(192, 576)
        self.frequency_embedding = nn.Parameter(torch.zeros(8, 576))
        self.last_layout = {}

    def reset_parameters(self):
        nn.init.ones_(self.stage2_norm.weight)
        nn.init.zeros_(self.stage2_norm.bias)
        nn.init.xavier_uniform_(self.stage2_proj.weight)
        nn.init.zeros_(self.stage2_proj.bias)
        nn.init.zeros_(self.frequency_embedding)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape:
            raise ValueError("the two stage2 halves must have identical shapes")
        # Concatenate chronological halves BEFORE flattening frequency/time.
        grid = torch.cat((ungroup_stage2(first), ungroup_stage2(second)), dim=2)
        grid = grid.to(device=self.stage2_proj.weight.device, dtype=self.stage2_proj.weight.dtype)
        features = self.stage2_proj(self.stage2_norm(grid))
        features = (features + temporal_encoding(device=features.device, dtype=features.dtype)
                    + self.frequency_embedding.view(1, 8, 1, 576))
        memory = features.reshape(features.shape[0], 2048, 576)
        self.last_layout = {"halves": 2, "stage2_per_half": list(first.shape[1:]),
                            "grid": [8, 256], "memory": list(memory.shape),
                            "flatten_order": "frequency_major_time_fastest",
                            "time_step_seconds": 20.0 / 256,
                            "frequency_embedding": list(self.frequency_embedding.shape)}
        return memory


class AudioAttention(nn.Module):
    """Two Q/O sets, one shared K/V set; no residual/FFN outside MeSH memory."""
    def __init__(self):
        super().__init__()
        self.heads, self.head_dim = 9, 64
        self.q_norms = nn.ModuleList([nn.LayerNorm(576) for _ in range(2)])
        self.q_projs = nn.ModuleList([nn.Linear(576, 576) for _ in range(2)])
        self.memory_norm = nn.LayerNorm(576)
        self.k_proj = nn.Linear(576, 576)
        self.v_proj = nn.Linear(576, 576)
        self.o_projs = nn.ModuleList([nn.Linear(576, 576) for _ in range(2)])

    def reset_parameters(self):
        for norm in list(self.q_norms) + [self.memory_norm]:
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        for proj in list(self.q_projs) + [self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        for proj in self.o_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def _heads(self, tensor):
        return tensor.reshape(tensor.shape[0], tensor.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def prepare_kv(self, memory):
        memory = self.memory_norm(memory)
        return self._heads(self.k_proj(memory)), self._heads(self.v_proj(memory))

    def forward(self, query, kv, round_index):
        q = self._heads(self.q_projs[round_index](self.q_norms[round_index](query)))
        # Audio is conditioning, so every query may see the complete recording.
        attended = F.scaled_dot_product_attention(q, kv[0], kv[1], dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).contiguous().reshape(query.shape[0], query.shape[1], 576)
        return self.o_projs[round_index](attended)
