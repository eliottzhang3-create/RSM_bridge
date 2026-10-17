\"\"\"Audio memory builder and gated Perceiver readers for the isolated route.\"\"\"

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def _fixed_temporal_encoding(tokens: torch.Tensor, *, seconds: float = 10.0, bin_seconds: float = 0.08) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"audio memory tokens must be [B,N,C], got {tuple(tokens.shape)}")
    batch, count, width = tokens.shape
    side = int(math.isqrt(count))
    if side * side != count:
        raise ValueError(f"expected square HTSAT grid, got N={count}")
    # HTSAT flattens [frequency,time] with time as the fastest dimension.
    time_index = torch.arange(count, device=tokens.device, dtype=tokens.dtype) % side
    time_seconds = (time_index + 0.5) / float(side) * float(seconds)
    temporal_id = time_seconds / float(bin_seconds)
    half = width // 2
    if half == 0:
        return tokens
    frequency = torch.exp(
        torch.arange(half, device=tokens.device, dtype=tokens.dtype)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = temporal_id[:, None] * frequency[None, :]
    encoding = torch.zeros((count, width), device=tokens.device, dtype=tokens.dtype)
    encoding[:, 0 : 2 * (half // 2) : 2] = torch.sin(angles[:, : half // 2])
    encoding[:, 1 : 2 * (half // 2) : 2] = torch.cos(angles[:, : half // 2])
    if width % 2:
        encoding[:, -1] = torch.sin(temporal_id)
    return encoding.unsqueeze(0).expand(batch, -1, -1)


class AudioMemoryBuilder(nn.Module):
    """Normalize/project the two frozen HTSAT stages independently."""

    def __init__(self, stage2_dim: int = 192, stage3_dim: int = 384, hidden_size: int = 576) -> None:
        super().__init__()
        self.stage2_norm = nn.LayerNorm(stage2_dim)
        self.stage3_norm = nn.LayerNorm(stage3_dim)
        self.stage2_proj = nn.Linear(stage2_dim, hidden_size)
        self.stage3_proj = nn.Linear(stage3_dim, hidden_size)
        self.hidden_size = int(hidden_size)
        self.stage2_dim = int(stage2_dim)
        self.stage3_dim = int(stage3_dim)

    def forward(self, stage2: torch.Tensor, stage3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if stage2.ndim != 3 or stage3.ndim != 3:
            raise ValueError("Stage 2/3 memory inputs must be rank-3 token tensors")
        if stage2.shape[0] != stage3.shape[0]:
            raise ValueError("Stage 2/3 memory batch sizes differ")
        if stage2.shape[-1] != self.stage2_dim or stage3.shape[-1] != self.stage3_dim:
            raise ValueError(
                f"unexpected HTSAT channels: stage2={stage2.shape[-1]}, stage3={stage3.shape[-1]}"
            )
        m2 = self.stage2_proj(self.stage2_norm(stage2))
        m3 = self.stage3_proj(self.stage3_norm(stage3))
        m2 = m2 + _fixed_temporal_encoding(m2)
        m3 = m3 + _fixed_temporal_encoding(m3)
        return m2, m3


class Perceiver(nn.Module):
    """One recurrent-loop audio reader with independent Stage 2/3 banks."""

    def __init__(self, hidden_size: int = 576, num_heads: int = 8) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.q_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k2_proj = nn.Linear(hidden_size, hidden_size)
        self.v2_proj = nn.Linear(hidden_size, hidden_size)
        self.k3_proj = nn.Linear(hidden_size, hidden_size)
        self.v3_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.alpha_stage2 = nn.Parameter(torch.zeros(()))
        self.alpha_stage3 = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.out_proj.bias)

    def _attend(self, query: torch.Tensor, memory: torch.Tensor, key: nn.Linear, value: nn.Linear) -> torch.Tensor:
        batch, length, _ = query.shape
        heads = self.num_heads
        head_dim = self.hidden_size // heads
        q = self.q_proj(self.q_norm(query)).view(batch, length, heads, head_dim).transpose(1, 2)
        k = key(memory).view(batch, memory.shape[1], heads, head_dim).transpose(1, 2)
        v = value(memory).view(batch, memory.shape[1], heads, head_dim).transpose(1, 2)
        attended = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return attended.transpose(1, 2).reshape(batch, length, self.hidden_size)

    def forward(self, hidden: torch.Tensor, memory2: torch.Tensor, memory3: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        c2 = self.out_proj(self._attend(hidden, memory2, self.k2_proj, self.v2_proj))
        c3 = self.out_proj(self._attend(hidden, memory3, self.k3_proj, self.v3_proj))
        gate2 = torch.tanh(self.alpha_stage2)
        gate3 = torch.tanh(self.alpha_stage3)
        result = hidden + gate2 * c2 + gate3 * c3
        trace = {
            "stage2_gate": gate2.detach(),
            "stage3_gate": gate3.detach(),
            "stage2_shape": tuple(memory2.shape),
            "stage3_shape": tuple(memory3.shape),
        }
        return result, trace


__all__ = ["AudioMemoryBuilder", "Perceiver"]
