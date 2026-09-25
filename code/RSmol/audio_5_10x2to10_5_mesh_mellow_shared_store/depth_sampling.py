"""Deterministic rank-synchronized recursive-depth sampling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from recursive_model_5_10x2to10_5_mesh import MAX_RECURSIVE_DEPTH, MIN_RECURSIVE_DEPTH


@dataclass
class SynchronizedDepthSampler:
    """Sample on rank zero once per micro-step and broadcast the integer T."""

    seed: int
    minimum: int = MIN_RECURSIVE_DEPTH
    maximum: int = MAX_RECURSIVE_DEPTH
    draws: int = 0
    histogram: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (int(self.minimum), int(self.maximum)) != (MIN_RECURSIVE_DEPTH, MAX_RECURSIVE_DEPTH):
            raise ValueError("the formal route samples exactly Uniform{2,...,10}")
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(self.seed))
        if not self.histogram:
            self.histogram = {depth: 0 for depth in range(self.minimum, self.maximum + 1)}

    def sample(self, device: torch.device | str) -> int:
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        if rank == 0:
            value = int(torch.randint(
                self.minimum, self.maximum + 1, (1,), generator=self.generator, device="cpu"
            ).item())
        else:
            value = 0
        depth = torch.tensor([value], dtype=torch.int64, device=device)
        if distributed:
            dist.broadcast(depth, src=0)
        value = int(depth.item())
        self.draws += 1
        self.histogram[value] = self.histogram.get(value, 0) + 1
        return value

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed), "minimum": int(self.minimum),
            "maximum": int(self.maximum), "draws": int(self.draws),
            "histogram": {str(k): int(v) for k, v in sorted(self.histogram.items())},
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = (int(self.seed), int(self.minimum), int(self.maximum))
        actual = (int(state["seed"]), int(state["minimum"]), int(state["maximum"]))
        if actual != expected:
            raise ValueError("recursive-depth sampler resume contract mismatch")
        self.draws = int(state["draws"])
        self.histogram = {int(k): int(v) for k, v in state["histogram"].items()}
        self.generator.set_state(state["generator_state"])
