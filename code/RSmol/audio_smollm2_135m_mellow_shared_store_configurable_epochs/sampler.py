"""Official-Mellow-style epoch permutation with contiguous per-rank slices."""
from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class ContiguousDistributedEpochSampler(Sampler[int]):
    """Use torch.randperm(seed=epoch), then give each rank one contiguous slice."""

    def __init__(
        self,
        dataset_size: int,
        *,
        num_replicas: int,
        rank: int,
        per_rank_batch_size: int,
        gradient_accumulation_steps: int,
        epoch: int,
        start_optimizer_step: int = 0,
    ) -> None:
        if dataset_size <= 0 or num_replicas <= 0:
            raise ValueError("dataset_size and num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank is outside the replica range")
        if per_rank_batch_size <= 0 or gradient_accumulation_steps <= 0:
            raise ValueError("batch size and accumulation must be positive")
        global_batch = num_replicas * per_rank_batch_size * gradient_accumulation_steps
        self.steps_per_epoch = dataset_size // global_batch
        if self.steps_per_epoch <= 0:
            raise ValueError("dataset is shorter than one effective global batch")
        if start_optimizer_step < 0 or start_optimizer_step > self.steps_per_epoch:
            raise ValueError("start_optimizer_step is outside this epoch")
        self.dataset_size = int(dataset_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.per_rank_batch_size = int(per_rank_batch_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.epoch = int(epoch)
        self.start_optimizer_step = int(start_optimizer_step)
        self.samples_per_rank = (
            self.steps_per_epoch * self.per_rank_batch_size * self.gradient_accumulation_steps
        )
        self.start_sample = (
            self.start_optimizer_step * self.per_rank_batch_size * self.gradient_accumulation_steps
        )

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.epoch)
        permutation = torch.randperm(self.dataset_size, generator=generator).tolist()
        rank_start = self.rank * self.samples_per_rank
        rank_end = rank_start + self.samples_per_rank
        rank_indices = permutation[rank_start:rank_end]
        if len(rank_indices) != self.samples_per_rank:
            raise RuntimeError("sampler failed to construct a complete contiguous rank slice")
        return iter(rank_indices[self.start_sample:])

    def __len__(self) -> int:
        return self.samples_per_rank - self.start_sample

    def audit(self) -> dict[str, int | str]:
        return {
            "kind": "torch_randperm_seed_epoch_contiguous_rank_slice",
            "epoch": self.epoch,
            "rank": self.rank,
            "dataset_size": self.dataset_size,
            "samples_per_rank": self.samples_per_rank,
            "start_optimizer_step": self.start_optimizer_step,
            "start_sample_within_rank": self.start_sample,
            "remaining_samples": len(self),
        }
