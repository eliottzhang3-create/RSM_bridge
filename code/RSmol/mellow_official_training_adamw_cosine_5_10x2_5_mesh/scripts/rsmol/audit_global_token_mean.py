#!/usr/bin/env python3
"""CPU arithmetic audit for the global-token-mean DDP reduction contract."""

from __future__ import annotations

import torch


def main() -> int:
    # Unequal token counts make the old mean-of-means visibly different.
    microbatch_sums = torch.tensor([2.0, 6.0], dtype=torch.float32)
    microbatch_counts = torch.tensor([1.0, 3.0], dtype=torch.float32)
    token_mean = microbatch_sums.sum() / microbatch_counts.sum()
    mean_of_means = (microbatch_sums / microbatch_counts).mean()
    if not torch.allclose(token_mean, torch.tensor(2.0)):
        raise AssertionError(f"unexpected token mean: {token_mean}")
    if torch.allclose(token_mean, mean_of_means):
        raise AssertionError("audit case did not distinguish token mean from mean-of-means")

    # DDP averages rank gradients.  The trainer's world-size factor must
    # cancel that averaging so the result remains global CE/global tokens.
    rank_sums = torch.tensor([3.0, 5.0, 7.0, 11.0], dtype=torch.float32)
    rank_counts = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
    world_size = rank_sums.numel()
    expected = rank_sums.sum() / rank_counts.sum()
    ddp_result = (rank_sums * world_size / rank_counts.sum()).mean()
    if not torch.allclose(expected, ddp_result):
        raise AssertionError(f"DDP scaling mismatch: expected={expected} got={ddp_result}")
    print(
        "global_token_mean audit PASS: "
        f"token_mean={float(token_mean):.6f} ddp_mean={float(ddp_result):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
