"""Dependency-free torch.profiler schedule arithmetic for PERF20.

The training entry point uses one-indexed optimizer steps for reports while
``torch.profiler.schedule`` consumes the same sequence through ``step()``.
Keeping the arithmetic here makes the boundary testable without importing
torch or constructing a distributed process group.
"""
from __future__ import annotations


def _validate(*, skip_first: int, wait: int, warmup: int, active: int, repeat: int, max_steps: int) -> None:
    values = (skip_first, wait, warmup, active, repeat, max_steps)
    if any(int(value) < 0 for value in values):
        raise ValueError("schedule values and max_steps must be non-negative")
    if int(active) <= 0 or int(repeat) <= 0:
        raise ValueError("active and repeat must be positive")


def active_steps(*, skip_first: int, wait: int, warmup: int, active: int, repeat: int, max_steps: int) -> list[int]:
    """Return one-indexed optimizer steps in profiler ``active`` windows."""
    _validate(skip_first=skip_first, wait=wait, warmup=warmup, active=active, repeat=repeat, max_steps=max_steps)
    step = int(skip_first) + 1
    result: list[int] = []
    for _ in range(int(repeat)):
        step += int(wait) + int(warmup)
        result.extend(range(step, min(int(max_steps), step + int(active) - 1) + 1))
        step += int(active)
    return result


def affected_steps(*, skip_first: int, wait: int, warmup: int, active: int, repeat: int, max_steps: int) -> list[int]:
    """Return one-indexed wait/warmup/active steps affected by profiling."""
    _validate(skip_first=skip_first, wait=wait, warmup=warmup, active=active, repeat=repeat, max_steps=max_steps)
    step = int(skip_first) + 1
    result: list[int] = []
    for _ in range(int(repeat)):
        result.extend(range(step, min(int(max_steps), step + int(wait) - 1) + 1))
        step += int(wait)
        result.extend(range(step, min(int(max_steps), step + int(warmup) - 1) + 1))
        step += int(warmup)
        result.extend(range(step, min(int(max_steps), step + int(active) - 1) + 1))
        step += int(active)
    return sorted(set(result))
