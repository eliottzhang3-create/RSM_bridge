"""Pinned Mellow GradNormTracker behavior from commit c8204d8."""
from __future__ import annotations

import math
from typing import Any, Iterable

import torch

MELLOW_REFERENCE_COMMIT = "c8204d8eb99b4384fd7a76ad57995731e0c0c2bf"
MELLOW_GRAD_NORM_SOURCE = "utils/utils.py::GradNormTracker"


class GradNormTracker:
    """Track per-parameter gradient norms and apply Mellow's shared scaling."""

    def __init__(
        self,
        initial_l2_norm: float,
        initial_max_norm: float,
        *,
        overdrive_factor: float = 2.5,
        momentum: float = 0.995,
    ) -> None:
        if initial_l2_norm <= 0 or initial_max_norm <= 0:
            raise ValueError("initial gradient norms must be positive")
        if overdrive_factor <= 1 or not 0 < momentum < 1:
            raise ValueError("invalid GradNormTracker overdrive/momentum")
        self.initial_norm = (float(initial_l2_norm), float(initial_max_norm))
        self.overdrive_factor = float(overdrive_factor)
        self.momentum = float(momentum)
        self.running_norm: dict[str, tuple[float, float]] = {}
        self.history: dict[str, list[tuple[float, float]]] = {}

    def track_and_clip_(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    ) -> tuple[float, float]:
        parameters = list(named_parameters)
        min_scale = 1.0
        total_l2_squared = 0.0

        for name, parameter in parameters:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().reshape(-1)
            l2_norm = float(torch.linalg.vector_norm(gradient, ord=2).item())
            max_norm = float(torch.linalg.vector_norm(gradient, ord=float("inf")).item())
            if not math.isfinite(l2_norm) or not math.isfinite(max_norm):
                raise FloatingPointError(f"nonfinite gradient norm for {name}")
            total_l2_squared += l2_norm * l2_norm

            running_l2, running_max = self.running_norm.get(name, self.initial_norm)
            l2_limit = running_l2 * self.overdrive_factor
            max_limit = running_max * self.overdrive_factor
            if l2_norm > l2_limit and l2_norm > 0:
                min_scale = min(min_scale, l2_limit / l2_norm)
            if max_norm > max_limit and max_norm > 0:
                min_scale = min(min_scale, max_limit / max_norm)

            bounded_l2 = min(l2_norm, l2_limit * self.overdrive_factor)
            bounded_max = min(max_norm, max_limit * self.overdrive_factor)
            updated_l2 = self.momentum * running_l2 + (1.0 - self.momentum) * bounded_l2
            updated_max = self.momentum * running_max + (1.0 - self.momentum) * bounded_max
            self.running_norm[name] = (updated_l2, updated_max)
            self.history.setdefault(name, []).append((l2_norm, max_norm))

        if min_scale < 1.0:
            for _, parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(min_scale)
        return math.sqrt(total_l2_squared), min_scale

    def state_dict(self) -> dict[str, tuple[float, float]]:
        return {name: tuple(values) for name, values in self.running_norm.items()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        restored: dict[str, tuple[float, float]] = {}
        for name, values in state.items():
            if not isinstance(name, str) or not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ValueError("invalid GradNormTracker state")
            l2_norm, max_norm = float(values[0]), float(values[1])
            if l2_norm <= 0 or max_norm <= 0 or not math.isfinite(l2_norm + max_norm):
                raise ValueError(f"invalid GradNormTracker norms for {name}")
            restored[name] = (l2_norm, max_norm)
        self.running_norm = restored

    def truncate_history(self) -> None:
        self.history = {}

    def contract(self) -> dict[str, Any]:
        return {
            "source_commit": MELLOW_REFERENCE_COMMIT,
            "source_symbol": MELLOW_GRAD_NORM_SOURCE,
            "initial_l2_norm": self.initial_norm[0],
            "initial_max_norm": self.initial_norm[1],
            "overdrive_factor": self.overdrive_factor,
            "momentum": self.momentum,
            "running_parameter_count": len(self.running_norm),
        }


__all__ = ["GradNormTracker", "MELLOW_GRAD_NORM_SOURCE", "MELLOW_REFERENCE_COMMIT"]
