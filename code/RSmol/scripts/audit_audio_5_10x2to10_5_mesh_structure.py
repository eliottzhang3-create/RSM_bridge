#!/usr/bin/env python3
"""Automatically audit all depths and representative backward paths."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers.models.llama.configuration_llama import LlamaConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recursive_model_5_10x2to10_5_mesh import (  # noqa: E402
    RecursiveLlamaForCausalLM, build_mesh_schedule, logical_layer_count,
)


def _tiny_config() -> LlamaConfig:
    config = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=110, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        attention_dropout=0.0, use_cache=False,
    )
    config.recursive_layer_count = 20
    config.recursive_min_depth = 2
    config.recursive_max_depth = 10
    config.recursive_default_depth = 2
    return config


def _expected_calls(depth: int) -> dict[str, int]:
    return {
        "pre_write": 1, "pre_read": 1, "loop1_write": 1, "loop1_read": 1,
        "refine_write": depth - 1, "refine_read": max(0, depth - 2), "out_read": 1,
    }


def _gradient_audit(model: Any, depth: int, input_ids: torch.Tensor) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    model.model.gradient_audit_mode = True
    output = model(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids.clone(),
        recursive_depth=depth, use_cache=False, return_dict=True,
    )
    output.loss.backward()
    required_routers = [
        "pre_write", "pre_read", "loop1_write", "loop1_read", "refine_write", "out_read",
    ]
    if depth > 2:
        required_routers.append("refine_read")
    router_grad = {
        name: bool(
            getattr(model.model, name).weight.grad is not None
            and torch.isfinite(getattr(model.model, name).weight.grad).all()
        )
        for name in required_routers
    }
    layer_grad = {
        str(index): bool(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in layer.parameters()
        ))
        for index, layer in enumerate(model.model.layers)
    }
    boundaries = model.model.last_core_input_refs + model.model.last_core_output_refs
    boundary_grad = bool(boundaries) and all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in boundaries
    )
    passed = all(router_grad.values()) and all(layer_grad.values()) and boundary_grad
    if not passed:
        raise RuntimeError(
            f"gradient audit failed at T={depth}: routers={router_grad} layers={layer_grad}"
        )
    return {
        "depth": depth, "loss": float(output.loss.detach()),
        "router_gradients": router_grad, "physical_layer_gradients": layer_grad,
        "all_loop_boundaries_have_gradients": boundary_grad,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(20260925)
    model = RecursiveLlamaForCausalLM(_tiny_config())
    model.eval()
    input_ids = torch.randint(3, 128, (2, 12))
    depth_reports = []
    with torch.no_grad():
        for depth in range(2, 11):
            output = model(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                recursive_depth=depth, use_cache=False, return_dict=True,
            )
            trace = [entry["physical_index"] for entry in model.model.last_forward_trace]
            expected_trace = list(build_mesh_schedule(depth))
            calls = dict(model.model.last_router_call_counts)
            expected_calls = _expected_calls(depth)
            passed = (
                trace == expected_trace and calls == expected_calls
                and len(trace) == logical_layer_count(depth)
                and tuple(output.logits.shape) == (2, 12, 128)
            )
            if not passed:
                raise RuntimeError(
                    f"structure audit failed at T={depth}: trace={trace} calls={calls}"
                )
            depth_reports.append({
                "depth": depth, "logical_layers": len(trace),
                "router_calls": calls, "status": "PASS",
            })
    model.train()
    gradients = [_gradient_audit(model, depth, input_ids) for depth in (2, 3, 6, 10)]
    artifact = None
    if args.init_artifact is not None:
        root = args.init_artifact.resolve(strict=True)
        marker = json.loads((root / "artifact_complete.json").read_text(encoding="utf-8"))
        migration = json.loads(
            (root / "variable_depth_init_report.json").read_text(encoding="utf-8")
        )
        if (
            marker.get("status") != "complete"
            or migration.get("t2_exact_parity", {}).get("status") != "PASS"
        ):
            raise RuntimeError("initialization artifact lacks completion/T2-parity proof")
        artifact = {
            "path": str(root), "marker": marker,
            "t2_exact_parity": migration["t2_exact_parity"],
        }
    report = {
        "status": "PASS", "all_depths": depth_reports,
        "gradient_sample_depths": gradients, "initialization_artifact": artifact,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-artifact", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
