#!/usr/bin/env python3
"""Audit a full checkpoint produced by the isolated two-stage route."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


ROUTE = "mellow_v0_official_adamw_cosine_two_stage_5_10x2_5_mesh_v1"
TEXT = "logical_30_physical_20_5_10x2_5"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--expected-epochs", type=int, default=None)
    parser.add_argument("--expected-total-step", type=int, default=None)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required = {
        "schema_version", "state_dict", "optimizer", "optimizer_contract", "scheduler",
        "grad_scaler", "grad_norm_tracker", "loss_tracker", "epoch_completed", "total_step",
        "num_epochs", "batch_geometry", "loss_reduction", "route_contract",
        "text_model_contract", "training_stage", "optimizer_group_contract",
        "trainability_contract", "random_state_by_rank",
    }
    missing = sorted(required.difference(checkpoint))
    if checkpoint.get("schema_version") != 2 or missing:
        raise SystemExit(f"invalid checkpoint schema={checkpoint.get('schema_version')} missing={missing}")
    if checkpoint["route_contract"] != ROUTE or checkpoint["text_model_contract"] != TEXT:
        raise SystemExit("checkpoint belongs to another route or text model")
    if checkpoint["training_stage"] != args.stage:
        raise SystemExit(f"stage mismatch: checkpoint={checkpoint['training_stage']!r} expected={args.stage!r}")
    if checkpoint["loss_reduction"] != "global_token_mean":
        raise SystemExit("checkpoint does not declare global_token_mean")
    per_rank_batch_size, accumulation_steps = {"stage1": (8, 4), "stage2": (4, 1)}[args.stage]
    expected_geometry = {
        "per_rank_batch_size": per_rank_batch_size,
        "world_size": 8,
        "gradient_accumulation_steps": accumulation_steps,
    }
    if checkpoint["batch_geometry"] != expected_geometry:
        raise SystemExit(f"unexpected batch geometry: {checkpoint['batch_geometry']!r}")
    if len(checkpoint["random_state_by_rank"]) != 8:
        raise SystemExit("checkpoint does not contain RNG state for all ranks")
    trainability = checkpoint["trainability_contract"]
    if trainability.get("htsat_trainable") is not False:
        raise SystemExit("HTSAT backbone is not frozen in checkpoint contract")
    for key in ("c2l_trainable", "bridge_trainable", "text_trainable", "router_trainable"):
        if trainability.get(key) is not True:
            raise SystemExit(f"expected trainable component missing: {key}")
    if args.expected_epochs is not None and int(checkpoint["num_epochs"]) != args.expected_epochs:
        raise SystemExit(f"unexpected epoch horizon: {checkpoint['num_epochs']}")
    if args.require_complete and int(checkpoint["epoch_completed"]) != int(checkpoint["num_epochs"]):
        raise SystemExit(
            f"checkpoint is not complete: epoch_completed={checkpoint['epoch_completed']} "
            f"num_epochs={checkpoint['num_epochs']}"
        )
    if args.require_complete:
        nonfinite = [
            name for name, tensor in checkpoint["state_dict"].items()
            if isinstance(tensor, torch.Tensor)
            and (tensor.is_floating_point() or tensor.is_complex())
            and not torch.isfinite(tensor).all().item()
        ]
        if nonfinite:
            raise SystemExit(f"checkpoint has non-finite model tensors: {nonfinite[:8]}")
    if args.expected_total_step is not None and int(checkpoint["total_step"]) != args.expected_total_step:
        raise SystemExit(f"unexpected total_step: {checkpoint['total_step']}")
    scheduler = checkpoint["scheduler"]
    if scheduler.get("scheduler_type") != "step_cosine_warmup":
        raise SystemExit("unexpected scheduler type")
    if int(scheduler.get("last_step", -1)) != int(checkpoint["total_step"]):
        raise SystemExit("scheduler and checkpoint step disagree")
    if not (0 < int(scheduler.get("warmup_steps", 0)) <= int(scheduler.get("total_steps", 0))):
        raise SystemExit(f"invalid scheduler horizon: {scheduler!r}")
    if args.require_complete and int(checkpoint["total_step"]) != int(scheduler["total_steps"]):
        raise SystemExit("completed checkpoint step does not match scheduler horizon")
    expected_bounds = {
        "stage1": {"routers": (1e-3, 1e-4), "other": (1e-4, 1e-5)},
        "stage2": {"all": (5e-4, 5e-5)},
    }[args.stage]
    for bounds in scheduler.get("group_bounds", []):
        name = str(bounds.get("name"))
        if name not in expected_bounds:
            raise SystemExit(f"unexpected scheduler parameter group: {name}")
        expected_max, expected_min = expected_bounds[name]
        if not math.isclose(float(bounds.get("max_lr", -1)), expected_max, abs_tol=1e-12):
            raise SystemExit(f"wrong max LR for {name}: {bounds}")
        if not math.isclose(float(bounds.get("min_lr", -1)), expected_min, abs_tol=1e-12):
            raise SystemExit(f"wrong min LR for {name}: {bounds}")
    optimizer_groups = checkpoint["optimizer_contract"].get("parameter_groups", [])
    if not optimizer_groups or set(group.get("name") for group in optimizer_groups) != set(expected_bounds):
        raise SystemExit("optimizer parameter-group contract mismatch")
    if checkpoint["optimizer_group_contract"] != optimizer_groups:
        raise SystemExit("duplicated optimizer group contracts differ")
    print(f"PASS: {args.checkpoint} stage={args.stage} epochs={checkpoint['num_epochs']} step={checkpoint['total_step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
