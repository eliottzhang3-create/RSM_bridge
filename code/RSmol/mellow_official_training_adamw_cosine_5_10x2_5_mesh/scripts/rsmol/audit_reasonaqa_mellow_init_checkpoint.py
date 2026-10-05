#!/usr/bin/env python3
"""Audit a Mellow checkpoint for model-only MCQ fine-tuning initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint does not contain a non-empty state_dict")
    tensor_count = 0
    total_elements = 0
    state_sha = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"state_dict entry is not a named tensor: {name!r}")
        tensor_count += 1
        total_elements += value.numel()
        state_sha.update(name.encode("utf-8"))
        state_sha.update(str(value.dtype).encode("ascii"))
        state_sha.update(repr(tuple(value.shape)).encode("ascii"))
    full_fields = {
        "optimizer", "optimizer_contract", "scheduler", "grad_scaler",
        "grad_norm_tracker", "loss_tracker", "epoch_completed", "total_step",
        "num_epochs", "batch_geometry", "random_state_by_rank",
    }
    missing_full_fields = sorted(full_fields.difference(checkpoint))
    report = {
        "status": "PASS",
        "contract": "mellow_model_initialization_checkpoint_audit_v1",
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_bytes": path.stat().st_size,
        "schema_version": checkpoint.get("schema_version"),
        "is_schema_v2_full_checkpoint": checkpoint.get("schema_version") == 2 and not missing_full_fields,
        "missing_full_checkpoint_fields": missing_full_fields,
        "state_dict_tensor_count": tensor_count,
        "state_dict_total_elements": total_elements,
        "state_dict_structural_sha256": state_sha.hexdigest(),
        "epoch_completed": checkpoint.get("epoch_completed"),
        "total_step": checkpoint.get("total_step"),
        "num_epochs": checkpoint.get("num_epochs"),
        "optimizer_contract": checkpoint.get("optimizer_contract"),
        "scheduler": checkpoint.get("scheduler"),
        "batch_geometry": checkpoint.get("batch_geometry"),
        "initialization_semantics": "load state_dict only; create fresh optimizer/scheduler/RNG for MCQ fine-tuning",
        "gpu_required": False,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint is not a regular file: {checkpoint}")
    report = audit(checkpoint)
    output = args.output_report.expanduser().resolve() if args.output_report else checkpoint.with_suffix(checkpoint.suffix + ".audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
