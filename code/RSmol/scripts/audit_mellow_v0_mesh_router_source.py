#!/usr/bin/env python3
"""Read-only CPU audit for epoch-17 router transfer into Mellow-v0 MeSH."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch

SOURCE = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_official_reasonaqa_adamw_cosine_5_10x2_5_mesh_5090/formal_30epochs_20261005_175504/checkpoints/mellow_adamw_cosine_reasonaqa_mesh_formal_20_20261005_095511181778133/model--epo-17.ckpt")
BASE = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/converted/mellow_v0_5_10x2_5_mesh_initialization/mellow_v0_5_10x2_5_mesh_init.pt")
REPORT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0/preflight/epoch17_router_source_audit.json")
CONTRACT = "mellow_v0_mesh_epoch17_router_source_audit_v1"
ROUTE = "mellow_official_adamw_cosine_5_10x2_5_mesh_v1"
TEXT = "logical_30_physical_20_5_10x2_5"
KEY_PATTERN = re.compile(r"^caption_decoder\.lm\.model\.(write_routers|read_routers)\.([012])\.(weight|bias)$")
ROUTER_KEYS = {
    f"caption_decoder.lm.model.{direction}.{index}.{suffix}"
    for direction in ("write_routers", "read_routers")
    for index in range(3)
    for suffix in ("weight", "bias")
}


def load(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def router_keys(state: Mapping[str, Any]) -> set[str]:
    return {key for key in state if "write_routers." in key or "read_routers." in key}


def audit(source_path: Path, base_path: Path) -> dict[str, Any]:
    source_path = source_path.expanduser().resolve(strict=True)
    base_path = base_path.expanduser().resolve(strict=True)
    if source_path == base_path or source_path.name != "model--epo-17.ckpt":
        raise ValueError("source must be a distinct epoch-17 full checkpoint")
    if not source_path.is_file() or not base_path.is_file():
        raise ValueError("checkpoint paths must be regular files")
    source = load(source_path)
    if source.get("schema_version") != 2 or source.get("route_contract") != ROUTE:
        raise ValueError("source schema or route contract mismatch")
    required = {"state_dict", "optimizer", "scheduler", "random_state_by_rank", "epoch_completed", "total_step", "num_epochs"}
    missing = sorted(required.difference(source))
    if missing:
        raise ValueError(f"source is missing full-checkpoint fields: {missing}")
    if source.get("text_model_contract") != TEXT:
        raise ValueError("source text contract mismatch")
    if int(source.get("epoch_completed", -1)) != 17 or int(source.get("num_epochs", -1)) != 30:
        raise ValueError("source must be epoch 17 of a 30-epoch run")
    step = int(source.get("total_step", -1))
    scheduler = source.get("scheduler")
    if step <= 0 or not isinstance(scheduler, Mapping) or int(scheduler.get("last_step", -1)) != step:
        raise ValueError("source scheduler and training step disagree")
    src_state = source.get("state_dict")
    if not isinstance(src_state, Mapping) or not src_state:
        raise ValueError("source state_dict missing")
    base = load(base_path)
    metadata = base.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("artifact_contract") != "mellow_v0_to_5_10x2_5_mesh_initialization_v1":
        raise ValueError("baseline initialization contract mismatch")
    if metadata.get("text_model_contract") != TEXT or int(metadata.get("memory_slots", -1)) != 5:
        raise ValueError("baseline text or memory-slot contract mismatch")
    dst_state = base.get("state_dict")
    if not isinstance(dst_state, Mapping) or not dst_state:
        raise ValueError("baseline state_dict missing")
    if set(src_state) != set(dst_state):
        raise ValueError(f"state keys differ: source_only={sorted(set(src_state)-set(dst_state))[:8]}, target_only={sorted(set(dst_state)-set(src_state))[:8]}")
    if router_keys(src_state) != ROUTER_KEYS or router_keys(dst_state) != ROUTER_KEYS:
        raise ValueError("router keys must be exactly 3 write/read pairs, 12 tensors")
    for key in src_state:
        left, right = src_state[key], dst_state[key]
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise ValueError(f"non-tensor state entry: {key}")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"source/target shape or dtype mismatch: {key}")
    routers = {}
    for key in sorted(ROUTER_KEYS):
        match = KEY_PATTERN.fullmatch(key)
        assert match is not None
        value = src_state[key].detach().cpu()
        shape = (5, 576) if match.group(3) == "weight" else (5,)
        if tuple(value.shape) != shape or value.dtype != torch.float32:
            raise ValueError(f"router geometry or dtype mismatch: {key}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"non-finite source router: {key}")
        routers[key] = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha256_tensor(value), "equal_to_baseline": bool(torch.equal(value, dst_state[key])), "min": float(value.min()), "max": float(value.max())}
    different = sum(not info["equal_to_baseline"] for info in routers.values())
    return {"status": "PASS", "artifact_contract": CONTRACT, "cuda_used": False, "source_checkpoint": str(source_path), "source_sha256": sha256_file(source_path), "baseline_initialization": str(base_path), "baseline_sha256": sha256_file(base_path), "epoch_completed": 17, "num_epochs": 30, "total_step": step, "state_tensor_count": len(src_state), "router_tensor_count": len(routers), "routers_different_from_baseline": different, "router_objects": 6, "router_groups": 3, "memory_slots": 5, "router_tensors": routers, "conversion_scope": "copy only these 12 router tensors", "memory_slot_zero_contract": "runtime sets slot 0 to raw input embeddings on every forward"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE)
    parser.add_argument("--baseline-init", type=Path, default=BASE)
    parser.add_argument("--report-path", type=Path, default=REPORT)
    args = parser.parse_args()
    try:
        result = audit(args.source_checkpoint, args.baseline_init)
    except Exception as exc:
        result = {"status": "FAILED", "artifact_contract": CONTRACT, "error": repr(exc), "traceback": traceback.format_exc(), "cuda_used": False}
    report_path = args.report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    fields = ("status", "artifact_contract", "error", "source_checkpoint", "source_sha256", "baseline_initialization", "baseline_sha256", "epoch_completed", "num_epochs", "total_step", "state_tensor_count", "router_tensor_count", "routers_different_from_baseline", "router_objects", "router_groups", "memory_slots", "cuda_used")
    summary = {key: result[key] for key in fields if key in result}
    summary["report_path"] = str(report_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
