#!/usr/bin/env python3
"""Model-free eight-GPU NCCL transport audit for the 5_10x4_5 text route."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("baseline", "p2p_off", "p2p_cumem_off"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--file-rendezvous-seconds", type=int, default=120)
    parser.add_argument("--broadcast-mib", type=int, default=64)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_phase(output_dir: Path, rank: int, name: str, **detail: Any) -> None:
    atomic_json(output_dir / "phases" / f"rank{rank}.json", {
        "rank": rank,
        "phase": name,
        "timestamp": time.time(),
        "detail": detail,
    })
    print(f"[nccl-transport][rank={rank}] phase={name} detail={detail}", flush=True)


def wait_for(paths: list[Path], timeout_seconds: int, description: str) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        missing = [path for path in paths if not path.is_file()]
        if not missing:
            return
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {description}: {[str(path) for path in missing]}")


def device_record(rank: int, local_rank: int, device: torch.device, profile: str) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    nccl_version = torch.cuda.nccl.version() if torch.cuda.nccl.is_available([]) else None
    return {
        "rank": rank,
        "local_rank": local_rank,
        "profile": profile,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": properties.name,
        "device_uuid": str(getattr(properties, "uuid", "")),
        "device_properties": str(properties),
        "compute_capability": [properties.major, properties.minor],
        "total_memory": properties.total_memory,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nccl_version": nccl_version,
        "environment": {name: os.environ.get(name) for name in (
            "NCCL_P2P_DISABLE",
            "NCCL_CUMEM_ENABLE",
            "NCCL_SHM_DISABLE",
            "NCCL_IB_DISABLE",
            "NCCL_DEBUG",
            "NCCL_DEBUG_SUBSYS",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        )},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
    if world_size != args.world_size:
        raise RuntimeError(f"expected world_size={args.world_size}, got {world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    initialized = False

    try:
        write_phase(output_dir, rank, "process_started")
        inventory = device_record(rank, local_rank, device, args.profile)
        inventory_path = output_dir / "devices" / f"rank{rank}.json"
        atomic_json(inventory_path, inventory)
        inventory_paths = [output_dir / "devices" / f"rank{value}.json" for value in range(world_size)]
        wait_for(inventory_paths, args.file_rendezvous_seconds, "device inventories")

        inventory_report_path = output_dir / "device_world_report.json"
        if rank == 0:
            inventories = [json.loads(path.read_text(encoding="utf-8")) for path in inventory_paths]
            local_ranks = [item["local_rank"] for item in inventories]
            uuids = [item["device_uuid"] for item in inventories]
            status = "PASS" if sorted(local_ranks) == list(range(world_size)) and all(uuids) and len(set(uuids)) == world_size else "FAIL"
            atomic_json(inventory_report_path, {
                "status": status,
                "local_ranks": local_ranks,
                "device_uuids": uuids,
                "inventories": inventories,
            })
        wait_for([inventory_report_path], args.file_rendezvous_seconds, "device world report")
        inventory_report = json.loads(inventory_report_path.read_text(encoding="utf-8"))
        if inventory_report["status"] != "PASS":
            raise RuntimeError(f"invalid rank-to-GPU mapping: {inventory_report}")
        write_phase(output_dir, rank, "device_mapping_pass", uuid=inventory["device_uuid"])

        write_phase(output_dir, rank, "process_group_init_start")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=args.timeout_seconds),
        )
        initialized = True
        write_phase(output_dir, rank, "process_group_ready")

        scalar = torch.tensor(float(rank + 1), dtype=torch.float32, device=device)
        torch.cuda.synchronize(device)
        write_phase(output_dir, rank, "scalar_all_reduce_start", input=float(scalar.item()))
        work = dist.all_reduce(scalar, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()
        torch.cuda.synchronize(device)
        actual_sum = float(scalar.item())
        expected_sum = float(world_size * (world_size + 1) / 2)
        if not math.isclose(actual_sum, expected_sum, rel_tol=0.0, abs_tol=1e-5):
            raise RuntimeError(f"scalar all_reduce mismatch: expected={expected_sum}, actual={actual_sum}")
        write_phase(output_dir, rank, "scalar_all_reduce_pass", actual=actual_sum)

        gathered = [torch.empty(1, dtype=torch.int64, device=device) for _ in range(world_size)]
        source = torch.tensor([rank], dtype=torch.int64, device=device)
        work = dist.all_gather(gathered, source, async_op=True)
        work.wait()
        torch.cuda.synchronize(device)
        gathered_values = [int(value.item()) for value in gathered]
        if gathered_values != list(range(world_size)):
            raise RuntimeError(f"all_gather mismatch: {gathered_values}")
        write_phase(output_dir, rank, "all_gather_pass", values=gathered_values)

        count = args.broadcast_mib * 1024 * 1024 // 4
        tensor = torch.full((count,), 7.25 if rank == 0 else 0.0, dtype=torch.float32, device=device)
        work = dist.broadcast(tensor, src=0, async_op=True)
        work.wait()
        torch.cuda.synchronize(device)
        endpoints = [float(tensor[0].item()), float(tensor[-1].item())]
        if endpoints != [7.25, 7.25]:
            raise RuntimeError(f"broadcast mismatch: {endpoints}")
        write_phase(output_dir, rank, "fixed_broadcast_pass", mib=args.broadcast_mib)

        result_path = output_dir / "results" / f"rank{rank}.json"
        atomic_json(result_path, {
            "status": "PASS",
            "rank": rank,
            "profile": args.profile,
            "device_uuid": inventory["device_uuid"],
            "scalar_sum": actual_sum,
            "all_gather": gathered_values,
            "broadcast_mib": args.broadcast_mib,
        })
        result_paths = [output_dir / "results" / f"rank{value}.json" for value in range(world_size)]
        wait_for(result_paths, args.file_rendezvous_seconds, "rank results")
        report_path = output_dir / "nccl_transport_report.json"
        if rank == 0:
            atomic_json(report_path, {
                "status": "PASS",
                "profile": args.profile,
                "world_size": world_size,
                "device_report": inventory_report,
                "rank_results": [json.loads(path.read_text(encoding="utf-8")) for path in result_paths],
            })
        wait_for([report_path], args.file_rendezvous_seconds, "final transport report")
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        atomic_json(output_dir / "failures" / f"rank{rank}.json", {
            "status": "FAIL",
            "rank": rank,
            "profile": args.profile,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    args = parse_args()
    report = run(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
