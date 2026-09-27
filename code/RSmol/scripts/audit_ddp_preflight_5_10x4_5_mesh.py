#!/usr/bin/env python3
"""Identical eight-rank DDP preflight for x2 and x4 text MeSH checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("x2", "x4"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--file-rendezvous-seconds", type=int, default=300)
    parser.add_argument("--broadcast-mib", type=int, default=64)
    return parser.parse_args(argv)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def phase(
    output_dir: Path,
    *,
    rank: int,
    name: str,
    detail: dict[str, Any] | None = None,
) -> None:
    payload = {
        "rank": rank,
        "phase": name,
        "timestamp": time.time(),
        "pid": os.getpid(),
        "local_rank": int(os.environ.get("LOCAL_RANK", rank)),
        "detail": detail or {},
    }
    atomic_json(output_dir / "phases" / f"rank{rank}.json", payload)
    print(f"[ddp-preflight][rank={rank}] phase={name}", flush=True)


def wait_for(paths: list[Path], timeout_seconds: int, description: str) -> None:
    deadline = time.monotonic() + timeout_seconds
    missing = paths
    while time.monotonic() < deadline:
        missing = [path for path in paths if not path.is_file()]
        if not missing:
            return
        time.sleep(0.25)
    raise TimeoutError(
        f"timed out waiting for {description}: {[str(path) for path in missing]}"
    )


def load_variant(variant: str):
    if variant == "x2":
        from recursive_model_5_10x2_5_mesh import (  # type: ignore
            RecursiveLlamaForCausalLM,
            parameter_audit,
            register_auto_class,
        )
    else:
        from recursive_model_5_10x4_5_mesh import (  # type: ignore
            RecursiveLlamaForCausalLM,
            parameter_audit,
            register_auto_class,
        )
    return RecursiveLlamaForCausalLM, parameter_audit, register_auto_class


def normalized_loading_info(raw: dict[str, Any]) -> dict[str, list[str]]:
    return {
        key: [str(value) for value in raw.get(key, [])]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }


def parameter_fingerprint(model: torch.nn.Module) -> dict[str, Any]:
    try:
        references = list(model.named_parameters(remove_duplicate=False))
    except TypeError:
        references = list(model.named_parameters())

    contract_hash = hashlib.sha256()
    value_hash = hashlib.sha256()
    storage_owners: dict[int, str] = {}
    metadata: list[dict[str, Any]] = []
    dtype_numel: dict[str, int] = {}
    noncontiguous: list[str] = []

    for name, parameter in references:
        if parameter.is_meta:
            raise RuntimeError(f"meta parameter remains after checkpoint load: {name}")
        tensor = parameter.detach().cpu()
        storage_pointer = int(tensor.untyped_storage().data_ptr())
        alias_of = storage_owners.setdefault(storage_pointer, name)
        entry = {
            "name": name,
            "shape": [int(value) for value in tensor.shape],
            "stride": [int(value) for value in tensor.stride()],
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
            "requires_grad": bool(parameter.requires_grad),
            "alias_of": alias_of,
        }
        metadata.append(entry)
        contract_hash.update(
            json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        dtype_numel[entry["dtype"]] = dtype_numel.get(entry["dtype"], 0) + entry["numel"]
        if not tensor.is_contiguous():
            noncontiguous.append(name)
        value_hash.update(name.encode("utf-8"))
        if tensor.numel():
            raw_bytes = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            value_hash.update(raw_bytes)

    unique_parameters = list(model.parameters())
    return {
        "contract_sha256": contract_hash.hexdigest(),
        "values_sha256": value_hash.hexdigest(),
        "parameter_references": len(references),
        "unique_parameters": len(unique_parameters),
        "total_numel_references": int(sum(parameter.numel() for _, parameter in references)),
        "total_numel_unique": int(sum(parameter.numel() for parameter in unique_parameters)),
        "total_bytes_unique": int(
            sum(parameter.numel() * parameter.element_size() for parameter in unique_parameters)
        ),
        "dtype_numel": dtype_numel,
        "noncontiguous_parameters": noncontiguous,
        "parameters": metadata,
    }


def compare_fingerprints(output_dir: Path, world_size: int) -> dict[str, Any]:
    records = [
        json.loads(
            (output_dir / "fingerprints" / f"rank{rank}.json").read_text(
                encoding="utf-8"
            )
        )
        for rank in range(world_size)
    ]
    contracts = {record["fingerprint"]["contract_sha256"] for record in records}
    values = {record["fingerprint"]["values_sha256"] for record in records}
    audits = {
        json.dumps(record["parameter_audit"], sort_keys=True, default=str)
        for record in records
    }
    loading_failures = [
        {"rank": record["rank"], "loading_info": record["loading_info"]}
        for record in records
        if any(record["loading_info"].values())
    ]
    status = (
        "PASS"
        if len(contracts) == 1
        and len(values) == 1
        and len(audits) == 1
        and not loading_failures
        else "FAIL"
    )
    return {
        "status": status,
        "world_size": world_size,
        "contract_sha256": sorted(contracts),
        "values_sha256": sorted(values),
        "parameter_audit_variants": len(audits),
        "loading_failures": loading_failures,
        "rank_summaries": [
            {
                "rank": record["rank"],
                "unique_parameters": record["fingerprint"]["unique_parameters"],
                "total_numel_unique": record["fingerprint"]["total_numel_unique"],
                "total_bytes_unique": record["fingerprint"]["total_bytes_unique"],
                "dtype_numel": record["fingerprint"]["dtype_numel"],
                "noncontiguous_parameters": record["fingerprint"][
                    "noncontiguous_parameters"
                ],
            }
            for record in records
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    if world_size != args.world_size:
        raise RuntimeError(f"expected world_size={args.world_size}, got {world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("DDP preflight requires CUDA")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    process_group_initialized = False

    try:
        phase(output_dir, rank=rank, name="process_started")
        ModelClass, parameter_audit, register_auto_class = load_variant(args.variant)
        register_auto_class()

        phase(output_dir, rank=rank, name="model_load_start")
        model, raw_loading_info = ModelClass.from_pretrained(
            args.model_path.resolve(strict=True),
            local_files_only=True,
            output_loading_info=True,
        )
        loading_info = normalized_loading_info(raw_loading_info)
        phase(output_dir, rank=rank, name="model_loaded_cpu")

        fingerprint = parameter_fingerprint(model)
        rank_record = {
            "rank": rank,
            "variant": args.variant,
            "model_path": str(args.model_path.resolve()),
            "loading_info": loading_info,
            "fingerprint": fingerprint,
            "parameter_audit": parameter_audit(model),
        }
        atomic_json(output_dir / "fingerprints" / f"rank{rank}.json", rank_record)
        fingerprint_paths = [
            output_dir / "fingerprints" / f"rank{value}.json"
            for value in range(world_size)
        ]
        wait_for(
            fingerprint_paths,
            args.file_rendezvous_seconds,
            "all rank parameter fingerprints",
        )

        comparison_path = output_dir / "parameter_world_report.json"
        if rank == 0:
            atomic_json(comparison_path, compare_fingerprints(output_dir, world_size))
        wait_for(
            [comparison_path],
            args.file_rendezvous_seconds,
            "parameter world report",
        )
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison.get("status") != "PASS":
            raise RuntimeError(f"cross-rank parameter comparison failed: {comparison}")
        phase(output_dir, rank=rank, name="parameter_contract_ready")

        model.to(device)
        torch.cuda.synchronize(device)
        phase(output_dir, rank=rank, name="model_on_device")

        phase(output_dir, rank=rank, name="process_group_init_start")
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=args.timeout_seconds),
        )
        process_group_initialized = True
        phase(output_dir, rank=rank, name="process_group_ready")

        scalar = torch.tensor(float(rank + 1), dtype=torch.float32, device=device)
        dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
        expected_sum = world_size * (world_size + 1) / 2
        if not math.isclose(
            float(scalar.item()), float(expected_sum), rel_tol=0.0, abs_tol=1e-5
        ):
            raise RuntimeError(
                f"scalar all_reduce mismatch: expected={expected_sum}, actual={scalar.item()}"
            )
        phase(output_dir, rank=rank, name="scalar_all_reduce_pass")

        broadcast_numel = args.broadcast_mib * 1024 * 1024 // 4
        broadcast_tensor = torch.empty(
            broadcast_numel, dtype=torch.float32, device=device
        )
        if rank == 0:
            broadcast_tensor.fill_(7.25)
        else:
            broadcast_tensor.zero_()
        dist.broadcast(broadcast_tensor, src=0)
        torch.cuda.synchronize(device)
        if not math.isclose(float(broadcast_tensor[0].item()), 7.25) or not math.isclose(
            float(broadcast_tensor[-1].item()), 7.25
        ):
            raise RuntimeError("fixed-size broadcast contents differ from rank 0")
        del broadcast_tensor
        phase(
            output_dir,
            rank=rank,
            name="fixed_broadcast_pass",
            detail={"mib": args.broadcast_mib},
        )

        torch.cuda.synchronize(device)
        phase(output_dir, rank=rank, name="ddp_init_start")
        ddp_start = time.perf_counter()
        ddp = DDP(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        torch.cuda.synchronize(device)
        ddp_seconds = time.perf_counter() - ddp_start
        phase(
            output_dir,
            rank=rank,
            name="ddp_ready",
            detail={"seconds": ddp_seconds},
        )
        dist.barrier()

        rank_result = {
            "status": "PASS",
            "rank": rank,
            "variant": args.variant,
            "device": str(device),
            "ddp_init_seconds": ddp_seconds,
            "parameter_contract_sha256": fingerprint["contract_sha256"],
            "parameter_values_sha256": fingerprint["values_sha256"],
            "scalar_all_reduce": float(scalar.item()),
            "broadcast_mib": args.broadcast_mib,
            "ddp_type": type(ddp).__name__,
            "ddp_init_sync": "default_true",
        }
        atomic_json(output_dir / "rank_results" / f"rank{rank}.json", rank_result)
        result_paths = [
            output_dir / "rank_results" / f"rank{value}.json"
            for value in range(world_size)
        ]
        wait_for(result_paths, args.file_rendezvous_seconds, "all rank results")

        final_path = output_dir / "ddp_preflight_report.json"
        if rank == 0:
            rank_results = [
                json.loads(path.read_text(encoding="utf-8")) for path in result_paths
            ]
            atomic_json(
                final_path,
                {
                    "status": "PASS",
                    "variant": args.variant,
                    "model_path": str(args.model_path.resolve()),
                    "world_size": world_size,
                    "parameter_comparison": comparison,
                    "rank_results": rank_results,
                    "standard_ddp": {
                        "broadcast_buffers": False,
                        "find_unused_parameters": False,
                        "init_sync": True,
                    },
                },
            )
        wait_for([final_path], args.file_rendezvous_seconds, "final report")
        return json.loads(final_path.read_text(encoding="utf-8"))
    except Exception as exc:
        atomic_json(
            output_dir / "failures" / f"rank{rank}.json",
            {
                "status": "FAIL",
                "rank": rank,
                "variant": args.variant,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if process_group_initialized and dist.is_initialized():
            dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
