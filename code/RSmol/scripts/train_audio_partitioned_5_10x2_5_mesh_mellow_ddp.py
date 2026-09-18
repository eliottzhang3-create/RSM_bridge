#!/usr/bin/env python3
"""Six-partition rank-RAM training for the 5-10x2-5 audio MeSH model.

This is deliberately separate from the historical fixed-260-prefix trainer.
Only checkpoints produced by this route can resume it.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import train_audio_5_10x2_5_mesh_mellow_ddp as base
from audio_5_10x2_5_mesh_mellow.data import ReasonAQADataset, collate_reasonaqa
from audio_5_10x2_5_mesh_mellow.model import (
    ARCHITECTURE_CONTRACT, MAPPER_CONTRACT, AUDIO_SINGLE_PREFIX_TOKENS,
    AUDIO_DUAL_PREFIX_TOKENS,
)

CONTRACT = "component_partitions6_rank_ram_compact_audio_v1"
SMOKE_SEGMENTS = ((2, 10), (0, 10), (1, 2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "formal"), required=True)
    p.add_argument("--partition-store-root", type=Path, default=Path(base.DEFAULT_COMPONENT_PARTITION_STORE_ROOT))
    p.add_argument("--model-path", "--mesh-checkpoint", dest="model_path", type=Path, default=Path(base.DEFAULT_MESH))
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--smoke20-report", type=Path, help="Formal gate: audited 20-step smoke report")
    p.add_argument("--smoke-resume-report", type=Path, help="Formal gate: audited 20-to-22 resume report")
    p.add_argument("--tokenizer-path", type=Path)
    p.add_argument("--htsat-checkpoint", type=Path, default=Path(base.DEFAULT_HTSAT))
    p.add_argument("--mellow-root", type=Path, default=Path(base.DEFAULT_MELLOW))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--max-lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--checkpoint-retention", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dist-timeout-minutes", type=int, default=30)
    p.add_argument("--release-min-fraction", type=float, default=0.70,
                   help="Minimum per-rank RSS drop as a fraction of partition payload bytes")
    p.add_argument("--release-timeout-seconds", type=int, default=120)
    return p.parse_args(argv)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    if (root / "BUILDING").exists():
        raise RuntimeError("partition root is BUILDING")
    report = json.loads((root / "materialization_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("format") != "reasonaqa_component_partition_stores_v1" or report.get("duplicated_audio") != 0 or report.get("source_payload_sha256_reverified") is not True:
        raise RuntimeError("materialization root is not an audited zero-copy PASS")
    partitions = report.get("partitions", [])
    if len(partitions) != 6 or sorted(int(x["partition_id"]) for x in partitions) != list(range(6)):
        raise RuntimeError("expected exactly partitions 0..5")
    result = {"root": str(root), "report_sha256": _sha(root / "materialization_report.json"), "partitions": []}
    for pid in range(6):
        directory = root / f"partition_{pid}"
        if (directory / "BUILDING").exists():
            raise RuntimeError(f"partition {pid} is BUILDING")
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        entry = next(item for item in partitions if int(item["partition_id"]) == pid)
        if meta.get("status") != "PASS" or meta.get("waveform_verification", {}).get("passed") is not True or entry.get("waveform_verification", {}).get("passed") is not True:
            raise RuntimeError(f"partition {pid} is not verified PASS")
        if int(meta.get("partition_id", -1)) != pid or int(entry["partition_id"]) != pid:
            raise RuntimeError("partition metadata order mismatch")
        for key in ("manifest_sha256", "index_sha256", "waveform_sha256", "num_unique_audio_files", "total_waveform_bytes"):
            if meta.get(key) != entry.get(key):
                raise RuntimeError(f"partition {pid} report/metadata {key} mismatch")
        if _sha(directory / "rows.jsonl") != meta["manifest_sha256"] or _sha(directory / "index.jsonl") != meta["index_sha256"]:
            raise RuntimeError(f"partition {pid} manifest/index hash mismatch")
        if (directory / "waveforms.f32").stat().st_size != int(meta["total_waveform_bytes"]):
            raise RuntimeError(f"partition {pid} payload byte mismatch")
        result["partitions"].append({"id": pid, "rows": int(entry["qa_rows"]), "audio": int(meta["num_unique_audio_files"]), "bytes": int(meta["total_waveform_bytes"]), "manifest_sha256": meta["manifest_sha256"], "index_sha256": meta["index_sha256"], "waveform_sha256": meta["waveform_sha256"]})
    if sum(x["rows"] for x in result["partitions"]) != 968059 or sum(x["bytes"] for x in result["partitions"]) != int(report["total_materialized_waveform_bytes"]):
        raise RuntimeError("partition totals disagree with materialization")
    return result


def _quotas(counts: list[int], epochs: int, global_batch: int) -> list[list[int]]:
    if len(counts) != 6 or epochs <= 0:
        raise ValueError("six partitions and positive epochs required")
    total = sum(counts) // global_batch
    large = counts[0] // global_batch
    remaining = total - large
    weights = counts[1:]
    weight_total = sum(weights)
    cumulative = [0] * 5
    result = []
    for epoch in range(epochs):
        ideal = [remaining * count / weight_total for count in weights]
        current = [math.floor(value) for value in ideal]
        extras = remaining - sum(current)
        priority = sorted(range(5), key=lambda i: (-(epoch + 1) * ideal[i] + cumulative[i] + current[i], (i + epoch) % 5))
        for index in priority[:extras]:
            current[index] += 1
        cumulative = [a + b for a, b in zip(cumulative, current)]
        quotas = [large, *current]
        if sum(quotas) != total or quotas[0] * global_batch > counts[0]:
            raise AssertionError("epoch quota allocation failed")
        result.append(quotas)
    return result


def _order(seed: int, epoch: int) -> list[int]:
    rng = random.Random((seed + 1) * 1_000_003 + epoch * 10_007)
    rest = list(range(1, 6))
    rng.shuffle(rest)
    rest.insert(rng.randrange(5), 0)
    assert rest[-1] != 0
    return rest


def _schedule(args: argparse.Namespace, inventory: dict[str, Any]) -> list[dict[str, int]]:
    if args.mode == "smoke":
        return [{"epoch": 0, "position": position, "partition_id": pid, "steps": count}
                for position, (pid, count) in enumerate(SMOKE_SEGMENTS)]
    counts = [entry["rows"] for entry in inventory["partitions"]]
    quotas = _quotas(counts, args.epochs, args.world_size * args.micro_batch_size * args.gradient_accumulation_steps)
    return [{"epoch": epoch, "position": position, "partition_id": pid, "steps": quotas[epoch][pid]}
            for epoch in range(args.epochs) for position, pid in enumerate(_order(args.seed, epoch))]


def _formal_smoke_gate(args: argparse.Namespace, inventory: dict[str, Any]) -> dict[str, Any]:
    if args.smoke20_report is None or args.smoke_resume_report is None:
        raise ValueError("formal training requires --smoke20-report and --smoke-resume-report")
    initial = json.loads(args.smoke20_report.read_text(encoding="utf-8"))
    resumed = json.loads(args.smoke_resume_report.read_text(encoding="utf-8"))
    for name, report, expected_start, expected_end in (
        ("smoke20", initial, {"segment": 0, "segment_step": 0, "global_step": 0}, {"segment": 2, "segment_step": 0, "global_step": 20}),
        ("resume2", resumed, {"segment": 2, "segment_step": 0, "global_step": 20}, {"segment": 3, "segment_step": 0, "global_step": 22}),
    ):
        if report.get("status") != "PASS" or report.get("mode") != "smoke" or report.get("hard_failures") or report.get("inventory") != inventory or report.get("start_cursor") != expected_start or report.get("end_cursor") != expected_end:
            raise RuntimeError(f"formal gate rejects {name} report")
        if len(report.get("segments", [])) != (2 if name == "smoke20" else 1):
            raise RuntimeError(f"formal gate rejects {name} segment count")
        if not report.get("first_step_gradient_audit", {}).get("trace_matches_5_10_10_5"):
            raise RuntimeError(f"formal gate rejects {name} gradient/trace audit")
        if any(len(segment.get("release", [])) != 8 or not all(item.get("passed") for item in segment["release"]) or segment.get("cgroup_release", {}).get("anon_drop_bytes") is None or segment["cgroup_release"]["anon_drop_bytes"] < segment["cgroup_release"]["required_bytes"] or len(segment.get("steps", [])) != segment["segment"]["steps"] for segment in report["segments"]):
            raise RuntimeError(f"formal gate rejects {name} release audit")
    checkpoint = str(Path(initial["checkpoints"][-1]).resolve())
    if resumed.get("resume_checkpoint") != checkpoint or resumed.get("resume_verified_two_steps") is not True:
        raise RuntimeError("resume report does not prove continuation from smoke step-20 checkpoint")
    if initial.get("seed") != args.seed or resumed.get("seed") != args.seed:
        raise RuntimeError("smoke/formal seed mismatch")
    return {"smoke20_report": str(args.smoke20_report.resolve()), "smoke_resume_report": str(args.smoke_resume_report.resolve()), "checkpoint20": checkpoint}


def _plan(dataset: ReasonAQADataset, *, seed: int, epoch: int, pid: int, steps: int, global_batch: int) -> tuple[list[list[int]], dict[str, Any]]:
    rows = list(range(len(dataset)))
    random.Random((seed + 1) * 1000003 + epoch * 10007 + pid * 101).shuffle(rows)
    required = steps * global_batch
    if pid == 0 and required > len(rows):
        raise RuntimeError("partition 0 repetition forbidden")
    if not rows:
        raise RuntimeError("empty partition")
    selected = [rows[index % len(rows)] for index in range(required)]
    chunks = [selected[index:index + global_batch] for index in range(0, required, global_batch)]
    if len(chunks) != steps or any(len(chunk) != global_batch for chunk in chunks):
        raise AssertionError("incomplete optimizer window")
    digest = hashlib.sha256(",".join(str(index) for chunk in chunks for index in chunk).encode("ascii")).hexdigest()
    audit = {"plan_sha256": digest, "source_rows": len(dataset), "selected_rows": required, "repeated_rows": max(0, required - len(rows)), "dropped_rows": max(0, len(rows) - required), "shuffle": "all partition rows together, then same-partition wraparound only", "global_batch": global_batch}
    return chunks, audit


def _rss() -> int:
    with Path("/proc/self/status").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/self/status lacks VmRSS")


def _memory() -> dict[str, Any]:
    return {"rss_bytes": _rss(), "cgroup": base._cgroup_memory_snapshot()}


def _anon(snapshot: dict[str, Any]) -> int | None:
    cgroup = snapshot.get("cgroup") or {}
    if cgroup.get("status") != "PASS":
        return None
    value = cgroup.get("memory_stat_bytes", {}).get("anon")
    return None if value is None else int(value)


def _trim() -> None:
    if ctypes.CDLL(None).malloc_trim(0) not in (0, 1):
        raise RuntimeError("malloc_trim returned an invalid result")


def _gather(local: Any, world: int) -> list[Any]:
    gathered: list[Any] = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered[0] = local
    return gathered


def _checkpoint(path: Path, model: Any, tokenizer: Any, optimizer: Any, scheduler: Any,
                args: argparse.Namespace, inventory: dict[str, Any], schedule: list[dict[str, int]],
                cursor: dict[str, int], rank: int, world: int, device: torch.device,
                plan_hash: str | None) -> None:
    rng = _gather(base._rng_state(device), world)
    if rank != 0:
        return
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent))
    published = False
    try:
        model.mesh_model.save_pretrained(tmp / "mesh_model", safe_serialization=False)
        tokenizer.save_pretrained(tmp / "tokenizer")
        torch.save(base._trainable_state(model), tmp / "audio_bridge.pt")
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "global_step": cursor["global_step"], "cursor": cursor, "plan_hash": plan_hash, "rng_states_by_rank": {str(i): state for i, state in enumerate(rng)}}, tmp / "training_state.pt")
        config = {"contract": CONTRACT, "architecture_contract": ARCHITECTURE_CONTRACT, "mapper_contract": MAPPER_CONTRACT, "compact_single_audio_prefix": True, "prefix_tokens": {"single": AUDIO_SINGLE_PREFIX_TOKENS, "dual": AUDIO_DUAL_PREFIX_TOKENS}, "mode": args.mode, "inventory": inventory, "schedule": schedule, "epochs": args.epochs, "world_size": world, "micro_batch_size": args.micro_batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps, "seed": args.seed, "max_lr": args.max_lr, "min_lr": args.min_lr, "warmup_steps": args.warmup_steps, "total_steps": sum(x["steps"] for x in schedule), "save_every": args.save_every, "checkpoint_retention": args.checkpoint_retention, "dist_timeout_minutes": args.dist_timeout_minutes, "release_min_fraction": args.release_min_fraction, "release_timeout_seconds": args.release_timeout_seconds, "htsat_checkpoint": str(args.htsat_checkpoint.resolve()), "mellow_root": str(args.mellow_root.resolve()), "mellow_provenance": model._audio_provenance}
        (tmp / "audio_mesh_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (tmp / "checkpoint_complete.json").write_text(json.dumps({"status": "complete", "global_step": cursor["global_step"], "contract": CONTRACT}) + "\n", encoding="utf-8")
        for file in ("mesh_model/config.json", "tokenizer/tokenizer_config.json", "audio_bridge.pt", "training_state.pt", "audio_mesh_config.json", "checkpoint_complete.json"):
            if not (tmp / file).is_file():
                raise RuntimeError(f"checkpoint missing {file}")
        tmp.replace(path)
        published = True
    finally:
        if not published:
            shutil.rmtree(tmp, ignore_errors=True)


def _resume(path: Path, args: argparse.Namespace, inventory: dict[str, Any], schedule: list[dict[str, int]], optimizer: Any, scheduler: Any, rank: int, device: torch.device, provenance: dict[str, Any]) -> tuple[dict[str, int], str | None]:
    config = json.loads((path / "audio_mesh_config.json").read_text(encoding="utf-8"))
    marker = json.loads((path / "checkpoint_complete.json").read_text(encoding="utf-8"))
    if config.get("contract") != CONTRACT or marker.get("status") != "complete" or marker.get("contract") != CONTRACT or config.get("compact_single_audio_prefix") is not True:
        raise RuntimeError("not a complete compatible partition checkpoint")
    if config.get("architecture_contract") != ARCHITECTURE_CONTRACT or config.get("mapper_contract") != MAPPER_CONTRACT or config.get("prefix_tokens") != {"single": 130, "dual": 260}:
        raise RuntimeError("resume architecture/prefix contract mismatch")
    if config.get("mellow_provenance", {}).get("mellow_htsat_sha256") != provenance.get("mellow_htsat_sha256"):
        raise RuntimeError("resume Mellow implementation SHA256 mismatch")
    for key, expected in {"mode": args.mode, "inventory": inventory, "schedule": schedule, "epochs": args.epochs, "world_size": args.world_size, "micro_batch_size": args.micro_batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps, "seed": args.seed, "max_lr": args.max_lr, "min_lr": args.min_lr, "warmup_steps": args.warmup_steps, "total_steps": sum(x["steps"] for x in schedule), "save_every": args.save_every, "checkpoint_retention": args.checkpoint_retention, "dist_timeout_minutes": args.dist_timeout_minutes, "release_min_fraction": args.release_min_fraction, "release_timeout_seconds": args.release_timeout_seconds, "htsat_checkpoint": str(args.htsat_checkpoint.resolve()), "mellow_root": str(args.mellow_root.resolve())}.items():
        if config.get(key) != expected:
            raise RuntimeError(f"resume contract differs in {key}")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    cursor = state["cursor"]
    if marker["global_step"] != cursor["global_step"] or cursor["global_step"] != sum(x["steps"] for x in schedule[:cursor["segment"]]) + cursor["segment_step"]:
        raise RuntimeError("checkpoint cursor/global step mismatch")
    rng = state["rng_states_by_rank"]
    if set(rng) != {str(i) for i in range(args.world_size)}:
        raise RuntimeError("checkpoint per-rank RNG coverage mismatch")
    base._restore_rng_state(rng[str(rank)], device)
    return cursor, state.get("plan_hash")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
    if not torch.cuda.is_available() or world != args.world_size or args.world_size != 8 or args.micro_batch_size != 8 or args.gradient_accumulation_steps != 4:
        raise RuntimeError("partition training requires exactly 8 GPUs, microbatch 8, GA 4")
    if args.epochs <= 0 or args.dist_timeout_minutes < 30 or args.save_every <= 0 or args.checkpoint_retention <= 0 or not 0 < args.release_min_fraction <= 1 or args.release_timeout_seconds <= 0:
        raise ValueError("invalid epochs, timeout, checkpoint or release threshold")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(minutes=args.dist_timeout_minutes))
    base._seed(args.seed, rank)
    report: dict[str, Any] = {"status": "FAIL", "mode": args.mode, "rank": rank, "segments": [], "checkpoints": [], "hard_failures": []}
    output_available = not args.output_dir.exists() or not any(args.output_dir.iterdir())
    try:
        if not output_available:
            raise FileExistsError(f"refusing nonempty output directory: {args.output_dir}")
        inventory = _inventory(args.partition_store_root)
        smoke_gate = _formal_smoke_gate(args, inventory) if args.mode == "formal" else None
        schedule = _schedule(args, inventory)
        total = sum(x["steps"] for x in schedule)
        if args.mode == "formal" and total != (968059 // 256) * args.epochs:
            raise RuntimeError("formal step budget differs from global epoch floor")
        args.warmup_steps = args.warmup_steps if args.warmup_steps is not None else math.ceil(total * .05)
        if args.warmup_steps != math.ceil(total * .05):
            raise ValueError("warmup must equal ceil(5% of total optimizer steps)")
        args.compact_single_audio_prefix = True
        model, tokenizer = base._load_model(args, device)
        model.train()
        if not model.trainable_parameter_audit()["training_mode_contract"]:
            raise RuntimeError("trainable parameter audit failed")
        model.mesh_model.model.routing_stats_mode = False
        model.mesh_model.model.gradient_audit_mode = True
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.max_lr, betas=(.9, .95), weight_decay=.1)
        scheduler = base._make_scheduler(optimizer, max_lr=args.max_lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps, total_steps=total)
        cursor = {"segment": 0, "segment_step": 0, "global_step": 0}
        saved_plan_hash = None
        if args.resume_from:
            cursor, saved_plan_hash = _resume(args.resume_from, args, inventory, schedule, optimizer, scheduler, rank, device, model._audio_provenance)
            if args.mode == "smoke" and cursor != {"segment": 2, "segment_step": 0, "global_step": 20}:
                raise RuntimeError("smoke resume requires released step-20 boundary checkpoint")
        elif args.mode == "smoke":
            assert total == 22
        ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        report.update({"inventory": inventory, "schedule": schedule, "total_steps": total, "warmup_steps": args.warmup_steps, "start_cursor": dict(cursor), "prefix_contract": {"single": 130, "dual": 260}, "seed": args.seed, "resume_checkpoint": str(args.resume_from.resolve()) if args.resume_from else None, "smoke_gate": smoke_gate})
        global_batch = world * args.micro_batch_size * args.gradient_accumulation_steps
        first_audit = None
        stop_step = 20 if args.mode == "smoke" and not args.resume_from else total
        for segment_index in range(cursor["segment"], len(schedule)):
            segment = schedule[segment_index]
            if cursor["global_step"] >= stop_step:
                break
            pid = segment["partition_id"]
            part = inventory["partitions"][pid]
            path = Path(inventory["root"]) / f"partition_{pid}"
            before = _memory()
            started = time.perf_counter()
            dataset = ReasonAQADataset(path / "rows.jsonl", tokenizer, unique_waveform_store_dir=path)
            if len(dataset) != part["rows"]:
                raise RuntimeError("partition QA cardinality changed")
            cache = base._RankLocalStoreWaveforms(dataset)
            for audio_id in range(part["audio"]):
                cache._get_audio_id(audio_id)
            if cache.current_bytes != part["bytes"] or cache.misses != part["audio"] or cache.evictions:
                raise RuntimeError("partition preload cardinality/bytes mismatch")
            chunks, plan_audit = _plan(dataset, seed=args.seed, epoch=segment["epoch"], pid=pid, steps=segment["steps"], global_batch=global_batch)
            if segment_index == cursor["segment"] and cursor["segment_step"] and saved_plan_hash != plan_audit["plan_sha256"]:
                raise RuntimeError("resume optimizer-window plan hash mismatch")
            dist.barrier()
            after_load = _memory()
            segment_report: dict[str, Any] = {"segment": segment, "plan": plan_audit, "preload_seconds": time.perf_counter() - started, "memory_before": before, "memory_after_load": after_load, "steps": []}
            if rank == 0:
                print(f"[partition-train] loaded p{pid} epoch={segment['epoch']} steps={segment['steps']} bytes/rank={part['bytes']} seconds={segment_report['preload_seconds']:.1f}", flush=True)
            initial_stats = cache.stats()
            start_at = cursor["segment_step"] if segment_index == cursor["segment"] else 0
            for local_step in range(start_at, segment["steps"]):
                if cursor["global_step"] >= stop_step:
                    break
                step_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                step_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
                chunk = chunks[local_step]
                rank_rows = chunk[rank * 32:(rank + 1) * 32]
                if len(rank_rows) != 32:
                    raise AssertionError("rank optimizer window is not 32 rows")
                slot_kind = {dataset.audio_structure(index)[0] for index in chunk}
                for micro in range(4):
                    indices = rank_rows[micro * 8:(micro + 1) * 8]
                    batch = collate_reasonaqa([cache.materialize_from_cached_metadata(index) for index in indices], tokenizer)
                    # Length/slot controls stay on CPU so variable-prefix
                    # packing does not trigger per-row GPU scalar syncs.
                    device_keys = {"audio1", "audio2", "text_ids"}
                    batch = {key: (value.to(device) if torch.is_tensor(value) and key in device_keys else value) for key, value in batch.items()}
                    with (ddp.no_sync() if micro != 3 else contextlib.nullcontext()):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            output = ddp(**{key: value for key, value in batch.items() if key not in {"row_indices", "audio2_reused"}})
                        if output.loss is None or not bool(torch.isfinite(output.loss)):
                            raise RuntimeError("nonfinite partition training loss")
                        step_loss_sum.add_(output.loss.detach().float())
                        (output.loss / 4).backward()
                    del batch, output
                owner = ddp.module
                if first_audit is None:
                    first_audit = base._mesh_runtime_gradient_audit(owner, require_router_stats=False)
                    owner.mesh_model.model.gradient_audit_mode = False
                torch.nn.utils.clip_grad_norm_(ddp.parameters(), .5, error_if_nonfinite=True)
                lr_used = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
                cursor = {"segment": segment_index, "segment_step": local_step + 1, "global_step": cursor["global_step"] + 1}
                torch.cuda.synchronize(device)
                step_loss = float((step_loss_sum / args.gradient_accumulation_steps).item())
                segment_report["steps"].append({"global_step": cursor["global_step"], "partition_step": local_step + 1, "slot": "mixed" if len(slot_kind) == 2 else "single" if True in slot_kind else "dual", "loss": step_loss, "lr": lr_used, "seconds": time.perf_counter() - step_started})
                at_boundary = local_step + 1 == segment["steps"]
                if rank == 0 and (cursor["global_step"] % 10 == 0 or at_boundary):
                    print(f"[partition-train] step={cursor['global_step']}/{total} p{pid} {local_step+1}/{segment['steps']} loss={step_loss:.6f} lr={lr_used:.8g} {segment_report['steps'][-1]['seconds']:.3f}s", flush=True)
                if args.mode == "formal" and cursor["global_step"] % args.save_every == 0 and not at_boundary:
                    checkpoint = args.output_dir / f"checkpoint-{cursor['global_step']:06d}"
                    _checkpoint(checkpoint, owner, tokenizer, optimizer, scheduler, args, inventory, schedule, cursor, rank, world, device, plan_audit["plan_sha256"])
                    if rank == 0:
                        report["checkpoints"].append(str(checkpoint))
                        report["retained_checkpoints"] = base._prune_checkpoints(args.output_dir, args.checkpoint_retention)
                    dist.barrier()
            after_stats = cache.stats()
            if after_stats["waveform_misses"] != initial_stats["waveform_misses"] or after_stats["cloned_bytes"] != initial_stats["cloned_bytes"] or after_stats["resident_unique_audio"] != part["audio"]:
                raise RuntimeError("training accessed store or changed complete rank-RAM residency")
            # No batch/iterator/prefetch references survive the boundary. The
            # owned waveform tensors must disappear before another load starts.
            torch.cuda.synchronize(device)
            del chunks, cache.items
            dataset.unique_waveform_store.close()
            del cache, dataset
            gc.collect()
            _trim()
            dist.barrier()
            threshold = int(part["bytes"] * args.release_min_fraction)
            cgroup_required = int(part["bytes"] * world * args.release_min_fraction)
            release_deadline = time.monotonic() + args.release_timeout_seconds
            release_polls = 0
            while True:
                after_release = _memory()
                release_polls += 1
                current_anon = _anon(after_release)
                loaded_anon = _anon(after_load)
                rss_ok = after_load["rss_bytes"] - after_release["rss_bytes"] >= threshold
                cgroup_ok = current_anon is not None and loaded_anon is not None and loaded_anon - current_anon >= cgroup_required
                if rss_ok and cgroup_ok or time.monotonic() >= release_deadline:
                    break
                time.sleep(2.0)
            released = after_load["rss_bytes"] - after_release["rss_bytes"]
            local_release = {"rank": rank, "rss_drop_bytes": released, "required_drop_bytes": threshold, "before": before, "after_load": after_load, "after_release": after_release, "polls": release_polls, "passed": released >= threshold}
            releases = _gather(local_release, world)
            segment_report["release"] = releases
            cgroup_loaded = _anon(after_load)
            cgroup_released = _anon(after_release)
            cgroup_drop = None if cgroup_loaded is None or cgroup_released is None else cgroup_loaded - cgroup_released
            segment_report["cgroup_release"] = {"anon_drop_bytes": cgroup_drop, "required_bytes": cgroup_required, "available": cgroup_drop is not None, "timeout_seconds": args.release_timeout_seconds}
            if cgroup_drop is None:
                raise RuntimeError(f"partition p{pid} cannot audit cgroup anon release; refusing next preload")
            if not all(item["passed"] for item in releases):
                raise RuntimeError(f"partition p{pid} RAM release failed: {[(x['rank'], x['rss_drop_bytes'], x['required_drop_bytes']) for x in releases]}")
            if cgroup_drop is not None and cgroup_drop < cgroup_required:
                raise RuntimeError(f"partition p{pid} cgroup anon failed to fall: {cgroup_drop} < {cgroup_required}")
            report["segments"].append(segment_report)
            cursor = {"segment": segment_index + 1, "segment_step": 0, "global_step": cursor["global_step"]}
            at_end = cursor["global_step"] == total
            save_boundary = (args.mode == "smoke" and cursor["global_step"] in (20, 22)) or (args.mode == "formal" and (cursor["global_step"] % args.save_every == 0 or at_end))
            if save_boundary:
                checkpoint = args.output_dir / f"checkpoint-{cursor['global_step']:06d}"
                _checkpoint(checkpoint, owner, tokenizer, optimizer, scheduler, args, inventory, schedule, cursor, rank, world, device, None)
                if rank == 0:
                    report["checkpoints"].append(str(checkpoint))
                    report["retained_checkpoints"] = base._prune_checkpoints(args.output_dir, args.checkpoint_retention)
                dist.barrier()
            if rank == 0:
                print(f"[partition-train] released p{pid} RSS drop/rank={[round(x['rss_drop_bytes']/1024**3, 2) for x in releases]} GiB", flush=True)
        if cursor["global_step"] != stop_step:
            raise RuntimeError(f"run ended at step {cursor['global_step']} rather than {stop_step}")
        report.update({"status": "PASS", "end_cursor": cursor, "first_step_gradient_audit": first_audit, "resume_verified_two_steps": args.mode == "smoke" and args.resume_from is not None and cursor["global_step"] == 22})
        return report
    except Exception as exc:
        report["hard_failures"].append({"error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        if rank == 0 and output_available:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "partition_training_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run(parse_args())
