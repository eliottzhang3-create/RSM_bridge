#!/usr/bin/env python3
"""DDP trainer and 8-GPU smoke gates for ReasonAQA + MeSH audio."""
from __future__ import annotations

import argparse
import csv
import contextlib
import copy
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
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from audio_5_10x2_5_mesh_mellow.data import ReasonAQADataset, collate_reasonaqa  # noqa: E402
from audio_5_10x2_5_mesh_mellow.model import (  # noqa: E402
    AUDIO_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    ARCHITECTURE_CONTRACT,
    MAPPER_CONTRACT,
    MESH_HIDDEN_SIZE,
    AudioMeshConfig,
    AudioMeshModel,
    _load_mellow_wrapper,
    write_config,
)
from perf20_schedule import active_steps as _schedule_active_steps, affected_steps as _schedule_affected_steps  # noqa: E402
from recursive_model_5_10x2_5_mesh import RecursiveLlamaForCausalLM, register_auto_class  # noqa: E402


DEFAULT_MESH = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244"
DEFAULT_HTSAT = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt"
DEFAULT_MELLOW = "/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main"
PERF20_STEPS = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", choices=("STAGE5", "STAGE7", "FORMAL", "PERF20"), default="STAGE5")
    # ``--mesh-checkpoint`` is the public name used by the repository
    # submission wrappers and operator commands; keep ``--model-path`` as a
    # backward-compatible alias for older invocations.
    p.add_argument("--model-path", "--mesh-checkpoint", dest="model_path", type=Path, default=Path(DEFAULT_MESH))
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--tokenizer-path", type=Path)
    p.add_argument("--htsat-checkpoint", type=Path, default=Path(DEFAULT_HTSAT))
    p.add_argument("--mellow-root", type=Path, default=Path(DEFAULT_MELLOW))
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--report-path", type=Path)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--checkpoint-retention", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers per DDP rank; 0 keeps loading in the rank process")
    p.add_argument("--steady-state-start-step", type=int, default=6, help="First optimizer step included in steady-state summaries (PERF20 default: 6, excluding steps 1-5)")
    profiler_group = p.add_mutually_exclusive_group()
    profiler_group.add_argument("--profiler", "--enable-profiler", dest="profiler", action="store_true", help="Enable rank0 torch.profiler collection for PERF20")
    profiler_group.add_argument("--no-profiler", "--disable-profiler", dest="profiler", action="store_false", help="Disable torch.profiler collection")
    p.set_defaults(profiler=False)
    p.add_argument("--profiler-skip-first", type=int, default=4)
    p.add_argument("--profiler-wait", type=int, default=1)
    p.add_argument("--profiler-warmup", type=int, default=1)
    p.add_argument("--profiler-active", type=int, default=2)
    p.add_argument("--profiler-repeat", type=int, default=1)
    p.add_argument("--profiler-with-stack", action="store_true", help="Collect Python/C++ stacks; off by default for the first profile")
    p.add_argument("--profiler-profile-memory", action="store_true", help="Collect profiler memory events; off by default for the first profile")
    p.add_argument("--profiler-record-shapes", action="store_true", help="Record operator input shapes; off by default for the first profile")
    return p.parse_args(argv)


def _validate_profiler_options(args: argparse.Namespace, max_steps: int) -> None:
    values = (
        args.profiler_skip_first,
        args.profiler_wait,
        args.profiler_warmup,
        args.profiler_active,
        args.profiler_repeat,
    )
    if any(int(value) < 0 for value in values):
        raise ValueError("profiler schedule values must be non-negative")
    if int(args.profiler_active) <= 0 or int(args.profiler_repeat) <= 0:
        raise ValueError("profiler active and repeat must be positive")
    required = int(args.profiler_skip_first)
    required += int(args.profiler_repeat) * (int(args.profiler_wait) + int(args.profiler_warmup) + int(args.profiler_active))
    if args.profiler and required > int(max_steps):
        raise ValueError(
            "profiler schedule cannot complete within the bounded run: "
            f"needs {required} optimizer steps, run has {max_steps}"
        )


def _profile_active_steps(args: argparse.Namespace, max_steps: int) -> list[int]:
    """Return one-indexed optimizer steps in the active profiler window."""
    return _schedule_active_steps(
        skip_first=int(args.profiler_skip_first),
        wait=int(args.profiler_wait),
        warmup=int(args.profiler_warmup),
        active=int(args.profiler_active),
        repeat=int(args.profiler_repeat),
        max_steps=int(max_steps),
    )


def _profile_overhead_steps(args: argparse.Namespace, max_steps: int) -> list[int]:
    """Return one-indexed wait/warmup/active steps excluded from throughput."""
    return _schedule_affected_steps(
        skip_first=int(args.profiler_skip_first),
        wait=int(args.profiler_wait),
        warmup=int(args.profiler_warmup),
        active=int(args.profiler_active),
        repeat=int(args.profiler_repeat),
        max_steps=int(max_steps),
    )


def _environment_report(device: torch.device, world: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(getattr(torch.version, "cuda", None)),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "device_index": int(device.index or 0),
        "world_size": int(world),
    }
    try:
        payload["gpu_name"] = str(torch.cuda.get_device_name(device))
        payload["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    except Exception as exc:
        payload["gpu_name_error"] = repr(exc)
    return payload


class _CudaPhaseEvents:
    """Collect device elapsed times without relying on asynchronous wall clocks."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def measure(self, name: str, fn: Any) -> Any:
        if not self.enabled:
            return fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            return fn()
        finally:
            end.record()
            self.events.setdefault(name, []).append((start, end))

    def seconds(self, device: torch.device) -> dict[str, float]:
        if not self.enabled:
            return {}
        torch.cuda.synchronize(device)
        return {
            name: sum(float(start.elapsed_time(end)) / 1000.0 for start, end in pairs)
            for name, pairs in self.events.items()
        }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "median": _percentile(values, 0.50),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
    }


def _make_profiler(args: argparse.Namespace, profile_dir: Path, artifact_paths: list[dict[str, Any]]) -> Any:
    profile_dir.mkdir(parents=True, exist_ok=True)

    def _on_trace_ready(active_profiler: Any) -> None:
        """Export one completed schedule cycle before profiler rotates it."""
        cycle_index = len(artifact_paths) + 1
        step_num = int(getattr(active_profiler, "step_num", 0))
        cycle_dir = profile_dir / f"cycle_{cycle_index:02d}_step_{step_num:04d}"
        # PERF20 output directories are unique, and cycle directories are
        # nevertheless created exclusively so a repeated callback can never
        # silently overwrite an earlier cycle.
        cycle_dir.mkdir(parents=True, exist_ok=False)
        worker_name = f"rank0-cycle{cycle_index:02d}-step{step_num:04d}"
        before_trace = set(cycle_dir.iterdir())
        tensorboard_trace_handler(str(cycle_dir), worker_name=worker_name)(active_profiler)
        trace_paths = sorted(str(path) for path in cycle_dir.iterdir() if path not in before_trace and path.is_file())
        artifact: dict[str, Any] = {
            "cycle": cycle_index,
            "profiler_step_num": step_num,
            "cycle_dir": str(cycle_dir),
            "trace_dir": str(cycle_dir),
            "trace_paths": trace_paths,
        }
        # Record the trace path immediately; if summary generation itself
        # fails, the failure report still identifies the completed export.
        artifact_paths.append(artifact)
        # key_averages() is consumed here, while this cycle's events are still
        # available.  It must not be deferred until profiler.__exit__().
        artifact.update(_write_profiler_summary(active_profiler, cycle_dir, cycle_index, step_num))

    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            skip_first=int(args.profiler_skip_first),
            wait=int(args.profiler_wait),
            warmup=int(args.profiler_warmup),
            active=int(args.profiler_active),
            repeat=int(args.profiler_repeat),
        ),
        on_trace_ready=_on_trace_ready,
        record_shapes=bool(args.profiler_record_shapes),
        profile_memory=bool(args.profiler_profile_memory),
        with_stack=bool(args.profiler_with_stack),
    )


def _profiler_event_value(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def _write_profiler_summary(profiler: Any, profile_dir: Path, cycle_index: int, step_num: int) -> dict[str, Any]:
    """Write one schedule-cycle summary while its events are still live."""
    summary_txt = profile_dir / f"operator_summary_cycle{int(cycle_index):02d}_step{int(step_num):04d}.txt"
    summary_csv = profile_dir / f"operator_summary_cycle{int(cycle_index):02d}_step{int(step_num):04d}.csv"
    averages = profiler.key_averages()
    try:
        table = averages.table(sort_by="self_cuda_time_total", row_limit=-1)
    except (AttributeError, KeyError, RuntimeError, ValueError):
        table = averages.table(sort_by="self_device_time_total", row_limit=-1)
    summary_txt.write_text(table + "\n", encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("operator", "count", "self_cpu_time_us", "cpu_time_us", "self_cuda_time_us", "cuda_time_us"))
        for event in averages:
            writer.writerow((
                str(getattr(event, "key", "")),
                int(getattr(event, "count", 0)),
                _profiler_event_value(event, "self_cpu_time_total"),
                _profiler_event_value(event, "cpu_time_total"),
                _profiler_event_value(event, "self_device_time_total", "self_cuda_time_total"),
                _profiler_event_value(event, "device_time_total", "cuda_time_total"),
            ))
    return {
        "operator_summary_txt": str(summary_txt),
        "operator_summary_csv": str(summary_csv),
        "summary_cycle": int(cycle_index),
        "summary_profiler_step_num": int(step_num),
    }


def _perf_steady_summary(metrics: list[dict[str, Any]], args: argparse.Namespace, max_steps: int) -> dict[str, Any]:
    profiler_affected_steps = set(_profile_overhead_steps(args, max_steps)) if args.profiler else set()
    configured_start = max(1, int(args.steady_state_start_step))
    steady = [item for item in metrics if int(item["step"]) >= configured_start and int(item["step"]) not in profiler_affected_steps]
    host_phases = ("data_wait", "host_to_device_enqueue", "forward_enqueue", "backward_enqueue", "grad_clip_host", "optimizer_enqueue", "scheduler_host", "metrics_enqueue_host")
    device_phases = ("host_to_device", "forward", "backward", "grad_clip", "optimizer", "metrics_collectives")
    host_phase_summary = {
        phase: _distribution([float(item.get("phase_timings_host_seconds", {}).get(phase, 0.0)) for item in steady])
        for phase in host_phases
    }
    device_phase_summary = {
        phase: _distribution([float(item.get("phase_timings_device_seconds", {}).get(phase, 0.0)) for item in steady])
        for phase in device_phases
    }
    return {
        "definition": {
            "start_step_inclusive": configured_start,
            "end_step_inclusive": int(max_steps),
            "excluded_first_steps": list(range(1, configured_start)),
            "excluded_profiler_steps": sorted(profiler_affected_steps),
            "timing_scope": "rank0 timings; throughput/token counts use DDP-global reductions",
        },
        "included_steps": [int(item["step"]) for item in steady],
        "step_time_seconds": _distribution([float(item["step_time_seconds"]) for item in steady]),
        "samples_per_second": _distribution([float(item["samples_per_second"]) for item in steady]),
        "audio_seconds_per_second": _distribution([float(item["audio_seconds_per_second"]) for item in steady]),
        "multimodal_tokens_per_second": _distribution([float(item["multimodal_tokens_per_second"]) for item in steady]),
        "nonpadding_tokens_per_second": _distribution([float(item["nonpadding_tokens_per_second"]) for item in steady]),
        "answer_tokens_per_second": _distribution([float(item["answer_tokens_per_second"]) for item in steady]),
        "phase_timings_host_seconds": host_phase_summary,
        "phase_timings_device_seconds": device_phase_summary,
        "peak_gpu_memory_allocated_gib": max((float(item.get("gpu_memory_max_allocated_gib", 0.0)) for item in metrics), default=0.0),
        "peak_gpu_memory_reserved_gib": max((float(item.get("gpu_memory_max_reserved_gib", 0.0)) for item in metrics), default=0.0),
        "profiler_affected_steps": sorted(profiler_affected_steps),
    }


def _local_rank_timing_payload(rank: int, metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the compact timing payload gathered once after PERF20 training."""

    steps: list[dict[str, Any]] = []
    for item in metrics:
        host = item.get("phase_timings_host_seconds", {})
        device = item.get("phase_timings_device_seconds", {})
        steps.append({
            "step": int(item["step"]),
            "data_wait_seconds": float(host.get("data_wait", 0.0)),
            "forward_device_seconds": float(device.get("forward", 0.0)),
            "backward_device_seconds": float(device.get("backward", 0.0)),
            "step_wall_seconds": float(item["step_time_seconds"]),
        })
    return {"rank": int(rank), "steps": steps}


def _per_rank_timing_report(payloads: list[dict[str, Any]], expected_world: int) -> dict[str, Any]:
    """Summarize rank-local timings without adding collectives to measured steps."""

    ordered = sorted(payloads, key=lambda item: int(item["rank"]))
    actual_ranks = [int(item["rank"]) for item in ordered]
    expected_ranks = list(range(int(expected_world)))
    if actual_ranks != expected_ranks:
        raise RuntimeError(f"PERF20 per-rank timing ranks mismatch: expected={expected_ranks} actual={actual_ranks}")
    step_sets = [{int(step["step"]) for step in item["steps"]} for item in ordered]
    if not step_sets or any(steps != step_sets[0] for steps in step_sets[1:]):
        raise RuntimeError(f"PERF20 per-rank timing step sets mismatch: {step_sets}")

    fields = (
        "data_wait_seconds",
        "forward_device_seconds",
        "backward_device_seconds",
        "step_wall_seconds",
    )
    per_rank_summary = []
    by_step: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for payload in ordered:
        rank = int(payload["rank"])
        steps = sorted(payload["steps"], key=lambda item: int(item["step"]))
        per_rank_summary.append({
            "rank": rank,
            "step_count": len(steps),
            "distributions": {
                field: _distribution([float(step[field]) for step in steps])
                for field in fields
            },
        })
        for step in steps:
            by_step.setdefault(int(step["step"]), []).append((rank, step))

    per_step_rank_skew = []
    for step_number in sorted(by_step):
        ranked_steps = sorted(by_step[step_number], key=lambda item: item[0])
        phases: dict[str, Any] = {}
        for field in fields:
            ranked_values = [(rank, float(step[field])) for rank, step in ranked_steps]
            values = [value for _, value in ranked_values]
            slowest_rank, maximum = max(ranked_values, key=lambda item: item[1])
            fastest_rank, minimum = min(ranked_values, key=lambda item: item[1])
            phases[field] = {
                **_distribution(values),
                "minimum": minimum,
                "maximum": maximum,
                "fastest_rank": fastest_rank,
                "slowest_rank": slowest_rank,
                "max_over_min": maximum / minimum if minimum > 0.0 else None,
            }
        per_step_rank_skew.append({"step": step_number, "phases": phases})

    return {
        "collection": "one dist.gather_object after the final optimizer step",
        "included_in_step_timing": False,
        "per_step_collectives_added": 0,
        "timing_sources": {
            "data_wait_seconds": "host wall time around next(data_iter), summed across microbatches",
            "forward_device_seconds": "CUDA events, summed across microbatches and resolved by the existing end-of-step synchronize",
            "backward_device_seconds": "CUDA events, summed across microbatches and resolved by the existing end-of-step synchronize; includes DDP gradient communication dependencies",
            "step_wall_seconds": "rank-local completion-inclusive optimizer-step wall time",
        },
        "raw_by_rank": ordered,
        "per_rank_summary": per_rank_summary,
        "per_step_rank_skew": per_step_rank_skew,
    }


def _gather_perf20_rank_timings(rank: int, world: int, metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Gather all rank-local PERF20 timings exactly once after measurement."""

    local_payload = _local_rank_timing_payload(rank, metrics)
    if world > 1:
        gathered: list[dict[str, Any] | None] | None = [None] * world if rank == 0 else None
        dist.gather_object(local_payload, gathered, dst=0)
        if rank != 0:
            return None
        if gathered is None or any(item is None for item in gathered):
            raise RuntimeError("PERF20 per-rank timing gather returned an incomplete payload")
        payloads = [item for item in gathered if item is not None]
    else:
        payloads = [local_payload]
    return _per_rank_timing_report(payloads, world)


def _init_dist(args: argparse.Namespace) -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return rank, world, device


def _seed(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _load_model(args: argparse.Namespace, device: torch.device) -> tuple[AudioMeshModel, Any]:
    register_auto_class()
    from transformers import AutoTokenizer
    model_path = args.resume_from / "mesh_model" if args.resume_from else args.model_path
    tokenizer_path = args.tokenizer_path or (args.resume_from / "tokenizer" if args.resume_from else model_path)
    mesh = RecursiveLlamaForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper, htsat, provenance = _load_mellow_wrapper(args.mellow_root, args.htsat_checkpoint, device)
    model = AudioMeshModel(mesh.to(device), tokenizer, wrapper, htsat, AudioMeshConfig())
    if args.resume_from:
        audio_state = torch.load(args.resume_from / "audio_bridge.pt", map_location=device, weights_only=False)
        if not isinstance(audio_state.get("bridge"), dict) or not audio_state["bridge"]:
            raise RuntimeError("resume checkpoint has no non-empty bridge state")
        if not isinstance(audio_state.get("c2l"), dict) or not audio_state["c2l"]:
            raise RuntimeError("resume checkpoint has no non-empty c2l state")
        model.bridge.load_state_dict(audio_state["bridge"], strict=True)
        c2l = getattr(model.htsat_wrapper, "c2l", None)
        if c2l is None:
            raise RuntimeError("resume checkpoint requires wrapper.c2l")
        c2l.load_state_dict(audio_state["c2l"], strict=True)
    model._audio_provenance = provenance
    return model.to(device), tokenizer


def _trainable_state(model: AudioMeshModel) -> dict[str, Any]:
    c2l = getattr(model.htsat_wrapper, "c2l", None)
    return {"bridge": model.bridge.state_dict(), "c2l": c2l.state_dict() if c2l is not None else {}}


def _save_checkpoint(path: Path, model: AudioMeshModel, tokenizer: Any, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR, step: int, epoch: int, batch_in_epoch: int, args: argparse.Namespace, manifest_hash: str, rng_states_by_rank: dict[str, Any], total_steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {path}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)))
    published = False
    try:
        model.mesh_model.save_pretrained(temporary / "mesh_model", safe_serialization=False)
        tokenizer.save_pretrained(temporary / "tokenizer")
        torch.save(_trainable_state(model), temporary / "audio_bridge.pt")
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "global_step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch, "rng_states_by_rank": rng_states_by_rank}, temporary / "training_state.pt")
        provenance = getattr(model, "_audio_provenance", {})
        if int(model.mesh_model.config.hidden_size) != MESH_HIDDEN_SIZE:
            raise RuntimeError("refusing to save checkpoint with non-contract MeSH hidden size")
        if model.last_audio_tokens_per_clip is not None and tuple(model.last_audio_tokens_per_clip) != (AUDIO_TOKENS_PER_CLIP, AUDIO_TOKENS_PER_CLIP):
            raise RuntimeError("refusing to save checkpoint with non-contract audio token count")
        config = {
            "architecture_contract": ARCHITECTURE_CONTRACT,
            "mapper_contract": MAPPER_CONTRACT,
            "mapper_initialization": "random_c2l_and_xavier_projection",
            "mesh_hidden_size": MESH_HIDDEN_SIZE,
            "audio_tokens_per_clip": AUDIO_TOKENS_PER_CLIP,
            "audio_prefix_tokens_with_separators": AUDIO_PREFIX_TOKENS,
            "manifest_sha256": manifest_hash,
            "htsat_checkpoint": str(args.htsat_checkpoint.resolve()),
            "mellow_root": str(args.mellow_root.resolve()),
            "mellow_provenance": provenance,
            "mesh_model_path": str(args.model_path),
            "epochs": args.epochs,
            "max_lr": args.max_lr,
            "min_lr": args.min_lr,
            "warmup_steps": args.warmup_steps,
            "total_steps": total_steps,
            "global_step": step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "world_size": args.world_size,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "save_every": args.save_every,
            "checkpoint_retention": args.checkpoint_retention,
        }
        (temporary / "audio_mesh_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (temporary / "checkpoint_complete.json").write_text(json.dumps({"status": "complete", "global_step": step, "required": ["mesh_model", "tokenizer", "audio_bridge.pt", "training_state.pt", "audio_mesh_config.json"]}, indent=2) + "\n", encoding="utf-8")
        required = (temporary / "mesh_model" / "config.json", temporary / "tokenizer", temporary / "audio_bridge.pt", temporary / "training_state.pt", temporary / "audio_mesh_config.json", temporary / "checkpoint_complete.json")
        if any(not item.exists() for item in required):
            raise RuntimeError("refusing to publish incomplete checkpoint")
        temporary.replace(path)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def _prune_checkpoints(output_dir: Path, retention: int) -> list[str]:
    """Keep only the newest complete checkpoints inside this exact run dir."""
    if retention <= 0:
        raise ValueError("checkpoint_retention must be positive")
    output_resolved = output_dir.resolve()
    # Remove only our own interrupted staging directories; incomplete
    # ``checkpoint-*`` directories are never treated as complete or deleted.
    for temporary in output_dir.glob(".checkpoint-*.tmp"):
        if temporary.is_dir() and temporary.resolve().parent == output_resolved:
            shutil.rmtree(temporary, ignore_errors=True)
    complete: list[tuple[int, Path]] = []
    for candidate in output_dir.glob("checkpoint-*"):
        suffix = candidate.name.removeprefix("checkpoint-")
        if not candidate.is_dir() or not suffix.isdigit():
            continue
        resolved = candidate.resolve()
        if resolved.parent != output_resolved:
            raise RuntimeError(f"refusing to prune checkpoint outside output directory: {resolved}")
        marker = candidate / "checkpoint_complete.json"
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("status") == "complete":
            complete.append((int(suffix), candidate))
    complete.sort(key=lambda item: item[0])
    for _, stale in complete[:-retention]:
        shutil.rmtree(stale)
    return [str(path) for _, path in complete[-retention:]]


def _load_training_state(path: Path, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR) -> dict[str, Any]:
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state


def _validate_resume_artifacts(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"resume checkpoint directory not found: {path}")
    _audit_saved_checkpoint(path)
    marker = json.loads((path / "checkpoint_complete.json").read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise RuntimeError("resume checkpoint marker is not complete")
    config = json.loads((path / "audio_mesh_config.json").read_text(encoding="utf-8"))
    for key in ("architecture_contract", "mapper_contract", "mesh_hidden_size", "audio_tokens_per_clip", "audio_prefix_tokens_with_separators", "manifest_sha256", "htsat_checkpoint", "mellow_root", "mellow_provenance", "world_size", "micro_batch_size", "gradient_accumulation_steps", "epochs", "max_lr", "min_lr", "warmup_steps", "total_steps", "save_every", "checkpoint_retention"):
        if key not in config:
            raise RuntimeError(f"resume checkpoint config missing {key}")
    if config["architecture_contract"] != ARCHITECTURE_CONTRACT or config["mapper_contract"] != MAPPER_CONTRACT:
        raise RuntimeError("resume checkpoint architecture/mapper contract mismatch")
    if int(config["mesh_hidden_size"]) != MESH_HIDDEN_SIZE or int(config["audio_tokens_per_clip"]) != AUDIO_TOKENS_PER_CLIP or int(config["audio_prefix_tokens_with_separators"]) != AUDIO_PREFIX_TOKENS:
        raise RuntimeError("resume checkpoint audio shape contract mismatch")
    audio = torch.load(path / "audio_bridge.pt", map_location="cpu", weights_only=False)
    if not isinstance(audio.get("bridge"), dict) or not audio["bridge"] or not isinstance(audio.get("c2l"), dict) or not audio["c2l"]:
        raise RuntimeError("resume checkpoint must contain non-empty bridge and c2l state")
    return config


def _audit_saved_checkpoint(path: Path) -> dict[str, Any]:
    required = ("mesh_model/config.json", "tokenizer/tokenizer_config.json", "audio_bridge.pt", "training_state.pt", "audio_mesh_config.json", "checkpoint_complete.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"composite checkpoint missing files: {missing}")
    marker = json.loads((path / "checkpoint_complete.json").read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise RuntimeError("composite checkpoint completion marker is invalid")
    training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    config = json.loads((path / "audio_mesh_config.json").read_text(encoding="utf-8"))
    for key in ("optimizer", "scheduler", "global_step", "epoch", "batch_in_epoch", "rng_states_by_rank"):
        if key not in training:
            raise RuntimeError(f"composite checkpoint training_state missing {key}")
    audio = torch.load(path / "audio_bridge.pt", map_location="cpu", weights_only=False)
    if "bridge" not in audio or "c2l" not in audio or not audio["bridge"] or not audio["c2l"]:
        raise RuntimeError("composite checkpoint missing bridge/c2l state")
    if not training["rng_states_by_rank"]:
        raise RuntimeError("composite checkpoint has no per-rank RNG states")
    expected_rng = {str(index) for index in range(int(config.get("world_size", 0)))}
    actual_rng = {str(key) for key in training["rng_states_by_rank"]}
    if actual_rng != expected_rng:
        raise RuntimeError(f"composite checkpoint RNG ranks mismatch: expected={sorted(expected_rng)} actual={sorted(actual_rng)}")
    return {"passed": True, "path": str(path), "global_step": int(training["global_step"]), "epoch": int(training["epoch"]), "batch_in_epoch": int(training["batch_in_epoch"]), "rng_ranks": sorted(actual_rng), "required_files": list(required)}


def _router_stats(model: AudioMeshModel) -> dict[str, Any]:
    owner = model.mesh_model.model
    return getattr(owner, "last_routing_stats", {})


def _mesh_runtime_gradient_audit(model: AudioMeshModel, *, require_router_stats: bool = True) -> dict[str, Any]:
    """Validate one forward/backward traversed both MeSH middle loops.

    PERF20 intentionally disables per-router CPU statistics so its timing path
    contains only the unified end-of-step CUDA synchronization.  Router
    parameter gradients and the full logical trace remain mandatory there;
    only the optional statistics side channel is skipped.
    """
    mesh_owner = model.mesh_model.model
    trace = list(getattr(mesh_owner, "last_forward_trace", []))
    expected_trace = [
        *({"logical_index": i, "physical_index": i} for i in range(5)),
        *({"logical_index": 5 + i, "physical_index": 5 + i} for i in range(10)),
        *({"logical_index": 15 + i, "physical_index": 5 + i} for i in range(10)),
        *({"logical_index": 25 + i, "physical_index": 15 + i} for i in range(5)),
    ]
    trace_ok = trace == expected_trace
    input_refs = list(getattr(mesh_owner, "last_core_input_refs", []))
    output_refs = list(getattr(mesh_owner, "last_core_output_refs", []))
    loop_input_grads = [
        ref.grad is not None and torch.isfinite(ref.grad).all() for ref in input_refs
    ]
    loop_output_grads = [
        ref.grad is not None and torch.isfinite(ref.grad).all() for ref in output_refs
    ]
    router_grads = {
        name: any(p.grad is not None and torch.isfinite(p.grad).all() for p in router.parameters())
        for group, routers in (("write", mesh_owner.write_routers), ("read", mesh_owner.read_routers))
        for name, router in ((f"{group}_{index}", router) for index, router in enumerate(routers))
    }
    core_grads = [
        any(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
        for layer in mesh_owner.layers[5:15]
    ]
    result = {
        "trace_length": len(trace),
        "expected_trace_length": 30,
        "trace_matches_5_10_10_5": trace_ok,
        "loop_input_finite_gradients": loop_input_grads,
        "loop_output_finite_gradients": loop_output_grads,
        "both_middle_loops_have_finite_gradients": len(input_refs) == 2 and len(output_refs) == 2 and all(loop_input_grads) and all(loop_output_grads),
        "router_stats_nonempty": len(_router_stats(model)) == 6,
        "router_stats_required": bool(require_router_stats),
        "router_stats_names": sorted(_router_stats(model)),
        "router_finite_gradients": router_grads,
        "all_router_gradients_finite": all(router_grads.values()),
        "middle_core_layer_gradients_finite": core_grads,
        "all_middle_core_gradients_finite": all(core_grads),
    }
    required_checks = (
        result["trace_matches_5_10_10_5"],
        result["both_middle_loops_have_finite_gradients"],
        result["all_router_gradients_finite"],
        result["all_middle_core_gradients_finite"],
    )
    if require_router_stats:
        required_checks = (*required_checks, result["router_stats_nonempty"])
    if not all(required_checks):
        raise RuntimeError(f"MeSH runtime gradient/router/trace audit failed: {result}")
    return result


def _make_scheduler(optimizer: torch.optim.Optimizer, *, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    if max_lr <= 0 or min_lr < 0 or min_lr > max_lr:
        raise ValueError("learning rates must satisfy 0 <= min_lr <= max_lr and max_lr > 0")
    def scale(step: int) -> float:
        if step < warmup_steps:
            return min(1.0, float(step + 1) / max(1, warmup_steps))
        # LambdaLR's ``step`` is zero at construction and increments after an
        # optimizer update.  The LR used by update N is therefore represented
        # by scheduler index N-1; use ``step + 1`` for the contract's update
        # index so the final update really uses min_lr.
        progress = min(1.0, max(0.0, float(step + 1 - warmup_steps) / max(1, total_steps - warmup_steps)))
        ratio = min_lr / max_lr
        return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state().cpu(),
        "cuda": torch.cuda.get_rng_state(device).cpu(),
        "python": random.getstate(),
    }


def _restore_rng_state(state: dict[str, Any] | None, device: torch.device) -> None:
    if not state:
        return
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"], device=device)
    if state.get("python") is not None:
        random.setstate(state["python"])


def _gather_rng_states(rank: int, world: int, device: torch.device) -> dict[str, Any]:
    local = _rng_state(device)
    gathered: list[Any] = [None for _ in range(world)]
    if world > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered[0] = local
    return {str(index): value for index, value in enumerate(gathered)}


def _skip_batches(data_iter: Any, count: int) -> None:
    for _ in range(max(0, int(count))):
        next(data_iter)


def _actual_resume_audit(path: Path, args: argparse.Namespace, batch_cpu: dict[str, Any], device: torch.device, expected_step: int, expected_lr: float, rank: int) -> dict[str, Any]:
    """Reload the complete composite checkpoint and execute one real batch."""
    saved_rng = _rng_state(device)
    reload_args = copy.copy(args)
    reload_args.resume_from = path
    reload_model, reload_tokenizer = _load_model(reload_args, device)
    try:
        saved_config = json.loads((path / "audio_mesh_config.json").read_text(encoding="utf-8"))
        optimizer = torch.optim.AdamW([p for p in reload_model.parameters() if p.requires_grad], lr=float(saved_config["max_lr"]), betas=(0.9, 0.95), weight_decay=0.1)
        scheduler = _make_scheduler(optimizer, max_lr=float(saved_config["max_lr"]), min_lr=float(saved_config.get("min_lr", 0.0)), warmup_steps=int(saved_config["warmup_steps"]), total_steps=max(1, int(saved_config.get("total_steps", expected_step))))
        state = _load_training_state(path, optimizer, scheduler)
        if int(state["global_step"]) != int(expected_step):
            raise RuntimeError(f"reloaded global step mismatch: {state['global_step']} != {expected_step}")
        loaded_lr = float(optimizer.param_groups[0]["lr"])
        if not math.isclose(loaded_lr, float(expected_lr), rel_tol=1e-6, abs_tol=1e-10):
            raise RuntimeError(f"reloaded learning-rate mismatch: {loaded_lr} != {expected_lr}")
        rank_state = state["rng_states_by_rank"].get(str(rank)) or state["rng_states_by_rank"].get("0")
        _restore_rng_state(rank_state, device)
        reload_model.train()
        reload_model.mesh_model.model.routing_stats_mode = True
        reload_model.mesh_model.model.gradient_audit_mode = True
        moved = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch_cpu.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = reload_model(**{key: value for key, value in moved.items() if key not in {"row_indices", "audio2_reused"}})
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("reloaded checkpoint produced a nonfinite loss")
        labels = reload_model.last_labels
        prefix_length = int(reload_model.last_prefix_length or 0)
        text_ids = moved["text_ids"]
        prompt_lengths = moved["prompt_lengths"]
        answer_lengths = moved["answer_lengths"]
        if labels is None:
            raise RuntimeError("reloaded checkpoint did not expose labels")
        for row_index in range(text_ids.shape[0]):
            prompt_length = int(prompt_lengths[row_index].item())
            answer_length = int(answer_lengths[row_index].item())
            answer_start = prefix_length + prompt_length
            answer_end = answer_start + answer_length
            if bool((labels[row_index, :answer_start] != -100).any()) or bool((labels[row_index, answer_end:] != -100).any()):
                raise RuntimeError("reloaded checkpoint violated answer-only label mask")
            if not torch.equal(labels[row_index, answer_start:answer_end], text_ids[row_index, prompt_length:prompt_length + answer_length]):
                raise RuntimeError("reloaded checkpoint answer labels are misaligned with unified text")
        output.loss.backward()
        runtime_audit = _mesh_runtime_gradient_audit(reload_model)
        trainable_audit = reload_model.trainable_parameter_audit()
        if not trainable_audit["training_mode_contract"]:
            raise RuntimeError(f"reloaded checkpoint training mode contract failed: {trainable_audit}")
        bridge_grad = {
            name: p.grad is not None and torch.isfinite(p.grad).all()
            for name, p in reload_model.bridge.named_parameters()
        }
        c2l = getattr(reload_model.htsat_wrapper, "c2l", None)
        c2l_grad = {
            name: p.grad is not None and torch.isfinite(p.grad).all()
            for name, p in c2l.named_parameters()
        } if c2l is not None else {}
        if not bridge_grad or not all(bridge_grad.values()) or not c2l_grad or not all(c2l_grad.values()):
            raise RuntimeError(f"reloaded checkpoint audio gradients failed: bridge={bridge_grad} c2l={c2l_grad}")
        grad_norm = torch.nn.utils.clip_grad_norm_(reload_model.parameters(), 0.5, error_if_nonfinite=True)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("reloaded checkpoint produced a nonfinite gradient")
        return {"passed": True, "global_step": int(state["global_step"]), "epoch": int(state["epoch"]), "batch_in_epoch": int(state["batch_in_epoch"]), "learning_rate": loaded_lr, "loss": float(output.loss.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu()), "optimizer_state_loaded": bool(optimizer.state_dict()["state"]), "scheduler_last_epoch": int(scheduler.last_epoch), "forward_backward": True, "answer_only_labels": True, "training_mode_contract": trainable_audit, "audio_gradients": {"bridge": bridge_grad, "c2l": c2l_grad}, "mesh_runtime_gradient_audit": runtime_audit}
    finally:
        del reload_model
        gc.collect()
        torch.cuda.empty_cache()
        _restore_rng_state(saved_rng, device)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank, world, device = _init_dist(args)
    _seed(args.seed, rank)
    report: dict[str, Any] = {"stage": f"{args.gate.lower()}_audio_5_10x2_5_mesh_mellow", "status": "FAIL", "configuration": vars(args), "rank": rank, "world_size": world, "checks": [], "warnings": [], "hard_failures": [], "environment": _environment_report(device, world)}
    profiler: Any | None = None
    profiler_artifacts: list[dict[str, Any]] = []
    perf_output_preexisting = False
    try:
        if world != args.world_size:
            raise RuntimeError(f"world size mismatch: launcher={world} requested={args.world_size}")
        if args.gate != "PERF20" and (args.save_every <= 0 or args.checkpoint_retention <= 0):
            raise ValueError("save_every and checkpoint_retention must be positive")
        if args.gate == "PERF20":
            canonical = {
                "world_size": 8,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "num_workers": 0,
                "epochs": 1,
                "max_lr": 1e-3,
                "min_lr": 0.0,
            }
            for key, expected in canonical.items():
                if getattr(args, key) != expected:
                    raise ValueError(f"PERF20 requires {key}={expected}, got {getattr(args, key)}")
            if args.resume_from is not None:
                raise ValueError("PERF20 starts from the text MeSH checkpoint and does not accept --resume-from")
            if args.max_steps is not None and int(args.max_steps) != PERF20_STEPS:
                raise ValueError(f"PERF20 requires --max-steps {PERF20_STEPS}")
            args.max_steps = PERF20_STEPS
            if int(args.steady_state_start_step) < 1 or int(args.steady_state_start_step) > PERF20_STEPS:
                raise ValueError(f"PERF20 steady-state start must be in [1, {PERF20_STEPS}]")
            if args.output_dir.exists():
                perf_output_preexisting = True
                raise FileExistsError(f"PERF20 refuses to reuse an existing output directory: {args.output_dir}")
            _validate_profiler_options(args, PERF20_STEPS)
            report["profiler"] = {
                "enabled": bool(args.profiler),
                "rank": 0,
                "precision": "bf16",
                "activities": ["CPU", "CUDA"],
                "step_granularity": "optimizer_step",
                "schedule": {
                    "skip_first": int(args.profiler_skip_first),
                    "wait": int(args.profiler_wait),
                    "warmup": int(args.profiler_warmup),
                    "active": int(args.profiler_active),
                    "repeat": int(args.profiler_repeat),
                },
                "options": {
                    "with_stack": bool(args.profiler_with_stack),
                    "profile_memory": bool(args.profiler_profile_memory),
                    "record_shapes": bool(args.profiler_record_shapes),
                },
                "profile_dir": str(args.output_dir / "profile"),
                "worker_name": "rank0",
                "active_steps": _profile_active_steps(args, PERF20_STEPS),
                "overhead_steps_excluded_from_steady_state": _profile_overhead_steps(args, PERF20_STEPS),
                "artifacts": profiler_artifacts,
            }
            report["correctness_audit"] = {"status": "pending_until_first_optimizer_step", "mesh_runtime_gradient_audit": None}
        elif args.gate == "FORMAL":
            canonical = {
                "world_size": 8,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "epochs": 3,
                "max_lr": 1e-3,
                "min_lr": 0.0,
                "save_every": 500,
                "checkpoint_retention": 4,
            }
            for key, expected in canonical.items():
                if getattr(args, key) != expected:
                    raise ValueError(f"FORMAL requires {key}={expected}, got {getattr(args, key)}")
            if args.max_steps is not None:
                raise ValueError("FORMAL does not accept --max-steps; use STAGE5/STAGE7 for bounded smoke")
        elif args.max_steps is not None and args.max_steps <= 0:
            raise ValueError("bounded smoke --max-steps must be positive")
        if args.gate != "PERF20" and args.profiler:
            raise ValueError("--profiler is isolated to PERF20; standard smoke/formal gates are unchanged")
        model, tokenizer = _load_model(args, device)
        model.train()
        if not model.trainable_parameter_audit()["training_mode_contract"]:
            raise RuntimeError(f"audio training mode contract failed: {model.trainable_parameter_audit()}")
        # PERF20 must measure the actual training path.  Router statistics call
        # ``.cpu()`` inside every router forward, which introduces extra CUDA
        # synchronizations and would invalidate the single-sync timing
        # contract.  Keep the statistics for the normal audit/training gates,
        # but disable them for the performance gate.
        if args.gate == "PERF20":
            model.mesh_model.model.routing_stats_mode = False
        else:
            model.mesh_model.model.routing_stats_mode = True
        model.mesh_model.model.gradient_audit_mode = True
        dataset = ReasonAQADataset(args.train_manifest, tokenizer)
        # Shuffle deterministically per epoch; set_epoch(epoch) below changes
        # the permutation while DistributedSampler keeps rank partitions
        # disjoint and equally sized.
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        loader = DataLoader(dataset, batch_size=args.micro_batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=lambda rows: collate_reasonaqa(rows, tokenizer))
        if args.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        # Only complete accumulation windows become optimizer steps.  This
        # keeps every optimizer step at the configured effective batch size;
        # a short tail of micro-batches is intentionally dropped at the epoch
        # boundary and will be reshuffled into the next epoch.
        steps_epoch = len(loader) // args.gradient_accumulation_steps
        dropped_microbatches = len(loader) % args.gradient_accumulation_steps
        if steps_epoch <= 0:
            raise ValueError(f"loader has {len(loader)} batches, fewer than one accumulation window of {args.gradient_accumulation_steps}")
        formal_steps = steps_epoch * args.epochs
        max_steps = args.max_steps or (2 if args.gate == "STAGE5" else 10 if args.gate == "STAGE7" else PERF20_STEPS if args.gate == "PERF20" else formal_steps)
        total_steps = formal_steps if args.gate == "FORMAL" else max_steps
        required_warmup = math.ceil(total_steps * 0.05)
        if args.gate == "FORMAL" and args.warmup_steps is not None and args.warmup_steps != required_warmup:
            raise ValueError(f"FORMAL warmup must equal ceil(actual_total_steps*0.05)={required_warmup}, got {args.warmup_steps}")
        args.warmup_steps = required_warmup
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.max_lr, betas=(0.9, 0.95), weight_decay=0.1)
        scheduler = _make_scheduler(optimizer, max_lr=args.max_lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps, total_steps=total_steps)
        start_step, start_epoch, start_batch_in_epoch = 0, 0, 0
        if args.resume_from:
            saved_config = _validate_resume_artifacts(args.resume_from)
            # Smoke resume may extend the bounded run (20 -> 22) while
            # retaining the original checkpoint's optimizer/scheduler
            # contract.  FORMAL never permits this; its total schedule must
            # remain the canonical three-epoch value.
            if args.gate != "FORMAL" and int(saved_config["total_steps"]) <= int(max_steps):
                total_steps = int(saved_config["total_steps"])
                args.warmup_steps = int(saved_config["warmup_steps"])
                scheduler = _make_scheduler(
                    optimizer,
                    max_lr=float(saved_config["max_lr"]),
                    min_lr=float(saved_config["min_lr"]),
                    warmup_steps=int(saved_config["warmup_steps"]),
                    total_steps=max(1, int(saved_config["total_steps"])),
                )
            saved_manifest_hash = str(saved_config["manifest_sha256"])
            current_manifest_hash = hashlib.sha256(args.train_manifest.read_bytes()).hexdigest()
            if saved_manifest_hash != current_manifest_hash:
                raise RuntimeError("resume manifest SHA256 mismatch")
            for key in ("world_size", "micro_batch_size", "gradient_accumulation_steps", "epochs", "max_lr", "min_lr", "save_every", "checkpoint_retention"):
                if str(saved_config[key]) != str(getattr(args, key)):
                    raise RuntimeError(f"resume {key} mismatch: saved={saved_config[key]} current={getattr(args, key)}")
            if int(saved_config["warmup_steps"]) != int(args.warmup_steps):
                raise RuntimeError(f"resume warmup_steps mismatch: saved={saved_config['warmup_steps']} current={args.warmup_steps}")
            if args.gate == "FORMAL" and int(saved_config["total_steps"]) != int(total_steps):
                raise RuntimeError(f"FORMAL resume total_steps mismatch: saved={saved_config['total_steps']} current={total_steps}")
            if args.gate != "FORMAL" and int(saved_config["total_steps"]) > int(total_steps):
                raise RuntimeError(f"resume smoke cannot shorten saved schedule: saved={saved_config['total_steps']} current={total_steps}")
            if Path(str(saved_config["htsat_checkpoint"])).resolve() != args.htsat_checkpoint.resolve():
                raise RuntimeError("resume HTSAT checkpoint mismatch")
            if Path(str(saved_config["mellow_root"])).resolve() != args.mellow_root.resolve():
                raise RuntimeError("resume Mellow root mismatch")
            if not saved_config.get("mellow_provenance"):
                raise RuntimeError("resume checkpoint has no Mellow provenance")
            if getattr(model, "_audio_provenance", {}).get("mellow_htsat_sha256") != saved_config["mellow_provenance"].get("mellow_htsat_sha256"):
                raise RuntimeError("resume Mellow provenance mismatch")
            state = _load_training_state(args.resume_from, optimizer, scheduler)
            start_step = int(state["global_step"])
            start_epoch = int(state.get("epoch", 0))
            start_batch_in_epoch = int(state.get("batch_in_epoch", 0))
            if start_batch_in_epoch % args.gradient_accumulation_steps != 0:
                raise RuntimeError("resume batch cursor is not aligned to the requested gradient accumulation steps")
            rank_rng = state.get("rng_states_by_rank", {}).get(str(rank)) or state.get("rng_states_by_rank", {}).get("0")
            if str(rank) not in state.get("rng_states_by_rank", {}):
                raise RuntimeError(f"resume checkpoint has no RNG state for rank {rank}")
            _restore_rng_state(rank_rng, device)
        ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False) if world > 1 else model
        metrics: list[dict[str, Any]] = []
        runtime_gradient_audit: dict[str, Any] | None = None
        optimizer_step = start_step
        epoch = start_epoch
        batch_in_epoch = start_batch_in_epoch
        perf_mode = args.gate == "PERF20"
        if args.gate == "PERF20" and rank == 0 and args.profiler:
            profile_dir = args.output_dir / "profile"
            profiler = _make_profiler(args, profile_dir, profiler_artifacts)
            profiler.__enter__()
            report["profiler"] = {
                "enabled": True,
                "rank": 0,
                "precision": "bf16",
                "activities": ["CPU", "CUDA"],
                "step_granularity": "optimizer_step",
                "schedule": {
                    "skip_first": int(args.profiler_skip_first),
                    "wait": int(args.profiler_wait),
                    "warmup": int(args.profiler_warmup),
                    "active": int(args.profiler_active),
                    "repeat": int(args.profiler_repeat),
                },
                "options": {
                    "with_stack": bool(args.profiler_with_stack),
                    "profile_memory": bool(args.profiler_profile_memory),
                    "record_shapes": bool(args.profiler_record_shapes),
                },
                "profile_dir": str(profile_dir),
                "worker_name": "rank0",
                "active_steps": _profile_active_steps(args, max_steps),
                "overhead_steps_excluded_from_steady_state": _profile_overhead_steps(args, max_steps),
                "artifacts": profiler_artifacts,
            }
        elif args.gate == "PERF20":
            report["profiler"] = {
                "enabled": False,
                "rank": 0,
                "precision": "bf16",
                "activities": ["CPU", "CUDA"],
                "step_granularity": "optimizer_step",
                "schedule": {
                    "skip_first": int(args.profiler_skip_first),
                    "wait": int(args.profiler_wait),
                    "warmup": int(args.profiler_warmup),
                    "active": int(args.profiler_active),
                    "repeat": int(args.profiler_repeat),
                },
                "options": {
                    "with_stack": bool(args.profiler_with_stack),
                    "profile_memory": bool(args.profiler_profile_memory),
                    "record_shapes": bool(args.profiler_record_shapes),
                },
                "profile_dir": str(args.output_dir / "profile"),
                "worker_name": "rank0",
                "active_steps": _profile_active_steps(args, max_steps),
                "overhead_steps_excluded_from_steady_state": _profile_overhead_steps(args, max_steps),
                "artifacts": profiler_artifacts,
            }
        if batch_in_epoch >= len(loader):
            epoch += batch_in_epoch // len(loader)
            batch_in_epoch = batch_in_epoch % len(loader)
        while optimizer_step < max_steps:
            sampler.set_epoch(epoch)
            data_iter = iter(loader)
            _skip_batches(data_iter, batch_in_epoch)
            completed_optimizer_steps = batch_in_epoch // args.gradient_accumulation_steps
            for _ in range(completed_optimizer_steps, steps_epoch):
                if optimizer_step >= max_steps:
                    break
                step_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                optimizer.zero_grad(set_to_none=True)
                phase_events = _CudaPhaseEvents(enabled=perf_mode)
                data_wait_seconds = 0.0
                host_phase_seconds: dict[str, float] = {}
                local_answer_tokens = 0
                last_local_answer_tokens = 0
                local_text_nonpadding_tokens = 0
                local_multimodal_tokens = 0
                local_max_sequence_length = 0
                microbatch_metrics: list[dict[str, int]] = []
                for micro in range(args.gradient_accumulation_steps):
                    if perf_mode:
                        data_wait_started = time.perf_counter()
                        with record_function("data_wait"):
                            batch = next(data_iter)
                        data_wait_seconds += time.perf_counter() - data_wait_started
                        text_nonpadding = int(batch["text_attention_mask"].sum().item())
                        answer_nonpadding = int(batch["answer_attention_mask"].sum().item())
                        last_local_answer_tokens = answer_nonpadding
                        batch_size = int(batch["text_ids"].shape[0])
                        sequence_length = int(AUDIO_PREFIX_TOKENS + batch["text_ids"].shape[1])
                        multimodal_tokens = int(AUDIO_PREFIX_TOKENS * batch_size + text_nonpadding)
                        local_answer_tokens += answer_nonpadding
                        local_text_nonpadding_tokens += text_nonpadding
                        local_multimodal_tokens += multimodal_tokens
                        local_max_sequence_length = max(local_max_sequence_length, sequence_length)
                        microbatch_metrics.append({
                            "micro": int(micro),
                            "batch_size": batch_size,
                            "sequence_length": sequence_length,
                            "text_nonpadding_tokens": text_nonpadding,
                            "answer_tokens": answer_nonpadding,
                            "multimodal_tokens": multimodal_tokens,
                        })
                    else:
                        batch = next(data_iter)
                    # All legacy gates retain their original final-microbatch
                    # CPU snapshot for checkpoint/reload audits.  PERF20 is
                    # the only gate that deliberately avoids this copy.
                    if not perf_mode and micro == args.gradient_accumulation_steps - 1:
                        batch_cpu = {key: (value.detach().cpu().clone() if torch.is_tensor(value) else value) for key, value in batch.items()}
                    if perf_mode:
                        host_to_device_started = time.perf_counter()
                        with record_function("host_to_device"):
                            batch = phase_events.measure("host_to_device", lambda: {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()})
                        host_phase_seconds["host_to_device_enqueue"] = host_phase_seconds.get("host_to_device_enqueue", 0.0) + (time.perf_counter() - host_to_device_started)
                    else:
                        batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
                    sync = contextlib.nullcontext() if not hasattr(ddp, "no_sync") or micro == args.gradient_accumulation_steps - 1 else ddp.no_sync()
                    with sync:
                        def _forward() -> Any:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                return ddp(**{k: v for k, v in batch.items() if k not in {"row_indices", "audio2_reused"}})
                        if perf_mode:
                            forward_started = time.perf_counter()
                            with record_function("forward"):
                                output = phase_events.measure("forward", _forward)
                            host_phase_seconds["forward_enqueue"] = host_phase_seconds.get("forward_enqueue", 0.0) + (time.perf_counter() - forward_started)
                        else:
                            output = _forward()
                        if output.loss is None or not torch.isfinite(output.loss):
                            raise RuntimeError("nonfinite audio MeSH loss")
                        if perf_mode:
                            backward_started = time.perf_counter()
                            with record_function("backward"):
                                phase_events.measure("backward", lambda: (output.loss / args.gradient_accumulation_steps).backward())
                            host_phase_seconds["backward_enqueue"] = host_phase_seconds.get("backward_enqueue", 0.0) + (time.perf_counter() - backward_started)
                        else:
                            (output.loss / args.gradient_accumulation_steps).backward()
                owner = ddp.module if hasattr(ddp, "module") else ddp
                if perf_mode:
                    grad_clip_started = time.perf_counter()
                    with record_function("grad_clip"):
                        grad_norm = phase_events.measure("grad_clip", lambda: torch.nn.utils.clip_grad_norm_(ddp.parameters(), 0.5, error_if_nonfinite=True))
                    host_phase_seconds["grad_clip_host"] = time.perf_counter() - grad_clip_started
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 0.5, error_if_nonfinite=True)
                if runtime_gradient_audit is None:
                    runtime_gradient_audit = _mesh_runtime_gradient_audit(
                        owner,
                        require_router_stats=args.gate != "PERF20",
                    )
                    owner.mesh_model.model.gradient_audit_mode = False
                lr_before_optimizer_step = float(optimizer.param_groups[0]["lr"])
                optimizer_step += 1
                if perf_mode:
                    optimizer_started = time.perf_counter()
                    with record_function("optimizer"):
                        phase_events.measure("optimizer", optimizer.step)
                    host_phase_seconds["optimizer_enqueue"] = time.perf_counter() - optimizer_started
                    scheduler_started = time.perf_counter()
                    with record_function("scheduler"):
                        # scheduler.step() is a CPU/Python operation.  Do not
                        # label a near-zero CUDA event as its full duration.
                        scheduler.step()
                    host_phase_seconds["scheduler_host"] = time.perf_counter() - scheduler_started
                else:
                    optimizer.step()
                    scheduler.step()
                global_samples = int(args.micro_batch_size * world * args.gradient_accumulation_steps)
                if perf_mode:
                    metrics_started = time.perf_counter()
                    with record_function("metrics"):
                        def _collect_metrics() -> tuple[torch.Tensor, torch.Tensor]:
                            reduced_local = torch.tensor([local_answer_tokens, local_text_nonpadding_tokens, local_multimodal_tokens, last_local_answer_tokens], dtype=torch.long, device=device)
                            max_sequence_local = torch.tensor(local_max_sequence_length, dtype=torch.long, device=device)
                            if world > 1:
                                with record_function("DDP/collectives"):
                                    dist.all_reduce(reduced_local, op=dist.ReduceOp.SUM)
                                    dist.all_reduce(max_sequence_local, op=dist.ReduceOp.MAX)
                            return reduced_local, max_sequence_local
                        reduced, max_sequence = phase_events.measure("metrics_collectives", _collect_metrics)
                    # This is intentionally named enqueue: CUDA/NCCL may still
                    # be running.  The device event below is resolved only by
                    # the single end-of-step synchronize.
                    host_phase_seconds["metrics_enqueue_host"] = time.perf_counter() - metrics_started
                else:
                    answer_tensor = torch.tensor(int(batch["answer_attention_mask"].sum().item()), dtype=torch.long, device=device)
                    if world > 1:
                        dist.all_reduce(answer_tensor, op=dist.ReduceOp.SUM)
                    reduced = torch.tensor([int(answer_tensor.item()), 0, 0, int(answer_tensor.item())], dtype=torch.long, device=device)
                    max_sequence = torch.tensor(0, dtype=torch.long, device=device)
                phase_timings_device = phase_events.seconds(device)
                phase_timings_host = {}
                if perf_mode:
                    phase_timings_host = {"data_wait": float(data_wait_seconds), **{key: float(value) for key, value in host_phase_seconds.items()}}
                elapsed = max(time.perf_counter() - step_started, 1e-9)
                answer_tokens = int(reduced[0].item())
                reported_answer_tokens = answer_tokens if args.gate == "PERF20" else int(reduced[3].item())
                text_nonpadding_tokens = int(reduced[1].item()) if perf_mode else 0
                multimodal_tokens = int(reduced[2].item()) if perf_mode else 0
                max_sequence_length = int(max_sequence.item()) if perf_mode else 0
                item = {"step": optimizer_step, "total_steps": max_steps, "progress_percent": 100.0 * optimizer_step / max(1, max_steps), "epoch": epoch, "batch_in_epoch": batch_in_epoch, "steps_per_epoch": steps_epoch, "loss": float(output.loss.detach().cpu()), "lr": float(optimizer.param_groups[0]["lr"]), "lr_before_optimizer_step": lr_before_optimizer_step, "grad_norm": float(grad_norm), "effective_answer_tokens": reported_answer_tokens, "aggregated_answer_tokens": answer_tokens, "nonpadding_tokens": text_nonpadding_tokens, "multimodal_tokens": multimodal_tokens, "max_sequence_length": max_sequence_length, "microbatches": microbatch_metrics, "phase_timings": {"host_seconds": phase_timings_host, "device_seconds": phase_timings_device}, "phase_timings_host_seconds": phase_timings_host, "phase_timings_device_seconds": phase_timings_device, "step_time_seconds": elapsed, "samples_per_second": global_samples / elapsed, "audio_seconds_per_second": global_samples * 20.0 / elapsed, "multimodal_tokens_per_second": multimodal_tokens / elapsed, "nonpadding_tokens_per_second": text_nonpadding_tokens / elapsed, "answer_tokens_per_second": answer_tokens / elapsed, "gpu_memory_allocated_gib": float(torch.cuda.memory_allocated(device) / 1024**3), "gpu_memory_reserved_gib": float(torch.cuda.memory_reserved(device) / 1024**3), "gpu_memory_max_allocated_gib": float(torch.cuda.max_memory_allocated(device) / 1024**3), "gpu_memory_max_reserved_gib": float(torch.cuda.max_memory_reserved(device) / 1024**3), "router_stats": _router_stats(owner)}
                metrics.append(item)
                if profiler is not None:
                    profiler.step()
                batch_in_epoch += args.gradient_accumulation_steps
                if rank == 0 and (optimizer_step % 10 == 0 or optimizer_step == max_steps):
                    memory = torch.cuda.memory_allocated(device) / 1024**3
                    print(f"[audio-train] step={optimizer_step}/{max_steps} progress={item['progress_percent']:.2f}% epoch={epoch + 1}/{args.epochs if args.gate == 'FORMAL' else '?'} batch={batch_in_epoch}/{len(loader)} loss={item['loss']:.6f} lr={item['lr']:.8g} step_s={item['step_time_seconds']:.3f} samples/s={item['samples_per_second']:.2f} audio_s/s={item['audio_seconds_per_second']:.2f} multimodal_tokens/s={item['multimodal_tokens_per_second']:.2f} nonpadding_tokens/s={item['nonpadding_tokens_per_second']:.2f} answer_tokens={item['effective_answer_tokens']} gpu_alloc_gib={item['gpu_memory_allocated_gib']:.3f} gpu_reserved_gib={item['gpu_memory_reserved_gib']:.3f} gpu_max_alloc_gib={item['gpu_memory_max_allocated_gib']:.3f} gpu_max_reserved_gib={item['gpu_memory_max_reserved_gib']:.3f} router_stats={item['router_stats']}", flush=True)
                save_due = (args.gate == "FORMAL" and (optimizer_step % max(1, args.save_every) == 0 or optimizer_step == max_steps)) or (args.gate == "STAGE7" and (optimizer_step == 10 or optimizer_step == max_steps))
                if save_due:
                    out = args.output_dir / f"checkpoint-{optimizer_step:06d}"
                    rng_states = _gather_rng_states(rank, world, device)
                    if rank == 0:
                        _save_checkpoint(out, owner, tokenizer, optimizer, scheduler, optimizer_step, epoch, batch_in_epoch, args, hashlib.sha256(args.train_manifest.read_bytes()).hexdigest(), rng_states, total_steps)
                    if world > 1:
                        dist.barrier()
                    if rank == 0:
                        report.setdefault("checkpoints", []).append(str(out))
                        if args.gate == "STAGE7":
                            artifact_audit = _audit_saved_checkpoint(out)
                            resume_audit = _actual_resume_audit(out, args, batch_cpu, device, optimizer_step, float(optimizer.param_groups[0]["lr"]), rank)
                            report["checkpoint_reload_audit"] = {"artifact": artifact_audit, "actual_resume": resume_audit}
                        elif args.gate == "FORMAL":
                            report["checkpoints"] = _prune_checkpoints(args.output_dir, args.checkpoint_retention)
                    if world > 1:
                        dist.barrier()
                    if optimizer_step >= max_steps:
                        break
                if optimizer_step >= max_steps:
                    break
            if batch_in_epoch >= steps_epoch * args.gradient_accumulation_steps:
                epoch += 1
                batch_in_epoch = 0
        per_rank_timing = _gather_perf20_rank_timings(rank, world, metrics) if args.gate == "PERF20" else None
        report.update({"status": "PASS", "start_step": start_step, "end_step": optimizer_step, "optimizer_steps": optimizer_step, "steps_per_epoch": steps_epoch, "dropped_microbatches_per_epoch": dropped_microbatches, "total_formal_steps": formal_steps, "warmup_steps": args.warmup_steps, "effective_global_batch_size": int(args.micro_batch_size * world * args.gradient_accumulation_steps), "metrics": metrics if rank == 0 else [], "ddp_broadcast_buffers": False, "router_policy": "warning_only", "routing_stats": {"enabled": args.gate != "PERF20", "mode": "disabled_for_perf20" if args.gate == "PERF20" else "continuous_per_forward", "reported_in_each_step": args.gate != "PERF20", "reason": "per-router .cpu() statistics would add CUDA synchronizations to the PERF20 timing path" if args.gate == "PERF20" else None}, "model_trainable_audit": (ddp.module if hasattr(ddp, "module") else ddp).trainable_parameter_audit(), "runtime_gradient_audit": runtime_gradient_audit, "resume_position": {"epoch": epoch, "batch_in_epoch": batch_in_epoch}, "checkpoints": report.get("checkpoints", [])})
        report["correctness_audit"] = {
            "mesh_runtime_gradient_audit": runtime_gradient_audit,
            "answer_only_labels": "build_labels enforces -100 outside real answer intervals and exact answer token count",
            "routing_stats": "disabled for PERF20 timing to avoid per-router CUDA synchronizations; first-step gradient audit retained, then gradient_audit_mode disabled" if args.gate == "PERF20" else "continuous per forward; first-step gradient audit retained, then gradient_audit_mode disabled",
        }
        if args.gate == "PERF20" and rank == 0:
            report["per_rank_timing"] = per_rank_timing
            report["steady_state_summary"] = _perf_steady_summary(metrics, args, max_steps)
            report["timing_semantics"] = {
                "step_time_seconds": "rank0 wall-clock from before the first microbatch data wait through the single CUDA synchronize after all metrics/collectives; this is the completion-inclusive step duration",
                "phase_timings_host_seconds": "host wall/enqueue timings; per-microbatch data_wait/dispatch/forward/backward values are summed within the optimizer step; data_wait includes next(data_iter), scheduler_host is CPU/Python scheduler.step(), and metrics_enqueue_host ends after collective launch rather than after CUDA/NCCL completion",
                "phase_timings_device_seconds": "CUDA event elapsed timings resolved after one unified end-of-step synchronize; per-microbatch device values are summed within the optimizer step; metrics_collectives includes the device/NCCL work through its event and therefore is completion-inclusive",
                "host_to_device": "rank0 CUDA event elapsed time around tensor .to(device) for every microbatch; host_to_device_enqueue separately records host dispatch time",
                "forward_backward_optimizer": "rank0 CUDA event elapsed time; DDP gradient collectives are included in backward; host enqueue fields are reported separately",
                "metrics_collectives": "global token counts use one identical all_reduce sequence on every rank; no rank-specific collective is introduced, and its device timing is not the enqueue-only host latency",
                "per_rank_timing": "all ranks retain local data/forward/backward/step timings during training; one gather_object runs only after the final measured optimizer step and is excluded from every step duration",
                "steady_state": "optimizer steps 6-20 by default, excluding profiler wait/warmup/active steps when profiling is enabled",
            }
    except Exception as exc:
        if args.gate == "PERF20":
            report.setdefault("correctness_audit", {})["status"] = "not_completed"
        report["hard_failures"].append({"error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        if profiler is not None:
            try:
                profiler.__exit__(None, None, None)
                if rank == 0:
                    # All trace and operator-summary files were emitted by
                    # _on_trace_ready before each schedule cycle rotated.
                    # Do not call key_averages() after exit: events may have
                    # been consumed or cleared by the profiler lifecycle.
                    report.setdefault("profiler", {}).update({"artifacts": profiler_artifacts})
            except Exception as exc:
                report["status"] = "FAIL"
                report["hard_failures"].append({"error": f"profiler finalization failed: {exc!r}", "traceback": traceback.format_exc()})
        if dist.is_initialized():
            dist.destroy_process_group()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.gate == "PERF20" and perf_output_preexisting:
            report_path = args.output_dir.parent / f"{args.output_dir.name}.FAIL.{os.getpid()}.json"
            report["report_path"] = str(report_path)
        else:
            default_report_name = "perf20_report.json" if args.gate == "PERF20" else f"{args.gate.lower()}_audit.json"
            report_path = args.report_path or args.output_dir / default_report_name
            report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    if int(os.environ.get("RANK", "0")) == 0:
        if args.gate == "PERF20":
            if report.get("status") == "PASS":
                summary = report.get("steady_state_summary", {})
                step_summary = summary.get("step_time_seconds", {})
                throughput = summary.get("samples_per_second", {})
                print(
                    "[audio-perf20] PASS "
                    f"steps={report.get('optimizer_steps')} "
                    f"steady_steps={len(summary.get('included_steps', []))} "
                    f"step_median_s={step_summary.get('median')} "
                    f"samples_per_s_median={throughput.get('median')} "
                    f"profiler={'on' if report.get('profiler', {}).get('enabled') else 'off'}",
                    flush=True,
                )
            else:
                print(f"[audio-perf20] FAIL errors={len(report.get('hard_failures', []))}", flush=True)
        print(json.dumps({"stage": report["stage"], "status": report["status"], "summary": {"steps": report.get("optimizer_steps"), "hard_failures": len(report.get("hard_failures", []))}, "report": report.get("report_path", str(args.report_path or args.output_dir))}, default=str))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
