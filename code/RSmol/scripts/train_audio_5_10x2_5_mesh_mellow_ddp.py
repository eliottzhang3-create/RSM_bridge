#!/usr/bin/env python3
"""DDP trainer and 8-GPU smoke gates for ReasonAQA + MeSH audio."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import contextlib
import copy
import gc
import hashlib
import json
import math
import os
import queue
import random
import shutil
import tempfile
import threading
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

from audio_5_10x2_5_mesh_mellow.data import ReasonAQADataset, UniqueWaveformStore, collate_reasonaqa  # noqa: E402
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
DEFAULT_SHARED_WAVEFORM_STORE = "/hpc_stor03/sjtu_home/jinwei.zhang/data/rsmol_reasonaqa_train_unique_waveforms_32k_10s_f32_v2"
DEFAULT_COMPONENT_PARTITION_STORE_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/data/rsmol_reasonaqa_train_component_partitions6_32k_10s_f32_v2"
PERF20_STEPS = 20
PERF20_INPUT_MODES = ("online", "warm_online", "waveform_preload", "full_preload", "shared_waveform_store", "store_rank_ram_preload", "store_rank_ram_prefetch", "partition_rank_ram_preload")


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
    p.add_argument(
        "--waveform-cache-dir",
        type=Path,
        help="PERF20 only: read fixed float32 waveforms from the completed 64-shard cache",
    )
    p.add_argument(
        "--shared-waveform-store-dir",
        type=Path,
        default=Path(DEFAULT_SHARED_WAVEFORM_STORE),
        help="PERF20 shared-store and rank-RAM modes: manifest-scoped single-file float32 waveform store",
    )
    p.add_argument(
        "--perf20-partition-store-root",
        type=Path,
        default=Path(DEFAULT_COMPONENT_PARTITION_STORE_ROOT),
        help="partition_rank_ram_preload only: root containing audited partition_<id> stores",
    )
    p.add_argument(
        "--perf20-partition-id",
        type=int,
        default=0,
        help="partition_rank_ram_preload only: zero-based materialized component partition (default: 0)",
    )
    p.add_argument(
        "--preload-data",
        action="store_true",
        help="Deprecated PERF20 alias for --perf20-input-mode full_preload",
    )
    p.add_argument(
        "--perf20-input-mode",
        choices=PERF20_INPUT_MODES,
        default="online",
        help=(
            "PERF20 causal input control: online; exact-file warm_online; waveform-only "
            "rank-local preload with timed tokenization/collate; fully collated full_preload; "
            "rank0-warmed shared mmap; exact-row store-to-rank-RAM preload; bounded asynchronous "
            "store-to-rank-RAM prefetch; or whole-component-partition rank-RAM preload"
        ),
    )
    p.add_argument("--perf20-prefetch-microbatches", type=int, default=8,
                   help="store_rank_ram_prefetch only: bounded queue depth, including initial priming")
    p.add_argument("--perf20-rank-cache-gib", type=float, default=2.0,
                   help="store_rank_ram_prefetch only: rank-local unique-waveform LRU capacity")
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
    host_phases = (
        "data_wait", "queue_get", "collate", "tokenize", "tokenize_thread_cpu",
        "text_tensor_build", "text_tensor_build_thread_cpu", "waveform_stack", "waveform_stack_thread_cpu",
        "batch_metadata", "batch_metadata_thread_cpu", "store_locate", "store_mmap_view", "store_clone", "store_dataset_item",
        "host_to_device_enqueue", "forward_enqueue", "backward_enqueue", "grad_clip_host",
        "optimizer_enqueue", "scheduler_host", "metrics_enqueue_host",
    )
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


def _local_rank_timing_payload(
    rank: int,
    metrics: list[dict[str, Any]],
    input_preparation: dict[str, Any] | None = None,
    steady_steps: set[int] | None = None,
) -> dict[str, Any]:
    """Build the compact timing payload gathered once after PERF20 training."""

    steps: list[dict[str, Any]] = []
    for item in metrics:
        host = item.get("phase_timings_host_seconds", {})
        device = item.get("phase_timings_device_seconds", {})
        steps.append({
            "step": int(item["step"]),
            "included_in_steady_state": steady_steps is None or int(item["step"]) in steady_steps,
            "data_wait_seconds": float(host.get("data_wait", 0.0)),
            "queue_get_seconds": float(host.get("queue_get", 0.0)),
            "collate_seconds": float(host.get("collate", 0.0)),
            "tokenize_seconds": float(host.get("tokenize", 0.0)),
            "tokenize_thread_cpu_seconds": float(host.get("tokenize_thread_cpu", 0.0)),
            "text_tensor_build_seconds": float(host.get("text_tensor_build", 0.0)),
            "waveform_stack_seconds": float(host.get("waveform_stack", 0.0)),
            "waveform_stack_thread_cpu_seconds": float(host.get("waveform_stack_thread_cpu", 0.0)),
            "store_clone_seconds": float(host.get("store_clone", 0.0)),
            "forward_device_seconds": float(device.get("forward", 0.0)),
            "backward_device_seconds": float(device.get("backward", 0.0)),
            "step_wall_seconds": float(item["step_time_seconds"]),
            "process_runtime_delta": item.get("process_runtime_delta"),
            "cgroup_cpu_delta": item.get("cgroup_cpu_delta"),
        })
    return {"rank": int(rank), "steps": steps, "input_preparation": input_preparation}


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
        "queue_get_seconds",
        "collate_seconds",
        "tokenize_seconds",
        "tokenize_thread_cpu_seconds",
        "text_tensor_build_seconds",
        "waveform_stack_seconds",
        "waveform_stack_thread_cpu_seconds",
        "store_clone_seconds",
        "forward_device_seconds",
        "backward_device_seconds",
        "step_wall_seconds",
    )
    per_rank_summary = []
    by_step: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for payload in ordered:
        rank = int(payload["rank"])
        steps = sorted(payload["steps"], key=lambda item: int(item["step"]))
        steady = [step for step in steps if bool(step.get("included_in_steady_state", True))]
        per_rank_summary.append({
            "rank": rank,
            "step_count": len(steady),
            "included_steps": [int(step["step"]) for step in steady],
            "distributions": {
                field: _distribution([float(step.get(field, 0.0)) for step in steady])
                for field in fields
            },
        })
        for step in steps:
            if bool(step.get("included_in_steady_state", True)):
                by_step.setdefault(int(step["step"]), []).append((rank, step))

    per_step_rank_skew = []
    for step_number in sorted(by_step):
        ranked_steps = sorted(by_step[step_number], key=lambda item: item[0])
        phases: dict[str, Any] = {}
        for field in fields:
            ranked_values = [(rank, float(step.get(field, 0.0))) for rank, step in ranked_steps]
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

    preparation_by_rank = [
        {"rank": int(payload["rank"]), **dict(payload["input_preparation"])}
        for payload in ordered
        if payload.get("input_preparation") is not None
    ]
    preparation_summary = None
    if preparation_by_rank:
        if len(preparation_by_rank) != int(expected_world):
            raise RuntimeError(
                "PERF20 input-preparation metadata is present for only part of the world: "
                f"expected={expected_world} actual={len(preparation_by_rank)}"
            )
        modes = sorted({str(item["mode"]) for item in preparation_by_rank})
        if len(modes) != 1:
            raise RuntimeError(f"PERF20 ranks disagree on input mode: {modes}")
        preparation_summary = {
            "mode": modes[0],
            "rank_count": len(preparation_by_rank),
            "duration_seconds": _distribution([float(item["duration_seconds"]) for item in preparation_by_rank]),
            "maximum_duration_seconds": max(float(item["duration_seconds"]) for item in preparation_by_rank),
            "barrier_wait_seconds": _distribution([float(item["barrier_wait_seconds"]) for item in preparation_by_rank]),
            "cpu_tensor_gib": _distribution([float(item.get("cpu_tensor_bytes", 0)) / 1024**3 for item in preparation_by_rank]),
            "row_indices_sha256_by_rank": [str(item["row_indices_sha256"]) for item in preparation_by_rank],
        }
        training_seconds = [float(item.get("training_measurement_seconds", 0.0)) for item in preparation_by_rank]
        preparation_summary["training_measurement_seconds"] = _distribution(training_seconds)
        preparation_summary["maximum_training_measurement_seconds"] = max(training_seconds)
        sample_counts = {int(item.get("global_samples_processed", 0)) for item in preparation_by_rank}
        if len(sample_counts) == 1:
            global_samples = sample_counts.pop()
            preparation_summary["global_samples_processed"] = global_samples
            preparation_summary["completion_inclusive_samples_per_second"] = (
                global_samples / max(training_seconds) if global_samples and max(training_seconds) > 0 else None
            )
        end_to_end = [
            float(item["duration_seconds"]) + float(item["barrier_wait_seconds"]) + float(item.get("training_measurement_seconds", 0.0))
            for item in preparation_by_rank
        ]
        preparation_summary["preparation_barrier_training_seconds"] = _distribution(end_to_end)
        preparation_summary["maximum_preparation_barrier_training_seconds"] = max(end_to_end)
        rank0_preparation = next(item for item in preparation_by_rank if int(item["rank"]) == 0)
        preparation_summary["rank0_duration_seconds"] = float(rank0_preparation["duration_seconds"])
        preparation_summary["rank0_warm_read_seconds"] = float(rank0_preparation.get("warm_read_seconds", 0.0))
        preparation_summary["rank0_warmed_file_gib"] = float(rank0_preparation.get("warmed_file_bytes") or 0) / 1024**3

    return {
        "collection": "one dist.gather_object after the final optimizer step",
        "included_in_step_timing": False,
        "per_step_collectives_added": 0,
        "timing_sources": {
            "data_wait_seconds": (
                "host wall time around next(iter(preloaded_cpu_batches)), summed across microbatches; "
                "shared-storage reads, audio decode/resample, tokenization, and collate completed before timing"
                if preparation_summary and preparation_summary["mode"] == "full_preload" else
                "host wall time around next(data_iter), summed across microbatches"
            ),
            "collate_seconds": "rank-process tokenizer plus collate wall time; zero during measured full_preload iteration",
            "queue_get_seconds": "prefetch consumer wall time inside queue.get only; zero for non-prefetch modes",
            "forward_device_seconds": "CUDA events, summed across microbatches and resolved by the existing end-of-step synchronize",
            "backward_device_seconds": "CUDA events, summed across microbatches and resolved by the existing end-of-step synchronize; includes DDP gradient communication dependencies",
            "step_wall_seconds": "rank-local completion-inclusive optimizer-step wall time",
        },
        "raw_by_rank": ordered,
        "per_rank_summary": per_rank_summary,
        "per_step_rank_skew": per_step_rank_skew,
        "input_preparation_by_rank": preparation_by_rank,
        "input_preparation_summary": preparation_summary,
    }


def _gather_perf20_rank_timings(
    rank: int,
    world: int,
    metrics: list[dict[str, Any]],
    input_preparation: dict[str, Any] | None = None,
    args: argparse.Namespace | None = None,
    max_steps: int | None = None,
) -> dict[str, Any] | None:
    """Gather all rank-local PERF20 timings exactly once after measurement."""

    steady_steps = None
    if args is not None and max_steps is not None:
        excluded = set(_profile_overhead_steps(args, max_steps)) if args.profiler else set()
        steady_steps = {step for step in range(max(1, int(args.steady_state_start_step)), int(max_steps) + 1) if step not in excluded}
    local_payload = _local_rank_timing_payload(rank, metrics, input_preparation, steady_steps)
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


def _batch_tensor_bytes(batch: dict[str, Any]) -> int:
    """Return bytes held by tensor values in one collated CPU batch."""

    return sum(
        int(value.numel()) * int(value.element_size())
        for value in batch.values()
        if torch.is_tensor(value)
    )


class _TimedReasonAQACollator:
    """Accumulate collate/tokenization wall time in the rank process."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.total_seconds = 0.0
        self.calls = 0
        self.phase_seconds = {
            "tokenize": 0.0,
            "tokenize_thread_cpu": 0.0,
            "text_tensor_build": 0.0,
            "text_tensor_build_thread_cpu": 0.0,
            "waveform_stack": 0.0,
            "waveform_stack_thread_cpu": 0.0,
            "batch_metadata": 0.0,
            "batch_metadata_thread_cpu": 0.0,
        }

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return collate_reasonaqa(rows, self.tokenizer, timing_accumulator=self.phase_seconds)
        finally:
            self.total_seconds += time.perf_counter() - started
            self.calls += 1

    def snapshot(self) -> dict[str, float]:
        return {"collate": float(self.total_seconds), **{key: float(value) for key, value in self.phase_seconds.items()}}


class _Perf20WaveformPreloadedDataset:
    """Serve only the planned PERF20 rows from rank-local decoded waveform RAM."""

    def __init__(self, base: ReasonAQADataset) -> None:
        self.base = base
        self.items: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            return self.items[int(index)]
        except KeyError as exc:
            raise RuntimeError(f"PERF20 waveform preload lacks planned row {index}") from exc


class _RankLocalStoreWaveforms:
    """Copy store mmap views into owned CPU tensors, with optional bounded LRU."""

    def __init__(self, dataset: ReasonAQADataset, max_bytes: int | None = None) -> None:
        if dataset.unique_waveform_store is None:
            raise RuntimeError("rank-local waveform cache requires the unique waveform store")
        self.dataset = dataset
        self.store = dataset.unique_waveform_store
        self.max_bytes = max_bytes
        self.items: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.hits = self.misses = self.evictions = self.cloned_bytes = 0
        self.current_bytes = self.peak_bytes = self.copy_seconds = 0
        self.locate_seconds = self.mmap_view_seconds = self.clone_seconds = 0.0
        self.clone_thread_cpu_seconds = 0.0
        self.dataset_item_seconds = self.audio_paths_seconds = 0.0

    def _get(self, path: str) -> torch.Tensor:
        started = time.perf_counter()
        audio_id = self.store.locate(path)
        self.locate_seconds += time.perf_counter() - started
        return self._get_audio_id(audio_id)

    def _get_audio_id(self, audio_id: int) -> torch.Tensor:
        """Return an owned tensor, cloning the store row only on first use."""

        audio_id = int(audio_id)
        value = self.items.get(audio_id)
        if value is not None:
            self.hits += 1
            self.items.move_to_end(audio_id)
            return value
        self.misses += 1
        copy_started = time.perf_counter()
        view_started = time.perf_counter()
        view = self.store.load_audio_id(audio_id)
        self.mmap_view_seconds += time.perf_counter() - view_started
        clone_started = time.perf_counter()
        thread_cpu_started = time.thread_time()
        value = view.clone()
        self.clone_thread_cpu_seconds += time.thread_time() - thread_cpu_started
        self.clone_seconds += time.perf_counter() - clone_started
        self.copy_seconds += time.perf_counter() - copy_started
        size = int(value.numel() * value.element_size())
        self.cloned_bytes += size
        if self.max_bytes is not None:
            while self.items and self.current_bytes + size > self.max_bytes:
                _, old = self.items.popitem(last=False)
                self.current_bytes -= int(old.numel() * old.element_size())
                self.evictions += 1
        self.items[audio_id] = value
        self.current_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        return value

    def materialize(self, row: int) -> dict[str, Any]:
        started = time.perf_counter()
        item = self.dataset[int(row)]  # text/answer contract; mmap views are not retained.
        self.dataset_item_seconds += time.perf_counter() - started
        started = time.perf_counter()
        audio1, audio2 = self.dataset.audio_paths(int(row))
        self.audio_paths_seconds += time.perf_counter() - started
        item["audio1"] = self._get(audio1)
        item["audio2"] = None if audio2 == audio1 else self._get(audio2)
        return item

    def materialize_from_cached_metadata(self, row: int) -> dict[str, Any]:
        """Build a row without asking the base dataset to touch mmap waveform views."""

        row_index = int(row)
        started = time.perf_counter()
        source = self.dataset.rows[row_index]
        prompt = str(source.get("prompt") or source.get("question") or source.get("input") or "")
        answer = str(
            source.get("answer") or source.get("target") or source.get("output")
            or source.get("caption1") or ""
        )
        if not answer:
            raise ValueError(f"manifest row {row_index} lacks answer")
        self.dataset_item_seconds += time.perf_counter() - started
        started = time.perf_counter()
        audio1, audio2 = self.dataset.audio_paths(row_index)
        self.audio_paths_seconds += time.perf_counter() - started
        return {
            "audio1": self._get(audio1),
            "audio2": None if audio2 == audio1 else self._get(audio2),
            "prompt": prompt,
            "answer": answer,
            "row_index": row_index,
            "audio2_reused": audio2 == audio1,
        }

    def stats(self) -> dict[str, Any]:
        lookups = self.hits + self.misses
        return {
            "waveform_lookups": lookups, "waveform_hits": self.hits,
            "waveform_misses": self.misses, "hit_rate": self.hits / lookups if lookups else 0.0,
            "waveform_evictions": self.evictions, "cloned_bytes": self.cloned_bytes,
            "copy_seconds": self.copy_seconds, "resident_unique_audio": len(self.items),
            "locate_seconds": self.locate_seconds, "mmap_view_seconds": self.mmap_view_seconds,
            "clone_seconds": self.clone_seconds, "clone_thread_cpu_seconds": self.clone_thread_cpu_seconds,
            "dataset_item_seconds": self.dataset_item_seconds, "audio_paths_seconds": self.audio_paths_seconds,
            "cache_current_bytes": self.current_bytes, "cache_peak_bytes": self.peak_bytes,
            "cache_capacity_bytes": self.max_bytes,
        }


class _Perf20StorePrefetcher:
    """One bounded producer per rank; only owned CPU waveform tensors cross the queue."""

    def __init__(self, cache: _RankLocalStoreWaveforms, rows: list[int],
                 batch_size: int, depth: int, collator: _TimedReasonAQACollator) -> None:
        if len(rows) % batch_size:
            raise ValueError("planned PERF20 row count is not a whole number of microbatches")
        self.cache, self.rows, self.batch_size, self.collator = cache, rows, batch_size, collator
        self.total = len(rows) // batch_size
        self.depth = min(depth, self.total)
        self.pending: queue.Queue[list[dict[str, Any]]] = queue.Queue(maxsize=depth)
        self.slots = threading.BoundedSemaphore(depth)
        self.finished = threading.Event()
        self.stopped = threading.Event()
        self.error: BaseException | None = None
        self.produced = self.consumed = 0
        self.queue_get_seconds = 0.0
        self.queue_empty_polls = 0
        self.queue_depth_before_get: list[int] = []
        self.queue_depth_after_get: list[int] = []
        self.thread = threading.Thread(target=self._produce, name="perf20-waveform-prefetch", daemon=True)

    def _produce(self) -> None:
        try:
            for offset in range(0, len(self.rows), self.batch_size):
                while not self.stopped.is_set() and not self.slots.acquire(timeout=0.2):
                    continue
                if self.stopped.is_set():
                    break
                items = [self.cache.materialize(row) for row in self.rows[offset:offset + self.batch_size]]
                while not self.stopped.is_set():
                    try:
                        self.pending.put(items, timeout=0.2)
                        self.produced += 1
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:
            self.error = exc
        finally:
            self.finished.set()

    def start_and_prime(self) -> None:
        self.thread.start()
        while self.pending.qsize() < self.depth and not self.finished.is_set():
            self.finished.wait(0.1)
        if self.error is not None:
            raise RuntimeError("rank-local waveform prefetch failed during priming") from self.error
        if self.pending.qsize() < self.depth:
            raise RuntimeError(f"rank-local waveform prefetch primed {self.pending.qsize()}/{self.depth}")

    def __iter__(self) -> _Perf20StorePrefetcher:
        return self

    def __next__(self) -> dict[str, Any]:
        if self.consumed >= self.total:
            raise StopIteration
        self.queue_depth_before_get.append(self.pending.qsize())
        get_started = time.perf_counter()
        while True:
            try:
                items = self.pending.get(timeout=0.2)
                self.slots.release()
                break
            except queue.Empty:
                self.queue_empty_polls += 1
                if self.finished.is_set():
                    raise RuntimeError("rank-local waveform prefetch stopped early") from self.error
        self.queue_get_seconds += time.perf_counter() - get_started
        self.queue_depth_after_get.append(self.pending.qsize())
        self.consumed += 1
        return self.collator(items)

    def timing_snapshot(self) -> dict[str, float]:
        return {
            "queue_get": float(self.queue_get_seconds),
            "queue_empty_polls": float(self.queue_empty_polls),
            "depth_observations": float(len(self.queue_depth_before_get)),
        }

    def stats(self) -> dict[str, Any]:
        return {
            "produced_microbatches": self.produced,
            "consumed_microbatches": self.consumed,
            "queue_get_seconds": self.queue_get_seconds,
            "queue_empty_polls": self.queue_empty_polls,
            "queue_depth_before_get": list(self.queue_depth_before_get),
            "queue_depth_after_get": list(self.queue_depth_after_get),
            "queue_depth_before_get_distribution": _distribution([float(value) for value in self.queue_depth_before_get]),
            "queue_depth_after_get_distribution": _distribution([float(value) for value in self.queue_depth_after_get]),
        }

    def close(self) -> None:
        self.stopped.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)


def _planned_perf20_rows(
    sampler: DistributedSampler,
    *,
    epoch: int,
    batch_in_epoch: int,
    micro_batch_size: int,
    required_microbatches: int,
) -> list[int]:
    """Resolve the exact rank-local sample stream without touching the dataset."""

    sampler.set_epoch(int(epoch))
    indices = [int(index) for index in sampler]
    start = int(batch_in_epoch) * int(micro_batch_size)
    count = int(required_microbatches) * int(micro_batch_size)
    rows = indices[start:start + count]
    if len(rows) != count:
        raise RuntimeError(
            "PERF20 planned row stream is shorter than the bounded experiment: "
            f"start={start} required={count} actual={len(rows)}"
        )
    return rows


def _row_stream_metadata(rows: list[int]) -> dict[str, Any]:
    encoded = ",".join(str(row) for row in rows).encode("ascii")
    return {
        "planned_samples": len(rows),
        "unique_planned_rows": len(set(rows)),
        "row_indices_sha256": hashlib.sha256(encoded).hexdigest(),
        "first_row_indices": rows[: min(8, len(rows))],
        "last_row_indices": rows[-min(8, len(rows)):],
    }


def _process_fault_snapshot() -> dict[str, int] | None:
    """Read process minor/major fault counters without adding a dependency."""

    try:
        # /proc/<pid>/stat fields 10 and 12 are minflt and majflt.  Split only
        # after the final ')' because the comm field may contain whitespace.
        fields = Path("/proc/self/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return {"minor_faults": int(fields[7]), "major_faults": int(fields[9])}
    except (OSError, ValueError, IndexError):
        return None


def _process_io_snapshot() -> dict[str, int] | None:
    """Read Linux process I/O counters; remote clients may report a subset."""

    try:
        values = {}
        for line in Path("/proc/self/io").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key in {"rchar", "syscr", "read_bytes"}:
                values[key] = int(value.strip())
        return values
    except (OSError, ValueError):
        return None


def _process_runtime_snapshot() -> dict[str, Any]:
    """Capture scheduler/context-switch evidence without adding dependencies."""

    result: dict[str, Any] = {}
    try:
        values = {}
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"Threads", "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches", "VmRSS", "VmHWM"}:
                values[key] = int(value.strip().split()[0])
        result["proc_status"] = values
    except (OSError, ValueError, IndexError) as exc:
        result["proc_status_error"] = repr(exc)
    try:
        fields = Path("/proc/self/schedstat").read_text(encoding="utf-8").split()
        result["schedstat"] = {
            "cpu_run_time_ns": int(fields[0]),
            "runqueue_wait_time_ns": int(fields[1]),
            "timeslices": int(fields[2]),
        }
    except (OSError, ValueError, IndexError) as exc:
        result["schedstat_error"] = repr(exc)
    return result


def _thread_runtime_configuration() -> dict[str, Any]:
    result: dict[str, Any] = {
        "os_cpu_count": os.cpu_count(),
        "python_active_threads": threading.active_count(),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM")
        },
    }
    try:
        affinity = sorted(os.sched_getaffinity(0))
        result.update({"cpu_affinity": affinity, "cpu_affinity_count": len(affinity)})
    except (AttributeError, OSError) as exc:
        result["cpu_affinity_error"] = repr(exc)
    return result


def _nested_numeric_delta(before: Any, after: Any) -> Any:
    """Subtract matching numeric leaves while retaining only comparable counters."""

    if isinstance(before, dict) and isinstance(after, dict):
        return {
            key: value
            for key in before.keys() & after.keys()
            if (value := _nested_numeric_delta(before[key], after[key])) is not None
        }
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    return None


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _rng_fingerprint(device: torch.device) -> dict[str, str]:
    result = {
        "python_random": hashlib.sha256(repr(random.getstate()).encode("utf-8")).hexdigest(),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()
    return result


def _batch_fingerprint(batch: dict[str, Any]) -> dict[str, Any]:
    tensor_hashes = {
        key: _tensor_sha256(value)
        for key, value in batch.items()
        if torch.is_tensor(value)
    }
    encoded = json.dumps({"row_indices": batch.get("row_indices"), "tensor_hashes": tensor_hashes}, sort_keys=True).encode("utf-8")
    return {
        "row_indices": [int(value) for value in batch.get("row_indices", [])],
        "tensor_sha256": tensor_hashes,
        "combined_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _cgroup_memberships_and_mounts() -> tuple[list[tuple[str, str]], list[dict[str, str]], list[str]]:
    """Return controller memberships and cgroup mounts with diagnostics."""

    errors: list[str] = []
    memberships: list[tuple[str, str]] = []
    mounts: list[dict[str, str]] = []
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            _, controllers, path = line.split(":", 2)
            memberships.append((controllers, path))
    except (OSError, ValueError) as exc:
        errors.append(f"cgroup memberships: {exc!r}")
    try:
        for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
            left, right = line.split(" - ", 1)
            left_fields, right_fields = left.split(), right.split()
            if right_fields[0] not in {"cgroup", "cgroup2"}:
                continue
            mounts.append({
                "root": left_fields[3],
                "mountpoint": left_fields[4],
                "fstype": right_fields[0],
                "source": right_fields[1],
                "super_options": right_fields[2] if len(right_fields) > 2 else "",
            })
    except (OSError, ValueError, IndexError) as exc:
        errors.append(f"cgroup mounts: {exc!r}")
    return memberships, mounts, errors


def _cgroup_candidate_paths(controller: str) -> tuple[list[tuple[int, Path]], list[str]]:
    cache = globals().setdefault("_PERF20_CGROUP_CANDIDATE_CACHE", {})
    if controller in cache:
        return cache[controller]
    memberships, mounts, errors = _cgroup_memberships_and_mounts()
    candidates: list[tuple[int, Path]] = []
    for controllers, membership in memberships:
        version = 2 if controllers == "" else 1
        if version == 1 and controller not in controllers.split(","):
            continue
        for mount in mounts:
            if mount["fstype"] != ("cgroup2" if version == 2 else "cgroup"):
                continue
            if version == 1 and controller not in (mount["source"] + "," + mount["super_options"]).split(","):
                continue
            mount_root = mount["root"].rstrip("/") or "/"
            member = membership.rstrip("/") or "/"
            if mount_root == "/":
                relative = member.lstrip("/")
            elif member == mount_root:
                relative = ""
            elif member.startswith(mount_root + "/"):
                relative = member[len(mount_root):].lstrip("/")
            else:
                continue
            candidates.append((version, Path(mount["mountpoint"]) / relative))
            candidates.append((version, Path(mount["mountpoint"])))
    # Conventional paths cover restricted container mountinfo views.
    for controllers, membership in memberships:
        if controllers == "":
            candidates.extend(((2, Path("/sys/fs/cgroup") / membership.lstrip("/")), (2, Path("/sys/fs/cgroup"))))
        elif controller in controllers.split(","):
            base = Path("/sys/fs/cgroup") / controller
            candidates.extend(((1, base / membership.lstrip("/")), (1, base)))
    unique: list[tuple[int, Path]] = []
    seen: set[tuple[int, str]] = set()
    for version, path in candidates:
        key = (version, str(path))
        if key not in seen:
            seen.add(key)
            unique.append((version, path))
    result = (unique, errors)
    cache[controller] = result
    return result


def _cgroup_memory_snapshot() -> dict[str, Any] | None:
    """Best-effort hybrid-safe memory snapshot; failures remain observable."""

    attempts: list[dict[str, Any]] = []
    candidates, discovery_errors = _cgroup_candidate_paths("memory")
    for version, root in candidates:
        current_name, maximum_name = (("memory.current", "memory.max") if version == 2 else
                                      ("memory.usage_in_bytes", "memory.limit_in_bytes"))
        try:
            current = int((root / current_name).read_text(encoding="utf-8").strip())
            maximum_text = (root / maximum_name).read_text(encoding="utf-8").strip()
            raw_stats = {}
            for line in (root / "memory.stat").read_text(encoding="utf-8").splitlines():
                key, value = line.split()
                raw_stats[key] = int(value)
        except (OSError, ValueError, IndexError) as exc:
            attempts.append({"version": version, "path": str(root), "error": repr(exc)})
            continue
        maximum = None if maximum_text == "max" else int(maximum_text)
        if maximum is not None and maximum >= 1 << 60:
            maximum = None  # v1 uses a very large sentinel for unlimited memory.
        if version == 1:
            # total_* includes child cgroups, matching the hierarchical usage limit.
            stats = {
                "anon": raw_stats.get("total_rss", raw_stats.get("rss", 0)),
                "file": raw_stats.get("total_cache", raw_stats.get("cache", 0)),
                "file_mapped": raw_stats.get("total_mapped_file", raw_stats.get("mapped_file", 0)),
                "active_file": raw_stats.get("total_active_file", raw_stats.get("active_file", 0)),
                "inactive_file": raw_stats.get("total_inactive_file", raw_stats.get("inactive_file", 0)),
            }
        else:
            stats = {key: raw_stats[key] for key in
                     ("anon", "file", "file_mapped", "file_dirty", "file_writeback", "active_file", "inactive_file")
                     if key in raw_stats}
        return {
            "status": "PASS",
            "cgroup_version": version,
            "cgroup_path": str(root),
            "memory_current_bytes": current,
            "memory_max_bytes": maximum,
            "memory_stat_bytes": stats,
            "discovery_errors": discovery_errors,
            "failed_candidates": attempts,
        }
    return {
        "status": "UNAVAILABLE",
        "cgroup_version": None,
        "cgroup_path": None,
        "memory_current_bytes": None,
        "memory_max_bytes": None,
        "memory_stat_bytes": {},
        "discovery_errors": discovery_errors,
        "failed_candidates": attempts,
    }


def _cgroup_cpu_snapshot() -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    candidates, discovery_errors = _cgroup_candidate_paths("cpu")
    for version, root in candidates:
        try:
            if version == 2:
                quota_text, period_text = (root / "cpu.max").read_text(encoding="utf-8").split()
                quota = None if quota_text == "max" else int(quota_text)
                period = int(period_text)
            else:
                raw_quota = int((root / "cpu.cfs_quota_us").read_text(encoding="utf-8").strip())
                quota = None if raw_quota < 0 else raw_quota
                period = int((root / "cpu.cfs_period_us").read_text(encoding="utf-8").strip())
            stats: dict[str, int] = {}
            for line in (root / "cpu.stat").read_text(encoding="utf-8").splitlines():
                key, value = line.split()
                stats[key] = int(value)
        except (OSError, ValueError, IndexError) as exc:
            attempts.append({"version": version, "path": str(root), "error": repr(exc)})
            continue
        return {
            "status": "PASS", "cgroup_version": version, "cgroup_path": str(root),
            "quota_us": quota, "period_us": period,
            "quota_cpus": quota / period if quota is not None and period > 0 else None,
            "cpu_stat": stats, "discovery_errors": discovery_errors, "failed_candidates": attempts,
        }
    return {
        "status": "UNAVAILABLE", "cgroup_version": None, "cgroup_path": None,
        "quota_us": None, "period_us": None, "quota_cpus": None, "cpu_stat": {},
        "discovery_errors": discovery_errors, "failed_candidates": attempts,
    }


def _warm_exact_perf20_files(
    dataset: ReasonAQADataset,
    local_rows: list[int],
    *,
    rank: int,
    world: int,
) -> dict[str, Any]:
    """Read the all-rank 20-step audio-file union once, sequentially per file."""

    local_paths = sorted({path for row in local_rows for path in dataset.audio_paths(row)})
    gathered: list[list[str] | None] = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, local_paths)
    else:
        gathered[0] = local_paths

    started = time.perf_counter()
    warmed_files = 0
    warmed_bytes = 0
    if rank == 0:
        global_paths = sorted({path for paths in gathered if paths is not None for path in paths})
        buffer = bytearray(8 * 1024 * 1024)
        view = memoryview(buffer)
        for value in global_paths:
            with Path(value).open("rb", buffering=0) as handle:
                while True:
                    size = handle.readinto(view)
                    if not size:
                        break
                    warmed_bytes += int(size)
            warmed_files += 1
    return {
        "local_unique_audio_paths": len(local_paths),
        "global_unique_audio_paths": warmed_files if rank == 0 else None,
        "warmed_file_bytes": warmed_bytes if rank == 0 else None,
        "warm_read_seconds": float(time.perf_counter() - started) if rank == 0 else 0.0,
        "warm_strategy": "rank0 reads the exact all-rank file union; lexical file order; sequential bytes within each file",
    }


def _warm_shared_waveform_store(
    dataset: ReasonAQADataset, local_rows: list[int], *, rank: int, world: int,
) -> dict[str, Any]:
    """Warm exactly the all-rank bounded-run audio IDs, ordered by file offset."""

    store = dataset.unique_waveform_store
    if store is None:
        raise RuntimeError("shared waveform store is not installed")
    local_ids: list[int] = []
    local_error: str | None = None
    try:
        local_ids = sorted({store.locate(path) for row in local_rows for path in dataset.audio_paths(row)})
    except Exception as exc:
        local_error = repr(exc)
    gathered: list[dict[str, Any] | None] = [None] * world
    payload = {"audio_ids": local_ids, "error": local_error}
    if world > 1:
        dist.all_gather_object(gathered, payload)
    else:
        gathered[0] = payload
    failures = {i: item["error"] for i, item in enumerate(gathered) if item is not None and item["error"]}
    if failures:
        raise RuntimeError(f"shared waveform store planned audio lookup failed: {failures}")
    if any(item is None for item in gathered):
        raise RuntimeError("shared waveform store received an incomplete all-rank audio union")
    global_ids = sorted({audio_id for item in gathered if item is not None for audio_id in item["audio_ids"]})
    if not global_ids:
        raise RuntimeError("shared waveform store planned audio union is empty")
    expected_bytes = len(global_ids) * int(store.bytes_per_audio)
    started = time.perf_counter()
    warmed_bytes = 0
    if int(rank) == 0:
        buffer = bytearray(store.bytes_per_audio)
        view = memoryview(buffer)
        with store.data_path.open("rb", buffering=0) as handle:
            for count, audio_id in enumerate(global_ids, start=1):
                handle.seek(audio_id * store.bytes_per_audio)
                offset = 0
                while offset < store.bytes_per_audio:
                    size = handle.readinto(view[offset:])
                    if not size:
                        raise RuntimeError(f"short waveform store read for audio_id={audio_id}: {offset}")
                    offset += int(size)
                warmed_bytes += offset
                if count % 1024 == 0:
                    print(f"[audio-perf20] shared waveform target warm "
                          f"audio={count}/{len(global_ids)} gib={warmed_bytes / 1024**3:.3f}", flush=True)
        if warmed_bytes != expected_bytes:
            raise RuntimeError(
                "shared waveform store warm byte count mismatch: "
                f"actual={warmed_bytes} expected={expected_bytes}"
            )
    return {
        "shared_store_path": str(store.store_dir),
        "shared_store_data_path": str(store.data_path),
        "shared_store_manifest_sha256": str(store.metadata.get("manifest_sha256")),
        "shared_store_waveform_sha256": str(store.metadata.get("waveform_sha256")),
        "shared_store_unique_audio": int(store.num_audio),
        "local_planned_unique_audio": len(local_ids),
        "global_planned_unique_audio": len(global_ids),
        "global_audio_ids_sha256": hashlib.sha256(",".join(map(str, global_ids)).encode("ascii")).hexdigest(),
        "shared_store_expected_bytes": expected_bytes,
        "shared_store_expected_gib": expected_bytes / 1024**3,
        "warmed_file_bytes": warmed_bytes if int(rank) == 0 else None,
        "warm_read_seconds": float(time.perf_counter() - started) if int(rank) == 0 else 0.0,
        "warm_owner_rank": 0,
        "warm_strategy": "rank0 reads only the exact all-rank 20-step unique waveform regions in increasing audio_id order before the common pre-measurement barrier",
    }


def _preload_perf20_waveforms(
    dataset: ReasonAQADataset,
    rows: list[int],
    target: _Perf20WaveformPreloadedDataset | None = None,
) -> tuple[_Perf20WaveformPreloadedDataset, dict[str, Any]]:
    """Decode only planned rows; leave tokenization/collate inside timed next()."""

    cached = target if target is not None else _Perf20WaveformPreloadedDataset(dataset)
    started = time.perf_counter()
    for row in dict.fromkeys(int(value) for value in rows):
        cached.items[row] = dataset[row]
    duration = time.perf_counter() - started
    tensor_bytes = sum(_batch_tensor_bytes(item) for item in cached.items.values())
    return cached, {
        "loaded_rows": len(cached.items),
        "cpu_tensor_bytes": tensor_bytes,
        "waveform_decode_seconds": float(duration),
        "timed_tokenization_and_collate": True,
    }


def _preload_store_rank_ram(
    cache: _RankLocalStoreWaveforms, target: _Perf20WaveformPreloadedDataset, rows: list[int],
) -> dict[str, Any]:
    """Own exactly the planned waveforms in rank-local RAM before measurement."""

    started = time.perf_counter()
    for row in dict.fromkeys(int(value) for value in rows):
        target.items[row] = cache.materialize(row)
    return {
        "loaded_rows": len(target.items),
        "cpu_tensor_bytes": cache.current_bytes,
        "store_to_rank_ram_seconds": time.perf_counter() - started,
        "waveform_cache_stats": cache.stats(),
        "timed_tokenization_and_collate": True,
    }


def _preload_partition_store_rank_ram(
    cache: _RankLocalStoreWaveforms,
    target: _Perf20WaveformPreloadedDataset,
    rows: list[int],
    *,
    rank: int,
    world: int,
) -> dict[str, Any]:
    """Clone the complete selected partition into each rank's anonymous RAM."""

    store = cache.store
    expected_bytes = int(store.num_audio) * int(store.bytes_per_audio)
    preload_started = time.perf_counter()
    for audio_id in range(int(store.num_audio)):
        cache._get_audio_id(audio_id)
        if int(rank) == 0 and (audio_id + 1) % 1024 == 0:
            print(
                "[audio-perf20] partition store-to-rank-RAM preload "
                f"audio={audio_id + 1}/{store.num_audio} "
                f"gib={cache.current_bytes / 1024**3:.3f}",
                flush=True,
            )
    preload_seconds = time.perf_counter() - preload_started
    if len(cache.items) != int(store.num_audio) or cache.current_bytes != expected_bytes:
        raise RuntimeError(
            "whole-partition rank-RAM preload cardinality/byte mismatch: "
            f"resident={len(cache.items)}/{store.num_audio} "
            f"bytes={cache.current_bytes}/{expected_bytes}"
        )
    if cache.evictions:
        raise RuntimeError(f"whole-partition rank-RAM preload unexpectedly evicted {cache.evictions} tensors")

    rows_started = time.perf_counter()
    for row in dict.fromkeys(int(value) for value in rows):
        target.items[row] = cache.materialize_from_cached_metadata(row)
    row_materialization_seconds = time.perf_counter() - rows_started
    stats = cache.stats()
    if stats["waveform_misses"] != int(store.num_audio):
        raise RuntimeError(
            "whole-partition preload did not clone each store row exactly once: "
            f"misses={stats['waveform_misses']} expected={store.num_audio}"
        )
    return {
        "loaded_rows": len(target.items),
        "cpu_tensor_bytes": cache.current_bytes,
        "whole_partition_unique_audio": int(store.num_audio),
        "whole_partition_expected_bytes_per_rank": expected_bytes,
        "whole_partition_expected_gib_per_rank": expected_bytes / 1024**3,
        "whole_partition_expected_bytes_all_ranks": expected_bytes * int(world),
        "whole_partition_expected_gib_all_ranks": expected_bytes * int(world) / 1024**3,
        "store_to_rank_ram_seconds": float(preload_seconds),
        "planned_row_materialization_seconds": float(row_materialization_seconds),
        "waveform_cache_stats_after_preload": stats,
        "training_storage_reads_expected": 0,
        "timed_tokenization_and_collate": True,
    }


def _preload_perf20_batches(
    loader: DataLoader,
    sampler: DistributedSampler,
    *,
    epoch: int,
    batch_in_epoch: int,
    required_microbatches: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize the exact bounded PERF20 input stream in rank-local RAM."""

    required = int(required_microbatches)
    if required <= 0:
        raise ValueError(f"PERF20 preload requires a positive batch count, got {required}")
    available = len(loader) - int(batch_in_epoch)
    if required > available:
        raise RuntimeError(
            "PERF20 preload would cross the current sampler epoch: "
            f"required={required} available={available}"
        )

    sampler.set_epoch(int(epoch))
    data_iter = iter(loader)
    _skip_batches(data_iter, int(batch_in_epoch))
    started = time.perf_counter()
    batches = [next(data_iter) for _ in range(required)]
    duration = time.perf_counter() - started

    row_indices = [int(row) for batch in batches for row in batch.get("row_indices", [])]
    row_stream = ",".join(str(row) for row in row_indices).encode("ascii")
    metadata = {
        "mode": "full_preload",
        "required_microbatches": required,
        "loaded_microbatches": len(batches),
        "consumed_microbatches": 0,
        "samples": len(row_indices),
        "duration_seconds": float(duration),
        "barrier_wait_seconds": 0.0,
        "cpu_tensor_bytes": sum(_batch_tensor_bytes(batch) for batch in batches),
        "row_indices_sha256": hashlib.sha256(row_stream).hexdigest(),
        "first_row_indices": row_indices[: min(8, len(row_indices))],
        "last_row_indices": row_indices[-min(8, len(row_indices)):],
    }
    return batches, metadata


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
            output = reload_model(**{key: value for key, value in moved.items() if key not in {"row_indices", "audio2_reused", "waveform_cache_shard_ids"}})
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


def _configure_perf20_partition_inputs(args: argparse.Namespace) -> dict[str, Any] | None:
    """Select and validate one self-contained materialized partition for PERF20."""

    if args.gate != "PERF20" or args.perf20_input_mode != "partition_rank_ram_preload":
        return None
    partition_id = int(args.perf20_partition_id)
    if partition_id < 0:
        raise ValueError(f"--perf20-partition-id must be non-negative, got {partition_id}")
    root = args.perf20_partition_store_root.expanduser().resolve(strict=True)
    if (root / "BUILDING").exists():
        raise RuntimeError(f"partition store root is still being built: {root}")
    report_path = root / "materialization_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"partition store root lacks materialization_report.json: {root}")
    materialization_report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_root = {
        "status": "PASS",
        "format": "reasonaqa_component_partition_stores_v1",
        "duplicated_audio": 0,
        "source_payload_sha256_reverified": True,
    }
    root_mismatches = {
        key: {"expected": expected, "actual": materialization_report.get(key)}
        for key, expected in expected_root.items()
        if materialization_report.get(key) != expected
    }
    if root_mismatches:
        raise RuntimeError(f"partition store root contract mismatch: {root_mismatches}")

    partition_dir = root / f"partition_{partition_id}"
    metadata_path = partition_dir / "metadata.json"
    manifest_path = partition_dir / "rows.jsonl"
    if (partition_dir / "BUILDING").exists():
        raise RuntimeError(f"selected partition is still being built: {partition_dir}")
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"selected partition lacks metadata.json or rows.jsonl: {partition_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_partition = {
        "status": "PASS",
        "format": "manifest_unique_fixed_waveform_store_v1",
        "partition_materialization_format": "reasonaqa_component_partition_stores_v1",
        "partition_id": partition_id,
    }
    partition_mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in expected_partition.items()
        if metadata.get(key) != expected
    }
    if partition_mismatches:
        raise RuntimeError(f"selected partition contract mismatch: {partition_mismatches}")
    if metadata.get("waveform_verification", {}).get("passed") is not True:
        raise RuntimeError(f"selected partition has no passing waveform verification: {partition_dir}")
    partition_reports = materialization_report.get("partitions", [])
    report_entry = next(
        (item for item in partition_reports if int(item.get("partition_id", -1)) == partition_id),
        None,
    )
    if report_entry is None:
        raise RuntimeError(f"materialization report has no partition {partition_id}")
    if report_entry.get("waveform_verification", {}).get("passed") is not True:
        raise RuntimeError(f"materialization report has no passing verification for partition {partition_id}")
    for key in ("manifest_sha256", "index_sha256", "waveform_sha256", "num_unique_audio_files", "total_waveform_bytes"):
        if report_entry.get(key) != metadata.get(key):
            raise RuntimeError(
                f"partition {partition_id} report/metadata mismatch for {key}: "
                f"report={report_entry.get(key)!r} metadata={metadata.get(key)!r}"
            )

    args.train_manifest = manifest_path
    args.shared_waveform_store_dir = partition_dir
    return {
        "partition_id": partition_id,
        "partition_store_root": str(root),
        "partition_store_dir": str(partition_dir),
        "partition_manifest": str(manifest_path),
        "partition_manifest_sha256": str(metadata["manifest_sha256"]),
        "partition_index_sha256": str(metadata["index_sha256"]),
        "partition_waveform_sha256": str(metadata["waveform_sha256"]),
        "partition_plan_sha256": str(metadata.get("partition_plan_sha256")),
        "qa_rows": int(report_entry["qa_rows"]),
        "num_unique_audio_files": int(metadata["num_unique_audio_files"]),
        "total_waveform_bytes": int(metadata["total_waveform_bytes"]),
        "total_waveform_gib": int(metadata["total_waveform_bytes"]) / 1024**3,
        "duplicated_audio_across_partitions": int(materialization_report["duplicated_audio"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank, world, device = _init_dist(args)
    _seed(args.seed, rank)
    preload_alias_conflict = args.preload_data and args.perf20_input_mode not in {"online", "full_preload"}
    if args.preload_data and not preload_alias_conflict:
        args.perf20_input_mode = "full_preload"
    report: dict[str, Any] = {"stage": f"{args.gate.lower()}_audio_5_10x2_5_mesh_mellow", "status": "FAIL", "configuration": vars(args), "rank": rank, "world_size": world, "checks": [], "warnings": [], "hard_failures": [], "environment": _environment_report(device, world)}
    profiler: Any | None = None
    store_prefetcher: _Perf20StorePrefetcher | None = None
    profiler_artifacts: list[dict[str, Any]] = []
    perf_output_preexisting = False
    partition_scope: dict[str, Any] | None = None
    try:
        if preload_alias_conflict:
            raise ValueError("--preload-data cannot be combined with a non-full --perf20-input-mode")
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
            if args.perf20_prefetch_microbatches <= 0 or args.perf20_rank_cache_gib <= 0:
                raise ValueError("PERF20 prefetch depth and rank cache GiB must be positive")
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
        if args.gate != "PERF20" and args.preload_data:
            raise ValueError("--preload-data is isolated to PERF20; standard smoke/formal gates are unchanged")
        if args.gate != "PERF20" and args.perf20_input_mode != "online":
            raise ValueError("--perf20-input-mode is isolated to PERF20; standard smoke/formal gates are unchanged")
        if args.waveform_cache_dir is not None:
            raise ValueError(
                "--waveform-cache-dir has been retired from PERF20; the 64-shard mmap experiment is abandoned"
            )
        if args.gate != "PERF20" and args.shared_waveform_store_dir != Path(DEFAULT_SHARED_WAVEFORM_STORE):
            raise ValueError("--shared-waveform-store-dir is isolated to PERF20")
        if args.gate != "PERF20" and (
            args.perf20_partition_store_root != Path(DEFAULT_COMPONENT_PARTITION_STORE_ROOT)
            or int(args.perf20_partition_id) != 0
        ):
            raise ValueError("--perf20-partition-store-root/--perf20-partition-id are isolated to PERF20")
        partition_scope = _configure_perf20_partition_inputs(args)
        report["configuration"] = vars(args)
        if partition_scope is not None:
            report["partition_scope"] = partition_scope
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
        dataset = ReasonAQADataset(
            args.train_manifest,
            tokenizer,
            unique_waveform_store_dir=(
                args.shared_waveform_store_dir
                if args.gate == "PERF20" and args.perf20_input_mode in {
                    "shared_waveform_store", "store_rank_ram_preload", "store_rank_ram_prefetch",
                    "partition_rank_ram_preload"}
                else None
            ),
        )
        # All causal controls use the identical global DistributedSampler
        # contract.  The shared-store control changes only waveform delivery;
        # no retired 64-shard cache or shard-owned sampler participates.
        sampler = DistributedSampler(
            dataset,
            num_replicas=world,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        waveform_preloaded_dataset = (
            _Perf20WaveformPreloadedDataset(dataset)
            if args.gate == "PERF20" and args.perf20_input_mode in {
                "waveform_preload", "store_rank_ram_preload", "partition_rank_ram_preload"}
            else None
        )
        loader_dataset = waveform_preloaded_dataset or dataset
        timed_collator = _TimedReasonAQACollator(tokenizer) if args.gate == "PERF20" else None
        collate_fn = timed_collator if timed_collator is not None else lambda rows: collate_reasonaqa(rows, tokenizer)
        perf_loader_generator = None
        if args.gate == "PERF20":
            # Iterator construction consumes only this dedicated RNG, never
            # the model/dropout CPU RNG.  Modes that materialize iterators at
            # different times therefore remain comparable.
            perf_loader_generator = torch.Generator()
            perf_loader_generator.manual_seed(int(args.seed) + 100_003 * int(rank) + 17)
        loader = DataLoader(
            loader_dataset,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            generator=perf_loader_generator,
        )
        sampler_audit: dict[str, Any] = {
            "kind": "torch_distributed_sampler",
            "shuffle": True,
            "seed": int(args.seed),
            "drop_last": True,
            "batches_per_rank": len(loader),
        }
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
        if batch_in_epoch >= len(loader):
            epoch += batch_in_epoch // len(loader)
            batch_in_epoch = batch_in_epoch % len(loader)

        preloaded_batches: list[dict[str, Any]] | None = None
        rank_ram_cache: _RankLocalStoreWaveforms | None = None
        input_preparation: dict[str, Any] | None = None
        preloaded_consumed = 0
        first_forward_input_audit: dict[str, Any] | None = None
        if perf_mode:
            required_microbatches = (int(max_steps) - int(optimizer_step)) * int(args.gradient_accumulation_steps)
            planned_rows = _planned_perf20_rows(
                sampler,
                epoch=epoch,
                batch_in_epoch=batch_in_epoch,
                micro_batch_size=args.micro_batch_size,
                required_microbatches=required_microbatches,
            )
            input_preparation = {
                "mode": str(args.perf20_input_mode),
                "required_microbatches": int(required_microbatches),
                "duration_seconds": 0.0,
                "barrier_wait_seconds": 0.0,
                "cpu_tensor_bytes": 0,
                "filesystem_cache_contract": (
                    "no explicit warm; OS cache state is uncontrolled and this mode must not be described as guaranteed cold"
                    if args.perf20_input_mode == "online" else
                    "exact all-rank bounded-run file union is read before timing; online decode/resample remains timed"
                    if args.perf20_input_mode == "warm_online" else
                    "rank-local decoded waveforms for exact bounded rows are resident before timing"
                    if args.perf20_input_mode == "waveform_preload" else
                    "rank0 reads only the all-rank bounded-run unique decoded waveform regions into node page cache before timing"
                    if args.perf20_input_mode == "shared_waveform_store" else
                    "exact rank-local store waveforms are copied into owned CPU tensors before timing"
                    if args.perf20_input_mode == "store_rank_ram_preload" else
                    "the complete selected component partition is copied into every rank's owned CPU tensors before timing"
                    if args.perf20_input_mode == "partition_rank_ram_preload" else
                    "bounded rank-local owned CPU tensor LRU plus asynchronous microbatch waveform prefetch"
                    if args.perf20_input_mode == "store_rank_ram_prefetch" else
                    "rank-local fully collated CPU batches are resident before timing"
                ),
                **_row_stream_metadata(planned_rows),
                "process_faults_before": _process_fault_snapshot(),
                "process_io_before": _process_io_snapshot(),
                "cgroup_memory_before": _cgroup_memory_snapshot(),
                "cgroup_cpu_before": _cgroup_cpu_snapshot(),
                "process_runtime_before": _process_runtime_snapshot(),
                "thread_runtime_configuration": _thread_runtime_configuration(),
                "perf20_dataloader_uses_dedicated_generator": perf_loader_generator is not None,
            }
            saved_preparation_rng = _rng_state(device)
            preparation_started = time.perf_counter()
            local_preparation_error: str | None = None
            try:
                try:
                    if args.perf20_input_mode == "warm_online":
                        input_preparation.update(
                            _warm_exact_perf20_files(dataset, planned_rows, rank=rank, world=world)
                        )
                    elif args.perf20_input_mode == "waveform_preload":
                        if waveform_preloaded_dataset is None:
                            raise RuntimeError("waveform_preload mode did not install its dataset wrapper")
                        _, waveform_metadata = _preload_perf20_waveforms(
                            dataset,
                            planned_rows,
                            target=waveform_preloaded_dataset,
                        )
                        input_preparation.update(waveform_metadata)
                    elif args.perf20_input_mode == "shared_waveform_store":
                        if dataset.unique_waveform_store is None:
                            raise RuntimeError("shared_waveform_store mode did not install its store reader")
                        input_preparation.update(
                            _warm_shared_waveform_store(dataset, planned_rows, rank=rank, world=world)
                        )
                    elif args.perf20_input_mode == "store_rank_ram_preload":
                        if waveform_preloaded_dataset is None:
                            raise RuntimeError("store rank-RAM preload did not install its dataset wrapper")
                        rank_ram_cache = _RankLocalStoreWaveforms(dataset)
                        input_preparation.update(_preload_store_rank_ram(
                            rank_ram_cache, waveform_preloaded_dataset, planned_rows,
                        ))
                    elif args.perf20_input_mode == "partition_rank_ram_preload":
                        if waveform_preloaded_dataset is None:
                            raise RuntimeError("partition rank-RAM preload did not install its dataset wrapper")
                        if partition_scope is None:
                            raise RuntimeError("partition rank-RAM preload has no validated partition scope")
                        rank_ram_cache = _RankLocalStoreWaveforms(dataset)
                        input_preparation.update(partition_scope)
                        input_preparation.update(_preload_partition_store_rank_ram(
                            rank_ram_cache, waveform_preloaded_dataset, planned_rows,
                            rank=rank, world=world,
                        ))
                    elif args.perf20_input_mode == "store_rank_ram_prefetch":
                        if timed_collator is None:
                            raise RuntimeError("store rank-RAM prefetch requires the timed collator")
                        rank_ram_cache = _RankLocalStoreWaveforms(
                            dataset, max_bytes=int(args.perf20_rank_cache_gib * 1024**3),
                        )
                        store_prefetcher = _Perf20StorePrefetcher(
                            rank_ram_cache, planned_rows, args.micro_batch_size,
                            args.perf20_prefetch_microbatches, timed_collator,
                        )
                        started_priming = time.perf_counter()
                        store_prefetcher.start_and_prime()
                        input_preparation.update({
                            "initial_primed_microbatches": store_prefetcher.depth,
                            "prefetch_queue_capacity_microbatches": args.perf20_prefetch_microbatches,
                            "rank_cache_capacity_bytes": rank_ram_cache.max_bytes,
                            "cpu_tensor_bytes": rank_ram_cache.current_bytes,
                            "initial_prime_seconds": time.perf_counter() - started_priming,
                            "waveform_cache_stats_after_prime": rank_ram_cache.stats(),
                        })
                    elif args.perf20_input_mode == "full_preload":
                        preloaded_batches, full_metadata = _preload_perf20_batches(
                            loader,
                            sampler,
                            epoch=epoch,
                            batch_in_epoch=batch_in_epoch,
                            required_microbatches=required_microbatches,
                        )
                        if full_metadata["row_indices_sha256"] != input_preparation["row_indices_sha256"]:
                            raise RuntimeError("full preload row stream differs from the planned causal-control stream")
                        input_preparation.update(full_metadata)
                    elif args.perf20_input_mode != "online":
                        raise RuntimeError(f"unsupported PERF20 input mode: {args.perf20_input_mode}")
                except Exception as exc:
                    local_preparation_error = repr(exc)
            finally:
                # Data preparation must not change the stochastic training
                # stream (notably bridge dropout) across all input controls.
                _restore_rng_state(saved_preparation_rng, device)
            input_preparation["duration_seconds"] = float(time.perf_counter() - preparation_started)
            preparation_errors: list[str | None] = [None] * world
            if world > 1:
                dist.all_gather_object(preparation_errors, local_preparation_error)
            else:
                preparation_errors[0] = local_preparation_error
            failed_preparations = {
                failed_rank: error
                for failed_rank, error in enumerate(preparation_errors)
                if error is not None
            }
            if failed_preparations:
                raise RuntimeError(f"PERF20 input preparation failed by rank: {failed_preparations}")
            # Every control uses the same pre-measurement barrier so rank skew
            # during preparation never leaks into optimizer-step timing.
            if world > 1:
                barrier_started = time.perf_counter()
                dist.barrier()
                input_preparation["barrier_wait_seconds"] = float(time.perf_counter() - barrier_started)
            input_preparation["process_faults_after"] = _process_fault_snapshot()
            input_preparation["process_io_after"] = _process_io_snapshot()
            input_preparation["cgroup_memory_after"] = _cgroup_memory_snapshot()
            input_preparation["cgroup_cpu_after"] = _cgroup_cpu_snapshot()
            input_preparation["process_runtime_after"] = _process_runtime_snapshot()
            if rank == 0:
                print(
                    "[audio-perf20] causal input preparation complete "
                    f"mode={args.perf20_input_mode} microbatches={required_microbatches} "
                    f"row_sha256={input_preparation['row_indices_sha256']}",
                    flush=True,
                )

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
        preloaded_data_iter = (
            iter(preloaded_batches) if preloaded_batches is not None else
            iter(store_prefetcher) if store_prefetcher is not None else None
        )
        training_measurement_started = time.perf_counter()
        while optimizer_step < max_steps:
            if preloaded_data_iter is None:
                sampler.set_epoch(epoch)
                data_iter = iter(loader)
                _skip_batches(data_iter, batch_in_epoch)
            else:
                data_iter = preloaded_data_iter
            completed_optimizer_steps = batch_in_epoch // args.gradient_accumulation_steps
            for _ in range(completed_optimizer_steps, steps_epoch):
                if optimizer_step >= max_steps:
                    break
                collator_before = timed_collator.snapshot() if timed_collator is not None else {}
                prefetch_before = store_prefetcher.timing_snapshot() if store_prefetcher is not None else {}
                cache_before = rank_ram_cache.stats() if rank_ram_cache is not None else {}
                process_runtime_before_step = _process_runtime_snapshot() if perf_mode else {}
                cgroup_cpu_before_step = _cgroup_cpu_snapshot() if perf_mode else {}
                step_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                optimizer.zero_grad(set_to_none=True)
                phase_events = _CudaPhaseEvents(enabled=perf_mode)
                data_wait_seconds = 0.0
                host_phase_seconds: dict[str, float] = {}
                input_pipeline_step: dict[str, Any] = {}
                local_answer_tokens = 0
                last_local_answer_tokens = 0
                local_text_nonpadding_tokens = 0
                local_multimodal_tokens = 0
                local_max_sequence_length = 0
                microbatch_metrics: list[dict[str, Any]] = []
                for micro in range(args.gradient_accumulation_steps):
                    if perf_mode:
                        data_wait_started = time.perf_counter()
                        with record_function("data_wait"):
                            batch = next(data_iter)
                        if preloaded_data_iter is not None:
                            preloaded_consumed += 1
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
                        microbatch_item: dict[str, Any] = {
                            "micro": int(micro),
                            "batch_size": batch_size,
                            "sequence_length": sequence_length,
                            "text_nonpadding_tokens": text_nonpadding,
                            "answer_tokens": answer_nonpadding,
                            "multimodal_tokens": multimodal_tokens,
                        }
                        cache_shard_pairs = batch.get("waveform_cache_shard_ids")
                        if cache_shard_pairs is not None:
                            microbatch_item["waveform_cache_primary_shards"] = sorted({int(pair[0]) for pair in cache_shard_pairs})
                            microbatch_item["waveform_cache_shards_touched"] = sorted({int(shard_id) for pair in cache_shard_pairs for shard_id in pair})
                        microbatch_metrics.append(microbatch_item)
                    else:
                        batch = next(data_iter)
                    if perf_mode and first_forward_input_audit is None:
                        first_forward_input_audit = {
                            "optimizer_step": int(optimizer_step + 1),
                            "microbatch": int(micro),
                            "rng_before_first_forward": _rng_fingerprint(device),
                            "batch_before_device_transfer": _batch_fingerprint(batch),
                        }
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
                                return ddp(**{k: v for k, v in batch.items() if k not in {"row_indices", "audio2_reused", "waveform_cache_shard_ids"}})
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
                if perf_mode:
                    if timed_collator is None:
                        raise RuntimeError("PERF20 timed collator is unavailable")
                    collator_after = timed_collator.snapshot()
                    for phase, value in collator_after.items():
                        host_phase_seconds[phase] = max(0.0, float(value) - float(collator_before.get(phase, 0.0)))
                    if store_prefetcher is not None:
                        prefetch_after = store_prefetcher.timing_snapshot()
                        host_phase_seconds["queue_get"] = max(
                            0.0, float(prefetch_after["queue_get"]) - float(prefetch_before.get("queue_get", 0.0))
                        )
                        host_phase_seconds["queue_empty_polls"] = max(
                            0.0, float(prefetch_after["queue_empty_polls"]) - float(prefetch_before.get("queue_empty_polls", 0.0))
                        )
                        depth_start = int(prefetch_before.get("depth_observations", 0.0))
                        depth_end = int(prefetch_after.get("depth_observations", 0.0))
                        input_pipeline_step["queue_depth_before_get"] = store_prefetcher.queue_depth_before_get[depth_start:depth_end]
                        input_pipeline_step["queue_depth_after_get"] = store_prefetcher.queue_depth_after_get[depth_start:depth_end]
                    if rank_ram_cache is not None:
                        cache_after = rank_ram_cache.stats()
                        for source, target in (
                            ("locate_seconds", "store_locate"),
                            ("mmap_view_seconds", "store_mmap_view"),
                            ("clone_seconds", "store_clone"),
                            ("dataset_item_seconds", "store_dataset_item"),
                        ):
                            host_phase_seconds[target] = max(
                                0.0, float(cache_after[source]) - float(cache_before.get(source, 0.0))
                            )
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
                process_runtime_after_step = _process_runtime_snapshot() if perf_mode else {}
                cgroup_cpu_after_step = _cgroup_cpu_snapshot() if perf_mode else {}
                answer_tokens = int(reduced[0].item())
                reported_answer_tokens = answer_tokens if args.gate == "PERF20" else int(reduced[3].item())
                text_nonpadding_tokens = int(reduced[1].item()) if perf_mode else 0
                multimodal_tokens = int(reduced[2].item()) if perf_mode else 0
                max_sequence_length = int(max_sequence.item()) if perf_mode else 0
                item = {"step": optimizer_step, "total_steps": max_steps, "progress_percent": 100.0 * optimizer_step / max(1, max_steps), "epoch": epoch, "batch_in_epoch": batch_in_epoch, "steps_per_epoch": steps_epoch, "loss": float(output.loss.detach().cpu()), "lr": float(optimizer.param_groups[0]["lr"]), "lr_before_optimizer_step": lr_before_optimizer_step, "grad_norm": float(grad_norm), "effective_answer_tokens": reported_answer_tokens, "aggregated_answer_tokens": answer_tokens, "nonpadding_tokens": text_nonpadding_tokens, "multimodal_tokens": multimodal_tokens, "max_sequence_length": max_sequence_length, "microbatches": microbatch_metrics, "input_pipeline_step": input_pipeline_step, "phase_timings": {"host_seconds": phase_timings_host, "device_seconds": phase_timings_device}, "phase_timings_host_seconds": phase_timings_host, "phase_timings_device_seconds": phase_timings_device, "step_time_seconds": elapsed, "samples_per_second": global_samples / elapsed, "audio_seconds_per_second": global_samples * 20.0 / elapsed, "multimodal_tokens_per_second": multimodal_tokens / elapsed, "nonpadding_tokens_per_second": text_nonpadding_tokens / elapsed, "answer_tokens_per_second": answer_tokens / elapsed, "gpu_memory_allocated_gib": float(torch.cuda.memory_allocated(device) / 1024**3), "gpu_memory_reserved_gib": float(torch.cuda.memory_reserved(device) / 1024**3), "gpu_memory_max_allocated_gib": float(torch.cuda.max_memory_allocated(device) / 1024**3), "gpu_memory_max_reserved_gib": float(torch.cuda.max_memory_reserved(device) / 1024**3), "process_runtime_delta": _nested_numeric_delta(process_runtime_before_step, process_runtime_after_step) if perf_mode else None, "cgroup_cpu_delta": _nested_numeric_delta(cgroup_cpu_before_step, cgroup_cpu_after_step) if perf_mode else None, "router_stats": _router_stats(owner)}
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
        training_measurement_seconds = time.perf_counter() - training_measurement_started
        if input_preparation is not None:
            input_preparation["consumed_preloaded_microbatches"] = int(preloaded_consumed)
            input_preparation["training_measurement_seconds"] = float(training_measurement_seconds)
            input_preparation["global_samples_processed"] = int(
                (optimizer_step - start_step) * args.micro_batch_size * world * args.gradient_accumulation_steps
            )
            if store_prefetcher is not None:
                store_prefetcher.close()
                if store_prefetcher.error is not None or store_prefetcher.produced != required_microbatches or store_prefetcher.consumed != required_microbatches:
                    raise RuntimeError(
                        "PERF20 rank-local waveform prefetch stream incomplete: "
                        f"produced={store_prefetcher.produced} consumed={store_prefetcher.consumed} "
                        f"expected={required_microbatches} error={store_prefetcher.error!r}"
                    )
                input_preparation["waveform_cache_stats_after_training"] = rank_ram_cache.stats()
                input_preparation["prefetch_stats_after_training"] = store_prefetcher.stats()
            elif rank_ram_cache is not None:
                cache_after_training = rank_ram_cache.stats()
                input_preparation["waveform_cache_stats_after_training"] = cache_after_training
                if args.perf20_input_mode == "partition_rank_ram_preload":
                    cache_after_preload = input_preparation["waveform_cache_stats_after_preload"]
                    unchanged = all(
                        cache_after_training[key] == cache_after_preload[key]
                        for key in ("waveform_misses", "cloned_bytes", "waveform_evictions", "resident_unique_audio")
                    )
                    input_preparation["training_cache_population_unchanged"] = unchanged
                    if not unchanged:
                        raise RuntimeError(
                            "partition rank-RAM cache population changed during measured training: "
                            f"before={cache_after_preload} after={cache_after_training}"
                        )
            input_preparation["process_faults_after_training"] = _process_fault_snapshot()
            input_preparation["process_io_after_training"] = _process_io_snapshot()
            input_preparation["cgroup_memory_after_training"] = _cgroup_memory_snapshot()
            input_preparation["cgroup_cpu_after_training"] = _cgroup_cpu_snapshot()
            input_preparation["process_runtime_after_training"] = _process_runtime_snapshot()
            input_preparation["thread_runtime_configuration_after_training"] = _thread_runtime_configuration()
            input_preparation["first_forward_input_audit"] = first_forward_input_audit
            input_preparation["process_runtime_training_delta"] = _nested_numeric_delta(
                input_preparation.get("process_runtime_after"), input_preparation.get("process_runtime_after_training")
            )
            input_preparation["cgroup_cpu_training_delta"] = _nested_numeric_delta(
                input_preparation.get("cgroup_cpu_after"), input_preparation.get("cgroup_cpu_after_training")
            )
        if preloaded_batches is not None and input_preparation is not None:
            if preloaded_consumed != int(input_preparation["required_microbatches"]):
                raise RuntimeError(
                    "PERF20 did not consume the exact preloaded stream: "
                    f"expected={input_preparation['required_microbatches']} actual={preloaded_consumed}"
                )
        per_rank_timing = _gather_perf20_rank_timings(
            rank, world, metrics, input_preparation, args=args, max_steps=max_steps,
        ) if args.gate == "PERF20" else None
        report.update({"status": "PASS", "start_step": start_step, "end_step": optimizer_step, "optimizer_steps": optimizer_step, "steps_per_epoch": steps_epoch, "dropped_microbatches_per_epoch": dropped_microbatches, "total_formal_steps": formal_steps, "warmup_steps": args.warmup_steps, "effective_global_batch_size": int(args.micro_batch_size * world * args.gradient_accumulation_steps), "metrics": metrics if rank == 0 else [], "ddp_broadcast_buffers": False, "router_policy": "warning_only", "routing_stats": {"enabled": args.gate != "PERF20", "mode": "disabled_for_perf20" if args.gate == "PERF20" else "continuous_per_forward", "reported_in_each_step": args.gate != "PERF20", "reason": "per-router .cpu() statistics would add CUDA synchronizations to the PERF20 timing path" if args.gate == "PERF20" else None}, "model_trainable_audit": (ddp.module if hasattr(ddp, "module") else ddp).trainable_parameter_audit(), "runtime_gradient_audit": runtime_gradient_audit, "resume_position": {"epoch": epoch, "batch_in_epoch": batch_in_epoch}, "checkpoints": report.get("checkpoints", [])})
        report["correctness_audit"] = {
            "mesh_runtime_gradient_audit": runtime_gradient_audit,
            "answer_only_labels": "build_labels enforces -100 outside real answer intervals and exact answer token count",
            "routing_stats": "disabled for PERF20 timing to avoid per-router CUDA synchronizations; first-step gradient audit retained, then gradient_audit_mode disabled" if args.gate == "PERF20" else "continuous per forward; first-step gradient audit retained, then gradient_audit_mode disabled",
        }
        if args.gate == "PERF20" and rank == 0:
            report["per_rank_timing"] = per_rank_timing
            input_mode = str(args.perf20_input_mode)
            timed_source = {
                "online": "DataLoader reads/decodes/resamples encoded audio and tokenizes/collates during each measured step",
                "warm_online": "same online DataLoader path after exact bounded-run file bytes were read before timing",
                "waveform_preload": "rank-local waveform RAM lookup plus timed tokenization/collate",
                "full_preload": "rank-local fully collated CPU batch list",
                "shared_waveform_store": "node-shared fixed-stride waveform mmap after rank0 warmed the exact all-rank 20-step audio union by file offset",
                "store_rank_ram_preload": "store mmap views cloned into rank-owned CPU tensors before timing; timed tokenization and collate",
                "store_rank_ram_prefetch": "bounded rank-owned CPU waveform LRU with asynchronous producer and timed consumer-side collate",
                "partition_rank_ram_preload": "complete selected component partition cloned into each rank's anonymous CPU RAM before timing; timed tokenization and collate",
            }[input_mode]
            report["data_pipeline"] = {
                "mode": input_mode,
                "preloaded": input_mode in {"waveform_preload", "full_preload", "shared_waveform_store", "store_rank_ram_preload", "partition_rank_ram_preload"},
                "waveform_cache_enabled": False,
                "shared_waveform_store_enabled": input_mode in {"shared_waveform_store", "store_rank_ram_preload", "store_rank_ram_prefetch", "partition_rank_ram_preload"},
                "whole_partition_rank_ram_enabled": input_mode == "partition_rank_ram_preload",
                "retired_waveform_shard_experiment": True,
                "timed_source": timed_source,
                "training_dataloader_accesses": 0 if input_mode in {"full_preload", "store_rank_ram_prefetch"} else PERF20_STEPS * args.gradient_accumulation_steps,
                "synchronization": "one barrier in every causal-control mode after preparation and before profiler/timing",
                "sampler": sampler_audit,
                "input_preparation_by_rank": per_rank_timing.get("input_preparation_by_rank", []) if per_rank_timing else [],
                "input_preparation_summary": per_rank_timing.get("input_preparation_summary") if per_rank_timing else None,
            }
            report["steady_state_summary"] = _perf_steady_summary(metrics, args, max_steps)
            report["timing_semantics"] = {
                "step_time_seconds": "rank0 wall-clock from before the first microbatch data wait through the single CUDA synchronize after all metrics/collectives; this is the completion-inclusive step duration",
                "phase_timings_host_seconds": "host wall/enqueue timings; data_wait surrounds next(data_iter); queue_get is measured directly inside the prefetch consumer; collate is decomposed into tokenize, text_tensor_build, waveform_stack, and batch_metadata; concurrent store clone phases can overlap consumer phases and must not be summed",
                "store_copy": "store_clone is load_audio_id mmap-view clone wall time accumulated by the producer; *_thread_cpu fields cover only the calling Python thread, not native helper threads; neither clone wall nor thread CPU is pure storage I/O time",
                "system_runtime": "per-step process scheduler/context-switch and cgroup CPU counter deltas are sampled outside the measured step wall interval; unavailable cgroup candidates retain path and exception diagnostics",
                "phase_timings_device_seconds": "CUDA event elapsed timings resolved after one unified end-of-step synchronize; per-microbatch device values are summed within the optimizer step; metrics_collectives includes the device/NCCL work through its event and therefore is completion-inclusive",
                "host_to_device": "rank0 CUDA event elapsed time around tensor .to(device) for every microbatch; host_to_device_enqueue separately records host dispatch time",
                "forward_backward_optimizer": "rank0 CUDA event elapsed time; DDP gradient collectives are included in backward; host enqueue fields are reported separately",
                "metrics_collectives": "global token counts use one identical all_reduce sequence on every rank; no rank-specific collective is introduced, and its device timing is not the enqueue-only host latency",
                "per_rank_timing": "all ranks retain local timings; per-rank distributions and straggler tables use exactly the same steady-state step set as rank0, and one gather_object runs only after the final measured optimizer step",
                "causal_input_controls": "within a fixed effective manifest, PERF20 reruns preserve the per-rank row hash, restored model RNG state, dedicated DataLoader generator, first-forward RNG/batch fingerprints, and one pre-measurement barrier; partition_rank_ram_preload intentionally selects a partition manifest and must be compared by its separately reported row hash/token load",
                "steady_state": "optimizer steps 6-20 by default, excluding profiler wait/warmup/active steps when profiling is enabled",
            }
    except Exception as exc:
        if args.gate == "PERF20":
            report.setdefault("correctness_audit", {})["status"] = "not_completed"
        report["hard_failures"].append({"error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        if store_prefetcher is not None:
            store_prefetcher.close()
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
