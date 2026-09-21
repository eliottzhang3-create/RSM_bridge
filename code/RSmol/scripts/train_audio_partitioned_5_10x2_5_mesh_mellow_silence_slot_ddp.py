#!/usr/bin/env python3
"""Six-partition MeSH training with fixed 260-token runtime-silence slots.

This is an isolated experiment.  It shares only audited partition lifecycle
helpers with the compact MeSH route; checkpoints, reports, model/data modules,
configuration filenames, and smoke gates use a distinct contract.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
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
import train_audio_partitioned_5_10x2_5_mesh_mellow_ddp as partition_common
from audio_5_10x2_5_mesh_mellow_silence_slot.data import (
    ReasonAQADataset,
    collate_reasonaqa,
)
from audio_5_10x2_5_mesh_mellow_silence_slot.model import (
    AUDIO_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    FIXED_PREFIX_TOKEN_CONTRACT,
    MAPPER_CONTRACT,
    SILENCE_SLOT_ARCHITECTURE_CONTRACT,
    AudioMeshSilenceSlotConfig,
    AudioMeshSilenceSlotModel,
    _load_mellow_wrapper,
)
from recursive_model_5_10x2_5_mesh import RecursiveLlamaForCausalLM, register_auto_class


CONTRACT = "component_partitions6_rank_ram_fixed260_runtime_silence_second_slot_answer_eos_v2"
CONFIG_FILENAME = "audio_mesh_fixed260_silence_slot_config.json"
ANSWER_TERMINATION_CONTRACT = {
    "token": "<|endoftext|>",
    "included_in_max_answer_tokens": True,
    "supervised": True,
}
SILENCE_SLOT_CONTRACT = {
    "single_audio_second_slot": "one exact-zero waveform created on GPU per containing microbatch",
    "silence_persisted_or_loaded": False,
    "silence_h2d_transfer": False,
    "silence_shape": [1, 1, 320000],
    "explicit_identical_dual_audio": "reuse audio1 encoder embedding; retain second slot",
    "distinct_dual_audio": "encode real audio2",
    "fixed_prefix_tokens_per_row": AUDIO_PREFIX_TOKENS,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument(
        "--partition-store-root",
        type=Path,
        default=Path(base.DEFAULT_COMPONENT_PARTITION_STORE_ROOT),
    )
    parser.add_argument(
        "--model-path", "--mesh-checkpoint", dest="model_path", type=Path,
        default=Path(base.DEFAULT_MESH),
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke20-report", type=Path)
    parser.add_argument("--smoke-resume-report", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--htsat-checkpoint", type=Path, default=Path(base.DEFAULT_HTSAT))
    parser.add_argument("--mellow-root", type=Path, default=Path(base.DEFAULT_MELLOW))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--checkpoint-retention", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dist-timeout-minutes", type=int, default=30)
    parser.add_argument("--release-min-fraction", type=float, default=0.70)
    parser.add_argument("--release-timeout-seconds", type=int, default=120)
    return parser.parse_args(argv)


def _validate_resume_artifact(path: Path) -> dict[str, Any]:
    required = (
        "mesh_model/config.json",
        "tokenizer/tokenizer_config.json",
        "audio_bridge.pt",
        "training_state.pt",
        CONFIG_FILENAME,
        "checkpoint_complete.json",
    )
    if not path.is_dir():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"silence-slot resume checkpoint is incomplete: {missing}")
    config = json.loads((path / CONFIG_FILENAME).read_text(encoding="utf-8"))
    marker = json.loads((path / "checkpoint_complete.json").read_text(encoding="utf-8"))
    expected = {
        "contract": CONTRACT,
        "architecture_contract": SILENCE_SLOT_ARCHITECTURE_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "compact_single_audio_prefix": False,
        "prefix_tokens": FIXED_PREFIX_TOKEN_CONTRACT,
        "answer_termination": ANSWER_TERMINATION_CONTRACT,
        "silence_slot": SILENCE_SLOT_CONTRACT,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items() if config.get(key) != value
    }
    if marker.get("status") != "complete" or marker.get("contract") != CONTRACT or mismatches:
        raise RuntimeError(
            "not a complete compatible fixed260 silence-slot checkpoint: "
            f"marker={marker} mismatches={mismatches}"
        )
    return config


def _validate_launch_contract(args: argparse.Namespace) -> None:
    expected_scalars = {
        "epochs": 10,
        "world_size": 8,
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "max_lr": 1e-3,
        "min_lr": 1e-4,
        "save_every": 500,
        "checkpoint_retention": 4,
        "seed": 0,
    }
    mismatches = {
        key: {"expected": expected, "actual": getattr(args, key)}
        for key, expected in expected_scalars.items()
        if getattr(args, key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "fixed260 silence-slot route requires the current MeSH six-partition "
            f"training configuration: {mismatches}"
        )
    expected_paths = {
        "partition_store_root": Path(base.DEFAULT_COMPONENT_PARTITION_STORE_ROOT).resolve(),
        "htsat_checkpoint": Path(base.DEFAULT_HTSAT).resolve(),
        "mellow_root": Path(base.DEFAULT_MELLOW).resolve(),
    }
    if args.resume_from is None:
        expected_paths["model_path"] = Path(base.DEFAULT_MESH).resolve()
    actual_paths = {key: getattr(args, key).resolve() for key in expected_paths}
    if actual_paths != expected_paths:
        raise RuntimeError(
            "fixed260 silence-slot route requires canonical MeSH/data/audio sources: "
            f"actual={actual_paths} expected={expected_paths}"
        )
    if args.tokenizer_path is not None and args.resume_from is None:
        raise ValueError("fresh silence-slot training uses the canonical MeSH checkpoint tokenizer")


def _load_model(args: argparse.Namespace, device: torch.device) -> tuple[AudioMeshSilenceSlotModel, Any]:
    register_auto_class()
    from transformers import AutoTokenizer

    if args.resume_from is not None:
        _validate_resume_artifact(args.resume_from)
    composite = args.resume_from
    model_path = composite / "mesh_model" if composite is not None else args.model_path
    tokenizer_path = args.tokenizer_path or (composite / "tokenizer" if composite is not None else model_path)
    if composite is not None and args.tokenizer_path is not None:
        if args.tokenizer_path.resolve() != (composite / "tokenizer").resolve():
            raise ValueError("silence-slot resume must use its checkpoint tokenizer")
    mesh = RecursiveLlamaForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper, htsat, provenance = _load_mellow_wrapper(
        args.mellow_root, args.htsat_checkpoint, device
    )
    model = AudioMeshSilenceSlotModel(
        mesh.to(device),
        tokenizer,
        wrapper,
        htsat,
        config=AudioMeshSilenceSlotConfig(),
    )
    if composite is not None:
        audio_state = torch.load(
            composite / "audio_bridge.pt", map_location=device, weights_only=False
        )
        if not isinstance(audio_state.get("bridge"), dict) or not audio_state["bridge"]:
            raise RuntimeError("silence-slot checkpoint has no bridge state")
        if not isinstance(audio_state.get("c2l"), dict) or not audio_state["c2l"]:
            raise RuntimeError("silence-slot checkpoint has no c2l state")
        model.bridge.load_state_dict(audio_state["bridge"], strict=True)
        model.htsat_wrapper.c2l.load_state_dict(audio_state["c2l"], strict=True)
    model._audio_provenance = provenance
    return model.to(device), tokenizer


def _checkpoint(
    path: Path,
    model: AudioMeshSilenceSlotModel,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    args: argparse.Namespace,
    inventory: dict[str, Any],
    schedule: list[dict[str, int]],
    cursor: dict[str, int],
    rank: int,
    world: int,
    device: torch.device,
    plan_hash: str | None,
) -> None:
    rng = partition_common._gather(base._rng_state(device), world)
    if rank != 0:
        return
    if path.exists():
        raise FileExistsError(f"refusing checkpoint overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent))
    published = False
    try:
        model.mesh_model.save_pretrained(temporary / "mesh_model", safe_serialization=False)
        tokenizer.save_pretrained(temporary / "tokenizer")
        torch.save(base._trainable_state(model), temporary / "audio_bridge.pt")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": cursor["global_step"],
                "cursor": cursor,
                "plan_hash": plan_hash,
                "rng_states_by_rank": {str(index): state for index, state in enumerate(rng)},
            },
            temporary / "training_state.pt",
        )
        config = {
            "contract": CONTRACT,
            "architecture_contract": SILENCE_SLOT_ARCHITECTURE_CONTRACT,
            "mapper_contract": MAPPER_CONTRACT,
            "compact_single_audio_prefix": False,
            "prefix_tokens": FIXED_PREFIX_TOKEN_CONTRACT,
            "answer_termination": ANSWER_TERMINATION_CONTRACT,
            "silence_slot": SILENCE_SLOT_CONTRACT,
            "mode": args.mode,
            "inventory": inventory,
            "schedule": schedule,
            "epochs": args.epochs,
            "world_size": world,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_global_batch_size": world * args.micro_batch_size * args.gradient_accumulation_steps,
            "seed": args.seed,
            "max_lr": args.max_lr,
            "min_lr": args.min_lr,
            "warmup_steps": args.warmup_steps,
            "total_steps": sum(item["steps"] for item in schedule),
            "optimizer": "AdamW",
            "optimizer_betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "gradient_clip_norm": 0.5,
            "save_every": args.save_every,
            "checkpoint_retention": args.checkpoint_retention,
            "dist_timeout_minutes": args.dist_timeout_minutes,
            "release_min_fraction": args.release_min_fraction,
            "release_timeout_seconds": args.release_timeout_seconds,
            "htsat_checkpoint": str(args.htsat_checkpoint.resolve()),
            "mellow_root": str(args.mellow_root.resolve()),
            "mellow_provenance": model._audio_provenance,
        }
        (temporary / CONFIG_FILENAME).write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "checkpoint_complete.json").write_text(
            json.dumps({"status": "complete", "global_step": cursor["global_step"], "contract": CONTRACT}) + "\n",
            encoding="utf-8",
        )
        for name in (
            "mesh_model/config.json", "tokenizer/tokenizer_config.json", "audio_bridge.pt",
            "training_state.pt", CONFIG_FILENAME, "checkpoint_complete.json",
        ):
            if not (temporary / name).is_file():
                raise RuntimeError(f"checkpoint missing {name}")
        temporary.replace(path)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def _resume(
    path: Path,
    args: argparse.Namespace,
    inventory: dict[str, Any],
    schedule: list[dict[str, int]],
    optimizer: Any,
    scheduler: Any,
    rank: int,
    device: torch.device,
    provenance: dict[str, Any],
) -> tuple[dict[str, int], str | None]:
    config = _validate_resume_artifact(path)
    checks = {
        "mode": args.mode,
        "inventory": inventory,
        "schedule": schedule,
        "epochs": args.epochs,
        "world_size": args.world_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "max_lr": args.max_lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "total_steps": sum(item["steps"] for item in schedule),
        "save_every": args.save_every,
        "checkpoint_retention": args.checkpoint_retention,
        "dist_timeout_minutes": args.dist_timeout_minutes,
        "release_min_fraction": args.release_min_fraction,
        "release_timeout_seconds": args.release_timeout_seconds,
        "htsat_checkpoint": str(args.htsat_checkpoint.resolve()),
        "mellow_root": str(args.mellow_root.resolve()),
    }
    for key, expected in checks.items():
        if config.get(key) != expected:
            raise RuntimeError(f"silence-slot resume contract differs in {key}")
    if config.get("mellow_provenance", {}).get("mellow_htsat_sha256") != provenance.get("mellow_htsat_sha256"):
        raise RuntimeError("silence-slot resume Mellow implementation SHA256 mismatch")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    cursor = state["cursor"]
    marker = json.loads((path / "checkpoint_complete.json").read_text(encoding="utf-8"))
    expected_step = sum(item["steps"] for item in schedule[:cursor["segment"]]) + cursor["segment_step"]
    path_step_text = path.name.removeprefix("checkpoint-")
    path_step = int(path_step_text) if path.name.startswith("checkpoint-") and path_step_text.isdigit() else None
    if (
        int(state["global_step"]) != int(cursor["global_step"])
        or int(marker.get("global_step", -1)) != int(cursor["global_step"])
        or path_step != int(cursor["global_step"])
        or int(cursor["global_step"]) != expected_step
    ):
        raise RuntimeError("silence-slot checkpoint cursor/global step mismatch")
    rng = state["rng_states_by_rank"]
    if set(rng) != {str(index) for index in range(args.world_size)}:
        raise RuntimeError("silence-slot checkpoint per-rank RNG coverage mismatch")
    base._restore_rng_state(rng[str(rank)], device)
    return cursor, state.get("plan_hash")


def _formal_smoke_gate(args: argparse.Namespace, inventory: dict[str, Any]) -> dict[str, Any]:
    if args.smoke20_report is None or args.smoke_resume_report is None:
        raise ValueError("formal training requires this route's --smoke20-report and --smoke-resume-report")
    initial = json.loads(args.smoke20_report.read_text(encoding="utf-8"))
    resumed = json.loads(args.smoke_resume_report.read_text(encoding="utf-8"))
    expectations = (
        ("smoke20", initial, {"segment": 0, "segment_step": 0, "global_step": 0}, {"segment": 2, "segment_step": 0, "global_step": 20}, 2),
        ("resume2", resumed, {"segment": 2, "segment_step": 0, "global_step": 20}, {"segment": 3, "segment_step": 0, "global_step": 22}, 1),
    )
    for name, report, start, end, segments in expectations:
        if (
            report.get("status") != "PASS"
            or report.get("mode") != "smoke"
            or report.get("training_contract") != CONTRACT
            or report.get("hard_failures")
            or report.get("inventory") != inventory
            or report.get("start_cursor") != start
            or report.get("end_cursor") != end
            or report.get("prefix_contract") != FIXED_PREFIX_TOKEN_CONTRACT
            or report.get("silence_slot_contract") != SILENCE_SLOT_CONTRACT
            or len(report.get("segments", [])) != segments
        ):
            raise RuntimeError(f"formal gate rejects {name} report")
        audit = report.get("first_step_gradient_audit", {})
        required_audits = (
            "trace_matches_5_10_10_5",
            "fixed260_silence_slot_contract",
            "all_bridge_gradients_finite",
            "all_c2l_gradients_finite",
            "htsat_frozen_and_gradient_free",
            "fixed260_prefix_observed",
            "answer_labels_start_after_fixed260_prefix",
        )
        if not all(audit.get(key) is True for key in required_audits):
            raise RuntimeError(f"formal gate rejects {name} architecture/silence audit")
        slot_totals = report.get("slot_audit_totals", {})
        single_rows = int(slot_totals.get("global_single_audio_rows", -1))
        dual_rows = int(slot_totals.get("global_dual_audio_rows", -1))
        same_rows = int(slot_totals.get("global_same_real_dual_audio_rows", -1))
        distinct_rows = int(slot_totals.get("global_distinct_real_dual_audio_rows", -1))
        if single_rows + dual_rows != (20 if name == "smoke20" else 2) * 256:
            raise RuntimeError(f"formal gate rejects {name} slot cardinality: {slot_totals}")
        if same_rows + distinct_rows != dual_rows:
            raise RuntimeError(f"formal gate rejects {name} dual-slot classification: {slot_totals}")
        if name == "smoke20" and not all(
            int(slot_totals.get(key, 0)) > 0
            for key in (
                "global_single_audio_rows",
                "global_dual_audio_rows",
                "global_distinct_real_dual_audio_rows",
                "runtime_silence_encoder_calls_on_rank0",
                "runtime_silence_rows_on_rank0",
            )
        ):
            raise RuntimeError(f"formal gate rejects {name} incomplete slot coverage: {slot_totals}")
        if any(
            len(segment.get("release", [])) != 8
            or not all(item.get("passed") for item in segment["release"])
            or segment.get("cgroup_release", {}).get("anon_drop_bytes") is None
            or segment["cgroup_release"]["anon_drop_bytes"] < segment["cgroup_release"]["required_bytes"]
            or len(segment.get("steps", [])) != segment["segment"]["steps"]
            for segment in report["segments"]
        ):
            raise RuntimeError(f"formal gate rejects {name} release/step audit")
    checkpoint = Path(initial["checkpoints"][-1]).resolve()
    _validate_resume_artifact(checkpoint)
    if resumed.get("resume_checkpoint") != str(checkpoint) or resumed.get("resume_verified_two_steps") is not True:
        raise RuntimeError("silence-slot resume report does not prove step-20 continuation")
    if initial.get("seed") != args.seed or resumed.get("seed") != args.seed:
        raise RuntimeError("silence-slot smoke/formal seed mismatch")
    return {
        "smoke20_report": str(args.smoke20_report.resolve()),
        "smoke_resume_report": str(args.smoke_resume_report.resolve()),
        "checkpoint20": str(checkpoint),
    }


def _microbatch_slot_audit(batch: dict[str, Any], model: AudioMeshSilenceSlotModel) -> dict[str, Any]:
    silence = int(batch["silence_second_slot_mask"].sum().item())
    same = int(batch["same_real_audio_mask"].sum().item())
    distinct = int(batch["text_ids"].shape[0]) - silence - same
    actual = dict(model.last_audio_slot_audit)
    expected_encoder_batch = (1 if silence else 0) + distinct
    if (
        actual.get("silence_second_slot_rows") != silence
        or actual.get("same_real_audio_rows") != same
        or actual.get("distinct_real_audio_rows") != distinct
        or actual.get("runtime_silence_waveforms_created") != (1 if silence else 0)
        or actual.get("second_encoder_input_batch_size") != expected_encoder_batch
        or actual.get("fixed_prefix_tokens") != AUDIO_PREFIX_TOKENS
        or actual.get("all_rows_classified") is not True
    ):
        raise RuntimeError(f"runtime silence-slot audit mismatch: expected={silence,same,distinct,expected_encoder_batch} actual={actual}")
    return actual


def _audio_runtime_gradient_audit(model: AudioMeshSilenceSlotModel) -> dict[str, Any]:
    bridge_gradients = {
        name: parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for name, parameter in model.bridge.named_parameters()
    }
    c2l = getattr(model.htsat_wrapper, "c2l", None)
    if not isinstance(c2l, torch.nn.Module):
        raise RuntimeError("runtime audio audit found no c2l module")
    c2l_gradients = {
        name: parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for name, parameter in c2l.named_parameters()
    }
    htsat_frozen = all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in model.htsat_backbone.parameters()
    )
    prefix_lengths = model.last_prefix_lengths
    fixed_prefix = (
        model.last_prefix_length == AUDIO_PREFIX_TOKENS
        and prefix_lengths is not None
        and int(prefix_lengths.numel()) > 0
        and bool((prefix_lengths == AUDIO_PREFIX_TOKENS).all())
    )
    result = {
        "bridge_finite_gradients": bridge_gradients,
        "all_bridge_gradients_finite": bool(bridge_gradients) and all(bridge_gradients.values()),
        "c2l_finite_gradients": c2l_gradients,
        "all_c2l_gradients_finite": bool(c2l_gradients) and all(c2l_gradients.values()),
        "htsat_frozen_and_gradient_free": htsat_frozen,
        "fixed260_prefix_observed": fixed_prefix,
        "answer_labels_start_after_fixed260_prefix": (
            model.last_labels is not None
            and model.last_labels.shape[1] >= AUDIO_PREFIX_TOKENS
            and bool((model.last_labels[:, :AUDIO_PREFIX_TOKENS] == -100).all())
        ),
    }
    if not all(
        (
            result["all_bridge_gradients_finite"],
            result["all_c2l_gradients_finite"],
            result["htsat_frozen_and_gradient_free"],
            result["fixed260_prefix_observed"],
            result["answer_labels_start_after_fixed260_prefix"],
        )
    ):
        raise RuntimeError(f"fixed260 silence-slot audio gradient/label audit failed: {result}")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
    if not torch.cuda.is_available() or world != args.world_size or args.world_size != 8:
        raise RuntimeError("silence-slot partition training requires exactly 8 CUDA ranks")
    if args.micro_batch_size != 8 or args.gradient_accumulation_steps != 4:
        raise RuntimeError("silence-slot partition training requires microbatch 8 and GA 4")
    if (
        args.epochs <= 0 or args.dist_timeout_minutes < 30 or args.save_every <= 0
        or args.checkpoint_retention <= 0 or not 0 < args.release_min_fraction <= 1
        or args.release_timeout_seconds <= 0
    ):
        raise ValueError("invalid epochs, timeout, checkpoint, or release threshold")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group(
            "nccl", rank=rank, world_size=world,
            timeout=timedelta(minutes=args.dist_timeout_minutes),
        )
    base._seed(args.seed, rank)
    report: dict[str, Any] = {
        "status": "FAIL",
        "mode": args.mode,
        "training_contract": CONTRACT,
        "answer_termination": ANSWER_TERMINATION_CONTRACT,
        "prefix_contract": FIXED_PREFIX_TOKEN_CONTRACT,
        "silence_slot_contract": SILENCE_SLOT_CONTRACT,
        "rank": rank,
        "segments": [],
        "checkpoints": [],
        "hard_failures": [],
    }
    output_available = not args.output_dir.exists() or not any(args.output_dir.iterdir())
    try:
        if not output_available:
            raise FileExistsError(f"refusing nonempty output directory: {args.output_dir}")
        _validate_launch_contract(args)
        inventory = partition_common._inventory(args.partition_store_root)
        smoke_gate = _formal_smoke_gate(args, inventory) if args.mode == "formal" else None
        schedule = partition_common._schedule(args, inventory)
        total = sum(item["steps"] for item in schedule)
        if args.mode == "formal" and total != (968059 // 256) * args.epochs:
            raise RuntimeError("formal step budget differs from global epoch floor")
        args.warmup_steps = args.warmup_steps if args.warmup_steps is not None else math.ceil(total * 0.05)
        if args.warmup_steps != math.ceil(total * 0.05):
            raise ValueError("warmup must equal ceil(5% of total optimizer steps)")
        model, tokenizer = _load_model(args, device)
        model.train()
        if not model.trainable_parameter_audit()["training_mode_contract"]:
            raise RuntimeError("trainable parameter audit failed")
        model.mesh_model.model.routing_stats_mode = False
        model.mesh_model.model.gradient_audit_mode = True
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.max_lr, betas=(0.9, 0.95), weight_decay=0.1,
        )
        scheduler = base._make_scheduler(
            optimizer, max_lr=args.max_lr, min_lr=args.min_lr,
            warmup_steps=args.warmup_steps, total_steps=total,
        )
        cursor = {"segment": 0, "segment_step": 0, "global_step": 0}
        saved_plan_hash = None
        if args.resume_from:
            cursor, saved_plan_hash = _resume(
                args.resume_from, args, inventory, schedule, optimizer, scheduler,
                rank, device, model._audio_provenance,
            )
            if args.mode == "smoke" and cursor != {"segment": 2, "segment_step": 0, "global_step": 20}:
                raise RuntimeError("smoke resume requires this route's released step-20 checkpoint")
        elif args.mode == "smoke":
            assert total == 22
        ddp = DDP(
            model, device_ids=[local_rank], broadcast_buffers=False,
            find_unused_parameters=False,
        )
        report.update({
            "inventory": inventory,
            "schedule": schedule,
            "total_steps": total,
            "warmup_steps": args.warmup_steps,
            "start_cursor": dict(cursor),
            "seed": args.seed,
            "resume_checkpoint": str(args.resume_from.resolve()) if args.resume_from else None,
            "smoke_gate": smoke_gate,
        })
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
            before = partition_common._memory()
            started = time.perf_counter()
            dataset = ReasonAQADataset(path / "rows.jsonl", tokenizer, unique_waveform_store_dir=path)
            if len(dataset) != part["rows"]:
                raise RuntimeError("partition QA cardinality changed")
            cache = base._RankLocalStoreWaveforms(dataset)
            for audio_id in range(part["audio"]):
                cache._get_audio_id(audio_id)
            if cache.current_bytes != part["bytes"] or cache.misses != part["audio"] or cache.evictions:
                raise RuntimeError("partition preload cardinality/bytes mismatch")
            chunks, plan_audit = partition_common._plan(
                dataset, seed=args.seed, epoch=segment["epoch"], pid=pid,
                steps=segment["steps"], global_batch=global_batch,
            )
            if segment_index == cursor["segment"] and cursor["segment_step"] and saved_plan_hash != plan_audit["plan_sha256"]:
                raise RuntimeError("resume optimizer-window plan hash mismatch")
            dist.barrier()
            after_load = partition_common._memory()
            segment_report: dict[str, Any] = {
                "segment": segment,
                "plan": plan_audit,
                "preload_seconds": time.perf_counter() - started,
                "memory_before": before,
                "memory_after_load": after_load,
                "steps": [],
            }
            if rank == 0:
                print(
                    f"[silence-slot] loaded p{pid} epoch={segment['epoch']} "
                    f"steps={segment['steps']} bytes/rank={part['bytes']} "
                    f"seconds={segment_report['preload_seconds']:.1f}", flush=True,
                )
            initial_stats = cache.stats()
            start_at = cursor["segment_step"] if segment_index == cursor["segment"] else 0
            for local_step in range(start_at, segment["steps"]):
                if cursor["global_step"] >= stop_step:
                    break
                step_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                optimizer.zero_grad(set_to_none=True)
                step_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
                chunk = chunks[local_step]
                rank_rows = chunk[rank * 32:(rank + 1) * 32]
                if len(rank_rows) != 32:
                    raise AssertionError("rank optimizer window is not 32 rows")
                global_structures = [dataset.audio_structure(index) for index in chunk]
                step_counts = {
                    "single_audio_rows": sum(1 for single, _ in global_structures if single),
                    "dual_audio_rows": sum(1 for single, _ in global_structures if not single),
                    "same_real_dual_audio_rows": sum(1 for single, same in global_structures if not single and same),
                }
                micro_audits: list[dict[str, Any]] = []
                for micro in range(4):
                    indices = rank_rows[micro * 8:(micro + 1) * 8]
                    batch = collate_reasonaqa(
                        [cache.materialize_from_cached_metadata(index) for index in indices], tokenizer
                    )
                    device_keys = {"audio1", "audio2", "text_ids"}
                    batch = {
                        key: (value.to(device) if torch.is_tensor(value) and key in device_keys else value)
                        for key, value in batch.items()
                    }
                    with ddp.no_sync() if micro != 3 else contextlib.nullcontext():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            output = ddp(**{
                                key: value for key, value in batch.items()
                                if key not in {"row_indices", "audio2_reused"}
                            })
                        micro_audits.append(_microbatch_slot_audit(batch, ddp.module))
                        if output.loss is None or not bool(torch.isfinite(output.loss)):
                            raise RuntimeError("nonfinite silence-slot training loss")
                        step_loss_sum.add_(output.loss.detach().float())
                        (output.loss / args.gradient_accumulation_steps).backward()
                    del batch, output
                owner = ddp.module
                if first_audit is None:
                    first_audit = base._mesh_runtime_gradient_audit(owner, require_router_stats=False)
                    first_audit.update(_audio_runtime_gradient_audit(owner))
                    first_audit.update({
                        "fixed260_silence_slot_contract": True,
                        "prefix_tokens_all_rows": AUDIO_PREFIX_TOKENS,
                        "runtime_silence_not_in_partition_store": True,
                        "microbatch_slot_audits": micro_audits,
                    })
                    owner.mesh_model.model.gradient_audit_mode = False
                torch.nn.utils.clip_grad_norm_(ddp.parameters(), 0.5, error_if_nonfinite=True)
                lr_used = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
                cursor = {
                    "segment": segment_index,
                    "segment_step": local_step + 1,
                    "global_step": cursor["global_step"] + 1,
                }
                torch.cuda.synchronize(device)
                step_loss = float((step_loss_sum / args.gradient_accumulation_steps).item())
                segment_report["steps"].append({
                    "global_step": cursor["global_step"],
                    "partition_step": local_step + 1,
                    **step_counts,
                    "runtime_silence_encoder_calls_on_rank": sum(
                        audit["runtime_silence_waveforms_created"] for audit in micro_audits
                    ),
                    "runtime_silence_rows_on_rank": sum(
                        audit["silence_second_slot_rows"] for audit in micro_audits
                    ),
                    "maximum_second_encoder_input_batch_on_rank": max(
                        audit["second_encoder_input_batch_size"] for audit in micro_audits
                    ),
                    "loss": step_loss,
                    "lr": lr_used,
                    "seconds": time.perf_counter() - step_started,
                    "max_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                })
                at_boundary = local_step + 1 == segment["steps"]
                if rank == 0 and (cursor["global_step"] % 10 == 0 or at_boundary):
                    latest = segment_report["steps"][-1]
                    print(
                        f"[silence-slot] step={cursor['global_step']}/{total} p{pid} "
                        f"{local_step+1}/{segment['steps']} single={step_counts['single_audio_rows']} "
                        f"dual={step_counts['dual_audio_rows']} loss={step_loss:.6f} "
                        f"lr={lr_used:.8g} {latest['seconds']:.3f}s", flush=True,
                    )
                if args.mode == "formal" and cursor["global_step"] % args.save_every == 0 and not at_boundary:
                    checkpoint = args.output_dir / f"checkpoint-{cursor['global_step']:06d}"
                    _checkpoint(
                        checkpoint, owner, tokenizer, optimizer, scheduler, args, inventory,
                        schedule, cursor, rank, world, device, plan_audit["plan_sha256"],
                    )
                    if rank == 0:
                        report["checkpoints"].append(str(checkpoint))
                        report["retained_checkpoints"] = base._prune_checkpoints(
                            args.output_dir, args.checkpoint_retention
                        )
                    dist.barrier()
            after_stats = cache.stats()
            if (
                after_stats["waveform_misses"] != initial_stats["waveform_misses"]
                or after_stats["cloned_bytes"] != initial_stats["cloned_bytes"]
                or after_stats["resident_unique_audio"] != part["audio"]
            ):
                raise RuntimeError("training accessed store or changed complete rank-RAM residency")
            torch.cuda.synchronize(device)
            del chunks, cache.items
            dataset.unique_waveform_store.close()
            del cache, dataset
            gc.collect()
            partition_common._trim()
            dist.barrier()
            threshold = int(part["bytes"] * args.release_min_fraction)
            cgroup_required = int(part["bytes"] * world * args.release_min_fraction)
            deadline = time.monotonic() + args.release_timeout_seconds
            polls = 0
            while True:
                after_release = partition_common._memory()
                polls += 1
                current_anon = partition_common._anon(after_release)
                loaded_anon = partition_common._anon(after_load)
                rss_ok = after_load["rss_bytes"] - after_release["rss_bytes"] >= threshold
                cgroup_ok = (
                    current_anon is not None and loaded_anon is not None
                    and loaded_anon - current_anon >= cgroup_required
                )
                if (rss_ok and cgroup_ok) or time.monotonic() >= deadline:
                    break
                time.sleep(2.0)
            released = after_load["rss_bytes"] - after_release["rss_bytes"]
            local_release = {
                "rank": rank,
                "rss_drop_bytes": released,
                "required_drop_bytes": threshold,
                "before": before,
                "after_load": after_load,
                "after_release": after_release,
                "polls": polls,
                "passed": released >= threshold,
            }
            releases = partition_common._gather(local_release, world)
            segment_report["release"] = releases
            loaded_cgroup = partition_common._anon(after_load)
            released_cgroup = partition_common._anon(after_release)
            cgroup_drop = (
                None if loaded_cgroup is None or released_cgroup is None
                else loaded_cgroup - released_cgroup
            )
            segment_report["cgroup_release"] = {
                "anon_drop_bytes": cgroup_drop,
                "required_bytes": cgroup_required,
                "available": cgroup_drop is not None,
                "timeout_seconds": args.release_timeout_seconds,
            }
            if cgroup_drop is None:
                raise RuntimeError(f"partition p{pid} cannot audit cgroup anon release")
            if not all(item["passed"] for item in releases):
                raise RuntimeError(f"partition p{pid} per-rank RAM release failed")
            if cgroup_drop < cgroup_required:
                raise RuntimeError(f"partition p{pid} cgroup anon failed to fall")
            report["segments"].append(segment_report)
            cursor = {
                "segment": segment_index + 1,
                "segment_step": 0,
                "global_step": cursor["global_step"],
            }
            at_end = cursor["global_step"] == total
            save_boundary = (
                (args.mode == "smoke" and cursor["global_step"] in (20, 22))
                or (args.mode == "formal" and (cursor["global_step"] % args.save_every == 0 or at_end))
            )
            if save_boundary:
                checkpoint = args.output_dir / f"checkpoint-{cursor['global_step']:06d}"
                _checkpoint(
                    checkpoint, owner, tokenizer, optimizer, scheduler, args, inventory,
                    schedule, cursor, rank, world, device, None,
                )
                if rank == 0:
                    report["checkpoints"].append(str(checkpoint))
                    report["retained_checkpoints"] = base._prune_checkpoints(
                        args.output_dir, args.checkpoint_retention
                    )
                dist.barrier()
            if rank == 0:
                print(f"[silence-slot] released p{pid}", flush=True)
        if cursor["global_step"] != stop_step:
            raise RuntimeError(f"run ended at step {cursor['global_step']} rather than {stop_step}")
        all_step_reports = [
            step for segment in report["segments"] for step in segment["steps"]
        ]
        global_dual = sum(int(step["dual_audio_rows"]) for step in all_step_reports)
        global_same = sum(
            int(step["same_real_dual_audio_rows"]) for step in all_step_reports
        )
        slot_audit_totals = {
            "global_single_audio_rows": sum(
                int(step["single_audio_rows"]) for step in all_step_reports
            ),
            "global_dual_audio_rows": global_dual,
            "global_same_real_dual_audio_rows": global_same,
            "global_distinct_real_dual_audio_rows": global_dual - global_same,
            "runtime_silence_encoder_calls_on_rank0": sum(
                int(step["runtime_silence_encoder_calls_on_rank"])
                for step in all_step_reports
            ),
            "runtime_silence_rows_on_rank0": sum(
                int(step["runtime_silence_rows_on_rank"])
                for step in all_step_reports
            ),
        }
        report.update({
            "status": "PASS",
            "end_cursor": cursor,
            "first_step_gradient_audit": first_audit,
            "slot_audit_totals": slot_audit_totals,
            "resume_verified_two_steps": (
                args.mode == "smoke" and args.resume_from is not None
                and cursor["global_step"] == 22
            ),
        })
        return report
    except Exception as exc:
        report["hard_failures"].append({
            "error": repr(exc), "traceback": traceback.format_exc()
        })
        raise
    finally:
        if rank == 0 and output_available:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "partition_training_report.json").write_text(
                json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
            )
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run(parse_args())
