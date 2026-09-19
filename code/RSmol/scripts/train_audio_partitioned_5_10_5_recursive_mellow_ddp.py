#!/usr/bin/env python3
"""Six-partition training for the fixed two-pass 5-10-5 audio baseline.

This route mirrors the current Audio MeSH partition lifecycle and optimizer
contract.  Its only model difference is the text backbone: twenty physical
decoder modules execute the exact logical 5-10-10-5 schedule, without MeSH
memory or routers.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
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

import train_audio_partitioned_smollm2_135m_mellow_ddp as partition_common
import train_audio_smollm2_135m_mellow_ddp as base
from audio_5_10_5_recursive_mellow.data import ReasonAQADataset, collate_reasonaqa
from audio_5_10_5_recursive_mellow.model import (
    AUDIO_DUAL_PREFIX_TOKENS,
    AUDIO_SINGLE_PREFIX_TOKENS,
    MAPPER_CONTRACT,
    RECURSIVE_AUDIO_CONTRACT,
    AudioRecursive5_10_5Config,
    AudioRecursive5_10_5Model,
    _load_mellow_wrapper,
    validate_recursive_5_10_5,
)
from recursive_model_5_10_5 import (
    LOGICAL_LAYER_COUNT,
    LOGICAL_TO_PHYSICAL,
    MIDDLE_LAYER_COUNT,
    PHYSICAL_LAYER_COUNT,
    PREFIX_LAYER_COUNT,
    RECURSIVE_LOOPS,
    SOURCE_LAYER_INDICES_0BASED,
    SUFFIX_LAYER_COUNT,
    RecursiveLlamaForCausalLM,
    register_auto_class,
)


CONTRACT = "recursive_5_10_5_component_partitions6_rank_ram_compact_audio_answer_eos_v2"
CONFIG_FILENAME = "audio_recursive_5_10_5_partition_config.json"
PARTITION_STORE_ROOT = partition_common.PARTITION_STORE_ROOT
SMOKE_SEGMENTS = ((2, 10), (0, 10), (1, 2))
DEFAULT_RECURSIVE_CHECKPOINT = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/"
    "formal-epoch2-continue-20260902_184936/checkpoint-step-009244"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--partition-store-root", type=Path, default=Path(PARTITION_STORE_ROOT))
    parser.add_argument("--model-path", "--recursive-checkpoint", dest="model_path", type=Path, default=Path(DEFAULT_RECURSIVE_CHECKPOINT))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--expected-resume-step", type=int)
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
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--checkpoint-retention", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dist-timeout-minutes", type=int, default=30)
    parser.add_argument("--release-min-fraction", type=float, default=0.70)
    parser.add_argument("--release-timeout-seconds", type=int, default=120)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _audio_state_hashes(model: AudioRecursive5_10_5Model) -> dict[str, str]:
    c2l = getattr(model.htsat_wrapper, "c2l", None)
    if not isinstance(c2l, torch.nn.Module):
        raise RuntimeError("fixed 5-10-5 audio model has no c2l module")
    return {
        "bridge_sha256": _module_state_sha256(model.bridge),
        "c2l_sha256": _module_state_sha256(c2l),
    }


def _validate_sources(args: argparse.Namespace) -> None:
    expected = {
        "partition_store_root": Path(PARTITION_STORE_ROOT).resolve(),
        "model_path": Path(DEFAULT_RECURSIVE_CHECKPOINT).resolve(),
        "htsat_checkpoint": Path(base.DEFAULT_HTSAT).resolve(),
        "mellow_root": Path(base.DEFAULT_MELLOW).resolve(),
    }
    actual = {
        "partition_store_root": args.partition_store_root.resolve(),
        "model_path": args.model_path.resolve(),
        "htsat_checkpoint": args.htsat_checkpoint.resolve(),
        "mellow_root": args.mellow_root.resolve(),
    }
    if actual != expected:
        raise RuntimeError(f"fixed 5-10-5 partition route requires canonical sources: actual={actual} expected={expected}")
    if args.tokenizer_path is not None:
        raise ValueError("fixed 5-10-5 partition route uses the canonical checkpoint tokenizer and rejects --tokenizer-path")
    missing_files = [str(path) for path in (args.htsat_checkpoint,) if not path.is_file()]
    missing_dirs = [str(path) for path in (args.partition_store_root, args.model_path, args.mellow_root) if not path.is_dir()]
    if missing_files or missing_dirs:
        raise FileNotFoundError(f"canonical fixed-recursive sources are unavailable: files={missing_files} dirs={missing_dirs}")


def _schedule(args: argparse.Namespace, inventory: dict[str, Any]) -> list[dict[str, int]]:
    if args.mode == "smoke":
        return [
            {"epoch": 0, "position": position, "partition_id": partition_id, "steps": steps}
            for position, (partition_id, steps) in enumerate(SMOKE_SEGMENTS)
        ]
    counts = [entry["rows"] for entry in inventory["partitions"]]
    global_batch = args.world_size * args.micro_batch_size * args.gradient_accumulation_steps
    quotas = partition_common._quotas(counts, args.epochs, global_batch)
    return [
        {"epoch": epoch, "position": position, "partition_id": partition_id, "steps": quotas[epoch][partition_id]}
        for epoch in range(args.epochs)
        for position, partition_id in enumerate(partition_common._order(args.seed, epoch))
    ]


def _recursive_metadata() -> dict[str, Any]:
    return {
        "logical_layer_count": LOGICAL_LAYER_COUNT,
        "physical_layer_count": PHYSICAL_LAYER_COUNT,
        "recursive_loops": RECURSIVE_LOOPS,
        "recursive_loops_scope": "middle_only",
        "prefix_layer_count": PREFIX_LAYER_COUNT,
        "middle_layer_count": MIDDLE_LAYER_COUNT,
        "suffix_layer_count": SUFFIX_LAYER_COUNT,
        "logical_to_physical": list(LOGICAL_TO_PHYSICAL),
        "source_layer_indices_0based": list(SOURCE_LAYER_INDICES_0BASED),
        "has_mesh_router_or_memory": False,
    }


def _load_model(args: argparse.Namespace, device: torch.device) -> tuple[AudioRecursive5_10_5Model, Any]:
    from transformers import AutoTokenizer

    register_auto_class()
    model_path = args.resume_from / "text_model" if args.resume_from else args.model_path
    tokenizer_path = args.resume_from / "tokenizer" if args.resume_from else args.model_path
    text_model = RecursiveLlamaForCausalLM.from_pretrained(model_path, local_files_only=True)
    validate_recursive_5_10_5(text_model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("fixed 5-10-5 tokenizer must define eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(text_model.config, "pad_token_id", None) is None:
        text_model.config.pad_token_id = int(tokenizer.pad_token_id)
    wrapper, htsat, provenance = _load_mellow_wrapper(args.mellow_root, args.htsat_checkpoint, device)
    model = AudioRecursive5_10_5Model(
        text_model.to(device),
        tokenizer,
        wrapper,
        htsat,
        AudioRecursive5_10_5Config(compact_single_audio_prefix=True),
    )
    if args.resume_from:
        audio_state = torch.load(args.resume_from / "audio_bridge.pt", map_location=device, weights_only=False)
        if not isinstance(audio_state.get("bridge"), dict) or not audio_state["bridge"]:
            raise RuntimeError("resume checkpoint has no bridge state")
        if not isinstance(audio_state.get("c2l"), dict) or not audio_state["c2l"]:
            raise RuntimeError("resume checkpoint has no c2l state")
        model.bridge.load_state_dict(audio_state["bridge"], strict=True)
        c2l = getattr(model.htsat_wrapper, "c2l", None)
        if c2l is None:
            raise RuntimeError("resume checkpoint requires wrapper.c2l")
        c2l.load_state_dict(audio_state["c2l"], strict=True)
    model._audio_provenance = provenance
    model._text_model_source = str(args.model_path.resolve())
    return model.to(device), tokenizer


def _sequence_and_gradient_audit(model: AudioRecursive5_10_5Model, batch: dict[str, Any]) -> dict[str, Any]:
    gradient = model.runtime_gradient_audit()
    trainable = model.trainable_parameter_audit()
    sequence = partition_common._sequence_audit(model, batch)
    recursive = trainable.get("recursive_text_contract", {})
    result = {
        **gradient,
        **sequence,
        "exact_fixed_recursive_5_10x2_5": (
            recursive.get("logical_layer_count") == 30
            and recursive.get("physical_decoder_layer_count") == 20
            and recursive.get("logical_to_physical") == list(LOGICAL_TO_PHYSICAL)
            and trainable.get("distinct_physical_decoder_layers") is True
            and trainable.get("has_router_parameters") is False
            and gradient.get("forward_trace_matches_exact_5_10x2_5") is True
            and gradient.get("both_middle_loops_have_finite_gradients") is True
        ),
        "all_bridge_gradients_finite": bool(gradient.get("bridge_gradients")) and all(gradient["bridge_gradients"].values()),
        "all_c2l_gradients_finite": bool(gradient.get("c2l_gradients")) and all(gradient["c2l_gradients"].values()),
        "training_mode_contract": trainable.get("training_mode_contract") is True,
    }
    required = (
        result["exact_fixed_recursive_5_10x2_5"],
        result.get("all_decoder_layers_have_finite_gradient") is True,
        result.get("embedding_has_finite_gradient") is True,
        result.get("lm_head_has_finite_gradient") is True,
        result["all_bridge_gradients_finite"],
        result["all_c2l_gradients_finite"],
        result.get("htsat_frozen_and_gradient_free") is True,
        result["training_mode_contract"],
        result.get("compact_prefix_contract") is True,
        result.get("answer_eos_contract") is True,
    )
    if not all(required):
        raise RuntimeError(f"fixed 5-10-5 first-step audit failed: {result}")
    return result


def _formal_smoke_gate(args: argparse.Namespace, inventory: dict[str, Any]) -> dict[str, Any]:
    if args.smoke20_report is None or args.smoke_resume_report is None:
        raise ValueError("formal training requires --smoke20-report and --smoke-resume-report")
    initial = json.loads(args.smoke20_report.read_text(encoding="utf-8"))
    resumed = json.loads(args.smoke_resume_report.read_text(encoding="utf-8"))
    expected_schedule = [
        {"epoch": 0, "position": position, "partition_id": partition_id, "steps": steps}
        for position, (partition_id, steps) in enumerate(SMOKE_SEGMENTS)
    ]
    for name, report, start, end, expected_segments in (
        ("smoke20", initial, {"segment": 0, "segment_step": 0, "global_step": 0}, {"segment": 2, "segment_step": 0, "global_step": 20}, expected_schedule[:2]),
        ("resume2", resumed, {"segment": 2, "segment_step": 0, "global_step": 20}, {"segment": 3, "segment_step": 0, "global_step": 22}, expected_schedule[2:]),
    ):
        if (
            report.get("status") != "PASS"
            or report.get("mode") != "smoke"
            or report.get("training_contract") != CONTRACT
            or report.get("hard_failures")
            or report.get("inventory") != inventory
            or report.get("schedule") != expected_schedule
            or report.get("start_cursor") != start
            or report.get("end_cursor") != end
            or report.get("seed") != args.seed
        ):
            raise RuntimeError(f"formal gate rejects {name} report")
        segments = report.get("segments", [])
        if [segment.get("segment") for segment in segments] != expected_segments:
            raise RuntimeError(f"formal gate rejects {name} partition order/step contract")
        if any(
            len(segment.get("release", [])) != 8
            or not all(item.get("passed") for item in segment["release"])
            or segment.get("cgroup_release", {}).get("anon_drop_bytes") is None
            or segment["cgroup_release"]["anon_drop_bytes"] < segment["cgroup_release"]["required_bytes"]
            or len(segment.get("steps", [])) != segment["segment"]["steps"]
            for segment in segments
        ):
            raise RuntimeError(f"formal gate rejects {name} release audit")
        audit = report.get("first_step_gradient_audit", {})
        required_audit = (
            audit.get("exact_fixed_recursive_5_10x2_5") is True,
            audit.get("forward_trace_matches_exact_5_10x2_5") is True,
            audit.get("both_middle_loops_have_finite_gradients") is True,
            audit.get("all_decoder_layers_have_finite_gradient") is True,
            audit.get("embedding_has_finite_gradient") is True,
            audit.get("lm_head_has_finite_gradient") is True,
            audit.get("all_bridge_gradients_finite") is True,
            audit.get("all_c2l_gradients_finite") is True,
            audit.get("htsat_frozen_and_gradient_free") is True,
            audit.get("no_mesh_router_or_memory_parameters") is True,
            audit.get("compact_prefix_contract") is True,
            audit.get("answer_eos_contract") is True,
            audit.get("training_mode_contract") is True,
        )
        if not all(required_audit):
            raise RuntimeError(f"formal gate rejects {name} recursive gradient/sequence audit")
    initial_checkpoints = initial.get("checkpoints", [])
    resumed_checkpoints = resumed.get("checkpoints", [])
    if len(initial_checkpoints) != 1 or len(resumed_checkpoints) != 1:
        raise RuntimeError("smoke reports must expose exactly checkpoint-000020 and checkpoint-000022")
    checkpoint20 = Path(initial_checkpoints[0]).resolve()
    checkpoint22 = Path(resumed_checkpoints[0]).resolve()
    if checkpoint20.name != "checkpoint-000020" or checkpoint22.name != "checkpoint-000022":
        raise RuntimeError("smoke checkpoint names do not match the 20+2 contract")
    artifact_configs: dict[int, dict[str, Any]] = {}
    for checkpoint_path, expected_step in ((checkpoint20, 20), (checkpoint22, 22)):
        marker = json.loads((checkpoint_path / "checkpoint_complete.json").read_text(encoding="utf-8"))
        config = json.loads((checkpoint_path / CONFIG_FILENAME).read_text(encoding="utf-8"))
        artifact_configs[expected_step] = config
        if (
            marker.get("status") != "complete"
            or marker.get("global_step") != expected_step
            or marker.get("contract") != CONTRACT
            or config.get("contract") != CONTRACT
            or config.get("mode") != "smoke"
            or config.get("architecture_contract") != RECURSIVE_AUDIO_CONTRACT
            or config.get("mapper_contract") != MAPPER_CONTRACT
            or config.get("recursive_text_contract") != _recursive_metadata()
            or config.get("compact_single_audio_prefix") is not True
            or config.get("prefix_tokens") != {"single": 130, "dual": 260}
            or config.get("answer_termination") != {"token": "<|endoftext|>", "included_in_max_answer_tokens": True, "supervised": True}
            or config.get("inventory") != inventory
            or config.get("schedule") != expected_schedule
            or config.get("epochs") != 10
            or config.get("world_size") != 8
            or config.get("micro_batch_size") != 8
            or config.get("gradient_accumulation_steps") != 4
            or config.get("effective_global_batch_size") != 256
            or config.get("seed") != args.seed
            or config.get("total_steps") != 22
            or config.get("warmup_steps") != 2
        ):
            raise RuntimeError(f"smoke checkpoint artifact failed validation: {checkpoint_path}")
    checkpoint = str(checkpoint20)
    if resumed.get("resume_checkpoint") != checkpoint or resumed.get("resume_verified_two_steps") is not True:
        raise RuntimeError("resume report does not prove continuation from checkpoint-000020")
    change = resumed.get("resume_parameter_change_audit", {})
    if change.get("all_groups_changed") is not True or not all(
        isinstance(change.get(group), dict)
        and change[group].get("changed") is True
        and change[group].get("finite") is True
        and change[group].get("exact_equal") is False
        and float(change[group].get("max_abs_delta", 0.0)) > 0.0
        for group in ("text", "bridge", "c2l")
    ):
        raise RuntimeError("resume report does not prove text/bridge/c2l updates")
    if (
        artifact_configs[20].get("resume_from") is not None
        or artifact_configs[22].get("resume_from") != checkpoint
        or artifact_configs[22].get("resume_parameter_change_audit") != change
    ):
        raise RuntimeError("smoke checkpoint configs do not prove exact resume lineage/update audit")
    return {
        "smoke20_report": str(args.smoke20_report.resolve()),
        "smoke_resume_report": str(args.smoke_resume_report.resolve()),
        "checkpoint20": checkpoint,
    }


def _checkpoint(
    path: Path,
    model: AudioRecursive5_10_5Model,
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
    parameter_change_audit: dict[str, Any] | None,
) -> None:
    rng_states = partition_common._gather(base._rng_state(device), world)
    if rank != 0:
        return
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent))
    published = False
    try:
        model.text_model.save_pretrained(temporary / "text_model", safe_serialization=False)
        tokenizer.save_pretrained(temporary / "tokenizer")
        torch.save(base._trainable_state(model), temporary / "audio_bridge.pt")
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scheduler_name": "cosine_lambda",
            "optimizer_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
            "training_contract": CONTRACT,
            "global_step": cursor["global_step"],
            "cursor": cursor,
            "plan_hash": plan_hash,
            "rng_states_by_rank": {str(index): state for index, state in enumerate(rng_states)},
        }, temporary / "training_state.pt")
        source_config = args.model_path.resolve() / "config.json"
        config = {
            "contract": CONTRACT,
            "architecture_contract": RECURSIVE_AUDIO_CONTRACT,
            "mapper_contract": MAPPER_CONTRACT,
            "recursive_text_contract": _recursive_metadata(),
            "compact_single_audio_prefix": True,
            "answer_termination": {"token": "<|endoftext|>", "included_in_max_answer_tokens": True, "supervised": True},
            "prefix_tokens": {"single": AUDIO_SINGLE_PREFIX_TOKENS, "dual": AUDIO_DUAL_PREFIX_TOKENS},
            "text_model_runtime_contract": model.text_contract,
            "text_model_source_path": str(args.model_path.resolve()),
            "text_model_source_config_sha256": _sha256(source_config),
            "audio_initialization_hashes": model._audio_initialization_hashes,
            "checkpoint_audio_state_hashes": _audio_state_hashes(model),
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
            "optimizer": "AdamW",
            "optimizer_betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "gradient_clip_norm": 0.5,
            "scheduler": "cosine_lambda",
            "warmup_steps": args.warmup_steps,
            "total_steps": sum(item["steps"] for item in schedule),
            "periodic_validation": False,
            "frozen_audio_encoder": True,
            "save_every": args.save_every,
            "checkpoint_retention": args.checkpoint_retention,
            "dist_timeout_minutes": args.dist_timeout_minutes,
            "release_min_fraction": args.release_min_fraction,
            "release_timeout_seconds": args.release_timeout_seconds,
            "htsat_checkpoint": str(args.htsat_checkpoint.resolve()),
            "mellow_root": str(args.mellow_root.resolve()),
            "mellow_provenance": model._audio_provenance,
            "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
            "resume_parameter_change_audit": parameter_change_audit,
        }
        (temporary / CONFIG_FILENAME).write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
        required = [
            "text_model/config.json",
            "tokenizer/tokenizer_config.json",
            "audio_bridge.pt",
            "training_state.pt",
            CONFIG_FILENAME,
            "checkpoint_complete.json",
        ]
        marker = {"status": "complete", "global_step": cursor["global_step"], "contract": CONTRACT, "required": required}
        (temporary / "checkpoint_complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        if any(not (temporary / name).is_file() for name in required):
            raise RuntimeError("fixed-recursive partition checkpoint is incomplete")
        if not base._text_model_weight_files(temporary / "text_model"):
            raise RuntimeError("fixed-recursive partition checkpoint has no text weights")
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
    model: AudioRecursive5_10_5Model,
) -> tuple[dict[str, int], str | None]:
    config = json.loads((path / CONFIG_FILENAME).read_text(encoding="utf-8"))
    marker = json.loads((path / "checkpoint_complete.json").read_text(encoding="utf-8"))
    required = [
        "text_model/config.json",
        "tokenizer/tokenizer_config.json",
        "audio_bridge.pt",
        "training_state.pt",
        CONFIG_FILENAME,
        "checkpoint_complete.json",
    ]
    if config.get("contract") != CONTRACT or marker.get("status") != "complete" or marker.get("contract") != CONTRACT:
        raise RuntimeError("not a complete compatible fixed-recursive partition checkpoint")
    if marker.get("required") != required or any(not (path / name).is_file() for name in required):
        raise RuntimeError("fixed-recursive checkpoint required-file contract mismatch")
    if not base._text_model_weight_files(path / "text_model"):
        raise RuntimeError("fixed-recursive checkpoint has no text weights")
    if (
        config.get("architecture_contract") != RECURSIVE_AUDIO_CONTRACT
        or config.get("mapper_contract") != MAPPER_CONTRACT
        or config.get("recursive_text_contract") != _recursive_metadata()
        or config.get("compact_single_audio_prefix") is not True
        or config.get("prefix_tokens") != {"single": 130, "dual": 260}
        or config.get("answer_termination") != {"token": "<|endoftext|>", "included_in_max_answer_tokens": True, "supervised": True}
    ):
        raise RuntimeError("resume recursive architecture/prefix/EOS contract mismatch")
    runtime_contract = config.get("text_model_runtime_contract", {})
    if (
        runtime_contract.get("logical_layer_count") != 30
        or runtime_contract.get("physical_decoder_layer_count") != 20
        or runtime_contract.get("logical_to_physical") != list(LOGICAL_TO_PHYSICAL)
    ):
        raise RuntimeError("resume checkpoint is not the exact fixed 5-10-5 model")
    if config.get("mellow_provenance", {}).get("mellow_htsat_sha256") != provenance.get("mellow_htsat_sha256"):
        raise RuntimeError("resume Mellow implementation SHA256 mismatch")
    source_config = args.model_path.resolve() / "config.json"
    if Path(config["text_model_source_path"]).resolve() != args.model_path.resolve():
        raise RuntimeError("resume original recursive source path mismatch")
    if config.get("text_model_source_config_sha256") != _sha256(source_config):
        raise RuntimeError("resume original recursive source config SHA256 mismatch")
    expected_values = {
        "mode": args.mode,
        "inventory": inventory,
        "schedule": schedule,
        "epochs": args.epochs,
        "world_size": args.world_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_global_batch_size": 256,
        "seed": args.seed,
        "max_lr": args.max_lr,
        "min_lr": args.min_lr,
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "gradient_clip_norm": 0.5,
        "scheduler": "cosine_lambda",
        "warmup_steps": args.warmup_steps,
        "total_steps": sum(item["steps"] for item in schedule),
        "periodic_validation": False,
        "frozen_audio_encoder": True,
        "save_every": args.save_every,
        "checkpoint_retention": args.checkpoint_retention,
        "dist_timeout_minutes": args.dist_timeout_minutes,
        "release_min_fraction": args.release_min_fraction,
        "release_timeout_seconds": args.release_timeout_seconds,
        "htsat_checkpoint": str(args.htsat_checkpoint.resolve()),
        "mellow_root": str(args.mellow_root.resolve()),
    }
    for key, expected in expected_values.items():
        if config.get(key) != expected:
            raise RuntimeError(f"resume contract differs in {key}")
    if config.get("checkpoint_audio_state_hashes") != _audio_state_hashes(model):
        raise RuntimeError("loaded bridge/c2l state differs from checkpoint hashes")
    initialization_hashes = config.get("audio_initialization_hashes")
    if not isinstance(initialization_hashes, dict) or set(initialization_hashes) != {"bridge_sha256", "c2l_sha256"}:
        raise RuntimeError("resume checkpoint has invalid audio initialization hashes")
    model._audio_initialization_hashes = initialization_hashes
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    if state.get("training_contract") != CONTRACT or state.get("scheduler_name") != "cosine_lambda":
        raise RuntimeError("resume training-state contract/scheduler mismatch")
    base._validate_optimizer_coverage(state, model)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    cursor = state["cursor"]
    if marker["global_step"] != cursor["global_step"] or cursor["global_step"] != sum(item["steps"] for item in schedule[:cursor["segment"]]) + cursor["segment_step"]:
        raise RuntimeError("checkpoint cursor/global step mismatch")
    rng_states = state["rng_states_by_rank"]
    if set(rng_states) != {str(index) for index in range(args.world_size)}:
        raise RuntimeError("checkpoint per-rank RNG coverage mismatch")
    base._restore_rng_state(rng_states[str(rank)], device)
    return cursor, state.get("plan_hash")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
    if args.resume_from is not None:
        resume_path = args.resume_from.resolve()
        output_path = args.output_dir.resolve()
        if output_path in {resume_path, resume_path.parent} or resume_path in output_path.parents:
            raise ValueError("resume output-dir must be separate from the source checkpoint and its parent")
    if not torch.cuda.is_available() or world != 8 or args.world_size != 8 or args.micro_batch_size != 8 or args.gradient_accumulation_steps != 4:
        raise RuntimeError("fixed-recursive partition training requires 8 GPUs, microbatch 8, GA 4")
    if (
        args.epochs != 10
        or args.max_lr != 1e-3
        or args.min_lr != 0.0
        or args.save_every != 500
        or args.checkpoint_retention != 4
        or args.seed != 0
    ):
        raise ValueError("fixed-recursive partition training requires epochs=10, max_lr=1e-3, min_lr=0, save_every=500, retention=4, seed=0")
    if args.dist_timeout_minutes < 30 or not 0 < args.release_min_fraction <= 1 or args.release_timeout_seconds <= 0:
        raise ValueError("invalid distributed timeout or release threshold")
    _validate_sources(args)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(minutes=args.dist_timeout_minutes))
    base._seed(args.seed, rank)
    report: dict[str, Any] = {
        "status": "FAIL",
        "mode": args.mode,
        "training_contract": CONTRACT,
        "answer_termination": {"token": "<|endoftext|>", "included_in_max_answer_tokens": True, "supervised": True},
        "rank": rank,
        "segments": [],
        "checkpoints": [],
        "hard_failures": [],
    }
    output_available = not args.output_dir.exists() or (args.output_dir.is_dir() and not any(args.output_dir.iterdir()))
    try:
        if not output_available:
            raise FileExistsError(f"refusing nonempty output directory: {args.output_dir}")
        inventory = partition_common._inventory(args.partition_store_root)
        smoke_gate = _formal_smoke_gate(args, inventory) if args.mode == "formal" else None
        schedule = _schedule(args, inventory)
        total_steps = sum(item["steps"] for item in schedule)
        if args.mode == "formal" and total_steps != 37_810:
            raise RuntimeError("formal fixed-recursive step budget must equal 37,810")
        if args.mode == "smoke" and total_steps != 22:
            raise RuntimeError("smoke fixed-recursive step budget must equal 22")
        args.warmup_steps = args.warmup_steps if args.warmup_steps is not None else math.ceil(total_steps * 0.05)
        if args.warmup_steps != math.ceil(total_steps * 0.05):
            raise ValueError("warmup must equal ceil(5% of total optimizer steps)")
        model, tokenizer = _load_model(args, device)
        model.train()
        if not model.trainable_parameter_audit()["training_mode_contract"]:
            raise RuntimeError("fixed-recursive trainable parameter audit failed")
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.max_lr, betas=(0.9, 0.95), weight_decay=0.1)
        scheduler = base._make_scheduler(optimizer, max_lr=args.max_lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps, total_steps=total_steps)
        cursor = {"segment": 0, "segment_step": 0, "global_step": 0}
        saved_plan_hash = None
        if args.resume_from:
            cursor, saved_plan_hash = _resume(args.resume_from, args, inventory, schedule, optimizer, scheduler, rank, device, model._audio_provenance, model)
            if cursor["global_step"] >= total_steps:
                raise RuntimeError("resume checkpoint has no remaining optimizer steps")
            if args.mode == "smoke" and cursor != {"segment": 2, "segment_step": 0, "global_step": 20}:
                raise RuntimeError("smoke resume requires released checkpoint-000020")
            if args.expected_resume_step is not None and cursor["global_step"] != args.expected_resume_step:
                raise RuntimeError(f"resume parent step {cursor['global_step']} != expected {args.expected_resume_step}")
        elif args.expected_resume_step is not None:
            raise ValueError("--expected-resume-step requires --resume-from")
        resume_representatives = base._select_resume_representatives(model) if args.resume_from else None
        resume_snapshots = base._snapshot_resume_representatives(resume_representatives) if resume_representatives else None
        resume_parameter_change_audit = None
        ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        run_start_hashes = _audio_state_hashes(ddp.module)
        gathered_hashes = partition_common._gather(run_start_hashes, world)
        if any(item != gathered_hashes[0] for item in gathered_hashes):
            raise RuntimeError(f"DDP audio initialization hashes differ across ranks: {gathered_hashes}")
        if args.resume_from:
            parent_config = json.loads((args.resume_from / CONFIG_FILENAME).read_text(encoding="utf-8"))
            if run_start_hashes != parent_config.get("checkpoint_audio_state_hashes"):
                raise RuntimeError("resumed DDP bridge/c2l hashes differ from parent checkpoint")
        else:
            ddp.module._audio_initialization_hashes = run_start_hashes
        global_batch = world * args.micro_batch_size * args.gradient_accumulation_steps
        report.update({
            "inventory": inventory,
            "schedule": schedule,
            "total_steps": total_steps,
            "warmup_steps": args.warmup_steps,
            "optimizer": "AdamW",
            "optimizer_betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "gradient_clip_norm": 0.5,
            "scheduler": "cosine_lambda",
            "effective_global_batch_size": global_batch,
            "periodic_validation": False,
            "frozen_audio_encoder": True,
            "start_cursor": dict(cursor),
            "prefix_contract": {"single": 130, "dual": 260},
            "seed": args.seed,
            "resume_checkpoint": str(args.resume_from.resolve()) if args.resume_from else None,
            "smoke_gate": smoke_gate,
            "recursive_text_contract": _recursive_metadata(),
            "text_model_runtime_contract": ddp.module.text_contract,
            "audio_initialization_hashes": ddp.module._audio_initialization_hashes,
            "run_start_audio_state_hashes": run_start_hashes,
        })
        first_audit = None
        stop_step = 20 if args.mode == "smoke" and not args.resume_from else total_steps
        for segment_index in range(cursor["segment"], len(schedule)):
            segment = schedule[segment_index]
            if cursor["global_step"] >= stop_step:
                break
            partition_id = segment["partition_id"]
            partition = inventory["partitions"][partition_id]
            partition_path = Path(inventory["root"]) / f"partition_{partition_id}"
            memory_before = partition_common._memory()
            preload_started = time.perf_counter()
            dataset = ReasonAQADataset(partition_path / "rows.jsonl", tokenizer, unique_waveform_store_dir=partition_path)
            if len(dataset) != partition["rows"]:
                raise RuntimeError("partition QA cardinality changed")
            cache = partition_common._RankLocalStoreWaveforms(dataset)
            for audio_id in range(partition["audio"]):
                cache._get_audio_id(audio_id)
            if cache.current_bytes != partition["bytes"] or cache.misses != partition["audio"]:
                raise RuntimeError("partition preload cardinality/bytes mismatch")
            chunks, plan_audit = partition_common._plan(dataset, seed=args.seed, epoch=segment["epoch"], pid=partition_id, steps=segment["steps"], global_batch=global_batch)
            if segment_index == cursor["segment"] and cursor["segment_step"] and saved_plan_hash != plan_audit["plan_sha256"]:
                raise RuntimeError("resume optimizer-window plan hash mismatch")
            dist.barrier()
            memory_after_load = partition_common._memory()
            segment_report: dict[str, Any] = {
                "segment": segment,
                "plan": plan_audit,
                "preload_seconds": time.perf_counter() - preload_started,
                "memory_before": memory_before,
                "memory_after_load": memory_after_load,
                "steps": [],
            }
            if rank == 0:
                print(f"[recursive-partition] loaded p{partition_id} epoch={segment['epoch']} steps={segment['steps']} bytes/rank={partition['bytes']} seconds={segment_report['preload_seconds']:.1f}", flush=True)
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
                    device_keys = {"audio1", "audio2", "text_ids"}
                    batch = {key: (value.to(device) if torch.is_tensor(value) and key in device_keys else value) for key, value in batch.items()}
                    with (ddp.no_sync() if micro != 3 else contextlib.nullcontext()):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            output = ddp(**{key: value for key, value in batch.items() if key not in {"row_indices", "audio2_reused"}})
                        if output.loss is None or not bool(torch.isfinite(output.loss)):
                            raise RuntimeError("nonfinite fixed-recursive partition loss")
                        step_loss_sum.add_(output.loss.detach().float())
                        (output.loss / 4).backward()
                    if first_audit is None and micro == 3:
                        first_audit = _sequence_and_gradient_audit(ddp.module, batch)
                    del batch, output
                if resume_representatives is not None and resume_parameter_change_audit is None:
                    base._verify_resume_representative_gradients(resume_representatives)
                torch.nn.utils.clip_grad_norm_(ddp.parameters(), 0.5, error_if_nonfinite=True)
                lr_used = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
                if resume_representatives is not None and resume_snapshots is not None and resume_parameter_change_audit is None:
                    resume_parameter_change_audit = base._compute_resume_parameter_change_audit(resume_representatives, resume_snapshots)
                    base._validate_parameter_change_audit(resume_parameter_change_audit)
                cursor = {"segment": segment_index, "segment_step": local_step + 1, "global_step": cursor["global_step"] + 1}
                torch.cuda.synchronize(device)
                step_loss = float((step_loss_sum / args.gradient_accumulation_steps).item())
                segment_report["steps"].append({
                    "global_step": cursor["global_step"],
                    "partition_step": local_step + 1,
                    "slot": "mixed" if len(slot_kind) == 2 else "single" if True in slot_kind else "dual",
                    "loss": step_loss,
                    "lr": lr_used,
                    "seconds": time.perf_counter() - step_started,
                })
                at_boundary = local_step + 1 == segment["steps"]
                if rank == 0 and (cursor["global_step"] % 10 == 0 or at_boundary):
                    print(f"[recursive-partition] step={cursor['global_step']}/{total_steps} p{partition_id} {local_step+1}/{segment['steps']} loss={step_loss:.6f} lr={lr_used:.8g} {segment_report['steps'][-1]['seconds']:.3f}s", flush=True)
                if args.mode == "formal" and cursor["global_step"] % args.save_every == 0 and not at_boundary:
                    checkpoint = args.output_dir / f"checkpoint-{cursor['global_step']:06d}"
                    _checkpoint(checkpoint, ddp.module, tokenizer, optimizer, scheduler, args, inventory, schedule, cursor, rank, world, device, plan_audit["plan_sha256"], resume_parameter_change_audit)
                    if rank == 0:
                        report["checkpoints"].append(str(checkpoint))
                        report["retained_checkpoints"] = base._prune_checkpoints(args.output_dir, args.checkpoint_retention)
                    dist.barrier()
            after_stats = cache.stats()
            if after_stats["waveform_misses"] != initial_stats["waveform_misses"] or after_stats["cloned_bytes"] != initial_stats["cloned_bytes"] or after_stats["resident_unique_audio"] != partition["audio"]:
                raise RuntimeError("training changed complete rank-RAM residency")
            torch.cuda.synchronize(device)
            del chunks, cache.items
            dataset.unique_waveform_store.close()
            del cache, dataset
            gc.collect()
            partition_common._trim()
            dist.barrier()
            threshold = int(partition["bytes"] * args.release_min_fraction)
            cgroup_required = int(partition["bytes"] * world * args.release_min_fraction)
            deadline = time.monotonic() + args.release_timeout_seconds
            polls = 0
            while True:
                memory_after_release = partition_common._memory()
                polls += 1
                current_anon = partition_common._anon(memory_after_release)
                loaded_anon = partition_common._anon(memory_after_load)
                rss_ok = memory_after_load["rss_bytes"] - memory_after_release["rss_bytes"] >= threshold
                cgroup_ok = current_anon is not None and loaded_anon is not None and loaded_anon - current_anon >= cgroup_required
                if (rss_ok and cgroup_ok) or time.monotonic() >= deadline:
                    break
                time.sleep(2.0)
            released = memory_after_load["rss_bytes"] - memory_after_release["rss_bytes"]
            local_release = {
                "rank": rank,
                "rss_drop_bytes": released,
                "required_drop_bytes": threshold,
                "before": memory_before,
                "after_load": memory_after_load,
                "after_release": memory_after_release,
                "polls": polls,
                "passed": released >= threshold,
            }
            releases = partition_common._gather(local_release, world)
            segment_report["release"] = releases
            loaded_anon = partition_common._anon(memory_after_load)
            released_anon = partition_common._anon(memory_after_release)
            cgroup_drop = None if loaded_anon is None or released_anon is None else loaded_anon - released_anon
            segment_report["cgroup_release"] = {
                "anon_drop_bytes": cgroup_drop,
                "required_bytes": cgroup_required,
                "available": cgroup_drop is not None,
                "timeout_seconds": args.release_timeout_seconds,
            }
            if cgroup_drop is None:
                raise RuntimeError(f"partition p{partition_id} cannot audit cgroup anon release")
            if not all(item["passed"] for item in releases):
                raise RuntimeError(f"partition p{partition_id} per-rank RAM release failed")
            if cgroup_drop < cgroup_required:
                raise RuntimeError(f"partition p{partition_id} cgroup anon release failed: {cgroup_drop} < {cgroup_required}")
            report["segments"].append(segment_report)
            cursor = {"segment": segment_index + 1, "segment_step": 0, "global_step": cursor["global_step"]}
            at_end = cursor["global_step"] == total_steps
            save_boundary = (args.mode == "smoke" and cursor["global_step"] in (20, 22)) or (args.mode == "formal" and (cursor["global_step"] % args.save_every == 0 or at_end))
            if save_boundary:
                checkpoint = args.output_dir / f"checkpoint-{cursor['global_step']:06d}"
                _checkpoint(checkpoint, ddp.module, tokenizer, optimizer, scheduler, args, inventory, schedule, cursor, rank, world, device, None, resume_parameter_change_audit)
                if rank == 0:
                    report["checkpoints"].append(str(checkpoint))
                    report["retained_checkpoints"] = base._prune_checkpoints(args.output_dir, args.checkpoint_retention)
                dist.barrier()
            if rank == 0:
                print(f"[recursive-partition] released p{partition_id} RSS drop/rank={[round(item['rss_drop_bytes']/1024**3, 2) for item in releases]} GiB", flush=True)
        if cursor["global_step"] != stop_step:
            raise RuntimeError(f"run ended at step {cursor['global_step']} rather than {stop_step}")
        report.update({
            "status": "PASS",
            "end_cursor": cursor,
            "first_step_gradient_audit": first_audit,
            "resume_verified_two_steps": args.mode == "smoke" and args.resume_from is not None and cursor["global_step"] == 22,
            "resume_parameter_change_audit": resume_parameter_change_audit,
        })
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
