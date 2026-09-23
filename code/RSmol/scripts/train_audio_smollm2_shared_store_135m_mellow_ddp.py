#!/usr/bin/env python3
"""Train original SmolLM2-135M on the isolated node-shared audio store."""
from __future__ import annotations

import argparse
import contextlib
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
from torch.utils.data import DataLoader, DistributedSampler

import train_audio_smollm2_135m_mellow_ddp as baseline
from audio_smollm2_135m_mellow_shared_store import TRAINING_CONTRACT
from audio_smollm2_135m_mellow_shared_store.data import ReasonAQADataset, collate_reasonaqa
from audio_smollm2_135m_mellow_shared_store.model import (
    ORIGINAL_SMOLLM2_CONTRACT,
    MAPPER_CONTRACT,
    AUDIO_PREFIX_TOKENS,
    AUDIO_DUAL_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    SMOLLM2_HIDDEN_SIZE,
)


CONFIG_FILENAME = "audio_smollm2_shared_store_config.json"
DEFAULT_MANIFEST = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/"
    "preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl"
)
DEFAULT_STORE = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/"
    "rsmol_reasonaqa_train_unique_waveforms_32k_10s_f32_v3"
)
ANSWER_TERMINATION = {
    "token": "<|endoftext|>",
    "included_in_max_answer_tokens": True,
    "supervised": True,
}
PREFIX_TOKENS = {"single": AUDIO_DUAL_PREFIX_TOKENS, "dual": AUDIO_DUAL_PREFIX_TOKENS}
SMOKE_TOTAL_STEPS = 22
SMOKE_FIRST_STOP = 20
FORMAL_EPOCHS = 30
CANONICAL_MAX_LR = 1e-3
CANONICAL_MIN_LR = 1e-4
AUDIO_SLOT_SEMANTICS = (
    "fixed260_second_slot_reuses_audio1_htsat_embedding_then_runs_bridge_separately"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "formal"), required=True)
    p.add_argument("--train-manifest", type=Path, required=True,
                   help="Staged manifest under /dev/shm")
    p.add_argument("--unique-waveform-store-dir", type=Path, required=True,
                   help="Staged complete unique store under /dev/shm")
    p.add_argument("--persistent-manifest-source", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--persistent-store-source", type=Path, default=DEFAULT_STORE)
    p.add_argument("--store-copy-seconds", type=float, required=True)
    p.add_argument("--manifest-copy-seconds", type=float, required=True)
    p.add_argument("--staging-total-seconds", type=float, required=True)
    p.add_argument("--model-path", "--smollm2-model", dest="model_path",
                   type=Path, default=Path(baseline.DEFAULT_MODEL))
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--smoke20-report", type=Path)
    p.add_argument("--smoke-resume-report", type=Path)
    p.add_argument("--tokenizer-path", type=Path)
    p.add_argument("--htsat-checkpoint", type=Path, default=Path(baseline.DEFAULT_HTSAT))
    p.add_argument("--mellow-root", type=Path, default=Path(baseline.DEFAULT_MELLOW))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--checkpoint-retention", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dist-timeout-minutes", type=int, default=30)
    return p.parse_args(argv)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _store_inventory(args: argparse.Namespace) -> dict[str, Any]:
    staged = args.unique_waveform_store_dir.expanduser().resolve(strict=True)
    source = args.persistent_store_source.expanduser().resolve(strict=True)
    staged_manifest = args.train_manifest.expanduser().resolve(strict=True)
    source_manifest = args.persistent_manifest_source.expanduser().resolve(strict=True)
    if Path("/dev/shm") not in staged.parents:
        raise RuntimeError(f"staged unique store must be under /dev/shm: {staged}")
    if Path("/dev/shm") not in staged_manifest.parents:
        raise RuntimeError(f"staged manifest must be under /dev/shm: {staged_manifest}")
    for root in (staged, source):
        if (root / "BUILDING").exists():
            raise RuntimeError(f"unique store is still BUILDING: {root}")
        for name in ("metadata.json", "index.jsonl", "waveforms.f32"):
            if not (root / name).is_file():
                raise FileNotFoundError(f"unique store lacks {name}: {root}")
    staged_meta = _json(staged / "metadata.json")
    source_meta = _json(source / "metadata.json")
    source_metadata_sha = _sha(source / "metadata.json")
    staged_metadata_sha = _sha(staged / "metadata.json")
    if staged_metadata_sha != source_metadata_sha:
        raise RuntimeError("source/staged metadata SHA256 audit failed")
    required_meta = {
        "status": "PASS",
        "format": "manifest_unique_fixed_waveform_store_v1",
        "sample_rate": 32000,
        "seconds": 10,
        "samples_per_audio": 320000,
        "bytes_per_audio": 1280000,
        "dtype": "float32",
        "byte_order": "little",
        "data_file": "waveforms.f32",
    }
    for label, meta in (("source", source_meta), ("staged", staged_meta)):
        mismatch = {key: (expected, meta.get(key)) for key, expected in required_meta.items()
                    if meta.get(key) != expected}
        if mismatch:
            raise RuntimeError(f"{label} unique-store contract mismatch: {mismatch}")
    identity_keys = (
        "manifest_sha256", "source_inventory_sha256", "index_sha256",
        "waveform_sha256", "num_unique_audio_files", "total_waveform_bytes",
    )
    mismatch = {key: (source_meta.get(key), staged_meta.get(key)) for key in identity_keys
                if source_meta.get(key) != staged_meta.get(key)}
    if mismatch:
        raise RuntimeError(f"source/staged unique-store identity mismatch: {mismatch}")
    source_index_sha = _sha(source / "index.jsonl")
    staged_index_sha = _sha(staged / "index.jsonl")
    if source_index_sha != source_meta["index_sha256"] or staged_index_sha != source_index_sha:
        raise RuntimeError("source/staged index SHA256 audit failed")
    manifest_sha = _sha(source_manifest)
    if _sha(staged_manifest) != manifest_sha or manifest_sha != source_meta["manifest_sha256"]:
        raise RuntimeError("source/staged manifest SHA256 differs from unique-store metadata")
    expected_bytes = int(source_meta["total_waveform_bytes"])
    source_bytes = (source / "waveforms.f32").stat().st_size
    staged_bytes = (staged / "waveforms.f32").stat().st_size
    if source_bytes != expected_bytes or staged_bytes != expected_bytes:
        raise RuntimeError(
            f"source/staged waveform size mismatch: expected={expected_bytes} "
            f"source={source_bytes} staged={staged_bytes}"
        )
    return {
        "persistent_store_source": str(source),
        "persistent_manifest_source": str(source_manifest),
        "staged_store_path": str(staged),
        "staged_manifest_path": str(staged_manifest),
        "manifest_sha256": manifest_sha,
        "metadata_sha256": source_metadata_sha,
        "index_sha256": source_index_sha,
        "waveform_sha256": str(source_meta["waveform_sha256"]),
        "num_unique_audio_files": int(source_meta["num_unique_audio_files"]),
        "total_waveform_bytes": expected_bytes,
        "total_waveform_gib": expected_bytes / 1024**3,
        "preprocessing_contract": source_meta.get("preprocessing_contract"),
        "sharing_contract": "one complete immutable store staged in node-shared /dev/shm; no rank-local full-store clone",
    }


def _gather(local: Any, world: int) -> list[Any]:
    gathered: list[Any] = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered[0] = local
    return gathered


def _staging_rank_audit(inventory: dict[str, Any], rank: int, world: int) -> list[dict[str, Any]]:
    store = Path(inventory["staged_store_path"])
    data = store / "waveforms.f32"
    stat = data.stat()
    local = {
        "rank": rank,
        "store_path": str(store),
        "data_path": str(data),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "bytes": int(stat.st_size),
    }
    rows = _gather(local, world)
    identities = {(item["store_path"], item["device"], item["inode"], item["bytes"]) for item in rows}
    if len(identities) != 1:
        raise RuntimeError(f"DDP ranks do not see one shared staged waveform inode: {rows}")
    return rows


def _dataset_audio_slot_audit(dataset: ReasonAQADataset, rank: int, world: int) -> dict[str, Any]:
    payload: dict[str, Any] | None = None
    if rank == 0:
        try:
            single_rows = 0
            dual_rows = 0
            dual_same_waveform_rows = 0
            invalid_single_rows: list[int] = []
            for index in range(len(dataset)):
                single_slot, same_waveform = dataset.audio_structure(index)
                if single_slot:
                    single_rows += 1
                    if not same_waveform and len(invalid_single_rows) < 20:
                        invalid_single_rows.append(index)
                else:
                    dual_rows += 1
                    dual_same_waveform_rows += int(same_waveform)
            payload = {
                "passed": not invalid_single_rows and single_rows > 0 and dual_rows > 0,
                "total_rows": len(dataset),
                "single_audio_rows": single_rows,
                "dual_audio_rows": dual_rows,
                "dual_rows_with_identical_explicit_paths": dual_same_waveform_rows,
                "invalid_single_row_indices": invalid_single_rows,
                "single_audio_contract": "audio2 absent structurally; reuse audio1 HTSAT embedding for slot two",
                "dual_audio_contract": "preserve both explicit waveform slots",
            }
        except Exception as exc:
            payload = {
                "passed": False,
                "total_rows": len(dataset),
                "error": repr(exc),
            }
    shared = [payload]
    if world > 1:
        dist.broadcast_object_list(shared, src=0)
    result = shared[0]
    if not isinstance(result, dict) or result.get("passed") is not True:
        raise RuntimeError(f"ReasonAQA audio-slot manifest audit failed: {result}")
    return result


def _training_shape(args: argparse.Namespace, dataset_rows: int) -> dict[str, int]:
    global_batch = args.world_size * args.micro_batch_size * args.gradient_accumulation_steps
    epochs = FORMAL_EPOCHS if args.epochs is None else int(args.epochs)
    steps_per_epoch = dataset_rows // global_batch
    total_steps = steps_per_epoch * epochs
    if epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("epochs and steps_per_epoch must be positive")
    if epochs != FORMAL_EPOCHS:
        raise ValueError(f"shared-store fixed260 route requires epochs={FORMAL_EPOCHS}")
    args.epochs = epochs
    return {
        "global_batch_size": global_batch,
        "steps_per_epoch": steps_per_epoch,
        "microbatches_per_epoch": steps_per_epoch * args.gradient_accumulation_steps,
        "dropped_rows_per_epoch": dataset_rows - steps_per_epoch * global_batch,
        "total_steps": total_steps,
    }


def _formal_gate(args: argparse.Namespace, inventory: dict[str, Any],
                 shape: dict[str, int]) -> dict[str, Any] | None:
    if args.mode != "formal":
        return None
    if args.smoke20_report is None or args.smoke_resume_report is None:
        raise ValueError("formal training requires --smoke20-report and --smoke-resume-report")
    first = _json(args.smoke20_report)
    resumed = _json(args.smoke_resume_report)
    for label, report, start, end in (
        ("smoke20", first, 0, 20),
        ("resume2", resumed, 20, 22),
    ):
        if (report.get("status") != "PASS" or report.get("mode") != "smoke"
                or report.get("training_contract") != TRAINING_CONTRACT
                or report.get("start_global_step") != start
                or report.get("end_global_step") != end
                or report.get("hard_failures")
                or report.get("epochs") != FORMAL_EPOCHS
                or report.get("max_lr") != CANONICAL_MAX_LR
                or report.get("min_lr") != CANONICAL_MIN_LR
                or report.get("compact_single_audio_prefix") is not False
                or report.get("prefix_tokens") != PREFIX_TOKENS
                or report.get("single_audio_slot_semantics") != AUDIO_SLOT_SEMANTICS
                or report.get("shape") != shape
                or report.get("warmup_steps") != args.warmup_steps
                or report.get("warmup_default_ceil_5_percent") != args.warmup_steps
                or report.get("sampler", {}).get("seed") != args.seed
                or report.get("initialization", {}).get("fresh_source") != str(args.model_path.resolve())):
            raise RuntimeError(f"formal gate rejects {label} report")
        report_inventory = report.get("store_inventory", {})
        for key in ("manifest_sha256", "index_sha256", "waveform_sha256", "total_waveform_bytes"):
            if report_inventory.get(key) != inventory.get(key):
                raise RuntimeError(f"formal gate {label} store differs in {key}")
        gradient = report.get("first_step_gradient_audit", {})
        required_gradient_audit = (
            gradient.get("standard_30_layer_smollm2") is True,
            gradient.get("all_decoder_layers_have_finite_gradient") is True,
            gradient.get("embedding_has_finite_gradient") is True,
            gradient.get("lm_head_has_finite_gradient") is True,
            gradient.get("all_bridge_gradients_finite") is True,
            gradient.get("all_c2l_gradients_finite") is True,
            gradient.get("htsat_frozen_and_gradient_free") is True,
            gradient.get("has_router_parameters") is False,
            gradient.get("training_mode_contract") is True,
            gradient.get("fixed260_audio_reuse_contract") is True,
        )
        if not all(required_gradient_audit):
            raise RuntimeError(f"formal gate rejects {label} SmolLM2 gradient audit")
        trainable = report.get("model_trainable_audit", {})
        if (
            trainable.get("training_mode_contract") is not True
            or trainable.get("all_text_trainable") is not True
            or trainable.get("bridge_trainable") is not True
            or trainable.get("c2l_trainable") is not True
            or trainable.get("htsat_frozen") is not True
            or trainable.get("independent_decoder_layers") is not True
            or trainable.get("decoder_layer_count") != 30
            or trainable.get("has_router_parameters") is not False
        ):
            raise RuntimeError(f"formal gate rejects {label} trainable-parameter audit")
        if report.get("answer_only_label_audit", {}).get("passed") is not True:
            raise RuntimeError(f"formal gate rejects {label} label audit")
        if report.get("answer_only_label_audit", {}).get("terminal_eos_supervised") is not True:
            raise RuntimeError(f"formal gate rejects {label} terminal EOS audit")
        slot_audit = report.get("answer_only_label_audit", {}).get("audio_slot_audit", {})
        if (
            slot_audit.get("passed") is not True
            or slot_audit.get("single_rows_reuse_audio1") is not True
            or slot_audit.get("audio_tokens_per_slot")
            != [AUDIO_TOKENS_PER_CLIP, AUDIO_TOKENS_PER_CLIP]
            or slot_audit.get("two_bridge_invocations_contract") is not True
        ):
            raise RuntimeError(f"formal gate rejects {label} fixed260 audio-slot audit")
        manifest_slot_audit = report.get("dataset_audio_slot_audit", {})
        if (
            manifest_slot_audit.get("passed") is not True
            or int(manifest_slot_audit.get("single_audio_rows", 0)) <= 0
            or int(manifest_slot_audit.get("dual_audio_rows", 0)) <= 0
            or manifest_slot_audit.get("invalid_single_row_indices") != []
        ):
            raise RuntimeError(f"formal gate rejects {label} manifest audio-slot audit")
    first_checkpoints = first.get("checkpoints", [])
    resumed_checkpoints = resumed.get("checkpoints", [])
    if len(first_checkpoints) != 1 or len(resumed_checkpoints) != 1:
        raise RuntimeError("smoke reports must expose exactly checkpoint-000020 and checkpoint-000022")
    first_checkpoint = str(Path(first_checkpoints[0]).resolve())
    if resumed.get("resume_checkpoint") != first_checkpoint or resumed.get("resume_verified_two_steps") is not True:
        raise RuntimeError("resume smoke does not prove exact continuation from checkpoint-000020")
    _audit_checkpoint(
        Path(first_checkpoint),
        expected_step=20,
        expected_mode="smoke",
        args=args,
        inventory=inventory,
        shape=shape,
    )
    _audit_checkpoint(
        Path(resumed_checkpoints[0]),
        expected_step=22,
        expected_mode="smoke",
        args=args,
        inventory=inventory,
        shape=shape,
    )
    baseline._validate_parameter_change_audit(
        resumed.get("resume_parameter_change_audit")
    )
    return {
        "smoke20_report": str(args.smoke20_report.resolve()),
        "smoke_resume_report": str(args.smoke_resume_report.resolve()),
        "checkpoint20": first_checkpoint,
    }


def _checkpoint_config(args: argparse.Namespace, inventory: dict[str, Any], shape: dict[str, int],
                       provenance: dict[str, Any], model: Any) -> dict[str, Any]:
    source_config = args.model_path.resolve() / "config.json"
    return {
        "contract": TRAINING_CONTRACT,
        "architecture_contract": ORIGINAL_SMOLLM2_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "mapper_initialization": "random_c2l_and_xavier_projection",
        "text_model_source_path": str(args.model_path.resolve()),
        "text_model_source_config_sha256": (
            _sha(source_config) if source_config.is_file() else None
        ),
        "standard_text_contract": model.text_contract,
        "compact_single_audio_prefix": False,
        "single_audio_slot_semantics": AUDIO_SLOT_SEMANTICS,
        "answer_termination": ANSWER_TERMINATION,
        "prefix_tokens": PREFIX_TOKENS,
        "text_hidden_size": SMOLLM2_HIDDEN_SIZE,
        "audio_tokens_per_clip": AUDIO_TOKENS_PER_CLIP,
        "audio_prefix_tokens_with_separators": AUDIO_PREFIX_TOKENS,
        "data_pipeline": {
            "kind": "node_shared_tmpfs_unique_waveform_store",
            "preprocessed_waveforms": "mono 32kHz first-10s crop/right-zero-pad float32",
            "store_residency": "one complete node-shared /dev/shm inode",
            "rank_local_full_store_clone": False,
            "shuffle_scope": "entire manifest independently each epoch before disjoint DistributedSampler rank partition",
            "sampler": "torch.utils.data.DistributedSampler",
            "shuffle": True,
            "drop_last": True,
        },
        "store_identity": {key: inventory[key] for key in (
            "persistent_store_source", "persistent_manifest_source", "manifest_sha256",
            "metadata_sha256", "index_sha256", "waveform_sha256",
            "num_unique_audio_files", "total_waveform_bytes",
        )},
        "mode": args.mode,
        "epochs": args.epochs,
        "world_size": args.world_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "max_lr": args.max_lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "total_steps": shape["total_steps"],
        "steps_per_epoch": shape["steps_per_epoch"],
        "save_every": args.save_every,
        "checkpoint_retention": args.checkpoint_retention,
        "dist_timeout_minutes": args.dist_timeout_minutes,
        "htsat_checkpoint": str(args.htsat_checkpoint.resolve()),
        "mellow_root": str(args.mellow_root.resolve()),
        "mellow_provenance": provenance,
    }


def _required_checkpoint_entries() -> list[str]:
    return [
        "text_model",
        "tokenizer",
        "audio_bridge.pt",
        "training_state.pt",
        CONFIG_FILENAME,
    ]


def _audit_checkpoint(
    path: Path,
    *,
    expected_step: int,
    expected_mode: str,
    args: argparse.Namespace,
    inventory: dict[str, Any],
    shape: dict[str, int],
) -> dict[str, Any]:
    path = path.resolve(strict=True)
    marker = _json(path / "checkpoint_complete.json")
    config = _json(path / CONFIG_FILENAME)
    required = _required_checkpoint_entries()
    source_config = args.model_path.resolve() / "config.json"
    expected_source_config_sha = _sha(source_config) if source_config.is_file() else None
    if (
        marker.get("status") != "complete"
        or marker.get("contract") != TRAINING_CONTRACT
        or marker.get("global_step") != expected_step
        or marker.get("required") != required
        or config.get("contract") != TRAINING_CONTRACT
        or config.get("global_step") != expected_step
        or config.get("mode") != expected_mode
        or config.get("architecture_contract") != ORIGINAL_SMOLLM2_CONTRACT
        or config.get("mapper_contract") != MAPPER_CONTRACT
        or config.get("compact_single_audio_prefix") is not False
        or config.get("single_audio_slot_semantics") != AUDIO_SLOT_SEMANTICS
        or config.get("prefix_tokens") != PREFIX_TOKENS
        or config.get("answer_termination") != ANSWER_TERMINATION
        or config.get("text_hidden_size") != SMOLLM2_HIDDEN_SIZE
        or config.get("audio_tokens_per_clip") != AUDIO_TOKENS_PER_CLIP
        or config.get("audio_prefix_tokens_with_separators") != AUDIO_PREFIX_TOKENS
        or config.get("epochs") != FORMAL_EPOCHS
        or config.get("world_size") != args.world_size
        or config.get("micro_batch_size") != args.micro_batch_size
        or config.get("gradient_accumulation_steps") != args.gradient_accumulation_steps
        or config.get("num_workers") != args.num_workers
        or config.get("seed") != args.seed
        or config.get("max_lr") != args.max_lr
        or config.get("min_lr") != args.min_lr
        or config.get("warmup_steps") != args.warmup_steps
        or config.get("total_steps") != shape["total_steps"]
        or config.get("steps_per_epoch") != shape["steps_per_epoch"]
        or config.get("save_every") != args.save_every
        or config.get("checkpoint_retention") != args.checkpoint_retention
        or config.get("dist_timeout_minutes") != args.dist_timeout_minutes
        or config.get("htsat_checkpoint") != str(args.htsat_checkpoint.resolve())
        or config.get("mellow_root") != str(args.mellow_root.resolve())
        or config.get("store_identity", {}).get("manifest_sha256")
        != inventory.get("manifest_sha256")
        or config.get("store_identity", {}).get("index_sha256")
        != inventory.get("index_sha256")
        or config.get("store_identity", {}).get("waveform_sha256")
        != inventory.get("waveform_sha256")
        or config.get("store_identity", {}).get("total_waveform_bytes")
        != inventory.get("total_waveform_bytes")
        or Path(str(config.get("text_model_source_path", ""))).resolve()
        != args.model_path.resolve()
        or config.get("text_model_source_config_sha256") != expected_source_config_sha
    ):
        raise RuntimeError(f"SmolLM2 shared-store checkpoint contract mismatch: {path}")
    standard = config.get("standard_text_contract", {})
    if (
        standard.get("model_type") != "llama"
        or standard.get("num_hidden_layers") != 30
        or standard.get("physical_decoder_layer_count") != 30
        or standard.get("independent_decoder_layers") is not True
    ):
        raise RuntimeError(f"checkpoint is not the standard 30-layer SmolLM2: {path}")
    required_paths = [path / item for item in required]
    if any(not item.exists() for item in required_paths):
        raise RuntimeError(f"SmolLM2 shared-store checkpoint is incomplete: {path}")
    if not baseline._text_model_weight_files(path / "text_model"):
        raise RuntimeError(f"SmolLM2 checkpoint has no text-model weights: {path}")
    return {"path": str(path), "global_step": expected_step, "status": "PASS"}


def _save_checkpoint(path: Path, model: Any, tokenizer: Any, optimizer: Any, scheduler: Any,
                     args: argparse.Namespace, inventory: dict[str, Any], shape: dict[str, int],
                     cursor: dict[str, int], rank: int, world: int, device: torch.device) -> None:
    rng_states = _gather(baseline._rng_state(device), world)
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
        torch.save(baseline._trainable_state(model), temporary / "audio_bridge.pt")
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scheduler_name": "cosine_lambda",
            "optimizer_parameter_names": [
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ],
            "training_contract": TRAINING_CONTRACT,
            "global_step": cursor["global_step"],
            "cursor": cursor,
            "rng_states_by_rank": {str(i): state for i, state in enumerate(rng_states)},
        }, temporary / "training_state.pt")
        config = _checkpoint_config(args, inventory, shape, model._audio_provenance, model)
        config.update({
            "global_step": cursor["global_step"],
            "epoch": cursor["epoch"],
            "batch_in_epoch": cursor["batch_in_epoch"],
        })
        (temporary / CONFIG_FILENAME).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        required_entries = _required_checkpoint_entries()
        marker = {
            "status": "complete",
            "global_step": cursor["global_step"],
            "contract": TRAINING_CONTRACT,
            "required": required_entries,
        }
        (temporary / "checkpoint_complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        required = (
            temporary / "text_model" / "config.json",
            temporary / "tokenizer" / "tokenizer_config.json",
            temporary / "audio_bridge.pt",
            temporary / "training_state.pt",
            temporary / CONFIG_FILENAME,
            temporary / "checkpoint_complete.json",
        )
        if any(not item.is_file() for item in required):
            raise RuntimeError("refusing to publish incomplete SmolLM2 shared-store checkpoint")
        if not baseline._text_model_weight_files(temporary / "text_model"):
            raise RuntimeError("refusing checkpoint without SmolLM2 text-model weights")
        temporary.replace(path)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def _resume(path: Path, args: argparse.Namespace, inventory: dict[str, Any], shape: dict[str, int],
            optimizer: Any, scheduler: Any, provenance: dict[str, Any], rank: int,
            device: torch.device, model: Any) -> dict[str, int]:
    path = path.resolve(strict=True)
    marker_before_audit = _json(path / "checkpoint_complete.json")
    resume_step = int(marker_before_audit.get("global_step", -1))
    if args.mode == "smoke" and resume_step != SMOKE_FIRST_STOP:
        raise RuntimeError("smoke resume requires checkpoint-000020")
    if args.mode == "formal" and not (0 < resume_step < shape["total_steps"]):
        raise RuntimeError("formal resume step must be inside the configured training horizon")
    _audit_checkpoint(
        path,
        expected_step=resume_step,
        expected_mode=args.mode,
        args=args,
        inventory=inventory,
        shape=shape,
    )
    config = _json(path / CONFIG_FILENAME)
    marker = _json(path / "checkpoint_complete.json")
    if (config.get("contract") != TRAINING_CONTRACT or marker.get("contract") != TRAINING_CONTRACT
            or marker.get("status") != "complete"):
        raise RuntimeError("resume source is not a complete SmolLM2 shared-store checkpoint")
    expected = _checkpoint_config(args, inventory, shape, provenance, model)
    for key in (
        "contract", "architecture_contract", "mapper_contract", "mapper_initialization",
        "text_model_source_path", "text_model_source_config_sha256", "standard_text_contract",
        "compact_single_audio_prefix", "text_hidden_size",
        "audio_tokens_per_clip", "audio_prefix_tokens_with_separators",
        "single_audio_slot_semantics", "answer_termination", "prefix_tokens", "store_identity", "mode", "epochs", "world_size",
        "micro_batch_size", "gradient_accumulation_steps", "num_workers", "seed", "max_lr",
        "min_lr", "warmup_steps", "total_steps", "steps_per_epoch", "save_every",
        "checkpoint_retention", "dist_timeout_minutes", "htsat_checkpoint", "mellow_root",
    ):
        if config.get(key) != expected.get(key):
            raise RuntimeError(f"resume contract differs in {key}")
    if config.get("mellow_provenance", {}).get("mellow_htsat_sha256") != provenance.get("mellow_htsat_sha256"):
        raise RuntimeError("resume Mellow implementation SHA256 mismatch")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    baseline._validate_optimizer_coverage(state, model)
    if state.get("training_contract") != TRAINING_CONTRACT:
        raise RuntimeError("resume training-state contract mismatch")
    if state.get("scheduler_name") != "cosine_lambda":
        raise RuntimeError("resume scheduler type mismatch")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    for optimizer_state in optimizer.state.values():
        for key, value in tuple(optimizer_state.items()):
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    cursor = {key: int(state["cursor"][key]) for key in ("epoch", "batch_in_epoch", "global_step")}
    expected_global = cursor["epoch"] * shape["steps_per_epoch"] + cursor["batch_in_epoch"] // args.gradient_accumulation_steps
    if (cursor["batch_in_epoch"] % args.gradient_accumulation_steps or
            cursor["batch_in_epoch"] > shape["microbatches_per_epoch"] or
            expected_global != cursor["global_step"] or
            int(config.get("global_step", -1)) != cursor["global_step"] or
            int(config.get("epoch", -1)) != cursor["epoch"] or
            int(config.get("batch_in_epoch", -1)) != cursor["batch_in_epoch"] or
            int(marker.get("global_step", -1)) != cursor["global_step"] or
            int(state.get("global_step", -1)) != cursor["global_step"]):
        raise RuntimeError(f"resume cursor/global-step proof failed: {cursor}")
    rng = state.get("rng_states_by_rank", {})
    if set(rng) != {str(i) for i in range(args.world_size)}:
        raise RuntimeError("resume checkpoint per-rank RNG coverage mismatch")
    baseline._restore_rng_state(rng[str(rank)], device)
    return cursor


def _resume_model_args(args: argparse.Namespace) -> argparse.Namespace:
    """Point the baseline loader at this route's checkpoint layout."""

    if args.resume_from is None:
        return args
    proxy = argparse.Namespace(**vars(args))
    proxy.resume_from = args.resume_from
    return proxy


def _answer_label_audit(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    labels = model.last_labels
    prefix_lengths = model.last_prefix_lengths
    if labels is None:
        raise RuntimeError("model did not expose labels")
    if prefix_lengths is None:
        prefix_length = model.last_prefix_length
        if prefix_length is None:
            raise RuntimeError("model did not expose fixed prefix length")
        prefix_lengths = torch.full(
            (labels.shape[0],), int(prefix_length), dtype=torch.long, device=labels.device
        )
    text_ids = batch["text_ids"]
    prompts = batch["prompt_lengths"]
    answers = batch["answer_lengths"]
    single_slot_mask = batch["single_audio_slot_mask"].to(dtype=torch.bool)
    reused_mask = batch["audio2_reused_mask"].to(dtype=torch.bool)
    if single_slot_mask.shape != reused_mask.shape or single_slot_mask.numel() != text_ids.shape[0]:
        raise RuntimeError("audio slot masks do not match the text batch")
    if bool((single_slot_mask & ~reused_mask).any()):
        raise RuntimeError("a structural single-audio row did not reuse audio1 for slot two")
    if batch["audio2"] is None and not bool(reused_mask.all()):
        raise RuntimeError("audio2 is absent even though the batch contains a distinct second waveform")
    if model.last_audio_tokens_per_clip != (AUDIO_TOKENS_PER_CLIP, AUDIO_TOKENS_PER_CLIP):
        raise RuntimeError(
            "fixed260 training requires two independently bridged 129-token audio slots"
        )
    supervised = 0
    for row in range(text_ids.shape[0]):
        prefix = int(prefix_lengths[row].item())
        if prefix != AUDIO_PREFIX_TOKENS:
            raise RuntimeError("shared-store training requires fixed 260-token prefixes")
        prompt = int(prompts[row].item())
        answer = int(answers[row].item())
        if answer <= 0:
            raise RuntimeError("answer has no supervised terminal EOS")
        start, end = prefix + prompt, prefix + prompt + answer
        if bool((labels[row, :start] != -100).any()) or bool((labels[row, end:] != -100).any()):
            raise RuntimeError("answer-only label mask has supervision outside the answer interval")
        if not torch.equal(labels[row, start:end], text_ids[row, prompt:prompt + answer]):
            raise RuntimeError("answer-only labels are not aligned to answer tokens")
        eos_id = int(model.tokenizer.eos_token_id)
        if int(labels[row, end - 1].item()) != eos_id:
            raise RuntimeError("answer terminal EOS is not supervised")
        if answer > 1 and int(labels[row, end - 2].item()) == eos_id:
            raise RuntimeError("answer has duplicate trailing EOS tokens")
        supervised += answer
    return {
        "passed": True, "rows": int(text_ids.shape[0]),
        "prefix_tokens": AUDIO_PREFIX_TOKENS,
        "supervised_answer_tokens": supervised, "terminal_eos_supervised": True,
        "audio_slot_audit": {
            "passed": True,
            "single_audio_rows": int(single_slot_mask.sum().item()),
            "reused_embedding_rows": int(reused_mask.sum().item()),
            "single_rows_reuse_audio1": True,
            "audio_tokens_per_slot": list(model.last_audio_tokens_per_clip),
            "two_bridge_invocations_contract": True,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
    report: dict[str, Any] = {
        "status": "FAIL",
        "mode": args.mode,
        "training_contract": TRAINING_CONTRACT,
        "rank": rank,
        "checkpoints": [],
        "metrics": [],
        "hard_failures": [],
    }
    output_available = not args.output_dir.exists() or not any(args.output_dir.iterdir())
    try:
        if not output_available:
            raise FileExistsError(f"refusing nonempty output directory: {args.output_dir}")
        if (not torch.cuda.is_available() or world != 8 or args.world_size != 8
                or args.micro_batch_size != 8 or args.gradient_accumulation_steps != 4
                or args.num_workers != 0):
            raise RuntimeError("shared-store training requires 8 GPUs, microbatch 8, GA 4, num_workers 0")
        if (args.max_lr != CANONICAL_MAX_LR or args.min_lr != CANONICAL_MIN_LR
                or args.save_every <= 0 or args.checkpoint_retention <= 0
                or args.dist_timeout_minutes < 30):
            raise ValueError(
                "fixed260 shared-store route requires max_lr=1e-3 and min_lr=1e-4; "
                "checkpoint, retention, and distributed timeout must also be valid"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", rank=rank, world_size=world,
                                timeout=timedelta(minutes=args.dist_timeout_minutes))
        baseline._seed(args.seed, rank)

        inventory = _store_inventory(args)
        rank_staging_audit = _staging_rank_audit(inventory, rank, world)
        # The shared-store route changes only waveform residency and delivery.
        # Keep the historical fixed-260 two-slot model contract: a structural
        # single-audio row reuses audio1's HTSAT embedding, then both slots run
        # through the trainable bridge independently.
        args.compact_single_audio_prefix = False
        args.init_from_audio_checkpoint = None
        dataset_started = time.perf_counter()
        # This constructor re-hashes the staged manifest, verifies it against
        # metadata, verifies index SHA256, and builds the path-to-audio map.
        dataset = ReasonAQADataset(
            args.train_manifest,
            tokenizer=None,
            unique_waveform_store_dir=args.unique_waveform_store_dir,
        )
        dataset_init_seconds = time.perf_counter() - dataset_started
        dataset_audio_slot_audit = _dataset_audio_slot_audit(dataset, rank, world)
        shape = _training_shape(args, len(dataset))
        default_warmup = math.ceil(shape["total_steps"] * 0.05)
        args.warmup_steps = default_warmup if args.warmup_steps is None else int(args.warmup_steps)
        if args.warmup_steps != default_warmup:
            raise ValueError(f"warmup_steps must equal ceil(total_steps * 0.05)={default_warmup}")
        formal_gate = _formal_gate(args, inventory, shape)

        model, tokenizer = baseline._load_model(_resume_model_args(args), device)
        dataset.tokenizer = tokenizer
        model.train()
        if not model.trainable_parameter_audit()["training_mode_contract"]:
            raise RuntimeError("trainable parameter audit failed")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.max_lr, betas=(0.9, 0.95), weight_decay=0.1,
        )
        scheduler = baseline._make_scheduler(
            optimizer, max_lr=args.max_lr, min_lr=args.min_lr,
            warmup_steps=args.warmup_steps, total_steps=shape["total_steps"],
        )
        # Construct DDP before restoring RNG so process-group/module setup
        # cannot perturb the exact per-rank continuation state.
        ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        cursor = {"epoch": 0, "batch_in_epoch": 0, "global_step": 0}
        if args.resume_from is not None:
            cursor = _resume(
                args.resume_from, args, inventory, shape, optimizer, scheduler,
                model._audio_provenance, rank, device, model,
            )
        resume_representatives = None
        resume_snapshots = None
        resume_gradient_verification = None
        resume_parameter_change_audit = None
        if args.resume_from is not None:
            resume_representatives = baseline._select_resume_representatives(model)
            resume_snapshots = baseline._snapshot_resume_representatives(
                resume_representatives
            )
        if args.mode == "smoke":
            if args.resume_from is None and cursor["global_step"] != 0:
                raise RuntimeError("initial smoke must start at step zero")
            if args.resume_from is not None and cursor != {"epoch": 0, "batch_in_epoch": 80, "global_step": 20}:
                raise RuntimeError("resume smoke requires checkpoint-000020 at the exact batch cursor")
            stop_step = SMOKE_TOTAL_STEPS if args.resume_from is not None else SMOKE_FIRST_STOP
        else:
            stop_step = shape["total_steps"]

        sampler = DistributedSampler(
            dataset, num_replicas=world, rank=rank, shuffle=True,
            seed=args.seed, drop_last=True,
        )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed + rank)
        loader = DataLoader(
            dataset, batch_size=args.micro_batch_size, sampler=sampler,
            shuffle=False, drop_last=True, num_workers=0,
            collate_fn=lambda rows: collate_reasonaqa(rows, tokenizer),
            generator=loader_generator,
        )
        if len(loader) < shape["microbatches_per_epoch"]:
            raise RuntimeError("DataLoader is shorter than the audited epoch budget")
        report.update({
            "store_inventory": inventory,
            "staging": {
                "store_copy_seconds": args.store_copy_seconds,
                "manifest_copy_seconds": args.manifest_copy_seconds,
                "total_seconds": args.staging_total_seconds,
                "excluded_from_optimizer_step_timing": True,
            },
            "rank_staging_audit": rank_staging_audit if rank == 0 else None,
            "dataset_rows": len(dataset),
            "dataset_audio_slot_audit": dataset_audio_slot_audit,
            "dataset_init_seconds_by_rank": _gather(dataset_init_seconds, world),
            "shape": shape,
            "epochs": args.epochs,
            "warmup_steps": args.warmup_steps,
            "warmup_default_ceil_5_percent": default_warmup,
            "max_lr": args.max_lr,
            "min_lr": args.min_lr,
            "compact_single_audio_prefix": False,
            "prefix_tokens": PREFIX_TOKENS,
            "single_audio_slot_semantics": AUDIO_SLOT_SEMANTICS,
            "initialization": {
                "kind": "original_smollm2_plus_random_audio_mapper" if args.resume_from is None else "full_state_resume",
                "fresh_source": str(args.model_path.resolve()),
                "resume_source": str(args.resume_from.resolve()) if args.resume_from else None,
            },
            "start_global_step": cursor["global_step"],
            "start_cursor": dict(cursor),
            "resume_checkpoint": str(args.resume_from.resolve()) if args.resume_from else None,
            "formal_gate": formal_gate,
            "sampler": {
                "kind": "DistributedSampler",
                "shuffle": True,
                "seed": args.seed,
                "drop_last": True,
                "entire_manifest_each_epoch": True,
                "rank_disjoint": True,
            },
        })
        first_gradient_audit = None
        answer_label_audit = None

        while cursor["global_step"] < stop_step:
            epoch = cursor["epoch"]
            if epoch >= args.epochs:
                raise RuntimeError("training exhausted epochs before reaching stop_step")
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            for _ in range(cursor["batch_in_epoch"]):
                next(iterator)
            while (cursor["batch_in_epoch"] < shape["microbatches_per_epoch"]
                   and cursor["global_step"] < stop_step):
                epoch_completed = False
                step_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), dtype=torch.float32, device=device)
                last_batch: dict[str, Any] | None = None
                for micro in range(args.gradient_accumulation_steps):
                    batch = next(iterator)
                    # Every tensor consumed by the baseline composite must
                    # share the rank device; fixed-260 mode concatenates text
                    # masks with the GPU audio prefix.
                    batch = {key: (value.to(device) if torch.is_tensor(value) else value)
                             for key, value in batch.items()}
                    last_batch = batch
                    synchronization = ddp.no_sync() if micro + 1 < args.gradient_accumulation_steps else contextlib.nullcontext()
                    with synchronization:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            output = ddp(**{key: value for key, value in batch.items()
                                          if key not in {"row_indices", "audio2_reused", "waveform_cache_shard_ids"}})
                        if output.loss is None or not bool(torch.isfinite(output.loss)):
                            raise RuntimeError("nonfinite shared-store training loss")
                        loss_sum.add_(output.loss.detach().float())
                        (output.loss / args.gradient_accumulation_steps).backward()
                owner = ddp.module
                if answer_label_audit is None:
                    if last_batch is None:
                        raise AssertionError("first optimizer step has no batch")
                    answer_label_audit = _answer_label_audit(owner, last_batch)
                if first_gradient_audit is None:
                    gradient = owner.runtime_gradient_audit()
                    trainable = owner.trainable_parameter_audit()
                    standard = trainable.get("standard_text_contract", {})
                    first_gradient_audit = {
                        **gradient,
                        "standard_30_layer_smollm2": (
                            standard.get("model_type") == "llama"
                            and standard.get("num_hidden_layers") == 30
                            and trainable.get("independent_decoder_layers") is True
                            and trainable.get("has_router_parameters") is False
                        ),
                        "all_bridge_gradients_finite": bool(gradient.get("bridge_gradients"))
                        and all(gradient["bridge_gradients"].values()),
                        "all_c2l_gradients_finite": bool(gradient.get("c2l_gradients"))
                        and all(gradient["c2l_gradients"].values()),
                        "training_mode_contract": trainable.get("training_mode_contract") is True,
                        "fixed260_audio_reuse_contract": True,
                    }
                    if not (
                        first_gradient_audit["standard_30_layer_smollm2"]
                        and first_gradient_audit["training_mode_contract"]
                    ):
                        raise RuntimeError(
                            f"SmolLM2 shared-store gradient audit failed: {first_gradient_audit}"
                        )
                if (
                    resume_representatives is not None
                    and resume_gradient_verification is None
                ):
                    resume_gradient_verification = (
                        baseline._verify_resume_representative_gradients(
                            resume_representatives
                        )
                    )
                grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 0.5, error_if_nonfinite=True)
                lr_used = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
                cursor["global_step"] += 1
                cursor["batch_in_epoch"] += args.gradient_accumulation_steps
                if cursor["batch_in_epoch"] == shape["microbatches_per_epoch"]:
                    cursor["epoch"] += 1
                    cursor["batch_in_epoch"] = 0
                    epoch_completed = True
                torch.cuda.synchronize(device)
                metric = {
                    "step": cursor["global_step"],
                    "epoch": cursor["epoch"],
                    "batch_in_epoch": cursor["batch_in_epoch"],
                    "loss": float((loss_sum / args.gradient_accumulation_steps).item()),
                    "lr": lr_used,
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "seconds": time.perf_counter() - step_started,
                }
                if rank == 0:
                    report["metrics"].append(metric)
                    if cursor["global_step"] % 10 == 0 or cursor["global_step"] == stop_step:
                        print(
                            f"[shared-store-train] step={cursor['global_step']}/{stop_step} "
                            f"epoch={cursor['epoch']} batch={cursor['batch_in_epoch']} "
                            f"loss={metric['loss']:.6f} lr={lr_used:.8g} seconds={metric['seconds']:.3f}",
                            flush=True,
                        )
                save = (
                    args.mode == "smoke" and cursor["global_step"] in {20, 22}
                ) or (
                    args.mode == "formal" and
                    (cursor["global_step"] % args.save_every == 0 or cursor["global_step"] == shape["total_steps"])
                )
                if save:
                    checkpoint = args.output_dir / f"checkpoint-{cursor['global_step']:06d}"
                    _save_checkpoint(
                        checkpoint, owner, tokenizer, optimizer, scheduler,
                        args, inventory, shape, dict(cursor), rank, world, device,
                    )
                    if rank == 0:
                        report["checkpoints"].append(str(checkpoint))
                        report["retained_checkpoints"] = baseline._prune_checkpoints(
                            args.output_dir, args.checkpoint_retention,
                        )
                    dist.barrier()
                if epoch_completed:
                    break

        if cursor["global_step"] != stop_step:
            raise RuntimeError(f"run ended at step {cursor['global_step']} rather than {stop_step}")
        if resume_representatives is not None and resume_snapshots is not None:
            resume_parameter_change_audit = (
                baseline._compute_resume_parameter_change_audit(
                    resume_representatives,
                    resume_snapshots,
                )
            )
            baseline._validate_parameter_change_audit(
                resume_parameter_change_audit
            )
        report.update({
            "status": "PASS",
            "end_global_step": cursor["global_step"],
            "end_cursor": cursor,
            "first_step_gradient_audit": first_gradient_audit,
            "model_trainable_audit": ddp.module.trainable_parameter_audit(),
            "answer_only_label_audit": answer_label_audit,
            "resume_verified_two_steps": (
                args.mode == "smoke" and args.resume_from is not None
                and report["start_global_step"] == 20 and cursor["global_step"] == 22
            ),
            "resume_representative_gradient_verification": resume_gradient_verification,
            "resume_parameter_change_audit": resume_parameter_change_audit,
            "architecture_audit": "standard 30-layer SmolLM2; no router, memory, recursion, or shared loop",
        })
        return report
    except Exception as exc:
        report["hard_failures"].append({"error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        if rank == 0 and output_available:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "shared_store_training_report.json").write_text(
                json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8",
            )
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run(parse_args())
