#!/usr/bin/env python3
"""Uniform-T=2..10 Audio MeSH training on one node-shared waveform store."""
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

import train_audio_5_10x2_5_mesh_mellow_ddp as base
from audio_5_10x2to10_5_mesh_mellow_shared_store import DEFAULT_EPOCHS, TRAINING_CONTRACT
from audio_5_10x2to10_5_mesh_mellow_shared_store.data import ReasonAQADataset, collate_reasonaqa
from audio_5_10x2to10_5_mesh_mellow_shared_store.depth_sampling import SynchronizedDepthSampler
from audio_5_10x2to10_5_mesh_mellow_shared_store.model import (
    ARCHITECTURE_CONTRACT,
    MAPPER_CONTRACT,
    AUDIO_PREFIX_TOKENS,
    AUDIO_DUAL_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    MESH_HIDDEN_SIZE,
    AudioMeshConfig,
    AudioMeshModel,
    RecursiveLlamaForCausalLM,
)
from recursive_model_5_10x2to10_5_mesh import build_mesh_schedule


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
FORMAL_EPOCHS = DEFAULT_EPOCHS
CANONICAL_MAX_LR = 1e-3
CANONICAL_MIN_LR = 1e-4
AUDIO_SLOT_SEMANTICS = (
    "fixed260_second_slot_reuses_audio1_htsat_embedding_then_runs_bridge_separately"
)
DEFAULT_INIT_ARTIFACT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "audio_5_10x2to10_5_mesh_mellow_shared_store/initialization/from_checkpoint_011343"
)
DEFAULT_STRUCTURE_REPORT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "audio_5_10x2to10_5_mesh_mellow_shared_store/preflight/structure_gradient_report.json"
)
DEFAULT_MEMORY_REPORT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "audio_5_10x2to10_5_mesh_mellow_shared_store/preflight/t10_activation_memory.json"
)
DEPTH_RANGE = tuple(range(2, 11))
DEPTH_SAMPLING_CONTRACT = "rank0_cpu_generator_uniform_integer_2_10_broadcast_each_microstep_v1"


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
    p.add_argument("--init-artifact", type=Path, default=DEFAULT_INIT_ARTIFACT)
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--smoke20-report", type=Path)
    p.add_argument("--smoke-resume-report", type=Path)
    p.add_argument("--tokenizer-path", type=Path)
    p.add_argument("--htsat-checkpoint", type=Path, default=Path(base.DEFAULT_HTSAT))
    p.add_argument("--mellow-root", type=Path, default=Path(base.DEFAULT_MELLOW))
    p.add_argument("--structure-report", type=Path, default=DEFAULT_STRUCTURE_REPORT)
    p.add_argument("--memory-report", type=Path, default=DEFAULT_MEMORY_REPORT)
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


_CLUSTER_STORAGE_ALIASES = (
    "/hpc_stor03/sjtu_home",
    "/mnt/cloudstorfs/sjtu_home",
)
_CLUSTER_STORAGE_CANONICAL_ROOT = "/cluster/sjtu_home"


def _normalize_cluster_storage_path(value: str) -> str:
    """Map the two cluster bind-mount spellings to one identity string."""
    normalized = value.replace("\\", "/").rstrip("/")
    for prefix in _CLUSTER_STORAGE_ALIASES:
        if normalized == prefix:
            return _CLUSTER_STORAGE_CANONICAL_ROOT
        if normalized.startswith(prefix + "/"):
            return _CLUSTER_STORAGE_CANONICAL_ROOT + normalized[len(prefix):]
    return normalized


def _path_identity_candidates(path: Path) -> set[str]:
    """Return existing-path identities without trusting only one bind-mount name."""
    expanded = path.expanduser()
    try:
        resolved = expanded.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return set()
    return {
        _normalize_cluster_storage_path(expanded.absolute().as_posix()),
        _normalize_cluster_storage_path(resolved.as_posix()),
    }


def _same_path(left: Path, right: Path) -> bool:
    left_candidates = _path_identity_candidates(left)
    right_candidates = _path_identity_candidates(right)
    return bool(left_candidates and right_candidates and left_candidates & right_candidates)


def _validate_phase_gates(args: argparse.Namespace) -> dict[str, Any]:
    structure_path = args.structure_report.resolve(strict=True)
    memory_path = args.memory_report.resolve(strict=True)
    structure = _json(structure_path)
    memory = _json(memory_path)
    depth_rows = structure.get("all_depths", [])
    gradient_rows = structure.get("gradient_sample_depths", [])
    if (structure.get("status") != "PASS"
            or [int(row.get("depth", -1)) for row in depth_rows] != list(DEPTH_RANGE)
            or any(row.get("status") != "PASS" for row in depth_rows)
            or [int(row.get("depth", -1)) for row in gradient_rows] != [2, 3, 6, 10]
            or any(not row.get("all_loop_boundaries_have_gradients") for row in gradient_rows)):
        raise RuntimeError("phase-3 structure/gradient report is not a complete PASS")
    artifact = structure.get("initialization_artifact") or {}
    if not artifact or not _same_path(Path(str(artifact.get("path", ""))), args.init_artifact):
        raise RuntimeError("phase-3 report does not belong to the requested initialization artifact")
    init_root = args.init_artifact.resolve(strict=True)
    expected_marker = _json(init_root / "artifact_complete.json")
    expected_migration = _json(init_root / "variable_depth_init_report.json")
    if (artifact.get("marker") != expected_marker
            or artifact.get("t2_exact_parity") != expected_migration.get("t2_exact_parity")):
        raise RuntimeError("phase-3 report initialization evidence differs from the requested artifact")
    if (memory.get("status") != "PASS"
            or int(memory.get("recursive_depth", -1)) != 10
            or int(memory.get("micro_batch_size", -1)) != 8
            or int(memory.get("gradient_accumulation_steps", -1)) != 4
            or int(memory.get("sequence_length", -1)) != 639
            or memory.get("find_unused_parameters") is not True
            or float(memory.get("peak_reserved_ratio", 1.0)) > float(memory.get("max_reserved_ratio", 0.90))):
        raise RuntimeError("phase-4 T=10 activation-memory report is not a formal-config PASS")
    provenance = memory.get("audio_provenance") or {}
    if not provenance:
        raise RuntimeError("phase-4 report lacks exact HTSAT/Mellow provenance")
    if not _same_path(Path(str(provenance.get("htsat_checkpoint", ""))), args.htsat_checkpoint):
        raise RuntimeError("phase-4 report HTSAT provenance differs from this run")
    if not _same_path(Path(str(provenance.get("mellow_root", ""))), args.mellow_root):
        raise RuntimeError("phase-4 report Mellow provenance differs from this run")
    return {
        "structure_report": str(structure_path),
        "structure_report_sha256": _sha(structure_path),
        "memory_report": str(memory_path),
        "memory_report_sha256": _sha(memory_path),
        "memory_peak_reserved_ratio": float(memory["peak_reserved_ratio"]),
    }


def _validate_init_artifact(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    root = path.resolve(strict=True)
    required = (
        root / "mesh_model" / "config.json", root / "tokenizer" / "tokenizer_config.json",
        root / "audio_bridge.pt", root / "variable_depth_init_report.json",
        root / "artifact_complete.json",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise RuntimeError(f"variable-depth initialization artifact is incomplete: {missing}")
    marker = _json(root / "artifact_complete.json")
    report = _json(root / "variable_depth_init_report.json")
    if (marker.get("status") != "complete"
            or report.get("status") != "PASS"
            or report.get("t2_exact_parity", {}).get("status") != "PASS"
            or float(report["t2_exact_parity"].get("max_abs_diff", -1.0)) != 0.0):
        raise RuntimeError("variable-depth initialization artifact lacks exact T=2 parity proof")
    source_checkpoint = Path(str(report.get("source", {}).get("source", "")))
    source_config_path = source_checkpoint / "audio_mesh_config.json"
    source_config = _json(source_config_path.resolve(strict=True))
    if not _same_path(Path(str(source_config.get("htsat_checkpoint", ""))), args.htsat_checkpoint):
        raise RuntimeError("initialization artifact HTSAT provenance differs from this run")
    if not _same_path(Path(str(source_config.get("mellow_root", ""))), args.mellow_root):
        raise RuntimeError("initialization artifact Mellow provenance differs from this run")
    return {
        "path": str(root),
        "artifact_complete_sha256": _sha(root / "artifact_complete.json"),
        "migration_report_sha256": _sha(root / "variable_depth_init_report.json"),
        "source_checkpoint": str(source_checkpoint.resolve(strict=True)),
        "source_audio_config": str(source_config_path.resolve(strict=True)),
        "t2_exact_parity": report["t2_exact_parity"],
        "training_state_loaded": False,
    }


def _load_model(args: argparse.Namespace, device: torch.device) -> tuple[AudioMeshModel, Any]:
    from transformers import AutoTokenizer

    fresh_audit = _validate_init_artifact(args.init_artifact, args)
    source = args.resume_from.resolve(strict=True) if args.resume_from else args.init_artifact.resolve(strict=True)
    mesh = RecursiveLlamaForCausalLM.from_pretrained(
        source / "mesh_model", local_files_only=True, torch_dtype=torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(source / "tokenizer", local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper, htsat, provenance = base._load_mellow_wrapper(
        args.mellow_root, args.htsat_checkpoint, device,
    )
    model = AudioMeshModel(
        mesh, tokenizer, wrapper, htsat,
        AudioMeshConfig(compact_single_audio_prefix=False),
    ).to(device)
    audio_state = torch.load(source / "audio_bridge.pt", map_location=device, weights_only=False)
    if not isinstance(audio_state.get("bridge"), dict) or not isinstance(audio_state.get("c2l"), dict):
        raise RuntimeError("composite source lacks bridge/c2l state")
    model.bridge.load_state_dict(audio_state["bridge"], strict=True)
    model.htsat_wrapper.c2l.load_state_dict(audio_state["c2l"], strict=True)
    model._audio_provenance = provenance
    model._initialization_audit = fresh_audit
    return model, tokenizer


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
    for label, report, start, end, expected_phase in (
        ("smoke20", first, 0, 20, "phase6_smoke20"),
        ("resume2", resumed, 20, 22, "phase7_exact_resume2"),
    ):
        if (report.get("status") != "PASS" or report.get("mode") != "smoke"
                or report.get("phase") != expected_phase
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
                or report.get("initialization", {}).get("fresh_source") != str(args.init_artifact.resolve())
                or report.get("depth_sampling", {}).get("contract") != DEPTH_SAMPLING_CONTRACT
                or report.get("depth_sampling", {}).get("range") != list(DEPTH_RANGE)
                or report.get("depth_sampling", {}).get("ddp_sync_each_micro_step") is not True
                or report.get("phase_gates") != args._phase_gates):
            raise RuntimeError(f"formal gate rejects {label} report")
        report_inventory = report.get("store_inventory", {})
        for key in ("manifest_sha256", "index_sha256", "waveform_sha256", "total_waveform_bytes"):
            if report_inventory.get(key) != inventory.get(key):
                raise RuntimeError(f"formal gate {label} store differs in {key}")
        if not report.get("first_step_gradient_audit", {}).get("trace_matches_sampled_depth"):
            raise RuntimeError(f"formal gate rejects {label} gradient trace")
        if report.get("answer_only_label_audit", {}).get("passed") is not True:
            raise RuntimeError(f"formal gate rejects {label} label audit")
        if report.get("answer_only_label_audit", {}).get("terminal_eos_supervised") is not True:
            raise RuntimeError(f"formal gate rejects {label} terminal EOS audit")
        depth_audit = report.get("depth_sampling", {}).get("rank_consistency_audit")
        if not isinstance(depth_audit, list) or len(depth_audit) != args.world_size:
            raise RuntimeError(f"formal gate rejects {label} depth-rank coverage")
        depth_identities = {
            (int(row.get("draws", -1)), json.dumps(row.get("histogram", {}), sort_keys=True))
            for row in depth_audit
        }
        if len(depth_identities) != 1:
            raise RuntimeError(f"formal gate rejects {label} divergent depth samplers")
    first_checkpoint = str(Path(first["checkpoints"][-1]).resolve())
    if resumed.get("resume_checkpoint") != first_checkpoint or resumed.get("resume_verified_two_steps") is not True:
        raise RuntimeError("resume smoke does not prove exact continuation from checkpoint-000020")
    if int(first.get("depth_sampling", {}).get("draws", -1)) != 20 * args.gradient_accumulation_steps:
        raise RuntimeError("smoke20 depth-sampler draw cursor is not exactly 80")
    if int(resumed.get("depth_sampling", {}).get("start_draws", -1)) != 20 * args.gradient_accumulation_steps:
        raise RuntimeError("resume2 did not restore the depth-sampler cursor at draw 80")
    if int(resumed.get("depth_sampling", {}).get("draws", -1)) != 22 * args.gradient_accumulation_steps:
        raise RuntimeError("resume2 depth-sampler draw cursor is not exactly 88")
    return {
        "smoke20_report": str(args.smoke20_report.resolve()),
        "smoke_resume_report": str(args.smoke_resume_report.resolve()),
        "checkpoint20": first_checkpoint,
    }


def _checkpoint_config(args: argparse.Namespace, inventory: dict[str, Any], shape: dict[str, int],
                       provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": TRAINING_CONTRACT,
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "mapper_initialization": "checkpoint_011343_trained_c2l_and_bridge",
        "init_artifact": str(args.init_artifact.resolve()),
        "initialization_audit": getattr(args, "_initialization_audit", None),
        "phase_gates": getattr(args, "_phase_gates", None),
        "recursive_depth_sampling": {
            "contract": DEPTH_SAMPLING_CONTRACT,
            "distribution": "discrete_uniform",
            "values": list(DEPTH_RANGE),
            "sampling_unit": "micro_step",
            "rank0_sample_then_broadcast": True,
            "full_backpropagation": True,
            "ddp_sync_each_micro_step": True,
        },
        "compact_single_audio_prefix": False,
        "single_audio_slot_semantics": AUDIO_SLOT_SEMANTICS,
        "answer_termination": ANSWER_TERMINATION,
        "prefix_tokens": PREFIX_TOKENS,
        "mesh_hidden_size": MESH_HIDDEN_SIZE,
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


def _save_checkpoint(path: Path, model: Any, tokenizer: Any, optimizer: Any, scheduler: Any,
                     depth_sampler: SynchronizedDepthSampler,
                     args: argparse.Namespace, inventory: dict[str, Any], shape: dict[str, int],
                     cursor: dict[str, int], rank: int, world: int, device: torch.device) -> None:
    rng_states = _gather(base._rng_state(device), world)
    local_depth_summary = {
        "rank": rank,
        "draws": int(depth_sampler.draws),
        "histogram": {str(k): int(v) for k, v in sorted(depth_sampler.histogram.items())},
    }
    depth_summaries = _gather(local_depth_summary, world)
    depth_identities = {
        (int(row["draws"]), json.dumps(row["histogram"], sort_keys=True))
        for row in depth_summaries
    }
    if len(depth_identities) != 1:
        raise RuntimeError(f"recursive-depth sampler diverged across ranks: {depth_summaries}")
    if rank != 0:
        return
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent))
    published = False
    try:
        model.mesh_model.save_pretrained(temporary / "mesh_model", safe_serialization=False)
        tokenizer.save_pretrained(temporary / "tokenizer")
        torch.save(base._trainable_state(model), temporary / "audio_bridge.pt")
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": cursor["global_step"],
            "cursor": cursor,
            "rng_states_by_rank": {str(i): state for i, state in enumerate(rng_states)},
            "depth_sampler": depth_sampler.state_dict(),
            "depth_sampler_rank_summaries": depth_summaries,
        }, temporary / "training_state.pt")
        config = _checkpoint_config(args, inventory, shape, model._audio_provenance)
        config.update({
            "global_step": cursor["global_step"],
            "epoch": cursor["epoch"],
            "batch_in_epoch": cursor["batch_in_epoch"],
        })
        (temporary / "audio_mesh_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        marker = {
            "status": "complete",
            "global_step": cursor["global_step"],
            "contract": TRAINING_CONTRACT,
            "required": ["mesh_model", "tokenizer", "audio_bridge.pt", "training_state.pt", "audio_mesh_config.json"],
        }
        (temporary / "checkpoint_complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        required = (
            temporary / "mesh_model" / "config.json",
            temporary / "tokenizer" / "tokenizer_config.json",
            temporary / "audio_bridge.pt",
            temporary / "training_state.pt",
            temporary / "audio_mesh_config.json",
            temporary / "checkpoint_complete.json",
        )
        if any(not item.is_file() for item in required):
            raise RuntimeError("refusing to publish incomplete shared-store checkpoint")
        temporary.replace(path)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def _resume(path: Path, args: argparse.Namespace, inventory: dict[str, Any], shape: dict[str, int],
            optimizer: Any, scheduler: Any, depth_sampler: SynchronizedDepthSampler,
            provenance: dict[str, Any], rank: int,
            device: torch.device) -> dict[str, int]:
    path = path.resolve(strict=True)
    config = _json(path / "audio_mesh_config.json")
    marker = _json(path / "checkpoint_complete.json")
    if (config.get("contract") != TRAINING_CONTRACT or marker.get("contract") != TRAINING_CONTRACT
            or marker.get("status") != "complete"):
        raise RuntimeError("resume source is not a complete shared-store-v2 checkpoint")
    expected = _checkpoint_config(args, inventory, shape, provenance)
    for key in (
        "contract", "architecture_contract", "mapper_contract", "mapper_initialization",
        "init_artifact", "initialization_audit", "phase_gates", "recursive_depth_sampling",
        "compact_single_audio_prefix", "mesh_hidden_size",
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
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if "depth_sampler" not in state:
        raise RuntimeError("resume checkpoint lacks recursive-depth sampler state")
    summaries = state.get("depth_sampler_rank_summaries")
    if not isinstance(summaries, list) or len(summaries) != args.world_size:
        raise RuntimeError("resume checkpoint lacks complete depth-sampler rank coverage")
    summary_identities = {
        (int(row["draws"]), json.dumps(row["histogram"], sort_keys=True))
        for row in summaries
    }
    if len(summary_identities) != 1:
        raise RuntimeError("resume checkpoint records divergent depth samplers")
    depth_sampler.load_state_dict(state["depth_sampler"])
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
    expected_draws = cursor["global_step"] * args.gradient_accumulation_steps
    if depth_sampler.draws != expected_draws:
        raise RuntimeError(
            f"resume depth-sampler cursor mismatch: draws={depth_sampler.draws} expected={expected_draws}"
        )
    base._restore_rng_state(rng[str(rank)], device)
    return cursor


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
    }


def _variable_gradient_audit(model: AudioMeshModel, micro_depths: list[int]) -> dict[str, Any]:
    owner = model.mesh_model.model
    if not micro_depths:
        raise RuntimeError("first optimizer step has no sampled recursive depths")
    last_depth = int(micro_depths[-1])
    trace = [int(item["physical_index"]) for item in owner.last_forward_trace]
    expected_trace = list(build_mesh_schedule(last_depth))
    physical_gradients = {
        str(index): bool(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in layer.parameters()
        ))
        for index, layer in enumerate(owner.layers)
    }
    required_routers = [
        "pre_write", "pre_read", "loop1_write", "loop1_read", "refine_write", "out_read",
    ]
    if any(depth > 2 for depth in micro_depths):
        required_routers.append("refine_read")
    router_gradients = {
        name: bool(
            getattr(owner, name).weight.grad is not None
            and torch.isfinite(getattr(owner, name).weight.grad).all()
        )
        for name in required_routers
    }
    boundary_refs = owner.last_core_input_refs + owner.last_core_output_refs
    boundary_gradients = bool(boundary_refs) and all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in boundary_refs
    )
    passed = (
        trace == expected_trace and all(physical_gradients.values())
        and all(router_gradients.values()) and boundary_gradients
    )
    if not passed:
        raise RuntimeError(
            "variable-depth first-step gradient audit failed: "
            f"depths={micro_depths} trace_match={trace == expected_trace} "
            f"routers={router_gradients} layers={physical_gradients} "
            f"boundaries={boundary_gradients}"
        )
    return {
        "passed": True,
        "micro_step_depths": list(micro_depths),
        "last_micro_step_depth": last_depth,
        "trace_matches_sampled_depth": True,
        "logical_layer_count": len(trace),
        "physical_trace": trace,
        "router_gradients": router_gradients,
        "physical_layer_gradients": physical_gradients,
        "all_loop_boundaries_have_gradients": boundary_gradients,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
    report: dict[str, Any] = {
        "status": "FAIL",
        "mode": args.mode,
        "phase": (
            "phase8_formal_7epochs" if args.mode == "formal" else
            "phase7_exact_resume2" if args.resume_from is not None else
            "phase6_smoke20"
        ),
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
        base._seed(args.seed, rank)

        args._initialization_audit = _validate_init_artifact(args.init_artifact, args)
        args._phase_gates = _validate_phase_gates(args)
        inventory = _store_inventory(args)
        rank_staging_audit = _staging_rank_audit(inventory, rank, world)
        # The shared-store route changes only waveform residency and delivery.
        # Keep the historical fixed-260 two-slot model contract: a structural
        # single-audio row reuses audio1's HTSAT embedding, then both slots run
        # through the trainable bridge independently.
        args.compact_single_audio_prefix = False
        dataset_started = time.perf_counter()
        # This constructor re-hashes the staged manifest, verifies it against
        # metadata, verifies index SHA256, and builds the path-to-audio map.
        dataset = ReasonAQADataset(
            args.train_manifest,
            tokenizer=None,
            unique_waveform_store_dir=args.unique_waveform_store_dir,
        )
        dataset_init_seconds = time.perf_counter() - dataset_started
        shape = _training_shape(args, len(dataset))
        default_warmup = math.ceil(shape["total_steps"] * 0.05)
        args.warmup_steps = default_warmup if args.warmup_steps is None else int(args.warmup_steps)
        if args.warmup_steps != default_warmup:
            raise ValueError(f"warmup_steps must equal ceil(total_steps * 0.05)={default_warmup}")
        formal_gate = _formal_gate(args, inventory, shape)

        model, tokenizer = _load_model(args, device)
        dataset.tokenizer = tokenizer
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
            warmup_steps=args.warmup_steps, total_steps=shape["total_steps"],
        )
        # Construct DDP before restoring RNG so process-group/module setup
        # cannot perturb the exact per-rank continuation state.
        depth_sampler = SynchronizedDepthSampler(seed=args.seed)
        ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
        cursor = {"epoch": 0, "batch_in_epoch": 0, "global_step": 0}
        start_depth_draws = 0
        if args.resume_from is not None:
            cursor = _resume(
                args.resume_from, args, inventory, shape, optimizer, scheduler, depth_sampler,
                model._audio_provenance, rank, device,
            )
            start_depth_draws = depth_sampler.draws
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
            "phase_gates": args._phase_gates,
            "initialization": {
                "kind": "migrated_checkpoint_011343" if args.resume_from is None else "full_state_resume",
                "fresh_source": str(args.init_artifact.resolve()),
                "audit": args._initialization_audit,
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
            "depth_sampling": {
                "contract": DEPTH_SAMPLING_CONTRACT,
                "range": list(DEPTH_RANGE),
                "unit": "micro_step",
                "rank0_sample_then_broadcast": True,
                "full_backpropagation": True,
                "ddp_sync_each_micro_step": True,
                "start_draws": start_depth_draws,
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
                step_depths: list[int] = []
                for micro in range(args.gradient_accumulation_steps):
                    batch = next(iterator)
                    # Every tensor consumed by AudioMeshModel must share the
                    # rank device; fixed-260 mode concatenates text masks with
                    # the GPU audio prefix.
                    batch = {key: (value.to(device) if torch.is_tensor(value) else value)
                             for key, value in batch.items()}
                    recursive_depth = depth_sampler.sample(device)
                    step_depths.append(recursive_depth)
                    last_batch = batch
                    # T=2 intentionally leaves refine_read unused while T>2 uses it.
                    # DDP must therefore reduce every micro-step independently; a
                    # no_sync accumulation window whose last micro-step is T=2 can
                    # otherwise leave earlier refine_read gradients rank-local.
                    with contextlib.nullcontext():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            output = ddp(recursive_depth=recursive_depth, **{key: value for key, value in batch.items()
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
                    first_gradient_audit = _variable_gradient_audit(owner, step_depths)
                    owner.mesh_model.model.gradient_audit_mode = False
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
                    "recursive_depths": list(step_depths),
                    "seconds": time.perf_counter() - step_started,
                }
                if rank == 0:
                    report["metrics"].append(metric)
                    if cursor["global_step"] % 10 == 0 or cursor["global_step"] == stop_step:
                        print(
                            f"[shared-store-train] step={cursor['global_step']}/{stop_step} "
                            f"epoch={cursor['epoch']} batch={cursor['batch_in_epoch']} "
                            f"depths={step_depths} loss={metric['loss']:.6f} "
                            f"lr={lr_used:.8g} seconds={metric['seconds']:.3f}",
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
                        depth_sampler,
                        args, inventory, shape, dict(cursor), rank, world, device,
                    )
                    if rank == 0:
                        report["checkpoints"].append(str(checkpoint))
                        report["retained_checkpoints"] = base._prune_checkpoints(
                            args.output_dir, args.checkpoint_retention,
                        )
                    dist.barrier()
                if epoch_completed:
                    break

        if cursor["global_step"] != stop_step:
            raise RuntimeError(f"run ended at step {cursor['global_step']} rather than {stop_step}")
        local_depth_summary = {
            "rank": rank,
            "draws": int(depth_sampler.draws),
            "histogram": {str(k): int(v) for k, v in sorted(depth_sampler.histogram.items())},
        }
        depth_rank_audit = _gather(local_depth_summary, world)
        depth_identities = {
            (int(row["draws"]), json.dumps(row["histogram"], sort_keys=True))
            for row in depth_rank_audit
        }
        if len(depth_identities) != 1:
            raise RuntimeError(f"recursive-depth sampler diverged across ranks: {depth_rank_audit}")
        report.update({
            "status": "PASS",
            "end_global_step": cursor["global_step"],
            "end_cursor": cursor,
            "first_step_gradient_audit": first_gradient_audit,
            "answer_only_label_audit": answer_label_audit,
            "depth_sampling": {
                **report["depth_sampling"],
                "draws": depth_sampler.draws,
                "histogram": {str(key): value for key, value in sorted(depth_sampler.histogram.items())},
                "rank_consistency_audit": depth_rank_audit if rank == 0 else None,
            },
            "resume_verified_two_steps": (
                args.mode == "smoke" and args.resume_from is not None
                and report["start_global_step"] == 20 and cursor["global_step"] == 22
            ),
            "routing_stats": "per-forward statistics disabled; sampled-depth trace and gradient audit retained",
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
