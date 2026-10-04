#!/usr/bin/env python3
"""Isolated full-shuffle training on one node-shared preprocessed audio store."""
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
from audio_5_10x5_5_mesh_8slot_mellow_shared_store_configurable_epochs import TRAINING_CONTRACT
from audio_5_10x5_5_mesh_8slot_mellow_shared_store_configurable_epochs.data import (
    ReasonAQADataset,
    collate_reasonaqa,
)
from audio_5_10x5_5_mesh_8slot_mellow_shared_store_configurable_epochs.model import (
    ARCHITECTURE_CONTRACT,
    MAPPER_CONTRACT,
    AUDIO_PREFIX_TOKENS,
    AUDIO_DUAL_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    MESH_HIDDEN_SIZE,
    AudioMeshX5EightSlotZeroConfig,
    AudioMeshX5EightSlotZeroModel,
    _load_mellow_wrapper,
)
from recursive_model_5_10x5_5_mesh_8slot import (
    LOGICAL_LAYER_COUNT,
    PHYSICAL_LAYER_COUNT,
    RECURSIVE_LOOPS,
    MEMORY_SLOT_COUNT,
    ROUTER_COUNT,
    ROUTER_PARAMETER_COUNT,
    LOGICAL_TO_PHYSICAL,
    MODEL_ARCHITECTURE_CONTRACT as TEXT_MODEL_ARCHITECTURE_CONTRACT,
    RecursiveLlamaForCausalLM,
    register_auto_class as register_x5_8slot_auto_class,
)


DEFAULT_MANIFEST = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/"
    "preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl"
)
DEFAULT_STORE = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/"
    "rsmol_reasonaqa_train_unique_waveforms_32k_10s_f32_v3"
)
DEFAULT_X5_8SLOT_TEXT_MODEL = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "stage4_5_10x5_5_mesh_8slot/formal_third_epoch_3081steps_20261003_222648_3090_v1/"
    "checkpoint-003081"
)
CHECKPOINT_CONFIG_FILENAME = "audio_mesh_x5_8slot_fixed260_zero_slot_config.json"
ANSWER_TERMINATION = {
    "token": "<|endoftext|>",
    "included_in_max_answer_tokens": True,
    "supervised": True,
}
PREFIX_TOKENS = {"single": AUDIO_DUAL_PREFIX_TOKENS, "dual": AUDIO_DUAL_PREFIX_TOKENS}
FORMAL_EPOCHS = 3
CANONICAL_MAX_LR = 1e-3
CANONICAL_MIN_LR = 1e-4
AUDIO_SLOT_SEMANTICS = (
    "fixed260_structural_single_uses_runtime_zero_waveform_second_slot_"
    "dual_uses_two_real_audio_slots"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("formal",), required=True)
    p.add_argument("--train-manifest", type=Path, required=True,
                   help="Staged manifest under /dev/shm")
    p.add_argument("--unique-waveform-store-dir", type=Path, required=True,
                   help="Staged complete unique store under /dev/shm")
    p.add_argument("--persistent-manifest-source", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--persistent-store-source", type=Path, default=DEFAULT_STORE)
    p.add_argument("--store-copy-seconds", type=float, required=True)
    p.add_argument("--manifest-copy-seconds", type=float, required=True)
    p.add_argument("--staging-total-seconds", type=float, required=True)
    p.add_argument("--model-path", "--mesh-checkpoint", dest="model_path",
                   type=Path, default=DEFAULT_X5_8SLOT_TEXT_MODEL)
    p.add_argument("--tokenizer-path", type=Path)
    p.add_argument("--htsat-checkpoint", type=Path, default=Path(base.DEFAULT_HTSAT))
    p.add_argument("--mellow-root", type=Path, default=Path(base.DEFAULT_MELLOW))
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


def _formal_audit_metadata(args: argparse.Namespace, inventory: dict[str, Any],
                           shape: dict[str, int]) -> dict[str, Any]:
    return {
        "required": True,
        "smoke_resume_gate": False,
        "reason": "formal-only x5/8-slot route; runtime audits run in the formal job",
        "source_checkpoint": str(args.model_path.resolve()),
        "shape": shape,
    }


def _checkpoint_config(args: argparse.Namespace, inventory: dict[str, Any], shape: dict[str, int],
                       provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": TRAINING_CONTRACT,
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "text_model_architecture_contract": TEXT_MODEL_ARCHITECTURE_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "mapper_initialization": "random_c2l_and_xavier_projection",
        "mesh_model_path": str(args.model_path.resolve()),
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
                     args: argparse.Namespace, inventory: dict[str, Any], shape: dict[str, int],
                     cursor: dict[str, int], rank: int, world: int, device: torch.device) -> None:
    rng_states = _gather(base._rng_state(device), world)
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
        }, temporary / "training_state.pt")
        config = _checkpoint_config(args, inventory, shape, model._audio_provenance)
        config.update({
            "global_step": cursor["global_step"],
            "epoch": cursor["epoch"],
            "batch_in_epoch": cursor["batch_in_epoch"],
        })
        (temporary / CHECKPOINT_CONFIG_FILENAME).write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        marker = {
            "status": "complete",
            "global_step": cursor["global_step"],
            "contract": TRAINING_CONTRACT,
            "required": [
                "mesh_model", "tokenizer", "audio_bridge.pt", "training_state.pt",
                CHECKPOINT_CONFIG_FILENAME,
            ],
        }
        (temporary / "checkpoint_complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        required = (
            temporary / "mesh_model" / "config.json",
            temporary / "tokenizer" / "tokenizer_config.json",
            temporary / "audio_bridge.pt",
            temporary / "training_state.pt",
            temporary / CHECKPOINT_CONFIG_FILENAME,
            temporary / "checkpoint_complete.json",
        )
        if any(not item.is_file() for item in required):
            raise RuntimeError("refusing to publish incomplete shared-store checkpoint")
        temporary.replace(path)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


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


def _validate_x5_8slot_text_checkpoint(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    required = (
        "config.json",
        "training_state.pt",
        "checkpoint_complete.json",
        "checkpoint_manifest.json",
        "mesh_checkpoint_metadata.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"x5/8-slot text initialization checkpoint is incomplete: {missing}")
    marker = _json(path / "checkpoint_complete.json")
    manifest = _json(path / "checkpoint_manifest.json")
    metadata = _json(path / "mesh_checkpoint_metadata.json")
    model_config = _json(path / "config.json")
    if (marker.get("status") != "complete" or marker.get("complete_marker") is not True
            or manifest.get("status") != "complete"):
        raise RuntimeError("x5/8-slot text initialization completion/manifest marker is invalid")
    listed_missing = [
        name for name in manifest.get("files", [])
        if not (path / str(name)).is_file()
    ]
    if listed_missing:
        raise RuntimeError(f"x5/8-slot text initialization manifest lists missing files: {listed_missing}")
    for label, payload in (("marker", marker), ("manifest", manifest), ("metadata", metadata)):
        if payload.get("architecture_contract") != TEXT_MODEL_ARCHITECTURE_CONTRACT:
            raise RuntimeError(f"x5/8-slot text initialization {label} architecture mismatch")
    if metadata.get("router_parameters_in_optimizer") is not True:
        raise RuntimeError("x5/8-slot text checkpoint does not prove router optimization")
    if int(metadata.get("memory_slots", -1)) != MEMORY_SLOT_COUNT:
        raise RuntimeError("x5/8-slot text checkpoint does not use eight memory slots")
    if int(metadata.get("router_groups", -1)) != ROUTER_COUNT:
        raise RuntimeError("x5/8-slot text checkpoint does not use six router groups")
    if int(metadata.get("router_module_count", -1)) != ROUTER_PARAMETER_COUNT:
        raise RuntimeError("x5/8-slot text checkpoint does not use twelve router modules")

    # The x5/8-slot text trainer's metadata intentionally stores the canonical
    # schedule rather than duplicating logical/physical/loop counts.  Derive
    # those counts from that schedule when the optional summary fields are
    # absent, while still rejecting any non-canonical schedule.
    schedule_value = metadata.get("logical_to_physical")
    if not isinstance(schedule_value, list):
        raise RuntimeError("x5/8-slot text checkpoint lacks logical_to_physical schedule")
    try:
        logical_to_physical = tuple(int(value) for value in schedule_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("x5/8-slot text checkpoint has an invalid logical_to_physical schedule") from exc
    if logical_to_physical != tuple(LOGICAL_TO_PHYSICAL):
        raise RuntimeError("x5/8-slot text checkpoint logical_to_physical schedule mismatch")
    config_contract = {
        "num_hidden_layers": LOGICAL_LAYER_COUNT,
        "recursive_layer_count": PHYSICAL_LAYER_COUNT,
        "recursive_loops": RECURSIVE_LOOPS,
        "logical_to_physical": list(LOGICAL_TO_PHYSICAL),
    }
    config_mismatch = {
        key: (expected, model_config.get(key))
        for key, expected in config_contract.items()
        if model_config.get(key) != expected
    }
    if config_mismatch:
        raise RuntimeError(f"x5/8-slot text config contract mismatch: {config_mismatch}")
    logical_layer_count = int(metadata.get("logical_layer_count", len(logical_to_physical)))
    physical_layer_count = int(metadata.get("physical_layer_count", max(logical_to_physical) + 1))
    recursive_loops = int(metadata.get("recursive_loops", RECURSIVE_LOOPS))
    if logical_layer_count != LOGICAL_LAYER_COUNT:
        raise RuntimeError("x5/8-slot text checkpoint does not use sixty logical layers")
    if physical_layer_count != PHYSICAL_LAYER_COUNT:
        raise RuntimeError("x5/8-slot text checkpoint does not use twenty physical layers")
    if recursive_loops != RECURSIVE_LOOPS:
        raise RuntimeError("x5/8-slot text checkpoint does not use five recursive loops")
    return {
        "passed": True,
        "path": str(path),
        "architecture_contract": TEXT_MODEL_ARCHITECTURE_CONTRACT,
        "optimizer_step": int(metadata.get("optimizer_step", -1)),
        "training_state_loaded": False,
        "optimizer_scheduler_rng_loaded": False,
        "memory_slots": int(metadata["memory_slots"]),
        "router_groups": int(metadata["router_groups"]),
        "router_module_count": int(metadata["router_module_count"]),
        "logical_layer_count": logical_layer_count,
        "physical_layer_count": physical_layer_count,
        "recursive_loops": recursive_loops,
        "logical_to_physical": list(logical_to_physical),
        "derived_optional_fields": [
            key for key in ("logical_layer_count", "physical_layer_count", "recursive_loops")
            if key not in metadata
        ],
        "fresh_audio_global_step": 0,
    }


def _load_x5_8slot_model(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    register_x5_8slot_auto_class()
    from transformers import AutoTokenizer

    initialization_audit = _validate_x5_8slot_text_checkpoint(args.model_path)
    model_path = args.model_path.expanduser().resolve(strict=True)
    tokenizer_path = (args.tokenizer_path or model_path).expanduser().resolve(strict=True)

    mesh = RecursiveLlamaForCausalLM.from_pretrained(model_path, local_files_only=True)
    if int(mesh.config.hidden_size) != MESH_HIDDEN_SIZE:
        raise RuntimeError("x5/8-slot text checkpoint hidden size differs from mapper contract")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper, htsat, provenance = _load_mellow_wrapper(
        args.mellow_root, args.htsat_checkpoint, device
    )
    model = AudioMeshX5EightSlotZeroModel(
        mesh.to(device),
        tokenizer,
        wrapper,
        htsat,
        config=AudioMeshX5EightSlotZeroConfig(),
    )
    model._audio_provenance = provenance
    model._initialization_audit = initialization_audit
    return model.to(device), tokenizer


def _finite_gradients(module: torch.nn.Module, *, require_nonzero: bool = True) -> bool:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    if not gradients or any(value is None for value in gradients):
        return False
    finite = all(bool(torch.isfinite(value).all()) for value in gradients if value is not None)
    nonzero = any(bool(torch.count_nonzero(value).item()) for value in gradients if value is not None)
    return finite and (nonzero if require_nonzero else True)


def _mesh_runtime_gradient_audit_x5_8slot(model: Any) -> dict[str, Any]:
    mesh_owner = model.mesh_model.model
    trace = list(getattr(mesh_owner, "last_forward_trace", []))
    expected_trace = [
        *({"logical_index": i, "physical_index": i} for i in range(5)),
        *(
            {"logical_index": 5 + loop * 10 + offset, "physical_index": 5 + offset}
            for loop in range(5)
            for offset in range(10)
        ),
        *({"logical_index": 55 + offset, "physical_index": 15 + offset} for offset in range(5)),
    ]
    input_refs = list(getattr(mesh_owner, "last_core_input_refs", []))
    output_refs = list(getattr(mesh_owner, "last_core_output_refs", []))
    input_gradients = [
        ref.grad is not None and bool(torch.isfinite(ref.grad).all()) for ref in input_refs
    ]
    output_gradients = [
        ref.grad is not None and bool(torch.isfinite(ref.grad).all()) for ref in output_refs
    ]
    router_gradients = {
        f"{group}_{index}": _finite_gradients(router, require_nonzero=False)
        for group, routers in (("write", mesh_owner.write_routers), ("read", mesh_owner.read_routers))
        for index, router in enumerate(routers)
    }
    middle_gradients = [
        _finite_gradients(layer, require_nonzero=False) for layer in mesh_owner.layers[5:15]
    ]
    result = {
        "trace_length": len(trace),
        "expected_trace_length": 60,
        "trace_matches_5_10x5_5_mesh_8slot": trace == expected_trace,
        "five_middle_loop_inputs_have_finite_gradients": len(input_refs) == 5 and all(input_gradients),
        "five_middle_loop_outputs_have_finite_gradients": len(output_refs) == 5 and all(output_gradients),
        "router_gradient_entries": len(router_gradients),
        "all_twelve_router_modules_have_finite_gradients": len(router_gradients) == 12 and all(router_gradients.values()),
        "router_finite_gradients": router_gradients,
        "all_middle_core_layers_have_finite_gradients": all(middle_gradients),
        "bridge_has_finite_nonzero_gradients": _finite_gradients(model.bridge),
        "c2l_has_finite_nonzero_gradients": _finite_gradients(model.htsat_wrapper.c2l),
        "htsat_backbone_has_no_gradients": all(
            parameter.grad is None for parameter in model.htsat_backbone.parameters()
        ),
        "memory_slots": int(mesh_owner.memory_slots),
        "write_router_groups": len(mesh_owner.write_routers),
        "read_router_groups": len(mesh_owner.read_routers),
    }
    required = (
        result["trace_matches_5_10x5_5_mesh_8slot"],
        result["five_middle_loop_inputs_have_finite_gradients"],
        result["five_middle_loop_outputs_have_finite_gradients"],
        result["all_twelve_router_modules_have_finite_gradients"],
        result["all_middle_core_layers_have_finite_gradients"],
        result["bridge_has_finite_nonzero_gradients"],
        result["c2l_has_finite_nonzero_gradients"],
        result["htsat_backbone_has_no_gradients"],
        result["memory_slots"] == 8,
        result["write_router_groups"] == 6,
        result["read_router_groups"] == 6,
    )
    if not all(required):
        raise RuntimeError(f"x5/8-slot audio runtime gradient/trace audit failed: {result}")
    return result


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
        if args.mode != "formal":
            raise ValueError("this isolated x5/8-slot deliverable supports formal training only")
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

        inventory = _store_inventory(args)
        rank_staging_audit = _staging_rank_audit(inventory, rank, world)
        # Reuse the audited shared-store residency and full-manifest shuffle.
        # Structurally single-audio rows receive one exact runtime zero waveform
        # in slot two; explicit dual-audio rows retain both real inputs.
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
        shape = _training_shape(args, len(dataset))
        default_warmup = math.ceil(shape["total_steps"] * 0.05)
        args.warmup_steps = default_warmup if args.warmup_steps is None else int(args.warmup_steps)
        if args.warmup_steps != default_warmup:
            raise ValueError(f"warmup_steps must equal ceil(total_steps * 0.05)={default_warmup}")
        formal_audit = _formal_audit_metadata(args, inventory, shape)

        model, tokenizer = _load_x5_8slot_model(args, device)
        dataset.tokenizer = tokenizer
        model.train()
        model_trainable_audit = model.trainable_parameter_audit()
        if not model_trainable_audit["training_mode_contract"]:
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
        ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        cursor = {"epoch": 0, "batch_in_epoch": 0, "global_step": 0}
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
            "architecture_contract": ARCHITECTURE_CONTRACT,
            "text_model_architecture_contract": TEXT_MODEL_ARCHITECTURE_CONTRACT,
            "mapper_contract": MAPPER_CONTRACT,
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
            "initialization": {
                "kind": "text_mesh_plus_random_audio_mapper",
                "fresh_source": str(args.model_path.resolve()),
            },
            "text_initialization_audit": model._initialization_audit,
            "model_trainable_audit": model_trainable_audit,
            "start_global_step": cursor["global_step"],
            "start_cursor": dict(cursor),
            "formal_audit": formal_audit,
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
        first_step_audio_slot_audits: list[dict[str, Any]] = []

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
                    # Every tensor consumed by AudioMeshModel must share the
                    # rank device; fixed-260 mode concatenates text masks with
                    # the GPU audio prefix.
                    batch = {key: (value.to(device) if torch.is_tensor(value) else value)
                             for key, value in batch.items()}
                    last_batch = batch
                    synchronization = ddp.no_sync() if micro + 1 < args.gradient_accumulation_steps else contextlib.nullcontext()
                    with synchronization:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            output = ddp(**{key: value for key, value in batch.items()
                                          if key not in {"row_indices", "audio2_reused", "waveform_cache_shard_ids"}})
                        if len(first_step_audio_slot_audits) < args.gradient_accumulation_steps:
                            first_step_audio_slot_audits.append(dict(ddp.module.last_audio_slot_audit))
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
                    first_gradient_audit = _mesh_runtime_gradient_audit_x5_8slot(owner)
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
                    cursor["global_step"] % args.save_every == 0
                    or cursor["global_step"] == shape["total_steps"]
                )
                if save:
                    checkpoint = args.output_dir / f"checkpoint-{cursor['global_step']:06d}"
                    _save_checkpoint(
                        checkpoint, owner, tokenizer, optimizer, scheduler,
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
        report.update({
            "status": "PASS",
            "end_global_step": cursor["global_step"],
            "end_cursor": cursor,
            "first_step_gradient_audit": first_gradient_audit,
            "first_step_audio_slot_audits": first_step_audio_slot_audits,
            "answer_only_label_audit": answer_label_audit,
            "routing_stats": "disabled to avoid per-forward GPU-to-CPU synchronization; first-step trace/gradient audit retained",
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

