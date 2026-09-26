#!/usr/bin/env python3
"""Train the Mellow-faithful SmolLM2 baseline from a node-shared store."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
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
from torch.utils.data import DataLoader

import train_audio_smollm2_135m_mellow_ddp as baseline
from audio_smollm2_135m_mellow.model import AudioSmolLM2Config
from audio_smollm2_135m_mellow_shared_store_configurable_epochs import TRAINING_CONTRACT
from audio_smollm2_135m_mellow_shared_store_configurable_epochs.data import (
    ANSWER_TOKENS,
    MELLOW_REFERENCE_COMMIT,
    MELLOW_TEMPLATE_BLOB_SHA,
    MELLOW_VARIABLE_STORE_FORMAT,
    PROMPT_TOKENS,
    ReasonAQADataset,
    SEGMENT_SAMPLES,
    collate_reasonaqa,
)
from audio_smollm2_135m_mellow_shared_store_configurable_epochs.model import (
    AUDIO_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    MAPPER_CONTRACT,
    ORIGINAL_SMOLLM2_CONTRACT,
    SMOLLM2_HIDDEN_SIZE,
    AudioSmolLM2Model,
)
from audio_smollm2_135m_mellow_shared_store_configurable_epochs.sampler import (
    ContiguousDistributedEpochSampler,
)

CONFIG_FILENAME = "audio_smollm2_shared_store_config.json"
DEFAULT_MANIFEST = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl")
DEFAULT_STORE = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/rsmol_reasonaqa_mellow_faithful_full_waveforms_32k_f32_v2")
SMOKE_FIRST_STOP = 20
SMOKE_TOTAL_STEPS = 22
FORMAL_EPOCHS = 30
CANONICAL_LR = 1e-3
CANONICAL_WEIGHT_DECAY = 1e-4
CANONICAL_MICRO_BATCH = 4
CANONICAL_GRAD_ACCUM = 1
CANONICAL_GLOBAL_BATCH = 32
CANONICAL_SEED = 1234
CANONICAL_CHECKPOINT_RETENTION = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "reference", "formal"), required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--unique-waveform-store-dir", type=Path, required=True)
    parser.add_argument("--persistent-manifest-source", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--persistent-store-source", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--store-copy-seconds", type=float, required=True)
    parser.add_argument("--manifest-copy-seconds", type=float, required=True)
    parser.add_argument("--staging-total-seconds", type=float, required=True)
    parser.add_argument("--model-path", type=Path, default=Path(baseline.DEFAULT_MODEL))
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--htsat-checkpoint", type=Path, default=Path(baseline.DEFAULT_HTSAT))
    parser.add_argument("--mellow-root", type=Path, default=Path(baseline.DEFAULT_MELLOW))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke20-report", type=Path)
    parser.add_argument("--smoke-resume-report", type=Path)
    parser.add_argument("--reference22-report", type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=CANONICAL_MICRO_BATCH)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=CANONICAL_GRAD_ACCUM)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=CANONICAL_LR)
    parser.add_argument("--weight-decay", type=float, default=CANONICAL_WEIGHT_DECAY)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--checkpoint-retention", type=int, default=CANONICAL_CHECKPOINT_RETENTION)
    parser.add_argument("--seed", type=int, default=CANONICAL_SEED)
    parser.add_argument("--dist-timeout-minutes", type=int, default=30)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route_code_identity() -> dict[str, str]:
    route_root = Path(__file__).resolve().parents[1]
    files = {
        "trainer": Path(__file__).resolve(),
        "entry": Path(__file__).resolve().with_name(
            "train_audio_smollm2_shared_store_configurable_epochs_135m_mellow_ddp.py"
        ),
        "package_init": route_root / "audio_smollm2_135m_mellow_shared_store_configurable_epochs" / "__init__.py",
        "data": route_root / "audio_smollm2_135m_mellow_shared_store_configurable_epochs" / "data.py",
        "model": route_root / "audio_smollm2_135m_mellow_shared_store_configurable_epochs" / "model.py",
        "sampler": route_root / "audio_smollm2_135m_mellow_shared_store_configurable_epochs" / "sampler.py",
        "templates": route_root / "audio_smollm2_135m_mellow_shared_store_configurable_epochs" / "mellow_templates.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"route code identity lacks files: {missing}")
    return {name: sha256(path) for name, path in files.items()}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def gather(value: Any, world: int) -> list[Any]:
    values: list[Any] = [None] * world
    if world > 1:
        dist.all_gather_object(values, value)
    else:
        values[0] = value
    return values


def _hash_value(digest: Any, value: Any) -> None:
    """Add a deterministic nested Python/Torch value to a SHA256 digest."""
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("utf-8") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("utf-8") + b"\0")
        # Adam stores its per-parameter step as a zero-dimensional tensor.
        # Flatten first because PyTorch forbids changing the element size of a
        # zero-dimensional tensor directly with view(torch.uint8).
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_value(digest, str(key))
            _hash_value(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _hash_value(digest, item)
        return
    digest.update(type(value).__name__.encode("utf-8") + b"\0")
    digest.update(repr(value).encode("utf-8") + b"\0")


def training_state_fingerprint(model: Any, optimizer: Any, scheduler: Any) -> dict[str, Any]:
    model_digest = hashlib.sha256()
    trainable_names: list[str] = []
    trainable_elements = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_names.append(name)
        trainable_elements += int(parameter.numel())
        _hash_value(model_digest, name)
        _hash_value(model_digest, parameter)
    optimizer_digest = hashlib.sha256()
    _hash_value(optimizer_digest, optimizer.state_dict())
    scheduler_digest = hashlib.sha256()
    _hash_value(scheduler_digest, scheduler.state_dict())
    return {
        "trainable_model_sha256": model_digest.hexdigest(),
        "optimizer_sha256": optimizer_digest.hexdigest(),
        "scheduler_sha256": scheduler_digest.hexdigest(),
        "trainable_parameter_count": len(trainable_names),
        "trainable_element_count": trainable_elements,
    }


def compare_reference22(
    reference_path: Path,
    trace_tail: list[dict[str, Any]],
    fingerprint: dict[str, Any],
    *,
    shape: dict[str, int],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    reference = read_json(reference_path.resolve(strict=True))
    if reference.get("status") != "PASS" or reference.get("mode") != "reference":
        raise RuntimeError("reference22 report is not a passing uninterrupted reference run")
    if reference.get("start_global_step") != 0 or reference.get("end_global_step") != 22:
        raise RuntimeError("reference22 report has the wrong cursor range")
    if reference.get("training_contract") != TRAINING_CONTRACT:
        raise RuntimeError("reference22 report has the wrong training contract")
    if reference.get("shape") != shape or reference.get("route_code_sha256") != route_code_identity():
        raise RuntimeError("reference22 report differs in training shape or route code")
    expected_optimizer = {
        "name": "Adam",
        "lr": CANONICAL_LR,
        "weight_decay": CANONICAL_WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR_epoch_level",
        "warmup_steps": 0,
    }
    if reference.get("optimizer_contract") != expected_optimizer:
        raise RuntimeError("reference22 report has the wrong optimizer contract")
    for key in ("manifest_sha256", "index_sha256", "waveform_sha256", "total_waveform_bytes"):
        if reference.get("store_inventory", {}).get(key) != inventory.get(key):
            raise RuntimeError(f"reference22 store differs in {key}")
    expected_trace = reference.get("resume_comparison_trace")
    if expected_trace != trace_tail:
        mismatch = first_trace_mismatch(expected_trace, trace_tail)
        raise RuntimeError(
            "resumed stochastic row/audio/crop/template/loss trace differs from "
            f"reference22 at {mismatch}"
        )
    expected_fingerprint = reference.get("training_state_fingerprint")
    if expected_fingerprint != fingerprint:
        raise RuntimeError("resumed model/optimizer/scheduler fingerprint differs from reference22")
    return {
        "passed": True,
        "reference22_report": str(reference_path.resolve()),
        "trace_exact_match": True,
        "training_state_exact_match": True,
    }


def first_trace_mismatch(expected: Any, actual: Any, path: str = "trace") -> str:
    """Locate the first divergence without hiding data or numerical differences."""
    if type(expected) is not type(actual):
        return f"{path}: types {type(expected).__name__} != {type(actual).__name__}"
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return f"{path}: keys {sorted(expected)} != {sorted(actual)}"
        for key in expected:
            if expected[key] != actual[key]:
                return first_trace_mismatch(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: lengths {len(expected)} != {len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            if left != right:
                return first_trace_mismatch(left, right, f"{path}[{index}]")
    elif expected != actual:
        return f"{path}: expected={expected!r}, actual={actual!r}"
    return f"{path}: unknown mismatch"


def prune_formal_checkpoints(output_dir: Path, keep: int) -> list[str]:
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        marker_path = path / "checkpoint_complete.json"
        if not path.is_dir() or not marker_path.is_file():
            continue
        marker = read_json(marker_path)
        if marker.get("status") == "complete" and marker.get("contract") == TRAINING_CONTRACT:
            checkpoints.append((int(marker.get("global_step", -1)), path))
    checkpoints.sort()
    for _, path in checkpoints[:-keep]:
        shutil.rmtree(path)
    return [str(path) for _, path in checkpoints[-keep:]]


def store_inventory(args: argparse.Namespace) -> dict[str, Any]:
    staged = args.unique_waveform_store_dir.resolve(strict=True)
    source = args.persistent_store_source.resolve(strict=True)
    staged_manifest = args.train_manifest.resolve(strict=True)
    source_manifest = args.persistent_manifest_source.resolve(strict=True)
    if Path("/dev/shm") not in staged.parents or Path("/dev/shm") not in staged_manifest.parents:
        raise RuntimeError("staged store and manifest must be under /dev/shm")
    required_files = ("metadata.json", "index.jsonl", "waveforms.f32")
    for root in (source, staged):
        if (root / "BUILDING").exists():
            raise RuntimeError(f"store is BUILDING: {root}")
        missing = [name for name in required_files if not (root / name).is_file()]
        if missing:
            raise RuntimeError(f"store {root} lacks {missing}")
    source_meta, staged_meta = read_json(source / "metadata.json"), read_json(staged / "metadata.json")
    if sha256(source / "metadata.json") != sha256(staged / "metadata.json"):
        raise RuntimeError("source/staged metadata differs")
    expected = {
        "status": "PASS", "format": MELLOW_VARIABLE_STORE_FORMAT,
        "mellow_reference_commit": MELLOW_REFERENCE_COMMIT,
        "sample_rate": 32000, "dtype": "float32", "byte_order": "little",
        "data_file": "waveforms.f32",
    }
    for label, metadata in (("source", source_meta), ("staged", staged_meta)):
        mismatch = {key: (value, metadata.get(key)) for key, value in expected.items() if metadata.get(key) != value}
        if mismatch:
            raise RuntimeError(f"{label} store contract mismatch: {mismatch}")
    identity_keys = (
        "manifest_sha256", "source_inventory_sha256", "index_sha256",
        "waveform_sha256", "num_unique_audio_files", "filepath1_unique_pool_size",
        "total_waveform_bytes",
    )
    mismatch = {key: (source_meta.get(key), staged_meta.get(key)) for key in identity_keys if source_meta.get(key) != staged_meta.get(key)}
    if mismatch:
        raise RuntimeError(f"source/staged store identity differs: {mismatch}")
    if sha256(source_manifest) != source_meta["manifest_sha256"] or sha256(staged_manifest) != source_meta["manifest_sha256"]:
        raise RuntimeError("manifest/store identity mismatch")
    for name, metadata_key in (("index.jsonl", "index_sha256"),):
        if sha256(source / name) != source_meta[metadata_key] or sha256(staged / name) != source_meta[metadata_key]:
            raise RuntimeError(f"{name} SHA256 mismatch")
    expected_bytes = int(source_meta["total_waveform_bytes"])
    if (source / "waveforms.f32").stat().st_size != expected_bytes or (staged / "waveforms.f32").stat().st_size != expected_bytes:
        raise RuntimeError("waveform byte-size mismatch")
    return {
        "persistent_store_source": str(source), "persistent_manifest_source": str(source_manifest),
        "staged_store_path": str(staged), "staged_manifest_path": str(staged_manifest),
        "metadata_sha256": sha256(source / "metadata.json"),
        **{key: source_meta[key] for key in identity_keys},
        "total_waveform_gib": expected_bytes / 1024**3,
        "preprocessing_contract": source_meta.get("preprocessing_contract"),
    }


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[AudioSmolLM2Model, Any]:
    from transformers import AutoTokenizer
    model_path = args.resume_from / "text_model" if args.resume_from else args.model_path
    tokenizer_path = args.tokenizer_path or (args.resume_from / "tokenizer" if args.resume_from else model_path)
    text_model = baseline._load_text_backbone(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    old_vocab = len(tokenizer)
    tokenizer.add_special_tokens({"pad_token": "!"})
    if tokenizer.pad_token_id != tokenizer.convert_tokens_to_ids("!"):
        raise RuntimeError("Mellow tokenizer pad token must be literal !")
    if len(tokenizer) != old_vocab:
        text_model.resize_token_embeddings(len(tokenizer))
    text_model.config.pad_token_id = int(tokenizer.pad_token_id)
    wrapper, htsat, provenance = baseline._load_mellow_wrapper(args.mellow_root, args.htsat_checkpoint, device)
    model = AudioSmolLM2Model(text_model.to(device), tokenizer, wrapper, htsat, AudioSmolLM2Config(compact_single_audio_prefix=False))
    if args.resume_from:
        audio_state = torch.load(args.resume_from / "audio_bridge.pt", map_location=device, weights_only=False)
        model.bridge.load_state_dict(audio_state["bridge"], strict=True)
        model.htsat_wrapper.c2l.load_state_dict(audio_state["c2l"], strict=True)
    model._audio_provenance = provenance
    model._text_model_source = str(args.model_path.resolve())
    return model.to(device), tokenizer


def training_shape(args: argparse.Namespace, rows: int) -> dict[str, int]:
    global_batch = args.world_size * args.micro_batch_size * args.gradient_accumulation_steps
    if global_batch != CANONICAL_GLOBAL_BATCH:
        raise RuntimeError(
            f"Mellow reproduction requires effective global batch {CANONICAL_GLOBAL_BATCH}, "
            f"got {global_batch}"
        )
    steps_per_epoch = rows // global_batch
    if steps_per_epoch <= 0:
        raise RuntimeError("dataset is shorter than one global batch")
    return {
        "global_batch_size": global_batch,
        "steps_per_epoch": steps_per_epoch,
        "microbatches_per_epoch": steps_per_epoch * args.gradient_accumulation_steps,
        "dropped_rows_per_epoch": rows - steps_per_epoch * global_batch,
        "total_steps": steps_per_epoch * args.epochs,
    }


def checkpoint_config(args: argparse.Namespace, inventory: dict[str, Any], shape: dict[str, int], model: Any) -> dict[str, Any]:
    source_config = args.model_path.resolve() / "config.json"
    return {
        "contract": TRAINING_CONTRACT,
        "architecture_contract": ORIGINAL_SMOLLM2_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "mellow_reference_commit": MELLOW_REFERENCE_COMMIT,
        "mellow_template_blob_sha": MELLOW_TEMPLATE_BLOB_SHA,
        "route_code_sha256": route_code_identity(),
        "text_model_source_path": str(args.model_path.resolve()),
        "text_model_source_config_sha256": sha256(source_config) if source_config.is_file() else None,
        "standard_text_contract": model.text_contract,
        "mapper_initialization": "random_c2l_and_xavier_projection",
        "sequence_contract": {
            "audio1_tokens": 129, "separator1_tokens": 1, "audio2_tokens": 129,
            "separator2_tokens": 1, "prompt_tokens": PROMPT_TOKENS,
            "answer_tokens": ANSWER_TOKENS, "total_tokens": 639,
            "answer_logit_slice": [388, 638], "text_attention_mask_passed": False,
        },
        "audio_slot_contract": "missing audio2 samples uniformly from non-empty filepath1 pool; both HTSAT passes are independent",
        "crop_contract": "uniform random inclusive offset for clips longer than 320000 samples; right-zero-pad shorter clips",
        "template_contract": "verbatim public Mellow data/template.py and audiotext_dataset.py dispatch",
        "store_identity": {
            key: inventory[key]
            for key in (
                "persistent_store_source", "persistent_manifest_source", "manifest_sha256",
                "metadata_sha256", "index_sha256", "waveform_sha256",
                "num_unique_audio_files", "filepath1_unique_pool_size", "total_waveform_bytes",
            )
        },
        "mode": args.mode, "epochs": args.epochs, "world_size": args.world_size,
        "micro_batch_size": args.micro_batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_global_batch_size": shape["global_batch_size"], "num_workers": args.num_workers,
        "seed": args.seed, "optimizer": "Adam", "optimizer_betas": [0.9, 0.999],
        "optimizer_eps": 1e-8, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "scheduler": "CosineAnnealingLR_epoch_level",
        "scheduler_t_max_epochs": args.epochs, "scheduler_eta_min": 0.0,
        "warmup_steps": 0, "gradient_clip_norm": 0.5, "autocast": False,
        "total_steps": shape["total_steps"], "steps_per_epoch": shape["steps_per_epoch"],
        "save_every_epochs": args.save_every_epochs,
        "checkpoint_retention": args.checkpoint_retention,
        "htsat_checkpoint": str(args.htsat_checkpoint.resolve()), "mellow_root": str(args.mellow_root.resolve()),
        "mellow_provenance": model._audio_provenance,
    }


def save_checkpoint(path: Path, model: Any, tokenizer: Any, optimizer: Any, scheduler: Any, args: argparse.Namespace, inventory: dict[str, Any], shape: dict[str, int], cursor: dict[str, int], rank: int, world: int, device: torch.device) -> None:
    rng_states = gather(baseline._rng_state(device), world)
    if rank != 0:
        return
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent))
    published = False
    try:
        model.text_model.save_pretrained(temporary / "text_model", safe_serialization=False)
        tokenizer.save_pretrained(temporary / "tokenizer")
        torch.save(baseline._trainable_state(model), temporary / "audio_bridge.pt")
        torch.save({
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scheduler_name": "CosineAnnealingLR_epoch_level",
            "optimizer_name": "Adam", "training_contract": TRAINING_CONTRACT,
            "global_step": cursor["global_step"], "cursor": cursor,
            "rng_states_by_rank": {str(index): state for index, state in enumerate(rng_states)},
            "optimizer_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        }, temporary / "training_state.pt")
        config = checkpoint_config(args, inventory, shape, model)
        config.update({"global_step": cursor["global_step"], "epoch": cursor["epoch"], "batch_in_epoch": cursor["batch_in_epoch"], "scheduler_last_epoch": scheduler.last_epoch})
        (temporary / CONFIG_FILENAME).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        required = ["text_model", "tokenizer", "audio_bridge.pt", "training_state.pt", CONFIG_FILENAME]
        marker = {"status": "complete", "contract": TRAINING_CONTRACT, "global_step": cursor["global_step"], "required": required}
        (temporary / "checkpoint_complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        if not baseline._text_model_weight_files(temporary / "text_model"):
            raise RuntimeError("checkpoint has no text-model weights")
        temporary.replace(path)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def validate_checkpoint(path: Path, args: argparse.Namespace, inventory: dict[str, Any], shape: dict[str, int], model: Any, expected_step: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve(strict=True)
    marker, config = read_json(path / "checkpoint_complete.json"), read_json(path / CONFIG_FILENAME)
    if marker.get("status") != "complete" or marker.get("contract") != TRAINING_CONTRACT or config.get("contract") != TRAINING_CONTRACT:
        raise RuntimeError("checkpoint contract/completion marker failed")
    if expected_step is not None and int(marker.get("global_step", -1)) != expected_step:
        raise RuntimeError("checkpoint global step mismatch")
    expected = checkpoint_config(args, inventory, shape, model)
    for key in expected:
        if config.get(key) != expected.get(key):
            raise RuntimeError(f"checkpoint contract differs in {key}")
    for name in marker.get("required", []):
        if not (path / name).exists():
            raise RuntimeError(f"checkpoint lacks {name}")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    if state.get("training_contract") != TRAINING_CONTRACT or state.get("optimizer_name") != "Adam" or state.get("scheduler_name") != "CosineAnnealingLR_epoch_level":
        raise RuntimeError("checkpoint optimizer/scheduler contract failed")
    return config, state


def resume_checkpoint(path: Path, args: argparse.Namespace, inventory: dict[str, Any], shape: dict[str, int], optimizer: Any, scheduler: Any, rank: int, device: torch.device, model: Any) -> dict[str, int]:
    config, state = validate_checkpoint(path, args, inventory, shape, model)
    baseline._validate_optimizer_coverage(state, model)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    # Adam keeps its non-capturable scalar step on CPU. load_state_dict already
    # places moment tensors on their parameter device; moving every tensor to
    # CUDA changes the optimizer's state layout relative to an uninterrupted run.
    for optimizer_state in optimizer.state.values():
        step = optimizer_state.get("step")
        if torch.is_tensor(step) and step.device.type != "cpu":
            raise RuntimeError("resumed Adam step tensor must remain on CPU")
    cursor = {key: int(state["cursor"][key]) for key in ("epoch", "batch_in_epoch", "global_step")}
    if cursor["global_step"] != cursor["epoch"] * shape["steps_per_epoch"] + cursor["batch_in_epoch"]:
        raise RuntimeError(f"resume cursor is inconsistent: {cursor}")
    if int(config["global_step"]) != cursor["global_step"] or int(config["epoch"]) != cursor["epoch"] or int(config["batch_in_epoch"]) != cursor["batch_in_epoch"]:
        raise RuntimeError("resume config/cursor mismatch")
    rng = state.get("rng_states_by_rank", {})
    if set(rng) != {str(index) for index in range(args.world_size)}:
        raise RuntimeError("resume checkpoint lacks all rank RNG states")
    baseline._restore_rng_state(rng[str(rank)], device)
    return cursor


def data_contract_audit(dataset: ReasonAQADataset) -> dict[str, Any]:
    single = dual = explicit_same = 0
    for index in range(len(dataset)):
        is_single, same = dataset.audio_structure(index)
        single += int(is_single)
        dual += int(not is_single)
        explicit_same += int((not is_single) and same)
    if single <= 0 or dual <= 0:
        raise RuntimeError(
            "manifest must contain both single- and dual-audio rows after "
            f"restoring normalized filepath2 semantics: single={single}, dual={dual}, "
            f"rows={len(dataset)}"
        )
    return {
        "passed": True, "rows": len(dataset), "single_audio_rows": single,
        "dual_audio_rows": dual, "explicit_same_audio_rows": explicit_same,
        "single_audio_behavior": "sample audio2 from filepath1 pool; never reuse audio1 embedding",
        "random_process": "stateful process-wide Python random in public-Mellow call order",
        "random_audio_pool": "sorted unique non-empty filepath1 paths; uniform choice and self-selection allowed",
    }


def batch_contract_audit(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    if tuple(batch["prompt_input_ids"].shape[1:]) != (PROMPT_TOKENS,) or tuple(batch["answer_input_ids"].shape[1:]) != (ANSWER_TOKENS,):
        raise RuntimeError("fixed prompt/answer shapes failed")
    if bool(batch["audio2_reused_mask"].any()):
        raise RuntimeError("Mellow batch reused audio1 embedding")
    if model.last_multimodal_sequence_length != 639 or model.last_audio_tokens_per_clip != (AUDIO_TOKENS_PER_CLIP, AUDIO_TOKENS_PER_CLIP):
        raise RuntimeError("Mellow model sequence/audio-token contract failed")
    labels = model.last_labels
    if labels is None or labels.shape[1] != 639 or bool((labels[:, :389] != -100).any()):
        raise RuntimeError("Mellow answer-loss prefix mask failed")
    expected = batch["answer_input_ids"].masked_fill(batch["answer_input_ids"].eq(int(model.tokenizer.pad_token_id)), -100)
    if not torch.equal(labels[:, 389:], expected):
        raise RuntimeError("Mellow answer targets differ from fixed answer tokens")
    return {
        "passed": True, "sequence_tokens": 639, "audio_prefix_tokens": AUDIO_PREFIX_TOKENS,
        "prompt_tokens": PROMPT_TOKENS, "answer_tokens": ANSWER_TOKENS,
        "answer_start": 389, "text_attention_mask_passed": False,
        "random_audio2_rows": int(batch["audio2_random_mask"].sum().item()),
        "nonzero_crop_offsets": int((batch["audio1_crop_offsets"] > 0).sum().item() + (batch["audio2_crop_offsets"] > 0).sum().item()),
        "template_groups": sorted(set(batch["template_groups"])),
    }


def formal_gate(args: argparse.Namespace, inventory: dict[str, Any], shape: dict[str, int]) -> dict[str, Any] | None:
    if args.mode != "formal":
        return None
    if args.smoke20_report is None or args.reference22_report is None or args.smoke_resume_report is None:
        raise ValueError(
            "formal requires --smoke20-report, --reference22-report, and --smoke-resume-report"
        )
    first = read_json(args.smoke20_report)
    reference = read_json(args.reference22_report)
    resumed = read_json(args.smoke_resume_report)
    current_code_identity = route_code_identity()
    for label, report, start, end in (("smoke20", first, 0, 20), ("resume2", resumed, 20, 22)):
        if report.get("status") != "PASS" or report.get("training_contract") != TRAINING_CONTRACT or report.get("start_global_step") != start or report.get("end_global_step") != end:
            raise RuntimeError(f"formal gate rejects {label} status/cursor")
        if report.get("shape") != shape or report.get("epochs") != args.epochs:
            raise RuntimeError(f"formal gate rejects {label} training shape")
        if report.get("route_code_sha256") != current_code_identity:
            raise RuntimeError(f"formal gate rejects {label} code identity")
        if report.get("optimizer_contract") != {"name": "Adam", "lr": CANONICAL_LR, "weight_decay": CANONICAL_WEIGHT_DECAY, "scheduler": "CosineAnnealingLR_epoch_level", "warmup_steps": 0}:
            raise RuntimeError(f"formal gate rejects {label} optimizer")
        if report.get("batch_contract_audit", {}).get("passed") is not True or report.get("first_step_gradient_audit", {}).get("passed") is not True:
            raise RuntimeError(f"formal gate rejects {label} runtime audit")
        for key in ("manifest_sha256", "index_sha256", "waveform_sha256", "total_waveform_bytes"):
            if report.get("store_inventory", {}).get(key) != inventory.get(key):
                raise RuntimeError(f"formal gate {label} store differs in {key}")
    if (
        reference.get("status") != "PASS"
        or reference.get("mode") != "reference"
        or reference.get("training_contract") != TRAINING_CONTRACT
        or reference.get("start_global_step") != 0
        or reference.get("end_global_step") != 22
        or reference.get("shape") != shape
        or reference.get("epochs") != args.epochs
        or reference.get("route_code_sha256") != current_code_identity
    ):
        raise RuntimeError("formal gate rejects uninterrupted reference22 status/shape/cursor")
    for key in ("manifest_sha256", "index_sha256", "waveform_sha256", "total_waveform_bytes"):
        if reference.get("store_inventory", {}).get(key) != inventory.get(key):
            raise RuntimeError(f"formal gate reference22 store differs in {key}")
    checkpoints1, checkpoints2 = first.get("checkpoints", []), resumed.get("checkpoints", [])
    if len(checkpoints1) != 1 or len(checkpoints2) != 1 or resumed.get("resume_checkpoint") != str(Path(checkpoints1[0]).resolve()):
        raise RuntimeError("formal gate resume lineage failed")
    if resumed.get("resume_parameter_change_audit", {}).get("all_groups_changed") is not True:
        raise RuntimeError("formal gate resume parameter-change audit failed")
    if resumed.get("resume_equivalence", {}).get("passed") is not True:
        raise RuntimeError("formal gate exact resume/reference22 comparison failed")
    return {
        "smoke20_report": str(args.smoke20_report.resolve()),
        "reference22_report": str(args.reference22_report.resolve()),
        "smoke_resume_report": str(args.smoke_resume_report.resolve()),
        "checkpoint20": str(Path(checkpoints1[0]).resolve()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0")); local_rank = int(os.environ.get("LOCAL_RANK", str(rank))); world = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
    report: dict[str, Any] = {"status": "FAIL", "mode": args.mode, "training_contract": TRAINING_CONTRACT, "rank": rank, "checkpoints": [], "metrics": [], "hard_failures": []}
    output_available = not args.output_dir.exists() or not any(args.output_dir.iterdir())
    try:
        if not output_available:
            raise FileExistsError(f"refusing nonempty output: {args.output_dir}")
        if not torch.cuda.is_available() or world != 8 or args.world_size != 8:
            raise RuntimeError("Mellow reproduction requires one 8-GPU node")
        if args.micro_batch_size != CANONICAL_MICRO_BATCH or args.gradient_accumulation_steps != CANONICAL_GRAD_ACCUM or args.num_workers != 0:
            raise RuntimeError("this reproduction requires microbatch=4, grad_accum=1, effective global batch=32, num_workers=0")
        if args.learning_rate != CANONICAL_LR or args.weight_decay != CANONICAL_WEIGHT_DECAY or args.seed != CANONICAL_SEED:
            raise RuntimeError("Mellow optimizer/seed contract mismatch")
        if args.epochs != FORMAL_EPOCHS or args.save_every_epochs != 1 or args.checkpoint_retention != CANONICAL_CHECKPOINT_RETENTION:
            raise RuntimeError("this route requires 30 epochs, epoch saves, and retention of the newest three checkpoints")
        torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(minutes=args.dist_timeout_minutes))
        baseline._seed(args.seed, rank)
        inventory = store_inventory(args)
        data_stat = (args.unique_waveform_store_dir / "waveforms.f32").stat()
        rank_store = gather({"rank": rank, "path": str(args.unique_waveform_store_dir.resolve()), "device": int(data_stat.st_dev), "inode": int(data_stat.st_ino), "bytes": int(data_stat.st_size)}, world)
        if len({(item["path"], item["device"], item["inode"], item["bytes"]) for item in rank_store}) != 1:
            raise RuntimeError("ranks do not share one staged waveform inode")

        model, tokenizer = load_model(args, device)
        dataset = ReasonAQADataset(
            args.train_manifest,
            tokenizer,
            unique_waveform_store_dir=args.unique_waveform_store_dir,
        )
        data_audit = data_contract_audit(dataset)
        shape = training_shape(args, len(dataset))
        gate = formal_gate(args, inventory, shape)
        model.train()
        trainable_audit = model.trainable_parameter_audit()
        if trainable_audit.get("training_mode_contract") is not True:
            raise RuntimeError("trainable parameter audit failed")
        optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)
        ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        cursor = {"epoch": 0, "batch_in_epoch": 0, "global_step": 0}
        if args.resume_from:
            cursor = resume_checkpoint(args.resume_from, args, inventory, shape, optimizer, scheduler, rank, device, model)
        resume_representatives = baseline._select_resume_representatives(model) if args.resume_from else None
        resume_snapshots = baseline._snapshot_resume_representatives(resume_representatives) if resume_representatives else None
        if args.mode == "smoke":
            if args.resume_from is None and cursor["global_step"] != 0:
                raise RuntimeError("initial smoke must start at zero")
            if args.resume_from is not None and cursor != {"epoch": 0, "batch_in_epoch": 20, "global_step": 20}:
                raise RuntimeError("resume smoke requires checkpoint-000020")
            if args.resume_from is not None and args.reference22_report is None:
                raise RuntimeError("resume2 requires --reference22-report for exact comparison")
            stop_step = SMOKE_TOTAL_STEPS if args.resume_from else SMOKE_FIRST_STOP
        elif args.mode == "reference":
            if args.resume_from is not None or cursor["global_step"] != 0:
                raise RuntimeError("reference22 must be an uninterrupted fresh run")
            stop_step = SMOKE_TOTAL_STEPS
        else:
            if args.resume_from is not None:
                raise RuntimeError("formal training must initialize fresh, not resume from smoke")
            stop_step = shape["total_steps"]
        report.update({
            "store_inventory": inventory, "rank_store_audit": rank_store if rank == 0 else None,
            "dataset_rows": len(dataset), "data_contract_audit": data_audit, "shape": shape,
            "epochs": args.epochs, "optimizer_contract": {"name": "Adam", "lr": args.learning_rate, "weight_decay": args.weight_decay, "scheduler": "CosineAnnealingLR_epoch_level", "warmup_steps": 0},
            "precision_contract": "float32 forward/backward; no autocast and no GradScaler",
            "initialization": {"kind": "original_smollm2_plus_random_mellow_mapper" if not args.resume_from else "full_state_resume", "fresh_source": str(args.model_path.resolve()), "resume_source": str(args.resume_from.resolve()) if args.resume_from else None},
            "start_global_step": cursor["global_step"], "start_cursor": dict(cursor),
            "resume_checkpoint": str(args.resume_from.resolve()) if args.resume_from else None,
            "formal_gate": gate, "model_trainable_audit": trainable_audit,
            "sampler_contract": "torch.randperm(seed=epoch), truncate to complete effective global batches, contiguous per-rank slices",
            "effective_global_batch_size": shape["global_batch_size"],
            "checkpoint_policy": {"save_every_epochs": 1, "retain_newest": CANONICAL_CHECKPOINT_RETENTION},
            "route_code_sha256": route_code_identity(),
        })
        first_gradient_audit = None; first_batch_audit = None; resumed_gradient_audit = None
        resume_comparison_trace: list[dict[str, Any]] = []
        sampler_audits: list[dict[str, Any]] = []
        while cursor["global_step"] < stop_step:
            epoch = cursor["epoch"]
            if epoch >= args.epochs:
                raise RuntimeError("epochs exhausted before target step")
            while scheduler.last_epoch < epoch:
                scheduler.step()
            if scheduler.last_epoch != epoch:
                raise RuntimeError(f"scheduler epoch mismatch: scheduler={scheduler.last_epoch} data={epoch}")
            sampler = ContiguousDistributedEpochSampler(
                len(dataset),
                num_replicas=world,
                rank=rank,
                per_rank_batch_size=args.micro_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                epoch=epoch,
                start_optimizer_step=cursor["batch_in_epoch"],
            )
            sampler_audits.append(sampler.audit())
            loader = DataLoader(
                dataset,
                batch_size=args.micro_batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=True,
                num_workers=0,
                generator=torch.Generator().manual_seed(args.seed + rank + epoch),
                collate_fn=lambda rows: collate_reasonaqa(rows, tokenizer),
            )
            iterator = iter(loader)
            while cursor["batch_in_epoch"] < shape["steps_per_epoch"] and cursor["global_step"] < stop_step:
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                local_micro_trace: list[dict[str, Any]] = []
                micro_losses: list[float] = []
                owner = ddp.module
                for micro_index in range(args.gradient_accumulation_steps):
                    batch = next(iterator)
                    moved = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
                    sync_context = contextlib.nullcontext() if micro_index == args.gradient_accumulation_steps - 1 else ddp.no_sync()
                    with sync_context:
                        output = ddp(**{
                            key: moved[key]
                            for key in (
                                "audio1", "audio2", "prompt_input_ids", "prompt_attention_mask",
                                "answer_input_ids", "answer_attention_mask", "audio2_reused_mask",
                                "single_audio_slot_mask",
                            )
                        })
                        if output.loss is None or not bool(torch.isfinite(output.loss)):
                            raise RuntimeError("nonfinite Mellow training loss")
                        raw_loss = float(output.loss.detach().float().item())
                        micro_losses.append(raw_loss)
                        (output.loss / args.gradient_accumulation_steps).backward()
                    if first_batch_audit is None:
                        first_batch_audit = batch_contract_audit(owner, moved)
                    local_micro_trace.append({
                        "micro_index": micro_index,
                        "row_indices": list(batch["row_indices"]),
                        "audio1_ids": batch["audio1_ids"].tolist(),
                        "audio2_ids": batch["audio2_ids"].tolist(),
                        "audio1_crop_offsets": batch["audio1_crop_offsets"].tolist(),
                        "audio2_crop_offsets": batch["audio2_crop_offsets"].tolist(),
                        "template_groups": list(batch["template_groups"]),
                        "loss": raw_loss,
                    })
                if first_gradient_audit is None:
                    gradient = owner.runtime_gradient_audit()
                    first_gradient_audit = {
                        "passed": bool(gradient.get("all_decoder_layers_have_finite_gradient") and gradient.get("embedding_has_finite_gradient") and gradient.get("lm_head_has_finite_gradient") and gradient.get("htsat_frozen_and_gradient_free") and all(gradient.get("bridge_gradients", {}).values()) and all(gradient.get("c2l_gradients", {}).values())),
                        **gradient,
                    }
                    if not first_gradient_audit["passed"]:
                        raise RuntimeError("first-step gradient audit failed")
                if resume_representatives is not None and resumed_gradient_audit is None:
                    resumed_gradient_audit = baseline._verify_resume_representative_gradients(resume_representatives)
                grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 0.5, error_if_nonfinite=True)
                lr_used = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                cursor["global_step"] += 1; cursor["batch_in_epoch"] += 1
                epoch_completed = cursor["batch_in_epoch"] == shape["steps_per_epoch"]
                if epoch_completed:
                    cursor["epoch"] += 1; cursor["batch_in_epoch"] = 0
                    scheduler.step()
                    if scheduler.last_epoch != cursor["epoch"]:
                        raise RuntimeError("epoch-level scheduler did not advance with the completed epoch")
                torch.cuda.synchronize(device)
                step_loss = sum(micro_losses) / len(micro_losses)
                metric = {"step": cursor["global_step"], "epoch": cursor["epoch"], "batch_in_epoch": cursor["batch_in_epoch"], "loss": step_loss, "micro_losses": micro_losses, "lr": lr_used, "grad_norm": float(grad_norm.detach().cpu()), "seconds": time.perf_counter() - started}
                if cursor["global_step"] in {21, 22}:
                    resume_comparison_trace.append({
                        "step": cursor["global_step"],
                        "lr": lr_used,
                        "by_rank": gather(local_micro_trace, world),
                    })
                if rank == 0:
                    report["metrics"].append(metric)
                    if cursor["global_step"] % 10 == 0 or cursor["global_step"] == stop_step:
                        print(f"[mellow-faithful] step={cursor['global_step']}/{stop_step} epoch={cursor['epoch']} batch={cursor['batch_in_epoch']} loss={metric['loss']:.6f} lr={lr_used:.8g}", flush=True)
                save = (
                    (args.mode == "smoke" and cursor["global_step"] in {20, 22})
                    or (args.mode == "formal" and epoch_completed and cursor["epoch"] % args.save_every_epochs == 0)
                )
                if save:
                    checkpoint = args.output_dir / f"checkpoint-{cursor['global_step']:06d}"
                    save_checkpoint(checkpoint, owner, tokenizer, optimizer, scheduler, args, inventory, shape, dict(cursor), rank, world, device)
                    if rank == 0:
                        if args.mode == "formal":
                            report.setdefault("checkpoint_history", []).append(str(checkpoint))
                        else:
                            report["checkpoints"].append(str(checkpoint))
                    dist.barrier()
                    if args.mode == "formal":
                        retained = prune_formal_checkpoints(args.output_dir, args.checkpoint_retention) if rank == 0 else None
                        retained_by_rank = gather(retained, world)
                        if rank == 0:
                            report["retained_checkpoints"] = retained_by_rank[0]
                            report["checkpoints"] = retained_by_rank[0]
                if epoch_completed:
                    break
        resume_change = None
        if resume_representatives and resume_snapshots:
            resume_change = baseline._compute_resume_parameter_change_audit(resume_representatives, resume_snapshots)
            baseline._validate_parameter_change_audit(resume_change)
        local_fingerprint = training_state_fingerprint(owner, optimizer, scheduler)
        fingerprints = gather(local_fingerprint, world)
        if len({json.dumps(item, sort_keys=True) for item in fingerprints}) != 1:
            raise RuntimeError("DDP ranks ended with different model/optimizer/scheduler fingerprints")
        resume_equivalence = None
        if args.mode == "smoke" and args.resume_from is not None:
            if rank == 0:
                # Preserve both traces in the FAIL report if the comparison
                # rejects this run, so a remote divergence can be inspected.
                report["resume_comparison_trace"] = resume_comparison_trace
                report["training_state_fingerprint"] = local_fingerprint
            resume_equivalence = compare_reference22(
                args.reference22_report,
                resume_comparison_trace,
                local_fingerprint,
                shape=shape,
                inventory=inventory,
            )
        report.update({
            "status": "PASS", "end_global_step": cursor["global_step"], "end_cursor": cursor,
            "batch_contract_audit": first_batch_audit, "first_step_gradient_audit": first_gradient_audit,
            "resume_representative_gradient_verification": resumed_gradient_audit,
            "resume_parameter_change_audit": resume_change,
            "resume_verified_two_steps": bool(args.mode == "smoke" and args.resume_from and report["start_global_step"] == 20 and cursor["global_step"] == 22),
            "resume_comparison_trace": resume_comparison_trace,
            "training_state_fingerprint": local_fingerprint,
            "resume_equivalence": resume_equivalence,
            "sampler_audits": sampler_audits,
        })
        return report
    except Exception as exc:
        report["hard_failures"].append({"error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        if rank == 0 and output_available:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "shared_store_training_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run(parse_args())
