#!/usr/bin/env python3
"""DDP trainer for the original SmolLM2-135M audio baseline.

The trainer owns a separate artifact contract from the recursive text-model
routes.  It uses the same ReasonAQA/Mellow data and audio semantics, but loads
the local standard Hugging Face ``LlamaForCausalLM`` checkpoint directly.
"""
from __future__ import annotations

import argparse
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_smollm2_135m_mellow.data import ReasonAQADataset, collate_reasonaqa  # noqa: E402
from audio_smollm2_135m_mellow.model import (  # noqa: E402
    AUDIO_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    MAPPER_CONTRACT,
    ORIGINAL_SMOLLM2_CONTRACT,
    AudioSmolLM2Config,
    AudioSmolLM2Model,
    SMOLLM2_HIDDEN_SIZE,
    _load_mellow_wrapper,
    validate_original_smollm2,
)


DEFAULT_MODEL = "/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2"
DEFAULT_HTSAT = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt"
DEFAULT_MELLOW = "/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main"
DEFAULT_TRAIN_MANIFEST = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl"
DEFAULT_VAL_MANIFEST = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_val.jsonl"
DEFAULT_OUTPUT = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/manual_run"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("STAGE7", "FORMAL"), default="STAGE7")
    parser.add_argument("--model-path", "--smollm2-model", dest="model_path", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--htsat-checkpoint", type=Path, default=Path(DEFAULT_HTSAT))
    parser.add_argument("--mellow-root", type=Path, default=Path(DEFAULT_MELLOW))
    parser.add_argument("--train-manifest", type=Path, default=Path(DEFAULT_TRAIN_MANIFEST))
    parser.add_argument("--val-manifest", type=Path, default=Path(DEFAULT_VAL_MANIFEST))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--schedule-total-steps", type=int, help="short-run scheduler length; independent from the execution stop step")
    parser.add_argument("--expected-resume-step", type=int, help="strict parent step expected by a bounded resume wrapper")
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--checkpoint-retention", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers per DDP rank")
    return parser.parse_args(argv)


def _init_dist(args: argparse.Namespace) -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    if not torch.cuda.is_available():
        raise RuntimeError("the audio baseline trainer requires CUDA; submit it through the 5090 wrapper")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return rank, world, device


def _validate_launch_contract(args: argparse.Namespace) -> None:
    """Reject unsafe resume destinations before distributed/model initialization."""
    if args.resume_from is not None:
        resume_path = args.resume_from.resolve()
        output_path = args.output_dir.resolve()
        if output_path in {resume_path, resume_path.parent}:
            raise ValueError(
                "resume output-dir must be separate from the source checkpoint and its parent "
                "so parent lineage remains outside continuation retention pruning"
            )
        if args.tokenizer_path is not None:
            raise ValueError("--tokenizer-path cannot be combined with --resume-from; resume must use the checkpoint tokenizer")


def _seed(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_hashes(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "train": _sha256(args.train_manifest),
        "val": _sha256(args.val_manifest) if args.val_manifest.is_file() else None,
    }


def _load_model(args: argparse.Namespace, device: torch.device) -> tuple[AudioSmolLM2Model, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.resume_from / "text_model" if args.resume_from else args.model_path
    tokenizer_path = args.tokenizer_path or (args.resume_from / "tokenizer" if args.resume_from else model_path)
    text_model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    validate_original_smollm2(text_model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("SmolLM2 tokenizer must define eos_token_id when pad_token_id is absent")
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(text_model.config, "pad_token_id", None) is None:
        text_model.config.pad_token_id = int(tokenizer.pad_token_id)
    wrapper, htsat, provenance = _load_mellow_wrapper(args.mellow_root, args.htsat_checkpoint, device)
    model = AudioSmolLM2Model(text_model.to(device), tokenizer, wrapper, htsat, AudioSmolLM2Config())
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
    model._text_model_source = str(args.model_path.resolve())
    return model.to(device), tokenizer


def _trainable_state(model: AudioSmolLM2Model) -> dict[str, Any]:
    c2l = getattr(model.htsat_wrapper, "c2l", None)
    return {"bridge": model.bridge.state_dict(), "c2l": c2l.state_dict() if c2l is not None else {}}


def _model_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    ignored = {"row_indices", "audio2_reused"}
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items() if key not in ignored}


def _checkpoint_config(
    args: argparse.Namespace,
    model: AudioSmolLM2Model,
    step: int,
    epoch: int,
    batch_in_epoch: int,
    total_steps: int,
    manifest_hashes: dict[str, str | None],
    *,
    resume_from: Path | None,
    resume_start_step: int,
    run_target_step: int,
    parameter_change_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    source_config = model.text_model.config.to_dict()
    config_path = Path(str(args.model_path)) / "config.json"
    run_kind = "formal" if args.gate == "FORMAL" else "stage7"
    return {
        "artifact_contract": "audio_smollm2_135m_mellow_composite_v1",
        "gate": str(args.gate),
        "run_kind": run_kind,
        "architecture_contract": ORIGINAL_SMOLLM2_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "text_model_source_path": str(args.model_path.resolve()),
        "text_model_runtime_class": model.text_contract["model_class"],
        "text_model_type": model.text_contract["model_type"],
        "text_model_hidden_size": int(model.text_contract["hidden_size"]),
        "text_model_num_hidden_layers": int(model.text_contract["num_hidden_layers"]),
        "text_model_config": source_config,
        "text_model_config_sha256": _sha256(config_path) if config_path.is_file() else None,
        "embedding_lm_head_tied": bool(model.text_contract["embedding_lm_head_tied"]),
        "audio_tokens_per_clip": AUDIO_TOKENS_PER_CLIP,
        "audio_prefix_tokens_with_separators": AUDIO_PREFIX_TOKENS,
        "sample_rate": 32000,
        "audio_seconds": 10,
        "max_prompt_tokens": 129,
        "max_answer_tokens": 250,
        "max_context_length": 768,
        "train_manifest": str(args.train_manifest.resolve()),
        "val_manifest": str(args.val_manifest.resolve()),
        "manifest_sha256": manifest_hashes["train"],
        "manifest_hashes": manifest_hashes,
        "htsat_checkpoint": str(args.htsat_checkpoint.resolve()),
        "mellow_root": str(args.mellow_root.resolve()),
        "mellow_provenance": getattr(model, "_audio_provenance", {}),
        "epochs": int(args.epochs),
        "max_lr": float(args.max_lr),
        "min_lr": float(args.min_lr),
        "scheduler": "cosine_lambda",
        "warmup_steps": int(args.warmup_steps),
        "total_steps": int(total_steps),
        "schedule_total_steps": int(total_steps),
        "run_target_step": int(run_target_step),
        "execution_max_steps": int(run_target_step),
        "resume_from": str(resume_from.resolve()) if resume_from is not None else None,
        "resume_start_step": int(resume_start_step),
        "parent_checkpoint_global_step": int(resume_start_step) if resume_from is not None else None,
        "parameter_change_audit": parameter_change_audit,
        "global_step": int(step),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "world_size": int(args.world_size),
        "micro_batch_size": int(args.micro_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "effective_global_batch_size": int(args.micro_batch_size * args.world_size * args.gradient_accumulation_steps),
        "seed": int(args.seed),
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "gradient_clip_norm": 0.5,
        "autocast_dtype": "bfloat16",
        "ddp_broadcast_buffers": False,
        "ddp_find_unused_parameters": False,
        "save_every": int(args.save_every),
        "checkpoint_retention": int(args.checkpoint_retention),
        "num_workers": int(args.num_workers),
        "trainable_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        "frozen_audio_encoder": True,
        "periodic_validation": False,
    }


def _save_checkpoint(
    path: Path,
    model: AudioSmolLM2Model,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    epoch: int,
    batch_in_epoch: int,
    args: argparse.Namespace,
    manifest_hashes: dict[str, str | None],
    rng_states_by_rank: dict[str, Any],
    total_steps: int,
    *,
    resume_from: Path | None,
    resume_start_step: int,
    run_target_step: int,
    parameter_change_audit: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {path}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)))
    published = False
    try:
        run_kind = "formal" if args.gate == "FORMAL" else "stage7"
        model.text_model.save_pretrained(temporary / "text_model", safe_serialization=False)
        tokenizer.save_pretrained(temporary / "tokenizer")
        torch.save(_trainable_state(model), temporary / "audio_bridge.pt")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": int(step),
                "epoch": int(epoch),
                "batch_in_epoch": int(batch_in_epoch),
                "rng_states_by_rank": rng_states_by_rank,
                "optimizer_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
                "scheduler_name": "cosine_lambda",
                "gate": str(args.gate),
                "run_kind": run_kind,
            },
            temporary / "training_state.pt",
        )
        config = _checkpoint_config(
            args,
            model,
            step,
            epoch,
            batch_in_epoch,
            total_steps,
            manifest_hashes,
            resume_from=resume_from,
            resume_start_step=resume_start_step,
            run_target_step=run_target_step,
            parameter_change_audit=parameter_change_audit,
        )
        (temporary / "audio_smollm2_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
        required_names = [
            "text_model/config.json",
            "tokenizer/tokenizer_config.json",
            "audio_bridge.pt",
            "training_state.pt",
            "audio_smollm2_config.json",
            "checkpoint_complete.json",
        ]
        marker = {
            "status": "complete",
            "artifact_contract": "audio_smollm2_135m_mellow_composite_v1",
            "gate": str(args.gate),
            "run_kind": run_kind,
            "global_step": int(step),
            "required": required_names,
        }
        (temporary / "checkpoint_complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        required = tuple(temporary / name for name in required_names)
        if any(not item.exists() for item in required):
            raise RuntimeError("refusing to publish incomplete baseline checkpoint")
        temporary.replace(path)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def _validate_parameter_change_audit(audit: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(audit, dict) or audit.get("all_groups_changed") is not True:
        raise RuntimeError("baseline resume checkpoint is missing a successful parameter-change audit")
    for group in ("text", "bridge", "c2l"):
        item = audit.get(group)
        if not isinstance(item, dict):
            raise RuntimeError(f"baseline parameter-change audit missing group {group}")
        try:
            delta = float(item["max_abs_delta"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"baseline parameter-change audit has invalid delta for {group}") from exc
        if (
            not math.isfinite(delta)
            or delta <= 0.0
            or item.get("finite") is not True
            or item.get("exact_equal") is not False
            or item.get("changed") is not True
        ):
            raise RuntimeError(f"baseline parameter-change audit did not prove a finite nonzero {group} update: {item}")
        if not item.get("name"):
            raise RuntimeError(f"baseline parameter-change audit has no parameter name for {group}")
    return audit


def _audit_saved_checkpoint(path: Path) -> dict[str, Any]:
    required = (
        "text_model/config.json",
        "tokenizer/tokenizer_config.json",
        "audio_bridge.pt",
        "training_state.pt",
        "audio_smollm2_config.json",
        "checkpoint_complete.json",
    )
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise RuntimeError(f"baseline checkpoint missing files: {missing}")
    marker = json.loads((path / "checkpoint_complete.json").read_text(encoding="utf-8"))
    if marker.get("status") != "complete" or marker.get("artifact_contract") != "audio_smollm2_135m_mellow_composite_v1":
        raise RuntimeError("baseline checkpoint completion marker is invalid")
    if marker.get("required") != list(required):
        raise RuntimeError(f"baseline checkpoint marker.required is invalid: {marker.get('required')!r}")
    config = json.loads((path / "audio_smollm2_config.json").read_text(encoding="utf-8"))
    required_config = (
        "gate", "run_kind", "architecture_contract", "mapper_contract", "text_model_source_path", "text_model_runtime_class", "text_model_config_sha256", "text_model_config", "embedding_lm_head_tied", "text_model_type", "text_model_hidden_size",
        "text_model_num_hidden_layers", "audio_tokens_per_clip", "audio_prefix_tokens_with_separators",
        "sample_rate", "audio_seconds", "train_manifest", "val_manifest", "manifest_sha256", "manifest_hashes", "htsat_checkpoint", "mellow_root", "mellow_provenance",
        "world_size", "micro_batch_size", "gradient_accumulation_steps", "epochs", "max_lr", "min_lr", "scheduler",
        "warmup_steps", "total_steps", "schedule_total_steps", "run_target_step", "execution_max_steps", "resume_from", "resume_start_step", "parent_checkpoint_global_step", "parameter_change_audit",
        "global_step", "epoch", "batch_in_epoch", "seed", "num_workers",
        "optimizer", "optimizer_betas", "weight_decay", "gradient_clip_norm", "autocast_dtype",
        "ddp_broadcast_buffers", "ddp_find_unused_parameters", "max_prompt_tokens", "max_answer_tokens", "max_context_length",
        "effective_global_batch_size", "save_every", "checkpoint_retention", "trainable_parameter_names", "frozen_audio_encoder", "periodic_validation",
    )
    absent = [key for key in required_config if key not in config]
    if absent:
        raise RuntimeError(f"baseline checkpoint config missing {absent}")
    expected_run_kinds = {"STAGE7": "stage7", "FORMAL": "formal"}
    if config["gate"] not in expected_run_kinds or config["run_kind"] != expected_run_kinds[config["gate"]]:
        raise RuntimeError(f"baseline checkpoint gate/run_kind mismatch: gate={config['gate']!r} run_kind={config['run_kind']!r}")
    if marker.get("gate") != config["gate"] or marker.get("run_kind") != config["run_kind"]:
        raise RuntimeError("baseline checkpoint marker gate/run_kind disagrees with config")
    if config["architecture_contract"] != ORIGINAL_SMOLLM2_CONTRACT or config["mapper_contract"] != MAPPER_CONTRACT:
        raise RuntimeError("baseline checkpoint architecture/mapper contract mismatch")
    if not Path(str(config["text_model_source_path"])).is_absolute():
        raise RuntimeError("baseline checkpoint text_model_source_path must be absolute")
    for path_key in ("train_manifest", "val_manifest", "htsat_checkpoint", "mellow_root"):
        if not Path(str(config[path_key])).is_absolute():
            raise RuntimeError(f"baseline checkpoint {path_key} must be absolute")
    if config["text_model_type"] != "llama" or int(config["text_model_hidden_size"]) != SMOLLM2_HIDDEN_SIZE or int(config["text_model_num_hidden_layers"]) != 30:
        raise RuntimeError("baseline checkpoint standard text architecture contract mismatch")
    if int(config["audio_tokens_per_clip"]) != AUDIO_TOKENS_PER_CLIP or int(config["audio_prefix_tokens_with_separators"]) != AUDIO_PREFIX_TOKENS:
        raise RuntimeError("baseline checkpoint audio shape contract mismatch")
    if int(config["sample_rate"]) != 32000 or int(config["audio_seconds"]) != 10:
        raise RuntimeError("baseline checkpoint waveform contract mismatch")
    if not isinstance(config["text_model_config"], dict) or not config["text_model_config_sha256"]:
        raise RuntimeError("baseline checkpoint text-model config provenance is missing its SHA256")
    if not isinstance(config["manifest_hashes"], dict) or "val" not in config["manifest_hashes"]:
        raise RuntimeError("baseline checkpoint text/manifest provenance has invalid types")
    if config["manifest_hashes"].get("train") != config["manifest_sha256"]:
        raise RuntimeError("baseline checkpoint train manifest hash fields disagree")
    if config["manifest_hashes"].get("val") is not None and not isinstance(config["manifest_hashes"].get("val"), str):
        raise RuntimeError("baseline checkpoint validation manifest hash must be a string or null")
    if not isinstance(config["mellow_provenance"], dict) or not config["mellow_provenance"].get("mellow_htsat_sha256"):
        raise RuntimeError("baseline checkpoint Mellow provenance is missing mellow_htsat_sha256")
    if not isinstance(config["trainable_parameter_names"], list) or not config["trainable_parameter_names"] or any(not isinstance(name, str) for name in config["trainable_parameter_names"]):
        raise RuntimeError("baseline checkpoint trainable_parameter_names is invalid")
    if any(
        name.startswith("htsat_backbone.")
        or (name.startswith("htsat_wrapper.") and not name.startswith("htsat_wrapper.c2l."))
        for name in config["trainable_parameter_names"]
    ):
        raise RuntimeError("baseline checkpoint trainable_parameter_names contains frozen HTSAT parameters")
    if config["frozen_audio_encoder"] is not True or config["periodic_validation"] is not False:
        raise RuntimeError("baseline checkpoint frozen-audio/validation contract mismatch")
    expected_effective_batch = int(config["micro_batch_size"]) * int(config["world_size"]) * int(config["gradient_accumulation_steps"])
    if int(config["effective_global_batch_size"]) != expected_effective_batch:
        raise RuntimeError("baseline checkpoint effective_global_batch_size is inconsistent")
    if int(config["save_every"]) <= 0 or int(config["checkpoint_retention"]) <= 0:
        raise RuntimeError("baseline checkpoint save cadence/retention must be positive")
    if int(config["schedule_total_steps"]) != int(config["total_steps"]):
        raise RuntimeError("baseline checkpoint scheduler length disagrees with total_steps")
    if int(config["schedule_total_steps"]) < int(config["run_target_step"]):
        raise RuntimeError("baseline checkpoint scheduler is shorter than its execution target")
    if int(config["run_target_step"]) < int(config["global_step"]):
        raise RuntimeError("baseline checkpoint global_step exceeds its run target")
    if int(config["execution_max_steps"]) != int(config["run_target_step"]):
        raise RuntimeError("baseline checkpoint execution_max_steps disagrees with run_target_step")
    if config["gate"] == "FORMAL":
        formal_expected = {
            "world_size": 8,
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "effective_global_batch_size": 256,
            "epochs": 3,
            "max_lr": 1e-3,
            "min_lr": 0.0,
            "save_every": 500,
            "checkpoint_retention": 4,
        }
        for key, expected in formal_expected.items():
            if config[key] != expected:
                raise RuntimeError(f"FORMAL checkpoint canonical contract mismatch for {key}: {config[key]!r} != {expected!r}")
        expected_warmup = math.ceil(int(config["total_steps"]) * 0.05)
        if int(config["warmup_steps"]) != expected_warmup or int(config["schedule_total_steps"]) != int(config["run_target_step"]):
            raise RuntimeError("FORMAL checkpoint warmup/schedule/run-target contract mismatch")
    if config["resume_from"] is None:
        if int(config["resume_start_step"]) != 0 or config["parent_checkpoint_global_step"] is not None:
            raise RuntimeError("new baseline checkpoint has invalid resume lineage")
    else:
        if not Path(str(config["resume_from"])).is_absolute():
            raise RuntimeError("baseline resume_from lineage must be an absolute path")
        if int(config["resume_start_step"]) <= 0 or int(config["parent_checkpoint_global_step"]) != int(config["resume_start_step"]):
            raise RuntimeError("baseline resumed checkpoint has invalid parent step lineage")
        _validate_parameter_change_audit(config["parameter_change_audit"])
    expected_fixed = {
        "optimizer": "AdamW",
        "scheduler": "cosine_lambda",
        "optimizer_betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "gradient_clip_norm": 0.5,
        "autocast_dtype": "bfloat16",
        "ddp_broadcast_buffers": False,
        "ddp_find_unused_parameters": False,
        "max_prompt_tokens": 129,
        "max_answer_tokens": 250,
        "max_context_length": 768,
    }
    for key, expected in expected_fixed.items():
        if config[key] != expected:
            raise RuntimeError(f"baseline checkpoint fixed contract mismatch for {key}: {config[key]!r} != {expected!r}")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    for key in ("optimizer", "scheduler", "scheduler_name", "global_step", "epoch", "batch_in_epoch", "rng_states_by_rank", "optimizer_parameter_names", "gate", "run_kind"):
        if key not in state:
            raise RuntimeError(f"baseline training_state missing {key}")
    if state["scheduler_name"] != config["scheduler"]:
        raise RuntimeError("baseline training_state scheduler disagrees with config")
    if state["gate"] != config["gate"] or state["run_kind"] != config["run_kind"]:
        raise RuntimeError("baseline training_state gate/run_kind disagrees with config")
    if list(state["optimizer_parameter_names"]) != list(config["trainable_parameter_names"]):
        raise RuntimeError("baseline training_state optimizer coverage disagrees with config")
    expected_rng = {str(index) for index in range(int(config["world_size"]))}
    actual_rng = {str(key) for key in state["rng_states_by_rank"]}
    if actual_rng != expected_rng:
        raise RuntimeError(f"baseline RNG ranks mismatch: expected={sorted(expected_rng)} actual={sorted(actual_rng)}")
    step_values = {int(marker.get("global_step", -1)), int(config.get("global_step", -1)), int(state["global_step"])}
    if len(step_values) != 1:
        raise RuntimeError(f"baseline checkpoint step mismatch: marker/config/training={step_values}")
    if int(state["global_step"]) <= 0:
        raise RuntimeError("baseline checkpoint global_step must be positive")
    if int(state["batch_in_epoch"]) % int(config["gradient_accumulation_steps"]) != 0:
        raise RuntimeError("baseline checkpoint batch cursor is not aligned to gradient accumulation")
    if not state["optimizer"].get("state"):
        raise RuntimeError("baseline optimizer state is empty; checkpoint cannot prove restoration")
    if not state["scheduler"]:
        raise RuntimeError("baseline scheduler state is empty; checkpoint cannot prove restoration")
    audio = torch.load(path / "audio_bridge.pt", map_location="cpu", weights_only=False)
    if not isinstance(audio.get("bridge"), dict) or not audio["bridge"] or not isinstance(audio.get("c2l"), dict) or not audio["c2l"]:
        raise RuntimeError("baseline checkpoint missing non-empty bridge/c2l state")
    return {
        "passed": True,
        "path": str(path),
        "global_step": int(state["global_step"]),
        "epoch": int(state["epoch"]),
        "batch_in_epoch": int(state["batch_in_epoch"]),
        "rng_ranks": sorted(actual_rng),
        "required_files": list(required),
        "optimizer_state_entries": len(state["optimizer"].get("state", {})),
        "scheduler_keys": sorted(state["scheduler"]),
        "gate": config["gate"],
        "run_kind": config["run_kind"],
    }


def _validate_resume_artifacts(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"resume checkpoint directory not found: {path}")
    _audit_saved_checkpoint(path)
    return json.loads((path / "audio_smollm2_config.json").read_text(encoding="utf-8"))


def _prune_checkpoints(output_dir: Path, retention: int) -> list[str]:
    if retention <= 0:
        raise ValueError("checkpoint_retention must be positive")
    output_resolved = output_dir.resolve()
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


def _validate_optimizer_coverage(state: dict[str, Any], model: AudioSmolLM2Model) -> None:
    saved_names = list(state.get("optimizer_parameter_names", []))
    current_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if saved_names != current_names:
        raise RuntimeError("checkpoint optimizer parameter coverage differs from the current trainable model")
    if any(name.startswith("htsat_backbone.") or name.startswith("htsat_wrapper.") and not name.startswith("htsat_wrapper.c2l.") for name in saved_names):
        raise RuntimeError("checkpoint optimizer unexpectedly contains frozen HTSAT parameters")


def _select_resume_representatives(model: AudioSmolLM2Model) -> dict[str, tuple[str, torch.nn.Parameter]]:
    """Select three small, stable tensors for proving a real resume update."""
    preferred = {
        "text": "text_model.model.layers.0.self_attn.q_proj.weight",
        "bridge": "bridge.linear1.weight",
        "c2l": "htsat_wrapper.c2l.weight",
    }
    named = dict(model.named_parameters())
    selected: dict[str, tuple[str, torch.nn.Parameter]] = {}
    for group, name in preferred.items():
        parameter = named.get(name)
        if parameter is None or not parameter.requires_grad:
            raise RuntimeError(f"resume parameter-change audit representative is unavailable: {group}={name}")
        selected[group] = (name, parameter)
    return selected


def _snapshot_resume_representatives(selected: dict[str, tuple[str, torch.nn.Parameter]]) -> dict[str, dict[str, Any]]:
    return {
        group: {"name": name, "before": parameter.detach().cpu().clone()}
        for group, (name, parameter) in selected.items()
    }


def _verify_resume_representative_gradients(selected: dict[str, tuple[str, torch.nn.Parameter]]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for group, (name, parameter) in selected.items():
        finite = parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        if not finite:
            raise RuntimeError(f"resume parameter-change representative has no finite first-update gradient: {group}={name}")
        result[group] = finite
    return result


def _compute_resume_parameter_change_audit(
    selected: dict[str, tuple[str, torch.nn.Parameter]],
    snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for group, (name, parameter) in selected.items():
        before = snapshots[group]["before"]
        after = parameter.detach().cpu()
        delta_tensor = (after.float() - before.float()).abs()
        max_abs_delta = float(delta_tensor.max().item()) if delta_tensor.numel() else 0.0
        exact_equal = bool(torch.equal(after, before))
        changed = bool((not exact_equal) and max_abs_delta > 0.0)
        audit[group] = {
            "name": name,
            "max_abs_delta": max_abs_delta,
            "exact_equal": exact_equal,
            "changed": changed,
            "finite": bool(math.isfinite(max_abs_delta)),
        }
    audit["all_groups_changed"] = all(bool(audit[group]["changed"]) for group in ("text", "bridge", "c2l"))
    return audit


def _make_scheduler(optimizer: torch.optim.Optimizer, *, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    if max_lr <= 0 or min_lr < 0 or min_lr > max_lr:
        raise ValueError("learning rates must satisfy 0 <= min_lr <= max_lr and max_lr > 0")

    def scale(step: int) -> float:
        if step < warmup_steps:
            return min(1.0, float(step + 1) / max(1, warmup_steps))
        progress = min(1.0, max(0.0, float(step + 1 - warmup_steps) / max(1, total_steps - warmup_steps)))
        ratio = min_lr / max_lr
        return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {"torch": torch.get_rng_state().cpu(), "cuda": torch.cuda.get_rng_state(device).cpu(), "python": random.getstate()}


def _restore_rng_state(state: dict[str, Any] | None, device: torch.device) -> None:
    if not state:
        return
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"], device=device)
    if state.get("python") is not None:
        random.setstate(state["python"])


def _gather_rng_states(world: int, device: torch.device) -> dict[str, Any]:
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


def _check_answer_labels(model: AudioSmolLM2Model, batch: dict[str, Any]) -> None:
    labels = model.last_labels
    if labels is None:
        raise RuntimeError("baseline forward did not expose labels")
    prefix_length = int(model.last_prefix_length or 0)
    text_ids = batch["text_ids"]
    for row_index in range(text_ids.shape[0]):
        prompt_length = int(batch["prompt_lengths"][row_index].item())
        answer_length = int(batch["answer_lengths"][row_index].item())
        start = prefix_length + prompt_length
        end = start + answer_length
        if bool((labels[row_index, :start] != -100).any()) or bool((labels[row_index, end:] != -100).any()):
            raise RuntimeError("baseline answer-only labels contain non-answer supervision")
        if not torch.equal(labels[row_index, start:end], text_ids[row_index, prompt_length:prompt_length + answer_length]):
            raise RuntimeError("baseline answer-only labels are misaligned")


def _actual_resume_audit(path: Path, args: argparse.Namespace, batch_cpu: dict[str, Any], device: torch.device, expected_step: int, expected_lr: float) -> dict[str, Any]:
    saved_rng = _rng_state(device)
    reload_args = copy.copy(args)
    reload_args.resume_from = path
    reload_model, _ = _load_model(reload_args, device)
    try:
        saved_config = json.loads((path / "audio_smollm2_config.json").read_text(encoding="utf-8"))
        optimizer = torch.optim.AdamW(
            [parameter for parameter in reload_model.parameters() if parameter.requires_grad],
            lr=float(saved_config["max_lr"]),
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        scheduler = _make_scheduler(
            optimizer,
            max_lr=float(saved_config["max_lr"]),
            min_lr=float(saved_config["min_lr"]),
            warmup_steps=int(saved_config["warmup_steps"]),
            total_steps=max(1, int(saved_config["total_steps"])),
        )
        state = _load_training_state(path, optimizer, scheduler)
        _validate_optimizer_coverage(state, reload_model)
        if int(state["global_step"]) != int(expected_step):
            raise RuntimeError(f"reloaded global step mismatch: {state['global_step']} != {expected_step}")
        loaded_lr = float(optimizer.param_groups[0]["lr"])
        if not math.isclose(loaded_lr, float(expected_lr), rel_tol=1e-6, abs_tol=1e-10):
            raise RuntimeError(f"reloaded learning-rate mismatch: {loaded_lr} != {expected_lr}")
        rank_state = state["rng_states_by_rank"].get("0")
        if rank_state is None:
            raise RuntimeError("reloaded checkpoint has no rank-0 RNG state")
        _restore_rng_state(rank_state, device)
        reload_model.train()
        moved = _model_batch(batch_cpu, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = reload_model(**moved)
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("reloaded baseline checkpoint produced a nonfinite loss")
        _check_answer_labels(reload_model, moved)
        output.loss.backward()
        gradient_audit = reload_model.runtime_gradient_audit()
        trainable_audit = reload_model.trainable_parameter_audit()
        if not trainable_audit["training_mode_contract"]:
            raise RuntimeError(f"reloaded baseline training mode contract failed: {trainable_audit}")
        grad_norm = torch.nn.utils.clip_grad_norm_(reload_model.parameters(), 0.5, error_if_nonfinite=True)
        return {
            "passed": True,
            "global_step": int(state["global_step"]),
            "epoch": int(state["epoch"]),
            "batch_in_epoch": int(state["batch_in_epoch"]),
            "learning_rate": loaded_lr,
            "loss": float(output.loss.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "optimizer_state_loaded": bool(optimizer.state_dict()["state"]),
            "scheduler_last_epoch": int(scheduler.last_epoch),
            "forward_backward": True,
            "answer_only_labels": True,
            "standard_text_contract": reload_model.text_contract,
            "training_mode_contract": trainable_audit,
            "gradient_audit": gradient_audit,
        }
    finally:
        del reload_model
        gc.collect()
        torch.cuda.empty_cache()
        _restore_rng_state(saved_rng, device)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_launch_contract(args)
    rank, world, device = _init_dist(args)
    _seed(args.seed, rank)
    report: dict[str, Any] = {
        "stage": f"{args.gate.lower()}_audio_smollm2_135m_mellow",
        "status": "FAIL",
        "configuration": vars(args),
        "rank": rank,
        "world_size": world,
        "checks": [],
        "warnings": [],
        "hard_failures": [],
    }
    try:
        if world != args.world_size:
            raise RuntimeError(f"world size mismatch: launcher={world} requested={args.world_size}")
        if args.save_every <= 0 or args.checkpoint_retention <= 0:
            raise ValueError("save_every and checkpoint_retention must be positive")
        if args.schedule_total_steps is not None and args.schedule_total_steps <= 0:
            raise ValueError("schedule-total-steps must be positive")
        if args.gate == "FORMAL":
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
                raise ValueError("FORMAL does not accept --max-steps; use bounded STAGE7 smoke")
        elif args.max_steps is not None and args.max_steps <= 0:
            raise ValueError("bounded smoke --max-steps must be positive")
        if not args.train_manifest.is_file():
            raise FileNotFoundError(f"train manifest not found: {args.train_manifest}")
        model, tokenizer = _load_model(args, device)
        model.train()
        if not model.trainable_parameter_audit()["training_mode_contract"]:
            raise RuntimeError(f"audio baseline training mode contract failed: {model.trainable_parameter_audit()}")
        dataset = ReasonAQADataset(args.train_manifest, tokenizer)
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True, seed=args.seed)
        loader = DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=lambda rows: collate_reasonaqa(rows, tokenizer),
        )
        if args.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        steps_epoch = len(loader) // args.gradient_accumulation_steps
        dropped_microbatches = len(loader) % args.gradient_accumulation_steps
        if steps_epoch <= 0:
            raise ValueError(f"loader has {len(loader)} batches, fewer than one accumulation window")
        formal_steps = steps_epoch * args.epochs
        max_steps = args.max_steps or (10 if args.gate == "STAGE7" else formal_steps)
        if args.gate == "FORMAL":
            if args.schedule_total_steps is not None and args.schedule_total_steps != formal_steps:
                raise ValueError(f"FORMAL schedule-total-steps must equal actual total steps {formal_steps}")
            schedule_total_steps = formal_steps
        else:
            schedule_total_steps = args.schedule_total_steps or max_steps
            if schedule_total_steps < max_steps:
                raise ValueError(f"schedule-total-steps={schedule_total_steps} must be >= execution max-steps={max_steps}")
        args.schedule_total_steps = int(schedule_total_steps)
        total_steps = int(schedule_total_steps)
        required_warmup = math.ceil(total_steps * 0.05)
        if args.gate == "FORMAL" and args.warmup_steps is not None and args.warmup_steps != required_warmup:
            raise ValueError(f"FORMAL warmup must equal ceil(actual_total_steps*0.05)={required_warmup}, got {args.warmup_steps}")
        args.warmup_steps = required_warmup
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.max_lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        scheduler = _make_scheduler(optimizer, max_lr=args.max_lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps, total_steps=total_steps)
        start_step, start_epoch, start_batch_in_epoch = 0, 0, 0
        manifest_hashes = _manifest_hashes(args)
        if args.resume_from:
            saved_config = _validate_resume_artifacts(args.resume_from)
            expected_run_kind = "formal" if args.gate == "FORMAL" else "stage7"
            if saved_config.get("gate") != args.gate or saved_config.get("run_kind") != expected_run_kind:
                raise RuntimeError(
                    "resume gate/run_kind mismatch: "
                    f"saved=({saved_config.get('gate')!r}, {saved_config.get('run_kind')!r}) "
                    f"current=({args.gate!r}, {expected_run_kind!r}); cross-gate resume is forbidden "
                    "(STAGE7 checkpoints cannot resume FORMAL)"
                )
            if args.gate == "STAGE7" and max_steps == 22 and args.expected_resume_step is None:
                raise RuntimeError("the 20-to-22 resume wrapper must declare --expected-resume-step 20")
            if args.expected_resume_step is not None and int(saved_config["global_step"]) != int(args.expected_resume_step):
                raise RuntimeError(f"resume parent global_step={saved_config['global_step']} != expected {args.expected_resume_step}")
            if int(saved_config["schedule_total_steps"]) != int(total_steps) or int(saved_config["total_steps"]) != int(total_steps):
                raise RuntimeError(f"resume scheduler length mismatch: saved={saved_config['schedule_total_steps']} current={total_steps}")
            if saved_config.get("scheduler") != "cosine_lambda":
                raise RuntimeError(f"resume scheduler kind mismatch: saved={saved_config.get('scheduler')!r}")
            if saved_config["manifest_sha256"] != manifest_hashes["train"]:
                raise RuntimeError("resume train manifest SHA256 mismatch")
            if saved_config.get("manifest_hashes", {}).get("val") != manifest_hashes["val"]:
                raise RuntimeError("resume validation manifest SHA256 mismatch")
            for key in ("world_size", "micro_batch_size", "gradient_accumulation_steps", "epochs", "max_lr", "min_lr", "save_every", "checkpoint_retention", "seed", "num_workers"):
                if str(saved_config[key]) != str(getattr(args, key)):
                    raise RuntimeError(f"resume {key} mismatch: saved={saved_config[key]} current={getattr(args, key)}")
            if int(saved_config["warmup_steps"]) != int(args.warmup_steps):
                raise RuntimeError(f"resume warmup_steps mismatch: saved={saved_config['warmup_steps']} current={args.warmup_steps}")
            if args.gate == "FORMAL" and int(saved_config["total_steps"]) != int(total_steps):
                raise RuntimeError(f"FORMAL resume total_steps mismatch: saved={saved_config['total_steps']} current={total_steps}")
            if args.gate == "FORMAL":
                if int(saved_config["run_target_step"]) != int(formal_steps) or int(saved_config["execution_max_steps"]) != int(formal_steps):
                    raise RuntimeError(
                        "FORMAL resume checkpoint is not a full canonical three-epoch run: "
                        f"run_target={saved_config['run_target_step']} execution_max={saved_config['execution_max_steps']} expected={formal_steps}"
                    )
            if Path(str(saved_config["htsat_checkpoint"])).resolve() != args.htsat_checkpoint.resolve():
                raise RuntimeError("resume HTSAT checkpoint mismatch")
            if Path(str(saved_config["mellow_root"])).resolve() != args.mellow_root.resolve():
                raise RuntimeError("resume Mellow root mismatch")
            saved_mellow_sha = saved_config.get("mellow_provenance", {}).get("mellow_htsat_sha256")
            current_mellow_sha = getattr(model, "_audio_provenance", {}).get("mellow_htsat_sha256")
            if not saved_mellow_sha or current_mellow_sha != saved_mellow_sha:
                raise RuntimeError("resume Mellow source SHA256 mismatch")
            saved_source = Path(str(saved_config["text_model_source_path"])).resolve()
            current_source = args.model_path.resolve()
            if saved_source != current_source:
                raise RuntimeError(f"resume text model source mismatch: saved={saved_source} current={current_source}")
            saved_source_config_sha = saved_config.get("text_model_config_sha256")
            current_source_config = current_source / "config.json"
            if not saved_source_config_sha:
                raise RuntimeError("resume checkpoint has no text model config SHA256 provenance")
            if not current_source_config.is_file() or _sha256(current_source_config) != saved_source_config_sha:
                raise RuntimeError("resume source model config SHA256 mismatch")
            if saved_config.get("architecture_contract") != ORIGINAL_SMOLLM2_CONTRACT:
                raise RuntimeError("resume original-text architecture contract mismatch")
            if model.text_contract["model_type"] != "llama" or model.text_contract["num_hidden_layers"] != 30:
                raise RuntimeError("resume loaded a non-standard text model")
            fixed_contract = {
                "optimizer": "AdamW",
                "scheduler": "cosine_lambda",
                "optimizer_betas": [0.9, 0.95],
                "weight_decay": 0.1,
                "gradient_clip_norm": 0.5,
                "autocast_dtype": "bfloat16",
                "ddp_broadcast_buffers": False,
                "ddp_find_unused_parameters": False,
                "audio_tokens_per_clip": AUDIO_TOKENS_PER_CLIP,
                "audio_prefix_tokens_with_separators": AUDIO_PREFIX_TOKENS,
                "sample_rate": 32000,
                "audio_seconds": 10,
                "max_prompt_tokens": 129,
                "max_answer_tokens": 250,
                "max_context_length": 768,
                "frozen_audio_encoder": True,
                "periodic_validation": False,
            }
            for key, expected in fixed_contract.items():
                if saved_config.get(key) != expected:
                    raise RuntimeError(f"resume fixed contract mismatch for {key}: saved={saved_config.get(key)!r} expected={expected!r}")
            state = _load_training_state(args.resume_from, optimizer, scheduler)
            _validate_optimizer_coverage(state, model)
            if state.get("gate") != args.gate or state.get("run_kind") != expected_run_kind:
                raise RuntimeError("resume training_state gate/run_kind is not compatible with the requested gate")
            if state.get("scheduler_name") != "cosine_lambda":
                raise RuntimeError("resume training_state scheduler_name is not cosine_lambda")
            start_step = int(state["global_step"])
            start_epoch = int(state.get("epoch", 0))
            start_batch_in_epoch = int(state.get("batch_in_epoch", 0))
            if start_batch_in_epoch % args.gradient_accumulation_steps != 0:
                raise RuntimeError("resume batch cursor is not aligned to gradient accumulation")
            if str(rank) not in state.get("rng_states_by_rank", {}):
                raise RuntimeError(f"resume checkpoint has no RNG state for rank {rank}")
            _restore_rng_state(state["rng_states_by_rank"][str(rank)], device)
        resume_representatives: dict[str, tuple[str, torch.nn.Parameter]] | None = None
        resume_snapshots: dict[str, dict[str, Any]] | None = None
        resume_gradient_verification: dict[str, bool] | None = None
        if args.resume_from:
            resume_representatives = _select_resume_representatives(model)
            resume_snapshots = _snapshot_resume_representatives(resume_representatives)
        ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False) if world > 1 else model
        owner = ddp.module if hasattr(ddp, "module") else ddp
        metrics: list[dict[str, Any]] = []
        runtime_gradient_audit: dict[str, Any] | None = None
        optimizer_step = start_step
        epoch = start_epoch
        batch_in_epoch = start_batch_in_epoch
        parameter_change_audit: dict[str, Any] | None = None
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
                batch_cpu: dict[str, Any] | None = None
                for micro in range(args.gradient_accumulation_steps):
                    batch = next(data_iter)
                    if micro == args.gradient_accumulation_steps - 1:
                        batch_cpu = {key: (value.detach().cpu().clone() if torch.is_tensor(value) else value) for key, value in batch.items()}
                    moved = _model_batch(batch, device)
                    sync = contextlib.nullcontext() if not hasattr(ddp, "no_sync") or micro == args.gradient_accumulation_steps - 1 else ddp.no_sync()
                    with sync:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            output = ddp(**moved)
                        if output.loss is None or not torch.isfinite(output.loss):
                            raise RuntimeError("nonfinite original SmolLM2 audio loss")
                        (output.loss / args.gradient_accumulation_steps).backward()
                if batch_cpu is None:
                    raise RuntimeError("missing last microbatch for checkpoint audit")
                grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 0.5, error_if_nonfinite=True)
                if resume_representatives is not None and resume_gradient_verification is None:
                    resume_gradient_verification = _verify_resume_representative_gradients(resume_representatives)
                if runtime_gradient_audit is None:
                    runtime_gradient_audit = owner.runtime_gradient_audit()
                lr_before_optimizer_step = float(optimizer.param_groups[0]["lr"])
                next_optimizer_step = optimizer_step + 1
                if args.resume_from is not None and next_optimizer_step == 21 and (
                    not math.isfinite(lr_before_optimizer_step) or lr_before_optimizer_step <= 0.0
                ):
                    raise RuntimeError(
                        "resume scheduler contract failed: step 21 must have a nonzero "
                        f"lr_before_optimizer_step, got {lr_before_optimizer_step}"
                    )
                optimizer_step += 1
                optimizer.step()
                scheduler.step()
                elapsed = max(time.perf_counter() - step_started, 1e-9)
                global_samples = int(args.micro_batch_size * world * args.gradient_accumulation_steps)
                answer_tokens = torch.tensor(int(batch["answer_attention_mask"].sum().item()), dtype=torch.long, device=device)
                if world > 1:
                    dist.all_reduce(answer_tokens, op=dist.ReduceOp.SUM)
                item = {
                    "step": optimizer_step,
                    "total_steps": max_steps,
                    "schedule_total_steps": total_steps,
                    "progress_percent": 100.0 * optimizer_step / max(1, max_steps),
                    "epoch": epoch,
                    "batch_in_epoch": batch_in_epoch,
                    "steps_per_epoch": steps_epoch,
                    "loss": float(output.loss.detach().cpu()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "lr_before_optimizer_step": lr_before_optimizer_step,
                    "grad_norm": float(grad_norm),
                    "effective_answer_tokens": int(answer_tokens.item()),
                    "step_time_seconds": elapsed,
                    "samples_per_second": global_samples / elapsed,
                    "audio_seconds_per_second": global_samples * 20.0 / elapsed,
                    "gpu_memory_allocated_gib": float(torch.cuda.memory_allocated(device) / 1024**3),
                    "gpu_memory_reserved_gib": float(torch.cuda.memory_reserved(device) / 1024**3),
                    "gpu_memory_max_allocated_gib": float(torch.cuda.max_memory_allocated(device) / 1024**3),
                    "gpu_memory_max_reserved_gib": float(torch.cuda.max_memory_reserved(device) / 1024**3),
                }
                metrics.append(item)
                batch_in_epoch += args.gradient_accumulation_steps
                if rank == 0 and (optimizer_step % 10 == 0 or optimizer_step == max_steps):
                    print(
                        f"[audio-smollm2] step={optimizer_step}/{max_steps} progress={item['progress_percent']:.2f}% "
                        f"epoch={epoch + 1}/{args.epochs if args.gate == 'FORMAL' else '?'} "
                        f"batch={batch_in_epoch}/{len(loader)} loss={item['loss']:.6f} lr={item['lr']:.8g} "
                        f"step_s={item['step_time_seconds']:.3f} samples/s={item['samples_per_second']:.2f} "
                        f"answer_tokens={item['effective_answer_tokens']} gpu_max_alloc_gib={item['gpu_memory_max_allocated_gib']:.3f}",
                        flush=True,
                    )
                save_due = (
                    args.gate == "FORMAL" and (optimizer_step % max(1, args.save_every) == 0 or optimizer_step == max_steps)
                ) or (
                    args.gate == "STAGE7" and (optimizer_step == 10 or optimizer_step == max_steps)
                )
                if save_due:
                    if resume_representatives is not None and resume_snapshots is not None:
                        parameter_change_audit = _compute_resume_parameter_change_audit(resume_representatives, resume_snapshots)
                        _validate_parameter_change_audit(parameter_change_audit)
                    out = args.output_dir / f"checkpoint-{optimizer_step:06d}"
                    rng_states = _gather_rng_states(world, device)
                    if rank == 0:
                        _save_checkpoint(
                            out,
                            owner,
                            tokenizer,
                            optimizer,
                            scheduler,
                            optimizer_step,
                            epoch,
                            batch_in_epoch,
                            args,
                            manifest_hashes,
                            rng_states,
                            total_steps,
                            resume_from=args.resume_from,
                            resume_start_step=start_step,
                            run_target_step=max_steps,
                            parameter_change_audit=parameter_change_audit,
                        )
                    if world > 1:
                        dist.barrier()
                    if rank == 0:
                        report.setdefault("checkpoints", []).append(str(out))
                        artifact_audit = _audit_saved_checkpoint(out)
                        if args.gate == "STAGE7":
                            reload_audit = {
                                "checkpoint": str(out),
                                "artifact": artifact_audit,
                                "actual_resume": _actual_resume_audit(out, args, batch_cpu, device, optimizer_step, float(optimizer.param_groups[0]["lr"])),
                            }
                            report.setdefault("checkpoint_reload_audits", []).append(reload_audit)
                            report["checkpoint_reload_audit"] = reload_audit
                        elif args.gate == "FORMAL":
                            report["checkpoints"] = _prune_checkpoints(args.output_dir, args.checkpoint_retention)
                    if world > 1:
                        dist.barrier()
                if optimizer_step >= max_steps:
                    break
            if batch_in_epoch >= steps_epoch * args.gradient_accumulation_steps:
                epoch += 1
                batch_in_epoch = 0
        report.update({
            "status": "PASS",
            "gate": args.gate,
            "run_kind": "formal" if args.gate == "FORMAL" else "stage7",
            "start_step": start_step,
            "end_step": optimizer_step,
            "optimizer_steps": optimizer_step,
            "steps_per_epoch": steps_epoch,
            "dropped_microbatches_per_epoch": dropped_microbatches,
            "total_formal_steps": formal_steps,
            "warmup_steps": args.warmup_steps,
            "schedule_total_steps": total_steps,
            "run_target_step": max_steps,
            "resume_lr_before_optimizer_step": {
                str(item["step"]): item["lr_before_optimizer_step"]
                for item in metrics
                if int(item["step"]) in {21, 22}
            } if args.resume_from else {},
            "effective_global_batch_size": int(args.micro_batch_size * world * args.gradient_accumulation_steps),
            "metrics": metrics if rank == 0 else [],
            "ddp_broadcast_buffers": False,
            "ddp_find_unused_parameters": False,
            "model_trainable_audit": owner.trainable_parameter_audit(),
            "runtime_gradient_audit": runtime_gradient_audit,
            "resume_lineage": {
                "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
                "resume_start_step": start_step,
                "parameter_change_audit": parameter_change_audit,
                "representative_gradient_verification": resume_gradient_verification,
            },
            "resume_position": {"epoch": epoch, "batch_in_epoch": batch_in_epoch},
            "checkpoints": report.get("checkpoints", []),
        })
    except Exception as exc:
        report["hard_failures"].append({"error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.report_path or args.output_dir / f"{args.gate.lower()}_audit.json"
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({"stage": report["stage"], "status": report["status"], "summary": {"steps": report.get("optimizer_steps"), "hard_failures": len(report.get("hard_failures", []))}, "report": str(args.report_path or args.output_dir)}, default=str))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
