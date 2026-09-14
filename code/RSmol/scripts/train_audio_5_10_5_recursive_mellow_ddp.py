#!/usr/bin/env python3
"""Formal ReasonAQA trainer for the fixed two-pass 5-10-5 audio model.

This route intentionally exposes only the canonical three-epoch FORMAL gate.
It reuses the established original-SmolLM2 audio trainer implementation while
replacing the text loader, composite model, route identity, and artifact
contract with strict fixed-recursion equivalents.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import train_audio_smollm2_135m_mellow_ddp as core  # noqa: E402
from audio_5_10_5_recursive_mellow.data import (  # noqa: E402
    ReasonAQADataset,
    collate_reasonaqa,
)
from audio_5_10_5_recursive_mellow.model import (  # noqa: E402
    AUDIO_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    MAPPER_CONTRACT,
    RECURSIVE_AUDIO_CONTRACT,
    RECURSIVE_HIDDEN_SIZE,
    AudioRecursive5_10_5Config,
    AudioRecursive5_10_5Model,
    _load_mellow_wrapper,
    validate_recursive_5_10_5,
)
from recursive_model_5_10_5 import (  # noqa: E402
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


DEFAULT_RECURSIVE_CHECKPOINT = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/"
    "formal-epoch2-continue-20260902_184936/checkpoint-step-009244"
)
DEFAULT_OUTPUT = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "audio_5_10_5_recursive_mellow/manual_run"
)
CANONICAL_TRAIN_MANIFEST = core.DEFAULT_TRAIN_MANIFEST
CANONICAL_VAL_MANIFEST = core.DEFAULT_VAL_MANIFEST
CANONICAL_HTSAT_CHECKPOINT = core.DEFAULT_HTSAT
CANONICAL_MELLOW_ROOT = core.DEFAULT_MELLOW
ARTIFACT_CONTRACT = "audio_5_10_5_recursive_mellow_composite_v1"
CONFIG_FILENAME = "audio_recursive_5_10_5_config.json"
_BASE_VALIDATE_LAUNCH_CONTRACT = core._validate_launch_contract


def _load_recursive_text_backbone(model_path: Path) -> Any:
    register_auto_class()
    model = RecursiveLlamaForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
    )
    validate_recursive_5_10_5(model)
    return model


def _validate_recursive_source_contract(args: Any) -> None:
    if args.tokenizer_path is not None:
        raise ValueError(
            "fixed 5-10-5 FORMAL uses the tokenizer packaged with the canonical "
            "text checkpoint and does not accept --tokenizer-path"
        )
    expected_paths = {
        "model_path": Path(DEFAULT_RECURSIVE_CHECKPOINT).resolve(),
        "train_manifest": Path(CANONICAL_TRAIN_MANIFEST).resolve(),
        "val_manifest": Path(CANONICAL_VAL_MANIFEST).resolve(),
        "htsat_checkpoint": Path(CANONICAL_HTSAT_CHECKPOINT).resolve(),
        "mellow_root": Path(CANONICAL_MELLOW_ROOT).resolve(),
    }
    actual_paths = {
        "model_path": args.model_path.resolve(),
        "train_manifest": args.train_manifest.resolve(),
        "val_manifest": args.val_manifest.resolve(),
        "htsat_checkpoint": args.htsat_checkpoint.resolve(),
        "mellow_root": args.mellow_root.resolve(),
    }
    if actual_paths != expected_paths:
        raise RuntimeError(
            "fixed 5-10-5 FORMAL requires the canonical text/data/audio sources: "
            f"actual={actual_paths} expected={expected_paths}"
        )
    if int(args.seed) != 0 or int(args.num_workers) != 0:
        raise RuntimeError(
            "fixed 5-10-5 FORMAL requires seed=0 and num_workers=0; "
            f"got seed={args.seed} num_workers={args.num_workers}"
        )
    required_files = (args.train_manifest, args.val_manifest, args.htsat_checkpoint)
    missing_files = [str(path) for path in required_files if not path.is_file()]
    required_dirs = (args.model_path, args.mellow_root)
    missing_dirs = [str(path) for path in required_dirs if not path.is_dir()]
    if missing_files or missing_dirs:
        raise FileNotFoundError(
            "fixed 5-10-5 canonical sources are not all visible: "
            f"missing_files={missing_files} missing_dirs={missing_dirs}"
        )


def _validate_recursive_launch_contract(args: Any) -> None:
    _BASE_VALIDATE_LAUNCH_CONTRACT(args)
    _validate_recursive_source_contract(args)
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise NotADirectoryError(f"fixed 5-10-5 output path is not a directory: {args.output_dir}")
    if args.output_dir.is_dir() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            "fixed 5-10-5 FORMAL refuses a non-empty output directory: "
            f"{args.output_dir}"
        )


def _module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _after_recursive_ddp_initialization(
    model: AudioRecursive5_10_5Model,
    args: Any,
    *,
    rank: int,
    world: int,
) -> None:
    c2l = getattr(model.htsat_wrapper, "c2l", None)
    if not isinstance(c2l, torch.nn.Module):
        raise RuntimeError("fixed 5-10-5 audio model has no trainable c2l module")
    run_start_hashes = {
        "bridge_sha256": _module_state_sha256(model.bridge),
        "c2l_sha256": _module_state_sha256(c2l),
    }
    gathered: list[Any] = [None for _ in range(int(world))]
    if int(world) > 1:
        dist.all_gather_object(gathered, run_start_hashes)
    else:
        gathered[0] = run_start_hashes
    if any(item != gathered[0] for item in gathered):
        raise RuntimeError(f"DDP audio initialization hashes differ across ranks: {gathered}")
    if args.resume_from is None:
        initialization_hashes = dict(run_start_hashes)
    else:
        parent_config = json.loads(
            (args.resume_from / CONFIG_FILENAME).read_text(encoding="utf-8")
        )
        initialization_hashes = parent_config.get("audio_initialization_hashes")
        if not isinstance(initialization_hashes, dict):
            raise RuntimeError("FORMAL resume parent has no audio_initialization_hashes")
    model._audio_initialization_hashes = initialization_hashes
    model._run_start_audio_state_hashes = run_start_hashes


def _extra_recursive_checkpoint_config(model: AudioRecursive5_10_5Model) -> dict[str, Any]:
    contract = model.text_contract
    return {
        "text_model_logical_layer_count": int(contract["logical_layer_count"]),
        "text_model_physical_layer_count": int(contract["physical_decoder_layer_count"]),
        "text_model_recursive_loops": int(contract["recursive_loops"]),
        "text_model_recursive_loops_scope": str(contract["recursive_loops_scope"]),
        "text_model_prefix_layer_count": int(contract["prefix_layer_count"]),
        "text_model_middle_layer_count": int(contract["middle_layer_count"]),
        "text_model_suffix_layer_count": int(contract["suffix_layer_count"]),
        "text_model_logical_to_physical": list(contract["logical_to_physical"]),
        "text_model_source_layer_indices_0based": list(contract["source_layer_indices_0based"]),
        "text_model_has_mesh_router_or_memory": False,
        "audio_initialization_hashes": getattr(model, "_audio_initialization_hashes", None),
        "run_start_audio_state_hashes": getattr(model, "_run_start_audio_state_hashes", None),
    }


def _validate_recursive_checkpoint_config(config: dict[str, Any]) -> None:
    expected = {
        "text_model_logical_layer_count": LOGICAL_LAYER_COUNT,
        "text_model_physical_layer_count": PHYSICAL_LAYER_COUNT,
        "text_model_recursive_loops": RECURSIVE_LOOPS,
        "text_model_recursive_loops_scope": "middle_only",
        "text_model_prefix_layer_count": PREFIX_LAYER_COUNT,
        "text_model_middle_layer_count": MIDDLE_LAYER_COUNT,
        "text_model_suffix_layer_count": SUFFIX_LAYER_COUNT,
        "text_model_logical_to_physical": list(LOGICAL_TO_PHYSICAL),
        "text_model_source_layer_indices_0based": list(SOURCE_LAYER_INDICES_0BASED),
        "text_model_has_mesh_router_or_memory": False,
    }
    missing = [key for key in expected if key not in config]
    if missing:
        raise RuntimeError(f"fixed 5-10-5 checkpoint config missing {missing}")
    for key, value in expected.items():
        if config[key] != value:
            raise RuntimeError(
                f"fixed 5-10-5 checkpoint contract mismatch for {key}: "
                f"{config[key]!r} != {value!r}"
            )
    text_config = config.get("text_model_config")
    if not isinstance(text_config, dict):
        raise RuntimeError("fixed 5-10-5 checkpoint has no text_model_config mapping")
    text_config_expected = {
        "num_hidden_layers": LOGICAL_LAYER_COUNT,
        "recursive_layer_count": PHYSICAL_LAYER_COUNT,
        "recursive_loops": RECURSIVE_LOOPS,
        "recursive_loops_scope": "middle_only",
        "recursive_prefix_layer_count": PREFIX_LAYER_COUNT,
        "recursive_middle_layer_count": MIDDLE_LAYER_COUNT,
        "recursive_suffix_layer_count": SUFFIX_LAYER_COUNT,
        "logical_to_physical": list(LOGICAL_TO_PHYSICAL),
        "recursive_source_layer_indices_0based": list(SOURCE_LAYER_INDICES_0BASED),
    }
    for key, value in text_config_expected.items():
        if text_config.get(key) != value:
            raise RuntimeError(
                f"fixed 5-10-5 saved HF config mismatch for {key}: "
                f"{text_config.get(key)!r} != {value!r}"
            )
    for key in ("audio_initialization_hashes", "run_start_audio_state_hashes"):
        value = config.get(key)
        if not isinstance(value, dict) or set(value) != {"bridge_sha256", "c2l_sha256"}:
            raise RuntimeError(f"fixed 5-10-5 checkpoint has invalid {key}: {value!r}")
        if any(not isinstance(digest, str) or len(digest) != 64 for digest in value.values()):
            raise RuntimeError(f"fixed 5-10-5 checkpoint has invalid hashes in {key}: {value!r}")


def _validate_recursive_training_plan(
    args: Any,
    *,
    dataset_rows: int,
    loader_batches: int,
    steps_per_epoch: int,
    formal_steps: int,
    warmup_steps: int,
) -> None:
    expected = {
        "dataset_rows": 968059,
        "loader_batches": 15126,
        "steps_per_epoch": 3781,
        "formal_steps": 11343,
        "warmup_steps": 568,
    }
    actual = {
        "dataset_rows": int(dataset_rows),
        "loader_batches": int(loader_batches),
        "steps_per_epoch": int(steps_per_epoch),
        "formal_steps": int(formal_steps),
        "warmup_steps": int(warmup_steps),
    }
    if str(args.gate) != "FORMAL":
        raise RuntimeError(f"fixed 5-10-5 route only permits FORMAL, got {args.gate!r}")
    _validate_recursive_source_contract(args)
    if actual != expected:
        raise RuntimeError(
            "fixed 5-10-5 formal data/step contract mismatch: "
            f"actual={actual} expected={expected}"
        )


def _configure_core() -> None:
    core.TRAINER_DESCRIPTION = __doc__
    core.GATE_CHOICES = ("FORMAL",)
    core.DEFAULT_GATE = "FORMAL"
    core.MODEL_PATH_OPTIONS = ("--model-path", "--recursive-checkpoint")
    core.ROUTE_NAME = "audio_5_10_5_recursive_mellow"
    core.LOG_PREFIX = "audio-5-10-5-recursive"
    core.ARTIFACT_CONTRACT = ARTIFACT_CONTRACT
    core.CONFIG_FILENAME = CONFIG_FILENAME
    core.ARCHITECTURE_CONTRACT = RECURSIVE_AUDIO_CONTRACT
    core.DEFAULT_MODEL = DEFAULT_RECURSIVE_CHECKPOINT
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.ReasonAQADataset = ReasonAQADataset
    core.collate_reasonaqa = collate_reasonaqa
    core.AudioSmolLM2Model = AudioRecursive5_10_5Model
    core.AudioSmolLM2Config = AudioRecursive5_10_5Config
    core.SMOLLM2_HIDDEN_SIZE = RECURSIVE_HIDDEN_SIZE
    core.AUDIO_TOKENS_PER_CLIP = AUDIO_TOKENS_PER_CLIP
    core.AUDIO_PREFIX_TOKENS = AUDIO_PREFIX_TOKENS
    core.MAPPER_CONTRACT = MAPPER_CONTRACT
    core._load_mellow_wrapper = _load_mellow_wrapper
    core._load_text_backbone = _load_recursive_text_backbone
    core._validate_launch_contract = _validate_recursive_launch_contract
    core._extra_checkpoint_config = _extra_recursive_checkpoint_config
    core._validate_route_checkpoint_config = _validate_recursive_checkpoint_config
    core._validate_route_training_plan = _validate_recursive_training_plan
    core._after_ddp_initialization = _after_recursive_ddp_initialization


_configure_core()

parse_args = core.parse_args
run = core.run


def main(argv: list[str] | None = None) -> int:
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
