#!/usr/bin/env python3
"""Evaluate the x2/x3/x4/x5 7-slot and 8-slot fixed-260 Audio MeSH checkpoints on MMAU.

The benchmark traversal and official scorer are shared with the existing
MMAU evaluator.  This adapter owns the route-specific MeSH loaders and enforces
the evaluation contract requested for the formal audio checkpoints:

* FP32 inference;
* two materialized 129-token audio slots (260 tokens including separators);
* a runtime all-zero GPU waveform in slot two for every single-audio row;
* one generation pass, followed by both the Mellow prefix scorer and the
  official MMAU scorer.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as official  # noqa: E402


@dataclass(frozen=True)
class RouteSpec:
    name: str
    audio_package: str
    audio_model_module: str
    text_model_module: str
    audio_config_filename: str
    audio_model_class: str
    audio_config_class: str
    training_contract: str
    audio_architecture_contract: str
    logical_layer_count: int
    recursive_loops: int
    memory_slots: int
    router_groups: int


ROUTES: dict[str, RouteSpec] = {
    "x2_7slot": RouteSpec(
        name="x2_7slot",
        audio_package="audio_5_10x2_5_mesh_7slot_mellow_shared_store_configurable_epochs",
        audio_model_module="audio_5_10x2_5_mesh_7slot_mellow_shared_store_configurable_epochs.model",
        text_model_module="recursive_model_5_10x2_5_mesh_7slot",
        audio_config_filename="audio_mesh_x2_7slot_fixed260_zero_slot_config.json",
        audio_model_class="AudioMeshX2SevenSlotZeroModel",
        audio_config_class="AudioMeshX2SevenSlotZeroConfig",
        training_contract=(
            "node_shared_unique_store_fullshuffle_fixed260_runtime_zero_second_slot_"
            "answer_eos_x2_7slot_v1"
        ),
        audio_architecture_contract=(
            "logical_30_physical_20_5_10x2_5_mesh_7slot_3router_audio_mellow_"
            "fixed260_runtime_zero_second_slot"
        ),
        logical_layer_count=30,
        recursive_loops=2,
        memory_slots=7,
        router_groups=3,
    ),
    "x4": RouteSpec(
        name="x4",
        audio_package="audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs",
        audio_model_module="audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs.model",
        text_model_module="recursive_model_5_10x4_5_mesh",
        audio_config_filename="audio_mesh_x4_fixed260_zero_slot_config.json",
        audio_model_class="AudioMeshX4ZeroSlotModel",
        audio_config_class="AudioMeshX4ZeroSlotConfig",
        training_contract=(
            "node_shared_unique_store_fullshuffle_fixed260_runtime_zero_second_slot_"
            "answer_eos_x4_v1"
        ),
        audio_architecture_contract=(
            "logical_50_physical_20_5_10x4_5_mesh_audio_mellow_"
            "fixed260_runtime_zero_second_slot"
        ),
        logical_layer_count=50,
        recursive_loops=4,
        memory_slots=7,
        router_groups=5,
    ),
    "x4_8slot": RouteSpec(
        name="x4_8slot",
        audio_package="audio_5_10x4_5_mesh_8slot_mellow_shared_store_configurable_epochs",
        audio_model_module="audio_5_10x4_5_mesh_8slot_mellow_shared_store_configurable_epochs.model",
        text_model_module="recursive_model_5_10x4_5_mesh_8slot",
        audio_config_filename="audio_mesh_x4_8slot_fixed260_zero_slot_config.json",
        audio_model_class="AudioMeshX4EightSlotZeroModel",
        audio_config_class="AudioMeshX4EightSlotZeroConfig",
        training_contract=(
            "node_shared_unique_store_fullshuffle_fixed260_runtime_zero_second_slot_"
            "answer_eos_x4_8slot_v1"
        ),
        audio_architecture_contract=(
            "logical_50_physical_20_5_10x4_5_mesh_8slot_audio_mellow_"
            "fixed260_runtime_zero_second_slot"
        ),
        logical_layer_count=50,
        recursive_loops=4,
        memory_slots=8,
        router_groups=5,
    ),
    "x3_7slot": RouteSpec(
        name="x3_7slot",
        audio_package="audio_5_10x3_5_mesh_7slot_mellow_shared_store_configurable_epochs",
        audio_model_module="audio_5_10x3_5_mesh_7slot_mellow_shared_store_configurable_epochs.model",
        text_model_module="recursive_model_5_10x3_5_mesh_7slot",
        audio_config_filename="audio_mesh_x3_7slot_fixed260_zero_slot_config.json",
        audio_model_class="AudioMeshX3SevenSlotZeroModel",
        audio_config_class="AudioMeshX3SevenSlotZeroConfig",
        training_contract=(
            "node_shared_unique_store_fullshuffle_fixed260_runtime_zero_second_slot_"
            "answer_eos_x3_7slot_v1"
        ),
        audio_architecture_contract=(
            "logical_40_physical_20_5_10x3_5_mesh_7slot_4router_audio_mellow_"
            "fixed260_runtime_zero_second_slot"
        ),
        logical_layer_count=40,
        recursive_loops=3,
        memory_slots=7,
        router_groups=4,
    ),
    "x5_7slot": RouteSpec(
        name="x5_7slot",
        audio_package="audio_5_10x5_5_mesh_7slot_mellow_shared_store_configurable_epochs",
        audio_model_module="audio_5_10x5_5_mesh_7slot_mellow_shared_store_configurable_epochs.model",
        text_model_module="recursive_model_5_10x5_5_mesh",
        audio_config_filename="audio_mesh_x5_7slot_fixed260_zero_slot_config.json",
        audio_model_class="AudioMeshX5SevenSlotZeroModel",
        audio_config_class="AudioMeshX5SevenSlotZeroConfig",
        training_contract=(
            "node_shared_unique_store_fullshuffle_fixed260_runtime_zero_second_slot_"
            "answer_eos_x5_7slot_v1"
        ),
        audio_architecture_contract=(
            "logical_60_physical_20_5_10x5_5_mesh_7slot_6router_audio_mellow_"
            "fixed260_runtime_zero_second_slot"
        ),
        logical_layer_count=60,
        recursive_loops=5,
        memory_slots=7,
        router_groups=6,
    ),
    "x5_8slot": RouteSpec(
        name="x5_8slot",
        audio_package="audio_5_10x5_5_mesh_8slot_mellow_shared_store_configurable_epochs",
        audio_model_module="audio_5_10x5_5_mesh_8slot_mellow_shared_store_configurable_epochs.model",
        text_model_module="recursive_model_5_10x5_5_mesh_8slot",
        audio_config_filename="audio_mesh_x5_8slot_fixed260_zero_slot_config.json",
        audio_model_class="AudioMeshX5EightSlotZeroModel",
        audio_config_class="AudioMeshX5EightSlotZeroConfig",
        training_contract=(
            "node_shared_unique_store_fullshuffle_fixed260_runtime_zero_second_slot_"
            "answer_eos_x5_8slot_v1"
        ),
        audio_architecture_contract=(
            "logical_60_physical_20_5_10x5_5_mesh_8slot_6router_audio_mellow_"
            "fixed260_runtime_zero_second_slot"
        ),
        logical_layer_count=60,
        recursive_loops=5,
        memory_slots=8,
        router_groups=6,
    ),
    "x5_9slot": RouteSpec(
        name="x5_9slot",
        audio_package="audio_5_10x5_5_mesh_9slot_mellow_shared_store_configurable_epochs",
        audio_model_module="audio_5_10x5_5_mesh_9slot_mellow_shared_store_configurable_epochs.model",
        text_model_module="recursive_model_5_10x5_5_mesh_9slot",
        audio_config_filename="audio_mesh_x5_9slot_fixed260_zero_slot_config.json",
        audio_model_class="AudioMeshX5NineSlotZeroModel",
        audio_config_class="AudioMeshX5NineSlotZeroConfig",
        training_contract=(
            "node_shared_unique_store_fullshuffle_fixed260_runtime_zero_second_slot_"
            "answer_eos_x5_9slot_v1"
        ),
        audio_architecture_contract=(
            "logical_60_physical_20_5_10x5_5_mesh_9slot_6router_audio_mellow_"
            "fixed260_runtime_zero_second_slot"
        ),
        logical_layer_count=60,
        recursive_loops=5,
        memory_slots=9,
        router_groups=6,
    ),
}

DEFAULT_CHECKPOINTS = {
    "x2_7slot": (
        "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
        "audio_5_10x2_5_mesh_7slot_mellow_shared_store_configurable_epochs/"
        "formal_3epochs_20261002_metadatafix_v1/checkpoint-011343"
    ),
    "x4": (
        "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
        "audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs/"
        "formal_3epochs_20261002_configfix_v3/checkpoint-011343"
    ),
    "x3_7slot": (
        "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
        "audio_5_10x3_5_mesh_7slot_mellow_shared_store_configurable_epochs/"
        "formal_3epochs_20261003_125649710077883-20/checkpoint-011343"
    ),
    "x5_7slot": (
        "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
        "audio_5_10x5_5_mesh_7slot_mellow_shared_store_configurable_epochs/"
        "formal_3epochs_20261002_151018142328161-20/checkpoint-011343"
    ),
    "x4_8slot": (
        "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
        "audio_5_10x4_5_mesh_8slot_mellow_shared_store_configurable_epochs/"
        "formal_3epochs_20261004_x4_8slot_v2/checkpoint-011343"
    ),
    "x5_8slot": (
        "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
        "audio_5_10x5_5_mesh_8slot_mellow_shared_store_configurable_epochs/"
        "formal_3epochs_20261004_x5_8slot_v1/checkpoint-011343"
    ),
    "x5_9slot": (
        "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
        "audio_5_10x5_5_mesh_9slot_mellow_shared_store_configurable_epochs/"
        "formal_3epochs_20261006_054353274259875-20/checkpoint-011343"
    ),
}

DEFAULT_DATASET_DIR = official.DEFAULT_DATASET_DIR
DEFAULT_HTSAT = official.DEFAULT_HTSAT
DEFAULT_MELLOW = official.DEFAULT_MELLOW
DEFAULT_MAX_PROMPT_TOKENS = official.DEFAULT_MAX_PROMPT_TOKENS
DEFAULT_MAX_NEW_TOKENS = 300
DEFAULT_MAX_CONTEXT_LENGTH = official.DEFAULT_MAX_CONTEXT_LENGTH
EXPECTED_PREFIX_TOKENS = {"single": 260, "dual": 260}
CONTRACT = "mmau_fixed260_runtime_zero_second_slot_fp32_dual_scoring_v1"
PREDICTION_FORMAT = "mellow_author_reply_raw_generation_zero_slot_fp32_v1"
MMAU_PROTOCOL_CONTRACT = "mellow_issue5_author_reply_fixed260_runtime_zero_fp32_v1"
SINGLE_AUDIO_SLOT_SEMANTICS = (
    "fixed260_second_slot_is_one_runtime_all_zero_gpu_waveform_"
    "encoded_once_then_expanded_before_the_bridge"
)


def _route(name: str) -> RouteSpec:
    try:
        return ROUTES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown MeSH MMAU route: {name!r}") from exc


def _infer_route(raw: Sequence[str]) -> str:
    checkpoint: str | None = None
    for index, value in enumerate(raw):
        if value == "--checkpoint" and index + 1 < len(raw):
            checkpoint = raw[index + 1]
        elif value.startswith("--checkpoint="):
            checkpoint = value.split("=", 1)[1]
    if checkpoint:
        path = Path(checkpoint)
        candidates = [
            name for name, spec in ROUTES.items()
            if (path / spec.audio_config_filename).is_file()
        ]
        if len(candidates) == 1:
            return candidates[0]
        normalized = str(path).replace("\\", "/").lower()
        if "10x2" in normalized and "7slot" in normalized:
            return "x2_7slot"
        if "10x3" in normalized and "7slot" in normalized:
            return "x3_7slot"
        if "10x4" in normalized and "8slot" in normalized:
            return "x4_8slot"
        if "10x5" in normalized and "8slot" in normalized:
            return "x5_8slot"
        if "10x5" in normalized and "9slot" in normalized:
            return "x5_9slot"
        if "10x4" in normalized:
            return "x4"
        if "10x5" in normalized and "7slot" in normalized:
            return "x5_7slot"
        raise ValueError(
            "cannot infer --route from checkpoint; pass one of "
            f"{tuple(ROUTES)} explicitly"
        )
    return "x5_7slot"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _torch_load(path: Path, *, map_location: Any) -> Any:
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _weight_files(mesh_dir: Path) -> list[Path]:
    suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    return sorted(
        path for path in mesh_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _route_modules(spec: RouteSpec) -> tuple[Any, Any, Any]:
    package = importlib.import_module(spec.audio_package)
    audio = importlib.import_module(spec.audio_model_module)
    text = importlib.import_module(spec.text_model_module)
    return package, audio, text


def _expected_trace(text_module: Any, spec: RouteSpec) -> list[dict[str, int]]:
    schedule = tuple(int(value) for value in text_module.LOGICAL_TO_PHYSICAL)
    if len(schedule) != spec.logical_layer_count:
        raise RuntimeError(
            f"{spec.name} logical trace length mismatch: "
            f"expected={spec.logical_layer_count} actual={len(schedule)}"
        )
    return [
        {"logical_index": logical, "physical_index": physical}
        for logical, physical in enumerate(schedule)
    ]


def _audit_checkpoint(args: argparse.Namespace, spec: RouteSpec) -> dict[str, Any]:
    """Fail closed unless the checkpoint belongs to the selected route."""
    import torch

    checkpoint = args.checkpoint.expanduser().resolve()
    package, audio_module, text_module = _route_modules(spec)
    required = [
        "mesh_model/config.json",
        "tokenizer/tokenizer_config.json",
        "audio_bridge.pt",
        "training_state.pt",
        spec.audio_config_filename,
        "checkpoint_complete.json",
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"{spec.name} checkpoint is missing required files: {missing}")
    empty = [name for name in required if (checkpoint / name).stat().st_size <= 0]
    if empty:
        raise RuntimeError(f"{spec.name} checkpoint has empty required files: {empty}")
    weights = _weight_files(checkpoint / "mesh_model")
    if not weights:
        raise RuntimeError(f"{spec.name} checkpoint has no MeSH model weight files")

    config = _json(checkpoint / spec.audio_config_filename)
    marker = _json(checkpoint / "checkpoint_complete.json")
    expected = {
        "contract": spec.training_contract,
        "architecture_contract": spec.audio_architecture_contract,
        "text_model_architecture_contract": text_module.MODEL_ARCHITECTURE_CONTRACT,
        "compact_single_audio_prefix": False,
        "prefix_tokens": EXPECTED_PREFIX_TOKENS,
        "mesh_hidden_size": int(audio_module.MESH_HIDDEN_SIZE),
        "audio_tokens_per_clip": int(audio_module.AUDIO_TOKENS_PER_CLIP),
        "audio_prefix_tokens_with_separators": int(audio_module.AUDIO_PREFIX_TOKENS),
        "mode": "formal",
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    suffix = checkpoint.name.removeprefix("checkpoint-")
    directory_step = int(suffix) if suffix.isdigit() else -1
    if directory_step <= 0:
        mismatches["checkpoint_directory"] = {
            "expected": "checkpoint-<positive optimizer step>",
            "actual": checkpoint.name,
        }
    if (
        marker.get("status") != "complete"
        or marker.get("contract") != spec.training_contract
        or int(marker.get("global_step", -1)) != directory_step
    ):
        mismatches["completion_marker"] = {
            "expected": {
                "status": "complete",
                "contract": spec.training_contract,
                "global_step": directory_step,
            },
            "actual": marker,
        }
    marker_required = marker.get("required")
    expected_marker_required = [
        "mesh_model",
        "tokenizer",
        "audio_bridge.pt",
        "training_state.pt",
        spec.audio_config_filename,
    ]
    if marker_required is not None and not set(expected_marker_required).issubset(set(marker_required)):
        mismatches["completion_marker_required"] = {
            "expected_at_least": expected_marker_required,
            "actual": marker_required,
        }
    if int(audio_module.AUDIO_TOKENS_PER_CLIP) != 129 or int(audio_module.AUDIO_PREFIX_TOKENS) != 260:
        mismatches["runtime_audio_constants"] = {
            "expected": {"per_clip": 129, "prefix": 260},
            "actual": {
                "per_clip": int(audio_module.AUDIO_TOKENS_PER_CLIP),
                "prefix": int(audio_module.AUDIO_PREFIX_TOKENS),
            },
        }
    for key, expected_value in (
        ("logical_layer_count", spec.logical_layer_count),
        ("recursive_loops", spec.recursive_loops),
        ("memory_slots", spec.memory_slots),
        ("router_groups", spec.router_groups),
    ):
        if key in config and int(config[key]) != expected_value:
            mismatches[key] = {"expected": expected_value, "actual": config[key]}
    if mismatches:
        raise RuntimeError(f"{spec.name} checkpoint contract mismatch: {mismatches}")

    state = _torch_load(checkpoint / "training_state.pt", map_location="cpu")
    if not isinstance(state, Mapping):
        raise RuntimeError("training_state.pt must contain a mapping")
    required_state = {"optimizer", "scheduler", "global_step", "cursor", "rng_states_by_rank"}
    missing_state = sorted(required_state.difference(state))
    if missing_state:
        raise RuntimeError(f"{spec.name} training state is missing keys: {missing_state}")
    cursor = state.get("cursor")
    if not isinstance(cursor, Mapping):
        raise RuntimeError(f"{spec.name} training state cursor is not a mapping")
    if int(state.get("global_step", -1)) != directory_step or int(cursor.get("global_step", -1)) != directory_step:
        raise RuntimeError(f"{spec.name} training cursor/global step mismatch")
    rng_ranks = {str(key) for key in state.get("rng_states_by_rank", {})}
    if rng_ranks != {str(index) for index in range(8)}:
        raise RuntimeError(f"{spec.name} RNG rank coverage mismatch: {sorted(rng_ranks)}")
    del state

    audio_state = _torch_load(checkpoint / "audio_bridge.pt", map_location="cpu")
    if not isinstance(audio_state, Mapping) or set(audio_state) != {"bridge", "c2l"}:
        raise RuntimeError(f"{spec.name} audio_bridge.pt must contain bridge and c2l states")
    if not audio_state["bridge"] or not audio_state["c2l"]:
        raise RuntimeError(f"{spec.name} audio_bridge.pt contains an empty state")
    del audio_state

    return {
        "status": "PASS",
        "route": spec.name,
        "artifact_kind": "fixed260_runtime_zero_slot_audio_mesh_checkpoint",
        "path": str(checkpoint),
        "config_path": str(checkpoint / spec.audio_config_filename),
        "global_step": directory_step,
        "training_contract": spec.training_contract,
        "audio_architecture_contract": spec.audio_architecture_contract,
        "text_model_architecture_contract": text_module.MODEL_ARCHITECTURE_CONTRACT,
        "logical_layer_count": spec.logical_layer_count,
        "recursive_loops": spec.recursive_loops,
        "memory_slots": spec.memory_slots,
        "router_groups": spec.router_groups,
        "required_files": required,
        "text_model_weight_files": [str(path) for path in weights],
        "compact_single_audio_prefix": False,
        "prefix_tokens": EXPECTED_PREFIX_TOKENS,
        "single_audio_slot_semantics": SINGLE_AUDIO_SLOT_SEMANTICS,
    }


def _load_runtime_model(
    args: argparse.Namespace,
    spec: RouteSpec,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"{spec.name} MMAU evaluation requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    checkpoint_audit = _audit_checkpoint(args, spec)
    package, audio_module, text_module = _route_modules(spec)
    text_module.register_auto_class()
    from transformers import AutoTokenizer

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    mesh = text_module.RecursiveLlamaForCausalLM.from_pretrained(
        checkpoint / "mesh_model", local_files_only=True
    )
    if int(mesh.config.hidden_size) != int(audio_module.MESH_HIDDEN_SIZE):
        raise RuntimeError(f"{spec.name} MeSH hidden size differs from audio mapper contract")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint / "tokenizer", local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper, htsat, provenance = audio_module._load_mellow_wrapper(
        args.mellow_root, args.htsat_checkpoint, device
    )
    model_config = getattr(audio_module, spec.audio_config_class)()
    model = getattr(audio_module, spec.audio_model_class)(
        mesh.to(device), tokenizer, wrapper, htsat, config=model_config
    )
    audio_state = _torch_load(checkpoint / "audio_bridge.pt", map_location=device)
    model.bridge.load_state_dict(audio_state["bridge"], strict=True)
    model.htsat_wrapper.c2l.load_state_dict(audio_state["c2l"], strict=True)
    model._audio_provenance = provenance
    model._route_name = spec.name
    model.to(device)
    model.eval()

    owner = model.mesh_model.model
    for attribute in ("audit_mode", "gradient_audit_mode", "routing_stats_mode"):
        if hasattr(owner, attribute):
            setattr(owner, attribute, False)
    modes = {
        "composite": bool(model.training),
        "mesh": bool(model.mesh_model.training),
        "bridge": bool(model.bridge.training),
        "wrapper": bool(model.htsat_wrapper.training),
        "htsat": bool(model.htsat_backbone.training),
        "c2l": bool(model.htsat_wrapper.c2l.training),
    }
    if any(modes.values()):
        raise RuntimeError(f"{spec.name} inference modules are not all in eval mode: {modes}")
    if bool(model.config_audio.compact_single_audio_prefix):
        raise RuntimeError(f"{spec.name} evaluator requires a fixed 260-token prefix")
    if int(model.config_audio.max_context_length) != DEFAULT_MAX_CONTEXT_LENGTH:
        raise RuntimeError(
            f"{spec.name} inference context mismatch: "
            f"expected={DEFAULT_MAX_CONTEXT_LENGTH} actual={model.config_audio.max_context_length}"
        )
    parameter_devices = {
        str(parameter.device) for parameter in model.parameters() if parameter.requires_grad
    }
    if parameter_devices != {str(device)}:
        raise RuntimeError(
            f"{spec.name} trainable parameters are not colocated on cuda:0: "
            f"{sorted(parameter_devices)}"
        )
    config = _json(checkpoint / spec.audio_config_filename)
    saved_provenance = config.get("mellow_provenance") or {}
    for key in ("module", "mellow_htsat_source", "mellow_htsat_sha256"):
        if key in saved_provenance and saved_provenance.get(key) != provenance.get(key):
            raise RuntimeError(
                f"{spec.name} Mellow provenance mismatch for {key}: "
                f"saved={saved_provenance.get(key)!r} runtime={provenance.get(key)!r}"
            )
    config["checkpoint_artifact_audit"] = checkpoint_audit
    config["runtime_audio_provenance"] = provenance
    config["runtime_max_context_length"] = int(model.config_audio.max_context_length)
    config["runtime_inference_dtype"] = "float32"
    return model, tokenizer, device, config


def _build_fixed260_zero_prefix(
    model: Any,
    waveform: Any,
    device: Any,
    spec: RouteSpec,
) -> tuple[Any, dict[str, Any]]:
    """Build slot two from one exact zero waveform on the current GPU."""
    import torch

    from audio_5_10x2_5_mesh_mellow.model import _find_embedding

    audio_module = importlib.import_module(spec.audio_model_module)
    audio1 = waveform.to(device=device, dtype=torch.float32, non_blocking=True)
    if audio1.ndim == 2:
        audio1 = audio1.unsqueeze(0)
    if audio1.ndim != 3:
        raise RuntimeError(f"MMAU waveform must be [B,C,T] after batching, got {tuple(audio1.shape)}")
    silence_mask = torch.ones((int(audio1.shape[0]),), dtype=torch.bool, device=device)
    same_real_mask = torch.zeros((int(audio1.shape[0]),), dtype=torch.bool, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
        first, second = model.encode_audio(
            audio1,
            None,
            silence_mask,
            same_real_mask,
        )
        separator_ids = torch.full(
            (int(audio1.shape[0]), 1),
            int(model.separator_token_id),
            dtype=torch.long,
            device=device,
        )
        separator = _find_embedding(model.mesh_model, separator_ids)
        prefix = torch.cat((first, separator, second, separator), dim=1)
    expected_slot = (int(audio_module.AUDIO_TOKENS_PER_CLIP), int(audio_module.MESH_HIDDEN_SIZE))
    if tuple(first.shape[1:]) != expected_slot or tuple(second.shape[1:]) != expected_slot:
        raise RuntimeError(
            f"{spec.name} zero-slot shape mismatch: "
            f"first={tuple(first.shape)} second={tuple(second.shape)}"
        )
    if tuple(prefix.shape[1:]) != (260, int(audio_module.MESH_HIDDEN_SIZE)):
        raise RuntimeError(f"{spec.name} fixed260 prefix shape mismatch: {tuple(prefix.shape)}")
    if not bool(torch.isfinite(prefix).all()):
        raise RuntimeError(f"{spec.name} fixed260 prefix contains non-finite values")
    slot_audit = dict(getattr(model, "last_audio_slot_audit", {}))
    expected_audit = {
        "silence_second_slot_rows": int(audio1.shape[0]),
        "same_real_audio_rows": 0,
        "runtime_silence_waveforms_created": 1,
        "second_encoder_input_batch_size": 1,
    }
    for key, expected in expected_audit.items():
        if int(slot_audit.get(key, -1)) != expected:
            raise RuntimeError(
                f"{spec.name} zero-slot audit mismatch for {key}: "
                f"expected={expected} actual={slot_audit.get(key)}"
            )
    return prefix, {
        "route": spec.name,
        "audio1_prefix_shape": list(first.shape),
        "audio2_prefix_shape": list(second.shape),
        "combined_prefix_shape": list(prefix.shape),
        "prefix_token_count": 260,
        "prefix_layout": "audio1 + separator1 + runtime_zero_wave + separator2",
        "separator_token_id": int(model.separator_token_id),
        "separator_token": model.tokenizer.decode(
            [int(model.separator_token_id)], skip_special_tokens=False
        ),
        "compact_single_audio_prefix_used": False,
        "audio2_prefix_materialized": True,
        "audio2_reused": False,
        "runtime_zero_second_slot": True,
        "runtime_zero_created_on_gpu": True,
        "runtime_zero_shape": slot_audit.get("runtime_silence_shape"),
        "audio_slot_audit": slot_audit,
        "inference_dtype": "float32",
    }


def _run_author_reply_generation(
    spec: RouteSpec,
    model: Any,
    tokenizer: Any,
    device: Any,
    sample: Mapping[str, Any],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    from generate_audio_checkpoint_reasonaqa import _greedy_decode

    prompt_ids_cpu, prompt_original_token_count, prompt_truncated = (
        official.tokenize_mellow_author_reply_prompt(
            tokenizer,
            str(sample["prompt"]),
            max_prompt_tokens=max_prompt_tokens,
        )
    )
    prompt_ids = prompt_ids_cpu.to(device)
    waveform, audio_segment = official.mellow_author_reply_audio_segment(sample["waveform"])
    _, _, text_module = _route_modules(spec)
    expected_trace = _expected_trace(text_module, spec)
    with torch.inference_mode():
        prefix, prefix_audit = _build_fixed260_zero_prefix(model, waveform, device, spec)
        generation = _greedy_decode(
            model,
            tokenizer,
            prefix,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            autocast_enabled=False,
            top_p=0.8,
            temperature=1.0,
            expected_trace=expected_trace,
        )
    generated_text_raw = tokenizer.decode(
        generation["generated_token_ids"], skip_special_tokens=False
    )
    generation["generated_text_raw"] = generated_text_raw
    generation["generated_text"] = generated_text_raw.split(
        tokenizer.eos_token or "<|endoftext|>"
    )[0]
    generation["decode_policy"] = "mellow_wrapper_decode_then_split_stop_token"
    generation["prompt_token_count"] = int(prompt_ids.shape[1])
    generation["prompt_original_token_count"] = prompt_original_token_count
    generation["prompt_truncated"] = prompt_truncated
    generation["audio1_segment"] = audio_segment
    generation["audio2_segment"] = {
        **audio_segment,
        "policy": "runtime_zero_waveform_on_gpu",
    }
    generation.update(prefix_audit)
    if generation.get("logical_trace") != expected_trace:
        raise RuntimeError(f"{spec.name} generation trace audit failed")
    if (
        generation.get("prefix_token_count") != 260
        or generation.get("runtime_zero_second_slot") is not True
        or generation.get("audio2_reused") is not False
        or generation.get("inference_dtype") != "float32"
    ):
        raise RuntimeError(f"{spec.name} fixed260 zero-slot generation audit failed: {generation}")
    return generation


def prepare_model_output_for_official_scorer(value: Any) -> str:
    """Keep raw generated text; each scorer owns its own normalization."""
    return str(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    route_parser = argparse.ArgumentParser(add_help=False)
    route_parser.add_argument("--route", choices=tuple(ROUTES))
    route_args, remaining = route_parser.parse_known_args(raw)
    route_name = route_args.route or _infer_route(raw)
    if not any(item == "--mode" or item.startswith("--mode=") for item in remaining):
        remaining = ["--mode", "full", *remaining]
    args = official.parse_args(
        remaining,
        default_checkpoint=DEFAULT_CHECKPOINTS[route_name],
        default_max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        add_audio_root=True,
        description=__doc__,
    )
    if args.dtype != "fp32":
        raise ValueError("this evaluator is fixed to --dtype fp32")
    args.route = route_name
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    spec = _route(args.route)

    def load_runtime_model(current_args: argparse.Namespace):
        return _load_runtime_model(current_args, spec)

    def run_model_generation(model: Any, tokenizer: Any, device: Any, sample: Mapping[str, Any], **kwargs: Any):
        return _run_author_reply_generation(spec, model, tokenizer, device, sample, **kwargs)

    report = official.run(
        args,
        load_runtime_model=load_runtime_model,
        run_model_generation=run_model_generation,
        prepare_prediction=prepare_model_output_for_official_scorer,
        prediction_format=PREDICTION_FORMAT,
        prompt_builder=official.build_mellow_author_reply_prompt,
        audio_decoder=official.decode_mellow_author_reply_audio,
        audio_root=args.audio_root,
        prefer_official_audio_file=True,
        prompt_format=official.MELLOW_AUTHOR_REPLY_PROMPT_FORMAT,
        audio_format=(
            official.MELLOW_AUTHOR_REPLY_AUDIO_FORMAT
            + "__single_segment_runtime_zero_second_slot"
        ),
        protocol_contract=MMAU_PROTOCOL_CONTRACT,
        generation_protocol={
            "decoder": "mellow_wrapper_top_p_filter_then_argmax_full_recompute",
            "top_p": 0.8,
            "temperature": 1.0,
            "do_sample": False,
            "use_cache": False,
            "inference_dtype": "float32",
            "audio_slot_policy": "single_audio_second_slot_runtime_zero_waveform_on_gpu",
            "prefix_tokens": 260,
            "route": spec.name,
        },
        audio_prefix_tokens=260,
        stage=f"mmau_test_mini_audio_mesh_{spec.name}_fixed260_zero_slot",
        logical_trace=(
            f"exact {spec.name} MeSH trace verified by route-aware greedy decoder"
        ),
    )
    inference_failures = int(
        report.get("records", {}).get("skip_reasons", {}).get("sample_exception", 0)
    )
    if inference_failures:
        report["status"] = "FAILED"
        report["comparable_official_score"] = False
        report["fatal_error"] = {
            "error": f"{inference_failures} {spec.name} generation failures were recorded as skipped rows",
            "detail": "Inspect skipped.jsonl; do not interpret either score as a valid complete-model score.",
        }

    predictions_path = args.output_dir / "predictions_fixed_order.json"
    if predictions_path.is_file() and report.get("inference_coverage", {}).get("status") == "PASS":
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        author_score = official.write_mellow_author_reply_evaluation(args.output_dir, predictions)
        report["route"] = spec.name
        report["mellow_author_reply_evaluation"] = author_score
        report["mellow_author_reply_context"] = official.MELLOW_AUTHOR_REPLY_CONTEXT
        report["primary_comparison_score"] = {
            "scorer": official.MELLOW_AUTHOR_REPLY_SCORER,
            "normalization": "prediction.split(')')[0].lower()",
            "comparable": bool(
                args.mode == "full"
                and inference_failures == 0
                and int(author_score["total"]["total"]) == official.EXPECTED_FULL_ROWS
            ),
            "record_errors_counted_incorrect": int(
                author_score.get("record_errors", {}).get("total", 0)
            ),
            **author_score["total"],
        }
        report["mmau_v051525_evaluation"] = report.get("official_evaluation", {})
        report["zero_slot_inference_contract"] = {
            "status": "PASS",
            "inference_dtype": "float32",
            "prefix_tokens": 260,
            "single_audio_second_slot": "runtime_zero_waveform_on_gpu",
            "prompt_answer_layout": "prompt_and_answer_concatenated_before_batch_right_padding",
        }
        official._write_json(args.output_dir / "evaluation_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "route": report.get("route", args.route),
        "stage": report.get("stage"),
        "status": report.get("status"),
        "mode": report.get("mode"),
        "records": report.get("records", {}),
        "primary_comparison_score": report.get("primary_comparison_score", {}),
        "official_evaluation": report.get("official_evaluation", {}),
        "report": str(args.output_dir / "evaluation_report.json"),
    }, ensure_ascii=False, default=official._json_default))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
