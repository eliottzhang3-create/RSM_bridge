#!/usr/bin/env python3
"""CPU-only preflight for initializing the 5-10x2-5 MeSH model from Mellow-v0.

This audit does not create a converted checkpoint.  It loads the released
Mellow-v0 state dictionary into the native Mellow model, then constructs the
target MeSH text model in memory and checks the complete source-to-target
mapping.  It also validates the Mellow HTSAT/c2l/projection groups against the
audio bridge contract used by the official MeSH training route.

Only the JSON report is written.  The source checkpoint and source model
directories are never modified.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
RSMOL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(RSMOL_ROOT) not in sys.path:
    sys.path.insert(0, str(RSMOL_ROOT))

from audit_mellow_v0_artifact import (  # noqa: E402
    DEFAULT_BASE_SMOLLM2,
    DEFAULT_MELLOW_CHECKPOINT,
    DEFAULT_MELLOW_SNAPSHOT,
    SOURCE_PREFIXES as NATIVE_SOURCE_PREFIXES,
    _instantiate_native_model,
    _load_state,
    _sha256,
    _smollm2_inventory,
    _source_inventory,
)
from convert_stepwise_5_10x2_5_mesh import (  # noqa: E402
    _target_config,
)
from recursive_model_5_10x2_5_mesh import (  # noqa: E402
    LOGICAL_TO_PHYSICAL,
    MEMORY_SLOT_COUNT,
    PHYSICAL_LAYER_COUNT,
    ROUTER_COUNT,
    ROUTER_PARAMETER_COUNT,
    SOURCE_LAYER_INDICES_0BASED,
    RecursiveLlamaForCausalLM,
    parameter_audit,
    register_auto_class,
)


DEFAULT_REPORT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0/preflight/"
    "mellow_v0_mesh_initialization_audit.json"
)
# This is the complete Mellow source checkout.  The Hugging Face snapshot
# containing config.json and v0.ckpt is a separate artifact and must not be
# used as --mellow-source-root.
DEFAULT_MELLOW_SOURCE_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/mellow-main"
)

ARTIFACT_CONTRACT = "mellow_v0_to_5_10x2_5_mesh_initialization_audit_v1"
TEXT_PREFIX = "caption_decoder.lm."
BRIDGE_GROUP_NAME = "projection"
EXPECTED_BRIDGE_KEYS = {
    "linear1.weight",
    "linear2.weight",
    "layer_norm.weight",
    "layer_norm.bias",
}
EXPECTED_C2L_KEYS = {"weight", "bias"}
EXPECTED_DISCARDED_SOURCE_LAYERS = tuple(
    index for index in range(30) if index not in SOURCE_LAYER_INDICES_0BASED
)
ROUTER_NAME_MARKERS = ("write_routers.", "read_routers.")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _tensor_digest(value: Any) -> str:
    import torch

    if not torch.is_tensor(value):
        raise TypeError(f"digest expects a tensor, got {type(value).__name__}")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _extract_group(state: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def _dtypes(values: Mapping[str, Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value.dtype) for value in values.values()).items()))


def _shape_inventory(values: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    if not values:
        return {"tensor_count": 0, "parameter_count": 0, "dtypes": {}}
    if not all(torch.is_tensor(value) for value in values.values()):
        raise RuntimeError("state group contains a non-tensor value")
    return {
        "tensor_count": len(values),
        "parameter_count": int(sum(value.numel() for value in values.values())),
        "dtypes": _dtypes(values),
        "shapes": {
            key: list(value.shape)
            for key, value in sorted(values.items())
        },
    }


def _source_key_for_target(target_key: str) -> str:
    """Map a target MeSH state key to its original Mellow text key."""

    if target_key.startswith("model.layers."):
        tail = target_key[len("model.layers."):]
        parts = tail.split(".", 1)
        if not parts[0].isdigit():
            raise RuntimeError(f"target layer key has a non-numeric index: {target_key}")
        target_index = int(parts[0])
        if target_index < 0 or target_index >= PHYSICAL_LAYER_COUNT:
            raise RuntimeError(f"target layer index is outside the physical model: {target_key}")
        source_index = SOURCE_LAYER_INDICES_0BASED[target_index]
        suffix = parts[1] if len(parts) == 2 else ""
        return f"{TEXT_PREFIX}model.layers.{source_index}.{suffix}"
    if target_key.startswith("model.") or target_key.startswith("lm_head."):
        return f"{TEXT_PREFIX}{target_key}"
    raise RuntimeError(f"target state key is not an inherited text parameter: {target_key}")


def _is_router_key(key: str) -> bool:
    return any(marker in key for marker in ROUTER_NAME_MARKERS)


def _check_tied_embeddings(text_state: Mapping[str, Any], config: Any) -> dict[str, Any]:
    import torch

    embedding_key = "model.embed_tokens.weight"
    lm_head_key = "lm_head.weight"
    embedding = text_state.get(embedding_key)
    lm_head = text_state.get(lm_head_key)
    if embedding is None or lm_head is None:
        raise RuntimeError(
            "Mellow text group must contain both model.embed_tokens.weight and lm_head.weight"
        )
    if tuple(embedding.shape) != tuple(lm_head.shape):
        raise RuntimeError(
            f"Mellow embedding/LM-head shapes differ: {tuple(embedding.shape)} vs {tuple(lm_head.shape)}"
        )
    equal = bool(torch.equal(embedding, lm_head))
    if not equal:
        raise RuntimeError("Mellow input embedding and LM head are not exactly tied")
    tie_config = bool(getattr(config, "tie_word_embeddings", False))
    return {
        "embedding_key": embedding_key,
        "lm_head_key": lm_head_key,
        "shape": list(embedding.shape),
        "exact_tensor_equality": equal,
        "config_tie_word_embeddings": tie_config,
    }


def _load_target_text_model(source_config: Any, source_path: Path, text_state: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    import torch

    target_config = _target_config(
        source_config,
        {"kind": "native_mellow_v0_text", "mapping": SOURCE_LAYER_INDICES_0BASED},
        source_path,
    )
    register_auto_class()
    target_model = RecursiveLlamaForCausalLM(target_config)
    floating = next(
        (
            value
            for value in text_state.values()
            if torch.is_tensor(value) and value.is_floating_point()
        ),
        None,
    )
    if floating is None:
        raise RuntimeError("Mellow text group contains no floating-point tensor")
    target_model.to(dtype=floating.dtype)
    return target_model, {
        "target_config": {
            "model_type": str(getattr(target_config, "model_type", "")),
            "hidden_size": int(getattr(target_config, "hidden_size", -1)),
            "vocab_size": int(getattr(target_config, "vocab_size", -1)),
            "logical_layer_count": int(getattr(target_config, "num_hidden_layers", -1)),
            "physical_layer_count": int(getattr(target_config, "recursive_layer_count", -1)),
            "recursive_loops": int(getattr(target_config, "recursive_loops", -1)),
            "memory_slots": int(getattr(target_config, "mesh_memory_slots", -1)),
            "router_count": int(getattr(target_config, "mesh_router_count", -1)),
            "mesh_architecture_contract": str(getattr(target_config, "mesh_architecture_contract", "")),
            "architectures": list(getattr(target_config, "architectures", []) or []),
        },
        "source_float_dtype": str(floating.dtype),
    }


def _audit_text_mapping(
    target_model: Any,
    source_state: Mapping[str, Any],
    source_config: Any,
) -> dict[str, Any]:
    import torch

    target_state = target_model.state_dict()
    source_text = _extract_group(source_state, TEXT_PREFIX)
    if not source_text:
        raise RuntimeError("Mellow checkpoint has no caption_decoder.lm text group")

    tied = _check_tied_embeddings(source_text, source_config)
    mapped: dict[str, Any] = {}
    missing: list[str] = []
    shape_mismatches: dict[str, Any] = {}
    dtype_mismatches: dict[str, Any] = {}
    source_keys_used: set[str] = set()
    key_map: dict[str, str] = {}
    inherited_target_keys = [key for key in target_state if not _is_router_key(key)]
    for target_key in inherited_target_keys:
        source_key = _source_key_for_target(target_key)
        source_relative = source_key[len(TEXT_PREFIX):]
        value = source_text.get(source_relative)
        if value is None:
            # Safe serialization can omit one side of a tied pair.  The
            # native Mellow audit already proved exact tying; reuse the side
            # that is present rather than silently accepting an arbitrary miss.
            alias = {
                "model.embed_tokens.weight": "lm_head.weight",
                "lm_head.weight": "model.embed_tokens.weight",
            }.get(source_relative)
            if alias is not None:
                value = source_text.get(alias)
                if value is not None:
                    source_key = f"{TEXT_PREFIX}{alias}"
        if value is None:
            missing.append(target_key)
            continue
        expected = target_state[target_key]
        source_keys_used.add(source_key[len(TEXT_PREFIX):])
        key_map[target_key] = source_key
        if tuple(value.shape) != tuple(expected.shape):
            shape_mismatches[target_key] = {
                "source_key": source_key,
                "source_shape": list(value.shape),
                "target_shape": list(expected.shape),
            }
            continue
        if value.dtype != expected.dtype:
            dtype_mismatches[target_key] = {
                "source_key": source_key,
                "source_dtype": str(value.dtype),
                "target_dtype": str(expected.dtype),
            }
        mapped[target_key] = value

    if missing or shape_mismatches:
        raise RuntimeError(
            "Mellow-to-MeSH text mapping is incomplete: "
            f"missing={missing[:8]} shape_mismatches={list(shape_mismatches)[:8]}"
        )
    if dtype_mismatches:
        raise RuntimeError(
            "Mellow-to-MeSH text mapping requires implicit dtype conversion: "
            f"{list(dtype_mismatches.items())[:8]}"
        )

    # Every source tensor must either feed a target parameter or belong to one
    # of the ten deliberately discarded source layers.  This catches silent
    # key-name drift in the released checkpoint.
    unexpected_source_keys: list[str] = []
    discarded_source_keys: list[str] = []
    for relative_key in source_text:
        if relative_key in source_keys_used:
            continue
        if relative_key.startswith("model.layers."):
            parts = relative_key.split(".", 3)
            if len(parts) >= 3 and parts[2].isdigit() and int(parts[2]) in EXPECTED_DISCARDED_SOURCE_LAYERS:
                discarded_source_keys.append(relative_key)
                continue
        unexpected_source_keys.append(relative_key)
    if unexpected_source_keys:
        raise RuntimeError(
            "Mellow text group contains source keys that are neither mapped nor "
            f"an explicitly discarded layer: {unexpected_source_keys[:12]}"
        )

    missing_after_load, unexpected_after_load = [], []
    incompatible = target_model.load_state_dict(mapped, strict=False)
    missing_after_load = sorted(incompatible.missing_keys)
    unexpected_after_load = sorted(incompatible.unexpected_keys)
    expected_router_keys = sorted(key for key in target_state if _is_router_key(key))
    if missing_after_load != expected_router_keys or unexpected_after_load:
        raise RuntimeError(
            "target dry-run load has an unexpected key result: "
            f"missing={missing_after_load[:12]} unexpected={unexpected_after_load[:12]}"
        )
    target_model.tie_weights()
    loaded_state = target_model.state_dict()
    unequal_loaded: list[str] = []
    for target_key, source_key in key_map.items():
        source_relative = source_key[len(TEXT_PREFIX):]
        source_value = source_text[source_relative]
        if not torch.equal(loaded_state[target_key].cpu(), source_value.cpu()):
            unequal_loaded.append(target_key)
    if unequal_loaded:
        raise RuntimeError(f"target dry-run load changed inherited values: {unequal_loaded[:8]}")

    audit = parameter_audit(target_model)
    routers = list(target_model.model.write_routers) + list(target_model.model.read_routers)
    hidden_size = int(target_model.config.hidden_size)
    expected_router_std = (2.0 / (MEMORY_SLOT_COUNT * hidden_size)) ** 0.5
    router_min = min(float(router.weight.detach().min().cpu()) for router in routers)
    router_max = max(float(router.weight.detach().max().cpu()) for router in routers)
    router_audit = {
        "router_objects": len(routers),
        "router_objects_independent": len({id(router) for router in routers}) == ROUTER_PARAMETER_COUNT,
        "expected_router_objects": ROUTER_PARAMETER_COUNT,
        "expected_router_groups": ROUTER_COUNT,
        "memory_slots": MEMORY_SLOT_COUNT,
        "weight_shapes": [list(router.weight.shape) for router in routers],
        "bias_shapes": [list(router.bias.shape) for router in routers],
        "bias_all_zero": all(bool(torch.equal(router.bias.detach().cpu(), torch.zeros_like(router.bias.detach().cpu()))) for router in routers),
        "weights_finite": all(bool(torch.isfinite(router.weight.detach()).all()) for router in routers),
        "weights_source_initialized": False,
        "initialization_contract": "truncated_normal_std_sqrt_2_over_5d_clamped_3std_bias_zero",
        "expected_std": expected_router_std,
        "allowed_abs_bound": 3.0 * expected_router_std,
        "observed_global_min": router_min,
        "observed_global_max": router_max,
        "observed_within_bound": router_min >= -3.0 * expected_router_std and router_max <= 3.0 * expected_router_std,
    }
    return {
        "status": "PASS",
        "source_text_tensor_count": len(source_text),
        "target_state_tensor_count": len(target_state),
        "inherited_target_tensor_count": len(inherited_target_keys),
        "mapped_target_tensor_count": len(mapped),
        "source_keys_used_count": len(source_keys_used),
        "discarded_source_layer_indices_0based": list(EXPECTED_DISCARDED_SOURCE_LAYERS),
        "discarded_source_tensor_count": len(discarded_source_keys),
        "discarded_source_keys_sample": sorted(discarded_source_keys)[:12],
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches_after_target_cast": dtype_mismatches,
        "missing_target_keys": missing,
        "unexpected_source_keys": unexpected_source_keys,
        "dry_run_missing_keys": missing_after_load,
        "dry_run_unexpected_keys": unexpected_after_load,
        "tied_embeddings": tied,
        "parameter_audit": audit,
        "router_memory_audit": router_audit,
        "source_layer_mapping_0based": [
            {"target_physical_layer": target, "source_mellow_layer": source}
            for target, source in enumerate(SOURCE_LAYER_INDICES_0BASED)
        ],
        "logical_to_physical_schedule": list(LOGICAL_TO_PHYSICAL),
    }


def _audit_audio_groups(state: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from torch import nn

    groups = {
        name: _extract_group(state, prefix)
        for name, prefix in NATIVE_SOURCE_PREFIXES.items()
    }
    projection = groups[BRIDGE_GROUP_NAME]
    c2l = groups["c2l"]
    htsat = groups["htsat"]
    if set(projection) != set(EXPECTED_BRIDGE_KEYS):
        raise RuntimeError(
            "Mellow projection key contract differs from the MeSH bridge: "
            f"missing={sorted(set(EXPECTED_BRIDGE_KEYS) - set(projection))} "
            f"unexpected={sorted(set(projection) - set(EXPECTED_BRIDGE_KEYS))}"
        )
    if set(c2l) != EXPECTED_C2L_KEYS:
        raise RuntimeError(
            f"Mellow c2l key contract differs from Linear(527,768): "
            f"missing={sorted(EXPECTED_C2L_KEYS - set(c2l))} "
            f"unexpected={sorted(set(c2l) - EXPECTED_C2L_KEYS)}"
        )
    bridge_state = {
        "linear1.weight": projection["linear1.weight"],
        "linear2.weight": projection["linear2.weight"],
        "norm.weight": projection["layer_norm.weight"],
        "norm.bias": projection["layer_norm.bias"],
    }
    # Importing the route's bridge is CPU-only and validates the exact target
    # module shape without loading HTSAT or running an audio forward pass.
    from audio_5_10x2_5_mesh_mellow.model import AudioBridge

    bridge = AudioBridge(768, 576, kernel=8, dropout=0.5)
    bridge.load_state_dict(bridge_state, strict=True)
    c2l_layer = nn.Linear(527, 768)
    c2l_layer.load_state_dict(c2l, strict=True)
    target_bridge_state = bridge.state_dict()
    target_c2l_state = c2l_layer.state_dict()
    bridge_dtype_mismatches = {
        key: {"source": str(bridge_state[key].dtype), "target": str(target_bridge_state[key].dtype)}
        for key in bridge_state
        if bridge_state[key].dtype != target_bridge_state[key].dtype
    }
    c2l_dtype_mismatches = {
        key: {"source": str(c2l[key].dtype), "target": str(target_c2l_state[key].dtype)}
        for key in c2l
        if c2l[key].dtype != target_c2l_state[key].dtype
    }
    if bridge_dtype_mismatches or c2l_dtype_mismatches:
        raise RuntimeError(
            "Mellow audio mapper requires implicit dtype conversion: "
            f"bridge={bridge_dtype_mismatches} c2l={c2l_dtype_mismatches}"
        )
    return {
        "status": "PASS",
        "groups": {
            name: _shape_inventory(group)
            for name, group in groups.items()
        },
        "htsat_frozen_policy": {
            "source_group_present": bool(htsat),
            "source_parameter_count": int(sum(value.numel() for value in htsat.values())),
            "runtime_policy": "copy Mellow-v0 HTSAT tensors, set requires_grad=False, preserve source buffers",
        },
        "c2l_contract": {
            "target": "Linear(527,768)",
            "loaded_strictly": True,
            "source_target_dtypes_match": True,
            "target_dtypes": _dtypes(target_c2l_state),
            "state_tensor_digests": {key: _tensor_digest(value) for key, value in sorted(c2l.items())},
        },
        "bridge_contract": {
            "target": "AudioBridge(768,576,kernel=8,dropout=0.5)",
            "loaded_strictly": True,
            "source_target_dtypes_match": True,
            "target_dtypes": _dtypes(target_bridge_state),
            "source_to_target_keys": {
                "audio_encoder.projection.linear1.weight": "bridge.linear1.weight",
                "audio_encoder.projection.linear2.weight": "bridge.linear2.weight",
                "audio_encoder.projection.layer_norm.weight": "bridge.norm.weight",
                "audio_encoder.projection.layer_norm.bias": "bridge.norm.bias",
            },
            "state_tensor_digests": {key: _tensor_digest(value) for key, value in sorted(bridge_state.items())},
        },
    }


def _audit_tokenizer(base_smollm2: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_smollm2, local_files_only=True)
    original_size = len(tokenizer)
    tokenizer.add_special_tokens({"pad_token": "!"})
    if len(tokenizer) != original_size:
        raise RuntimeError("adding Mellow's '!' pad token changes the tokenizer vocabulary")
    pad_id = tokenizer.pad_token_id
    separator_id = tokenizer.convert_tokens_to_ids("!")
    if pad_id is None or separator_id is None or int(pad_id) != int(separator_id):
        raise RuntimeError(
            f"Mellow tokenizer contract failed: pad_id={pad_id!r} separator_id={separator_id!r}"
        )
    mesh_separator_id = 0
    if mesh_separator_id >= len(tokenizer):
        raise RuntimeError("MeSH separator token id 0 is outside the tokenizer vocabulary")
    return {
        "size": len(tokenizer),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": int(pad_id),
        "padding_token": "!",
        "padding_token_id": int(pad_id),
        "native_mellow_inter_audio_separator_id": 0,
        "native_mellow_inter_audio_separator_token": tokenizer.convert_ids_to_tokens(0),
        "mesh_recursive_decoder_inter_audio_separator_id": mesh_separator_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": int(tokenizer.eos_token_id),
    }


def _audit(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    source_root = args.mellow_source_root.expanduser().resolve()
    snapshot = args.mellow_snapshot.expanduser().resolve()
    checkpoint = args.mellow_checkpoint.expanduser().resolve()
    base_smollm2 = args.base_smollm2.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Mellow-v0 checkpoint not found: {checkpoint}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Mellow source checkout not found: {source_root}")
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Mellow Hugging Face snapshot not found: {snapshot}")
    if checkpoint.parent != snapshot:
        raise RuntimeError(f"checkpoint must be inside the selected snapshot: {checkpoint} vs {snapshot}")
    if not base_smollm2.is_dir():
        raise FileNotFoundError(f"base SmolLM2 directory not found: {base_smollm2}")
    source_hashes = _source_inventory(source_root)
    smollm2_inventory = _smollm2_inventory(base_smollm2)
    source_state = _load_state(checkpoint)
    known_prefixes = tuple(NATIVE_SOURCE_PREFIXES.values())
    unknown_keys = sorted(key for key in source_state if not key.startswith(known_prefixes))
    if unknown_keys:
        raise RuntimeError(f"Mellow-v0 state contains unknown keys: {unknown_keys[:20]}")
    group_counts = {
        name: sum(key.startswith(prefix) for key in source_state)
        for name, prefix in NATIVE_SOURCE_PREFIXES.items()
    }
    if any(count <= 0 for count in group_counts.values()):
        raise RuntimeError(f"Mellow-v0 state has an empty required group: {group_counts}")

    native_model, native_tokenizer = _instantiate_native_model(source_root, base_smollm2)
    incompatible = native_model.load_state_dict(source_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"native Mellow strict load mismatch: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    native_runtime_state = native_model.state_dict()
    native_shape_mismatches = {
        key: {"source": list(source_state[key].shape), "runtime": list(native_runtime_state[key].shape)}
        for key in source_state
        if tuple(source_state[key].shape) != tuple(native_runtime_state[key].shape)
    }
    native_dtype_mismatches = {
        key: {"source": str(source_state[key].dtype), "runtime": str(native_runtime_state[key].dtype)}
        for key in source_state
        if source_state[key].dtype != native_runtime_state[key].dtype
    }
    if native_shape_mismatches or native_dtype_mismatches:
        raise RuntimeError(
            "native Mellow runtime tensor metadata differs from checkpoint: "
            f"shapes={list(native_shape_mismatches.items())[:8]} "
            f"dtypes={list(native_dtype_mismatches.items())[:8]}"
        )
    native_model.eval()
    source_config = native_model.caption_decoder.lm.config
    native_tokenizer_size = len(native_tokenizer)
    source_geometry = {
        "model_type": str(getattr(source_config, "model_type", "")),
        "hidden_size": int(getattr(source_config, "hidden_size", -1)),
        "vocab_size": int(getattr(source_config, "vocab_size", -1)),
        "num_hidden_layers": int(getattr(source_config, "num_hidden_layers", -1)),
    }
    if (
        source_geometry["model_type"] != "llama"
        or source_geometry["hidden_size"] != 576
        or source_geometry["num_hidden_layers"] != 30
        or source_geometry["vocab_size"] <= 0
    ):
        raise RuntimeError(
            "Mellow text decoder is not the expected 30-layer SmolLM2/Llama geometry: "
            f"{source_geometry}"
        )
    del native_runtime_state
    del native_shape_mismatches
    del native_dtype_mismatches
    del native_model
    del native_tokenizer
    gc.collect()
    text_state = _extract_group(source_state, TEXT_PREFIX)
    target_model, target_inventory = _load_target_text_model(source_config, checkpoint, text_state)
    text_mapping = _audit_text_mapping(target_model, source_state, source_config)
    audio_groups = _audit_audio_groups(source_state)
    tokenizer_audit = _audit_tokenizer(base_smollm2)
    if int(tokenizer_audit["size"]) != int(getattr(source_config, "vocab_size", -1)):
        raise RuntimeError(
            f"tokenizer/model vocabulary mismatch: tokenizer={tokenizer_audit['size']} "
            f"model={getattr(source_config, 'vocab_size', None)}"
        )

    return {
        "status": "PASS",
        "stage": "mellow_v0_mesh_initialization_audit",
        "artifact_contract": ARTIFACT_CONTRACT,
        "offline": {
            "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
            "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
            "cuda_used": False,
        },
        "mellow_source_root": str(source_root),
        "mellow_source_sha256": source_hashes,
        "mellow_snapshot": str(snapshot),
        "snapshot_config_sha256": _sha256(snapshot / "config.json"),
        "mellow_checkpoint": str(checkpoint),
        "mellow_checkpoint_sha256": _sha256(checkpoint),
        "base_smollm2": str(base_smollm2),
        "base_smollm2_inventory": smollm2_inventory,
        "native_mellow_strict_state_load": {
            "passed": True,
            "state_shapes_match": True,
            "state_dtypes_match": True,
            "state_tensor_count": len(source_state),
            "state_parameter_count": int(sum(value.numel() for value in source_state.values())),
            "group_tensor_counts": group_counts,
            "unknown_keys": [],
            "native_tokenizer_size": native_tokenizer_size,
        },
        "source_text_config": {
            "model_type": str(getattr(source_config, "model_type", "")),
            "hidden_size": int(getattr(source_config, "hidden_size", -1)),
            "vocab_size": int(getattr(source_config, "vocab_size", -1)),
            "num_hidden_layers": int(getattr(source_config, "num_hidden_layers", -1)),
            "intermediate_size": int(getattr(source_config, "intermediate_size", -1)),
            "num_attention_heads": int(getattr(source_config, "num_attention_heads", -1)),
            "num_key_value_heads": int(getattr(source_config, "num_key_value_heads", -1)),
            "tie_word_embeddings": bool(getattr(source_config, "tie_word_embeddings", False)),
        },
        "target_mesh": target_inventory,
        "text_mapping": text_mapping,
        "audio_groups": audio_groups,
        "tokenizer": tokenizer_audit,
        "source_prefix_contract": dict(NATIVE_SOURCE_PREFIXES),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mellow-source-root", type=Path, default=DEFAULT_MELLOW_SOURCE_ROOT)
    parser.add_argument("--mellow-snapshot", type=Path, default=DEFAULT_MELLOW_SNAPSHOT)
    parser.add_argument("--mellow-checkpoint", type=Path, default=DEFAULT_MELLOW_CHECKPOINT)
    parser.add_argument("--base-smollm2", type=Path, default=DEFAULT_BASE_SMOLLM2)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = _audit(args)
    except Exception as exc:
        import traceback

        report = {
            "status": "FAILED",
            "stage": "mellow_v0_mesh_initialization_audit",
            "artifact_contract": ARTIFACT_CONTRACT,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "cuda_used": False,
        }
    _write_json(args.report_path.expanduser().resolve(), report)
    print(json.dumps(_json_safe(report), ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
