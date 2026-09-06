#!/usr/bin/env python3
"""Convert a clean SmolLM2/5-10-5 directory to the isolated MeSH model.

The converter accepts either the original 30-layer SmolLM2 checkpoint or a
conversion-only 5-10-5 directory containing twenty physical layers.  Training
state markers are rejected by default so a formal checkpoint cannot silently
become a MeSH initialization source.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from recursive_model_5_10x2_5_mesh import (  # noqa: E402
    LOGICAL_TO_PHYSICAL,
    MEMORY_SLOT_COUNT,
    MAPPING_POLICY,
    PHYSICAL_LAYER_COUNT,
    ROUTER_PARAMETER_COUNT,
    SOURCE_LAYER_INDICES_0BASED,
    RecursiveLlamaForCausalLM,
    parameter_audit,
    register_auto_class,
)

FORBIDDEN_CHECKOUT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM")
LOCAL_CHECKOUT = SCRIPT_ROOT.parents[1]
DEFAULT_SOURCE_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5")
DEFAULT_OUTPUT_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x2-5-mesh")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
TOKENIZER_NAMES = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model", "spiece.model", "vocab.json", "merges.txt", "added_tokens.json", "generation_config.json"}
TRAINING_MARKERS = {"training_state.pt", "optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin", "rng_state.pth", "scaler.pt", "checkpoint_state.pt"}


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def reject_forbidden_output(path: Path) -> Path:
    candidate = _resolved(path)
    for forbidden in (_resolved(FORBIDDEN_CHECKOUT), _resolved(LOCAL_CHECKOUT)):
        try:
            candidate.relative_to(forbidden)
        except ValueError:
            continue
        raise ValueError(f"Refusing to write model artifacts inside Git checkout: {candidate}")
    return candidate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--allow-training-state", action="store_true", help="unsafe escape hatch; never use for formal initialization")
    return parser.parse_args(argv)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"<unavailable:{type(exc).__name__}>"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception as exc:
        return f"<unavailable:{type(exc).__name__}>"


def _training_markers(source: Path) -> list[str]:
    found: list[str] = []
    names = {str(path.name).lower() for path in source.rglob("*")}
    if source.name.lower().startswith("checkpoint-"):
        found.append(source.name)
    for name in names:
        if name in {"checkpoint_complete.json", "mesh_checkpoint_metadata.json", "mesh_conversion_metadata.json"}:
            found.append(name)
        if name in {item.lower() for item in TRAINING_MARKERS} or name.startswith("checkpoint-"):
            found.append(name)
        if any(token in name for token in ("optimizer", "scheduler", "trainer_state", "training_state", "training_args", "rng_state", "scaler_state")):
            found.append(name)
    return sorted(found)


def detect_source(source: Path, *, allow_training_state: bool = False) -> dict[str, Any]:
    config_path = source / "config.json"
    if not source.is_dir() or not config_path.is_file():
        raise FileNotFoundError(f"clean source directory/config.json not found: {source}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    markers = _training_markers(source)
    metadata_path = source / "recursive_conversion_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    recursive_count = raw.get("recursive_layer_count")
    if recursive_count is not None and int(recursive_count) == 20:
        kind = "conversion_only_5_10_5"
        physical_layers = 20
        mapping = tuple(range(20))
        if not metadata_path.is_file() or not str(metadata.get("conversion", "")).startswith("SmolLM2-5-10-5"):
            raise ValueError("20-layer source must contain explicit conversion-only 5-10-5 metadata")
        if metadata.get("status") not in (None, "ok") or any(key in metadata for key in ("optimizer_step", "scheduler", "checkpoint", "training", "resume_from")):
            raise ValueError("20-layer source metadata indicates training/checkpoint state")
        if raw.get("architectures") and not any("RecursiveLlama" in str(item) for item in raw["architectures"]):
            raise ValueError("20-layer conversion-only source has incompatible architectures metadata")
    elif int(raw.get("num_hidden_layers", -1)) == 30:
        kind = "original_smolLM2_30_layer"
        physical_layers = 30
        mapping = SOURCE_LAYER_INDICES_0BASED
    else:
        raise ValueError("source must be original 30-layer SmolLM2 or conversion-only 20-layer 5-10-5")
    if markers and not allow_training_state:
        raise ValueError(f"source contains training/checkpoint markers: {markers}")
    if kind == "conversion_only_5_10_5" and raw.get("recursive_loops") not in (None, 2):
        raise ValueError("conversion-only source has an incompatible recursive_loops value")
    return {"kind": kind, "physical_layers": physical_layers, "mapping": mapping, "raw_config": raw, "metadata": metadata, "training_markers": markers}


def _load_weights(source: Path) -> dict[str, Any]:
    safetensors_files = sorted(source.glob("*.safetensors"))
    if safetensors_files:
        from safetensors.torch import load_file
        state: dict[str, Any] = {}
        for path in safetensors_files:
            state.update(load_file(str(path), device="cpu"))
        return state
    bins = sorted(source.glob("*.bin")) + sorted(source.glob("*.pt"))
    bins = [p for p in bins if p.name not in TRAINING_MARKERS]
    if not bins:
        raise FileNotFoundError(f"no model weight files found under {source}")
    import torch
    state = {}
    for path in bins:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict) and "state_dict" in loaded:
            loaded = loaded["state_dict"]
        if not isinstance(loaded, dict):
            raise ValueError(f"unsupported weight file: {path}")
        state.update(loaded)
    return state


def _remap_key(key: str, source_kind: str, source_to_target: dict[int, int]) -> str | None:
    prefix = "model.layers."
    if key.startswith(prefix):
        tail = key[len(prefix):]
        parts = tail.split(".", 1)
        if not parts[0].isdigit():
            return None
        source_index = int(parts[0])
        if source_index not in source_to_target:
            return None
        suffix = parts[1] if len(parts) == 2 else ""
        return f"model.layers.{source_to_target[source_index]}.{suffix}"
    if key.startswith("transformer."):
        key = key[len("transformer."):]
    return key if (key.startswith("model.") or key.startswith("lm_head.")) else None


def _restore_tied_weight_aliases(remapped: dict[str, Any], target_state: dict[str, Any], source_config: Any) -> list[str]:
    """Restore shared tensor aliases omitted by safe serialization."""

    if not bool(getattr(source_config, "tie_word_embeddings", False)):
        return []
    embedding_key = "model.embed_tokens.weight"
    lm_head_key = "lm_head.weight"
    present_key: str | None = None
    missing_key: str | None = None
    if embedding_key in remapped and lm_head_key not in remapped:
        present_key, missing_key = embedding_key, lm_head_key
    elif lm_head_key in remapped and embedding_key not in remapped:
        present_key, missing_key = lm_head_key, embedding_key
    if present_key is None or missing_key is None:
        return []
    if present_key not in target_state or missing_key not in target_state:
        raise ValueError("tied embedding/LM-head keys are absent from the target model")
    source_tensor = remapped[present_key]
    if tuple(source_tensor.shape) != tuple(target_state[missing_key].shape):
        raise ValueError(
            f"cannot restore tied weight {missing_key}: source shape={tuple(source_tensor.shape)} "
            f"target shape={tuple(target_state[missing_key].shape)}"
        )
    remapped[missing_key] = source_tensor
    return [missing_key]


def _copy_tokenizer(source: Path, target: Path) -> list[str]:
    copied: list[str] = []
    for path in source.iterdir():
        if not path.is_file() or path.name == "config.json" or path.suffix in WEIGHT_SUFFIXES:
            continue
        if path.name in TOKENIZER_NAMES or path.name.startswith("tokenizer") or path.name.endswith((".model", ".jinja")):
            shutil.copy2(path, target / path.name)
            copied.append(path.name)
    return sorted(set(copied))


def _target_config(source_config: Any, source_info: dict[str, Any], source: Path) -> Any:
    target = copy.deepcopy(source_config)
    target.num_hidden_layers = 30
    target.recursive_layer_count = 20
    target.recursive_loops = 2
    target.recursive_prefix_layer_count = 5
    target.recursive_middle_layer_count = 10
    target.recursive_suffix_layer_count = 5
    target.middle_recurrent_count = 10
    target.recursive_loops_scope = "middle_only"
    target.logical_to_physical = list(LOGICAL_TO_PHYSICAL)
    target.logical_to_physical_schedule = list(LOGICAL_TO_PHYSICAL)
    target.recursive_logical_to_physical = list(LOGICAL_TO_PHYSICAL)
    target.recursive_source_layer_indices_0based = list(SOURCE_LAYER_INDICES_0BASED)
    target.recursive_source_layer_indices_1based = [i + 1 for i in SOURCE_LAYER_INDICES_0BASED]
    target.recursive_mapping_policy = MAPPING_POLICY
    target.mesh_architecture_contract = "logical_30_physical_20_5_10x2_5"
    target.mesh_memory_slots = MEMORY_SLOT_COUNT
    target.mesh_router_count = ROUTER_PARAMETER_COUNT
    target.mesh_router_init = "truncated_normal_std_sqrt_2_over_5d_clamped_3std_bias_zero"
    target.mesh_transition_query = "prefix_output"
    target.mesh_embedding_scale = "disabled"
    target.mesh_memory_persistent = False
    target.mesh_source_kind = source_info["kind"]
    target.mesh_source_path = str(source)
    target.architectures = ["RecursiveLlama5_10x2_5MeshForCausalLM"]
    return target


def convert(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig

    source = _resolved(args.source_checkpoint)
    output = reject_forbidden_output(args.output_dir)
    info = detect_source(source, allow_training_state=args.allow_training_state)
    if output.exists() and not args.allow_overwrite:
        raise FileExistsError(f"output already exists: {output}; pass --allow-overwrite")
    source_config = AutoConfig.from_pretrained(source, local_files_only=True)
    if getattr(source_config, "model_type", None) != "llama":
        raise ValueError("MeSH converter currently supports only Llama/SmolLM2 checkpoints")
    torch.manual_seed(args.seed)
    state = _load_weights(source)
    target_config = _target_config(source_config, info, source)
    register_auto_class()
    target_model = RecursiveLlamaForCausalLM(target_config)
    first_tensor = next((value for value in state.values() if isinstance(value, torch.Tensor) and value.is_floating_point()), None)
    if first_tensor is not None:
        target_model.to(dtype=first_tensor.dtype)
    target_state = target_model.state_dict()
    source_to_target = {source_index: target_index for target_index, source_index in enumerate(info["mapping"])}
    remapped: dict[str, Any] = {}
    for key, value in state.items():
        new_key = _remap_key(str(key), info["kind"], source_to_target)
        if new_key is not None and new_key in target_state:
            remapped[new_key] = value
    restored_tied_weight_aliases = _restore_tied_weight_aliases(remapped, target_state, source_config)
    required_inherited = [key for key in target_state if not ("write_routers." in key or "read_routers." in key)]
    missing = sorted(set(required_inherited) - set(remapped))
    if missing:
        raise ValueError(f"source is missing inherited parameters, first missing={missing[:8]}")
    target_model.load_state_dict(remapped, strict=False)
    target_model.tie_weights()
    audit = parameter_audit(target_model)
    target_model.config.mesh_parameter_audit = audit
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        target_model.save_pretrained(staging, safe_serialization=True)
        copied = _copy_tokenizer(source, staging)
        metadata = {
            "status": "ok", "conversion": "SmolLM2-5-10x2-5-mesh", "source_checkpoint": str(source), "target_output": str(output),
            "source_kind": info["kind"], "source_training_markers": info["training_markers"], "source_config": info["raw_config"],
            "logical_layer_count": 30, "physical_layer_count": PHYSICAL_LAYER_COUNT, "prefix_layer_count": 5, "middle_layer_count": 10, "suffix_layer_count": 5, "loops": 2,
            "logical_to_physical": list(LOGICAL_TO_PHYSICAL), "source_layer_indices_0based": list(SOURCE_LAYER_INDICES_0BASED), "mapping_policy": MAPPING_POLICY,
            "memory_slots": MEMORY_SLOT_COUNT, "router_parameter_count": ROUTER_PARAMETER_COUNT, "transition_query": "prefix_output", "embedding_scale": "disabled",
            "router_init": "truncated_normal_std_sqrt_2_over_5d_clamped_3std_bias_zero", "memory_persistent": False, "architectures": list(target_config.architectures),
            "parameter_audit": audit, "copied_tokenizer_files": copied, "seed": args.seed, "code_commit": git_commit(), "python": sys.version,
            "restored_tied_weight_aliases": restored_tied_weight_aliases,
            "platform": platform.platform(), "torch": torch.__version__, "transformers": _package_version("transformers"), "conversion_time_utc": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "mesh_conversion_metadata.json").write_text(json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metadata


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = convert(args)
    print(f"[result] status=PASS output={_resolved(args.output_dir)}", flush=True)
    print(json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
