#!/usr/bin/env python3
"""Convert a local SmolLM2/Llama checkpoint to a strict shared recursive model.

The source and destination are deliberately explicit.  This script never
downloads from the Hub and refuses to write model parameters inside the remote
Git checkout used to synchronize this repository.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The script is normally launched as ``python code/RSmol/scripts/...``; add its
# package parent explicitly so this works independently of the caller's cwd.
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))


FORBIDDEN_CHECKOUT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
TOKENIZER_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "generation_config.json",
}


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def reject_forbidden_output(path: Path) -> None:
    candidate = _resolved(path)
    forbidden = _resolved(FORBIDDEN_CHECKOUT)
    try:
        candidate.relative_to(forbidden)
    except ValueError:
        return
    raise ValueError(
        "Refusing to write model parameters inside the remote Git checkout: "
        f"output={candidate} forbidden_root={forbidden}. Choose an external model directory."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument(
        "--source-layer-indices",
        default=None,
        help="Comma-separated 0-based indices, only for an explicit tiny fixture mapping.",
    )
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - git may not be available in a container
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def json_safe(value: Any) -> Any:
    # Keep path and metadata checks importable on a workstation without the
    # remote Torch/Transformers environment.  The conversion path imports
    # Torch lazily below, after all dependency-free validation has completed.
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised by dependency-free checks
        torch = None
    if isinstance(value, Path):
        return str(value)
    if torch is not None and isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def copy_module_checked(source: torch.nn.Module, target: torch.nn.Module, name: str) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    extra = sorted(set(source_state) - set(target_state))
    if missing or extra:
        raise ValueError(f"{name} state keys differ: missing_source={missing} extra_source={extra}")
    for key, target_tensor in target_state.items():
        source_tensor = source_state[key]
        if source_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"{name}.{key} shape mismatch: source={tuple(source_tensor.shape)} "
                f"target={tuple(target_tensor.shape)}"
            )
        if source_tensor.dtype != target_tensor.dtype:
            raise ValueError(
                f"{name}.{key} dtype mismatch: source={source_tensor.dtype} target={target_tensor.dtype}"
            )
    target.load_state_dict(source_state, strict=True)


def copy_non_layer_modules(source: Any, target: Any) -> None:
    for name in ("embed_tokens", "norm"):
        copy_module_checked(getattr(source.model, name), getattr(target.model, name), f"model.{name}")
    copy_module_checked(source.lm_head, target.lm_head, "lm_head")


def source_config_summary(config: Any) -> dict[str, Any]:
    keys = (
        "model_type",
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "torch_dtype",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    )
    result = {}
    for key in keys:
        result[key] = json_safe(getattr(config, key, None))
    return result


def build_target_config(
    source_config: Any,
    mapping: tuple[int, ...],
    *,
    loops: int,
    mapping_policy: str,
) -> Any:
    logical_layer_count = int(getattr(source_config, "num_hidden_layers", 0))
    physical_layer_count = len(mapping)
    if logical_layer_count <= 0:
        raise ValueError(
            "Source config num_hidden_layers must be a positive logical layer count, "
            f"got {logical_layer_count}"
        )
    recursive_loops = int(loops)
    if recursive_loops <= 0:
        raise ValueError(f"recursive_loops must be positive, got {loops}")
    if logical_layer_count != physical_layer_count * recursive_loops:
        raise ValueError(
            "Recursive target depth mismatch: source logical num_hidden_layers must equal "
            f"len(mapping) * loops ({physical_layer_count} * {recursive_loops}), "
            f"got {logical_layer_count}"
        )
    target_config = copy.deepcopy(source_config)
    # Keep the HF config depth logical.  GenerationMixin and DynamicCache use
    # num_hidden_layers to represent cache slots; the unique ModuleList size
    # is carried separately in recursive_layer_count.
    target_config.num_hidden_layers = logical_layer_count
    target_config.recursive_loops = recursive_loops
    target_config.recursive_layer_count = physical_layer_count
    target_config.recursive_source_layer_indices_0based = list(mapping)
    target_config.recursive_source_layer_indices_1based = [index + 1 for index in mapping]
    target_config.recursive_mapping_policy = mapping_policy
    target_config.architectures = ["RecursiveLlamaForCausalLM"]
    return target_config


def copy_tokenizer_files(source_dir: Path, target_dir: Path) -> list[str]:
    copied: list[str] = []
    for source_file in source_dir.iterdir():
        if not source_file.is_file():
            continue
        name = source_file.name
        # ``config.json`` was already written by target_model.save_pretrained;
        # copying the source one here would silently undo the recursive fields
        # and the recursive metadata fields.
        is_config = name == "generation_config.json"
        is_tokenizer = (
            name in TOKENIZER_NAMES
            or name.startswith("tokenizer")
            or name.endswith((".model", ".jinja"))
        )
        # Keep small non-weight auxiliary files, but never copy source model
        # weights into the temporary destination before save_pretrained.
        if (is_config or is_tokenizer) and not name.endswith(WEIGHT_SUFFIXES):
            shutil.copy2(source_file, target_dir / name)
            copied.append(name)
    return sorted(set(copied))


def atomic_replace_directory(staging: Path, output: Path, allow_overwrite: bool) -> None:
    if output.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Pass --allow-overwrite explicitly to replace it."
        )
    if output.exists():
        shutil.rmtree(output)
    staging.replace(output)


def convert(args: argparse.Namespace) -> dict[str, Any]:
    source = _resolved(args.source_checkpoint)
    output = _resolved(args.output_dir)
    reject_forbidden_output(output)
    if not source.is_dir():
        raise FileNotFoundError(f"Source checkpoint directory does not exist: {source}")
    config_file = source / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"Source checkpoint is missing config.json: {config_file}")
    if output.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Pass --allow-overwrite explicitly to replace it."
        )

    import torch
    from recursive_model import (
        DEFAULT_LOOPS,
        MAPPING_POLICY,
        PROJECT_SOURCE_LAYERS,
        RecursiveLlamaForCausalLM,
        build_stepwise_mapping,
        parameter_audit,
    )

    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        source_config_json = json.loads(config_file.read_text(encoding="utf-8"))
        raw_source_layers = source_config_json["num_hidden_layers"]
        if isinstance(raw_source_layers, bool) or not isinstance(raw_source_layers, int):
            raise TypeError(
                "num_hidden_layers must be a JSON integer, "
                f"got {raw_source_layers!r}"
            )
        source_layers_from_json = raw_source_layers
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Source config.json must contain an integer num_hidden_layers: {config_file}"
        ) from exc
    source_config = AutoConfig.from_pretrained(source, local_files_only=True)
    source_layers = source_layers_from_json
    loaded_source_layers = int(getattr(source_config, "num_hidden_layers", -1))
    if loaded_source_layers != source_layers:
        raise ValueError(
            "AutoConfig disagrees with source config.json: "
            f"json={source_layers} loaded={loaded_source_layers}"
        )
    if getattr(source_config, "model_type", None) != "llama":
        raise ValueError(
            "Stage 1 currently converts only Llama/SmolLM2 checkpoints; "
            f"source model_type={getattr(source_config, 'model_type', None)!r}"
        )
    explicit = None
    if args.source_layer_indices is not None:
        explicit = tuple(int(part.strip()) for part in args.source_layer_indices.split(",") if part.strip())
    mapping = build_stepwise_mapping(source_layers, source_layer_indices_0based=explicit)
    production_mapping = build_stepwise_mapping(PROJECT_SOURCE_LAYERS)
    if source_layers == PROJECT_SOURCE_LAYERS and mapping != production_mapping:
        raise ValueError(
            "The production L=30 mapping is fixed to source 0-based "
            f"{list(production_mapping)}; got {list(mapping)}"
        )
    if source_layers != PROJECT_SOURCE_LAYERS and explicit is None:
        raise ValueError(
            f"Current production conversion requires L={PROJECT_SOURCE_LAYERS}, got L={source_layers}; "
            "provide an explicit mapping only for a local fixture."
        )

    torch.manual_seed(args.seed)
    source_model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        use_safetensors=True,
    )
    if not hasattr(source_model, "model") or not hasattr(source_model.model, "layers"):
        raise TypeError("Source checkpoint is not a Llama-style causal LM with model.layers")
    if len(source_model.model.layers) != source_layers:
        raise ValueError(
            f"Source config/model layer mismatch: config={source_layers} actual={len(source_model.model.layers)}"
        )

    mapping_policy = MAPPING_POLICY if source_layers == PROJECT_SOURCE_LAYERS else "explicit_indices"
    target_config = build_target_config(
        source_config,
        mapping,
        loops=DEFAULT_LOOPS,
        mapping_policy=mapping_policy,
    )
    target_model = RecursiveLlamaForCausalLM(target_config)
    source_dtype = next(source_model.parameters()).dtype
    target_model.to(dtype=source_dtype)
    copy_non_layer_modules(source_model, target_model)
    for target_index, source_index in enumerate(mapping):
        copy_module_checked(
            source_model.model.layers[source_index],
            target_model.model.layers[target_index],
            f"model.layers[{target_index}]<-source.model.layers[{source_index}]",
        )
    target_model.tie_weights()

    audit = parameter_audit(target_model)
    if audit["physical_layer_count"] != len(mapping):
        raise AssertionError("Recursive model physical layer count does not match mapping")
    if audit["logical_layer_count"] != source_layers:
        raise AssertionError("Recursive model logical layer count does not match source config")
    if not audit["depth_consistent"]:
        raise AssertionError("Recursive model logical/physical layer depth is inconsistent")
    target_model.config.recursive_parameter_audit = audit
    target_model.config.recursive_source_layer_indices_0based = list(mapping)
    target_model.config.recursive_source_layer_indices_1based = [index + 1 for index in mapping]
    target_model.config.recursive_mapping_policy = mapping_policy

    staging_parent = output.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=staging_parent))
    try:
        target_model.save_pretrained(staging, safe_serialization=True)
        copied = copy_tokenizer_files(source, staging)
        metadata = {
            "status": "ok",
            "source_checkpoint": str(source),
            "source_config": source_config_summary(source_config),
            # Keep the old keys as explicit aliases for compatibility, while
            # the named fields below remove any ambiguity between logical and
            # physical depth.
            "target_layers": len(mapping),
            "source_layers": source_layers,
            "source_logical_layers": source_layers,
            "target_logical_layers": int(target_config.num_hidden_layers),
            "target_physical_unique_layers": len(mapping),
            "logical_layer_count": int(target_config.num_hidden_layers),
            "physical_layer_count": len(mapping),
            "recursive_layer_count": len(mapping),
            "recursive_loops": DEFAULT_LOOPS,
            "loops": DEFAULT_LOOPS,
            "mapping_policy": mapping_policy,
            "mapping": {
                "source_layer_indices_1based": [index + 1 for index in mapping],
                "source_layer_indices_0based": list(mapping),
                "target_physical_indices_0based": list(range(len(mapping))),
            },
            "seed": args.seed,
            "code_commit": git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "transformers_api_expected": "4.54.1",
            "parameter_audit": audit,
            "copied_tokenizer_files": copied,
            "conversion_time_utc": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "recursive_conversion_metadata.json").write_text(
            json.dumps(json_safe(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        atomic_replace_directory(staging, output, args.allow_overwrite)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metadata


def main() -> None:
    args = parse_args()
    try:
        metadata = convert(args)
    except Exception:
        print("[result] status=FAIL", file=sys.stderr, flush=True)
        raise
    print(f"[result] status=PASS output={args.output_dir.resolve()}", flush=True)
    print(json.dumps(json_safe(metadata), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
