#!/usr/bin/env python3
"""Convert a 30-layer SmolLM2 checkpoint to the dynamic 5-10xr-5-Poisson model.

Only the twenty physical layers in the fixed source mapping are copied. The
destination is assembled in a temporary sibling directory and atomically
renamed into the requested external model directory. Existing 15R files and
the Git checkout are never used as an output location.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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

FORBIDDEN_CHECKOUT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM")
LOCAL_CHECKOUT = SCRIPT_ROOT.parents[1]
DEFAULT_OUTPUT_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10xr-5-poisson")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
TOKENIZER_NAMES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "tokenizer.model", "spiece.model", "vocab.json", "merges.txt",
    "added_tokens.json", "generation_config.json",
}
SOURCE_LAYER_INDICES_0BASED = (0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29)
LOGICAL_TO_PHYSICAL = (tuple(range(5)) + tuple(range(5, 15)) * 10 + tuple(range(15, 20)))
MAPPING_POLICY = "explicit_5_10xr_5_source_layers"
SAMPLING_POLICY = "truncated_poisson"
SAMPLER_VERSION = "truncated_poisson_lambda7_support4_10_v1"
SAMPLER_KEY = "sha256_cpu_torch_generator_base_seed_rank_optimizer_step_microbatch_v1"
POISSON_LAMBDA = 7.0
POISSON_SUPPORT = tuple(range(4, 11))
POISSON_NORMALIZATION_Z = sum(
    math.exp(-POISSON_LAMBDA) * POISSON_LAMBDA**k / math.factorial(k)
    for k in POISSON_SUPPORT
)
POISSON_PROBABILITIES = tuple(
    (math.exp(-POISSON_LAMBDA) * POISSON_LAMBDA**k / math.factorial(k)) / POISSON_NORMALIZATION_Z
    for k in POISSON_SUPPORT
)


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def reject_forbidden_output(path: Path) -> Path:
    candidate = _resolved(path)
    for forbidden in (_resolved(FORBIDDEN_CHECKOUT), _resolved(LOCAL_CHECKOUT)):
        try:
            candidate.relative_to(forbidden)
        except ValueError:
            continue
        raise ValueError(
            "Refusing to write model parameters inside a Git checkout: "
            f"output={candidate} forbidden_root={forbidden}"
        )
    return candidate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def json_safe(value: Any) -> Any:
    try:
        import torch
    except ImportError:
        torch = None
    if isinstance(value, Path):
        return str(value)
    if torch is not None and isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def source_config_summary(config: Any) -> dict[str, Any]:
    keys = (
        "model_type", "architectures", "num_hidden_layers", "hidden_size",
        "intermediate_size", "num_attention_heads", "num_key_value_heads",
        "vocab_size", "max_position_embeddings", "torch_dtype", "bos_token_id",
        "eos_token_id", "pad_token_id",
    )
    return {key: json_safe(getattr(config, key, None)) for key in keys}


def copy_module_checked(source: Any, target: Any, name: str) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    extra = sorted(set(source_state) - set(target_state))
    if missing or extra:
        raise ValueError(f"{name} state keys differ: missing_source={missing} extra_source={extra}")
    for key, target_tensor in target_state.items():
        source_tensor = source_state[key]
        if source_tensor.shape != target_tensor.shape or source_tensor.dtype != target_tensor.dtype:
            raise ValueError(
                f"{name}.{key} mismatch: source={tuple(source_tensor.shape)}/{source_tensor.dtype} "
                f"target={tuple(target_tensor.shape)}/{target_tensor.dtype}"
            )
    target.load_state_dict(source_state, strict=True)


def copy_non_layer_modules(source: Any, target: Any) -> None:
    for name in ("embed_tokens", "norm"):
        copy_module_checked(getattr(source.model, name), getattr(target.model, name), f"model.{name}")
    copy_module_checked(source.lm_head, target.lm_head, "lm_head")


def copy_tokenizer_files(source_dir: Path, target_dir: Path) -> list[str]:
    copied: list[str] = []
    for source_file in source_dir.iterdir():
        if not source_file.is_file():
            continue
        name = source_file.name
        if name == "config.json":
            continue
        is_tokenizer = name in TOKENIZER_NAMES or name.startswith("tokenizer") or name.endswith((".model", ".jinja"))
        if is_tokenizer and not name.endswith(WEIGHT_SUFFIXES):
            shutil.copy2(source_file, target_dir / name)
            copied.append(name)
    return sorted(set(copied))


def build_target_config(source_config: Any) -> Any:
    """Set dynamic logical/cache and selective-gradient metadata."""

    if int(getattr(source_config, "num_hidden_layers", -1)) != 30:
        raise ValueError("SmolLM2-5-10xr-5 requires source num_hidden_layers=30")
    target = copy.deepcopy(source_config)
    # HF's layer count is used here as the maximum logical/cache namespace;
    # each forward selects a concrete r and executes 50..110 entries.
    target.num_hidden_layers = 110
    target.recursive_source_num_hidden_layers = 30
    target.recursive_source_layer_count = 30
    target.recursive_layer_count = 20
    target.recursive_loops = 10
    target.recursive_min_middle_loops = 4
    target.recursive_max_middle_loops = 10
    target.recursive_default_inference_middle_loops = 7
    target.recursive_parameter_gradient_tail_loops = 4
    target.recursive_backward_policy = "selective_parameter_gradients_final_four_middle_calls_v1"
    target.recursive_training_loop_mode = "per_local_microbatch_per_sequence_truncated_poisson"
    target.recursive_sampling_policy = SAMPLING_POLICY
    target.recursive_sampler_version = SAMPLER_VERSION
    target.recursive_poisson_lambda = POISSON_LAMBDA
    target.recursive_poisson_support = list(POISSON_SUPPORT)
    target.recursive_poisson_normalization_z = POISSON_NORMALIZATION_Z
    target.recursive_poisson_Z = POISSON_NORMALIZATION_Z
    target.recursive_poisson_probabilities = list(POISSON_PROBABILITIES)
    target.recursive_sampler_key = SAMPLER_KEY
    target.recursive_min_logical_layer_count = 50
    target.recursive_max_logical_layer_count = 110
    target.recursive_prefix_layer_count = 5
    target.recursive_middle_layer_count = 10
    target.recursive_suffix_layer_count = 5
    target.middle_recurrent_count = 10
    target.recursive_loops_scope = "middle_only"
    target.recursive_prefix_source_layers_1based = [1, 2, 3, 4, 5]
    target.recursive_middle_source_layers_1based = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    target.recursive_suffix_source_layers_1based = [26, 27, 28, 29, 30]
    target.logical_to_physical = list(LOGICAL_TO_PHYSICAL)
    target.recursive_logical_to_physical = list(LOGICAL_TO_PHYSICAL)
    target.logical_to_physical_schedule = list(LOGICAL_TO_PHYSICAL)
    target.recursive_logical_to_physical_schedule = list(LOGICAL_TO_PHYSICAL)
    target.recursive_source_layer_indices_0based = list(SOURCE_LAYER_INDICES_0BASED)
    target.recursive_source_layer_indices_1based = [index + 1 for index in SOURCE_LAYER_INDICES_0BASED]
    target.recursive_source_mapping_0based = list(SOURCE_LAYER_INDICES_0BASED)
    target.recursive_source_mapping_1based = [index + 1 for index in SOURCE_LAYER_INDICES_0BASED]
    target.recursive_mapping_policy = MAPPING_POLICY
    target.architectures = ["RecursiveLlama5_10xr_5ForCausalLM"]
    return target


def build_source_mapping(num_hidden_layers: int = 30) -> tuple[int, ...]:
    """Return the fixed twenty-module source mapping after depth validation."""

    if int(num_hidden_layers) != 30:
        raise ValueError(f"SmolLM2-5-10xr-5 requires source num_hidden_layers=30, got {num_hidden_layers}")
    return SOURCE_LAYER_INDICES_0BASED


def _parameter_audit(model: Any) -> dict[str, Any]:
    names = list(model.named_parameters(remove_duplicate=False))
    unique = {id(parameter): parameter for _, parameter in names}
    return {
        "parameter_count_unique": sum(parameter.numel() for parameter in unique.values()),
        "parameter_count_references": sum(parameter.numel() for _, parameter in names),
        "source_logical_layer_count": 30,
        "source_physical_layer_count": 30,
        "physical_layer_count": len(model.model.layers),
        "logical_layer_count": int(model.config.num_hidden_layers),
        "logical_cache_slot_count": 110,
        "recursive_loops": 10,
        "min_middle_loops": 4,
        "max_middle_loops": 10,
        "default_inference_middle_loops": 7,
        "parameter_gradient_tail_loops": 4,
        "sampling_policy": SAMPLING_POLICY,
        "sampler_version": SAMPLER_VERSION,
        "poisson_lambda": POISSON_LAMBDA,
        "poisson_support": list(POISSON_SUPPORT),
        "poisson_normalization_z": POISSON_NORMALIZATION_Z,
        "poisson_probabilities": list(POISSON_PROBABILITIES),
        "middle_recurrent_count": 10,
        "schedule": list(LOGICAL_TO_PHYSICAL),
        "shared_storage_is_unique": len(unique) == len(list(model.parameters())),
    }


def convert(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from recursive_model_5_10xr_5 import RecursiveLlama5_10xr_5ForCausalLM, register_auto_class

    source = _resolved(args.source_checkpoint)
    output = reject_forbidden_output(args.output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Source checkpoint directory does not exist: {source}")
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Source checkpoint is missing config.json: {config_path}")
    if output.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --allow-overwrite explicitly")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw_layers = raw.get("num_hidden_layers")
    if isinstance(raw_layers, bool) or not isinstance(raw_layers, int):
        raise ValueError("source config.json num_hidden_layers must be an integer")
    if raw_layers != 30:
        raise ValueError(f"SmolLM2-5-10xr-5 requires source config num_hidden_layers=30, got {raw_layers}")
    source_config = AutoConfig.from_pretrained(source, local_files_only=True)
    actual_layers = int(getattr(source_config, "num_hidden_layers", -1))
    if actual_layers != raw_layers:
        raise ValueError(f"AutoConfig disagrees with source config.json: json={raw_layers} loaded={actual_layers}")
    if actual_layers != 30:
        raise ValueError(f"AutoConfig source layer count must be 30, got {actual_layers}")
    if getattr(source_config, "model_type", None) != "llama":
        raise ValueError("5-10xr-5 conversion currently supports only Llama/SmolLM2 checkpoints")
    mapping = build_source_mapping(actual_layers)
    torch.manual_seed(args.seed)
    source_model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, use_safetensors=True)
    source_layers_module = getattr(getattr(source_model, "model", None), "layers", ())
    if len(source_layers_module) != 30:
        raise ValueError(f"Source model layer count must be 30, got {len(source_layers_module)}")
    register_auto_class()
    target_config = build_target_config(source_config)
    target_model = RecursiveLlama5_10xr_5ForCausalLM(target_config)
    target_model.to(dtype=next(source_model.parameters()).dtype)
    copy_non_layer_modules(source_model, target_model)
    for target_index, source_index in enumerate(mapping):
        copy_module_checked(source_model.model.layers[source_index], target_model.model.layers[target_index], f"model.layers[{target_index}]<-source.model.layers[{source_index}]")
    target_model.tie_weights()
    audit = _parameter_audit(target_model)
    target_model.config.recursive_parameter_audit = audit
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        target_model.save_pretrained(staging, safe_serialization=True)
        copied = copy_tokenizer_files(source, staging)
        metadata = {
            "status": "ok", "conversion": "SmolLM2-5-10xr-5", "source_checkpoint": str(source),
            "target_output": str(output), "source_config": source_config_summary(source_config),
            "source_config_json": raw, "target_config": target_config.to_dict(),
            "source_logical_layer_count": 30, "target_logical_layer_count": 110,
            "physical_layer_count": 20, "recursive_layer_count": 20,
            "source_layers": 30, "target_layers": 20,
            "target_physical_unique_layers": 20,
            "prefix_layer_count": 5, "middle_recurrent_count": 10, "suffix_layer_count": 5,
            "loops": 10, "recursive_loops": 10, "min_middle_loops": 4, "max_middle_loops": 10,
            "default_inference_middle_loops": 7, "parameter_gradient_tail_loops": 4,
            "loops_scope": "middle_only", "recursive_loops_scope": "middle_only",
            "logical_to_physical": list(LOGICAL_TO_PHYSICAL),
            "logical_to_physical_schedule": list(LOGICAL_TO_PHYSICAL),
            "source_layer_indices_0based": list(mapping),
            "source_layer_indices_1based": [index + 1 for index in mapping],
            "prefix_source_layers_1based": [1, 2, 3, 4, 5],
            "middle_source_layers_1based": [6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
            "suffix_source_layers_1based": [26, 27, 28, 29, 30],
            "prefix_physical_indices_0based": [0, 1, 2, 3, 4],
            "middle_physical_indices_0based": list(range(5, 15)),
            "suffix_physical_indices_0based": [15, 16, 17, 18, 19],
            "architectures": ["RecursiveLlama5_10xr_5ForCausalLM"], "mapping_policy": MAPPING_POLICY,
            "backward_policy": "selective_parameter_gradients_final_four_middle_calls_v1",
            "sampling_policy": SAMPLING_POLICY, "sampler_version": SAMPLER_VERSION,
            "poisson_lambda": POISSON_LAMBDA, "poisson_support": list(POISSON_SUPPORT),
            "poisson_normalization_z": POISSON_NORMALIZATION_Z,
            "poisson_probabilities": list(POISSON_PROBABILITIES), "sampler_key": SAMPLER_KEY,
            "parameter_audit": audit, "copied_tokenizer_files": copied, "seed": args.seed,
            "code_commit": git_commit(), "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "transformers": package_version("transformers"),
            "transformers_api_expected": "4.54.1", "conversion_time_utc": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "recursive_conversion_metadata.json").write_text(json.dumps(json_safe(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            if not args.allow_overwrite:
                raise FileExistsError(f"Output already exists: {output}")
            shutil.rmtree(output)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metadata


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        metadata = convert(args)
    except Exception:
        print("[result] status=FAIL", file=sys.stderr, flush=True)
        raise
    print(f"[result] status=PASS output={_resolved(args.output_dir)}", flush=True)
    print(json.dumps(json_safe(metadata), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
