#!/usr/bin/env python3
"""Convert 30-layer SmolLM2 to isolated 5-10-5 adjacent-pair average.

Source layers 0--4 and 25--29 are copied exactly.  Each target middle layer
is initialized by the elementwise FP32 arithmetic mean of one adjacent source
pair: (5,6), (7,8), ..., (23,24).  The target keeps the established 20
physical / 30 logical 5-10-5 execution schedule and is written atomically to
an external directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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

FORBIDDEN_CHECKOUT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM")
LOCAL_CHECKOUT = SCRIPT_ROOT.parents[1]
DEFAULT_OUTPUT_DIR = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5-adjacent-average"
)
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

LOGICAL_TO_PHYSICAL = (
    0, 1, 2, 3, 4,
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19,
)
PREFIX_SOURCE_LAYERS_0BASED = (0, 1, 2, 3, 4)
MIDDLE_SOURCE_LAYER_PAIRS_0BASED = (
    (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
    (15, 16), (17, 18), (19, 20), (21, 22), (23, 24),
)
SUFFIX_SOURCE_LAYERS_0BASED = (25, 26, 27, 28, 29)
SOURCE_LAYER_COVERAGE_0BASED = tuple(range(30))
INITIALIZATION_POLICY = "adjacent_layer_parameter_average_fp32_v1"
INITIALIZATION_CONTRACT = (
    "prefix_exact_middle_adjacent_pair_fp32_mean_suffix_exact_v1"
)
AVERAGE_ACCUMULATOR_DTYPE = "float32"
ARCHITECTURE_CONTRACT = "logical_30_physical_20_5_10_5_loops_2"


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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True
        ).strip()
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


def _state_schema(module: Any) -> dict[str, tuple[tuple[int, ...], Any]]:
    return {
        key: (tuple(tensor.shape), tensor.dtype)
        for key, tensor in module.state_dict().items()
    }


def copy_module_checked(source: Any, target: Any, name: str) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    if set(source_state) != set(target_state):
        raise ValueError(
            f"{name} state keys differ: "
            f"missing_source={sorted(set(target_state) - set(source_state))} "
            f"extra_source={sorted(set(source_state) - set(target_state))}"
        )
    if _state_schema(source) != _state_schema(target):
        raise ValueError(f"{name} state tensor schemas differ")
    target.load_state_dict(source_state, strict=True)


def average_modules_checked(source_a: Any, source_b: Any, target: Any, name: str) -> dict[str, Any]:
    """Load an exact FP32 elementwise mean into one target decoder layer."""

    import torch

    state_a = source_a.state_dict()
    state_b = source_b.state_dict()
    target_state = target.state_dict()
    keys_a = set(state_a)
    if keys_a != set(state_b) or keys_a != set(target_state):
        raise ValueError(f"{name} state keys differ across source pair and target")
    averaged: dict[str, torch.Tensor] = {}
    floating_keys: list[str] = []
    copied_nonfloating_keys: list[str] = []
    for key in sorted(target_state):
        left = state_a[key]
        right = state_b[key]
        destination = target_state[key]
        if left.shape != right.shape or left.shape != destination.shape:
            raise ValueError(
                f"{name}.{key} shape mismatch: "
                f"left={tuple(left.shape)} right={tuple(right.shape)} "
                f"target={tuple(destination.shape)}"
            )
        if left.dtype != right.dtype or left.dtype != destination.dtype:
            raise ValueError(
                f"{name}.{key} dtype mismatch: "
                f"left={left.dtype} right={right.dtype} target={destination.dtype}"
            )
        if left.is_floating_point():
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise ValueError(f"{name}.{key} source contains non-finite values")
            mean = (left.float() + right.float()).mul_(0.5)
            if not torch.isfinite(mean).all():
                raise ValueError(f"{name}.{key} FP32 mean contains non-finite values")
            averaged[key] = mean.to(dtype=destination.dtype)
            floating_keys.append(key)
        else:
            if not torch.equal(left, right):
                raise ValueError(
                    f"{name}.{key} non-floating pair values differ and cannot be averaged"
                )
            averaged[key] = left.clone()
            copied_nonfloating_keys.append(key)
    target.load_state_dict(averaged, strict=True)
    loaded = target.state_dict()
    for key, expected in averaged.items():
        if not torch.equal(loaded[key], expected):
            raise AssertionError(f"{name}.{key} failed exact post-load verification")
    return {
        "floating_tensor_count": len(floating_keys),
        "nonfloating_tensor_count": len(copied_nonfloating_keys),
        "floating_keys": floating_keys,
        "copied_nonfloating_keys": copied_nonfloating_keys,
    }


def verify_module_equal_checked(source: Any, target: Any, name: str) -> None:
    import torch

    source_state = source.state_dict()
    target_state = target.state_dict()
    if set(source_state) != set(target_state):
        raise ValueError(f"{name} state keys differ during saved-artifact verification")
    for key in sorted(target_state):
        if not torch.equal(source_state[key], target_state[key]):
            difference = (
                float((source_state[key].float() - target_state[key].float()).abs().max().item())
                if source_state[key].numel()
                else 0.0
            )
            raise AssertionError(
                f"{name}.{key} is not an exact saved copy; max_abs_diff={difference}"
            )


def verify_module_average_checked(source_a: Any, source_b: Any, target: Any, name: str) -> None:
    import torch

    state_a = source_a.state_dict()
    state_b = source_b.state_dict()
    target_state = target.state_dict()
    if set(state_a) != set(state_b) or set(state_a) != set(target_state):
        raise ValueError(f"{name} state keys differ during saved-average verification")
    for key in sorted(target_state):
        left = state_a[key]
        right = state_b[key]
        actual = target_state[key]
        if left.is_floating_point():
            expected = ((left.float() + right.float()) * 0.5).to(dtype=actual.dtype)
        else:
            if not torch.equal(left, right):
                raise ValueError(f"{name}.{key} non-floating source values differ")
            expected = left
        if not torch.equal(expected, actual):
            difference = (
                float((expected.float() - actual.float()).abs().max().item())
                if expected.numel()
                else 0.0
            )
            raise AssertionError(
                f"{name}.{key} saved mean mismatch; max_abs_diff={difference}"
            )


def verify_saved_model(source_model: Any, saved_model: Any) -> dict[str, Any]:
    """Verify the serialized checkpoint, not only the in-memory target."""

    verify_module_equal_checked(
        source_model.model.embed_tokens, saved_model.model.embed_tokens,
        "saved.model.embed_tokens",
    )
    verify_module_equal_checked(source_model.model.norm, saved_model.model.norm, "saved.model.norm")
    verify_module_equal_checked(source_model.lm_head, saved_model.lm_head, "saved.lm_head")
    source_layers = source_model.model.layers
    saved_layers = saved_model.model.layers
    for target_index, source_index in enumerate(PREFIX_SOURCE_LAYERS_0BASED):
        verify_module_equal_checked(
            source_layers[source_index], saved_layers[target_index],
            f"saved.model.layers[{target_index}]<-source.layers[{source_index}]",
        )
    for offset, pair in enumerate(MIDDLE_SOURCE_LAYER_PAIRS_0BASED):
        target_index = 5 + offset
        verify_module_average_checked(
            source_layers[pair[0]], source_layers[pair[1]], saved_layers[target_index],
            f"saved.model.layers[{target_index}]<-mean(source.layers[{pair[0]}],source.layers[{pair[1]}])",
        )
    for offset, source_index in enumerate(SUFFIX_SOURCE_LAYERS_0BASED):
        target_index = 15 + offset
        verify_module_equal_checked(
            source_layers[source_index], saved_layers[target_index],
            f"saved.model.layers[{target_index}]<-source.layers[{source_index}]",
        )
    return {
        "status": "PASS",
        "non_layer_exact_modules": ["embed_tokens", "norm", "lm_head"],
        "exact_decoder_layer_count": 10,
        "averaged_decoder_layer_count": 10,
        "verification": "exact tensor equality after save_pretrained/from_pretrained",
    }


def copy_non_layer_modules(source: Any, target: Any) -> None:
    for name in ("embed_tokens", "norm"):
        copy_module_checked(
            getattr(source.model, name), getattr(target.model, name), f"model.{name}"
        )
    copy_module_checked(source.lm_head, target.lm_head, "lm_head")


def copy_tokenizer_files(source_dir: Path, target_dir: Path) -> list[str]:
    copied: list[str] = []
    for source_file in source_dir.iterdir():
        if not source_file.is_file() or source_file.name == "config.json":
            continue
        name = source_file.name
        is_tokenizer = (
            name in TOKENIZER_NAMES
            or name.startswith("tokenizer")
            or name.endswith((".model", ".jinja"))
        )
        if is_tokenizer and not name.endswith(WEIGHT_SUFFIXES):
            shutil.copy2(source_file, target_dir / name)
            copied.append(name)
    return sorted(set(copied))


def source_layer_coverage() -> tuple[int, ...]:
    coverage = (
        PREFIX_SOURCE_LAYERS_0BASED
        + tuple(index for pair in MIDDLE_SOURCE_LAYER_PAIRS_0BASED for index in pair)
        + SUFFIX_SOURCE_LAYERS_0BASED
    )
    if coverage != SOURCE_LAYER_COVERAGE_0BASED or len(set(coverage)) != 30:
        raise AssertionError(f"invalid adjacent-average source coverage: {coverage}")
    return coverage


def build_target_config(source_config: Any) -> Any:
    if int(getattr(source_config, "num_hidden_layers", -1)) != 30:
        raise ValueError("Adjacent-average 5-10-5 requires source num_hidden_layers=30")
    source_layer_coverage()
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
    target.recursive_logical_to_physical = list(LOGICAL_TO_PHYSICAL)
    target.logical_to_physical_schedule = list(LOGICAL_TO_PHYSICAL)
    target.recursive_logical_to_physical_schedule = list(LOGICAL_TO_PHYSICAL)
    target.recursive_mapping_policy = INITIALIZATION_POLICY
    target.recursive_initialization_policy = INITIALIZATION_POLICY
    target.recursive_initialization_contract = INITIALIZATION_CONTRACT
    target.recursive_average_accumulator_dtype = AVERAGE_ACCUMULATOR_DTYPE
    target.recursive_prefix_source_layers_0based = list(PREFIX_SOURCE_LAYERS_0BASED)
    target.recursive_middle_source_layer_pairs_0based = [
        list(pair) for pair in MIDDLE_SOURCE_LAYER_PAIRS_0BASED
    ]
    target.recursive_suffix_source_layers_0based = list(SUFFIX_SOURCE_LAYERS_0BASED)
    target.recursive_source_layer_coverage_0based = list(SOURCE_LAYER_COVERAGE_0BASED)
    target.recursive_source_layer_indices_0based = None
    target.recursive_source_layer_indices_1based = None
    target.recursive_source_mapping_0based = None
    target.recursive_source_mapping_1based = None
    target.architectures = ["RecursiveLlamaForCausalLM"]
    return target


def _parameter_audit(model: Any) -> dict[str, Any]:
    names = list(model.named_parameters(remove_duplicate=False))
    unique = {id(parameter): parameter for _, parameter in names}
    return {
        "parameter_count_unique": sum(parameter.numel() for parameter in unique.values()),
        "parameter_count_references": sum(parameter.numel() for _, parameter in names),
        "physical_layer_count": len(model.model.layers),
        "logical_layer_count": int(model.config.num_hidden_layers),
        "logical_cache_slot_count": 30,
        "recursive_loops": 2,
        "middle_recurrent_count": 10,
        "schedule": list(LOGICAL_TO_PHYSICAL),
        "parameter_storage_unique": len(unique) == len(list(model.parameters())),
        "initialization_policy": INITIALIZATION_POLICY,
        "initialization_contract": INITIALIZATION_CONTRACT,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from recursive_model_5_10_5_adjacent_average import (
        RecursiveLlamaForCausalLM,
        register_auto_class,
    )

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
    if isinstance(raw_layers, bool) or not isinstance(raw_layers, int) or raw_layers != 30:
        raise ValueError(f"Source config.json must declare exactly 30 layers, got {raw_layers!r}")
    recursive_source_fields = sorted(
        key for key in raw
        if key.startswith("recursive_") or key in {
            "logical_to_physical", "logical_to_physical_schedule", "middle_recurrent_count"
        }
    )
    if recursive_source_fields:
        raise ValueError(
            "Adjacent-average source must be the original 30-physical-layer SmolLM2, "
            f"not a recursive checkpoint; recursive_fields={recursive_source_fields}"
        )
    source_config = AutoConfig.from_pretrained(source, local_files_only=True)
    actual_layers = int(getattr(source_config, "num_hidden_layers", -1))
    if actual_layers != raw_layers:
        raise ValueError(
            f"AutoConfig disagrees with source config.json: json={raw_layers} loaded={actual_layers}"
        )
    if getattr(source_config, "model_type", None) != "llama":
        raise ValueError("Adjacent-average conversion supports only Llama/SmolLM2 checkpoints")
    source_layer_coverage()
    torch.manual_seed(args.seed)
    source_model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, use_safetensors=True
    )
    source_layers = getattr(getattr(source_model, "model", None), "layers", ())
    if len(source_layers) != 30:
        raise ValueError(f"Source model layer count must be 30, got {len(source_layers)}")
    register_auto_class()
    target_config = build_target_config(source_config)
    target_model = RecursiveLlamaForCausalLM(target_config)
    target_model.to(dtype=next(source_model.parameters()).dtype)
    copy_non_layer_modules(source_model, target_model)
    exact_assignments: list[dict[str, int]] = []
    for target_index, source_index in enumerate(PREFIX_SOURCE_LAYERS_0BASED):
        copy_module_checked(
            source_layers[source_index], target_model.model.layers[target_index],
            f"model.layers[{target_index}]<-source.model.layers[{source_index}]",
        )
        exact_assignments.append({"target": target_index, "source": source_index})
    pair_audits: list[dict[str, Any]] = []
    for offset, pair in enumerate(MIDDLE_SOURCE_LAYER_PAIRS_0BASED):
        target_index = 5 + offset
        tensor_audit = average_modules_checked(
            source_layers[pair[0]], source_layers[pair[1]],
            target_model.model.layers[target_index],
            f"model.layers[{target_index}]<-mean(source.layers[{pair[0]}],source.layers[{pair[1]}])",
        )
        pair_audits.append(
            {"target": target_index, "source_pair": list(pair), **tensor_audit}
        )
    for offset, source_index in enumerate(SUFFIX_SOURCE_LAYERS_0BASED):
        target_index = 15 + offset
        copy_module_checked(
            source_layers[source_index], target_model.model.layers[target_index],
            f"model.layers[{target_index}]<-source.model.layers[{source_index}]",
        )
        exact_assignments.append({"target": target_index, "source": source_index})
    target_model.tie_weights()
    audit = _parameter_audit(target_model)
    if not audit["parameter_storage_unique"] or audit["physical_layer_count"] != 20:
        raise RuntimeError(f"Invalid target parameter audit: {audit}")
    target_model.config.recursive_parameter_audit = audit
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        target_model.save_pretrained(staging, safe_serialization=True)
        reloaded_model = AutoModelForCausalLM.from_pretrained(
            staging, local_files_only=True, use_safetensors=True
        )
        saved_artifact_verification = verify_saved_model(source_model, reloaded_model)
        del reloaded_model
        copied = copy_tokenizer_files(source, staging)
        weight_files = sorted(
            path for path in staging.iterdir()
            if path.is_file() and path.name.endswith((".safetensors", ".bin"))
        )
        if not weight_files:
            raise RuntimeError("Converted model did not produce a weight artifact")
        metadata = {
            "status": "ok",
            "conversion": "SmolLM2-5-10-5-adjacent-average",
            "architecture_contract": ARCHITECTURE_CONTRACT,
            "initialization_policy": INITIALIZATION_POLICY,
            "initialization_contract": INITIALIZATION_CONTRACT,
            "average_accumulator_dtype": AVERAGE_ACCUMULATOR_DTYPE,
            "source_checkpoint": str(source),
            "target_output": str(output),
            "source_config": source_config_summary(source_config),
            "source_config_json": raw,
            "target_config": target_config.to_dict(),
            "source_logical_layer_count": 30,
            "target_logical_layer_count": 30,
            "physical_layer_count": 20,
            "recursive_layer_count": 20,
            "prefix_layer_count": 5,
            "middle_recurrent_count": 10,
            "suffix_layer_count": 5,
            "recursive_loops": 2,
            "recursive_loops_scope": "middle_only",
            "logical_to_physical": list(LOGICAL_TO_PHYSICAL),
            "prefix_source_layers_0based": list(PREFIX_SOURCE_LAYERS_0BASED),
            "middle_source_layer_pairs_0based": [
                list(pair) for pair in MIDDLE_SOURCE_LAYER_PAIRS_0BASED
            ],
            "suffix_source_layers_0based": list(SUFFIX_SOURCE_LAYERS_0BASED),
            "source_layer_coverage_0based": list(SOURCE_LAYER_COVERAGE_0BASED),
            "source_layer_coverage_exactly_once": True,
            "exact_assignments": exact_assignments,
            "pair_average_audits": pair_audits,
            "saved_artifact_verification": saved_artifact_verification,
            "parameter_audit": audit,
            "copied_tokenizer_files": copied,
            "saved_weight_files": [
                {"name": path.name, "size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}
                for path in weight_files
            ],
            "seed": args.seed,
            "code_commit": git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "transformers_api_expected": "4.54.1",
            "conversion_time_utc": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "adjacent_average_conversion_metadata.json").write_text(
            json.dumps(json_safe(metadata), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
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
