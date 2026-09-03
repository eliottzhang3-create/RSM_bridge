#!/usr/bin/env python3
"""Atomically convert a local 30-layer Llama checkpoint to 5-10xpoisson-parcae.

The converter only writes a new, explicitly named output directory.  It does
not touch any historical 5-10xr-5 or fixed-depth artifacts.
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

FORBIDDEN_CHECKOUT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM")
LOCAL_CHECKOUT = SCRIPT_ROOT.parents[1]
DEFAULT_OUTPUT_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10xpoisson-parcae")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
TOKENIZER_NAMES = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model", "spiece.model", "vocab.json", "merges.txt", "added_tokens.json", "generation_config.json"}

SOURCE_LAYER_INDICES_0BASED = (0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29)
LOGICAL_TO_PHYSICAL = tuple(range(5)) + tuple(range(5, 15)) * 10 + tuple(range(15, 20))
MAPPING_POLICY = "explicit_5_10xpoisson_parcae_source_layers"
SAMPLING_POLICY = "truncated_poisson"
SAMPLER_VERSION = "truncated_poisson_lambda7_support4_10_v1"
SAMPLER_KEY = "sha256_cpu_torch_generator_base_seed_rank_optimizer_step_microbatch_v1"
POISSON_LAMBDA = 7.0
POISSON_SUPPORT = tuple(range(4, 11))
POISSON_NORMALIZATION_Z = sum(__import__("math").exp(-POISSON_LAMBDA) * POISSON_LAMBDA**k / __import__("math").factorial(k) for k in POISSON_SUPPORT)
POISSON_PROBABILITIES = tuple((__import__("math").exp(-POISSON_LAMBDA) * POISSON_LAMBDA**k / __import__("math").factorial(k)) / POISSON_NORMALIZATION_Z for k in POISSON_SUPPORT)
DEFAULT_SSM_DECAY = __import__("math").sqrt(1.0 / 5.0)
DEFAULT_TARGET_PRODUCT = -__import__("math").log(DEFAULT_SSM_DECAY)


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def reject_forbidden_output(path: Path) -> Path:
    candidate = _resolved(path)
    for forbidden in (_resolved(FORBIDDEN_CHECKOUT), _resolved(LOCAL_CHECKOUT)):
        try:
            candidate.relative_to(forbidden)
        except ValueError:
            continue
        raise ValueError(f"Refusing to write model parameters inside a Git checkout: output={candidate}")
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
    if isinstance(value, Path):
        return str(value)
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
    except Exception:
        return "<unavailable>"


def source_config_summary(config: Any) -> dict[str, Any]:
    keys = ("model_type", "architectures", "num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "vocab_size", "max_position_embeddings", "torch_dtype", "bos_token_id", "eos_token_id", "pad_token_id")
    return {key: json_safe(getattr(config, key, None)) for key in keys}


def copy_module_checked(source: Any, target: Any, name: str) -> None:
    source_state, target_state = source.state_dict(), target.state_dict()
    if set(source_state) != set(target_state):
        raise ValueError(f"{name} state keys differ")
    for key, target_tensor in target_state.items():
        source_tensor = source_state[key]
        if source_tensor.shape != target_tensor.shape or source_tensor.dtype != target_tensor.dtype:
            raise ValueError(f"{name}.{key} mismatch: source={tuple(source_tensor.shape)}/{source_tensor.dtype} target={tuple(target_tensor.shape)}/{target_tensor.dtype}")
    target.load_state_dict(source_state, strict=True)


def copy_non_layer_modules(source: Any, target: Any) -> None:
    for name in ("embed_tokens", "norm"):
        copy_module_checked(getattr(source.model, name), getattr(target.model, name), f"model.{name}")
    copy_module_checked(source.lm_head, target.lm_head, "lm_head")


def copy_tokenizer_files(source_dir: Path, target_dir: Path) -> list[str]:
    copied = []
    for path in source_dir.iterdir():
        if not path.is_file() or path.name == "config.json":
            continue
        is_tokenizer = path.name in TOKENIZER_NAMES or path.name.startswith("tokenizer") or path.name.endswith((".model", ".jinja"))
        if is_tokenizer and not path.name.endswith(WEIGHT_SUFFIXES):
            shutil.copy2(path, target_dir / path.name)
            copied.append(path.name)
    return sorted(set(copied))


def build_target_config(source_config: Any) -> Any:
    if int(getattr(source_config, "num_hidden_layers", -1)) != 30:
        raise ValueError("5-10xpoisson-parcae requires source num_hidden_layers=30")
    target = copy.deepcopy(source_config)
    target.num_hidden_layers = 110
    target.recursive_source_num_hidden_layers = 30
    target.recursive_source_layer_count = 30
    target.recursive_layer_count = 20
    target.recursive_loops = 10
    target.recursive_min_middle_loops = 4
    target.recursive_max_middle_loops = 10
    target.recursive_default_inference_middle_loops = 7
    target.recursive_parameter_gradient_tail_loops = 4
    target.recursive_backward_policy = "hidden_path_all_calls_parameter_gradients_final_four_aligned_calls_v1"
    target.recursive_training_loop_mode = "per_local_microbatch_per_sequence_truncated_poisson"
    target.recursive_sampling_policy = SAMPLING_POLICY
    target.recursive_sampler_version = SAMPLER_VERSION
    target.recursive_sampler_key = SAMPLER_KEY
    target.recursive_poisson_lambda = POISSON_LAMBDA
    target.recursive_poisson_support = list(POISSON_SUPPORT)
    target.recursive_poisson_normalization_z = POISSON_NORMALIZATION_Z
    target.recursive_poisson_Z = POISSON_NORMALIZATION_Z
    target.recursive_poisson_probabilities = list(POISSON_PROBABILITIES)
    target.recursive_prelude_norm = "LlamaRMSNorm"
    target.recursive_state_init = "like-init"
    target.recursive_state_init_std = float(getattr(source_config, "initializer_range", 0.02))
    target.recursive_embedding_scale = float(getattr(source_config, "embedding_scale", getattr(source_config, "embed_scale", 1.0)))
    target.recursive_injection_init = "parcae_exact_ssm_decay_sqrt_1_over_5_identity_B_no_weight_decay"
    target.recursive_injection_formula = "h*decay + dt*(PN(e) @ B.T)"
    target.recursive_injection_no_weight_decay = True
    target.recursive_B_init = "identity"
    target.recursive_learned_h0 = False
    target.recursive_ssm_decay = DEFAULT_SSM_DECAY
    target.recursive_target_product = DEFAULT_TARGET_PRODUCT
    target.recursive_initial_dt = DEFAULT_TARGET_PRODUCT
    target.recursive_initial_decay = DEFAULT_SSM_DECAY
    target.recursive_min_logical_layer_count = 50
    target.recursive_max_logical_layer_count = 110
    target.recursive_prefix_layer_count = 5
    target.recursive_middle_layer_count = 10
    target.recursive_suffix_layer_count = 5
    target.middle_recurrent_count = 10
    target.recursive_loops_scope = "middle_only"
    target.recursive_local_tmax = True
    target.recursive_noop_left_alignment = True
    target.logical_to_physical = list(LOGICAL_TO_PHYSICAL)
    target.recursive_logical_to_physical = list(LOGICAL_TO_PHYSICAL)
    target.logical_to_physical_schedule = list(LOGICAL_TO_PHYSICAL)
    target.recursive_logical_to_physical_schedule = list(LOGICAL_TO_PHYSICAL)
    target.recursive_source_layer_indices_0based = list(SOURCE_LAYER_INDICES_0BASED)
    target.recursive_source_layer_indices_1based = [i + 1 for i in SOURCE_LAYER_INDICES_0BASED]
    target.recursive_mapping_policy = MAPPING_POLICY
    target.architectures = ["RecursiveLlama5_10xpoisson_parcaeForCausalLM"]
    return target


def build_source_mapping(num_hidden_layers: int = 30) -> tuple[int, ...]:
    if int(num_hidden_layers) != 30:
        raise ValueError(f"source num_hidden_layers must be 30, got {num_hidden_layers}")
    return SOURCE_LAYER_INDICES_0BASED


def convert(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from recursive_model_5_10xpoisson_parcae import RecursiveLlama5_10xpoisson_parcaeForCausalLM, register_auto_class, parameter_audit

    source, output = _resolved(args.source_checkpoint), reject_forbidden_output(args.output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Source checkpoint directory does not exist: {source}")
    if not (source / "config.json").is_file():
        raise FileNotFoundError("source checkpoint is missing config.json")
    if output.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --allow-overwrite explicitly")
    raw = json.loads((source / "config.json").read_text(encoding="utf-8"))
    raw_layers = raw.get("num_hidden_layers")
    if isinstance(raw_layers, bool) or not isinstance(raw_layers, int) or raw_layers != 30:
        raise ValueError("source config.json num_hidden_layers must be exactly 30")
    source_config = AutoConfig.from_pretrained(source, local_files_only=True)
    actual_layers = int(getattr(source_config, "num_hidden_layers", -1))
    if actual_layers != raw_layers or actual_layers != 30:
        raise ValueError(f"AutoConfig/source config layer mismatch: json={raw_layers} loaded={actual_layers}")
    source_model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, use_safetensors=True)
    source_layers_module = getattr(getattr(source_model, "model", None), "layers", ())
    if len(source_layers_module) != 30:
        raise ValueError(f"Source model layer count must be 30, got {len(source_layers_module)}")
    torch.manual_seed(args.seed)
    register_auto_class()
    target_config = build_target_config(source_config)
    target_model = RecursiveLlama5_10xpoisson_parcaeForCausalLM(target_config)
    target_model.to(dtype=next(source_model.parameters()).dtype)
    copy_non_layer_modules(source_model, target_model)
    for target_index, source_index in enumerate(build_source_mapping(actual_layers)):
        copy_module_checked(source_layers_module[source_index], target_model.model.layers[target_index], f"model.layers[{target_index}]<-source.model.layers[{source_index}]")
    target_model.tie_weights()
    audit = parameter_audit(target_model)
    target_model.config.recursive_parameter_audit = audit
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    metadata = {
        "status": "ok", "conversion": MODEL_LABEL, "source_checkpoint": str(source), "target_output": str(output),
        "source_config": source_config_summary(source_config), "source_config_json": raw, "target_config": target_config.to_dict(),
        "source_logical_layer_count": 30, "target_logical_layer_count": 110, "physical_layer_count": 20,
        "prefix_layer_count": 5, "middle_recurrent_count": 10, "suffix_layer_count": 5,
        "min_middle_loops": 4, "max_middle_loops": 10, "default_inference_middle_loops": 7, "parameter_gradient_tail_loops": 4,
        "logical_depth_range": [50, 110], "logical_to_physical": list(LOGICAL_TO_PHYSICAL), "source_layer_indices_0based": list(SOURCE_LAYER_INDICES_0BASED),
        "mapping_policy": MAPPING_POLICY, "backward_policy": target_config.recursive_backward_policy,
        "sampling_policy": SAMPLING_POLICY, "sampler_version": SAMPLER_VERSION, "sampler_key": SAMPLER_KEY,
        "poisson_lambda": POISSON_LAMBDA, "poisson_support": list(POISSON_SUPPORT), "poisson_probabilities": list(POISSON_PROBABILITIES), "recursive_poisson_probabilities": list(POISSON_PROBABILITIES), "poisson_normalization_z": POISSON_NORMALIZATION_Z, "poisson_Z": POISSON_NORMALIZATION_Z, "Z": POISSON_NORMALIZATION_Z,
        "prelude_norm": "LlamaRMSNorm", "injection_formula": "u_t = Abar(h_t) + Bbar(PN(e))", "state_init": "like-init", "state_init_std": float(getattr(source_config, "initializer_range", 0.02)), "embedding_scale": float(getattr(source_config, "embedding_scale", getattr(source_config, "embed_scale", 1.0))), "injection_init": "parcae_exact_ssm_decay_sqrt_1_over_5_identity_B_no_weight_decay", "ssm_decay": DEFAULT_SSM_DECAY, "target_product": DEFAULT_TARGET_PRODUCT, "initial_dt": DEFAULT_TARGET_PRODUCT, "initial_decay": DEFAULT_SSM_DECAY, "B_init": "identity", "injection_no_weight_decay": True, "learned_h0": False, "recursive_injection_no_weight_decay": True, "recursive_B_init": "identity", "recursive_learned_h0": False,
        "parameter_audit": audit, "seed": args.seed, "copied_tokenizer_files": [], "code_commit": git_commit(),
        "python": sys.version, "platform": platform.platform(), "transformers": package_version("transformers"), "torch": package_version("torch"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        target_model.save_pretrained(staging, safe_serialization=True)
        metadata["copied_tokenizer_files"] = copy_tokenizer_files(source, staging)
        (staging / "conversion_metadata.json").write_text(json.dumps(json_safe(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "checkpoint_complete.json").write_text(json.dumps({"complete": True, "conversion": MODEL_LABEL, "metadata": "conversion_metadata.json"}, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            if not args.allow_overwrite:
                raise FileExistsError(output)
            backup = output.with_name(output.name + ".old")
            if backup.exists():
                shutil.rmtree(backup)
            output.replace(backup)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metadata


def main(argv: list[str] | None = None) -> int:
    try:
        print(json.dumps(convert(parse_args(argv)), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"[result] status=FAIL error={exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
