#!/usr/bin/env python3
"""Offline Stage 3 evaluation for the isolated 5-10xpoisson-Parcae model.

The benchmark protocol is owned by :mod:`evaluate_stage3`; this entry point
only reuses its snapshot, task-overlay, logging, and result helpers.  Model
registration and all architecture/preflight checks are local to the Parcae
implementation.  A Stage 4 checkpoint is accepted only when its config,
root weights, and nested tokenizer can be loaded locally in offline mode.
Each run writes ``lm_eval_results.json``, ``log_samples.json``,
``summary.json``, ``summary.csv``, ``audit_report.json``, and
``run_config.json`` under the external output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module_from_path(name: str, path: Path) -> Any:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_stage3 = _load_module_from_path(
    "rsmol_stage3_benchmark_helpers", SCRIPT_ROOT / "scripts" / "evaluate_stage3.py"
)


def _load_parcae() -> Any:
    """Load only the Parcae model module under its canonical namespace."""

    return _load_module_from_path(
        "code.RSmol.recursive_model_5_10xpoisson_parcae",
        SCRIPT_ROOT / "recursive_model_5_10xpoisson_parcae.py",
    )


DEFAULT_MODEL = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "stage4_5_10xpoisson_parcae_lr8e-4_mb2_ga64_20260903_172047/checkpoint-009244"
)
DEFAULT_BENCHMARK_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/eval_datasets"
)
DEFAULT_LOG_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/eval_logs/"
    "stage3_5_10xpoisson_parcae_009244"
)
DEFAULT_CACHE_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/eval_cache/"
    "stage3_5_10xpoisson_parcae_009244"
)
STAGE3_TASKS = _stage3.STAGE3_TASKS
EXPECTED_MMLU_SUBJECTS = _stage3.EXPECTED_MMLU_SUBJECTS
EXPECTED_LM_EVAL_VERSION = _stage3.EXPECTED_LM_EVAL_VERSION
EXPECTED_TRANSFORMERS_VERSION = _stage3.EXPECTED_TRANSFORMERS_VERSION
EXPECTED_DATASETS_VERSION = _stage3.EXPECTED_DATASETS_VERSION

MODEL_LABEL = "5_10xpoisson_parcae"
ARCHITECTURE_CONTRACT = "logical_50_110_physical_20_5_10xpoisson_parcae_tail4"
EXPECTED_ARCHITECTURES = ("RecursiveLlama5_10xpoisson_parcaeForCausalLM",)
PHYSICAL_LAYER_COUNT = 20
PREFIX_LAYER_COUNT = 5
MIDDLE_LAYER_COUNT = 10
SUFFIX_LAYER_COUNT = 5
MIN_MIDDLE_LOOPS = 4
MAX_MIDDLE_LOOPS = 10
DEFAULT_INFERENCE_MIDDLE_LOOPS = 7
MIN_LOGICAL_LAYER_COUNT = 50
MAX_LOGICAL_LAYER_COUNT = 110
PARAMETER_GRADIENT_TAIL_LOOPS = 4
SAMPLER_VERSION = "truncated_poisson_lambda7_support4_10_v1"
SAMPLER_KEY = "sha256_cpu_torch_generator_base_seed_rank_optimizer_step_microbatch_v1"
BACKWARD_POLICY = "hidden_path_all_calls_parameter_gradients_final_four_aligned_calls_v1"
DEFAULT_SSM_DECAY = math.sqrt(1.0 / 5.0)
DEFAULT_TARGET_PRODUCT = -math.log(DEFAULT_SSM_DECAY)
SOURCE_MAPPING_0BASED = (
    0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,
    25, 26, 27, 28, 29,
)
SOURCE_MAPPING_1BASED = tuple(item + 1 for item in SOURCE_MAPPING_0BASED)
BF16_INCREMENTAL_MAX_ABS = 1.0
BF16_INCREMENTAL_MIN_COSINE = 0.999
CACHE_ATOL = 1e-5
CACHE_RTOL = 1e-4
POISSON_LAMBDA = 7.0
POISSON_SUPPORT = tuple(range(4, 11))
POISSON_NORMALIZATION_Z = sum(math.exp(-POISSON_LAMBDA) * POISSON_LAMBDA**k / math.factorial(k) for k in POISSON_SUPPORT)
POISSON_PROBABILITIES = tuple((math.exp(-POISSON_LAMBDA) * POISSON_LAMBDA**k / math.factorial(k)) / POISSON_NORMALIZATION_Z for k in POISSON_SUPPORT)
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
RECURSIVE_LOOPS = MAX_MIDDLE_LOOPS
LOOPS_SCOPE = "middle_only"
LOGICAL_TO_PHYSICAL = tuple(range(5)) + tuple(range(5, 15)) * MAX_MIDDLE_LOOPS + tuple(range(15, 20))
LOGICAL_TO_PHYSICAL_SCHEDULE = LOGICAL_TO_PHYSICAL
RECURSIVE_LOGICAL_TO_PHYSICAL = LOGICAL_TO_PHYSICAL
RECURSIVE_LOGICAL_TO_PHYSICAL_SCHEDULE = LOGICAL_TO_PHYSICAL

EvaluationConfig = _stage3.EvaluationConfig
json_safe = _stage3.json_safe
write_json = _stage3.write_json
ensure_external_path = _stage3.ensure_external_path
ensure_external_output = _stage3.ensure_external_output
ensure_log_root = _stage3.ensure_log_root
set_offline_environment = _stage3.set_offline_environment
inspect_pinned_versions = _stage3.inspect_pinned_versions
validate_benchmark_layout = _stage3.validate_benchmark_layout
prepare_local_task_overlays = _stage3.prepare_local_task_overlays
discover_mmlu_subjects = _stage3.discover_mmlu_subjects
_task_log_path = _stage3._task_log_path
_runtime_log_path = _stage3._runtime_log_path
_append_text = _stage3._append_text
_flatten_result_rows = _stage3._flatten_result_rows
_result_sample_counts = _stage3._result_sample_counts
_write_summary = _stage3._write_summary
_gpu_info = _stage3._gpu_info
package_version = _stage3.package_version
git_commit = _stage3.git_commit
utc_now = _stage3.utc_now


def _read_json(path: Path) -> dict[str, Any]:
    return _stage3._read_json(path)


def _model_file_manifest(model_dir: Path) -> list[str]:
    return _stage3._model_file_manifest(model_dir)


def _tokenizer_file_manifest(model_dir: Path) -> list[str]:
    return _stage3._tokenizer_file_manifest(model_dir)


def _exact_tuple(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None


def _expected_schedule(loops: int) -> tuple[int, ...]:
    loops = int(loops)
    if not MIN_MIDDLE_LOOPS <= loops <= MAX_MIDDLE_LOOPS:
        raise ValueError(f"middle loop count must be in [4, 10], got {loops}")
    return tuple(range(5)) + tuple(range(5, 15)) * loops + tuple(range(15, 20))


def _strict_config_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate serialized Parcae metadata before importing model weights."""

    required: dict[str, Any] = {
        "model_type": "llama",
        "num_hidden_layers": MAX_LOGICAL_LAYER_COUNT,
        "recursive_source_num_hidden_layers": 30,
        "recursive_source_layer_count": 30,
        "recursive_layer_count": PHYSICAL_LAYER_COUNT,
        "recursive_loops": MAX_MIDDLE_LOOPS,
        "recursive_loops_scope": LOOPS_SCOPE,
        "recursive_min_middle_loops": MIN_MIDDLE_LOOPS,
        "recursive_max_middle_loops": MAX_MIDDLE_LOOPS,
        "recursive_default_inference_middle_loops": DEFAULT_INFERENCE_MIDDLE_LOOPS,
        "recursive_parameter_gradient_tail_loops": PARAMETER_GRADIENT_TAIL_LOOPS,
        "recursive_prefix_layer_count": PREFIX_LAYER_COUNT,
        "recursive_middle_layer_count": MIDDLE_LAYER_COUNT,
        "recursive_suffix_layer_count": SUFFIX_LAYER_COUNT,
        "recursive_min_logical_layer_count": MIN_LOGICAL_LAYER_COUNT,
        "recursive_max_logical_layer_count": MAX_LOGICAL_LAYER_COUNT,
        "recursive_mapping_policy": "explicit_5_10xpoisson_parcae_source_layers",
        "recursive_sampling_policy": "truncated_poisson",
        "recursive_prelude_norm": "LlamaRMSNorm",
        "recursive_state_init": "like-init",
        "recursive_learned_h0": False,
        "recursive_training_loop_mode": "per_local_microbatch_per_sequence_truncated_poisson",
        "recursive_local_tmax": True,
        "recursive_noop_left_alignment": True,
        "recursive_injection_no_weight_decay": True,
        "recursive_B_init": "identity",
        "recursive_injection_formula": "h*decay + dt*(PN(e) @ B.T)",
    }
    for key, expected in required.items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(f"strict Parcae contract mismatch/missing {key}: got={actual!r} expected={expected!r}")
    architectures = tuple(str(item) for item in config.get("architectures", ()))
    if architectures != EXPECTED_ARCHITECTURES:
        raise ValueError(f"strict Parcae architectures mismatch: {architectures!r}")
    support = tuple(int(item) for item in config.get("recursive_poisson_support", ()))
    if support != tuple(range(4, 11)):
        raise ValueError(f"strict Parcae Poisson support must be 4..10, got {support!r}")
    probabilities = tuple(float(item) for item in config.get("recursive_poisson_probabilities", ()))
    if len(probabilities) != len(POISSON_PROBABILITIES) or any(
        abs(a - b) > 1e-14 for a, b in zip(probabilities, POISSON_PROBABILITIES)
    ):
        raise ValueError("strict Parcae truncated-Poisson probabilities mismatch")
    if abs(float(config.get("recursive_poisson_lambda", -1.0)) - 7.0) > 1e-14:
        raise ValueError("strict Parcae Poisson lambda must be 7")
    for key in ("recursive_poisson_normalization_z", "recursive_poisson_Z"):
        if abs(float(config.get(key, -1.0)) - POISSON_NORMALIZATION_Z) > 1e-14:
            raise ValueError(f"strict Parcae {key} mismatch")
    source = _exact_tuple(config.get("recursive_source_layer_indices_0based"))
    if source != SOURCE_MAPPING_0BASED:
        raise ValueError(f"strict Parcae source mapping mismatch: {source!r}")
    max_schedule = LOGICAL_TO_PHYSICAL
    for key in ("logical_to_physical", "recursive_logical_to_physical", "logical_to_physical_schedule", "recursive_logical_to_physical_schedule"):
        if _exact_tuple(config.get(key)) != max_schedule:
            raise ValueError(f"strict Parcae logical schedule mismatch/missing {key}")
    for key, expected in (
        ("recursive_sampler_version", SAMPLER_VERSION),
        ("recursive_sampler_key", SAMPLER_KEY),
        ("recursive_backward_policy", BACKWARD_POLICY),
        ("recursive_injection_init", "parcae_exact_ssm_decay_sqrt_1_over_5_identity_B_no_weight_decay"),
    ):
        if config.get(key) != expected:
            raise ValueError(f"strict Parcae metadata mismatch/missing {key}: {config.get(key)!r}")
    for key, expected in (
        ("recursive_prefix_layer_count", PREFIX_LAYER_COUNT),
        ("recursive_middle_layer_count", MIDDLE_LAYER_COUNT),
        ("recursive_suffix_layer_count", SUFFIX_LAYER_COUNT),
    ):
        if int(config.get(key, -1)) != expected:
            raise ValueError(f"strict Parcae layer partition mismatch: {key}")
    if float(config.get("recursive_state_init_std", 0.0)) <= 0:
        raise ValueError("strict Parcae contract requires positive recursive_state_init_std")
    if float(config.get("recursive_embedding_scale", 0.0)) <= 0:
        raise ValueError("strict Parcae contract requires positive recursive_embedding_scale")
    for key, expected in (("recursive_ssm_decay", DEFAULT_SSM_DECAY), ("recursive_initial_decay", DEFAULT_SSM_DECAY), ("recursive_target_product", DEFAULT_TARGET_PRODUCT), ("recursive_initial_dt", DEFAULT_TARGET_PRODUCT)):
        if abs(float(config.get(key, -1.0)) - expected) > 1e-14:
            raise ValueError(f"strict Parcae injection initialization mismatch: {key}")
    return {
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "architectures": list(architectures),
        "logical_depth_range": [MIN_LOGICAL_LAYER_COUNT, MAX_LOGICAL_LAYER_COUNT],
        "physical_layer_count": PHYSICAL_LAYER_COUNT,
        "scalar_depth_support": list(range(4, 11)),
        "default_inference_T": DEFAULT_INFERENCE_MIDDLE_LOOPS,
        "parameter_gradient_tail_loops": PARAMETER_GRADIENT_TAIL_LOOPS,
        "prelude_norm": config["recursive_prelude_norm"],
        "state_init": config["recursive_state_init"],
        "injection_formula": config["recursive_injection_formula"],
        "poisson_support": list(support),
        "poisson_normalization_z": float(config["recursive_poisson_normalization_z"]),
    }


def _tokenizer_path(model_dir: Path) -> Path:
    nested = model_dir / "tokenizer"
    return nested if nested.is_dir() else model_dir


def inspect_model_artifacts_5_10xpoisson_parcae(path: Path) -> dict[str, Any]:
    """Fail closed on an incomplete Stage 4 checkpoint."""

    model_dir = ensure_external_path(path, label="5-10xpoisson-parcae checkpoint")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Parcae checkpoint is not an existing directory: {model_dir}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Parcae checkpoint is missing config.json: {model_dir}")
    config = _read_json(config_path)
    contract = _strict_config_contract(config)
    model_files = _model_file_manifest(model_dir)
    if not model_files:
        raise FileNotFoundError(f"Parcae checkpoint has no root model weight artifact: {model_dir}")
    tokenizer_dir = _tokenizer_path(model_dir)
    tokenizer_files = _tokenizer_file_manifest(tokenizer_dir)
    if not tokenizer_files or not ({"tokenizer.json", "tokenizer.model", "vocab.json", "spiece.model"} & set(tokenizer_files)):
        raise FileNotFoundError(f"Parcae checkpoint tokenizer has no local vocabulary payload: {tokenizer_dir}")
    if "tokenizer_config.json" not in tokenizer_files:
        raise FileNotFoundError(f"Parcae checkpoint tokenizer is missing tokenizer_config.json: {tokenizer_dir}")
    tokenizer_config = _read_json(tokenizer_dir / "tokenizer_config.json")
    tokenizer_vocab_size: int | None = None
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if tokenizer_json.is_file():
        try:
            tokenizer_payload = _read_json(tokenizer_json)
            vocab = tokenizer_payload.get("model", {}).get("vocab", {})
            if isinstance(vocab, dict):
                tokenizer_vocab_size = len(vocab)
        except ValueError:
            tokenizer_vocab_size = None
    config_vocab_size = config.get("vocab_size")
    vocab_compatible = tokenizer_vocab_size is None or config_vocab_size is None or int(config_vocab_size) == tokenizer_vocab_size
    if not vocab_compatible:
        raise ValueError(f"Parcae tokenizer/model vocab mismatch: config={config_vocab_size} tokenizer={tokenizer_vocab_size}")
    return {
        "label": MODEL_LABEL,
        "model_label": MODEL_LABEL,
        "path": str(model_dir),
        "config": config,
        "tokenizer_path": str(tokenizer_dir),
        "tokenizer_config": tokenizer_config,
        "model_files": model_files,
        "tokenizer_files": tokenizer_files,
        "config_vocab_size": config_vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "vocab_compatible": vocab_compatible,
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "recursive_audit": {"is_recursive": True, "status": "PASS", **contract, "source_mapping_0based": list(SOURCE_MAPPING_0BASED), "logical_to_physical": list(LOGICAL_TO_PHYSICAL), "contract_checks": {"architecture_contract": True, "config_contract": True, "physical_layer_count": True, "logical_depth_range": True, "poisson_support": True, "default_inference_T": True, "prelude_norm": True, "state_init": True, "injection_formula": True, "parameter_gradient_tail_loops": True}},
    }


def _finite(name: str, tensor: Any) -> None:
    import torch

    if not torch.isfinite(tensor.float()).all().item():
        raise RuntimeError(f"{name} contains non-finite values")


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device


def _prompt(model: Any, *, length: int = 3) -> Any:
    import torch

    device = _model_device(model)
    vocab_size = int(getattr(model.config, "vocab_size", 128))
    return torch.arange(length, device=device, dtype=torch.long).view(1, -1) % vocab_size


def _physical_layers(recursive_model: Any) -> list[Any]:
    layers = list(getattr(recursive_model, "layers", ()))
    if len(layers) != PHYSICAL_LAYER_COUNT or len({id(layer) for layer in layers}) != PHYSICAL_LAYER_COUNT:
        raise RuntimeError(f"Parcae physical layer audit failed: count={len(layers)} unique={len({id(layer) for layer in layers})}")
    return layers


@contextlib.contextmanager
def _physical_trace(recursive_model: Any):
    trace: list[int] = []
    handles = [layer.register_forward_hook(lambda _m, _i, _o, index=index: trace.append(index)) for index, layer in enumerate(_physical_layers(recursive_model))]
    try:
        yield trace
    finally:
        for handle in handles:
            handle.remove()


def _assert_logits(outputs: Any, model: Any, *, batch: int, sequence: int) -> None:
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Parcae model output has no logits")
    expected = (batch, sequence, int(model.config.vocab_size))
    if tuple(logits.shape) != expected:
        raise RuntimeError(f"Parcae logits shape mismatch: got={tuple(logits.shape)} expected={expected}")
    _finite("Parcae logits", logits)


def _forward_with_trace(model: Any, input_ids: Any, *, use_cache: bool, middle_loop_count: int | None = None, past_key_values: Any | None = None) -> tuple[Any, list[int], dict[str, Any]]:
    recursive_model = getattr(model, "model", model)
    kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": bool(use_cache)}
    if middle_loop_count is not None:
        kwargs["middle_loop_count"] = int(middle_loop_count)
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    with _physical_trace(recursive_model) as trace:
        outputs = model(**kwargs)
    _assert_logits(outputs, model, batch=input_ids.shape[0], sequence=input_ids.shape[1])
    return outputs, trace, dict(getattr(recursive_model, "_last_forward_audit", {}))


def _cache_storage(cache: Any, slot: int) -> tuple[Any, Any]:
    layers = getattr(cache, "layers", None)
    if layers is not None and int(slot) < len(layers):
        layer = layers[int(slot)]
        return getattr(layer, "keys", getattr(layer, "key_cache", None)), getattr(layer, "values", getattr(layer, "value_cache", None))
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None and int(slot) < len(key_cache):
        return key_cache[int(slot)], value_cache[int(slot)]
    raise RuntimeError(f"cache has no logical slot {slot}")


def _cache_slot_audit(cache: Any, *, logical_depth: int, expected_length: int) -> dict[str, Any]:
    import torch

    if cache is None:
        raise RuntimeError("use_cache=True returned no cache")
    try:
        capacity = len(cache)
    except TypeError:
        capacity = logical_depth
    if capacity < logical_depth:
        raise RuntimeError(f"cache capacity {capacity} < logical depth {logical_depth}")
    slots = []
    for slot in range(logical_depth):
        try:
            length = int(cache.get_seq_length(slot))
        except TypeError:
            length = int(cache.get_seq_length(layer_idx=slot))
        keys, values = _cache_storage(cache, slot)
        if keys is None or values is None or not hasattr(keys, "shape") or not hasattr(values, "shape"):
            raise RuntimeError(f"logical cache slot {slot} has invalid K/V")
        if not torch.isfinite(keys.float()).all().item() or not torch.isfinite(values.float()).all().item():
            raise RuntimeError(f"logical cache slot {slot} is non-finite")
        if int(keys.shape[-2]) != length or int(values.shape[-2]) != length or length != expected_length:
            raise RuntimeError(f"logical cache slot {slot} length/shape mismatch")
        slots.append({"slot": slot, "length": length, "key_shape": list(keys.shape), "value_shape": list(values.shape), "finite": True})
    return {"logical_depth": logical_depth, "cache_capacity": capacity, "slots": slots, "all_lengths_expected": True}


@contextlib.contextmanager
def _seed_context(device: Any, seed: int):
    import torch

    devices = [device.index if getattr(device, "type", None) == "cuda" and device.index is not None else 0] if getattr(device, "type", None) == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if getattr(device, "type", None) == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        yield


def validate_scalar_traces(model: Any) -> dict[str, Any]:
    import torch

    model.eval()
    input_ids = _prompt(model)
    audits: dict[str, Any] = {}
    with torch.inference_mode():
        for loops in range(MIN_MIDDLE_LOOPS, MAX_MIDDLE_LOOPS + 1):
            outputs, trace, audit = _forward_with_trace(model, input_ids, use_cache=False, middle_loop_count=loops)
            expected = list(_expected_schedule(loops))
            if trace != expected or audit.get("local_tmax") != loops or audit.get("cache_enabled"):
                raise RuntimeError(f"Parcae scalar T={loops} trace/audit mismatch: trace={trace} audit={audit}")
            if int(audit.get("logical_layer_count", -1)) != len(expected):
                raise RuntimeError(f"Parcae scalar T={loops} logical depth mismatch")
            audits[str(loops)] = {"middle_loop_count": loops, "logical_depth": len(expected), "trace": trace, "finite": True, "logits_shape": list(outputs.logits.shape)}
    return {"T_values": list(range(4, 11)), "logical_depth_range": [50, 110], "audits": audits}


def validate_default_T7(model: Any) -> dict[str, Any]:
    import torch

    model.eval()
    input_ids = _prompt(model)
    with torch.inference_mode():
        outputs, trace, audit = _forward_with_trace(model, input_ids, use_cache=False)
    expected = list(_expected_schedule(DEFAULT_INFERENCE_MIDDLE_LOOPS))
    if trace != expected or audit.get("local_tmax") != DEFAULT_INFERENCE_MIDDLE_LOOPS:
        raise RuntimeError(f"Parcae default T=7 mismatch: trace={trace} audit={audit}")
    return {"default_inference_T": 7, "logical_depth": len(expected), "trace": trace, "finite": True, "logits_shape": list(outputs.logits.shape), "audit": audit}


def validate_semantics_and_gradient_metadata(model: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    recursive_model = getattr(model, "model", model)
    input_ids = torch.arange(4 * 8, device=_model_device(model), dtype=torch.long).view(4, 8) % int(model.config.vocab_size)
    model.train()
    model.zero_grad(set_to_none=True)
    recursive_model._collect_middle_gradient_audit = True
    try:
        counts = torch.tensor([4, 7, 10, 6], device=input_ids.device, dtype=torch.long)
        outputs = model(input_ids=input_ids, middle_loop_counts=counts, use_cache=False)
        state = getattr(recursive_model, "_last_state_init", None)
        pn_e = getattr(recursive_model, "_last_pn_e", None)
        audit = dict(getattr(recursive_model, "_last_forward_audit", {}))
        if state is None or pn_e is None or tuple(state.shape) != tuple(pn_e.shape) or not torch.any(state != 0).item():
            raise RuntimeError("Parcae h0 must be nonzero like-init state with e shape")
        if audit.get("state_init") != "like-init" or audit.get("prelude_norm") != "LlamaRMSNorm" or audit.get("prelude_norm_calls") != 1 or not audit.get("pn_e_reused"):
            raise RuntimeError(f"Parcae PN/h0 metadata mismatch: {audit}")
        if any(name.endswith("h0") or ".h0" in name for name, _ in recursive_model.named_parameters()):
            raise RuntimeError("learned h0 parameter is forbidden")
        trace = list(getattr(recursive_model, "_last_middle_gradient_audit", ()))
        if len(trace) != 10 or any(bool(item["parameter_grad_enabled"]) != (item["aligned_step"] >= 6) for item in trace):
            raise RuntimeError("Parcae tail-4 gradient metadata mismatch")
        if any(bool(item["inactive"].any()) for item in trace[-4:]):
            raise RuntimeError("final four Parcae aligned calls must be active")
        injection = recursive_model.recurrent.injection
        h = torch.randn(2, 3, injection.A_log.numel(), device=input_ids.device, dtype=next(injection.parameters()).dtype)
        e = torch.randn_like(h)
        actual = injection(h, e)
        dt = F.softplus(injection.dt_bias)
        decay = torch.exp(-dt * torch.exp(injection.A_log))
        expected = h * decay + dt * torch.matmul(e, injection.B.transpose(-1, -2))
        if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
            raise RuntimeError("Parcae A/B additive injection formula mismatch")
        if not torch.allclose(injection.B.detach(), torch.eye(injection.B.shape[0], device=injection.B.device, dtype=injection.B.dtype)):
            raise RuntimeError("Parcae B is not identity initialized")
        recurrent_parameters = [(name, parameter) for name, parameter in recursive_model.named_parameters() if "recurrent.injection" in name or "recurrent.middle" in name]
        loss = outputs.logits.float().square().mean()
        early_hidden_norms = []
        early_parameter_edges = []
        for item in trace[:-4]:
            hidden_input = item.get("input")
            gradient = torch.autograd.grad(loss, hidden_input, retain_graph=True, allow_unused=True)[0] if hidden_input is not None else None
            if gradient is None or not torch.isfinite(gradient).all().item() or not torch.any(gradient != 0).item():
                raise RuntimeError("early hidden-input gradient path was severed")
            early_hidden_norms.append(float(gradient.detach().norm().item()))
            param_grads = torch.autograd.grad(item["output"].float().sum(), [parameter for _, parameter in recurrent_parameters], retain_graph=True, allow_unused=True)
            early_parameter_edges.append(any(g is not None and torch.any(g != 0).item() for g in param_grads))
        if any(early_parameter_edges):
            raise RuntimeError("early recurrent calls established parameter gradient edges")
        loss.backward()
        recurrent_gradient_audit = {}
        for name, parameter in recurrent_parameters:
            gradient = parameter.grad
            if gradient is None or not torch.isfinite(gradient).all().item() or not torch.any(gradient != 0).item():
                raise RuntimeError(f"tail recurrent gradient invalid: {name}")
            recurrent_gradient_audit[name] = {"shape": list(gradient.shape), "norm": float(gradient.norm().item()), "finite": True, "nonzero": True}
        def _layer_gradients(prefix: str) -> list[str]:
            names: list[str] = []
            for name, parameter in recursive_model.named_parameters():
                if not name.startswith(prefix) or not parameter.requires_grad:
                    continue
                gradient = parameter.grad
                if gradient is None or not torch.isfinite(gradient).all().item() or not torch.any(gradient != 0).item():
                    raise RuntimeError(f"Parcae {prefix} gradient invalid: {name}")
                names.append(name)
            if not names:
                raise RuntimeError(f"Parcae {prefix} has no trainable parameters with gradients")
            return names
        prefix_layers_with_grad = _layer_gradients("prefix_layers")
        middle_layers_with_grad = _layer_gradients("recurrent.middle.layers")
        suffix_layers_with_grad = _layer_gradients("suffix_layers")
        return {"state_init": "like-init", "state_shape": list(state.shape), "state_nonzero": True, "prelude_norm": "LlamaRMSNorm", "pn_single_compute_reused": True, "injection_formula_match": True, "B_identity": True, "parameter_gradient_tail_loops": 4, "exact_parameter_gradient_tail": 4, "early_hidden_gradient_norms": early_hidden_norms, "early_parameter_gradient_edges_absent": True, "last_four_injection_middle_parameter_grads": True, "injection_gradient_audit": recurrent_gradient_audit, "prefix_layers_with_grad": prefix_layers_with_grad, "middle_layers_with_grad": middle_layers_with_grad, "suffix_layers_with_grad": suffix_layers_with_grad, "prefix_suffix_gradients_finite_nonzero": True, "forward_audit": audit}
    finally:
        recursive_model._collect_middle_gradient_audit = False
        model.eval()


def validate_cache_incremental(model: Any) -> dict[str, Any]:
    import torch

    model.eval()
    device = _model_device(model)
    input_ids = _prompt(model)
    loops = DEFAULT_INFERENCE_MIDDLE_LOOPS
    logical_depth = len(_expected_schedule(loops))
    seed = 1729
    with _seed_context(device, seed):
        with torch.inference_mode():
            no_cache, no_trace, _ = _forward_with_trace(model, input_ids, use_cache=False, middle_loop_count=loops)
            no_cache_state = getattr(getattr(model, "model", model), "_last_state_init", None)
            no_cache_state = None if no_cache_state is None else no_cache_state.detach().clone()
    with _seed_context(device, seed):
        with torch.inference_mode():
            cached, cache_trace, cache_audit = _forward_with_trace(model, input_ids, use_cache=True, middle_loop_count=loops)
            cache_state = getattr(getattr(model, "model", model), "_last_state_init", None)
            cache_state = None if cache_state is None else cache_state.detach().clone()
    if no_cache_state is None or cache_state is None or not torch.equal(no_cache_state, cache_state):
        raise RuntimeError("Parcae cache/no-cache comparison did not use identical fresh like-init h0")
    if no_trace != list(_expected_schedule(loops)) or cache_trace != list(_expected_schedule(loops)):
        raise RuntimeError("Parcae cache/no-cache physical trace mismatch")
    diff = (no_cache.logits.float() - cached.logits.float()).abs()
    if not torch.allclose(no_cache.logits.float(), cached.logits.float(), atol=CACHE_ATOL, rtol=CACHE_RTOL):
        raise RuntimeError(f"Parcae cache/no-cache prefill mismatch: max_diff={float(diff.max().item())}")
    cache = cached.past_key_values
    initial = _cache_slot_audit(cache, logical_depth=logical_depth, expected_length=input_ids.shape[1])
    with torch.inference_mode():
        incremental, increment_trace, increment_audit = _forward_with_trace(model, input_ids[:, -1:], use_cache=True, middle_loop_count=loops, past_key_values=cache)
    _assert_logits(incremental, model, batch=1, sequence=1)
    increment = _cache_slot_audit(cache, logical_depth=logical_depth, expected_length=input_ids.shape[1] + 1)
    if increment_trace != list(_expected_schedule(loops)):
        raise RuntimeError("Parcae incremental physical trace mismatch")
    rejected = False
    try:
        with torch.inference_mode():
            model(input_ids=input_ids[:, -1:], use_cache=True, middle_loop_count=4, past_key_values=cache)
    except (ValueError, RuntimeError, TypeError):
        rejected = True
    if not rejected:
        raise RuntimeError("Parcae cache reuse with different T was not rejected")
    return {"T": loops, "logical_depth": logical_depth, "cache_no_cache_logits_max_abs_diff": float(diff.max().item()), "cache_no_cache_logits_allclose": True, "cache_no_cache_h0_equal": True, "prefill": initial, "incremental": increment, "increment_trace": increment_trace, "increment_logits_finite": True, "cache_T_mismatch_rejected": True, "cache_audit": cache_audit, "increment_audit": increment_audit, "fresh_h0_seed": seed}


def _generation_audit_one(model: Any, input_ids: Any, loops: int | None) -> dict[str, Any]:
    import torch

    recursive_model = getattr(model, "model", model)
    physical_trace: list[int] = []
    resolved: list[int] = []
    explicit: list[int | None] = []
    finite_logits: list[bool] = []
    handles = [layer.register_forward_hook(lambda _m, _i, _o, index=index: physical_trace.append(index)) for index, layer in enumerate(_physical_layers(recursive_model))]
    def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        del args
        value = kwargs.get("middle_loop_count")
        explicit.append(None if value is None else int(value))
    def recursive_post(module: Any, args: tuple[Any, ...], output: Any):
        del args, output
        resolved.append(int(module._last_forward_audit.get("local_tmax", -1)))
    def outer_post(_module: Any, args: tuple[Any, ...], output: Any):
        del args
        logits = getattr(output, "logits", None)
        finite_logits.append(logits is not None and bool(torch.isfinite(logits.float()).all()))
    try:
        handles.append(recursive_model.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(recursive_model.register_forward_hook(recursive_post))
        handles.append(model.register_forward_hook(outer_post))
        pad_id = getattr(getattr(model, "generation_config", None), "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(model.config, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(model.config, "eos_token_id", None)
        if pad_id is None:
            raise RuntimeError("generation requires pad_token_id or eos_token_id")
        kwargs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids, dtype=torch.long), "pad_token_id": int(pad_id), "eos_token_id": getattr(model.config, "eos_token_id", None), "max_new_tokens": 2, "do_sample": False, "use_cache": True}
        if loops is not None:
            kwargs["middle_loop_count"] = int(loops)
        with torch.inference_mode():
            generated = model.generate(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    requested = DEFAULT_INFERENCE_MIDDLE_LOOPS if loops is None else int(loops)
    expected = list(_expected_schedule(requested)) * len(resolved)
    if generated.ndim != 2 or generated.shape[1] < input_ids.shape[1] + 1 or not resolved or any(item != requested for item in resolved):
        raise RuntimeError(f"Parcae generation depth/shape mismatch: shape={tuple(generated.shape)} resolved={resolved}")
    if physical_trace != expected or not all(finite_logits):
        raise RuntimeError("Parcae generation trace/logits audit failed")
    if loops is not None and any(value is not None and value != loops for value in explicit):
        raise RuntimeError(f"Parcae generation did not propagate T={loops}: {explicit}")
    return {"requested_T": requested, "resolved_depths": resolved, "explicit_argument_trace": explicit, "physical_trace": physical_trace, "generated_shape": list(generated.shape), "finite": True, "all_calls_use_requested_T": True}


def validate_generation(model: Any) -> dict[str, Any]:
    model.eval()
    input_ids = _prompt(model, length=2)
    return {"explicit_T_values": list(range(4, 11)), "explicit": {str(loops): _generation_audit_one(model, input_ids, loops) for loops in range(4, 11)}, "default_T7": _generation_audit_one(model, input_ids, None), "generate_called": True}


def validate_save_reload(model: Any, tokenizer: Any, model_path: Path, *, output_dir: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parcae = _load_parcae()
    parcae.register_auto_class()
    with tempfile.TemporaryDirectory(prefix="parcae-save-reload-", dir=str(output_dir)) as temporary:
        saved = Path(temporary)
        model.save_pretrained(saved)
        tokenizer.save_pretrained(saved / "tokenizer")
        reloaded = AutoModelForCausalLM.from_pretrained(saved, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        reloaded.to(_model_device(model))
        if not isinstance(reloaded, (parcae.RecursiveLlama5_10xpoisson_parcaeForCausalLM, parcae.RecursiveLlamaForCausalLM)):
            raise RuntimeError(f"Parcae save/reload returned wrong class: {type(reloaded)!r}")
        reloaded.eval()
        reloaded_tokenizer = AutoTokenizer.from_pretrained(saved / "tokenizer", local_files_only=True, use_fast=True)
        del reloaded_tokenizer
        input_ids = _prompt(reloaded)
        with torch.inference_mode():
            outputs, trace, audit = _forward_with_trace(reloaded, input_ids, use_cache=False)
        if trace != list(_expected_schedule(7)) or audit.get("local_tmax") != 7:
            raise RuntimeError("Parcae save/reload default T=7 trace mismatch")
        return {"status": "PASS", "class": f"{type(reloaded).__module__}.{type(reloaded).__name__}", "default_T7": {"logical_depth": len(trace), "trace": trace, "finite": True, "logits_shape": list(outputs.logits.shape)}, "local_files_only": True}


def recursive_runtime_audit_5_10xpoisson_parcae(model: Any, *, tokenizer: Any, model_path: Path, output_dir: Path) -> dict[str, Any]:
    import torch

    parcae = _load_parcae()
    expected_classes = tuple(item for item in (getattr(parcae, "RecursiveLlama5_10xpoisson_parcaeForCausalLM", None), getattr(parcae, "RecursiveLlamaForCausalLM", None)) if item is not None)
    if not expected_classes or not isinstance(model, expected_classes):
        raise TypeError(f"AutoModel resolved the wrong Parcae class: expected={expected_classes!r} got={type(model)!r}")
    config_contract = _strict_config_contract(model.config.to_dict() if hasattr(model.config, "to_dict") else vars(model.config))
    recursive_model = getattr(model, "model", model)
    layers = _physical_layers(recursive_model)
    if len(getattr(recursive_model, "prefix_layers", ())) != 5 or len(getattr(recursive_model.recurrent.middle, "layers", ())) != 10 or len(getattr(recursive_model, "suffix_layers", ())) != 5:
        raise RuntimeError("Parcae physical prefix/middle/suffix partition mismatch")
    parameter_audit = parcae.parameter_audit(model)
    if parameter_audit.get("has_learned_h0"):
        raise RuntimeError("Parcae parameter audit found learned h0")
    scalar = validate_scalar_traces(model)
    default = validate_default_T7(model)
    semantics = validate_semantics_and_gradient_metadata(model)
    cache = validate_cache_incremental(model)
    generation = validate_generation(model)
    reload_report = validate_save_reload(model, tokenizer, model_path, output_dir=output_dir)
    return {"status": "PASS", "model_class": type(model).__name__, "architecture_contract": ARCHITECTURE_CONTRACT, "logical_depth_range": [50, 110], "physical_layer_count": len(layers), "prefix_layer_count": 5, "middle_layer_count": 10, "suffix_layer_count": 5, "parameter_audit": parameter_audit, "config_contract": config_contract, "scalar_inference": scalar, "default_T7": default, "semantics": semantics, "cache_incremental": cache, "generation": generation, "save_reload": reload_report, "finite_logits": True}


def load_and_audit_model_5_10xpoisson_parcae(config: EvaluationConfig, *, output_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parcae = _load_parcae()
    parcae.register_auto_class()
    model = AutoModelForCausalLM.from_pretrained(config.model_path, local_files_only=True, torch_dtype=getattr(torch, config.dtype), low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_path(config.model_path), local_files_only=True, use_fast=True)
    model.to(torch.device(config.device))
    audit = recursive_runtime_audit_5_10xpoisson_parcae(model, tokenizer=tokenizer, model_path=config.model_path, output_dir=output_dir)
    return model, tokenizer, audit


def _run_single_task(config: EvaluationConfig, task: str, task_dir: Path, overlay_dir: Path, stderr_log_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_cache = (config.cache_dir or DEFAULT_CACHE_ROOT) / task
    set_offline_environment(task_cache)
    model_args = ",".join((f"pretrained={config.model_path}", f"tokenizer={_tokenizer_path(config.model_path)}", f"dtype={config.dtype}", "local_files_only=True"))
    started = time.time()
    captured_stderr = io.StringIO()
    result: Mapping[str, Any] | None = None
    print(f"[stage3][model={MODEL_LABEL}][task={task}] starting lm_eval; overlay={overlay_dir} stderr_log={stderr_log_path}", flush=True)
    try:
        with contextlib.redirect_stderr(captured_stderr):
            _load_parcae().register_auto_class()
            from lm_eval import evaluator
            from lm_eval.tasks import TaskManager
            evaluator.get_git_commit_hash = lambda: "<disabled:lm_eval_git_probe>"
            result = evaluator.simple_evaluate(model="hf", model_args=model_args, tasks=[task], batch_size=config.batch_size, device=config.device, limit=config.limit, log_samples=config.log_samples, task_manager=TaskManager(include_path=str(overlay_dir)), num_fewshot=5 if task in {"mmlu", "gsm8k"} else None, random_seed=config.seed, numpy_random_seed=config.seed, torch_random_seed=config.seed, fewshot_random_seed=config.seed)
        if task == "mmlu":
            result_tasks = result.get("results", {}) if isinstance(result, Mapping) else {}
            expected = {f"mmlu_{subject}" for subject in discover_mmlu_subjects(config.benchmark_root)}
            missing = sorted(expected - set(result_tasks))
            if missing:
                raise RuntimeError(f"lm_eval MMLU result omitted subject rows: missing_count={len(missing)} first={missing[:5]}")
    finally:
        stderr_text = captured_stderr.getvalue()
        stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_log_path.write_text(stderr_text, encoding="utf-8")
        if stderr_text:
            print(stderr_text, file=sys.stderr, end="", flush=True)
    if result is None:
        raise RuntimeError(f"lm_eval returned no result for task {task!r}")
    payload = {"requested_task": task, "model_label": MODEL_LABEL, "architecture_contract": ARCHITECTURE_CONTRACT, "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(), "finished_at": utc_now(), "task_config": str(overlay_dir), "stderr_log": str(stderr_log_path), "raw_lm_eval": result}
    write_json(task_dir / "lm_eval_results.json", payload)
    if config.log_samples and isinstance(result, Mapping) and "samples" in result:
        write_json(task_dir / "log_samples.json", {"samples": result["samples"]})
    return payload, _flatten_result_rows(task, result)


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    started_at = utc_now()
    output_dir = ensure_external_output(config.output_dir)
    log_root = ensure_log_root(config.log_root or DEFAULT_LOG_ROOT)
    set_offline_environment(config.cache_dir or DEFAULT_CACHE_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_info = inspect_model_artifacts_5_10xpoisson_parcae(config.model_path)
    versions = inspect_pinned_versions()
    benchmark = validate_benchmark_layout(config.benchmark_root)
    task_probe_dir = output_dir / ".task-config-probe"
    task_probe_dir.mkdir(parents=True, exist_ok=True)
    protocol = prepare_local_task_overlays(config.benchmark_root, task_probe_dir, config.tasks)
    runtime_audit: dict[str, Any] = {"status": "not_executed"}
    if not config.validation_only:
        if not config.device.startswith("cuda"):
            raise RuntimeError("formal Parcae Stage 3 evaluation requires one CUDA device")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("formal Parcae Stage 3 evaluation requires a submitted CUDA job")
        model, tokenizer, runtime_audit = load_and_audit_model_5_10xpoisson_parcae(config, output_dir=output_dir)
        del tokenizer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    task_results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    task_log_paths = {task: str(_task_log_path(log_root, config, task)) for task in config.tasks}
    print(f"[stage3] model={MODEL_LABEL} model_path={config.model_path} architecture_contract={ARCHITECTURE_CONTRACT} output={output_dir} log_root={log_root} tasks={','.join(config.tasks)}", flush=True)
    if not config.validation_only:
        for task in config.tasks:
            task_dir = output_dir / task
            stderr_log_path = _task_log_path(log_root, config, task)
            overlay_dir = task_dir / "lm_eval_include"
            try:
                task_dir = ensure_external_output(task_dir)
                task_dir.mkdir(parents=True, exist_ok=True)
                overlay_dir.mkdir(parents=True, exist_ok=True)
                task_protocol = prepare_local_task_overlays(config.benchmark_root, overlay_dir, (task,))
                write_json(task_dir / "task_protocol.json", task_protocol)
                payload, task_rows = _run_single_task(config, task, task_dir, overlay_dir, stderr_log_path)
                task_results[task] = payload
                rows.extend(task_rows)
            except Exception:
                failure_text = traceback.format_exc()
                failures[task] = failure_text
                _append_text(stderr_log_path, f"\n=== Stage 3 Parcae task failure: {task} ===\nmodel={config.model_path}\noutput={task_dir}\n{failure_text}")
                print(failure_text, file=sys.stderr, end="", flush=True)
    _write_summary(output_dir, rows)
    audit = {"status": "FAIL" if failures else "PASS", "stage": "stage3_benchmark_evaluation", "model_label": MODEL_LABEL, "model_path": str(config.model_path), "architecture_contract": ARCHITECTURE_CONTRACT, "started_at": started_at, "finished_at": utc_now(), "command": sys.argv, "git_commit": git_commit(), "platform": platform.platform(), "packages": versions, "model": model_info, "benchmark_root": str(config.benchmark_root.expanduser().resolve()), "benchmark_manifest": benchmark, "protocol": protocol, "configuration": asdict(config), "log_root": str(log_root), "task_log_paths": task_log_paths, "gpu": _gpu_info(config.device), "recursive_runtime_audit": runtime_audit, "tasks": list(config.tasks), "task_results": task_results, "summary_rows": rows, "sample_counts": {task: _result_sample_counts(payload.get("raw_lm_eval", {})) for task, payload in task_results.items()}, "skipped_count": len(config.tasks) if config.validation_only else 0, "failed_count": len(failures), "failures": failures, "output_dir": str(output_dir), "formal_eval_executed": not config.validation_only}
    write_json(output_dir / "audit_report.json", audit)
    write_json(output_dir / "run_config.json", {"configuration": asdict(config), "protocol": protocol})
    if failures:
        raise RuntimeError(f"Stage 3 Parcae task failures: {sorted(failures)}")
    return audit


def parse_args(argv: Sequence[str] | None = None) -> EvaluationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=STAGE3_TASKS, default=list(STAGE3_TASKS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--no-log-samples", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when supplied")
    if args.smoke and args.limit is None:
        args.limit = 2
    return EvaluationConfig(model_path=args.model_path, benchmark_root=args.benchmark_root, output_dir=args.output_dir, tasks=tuple(args.tasks), device=args.device, dtype=args.dtype, batch_size=args.batch_size, seed=args.seed, limit=args.limit, log_samples=not args.no_log_samples, validation_only=args.validation_only, smoke=args.smoke, cache_dir=args.cache_dir or DEFAULT_CACHE_ROOT, log_root=args.log_root or DEFAULT_LOG_ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    config: EvaluationConfig | None = None
    output_was_fresh = False
    try:
        config = parse_args(argv)
        try:
            candidate = ensure_external_path(config.output_dir, label="output")
            output_was_fresh = not candidate.exists() or (candidate.is_dir() and not any(candidate.iterdir()))
        except Exception:
            output_was_fresh = False
        audit = run_evaluation(config)
        print(json.dumps(json_safe(audit), ensure_ascii=False, indent=2), flush=True)
        print(f"[result] status={audit['status']} output={config.output_dir}", flush=True)
        return 0
    except Exception:
        failure_text = traceback.format_exc()
        print("[result] status=FAIL", file=sys.stderr, flush=True)
        print(failure_text, file=sys.stderr, end="", flush=True)
        if config is not None:
            try:
                _append_text(_runtime_log_path(ensure_log_root(config.log_root or DEFAULT_LOG_ROOT), config), f"\n=== Stage 3 Parcae process failure ===\nmodel={config.model_path}\noutput={config.output_dir}\n{failure_text}")
            except Exception:
                pass
            if output_was_fresh:
                try:
                    failure_dir = ensure_external_path(config.output_dir, label="output")
                    failure_dir.mkdir(parents=True, exist_ok=True)
                    report = failure_dir / "audit_report.json"
                    if not report.exists():
                        write_json(report, {"status": "FAIL", "stage": "stage3_benchmark_evaluation", "model_label": MODEL_LABEL, "model_path": str(config.model_path), "architecture_contract": ARCHITECTURE_CONTRACT, "started_at": utc_now(), "finished_at": utc_now(), "command": sys.argv, "configuration": asdict(config), "failure": failure_text, "formal_eval_executed": False})
                except Exception:
                    pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
