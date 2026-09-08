#!/usr/bin/env python3
"""Offline Stage 3 evaluation and MeSH preflight for 5-10x2-5 checkpoints.

This entry point is intentionally isolated from every other recursive model.
It imports only ``recursive_model_5_10x2_5_mesh.py`` by path, registers the
real MeSH class, and reuses only the generic benchmark/environment helpers
from :mod:`evaluate_stage3`.  A report is written even when artifact, runtime,
or benchmark checks fail.  Router collapse and slot imbalance are diagnostic
warnings, never global hard failures: those parameters are trainable.
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
from dataclasses import asdict, dataclass
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


# The benchmark helper contains no recursive-model import.  Loading it by
# path keeps this script usable as ``python scripts/...py`` from a checkout.
_stage3 = _load_module_from_path(
    "rsmol_stage3_benchmark_helpers_mesh",
    SCRIPT_ROOT / "scripts" / "evaluate_stage3.py",
)


def _load_mesh() -> Any:
    """Load exactly the MeSH architecture under its canonical namespace."""

    return _load_module_from_path(
        "code.RSmol.recursive_model_5_10x2_5_mesh",
        SCRIPT_ROOT / "recursive_model_5_10x2_5_mesh.py",
    )


DEFAULT_MODEL = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "stage4_5_10x2_5_mesh/formal_resume_000500_nonfatal_router_20260907_115533/"
    "checkpoint-009244"
)
DEFAULT_BENCHMARK_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/eval_datasets"
)
DEFAULT_OUTPUT_DIR = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "stage3_eval_5_10x2_5_mesh_009244"
)
DEFAULT_LOG_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/eval_logs/"
    "stage3_5_10x2_5_mesh_009244"
)
DEFAULT_CACHE_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/eval_cache/"
    "stage3_5_10x2_5_mesh_009244"
)
STAGE3_TASKS = _stage3.STAGE3_TASKS
EXPECTED_MMLU_SUBJECTS = _stage3.EXPECTED_MMLU_SUBJECTS
EXPECTED_LM_EVAL_VERSION = _stage3.EXPECTED_LM_EVAL_VERSION
EXPECTED_TRANSFORMERS_VERSION = _stage3.EXPECTED_TRANSFORMERS_VERSION
EXPECTED_DATASETS_VERSION = _stage3.EXPECTED_DATASETS_VERSION

# These are copied from the MeSH module at audit time as a defense against
# accidentally auditing a 5-10-5/Parcae contract.  The literal constants also
# make the contract independently inspectable before importing torch.
MODEL_LABEL = "5_10x2_5_mesh"
ARCHITECTURE_CONTRACT = "logical_30_physical_20_5_10x2_5_mesh"
EXPECTED_ARCHITECTURES = (
    "RecursiveLlama5_10x2_5MeshForCausalLM",
    "RecursiveLlamaForCausalLM",
)
LOGICAL_LAYER_COUNT = 30
PHYSICAL_LAYER_COUNT = 20
PREFIX_LAYER_COUNT = 5
MIDDLE_LAYER_COUNT = 10
SUFFIX_LAYER_COUNT = 5
RECURSIVE_LOOPS = 2
LOOPS_SCOPE = "middle_only"
MEMORY_SLOT_COUNT = 5
ROUTER_COUNT = 6
ROUTER_PARAMETER_COUNT = 6
TRANSITION_QUERY = "prefix_output"
MAPPING_POLICY = "explicit_5_10_5_source_layers_mesh"
SOURCE_MAPPING_0BASED = (
    0, 1, 2, 3, 4, 5, 7, 9, 11, 13,
    15, 17, 19, 21, 23, 25, 26, 27, 28, 29,
)
LOGICAL_TO_PHYSICAL = (
    0, 1, 2, 3, 4,
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19,
)


def _expected_schedule(loops: int = RECURSIVE_LOOPS) -> tuple[int, ...]:
    """Return the only supported MeSH logical schedule."""

    if int(loops) != RECURSIVE_LOOPS:
        raise ValueError(f"MeSH supports exactly {RECURSIVE_LOOPS} middle loops, got {loops}")
    return LOGICAL_TO_PHYSICAL
BF16_MAX_ABS_WARNING = 1.0
BF16_MIN_COSINE_WARNING = 0.999
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

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
git_commit = _stage3.git_commit
utc_now = _stage3.utc_now


@dataclass
class MeshEvaluationConfig:
    model_path: Path
    benchmark_root: Path
    output_dir: Path
    tokenizer_path: Path | None = None
    tasks: tuple[str, ...] = STAGE3_TASKS
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    batch_size: int = 1
    seed: int = 0
    limit: int | None = None
    max_new_tokens: int = 2
    log_samples: bool = True
    validation_only: bool = False
    smoke: bool = False
    cache_dir: Path | None = None
    log_root: Path | None = None
    report_path: Path | None = None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _model_file_manifest(model_dir: Path) -> list[str]:
    files: set[Path] = set()
    for pattern in ("*.safetensors", "pytorch_model*.bin", "*.bin"):
        files.update(model_dir.glob(pattern))
    return sorted(path.name for path in files if path.is_file())


def _tokenizer_file_manifest(tokenizer_dir: Path) -> list[str]:
    names = (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "tokenizer.model", "spiece.model", "vocab.json", "merges.txt",
    )
    return [name for name in names if (tokenizer_dir / name).is_file()]


def _tokenizer_path(model_dir: Path, override: Path | None = None) -> Path:
    if override is not None:
        return ensure_external_path(override, label="MeSH tokenizer")
    nested = model_dir / "tokenizer"
    return nested if nested.is_dir() else model_dir


def _exact_tuple(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None


def _first_present(config: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in config:
            return config[name]
    return None


def _strict_config_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the serialized MeSH contract before Transformers loading."""

    if config.get("model_type") != "llama":
        raise ValueError(f"MeSH model_type must be llama, got {config.get('model_type')!r}")
    architectures = tuple(str(item) for item in config.get("architectures", ()))
    if not architectures or not any(item in EXPECTED_ARCHITECTURES for item in architectures):
        raise ValueError(
            "MeSH architectures must include RecursiveLlama5_10x2_5MeshForCausalLM "
            f"(or its alias), got {architectures!r}"
        )
    required = {
        "num_hidden_layers": (LOGICAL_LAYER_COUNT,),
        "recursive_layer_count": (PHYSICAL_LAYER_COUNT,),
        "recursive_loops": (RECURSIVE_LOOPS,),
        "recursive_loops_scope": (LOOPS_SCOPE,),
        "recursive_prefix_layer_count": (PREFIX_LAYER_COUNT,),
        "recursive_middle_layer_count": (MIDDLE_LAYER_COUNT,),
        "recursive_suffix_layer_count": (SUFFIX_LAYER_COUNT,),
    }
    for key, expected_values in required.items():
        if config.get(key) not in expected_values:
            raise ValueError(f"MeSH config mismatch/missing {key}: got={config.get(key)!r} expected={expected_values!r}")
    memory_slots = _first_present(config, "mesh_memory_slots", "memory_slots", "recursive_memory_slots")
    if int(memory_slots if memory_slots is not None else -1) != MEMORY_SLOT_COUNT:
        raise ValueError(f"MeSH memory_slots must be {MEMORY_SLOT_COUNT}, got {memory_slots!r}")
    router_count = _first_present(config, "mesh_router_count", "router_count", "router_parameter_count")
    if int(router_count if router_count is not None else -1) != ROUTER_COUNT:
        raise ValueError(f"MeSH router_count must be {ROUTER_COUNT}, got {router_count!r}")
    transition = _first_present(config, "mesh_transition_query", "transition_query", "recursive_transition_query")
    if transition != TRANSITION_QUERY:
        raise ValueError(f"MeSH transition_query must be {TRANSITION_QUERY!r}, got {transition!r}")
    mapping = _exact_tuple(config.get("logical_to_physical"))
    if mapping != LOGICAL_TO_PHYSICAL:
        raise ValueError(f"MeSH logical_to_physical mismatch: got={mapping!r}")
    for key in ("logical_to_physical_schedule", "recursive_logical_to_physical", "recursive_logical_to_physical_schedule"):
        if key in config and _exact_tuple(config[key]) != LOGICAL_TO_PHYSICAL:
            raise ValueError(f"MeSH {key} mismatch")
    source = _exact_tuple(config.get("recursive_source_layer_indices_0based"))
    if source != SOURCE_MAPPING_0BASED:
        raise ValueError(f"MeSH source mapping mismatch: got={source!r} expected={SOURCE_MAPPING_0BASED!r}")
    if "recursive_source_layer_indices_1based" in config:
        expected_1 = tuple(item + 1 for item in SOURCE_MAPPING_0BASED)
        if _exact_tuple(config["recursive_source_layer_indices_1based"]) != expected_1:
            raise ValueError("MeSH 1-based source mapping mismatch")
    policy = _first_present(config, "recursive_mapping_policy", "mapping_policy")
    if policy is not None and policy != MAPPING_POLICY:
        raise ValueError(f"MeSH mapping policy mismatch: {policy!r}")
    return {
        "status": "PASS",
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "architectures": list(architectures),
        "model_type": "llama",
        "logical_layer_count": LOGICAL_LAYER_COUNT,
        "physical_layer_count": PHYSICAL_LAYER_COUNT,
        "recursive_loops": RECURSIVE_LOOPS,
        "loops_scope": LOOPS_SCOPE,
        "prefix_layer_count": PREFIX_LAYER_COUNT,
        "middle_layer_count": MIDDLE_LAYER_COUNT,
        "suffix_layer_count": SUFFIX_LAYER_COUNT,
        "memory_slots": MEMORY_SLOT_COUNT,
        "router_count": ROUTER_COUNT,
        "transition_query": TRANSITION_QUERY,
        "logical_to_physical": list(LOGICAL_TO_PHYSICAL),
        "source_mapping_0based": list(SOURCE_MAPPING_0BASED),
        "mapping_policy": policy,
    }


def _metadata_contract(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return mismatches for optional MeSH metadata (never trust it blindly)."""

    expected: dict[str, Any] = {
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "logical_layer_count": LOGICAL_LAYER_COUNT,
        "physical_layer_count": PHYSICAL_LAYER_COUNT,
        "prefix_layer_count": PREFIX_LAYER_COUNT,
        "middle_layer_count": MIDDLE_LAYER_COUNT,
        "suffix_layer_count": SUFFIX_LAYER_COUNT,
        "loops": RECURSIVE_LOOPS,
        "memory_slots": MEMORY_SLOT_COUNT,
        "router_parameter_count": ROUTER_PARAMETER_COUNT,
        "transition_query": TRANSITION_QUERY,
        "logical_to_physical": list(LOGICAL_TO_PHYSICAL),
        "source_layer_indices_0based": list(SOURCE_MAPPING_0BASED),
    }
    mismatches = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if key in metadata and metadata.get(key) != value
    }
    return {"present": True, "mismatches": mismatches, "keys": sorted(metadata)}


def inspect_model_artifacts_5_10x2_5_mesh(path: Path, *, tokenizer_path: Path | None = None) -> dict[str, Any]:
    """Hard-check core files/config; optional metadata absence is a warning."""

    model_dir = ensure_external_path(path, label="5-10x2-5 MeSH checkpoint")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"MeSH checkpoint is not an existing directory: {model_dir}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"MeSH checkpoint is missing config.json: {model_dir}")
    config = _read_json(config_path)
    contract = _strict_config_contract(config)
    model_files = _model_file_manifest(model_dir)
    if not model_files:
        raise FileNotFoundError(f"MeSH checkpoint has no local model weight artifact: {model_dir}")
    tokenizer_dir = _tokenizer_path(model_dir, tokenizer_path)
    tokenizer_files = _tokenizer_file_manifest(tokenizer_dir)
    if "tokenizer_config.json" not in tokenizer_files:
        raise FileNotFoundError(f"MeSH tokenizer is missing tokenizer_config.json: {tokenizer_dir}")
    if not ({"tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json"} & set(tokenizer_files)):
        raise FileNotFoundError(f"MeSH tokenizer has no local vocabulary payload: {tokenizer_dir}")
    tokenizer_config = _read_json(tokenizer_dir / "tokenizer_config.json")
    tokenizer_vocab_size: int | None = None
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if tokenizer_json.is_file():
        try:
            payload = _read_json(tokenizer_json)
            vocab = payload.get("model", {}).get("vocab", {})
            if isinstance(vocab, dict):
                tokenizer_vocab_size = len(vocab)
        except ValueError:
            tokenizer_vocab_size = None
    config_vocab_size = config.get("vocab_size")
    vocab_compatible = tokenizer_vocab_size is None or config_vocab_size is None or int(config_vocab_size) == tokenizer_vocab_size
    if not vocab_compatible:
        raise ValueError(f"MeSH tokenizer/model vocab mismatch: config={config_vocab_size} tokenizer={tokenizer_vocab_size}")
    warnings: list[str] = []
    metadata_reports: dict[str, Any] = {}
    for name in ("mesh_checkpoint_metadata.json", "mesh_conversion_metadata.json"):
        metadata_path = model_dir / name
        if metadata_path.is_file():
            report = _metadata_contract(_read_json(metadata_path))
            metadata_reports[name] = report
            if report["mismatches"]:
                raise ValueError(f"MeSH metadata contract mismatch in {name}: {report['mismatches']}")
        elif name == "mesh_checkpoint_metadata.json":
            warnings.append("mesh_checkpoint_metadata.json is absent; metadata audit is limited")
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
        "recursive_audit": {"status": "PASS", **contract, "metadata": metadata_reports, "warnings": warnings},
        "warnings": warnings,
    }


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device


def _prompt(model: Any, *, length: int = 4) -> Any:
    import torch
    vocab = int(getattr(model.config, "vocab_size", 128))
    return torch.arange(length, device=_model_device(model), dtype=torch.long).view(1, -1) % vocab


def _finite(name: str, tensor: Any) -> None:
    import torch
    if not bool(torch.isfinite(tensor.float()).all()):
        raise RuntimeError(f"{name} contains non-finite values")


def _assert_logits(outputs: Any, model: Any, *, batch: int, sequence: int) -> Any:
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("MeSH model output has no logits")
    expected = (batch, sequence, int(model.config.vocab_size))
    if tuple(logits.shape) != expected:
        raise RuntimeError(f"MeSH logits shape mismatch: got={tuple(logits.shape)} expected={expected}")
    _finite("MeSH logits", logits)
    return logits


def _physical_layers(model: Any) -> list[Any]:
    recursive_model = getattr(model, "model", model)
    layers = list(getattr(recursive_model, "layers", ()))
    if len(layers) != PHYSICAL_LAYER_COUNT or len({id(item) for item in layers}) != PHYSICAL_LAYER_COUNT:
        raise RuntimeError(f"MeSH physical layer audit failed: count={len(layers)} unique={len({id(item) for item in layers})}")
    return layers


@contextlib.contextmanager
def _physical_trace(model: Any):
    recursive_model = getattr(model, "model", model)
    trace: list[int] = []
    handles = [layer.register_forward_hook(lambda _m, _i, _o, index=index: trace.append(index)) for index, layer in enumerate(_physical_layers(model))]
    try:
        yield recursive_model, trace
    finally:
        for handle in handles:
            handle.remove()


def _cache_storage(cache: Any, slot: int) -> tuple[Any, Any]:
    layers = getattr(cache, "layers", None)
    if layers is not None and slot < len(layers):
        layer = layers[slot]
        return getattr(layer, "keys", getattr(layer, "key_cache", None)), getattr(layer, "values", getattr(layer, "value_cache", None))
    keys = getattr(cache, "key_cache", None)
    values = getattr(cache, "value_cache", None)
    if keys is not None and values is not None and slot < len(keys):
        return keys[slot], values[slot]
    raise RuntimeError(f"MeSH cache has no logical slot {slot}")


def _cache_slot_audit(cache: Any, *, expected_length: int) -> dict[str, Any]:
    try:
        capacity = len(cache)
    except TypeError:
        capacity = LOGICAL_LAYER_COUNT
    if capacity < LOGICAL_LAYER_COUNT:
        raise RuntimeError(f"MeSH cache capacity {capacity} < {LOGICAL_LAYER_COUNT}")
    slots: list[dict[str, Any]] = []
    for slot in range(LOGICAL_LAYER_COUNT):
        try:
            length = int(cache.get_seq_length(slot))
        except TypeError:
            length = int(cache.get_seq_length(layer_idx=slot))
        keys, values = _cache_storage(cache, slot)
        if keys is None or values is None or not hasattr(keys, "shape") or not hasattr(values, "shape"):
            raise RuntimeError(f"MeSH logical cache slot {slot} is empty")
        if length < expected_length:
            raise RuntimeError(f"MeSH cache slot {slot} length {length} < {expected_length}")
        slots.append({"slot": slot, "length": length, "key_shape": list(keys.shape), "value_shape": list(values.shape)})
    return {"logical_slot_count": len(slots), "expected_length": expected_length, "slots": slots}


def _cosine(a: Any, b: Any) -> float:
    import torch
    return float(torch.nn.functional.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item())


def _run_forward(model: Any, input_ids: Any, *, use_cache: bool, past_key_values: Any = None) -> tuple[Any, list[int], dict[str, Any]]:
    recursive_model = getattr(model, "model", model)
    kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": use_cache}
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    with _physical_trace(model) as (backbone, trace):
        outputs = model(**kwargs)
    _assert_logits(outputs, model, batch=input_ids.shape[0], sequence=input_ids.shape[1])
    forward_trace = list(getattr(backbone, "last_forward_trace", ()))
    return outputs, trace, {"logical_trace": forward_trace, "memory_shape": list(getattr(backbone, "last_memory_shape", ()) or ())}


def _audit_transition_and_memory(model: Any, *, logical_trace: list[dict[str, Any]], expected_shape: list[int]) -> dict[str, Any]:
    recursive_model = getattr(model, "model", model)
    if logical_trace != [{"logical_index": i, "physical_index": p} for i, p in enumerate(LOGICAL_TO_PHYSICAL)]:
        raise RuntimeError(f"MeSH logical forward trace mismatch: {logical_trace}")
    if list(getattr(recursive_model, "last_memory_shape", ())) != expected_shape:
        raise RuntimeError(f"MeSH memory shape mismatch: got={getattr(recursive_model, 'last_memory_shape', None)} expected={expected_shape}")
    queries = getattr(recursive_model, "last_router_queries", {})
    weights = getattr(recursive_model, "last_router_weights", {})
    if not {"write_pre", "read_pre"}.issubset(queries) or len(queries) != 6 or len(weights) != 6:
        raise RuntimeError(f"MeSH router audit did not observe six transition routers: queries={sorted(queries)} weights={sorted(weights)}")
    prefix = getattr(recursive_model, "last_prefix_output", None)
    if prefix is None:
        raise RuntimeError("MeSH prefix_output was not recorded")
    for name in ("write_pre", "read_pre"):
        if tuple(queries[name].shape) != tuple(prefix.shape):
            raise RuntimeError(f"MeSH {name} query is not prefix_output-shaped")
    return {"logical_steps": len(logical_trace), "middle_repeated": True, "memory_shape": expected_shape, "router_query_names": sorted(queries), "transition_query": TRANSITION_QUERY}


def _runtime_audit(model: Any, tokenizer: Any, model_path: Path, *, output_dir: Path, max_new_tokens: int) -> dict[str, Any]:
    import torch
    mesh = _load_mesh()
    module_contract = {
        "logical": getattr(mesh, "LOGICAL_LAYER_COUNT", None),
        "physical": getattr(mesh, "PHYSICAL_LAYER_COUNT", None),
        "prefix": getattr(mesh, "PREFIX_LAYER_COUNT", None),
        "middle": getattr(mesh, "MIDDLE_LAYER_COUNT", None),
        "suffix": getattr(mesh, "SUFFIX_LAYER_COUNT", None),
        "loops": getattr(mesh, "RECURSIVE_LOOPS", None),
        "memory_slots": getattr(mesh, "MEMORY_SLOT_COUNT", None),
        "router_stages": getattr(mesh, "ROUTER_COUNT", None),
        "router_parameters": getattr(mesh, "ROUTER_PARAMETER_COUNT", None),
        "transition_query": getattr(mesh, "TRANSITION_ROUTER_QUERY", None),
        "logical_to_physical": list(getattr(mesh, "LOGICAL_TO_PHYSICAL", ())),
        "source_mapping_0based": list(getattr(mesh, "SOURCE_MAPPING_0BASED", ())),
    }
    expected_module_contract = {
        "logical": LOGICAL_LAYER_COUNT, "physical": PHYSICAL_LAYER_COUNT,
        "prefix": PREFIX_LAYER_COUNT, "middle": MIDDLE_LAYER_COUNT,
        "suffix": SUFFIX_LAYER_COUNT, "loops": RECURSIVE_LOOPS,
        "memory_slots": MEMORY_SLOT_COUNT, "router_stages": 3,
        "router_parameters": ROUTER_PARAMETER_COUNT,
        "transition_query": TRANSITION_QUERY,
        "logical_to_physical": list(LOGICAL_TO_PHYSICAL),
        "source_mapping_0based": list(SOURCE_MAPPING_0BASED),
    }
    if module_contract != expected_module_contract:
        raise RuntimeError(f"loaded MeSH module constants disagree with evaluator contract: {module_contract}")
    expected_classes = tuple(item for item in (getattr(mesh, "RecursiveLlama5_10x2_5MeshForCausalLM", None), getattr(mesh, "RecursiveLlamaForCausalLM", None)) if item is not None)
    if not expected_classes or not isinstance(model, expected_classes):
        raise TypeError(f"AutoModel resolved the wrong MeSH class: expected={expected_classes!r} got={type(model)!r}")
    layers = _physical_layers(model)
    if _model_device(model).type != "cuda":
        raise RuntimeError(f"MeSH runtime audit requires GPU, got {_model_device(model)}")
    config_dict = model.config.to_dict() if hasattr(model.config, "to_dict") else vars(model.config)
    config_contract = _strict_config_contract(config_dict)
    parameter_audit = mesh.parameter_audit(model)
    if int(parameter_audit.get("physical_layer_count", -1)) != PHYSICAL_LAYER_COUNT or not parameter_audit.get("router_objects_independent"):
        raise RuntimeError(f"MeSH parameter audit failed: {parameter_audit}")
    if parameter_audit.get("schedule") != list(LOGICAL_TO_PHYSICAL) or parameter_audit.get("source_mapping_0based") != list(SOURCE_MAPPING_0BASED):
        raise RuntimeError("MeSH parameter audit schedule/source mapping disagrees with the model contract")
    if any(parameter_audit.get(key) != value for key, value in (("logical_layer_count", LOGICAL_LAYER_COUNT), ("physical_layer_count", PHYSICAL_LAYER_COUNT), ("logical_cache_slot_count", LOGICAL_LAYER_COUNT), ("recursive_loops", RECURSIVE_LOOPS), ("memory_slots", MEMORY_SLOT_COUNT), ("router_count", ROUTER_COUNT), ("transition_query", TRANSITION_QUERY))):
        raise RuntimeError(f"MeSH parameter audit contract mismatch: {parameter_audit}")
    router_modules = list(getattr(recursive_model, "write_routers", ())) + list(getattr(recursive_model, "read_routers", ()))
    names = [name for name, _ in model.named_parameters(remove_duplicate=False) if ".write_routers." in name or ".read_routers." in name]
    if len(router_modules) != ROUTER_COUNT or len(names) != ROUTER_COUNT * 2:
        raise RuntimeError(f"MeSH router parameter audit expected 6 router modules/12 tensors: modules={len(router_modules)} tensors={len(names)}")
    if not all(torch.isfinite(parameter.detach().float()).all() for name, parameter in model.named_parameters() if name in names):
        raise RuntimeError(f"MeSH router parameters are not finite: {names}")
    recursive_model = getattr(model, "model", model)
    recursive_model.audit_mode = True
    recursive_model.routing_stats_mode = True
    input_ids = _prompt(model)
    model.eval()
    with torch.inference_mode():
        no_cache, no_trace, no_audit = _run_forward(model, input_ids, use_cache=False)
        prefill, pre_trace, pre_audit = _run_forward(model, input_ids, use_cache=True)
    no_logits = no_cache.logits
    pre_logits = prefill.logits
    if tuple(no_logits.shape) != tuple(prefill.logits.shape):
        raise RuntimeError("MeSH no-cache/prefill logits shape mismatch")
    prefill_max_abs = float((no_logits.float() - pre_logits.float()).abs().max().item())
    prefill_mean_abs = float((no_logits.float() - pre_logits.float()).abs().mean().item())
    prefill_cosine = _cosine(no_logits, pre_logits)
    trace_audit = _audit_transition_and_memory(model, logical_trace=list(no_audit["logical_trace"]), expected_shape=[1, MEMORY_SLOT_COUNT, input_ids.shape[1], int(model.config.hidden_size)])
    cache_audit = _cache_slot_audit(prefill.past_key_values, expected_length=input_ids.shape[1])
    warnings: list[str] = []
    hard_failures: list[str] = []
    is_bf16 = getattr(model, "dtype", None) == torch.bfloat16
    if is_bf16 and (prefill_max_abs > BF16_MAX_ABS_WARNING or prefill_cosine < BF16_MIN_COSINE_WARNING):
        warnings.append(f"BF16 prefill/no-cache drift warning: max_abs={prefill_max_abs:.6g} cosine={prefill_cosine:.6g}")
    elif not is_bf16 and (prefill_max_abs > 1e-3 or prefill_cosine < 0.99999):
        raise RuntimeError(f"MeSH prefill/no-cache logits diverged beyond tolerance: max_abs={prefill_max_abs:.6g} cosine={prefill_cosine:.6g}")
    # Incremental-vs-full comparison; drift is warning-only for BF16.
    split = max(1, input_ids.shape[1] - 1)
    with torch.inference_mode():
        first, _, _ = _run_forward(model, input_ids[:, :split], use_cache=True)
        incremental, _, _ = _run_forward(model, input_ids[:, split:], use_cache=True, past_key_values=first.past_key_values)
        full, _, _ = _run_forward(model, input_ids, use_cache=False)
    inc_last = incremental.logits[:, -1:, :]
    full_last = full.logits[:, -1:, :]
    inc_max_abs = float((inc_last.float() - full_last.float()).abs().max().item())
    inc_mean_abs = float((inc_last.float() - full_last.float()).abs().mean().item())
    inc_cosine = _cosine(inc_last, full_last)
    inc_argmax_equal = bool(torch.equal(inc_last.argmax(dim=-1), full_last.argmax(dim=-1)))
    if is_bf16 and (inc_max_abs > BF16_MAX_ABS_WARNING or inc_cosine < BF16_MIN_COSINE_WARNING):
        warnings.append(f"BF16 incremental/full drift warning: max_abs={inc_max_abs:.6g} cosine={inc_cosine:.6g}")
    elif not is_bf16 and (inc_max_abs > 1e-3 or inc_cosine < 0.99999):
        raise RuntimeError(f"MeSH incremental/full logits diverged beyond tolerance: max_abs={inc_max_abs:.6g} cosine={inc_cosine:.6g}")
    # Router collapse/slot imbalance is intentionally a diagnostic warning.
    stats = getattr(recursive_model, "last_routing_stats", {})
    imbalance = []
    for name, item in stats.items():
        probs = item.get("slot_probabilities", [])
        if probs and max(probs) - min(probs) > 0.8:
            imbalance.append(name)
    if imbalance:
        warnings.append(f"router collapse/slot imbalance diagnostic (non-fatal): {imbalance}")
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or getattr(model.config, "eos_token_id", None)
    if pad_id is None:
        raise RuntimeError("MeSH generation requires pad_token_id or eos_token_id")
    with torch.inference_mode():
        generated_cache = model.generate(input_ids=input_ids[:, :2], max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=int(pad_id), eos_token_id=getattr(model.config, "eos_token_id", None))
        generated_no_cache = model.generate(input_ids=input_ids[:, :2], max_new_tokens=max_new_tokens, do_sample=False, use_cache=False, pad_token_id=int(pad_id), eos_token_id=getattr(model.config, "eos_token_id", None))
    if not torch.equal(generated_cache, generated_no_cache) and not is_bf16:
        raise RuntimeError("MeSH greedy generation use_cache/no-cache tokens diverged")
    if not torch.equal(generated_cache, generated_no_cache) and is_bf16:
        warnings.append("greedy generation use_cache/no-cache token drift warning")
    reload_report = _save_reload_audit(model, tokenizer, output_dir=output_dir)
    return {
        "status": "PASS",
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "device": str(_model_device(model)),
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "physical_layer_count": len(layers),
        "unique_physical_layers": len({id(item) for item in layers}),
        "parameter_audit": parameter_audit,
        "config_contract": config_contract,
        "router_parameter_names": names,
        "router_module_count": len(router_modules),
        "router_stats": stats,
        "trace": trace_audit,
        "no_cache": {"logits_shape": list(no_logits.shape), "finite": True, "logical_steps": len(no_trace)},
        "prefill": {"logits_shape": list(prefill.logits.shape), "finite": True, "max_abs": prefill_max_abs, "mean_abs": prefill_mean_abs, "cosine": prefill_cosine},
        "cache": cache_audit,
        "incremental_vs_full": {"max_abs": inc_max_abs, "mean_abs": inc_mean_abs, "cosine": inc_cosine, "argmax_equal": inc_argmax_equal, "warning_only_bf16_drift": True},
        "generation": {"use_cache_shape": list(generated_cache.shape), "no_cache_shape": list(generated_no_cache.shape), "greedy_equal": bool(torch.equal(generated_cache, generated_no_cache)), "max_new_tokens": max_new_tokens},
        "save_reload": reload_report,
        "warnings": warnings,
        "hard_failures": hard_failures,
    }


def recursive_runtime_audit_5_10x2_5_mesh(model: Any, *, tokenizer: Any, model_path: Path, output_dir: Path, max_new_tokens: int = 2) -> dict[str, Any]:
    """Public compatibility name for the isolated MeSH runtime audit."""

    return _runtime_audit(model, tokenizer, model_path, output_dir=output_dir, max_new_tokens=max_new_tokens)


def _save_reload_audit(model: Any, tokenizer: Any, *, output_dir: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mesh = _load_mesh()
    mesh.register_auto_class()
    with tempfile.TemporaryDirectory(prefix="mesh-save-reload-", dir=str(output_dir)) as temporary:
        saved = Path(temporary)
        model.save_pretrained(saved, safe_serialization=True)
        tokenizer.save_pretrained(saved)
        reloaded = AutoModelForCausalLM.from_pretrained(saved, local_files_only=True, torch_dtype=model.dtype, low_cpu_mem_usage=True)
        if not isinstance(reloaded, (mesh.RecursiveLlama5_10x2_5MeshForCausalLM, mesh.RecursiveLlamaForCausalLM)):
            raise RuntimeError(f"MeSH save/reload returned wrong class: {type(reloaded)!r}")
        reloaded.to(_model_device(model)).eval()
        AutoTokenizer.from_pretrained(saved, local_files_only=True, use_fast=True)
        ids = _prompt(reloaded)
        with torch.inference_mode():
            original = model(input_ids=ids, use_cache=False).logits
            restored = reloaded(input_ids=ids, use_cache=False).logits
        _finite("reloaded MeSH logits", restored)
        max_abs = float((original.float() - restored.float()).abs().max().item())
        cosine = _cosine(original, restored)
        if model.dtype != torch.bfloat16 and (max_abs > 1e-3 or cosine < 0.99999):
            raise RuntimeError(f"MeSH save/reload logits diverged beyond tolerance: max_abs={max_abs:.6g} cosine={cosine:.6g}")
        return {"status": "PASS", "class": f"{type(reloaded).__module__}.{type(reloaded).__name__}", "local_files_only": True, "logits_max_abs": max_abs, "logits_shape": list(restored.shape)}


def load_and_audit_model_5_10x2_5_mesh(config: MeshEvaluationConfig, *, output_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mesh = _load_mesh()
    mesh.register_auto_class()
    dtype = getattr(torch, config.dtype)
    model = AutoModelForCausalLM.from_pretrained(config.model_path, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_path(config.model_path, config.tokenizer_path), local_files_only=True, use_fast=True)
    model.to(torch.device(config.device))
    audit = recursive_runtime_audit_5_10x2_5_mesh(model, tokenizer=tokenizer, model_path=config.model_path, output_dir=output_dir, max_new_tokens=config.max_new_tokens)
    return model, tokenizer, audit


def _run_single_task(config: MeshEvaluationConfig, task: str, task_dir: Path, overlay_dir: Path, stderr_log_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import lm_eval
    task_cache = (config.cache_dir or DEFAULT_CACHE_ROOT) / task
    set_offline_environment(task_cache)
    model_args = ",".join((f"pretrained={config.model_path}", f"tokenizer={_tokenizer_path(config.model_path, config.tokenizer_path)}", f"dtype={config.dtype}", "local_files_only=True"))
    started = time.time()
    captured_stderr = io.StringIO()
    result: Mapping[str, Any] | None = None
    print(f"[stage3][model={MODEL_LABEL}][task={task}] starting lm_eval; overlay={overlay_dir}", flush=True)
    try:
        with contextlib.redirect_stderr(captured_stderr):
            _load_mesh().register_auto_class()
            from lm_eval import evaluator
            from lm_eval.tasks import TaskManager
            evaluator.get_git_commit_hash = lambda: "<disabled:lm_eval_git_probe>"
            result = evaluator.simple_evaluate(model="hf", model_args=model_args, tasks=[task], batch_size=config.batch_size, device=config.device, limit=config.limit, log_samples=config.log_samples, task_manager=TaskManager(include_path=str(overlay_dir)), num_fewshot=5 if task in {"mmlu", "gsm8k"} else None, random_seed=config.seed, numpy_random_seed=config.seed, torch_random_seed=config.seed, fewshot_random_seed=config.seed)
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


def run_evaluation(config: MeshEvaluationConfig) -> dict[str, Any]:
    started_at = utc_now()
    output_dir = ensure_external_output(config.output_dir)
    log_root = ensure_log_root(config.log_root or DEFAULT_LOG_ROOT)
    set_offline_environment(config.cache_dir or DEFAULT_CACHE_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    hard_failures: list[str] = []
    traceback_text: str | None = None
    model_info: dict[str, Any] | None = None
    runtime_audit: dict[str, Any] = {"status": "not_executed"}
    benchmark_outputs: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    try:
        model_info = inspect_model_artifacts_5_10x2_5_mesh(config.model_path, tokenizer_path=config.tokenizer_path)
        checks.append({"name": "artifact_contract", "status": "PASS"})
        warnings.extend(model_info.get("warnings", []))
        versions = inspect_pinned_versions()
        benchmark = validate_benchmark_layout(config.benchmark_root)
        task_probe_dir = output_dir / ".task-config-probe"
        task_probe_dir.mkdir(parents=True, exist_ok=True)
        protocol = prepare_local_task_overlays(config.benchmark_root, task_probe_dir, config.tasks)
        checks.append({"name": "benchmark_protocol", "status": "PASS", "tasks": list(config.tasks)})
        if not config.validation_only:
            if not config.device.startswith("cuda"):
                raise RuntimeError("formal MeSH Stage 3 evaluation requires one CUDA device")
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("formal MeSH Stage 3 evaluation requires a submitted CUDA job")
            model, tokenizer, runtime_audit = load_and_audit_model_5_10x2_5_mesh(config, output_dir=output_dir)
            warnings.extend(runtime_audit.get("warnings", []))
            checks.append({"name": "runtime_audit", "status": "PASS"})
            del tokenizer, model
            torch.cuda.empty_cache()
        task_log_paths = {task: str(_task_log_path(log_root, config, task)) for task in config.tasks}
        if not config.validation_only:
            for task in config.tasks:
                task_dir = output_dir / task
                stderr_log_path = _task_log_path(log_root, config, task)
                try:
                    task_dir = ensure_external_output(task_dir)
                    task_dir.mkdir(parents=True, exist_ok=True)
                    overlay_dir = task_dir / "lm_eval_include"
                    overlay_dir.mkdir(parents=True, exist_ok=True)
                    task_protocol = prepare_local_task_overlays(config.benchmark_root, overlay_dir, (task,))
                    write_json(task_dir / "task_protocol.json", task_protocol)
                    payload, task_rows = _run_single_task(config, task, task_dir, overlay_dir, stderr_log_path)
                    benchmark_outputs[task] = payload
                    rows.extend(task_rows)
                except Exception:
                    failure = traceback.format_exc()
                    failures[task] = failure
                    hard_failures.append(f"benchmark task {task} failed")
                    _append_text(stderr_log_path, f"\n=== Stage 3 MeSH task failure: {task} ===\n{failure}")
        _write_summary(output_dir, rows)
        status = "FAIL" if hard_failures else "PASS"
        report = {"status": status, "stage": "stage3_benchmark_evaluation", "model_label": MODEL_LABEL, "architecture_contract": ARCHITECTURE_CONTRACT, "model_path": str(config.model_path), "started_at": started_at, "finished_at": utc_now(), "command": sys.argv, "git_commit": git_commit(), "platform": platform.platform(), "packages": versions, "model": model_info, "benchmark_root": str(config.benchmark_root.expanduser().resolve()), "benchmark_manifest": benchmark, "protocol": protocol, "configuration": asdict(config), "log_root": str(log_root), "checks": checks, "warnings": warnings, "hard_failures": hard_failures, "traceback": traceback_text, "runtime_audit": runtime_audit, "recursive_runtime_audit": runtime_audit, "benchmark_outputs": benchmark_outputs, "task_results": benchmark_outputs, "summary_rows": rows, "sample_counts": {task: _result_sample_counts(payload.get("raw_lm_eval", {})) for task, payload in benchmark_outputs.items()}, "failed_count": len(failures), "failures": failures, "output_dir": str(output_dir), "formal_eval_executed": not config.validation_only}
        write_json(config.report_path or (output_dir / "audit_report.json"), report)
        write_json(output_dir / "run_config.json", {"configuration": asdict(config), "protocol": protocol})
        return report
    except Exception:
        traceback_text = traceback.format_exc()
        hard_failures.append(str(traceback_text.splitlines()[-1] if traceback_text else "evaluation failed"))
        report = {"status": "FAIL", "stage": "stage3_benchmark_evaluation", "model_label": MODEL_LABEL, "architecture_contract": ARCHITECTURE_CONTRACT, "model_path": str(config.model_path), "started_at": started_at, "finished_at": utc_now(), "command": sys.argv, "configuration": asdict(config), "checks": checks, "warnings": warnings, "hard_failures": hard_failures, "traceback": traceback_text, "runtime_audit": runtime_audit, "benchmark_outputs": benchmark_outputs, "formal_eval_executed": False}
        try:
            write_json(config.report_path or (output_dir / "audit_report.json"), report)
        except Exception:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> MeshEvaluationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", choices=STAGE3_TASKS, default=list(STAGE3_TASKS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--no-log-samples", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--batch-size and --max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when supplied")
    if args.smoke and args.limit is None:
        args.limit = 2
    return MeshEvaluationConfig(model_path=args.model_path, tokenizer_path=args.tokenizer_path, benchmark_root=args.benchmark_root, output_dir=args.output_dir, report_path=args.report_path, tasks=tuple(args.tasks), device=args.device, dtype=args.dtype, batch_size=args.batch_size, seed=args.seed, limit=args.limit, max_new_tokens=args.max_new_tokens, log_samples=not args.no_log_samples, validation_only=args.validation_only, smoke=args.smoke, cache_dir=args.cache_dir, log_root=args.log_root)


def main(argv: Sequence[str] | None = None) -> int:
    config: MeshEvaluationConfig | None = None
    try:
        config = parse_args(argv)
        report = run_evaluation(config)
        print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
        print(f"[result] status={report['status']} output={config.output_dir}", flush=True)
        return 0 if report["status"] == "PASS" else 1
    except Exception:
        print("[result] status=FAIL", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, end="", flush=True)
        if config is not None:
            try:
                _append_text(_runtime_log_path(ensure_log_root(config.log_root or DEFAULT_LOG_ROOT), config), f"\n=== Stage 3 MeSH process failure ===\n{traceback.format_exc()}")
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
