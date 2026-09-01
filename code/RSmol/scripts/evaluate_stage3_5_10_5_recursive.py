#!/usr/bin/env python3
"""Offline Stage 3 evaluation for the isolated SmolLM2 5-10-5 recursive baseline.

The benchmark implementation remains lm-evaluation-harness 0.4.12.  This
entry point reuses only the benchmark-snapshot, YAML-overlay, logging, and
result helpers from ``evaluate_stage3``; model inspection, registration, and
runtime auditing are local to the 5-10-5 architecture.  In particular, this
process registers only ``code.RSmol.recursive_model_5_10_5`` and never mixes
model registries with the linear 5-10-5 architecture.

The supported official tasks are ``hellaswag``, ``mmlu``, ``gsm8k``,
``arc_easy``, and ``arc_challenge``.  Their task YAMLs retain the official
prompting and metrics: HellaSwag/ARC multiple-choice log-likelihood, MMLU
test with task-scoped five-shot dev examples, and GSM8K main/test with
deterministic five-shot generation and exact-match answer extraction.
Each run writes ``lm_eval_results.json``, ``log_samples.json``,
``summary.json``, ``summary.csv``, ``audit_report.json``, and
``run_config.json`` under the external output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import importlib.util
import json
import os
import platform
import sys
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
    """Load a project helper without executing ``code.RSmol.__init__``.

    The package initializer imports the other recursive implementation.  The
    isolated evaluator must not import or register that implementation merely
    to reuse the dependency-light Stage 3 benchmark helpers.
    """

    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_stage3 = _load_module_from_path(
    "rsmol_stage3_benchmark_helpers", SCRIPT_ROOT / "scripts" / "evaluate_stage3.py"
)


def _load_recursive_5_10_5_recursive() -> Any:
    """Load the new model under its canonical name without package side effects."""

    return _load_module_from_path(
        "code.RSmol.recursive_model_5_10_5", SCRIPT_ROOT / "recursive_model_5_10_5.py"
    )


DEFAULT_MODEL = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5")
DEFAULT_BENCHMARK_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/eval_datasets"
)
DEFAULT_LOG_ROOT = SCRIPT_ROOT / "log"
STAGE3_TASKS = _stage3.STAGE3_TASKS
EXPECTED_MMLU_SUBJECTS = _stage3.EXPECTED_MMLU_SUBJECTS
EXPECTED_LM_EVAL_VERSION = _stage3.EXPECTED_LM_EVAL_VERSION
EXPECTED_TRANSFORMERS_VERSION = _stage3.EXPECTED_TRANSFORMERS_VERSION
EXPECTED_DATASETS_VERSION = _stage3.EXPECTED_DATASETS_VERSION

MODEL_LABEL = "recursive_5_10_5_middle_loop2"
ARCHITECTURE_CONTRACT = "logical_30_physical_20_5_10_5_loops_2"
LOGICAL_LAYER_COUNT = 30
PHYSICAL_LAYER_COUNT = 20
PREFIX_LAYER_COUNT = 5
MIDDLE_LAYER_COUNT = 10
SUFFIX_LAYER_COUNT = 5
RECURSIVE_LOOPS = 2
LOOPS_SCOPE = "middle_only"
SOURCE_MAPPING_0BASED = (
    0, 1, 2, 3, 4, 5, 7, 9, 11, 13,
    15, 17, 19, 21, 23, 25, 26, 27, 28, 29,
)
SOURCE_MAPPING_1BASED = tuple(index + 1 for index in SOURCE_MAPPING_0BASED)
LOGICAL_TO_PHYSICAL = (
    0, 1, 2, 3, 4,
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19,
)
MAPPING_POLICY = "explicit_5_10_5_source_layers"
EXPECTED_ARCHITECTURES = ("RecursiveLlamaForCausalLM",)
BF16_INCREMENTAL_MAX_ABS = 1.0
BF16_INCREMENTAL_MIN_COSINE = 0.999
CACHE_ATOL = 1e-5
CACHE_RTOL = 1e-4


EvaluationConfig = _stage3.EvaluationConfig
json_safe = _stage3.json_safe
write_json = _stage3.write_json
ensure_external_path = _stage3.ensure_external_path
ensure_external_output = _stage3.ensure_external_output
ensure_log_root = _stage3.ensure_log_root
set_offline_environment = _stage3.set_offline_environment
inspect_pinned_versions = _stage3.inspect_pinned_versions
load_tokenizer_runtime_metadata = _stage3.load_tokenizer_runtime_metadata
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


def inspect_model_artifacts_5_10_5_recursive(path: Path) -> dict[str, Any]:
    """Validate the complete external 5-10-5 checkpoint before loading it."""

    model_dir = ensure_external_path(path, label="5-10-5 model")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"5-10-5 model is not an existing directory: {model_dir}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"5-10-5 model is missing config.json: {model_dir}")
    config = _read_json(config_path)
    if config.get("model_type") != "llama":
        raise ValueError(
            "SmolLM2-5-10-5 checkpoint must retain model_type='llama' for HF compatibility; "
            f"got {config.get('model_type')!r}"
        )
    model_files = _model_file_manifest(model_dir)
    tokenizer_files = _tokenizer_file_manifest(model_dir)
    if not model_files:
        raise FileNotFoundError(f"5-10-5 model has no local weight artifact: {model_dir}")
    if "tokenizer.json" not in tokenizer_files and "tokenizer.model" not in tokenizer_files:
        raise FileNotFoundError(f"5-10-5 model needs tokenizer.json or tokenizer.model: {model_dir}")
    if "tokenizer_config.json" not in tokenizer_files:
        raise FileNotFoundError(f"5-10-5 model is missing tokenizer_config.json: {model_dir}")

    config_vocab_size = config.get("vocab_size")
    tokenizer_vocab_size: int | None = None
    tokenizer_json = model_dir / "tokenizer.json"
    if tokenizer_json.is_file():
        try:
            tokenizer_payload = _read_json(tokenizer_json)
            vocab = tokenizer_payload.get("model", {}).get("vocab", {})
            if isinstance(vocab, dict):
                tokenizer_vocab_size = len(vocab)
        except ValueError:
            # Keep the artifact audit useful; AutoTokenizer will report the
            # detailed tokenizer-format error during the runtime preflight.
            tokenizer_vocab_size = None
    vocab_compatible = (
        tokenizer_vocab_size is None
        or config_vocab_size is None
        or int(config_vocab_size) == tokenizer_vocab_size
    )
    if not vocab_compatible:
        raise ValueError(
            "Tokenizer/model vocab mismatch for 5-10-5 model: "
            f"config={config_vocab_size} tokenizer={tokenizer_vocab_size}"
        )

    architectures = tuple(str(item) for item in config.get("architectures", ()))
    logical = int(config.get("num_hidden_layers", 0))
    physical = int(config.get("recursive_layer_count", 0))
    loops = int(config.get("recursive_loops", 0))
    prefix = int(config.get("recursive_prefix_layer_count", 0))
    middle = int(config.get("recursive_middle_layer_count", 0))
    suffix = int(config.get("recursive_suffix_layer_count", 0))
    loops_scope = str(config.get("recursive_loops_scope", ""))
    source_mapping = _exact_tuple(config.get("recursive_source_layer_indices_0based"))
    source_mapping_1based = _exact_tuple(config.get("recursive_source_layer_indices_1based"))
    logical_schedule = _exact_tuple(config.get("logical_to_physical"))
    recursive_schedule = _exact_tuple(config.get("recursive_logical_to_physical"))
    schedule_alias = _exact_tuple(config.get("logical_to_physical_schedule"))
    recursive_schedule_alias = _exact_tuple(
        config.get("recursive_logical_to_physical_schedule")
    )
    checks = {
        "architectures": architectures == EXPECTED_ARCHITECTURES,
        "logical_layer_count": logical == LOGICAL_LAYER_COUNT,
        "physical_layer_count": physical == PHYSICAL_LAYER_COUNT,
        "recursive_loops": loops == RECURSIVE_LOOPS,
        "loops_scope": loops_scope == LOOPS_SCOPE,
        "prefix_layer_count": prefix == PREFIX_LAYER_COUNT,
        "middle_layer_count": middle == MIDDLE_LAYER_COUNT,
        "suffix_layer_count": suffix == SUFFIX_LAYER_COUNT,
        "source_mapping_0based": source_mapping == SOURCE_MAPPING_0BASED,
        "source_mapping_1based": source_mapping_1based == SOURCE_MAPPING_1BASED,
        "logical_to_physical": logical_schedule == LOGICAL_TO_PHYSICAL,
        "recursive_logical_to_physical": recursive_schedule == LOGICAL_TO_PHYSICAL,
        "logical_to_physical_schedule": schedule_alias == LOGICAL_TO_PHYSICAL,
        "recursive_logical_to_physical_schedule": recursive_schedule_alias == LOGICAL_TO_PHYSICAL,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "Invalid SmolLM2-5-10-5 checkpoint contract: "
            f"failed={failed} logical={logical} physical={physical} loops={loops} "
            f"architectures={architectures} source_mapping={source_mapping} "
            f"logical_schedule={logical_schedule}"
        )
    tokenizer_config = _read_json(model_dir / "tokenizer_config.json")
    return {
        "label": MODEL_LABEL,
        "path": str(model_dir),
        "config": config,
        "tokenizer_config": tokenizer_config,
        "model_files": model_files,
        "tokenizer_files": tokenizer_files,
        "config_vocab_size": config_vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "vocab_compatible": vocab_compatible,
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "recursive_audit": {
            "is_recursive": True,
            "status": "PASS",
            "architectures": list(architectures),
            "logical_layer_count": logical,
            "physical_layer_count": physical,
            "recursive_loops": loops,
            "loops_scope": loops_scope,
            "prefix_layer_count": prefix,
            "middle_layer_count": middle,
            "suffix_layer_count": suffix,
            "source_mapping_0based": list(source_mapping or ()),
            "source_mapping_1based": list(source_mapping_1based or ()),
            "logical_to_physical": list(logical_schedule or ()),
            "contract_checks": checks,
        },
    }


def _cache_slot_state(cache: Any, index: int) -> tuple[int, bool, bool]:
    """Read canonical DynamicCache layer fields without deprecated properties."""

    length = int(cache.get_seq_length(index))
    key_state: Any = None
    value_state: Any = None
    layers = getattr(cache, "layers", None)
    if layers is not None and index < len(layers):
        key_state = getattr(layers[index], "keys", None)
        value_state = getattr(layers[index], "values", None)
    else:
        key_cache = getattr(cache, "key_cache", ())
        value_cache = getattr(cache, "value_cache", ())
        if index < len(key_cache):
            key_state = key_cache[index]
        if index < len(value_cache):
            value_state = value_cache[index]
    return (
        length,
        hasattr(key_state, "numel") and int(key_state.numel()) > 0,
        hasattr(value_state, "numel") and int(value_state.numel()) > 0,
    )


def _finite(name: str, tensor: Any) -> None:
    import torch

    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"{name} contains non-finite values")


def recursive_runtime_audit_5_10_5_recursive(
    model: Any, *, tokenizer: Any, device: str, max_new_tokens: int = 1
) -> dict[str, Any]:
    """Audit class, mapping, storage, logical trace, cache, and generation."""

    import torch

    recursive_module = _load_recursive_5_10_5_recursive()
    RecursiveLlamaForCausalLM = recursive_module.RecursiveLlamaForCausalLM
    MODEL_LOGICAL_TO_PHYSICAL = recursive_module.LOGICAL_TO_PHYSICAL
    parameter_audit = recursive_module.parameter_audit

    expected_classes = tuple(
        item for item in (
            getattr(recursive_module, "RecursiveLlama5_10_5ForCausalLM", None),
            getattr(recursive_module, "RecursiveLlamaForCausalLM", None),
        ) if item is not None
    )
    if not expected_classes or not isinstance(model, expected_classes):
        raise TypeError(
            "AutoModel resolved the wrong recursive 5-10-5 class: "
            f"expected={[item.__name__ for item in expected_classes]!r} got={type(model)!r}"
        )
    config = model.config
    base = model.model
    layers = list(getattr(base, "layers", ()))
    schedule = tuple(int(item) for item in getattr(base, "logical_to_physical", ()))
    runtime_contract = (
        int(getattr(config, "num_hidden_layers", 0)),
        int(getattr(config, "recursive_layer_count", 0)),
        int(getattr(config, "recursive_loops", 0)),
        int(getattr(config, "recursive_prefix_layer_count", 0)),
        int(getattr(config, "recursive_middle_layer_count", 0)),
        int(getattr(config, "recursive_suffix_layer_count", 0)),
        str(getattr(config, "recursive_loops_scope", "")),
    )
    if runtime_contract != (30, 20, 2, 5, 10, 5, "middle_only"):
        raise ValueError(f"5-10-5 runtime architecture contract mismatch: {runtime_contract}")
    physical_layer_objects = len({id(layer) for layer in layers})
    if len(layers) != PHYSICAL_LAYER_COUNT or physical_layer_objects != PHYSICAL_LAYER_COUNT:
        raise ValueError(
            "5-10-5 physical layer object audit failed: "
            f"count={len(layers)} unique={physical_layer_objects}"
        )
    config_schedule = _exact_tuple(getattr(config, "logical_to_physical", ()))
    source_mapping = _exact_tuple(
        getattr(config, "recursive_source_layer_indices_0based", ())
    )
    if schedule != LOGICAL_TO_PHYSICAL or config_schedule != LOGICAL_TO_PHYSICAL:
        raise ValueError(
            f"5-10-5 runtime schedule mismatch: model={schedule} config={config_schedule}"
        )
    if source_mapping != SOURCE_MAPPING_0BASED:
        raise ValueError(f"5-10-5 runtime source mapping mismatch: {source_mapping}")
    audit = parameter_audit(model)
    if not audit.get("parameter_storage_unique", False):
        raise ValueError(f"5-10-5 parameter storage audit failed: {audit}")
    schedule_counts = {index: schedule.count(index) for index in range(PHYSICAL_LAYER_COUNT)}
    expected_counts = {
        index: (2 if 5 <= index < 15 else 1)
        for index in range(PHYSICAL_LAYER_COUNT)
    }
    if schedule_counts != expected_counts:
        raise ValueError(
            "5-10-5 recursive schedule reuse audit failed: "
            f"expected={expected_counts} got={schedule_counts}"
        )

    trace: list[int] = []
    hooks = [
        layer.register_forward_hook(lambda _m, _i, _o, index=index: trace.append(index))
        for index, layer in enumerate(layers)
    ]
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    model.eval()
    try:
        with torch.inference_mode():
            no_cache = model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    if trace != list(LOGICAL_TO_PHYSICAL):
        raise RuntimeError(
            f"5-10-5 forward trace mismatch: expected={list(LOGICAL_TO_PHYSICAL)} got={trace}"
        )
    _finite("no-cache logits", no_cache.logits)

    with torch.inference_mode():
        with_cache = model(input_ids=input_ids, use_cache=True)
    _finite("cache logits", with_cache.logits)
    prefill_diff = float((no_cache.logits - with_cache.logits).abs().max().item())
    if not torch.allclose(no_cache.logits, with_cache.logits, atol=CACHE_ATOL, rtol=CACHE_RTOL):
        raise RuntimeError(
            "5-10-5 cache/no-cache prefill mismatch: "
            f"max_diff={prefill_diff} atol={CACHE_ATOL} rtol={CACHE_RTOL}"
        )
    cache = with_cache.past_key_values
    if cache is None or len(cache) < LOGICAL_LAYER_COUNT:
        raise RuntimeError(
            f"5-10-5 cache has insufficient logical slots: {len(cache) if cache is not None else 0}"
        )
    prefill_slots = []
    for index in range(LOGICAL_LAYER_COUNT):
        slot = _cache_slot_state(cache, index)
        prefill_slots.append({"index": index, "length": slot[0], "key_nonempty": slot[1], "value_nonempty": slot[2]})
        if slot[0] != input_ids.shape[1] or not slot[1] or not slot[2]:
            raise RuntimeError(f"Invalid 5-10-5 prefill cache slot {index}: {slot}")

    next_input = {"input_ids": input_ids[:, -1:], "use_cache": True, "past_key_values": cache}
    incremental = model(**next_input)
    _finite("incremental logits", incremental.logits)
    extended = model(
        input_ids=torch.cat((input_ids, input_ids[:, -1:]), dim=1), use_cache=False
    )
    _finite("extended logits", extended.logits)
    incremental_logits = incremental.logits[:, -1].float()
    extended_logits = extended.logits[:, -1].float()
    differences = (incremental_logits - extended_logits).abs()
    incremental_max_diff = float(differences.max().item())
    incremental_mean_diff = float(differences.mean().item())
    incremental_cosine = float(
        torch.nn.functional.cosine_similarity(incremental_logits, extended_logits, dim=-1).min().item()
    )
    argmax_equal = bool(
        torch.equal(incremental_logits.argmax(dim=-1), extended_logits.argmax(dim=-1))
    )
    if (
        incremental_max_diff > BF16_INCREMENTAL_MAX_ABS
        or incremental_cosine < BF16_INCREMENTAL_MIN_COSINE
        or not argmax_equal
    ):
        raise RuntimeError(
            "5-10-5 incremental cache semantic audit failed: "
            f"max_diff={incremental_max_diff} mean_diff={incremental_mean_diff} "
            f"cosine={incremental_cosine} argmax_equal={argmax_equal}"
        )
    incremental_slots = []
    for index in range(LOGICAL_LAYER_COUNT):
        slot = _cache_slot_state(cache, index)
        incremental_slots.append({"index": index, "length": slot[0], "key_nonempty": slot[1], "value_nonempty": slot[2]})
        if slot[0] != input_ids.shape[1] + 1 or not slot[1] or not slot[2]:
            raise RuntimeError(f"Invalid 5-10-5 incremental cache slot {index}: {slot}")

    prompt = "Recursive models are"
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {name: value.to(device) for name, value in encoded.items()}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    with torch.inference_mode():
        generated = model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=True, pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
        )
        generated_no_cache = model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=False, pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
        )
    if generated.ndim != 2 or generated.shape[1] <= encoded["input_ids"].shape[1]:
        raise RuntimeError("5-10-5 generation produced no new token")
    cache_no_cache_tokens_equal = bool(torch.equal(generated, generated_no_cache))
    if not cache_no_cache_tokens_equal:
        print(
            "[generation-warning] 5-10-5 cache/no-cache greedy token IDs differ; "
            "retaining the supported cached generation path",
            flush=True,
        )
    precreated_cache = recursive_module.make_dynamic_cache()
    with torch.inference_mode():
        generated_with_precreated_cache = model.generate(
            **encoded,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            past_key_values=precreated_cache,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if len(precreated_cache) < LOGICAL_LAYER_COUNT:
        raise RuntimeError(
            "5-10-5 precreated DynamicCache did not expand through all logical slots: "
            f"capacity={len(precreated_cache)} expected={LOGICAL_LAYER_COUNT}"
        )
    precreated_slots = []
    for index in range(LOGICAL_LAYER_COUNT):
        slot = _cache_slot_state(precreated_cache, index)
        precreated_slots.append(
            {"index": index, "length": slot[0], "key_nonempty": slot[1], "value_nonempty": slot[2]}
        )
        if slot[0] != encoded["input_ids"].shape[1] or not slot[1] or not slot[2]:
            raise RuntimeError(f"Invalid 5-10-5 precreated generation cache slot {index}: {slot}")
    if generated_with_precreated_cache.shape[1] <= encoded["input_ids"].shape[1]:
        raise RuntimeError("5-10-5 precreated-cache generation produced no new token")
    return {
        "status": "PASS",
        "model_class": type(model).__name__,
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "logical_layer_count": LOGICAL_LAYER_COUNT,
        "physical_layer_count": PHYSICAL_LAYER_COUNT,
        "recursive_loops": RECURSIVE_LOOPS,
        "loops_scope": LOOPS_SCOPE,
        "logical_to_physical": list(MODEL_LOGICAL_TO_PHYSICAL),
        "forward_trace": trace,
        "expected_forward_trace": list(LOGICAL_TO_PHYSICAL),
        "physical_layer_object_count": len(layers),
        "unique_physical_layer_object_count": physical_layer_objects,
        "schedule_use_counts": schedule_counts,
        "expected_schedule_use_counts": expected_counts,
        "parameter_audit": audit,
        "cache": {
            "logical_slot_count": len(cache),
            "expected_logical_slot_count": LOGICAL_LAYER_COUNT,
            "prefill_slots": prefill_slots,
            "incremental_slots": incremental_slots,
            "prefill_max_diff": prefill_diff,
            "prefill_atol": CACHE_ATOL,
            "prefill_rtol": CACHE_RTOL,
            "incremental_max_diff": incremental_max_diff,
            "incremental_mean_diff": incremental_mean_diff,
            "incremental_cosine": incremental_cosine,
            "incremental_argmax_equal": argmax_equal,
            "precreated_cache_type": type(precreated_cache).__name__,
            "precreated_cache_slots": precreated_slots,
            "precreated_generation_output_shape": list(generated_with_precreated_cache.shape),
        },
        "generation": {
            "output_shape": list(generated.shape),
            "cache_no_cache_tokens_equal": cache_no_cache_tokens_equal,
            "generation_warning": None if cache_no_cache_tokens_equal else "cached/no-cache greedy IDs diverged",
        },
    }


def load_and_audit_recursive_model_5_10_5_recursive(
    model_path: Path, *, tokenizer: Any, device: str, dtype: str
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM
    recursive_module = _load_recursive_5_10_5_recursive()
    recursive_module.register_auto_class()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=getattr(torch, dtype),
        low_cpu_mem_usage=True,
    )
    # Deliberately no device_map: the submitted single GPU owns evaluation.
    model.to(torch.device(device))
    try:
        return recursive_runtime_audit_5_10_5_recursive(model, tokenizer=tokenizer, device=device)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_single_task_5_10_5_recursive(
    config: EvaluationConfig,
    task: str,
    task_dir: Path,
    overlay_dir: Path,
    stderr_log_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_cache = (
        (config.cache_dir / task)
        if config.cache_dir is not None
        else task_dir / ".hf-cache"
    )
    set_offline_environment(task_cache)
    model_args = ",".join(
        (
            f"pretrained={config.model_path}",
            f"dtype={config.dtype}",
            "local_files_only=True",
        )
    )
    started = time.time()
    captured_stderr = io.StringIO()
    result: Mapping[str, Any] | None = None
    print(
        f"[stage3][model={MODEL_LABEL}][task={task}] starting lm_eval; "
        f"overlay={overlay_dir} stderr_log={stderr_log_path}",
        flush=True,
    )
    try:
        with contextlib.redirect_stderr(captured_stderr):
            # This is the only model registration used by this isolated process.
            _load_recursive_5_10_5_recursive().register_auto_class()
            from lm_eval import evaluator
            from lm_eval.tasks import TaskManager

            evaluator.get_git_commit_hash = lambda: "<disabled:lm_eval_git_probe>"
            result = evaluator.simple_evaluate(
                model="hf",
                model_args=model_args,
                tasks=[task],
                batch_size=config.batch_size,
                device=config.device,
                limit=config.limit,
                log_samples=config.log_samples,
                task_manager=TaskManager(include_path=str(overlay_dir)),
                num_fewshot=5 if task in {"mmlu", "gsm8k"} else None,
                random_seed=config.seed,
                numpy_random_seed=config.seed,
                torch_random_seed=config.seed,
                fewshot_random_seed=config.seed,
            )
        if task == "mmlu":
            result_tasks = result.get("results", {}) if isinstance(result, Mapping) else {}
            expected = {f"mmlu_{subject}" for subject in discover_mmlu_subjects(config.benchmark_root)}
            missing = sorted(expected - set(result_tasks))
            if missing:
                raise RuntimeError(
                    "lm_eval MMLU result omitted subject rows; "
                    f"missing_count={len(missing)} first={missing[:5]}"
                )
    finally:
        stderr_text = captured_stderr.getvalue()
        stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_log_path.write_text(stderr_text, encoding="utf-8")
        if stderr_text:
            print(stderr_text, file=sys.stderr, end="", flush=True)
    if result is None:
        raise RuntimeError(f"lm_eval returned no result for task {task!r}")
    payload = {
        "requested_task": task,
        "model_label": MODEL_LABEL,
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_at": utc_now(),
        "task_config": str(overlay_dir),
        "stderr_log": str(stderr_log_path),
        "raw_lm_eval": result,
    }
    write_json(task_dir / "lm_eval_results.json", payload)
    if config.log_samples and isinstance(result, Mapping) and "samples" in result:
        write_json(task_dir / "log_samples.json", {"samples": result["samples"]})
    return payload, _flatten_result_rows(task, result)


def run_evaluation_5_10_5_recursive(config: EvaluationConfig) -> dict[str, Any]:
    started_at = utc_now()
    output_dir = ensure_external_output(config.output_dir)
    log_root = ensure_log_root(config.log_root)
    # The helper sets HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, and
    # HF_DATASETS_OFFLINE before importing lm_eval or Transformers.
    set_offline_environment(config.cache_dir or (output_dir / ".hf-cache"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_info = inspect_model_artifacts_5_10_5_recursive(config.model_path)
    versions = inspect_pinned_versions()
    load_tokenizer_runtime_metadata(model_info)
    benchmark = validate_benchmark_layout(config.benchmark_root)
    task_probe_dir = output_dir / ".task-config-probe"
    task_probe_dir.mkdir(parents=True, exist_ok=True)
    protocol = prepare_local_task_overlays(config.benchmark_root, task_probe_dir, config.tasks)
    runtime_audit: dict[str, Any] = {"status": "not_executed"}
    if not config.validation_only:
        if not config.device.startswith("cuda"):
            raise RuntimeError("Formal Stage 3 evaluation requires one CUDA device")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Formal Stage 3 evaluation requires a submitted CUDA job")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, local_files_only=True, use_fast=True
        )
        runtime_audit = load_and_audit_recursive_model_5_10_5_recursive(
            config.model_path, tokenizer=tokenizer, device=config.device, dtype=config.dtype
        )

    task_results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    task_log_paths = {task: str(_task_log_path(log_root, config, task)) for task in config.tasks}
    print(
        f"[stage3] model={MODEL_LABEL} model_path={config.model_path} "
        f"architecture_contract={ARCHITECTURE_CONTRACT} output={output_dir} "
        f"log_root={log_root} tasks={','.join(config.tasks)}",
        flush=True,
    )
    if not config.validation_only:
        for task in config.tasks:
            task_dir = output_dir / task
            stderr_log_path = _task_log_path(log_root, config, task)
            overlay_dir = task_dir / "lm_eval_include"
            try:
                task_dir = ensure_external_output(task_dir)
                task_dir.mkdir(parents=True, exist_ok=True)
                overlay_dir.mkdir(parents=True, exist_ok=True)
                task_protocol = prepare_local_task_overlays(
                    config.benchmark_root, overlay_dir, (task,)
                )
                write_json(task_dir / "task_protocol.json", task_protocol)
                payload, task_rows = _run_single_task_5_10_5_recursive(
                    config, task, task_dir, overlay_dir, stderr_log_path
                )
                task_results[task] = payload
                rows.extend(task_rows)
            except Exception:
                failure_text = traceback.format_exc()
                failures[task] = failure_text
                diagnostic = (
                    f"\n=== Stage 3 5-10-5 recursive task failure: {task} ===\n"
                    f"model={config.model_path}\noutput={output_dir / task}\n"
                    f"stderr_log={stderr_log_path}\n{failure_text}"
                )
                _append_text(stderr_log_path, diagnostic)
                print(diagnostic, file=sys.stderr, end="", flush=True)
    _write_summary(output_dir, rows)
    audit = {
        "status": "FAIL" if failures else "PASS",
        "stage": "stage3_benchmark_evaluation",
        "model_label": MODEL_LABEL,
        "model_path": str(config.model_path),
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "started_at": started_at,
        "finished_at": utc_now(),
        "command": sys.argv,
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "packages": versions,
        "model": model_info,
        "benchmark_root": str(config.benchmark_root.expanduser().resolve()),
        "benchmark_manifest": benchmark,
        "protocol": protocol,
        "configuration": asdict(config),
        "log_root": str(log_root),
        "task_log_paths": task_log_paths,
        "gpu": _gpu_info(config.device),
        "recursive_runtime_audit": runtime_audit,
        "tasks": list(config.tasks),
        "task_results": task_results,
        "summary_rows": rows,
        "sample_counts": {
            task: _result_sample_counts(payload.get("raw_lm_eval", {}))
            for task, payload in task_results.items()
        },
        "skipped_count": len(config.tasks) if config.validation_only else 0,
        "failed_count": len(failures),
        "failures": failures,
        "output_dir": str(output_dir),
        "formal_eval_executed": not config.validation_only,
    }
    write_json(output_dir / "audit_report.json", audit)
    write_json(output_dir / "run_config.json", {"configuration": asdict(config), "protocol": protocol})
    if failures:
        raise RuntimeError(f"Stage 3 5-10-5 recursive task failures: {sorted(failures)}")
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
    return EvaluationConfig(
        model_path=args.model_path,
        benchmark_root=args.benchmark_root,
        output_dir=args.output_dir,
        tasks=tuple(args.tasks),
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        seed=args.seed,
        limit=args.limit,
        log_samples=not args.no_log_samples,
        validation_only=args.validation_only,
        smoke=args.smoke,
        cache_dir=args.cache_dir,
        log_root=args.log_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config: EvaluationConfig | None = None
    output_was_fresh = False
    try:
        config = parse_args(argv)
        try:
            candidate = ensure_external_path(config.output_dir, label="output")
            output_was_fresh = not candidate.exists() or (
                candidate.is_dir() and not any(candidate.iterdir())
            )
        except Exception:
            output_was_fresh = False
        audit = run_evaluation_5_10_5_recursive(config)
        print(json.dumps(json_safe(audit), ensure_ascii=False, indent=2), flush=True)
        print(f"[result] status={audit['status']} output={config.output_dir}", flush=True)
        return 0
    except Exception:
        failure_text = traceback.format_exc()
        print("[result] status=FAIL", file=sys.stderr, flush=True)
        print(failure_text, file=sys.stderr, end="", flush=True)
        if config is not None:
            try:
                log_root = ensure_log_root(config.log_root)
                _append_text(
                    _runtime_log_path(log_root, config),
                    "\n=== Stage 3 5-10-5 recursive process failure ===\n"
                    f"model={config.model_path}\noutput={config.output_dir}\n{failure_text}",
                )
            except Exception:
                pass
        if config is not None and output_was_fresh:
            try:
                failure_dir = ensure_external_path(config.output_dir, label="output")
                failure_dir.mkdir(parents=True, exist_ok=True)
                failure_report = failure_dir / "audit_report.json"
                if not failure_report.exists():
                    write_json(
                        failure_report,
                        {
                            "status": "FAIL",
                            "stage": "stage3_benchmark_evaluation",
                            "model_label": MODEL_LABEL,
                            "model_path": str(config.model_path),
                            "architecture_contract": ARCHITECTURE_CONTRACT,
                            "started_at": utc_now(),
                            "finished_at": utc_now(),
                            "command": sys.argv,
                            "configuration": asdict(config),
                            "failure": failure_text,
                            "formal_eval_executed": False,
                        },
                    )
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
