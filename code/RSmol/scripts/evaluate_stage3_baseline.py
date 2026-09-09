#!/usr/bin/env python3
"""Offline Stage 3 evaluation for the Falcon-H1 and TinyBrainBot baselines.

The benchmark protocol is shared with the established Stage 3 evaluator:
``hellaswag,mmlu,gsm8k,arc_easy,arc_challenge`` using the installed official
lm-eval task YAMLs and the local parquet snapshot.  Only the dataset source is
overlaid; prompts, few-shot splits, document processing, answer extraction,
and metrics remain owned by lm-eval.

This entry point is deliberately independent of the project recursive/MeSH
Auto registries.  Baselines are loaded through the stock Transformers
``AutoConfig``, ``AutoTokenizer``, and ``AutoModelForCausalLM`` mappings with
``local_files_only=True``.  The preflight audit rejects a model whose local
configuration does not match the selected baseline contract, and rejects a
Transformers runtime too old to provide the requested architecture.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


# Reuse the benchmark/layout/task-overlay helpers from the original Stage 3
# implementation without importing or invoking any recursive model registry.
# The helper module has no import-time registration side effects.
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT / "scripts"))

from evaluate_stage3 import (  # noqa: E402
    EXPECTED_MMLU_SUBJECTS,
    OFFLINE_ENVIRONMENT,
    STAGE3_TASKS,
    _flatten_result_rows,
    _result_sample_counts,
    _runtime_log_path,
    _task_log_path,
    ensure_external_output,
    ensure_external_path,
    ensure_log_root,
    git_commit,
    json_safe,
    package_version,
    prepare_local_task_overlays,
    set_offline_environment,
    validate_benchmark_layout,
    write_json,
)


REMOTE_MODEL_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models")
DEFAULT_BENCHMARK_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/eval_datasets"
)
DEFAULT_LOG_ROOT = SCRIPT_ROOT / "log"
DEFAULT_MODEL_PATHS = {
    "falcon90m": REMOTE_MODEL_ROOT / "falcon90M",
    "tinybrainbot100m": REMOTE_MODEL_ROOT / "Tinybrainbot100M",
}
DEFAULT_OUTPUT_ROOTS = {
    "falcon90m": Path(
        "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage3-baseline-falcon90m"
    ),
    "tinybrainbot100m": Path(
        "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage3-baseline-tinybrainbot100m"
    ),
}

EXPECTED_LM_EVAL_VERSION = "0.4.12"
EXPECTED_DATASETS_VERSION = "3.6.0"
EXPECTED_DEVICE = "cuda:0"

# The Falcon values are taken from the model's published config.  TinyBrainBot
# values are its model-card architecture contract.  Keeping these checks
# explicit prevents a wrong checkpoint or a silent fallback to a different
# AutoModel implementation from producing a misleading baseline number.
MODEL_CONTRACTS: dict[str, dict[str, Any]] = {
    "falcon90m": {
        "display_name": "tiiuae/Falcon-H1-Tiny-90M-Base",
        "model_type": "falcon_h1",
        "architecture": "FalconH1ForCausalLM",
        "minimum_transformers": "4.57.0",
        "dimensions": {
            "hidden_size": 512,
            "intermediate_size": 768,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vocab_size": 32768,
            "max_position_embeddings": 262144,
            "tie_word_embeddings": True,
        },
        "weight_format": "safetensors_or_pytorch_bin",
    },
    "tinybrainbot100m": {
        "display_name": "nkthebass/tinybrainbot-100m-v3-base",
        "model_type": "llama",
        "architecture": "LlamaForCausalLM",
        # Llama is available in the pinned evaluation stack; the declared
        # config version, when present, is also checked at runtime.
        "minimum_transformers": "4.31.0",
        "dimensions": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "intermediate_size": 2048,
            "max_position_embeddings": 1024,
            "vocab_size": 32000,
            "tie_word_embeddings": True,
        },
        "weight_format": "safetensors_or_pytorch_bin",
    },
}

_VERSION_RE = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?")
_WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin", "*.bin")
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
)


@dataclass
class BaselineEvaluationConfig:
    model_key: str
    model_path: Path
    benchmark_root: Path
    output_dir: Path
    tasks: tuple[str, ...] = STAGE3_TASKS
    device: str = EXPECTED_DEVICE
    dtype: str = "bfloat16"
    batch_size: int = 1
    seed: int = 0
    limit: int | None = None
    log_samples: bool = True
    validation_only: bool = False
    smoke: bool = False
    cache_dir: Path | None = None
    log_root: Path | None = None


def _version_tuple(value: str | None) -> tuple[int, int, int] | None:
    if not value or value.startswith("<"):
        return None
    match = _VERSION_RE.match(str(value).strip())
    if match is None:
        return None
    return tuple(int(part or 0) for part in match.groups())


def _version_at_least(actual: str | None, required: str) -> bool:
    parsed_actual = _version_tuple(actual)
    parsed_required = _version_tuple(required)
    return parsed_actual is not None and parsed_required is not None and parsed_actual >= parsed_required


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "baseline"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _path_is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _model_weight_manifest(model_dir: Path) -> list[str]:
    files: set[Path] = set()
    for pattern in _WEIGHT_PATTERNS:
        files.update(path for path in model_dir.glob(pattern) if path.is_file())
    return sorted(path.name for path in files)


def _tokenizer_manifest(model_dir: Path) -> list[str]:
    return [name for name in _TOKENIZER_FILES if (model_dir / name).is_file()]


def _recursive_markers(config: Mapping[str, Any]) -> list[str]:
    markers = []
    for key in config:
        lowered = str(key).lower()
        if "recursive" in lowered or "mesh" in lowered:
            markers.append(str(key))
    architectures = config.get("architectures", [])
    if isinstance(architectures, (list, tuple)):
        for architecture in architectures:
            if "recursive" in str(architecture).lower() or "mesh" in str(architecture).lower():
                markers.append(str(architecture))
    return sorted(set(markers))


def _config_contract_audit(config: Mapping[str, Any], model_key: str) -> dict[str, Any]:
    if model_key not in MODEL_CONTRACTS:
        raise ValueError(f"Unknown baseline model key {model_key!r}")
    contract = MODEL_CONTRACTS[model_key]
    model_type = config.get("model_type")
    architectures_raw = config.get("architectures", [])
    architectures = [str(item) for item in architectures_raw] if isinstance(architectures_raw, (list, tuple)) else []
    checks: dict[str, Any] = {
        "model_type": model_type == contract["model_type"],
        "architecture": contract["architecture"] in architectures,
        "no_recursive_or_mesh_markers": not _recursive_markers(config),
    }
    if model_type != contract["model_type"]:
        raise ValueError(
            f"{model_key} model_type mismatch: expected {contract['model_type']!r}, got {model_type!r}"
        )
    if contract["architecture"] not in architectures:
        raise ValueError(
            f"{model_key} architecture mismatch: expected {contract['architecture']!r}, got {architectures!r}"
        )
    markers = _recursive_markers(config)
    if markers:
        raise ValueError(
            f"{model_key} baseline contains recursive/MeSH markers {markers}; refusing registry fallback"
        )

    dimension_checks: dict[str, Any] = {}
    dimension_mismatches: dict[str, dict[str, Any]] = {}
    for field, expected in contract["dimensions"].items():
        actual = config.get(field)
        equal = actual == expected
        dimension_checks[field] = equal
        if not equal:
            dimension_mismatches[field] = {"expected": expected, "actual": actual}
    if dimension_mismatches:
        raise ValueError(
            f"{model_key} config dimensions do not match the baseline contract: {dimension_mismatches}"
        )
    checks.update(dimension_checks)
    return {
        "model_key": model_key,
        "display_name": contract["display_name"],
        "expected_model_type": contract["model_type"],
        "expected_architecture": contract["architecture"],
        "minimum_transformers": contract["minimum_transformers"],
        "model_type": model_type,
        "architectures": architectures,
        "checks": checks,
        "dimensions": {
            field: {"expected": expected, "actual": config.get(field)}
            for field, expected in contract["dimensions"].items()
        },
        "declared_transformers_version": config.get("transformers_version"),
        "recursive_or_mesh_markers": markers,
    }


def inspect_model_artifacts(path: Path, model_key: str) -> dict[str, Any]:
    """Audit local HF artifacts and the selected baseline's JSON contract.

    A GGUF file is recorded as an ignored artifact but is never accepted as a
    model weight.  This is important for TinyBrainBot, whose directory also
    contains an F16 GGUF alongside the HF safetensors checkpoint.
    """

    if model_key not in MODEL_CONTRACTS:
        raise ValueError(f"Unknown baseline model key {model_key!r}")
    model_dir = ensure_external_path(path, label=f"{model_key} model")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"{model_key} model directory does not exist: {model_dir}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{model_key} model is missing config.json: {model_dir}")
    config = _read_json(config_path)
    contract_audit = _config_contract_audit(config, model_key)
    weight_files = _model_weight_manifest(model_dir)
    ignored_gguf = sorted(path.name for path in model_dir.glob("*.gguf") if path.is_file())
    if not weight_files:
        if ignored_gguf:
            raise FileNotFoundError(
                f"{model_key} has GGUF only ({ignored_gguf}); no HF safetensors/bin weights are available"
            )
        raise FileNotFoundError(f"{model_key} has no HF safetensors/bin weight artifact: {model_dir}")
    tokenizer_files = _tokenizer_manifest(model_dir)
    if "tokenizer.json" not in tokenizer_files and "tokenizer.model" not in tokenizer_files:
        raise FileNotFoundError(
            f"{model_key} needs tokenizer.json or tokenizer.model for offline loading: {model_dir}"
        )
    if "tokenizer_config.json" not in tokenizer_files:
        raise FileNotFoundError(f"{model_key} is missing tokenizer_config.json: {model_dir}")

    tokenizer_vocab_size: int | None = None
    tokenizer_json = model_dir / "tokenizer.json"
    if tokenizer_json.is_file():
        try:
            payload = _read_json(tokenizer_json)
            vocab = payload.get("model", {}).get("vocab", {})
            if isinstance(vocab, dict):
                tokenizer_vocab_size = len(vocab)
        except ValueError:
            # AutoTokenizer will provide a more detailed parse error during
            # runtime; retain the artifact audit context here.
            tokenizer_vocab_size = None
    config_vocab_size = config.get("vocab_size")
    vocab_compatible = (
        tokenizer_vocab_size is None
        or config_vocab_size is None
        or int(config_vocab_size) == tokenizer_vocab_size
    )
    if not vocab_compatible:
        raise ValueError(
            f"{model_key} tokenizer/model vocab mismatch: config={config_vocab_size} tokenizer={tokenizer_vocab_size}"
        )
    return {
        "label": model_key,
        "display_name": MODEL_CONTRACTS[model_key]["display_name"],
        "path": str(model_dir),
        "config": config,
        "config_contract": contract_audit,
        "model_files": weight_files,
        "weight_format": "HF safetensors/bin",
        "ignored_gguf_files": ignored_gguf,
        "tokenizer_config": _read_json(model_dir / "tokenizer_config.json"),
        "tokenizer_files": tokenizer_files,
        "config_vocab_size": config_vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "vocab_compatible": vocab_compatible,
    }


def inspect_runtime_versions(model_info: Mapping[str, Any]) -> dict[str, str]:
    """Check the pinned harness and the model's Transformers requirement."""

    actual = {
        "lm_eval_package": package_version("lm-eval"),
        "lm_eval": package_version("lm_eval"),
        "transformers": package_version("transformers"),
        "datasets": package_version("datasets"),
        "torch": package_version("torch"),
    }
    if actual["lm_eval_package"].startswith("<") and not actual["lm_eval"].startswith("<"):
        actual["lm_eval_package"] = actual["lm_eval"]
    # Preserve the probe even when a later compatibility check fails so the
    # machine-readable failure report identifies the offending runtime.
    if isinstance(model_info, dict):
        model_info["runtime_package_probe"] = actual
    required_lm = EXPECTED_LM_EVAL_VERSION
    actual_lm = actual["lm_eval_package"]
    if actual_lm != required_lm:
        raise RuntimeError(
            f"lm-eval version mismatch: required {required_lm}, found {actual_lm}; refusing protocol drift"
        )
    if actual["datasets"] != EXPECTED_DATASETS_VERSION:
        raise RuntimeError(
            f"datasets version mismatch: required {EXPECTED_DATASETS_VERSION}, found {actual['datasets']}; refusing protocol drift"
        )

    contract = MODEL_CONTRACTS[str(model_info["label"])]
    required_transformers = contract["minimum_transformers"]
    declared = model_info.get("config", {}).get("transformers_version")
    if _version_tuple(str(declared) if declared is not None else None) is not None and not _version_at_least(
        required_transformers, str(declared)
    ):
        required_transformers = str(declared)
    if not _version_at_least(actual["transformers"], required_transformers):
        raise RuntimeError(
            f"{model_info['label']} requires Transformers >= {required_transformers} for its "
            f"{model_info['config_contract']['expected_architecture']} architecture; found {actual['transformers']}"
        )
    if actual["torch"].startswith("<"):
        raise RuntimeError("PyTorch is unavailable in the rsmol evaluation environment")
    return actual


def load_tokenizer_runtime_metadata(model_info: dict[str, Any]) -> dict[str, Any]:
    """Load the tokenizer locally and verify its runtime vocabulary."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_info["path"],
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    metadata = {
        "class": type(tokenizer).__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "tokenizer_length": int(len(tokenizer)),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
    }
    config_vocab_size = model_info.get("config_vocab_size")
    if config_vocab_size is not None and metadata["vocab_size"] != int(config_vocab_size):
        raise ValueError(
            f"{model_info['label']} runtime tokenizer/model vocab mismatch: "
            f"config={config_vocab_size} tokenizer={metadata['vocab_size']}"
        )
    model_info["runtime_tokenizer"] = metadata
    return metadata


def _parameter_metadata(model: Any) -> dict[str, Any]:
    parameters = list(model.parameters())
    total = sum(int(parameter.numel()) for parameter in parameters)
    trainable = sum(int(parameter.numel()) for parameter in parameters if parameter.requires_grad)
    storage_keys: set[tuple[int, int]] = set()
    unique_storage_bytes = 0
    for parameter in parameters:
        storage = getattr(parameter, "untyped_storage", lambda: None)()
        if storage is None:
            continue
        try:
            key = (int(storage.data_ptr()), int(storage.nbytes()))
            if key not in storage_keys:
                storage_keys.add(key)
                unique_storage_bytes += key[1]
        except Exception:
            # Storage accounting is diagnostic only; logical parameter count
            # remains authoritative if a backend does not expose nbytes.
            continue
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
        "unique_parameter_storage_count": len(storage_keys),
        "unique_parameter_storage_bytes": unique_storage_bytes,
    }


def _architecture_metadata(model: Any, config: Mapping[str, Any], model_key: str) -> dict[str, Any]:
    fields = (
        "model_type",
        "architectures",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "vocab_size",
        "tie_word_embeddings",
        "torch_dtype",
        "transformers_version",
    )
    contract = MODEL_CONTRACTS[model_key]
    expected_class = contract["architecture"]
    actual_class = type(model).__name__
    if actual_class != expected_class:
        raise TypeError(
            f"{model_key} AutoModel resolved {actual_class!r}; expected {expected_class!r}. "
            "No fallback architecture is permitted."
        )
    result = {
        "model_class": actual_class,
        "config_class": type(getattr(model, "config", None)).__name__,
        "selected_config": {field: config.get(field) for field in fields},
        "expected_architecture": expected_class,
        "architecture_match": True,
    }
    return result


def load_and_audit_model(
    model_info: dict[str, Any], *, device: str, dtype: str
) -> dict[str, Any]:
    """Load one baseline via stock Auto mappings and run a CUDA forward test."""

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if device != EXPECTED_DEVICE:
        raise RuntimeError(
            f"Baseline runtime audit requires one visible GPU at {EXPECTED_DEVICE}; got {device!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Baseline Stage 3 formal evaluation requires a submitted CUDA job")
    torch_device = torch.device(device)
    if torch_device.index not in (None, 0):
        raise RuntimeError(f"Baseline runtime audit must use cuda:0, got {device!r}")

    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported torch dtype {dtype!r}")
    path = model_info["path"]
    model_key = str(model_info["label"])
    expected_class = MODEL_CONTRACTS[model_key]["architecture"]
    # AutoConfig is an explicit support probe.  An old Transformers build must
    # fail here with a model-specific message rather than silently selecting a
    # generic/incorrect model implementation.
    try:
        auto_config = AutoConfig.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
    except Exception as exc:
        raise RuntimeError(
            f"{model_key} Transformers AutoConfig cannot resolve model_type "
            f"{model_info['config'].get('model_type')!r} for {expected_class}: {exc}"
        ) from exc
    if type(auto_config).__name__.lower().replace("config", "") not in {
        str(model_info["config"].get("model_type", "")).lower(),
        "falconh1",
        "llama",
    }:
        raise TypeError(
            f"{model_key} AutoConfig resolved unexpected class {type(auto_config).__name__!r}; "
            f"expected a config for {model_info['config'].get('model_type')!r}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, use_fast=True, trust_remote_code=False
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{model_key} AutoModelForCausalLM could not load the declared "
            f"{expected_class} architecture from local HF weights: {exc}"
        ) from exc
    try:
        architecture = _architecture_metadata(model, model_info["config"], model_key)
        model.to(torch_device)
        model.eval()
        encoded = tokenizer(
            "The quick brown fox jumps over the lazy dog.",
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(torch_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch_device)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise RuntimeError(f"Unexpected tokenizer sanity shape: {tuple(input_ids.shape)}")
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError("Baseline forward output has no logits")
        expected_vocab = int(model_info["config"].get("vocab_size", 0))
        if tuple(logits.shape[:2]) != tuple(input_ids.shape) or logits.shape[-1] != expected_vocab:
            raise RuntimeError(
                "Baseline forward shape mismatch: "
                f"input={tuple(input_ids.shape)} logits={tuple(logits.shape)} expected_vocab={expected_vocab}"
            )
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("Baseline forward produced non-finite logits")
        runtime = {
            "status": "PASS",
            "device": str(torch_device),
            "input_shape": list(input_ids.shape),
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
            "model_parameter_dtype": str(next(model.parameters()).dtype),
            "architecture": architecture,
            "parameters": _parameter_metadata(model),
            "forward_trace": "one tokenizer -> model -> logits pass",
        }
        return runtime
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _write_summary(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_json(output_dir / "summary.json", {"rows": list(rows)})
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("requested_task", "task", "metric", "value")
        )
        writer.writeheader()
        writer.writerows(rows)


def _gpu_info(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {"requested_device": device}
    try:
        import torch

        info.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "device_count": int(torch.cuda.device_count()),
            }
        )
        if torch.cuda.is_available():
            index = torch.device(device).index or torch.cuda.current_device()
            info["device_index"] = int(index)
            info["device_name"] = torch.cuda.get_device_name(index)
            info["capability"] = list(torch.cuda.get_device_capability(index))
    except ImportError:
        info["torch"] = "unavailable"
    return info


def _run_single_task(
    config: BaselineEvaluationConfig,
    task: str,
    task_dir: Path,
    overlay_dir: Path,
    stderr_log_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one official lm-eval task, with no project model registration."""

    set_offline_environment(task_dir / ".hf-cache")
    model_args = ",".join(
        (
            f"pretrained={config.model_path}",
            f"dtype={config.dtype}",
            "local_files_only=True",
            "trust_remote_code=False",
        )
    )
    started = time.time()
    captured_stderr = io.StringIO()
    result: Mapping[str, Any] | None = None
    print(
        f"[stage3-baseline][model={config.model_key}][task={task}] starting lm_eval; "
        f"overlay={overlay_dir} stderr_log={stderr_log_path}",
        flush=True,
    )
    try:
        with contextlib.redirect_stderr(captured_stderr):
            from lm_eval import evaluator
            from lm_eval.tasks import TaskManager

            # lm-eval 0.4.12 probes git while constructing its result payload.
            # That probe is unrelated to scoring and is disabled exactly as in
            # the established Stage 3 protocol.
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
                num_fewshot=5 if task == "mmlu" else None,
                random_seed=config.seed,
                numpy_random_seed=config.seed,
                torch_random_seed=config.seed,
                fewshot_random_seed=config.seed,
            )
        if task == "mmlu":
            result_tasks = result.get("results", {}) if isinstance(result, Mapping) else {}
            expected_subject_tasks = {
                f"mmlu_{subject}"
                for subject in _discover_mmlu_subjects(config.benchmark_root)
            }
            missing_subject_tasks = sorted(expected_subject_tasks - set(result_tasks))
            if missing_subject_tasks:
                raise RuntimeError(
                    "lm_eval MMLU result omitted subject rows; refusing aggregate-only report: "
                    f"missing_count={len(missing_subject_tasks)} first={missing_subject_tasks[:5]}"
                )
    finally:
        stderr_text = captured_stderr.getvalue()
        try:
            stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_log_path.write_text(stderr_text, encoding="utf-8")
        except Exception as log_error:
            print(
                f"[stage3-baseline][task={task}][WARN] could not write stderr log "
                f"{stderr_log_path}: {log_error!r}",
                file=sys.stderr,
                flush=True,
            )
        if stderr_text:
            print(stderr_text, file=sys.stderr, end="", flush=True)
    if result is None:
        raise RuntimeError(f"lm_eval returned no result for task {task!r}")
    result_payload = {
        "model_key": config.model_key,
        "requested_task": task,
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "task_config": str(overlay_dir),
        "stderr_log": str(stderr_log_path),
        "raw_lm_eval": result,
    }
    write_json(task_dir / "lm_eval_results.json", result_payload)
    if config.log_samples and isinstance(result, Mapping) and "samples" in result:
        write_json(task_dir / "log_samples.json", {"samples": result["samples"]})
    return result_payload, _flatten_result_rows(task, result)


def _discover_mmlu_subjects(benchmark_root: Path) -> tuple[str, ...]:
    """Use the shared helper lazily to keep import-time dependencies small."""

    from evaluate_stage3 import discover_mmlu_subjects

    return discover_mmlu_subjects(benchmark_root)


def _base_audit(config: BaselineEvaluationConfig, started_at: str) -> dict[str, Any]:
    return {
        "status": "FAIL",
        "stage": "stage3_baseline_benchmark_evaluation",
        "model_key": config.model_key,
        "model_display_name": MODEL_CONTRACTS.get(config.model_key, {}).get("display_name"),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python": sys.version,
        "configuration": asdict(config),
        "model": None,
        "packages": None,
        "benchmark_root": str(config.benchmark_root.expanduser().resolve()),
        "benchmark_manifest": None,
        "protocol": None,
        "log_root": str(config.log_root) if config.log_root else str(DEFAULT_LOG_ROOT),
        "offline_environment": {
            name: os.environ.get(name) for name in OFFLINE_ENVIRONMENT
        },
        "gpu": _gpu_info(config.device),
        "runtime_audit": None,
        "tasks": list(config.tasks),
        "task_results": {},
        "summary_rows": [],
        "sample_counts": {},
        "skipped_count": len(config.tasks) if config.validation_only else 0,
        "failed_count": 0,
        "failures": {},
        "output_dir": str(config.output_dir),
        "stage4_status": "paused",
        "formal_eval_executed": not config.validation_only,
    }


def run_evaluation(config: BaselineEvaluationConfig) -> dict[str, Any]:
    """Run preflight, runtime audit, and the requested Stage 3 tasks."""

    started_at = datetime.now(timezone.utc).isoformat()
    output_dir = ensure_external_output(config.output_dir)
    log_root = ensure_log_root(config.log_root)
    config.output_dir = output_dir
    config.log_root = log_root
    set_offline_environment(config.cache_dir or (output_dir / ".hf-cache"))
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = _base_audit(config, started_at)
    model_info: dict[str, Any] | None = None
    versions: dict[str, str] | None = None
    protocol: dict[str, Any] | None = None
    benchmark: dict[str, Any] | None = None
    try:
        model_info = inspect_model_artifacts(config.model_path, config.model_key)
        audit["model"] = model_info
        versions = inspect_runtime_versions(model_info)
        audit["packages"] = versions
        load_tokenizer_runtime_metadata(model_info)
        benchmark = validate_benchmark_layout(config.benchmark_root)
        audit["benchmark_manifest"] = benchmark
        probe_dir = output_dir / ".task-config-probe"
        probe_dir.mkdir(parents=True, exist_ok=True)
        protocol = prepare_local_task_overlays(
            config.benchmark_root, probe_dir, config.tasks
        )
        audit["protocol"] = protocol
        if not config.validation_only:
            if config.device != EXPECTED_DEVICE:
                raise RuntimeError(
                    f"Formal baseline evaluation requires device {EXPECTED_DEVICE}; got {config.device!r}"
                )
            audit["runtime_audit"] = load_and_audit_model(
                model_info, device=config.device, dtype=config.dtype
            )
    except Exception:
        if versions is None and model_info is not None:
            audit["packages"] = model_info.get("runtime_package_probe")
        audit["failures"] = {"preflight_or_runtime": traceback.format_exc()}
        audit["failed_count"] = 1
        audit["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir / "audit_report.json", audit)
        write_json(
            output_dir / "run_config.json",
            {"configuration": asdict(config), "protocol": protocol},
        )
        _write_summary(output_dir, [])
        raise

    task_results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    task_log_paths = {
        task: str(_task_log_path(log_root, config, task)) for task in config.tasks
    }
    print(
        f"[stage3-baseline] model_key={config.model_key} model={config.model_path} "
        f"output={output_dir} log_root={log_root} tasks={','.join(config.tasks)}",
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
                payload, task_rows = _run_single_task(
                    config, task, task_dir, overlay_dir, stderr_log_path
                )
                task_results[task] = payload
                rows.extend(task_rows)
            except Exception:
                failure_text = traceback.format_exc()
                failures[task] = failure_text
                diagnostic = (
                    f"\n=== Stage 3 baseline task failure: {task} ===\n"
                    f"model_key={config.model_key}\n"
                    f"model={config.model_path}\n"
                    f"output={output_dir / task}\n"
                    f"overlay={overlay_dir}\n"
                    f"stderr_log={stderr_log_path}\n"
                    f"{failure_text}"
                )
                try:
                    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with stderr_log_path.open("a", encoding="utf-8") as handle:
                        handle.write(diagnostic)
                except Exception as log_error:
                    print(
                        f"[stage3-baseline][task={task}][WARN] failed to append diagnostic log "
                        f"{stderr_log_path}: {log_error!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                print(diagnostic, file=sys.stderr, end="", flush=True)

    _write_summary(output_dir, rows)
    audit.update(
        {
            "status": "FAIL" if failures else "PASS",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "task_log_paths": task_log_paths,
            "offline_environment": {
                name: os.environ.get(name) for name in OFFLINE_ENVIRONMENT
            },
            "gpu": _gpu_info(config.device),
            "task_results": task_results,
            "summary_rows": rows,
            "sample_counts": {
                task: _result_sample_counts(payload.get("raw_lm_eval", {}))
                for task, payload in task_results.items()
            },
            "failed_count": len(failures),
            "failures": failures,
        }
    )
    write_json(output_dir / "audit_report.json", audit)
    write_json(
        output_dir / "run_config.json",
        {"configuration": asdict(config), "protocol": protocol},
    )
    if failures:
        summaries = {
            task: next(
                (line.strip() for line in reversed(trace.splitlines()) if line.strip()),
                "unknown failure",
            )
            for task, trace in failures.items()
        }
        raise RuntimeError(
            f"Stage 3 baseline task failures: {summaries}; detailed tracebacks saved under {log_root}"
        )
    return audit


def _default_output_dir(model_key: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOTS[model_key].with_name(
        f"{DEFAULT_OUTPUT_ROOTS[model_key].name}-{timestamp}-{os.getpid()}"
    )


def parse_args(argv: Sequence[str] | None = None) -> BaselineEvaluationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-key",
        choices=tuple(MODEL_CONTRACTS),
        default=os.environ.get("RSMOL_BASELINE_MODEL_KEY", "falcon90m"),
        help="Baseline contract to audit and evaluate.",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=STAGE3_TASKS,
        default=list(STAGE3_TASKS),
    )
    parser.add_argument("--device", default=os.environ.get("RSMOL_BASELINE_DEVICE", EXPECTED_DEVICE))
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default=os.environ.get("RSMOL_BASELINE_DTYPE", "bfloat16"),
    )
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("RSMOL_BASELINE_BATCH_SIZE", "1")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("RSMOL_BASELINE_SEED", "0")))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--no-log-samples", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=None)
    args = parser.parse_args(argv)
    model_key = str(args.model_key)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when supplied")
    if args.smoke and args.limit is None:
        args.limit = 2
    model_path = args.model_path or DEFAULT_MODEL_PATHS[model_key]
    output_dir = args.output_dir or _default_output_dir(model_key)
    log_root = args.log_root or Path(
        os.environ.get("RSMOL_BASELINE_LOG_ROOT", str(DEFAULT_LOG_ROOT))
    )
    return BaselineEvaluationConfig(
        model_key=model_key,
        model_path=model_path,
        benchmark_root=args.benchmark_root,
        output_dir=output_dir,
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
        log_root=log_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config: BaselineEvaluationConfig | None = None
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
                log_root = ensure_log_root(config.log_root)
                runtime_log = _runtime_log_path(log_root, config)
                runtime_log.parent.mkdir(parents=True, exist_ok=True)
                with runtime_log.open("a", encoding="utf-8") as handle:
                    handle.write(
                        "\n=== Stage 3 baseline process failure ===\n"
                        f"model_key={config.model_key}\n"
                        f"model={config.model_path}\n"
                        f"output={config.output_dir}\n"
                        f"{failure_text}"
                    )
                print(
                    f"[stage3-baseline][process][FAIL] traceback saved to {runtime_log}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as log_error:
                print(
                    f"[stage3-baseline][process][WARN] could not write process log: {log_error!r}",
                    file=sys.stderr,
                    flush=True,
                )
        if config is not None and output_was_fresh:
            try:
                failure_dir = ensure_external_path(config.output_dir, label="output")
                failure_dir.mkdir(parents=True, exist_ok=True)
                failure_report = failure_dir / "audit_report.json"
                if not failure_report.exists():
                    audit = _base_audit(config, datetime.now(timezone.utc).isoformat())
                    audit["failures"] = {"process": failure_text}
                    audit["failed_count"] = 1
                    write_json(failure_report, audit)
                    write_json(
                        failure_dir / "run_config.json",
                        {"configuration": asdict(config), "protocol": None},
                    )
                    _write_summary(failure_dir, [])
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
