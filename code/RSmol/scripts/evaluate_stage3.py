#!/usr/bin/env python3
"""Offline Stage 3 benchmark evaluation for the two SmolLM2 checkpoints.

This entry point deliberately keeps benchmark semantics in lm-evaluation-
harness.  It discovers the installed 0.4.12 task YAMLs, copies them to an
external include directory, and changes only the dataset source to the local
parquet snapshot.  The task's prompt, document conversion, few-shot split,
answer extraction, and metrics therefore remain owned by the official task.

The script has a validation-only mode.  Validation checks model artifacts,
the exact local benchmark layout, installed task configs, and the protocol
contract before any model inference is attempted.  Formal runs must be made
through ``run_stage3_eval_5090.sh`` on the remote GPU host.
"""

from __future__ import annotations

import argparse
import copy
import csv
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_ROOT.parents[1]
REMOTE_CHECKOUT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM")
DEFAULT_LOG_ROOT = SCRIPT_ROOT / "log"

DEFAULT_ORIGINAL_MODEL = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2")
DEFAULT_RECURSIVE_MODEL = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-15R")
DEFAULT_BENCHMARK_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/eval_datasets"
)
STAGE3_TASKS = ("hellaswag", "mmlu", "gsm8k", "arc_easy", "arc_challenge")
EXPECTED_LM_EVAL_VERSION = "0.4.12"
EXPECTED_TRANSFORMERS_VERSION = "4.54.1"
EXPECTED_DATASETS_VERSION = "3.6.0"
EXPECTED_RECURSIVE_LOGICAL_LAYERS = 30
EXPECTED_RECURSIVE_PHYSICAL_LAYERS = 15
EXPECTED_RECURSIVE_LOOPS = 2
EXPECTED_MMLU_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)

OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

# Keys accepted by lm_eval task configs that may carry a ``!function`` tag.
# Keep this allow-list narrow: e.g. ``filter.function`` is a filter name, not
# a Python function reference, and must not be serialized as ``!function``.
_FUNCTION_REFERENCE_KEYS = frozenset(
    {
        "process_docs",
        "process_results",
        "doc_to_text",
        "doc_to_target",
        "doc_to_choice",
        "doc_to_decontamination_query",
        "doc_to_image",
        "doc_to_audio",
        "aggregation",
        "custom_dataset",
        "class",
    }
)
_RELATIVE_FUNCTION_REF = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$"
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/].+")


def _is_function_ref_string(value: str) -> bool:
    """Recognize the strings produced by lm_eval's function YAML loader.

    In 0.4.12, ``load_yaml(..., resolve_func=False)`` turns a tagged value
    such as ``utils.process_docs`` into ``<yaml-dir>/utils.process_docs``.
    The relative check also covers compatible loaders that preserve the
    original spelling, while rejecting templates/expressions containing
    whitespace or punctuation.
    """

    if not value:
        return False
    if value.startswith(("/", "\\")) or _WINDOWS_ABSOLUTE_PATH.match(value):
        # The official absolute form ends in ``module.function``.
        return "." in value and value.rsplit(".", 1)[-1].isidentifier()
    if any(character.isspace() for character in value):
        return False
    return bool(_RELATIVE_FUNCTION_REF.fullmatch(value))


def _normalise_function_reference(
    value: str, source_path: Path | None = None
) -> str:
    """Make official task functions importable from a copied overlay.

    The pinned loader materializes ``!function utils.process_docs`` as an
    absolute path when unresolved.  Keeping that path would make the second
    ``resolve_func=True`` pass sensitive to dotted interpreter directories
    such as ``python3.11``.  For functions shipped under ``lm_eval/tasks`` we
    therefore emit an importable dotted module path.  Other absolute paths are
    preserved verbatim, and unresolved relative refs remain relative when no
    source location is available.
    """

    if not _is_function_ref_string(value):
        return value
    module_name, separator, function_name = value.rpartition(".")
    if not separator or not function_name.isidentifier():
        return value
    module_path: Path | None = None
    is_absolute = value.startswith(("/", "\\")) or _WINDOWS_ABSOLUTE_PATH.match(value)
    if is_absolute:
        module_path = Path(module_name)
    elif source_path is not None:
        module_path = source_path.resolve().parent / Path(module_name.replace(".", "/"))
    if module_path is None:
        return value
    if module_path.suffix == ".py":
        candidate = module_path
    else:
        candidate = module_path if module_path.is_file() else module_path.with_suffix(".py")
    if not candidate.is_file():
        return value
    path_text = candidate.as_posix()
    marker = "/lm_eval/tasks/"
    marker_index = path_text.rfind(marker)
    if marker_index < 0:
        return value
    relative_module = path_text[marker_index + len(marker) :]
    if relative_module.endswith(".py"):
        relative_module = relative_module[:-3]
    dotted_module = "lm_eval.tasks." + relative_module.replace("/", ".")
    return f"{dotted_module}.{function_name}"


@dataclass
class EvaluationConfig:
    model_path: Path
    benchmark_root: Path
    output_dir: Path
    tasks: tuple[str, ...] = STAGE3_TASKS
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    batch_size: int = 1
    seed: int = 0
    limit: int | None = None
    log_samples: bool = True
    validation_only: bool = False
    smoke: bool = False
    cache_dir: Path | None = None
    reference_model_path: Path | None = None
    log_root: Path | None = None


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "<unavailable>"
    except Exception as exc:  # pragma: no cover - defensive metadata path
        return f"<unavailable:{type(exc).__name__}>"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - source archive/no git
        return f"<unavailable:{type(exc).__name__}>"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Convert common lm_eval/numpy values without importing numpy eagerly."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    for method in ("item", "tolist"):
        converter = getattr(value, method, None)
        if converter is not None:
            try:
                return json_safe(converter())
            except Exception:
                pass
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _path_is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def ensure_external_path(path: Path, *, label: str = "path") -> Path:
    candidate = path.expanduser().resolve()
    forbidden = (REPO_ROOT.resolve(), REMOTE_CHECKOUT.resolve())
    if any(_path_is_within(candidate, root) for root in forbidden):
        raise ValueError(
            f"Stage 3 refuses to use {label} inside a Git checkout: {candidate}. "
            "Use an external model/data/output directory."
        )
    return candidate


def ensure_log_root(path: Path | None = None) -> Path:
    """Create the diagnostic log root, allowing the requested checkout log.

    The benchmark outputs, model paths, and caches remain external.  The
    checkout's ``code/RSmol/log`` directory is an explicit user-requested
    exception for vc/task/runtime logs; arbitrary other checkout paths remain
    rejected.
    """

    candidate = (path or DEFAULT_LOG_ROOT).expanduser().resolve()
    permitted_checkout_log = DEFAULT_LOG_ROOT.resolve()
    if _path_is_within(candidate, REPO_ROOT) and not _path_is_within(
        candidate, permitted_checkout_log
    ):
        raise ValueError(
            "Stage 3 log root may be external, or under the explicit checkout "
            f"log exception {permitted_checkout_log}; got {candidate}"
        )
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _safe_log_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return component or "stage3"


def _task_log_path(log_root: Path, config: EvaluationConfig, task: str) -> Path:
    model_component = _safe_log_component(config.output_dir.name or config.model_path.name)
    return log_root / model_component / f"{_safe_log_component(task)}.stderr.log"


def _runtime_log_path(log_root: Path, config: EvaluationConfig) -> Path:
    model_component = _safe_log_component(config.output_dir.name or config.model_path.name)
    return log_root / model_component / "runtime.log"


def _append_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(content)


def ensure_external_output(path: Path) -> Path:
    """Return a fresh external output path; never overwrite non-empty output."""

    candidate = ensure_external_path(path, label="output")
    if candidate.exists() and not candidate.is_dir():
        raise FileExistsError(f"Output path is not a directory: {candidate}")
    if candidate.is_dir() and any(candidate.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty Stage 3 output directory: {candidate}"
        )
    return candidate


def set_offline_environment(cache_dir: Path | None = None) -> dict[str, str]:
    """Set all HF/datasets offline switches before importing lm_eval."""

    for name, value in OFFLINE_ENVIRONMENT.items():
        os.environ[name] = value
    if cache_dir is not None:
        cache = ensure_external_path(cache_dir, label="HF cache")
        cache.mkdir(parents=True, exist_ok=True)
        # These names are understood by the pinned HF stack and keep cache
        # writes out of both the checkout and shared model/data directories.
        for name in (
            "HF_HOME",
            "HF_DATASETS_CACHE",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
        ):
            os.environ[name] = str(cache)
    return {name: os.environ.get(name, "") for name in OFFLINE_ENVIRONMENT}


def _model_file_manifest(model_dir: Path) -> list[str]:
    patterns = ("*.safetensors", "pytorch_model*.bin", "*.bin")
    files: set[Path] = set()
    for pattern in patterns:
        files.update(model_dir.glob(pattern))
    return sorted(path.name for path in files if path.is_file())


def _tokenizer_file_manifest(model_dir: Path) -> list[str]:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
    )
    return [name for name in names if (model_dir / name).is_file()]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return value


def inspect_model_artifacts(path: Path, *, label: str = "model") -> dict[str, Any]:
    """Validate an external checkpoint before any Transformers loading."""

    model_dir = ensure_external_path(path, label=label)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"{label} is not an existing local directory: {model_dir}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{label} is missing config.json: {model_dir}")
    config = _read_json(config_path)
    model_files = _model_file_manifest(model_dir)
    tokenizer_files = _tokenizer_file_manifest(model_dir)
    if not model_files:
        raise FileNotFoundError(f"{label} has no local model weight artifact: {model_dir}")
    if not ("tokenizer.json" in tokenizer_files or "tokenizer.model" in tokenizer_files):
        raise FileNotFoundError(
            f"{label} needs tokenizer.json or tokenizer.model for offline loading: {model_dir}"
        )
    if "tokenizer_config.json" not in tokenizer_files:
        raise FileNotFoundError(f"{label} is missing tokenizer_config.json: {model_dir}")

    recursive_fields = {
        "recursive_layer_count",
        "recursive_loops",
        "recursive_mapping_policy",
    }
    is_recursive = any(field in config for field in recursive_fields) or any(
        "Recursive" in str(architecture) for architecture in config.get("architectures", [])
    )
    recursive_audit: dict[str, Any] = {"is_recursive": is_recursive}
    architectures = [str(item) for item in config.get("architectures", [])]
    if label == "model" and not is_recursive and architectures:
        if not any(item.endswith("LlamaForCausalLM") for item in architectures):
            raise ValueError(
                f"Original checkpoint must expose a LlamaForCausalLM architecture; got {architectures}"
            )
    if is_recursive:
        logical = int(config.get("num_hidden_layers", 0))
        physical = int(config.get("recursive_layer_count", 0))
        loops = int(config.get("recursive_loops", 0))
        mapping = list(config.get("recursive_source_layer_indices_0based", []))
        if (logical, physical, loops) != (
            EXPECTED_RECURSIVE_LOGICAL_LAYERS,
            EXPECTED_RECURSIVE_PHYSICAL_LAYERS,
            EXPECTED_RECURSIVE_LOOPS,
        ):
            raise ValueError(
                f"Recursive checkpoint must be logical=30 physical=15 loops=2; "
                f"got logical={logical} physical={physical} loops={loops}"
            )
        if len(mapping) != physical or len(set(mapping)) != physical:
            raise ValueError(f"Recursive source mapping must contain 15 unique indices: {mapping}")
        recursive_audit.update(
            {
                "architectures": architectures,
                "logical_layer_count": logical,
                "physical_layer_count": physical,
                "recursive_loops": loops,
                "source_mapping_0based": mapping,
                "depth_contract": logical == physical * loops,
            }
        )
    tokenizer_config = _read_json(model_dir / "tokenizer_config.json")
    tokenizer_vocab_size: int | None = None
    tokenizer_json = model_dir / "tokenizer.json"
    if tokenizer_json.is_file():
        try:
            tokenizer_payload = _read_json(tokenizer_json)
            vocab = tokenizer_payload.get("model", {}).get("vocab", {})
            if isinstance(vocab, dict):
                tokenizer_vocab_size = len(vocab)
        except ValueError:
            # Transformers will provide the detailed tokenizer error later;
            # retaining the artifact audit is more useful than masking it.
            tokenizer_vocab_size = None
    config_vocab_size = config.get("vocab_size")
    vocab_compatible = tokenizer_vocab_size is None or config_vocab_size is None or int(
        config_vocab_size
    ) == tokenizer_vocab_size
    if not vocab_compatible:
        raise ValueError(
            f"Tokenizer/model vocab mismatch for {model_dir}: "
            f"config={config_vocab_size} tokenizer={tokenizer_vocab_size}"
        )
    return {
        "label": label,
        "path": str(model_dir),
        "config": config,
        "tokenizer_config": tokenizer_config,
        "model_files": model_files,
        "tokenizer_files": tokenizer_files,
        "config_vocab_size": config_vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "vocab_compatible": vocab_compatible,
        "recursive_audit": recursive_audit,
    }


def compare_model_compatibility(
    original: Mapping[str, Any], recursive: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare architecture/tokenizer dimensions without copying checkpoints."""

    original_config = original.get("config", {})
    recursive_config = recursive.get("config", {})
    fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "hidden_act",
        "rms_norm_eps",
        "rope_theta",
        "attention_bias",
        "tie_word_embeddings",
    )
    comparisons = {
        field: {
            "original": original_config.get(field),
            "recursive": recursive_config.get(field),
            "equal": (
                original_config.get(field) == recursive_config.get(field)
                if original_config.get(field) is not None
                and recursive_config.get(field) is not None
                else "not_comparable"
            ),
        }
        for field in fields
    }
    mismatches = [field for field, item in comparisons.items() if item["equal"] is False]
    tokenizer_comparisons = {
        "vocab_size": {
            "original": original.get("tokenizer_vocab_size"),
            "recursive": recursive.get("tokenizer_vocab_size"),
            "equal": (
                original.get("tokenizer_vocab_size") == recursive.get("tokenizer_vocab_size")
                if original.get("tokenizer_vocab_size") is not None
                and recursive.get("tokenizer_vocab_size") is not None
                else "not_comparable"
            ),
        }
    }
    original_runtime_tokenizer = original.get("runtime_tokenizer", {})
    recursive_runtime_tokenizer = recursive.get("runtime_tokenizer", {})
    for field in (
        "vocab_size",
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "unk_token_id",
    ):
        tokenizer_comparisons[field] = {
            "original": original_runtime_tokenizer.get(field),
            "recursive": recursive_runtime_tokenizer.get(field),
            "equal": (
                original_runtime_tokenizer.get(field)
                == recursive_runtime_tokenizer.get(field)
                if original_runtime_tokenizer.get(field) is not None
                and recursive_runtime_tokenizer.get(field) is not None
                else "not_comparable"
            ),
        }
    for field, item in tokenizer_comparisons.items():
        if item["original"] is not None and item["recursive"] is not None and not item["equal"]:
            mismatches.append(f"tokenizer_{field}")
    if mismatches:
        raise ValueError(f"Original/recursive model compatibility mismatch: {mismatches}")
    return {"fields": comparisons, "tokenizer_fields": tokenizer_comparisons, "compatible": True}


def load_tokenizer_runtime_metadata(model_info: dict[str, Any]) -> dict[str, Any]:
    """Load the local tokenizer once and retain compatibility metadata."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_info["path"], local_files_only=True, use_fast=True
    )
    metadata = {
        "class": type(tokenizer).__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
    }
    config_vocab_size = model_info.get("config_vocab_size")
    if config_vocab_size is not None and metadata["vocab_size"] != int(config_vocab_size):
        raise ValueError(
            f"Runtime tokenizer/model vocab mismatch for {model_info['path']}: "
            f"config={config_vocab_size} tokenizer={metadata['vocab_size']}"
        )
    model_info["runtime_tokenizer"] = metadata
    return metadata


def _split_file(root: Path, *parts: str) -> Path:
    return root.joinpath(*parts)


def local_data_files(task: str, benchmark_root: Path) -> dict[str, str]:
    """Return local parquet split mapping consumed by ``datasets``.

    The returned mapping intentionally excludes MMLU ``auxiliary_train`` and
    the GSM8K ``socratic`` configuration.  ``dataset_path=parquet`` plus this
    mapping is the only dataset-source change made to official task YAMLs.
    """

    root = benchmark_root.expanduser().resolve()
    if task == "hellaswag":
        base = root / "Rowan_hellaswag" / "data"
        return {
            split: str(base / f"{split}-00000-of-00001.parquet")
            for split in ("train", "validation", "test")
        }
    if task.startswith("mmlu_"):
        subject = task.removeprefix("mmlu_")
        base = root / "cais_mmlu" / subject
        return {
            split: str(base / f"{split}-00000-of-00001.parquet")
            for split in ("dev", "test", "validation")
        }
    if task == "gsm8k":
        base = root / "openai_gsm8k" / "main"
        return {
            split: str(base / f"{split}-00000-of-00001.parquet")
            for split in ("train", "test")
        }
    if task in ("arc_easy", "arc_challenge"):
        dataset_name = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        base = root / "allenai_ai2_arc" / dataset_name
        return {
            split: str(base / f"{split}-00000-of-00001.parquet")
            for split in ("train", "validation", "test")
        }
    raise KeyError(f"No local benchmark mapping for task {task!r}")


def discover_mmlu_subjects(benchmark_root: Path) -> tuple[str, ...]:
    """Discover the 57 subject directories while excluding ``all`` helpers."""

    root = benchmark_root.expanduser().resolve() / "cais_mmlu"
    if not root.is_dir():
        raise FileNotFoundError(f"MMLU root does not exist: {root}")
    subjects = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name in {"all", "auxiliary_train"}:
            continue
        required = (
            path / "dev-00000-of-00001.parquet",
            path / "test-00000-of-00001.parquet",
        )
        if all(item.is_file() for item in required):
            subjects.append(path.name)
    if len(subjects) != 57:
        raise ValueError(
            "MMLU snapshot must expose exactly 57 subject directories with dev/test parquet; "
            f"found={len(subjects)} subjects={subjects[:5]}"
        )
    unexpected = sorted(set(subjects) ^ set(EXPECTED_MMLU_SUBJECTS))
    if unexpected:
        raise ValueError(
            "MMLU snapshot subject set does not match the original 57-subject benchmark: "
            f"unexpected_or_missing={unexpected}"
        )
    return tuple(subjects)


REQUIRED_COLUMNS = {
    # These are the raw columns consumed by the official task's
    # ``process_docs``; validating only the derived ``query`` would conceal a
    # malformed local snapshot until inference starts.
    "hellaswag": {"activity_label", "ctx_a", "ctx_b", "endings", "label"},
    "mmlu": {"question", "choices", "answer"},
    "gsm8k": {"question", "answer"},
    "arc_easy": {"question", "choices", "answerKey"},
    "arc_challenge": {"question", "choices", "answerKey"},
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _parquet_descriptor(path: Path, required_columns: set[str]) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to validate the local parquet schema before evaluation"
        ) from exc
    parquet_file = parquet.ParquetFile(path)
    names = {str(field.name) for field in parquet_file.schema_arrow}
    missing = sorted(required_columns - names)
    if missing:
        raise ValueError(f"{path} is missing required columns {missing}; found={sorted(names)}")
    return {
        "schema_check": "PASS",
        "columns": sorted(names),
        "num_rows": int(parquet_file.metadata.num_rows),
        "row_groups": int(parquet_file.metadata.num_row_groups),
    }


def build_dataset_manifest(benchmark_root: Path, *, checksums: bool = True) -> list[dict[str, Any]]:
    root = benchmark_root.expanduser().resolve()
    files = sorted(path for path in root.rglob("*.parquet") if path.is_file())
    manifest = []
    for path in files:
        entry: dict[str, Any] = {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        if checksums:
            entry["sha256"] = _sha256(path)
        manifest.append(entry)
    return manifest


def validate_benchmark_layout(
    benchmark_root: Path, *, checksums: bool = True
) -> dict[str, Any]:
    root = ensure_external_path(benchmark_root, label="benchmark root")
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark root is not an existing local directory: {root}")
    required_files: dict[str, Path] = {}
    for task in ("hellaswag", "gsm8k", "arc_easy", "arc_challenge"):
        for split, path in local_data_files(task, root).items():
            required_files[f"{task}:{split}"] = Path(path)
    mmlu_subjects = discover_mmlu_subjects(root)
    for subject in mmlu_subjects:
        for split, path in local_data_files(f"mmlu_{subject}", root).items():
            required_files[f"mmlu:{subject}:{split}"] = Path(path)
    missing = sorted(str(path) for path in required_files.values() if not path.is_file())
    if missing:
        raise FileNotFoundError(
            f"Benchmark snapshot is incomplete; missing {len(missing)} parquet files, "
            f"first entries={missing[:5]}"
        )

    split_descriptors: dict[str, Any] = {}
    for key, path in required_files.items():
        base_task = key.split(":", 1)[0]
        split_descriptors[key] = _parquet_descriptor(path, REQUIRED_COLUMNS[base_task])
    manifest = build_dataset_manifest(root, checksums=checksums)
    return {
        "root": str(root),
        "required_file_count": len(required_files),
        "required_files": {key: str(path) for key, path in required_files.items()},
        "split_descriptors": split_descriptors,
        "all_parquet_file_count": len(manifest),
        "all_parquet_manifest": manifest,
        "auxiliary_train_used": False,
        "gsm8k_socratic_used": False,
        "mmlu_subjects": list(mmlu_subjects),
    }


def _official_task_yaml_paths() -> list[Path]:
    import lm_eval.tasks

    roots = [Path(item) for item in lm_eval.tasks.__path__]
    return sorted(path for root in roots for path in root.rglob("*.yaml"))


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        # Use the pinned harness loader so includes are recursively resolved,
        # while keeping !function values as tagged path strings that can be
        # serialized into a local overlay without reimplementing task logic.
        from lm_eval.tasks._yaml_loader import load_yaml

        value = load_yaml(path, resolve_func=False, recursive=True)
    except ImportError as exc:  # pragma: no cover - lm_eval is remote-only here
        raise RuntimeError("lm_eval is required to inspect official task configs") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Official task config is not a mapping: {path}")
    return value


def discover_official_task_configs() -> dict[str, tuple[Path, dict[str, Any]]]:
    """Discover task YAMLs from the installed lm_eval package, not the Hub."""

    configs: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in _official_task_yaml_paths():
        try:
            config = _load_yaml(path)
        except (OSError, ValueError):
            continue
        task = config.get("task")
        if isinstance(task, str) and task and not task.startswith("_"):
            # Prefer the shortest path when a package ships an auxiliary copy.
            previous = configs.get(task)
            if previous is None or len(path.parts) < len(previous[0].parts):
                configs[task] = (path, config)
    return configs


def discover_official_group_config(group: str) -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in _official_task_yaml_paths():
        try:
            config = _load_yaml(path)
        except (OSError, ValueError):
            continue
        if config.get("group") == group or path.stem == group:
            candidates.append((path, config))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find installed official lm_eval group config for {group!r}"
        )
    return sorted(candidates, key=lambda item: (len(item[0].parts), str(item[0])))[0]


def _official_group_configs() -> dict[str, dict[str, Any]]:
    """Load official group configs for preflight membership expansion."""

    groups: dict[str, dict[str, Any]] = {}
    for path in _official_task_yaml_paths():
        try:
            config = _load_yaml(path)
        except (OSError, ValueError):
            continue
        name = config.get("group")
        if isinstance(name, str) and name and name not in groups:
            groups[name] = config
    return groups


def _official_task_tags(
    task_configs: Mapping[str, tuple[Path, Mapping[str, Any]]]
) -> dict[str, set[str]]:
    """Build the tag-to-leaf index used by official hierarchical groups."""

    tags: dict[str, set[str]] = {}
    for task_name, (_, config) in task_configs.items():
        values = config.get("tag")
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, (list, tuple)):
            continue
        for tag in values:
            if isinstance(tag, str) and tag:
                tags.setdefault(tag, set()).add(task_name)
    return tags


def _expand_official_group(
    name: str,
    groups: Mapping[str, Mapping[str, Any]],
    task_names: Mapping[str, Any],
    active: tuple[str, ...] = (),
    *,
    tags: Mapping[str, set[str]] | None = None,
) -> set[str]:
    """Expand a group using only the installed official group/task registry."""

    if name in active:
        raise ValueError(f"Cycle in official task groups: {active + (name,)}")
    config = groups.get(name)
    if config is None:
        raise KeyError(f"Official task group {name!r} is not registered")
    members = config.get("task", [])
    if isinstance(members, str):
        members = [members]
    expanded: set[str] = set()
    for member in members:
        # GroupConfig.task is a list of names in lm_eval 0.4.12, but accepting
        # the documented mapping form keeps this preflight compatible with
        # task registries that spell a member as {task: ...} or {group: ...}.
        if isinstance(member, Mapping):
            member = member.get("group", member.get("task"))
        if not isinstance(member, str):
            raise ValueError(
                f"Official task group {name!r} contains an invalid member {member!r}"
            )
        if member in groups:
            expanded.update(
                _expand_official_group(
                    member, groups, task_names, active + (name,), tags=tags
                )
            )
        elif tags is not None and member in tags:
            expanded.update(tags[member])
        elif member in task_names:
            expanded.add(member)
        else:
            raise KeyError(f"Official task group {name!r} references unknown member {member!r}")
    return expanded


def _dump_yaml(
    path: Path, payload: Mapping[str, Any], *, source_path: Path | None = None
) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to write local lm_eval overlays") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    class _FunctionRef(str):
        pass

    def represent_function(dumper: Any, data: _FunctionRef) -> Any:
        return dumper.represent_scalar("!function", str(data))

    yaml.SafeDumper.add_representer(_FunctionRef, represent_function)

    # ``load_yaml(resolve_func=False)`` returns plain strings, not a marker
    # object, so SafeDumper would otherwise silently turn every official
    # function reference into an ordinary scalar.  Re-tag only fields whose
    # task-config schema permits !function (including metric aggregation).
    def is_function_ref(value: str) -> bool:
        return _is_function_ref_string(value)

    def mark_function_refs(value: Any, key: str | None = None) -> Any:
        if key in _FUNCTION_REFERENCE_KEYS and isinstance(value, str) and is_function_ref(value):
            return _FunctionRef(_normalise_function_reference(value, source_path))
        if isinstance(value, Mapping):
            return {k: mark_function_refs(v, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [mark_function_refs(v, key) for v in value]
        return value

    marked = mark_function_refs(dict(payload))
    path.write_text(
        yaml.safe_dump(marked, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _task_name_for_subject(subject: str, configs: Mapping[str, Any]) -> str:
    expected = f"mmlu_{subject}"
    if expected in configs:
        return expected
    raise FileNotFoundError(
        f"Installed lm_eval does not expose the expected MMLU subject task {expected!r}"
    )


def _metric_names(config: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for item in config.get("metric_list", []) or []:
        if isinstance(item, Mapping) and item.get("metric") is not None:
            values.add(str(item["metric"]))
    return values


def _protocol_detail(task: str, config: Mapping[str, Any]) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "official_task": task,
        "dataset_path_before_overlay": config.get("dataset_path"),
        "dataset_name_before_overlay": config.get("dataset_name"),
        "test_split": config.get("test_split"),
        "validation_split": config.get("validation_split"),
        "fewshot_split": config.get("fewshot_split"),
        "num_fewshot": config.get("num_fewshot", "native_default"),
        "output_type": config.get("output_type"),
        "metric_names": sorted(_metric_names(config)),
    }
    if task == "hellaswag" and config.get("validation_split") not in (None, "validation"):
        raise ValueError(
            f"HellaSwag must evaluate validation, got {config.get('validation_split')!r}"
        )
    if task == "hellaswag":
        required = {"acc", "acc_norm"}
        if not required.issubset(_metric_names(config)):
            raise ValueError(f"HellaSwag official metrics must include {sorted(required)}")
    if task.startswith("mmlu_"):
        if config.get("test_split") not in (None, "test"):
            raise ValueError(f"MMLU {task} must evaluate test, got {config.get('test_split')!r}")
        if config.get("fewshot_split") not in (None, "dev"):
            raise ValueError(f"MMLU {task} must use dev few-shot, got {config.get('fewshot_split')!r}")
        if config.get("num_fewshot") not in (None, 5):
            raise ValueError(f"MMLU {task} has unsupported num_fewshot={config.get('num_fewshot')!r}")
        if config.get("num_fewshot") is None:
            # lm_eval==0.4.12's original MMLU template declares dev as the
            # few-shot split but omits num_fewshot (its evaluator default is
            # zero).  The requested original MMLU protocol is 5-shot, so the
            # run applies a task-scoped API override, never a global override.
            detail["num_fewshot_override"] = 5
        if "acc" not in _metric_names(config):
            raise ValueError(f"MMLU {task} official metrics must include acc")
    if task == "gsm8k":
        if config.get("dataset_name") not in (None, "main"):
            raise ValueError(f"GSM8K must use main, got {config.get('dataset_name')!r}")
        if config.get("test_split") not in (None, "test"):
            raise ValueError("GSM8K must evaluate main/test")
        if config.get("fewshot_split") not in (None, "train"):
            raise ValueError("GSM8K must use main/train for native few-shot")
        if config.get("num_fewshot") != 5:
            raise ValueError(
                "GSM8K must expose native num_fewshot=5 in the installed task config; "
                f"got {config.get('num_fewshot')!r}"
            )
        if not ({"exact_match", "exact-match"} & _metric_names(config)):
            raise ValueError("GSM8K official metrics must include exact_match")
        generation = dict(config.get("generation_kwargs") or {})
        if generation.get("do_sample") is not False or float(generation.get("temperature", -1.0)) != 0.0:
            raise ValueError(
                "GSM8K official task config must explicitly set do_sample=false and temperature=0; "
                f"got {generation!r}. The overlay refuses to change task semantics."
            )
        detail["generation_kwargs"] = {
            **generation,
            "do_sample": False,
            "temperature": 0.0,
        }
    if task in ("arc_easy", "arc_challenge"):
        if config.get("test_split") not in (None, "test"):
            raise ValueError(f"{task} must evaluate test, got {config.get('test_split')!r}")
        if not {"acc", "acc_norm"}.issubset(_metric_names(config)):
            raise ValueError(f"{task} official metrics must include acc and acc_norm")
    return detail


def prepare_local_task_overlays(
    benchmark_root: Path, overlay_dir: Path, requested_tasks: Sequence[str]
) -> dict[str, Any]:
    """Copy official configs and replace only local dataset source fields."""

    configs = discover_official_task_configs()
    mmlu_subjects = discover_mmlu_subjects(benchmark_root) if "mmlu" in requested_tasks else ()
    if "mmlu" in requested_tasks:
        group_path, group_config = discover_official_group_config("mmlu")
        group_tasks = group_config.get("task", [])
        has_group_tasks = isinstance(group_tasks, str) or (
            isinstance(group_tasks, (list, tuple)) and bool(group_tasks)
        )
        expected_names = [_task_name_for_subject(subject, configs) for subject in mmlu_subjects]
        # 0.4.12 stores mmlu hierarchically (stem/other/social_sciences/
        # humanities), with each subgroup referring to a tag such as
        # ``mmlu_stem_tasks``.  Expand that official hierarchy and require its leaves
        # to be exactly the discovered original 57 subjects.  This prevents a
        # similarly named auxiliary or multilingual group from slipping into
        # the formal run while preserving the official group YAML unchanged.
        official_groups = _official_group_configs()
        official_tags = _official_task_tags(configs)
        official_group_leaves = _expand_official_group(
            "mmlu", official_groups, configs, tags=official_tags
        )
        if official_group_leaves != set(expected_names):
            raise ValueError(
                "Installed official mmlu group is not the original 57-subject group: "
                f"missing={sorted(set(expected_names) - official_group_leaves)} "
                f"unexpected={sorted(official_group_leaves - set(expected_names))}"
        )
        local_group = copy.deepcopy(group_config)
        if not has_group_tasks:
            local_group["task"] = expected_names
        _dump_yaml(overlay_dir / "mmlu.yaml", local_group, source_path=group_path)

    protocol: dict[str, Any] = {
        "official_group": {},
        "tasks": {},
        "overlay_dir": str(overlay_dir.resolve()),
        "dataset_path_override": "parquet",
        "dataset_kwargs_data_files": True,
    }
    if "mmlu" in requested_tasks:
        group_path, group_config = discover_official_group_config("mmlu")
        protocol["official_group"] = {
            "name": "mmlu",
            "source": str(group_path),
            "subject_count": len(mmlu_subjects),
            "subjects": list(mmlu_subjects),
            "expanded_leaf_tasks": sorted(official_group_leaves),
        }
        tasks_to_write = [f"mmlu_{subject}" for subject in mmlu_subjects]
    else:
        tasks_to_write = []
    tasks_to_write.extend(task for task in requested_tasks if task != "mmlu")

    for task in tasks_to_write:
        task_name = task
        source_path, official = configs.get(task_name, (None, None))
        if source_path is None or official is None:
            raise FileNotFoundError(f"Official lm_eval task YAML not found for {task_name!r}")
        _protocol_detail(task_name, official)
        local = copy.deepcopy(official)
        local["dataset_path"] = "parquet"
        # Generic parquet has no Hugging Face config name.  Removing this
        # source-only field does not touch any task processing or metric field.
        local.pop("dataset_name", None)
        kwargs = dict(local.get("dataset_kwargs") or {})
        kwargs["data_files"] = local_data_files(task_name, benchmark_root)
        local["dataset_kwargs"] = kwargs
        _dump_yaml(
            overlay_dir / f"{task_name}.yaml", local, source_path=source_path
        )
        detail = _protocol_detail(task_name, official)
        detail.update(
            {
                "official_yaml": str(source_path),
                "overlay_yaml": str((overlay_dir / f"{task_name}.yaml").resolve()),
                "dataset_path_after_overlay": "parquet",
                "local_data_files": local_data_files(task_name, benchmark_root),
                "overlay_modified_fields": [
                    "dataset_path",
                    "dataset_name (removed source-only field)",
                    "dataset_kwargs.data_files",
                ],
                "uses_auxiliary_train": False,
                "uses_gsm8k_socratic": False,
            }
        )
        protocol["tasks"][task_name] = detail
    return protocol


def inspect_pinned_versions() -> dict[str, str]:
    actual = {
        name: package_version(name)
        for name in ("lm-eval", "lm_eval", "transformers", "datasets", "torch")
    }
    lm_version = actual.get("lm-eval", "<unavailable>")
    if lm_version == "<unavailable>":
        lm_version = actual.get("lm_eval", "<unavailable>")
    actual["lm_eval"] = lm_version
    expected = {
        "lm_eval": EXPECTED_LM_EVAL_VERSION,
        "transformers": EXPECTED_TRANSFORMERS_VERSION,
        "datasets": EXPECTED_DATASETS_VERSION,
    }
    mismatches = {
        name: {"expected": value, "actual": actual.get(name)}
        for name, value in expected.items()
        if actual.get(name) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Pinned evaluation dependency mismatch; refusing silent protocol drift: {mismatches}"
        )
    return actual


def recursive_forward_trace_audit(model: Any, *, device: str) -> dict[str, Any]:
    """Audit custom Auto registration, shared storage, and 0..14 twice trace."""

    import torch

    from code.RSmol.recursive_model import RecursiveLlamaForCausalLM, parameter_audit

    if not isinstance(model, RecursiveLlamaForCausalLM):
        raise TypeError(f"AutoModel resolved the wrong class: {type(model)!r}")
    config = model.config
    logical = int(getattr(config, "num_hidden_layers", 0))
    physical = int(getattr(config, "recursive_layer_count", 0))
    loops = int(getattr(config, "recursive_loops", 0))
    if (logical, physical, loops) != (
        EXPECTED_RECURSIVE_LOGICAL_LAYERS,
        EXPECTED_RECURSIVE_PHYSICAL_LAYERS,
        EXPECTED_RECURSIVE_LOOPS,
    ):
        raise ValueError("Recursive runtime config is not logical=30 physical=15 loops=2")
    if len(getattr(model.model, "layers", ())) != physical:
        raise ValueError(
            "Recursive runtime layer list does not match physical depth: "
            f"len={len(getattr(model.model, 'layers', ()))} physical={physical}"
        )
    audit = parameter_audit(model)
    if audit.get("physical_layer_count") != EXPECTED_RECURSIVE_PHYSICAL_LAYERS:
        raise ValueError(f"Unexpected recursive parameter audit: {audit}")
    physical_layer_objects = list(model.model.layers)
    unique_layer_object_count = len({id(layer) for layer in physical_layer_objects})
    if unique_layer_object_count != EXPECTED_RECURSIVE_PHYSICAL_LAYERS:
        raise ValueError(
            "Recursive model layer list contains duplicate or missing module objects: "
            f"count={len(physical_layer_objects)} unique={unique_layer_object_count}"
        )

    trace: list[int] = []
    hooks = []
    for index, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(lambda _module, _args, _out, i=index: trace.append(i)))
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    for hook in hooks:
        hook.remove()
    expected = list(range(EXPECTED_RECURSIVE_PHYSICAL_LAYERS)) * EXPECTED_RECURSIVE_LOOPS
    if trace != expected:
        raise RuntimeError(f"Recursive forward trace mismatch: expected={expected} got={trace}")
    no_duplicate_parameter_storage = bool(
        audit.get("recursive_layer_count") == physical
        and audit.get("physical_layer_count") == physical
        and unique_layer_object_count == physical
        and audit.get("depth_consistent") is True
        and audit.get("logical_cache_slot_count") == logical
        and audit.get("recursive_shared_parameter_count_logical_references")
        == audit.get("recursive_shared_parameter_count_unique") * loops
    )
    if not no_duplicate_parameter_storage:
        raise RuntimeError(
            "Recursive parameter audit found duplicate storage or inconsistent logical references: "
            f"{audit}"
        )
    return {
        "status": "PASS",
        "custom_auto_class": type(model).__name__,
        "logical_layer_count": logical,
        "physical_layer_count": physical,
        "recursive_loops": loops,
        "forward_trace": trace,
        "expected_forward_trace": expected,
        "no_duplicate_parameter_storage": no_duplicate_parameter_storage,
        "physical_layer_object_count": len(physical_layer_objects),
        "unique_physical_layer_object_count": unique_layer_object_count,
        "parameter_audit": audit,
    }


def load_and_audit_recursive_model(model_path: Path, *, device: str, dtype: str) -> dict[str, Any]:
    import torch

    from transformers import AutoModelForCausalLM

    from code.RSmol.recursive_model import register_auto_class

    register_auto_class()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=getattr(torch, dtype),
        low_cpu_mem_usage=True,
    )
    # Deliberately no device_map: one submitted GPU owns the evaluation.
    model.to(torch.device(device))
    try:
        return recursive_forward_trace_audit(model, device=device)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _flatten_result_rows(task: str, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    result_map = payload.get("results", {})
    if not isinstance(result_map, Mapping):
        return rows
    for result_task, metrics in result_map.items():
        if not isinstance(metrics, Mapping):
            continue
        for metric_name, value in metrics.items():
            if metric_name in {"alias", "group_subtasks", "group_alias"}:
                continue
            if isinstance(value, (int, float)) or hasattr(value, "item"):
                normalized_metric = str(metric_name).split(",", 1)[0]
                rows.append(
                    {
                        "requested_task": task,
                        "task": str(result_task),
                        "metric": normalized_metric,
                        "value": json_safe(value),
                    }
                )
    return rows


def _result_sample_counts(payload: Mapping[str, Any]) -> dict[str, Any]:
    result_map = payload.get("results", {})
    if not isinstance(result_map, Mapping):
        return {}
    return {
        str(task): metrics.get("n-samples")
        for task, metrics in result_map.items()
        if isinstance(metrics, Mapping) and "n-samples" in metrics
    }


def _write_summary(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_json(output_dir / "summary.json", {"rows": list(rows)})
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("requested_task", "task", "metric", "value"))
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
    config: EvaluationConfig,
    task: str,
    task_dir: Path,
    overlay_dir: Path,
    stderr_log_path: Path,
    *,
    register_recursive: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Keep dataset/request caches task-local and external.  The environment is
    # already offline before lm_eval is imported; resetting these locations
    # here also makes separate task outputs independently reproducible.
    set_offline_environment(task_dir / ".hf-cache")
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
        f"[stage3][task={task}] starting lm_eval; overlay={overlay_dir} "
        f"stderr_log={stderr_log_path}",
        flush=True,
    )
    # num_fewshot is intentionally omitted: official YAML native defaults are
    # required (MMLU/GSM8K 5-shot, HellaSwag/ARC zero-shot).
    try:
        with contextlib.redirect_stderr(captured_stderr):
            if register_recursive:
                from code.RSmol.recursive_model import register_auto_class

                # Re-assert the project Auto mapping immediately before the
                # harness imports its HF backend/model constructor.
                register_auto_class()
            from lm_eval import evaluator
            from lm_eval.tasks import TaskManager

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
                f"mmlu_{subject}" for subject in discover_mmlu_subjects(config.benchmark_root)
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
                f"[stage3][task={task}][WARN] could not write stderr log "
                f"{stderr_log_path}: {log_error!r}",
                file=sys.stderr,
                flush=True,
            )
        if stderr_text:
            print(stderr_text, file=sys.stderr, end="", flush=True)
    if result is None:
        raise RuntimeError(f"lm_eval returned no result for task {task!r}")
    result_payload = {
        "requested_task": task,
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_at": utc_now(),
        "task_config": str(overlay_dir),
        "stderr_log": str(stderr_log_path),
        "raw_lm_eval": result,
    }
    write_json(task_dir / "lm_eval_results.json", result_payload)
    if config.log_samples and isinstance(result, Mapping) and "samples" in result:
        write_json(task_dir / "log_samples.json", {"samples": result["samples"]})
    return result_payload, _flatten_result_rows(task, result)


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    started_at = utc_now()
    output_dir = ensure_external_output(config.output_dir)
    log_root = ensure_log_root(config.log_root)
    set_offline_environment(config.cache_dir or (output_dir / ".hf-cache"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_info = inspect_model_artifacts(config.model_path)
    if config.reference_model_path is not None:
        reference_info = inspect_model_artifacts(config.reference_model_path, label="reference model")
    else:
        reference_info = None
    versions = inspect_pinned_versions()
    load_tokenizer_runtime_metadata(model_info)
    if reference_info is not None:
        load_tokenizer_runtime_metadata(reference_info)
    if model_info["recursive_audit"]["is_recursive"] and reference_info is not None:
        compatibility = compare_model_compatibility(reference_info, model_info)
    else:
        compatibility = {"compatible": "not_requested"}
    benchmark = validate_benchmark_layout(config.benchmark_root)
    # Importing lm_eval and discovering the installed configs occurs after all
    # offline flags are set and before any model inference.
    task_probe_dir = output_dir / ".task-config-probe"
    task_probe_dir.mkdir(parents=True, exist_ok=True)
    protocol = prepare_local_task_overlays(
        config.benchmark_root, task_probe_dir, config.tasks
    )
    recursive_runtime_audit: dict[str, Any] = {"status": "not_applicable"}
    if not config.validation_only:
        if not config.device.startswith("cuda"):
            raise RuntimeError("Formal Stage 3 evaluation requires one CUDA device")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Formal Stage 3 evaluation requires a submitted CUDA job")
        if model_info["recursive_audit"]["is_recursive"]:
            recursive_runtime_audit = load_and_audit_recursive_model(
                config.model_path, device=config.device, dtype=config.dtype
            )

    task_results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    task_log_paths = {
        task: str(_task_log_path(log_root, config, task)) for task in config.tasks
    }
    print(
        f"[stage3] model={config.model_path} output={output_dir} "
        f"log_root={log_root} tasks={','.join(config.tasks)}",
        flush=True,
    )
    if not config.validation_only:
        for task in config.tasks:
            task_dir = output_dir / task
            stderr_log_path = _task_log_path(log_root, config, task)
            overlay_dir = task_dir / "lm_eval_include"
            print(
                f"[stage3][task={task}] preparing overlay={overlay_dir} "
                f"stderr_log={stderr_log_path}",
                flush=True,
            )
            try:
                task_dir = ensure_external_output(task_dir)
                task_dir.mkdir(parents=True, exist_ok=True)
                overlay_dir.mkdir(parents=True, exist_ok=True)
                task_protocol = prepare_local_task_overlays(
                    config.benchmark_root, overlay_dir, (task,)
                )
                write_json(task_dir / "task_protocol.json", task_protocol)
                payload, task_rows = _run_single_task(
                    config,
                    task,
                    task_dir,
                    overlay_dir,
                    stderr_log_path,
                    register_recursive=model_info["recursive_audit"]["is_recursive"],
                )
                task_results[task] = payload
                rows.extend(task_rows)
            except Exception:
                failure_text = traceback.format_exc()
                failures[task] = failure_text
                diagnostic = (
                    f"\n=== Stage 3 task failure: {task} ===\n"
                    f"model={config.model_path}\n"
                    f"output={output_dir / task}\n"
                    f"overlay={overlay_dir}\n"
                    f"stderr_log={stderr_log_path}\n"
                    f"{failure_text}"
                )
                try:
                    _append_text(stderr_log_path, diagnostic)
                except Exception as log_error:
                    print(
                        f"[stage3][task={task}][WARN] failed to append diagnostic log "
                        f"{stderr_log_path}: {log_error!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                print(
                    f"[stage3][task={task}][FAIL] detailed traceback follows; "
                    f"log={stderr_log_path}",
                    file=sys.stderr,
                    flush=True,
                )
                print(diagnostic, file=sys.stderr, end="", flush=True)
    _write_summary(output_dir, rows)
    skipped_count = len(config.tasks) if config.validation_only else 0
    audit = {
        "status": "FAIL" if failures else "PASS",
        "stage": "stage3_benchmark_evaluation",
        "started_at": started_at,
        "finished_at": utc_now(),
        "command": sys.argv,
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": versions,
        "model": model_info,
        "reference_model": reference_info,
        "model_compatibility": compatibility,
        "benchmark_root": str(config.benchmark_root.expanduser().resolve()),
        "benchmark_manifest": benchmark,
        "protocol": protocol,
        "configuration": asdict(config),
        "log_root": str(log_root),
        "task_log_paths": task_log_paths,
        "offline_environment": {name: os.environ.get(name) for name in OFFLINE_ENVIRONMENT},
        "gpu": _gpu_info(config.device),
        "recursive_runtime_audit": recursive_runtime_audit,
        "tasks": list(config.tasks),
        "task_results": task_results,
        "summary_rows": rows,
        "sample_counts": {
            task: _result_sample_counts(payload.get("raw_lm_eval", {}))
            for task, payload in task_results.items()
        },
        "skipped_count": skipped_count,
        "failed_count": len(failures),
        "failures": failures,
        "output_dir": str(output_dir),
        "stage4_status": "paused",
        "formal_eval_executed": not config.validation_only,
    }
    write_json(output_dir / "audit_report.json", audit)
    # Keep the probe overlays, but make their role explicit and prevent them
    # from being mistaken for a benchmark result directory.
    write_json(output_dir / "run_config.json", {"configuration": asdict(config), "protocol": protocol})
    if failures:
        failure_summaries = {
            task: next(
                (line.strip() for line in reversed(trace.splitlines()) if line.strip()),
                "unknown failure",
            )
            for task, trace in failures.items()
        }
        raise RuntimeError(
            "Stage 3 task failures: "
            f"{failure_summaries}; detailed tracebacks were printed above and saved under {log_root}"
        )
    return audit


def parse_args(argv: Sequence[str] | None = None) -> EvaluationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_ORIGINAL_MODEL)
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
    parser.add_argument("--reference-model-path", type=Path, default=None)
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="Diagnostic log root; defaults to code/RSmol/log (explicit checkout exception).",
    )
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
        reference_model_path=args.reference_model_path,
        log_root=args.log_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config: EvaluationConfig | None = None
    output_was_fresh = False
    try:
        config = parse_args(argv)
        # Remember the collision-guard state before run_evaluation creates its
        # probe/cache directories.  This lets a failed preflight leave an
        # audit report in a path that this invocation explicitly reserved,
        # while preserving the non-empty-output refusal contract.
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
                _append_text(
                    runtime_log,
                    "\n=== Stage 3 process failure ===\n"
                    f"model={config.model_path}\n"
                    f"output={config.output_dir}\n"
                    f"{failure_text}",
                )
                print(
                    f"[stage3][process][FAIL] detailed traceback saved to {runtime_log}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as log_error:
                print(
                    f"[stage3][process][WARN] could not write process log: {log_error!r}",
                    file=sys.stderr,
                    flush=True,
                )
        # A failed preflight should still leave a machine-readable reason when
        # the requested external output path can be safely created.  Never
        # overwrite an existing audit or any non-empty caller-owned directory.
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
                            "started_at": utc_now(),
                            "finished_at": utc_now(),
                            "command": sys.argv,
                            "configuration": asdict(config),
                            "failure": failure_text,
                            "formal_eval_executed": False,
                            "stage4_status": "paused",
                        },
                    )
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
