#!/usr/bin/env python3
"""Stage 4 distributed training for the SmolLM2-5-10xr-5 architecture.

The Stage 4 contract is deliberately implemented without ``Trainer`` or an
implicit dataset worker pool.  Each torchrun rank owns a deterministic subset
of the 85 Parquet shards, streams one shard at a time with a bounded shuffle
buffer, and contributes token-weighted gradients to a DDP model.  The same
entry point implements the five historical gates plus an independent
``FORMAL`` training mode used by the remote job:

``A`` synthetic 8-rank DDP audit, ``D`` the fixed ten-step real-data pilot,
``E`` a resume smoke, and ``FORMAL`` the 9,244-step training target. Gate B/C
are intentionally unsupported here: reuse the existing Gate B audit JSON and
do not rerun either audit from this entry point.

Only reports/checkpoints under an explicitly external output directory are
written.  In particular, neither the local checkout nor the remote Git
checkout is ever used as a model/checkpoint transfer directory.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import quantiles
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REMOTE_CHECKOUT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM")
MODEL_ARCHITECTURE_CONTRACT = "logical_50_110_physical_20_5_10xr_5_poisson_r4_10_tail4"
LOGICAL_LAYER_COUNT = 110
PHYSICAL_LAYER_COUNT = 20
MIDDLE_LAYER_COUNT = 10
MIN_MIDDLE_LOOPS = 4
MAX_MIDDLE_LOOPS = 10
DEFAULT_INFERENCE_MIDDLE_LOOPS = 7
PARAMETER_GRADIENT_TAIL_LOOPS = 4
LOGICAL_TO_PHYSICAL = tuple(range(5)) + tuple(range(5, 15)) * MAX_MIDDLE_LOOPS + tuple(range(15, 20))
SAMPLING_POLICY = "truncated_poisson"
SAMPLER_VERSION = "truncated_poisson_lambda7_support4_10_v1"
SAMPLER_KEY = "sha256_cpu_torch_generator_base_seed_rank_optimizer_step_microbatch_v1"
POISSON_LAMBDA = 7.0
POISSON_SUPPORT = tuple(range(MIN_MIDDLE_LOOPS, MAX_MIDDLE_LOOPS + 1))
POISSON_NORMALIZATION_Z = sum(
    math.exp(-POISSON_LAMBDA) * POISSON_LAMBDA**k / math.factorial(k)
    for k in POISSON_SUPPORT
)
POISSON_Z = POISSON_NORMALIZATION_Z
POISSON_PROBABILITIES = tuple(
    (math.exp(-POISSON_LAMBDA) * POISSON_LAMBDA**k / math.factorial(k)) / POISSON_NORMALIZATION_Z
    for k in POISSON_SUPPORT
)
DEFAULT_DATA_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset")
DEFAULT_OUTPUT_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10xr_5_poisson")
EXPECTED_PARQUET_COUNT = 85
EXPECTED_PARQUET_NAMES = tuple(
    f"train-{index:05d}-of-{EXPECTED_PARQUET_COUNT:05d}.parquet"
    for index in range(EXPECTED_PARQUET_COUNT)
)
DEFAULT_WORLD_SIZE = 8
DEFAULT_MICRO_BATCH_SIZE = 2
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 64
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_MAX_LR = 2e-4
DEFAULT_MIN_LR = 2e-5
DEFAULT_ADAMW_BETAS = (0.9, 0.95)
DEFAULT_ADAMW_EPS = 1e-8
DEFAULT_ADAMW_WEIGHT_DECAY = 0.1
DEFAULT_ADAMW_AMSGRAD = False
DEFAULT_CONTEXT_LENGTH = 1024
# Gate D is intentionally a bounded ten-step smoke.  Keep the planned
# 9,244-step run as an explicit formal target so it cannot silently
# turn a Stage 4 smoke into a Stage 5 implementation.
DEFAULT_MAX_OPTIMIZER_STEPS = 10
DEFAULT_FORMAL_OPTIMIZER_STEPS = 9244
DEFAULT_LOG_INTERVAL_STEPS = 10
SCHEDULER_TYPE = "linear_warmup_cosine"
DEFAULT_TARGET_SAMPLES_PER_RANK = (
    DEFAULT_MAX_OPTIMIZER_STEPS
    * DEFAULT_GRADIENT_ACCUMULATION_STEPS
    * DEFAULT_MICRO_BATCH_SIZE
)
# Human-readable Gate D smoke values: 1,280 effective samples/rank and
# 640 local microbatches for the default ten-step pilot.  The formal target
# values are retained separately in ``DEFAULT_FORMAL_*`` constants.
DEFAULT_LOCAL_MICROBATCHES = DEFAULT_MAX_OPTIMIZER_STEPS * DEFAULT_GRADIENT_ACCUMULATION_STEPS
DEFAULT_GLOBAL_EFFECTIVE_BATCH = DEFAULT_WORLD_SIZE * DEFAULT_MICRO_BATCH_SIZE * DEFAULT_GRADIENT_ACCUMULATION_STEPS
DEFAULT_FORMAL_GLOBAL_SAMPLES = DEFAULT_FORMAL_OPTIMIZER_STEPS * DEFAULT_GLOBAL_EFFECTIVE_BATCH
DEFAULT_FORMAL_SAMPLES_PER_RANK = (
    DEFAULT_FORMAL_OPTIMIZER_STEPS
    * DEFAULT_MICRO_BATCH_SIZE
    * DEFAULT_GRADIENT_ACCUMULATION_STEPS
)
DEFAULT_FORMAL_LOCAL_MICROBATCHES = DEFAULT_FORMAL_OPTIMIZER_STEPS * DEFAULT_GRADIENT_ACCUMULATION_STEPS
DEFAULT_FORMAL_WARMUP_STEPS = max(1, math.ceil(DEFAULT_FORMAL_OPTIMIZER_STEPS * 0.05))
DEFAULT_FORMAL_SAVE_EVERY = 500
DEFAULT_FORMAL_CHECKPOINT_RETENTION = 3
# The formal 9,244-step target consumes 9,465,856 global samples (1024 per
# optimizer step); Gate D itself consumes only 10 optimizer steps by default.
DEFAULT_RECORD_BUFFER_SIZE = 4096
# Explicitly pinned loader settings: Stage 4 intentionally does not create
# DataLoader workers or an HF Arrow cache during torchrun.
NUM_WORKERS = 0
PIN_MEMORY = False
PERSISTENT_WORKERS = False
DATASET_NUM_PROC = 1


def poisson_metadata() -> dict[str, Any]:
    return {
        "sampling_policy": SAMPLING_POLICY,
        "sampler_version": SAMPLER_VERSION,
        "sampler_key": SAMPLER_KEY,
        "poisson_lambda": POISSON_LAMBDA,
        "poisson_support": list(POISSON_SUPPORT),
        "poisson_normalization_z": POISSON_NORMALIZATION_Z,
        "poisson_probabilities": list(POISSON_PROBABILITIES),
    }


def _sampler_seed(base_seed: int, rank: int, optimizer_step: int, microbatch_index: int) -> int:
    payload = f"{int(base_seed)}:{int(rank)}:{int(optimizer_step)}:{int(microbatch_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def sample_middle_loop_counts(
    base_seed: int, rank: int, optimizer_step: int, microbatch_index: int, batch_size: int
) -> Any:
    """Sample one independent truncated-Poisson depth per local sequence."""

    import torch

    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_sampler_seed(base_seed, rank, optimizer_step, microbatch_index))
    probabilities = torch.tensor(POISSON_PROBABILITIES, dtype=torch.float64, device="cpu")
    indices = torch.multinomial(probabilities, batch_size, replacement=True, generator=generator)
    return torch.tensor(POISSON_SUPPORT, dtype=torch.long, device="cpu")[indices]


def _validate_exact_sampling_contract(value: Mapping[str, Any], *, label: str) -> None:
    """Reject any checkpoint/report sampling metadata drift."""

    support = tuple(int(item) for item in value.get("poisson_support", ()))
    probabilities = tuple(float(item) for item in value.get("poisson_probabilities", ()))
    if (
        value.get("sampling_policy") != SAMPLING_POLICY
        or value.get("sampler_version") != SAMPLER_VERSION
        or value.get("sampler_key") != SAMPLER_KEY
        or float(value.get("poisson_lambda", -1.0)) != POISSON_LAMBDA
        or support != POISSON_SUPPORT
        or len(probabilities) != len(POISSON_PROBABILITIES)
        or any(abs(a - b) > 1e-14 for a, b in zip(probabilities, POISSON_PROBABILITIES))
    ):
        raise ValueError(
            f"{label} sampling contract mismatch: "
            f"policy={value.get('sampling_policy')!r} version={value.get('sampler_version')!r} "
            f"lambda={value.get('poisson_lambda')!r} support={support} "
            f"sampler_key={value.get('sampler_key')!r}"
        )


@dataclass
class Stage4Config:
    """Serializable Stage 4 configuration.

    The numerical defaults are part of the experiment contract. Gate D is a
    fixed ten-step smoke; ``FORMAL`` selects the independent 9,244-step
    production training contract. Gate B/C are rejected by this entry point.
    """

    gate: str = "A"
    model_path: Path | None = None
    tokenizer_path: Path | None = None
    data_dir: Path = DEFAULT_DATA_DIR
    audit_report: Path | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    report_path: Path | None = None
    resume_from: Path | None = None
    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_lr: float = DEFAULT_MAX_LR
    min_lr: float = DEFAULT_MIN_LR
    context_length: int = DEFAULT_CONTEXT_LENGTH
    warmup_steps: int = 0
    max_optimizer_steps: int = DEFAULT_MAX_OPTIMIZER_STEPS
    formal_optimizer_steps: int = DEFAULT_FORMAL_OPTIMIZER_STEPS
    scheduler_total_steps: int | None = None
    log_interval_steps: int = DEFAULT_LOG_INTERVAL_STEPS
    seed: int = 0
    weight_decay: float = DEFAULT_ADAMW_WEIGHT_DECAY
    optimizer_betas: tuple[float, float] = DEFAULT_ADAMW_BETAS
    optimizer_eps: float = DEFAULT_ADAMW_EPS
    optimizer_amsgrad: bool = DEFAULT_ADAMW_AMSGRAD
    max_grad_norm: float = 1.0
    record_buffer_size: int = DEFAULT_RECORD_BUFFER_SIZE
    save_every: int = 500
    checkpoint_retention: int = DEFAULT_FORMAL_CHECKPOINT_RETENTION
    monitor_interval_seconds: float = 60.0
    backend: str = "nccl"  # NCCL is the production 8-GPU backend.
    world_size: int = DEFAULT_WORLD_SIZE
    local_rank: int = -1
    dry_run: bool = False
    allow_non8: bool = False
    synthetic_layers: int = 30
    synthetic_hidden_size: int = 32
    synthetic_vocab_size: int = 97
    sampling_policy: str = SAMPLING_POLICY
    sampler_key: str = SAMPLER_KEY


def _path_is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def ensure_external_output(path: Path) -> Path:
    """Resolve an output and reject both Git checkouts.

    This guard is intentionally applied to the output root, report path,
    atomic checkpoint staging path, and resume smoke directory.
    """

    candidate = path.expanduser().resolve()
    forbidden = (REPO_ROOT.resolve(), REMOTE_CHECKOUT.resolve())
    if any(_path_is_within(candidate, root) for root in forbidden):
        raise ValueError(
            "Stage 4 refuses to write model/checkpoint/report artifacts inside a Git checkout: "
            f"output={candidate}; choose an external task output directory"
        )
    return candidate


def ensure_external_resume(path: Path) -> Path:
    """Validate an external, complete checkpoint before model loading.

    Gate E intentionally uses the checkpoint directory itself as the model
    source.  Checking it here (before ``from_pretrained``) keeps an invalid or
    Git-local resume from turning into a less useful model-loading error and
    makes the resume-source contract independently auditable.
    """

    candidate = path.expanduser().resolve()
    forbidden = (REPO_ROOT.resolve(), REMOTE_CHECKOUT.resolve())
    if any(_path_is_within(candidate, root) for root in forbidden):
        raise ValueError(
            "Stage 4 refuses to read model/checkpoint state from a Git checkout: "
            f"resume_from={candidate}; choose an external complete checkpoint"
        )
    if not candidate.is_dir():
        raise FileNotFoundError(f"Resume checkpoint directory does not exist: {candidate}")
    required = ("config.json", "training_state.pt", "checkpoint_complete.json")
    missing = [name for name in required if not (candidate / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Resume checkpoint is incomplete: {candidate}; missing={missing}"
        )
    model_files = list(candidate.glob("*.safetensors")) + list(candidate.glob("pytorch_model*.bin"))
    tokenizer_files = (
        list(candidate.glob("tokenizer.json"))
        + list(candidate.glob("tokenizer.model"))
        + list(candidate.glob("vocab.*"))
    )
    if not model_files or not tokenizer_files:
        raise FileNotFoundError(
            "Resume checkpoint is missing model/tokenizer artifacts: "
            f"model_files={[item.name for item in model_files]}, "
            f"tokenizer_files={[item.name for item in tokenizer_files]}"
        )
    try:
        marker = json.loads((candidate / "checkpoint_complete.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Resume checkpoint completeness marker is unreadable: {candidate}") from exc
    if marker.get("complete") is not True:
        raise ValueError(
            f"Resume checkpoint completeness marker must declare complete=true: {candidate}"
        )
    return candidate


def expected_parquet_names() -> tuple[str, ...]:
    """Return the exact 85-shard manifest."""

    return EXPECTED_PARQUET_NAMES


def discover_parquet_files(data_dir: Path) -> list[Path]:
    """Validate the exact ``train-xxxxx-of-00085.parquet`` set."""

    root = data_dir.expanduser()
    parquet_root = root / "data" if (root / "data").is_dir() else root
    if not parquet_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {parquet_root}")
    found = sorted(path.name for path in parquet_root.glob("train-*.parquet"))
    expected = list(expected_parquet_names())
    if found != expected:
        missing = sorted(set(expected) - set(found))
        unexpected = sorted(set(found) - set(expected))
        raise ValueError(
            "Stage 4 requires exactly the 85 parquet shards train-00000-of-00085.parquet "
            f"through train-00084-of-00085.parquet; found={len(found)} "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    return [parquet_root / name for name in expected]


def _schema_descriptor(parquet_file: Any, path: Path) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    schema = getattr(parquet_file, "schema_arrow", None)
    if schema is None:
        raise ValueError(f"Parquet shard has no readable Arrow schema footer: {path}")
    fields = [
        {
            "name": str(field.name),
            "type": str(field.type),
            "nullable": bool(getattr(field, "nullable", True)),
        }
        for field in schema
    ]
    names = {field["name"] for field in fields}
    for required in ("text", "source"):
        if required not in names:
            raise ValueError(f"Parquet shard {path} is missing required '{required}' column")
        field_type = next(field["type"] for field in fields if field["name"] == required)
        if "string" not in field_type.lower():
            raise ValueError(
                f"Parquet shard {path} column '{required}' must be a string type; got {field_type!r}"
            )
    signature = tuple((field["name"], field["type"], field["nullable"]) for field in fields)
    return signature, fields


class _LengthStats:
    """Bounded exact counters plus a reservoir for token-length quantiles."""

    def __init__(self, reservoir_size: int = 10000, seed: int = 0) -> None:
        self.count = 0
        self.total = 0
        self.minimum: int | None = None
        self.maximum: int | None = None
        self.reservoir_size = int(reservoir_size)
        self.values: list[int] = []
        self.rng = random.Random(seed)

    def add(self, value: int) -> None:
        value = int(value)
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if len(self.values) < self.reservoir_size:
            self.values.append(value)
        else:
            index = self.rng.randrange(self.count)
            if index < self.reservoir_size:
                self.values[index] = value

    def as_dict(self) -> dict[str, Any]:
        if not self.values:
            return {"count": 0, "min": None, "max": None, "mean": None, "quantiles": {}}
        values = sorted(self.values)
        # statistics.quantiles requires at least two values; explicit order
        # statistics keep the result stable for one-row fixtures.
        if len(values) == 1:
            q = {"p50": values[0], "p90": values[0], "p95": values[0], "p99": values[0]}
        else:
            qtiles = quantiles(values, n=100, method="inclusive")
            q = {
                "p50": qtiles[49],
                "p90": qtiles[89],
                "p95": qtiles[94],
                "p99": qtiles[98],
            }
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.total / self.count,
            "quantiles": q,
            "quantiles_source": "bounded_reservoir",
            "reservoir_size": self.reservoir_size,
        }


def _bounded_counter_add(counter: Counter[str], key: str, *, limit: int = 10000) -> None:
    """Keep source-distribution audit bounded for URL-like source values."""

    if key in counter or len(counter) < limit:
        counter[key] += 1
    else:
        counter["<other>"] += 1


def _tokenize_ids(tokenizer: Any, text: str, context_length: int) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=context_length,
    )
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    result = [int(token) for token in ids]
    if len(result) > context_length:
        raise AssertionError("Tokenizer returned a sequence above context_length")
    return result


def _token_length_for_audit(tokenizer: Any, text: str) -> int:
    """Return the untruncated token length used by the Gate B sample audit."""

    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return len(ids)


def select_sample_shards(
    files: Sequence[Path], *, sample_shards: int = 3, seed: int = 0
) -> list[Path]:
    """Select a deterministic, bounded sample from the sorted shard list."""

    if not (0 < int(sample_shards) <= len(files)):
        raise ValueError(
            f"sample_shards must be in [1, {len(files)}], got {sample_shards}"
        )
    # ``discover_parquet_files`` is sorted, but sorting here also makes this
    # helper deterministic when called directly by tests or other callers.
    ordered = sorted((Path(path) for path in files), key=lambda path: path.name)
    selected = random.Random(int(seed)).sample(ordered, int(sample_shards))
    return sorted(selected, key=lambda path: path.name)


def audit_parquet_shards(
    files: Sequence[Path],
    *,
    tokenizer: Any | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    content: bool = True,
    reservoir_size: int = 10000,
    content_paths: Sequence[Path] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    progress_every_rows: int = 10000,
) -> dict[str, Any]:
    """Audit every footer and optionally stream content from selected shards.

    The footer section records schema, row groups, ``num_rows`` and bytes for
    all 85 shards.  If ``content_paths`` is supplied, the content section
    streams only those shards through ``ParquetFile.iter_batches``; it never
    reads or tokenizes the other shards.  No Arrow Dataset cache is created.
    """

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - remote runtime dependency
        raise ImportError("Stage 4 parquet audit requires pyarrow") from exc
    if len(files) != EXPECTED_PARQUET_COUNT:
        raise ValueError(f"Stage 4 footer audit requires {EXPECTED_PARQUET_COUNT} shards, got {len(files)}")
    if progress_every_rows <= 0:
        raise ValueError("progress_every_rows must be positive")
    ordered_files = [Path(path) for path in files]
    file_keys = {path.resolve() for path in ordered_files}
    if content_paths is None:
        selected_paths = tuple(ordered_files) if content else tuple()
        content_scope = "full_corpus" if content else "footer_only"
    else:
        if not content:
            raise ValueError("content_paths requires content=True")
        selected_paths = tuple(Path(path) for path in content_paths)
        selected_keys = [path.resolve() for path in selected_paths]
        if len(set(selected_keys)) != len(selected_keys) or not set(selected_keys) <= file_keys:
            raise ValueError("content_paths must be unique members of files")
        content_scope = "sampled_shards"
    selected_key_set = {path.resolve() for path in selected_paths}
    selected_index = {path.resolve(): index for index, path in enumerate(selected_paths, start=1)}

    def emit(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    shards: list[dict[str, Any]] = []
    reference_signature: tuple[tuple[Any, ...], ...] | None = None
    reference_name: str | None = None
    total_rows = 0
    total_bytes = 0
    total_null_text = 0
    total_empty_text = 0
    total_tokenizer_empty = 0
    total_tokenized_rows = 0
    total_payload_rows = 0
    source_types: Counter[str] = Counter()
    source_distribution: Counter[str] = Counter()
    source_missing = 0
    source_examples: list[str] = []
    length_stats = _LengthStats(reservoir_size=reservoir_size, seed=17)
    sampled_footer_rows = 0
    for footer_index, path in enumerate(ordered_files, start=1):
        emit(f"[gate-b] footer {footer_index}/{len(ordered_files)}: {path.name}")
        parquet_file = parquet.ParquetFile(path)
        signature, fields = _schema_descriptor(parquet_file, path)
        if reference_signature is None:
            reference_signature, reference_name = signature, path.name
        elif signature != reference_signature:
            raise ValueError(f"Parquet schema mismatch: reference={reference_name} current={path.name}")
        metadata = getattr(parquet_file, "metadata", None)
        num_rows = getattr(metadata, "num_rows", None)
        row_groups = getattr(parquet_file, "num_row_groups", None)
        if num_rows is None or row_groups is None:
            raise ValueError(f"Parquet footer lacks num_rows/row_groups metadata: {path}")
        shard = {
            "name": path.name,
            "path": str(path),
            "row_groups": int(row_groups),
            "num_rows": int(num_rows),
            "bytes": int(path.stat().st_size),
            "schema": fields,
        }
        sample_this_shard = path.resolve() in selected_key_set
        if sample_this_shard:
            sampled_footer_rows += int(num_rows)
            emit(
                f"[gate-b] sample shard {selected_index[path.resolve()]}/{len(selected_paths)} "
                f"start: {path.name} footer_rows={int(num_rows)}"
            )
            shard_lengths = _LengthStats(reservoir_size=max(128, reservoir_size // 8), seed=19)
            shard_null_text = shard_empty_text = shard_tokenizer_empty = 0
            shard_source_missing = 0
            shard_over_context = 0
            shard_payload_rows = 0
            shard_tokenized_rows = 0
            shard_source_types: Counter[str] = Counter()
            shard_source_distribution: Counter[str] = Counter()
            shard_source_examples: list[str] = []
            last_progress_rows = 0
            for record_batch in parquet_file.iter_batches(
                batch_size=1024, columns=["text", "source"], use_threads=False
            ):
                text_values = record_batch.column("text").to_pylist()
                source_values = record_batch.column("source").to_pylist()
                for text_value, source_value in zip(text_values, source_values):
                    shard_payload_rows += 1
                    source_type = type(source_value).__name__
                    source_types[source_type] += 1
                    shard_source_types[source_type] += 1
                    if source_value is None:
                        source_key = "<None>"
                    else:
                        source_key = str(source_value)
                    _bounded_counter_add(source_distribution, source_key)
                    _bounded_counter_add(shard_source_distribution, source_key)
                    if source_value is None or not str(source_value).strip():
                        source_missing += 1
                        shard_source_missing += 1
                    elif source_key not in source_examples and len(source_examples) < 20:
                        source_examples.append(source_key)
                    if source_value is not None and str(source_value).strip():
                        if source_key not in shard_source_examples and len(shard_source_examples) < 20:
                            shard_source_examples.append(source_key)
                    if text_value is None:
                        total_null_text += 1
                        shard_null_text += 1
                    elif not str(text_value).strip():
                        total_empty_text += 1
                        shard_empty_text += 1
                    elif tokenizer is not None:
                        token_length = _token_length_for_audit(tokenizer, str(text_value))
                        shard_tokenized_rows += 1
                        total_tokenized_rows += 1
                        if not token_length:
                            total_tokenizer_empty += 1
                            shard_tokenizer_empty += 1
                        else:
                            length_stats.add(token_length)
                            shard_lengths.add(token_length)
                            if token_length > context_length:
                                shard_over_context += 1
                    if shard_payload_rows - last_progress_rows >= progress_every_rows:
                        emit(
                            f"[gate-b] sample shard {selected_index[path.resolve()]}/{len(selected_paths)} "
                            f"{path.name}: read_rows={shard_payload_rows}/{int(num_rows)} "
                            f"tokenized_rows={shard_tokenized_rows}"
                        )
                        last_progress_rows = shard_payload_rows
            emit(
                f"[gate-b] sample shard {selected_index[path.resolve()]}/{len(selected_paths)} "
                f"done: {path.name} read_rows={shard_payload_rows}/{int(num_rows)} "
                f"tokenized_rows={shard_tokenized_rows}"
            )
            shard["content"] = {
                "scope": "sampled_shard" if content_scope == "sampled_shards" else "full_shard",
                "full_corpus_values_available": content_scope == "full_corpus",
                "audited": True,
                "payload_rows": shard_payload_rows,
                "payload_rows_match_footer": shard_payload_rows == shard["num_rows"],
                "none_text": shard_null_text,
                "empty_text": shard_empty_text,
                "tokenizer_empty_text": shard_tokenizer_empty,
                "valid_trainable_rows": int(shard_payload_rows - shard_null_text - shard_empty_text - shard_tokenizer_empty),
                "source_missing": shard_source_missing,
                "source_types": dict(sorted(shard_source_types.items())),
                "source_distribution": dict(sorted(shard_source_distribution.items())),
                "source_examples": shard_source_examples,
                "token_length": shard_lengths.as_dict() if tokenizer is not None else None,
                "over_context_length": shard_over_context if tokenizer is not None else None,
            }
        elif content:
            shard["content"] = {
                "scope": "not_sampled",
                "full_corpus_values_available": False,
                "audited": False,
            }
        shards.append(shard)
        if sample_this_shard:
            total_payload_rows += int(shard["content"]["payload_rows"])
        total_rows += int(num_rows)
        total_bytes += int(path.stat().st_size)
    if reference_signature is None:
        raise AssertionError("No parquet schema was observed")
    report: dict[str, Any] = {
        "file_count": len(shards),
        "total_rows": total_rows,
        "total_bytes": total_bytes,
        "schema": [
            {"name": name, "type": field_type, "nullable": nullable}
            for name, field_type, nullable in reference_signature
        ],
        "shards": shards,
        "footer_only": not content,
        "content_audit": bool(content),
        "content_scope": content_scope,
        "content_is_full_corpus": content_scope == "full_corpus",
        "sampled_shard_count": len(selected_paths) if content_scope == "sampled_shards" else None,
        "sampled_shards": [path.name for path in selected_paths] if content_scope == "sampled_shards" else [],
        "sampled_footer_rows": sampled_footer_rows if content_scope == "sampled_shards" else None,
        "required_columns": ["text", "source"],
    }
    if content:
        valid_trainable_rows = total_payload_rows - total_null_text - total_empty_text - total_tokenizer_empty
        report["content"] = {
            "scope": content_scope,
            "full_corpus_values_available": content_scope == "full_corpus",
            "audited_shard_count": len(selected_paths),
            "payload_rows": total_payload_rows,
            "payload_rows_match_footer": (
                total_payload_rows == total_rows if content_scope == "full_corpus" else
                total_payload_rows == sampled_footer_rows
            ),
            "none_text": total_null_text,
            "empty_text": total_empty_text,
            "tokenizer_empty_text": total_tokenizer_empty,
            "tokenized_rows": total_tokenized_rows,
            "valid_trainable_rows": valid_trainable_rows,
            "source_missing": source_missing,
            "source_types": dict(sorted(source_types.items())),
            "source_distribution": dict(sorted(source_distribution.items())),
            "source_examples": source_examples,
            "token_length": length_stats.as_dict() if tokenizer is not None else None,
            "over_context_length": (
                sum(
                    int(item.get("content", {}).get("over_context_length", 0))
                    for item in shards
                    if item.get("content", {}).get("audited")
                )
                if tokenizer is not None else None
            ),
            "over_context_ratio": (
                (
                    sum(
                        int(item.get("content", {}).get("over_context_length", 0))
                        for item in shards
                        if item.get("content", {}).get("audited")
                    ) / total_tokenized_rows
                )
                if tokenizer is not None and total_tokenized_rows else None
            ),
            "over_context_ratio_denominator": "tokenized_rows",
            "interpretation": (
                "Exact full-corpus content statistics"
                if content_scope == "full_corpus"
                else "Three-shard streaming sample only; not a full-corpus estimate"
            ),
        }
    return report


def assign_shards(files: Sequence[Path], *, world_size: int, seed: int) -> dict[str, Any]:
    """Shuffle all shard names once and split contiguously by rank.

    The first ``len(files) % world_size`` ranks receive one extra shard.  For
    85 and 8 this is exactly five ranks with 11 shards and three ranks with 10
    shards.  The complete assignment is written to every Stage 4 manifest.
    """

    if len(files) != EXPECTED_PARQUET_COUNT:
        raise ValueError(f"Expected 85 shards, got {len(files)}")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    base, remainder = divmod(len(shuffled), world_size)
    rank_shards: dict[str, list[str]] = {}
    offset = 0
    for rank in range(world_size):
        count = base + (1 if rank < remainder else 0)
        rank_shards[str(rank)] = [str(path) for path in shuffled[offset : offset + count]]
        offset += count
    return {
        "seed": int(seed),
        "world_size": int(world_size),
        "shuffle": "global_fixed_seed_then_contiguous_rank_split",
        "shards_shuffled": [str(path) for path in shuffled],
        "rank_shards": rank_shards,
        "rank_shard_counts": {rank: len(paths) for rank, paths in rank_shards.items()},
    }


def _iter_shard_texts(path: Path, rng: random.Random, buffer_size: int) -> Iterator[str]:
    """Yield text values using a bounded, per-shard shuffle buffer."""

    if buffer_size <= 0:
        raise ValueError("record_buffer_size must be positive")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Stage 4 streaming requires pyarrow") from exc
    parquet_file = parquet.ParquetFile(path)
    buffer: list[str] = []
    for record_batch in parquet_file.iter_batches(
        batch_size=min(buffer_size, 1024), columns=["text"], use_threads=False
    ):
        for value in record_batch.column("text").to_pylist():
            if value is None or not str(value).strip():
                continue
            text_value = str(value)
            if len(buffer) < buffer_size:
                buffer.append(text_value)
            else:
                index = rng.randrange(len(buffer))
                yield buffer[index]
                buffer[index] = text_value
    rng.shuffle(buffer)
    yield from buffer


def collate_dynamic_padding(
    token_sequences: Sequence[Sequence[int]], *, pad_token_id: int, device: Any | None = None
) -> dict[str, Any]:
    """Pad each microbatch to its own longest sample, with padding-only mask."""

    if not token_sequences or any(len(sequence) == 0 for sequence in token_sequences):
        raise ValueError("A Stage 4 microbatch must contain non-empty token sequences")
    max_length = max(len(sequence) for sequence in token_sequences)
    if max_length > DEFAULT_CONTEXT_LENGTH:
        raise ValueError(f"A Stage 4 microbatch exceeds context_length=1024: {max_length}")
    rows = []
    masks = []
    for sequence in token_sequences:
        sequence = [int(token) for token in sequence]
        padding = max_length - len(sequence)
        rows.append(sequence + [int(pad_token_id)] * padding)
        masks.append([1] * len(sequence) + [0] * padding)
    import torch

    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels.masked_fill_(attention_mask == 0, -100)
    if torch.any(labels[attention_mask == 1] == -100):
        raise AssertionError("Non-padding tokens must remain supervised")
    if torch.any(labels[attention_mask == 0] != -100):
        raise AssertionError("Every padding position must be ignored with label -100")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class DistributedParquetStream:
    """One-rank streaming iterator; it never reads another rank's shards.

    Documents are not packed: every yielded microbatch contains exactly eight
    independent examples and dynamic padding only to that batch's maximum.
    """

    def __init__(
        self,
        rank_shards: Sequence[Path],
        tokenizer: Any,
        *,
        rank: int,
        micro_batch_size: int,
        context_length: int,
        pad_token_id: int,
        seed: int,
        record_buffer_size: int,
        device: Any,
    ) -> None:
        self.rank_shards = tuple(rank_shards)
        self.tokenizer = tokenizer
        self.rank = int(rank)
        self.micro_batch_size = int(micro_batch_size)
        self.context_length = int(context_length)
        self.pad_token_id = int(pad_token_id)
        self.seed = int(seed)
        self.record_buffer_size = int(record_buffer_size)
        self.device = device
        self.epoch = 0
        self.shard_index = 0
        self.shard_microbatches_seen = 0
        self.samples_seen = 0
        self.microbatches_seen = 0
        self._rng = random.Random(self.seed + 1009 * self.rank)
        self._resume_cursor: dict[str, Any] | None = None
        self._resume_applied = False
        if self.micro_batch_size != 2:
            raise ValueError("Stage 4 requires micro_batch_size=2")
        if not (0 < self.context_length <= 1024):
            raise ValueError("Stage 4 context_length must be in [1, 1024]")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        # Global shard order is fixed in the manifest; only records within a
        # shard are shuffled with the bounded buffer.  Epoch rollover is
        # deterministic and rank-local.  On resume, skip the recorded epoch,
        # shard, and complete microbatches before yielding new training data.
        resume = self._resume_cursor if not self._resume_applied else None
        self._resume_applied = True
        if resume:
            target_epoch = max(1, int(resume.get("epoch", 1)))
            target_shard = int(resume.get("shard_index", 0))
            target_microbatches = int(resume.get("shard_microbatches_seen", 0))
            if target_shard >= len(self.rank_shards):
                target_epoch += 1
                target_shard = 0
                target_microbatches = 0
            first_epoch = True
        else:
            target_epoch, target_shard, target_microbatches = 1, 0, 0
            first_epoch = False
        while True:
            if first_epoch:
                self.epoch = target_epoch
                shard_indices = range(target_shard, len(self.rank_shards))
                first_epoch = False
            else:
                self.epoch += 1
                shard_indices = range(len(self.rank_shards))
            for index in shard_indices:
                shard_path = self.rank_shards[index]
                self.shard_index = index
                self.shard_microbatches_seen = target_microbatches if resume and index == target_shard else 0
                skip_microbatches = target_microbatches if resume and index == target_shard else 0
                batch: list[list[int]] = []
                for text_value in _iter_shard_texts(shard_path, self._rng, self.record_buffer_size):
                    token_ids = _tokenize_ids(self.tokenizer, text_value, self.context_length)
                    if not token_ids:
                        continue
                    batch.append(token_ids)
                    if len(batch) == self.micro_batch_size:
                        if skip_microbatches > 0:
                            # A bounded shuffle buffer is not bitwise
                            # reconstructible, but complete microbatch skip is
                            # deterministic and auditable at shard granularity.
                            skip_microbatches -= 1
                            batch = []
                            continue
                        result = collate_dynamic_padding(
                            batch, pad_token_id=self.pad_token_id, device=self.device
                        )
                        self.samples_seen += self.micro_batch_size
                        self.microbatches_seen += 1
                        self.shard_microbatches_seen += 1
                        yield result
                        batch = []
                # Deliberately drop an incomplete microbatch.  The training
                # loop performs all_reduce MIN before every accumulation slot,
                # so all ranks terminate together instead of deadlocking.
            self.shard_index = len(self.rank_shards)
            self.shard_microbatches_seen = 0
            resume = None

    def restore_cursor(self, cursor: Mapping[str, Any]) -> None:
        """Restore coarse epoch/shard/microbatch position for resume."""

        self.epoch = int(cursor.get("epoch", 0))
        self.shard_index = int(cursor.get("shard_index", 0))
        self.shard_microbatches_seen = int(cursor.get("shard_microbatches_seen", 0))
        self.samples_seen = int(cursor.get("samples_seen", 0))
        self.microbatches_seen = int(cursor.get("microbatches_seen", 0))
        shuffle_state = cursor.get("bounded_shuffle_rng_state")
        if shuffle_state is not None:
            self._rng.setstate(_tupleize(shuffle_state))
        self._resume_cursor = dict(cursor)
        self._resume_applied = False

    def cursor(self, *, include_rng: bool = True) -> dict[str, Any]:
        cursor = {
            "epoch": self.epoch,
            "shard_index": self.shard_index,
            "shard_microbatches_seen": self.shard_microbatches_seen,
            "samples_seen": self.samples_seen,
            "microbatches_seen": self.microbatches_seen,
            "bounded_shuffle_bitwise_exact": False,
        }
        if include_rng:
            cursor["bounded_shuffle_rng_state"] = _json_safe(self._rng.getstate())
        return cursor


def _tokenizer_pad(tokenizer: Any) -> int:
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token", None) is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        # This mirrors Stage 2's explicit synthetic-pad audit marker.
        setattr(tokenizer, "_stage4_synthetic_pad_token", True)
    return int(tokenizer.pad_token_id)


def _set_seed(seed: int, rank: int = 0) -> None:
    import torch

    random.seed(int(seed) + rank)
    try:
        import numpy as np

        np.random.seed((int(seed) + rank) % (2**32 - 1))
    except ImportError:
        pass
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _bf16_autocast(device: Any) -> Any:
    import torch

    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _causal_loss_sum(logits: Any, labels: Any) -> tuple[Any, Any]:
    """Return unreduced NTP loss sum and valid shifted-token count."""

    import torch.nn.functional as functional

    # Standard NTP shift: logits[:,:-1] supervises labels[:,1:].  There is
    # no source/prefix mask; only dynamic padding labels are -100.
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:].contiguous()
    valid_count = (shift_labels != -100).sum()
    loss_sum = functional.cross_entropy(
        shift_logits.contiguous().float().view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum, valid_count


def token_weighted_gradient_scale(*, world_size: int, global_window_tokens: int) -> float:
    """Scale one unreduced local loss sum for DDP token-weighted accumulation.

    DDP averages the per-rank gradients.  Therefore each rank contributes
    ``world_size / global_window_tokens`` times its local loss sum.  The
    denominator already covers every microbatch in the accumulation window;
    dividing by ``gradient_accumulation_steps`` here would scale the gradient
    down a second time.
    """

    world_size = int(world_size)
    global_window_tokens = int(global_window_tokens)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if global_window_tokens <= 0:
        raise ValueError("global_window_tokens must be positive")
    return float(world_size) / float(global_window_tokens)


def _all_reduce_scalar(value: Any, *, op: Any | None = None) -> Any:
    import torch.distributed as dist

    tensor = value if hasattr(value, "device") else __import__("torch").tensor(value)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=op or dist.ReduceOp.SUM)
    return tensor


def _all_reduce_min_flag(flag: bool, device: Any) -> bool:
    import torch
    import torch.distributed as dist

    value = torch.tensor(1 if flag else 0, device=device, dtype=torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return bool(int(value.item()))


def _trainable_parameters(model: Any) -> tuple[Any, ...]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    ids = [id(parameter) for parameter in parameters]
    if len(ids) != len(set(ids)):
        raise AssertionError("model.parameters() contains duplicate trainable Parameter objects")
    return parameters


def optimizer_parameter_audit(optimizer: Any, parameters: Sequence[Any]) -> dict[str, Any]:
    """Ensure AdamW contains each shared Parameter object exactly once."""

    if type(optimizer).__name__ != "AdamW":
        raise AssertionError(f"Stage 4 requires AdamW, got {type(optimizer).__name__}")
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    parameter_ids = [id(parameter) for parameter in parameters]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    exact = (
        len(parameter_ids) == len(set(parameter_ids))
        and optimizer_ids == parameter_ids
        and len(optimizer_ids) == len(set(optimizer_ids))
    )
    if not exact:
        raise AssertionError("AdamW must contain each unique trainable Parameter exactly once")
    if len(optimizer.param_groups) != 1:
        raise AssertionError("Stage 4 requires exactly one AdamW parameter group")
    group = optimizer.param_groups[0]
    betas = tuple(float(value) for value in group.get("betas", ()))
    eps = float(group.get("eps", float("nan")))
    amsgrad = bool(group.get("amsgrad", False))
    weight_decay = float(group.get("weight_decay", float("nan")))
    if betas != DEFAULT_ADAMW_BETAS:
        raise AssertionError(f"AdamW betas mismatch: expected={DEFAULT_ADAMW_BETAS} got={betas}")
    if eps != DEFAULT_ADAMW_EPS:
        raise AssertionError(f"AdamW eps mismatch: expected={DEFAULT_ADAMW_EPS} got={eps}")
    if amsgrad is not DEFAULT_ADAMW_AMSGRAD:
        raise AssertionError(f"AdamW amsgrad mismatch: expected={DEFAULT_ADAMW_AMSGRAD} got={amsgrad}")
    if weight_decay != DEFAULT_ADAMW_WEIGHT_DECAY:
        raise AssertionError(
            f"AdamW weight_decay mismatch: expected={DEFAULT_ADAMW_WEIGHT_DECAY} got={weight_decay}"
        )
    return {
        "trainable_parameter_count": len(parameters),
        "optimizer_parameter_count": len(optimizer_parameters),
        "optimizer_matches_model_exactly_once": True,
        "optimizer_type": type(optimizer).__name__,
        "betas": list(betas),
        "eps": eps,
        "weight_decay": weight_decay,
        "amsgrad": amsgrad,
    }


def _parameter_checksum(model: Any) -> Any:
    import torch

    checksum = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float64)
    for parameter in model.parameters():
        checksum = checksum + parameter.detach().float().sum(dtype=torch.float64)
    return checksum


def _assert_rank_equal(value: Any, *, name: str, device: Any, tolerance: float = 1e-5) -> float:
    import torch
    import torch.distributed as dist

    scalar = value.detach().float().to(device=device).reshape(()) if hasattr(value, "detach") else torch.tensor(float(value), device=device)
    if dist.is_available() and dist.is_initialized():
        lo = scalar.clone()
        hi = scalar.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        if float((hi - lo).abs().item()) > tolerance:
            raise AssertionError(f"{name} differs across ranks: min={lo.item()} max={hi.item()}")
    return float(scalar.item())


def _load_runtime_model(model_path: Path, device: Any, tokenizer_path: Path | None = None) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from code.RSmol.recursive_model_5_10xr_5 import register_auto_class

    model_path = _require_local_artifact_dir(model_path, kind="recursive model")
    tokenizer_source = _require_local_artifact_dir(tokenizer_path or model_path, kind="tokenizer")
    register_auto_class()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=__import__("torch").float32,
    )
    # No device_map: every rank owns one local process/device and DDP performs
    # synchronization explicitly.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=True,
        use_fast=True,
    )
    model.to(device=device, dtype=__import__("torch").float32)
    model.config.use_cache = False
    model.train()
    _validate_recursive_architecture(model)
    return model, tokenizer


def _validate_recursive_architecture(model: Any, *, production: bool = True) -> dict[str, Any]:
    recursive_model = getattr(model, "model", model)
    config = getattr(model, "config", None)
    source_logical = int(getattr(config, "recursive_source_num_hidden_layers", 0))
    source_physical = int(getattr(config, "recursive_source_layer_count", 0))
    logical = int(getattr(config, "num_hidden_layers", 0))
    physical = int(getattr(config, "recursive_layer_count", 0))
    loops = int(getattr(config, "recursive_loops", 0))
    schedule = list(getattr(config, "logical_to_physical", getattr(recursive_model, "logical_to_physical", ())))
    source_mapping = list(getattr(config, "recursive_source_layer_indices_0based", ()))
    loops_scope = str(getattr(config, "recursive_loops_scope", ""))
    expected_source_mapping = [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29]
    expected_schedule = list(LOGICAL_TO_PHYSICAL)
    if (source_logical, source_physical) != (30, 30) or loops != 10 or loops_scope != "middle_only" or (logical, physical) != (110, 20) or schedule != expected_schedule or source_mapping != expected_source_mapping:
        raise ValueError(
            "Stage 4 5-10xr-5-Poisson requires source=30, logical=110 physical=20 max loops=10 and the explicit schedule; "
            f"got source_logical={source_logical}, source_physical={source_physical}, logical={logical}, "
            f"physical={physical}, loops={loops}, loops_scope={loops_scope}, schedule={schedule}, source_mapping={source_mapping}"
        )
    if len(getattr(recursive_model, "layers", ())) != physical:
        raise ValueError("Recursive model physical layer module count does not match config")
    dynamic_metadata = (
        int(getattr(config, "recursive_min_middle_loops", -1)),
        int(getattr(config, "recursive_max_middle_loops", -1)),
        int(getattr(config, "recursive_default_inference_middle_loops", -1)),
        int(getattr(config, "recursive_parameter_gradient_tail_loops", -1)),
    )
    if dynamic_metadata != (4, 10, 7, 4):
        raise ValueError(f"Invalid dynamic 5-10xr-5 metadata: {dynamic_metadata}")
    config_sampling = {
        "sampling_policy": str(getattr(config, "recursive_sampling_policy", "")),
        "sampler_version": str(getattr(config, "recursive_sampler_version", "")),
        "sampler_key": str(getattr(config, "recursive_sampler_key", "")),
        "poisson_lambda": float(getattr(config, "recursive_poisson_lambda", -1.0)),
        "poisson_support": list(getattr(config, "recursive_poisson_support", ())),
        "poisson_probabilities": list(getattr(config, "recursive_poisson_probabilities", ())),
    }
    try:
        _validate_exact_sampling_contract(config_sampling, label="checkpoint")
    except (TypeError, ValueError):
        raise ValueError(
            "Stage 4 5-10xr-5-Poisson checkpoint sampling metadata mismatch: "
            f"metadata={config_sampling!r}"
        )
    try:
        named = list(model.named_parameters(remove_duplicate=False))
    except TypeError:
        named = list(model.named_parameters())
    ids = [id(parameter) for _, parameter in named]
    unique = len(set(ids))
    if unique != len({id(parameter) for parameter in model.parameters()}):
        raise AssertionError("Parameter identity audit failed")
    if not all(parameter.dtype == __import__("torch").float32 for parameter in model.parameters()):
        raise AssertionError("Stage 4 requires FP32 parameters")
    try:
        from code.RSmol.recursive_model_5_10xr_5 import parameter_audit

        recursive_parameter_audit = parameter_audit(model)
    except ImportError:
        recursive_parameter_audit = {"available": False}
    return {
        "logical_layer_count": logical,
        "source_logical_layer_count": source_logical,
        "source_physical_layer_count": source_physical,
        "physical_layer_count": physical,
        "recursive_loops": loops,
        "min_middle_loops": int(getattr(config, "recursive_min_middle_loops", 4)),
        "max_middle_loops": int(getattr(config, "recursive_max_middle_loops", 10)),
        "default_inference_middle_loops": int(getattr(config, "recursive_default_inference_middle_loops", 7)),
        "parameter_gradient_tail_loops": int(getattr(config, "recursive_parameter_gradient_tail_loops", 4)),
        "sampling_policy": SAMPLING_POLICY,
        "sampler_version": SAMPLER_VERSION,
        "poisson_lambda": POISSON_LAMBDA,
        "poisson_support": list(POISSON_SUPPORT),
        "poisson_probabilities": list(POISSON_PROBABILITIES),
        "sampler_key": SAMPLER_KEY,
        "physical_module_count": len(recursive_model.layers),
        "parameter_storage_unique": unique,
        "parameters_fp32": True,
        "use_cache": bool(getattr(config, "use_cache", True)),
        "forward_order": list(LOGICAL_TO_PHYSICAL),
        "logical_to_physical": list(LOGICAL_TO_PHYSICAL),
        "prefix_layer_count": 5,
        "middle_recurrent_count": 10,
        "suffix_layer_count": 5,
        "architecture_contract": MODEL_ARCHITECTURE_CONTRACT,
        "production": production,
        "recursive_parameter_audit": recursive_parameter_audit,
    }


class RuntimeMonitor:
    """Append-only process/resource monitor for long remote pilots."""

    def __init__(self, output_path: Path, *, interval_seconds: float, rank: int, device: Any) -> None:
        self.output_path = output_path
        self.interval_seconds = float(interval_seconds)
        self.rank = int(rank)
        self.device = device
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    @staticmethod
    def _proc_stats() -> dict[str, Any]:
        result: dict[str, Any] = {"rss_bytes": None, "fd_count": None, "mmap_count": None}
        status_path = Path("/proc/self/status")
        if status_path.exists():
            text = status_path.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if line.startswith("VmRSS:"):
                    result["rss_bytes"] = int(line.split()[1]) * 1024
        fd_path = Path("/proc/self/fd")
        mmap_path = Path("/proc/self/maps")
        if fd_path.exists():
            result["fd_count"] = len(list(fd_path.iterdir()))
        if mmap_path.exists():
            with mmap_path.open("r", encoding="utf-8", errors="replace") as mmap_stream:
                result["mmap_count"] = sum(1 for _ in mmap_stream)
        return result

    def _sample(self) -> dict[str, Any]:
        import torch

        sample: dict[str, Any] = {
            "timestamp": time.time(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "rank": self.rank,
            "device": str(self.device),
            "faulthandler_enabled": bool(faulthandler.is_enabled()),
            **self._proc_stats(),
            "shm_free_bytes": None,
            "tmp_free_bytes": None,
        }
        for key, path in (("shm_free_bytes", Path("/dev/shm")), ("tmp_free_bytes", Path("/tmp"))):
            try:
                sample[key] = int(shutil.disk_usage(path).free)
            except OSError:
                pass
        if torch.cuda.is_available() and getattr(self.device, "type", "") == "cuda":
            sample["gpu_memory_allocated_bytes"] = int(torch.cuda.memory_allocated(self.device))
            sample["gpu_memory_reserved_bytes"] = int(torch.cuda.memory_reserved(self.device))
            sample["gpu_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(self.device))
        else:
            sample["gpu_memory_allocated_bytes"] = None
            sample["gpu_memory_reserved_bytes"] = None
            sample["gpu_max_memory_allocated_bytes"] = None
        return sample

    def _run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        while not self.stop_event.is_set():
            with self.output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(self._sample(), ensure_ascii=False) + "\n")
                stream.flush()
            self.stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        if self.interval_seconds <= 0:
            return
        self.thread = threading.Thread(target=self._run, name=f"stage4-monitor-rank{self.rank}", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)


def _register_forward_trace(model: Any) -> tuple[list[int], list[Any]]:
    import torch

    recursive_model = getattr(model, "module", model)
    recursive_model = getattr(recursive_model, "model", recursive_model)
    sequence: list[int] = []
    handles: list[Any] = []
    for index, layer in enumerate(getattr(recursive_model, "layers", ())):
        handles.append(layer.register_forward_hook(lambda _module, _inputs, _output, i=index: sequence.append(i)))
    return sequence, handles


def recursive_forward_trace_audit(model: Any, *, device: Any, middle_loop_count: int = DEFAULT_INFERENCE_MIDDLE_LOOPS) -> dict[str, Any]:
    """Verify the exact dynamic physical execution trace for one r."""

    import torch

    base_model = getattr(model, "module", model)
    recursive_base = getattr(base_model, "model", base_model)
    physical = len(getattr(recursive_base, "layers", ()))
    sequence, handles = _register_forward_trace(model)
    training_state = bool(base_model.training)
    base_model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad(), _bf16_autocast(device):
        base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, middle_loop_count=middle_loop_count)
    for handle in handles:
        handle.remove()
    base_model.train(training_state)
    from code.RSmol.recursive_model_5_10xr_5 import build_5_10xr_5_schedule
    expected = list(build_5_10xr_5_schedule(middle_loop_count))
    if sequence != expected:
        raise AssertionError(f"Recursive forward trace mismatch: expected={expected} got={sequence}")
    return {
        "trace": sequence,
        "expected": expected,
        "logical_layers": LOGICAL_LAYER_COUNT,
        "physical_shared_layers": physical,
        "loops": middle_loop_count,
        "forward_trace_ok": True,
    }


def _synthetic_batch(*, rank: int, device: Any, vocab_size: int) -> dict[str, Any]:
    # Unequal lengths explicitly exercise dynamic padding and label masking.
    rows = [[3 + rank, 5, 7, 11, 13, 17], [19, 23, 29, 31 + rank]]
    rows = [[token % vocab_size for token in row] for row in rows]
    return collate_dynamic_padding(rows, pad_token_id=0, device=device)


def _selective_middle_gradient_audit(
    recursive_base: Any, *, middle_loop_count: int
) -> dict[str, Any]:
    """Prove one forward has hidden edges through early calls only.

    This probe is intentionally run before the regular loss backward.  For
    each early middle output, ``autograd.grad`` must find a finite nonzero
    derivative with respect to that call's hidden input while finding no edge
    to the shared layer Parameters.  The normal backward then traverses the
    complete schedule and accumulates the final four calls' parameter grads.
    """

    import torch

    details = list(getattr(recursive_base, "_last_middle_gradient_audit", ()))
    expected_entries = int(middle_loop_count) * MIDDLE_LAYER_COUNT
    if len(details) != expected_entries:
        raise AssertionError(
            "selective middle-gradient audit did not capture every physical middle call: "
            f"expected={expected_entries} (10r) got={len(details)}"
        )
    expected_enabled = list(
        range(int(middle_loop_count) - PARAMETER_GRADIENT_TAIL_LOOPS + 1, int(middle_loop_count) + 1)
    )
    expected_disabled = list(range(1, int(middle_loop_count) - PARAMETER_GRADIENT_TAIL_LOOPS + 1))
    early_input_gradient_norms: dict[str, float] = {}
    middle_physical_indices = list(range(5, 5 + MIDDLE_LAYER_COUNT))
    middle_call_audit_keys: list[str] = []
    for loop in range(1, int(middle_loop_count) + 1):
        loop_details = [detail for detail in details if int(detail["loop"]) == loop]
        physical_trace = [int(detail["physical_index"]) for detail in loop_details]
        if physical_trace != middle_physical_indices:
            raise AssertionError(
                f"middle loop {loop} must contain physical layers 5..14 exactly once; "
                f"got {physical_trace}"
            )
        for detail in loop_details:
            physical = int(detail["physical_index"])
            key = f"{loop}:{physical}"
            middle_call_audit_keys.append(key)
            expected = loop in expected_enabled
            if bool(detail["parameter_grad_enabled"]) != expected:
                raise AssertionError(
                    f"middle loop {key} parameter-gradient flag mismatch: "
                    f"expected={expected} detail={detail}"
                )
            if expected:
                continue
            probe = detail["output"].float().square().mean()
            input_gradient = torch.autograd.grad(
                probe, detail["input"], retain_graph=True, allow_unused=True
            )[0]
            if (
                input_gradient is None
                or not torch.isfinite(input_gradient).all()
                or float(input_gradient.norm().item()) <= 0
            ):
                raise AssertionError(f"early middle loop {key} lost its hidden-state gradient path")
            parameter_gradients = torch.autograd.grad(
                probe,
                tuple(detail["layer"].parameters()),
                retain_graph=True,
                allow_unused=True,
            )
            if any(gradient is not None for gradient in parameter_gradients):
                raise AssertionError(f"early middle loop {key} unexpectedly has parameter-gradient edges")
            early_input_gradient_norms[key] = float(input_gradient.norm().item())
    return {
        "all_middle_calls_captured": True,
        "middle_call_count": expected_entries,
        "middle_call_audit_keys": middle_call_audit_keys,
        "each_middle_loop_has_exactly_ten_physical_calls": True,
        "backward_traversed_middle_loops": list(range(1, int(middle_loop_count) + 1)),
        "parameter_gradient_enabled_middle_loops": expected_enabled,
        "parameter_gradient_disabled_middle_loops": expected_disabled,
        "early_hidden_input_gradient_norms": early_input_gradient_norms,
        "early_parameter_gradient_edges_absent": True,
        "exact_parameter_gradient_tail": len(expected_enabled) == PARAMETER_GRADIENT_TAIL_LOOPS,
    }


def _backward_trace_coverage_matches(actual: Sequence[int], expected: Sequence[int]) -> bool:
    """Check backward-hook coverage without assuming PyTorch hook ordering.

    ``functional_call`` with detached parameter views preserves the hidden-state
    autograd path but does not guarantee the same module-hook callback order as
    a regular parameter-owning call.  Coverage and multiplicity are the stable
    contract; selective gradient probes above verify the actual dependency path.
    """

    return len(actual) == len(expected) and Counter(int(value) for value in actual) == Counter(
        int(value) for value in expected
    )


def _all_r_backward_audit(
    model: Any,
    recursive_base: Any,
    *,
    batch: Mapping[str, Any],
    device: Any,
) -> list[dict[str, Any]]:
    """Run a dependency-light backward-path audit for every supported r."""

    import torch

    audits: list[dict[str, Any]] = []
    labels = batch["labels"]
    for middle_loop_count in range(MIN_MIDDLE_LOOPS, MAX_MIDDLE_LOOPS + 1):
        model.train()
        model.zero_grad(set_to_none=True)
        backward_sequence: list[int] = []
        handles = [
            layer.register_full_backward_hook(
                lambda _module, _grad_input, _grad_output, index=index: backward_sequence.append(index)
            )
            for index, layer in enumerate(recursive_base.layers)
        ]
        try:
            recursive_base._collect_middle_gradient_audit = True
            with _bf16_autocast(device):
                result = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                    middle_loop_counts=torch.full(
                        (batch["input_ids"].shape[0],), middle_loop_count,
                        dtype=torch.long, device=device,
                    ),
                )
                selective = _selective_middle_gradient_audit(
                    recursive_base, middle_loop_count=middle_loop_count
                )
                loss_sum, _valid_count = _causal_loss_sum(result.logits, labels)
            recursive_base._collect_middle_gradient_audit = False
            loss_sum.backward()
        finally:
            recursive_base._collect_middle_gradient_audit = False
            for handle in handles:
                handle.remove()
        from code.RSmol.recursive_model_5_10xr_5 import build_5_10xr_5_schedule

        expected_forward = list(build_5_10xr_5_schedule(middle_loop_count))
        expected_backward = list(reversed(expected_forward))
        if not _backward_trace_coverage_matches(backward_sequence, expected_backward):
            raise AssertionError(
                f"r={middle_loop_count} backward trace coverage mismatch: "
                f"expected_counts={dict(Counter(expected_backward))} "
                f"got_counts={dict(Counter(backward_sequence))}"
            )
        if not all(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and float(parameter.grad.detach().norm().item()) > 0
            for layer in recursive_base.layers
            for parameter in layer.parameters()
        ):
            raise AssertionError(f"r={middle_loop_count} has missing/non-finite/zero layer gradients")
        audits.append(
            {
                "r": middle_loop_count,
                "forward_trace": expected_forward,
                "backward_trace": backward_sequence,
                "backward_trace_ok": True,
                "backward_trace_order": (
                    "exact_reverse_schedule"
                    if backward_sequence == expected_backward
                    else "functional_call_early_hook_order"
                ),
                "backward_trace_coverage_ok": True,
                "selective_middle_gradient_audit": selective,
            }
        )
    model.zero_grad(set_to_none=True)
    return audits


def _synthetic_gate_a(model: Any, *, device: Any, config: Stage4Config, rank: int) -> dict[str, Any]:
    """Audit DDP initialization, recursive trace, masks, gradients and update."""

    import torch
    import torch.distributed as dist

    base_model = getattr(model, "module", model)
    base_model.config.use_cache = False
    sampled_counts = sample_middle_loop_counts(
        config.seed, rank, 0, 0, config.micro_batch_size
    )
    sampled_r = int(sampled_counts.max().item())
    parameters = _trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=DEFAULT_ADAMW_BETAS,
        eps=DEFAULT_ADAMW_EPS,
        weight_decay=DEFAULT_ADAMW_WEIGHT_DECAY,
        amsgrad=DEFAULT_ADAMW_AMSGRAD,
    )
    optimizer_audit = optimizer_parameter_audit(optimizer, parameters)
    recursive_base = getattr(base_model, "model", base_model)
    batch = _synthetic_batch(rank=rank, device=device, vocab_size=config.synthetic_vocab_size)
    labels = batch["labels"]
    if labels[0, 0].item() == -100 or labels[1, 1].item() == -100:
        raise AssertionError("Synthetic audit masked a non-padding label")
    all_r_backward_audits = _all_r_backward_audit(
        model, recursive_base, batch=batch, device=device
    )
    # Install the sampled-r trace hooks only after the all-r pre-audit.  This
    # keeps the 64-microbatch trace free of the seven standalone all-r audit runs.
    sequence, handles = _register_forward_trace(model)
    backward_sequence: list[int] = []
    backward_handles = [
        layer.register_full_backward_hook(
            lambda _module, _grad_input, _grad_output, i=index: backward_sequence.append(i)
        )
        for index, layer in enumerate(recursive_base.layers)
    ]
    optimizer.zero_grad(set_to_none=True)
    local_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    local_tokens = torch.zeros((), device=device, dtype=torch.int64)
    microbatches = 0
    sampled_counts_history: list[list[int]] = []
    sampled_tmax_history: list[int] = []
    selective_gradient_audit: dict[str, Any] | None = None
    # Obtain every microbatch's global valid-token denominator before any
    # backward pass, then use the window-global denominator for token-weighted
    # accumulation under DDP's default gradient averaging.
    local_valid_count = (labels[:, 1:] != -100).sum().to(dtype=torch.int64)
    global_counts = [int(_all_reduce_scalar(local_valid_count.detach().clone()).item()) for _ in range(config.gradient_accumulation_steps)]
    window_global_tokens = sum(global_counts)
    if window_global_tokens <= 0:
        raise FloatingPointError("Synthetic accumulation window has no supervised tokens")
    for micro_index in range(config.gradient_accumulation_steps):
        micro_counts = sample_middle_loop_counts(
            config.seed, rank, 0, micro_index, config.micro_batch_size
        )
        micro_tmax = int(micro_counts.max().item())
        sampled_counts_history.append([int(value) for value in micro_counts.tolist()])
        sampled_tmax_history.append(micro_tmax)
        recursive_base._collect_middle_gradient_audit = micro_index == 0
        with _bf16_autocast(device):
            result = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                middle_loop_counts=micro_counts.to(device=device),
            )
            if micro_index == 0:
                selective_gradient_audit = _selective_middle_gradient_audit(
                    recursive_base, middle_loop_count=micro_tmax
                )
        recursive_base._collect_middle_gradient_audit = False
        loss_sum, valid_count = _causal_loss_sum(result.logits, labels)
        if not torch.isfinite(loss_sum):
            raise FloatingPointError("Synthetic loss is non-finite")
        scale = token_weighted_gradient_scale(
            world_size=config.world_size,
            global_window_tokens=window_global_tokens,
        )
        (loss_sum * scale).backward()
        local_loss_sum += loss_sum.detach().double()
        local_tokens += valid_count.detach()
        microbatches += 1
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
    if not torch.isfinite(torch.as_tensor(gradient_norm)):
        raise FloatingPointError("Synthetic gradient norm is non-finite")
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer.step()
    update_norm = float(sum(torch.linalg.vector_norm(parameter.detach() - old).item() for parameter, old in zip(parameters, before)))
    if update_norm <= 0:
        raise AssertionError("Synthetic optimizer did not update parameters")
    for handle in handles + backward_handles:
        handle.remove()
    if selective_gradient_audit is None:
        raise AssertionError("Gate A did not run the selective middle-gradient audit")
    physical = len(recursive_base.layers)
    from code.RSmol.recursive_model_5_10xr_5 import build_5_10xr_5_schedule
    forward_expected = [list(build_5_10xr_5_schedule(tmax)) for tmax in sampled_tmax_history]
    backward_expected = [list(reversed(chunk)) for chunk in forward_expected]
    forward_chunks, backward_chunks = [], []
    forward_offset = backward_offset = 0
    for expected_forward, expected_backward in zip(forward_expected, backward_expected):
        forward_chunks.append(sequence[forward_offset:forward_offset + len(expected_forward)])
        backward_chunks.append(backward_sequence[backward_offset:backward_offset + len(expected_backward)])
        forward_offset += len(expected_forward)
        backward_offset += len(expected_backward)
    expected_trace_length = sum(len(chunk) for chunk in forward_expected)
    trace_forward_ok = (
        len(sequence) == expected_trace_length
        and len(forward_chunks) == config.gradient_accumulation_steps
        and all(actual == expected for actual, expected in zip(forward_chunks, forward_expected))
    )
    trace_backward_ok = (
        len(backward_sequence) == expected_trace_length
        and len(backward_chunks) == config.gradient_accumulation_steps
        and all(
            _backward_trace_coverage_matches(actual, expected)
            for actual, expected in zip(backward_chunks, backward_expected)
        )
    )
    backward_exact_order = (
        len(backward_sequence) == expected_trace_length
        and len(backward_chunks) == config.gradient_accumulation_steps
        and all(actual == expected for actual, expected in zip(backward_chunks, backward_expected))
    )
    if not trace_forward_ok:
        raise AssertionError(
            "Forward trace does not match the per-microbatch local-Tmax schedules: "
            f"expected_length={expected_trace_length} got_length={len(sequence)} chunks={forward_chunks}"
        )
    if not trace_backward_ok:
        raise AssertionError(
            "Backward does not match the per-microbatch reversed local-Tmax schedules: "
            f"expected_length={expected_trace_length} got_length={len(backward_sequence)} chunks={backward_chunks}"
        )
    layer_gradients = {}
    for index, layer in enumerate(recursive_base.layers):
        grads = [parameter.grad for parameter in layer.parameters()]
        norms = [float(torch.linalg.vector_norm(grad.detach()).item()) for grad in grads if grad is not None]
        if not norms or not all(math.isfinite(value) and value > 0 for value in norms):
            raise AssertionError(f"Physical layer {index} has missing/non-finite/zero gradient")
        layer_gradients[str(index)] = {"norm": max(norms), "finite_nonzero": True}
    prefix_layers_with_grad = [
        index for index in range(5)
        if layer_gradients.get(str(index), {}).get("finite_nonzero") is True
    ]
    if prefix_layers_with_grad != [0, 1, 2, 3, 4]:
        raise AssertionError(
            f"Gate A prefix layers must all receive finite nonzero gradients: {prefix_layers_with_grad}"
        )
    suffix_layers_with_grad = [
        index for index in range(15, 20)
        if layer_gradients.get(str(index), {}).get("finite_nonzero") is True
    ]
    if suffix_layers_with_grad != [15, 16, 17, 18, 19]:
        raise AssertionError(
            f"Gate A suffix layers must all receive finite nonzero gradients: {suffix_layers_with_grad}"
        )
    checksum = _parameter_checksum(model)
    if dist.is_available() and dist.is_initialized():
        lo, hi = checksum.clone(), checksum.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        if not torch.allclose(lo, hi, atol=1e-5, rtol=1e-5):
            raise AssertionError("DDP parameter checksum differs across ranks")
    global_loss = _all_reduce_scalar(local_loss_sum.clone())
    global_count = _all_reduce_scalar(local_tokens.clone())
    result = {
        "status": "PASS",
        "gate": "A",
        "rank": rank,
        "world_size": config.world_size,
        "backend": dist.get_backend() if dist.is_initialized() else None,
        "ddp_initialized": bool(dist.is_initialized()),
        "recursive_forward_trace": sequence[:LOGICAL_LAYER_COUNT],
        "recursive_forward_trace_expected": forward_expected[0],
        "recursive_backward_trace": backward_sequence[:LOGICAL_LAYER_COUNT],
        "recursive_backward_trace_expected": backward_expected[0],
        "sampled_r_forward_trace_total_length": len(sequence),
        "sampled_r_backward_trace_total_length": len(backward_sequence),
        "expected_sampled_r_trace_total_length": expected_trace_length,
        "sampled_r_trace_chunk_count": len(forward_chunks),
        "all_microbatches_same_r_trace": False,
        "all_microbatches_match_local_tmax_trace": bool(trace_forward_ok and trace_backward_ok),
        "forward_trace_ok": trace_forward_ok,
        "backward_second_loop_then_first": backward_exact_order,
        "backward_trace_coverage_ok": trace_backward_ok,
        "backward_trace_order_relaxed_for_functional_call": not backward_exact_order,
        "logical_layers": len(forward_expected[0]),
        "physical_layers": physical,
        "recursive_loops": sampled_r,
        "sampled_r": sampled_r,
        "sampled_middle_loop_counts_per_microbatch": sampled_counts_history,
        "sampled_local_tmax_per_microbatch": sampled_tmax_history,
        "sampling_policy": SAMPLING_POLICY,
        "sampler_metadata": poisson_metadata(),
        "all_rank_r_equal": False,
        "backward_traversed_middle_loops": list(range(1, sampled_r + 1)),
        "parameter_gradient_enabled_middle_loops": list(range(max(1, sampled_r - 3), sampled_r + 1)),
        "parameter_gradient_disabled_middle_loops": list(range(1, max(1, sampled_r - 3))),
        "label_shift": "logits[..., :-1, :] vs labels[..., 1:]",
        "padding_only_label_mask": True,
        "global_loss_sum": float(global_loss.item()),
        "global_valid_token_count": int(global_count.item()),
        "global_window_valid_token_denominator": window_global_tokens,
        "global_loss_count_consistent": True,
        "loss_finite": True,
        "gradient_norm": float(torch.as_tensor(gradient_norm).item()),
        "gradient_finite_nonzero": True,
        "physical_layer_gradients": layer_gradients,
        "prefix_layers_with_grad": prefix_layers_with_grad,
        "prefix_all_receive_finite_nonzero_grad": True,
        "suffix_layers_with_grad": suffix_layers_with_grad,
        "suffix_all_receive_finite_nonzero_grad": True,
        "optimizer_step_calls": 1,
        "optimizer_audit": optimizer_audit,
        "selective_middle_gradient_audit": selective_gradient_audit,
        "all_r_backward_audits": all_r_backward_audits,
        "exact_microbatches_per_update": microbatches == config.gradient_accumulation_steps,
        "optimizer_updated_parameters": True,
        "parameter_checksum": float(checksum.item()),
        "rank_checksum_consistent": True,
        "rank0_only_checkpoint": True,
        "parameters_fp32": all(parameter.dtype == torch.float32 for parameter in base_model.parameters()),
        "bf16_autocast": True,
        "use_cache": False,
        "find_unused_parameters": False,
        "no_device_map": True,
    }
    return result


# Public descriptive alias used by external audit harnesses.
def gate_a_synthetic_ddp_audit(model: Any, *, device: Any, config: Stage4Config, rank: int) -> dict[str, Any]:
    return _synthetic_gate_a(model, device=device, config=config, rank=rank)


def _prepare_distributed(config: Stage4Config) -> tuple[int, int, Any, bool]:
    import torch
    import torch.distributed as dist

    # Gate B is intentionally a standalone single-process audit.  Cluster
    # launchers may leave WORLD_SIZE/RANK/LOCAL_RANK inherited in the
    # environment even when no torchrun process group was started; never let
    # those stale variables turn Gate B into a false distributed launch.
    if config.gate == "B":
        rank = 0
        world_size = 1
        local_rank = 0
    else:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", str(config.world_size)))
        local_rank = int(os.environ.get("LOCAL_RANK", str(config.local_rank if config.local_rank >= 0 else rank)))
    config.world_size = world_size
    config.local_rank = local_rank
    if config.gate == "FORMAL":
        if world_size != DEFAULT_WORLD_SIZE:
            raise ValueError(f"FORMAL requires world_size=8, got {world_size}")
        if str(config.backend).lower() != "nccl":
            raise ValueError(f"FORMAL requires backend=nccl, got {config.backend!r}")
        if config.allow_non8:
            raise ValueError("FORMAL does not allow --allow-non8")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        backend = config.backend
        if backend == "nccl" and not use_cuda:
            raise RuntimeError("NCCL Stage 4 requires CUDA; use backend=gloo only for CPU static/unit audit")
        print(
            f"[stage4-lifecycle][rank={rank}] phase=distributed_init_start "
            f"backend={backend} world_size={world_size} local_rank={local_rank}",
            flush=True,
        )
        dist.init_process_group(backend=backend, init_method="env://")
        print(
            f"[stage4-lifecycle][rank={rank}] phase=distributed_init_done "
            f"backend={backend} world_size={world_size}",
            flush=True,
        )
    # Gate B is deliberately single-process; only the training/audit gates
    # that launch DDP require the production eight-rank topology.
    if config.gate == "B" and world_size != 1:
        raise ValueError(f"Stage 4 Gate B requires world_size=1, got {world_size}")
    if config.gate != "B" and world_size != DEFAULT_WORLD_SIZE and not (config.allow_non8 or config.dry_run):
        raise ValueError(f"Stage 4 Gate {config.gate} requires world_size=8, got {world_size}")
    return rank, world_size, device, use_cuda


def _ddp_wrap(model: Any, config: Stage4Config, device: Any) -> Any:
    import torch.nn.parallel as parallel

    import torch

    if config.world_size <= 1:
        return model
    if getattr(device, "type", "") == "cuda":
        return parallel.DistributedDataParallel(
            model,
            device_ids=[int(device.index)],
            output_device=int(device.index),
            find_unused_parameters=False,
        )
    return parallel.DistributedDataParallel(model, find_unused_parameters=False)


def _model_checksum_audit(model: Any, device: Any) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    checksum = _parameter_checksum(model)
    result = {"local": float(checksum.item()), "min": float(checksum.item()), "max": float(checksum.item()), "consistent": True}
    if dist.is_initialized():
        lo, hi = checksum.clone(), checksum.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        result.update({"min": float(lo.item()), "max": float(hi.item()), "consistent": bool(torch.allclose(lo, hi, atol=1e-5, rtol=1e-5))})
    return result


def compute_warmup_steps(total_steps: int, requested: int | None = None) -> int:
    """Return an auditable warmup count for one concrete training target.

    A zero/``None`` request means the Stage 4 contract default: ``ceil(5%)``.
    Positive explicit values remain accepted for backwards-compatible pilot
    experiments, but are always clamped to the concrete schedule length.
    """

    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if requested is None or int(requested) == 0:
        return max(1, min(total_steps, math.ceil(0.05 * total_steps)))
    requested = int(requested)
    if requested < 0:
        raise ValueError("warmup_steps must be non-negative")
    return max(1, min(total_steps, requested))


def cosine_warmup_factor(
    step: int,
    total_steps: int,
    warmup_steps: int,
    *,
    max_lr: float = DEFAULT_MAX_LR,
    min_lr: float = DEFAULT_MIN_LR,
) -> float:
    """Return the LR multiplier at a completed optimizer-step index.

    Step zero starts at ``min_lr``; the linear warmup reaches ``max_lr`` at
    ``warmup_steps`` (when there is a post-warmup interval), and cosine decay
    reaches ``min_lr`` at ``total_steps``.  This pure helper is intentionally
    dependency-free so static tests can audit boundaries and monotonicity.
    """

    total_steps = int(total_steps)
    warmup_steps = compute_warmup_steps(total_steps, warmup_steps)
    if max_lr <= 0 or min_lr <= 0 or min_lr > max_lr:
        raise ValueError("scheduler requires 0 < min_lr <= max_lr")
    step = max(0, min(int(step), total_steps))
    min_ratio = float(min_lr) / float(max_lr)
    if step >= total_steps:
        return min_ratio
    if warmup_steps < total_steps and step <= warmup_steps:
        return min_ratio + (1.0 - min_ratio) * (float(step) / float(warmup_steps))
    if warmup_steps >= total_steps:
        return min_ratio + (1.0 - min_ratio) * (float(step) / float(total_steps))
    progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def scheduler_metadata(
    *,
    total_steps: int,
    warmup_steps: int,
    max_lr: float = DEFAULT_MAX_LR,
    min_lr: float = DEFAULT_MIN_LR,
    target_steps: int | None = None,
    formal_steps: int = DEFAULT_FORMAL_OPTIMIZER_STEPS,
) -> dict[str, Any]:
    """Build the scheduler contract copied into reports and checkpoints."""

    total_steps = int(total_steps)
    warmup_steps = compute_warmup_steps(total_steps, warmup_steps)
    return {
        "type": SCHEDULER_TYPE,
        "scheduler_type": SCHEDULER_TYPE,
        "max_lr": float(max_lr),
        "min_lr": float(min_lr),
        "total_steps_for_schedule": total_steps,
        "warmup_steps": warmup_steps,
        "warmup_fraction": 0.05,
        "target_optimizer_steps": int(target_steps if target_steps is not None else total_steps),
        "formal_optimizer_steps": int(formal_steps),
    }


def validate_formal_resume_state(state: Mapping[str, Any]) -> dict[str, int]:
    """Validate metadata required to continue a FORMAL checkpoint.

    The model/optimizer/RNG tensors are restored by the torch-specific resume
    path; this dependency-light validator protects the scheduler domain and
    total-step semantics before that state is applied.
    """

    configuration = state.get("configuration", {})
    actual_gate = (
        str(configuration.get("gate", "")).upper()
        if isinstance(configuration, Mapping)
        else ""
    )
    if actual_gate != "FORMAL":
        checkpoint_mode = actual_gate or "UNKNOWN"
        raise ValueError(
            "FORMAL resume requires a checkpoint created by FORMAL mode; "
            f"got checkpoint gate={checkpoint_mode}. This is usually a Gate D/E checkpoint. "
            "For a fresh formal run, unset RSMOL_5_10XR_5_RESUME_FROM; for resume, pass a "
            "checkpoint-step-NNNNNN created by --gate FORMAL."
        )
    scheduler = state.get("scheduler_config")
    if not isinstance(scheduler, Mapping):
        raise ValueError("FORMAL resume checkpoint is missing scheduler_config")
    try:
        schedule_total_steps = int(scheduler.get("total_steps_for_schedule", -1))
        warmup_steps = int(scheduler.get("warmup_steps", -1))
        optimizer_step = int(state.get("optimizer_step", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("FORMAL resume checkpoint has invalid scheduler/optimizer metadata") from exc
    if schedule_total_steps != DEFAULT_FORMAL_OPTIMIZER_STEPS:
        raise ValueError("FORMAL resume checkpoint scheduler domain must be 9244")
    if warmup_steps != DEFAULT_FORMAL_WARMUP_STEPS:
        raise ValueError("FORMAL resume checkpoint warmup_steps must be 463")
    if optimizer_step < 0 or optimizer_step > DEFAULT_FORMAL_OPTIMIZER_STEPS:
        raise ValueError(
            "FORMAL resume checkpoint optimizer_step must be in [0, 9244]; "
            f"got {optimizer_step}"
        )
    return {
        "optimizer_step": optimizer_step,
        "scheduler_total_steps": DEFAULT_FORMAL_OPTIMIZER_STEPS,
        "warmup_steps": DEFAULT_FORMAL_WARMUP_STEPS,
    }


def _load_scheduler(
    optimizer: Any,
    warmup_steps: int,
    total_steps: int | None = None,
    *,
    max_lr: float = DEFAULT_MAX_LR,
    min_lr: float = DEFAULT_MIN_LR,
) -> Any:
    """Create the shared linear-warmup + cosine-decay scheduler.

    ``warmup_steps`` remains the second positional parameter for compatibility
    with older audit callers.  The schedule length defaults to the warmup
    length only when omitted; all real training paths pass an explicit target.
    """

    import torch

    if total_steps is None:
        total_steps = max(1, int(warmup_steps))
    total_steps = int(total_steps)
    warmup_steps = compute_warmup_steps(total_steps, warmup_steps)
    if max_lr <= 0 or min_lr <= 0 or min_lr > max_lr:
        raise ValueError("scheduler requires 0 < min_lr <= max_lr")

    def schedule(step: int) -> float:
        return cosine_warmup_factor(
            step,
            total_steps,
            warmup_steps,
            max_lr=max_lr,
            min_lr=min_lr,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    # Keep the pure, auditable contract on the object for checkpoint/report
    # code without depending on LambdaLR's private lambda serialization.
    scheduler.stage4_config = scheduler_metadata(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        max_lr=max_lr,
        min_lr=min_lr,
        target_steps=total_steps,
    )
    return scheduler


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _tupleize(value: Any) -> Any:
    """Rebuild nested tuples for ``random.Random.setstate`` after loading."""

    if isinstance(value, list):
        return tuple(_tupleize(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_tupleize(item) for item in value)
    return value


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _gather_rank_cursors(
    cursor: Mapping[str, Any], *, rank: int, world_size: int
) -> dict[str, dict[str, Any]]:
    """Collect one bounded-shuffle cursor from every rank before a save.

    ``all_gather_object`` is intentionally used only at checkpoint boundaries;
    it keeps the training data path rank-local while making a resume
    checkpoint self-contained for every DDP rank.
    """

    import torch.distributed as dist

    local = _json_safe(dict(cursor))
    if not dist.is_initialized():
        if int(world_size) != 1:
            raise RuntimeError("Per-rank cursor gathering requires an initialized process group")
        return {str(rank): dict(local)}
    if int(dist.get_world_size()) != int(world_size):
        raise RuntimeError(
            f"Cursor gather world-size mismatch: process_group={dist.get_world_size()} config={world_size}"
        )
    gathered: list[Any] = [None for _ in range(int(world_size))]
    dist.all_gather_object(gathered, local)
    if any(not isinstance(value, Mapping) for value in gathered):
        raise RuntimeError(f"Per-rank cursor gather returned invalid values: {gathered!r}")
    return {str(index): dict(value) for index, value in enumerate(gathered)}


def save_complete_checkpoint(
    checkpoint_dir: Path,
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    optimizer_step: int,
    cumulative_samples: int,
    cumulative_tokens: int,
    data_cursors_by_rank: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    config: Stage4Config,
    report: Mapping[str, Any],
    rank: int,
    scheduler_config: Mapping[str, Any] | None = None,
) -> Path | None:
    """Write a complete checkpoint on rank 0 through an atomic staging dir.

    A Stage 4 DDP checkpoint must carry a cursor for every rank.  Keeping only
    rank 0's cursor would make a later resume silently repeat data on the
    other ranks, so the contract is enforced at the save boundary.
    """

    import torch

    if rank != 0:
        return None
    expected_cursor_keys = {str(index) for index in range(int(config.world_size))}
    actual_cursor_keys = {str(key) for key in data_cursors_by_rank}
    if actual_cursor_keys != expected_cursor_keys:
        raise ValueError(
            "Complete checkpoint requires one data cursor per rank: "
            f"expected={sorted(expected_cursor_keys)} got={sorted(actual_cursor_keys)}"
        )
    if any(not isinstance(value, Mapping) for value in data_cursors_by_rank.values()):
        raise ValueError("Each data_cursors_by_rank entry must be a mapping")
    normalized_cursors = {
        str(key): dict(value) for key, value in data_cursors_by_rank.items()
    }
    if scheduler_config is None:
        candidate_scheduler = report.get("scheduler") if isinstance(report, Mapping) else None
        scheduler_config = candidate_scheduler if isinstance(candidate_scheduler, Mapping) else getattr(scheduler, "stage4_config", {})
    normalized_scheduler_config = _json_safe(dict(scheduler_config or {}))
    checkpoint_dir = ensure_external_output(checkpoint_dir)
    if checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_dir}")
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = checkpoint_dir.with_name(checkpoint_dir.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        base_model = getattr(model, "module", model)
        base_model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scheduler_config": normalized_scheduler_config,
                "optimizer_step": int(optimizer_step),
                "cumulative_samples": int(cumulative_samples),
                "cumulative_tokens": int(cumulative_tokens),
                "python_random_state": random.getstate(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "data_cursors_by_rank": _json_safe(normalized_cursors),
                "manifest": _json_safe(dict(manifest)),
                "configuration": _json_safe(asdict(config)),
                "report": _json_safe(dict(report)),
                "checkpoint_contract": {
                    "complete": True,
                    "mode": "formal" if config.gate == "FORMAL" else str(config.gate),
                    "model_config": True,
                    "tokenizer": True,
                    "optimizer": True,
                    "scheduler": True,
                    "scheduler_config": bool(normalized_scheduler_config),
                    "rng": True,
                    "step": True,
                    "data_cursors_by_rank": True,
                    "manifest": True,
                    "world_size": int(config.world_size),
                    "architecture": MODEL_ARCHITECTURE_CONTRACT,
                    "sampling_policy": SAMPLING_POLICY,
                    "sampler_version": SAMPLER_VERSION,
                    "poisson_lambda": POISSON_LAMBDA,
                    "poisson_support": list(POISSON_SUPPORT),
                    "poisson_normalization_z": POISSON_NORMALIZATION_Z,
                    "poisson_probabilities": list(POISSON_PROBABILITIES),
                    "sampler_key": SAMPLER_KEY,
                    "default_inference_r": DEFAULT_INFERENCE_MIDDLE_LOOPS,
                    "fixed_parameter_gradient_tail_loops": PARAMETER_GRADIENT_TAIL_LOOPS,
                    "bounded_shuffle_rng_state_saved": True,
                    "bounded_shuffle_bitwise_exact": False,
                },
            },
            staging / "training_state.pt",
        )
        (staging / "checkpoint_complete.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "mode": "formal" if config.gate == "FORMAL" else str(config.gate),
                    "optimizer_step": int(optimizer_step),
                    "world_size": int(config.world_size),
                    "architecture": MODEL_ARCHITECTURE_CONTRACT,
                    "sampling_policy": SAMPLING_POLICY,
                    "sampler_version": SAMPLER_VERSION,
                    "poisson_lambda": POISSON_LAMBDA,
                    "poisson_support": list(POISSON_SUPPORT),
                    "poisson_normalization_z": POISSON_NORMALIZATION_Z,
                    "poisson_probabilities": list(POISSON_PROBABILITIES),
                    "sampler_key": SAMPLER_KEY,
                    "default_inference_r": DEFAULT_INFERENCE_MIDDLE_LOOPS,
                    "fixed_parameter_gradient_tail_loops": PARAMETER_GRADIENT_TAIL_LOOPS,
                    "rank0_only": True,
                    "data_cursors_by_rank": sorted(normalized_cursors),
                    "scheduler_config": normalized_scheduler_config,
                    "files": sorted(path.name for path in staging.iterdir()) + ["checkpoint_complete.json"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        required_files = ("config.json", "training_state.pt", "checkpoint_complete.json")
        missing_files = [name for name in required_files if not (staging / name).is_file()]
        model_files = list(staging.glob("*.safetensors")) + list(staging.glob("pytorch_model*.bin"))
        tokenizer_files = list(staging.glob("tokenizer.json")) + list(staging.glob("tokenizer.model")) + list(staging.glob("vocab.*"))
        if missing_files or not model_files or not tokenizer_files:
            raise RuntimeError(
                "Checkpoint integrity audit failed: "
                f"missing={missing_files}, model_files={[path.name for path in model_files]}, "
                f"tokenizer_files={[path.name for path in tokenizer_files]}"
            )
        staging.replace(checkpoint_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return checkpoint_dir


_FORMAL_CHECKPOINT_NAME = re.compile(r"checkpoint-step-(?P<step>[0-9]{6})\Z")


def _complete_checkpoint_step(path: Path) -> int | None:
    """Return a checkpoint step only for an explicitly complete checkpoint.

    This deliberately does not use a broad glob or infer completeness from a
    directory name.  Staging directories, reports, latest pointers, and
    incomplete checkpoints are therefore never retention candidates.
    """

    match = _FORMAL_CHECKPOINT_NAME.fullmatch(path.name)
    if match is None or not path.is_dir():
        return None
    marker_path = path / "checkpoint_complete.json"
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(marker, Mapping):
        return None
    if marker.get("complete") is not True:
        return None
    try:
        _validate_exact_sampling_contract(marker, label="Checkpoint retention marker")
    except ValueError:
        return None
    marker_step = marker.get("optimizer_step")
    step = int(match.group("step"))
    if marker_step is not None:
        try:
            if int(marker_step) != step:
                return None
        except (TypeError, ValueError):
            return None
    return step


def formal_checkpoint_steps(output_dir: Path) -> list[int]:
    """List complete formal checkpoint steps in ascending order."""

    output_dir = ensure_external_output(Path(output_dir))
    if not output_dir.is_dir():
        return []
    steps = [
        step
        for path in output_dir.iterdir()
        if (step := _complete_checkpoint_step(path)) is not None
    ]
    return sorted(set(steps))


def formal_save_steps(total_steps: int = DEFAULT_FORMAL_OPTIMIZER_STEPS,
                      save_every: int = DEFAULT_FORMAL_SAVE_EVERY) -> list[int]:
    """Return periodic formal save steps plus the mandatory final step."""

    total_steps = int(total_steps)
    save_every = int(save_every)
    if total_steps <= 0 or save_every <= 0:
        raise ValueError("formal total_steps and save_every must be positive")
    steps = list(range(save_every, total_steps + 1, save_every))
    if not steps or steps[-1] != total_steps:
        steps.append(total_steps)
    return steps


def retain_formal_checkpoints(
    output_dir: Path,
    *,
    keep: int = DEFAULT_FORMAL_CHECKPOINT_RETENTION,
    latest_checkpoint: Path | None = None,
) -> list[Path]:
    """Retain only the newest complete formal checkpoints.

    The caller must invoke this on rank 0 after an atomic checkpoint save.
    Only exact ``checkpoint-step-NNNNNN`` directories with a verified
    ``complete=true`` marker may be removed.
    """

    keep = int(keep)
    if keep <= 0:
        raise ValueError("formal checkpoint retention must be positive")
    output_dir = ensure_external_output(Path(output_dir))
    candidates = [
        (step, output_dir / f"checkpoint-step-{step:06d}")
        for step in formal_checkpoint_steps(output_dir)
    ]
    candidates.sort(key=lambda item: item[0])
    retained = candidates[-keep:]
    retained_paths = [path for _, path in retained]
    retained_names = {path.name for path in retained_paths}
    for _, path in candidates[:-keep]:
        # Revalidate immediately before deletion so this helper cannot ever
        # remove a path that changed into a staging/incomplete directory.
        if path.name in retained_names or _complete_checkpoint_step(path) is None:
            continue
        shutil.rmtree(path)
    if retained_paths:
        newest = retained_paths[-1]
        if latest_checkpoint is not None and Path(latest_checkpoint).name == newest.name:
            newest = Path(latest_checkpoint)
        pointer = {
            "checkpoint": str(newest),
            "optimizer_step": int(retained[-1][0]),
            "complete": True,
            "retained_checkpoints": [str(path) for path in retained_paths],
        }
        _write_json(output_dir / "latest_complete_checkpoint.json", pointer)
        verified = json.loads(
            (output_dir / "latest_complete_checkpoint.json").read_text(encoding="utf-8")
        )
        if verified.get("complete") is not True or int(verified.get("optimizer_step", -1)) != retained[-1][0]:
            raise RuntimeError("latest_complete_checkpoint.json failed post-retention verification")
    return retained_paths


def _restore_training_state(
    resume_from: Path,
    *,
    optimizer: Any,
    scheduler: Any,
    config: Stage4Config,
    rank: int,
) -> tuple[int, int, int, dict[str, Any], dict[str, Any], str]:
    import torch

    resume_from = ensure_external_resume(resume_from)
    state_path = resume_from / "training_state.pt"
    complete_path = resume_from / "checkpoint_complete.json"
    if not state_path.is_file() or not complete_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint is incomplete: {resume_from}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("complete") is not True:
        raise ValueError("Resume checkpoint_complete.json does not declare complete=true")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    contract = dict(state.get("checkpoint_contract", {}))
    architecture_contract = MODEL_ARCHITECTURE_CONTRACT
    if contract.get("architecture") != architecture_contract:
        raise ValueError(
            "Resume checkpoint architecture contract mismatch: "
            f"expected={architecture_contract!r} got={contract.get('architecture')!r}"
        )
    _validate_exact_sampling_contract(contract, label="Resume checkpoint")
    if int(contract.get("default_inference_r", -1)) != DEFAULT_INFERENCE_MIDDLE_LOOPS:
        raise ValueError("Resume checkpoint default inference r mismatch")
    if int(contract.get("fixed_parameter_gradient_tail_loops", -1)) != PARAMETER_GRADIENT_TAIL_LOOPS:
        raise ValueError("Resume checkpoint fixed gradient-tail contract mismatch")
    if "world_size" not in contract or int(contract["world_size"]) != int(config.world_size):
        raise ValueError(
            "Resume checkpoint contract world_size mismatch: "
            f"checkpoint={contract.get('world_size')!r} current={config.world_size}"
        )
    if "optimizer_step" not in state:
        raise ValueError("Resume checkpoint is missing training_state.optimizer_step")
    state_step = int(state["optimizer_step"])
    if config.gate == "FORMAL":
        validate_formal_resume_state(state)
    if "world_size" not in complete or int(complete["world_size"]) != int(config.world_size):
        raise ValueError(
            "Resume checkpoint_complete.json world_size mismatch: "
            f"checkpoint={complete.get('world_size')!r} current={config.world_size}"
        )
    _validate_exact_sampling_contract(complete, label="Resume checkpoint_complete.json")
    if "optimizer_step" not in complete or int(complete["optimizer_step"]) != state_step:
        raise ValueError(
            "Resume checkpoint step mismatch between checkpoint_complete.json and training_state.pt: "
            f"marker={complete.get('optimizer_step')!r} state={state_step}"
        )
    state_configuration = dict(state.get("configuration", {}))
    if "world_size" in state_configuration and int(state_configuration["world_size"]) != int(config.world_size):
        raise ValueError(
            "Resume training_state configuration world_size mismatch: "
            f"checkpoint={state_configuration['world_size']} current={config.world_size}"
        )
    if "seed" in state_configuration and int(state_configuration["seed"]) != int(config.seed):
        raise ValueError(
            "Resume training_state sampler seed mismatch: "
            f"checkpoint={state_configuration['seed']} current={config.seed}"
        )
    has_per_rank_cursors = "data_cursors_by_rank" in state
    if config.gate == "E" and not has_per_rank_cursors:
        raise ValueError(
            "Gate E resume rejected: checkpoint contains legacy single data_cursor; "
            "a new Stage 4 checkpoint must contain data_cursors_by_rank"
        )
    if has_per_rank_cursors and "data_cursors_by_rank" in complete:
        marker_cursor_keys = {str(key) for key in complete["data_cursors_by_rank"]}
        expected_cursor_keys = {str(index) for index in range(int(config.world_size))}
        if marker_cursor_keys != expected_cursor_keys:
            raise ValueError(
                "Resume checkpoint_complete.json data_cursors_by_rank keys mismatch: "
                f"expected={sorted(expected_cursor_keys)} got={sorted(marker_cursor_keys)}"
            )
    required_contract = (
        "complete", "model_config", "tokenizer", "optimizer", "scheduler",
        "rng", "step", "manifest", "world_size", "architecture",
    )
    required_contract += ("data_cursors_by_rank" if has_per_rank_cursors else "data_cursor",)
    missing_contract = [name for name in required_contract if not contract.get(name)]
    if missing_contract:
        raise ValueError(f"Resume checkpoint failed completeness contract: missing={missing_contract}")
    if has_per_rank_cursors:
        raw_cursors = state.get("data_cursors_by_rank")
        if not isinstance(raw_cursors, Mapping):
            raise ValueError("Resume checkpoint data_cursors_by_rank must be a mapping")
        expected_cursor_keys = {str(index) for index in range(int(config.world_size))}
        actual_cursor_keys = {str(key) for key in raw_cursors}
        if actual_cursor_keys != expected_cursor_keys:
            raise ValueError(
                "Resume checkpoint data_cursors_by_rank keys mismatch: "
                f"expected={sorted(expected_cursor_keys)} got={sorted(actual_cursor_keys)}"
            )
        selected = raw_cursors.get(str(rank))
        if not isinstance(selected, Mapping):
            raise ValueError(f"Resume checkpoint is missing cursor for rank {rank}")
        cursor = dict(selected)
        cursor_source = "data_cursors_by_rank"
    else:
        legacy_cursor = state.get("data_cursor")
        if not isinstance(legacy_cursor, Mapping):
            raise ValueError("Legacy resume checkpoint data_cursor must be a mapping")
        if int(config.world_size) != 1:
            raise ValueError(
                "Legacy single data_cursor checkpoints are supported only for world_size=1; "
                f"current world_size={config.world_size}"
            )
        cursor = dict(legacy_cursor)
        cursor_source = "legacy_data_cursor"
    required_cursor = (
        "epoch", "shard_index", "shard_microbatches_seen", "samples_seen", "microbatches_seen"
    )
    missing_cursor = [name for name in required_cursor if name not in cursor]
    if missing_cursor:
        raise ValueError(f"Resume checkpoint data cursor is incomplete: missing={missing_cursor}")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if state.get("python_random_state") is not None:
        random.setstate(state["python_random_state"])
    if state.get("torch_random_state") is not None:
        torch.set_rng_state(state["torch_random_state"])
    if state.get("cuda_random_state_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_random_state_all"])
    return (
        state_step,
        int(state.get("cumulative_samples", 0)),
        int(state.get("cumulative_tokens", 0)),
        cursor,
        dict(state.get("manifest", {})),
        cursor_source,
    )


def _training_window(
    model: Any,
    iterator: Iterator[dict[str, Any]],
    *,
    config: Stage4Config,
    device: Any,
    rank: int,
    optimizer_step: int,
) -> tuple[bool, dict[str, Any]]:
    """Collect exactly 64 complete microbatches before doing backward."""

    import torch
    import torch.distributed as dist

    batches: list[dict[str, Any]] = []
    global_counts: list[int] = []
    for micro_index in range(config.gradient_accumulation_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            batch = None
        complete = _all_reduce_min_flag(batch is not None, device)
        if not complete:
            return False, {
                "partial_accumulation_window_discarded": True,
                "microbatches_collected": len(batches),
                "required_microbatches": config.gradient_accumulation_steps,
                "all_ranks_coordinated_end": True,
            }
        assert batch is not None
        batches.append(batch)
        local_count = (batch["labels"][:, 1:] != -100).sum().to(dtype=torch.int64)
        count = _all_reduce_scalar(local_count.clone())
        global_counts.append(int(count.item()))
    global_window_tokens = sum(global_counts)
    if global_window_tokens <= 0:
        raise FloatingPointError("Accumulation window has no supervised shifted tokens")
    local_loss_sum_total = torch.zeros((), device=device, dtype=torch.float64)
    local_valid_total = torch.zeros((), device=device, dtype=torch.int64)
    micro_records: list[dict[str, Any]] = []
    sampled_counts_history: list[list[int]] = []
    sampled_tmax_history: list[int] = []
    for micro_index, (batch, global_count) in enumerate(zip(batches, global_counts)):
        middle_loop_counts = sample_middle_loop_counts(
            config.seed, rank, optimizer_step, micro_index, batch["input_ids"].shape[0]
        )
        local_tmax = int(middle_loop_counts.max().item())
        sampled_counts_history.append([int(value) for value in middle_loop_counts.tolist()])
        sampled_tmax_history.append(local_tmax)
        with _bf16_autocast(device):
            result = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                middle_loop_counts=middle_loop_counts.to(device=device),
            )
            loss_sum, valid_count = _causal_loss_sum(result.logits, batch["labels"])
        if not torch.isfinite(loss_sum):
            raise FloatingPointError(f"Non-finite loss at rank={rank}")
        scale = token_weighted_gradient_scale(
            world_size=config.world_size,
            global_window_tokens=global_window_tokens,
        )
        (loss_sum * scale).backward()
        local_loss_sum_total += loss_sum.detach().double()
        local_valid_total += valid_count.detach()
        global_loss = _all_reduce_scalar(loss_sum.detach().double().clone())
        micro_records.append(
            {
                "microbatch": len(micro_records),
                "middle_loop_counts": sampled_counts_history[-1],
                "local_tmax": local_tmax,
                "tau": [local_tmax - int(value) for value in sampled_counts_history[-1]],
                "local_loss_sum": float(loss_sum.detach().float().item()),
                "local_valid_token_count": int(valid_count.detach().item()),
                "global_loss_sum": float(global_loss.item()),
                "global_valid_token_count": int(global_count),
                "gradient_scale": scale,
            }
        )
    global_loss_sum = _all_reduce_scalar(local_loss_sum_total.clone())
    global_valid_tokens = _all_reduce_scalar(local_valid_total.clone())
    return True, {
        "partial_accumulation_window_discarded": False,
        "microbatches_collected": len(batches),
        "exact_accumulation_microbatches": len(batches) == config.gradient_accumulation_steps,
        "window_global_loss_sum": float(global_loss_sum.item()),
        "window_global_valid_token_count": int(global_valid_tokens.item()),
        "window_global_loss_count_consistent": True,
        "microbatch_records": micro_records,
        "middle_loop_counts_per_microbatch": sampled_counts_history,
        "local_tmax_per_microbatch": sampled_tmax_history,
        "sampling_policy": SAMPLING_POLICY,
        "sampler_metadata": poisson_metadata(),
        "all_rank_r_equal": False,
    }


def _emit_progress_log(
    *,
    output_dir: Path,
    rank: int,
    optimizer_step: int,
    total_steps: int,
    loss: float,
    learning_rate: float,
    elapsed_seconds: float,
    samples_per_second: float,
    steps_per_second: float,
    reason: str,
) -> dict[str, Any] | None:
    """Print and persist one rank-0 training progress record."""

    if rank != 0:
        return None
    record = {
        "optimizer_step": int(optimizer_step),
        "total_steps": int(total_steps),
        "loss": float(loss),
        "learning_rate": float(learning_rate),
        "samples_per_second": float(samples_per_second),
        "steps_per_second": float(steps_per_second),
        "elapsed_seconds": float(elapsed_seconds),
        "reason": str(reason),
    }
    _append_jsonl(output_dir / "stage4_progress.jsonl", record)
    print(
        "[stage4-progress] "
        f"step={record['optimizer_step']}/{record['total_steps']} "
        f"loss={record['loss']:.6f} lr={record['learning_rate']:.8g} "
        f"speed={record['samples_per_second']:.3f} samples/sec "
        f"({record['steps_per_second']:.5f} steps/sec) "
        f"elapsed={record['elapsed_seconds']:.2f}s reason={record['reason']}",
        flush=True,
    )
    return record


def run_training(
    model: Any,
    tokenizer: Any,
    *,
    config: Stage4Config,
    device: Any,
    rank: int,
    manifest: Mapping[str, Any],
    output_dir: Path,
    resume_state: tuple[int, int, int, dict[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a real/synthetic pilot with exact token-weighted accumulation."""

    import torch
    import torch.distributed as dist

    if config.gate == "D" and config.max_optimizer_steps != DEFAULT_MAX_OPTIMIZER_STEPS:
        raise ValueError(
            "Stage 4 Gate D is a fixed ten-optimizer-step smoke; "
            f"got max_optimizer_steps={config.max_optimizer_steps}"
        )
    if config.gate == "FORMAL":
        if config.max_optimizer_steps != DEFAULT_FORMAL_OPTIMIZER_STEPS:
            raise ValueError("FORMAL requires exactly 9244 optimizer steps")
        if config.scheduler_total_steps != DEFAULT_FORMAL_OPTIMIZER_STEPS:
            raise ValueError("FORMAL requires scheduler_total_steps=9244")
        if compute_warmup_steps(config.scheduler_total_steps, config.warmup_steps) != DEFAULT_FORMAL_WARMUP_STEPS:
            raise ValueError("FORMAL requires warmup_steps=463")
        if config.save_every != DEFAULT_FORMAL_SAVE_EVERY:
            raise ValueError("FORMAL requires save_every=500")
        if config.checkpoint_retention != DEFAULT_FORMAL_CHECKPOINT_RETENTION:
            raise ValueError("FORMAL requires checkpoint_retention=3")
    model = _ddp_wrap(model, config, device)
    parameters = _trainable_parameters(model)
    schedule_total_steps = int(config.scheduler_total_steps or config.max_optimizer_steps)
    warmup_steps = compute_warmup_steps(schedule_total_steps, config.warmup_steps)
    # ``learning_rate`` is the historical CLI name and remains the max LR.
    if abs(float(config.learning_rate) - float(config.max_lr)) > 1e-12:
        raise ValueError("Stage 4 learning_rate and max_lr must match")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.max_lr,
        betas=config.optimizer_betas,
        eps=config.optimizer_eps,
        weight_decay=config.weight_decay,
        amsgrad=config.optimizer_amsgrad,
    )
    optimizer_audit = optimizer_parameter_audit(optimizer, parameters)
    scheduler = _load_scheduler(
        optimizer,
        warmup_steps,
        schedule_total_steps,
        max_lr=config.max_lr,
        min_lr=config.min_lr,
    )
    schedule = scheduler_metadata(
        total_steps=schedule_total_steps,
        warmup_steps=warmup_steps,
        max_lr=config.max_lr,
        min_lr=config.min_lr,
        target_steps=config.max_optimizer_steps,
        formal_steps=config.formal_optimizer_steps,
    )
    config.scheduler_total_steps = schedule_total_steps
    config.warmup_steps = warmup_steps
    start_step = 0
    cumulative_samples = 0
    cumulative_tokens = 0
    previous_cursor: dict[str, Any] = {}
    previous_manifest: dict[str, Any] = {}
    previous_cursor_source = "fresh"
    formal_resume = config.gate == "FORMAL" and resume_state is not None
    if resume_state is not None:
        _stage4_event(
            output_dir,
            rank,
            "restore_state_start",
            checkpoint=str(config.resume_from),
        )
        (
            start_step,
            cumulative_samples,
            cumulative_tokens,
            previous_cursor,
            previous_manifest,
            previous_cursor_source,
        ) = _restore_training_state(
            config.resume_from or Path("."),
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            rank=rank,
        )
        _stage4_event(
            output_dir,
            rank,
            "restore_state_done",
            start_step=start_step,
            cursor_source=previous_cursor_source,
        )
        if previous_manifest:
            manifest = previous_manifest
    stream = DistributedParquetStream(
        [Path(path) for path in manifest["rank_shards"][str(rank)]],
        tokenizer,
        rank=rank,
        micro_batch_size=config.micro_batch_size,
        context_length=config.context_length,
        pad_token_id=_tokenizer_pad(tokenizer),
        seed=config.seed,
        record_buffer_size=config.record_buffer_size,
        device=device,
    )
    if previous_cursor:
        # Restore the auditable logical cursor.  The iterator then performs a
        # coarse skip through the recorded epoch/shard/complete microbatches;
        # the bounded shuffle buffer itself is intentionally not claimed exact.
        stream.restore_cursor(previous_cursor)
        _stage4_event(
            output_dir,
            rank,
            "cursor_restore_done",
            epoch=previous_cursor.get("epoch"),
            shard_index=previous_cursor.get("shard_index"),
            shard_microbatches_seen=previous_cursor.get("shard_microbatches_seen"),
        )
    batch_iterator = iter(stream)
    _stage4_event(output_dir, rank, "stream_ready")
    # A bounded shuffle cannot restore the exact in-shard cursor.  Keep the
    # cursor for audit/restart bookkeeping but never claim bitwise data resume.
    monitor = RuntimeMonitor(output_dir / f"runtime_monitor_rank{rank}.jsonl", interval_seconds=config.monitor_interval_seconds, rank=rank, device=device)
    monitor.start()
    metrics: list[dict[str, Any]] = []
    progress_logs: list[dict[str, Any]] = []
    optimizer_step = start_step
    latest_checkpoint: Path | None = (
        config.resume_from.resolve() if formal_resume and config.resume_from is not None else None
    )
    retained_checkpoints: list[Path] = []
    stop_reason = "max_optimizer_steps"
    if formal_resume and start_step >= config.max_optimizer_steps:
        stop_reason = "already_at_formal_target"
    coordinated_stop = False
    training_start = time.perf_counter()
    try:
        while optimizer_step < config.max_optimizer_steps:
            optimizer.zero_grad(set_to_none=True)
            window_start = time.perf_counter()
            complete, window = _training_window(
                model, batch_iterator, config=config, device=device, rank=rank,
                optimizer_step=optimizer_step,
            )
            if not complete:
                # All ranks have observed the same MIN flag and reached this
                # point.  The partial window is intentionally not stepped.
                stop_reason = "coordinated_data_exhaustion"
                coordinated_stop = True
                break
            _assert_rank_equal(
                torch.tensor(window["window_global_loss_sum"], device=device),
                name="global_loss_sum",
                device=device,
                tolerance=1e-4,
            )
            _assert_rank_equal(
                torch.tensor(window["window_global_valid_token_count"], device=device),
                name="global_valid_token_count",
                device=device,
                tolerance=0.0,
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            if not torch.isfinite(torch.as_tensor(gradient_norm)):
                raise FloatingPointError(f"Non-finite gradient norm at optimizer_step={optimizer_step}")
            before_checksum = _parameter_checksum(model).detach().clone()
            representative_parameter = next(
                parameter
                for name, parameter in model.named_parameters()
                if ".layers." in name
            )
            representative_before = representative_parameter.detach().flatten()[0].clone()
            optimizer.step()
            scheduler.step()
            optimizer_step += 1
            after_checksum = _parameter_checksum(model).detach().clone()
            representative_delta = float(
                (representative_parameter.detach().flatten()[0] - representative_before).abs().item()
            )
            if representative_delta <= 0.0:
                raise FloatingPointError(
                    f"Shared recursive parameter did not update at optimizer_step={optimizer_step}"
                )
            cumulative_samples += config.micro_batch_size * config.gradient_accumulation_steps
            cumulative_tokens += int(window["window_global_valid_token_count"])
            elapsed = max(time.perf_counter() - window_start, 1e-9)
            metrics.append(
                {
                    "optimizer_step": optimizer_step,
                    "middle_loop_counts_per_microbatch": window["middle_loop_counts_per_microbatch"],
                    "local_tmax_per_microbatch": window["local_tmax_per_microbatch"],
                    "all_rank_r_equal": window["all_rank_r_equal"],
                    "parameter_gradient_tail_loops": PARAMETER_GRADIENT_TAIL_LOOPS,
                    "microbatches": config.gradient_accumulation_steps,
                    "exact_accumulation_microbatches": True,
                    "global_loss_sum": window["window_global_loss_sum"],
                    "global_valid_token_count": window["window_global_valid_token_count"],
                    "loss": window["window_global_loss_sum"] / max(window["window_global_valid_token_count"], 1),
                    "grad_norm": float(torch.as_tensor(gradient_norm).item()),
                    "loss_finite": True,
                    "grad_finite": True,
                    "parameter_checksum_before": float(before_checksum.item()),
                    "parameter_checksum_after": float(after_checksum.item()),
                    "parameter_updated": bool(not torch.equal(before_checksum, after_checksum)),
                    "shared_parameter_updated": True,
                    "shared_parameter_representative_delta": representative_delta,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_seconds": elapsed,
                    "global_samples_per_second": config.world_size * config.micro_batch_size * config.gradient_accumulation_steps / elapsed,
                    "global_tokens_per_second": window["window_global_valid_token_count"] / elapsed,
                    "data_cursor": stream.cursor(include_rng=False),
                }
            )
            if rank == 0:
                _append_jsonl(output_dir / "stage4_metrics.jsonl", metrics[-1])
                if optimizer_step % config.log_interval_steps == 0:
                    progress = _emit_progress_log(
                        output_dir=output_dir,
                        rank=rank,
                        optimizer_step=optimizer_step,
                        total_steps=config.max_optimizer_steps,
                        loss=metrics[-1]["loss"],
                        learning_rate=metrics[-1]["learning_rate"],
                        elapsed_seconds=max(time.perf_counter() - training_start, 1e-9),
                        samples_per_second=(optimizer_step - start_step)
                        * (config.world_size * config.micro_batch_size * config.gradient_accumulation_steps)
                        / max(time.perf_counter() - training_start, 1e-9),
                        steps_per_second=(optimizer_step - start_step)
                        / max(time.perf_counter() - training_start, 1e-9),
                        reason="interval",
                    )
                    if progress is not None:
                        progress_logs.append(progress)
            if optimizer_step % config.save_every == 0 or optimizer_step == config.max_optimizer_steps:
                checkpoint = save_complete_checkpoint(
                    output_dir / f"checkpoint-step-{optimizer_step:06d}",
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    optimizer_step=optimizer_step,
                    cumulative_samples=cumulative_samples,
                    cumulative_tokens=cumulative_tokens,
                    data_cursors_by_rank=_gather_rank_cursors(
                        stream.cursor(include_rng=True), rank=rank, world_size=config.world_size
                    ),
                    manifest=manifest,
                    config=config,
                    report={
                        "latest_metric": metrics[-1],
                        "scheduler": schedule,
                        "progress_logging": {
                            "interval_steps": config.log_interval_steps,
                            "last": progress_logs[-1] if progress_logs else None,
                        },
                        "architecture": _validate_recursive_architecture(getattr(model, "module", model)),
                    },
                    rank=rank,
                    scheduler_config=schedule,
                )
                if checkpoint is not None:
                    latest_checkpoint = checkpoint
                if rank == 0 and checkpoint is not None:
                    # Pointer is written only after the complete atomic rename.
                    if config.gate == "FORMAL":
                        retained_checkpoints = retain_formal_checkpoints(
                            output_dir,
                            keep=config.checkpoint_retention,
                            latest_checkpoint=checkpoint,
                        )
                    else:
                        _write_json(
                            output_dir / "latest_complete_checkpoint.json",
                            {"checkpoint": str(checkpoint), "optimizer_step": optimizer_step, "complete": True},
                        )
                if dist.is_initialized():
                    dist.barrier()

        # A coordinated data-exhaustion stop can happen between configured
        # periodic saves.  Preserve the last *complete* optimizer step before
        # stopping.  Only rank 0 writes the checkpoint; every rank enters the
        # barrier so no process races ahead into teardown.
        if coordinated_stop and optimizer_step > start_step:
            final_checkpoint_path = output_dir / f"checkpoint-step-{optimizer_step:06d}"
            # This is a collective and must be reached by every rank before
            # rank 0 enters the checkpoint-writing branch.
            data_cursors_by_rank = _gather_rank_cursors(
                stream.cursor(include_rng=True), rank=rank, world_size=config.world_size
            )
            if rank == 0:
                if _complete_checkpoint_step(final_checkpoint_path) is None:
                    final_checkpoint = save_complete_checkpoint(
                        final_checkpoint_path,
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        optimizer_step=optimizer_step,
                        cumulative_samples=cumulative_samples,
                        cumulative_tokens=cumulative_tokens,
                        data_cursors_by_rank=data_cursors_by_rank,
                        manifest=manifest,
                        config=config,
                        report={
                            "latest_metric": metrics[-1] if metrics else None,
                            "scheduler": schedule,
                            "progress_logging": {
                                "interval_steps": config.log_interval_steps,
                                "last": progress_logs[-1] if progress_logs else None,
                            },
                            "stop_reason": stop_reason,
                            "steps_at_stop": optimizer_step,
                            "architecture": _validate_recursive_architecture(
                                getattr(model, "module", model)
                            ),
                        },
                        rank=rank,
                        scheduler_config=schedule,
                    )
                    latest_checkpoint = final_checkpoint
                else:
                    latest_checkpoint = final_checkpoint_path
                if config.gate != "FORMAL":
                    _write_json(
                        output_dir / "latest_complete_checkpoint.json",
                        {
                            "checkpoint": str(latest_checkpoint),
                            "optimizer_step": optimizer_step,
                            "complete": True,
                            "stop_reason": stop_reason,
                            "steps_at_stop": optimizer_step,
                        },
                    )
                if config.gate == "FORMAL":
                    retained_checkpoints = retain_formal_checkpoints(
                        output_dir,
                        keep=config.checkpoint_retention,
                        latest_checkpoint=latest_checkpoint,
                    )
            if dist.is_initialized():
                dist.barrier()
            # Keep the path in every rank's report even though only rank 0
            # owns the physical write.
            latest_checkpoint = final_checkpoint_path
        if rank == 0 and metrics:
            # An interval record may already exist at this step.  Emit a
            # dedicated final record only when the final/stop step was not
            # covered by the interval, avoiding duplicate remote log noise.
            last_logged_step = progress_logs[-1]["optimizer_step"] if progress_logs else None
            if last_logged_step != optimizer_step:
                elapsed = max(time.perf_counter() - training_start, 1e-9)
                last_metric = metrics[-1]
                progress = _emit_progress_log(
                    output_dir=output_dir,
                    rank=rank,
                    optimizer_step=optimizer_step,
                    total_steps=config.max_optimizer_steps,
                    loss=last_metric["loss"],
                    learning_rate=last_metric["learning_rate"],
                    elapsed_seconds=elapsed,
                    samples_per_second=(optimizer_step - start_step)
                    * (config.world_size * config.micro_batch_size * config.gradient_accumulation_steps)
                    / elapsed,
                    steps_per_second=(optimizer_step - start_step) / elapsed,
                    reason="training_end" if not coordinated_stop else stop_reason,
                )
                if progress is not None:
                    progress_logs.append(progress)
    finally:
        monitor.stop()
    if (
        optimizer_step == start_step
        and config.max_optimizer_steps > start_step
        and config.gate != "FORMAL"
    ):
        raise RuntimeError("No complete optimizer step was available on all ranks")
    checksum = _model_checksum_audit(model, device)
    formal_target_reached = optimizer_step >= config.max_optimizer_steps
    training_status = "PASS" if formal_target_reached or config.gate != "FORMAL" else "FAIL"
    return {
        "status": training_status,
        "global_effective_batch": config.world_size * config.micro_batch_size * config.gradient_accumulation_steps,
        "local_samples_per_optimizer_step": config.micro_batch_size * config.gradient_accumulation_steps,
        "optimizer_step_start": start_step,
        "optimizer_steps": optimizer_step,
        "optimizer_step_increment": optimizer_step - start_step,
        "optimizer_step_calls": optimizer_step - start_step,
        "optimizer_scheduler_rng_restored": bool(resume_state is not None),
        "formal_resume": formal_resume,
        "formal_resume_source": str(config.resume_from) if formal_resume and config.resume_from else None,
        "resume_smoke_step_increment": optimizer_step - start_step if resume_state is not None else None,
        "cumulative_samples": cumulative_samples,
        "cumulative_samples_per_rank": cumulative_samples,
        "cumulative_global_samples": cumulative_samples * config.world_size,
        "cumulative_valid_tokens": cumulative_tokens,
        "cumulative_global_valid_tokens": cumulative_tokens,
        "target_optimizer_steps": config.max_optimizer_steps,
        "target_samples_per_rank": config.max_optimizer_steps * config.gradient_accumulation_steps * config.micro_batch_size,
        "target_global_samples": config.max_optimizer_steps * config.world_size * config.gradient_accumulation_steps * config.micro_batch_size,
        "target_local_microbatches": config.max_optimizer_steps * config.gradient_accumulation_steps,
        "formal_optimizer_steps": config.formal_optimizer_steps,
        "formal_save_steps": (
            formal_save_steps(config.max_optimizer_steps, config.save_every)
            if config.gate == "FORMAL"
            else []
        ),
        "checkpoint_retention": config.checkpoint_retention,
        "parameter_checksum_audit": checksum,
        "metrics": metrics,
        "scheduler": schedule,
        "progress_logging": {
            "interval_steps": config.log_interval_steps,
            "rank0_only": True,
            "records": progress_logs,
            "last": progress_logs[-1] if progress_logs else None,
        },
        "last_progress": progress_logs[-1] if progress_logs else None,
        "last_loss": progress_logs[-1]["loss"] if progress_logs else None,
        "last_learning_rate": progress_logs[-1]["learning_rate"] if progress_logs else None,
        "last_samples_per_second": progress_logs[-1]["samples_per_second"] if progress_logs else None,
        "optimizer_audit": optimizer_audit,
        "sampling_contract": {
            "policy": SAMPLING_POLICY,
            "sampler_version": SAMPLER_VERSION,
            "poisson_lambda": POISSON_LAMBDA,
            "poisson_support": list(POISSON_SUPPORT),
            "poisson_normalization_z": POISSON_NORMALIZATION_Z,
            "poisson_probabilities": list(POISSON_PROBABILITIES),
            "sampler_key": SAMPLER_KEY,
            "default_inference_r": DEFAULT_INFERENCE_MIDDLE_LOOPS,
            "fixed_parameter_gradient_tail_loops": PARAMETER_GRADIENT_TAIL_LOOPS,
            "local_tmax_histogram": dict(
                Counter(str(tmax) for item in metrics for tmax in item.get("local_tmax_per_microbatch", ()))
            ),
            "per_sequence_sampling": True,
            "all_rank_r_equal": False,
        },
        "rank0_only_checkpoint": True,
        "latest_complete_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
        "retained_checkpoints": [str(path) for path in retained_checkpoints],
        "formal_target_reached": formal_target_reached,
        "target_not_reached": bool(config.gate == "FORMAL" and not formal_target_reached),
        "stop_reason": stop_reason,
        "coordinated_stop": coordinated_stop,
        "steps_at_stop": optimizer_step,
        "final_checkpoint": (
            str(latest_checkpoint)
            if (coordinated_stop or config.gate == "FORMAL") and latest_checkpoint is not None
            else None
        ),
        "final_checkpoint_saved_after_coordinated_stop": bool(
            coordinated_stop and optimizer_step > start_step and latest_checkpoint is not None
        ),
        "manifest": _json_safe(dict(manifest)),
        "data_cursor": stream.cursor(),
        "data_cursor_rank": str(rank),
        "data_cursor_source": previous_cursor_source,
        "bounded_shuffle_bitwise_exact": False,
        "data_cursor_restored": False,
        "coarse_cursor_skip_applied": bool(previous_cursor),
        "data_cursor_restore_mode": (
            "coarse_epoch_shard_complete_microbatch_skip"
            if previous_cursor
            else "fresh"
        ),
        "resume_data_semantics": (
            "optimizer/scheduler/RNG/step metadata restored; coarse epoch/shard/"
            "complete-microbatch skip applied; bounded shuffle order is not bitwise exact"
            if previous_cursor
            else "fresh data cursor; no resume cursor was provided"
        ),
        "previous_cursor": previous_cursor,
    }


def _broadcast_object(value: Any, *, rank: int) -> Any:
    """Broadcast a JSON-compatible object, with a single-rank fast path."""

    import torch.distributed as dist

    if not dist.is_initialized():
        return value
    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _load_tokenizer_only(path: Path) -> Any:
    from transformers import AutoTokenizer

    path = _require_local_artifact_dir(path, kind="tokenizer")
    return AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)


def _require_local_artifact_dir(path: Path, *, kind: str) -> Path:
    """Return a canonical local directory before Transformers can parse it.

    Without this guard, a missing HPC mount/path is passed to
    ``from_pretrained`` and Hugging Face interprets the absolute path as a
    repo id, producing a misleading ``HFValidationError``.  Stage 4 is
    strictly offline, so fail with the actual path and expected artifacts.
    """

    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Stage 4 {kind} path is not an existing local directory: {candidate}. "
            "Check the external checkpoint mount/path or pass "
            "RSMOL_5_10XR_5_TOKENIZER_PATH explicitly."
        )
    if kind == "recursive model":
        if not (candidate / "config.json").is_file():
            raise FileNotFoundError(
                f"Stage 4 recursive model directory is missing config.json: {candidate}"
            )
        required_any = ("model.safetensors", "pytorch_model.bin")
    else:
        required_any = (
            "tokenizer.json", "tokenizer.model", "vocab.json", "spiece.model",
            "tokenizer_config.json",
        )
    present = [name for name in required_any if (candidate / name).is_file()]
    if not present:
        raise FileNotFoundError(
            f"Stage 4 {kind} directory has no recognized local artifacts: {candidate}; "
            f"expected one of {required_any}."
        )
    return candidate


def _dataset_preaudit(
    config: Stage4Config,
    *,
    tokenizer: Any,
    rank: int,
    world_size: int,
    require_full_content: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the single-process content audit and broadcast its manifest.

    Gate B calls this directly without a process group.  Gates C/D run the
    expensive footer/content scan once on rank 0 and broadcast only metadata;
    actual training remains rank-local and never centralizes Parquet payloads.
    """

    import torch.distributed as dist

    if rank == 0:
        if world_size > 1 and config.audit_report is None:
            raise RuntimeError(
                "Gate C/D multi-card startup requires --audit-report from Gate B; "
                "rank 0 is forbidden from scanning the full corpus as a training preflight"
            )
        if config.audit_report is not None:
            # Prefer the standalone Gate B artifact.  This avoids a second
            # full-corpus scan before a multi-card pilot while retaining all
            # footer/content evidence in the report consumed here.
            audit_bundle = json.loads(config.audit_report.expanduser().read_text(encoding="utf-8"))
            audit = dict(audit_bundle.get("dataset_audit", audit_bundle.get("audit", audit_bundle)))
            manifest = dict(audit_bundle.get("manifest", {}))
            if not manifest:
                files = discover_parquet_files(config.data_dir)
                manifest = assign_shards(files, world_size=world_size, seed=config.seed)
        else:
            files = discover_parquet_files(config.data_dir)
            audit = audit_parquet_shards(
                files,
                tokenizer=tokenizer,
                context_length=config.context_length,
                content=require_full_content,
            )
            manifest = assign_shards(files, world_size=world_size, seed=config.seed)
        # A Gate B manifest may have been produced with world_size=1.  Never
        # silently reuse that assignment for a different DDP topology.
        if int(manifest.get("world_size", world_size)) != world_size:
            files = discover_parquet_files(config.data_dir)
            manifest = assign_shards(files, world_size=world_size, seed=config.seed)
        by_name = {item["name"]: item for item in audit["shards"]}
        full_content = bool(
            audit.get(
                "content_is_full_corpus",
                audit.get("content_audit") and not audit.get("footer_only"),
            )
        )
        rank_valid: dict[str, int | None] = {}
        for rank_key, shard_paths in manifest["rank_shards"].items():
            rank_valid[rank_key] = (
                sum(
                    int(by_name[Path(path).name].get("content", {}).get("valid_trainable_rows", 0))
                    for path in shard_paths
                )
                if full_content
                else None
            )
        manifest["rank_raw_rows"] = {
            rank_key: sum(int(by_name[Path(path).name]["num_rows"]) for path in shard_paths)
            for rank_key, shard_paths in manifest["rank_shards"].items()
        }
        manifest["rank_valid_trainable_rows"] = rank_valid
        manifest["rank_effective_rows_scope"] = (
            "full_corpus" if full_content else "unavailable_sample_only"
        )
        target_samples_per_rank = (
            int(config.max_optimizer_steps)
            * config.gradient_accumulation_steps
            * config.micro_batch_size
        )
        manifest["target_optimizer_steps"] = int(config.max_optimizer_steps)
        manifest["target_samples_per_rank"] = target_samples_per_rank
        manifest["formal_optimizer_steps"] = int(config.formal_optimizer_steps)
        manifest["rank_has_target_samples"] = {
            rank_key: (value >= target_samples_per_rank if value is not None else None)
            for rank_key, value in rank_valid.items()
        }
        manifest["rank_has_raw_capacity"] = {
            rank_key: value >= target_samples_per_rank
            for rank_key, value in manifest["rank_raw_rows"].items()
        }
        manifest["raw_rows"] = int(audit["total_rows"])
        manifest["effective_trainable_rows"] = (
            int(audit["content"]["valid_trainable_rows"]) if full_content else None
        )
        manifest["effective_rows_scope"] = (
            "full_corpus" if full_content else "unavailable_sample_only"
        )
        manifest["formal_global_samples"] = int(
            config.formal_optimizer_steps
            * config.gradient_accumulation_steps
            * config.micro_batch_size
            * world_size
        )
        manifest["formal_remaining_raw_rows"] = int(
            audit["total_rows"] - manifest["formal_global_samples"]
        )
        bundle = {"audit": audit, "manifest": manifest}
    else:
        bundle = None
    bundle = _broadcast_object(bundle, rank=rank)
    if not bundle or "audit" not in bundle or "manifest" not in bundle:
        raise RuntimeError("Failed to broadcast Stage 4 dataset audit/manifest")
    audit, manifest = bundle["audit"], bundle["manifest"]
    has_full_effective_rows = manifest.get("effective_rows_scope") == "full_corpus"
    capacity_key = "rank_has_target_samples" if has_full_effective_rows else "rank_has_raw_capacity"
    missing = [
        rank_key
        for rank_key, ok in manifest.get(capacity_key, {}).items()
        if not ok
    ]
    if missing:
        if has_full_effective_rows:
            raise RuntimeError(
                "Fail-fast: rank shard(s) lack the required target effective samples: "
                f"{missing}; valid={manifest.get('rank_valid_trainable_rows')}"
            )
        raise RuntimeError(
            "Fail-fast: rank shard(s) lack the required target raw-row capacity: "
            f"{missing}; raw={manifest.get('rank_raw_rows')}"
        )
    if dist.is_initialized():
        dist.barrier()
    return audit, manifest


def gate_b_real_data_preaudit(
    config: Stage4Config,
    *,
    tokenizer: Any,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Public Gate B helper; call it from a non-torchrun process."""

    if world_size != 1:
        raise ValueError("Gate B pre-audit helper requires world_size=1")
    return _dataset_preaudit(config, tokenizer=tokenizer, rank=rank, world_size=world_size)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = ensure_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    """Append one auditable record without rewriting prior runtime records."""

    path = ensure_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_safe(value), ensure_ascii=False) + "\n")
        stream.flush()


def _stage4_event(output_dir: Path, rank: int, phase: str, **fields: Any) -> None:
    """Persist and print rank-local lifecycle events for diagnosing slow starts.

    Gate E can spend substantial time loading a checkpoint and replaying a
    coarse data cursor before the first optimizer step.  These events make
    that interval observable without changing the training protocol.
    """

    record = {
        "timestamp": time.time(),
        "rank": int(rank),
        "phase": str(phase),
        **{str(key): _json_safe(value) for key, value in fields.items()},
    }
    _append_jsonl(output_dir / f"stage4_lifecycle_rank{rank}.jsonl", record)
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(f"[stage4-lifecycle][rank={rank}] phase={phase}{suffix}", flush=True)


def _synthetic_model(config: Stage4Config, device: Any) -> Any:
    import torch
    from transformers import LlamaConfig

    from code.RSmol.recursive_model_5_10xr_5 import RecursiveLlama5_10xr_5ForCausalLM

    # Gate A is an exact architecture audit, not the legacy configurable
    # 4-layer fixture: it must instantiate all 20 physical modules and the
    # complete max-depth 5-10xr-5 logical schedule.
    logical = 110
    physical = 20
    llama_config = LlamaConfig(
        vocab_size=config.synthetic_vocab_size,
        hidden_size=config.synthetic_hidden_size,
        intermediate_size=config.synthetic_hidden_size * 2,
        num_hidden_layers=logical,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        _attn_implementation="eager",
    )
    llama_config.recursive_source_num_hidden_layers = 30
    llama_config.recursive_source_layer_count = 30
    llama_config.recursive_loops = 10
    llama_config.recursive_min_middle_loops = 4
    llama_config.recursive_max_middle_loops = 10
    llama_config.recursive_default_inference_middle_loops = 7
    llama_config.recursive_parameter_gradient_tail_loops = 4
    llama_config.recursive_sampling_policy = SAMPLING_POLICY
    llama_config.recursive_sampler_version = SAMPLER_VERSION
    llama_config.recursive_poisson_lambda = POISSON_LAMBDA
    llama_config.recursive_poisson_support = list(POISSON_SUPPORT)
    llama_config.recursive_poisson_normalization_z = POISSON_NORMALIZATION_Z
    llama_config.recursive_poisson_Z = POISSON_NORMALIZATION_Z
    llama_config.recursive_poisson_probabilities = list(POISSON_PROBABILITIES)
    llama_config.recursive_sampler_key = SAMPLER_KEY
    llama_config.recursive_loops_scope = "middle_only"
    llama_config.recursive_layer_count = physical
    llama_config.recursive_mapping_policy = "stage4_5_10xr_5_synthetic_fixture"
    llama_config.recursive_source_layer_indices_0based = [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29]
    llama_config.recursive_source_layer_indices_1based = [index + 1 for index in llama_config.recursive_source_layer_indices_0based]
    llama_config.recursive_prefix_layer_count = 5
    llama_config.recursive_middle_layer_count = 10
    llama_config.recursive_suffix_layer_count = 5
    llama_config.middle_recurrent_count = 10
    llama_config.logical_to_physical = list(LOGICAL_TO_PHYSICAL)
    llama_config.recursive_logical_to_physical = list(LOGICAL_TO_PHYSICAL)
    llama_config.use_cache = False
    model = RecursiveLlama5_10xr_5ForCausalLM(llama_config)
    model.to(device=device, dtype=torch.float32)
    model.train()
    _validate_recursive_architecture(model, production=False)
    return model


def _parse_args(argv: Sequence[str] | None = None) -> Stage4Config:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    def has_option(*names: str) -> bool:
        return any(
            argument == name or argument.startswith(name + "=")
            for argument in raw_argv
            for name in names
        )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate", type=str.upper, choices=("A", "B", "C", "D", "E", "FORMAL"), default="A"
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--audit-report", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=DEFAULT_MICRO_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-lr", dest="max_lr", type=float, default=DEFAULT_MAX_LR)
    parser.add_argument("--min-lr", dest="min_lr", type=float, default=DEFAULT_MIN_LR)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-optimizer-steps", "--max-steps", dest="max_optimizer_steps", type=int, default=DEFAULT_MAX_OPTIMIZER_STEPS)
    parser.add_argument("--formal-optimizer-steps", type=int, default=DEFAULT_FORMAL_OPTIMIZER_STEPS)
    parser.add_argument("--scheduler-total-steps", type=int, default=None)
    parser.add_argument("--log-interval-steps", type=int, default=DEFAULT_LOG_INTERVAL_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--record-buffer-size", type=int, default=DEFAULT_RECORD_BUFFER_SIZE)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--checkpoint-retention", type=int, default=DEFAULT_FORMAL_CHECKPOINT_RETENTION
    )
    parser.add_argument("--monitor-interval-seconds", type=float, default=60.0)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE)
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--dry-run", "--static-check", dest="dry_run", action="store_true")
    parser.add_argument("--allow-non8", action="store_true")
    parser.add_argument("--synthetic-layers", type=int, default=30)
    parser.add_argument("--synthetic-hidden-size", type=int, default=32)
    parser.add_argument("--synthetic-vocab-size", type=int, default=97)
    args = parser.parse_args(argv)
    config = Stage4Config(**vars(args))
    if config.gate in {"B", "C"}:
        raise ValueError(
            f"Stage 4 5-10xr-5 Gate {config.gate} is unsupported; reuse the existing Gate B audit JSON "
            "and run only Gate A, Gate D, Gate E, or FORMAL."
        )
    formal_explicit = config.gate == "FORMAL"
    if formal_explicit:
        # FORMAL is an independent training contract.  Defaults from the
        # pilot parser are not silently reused, and explicit conflicts fail
        # before distributed/model startup.
        max_steps_explicit = has_option("--max-optimizer-steps", "--max-steps")
        if max_steps_explicit and config.max_optimizer_steps != DEFAULT_FORMAL_OPTIMIZER_STEPS:
            raise ValueError(
                "FORMAL requires max_optimizer_steps=9244; refusing an explicit conflicting value"
            )
        if has_option("--formal-optimizer-steps") and config.formal_optimizer_steps != DEFAULT_FORMAL_OPTIMIZER_STEPS:
            raise ValueError("FORMAL requires formal_optimizer_steps=9244")
        if has_option("--scheduler-total-steps") and config.scheduler_total_steps != DEFAULT_FORMAL_OPTIMIZER_STEPS:
            raise ValueError("FORMAL requires scheduler_total_steps=9244")
        if has_option("--warmup-steps") and config.warmup_steps not in (0, DEFAULT_FORMAL_WARMUP_STEPS):
            raise ValueError("FORMAL requires warmup_steps=463 (or 0 for the automatic 5% default)")
        if has_option("--save-every") and config.save_every != DEFAULT_FORMAL_SAVE_EVERY:
            raise ValueError("FORMAL requires save_every=500")
        if has_option("--checkpoint-retention") and config.checkpoint_retention != DEFAULT_FORMAL_CHECKPOINT_RETENTION:
            raise ValueError("FORMAL requires checkpoint_retention=3")
        if config.world_size != DEFAULT_WORLD_SIZE:
            raise ValueError("FORMAL requires world_size=8")
        if str(config.backend).lower() != "nccl":
            raise ValueError("FORMAL requires backend=nccl")
        if config.allow_non8:
            raise ValueError("FORMAL does not allow --allow-non8")
        config.max_optimizer_steps = DEFAULT_FORMAL_OPTIMIZER_STEPS
        config.formal_optimizer_steps = DEFAULT_FORMAL_OPTIMIZER_STEPS
        config.scheduler_total_steps = DEFAULT_FORMAL_OPTIMIZER_STEPS
        config.warmup_steps = DEFAULT_FORMAL_WARMUP_STEPS
        config.save_every = DEFAULT_FORMAL_SAVE_EVERY
        config.checkpoint_retention = DEFAULT_FORMAL_CHECKPOINT_RETENTION
    if config.micro_batch_size != 2:
        raise ValueError("Stage 4 requires micro_batch_size=2")
    if config.gradient_accumulation_steps != 64:
        raise ValueError("Stage 4 requires gradient_accumulation_steps=64")
    if config.learning_rate != DEFAULT_MAX_LR:
        raise ValueError("Stage 4 requires learning_rate=2e-4")
    if config.max_lr != DEFAULT_MAX_LR:
        raise ValueError("Stage 4 requires max_lr=2e-4")
    if tuple(float(value) for value in config.optimizer_betas) != DEFAULT_ADAMW_BETAS:
        raise ValueError(f"Stage 4 requires AdamW betas={DEFAULT_ADAMW_BETAS}")
    if float(config.optimizer_eps) != DEFAULT_ADAMW_EPS:
        raise ValueError(f"Stage 4 requires AdamW eps={DEFAULT_ADAMW_EPS}")
    if float(config.weight_decay) != DEFAULT_ADAMW_WEIGHT_DECAY:
        raise ValueError(f"Stage 4 requires AdamW weight_decay={DEFAULT_ADAMW_WEIGHT_DECAY}")
    if bool(config.optimizer_amsgrad) is not DEFAULT_ADAMW_AMSGRAD:
        raise ValueError(f"Stage 4 requires AdamW amsgrad={DEFAULT_ADAMW_AMSGRAD}")
    if config.min_lr <= 0 or config.min_lr > config.max_lr:
        raise ValueError("Stage 4 requires 0 < min_lr <= max_lr")
    if config.context_length <= 0 or config.context_length > 1024:
        raise ValueError("Stage 4 context_length must be in [1, 1024]")
    if config.max_optimizer_steps <= 0:
        raise ValueError("max_optimizer_steps must be positive")
    if config.gate == "D" and config.max_optimizer_steps != DEFAULT_MAX_OPTIMIZER_STEPS:
        raise ValueError(
            "Stage 4 Gate D is a fixed ten-optimizer-step smoke; "
            f"got max_optimizer_steps={config.max_optimizer_steps}. "
            "The 9244-step value is formal_optimizer_steps only."
        )
    if config.gate == "FORMAL":
        if config.max_optimizer_steps != DEFAULT_FORMAL_OPTIMIZER_STEPS:
            raise ValueError("FORMAL requires exactly 9244 optimizer steps")
        if config.scheduler_total_steps != DEFAULT_FORMAL_OPTIMIZER_STEPS:
            raise ValueError("FORMAL requires scheduler_total_steps=9244")
        if config.warmup_steps != DEFAULT_FORMAL_WARMUP_STEPS:
            raise ValueError("FORMAL requires warmup_steps=463")
        if config.save_every != DEFAULT_FORMAL_SAVE_EVERY:
            raise ValueError("FORMAL requires save_every=500")
        if config.checkpoint_retention != DEFAULT_FORMAL_CHECKPOINT_RETENTION:
            raise ValueError("FORMAL requires checkpoint_retention=3")
    if config.formal_optimizer_steps <= 0:
        raise ValueError("formal_optimizer_steps must be positive")
    if config.scheduler_total_steps is not None and config.scheduler_total_steps <= 0:
        raise ValueError("scheduler_total_steps must be positive")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.log_interval_steps <= 0:
        raise ValueError("log_interval_steps must be positive")
    max_steps_explicit = any(
        option in raw_argv
        for option in ("--max-optimizer-steps", "--max-steps")
    )
    if (
        config.gate == "C"
        and not max_steps_explicit
        and config.max_optimizer_steps == DEFAULT_MAX_OPTIMIZER_STEPS
    ):
        config.max_optimizer_steps = 2
    config.scheduler_total_steps = int(config.scheduler_total_steps or config.max_optimizer_steps)
    config.warmup_steps = compute_warmup_steps(config.scheduler_total_steps, config.warmup_steps)
    if config.record_buffer_size <= 0:
        raise ValueError("record_buffer_size must be positive")
    if config.save_every <= 0:
        raise ValueError("save_every must be positive")
    if config.checkpoint_retention <= 0:
        raise ValueError("checkpoint_retention must be positive")
    if config.world_size <= 0:
        raise ValueError("world_size must be positive")
    if config.report_path is None:
        config.report_path = config.output_dir / "stage4_report.json"
    return config


def run(config: Stage4Config) -> dict[str, Any]:
    """Execute one Stage 4 gate and return its auditable JSON report."""

    faulthandler.enable()
    if str(config.gate).upper() in {"B", "C"}:
        raise ValueError(
            f"Stage 4 5-10xr-5 Gate {config.gate} is unsupported; Gate B/C are intentionally not rerun."
        )
    if config.gate == "D" and config.max_optimizer_steps != DEFAULT_MAX_OPTIMIZER_STEPS:
        raise ValueError(
            "Stage 4 Gate D is a fixed ten-optimizer-step smoke; "
            f"got max_optimizer_steps={config.max_optimizer_steps}. "
            "The 9244-step value is formal_optimizer_steps only."
        )
    if config.gate == "E" and not config.dry_run and config.resume_from is None:
        raise ValueError("Gate E requires --resume-from (the complete checkpoint is its model source)")
    if config.resume_from is not None and not config.dry_run:
        # Canonicalize and validate before any model loading.  In particular,
        # Gate E must not accidentally fall back to --model-path when the
        # resume directory is absent or incomplete.
        config.resume_from = ensure_external_resume(config.resume_from)
    output_dir = ensure_external_output(config.output_dir)
    report_path = ensure_external_output(config.report_path or output_dir / "stage4_report.json")
    if not config.dry_run and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Stage 4 refuses to overwrite a non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.dry_run:
        report = {
            "status": "PASS",
            "gate": config.gate,
            "mode": "formal" if config.gate == "FORMAL" else "gate",
            "dry_run": True,
            "configuration": asdict(config),
            "data_loader_contract": {
                "num_workers": NUM_WORKERS,
                "pin_memory": PIN_MEMORY,
                "persistent_workers": PERSISTENT_WORKERS,
                "dataset_num_proc": DATASET_NUM_PROC,
            },
            "expected_target_samples_per_rank": (
                config.max_optimizer_steps
                * config.gradient_accumulation_steps
                * config.micro_batch_size
            ),
            "expected_global_effective_batch": DEFAULT_GLOBAL_EFFECTIVE_BATCH,
            "expected_formal_samples_per_rank": DEFAULT_FORMAL_SAMPLES_PER_RANK,
            "expected_formal_local_microbatches": DEFAULT_FORMAL_LOCAL_MICROBATCHES,
            "expected_formal_global_samples": DEFAULT_FORMAL_GLOBAL_SAMPLES,
            "formal_optimizer_steps": config.formal_optimizer_steps,
            "gate_d_smoke_optimizer_steps": DEFAULT_MAX_OPTIMIZER_STEPS,
            "formal_save_steps": formal_save_steps(
                config.max_optimizer_steps if config.gate == "FORMAL" else config.formal_optimizer_steps,
                config.save_every,
            ) if config.gate == "FORMAL" else [],
            "checkpoint_retention": config.checkpoint_retention,
            "sampling_contract": {
                "policy": SAMPLING_POLICY,
                "sampler_version": SAMPLER_VERSION,
                "poisson_lambda": POISSON_LAMBDA,
                "poisson_support": list(POISSON_SUPPORT),
                "poisson_normalization_z": POISSON_NORMALIZATION_Z,
                "poisson_probabilities": list(POISSON_PROBABILITIES),
                "sampler_key": SAMPLER_KEY,
                "default_inference_r": DEFAULT_INFERENCE_MIDDLE_LOOPS,
                "fixed_parameter_gradient_tail_loops": PARAMETER_GRADIENT_TAIL_LOOPS,
            },
            "scheduler": scheduler_metadata(
                total_steps=int(config.scheduler_total_steps or config.max_optimizer_steps),
                warmup_steps=config.warmup_steps,
                max_lr=config.max_lr,
                min_lr=config.min_lr,
                target_steps=config.max_optimizer_steps,
                formal_steps=config.formal_optimizer_steps,
            ),
            "progress_logging": {
                "interval_steps": config.log_interval_steps,
                "rank0_only": True,
                "last": None,
            },
            "output_dir_external": True,
            "no_model_or_checkpoint_written": True,
        }
        _write_json(report_path, report)
        return report

    rank, world_size, device, use_cuda = _prepare_distributed(config)
    _set_seed(config.seed, rank=rank)
    process_group = __import__("torch.distributed", fromlist=["distributed"])
    try:
        _stage4_event(
            output_dir,
            rank,
            "distributed_ready",
            gate=config.gate,
            world_size=world_size,
            device=str(device),
        )
        if config.gate == "A":
            model = _synthetic_model(config, device)
            model = _ddp_wrap(model, config, device)
            gate_report = _synthetic_gate_a(model, device=device, config=config, rank=rank)
            report = {"configuration": asdict(config), "git_commit": _git_commit(), **gate_report}
            if rank == 0:
                # Gate A uses a tiny JSON checkpoint marker rather than
                # writing synthetic model parameters; this still audits the
                # rank-0-only checkpoint ownership contract.
                _write_json(
                    output_dir / "stage4_gate_A_rank0_checkpoint.json",
                    {"complete": True, "rank0_only": True, "synthetic": True},
                )
                _write_json(report_path, report)
                _write_json(output_dir / "stage4_gate_A_audit.json", report)
            if process_group.is_initialized():
                process_group.barrier()
            return report

        if config.gate == "B":
            if world_size != 1 or process_group.is_initialized():
                raise RuntimeError("Gate B is single-process only and must not be launched with torchrun")
            model_path = config.model_path
            if model_path is None:
                raise ValueError("Gate B requires --model-path for the converted recursive checkpoint")
            tokenizer = _load_tokenizer_only(config.tokenizer_path or model_path)
            files = discover_parquet_files(config.data_dir)
            sampled_files = select_sample_shards(files, sample_shards=3, seed=config.seed)
            audit = audit_parquet_shards(
                files,
                tokenizer=tokenizer,
                context_length=config.context_length,
                content=True,
                content_paths=sampled_files,
                progress_callback=lambda message: print(message, flush=True),
            )
            manifest = assign_shards(files, world_size=world_size, seed=config.seed)
            by_name = {item["name"]: item for item in audit["shards"]}
            target_samples_per_rank = (
                int(config.max_optimizer_steps)
                * config.gradient_accumulation_steps
                * config.micro_batch_size
            )
            manifest["rank_raw_rows"] = {"0": audit["total_rows"]}
            manifest["rank_valid_trainable_rows"] = {"0": None}
            manifest["rank_effective_rows_scope"] = "unavailable_sample_only"
            manifest["target_optimizer_steps"] = int(config.max_optimizer_steps)
            manifest["target_samples_per_rank"] = target_samples_per_rank
            manifest["formal_optimizer_steps"] = int(config.formal_optimizer_steps)
            manifest["rank_has_target_samples"] = {"0": None}
            manifest["rank_has_raw_capacity"] = {
                "0": audit["total_rows"] >= target_samples_per_rank
            }
            if not manifest["rank_has_raw_capacity"]["0"]:
                raise RuntimeError(
                    "Gate B fail-fast: the footer audit has fewer than "
                    f"{target_samples_per_rank} raw samples"
                )
            manifest["raw_rows"] = audit["total_rows"]
            manifest["effective_trainable_rows"] = None
            manifest["effective_rows_scope"] = "unavailable_sample_only"
            manifest["formal_optimizer_steps"] = int(config.formal_optimizer_steps)
            manifest["gate_d_smoke_optimizer_steps"] = DEFAULT_MAX_OPTIMIZER_STEPS
            manifest["formal_global_samples"] = DEFAULT_FORMAL_GLOBAL_SAMPLES
            manifest["formal_remaining_raw_rows"] = audit["total_rows"] - manifest["formal_global_samples"]
            report = {
                "status": "PASS",
                "gate": "B",
                "configuration": asdict(config),
                "dataset_audit": audit,
                "manifest": manifest,
                "raw_rows": audit["total_rows"],
                "effective_rows": None,
                "effective_rows_scope": "unavailable_sample_only",
                "formal_global_samples": manifest["formal_global_samples"],
                "formal_optimizer_steps": config.formal_optimizer_steps,
                "gate_d_smoke_optimizer_steps": DEFAULT_MAX_OPTIMIZER_STEPS,
                "scheduler": scheduler_metadata(
                    total_steps=int(config.scheduler_total_steps or config.max_optimizer_steps),
                    warmup_steps=config.warmup_steps,
                    max_lr=config.max_lr,
                    min_lr=config.min_lr,
                    target_steps=config.max_optimizer_steps,
                    formal_steps=config.formal_optimizer_steps,
                ),
                "progress_logging": {
                    "interval_steps": config.log_interval_steps,
                    "rank0_only": True,
                    "last": None,
                },
                "formal_remaining_raw_rows": manifest["formal_remaining_raw_rows"],
                "torchrun_started": False,
                "cache_files_audited": True,
                "cache_files_paths": [],
                "cache_policy": "datasets library not used; no HF Arrow cache created",
                "datasets_library_used": False,
                "git_commit": _git_commit(),
                "data_loader_contract": {
                    "num_workers": NUM_WORKERS,
                    "pin_memory": PIN_MEMORY,
                    "persistent_workers": PERSISTENT_WORKERS,
                    "dataset_num_proc": DATASET_NUM_PROC,
                },
            }
            _write_json(report_path, report)
            _write_json(output_dir / "stage4_gate_B_audit.json", report)
            return report

        if config.gate == "E":
            if config.resume_from is None:
                raise ValueError("Gate E requires --resume-from")
        elif config.model_path is None and config.resume_from is None:
            raise ValueError(f"Gate {config.gate} requires --model-path")
        formal_resume = config.gate == "FORMAL" and config.resume_from is not None
        if config.gate in ("C", "D", "FORMAL") and world_size > 1 and config.audit_report is None and not formal_resume:
            raise RuntimeError(
                "Gate C/D/FORMAL requires an external Gate B --audit-report before multi-card startup"
            )
        model_path = config.resume_from or config.model_path
        _stage4_event(output_dir, rank, "model_load_start", model_path=str(model_path))
        model, tokenizer = _load_runtime_model(model_path, device, config.tokenizer_path)
        _stage4_event(output_dir, rank, "model_load_done")
        architecture_audit = _validate_recursive_architecture(model)
        _stage4_event(output_dir, rank, "architecture_audit_done")
        forward_trace_audit = recursive_forward_trace_audit(model, device=device)
        _stage4_event(output_dir, rank, "forward_trace_done")
        if formal_resume:
            import torch

            state = torch.load(
                (config.resume_from or Path(".")) / "training_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            resume_metadata = validate_formal_resume_state(state)
            resume_step = resume_metadata["optimizer_step"]
            manifest = dict(state.get("manifest", {}))
            if not manifest:
                raise ValueError("FORMAL resume checkpoint does not include a shard manifest")
            audit = {
                "resume_manifest_source": str(config.resume_from),
                "formal_resume": True,
                "resume_optimizer_step": resume_step,
                "data_audit_reused_from_checkpoint": True,
            }
        elif config.gate in ("C", "D", "FORMAL"):
            audit, manifest = _dataset_preaudit(config, tokenizer=tokenizer, rank=rank, world_size=world_size)
        else:
            resume_state_path = (config.resume_from or Path(".")) / "training_state.pt"
            import torch

            state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
            state_configuration = state.get("configuration", {})
            if (
                config.gate == "E"
                and isinstance(state_configuration, Mapping)
                and str(state_configuration.get("gate", "")).upper() == "FORMAL"
            ):
                raise ValueError(
                    "Gate E cannot consume a FORMAL checkpoint: FORMAL resume is not implemented by Gate E; "
                    "use --gate FORMAL to continue the formal target"
                )
            _stage4_event(
                output_dir,
                rank,
                "resume_metadata_loaded",
                checkpoint=str(config.resume_from),
                checkpoint_step=int(state.get("optimizer_step", -1)),
            )
            manifest = dict(state.get("manifest", {}))
            if not manifest:
                raise ValueError("Resume checkpoint does not include a Stage 4 shard manifest")
            audit = {"resume_manifest_source": str(config.resume_from)}
            old_step = int(state["optimizer_step"])
            # Gate E is intentionally a 1--2-step continuation smoke, not a
            # second long run.  ``max_optimizer_steps`` is therefore treated
            # as an upper bound only for the non-resume gates.
            config.max_optimizer_steps = old_step + 2
            saved_scheduler = state.get("scheduler_config")
            if not isinstance(saved_scheduler, Mapping):
                saved_report = state.get("report", {})
                saved_scheduler = (
                    dict(saved_report.get("scheduler", {}))
                    if isinstance(saved_report, Mapping)
                    else {}
                )
            if not isinstance(saved_scheduler, Mapping):
                saved_scheduler = {}
            # Resume with the checkpoint's original schedule domain.  This is
            # what makes an old final-step LR remain continuous when Gate E
            # intentionally runs two additional smoke steps.
            config.scheduler_total_steps = int(
                saved_scheduler.get(
                    "total_steps_for_schedule",
                    state.get("configuration", {}).get("scheduler_total_steps", old_step),
                )
            )
            config.max_lr = float(saved_scheduler.get("max_lr", config.max_lr))
            config.min_lr = float(saved_scheduler.get("min_lr", config.min_lr))
            config.learning_rate = config.max_lr
            config.warmup_steps = int(
                saved_scheduler.get(
                    "warmup_steps",
                    state.get("configuration", {}).get("warmup_steps", 0),
                )
            )
        if config.gate == "C" and config.max_optimizer_steps == DEFAULT_MAX_OPTIMIZER_STEPS:
            config.max_optimizer_steps = 2
            config.scheduler_total_steps = 2
        if config.gate == "FORMAL":
            # Keep this assertion next to runtime schedule finalization so a
            # hand-built Stage4Config cannot bypass the parser contract.
            if config.max_optimizer_steps != DEFAULT_FORMAL_OPTIMIZER_STEPS:
                raise ValueError("FORMAL requires exactly 9244 optimizer steps")
            if config.scheduler_total_steps != DEFAULT_FORMAL_OPTIMIZER_STEPS:
                raise ValueError("FORMAL requires scheduler_total_steps=9244")
            if config.warmup_steps != DEFAULT_FORMAL_WARMUP_STEPS:
                raise ValueError("FORMAL requires warmup_steps=463")
            if config.save_every != DEFAULT_FORMAL_SAVE_EVERY:
                raise ValueError("FORMAL requires save_every=500")
            if config.checkpoint_retention != DEFAULT_FORMAL_CHECKPOINT_RETENTION:
                raise ValueError("FORMAL requires checkpoint_retention=3")
        config.scheduler_total_steps = int(config.scheduler_total_steps or config.max_optimizer_steps)
        config.warmup_steps = compute_warmup_steps(config.scheduler_total_steps, config.warmup_steps)
        if rank == 0:
            _write_json(output_dir / "stage4_manifest.json", manifest)
        if process_group.is_initialized():
            process_group.barrier()
        _stage4_event(output_dir, rank, "training_start", max_optimizer_steps=config.max_optimizer_steps)
        resume_state = None
        if config.resume_from is not None:
            resume_state = (0, 0, 0, {}, manifest)
        train_report = run_training(
            model,
            tokenizer,
            config=config,
            device=device,
            rank=rank,
            manifest=manifest,
            output_dir=output_dir,
            resume_state=resume_state,
        )
        _stage4_event(
            output_dir,
            rank,
            "training_done",
            optimizer_steps=train_report.get("optimizer_steps"),
            stop_reason=train_report.get("stop_reason"),
        )
        if config.gate == "E":
            # Gate E is a resume smoke, so make the intentionally coarse data
            # semantics a hard, reportable assertion rather than an implicit
            # claim that the bounded shuffle cursor was restored exactly.
            all_ranks_coarse_skip = _all_reduce_min_flag(
                bool(train_report.get("coarse_cursor_skip_applied")), device
            )
            if not all_ranks_coarse_skip:
                raise RuntimeError("Gate E resume did not apply a coarse data cursor skip")
            if train_report.get("data_cursor_restored") is not False:
                raise RuntimeError("Gate E must report data_cursor_restored=false for bounded shuffle")
            train_report["gate_e_resume_semantics_verified"] = True
            resume_start_step = int(train_report.get("optimizer_step_start", 0))
            resumed_depth_sequence = [
                item.get("middle_loop_counts_per_microbatch", [])
                for item in train_report.get("metrics", [])
            ]
            uninterrupted_reference_depth_sequence = [
                [
                    sample_middle_loop_counts(
                        config.seed, rank, resume_start_step + offset,
                        micro_index, config.micro_batch_size,
                    ).tolist()
                    for micro_index in range(config.gradient_accumulation_steps)
                ]
                for offset in range(len(resumed_depth_sequence))
            ]
            if resumed_depth_sequence != uninterrupted_reference_depth_sequence:
                raise RuntimeError(
                    "Gate E resume Poisson depth sequence differs from the uninterrupted keyed sequence: "
                    f"actual={resumed_depth_sequence} expected={uninterrupted_reference_depth_sequence}"
                )
            train_report["gate_e_resume_verification"] = {
                "rank": rank,
                "step_increment": train_report.get("resume_smoke_step_increment"),
                "coarse_cursor_skip_applied": True,
                "all_ranks_coarse_cursor_skip_applied": True,
                "data_cursor_restored": False,
                "bounded_shuffle_bitwise_exact": False,
                "data_cursor_source": train_report.get("data_cursor_source"),
                "shared_parameter_updated": bool(
                    any(item.get("shared_parameter_updated") for item in train_report.get("metrics", []))
                ),
                "resume_start_optimizer_step": resume_start_step,
                "resumed_depth_sequence": resumed_depth_sequence,
                "uninterrupted_reference_depth_sequence": uninterrupted_reference_depth_sequence,
                "depth_sequence_matches_uninterrupted": True,
            }
        report = {
            "status": train_report.get("status", "PASS"),
            "gate": config.gate,
            "mode": "formal" if config.gate == "FORMAL" else "gate",
            "configuration": asdict(config),
            "dataset_audit": audit,
            "training": train_report,
            "scheduler": train_report.get("scheduler"),
            "progress_logging": train_report.get("progress_logging"),
            "last_loss": train_report.get("last_loss"),
            "last_learning_rate": train_report.get("last_learning_rate"),
            "last_samples_per_second": train_report.get("last_samples_per_second"),
            "formal_optimizer_steps": config.formal_optimizer_steps,
            "formal_save_steps": train_report.get("formal_save_steps", []),
            "checkpoint_retention": train_report.get("checkpoint_retention", config.checkpoint_retention),
            "gate_d_smoke_optimizer_steps": DEFAULT_MAX_OPTIMIZER_STEPS,
            "stop_reason": train_report.get("stop_reason"),
            "steps_at_stop": train_report.get("steps_at_stop"),
            "actual_optimizer_steps": train_report.get("optimizer_steps"),
            "cumulative_samples_per_rank": train_report.get("cumulative_samples_per_rank"),
            "cumulative_global_samples": train_report.get("cumulative_global_samples"),
            "cumulative_valid_tokens": train_report.get("cumulative_valid_tokens"),
            "formal_resume": train_report.get("formal_resume", False),
            "formal_resume_source": train_report.get("formal_resume_source"),
            "final_checkpoint": train_report.get("final_checkpoint"),
            "latest_complete_checkpoint": train_report.get("latest_complete_checkpoint"),
            "retained_checkpoints": train_report.get("retained_checkpoints", []),
            "formal_target_reached": train_report.get("formal_target_reached"),
            "target_not_reached": train_report.get("target_not_reached", False),
            "architecture_audit": architecture_audit,
            "forward_trace_audit": forward_trace_audit,
            "model_path": str(model_path),
            "world_size": world_size,
            "device": str(device),
            "cuda": use_cuda,
            "ddp": world_size > 1,
            "rank0_only_checkpoint": True,
            "cache_files_audited": True,
            "cache_files_paths": [],
            "cache_policy": "datasets library not used; no HF Arrow cache created",
            "datasets_library_used": False,
            "data_loader_contract": {
                "num_workers": NUM_WORKERS,
                "pin_memory": PIN_MEMORY,
                "persistent_workers": PERSISTENT_WORKERS,
                "dataset_num_proc": DATASET_NUM_PROC,
            },
            "git_commit": _git_commit(),
        }
        _write_json(output_dir / f"stage4_rank{rank}_audit.json", report)
        if rank == 0:
            _write_json(report_path, report)
            _write_json(output_dir / f"stage4_gate_{config.gate}_audit.json", report)
        if process_group.is_initialized():
            process_group.barrier()
        return report
    except Exception as exc:
        error = {
            "status": "FAIL",
            "gate": config.gate,
            "rank": rank,
            "world_size": world_size,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        if rank == 0:
            try:
                _write_json(output_dir / "stage4_failure.json", error)
            except Exception:
                pass
        raise
    finally:
        if process_group.is_initialized():
            process_group.destroy_process_group()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        report = run(config)
        if report.get("status") != "PASS":
            return 1
        print("[result] status=PASS", flush=True)
        return 0
    except Exception:
        print("[result] status=FAIL", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
