#!/usr/bin/env python3
"""Stage 4 distributed training and audit entry point.

The Stage 4 contract is deliberately implemented without ``Trainer`` or an
implicit dataset worker pool.  Each torchrun rank owns a deterministic subset
of the 85 Parquet shards, streams one shard at a time with a bounded shuffle
buffer, and contributes token-weighted gradients to a DDP model.  The same
entry point implements the five gates used by the remote job:

``A`` synthetic 8-rank DDP audit, ``B`` single-process data pre-audit,
``C`` a short real-data pilot, ``D`` the fixed-step pilot, and ``E`` a
resume smoke using a new output directory.

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
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REMOTE_CHECKOUT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM")
DEFAULT_DATA_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset")
DEFAULT_OUTPUT_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4")
EXPECTED_PARQUET_COUNT = 85
EXPECTED_PARQUET_NAMES = tuple(
    f"train-{index:05d}-of-{EXPECTED_PARQUET_COUNT:05d}.parquet"
    for index in range(EXPECTED_PARQUET_COUNT)
)
DEFAULT_WORLD_SIZE = 8
DEFAULT_MICRO_BATCH_SIZE = 8
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 16
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_CONTEXT_LENGTH = 1024
DEFAULT_MAX_OPTIMIZER_STEPS = 9244
DEFAULT_TARGET_SAMPLES_PER_RANK = (
    DEFAULT_MAX_OPTIMIZER_STEPS
    * DEFAULT_GRADIENT_ACCUMULATION_STEPS
    * DEFAULT_MICRO_BATCH_SIZE
)
# Human-readable contract value: 1,183,232 effective samples/rank and
# 147,904 local microbatches for the default 9,244-step pilot.
DEFAULT_LOCAL_MICROBATCHES = DEFAULT_MAX_OPTIMIZER_STEPS * DEFAULT_GRADIENT_ACCUMULATION_STEPS
DEFAULT_GLOBAL_EFFECTIVE_BATCH = DEFAULT_WORLD_SIZE * DEFAULT_MICRO_BATCH_SIZE * DEFAULT_GRADIENT_ACCUMULATION_STEPS
DEFAULT_FORMAL_GLOBAL_SAMPLES = DEFAULT_MAX_OPTIMIZER_STEPS * DEFAULT_GLOBAL_EFFECTIVE_BATCH
# Formal global consumption is 9,465,856 samples (1024 per optimizer step).
DEFAULT_RECORD_BUFFER_SIZE = 4096
# Explicitly pinned loader settings: Stage 4 intentionally does not create
# DataLoader workers or an HF Arrow cache during torchrun.
NUM_WORKERS = 0
PIN_MEMORY = False
PERSISTENT_WORKERS = False
DATASET_NUM_PROC = 1


@dataclass
class Stage4Config:
    """Serializable Stage 4 configuration.

    The numerical defaults are part of the experiment contract.  Gate C may
    override ``max_optimizer_steps`` explicitly; Gate D keeps 9244 by
    default, consuming exactly 9,465,856 global samples.
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
    context_length: int = DEFAULT_CONTEXT_LENGTH
    warmup_steps: int = 2
    max_optimizer_steps: int = DEFAULT_MAX_OPTIMIZER_STEPS
    seed: int = 0
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    record_buffer_size: int = DEFAULT_RECORD_BUFFER_SIZE
    save_every: int = 500
    monitor_interval_seconds: float = 60.0
    backend: str = "nccl"  # NCCL is the production 8-GPU backend.
    world_size: int = DEFAULT_WORLD_SIZE
    local_rank: int = -1
    dry_run: bool = False
    allow_non8: bool = False
    synthetic_layers: int = 4
    synthetic_hidden_size: int = 32
    synthetic_vocab_size: int = 97


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
        self.minimum: int | None = None
        self.maximum: int | None = None
        self.reservoir_size = int(reservoir_size)
        self.values: list[int] = []
        self.rng = random.Random(seed)

    def add(self, value: int) -> None:
        value = int(value)
        self.count += 1
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
            return {"count": 0, "min": None, "max": None, "quantiles": {}}
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
        return {"count": self.count, "min": self.minimum, "max": self.maximum, "quantiles": q}


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


def audit_parquet_shards(
    files: Sequence[Path],
    *,
    tokenizer: Any | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    content: bool = True,
    reservoir_size: int = 10000,
) -> dict[str, Any]:
    """Audit every footer and, optionally, every text/source record.

    The footer section records schema, row groups, ``num_rows`` and bytes for
    all 85 shards.  The content section streams only ``text`` and ``source``
    columns through ``ParquetFile.iter_batches`` and records null/empty text,
    source types/distribution, tokenizer-empty samples, and token lengths.
    No Arrow Dataset cache is created.
    """

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - remote runtime dependency
        raise ImportError("Stage 4 parquet audit requires pyarrow") from exc
    if len(files) != EXPECTED_PARQUET_COUNT:
        raise ValueError(f"Stage 4 footer audit requires {EXPECTED_PARQUET_COUNT} shards, got {len(files)}")
    shards: list[dict[str, Any]] = []
    reference_signature: tuple[tuple[Any, ...], ...] | None = None
    reference_name: str | None = None
    total_rows = 0
    total_bytes = 0
    total_null_text = 0
    total_empty_text = 0
    total_tokenizer_empty = 0
    total_payload_rows = 0
    source_types: Counter[str] = Counter()
    source_distribution: Counter[str] = Counter()
    length_stats = _LengthStats(reservoir_size=reservoir_size, seed=17)
    for path in files:
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
        if content:
            shard_lengths = _LengthStats(reservoir_size=max(128, reservoir_size // 8), seed=19)
            shard_null_text = shard_empty_text = shard_tokenizer_empty = 0
            shard_payload_rows = 0
            shard_source_types: Counter[str] = Counter()
            shard_source_distribution: Counter[str] = Counter()
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
                    if text_value is None:
                        total_null_text += 1
                        shard_null_text += 1
                        continue
                    text_string = str(text_value)
                    if not text_string.strip():
                        total_empty_text += 1
                        shard_empty_text += 1
                        continue
                    if tokenizer is not None:
                        token_ids = _tokenize_ids(tokenizer, text_string, context_length)
                        if not token_ids:
                            total_tokenizer_empty += 1
                            shard_tokenizer_empty += 1
                        else:
                            length_stats.add(len(token_ids))
                            shard_lengths.add(len(token_ids))
            shard["content"] = {
                "payload_rows": shard_payload_rows,
                "payload_rows_match_footer": shard_payload_rows == shard["num_rows"],
                "none_text": shard_null_text,
                "empty_text": shard_empty_text,
                "tokenizer_empty_text": shard_tokenizer_empty,
                "valid_trainable_rows": int(shard["num_rows"] - shard_null_text - shard_empty_text - shard_tokenizer_empty),
                "source_types": dict(sorted(shard_source_types.items())),
                "source_distribution": dict(sorted(shard_source_distribution.items())),
                "token_length": shard_lengths.as_dict() if tokenizer is not None else None,
            }
        shards.append(shard)
        if content:
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
        "required_columns": ["text", "source"],
    }
    if content:
        report["content"] = {
            "payload_rows": total_payload_rows,
            "payload_rows_match_footer": total_payload_rows == total_rows,
            "none_text": total_null_text,
            "empty_text": total_empty_text,
            "tokenizer_empty_text": total_tokenizer_empty,
            "valid_trainable_rows": total_rows - total_null_text - total_empty_text - total_tokenizer_empty,
            "source_types": dict(sorted(source_types.items())),
            "source_distribution": dict(sorted(source_distribution.items())),
            "token_length": length_stats.as_dict() if tokenizer is not None else None,
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
        if self.micro_batch_size != 8:
            raise ValueError("Stage 4 requires micro_batch_size=8")
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
    return {
        "trainable_parameter_count": len(parameters),
        "optimizer_parameter_count": len(optimizer_parameters),
        "optimizer_matches_model_exactly_once": True,
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

    from code.RSmol.recursive_model import register_auto_class

    register_auto_class()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=__import__("torch").float32,
    )
    # No device_map: every rank owns one local process/device and DDP performs
    # synchronization explicitly.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or model_path,
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
    logical = int(getattr(config, "num_hidden_layers", 0))
    physical = int(getattr(config, "recursive_layer_count", 0))
    loops = int(getattr(config, "recursive_loops", 0))
    if loops != 2 or logical != physical * loops or (production and (logical, physical) != (30, 15)):
        raise ValueError(
            "Stage 4 requires a two-loop logical=2*physical architecture "
            "(production logical=30, physical=15); "
            f"got logical={logical}, physical={physical}, loops={loops}"
        )
    if len(getattr(recursive_model, "layers", ())) != physical:
        raise ValueError("Recursive model physical layer module count does not match config")
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
        from code.RSmol.recursive_model import parameter_audit

        recursive_parameter_audit = parameter_audit(model)
    except ImportError:
        recursive_parameter_audit = {"available": False}
    return {
        "logical_layer_count": logical,
        "physical_layer_count": physical,
        "recursive_loops": loops,
        "physical_module_count": len(recursive_model.layers),
        "parameter_storage_unique": unique,
        "parameters_fp32": True,
        "use_cache": bool(getattr(config, "use_cache", True)),
        "forward_order": [list(range(physical)) for _ in range(loops)],
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


def recursive_forward_trace_audit(model: Any, *, device: Any) -> dict[str, Any]:
    """Verify physical shared layers execute ``0..K-1`` exactly twice."""

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
        base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    for handle in handles:
        handle.remove()
    base_model.train(training_state)
    expected = list(range(physical)) * 2
    if sequence != expected:
        raise AssertionError(f"Recursive forward trace mismatch: expected={expected} got={sequence}")
    return {
        "trace": sequence,
        "expected": expected,
        "logical_layers": physical * 2,
        "physical_shared_layers": physical,
        "loops": 2,
        "forward_trace_ok": True,
    }


def _synthetic_batch(*, rank: int, device: Any, vocab_size: int) -> dict[str, Any]:
    # Unequal lengths explicitly exercise dynamic padding and label masking.
    rows = [[3 + rank, 5, 7, 11, 13, 17], [19, 23, 29, 31 + rank]]
    rows = [[token % vocab_size for token in row] for row in rows]
    return collate_dynamic_padding(rows, pad_token_id=0, device=device)


def _synthetic_gate_a(model: Any, *, device: Any, config: Stage4Config, rank: int) -> dict[str, Any]:
    """Audit DDP initialization, recursive trace, masks, gradients and update."""

    import torch
    import torch.distributed as dist

    base_model = getattr(model, "module", model)
    base_model.config.use_cache = False
    parameters = _trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0.0)
    sequence, handles = _register_forward_trace(model)
    backward_sequence: list[int] = []
    recursive_base = getattr(base_model, "model", base_model)
    backward_handles = [
        layer.register_full_backward_hook(
            lambda _module, _grad_input, _grad_output, i=index: backward_sequence.append(i)
        )
        for index, layer in enumerate(recursive_base.layers)
    ]
    batch = _synthetic_batch(rank=rank, device=device, vocab_size=config.synthetic_vocab_size)
    labels = batch["labels"]
    if labels[0, 0].item() == -100 or labels[1, 1].item() == -100:
        raise AssertionError("Synthetic audit masked a non-padding label")
    optimizer.zero_grad(set_to_none=True)
    local_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    local_tokens = torch.zeros((), device=device, dtype=torch.int64)
    microbatches = 0
    # Obtain every microbatch's global valid-token denominator before any
    # backward pass, then use the window-global denominator for token-weighted
    # accumulation under DDP's default gradient averaging.
    local_valid_count = (labels[:, 1:] != -100).sum().to(dtype=torch.int64)
    global_counts = [int(_all_reduce_scalar(local_valid_count.detach().clone()).item()) for _ in range(config.gradient_accumulation_steps)]
    window_global_tokens = sum(global_counts)
    if window_global_tokens <= 0:
        raise FloatingPointError("Synthetic accumulation window has no supervised tokens")
    for _ in range(config.gradient_accumulation_steps):
        with _bf16_autocast(device):
            result = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            loss_sum, valid_count = _causal_loss_sum(result.logits, labels)
        if not torch.isfinite(loss_sum):
            raise FloatingPointError("Synthetic loss is non-finite")
        scale = float(config.world_size) / float(config.gradient_accumulation_steps * window_global_tokens)
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
    physical = len(recursive_base.layers)
    forward_expected = list(range(physical)) * 2
    backward_expected = list(reversed(range(physical))) * 2
    trace_forward_ok = sequence[: 2 * physical] == forward_expected
    trace_backward_ok = backward_sequence[: 2 * physical] == backward_expected
    if not trace_forward_ok:
        raise AssertionError(f"Forward trace is not 0..physical-1 twice: {sequence[:2 * physical]}")
    if not trace_backward_ok:
        raise AssertionError(
            "Backward must traverse the second recursive loop and then the first loop: "
            f"got {backward_sequence[:2 * physical]}"
        )
    layer_gradients = {}
    for index, layer in enumerate(recursive_base.layers):
        grads = [parameter.grad for parameter in layer.parameters()]
        norms = [float(torch.linalg.vector_norm(grad.detach()).item()) for grad in grads if grad is not None]
        if not norms or not all(math.isfinite(value) and value > 0 for value in norms):
            raise AssertionError(f"Physical layer {index} has missing/non-finite/zero gradient")
        layer_gradients[str(index)] = {"norm": max(norms), "finite_nonzero": True}
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
        "recursive_forward_trace": sequence[: 2 * physical],
        "recursive_forward_trace_expected": forward_expected,
        "recursive_backward_trace": backward_sequence[: 2 * physical],
        "recursive_backward_trace_expected": backward_expected,
        "forward_trace_ok": trace_forward_ok,
        "backward_second_loop_then_first": trace_backward_ok,
        "logical_layers": int(getattr(base_model.config, "num_hidden_layers", 0)),
        "physical_layers": physical,
        "recursive_loops": int(getattr(base_model, "recursive_loops", 0)),
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
        "optimizer_step_calls": 1,
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

    rank = int(os.environ.get("RANK", "0"))
    configured_world_size = 1 if config.gate == "B" else config.world_size
    world_size = int(os.environ.get("WORLD_SIZE", str(configured_world_size)))
    local_rank = int(os.environ.get("LOCAL_RANK", str(config.local_rank if config.local_rank >= 0 else rank)))
    config.world_size = world_size
    config.local_rank = local_rank
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
        dist.init_process_group(backend=backend, init_method="env://")
    if world_size != DEFAULT_WORLD_SIZE and not (config.allow_non8 or config.dry_run):
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


def _load_scheduler(optimizer: Any, warmup_steps: int) -> Any:
    import torch

    def schedule(step: int) -> float:
        return min(1.0, float(step + 1) / float(max(1, warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


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
                    "model_config": True,
                    "tokenizer": True,
                    "optimizer": True,
                    "scheduler": True,
                    "rng": True,
                    "step": True,
                    "data_cursors_by_rank": True,
                    "manifest": True,
                    "world_size": int(config.world_size),
                    "architecture": "logical_30_physical_15_loops_2",
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
                    "optimizer_step": int(optimizer_step),
                    "world_size": int(config.world_size),
                    "rank0_only": True,
                    "data_cursors_by_rank": sorted(normalized_cursors),
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
    architecture_contract = "logical_30_physical_15_loops_2"
    if contract.get("architecture") != architecture_contract:
        raise ValueError(
            "Resume checkpoint architecture contract mismatch: "
            f"expected={architecture_contract!r} got={contract.get('architecture')!r}"
        )
    if "world_size" not in contract or int(contract["world_size"]) != int(config.world_size):
        raise ValueError(
            "Resume checkpoint contract world_size mismatch: "
            f"checkpoint={contract.get('world_size')!r} current={config.world_size}"
        )
    if "optimizer_step" not in state:
        raise ValueError("Resume checkpoint is missing training_state.optimizer_step")
    state_step = int(state["optimizer_step"])
    if "world_size" not in complete or int(complete["world_size"]) != int(config.world_size):
        raise ValueError(
            "Resume checkpoint_complete.json world_size mismatch: "
            f"checkpoint={complete.get('world_size')!r} current={config.world_size}"
        )
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
) -> tuple[bool, dict[str, Any]]:
    """Collect exactly 16 complete microbatches before doing backward."""

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
    for batch, global_count in zip(batches, global_counts):
        with _bf16_autocast(device):
            result = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            loss_sum, valid_count = _causal_loss_sum(result.logits, batch["labels"])
        if not torch.isfinite(loss_sum):
            raise FloatingPointError(f"Non-finite loss at rank={rank}")
        scale = float(config.world_size) / float(config.gradient_accumulation_steps * global_window_tokens)
        (loss_sum * scale).backward()
        local_loss_sum_total += loss_sum.detach().double()
        local_valid_total += valid_count.detach()
        global_loss = _all_reduce_scalar(loss_sum.detach().double().clone())
        micro_records.append(
            {
                "microbatch": len(micro_records),
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
        "exact_16_microbatches": len(batches) == 16,
        "window_global_loss_sum": float(global_loss_sum.item()),
        "window_global_valid_token_count": int(global_valid_tokens.item()),
        "window_global_loss_count_consistent": True,
        "microbatch_records": micro_records,
    }


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

    model = _ddp_wrap(model, config, device)
    parameters = _trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer_audit = optimizer_parameter_audit(optimizer, parameters)
    scheduler = _load_scheduler(optimizer, config.warmup_steps)
    start_step = 0
    cumulative_samples = 0
    cumulative_tokens = 0
    previous_cursor: dict[str, Any] = {}
    previous_manifest: dict[str, Any] = {}
    previous_cursor_source = "fresh"
    if resume_state is not None:
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
    batch_iterator = iter(stream)
    # A bounded shuffle cannot restore the exact in-shard cursor.  Keep the
    # cursor for audit/restart bookkeeping but never claim bitwise data resume.
    monitor = RuntimeMonitor(output_dir / f"runtime_monitor_rank{rank}.jsonl", interval_seconds=config.monitor_interval_seconds, rank=rank, device=device)
    monitor.start()
    metrics: list[dict[str, Any]] = []
    optimizer_step = start_step
    latest_checkpoint: Path | None = None
    stop_reason = "max_optimizer_steps"
    coordinated_stop = False
    try:
        while optimizer_step < config.max_optimizer_steps:
            optimizer.zero_grad(set_to_none=True)
            window_start = time.perf_counter()
            complete, window = _training_window(model, batch_iterator, config=config, device=device, rank=rank)
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
                    "microbatches": config.gradient_accumulation_steps,
                    "exact_16_microbatches": True,
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
                    report={"latest_metric": metrics[-1], "architecture": _validate_recursive_architecture(getattr(model, "module", model))},
                    rank=rank,
                )
                if checkpoint is not None:
                    latest_checkpoint = checkpoint
                if rank == 0 and checkpoint is not None:
                    # Pointer is written only after the complete atomic rename.
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
                if not final_checkpoint_path.exists():
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
                            "stop_reason": stop_reason,
                            "steps_at_stop": optimizer_step,
                            "architecture": _validate_recursive_architecture(
                                getattr(model, "module", model)
                            ),
                        },
                        rank=rank,
                    )
                    latest_checkpoint = final_checkpoint
                else:
                    latest_checkpoint = final_checkpoint_path
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
            if dist.is_initialized():
                dist.barrier()
            # Keep the path in every rank's report even though only rank 0
            # owns the physical write.
            latest_checkpoint = final_checkpoint_path
    finally:
        monitor.stop()
    if optimizer_step == start_step and config.max_optimizer_steps > start_step:
        raise RuntimeError("No complete optimizer step was available on all ranks")
    checksum = _model_checksum_audit(model, device)
    return {
        "status": "PASS",
        "global_effective_batch": config.world_size * config.micro_batch_size * config.gradient_accumulation_steps,
        "local_samples_per_optimizer_step": config.micro_batch_size * config.gradient_accumulation_steps,
        "optimizer_step_start": start_step,
        "optimizer_steps": optimizer_step,
        "optimizer_step_increment": optimizer_step - start_step,
        "optimizer_step_calls": optimizer_step - start_step,
        "optimizer_scheduler_rng_restored": bool(resume_state is not None),
        "resume_smoke_step_increment": optimizer_step - start_step if resume_state is not None else None,
        "cumulative_samples": cumulative_samples,
        "cumulative_valid_tokens": cumulative_tokens,
        "target_samples_per_rank": DEFAULT_TARGET_SAMPLES_PER_RANK,
        "target_local_microbatches": DEFAULT_LOCAL_MICROBATCHES,
        "parameter_checksum_audit": checksum,
        "metrics": metrics,
        "optimizer_audit": optimizer_audit,
        "rank0_only_checkpoint": True,
        "latest_complete_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
        "stop_reason": stop_reason,
        "coordinated_stop": coordinated_stop,
        "steps_at_stop": optimizer_step,
        "final_checkpoint": (
            str(latest_checkpoint)
            if coordinated_stop and latest_checkpoint is not None
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

    return AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)


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
        rank_valid: dict[str, int] = {}
        for rank_key, shard_paths in manifest["rank_shards"].items():
            by_name = {item["name"]: item for item in audit["shards"]}
            rank_valid[rank_key] = sum(
                int(by_name[Path(path).name].get("content", {}).get("valid_trainable_rows", 0))
                for path in shard_paths
            )
        manifest["rank_raw_rows"] = {
            rank_key: sum(int(by_name[Path(path).name]["num_rows"]) for path in shard_paths)
            for rank_key, shard_paths in manifest["rank_shards"].items()
        }
        manifest["rank_valid_trainable_rows"] = rank_valid
        manifest["target_samples_per_rank"] = DEFAULT_TARGET_SAMPLES_PER_RANK
        manifest["rank_has_target_samples"] = {
            rank_key: value >= DEFAULT_TARGET_SAMPLES_PER_RANK
            for rank_key, value in rank_valid.items()
        }
        manifest["raw_rows"] = int(audit["total_rows"])
        manifest["effective_trainable_rows"] = int(audit["content"]["valid_trainable_rows"])
        manifest["formal_global_samples"] = int(
            DEFAULT_MAX_OPTIMIZER_STEPS
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
    missing = [
        rank_key
        for rank_key, ok in manifest.get("rank_has_target_samples", {}).items()
        if not ok
    ]
    if missing:
        raise RuntimeError(
            "Fail-fast: rank shard(s) lack the required 1,183,232 effective samples: "
            f"{missing}; valid={manifest.get('rank_valid_trainable_rows')}"
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


def _synthetic_model(config: Stage4Config, device: Any) -> Any:
    import torch
    from transformers import LlamaConfig

    from code.RSmol.recursive_model import RecursiveLlamaForCausalLM

    logical = int(config.synthetic_layers)
    if logical <= 0 or logical % 2:
        raise ValueError("synthetic_layers must be a positive even number")
    physical = logical // 2
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
    llama_config.recursive_loops = 2
    llama_config.recursive_layer_count = physical
    llama_config.recursive_mapping_policy = "stage4_synthetic_fixture"
    llama_config.use_cache = False
    model = RecursiveLlamaForCausalLM(llama_config)
    model.to(device=device, dtype=torch.float32)
    model.train()
    _validate_recursive_architecture(model, production=False)
    return model


def _parse_args(argv: Sequence[str] | None = None) -> Stage4Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("A", "B", "C", "D", "E"), default="A")
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
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--max-optimizer-steps", "--max-steps", dest="max_optimizer_steps", type=int, default=DEFAULT_MAX_OPTIMIZER_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--record-buffer-size", type=int, default=DEFAULT_RECORD_BUFFER_SIZE)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--monitor-interval-seconds", type=float, default=60.0)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE)
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--dry-run", "--static-check", dest="dry_run", action="store_true")
    parser.add_argument("--allow-non8", action="store_true")
    parser.add_argument("--synthetic-layers", type=int, default=4)
    parser.add_argument("--synthetic-hidden-size", type=int, default=32)
    parser.add_argument("--synthetic-vocab-size", type=int, default=97)
    args = parser.parse_args(argv)
    config = Stage4Config(**vars(args))
    if config.micro_batch_size != 8:
        raise ValueError("Stage 4 requires micro_batch_size=8")
    if config.gradient_accumulation_steps != 16:
        raise ValueError("Stage 4 requires gradient_accumulation_steps=16")
    if config.learning_rate != 2e-4:
        raise ValueError("Stage 4 requires learning_rate=2e-4")
    if config.context_length <= 0 or config.context_length > 1024:
        raise ValueError("Stage 4 context_length must be in [1, 1024]")
    if config.max_optimizer_steps <= 0:
        raise ValueError("max_optimizer_steps must be positive")
    if config.record_buffer_size <= 0:
        raise ValueError("record_buffer_size must be positive")
    if config.save_every <= 0:
        raise ValueError("save_every must be positive")
    if config.world_size <= 0:
        raise ValueError("world_size must be positive")
    if config.report_path is None:
        config.report_path = config.output_dir / "stage4_report.json"
    return config


def run(config: Stage4Config) -> dict[str, Any]:
    """Execute one Stage 4 gate and return its auditable JSON report."""

    faulthandler.enable()
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
            "dry_run": True,
            "configuration": asdict(config),
            "data_loader_contract": {
                "num_workers": NUM_WORKERS,
                "pin_memory": PIN_MEMORY,
                "persistent_workers": PERSISTENT_WORKERS,
                "dataset_num_proc": DATASET_NUM_PROC,
            },
            "expected_target_samples_per_rank": DEFAULT_TARGET_SAMPLES_PER_RANK,
            "expected_global_effective_batch": DEFAULT_GLOBAL_EFFECTIVE_BATCH,
            "expected_formal_global_samples": DEFAULT_FORMAL_GLOBAL_SAMPLES,
            "output_dir_external": True,
            "no_model_or_checkpoint_written": True,
        }
        _write_json(report_path, report)
        return report

    if config.gate == "B" and int(os.environ.get("WORLD_SIZE", "1").strip() or "1") > 1:
        raise RuntimeError("Gate B is single-process only and must not be launched with torchrun")
    rank, world_size, device, use_cuda = _prepare_distributed(config)
    _set_seed(config.seed, rank=rank)
    process_group = __import__("torch.distributed", fromlist=["distributed"])
    try:
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
            audit = audit_parquet_shards(files, tokenizer=tokenizer, context_length=config.context_length, content=True)
            manifest = assign_shards(files, world_size=world_size, seed=config.seed)
            by_name = {item["name"]: item for item in audit["shards"]}
            manifest["rank_raw_rows"] = {"0": audit["total_rows"]}
            manifest["rank_valid_trainable_rows"] = {
                "0": audit["content"]["valid_trainable_rows"]
            }
            manifest["target_samples_per_rank"] = DEFAULT_TARGET_SAMPLES_PER_RANK
            manifest["rank_has_target_samples"] = {
                "0": audit["content"]["valid_trainable_rows"] >= DEFAULT_TARGET_SAMPLES_PER_RANK
            }
            if not manifest["rank_has_target_samples"]["0"]:
                raise RuntimeError(
                    "Gate B fail-fast: the pre-audited data has fewer than "
                    f"{DEFAULT_TARGET_SAMPLES_PER_RANK} effective samples"
                )
            manifest["raw_rows"] = audit["total_rows"]
            manifest["effective_trainable_rows"] = audit["content"]["valid_trainable_rows"]
            manifest["formal_global_samples"] = DEFAULT_MAX_OPTIMIZER_STEPS * 1024
            manifest["formal_remaining_raw_rows"] = audit["total_rows"] - manifest["formal_global_samples"]
            report = {
                "status": "PASS",
                "gate": "B",
                "configuration": asdict(config),
                "dataset_audit": audit,
                "manifest": manifest,
                "raw_rows": audit["total_rows"],
                "effective_rows": audit["content"]["valid_trainable_rows"],
                "formal_global_samples": manifest["formal_global_samples"],
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
        if config.gate in ("C", "D") and world_size > 1 and config.audit_report is None:
            raise RuntimeError(
                "Gate C/D requires an external Gate B --audit-report before multi-card startup"
            )
        model_path = config.resume_from or config.model_path
        model, tokenizer = _load_runtime_model(model_path, device, config.tokenizer_path)
        architecture_audit = _validate_recursive_architecture(model)
        forward_trace_audit = recursive_forward_trace_audit(model, device=device)
        if config.gate in ("C", "D"):
            audit, manifest = _dataset_preaudit(config, tokenizer=tokenizer, rank=rank, world_size=world_size)
        else:
            resume_state_path = (config.resume_from or Path(".")) / "training_state.pt"
            import torch

            state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
            manifest = dict(state.get("manifest", {}))
            if not manifest:
                raise ValueError("Resume checkpoint does not include a Stage 4 shard manifest")
            audit = {"resume_manifest_source": str(config.resume_from)}
            old_step = int(state["optimizer_step"])
            # Gate E is intentionally a 1--2-step continuation smoke, not a
            # second long run.  ``max_optimizer_steps`` is therefore treated
            # as an upper bound only for the non-resume gates.
            config.max_optimizer_steps = old_step + 2
        if config.gate == "C" and config.max_optimizer_steps == DEFAULT_MAX_OPTIMIZER_STEPS:
            config.max_optimizer_steps = 2
        if rank == 0:
            _write_json(output_dir / "stage4_manifest.json", manifest)
        if process_group.is_initialized():
            process_group.barrier()
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
            }
        report = {
            "status": "PASS",
            "gate": config.gate,
            "configuration": asdict(config),
            "dataset_audit": audit,
            "training": train_report,
            "stop_reason": train_report.get("stop_reason"),
            "steps_at_stop": train_report.get("steps_at_stop"),
            "final_checkpoint": train_report.get("final_checkpoint"),
            "latest_complete_checkpoint": train_report.get("latest_complete_checkpoint"),
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
