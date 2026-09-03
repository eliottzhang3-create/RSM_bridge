#!/usr/bin/env python3
"""Stage 4 training/smoke entry point for 5-10xpoisson-parcae.

The formal contract is 9,244 optimizer steps, 64 local microbatches per
window, two sequences per local microbatch, and ceil(5%)=463 warmup steps.
Every local microbatch samples an independent vector ``(T_1, ..., T_B)`` from
the exact truncated Poisson distribution.  The vector is deliberately not
broadcast between ranks.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

MODEL_ARCHITECTURE_CONTRACT = "logical_50_110_physical_20_5_10xpoisson_parcae_tail4"
MODEL_LABEL = "5_10xpoisson_parcae"
MIN_MIDDLE_LOOPS, MAX_MIDDLE_LOOPS = 4, 10
DEFAULT_INFERENCE_MIDDLE_LOOPS = 7
DEFAULT_INFERENCE_R = DEFAULT_INFERENCE_MIDDLE_LOOPS
PARAMETER_GRADIENT_TAIL_LOOPS = 4
MIN_LOGICAL_LAYER_COUNT, MAX_LOGICAL_LAYER_COUNT = 50, 110
PHYSICAL_LAYER_COUNT, PREFIX_LAYER_COUNT, MIDDLE_LAYER_COUNT, SUFFIX_LAYER_COUNT = 20, 5, 10, 5
DEFAULT_WORLD_SIZE = 8
DEFAULT_MICRO_BATCH_SIZE = 8
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 16
DEFAULT_FORMAL_MICRO_BATCH_SIZE = 2
DEFAULT_FORMAL_GRADIENT_ACCUMULATION_STEPS = 64
DEFAULT_CONTEXT_LENGTH = 1024
DEFAULT_MAX_OPTIMIZER_STEPS = 10
DEFAULT_FORMAL_OPTIMIZER_STEPS = 9244
DEFAULT_FORMAL_WARMUP_STEPS = 463  # ceil(5% of the formal 9244-step schedule)
DEFAULT_GLOBAL_EFFECTIVE_BATCH = DEFAULT_WORLD_SIZE * DEFAULT_MICRO_BATCH_SIZE * DEFAULT_GRADIENT_ACCUMULATION_STEPS
DEFAULT_TARGET_SAMPLES_PER_RANK = DEFAULT_MICRO_BATCH_SIZE * DEFAULT_GRADIENT_ACCUMULATION_STEPS * DEFAULT_MAX_OPTIMIZER_STEPS
DEFAULT_LOCAL_MICROBATCHES = DEFAULT_MAX_OPTIMIZER_STEPS * DEFAULT_GRADIENT_ACCUMULATION_STEPS
DEFAULT_FORMAL_SAMPLES_PER_RANK = DEFAULT_FORMAL_OPTIMIZER_STEPS * DEFAULT_FORMAL_GRADIENT_ACCUMULATION_STEPS * DEFAULT_FORMAL_MICRO_BATCH_SIZE
DEFAULT_FORMAL_LOCAL_MICROBATCHES = DEFAULT_FORMAL_OPTIMIZER_STEPS * DEFAULT_FORMAL_GRADIENT_ACCUMULATION_STEPS
DEFAULT_FORMAL_GLOBAL_SAMPLES = DEFAULT_FORMAL_SAMPLES_PER_RANK * DEFAULT_WORLD_SIZE
DEFAULT_LEARNING_RATE, DEFAULT_MIN_LR = 2e-4, 2e-5
DEFAULT_FORMAL_LEARNING_RATE, DEFAULT_FORMAL_MIN_LR = 8e-4, 8e-5
DEFAULT_SAVE_EVERY, DEFAULT_CHECKPOINT_RETENTION = 500, 3
SAMPLING_POLICY = "truncated_poisson"
SAMPLER_VERSION = "truncated_poisson_lambda7_support4_10_v1"
SAMPLER_KEY = "sha256_cpu_torch_generator_base_seed_rank_optimizer_step_microbatch_v1"
POISSON_LAMBDA = 7.0
POISSON_SUPPORT = tuple(range(4, 11))
POISSON_NORMALIZATION_Z = sum(math.exp(-POISSON_LAMBDA) * POISSON_LAMBDA**k / math.factorial(k) for k in POISSON_SUPPORT)
POISSON_Z = POISSON_NORMALIZATION_Z
POISSON_TRUNCATION_Z = POISSON_NORMALIZATION_Z
POISSON_PROBABILITIES = tuple((math.exp(-POISSON_LAMBDA) * POISSON_LAMBDA**k / math.factorial(k)) / POISSON_NORMALIZATION_Z for k in POISSON_SUPPORT)
EXPECTED_MAPPING_POLICY = "explicit_5_10xpoisson_parcae_source_layers"
EXPECTED_BACKWARD_POLICY = "hidden_path_all_calls_parameter_gradients_final_four_aligned_calls_v1"
EXPECTED_INJECTION_INIT = "parcae_exact_ssm_decay_sqrt_1_over_5_identity_B_no_weight_decay"
INJECTION_NO_WEIGHT_DECAY_SUFFIXES = (
    "recurrent.injection.a_log",
    "recurrent.injection.dt_bias",
    "recurrent.injection.b",
)


def formal_save_steps(total_steps: int = DEFAULT_FORMAL_OPTIMIZER_STEPS, every: int = DEFAULT_SAVE_EVERY) -> list[int]:
    return [*range(int(every), int(total_steps) + 1, int(every)), int(total_steps)] if int(total_steps) % int(every) else list(range(int(every), int(total_steps) + 1, int(every)))


def poisson_metadata() -> dict[str, Any]:
    return {"sampling_policy": SAMPLING_POLICY, "sampler_version": SAMPLER_VERSION, "sampler_key": SAMPLER_KEY, "poisson_lambda": POISSON_LAMBDA, "poisson_support": list(POISSON_SUPPORT), "poisson_normalization_z": POISSON_NORMALIZATION_Z, "lambda": POISSON_LAMBDA, "support": list(POISSON_SUPPORT), "Z": POISSON_NORMALIZATION_Z, "poisson_probabilities": list(POISSON_PROBABILITIES), "recursive_poisson_lambda": POISSON_LAMBDA, "recursive_poisson_support": list(POISSON_SUPPORT), "recursive_poisson_probabilities": list(POISSON_PROBABILITIES), "recursive_poisson_normalization_z": POISSON_NORMALIZATION_Z, "recursive_poisson_Z": POISSON_NORMALIZATION_Z}


def _sampler_seed(base_seed: int, rank: int, optimizer_step: int, microbatch_index: int) -> int:
    payload = f"{int(base_seed)}:{int(rank)}:{int(optimizer_step)}:{int(microbatch_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def sample_middle_loop_counts(base_seed: int, rank: int, optimizer_step: int, microbatch_index: int, batch_size: int):
    """Sample one T_i per local sequence with a private CPU torch.Generator."""

    import torch
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_sampler_seed(base_seed, rank, optimizer_step, microbatch_index))
    probabilities = torch.tensor(POISSON_PROBABILITIES, dtype=torch.float64, device="cpu")
    indices = torch.multinomial(probabilities, batch_size, replacement=True, generator=generator)
    return torch.tensor(POISSON_SUPPORT, dtype=torch.long, device="cpu")[indices]


def sample_middle_loop_count(*args: Any, **kwargs: Any):
    """Compatibility guard: scalar step sampling is forbidden in training."""

    raise ValueError("5-10xpoisson-parcae training requires sample_middle_loop_counts per local microbatch")


def validate_sampling_contract(metadata: dict[str, Any]) -> None:
    expected = poisson_metadata()
    for key in expected:
        if metadata.get(key) != expected[key]:
            raise ValueError(f"sampler metadata mismatch for {key}: got={metadata.get(key)!r} expected={expected[key]!r}")


_validate_exact_sampling_contract = validate_sampling_contract


def scheduler_metadata(*, total_steps: int, warmup_steps: int, max_lr: float = DEFAULT_LEARNING_RATE, min_lr: float = DEFAULT_MIN_LR) -> dict[str, Any]:
    return {"scheduler_type": "linear_warmup_cosine", "total_steps_for_schedule": int(total_steps), "warmup_steps": int(warmup_steps), "max_lr": float(max_lr), "min_lr": float(min_lr)}


def build_schedule(T: int) -> tuple[int, ...]:
    T = int(T)
    if not MIN_MIDDLE_LOOPS <= T <= MAX_MIDDLE_LOOPS:
        raise ValueError("T must be in support 4..10")
    return tuple(range(5)) + tuple(range(5, 15)) * T + tuple(range(15, 20))


def validate_schedule_contract() -> dict[str, Any]:
    values = {str(T): len(build_schedule(T)) for T in range(4, 11)}
    if min(values.values()) != MIN_LOGICAL_LAYER_COUNT or max(values.values()) != MAX_LOGICAL_LAYER_COUNT:
        raise AssertionError(f"schedule depth range is not 50..110: {values}")
    return {"schedule_logical_depths": values, "min_logical_layer_count": MIN_LOGICAL_LAYER_COUNT, "max_logical_layer_count": MAX_LOGICAL_LAYER_COUNT}


def expected_parquet_names(shard_count: int = 85) -> list[str]:
    return [f"train-{index:05d}-of-{int(shard_count):05d}.parquet" for index in range(int(shard_count))]


def discover_parquet_files(data_dir: Path) -> list[Path]:
    # The immutable 85-shard snapshot is distributed as
    # ``<data_dir>/data/train-xxxxx-of-00085.parquet``.  Accept the dataset
    # root used in README/configuration as well as a directly supplied
    # ``data`` directory, matching the established Stage 2/legacy loaders.
    root = Path(data_dir).expanduser().resolve()
    parquet_root = root / "data" if (root / "data").is_dir() else root
    files = sorted(parquet_root.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet shards found under {data_dir} (searched {parquet_root})")
    return files


def select_sample_shards(files: Sequence[Path], *, sample_shards: int = 3, seed: int = 0) -> list[Path]:
    keyed = sorted((hashlib.sha256(f"{int(seed)}:{path.name}".encode()).hexdigest(), path) for path in files)
    return [path for _, path in keyed[: int(sample_shards)]]


def assign_shards(files: Sequence[Path], *, world_size: int = DEFAULT_WORLD_SIZE, seed: int = 0) -> dict[str, Any]:
    ordered = [path for _, path in sorted((hashlib.sha256(f"{int(seed)}:{path.name}".encode()).hexdigest(), path) for path in files)]
    by_rank = {str(rank): [str(path) for path in ordered[rank:: int(world_size)]] for rank in range(int(world_size))}
    return {"world_size": int(world_size), "seed": int(seed), "rank_shards": by_rank, "rank_shard_counts": {rank: len(paths) for rank, paths in by_rank.items()}}


def audit_parquet_shards(files: Sequence[Path], *, tokenizer: Any | None = None, context_length: int = DEFAULT_CONTEXT_LENGTH, content_paths: Sequence[Path] | None = None) -> dict[str, Any]:
    """CPU-only footer/content audit using the sole real-data tokenizer policy."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("audit_parquet_shards requires pyarrow") from exc
    content_paths = {Path(path).resolve() for path in (content_paths or ())}
    shards, total_rows = [], 0
    content_audit = {"audited": bool(content_paths), "tokenized_rows": 0, "empty_after_tokenization": 0}
    for path in files:
        parquet = pq.ParquetFile(path)
        rows = int(parquet.metadata.num_rows)
        total_rows += rows
        item = {"name": Path(path).name, "rows": rows, "content_audited": Path(path).resolve() in content_paths}
        if Path(path).resolve() in content_paths and tokenizer is not None:
            for record_batch in parquet.iter_batches(columns=["text"], batch_size=128):
                for text_value in record_batch.column("text").to_pylist():
                    if not text_value:
                        continue
                    ids = _tokenize_ids(tokenizer, str(text_value), int(context_length))
                    content_audit["tokenized_rows"] += 1
                    if not ids:
                        content_audit["empty_after_tokenization"] += 1
        shards.append(item)
    return {"file_count": len(shards), "total_rows": total_rows, "shards": shards, "content_audit": content_audit, "content_paths": [str(path) for path in content_paths], "tokenizer_contract": "add_special_tokens=False,truncation=True,max_length=1024", "raw_rows_are_not_capacity": True}


def cosine_warmup_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return 0.1 + 0.9 * float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def token_weighted_gradient_scale(*, world_size: int, global_window_tokens: int) -> float:
    if global_window_tokens <= 0:
        raise ValueError("global_window_tokens must be positive")
    return float(world_size) / float(global_window_tokens)


@dataclass
class Stage4Config:
    gate: str = "D"
    model_path: Path | None = None
    tokenizer_path: Path | None = None
    data_dir: Path = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset")
    output_dir: Path = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10xpoisson_parcae")
    max_optimizer_steps: int = DEFAULT_MAX_OPTIMIZER_STEPS
    formal_optimizer_steps: int = DEFAULT_FORMAL_OPTIMIZER_STEPS
    scheduler_total_steps: int = DEFAULT_MAX_OPTIMIZER_STEPS
    warmup_steps: int = 1
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE
    context_length: int = DEFAULT_CONTEXT_LENGTH
    learning_rate: float = DEFAULT_LEARNING_RATE
    min_lr: float = DEFAULT_MIN_LR
    save_every: int = DEFAULT_SAVE_EVERY
    checkpoint_retention: int = DEFAULT_CHECKPOINT_RETENTION
    world_size: int = DEFAULT_WORLD_SIZE
    report_path: Path | None = None
    backend: str = "nccl"
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 0
    device: str = "cuda"
    dry_run: bool = False
    max_microbatches: int | None = None
    resume_from: Path | None = None
    audit_report: Path | None = None


def _parse_args(argv: Sequence[str] | None = None) -> Stage4Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", default="D", choices=("A", "B", "C", "D", "E", "FORMAL", "formal"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Stage4Config.data_dir)
    parser.add_argument("--output-dir", type=Path, default=Stage4Config.output_dir)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--scheduler-total-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--micro-batch-size", type=int, default=DEFAULT_MICRO_BATCH_SIZE)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-lr", type=float)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--checkpoint-retention", type=int, default=DEFAULT_CHECKPOINT_RETENTION)
    parser.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-microbatches", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    gate = str(args.gate).upper()
    requested_max_lr = args.max_lr if args.max_lr is not None else args.learning_rate
    if gate == "FORMAL":
        learning_rate = (
            DEFAULT_FORMAL_LEARNING_RATE
            if requested_max_lr is None
            else float(requested_max_lr)
        )
        min_lr = DEFAULT_FORMAL_MIN_LR if args.min_lr is None else float(args.min_lr)
        if learning_rate != DEFAULT_FORMAL_LEARNING_RATE:
            raise ValueError("FORMAL requires maximum learning rate=8e-4")
        if min_lr != DEFAULT_FORMAL_MIN_LR:
            raise ValueError("FORMAL requires minimum learning rate=8e-5")
    else:
        learning_rate = DEFAULT_LEARNING_RATE if requested_max_lr is None else float(requested_max_lr)
        min_lr = DEFAULT_MIN_LR if args.min_lr is None else float(args.min_lr)
    if learning_rate <= 0.0 or min_lr <= 0.0 or min_lr > learning_rate:
        raise ValueError("learning rates must satisfy 0 < min_lr <= max_lr")
    if not math.isclose(min_lr, learning_rate * 0.1, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("minimum learning rate must equal 0.1 * maximum learning rate")
    if args.max_steps is not None and args.max_optimizer_steps is not None and args.max_steps != args.max_optimizer_steps:
        raise ValueError("--max-steps and --max-optimizer-steps disagree")
    requested_steps = args.max_optimizer_steps if args.max_optimizer_steps is not None else args.max_steps
    if gate == "FORMAL":
        steps = DEFAULT_FORMAL_OPTIMIZER_STEPS if requested_steps is None else requested_steps
        if steps != DEFAULT_FORMAL_OPTIMIZER_STEPS:
            raise ValueError("FORMAL requires max_optimizer_steps=9244")
        scheduler_steps = DEFAULT_FORMAL_OPTIMIZER_STEPS if args.scheduler_total_steps is None else args.scheduler_total_steps
        if scheduler_steps != DEFAULT_FORMAL_OPTIMIZER_STEPS:
            raise ValueError("FORMAL requires scheduler_total_steps=9244")
        warmup = DEFAULT_FORMAL_WARMUP_STEPS if args.warmup_steps is None else args.warmup_steps
        if warmup != DEFAULT_FORMAL_WARMUP_STEPS:
            raise ValueError("FORMAL requires warmup_steps=463")
        if args.save_every != DEFAULT_SAVE_EVERY:
            raise ValueError("FORMAL requires save_every=500")
        if args.checkpoint_retention != DEFAULT_CHECKPOINT_RETENTION:
            raise ValueError("FORMAL requires checkpoint_retention=3")
        if args.gradient_accumulation_steps != DEFAULT_FORMAL_GRADIENT_ACCUMULATION_STEPS:
            raise ValueError("FORMAL requires gradient_accumulation_steps=64 for the memory-safe 2x64 topology")
        if args.micro_batch_size != DEFAULT_FORMAL_MICRO_BATCH_SIZE:
            raise ValueError("FORMAL requires micro_batch_size=2 for the memory-safe 2x64 topology")
        if args.context_length != DEFAULT_CONTEXT_LENGTH:
            raise ValueError("FORMAL requires context_length=1024")
        if args.world_size != DEFAULT_WORLD_SIZE:
            raise ValueError("FORMAL requires world_size=8")
        if args.max_microbatches is not None:
            raise ValueError("FORMAL requires max_microbatches=None")
    elif gate == "C":
        steps = 2 if requested_steps is None else requested_steps
        scheduler_steps = steps if args.scheduler_total_steps is None else args.scheduler_total_steps
        warmup = max(1, math.ceil(scheduler_steps * 0.05)) if args.warmup_steps is None else args.warmup_steps
    else:
        steps = DEFAULT_MAX_OPTIMIZER_STEPS if requested_steps is None else requested_steps
        if gate == "D" and requested_steps is not None and steps != DEFAULT_MAX_OPTIMIZER_STEPS:
            raise ValueError("Gate D is the fixed ten-optimizer-step real-data smoke")
        scheduler_steps = steps if args.scheduler_total_steps is None else args.scheduler_total_steps
        warmup = max(1, math.ceil(scheduler_steps * 0.05)) if args.warmup_steps is None else args.warmup_steps
    if gate == "E" and args.resume_from is None:
        raise ValueError("Gate E requires --resume-from")
    if gate == "FORMAL" and args.resume_from is not None:
        # Formal resume is supported, but must continue the exact 9244-step domain.
        pass
    return Stage4Config(gate=gate, model_path=args.model_path, tokenizer_path=args.tokenizer_path, data_dir=args.data_dir, output_dir=args.output_dir, max_optimizer_steps=int(steps), scheduler_total_steps=int(scheduler_steps), warmup_steps=int(warmup), gradient_accumulation_steps=args.gradient_accumulation_steps, micro_batch_size=args.micro_batch_size, context_length=args.context_length, learning_rate=learning_rate, min_lr=min_lr, save_every=args.save_every, checkpoint_retention=args.checkpoint_retention, world_size=args.world_size, report_path=args.report_path, backend=args.backend, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, resume_from=args.resume_from, audit_report=args.audit_report, seed=args.seed, device=args.device, dry_run=args.dry_run, max_microbatches=args.max_microbatches)


def _validate_formal_runtime_configuration(config: Stage4Config, *, world_size: int, dry_run: bool) -> None:
    """Fail closed on the complete formal topology/configuration contract."""

    if config.gate not in {"FORMAL", "D"}:
        return
    if config.world_size != DEFAULT_WORLD_SIZE:
        raise ValueError(f"{config.gate} requires config.world_size=8, got {config.world_size}")
    if not dry_run and world_size != DEFAULT_WORLD_SIZE:
        raise RuntimeError(f"{config.gate} requires WORLD_SIZE=8, got {world_size}")
    if config.gate == "FORMAL":
        expected = {
            "micro_batch_size": DEFAULT_FORMAL_MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": DEFAULT_FORMAL_GRADIENT_ACCUMULATION_STEPS,
            "context_length": DEFAULT_CONTEXT_LENGTH,
            "max_optimizer_steps": DEFAULT_FORMAL_OPTIMIZER_STEPS,
            "scheduler_total_steps": DEFAULT_FORMAL_OPTIMIZER_STEPS,
            "warmup_steps": DEFAULT_FORMAL_WARMUP_STEPS,
            "learning_rate": DEFAULT_FORMAL_LEARNING_RATE,
            "min_lr": DEFAULT_FORMAL_MIN_LR,
            "save_every": DEFAULT_SAVE_EVERY,
            "checkpoint_retention": DEFAULT_CHECKPOINT_RETENTION,
            "max_microbatches": None,
        }
        actual = {name: getattr(config, name) for name in expected}
        if actual != expected:
            raise ValueError(f"FORMAL requires exact configuration {expected}, got {actual}")


def _synthetic_batch(*, batch_size: int, context_length: int, vocab_size: int, device: Any):
    import torch
    ids = torch.arange(batch_size * context_length, device=device, dtype=torch.long).reshape(batch_size, context_length) % vocab_size
    return ids, torch.ones_like(ids)


def _tokenize_ids(tokenizer: Any, text: str, context_length: int) -> list[int]:
    """Tokenize one non-empty row using the Stage 4 truncation contract.

    FORMAL fixes ``context_length=1024``; the explicit argument keeps Gate D
    fixtures honest while retaining the same ``max_length=1024`` contract.
    """

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=int(context_length),
    )
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    result = [int(token) for token in ids]
    if len(result) > int(context_length):
        raise AssertionError("tokenizer returned a sequence above context_length")
    return result


def collate_dynamic_padding(
    token_sequences: Sequence[Sequence[int]], *, pad_token_id: int, context_length: int = DEFAULT_CONTEXT_LENGTH
) -> dict[str, Any]:
    """Pad exactly one local microbatch to its own maximum and emit valid masks."""

    if not token_sequences or any(len(sequence) == 0 for sequence in token_sequences):
        raise ValueError("a Stage 4 microbatch must contain non-empty tokenized rows")
    max_length = max(len(sequence) for sequence in token_sequences)
    if max_length > int(context_length):
        raise ValueError(f"microbatch exceeds context_length={context_length}: {max_length}")
    rows = []
    masks = []
    for sequence in token_sequences:
        sequence = [int(token) for token in sequence]
        padding = max_length - len(sequence)
        rows.append(sequence + [int(pad_token_id)] * padding)
        masks.append([1] * len(sequence) + [0] * padding)
    import torch
    input_ids = torch.tensor(rows, dtype=torch.long)
    valid_mask = torch.tensor(masks, dtype=torch.long)
    labels = input_ids.clone()
    labels.masked_fill_(valid_mask == 0, -100)
    return {
        "input_ids": input_ids,
        "attention_mask": valid_mask,
        "valid_mask": valid_mask,
        "labels": labels,
    }


class DistributedParquetStream:
    """Deterministic rank-local stream with exact row/shard cursor resume.

    Rows are never packed into 1024-token blocks and short rows are never
    discarded.  Each non-empty tokenized row is one independent example;
    dynamic padding occurs only inside a local microbatch.  Once a rank has
    consumed its assigned shards, the same fixed order rolls into the next
    deterministic epoch.  A partial batch is carried across the epoch
    boundary, so no valid row is dropped merely because a shard ended.
    """

    cursor_policy = "epoch_shard_row_pending_exact_deterministic_rollover"

    def __init__(
        self,
        data_dir: Path,
        tokenizer: Any,
        *,
        rank: int,
        world_size: int,
        seed: int,
        batch_size: int,
        context_length: int,
        pad_token_id: int,
        gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    ) -> None:
        self.tokenizer = tokenizer
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.context_length = int(context_length)
        self.pad_token_id = int(pad_token_id)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        files = discover_parquet_files(self.data_dir)
        self.rank_shards = tuple(Path(path) for path in assign_shards(files, world_size=self.world_size, seed=self.seed)["rank_shards"][str(self.rank)])
        if not self.rank_shards:
            raise FileNotFoundError(f"rank {self.rank} has no assigned parquet shards")
        self.epoch = 0
        self.shard_index = 0
        self.row_index = 0
        self.pending_token_ids: list[list[int]] = []
        self.samples_seen = 0
        self.microbatches_seen = 0
        self.nonempty_rows_seen = 0
        self._restored = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("real-data Stage 4 requires pyarrow on the remote runtime") from exc
        while True:
            start_shard = int(self.shard_index)
            for shard_index in range(start_shard, len(self.rank_shards)):
                path = self.rank_shards[shard_index]
                parquet = pq.ParquetFile(path)
                skip_before = int(self.row_index) if shard_index == start_shard else 0
                for record_batch in parquet.iter_batches(columns=["text"], batch_size=128, use_threads=False):
                    for local_row_index, text_value in enumerate(record_batch.column("text").to_pylist()):
                        # ``row_index`` is the next raw row to process, making
                        # the cursor exact even when the row is empty.
                        if local_row_index < 0:  # pragma: no cover - documents integer cursor semantics
                            continue
                        absolute_row = getattr(self, "_batch_row_base", 0) + local_row_index
                        if absolute_row < skip_before:
                            continue
                        self.shard_index = shard_index
                        self.row_index = absolute_row + 1
                        if text_value is None or not str(text_value).strip():
                            continue
                        token_ids = _tokenize_ids(self.tokenizer, str(text_value), self.context_length)
                        if not token_ids:
                            continue
                        self.nonempty_rows_seen += 1
                        self.pending_token_ids.append(token_ids)
                        if len(self.pending_token_ids) < self.batch_size:
                            continue
                        batch_rows = self.pending_token_ids[: self.batch_size]
                        del self.pending_token_ids[: self.batch_size]
                        self.samples_seen += self.batch_size
                        self.microbatches_seen += 1
                        yield collate_dynamic_padding(batch_rows, pad_token_id=self.pad_token_id, context_length=self.context_length)
                    self._batch_row_base = getattr(self, "_batch_row_base", 0) + len(record_batch.column("text"))
                self._batch_row_base = 0
                self.shard_index = shard_index + 1
                self.row_index = 0
                start_shard = shard_index + 1
            # Fixed rank-local shard order is an explicit infinite epoch
            # rollover policy; pending rows are carried into the next epoch.
            self.epoch += 1
            self.shard_index = 0
            self.row_index = 0
            start_shard = 0

    def restore_cursor(self, cursor: Mapping[str, Any]) -> None:
        expected_identity = {
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "cursor_policy": self.cursor_policy,
            "data_dir": str(self.data_dir),
            "batch_size": self.batch_size,
            "context_length": self.context_length,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        mismatches = {
            key: {"saved": cursor.get(key), "current": expected}
            for key, expected in expected_identity.items()
            if cursor.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"resume cursor identity mismatch: {mismatches}")
        self.epoch = int(cursor.get("epoch", 0))
        self.shard_index = int(cursor.get("shard_index", 0))
        self.row_index = int(cursor.get("row_index", 0))
        self.samples_seen = int(cursor.get("samples_seen", 0))
        self.microbatches_seen = int(cursor.get("microbatches_seen", 0))
        pending = cursor.get("pending_token_ids", [])
        if not isinstance(pending, list):
            raise ValueError("pending_token_ids cursor must be a list")
        self.pending_token_ids = [[int(token) for token in row] for row in pending]
        if len(self.pending_token_ids) >= self.batch_size:
            raise ValueError("pending_token_ids cursor must contain fewer than one microbatch")
        if not (0 <= self.shard_index <= len(self.rank_shards)) or self.row_index < 0:
            raise ValueError("invalid epoch/shard/row cursor")
        self._batch_row_base = 0
        self._restored = True

    def cursor(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "shard_index": int(self.shard_index),
            "row_index": int(self.row_index),
            "pending_token_ids": [list(row) for row in self.pending_token_ids],
            "samples_seen": int(self.samples_seen),
            "microbatches_seen": int(self.microbatches_seen),
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "cursor_policy": self.cursor_policy,
            "data_dir": str(self.data_dir),
            "batch_size": self.batch_size,
            "context_length": self.context_length,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }


def _iter_real_batches(data_dir: Path, tokenizer: Any, *, batch_size: int, context_length: int, rank: int = 0, world_size: int = 1, seed: int = 0) -> Iterable[dict[str, Any]]:
    """Compatibility wrapper around the exact-cursor rank-local stream."""

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("tokenizer requires pad_token_id or eos_token_id for dynamic padding")
    yield from DistributedParquetStream(
        data_dir,
        tokenizer,
        rank=rank,
        world_size=world_size,
        seed=seed,
        batch_size=batch_size,
        context_length=context_length,
        pad_token_id=int(pad_token_id),
    )


def _load_runtime_model(config: Stage4Config, device: Any) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # A resume must load checkpoint weights/configuration.  The optional base
    # model path is only for fresh runs; preferring it would restore optimizer
    # state against the wrong model parameters.
    model_path = config.resume_from or config.model_path
    if model_path is None:
        raise ValueError("Stage 4 C/D/FORMAL requires --model-path or --resume-from")
    from recursive_model_5_10xpoisson_parcae import register_auto_class
    register_auto_class()
    # Complete checkpoints carry their own tokenizer.  An explicit tokenizer
    # path remains an override for migrations, then the original/base path is
    # used only for legacy/incomplete artifacts (which fail checkpoint audit).
    checkpoint_tokenizer = Path(model_path) / "tokenizer" if config.resume_from is not None else None
    tokenizer_path = config.tokenizer_path or (checkpoint_tokenizer if checkpoint_tokenizer and checkpoint_tokenizer.is_dir() else None) or config.model_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype="auto")
    validate_runtime_model_contract(model)
    model.to(device)
    # Parameter objects can be replaced by HF loading/device movement, so
    # restore convenience flags while retaining the optimizer's name policy.
    recursive_model = getattr(model, "model", model)
    recursive_model.recurrent.injection.mark_no_weight_decay()
    return model, tokenizer


def validate_runtime_model_contract(model: Any) -> dict[str, Any]:
    """Fail closed if *any* required Poisson-Parcae metadata is absent/wrong."""

    model_config = getattr(model, "config", None)
    required = {
        "recursive_mapping_policy": EXPECTED_MAPPING_POLICY,
        "recursive_backward_policy": EXPECTED_BACKWARD_POLICY,
        "recursive_sampling_policy": SAMPLING_POLICY,
        "recursive_sampler_version": SAMPLER_VERSION,
        "recursive_sampler_key": SAMPLER_KEY,
        "recursive_prelude_norm": "LlamaRMSNorm",
        "recursive_state_init": "like-init",
        "recursive_learned_h0": False,
        "recursive_training_loop_mode": "per_local_microbatch_per_sequence_truncated_poisson",
        "recursive_local_tmax": True,
        "recursive_noop_left_alignment": True,
        "recursive_injection_init": EXPECTED_INJECTION_INIT,
        "recursive_injection_formula": "h*decay + dt*(PN(e) @ B.T)",
        "recursive_source_num_hidden_layers": 30,
        "recursive_source_layer_count": 30,
        "recursive_layer_count": PHYSICAL_LAYER_COUNT,
        "num_hidden_layers": MAX_LOGICAL_LAYER_COUNT,
        "recursive_min_middle_loops": MIN_MIDDLE_LOOPS,
        "recursive_max_middle_loops": MAX_MIDDLE_LOOPS,
        "recursive_default_inference_middle_loops": DEFAULT_INFERENCE_MIDDLE_LOOPS,
        "recursive_parameter_gradient_tail_loops": 4,
        "recursive_poisson_lambda": POISSON_LAMBDA,
        "recursive_injection_no_weight_decay": True,
        "recursive_B_init": "identity",
    }
    for key, expected in required.items():
        actual = getattr(model_config, key, None)
        if actual is None or (isinstance(expected, bool) and (type(actual) is not bool or actual is not expected)) or (isinstance(expected, float) and (type(actual) not in (int, float) or float(actual) != expected)) or (not isinstance(expected, (float, bool)) and actual != expected):
            raise ValueError(f"strict runtime contract mismatch/missing {key}: got={actual!r} expected={expected!r}")
    state_init_std = getattr(model_config, "recursive_state_init_std", None)
    embedding_scale = getattr(model_config, "recursive_embedding_scale", None)
    if state_init_std is None or float(state_init_std) <= 0:
        raise ValueError("strict runtime contract requires recursive_state_init_std")
    if embedding_scale is None or float(embedding_scale) <= 0:
        raise ValueError("strict runtime contract requires recursive_embedding_scale")
    if tuple(getattr(model_config, "recursive_poisson_support", ())) != POISSON_SUPPORT:
        raise ValueError("strict runtime contract mismatch for recursive_poisson_support")
    probabilities = tuple(float(value) for value in getattr(model_config, "recursive_poisson_probabilities", ()))
    if len(probabilities) != len(POISSON_PROBABILITIES) or any(abs(a - b) > 1e-14 for a, b in zip(probabilities, POISSON_PROBABILITIES)):
        raise ValueError("strict runtime contract mismatch for recursive_poisson_probabilities")
    expected_mapping = (0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29)
    if tuple(getattr(model_config, "recursive_source_layer_indices_0based", ())) != expected_mapping:
        raise ValueError("strict runtime contract mismatch for source layer mapping")
    if abs(float(getattr(model_config, "recursive_poisson_normalization_z", -1.0)) - POISSON_NORMALIZATION_Z) > 1e-14:
        raise ValueError("strict runtime contract mismatch for recursive_poisson_normalization_z")
    if abs(float(getattr(model_config, "recursive_poisson_Z", -1.0)) - POISSON_NORMALIZATION_Z) > 1e-14:
        raise ValueError("strict runtime contract mismatch for recursive_poisson_Z")
    for key, expected in {
        "recursive_prefix_layer_count": PREFIX_LAYER_COUNT,
        "recursive_middle_layer_count": MIDDLE_LAYER_COUNT,
        "recursive_suffix_layer_count": SUFFIX_LAYER_COUNT,
        "recursive_min_logical_layer_count": MIN_LOGICAL_LAYER_COUNT,
        "recursive_max_logical_layer_count": MAX_LOGICAL_LAYER_COUNT,
    }.items():
        if getattr(model_config, key, None) != expected:
            raise ValueError(f"strict runtime contract mismatch/missing {key}")
    if abs(float(getattr(model_config, "recursive_ssm_decay", -1.0)) - math.sqrt(1.0 / 5.0)) > 1e-14:
        raise ValueError("strict runtime contract mismatch for recursive_ssm_decay")
    target_product = -math.log(math.sqrt(1.0 / 5.0))
    if abs(float(getattr(model_config, "recursive_target_product", -1.0)) - target_product) > 1e-14:
        raise ValueError("strict runtime contract mismatch for recursive_target_product")
    if abs(float(getattr(model_config, "recursive_initial_dt", -1.0)) - target_product) > 1e-14:
        raise ValueError("strict runtime contract mismatch for recursive_initial_dt")
    if abs(float(getattr(model_config, "recursive_initial_decay", -1.0)) - math.sqrt(1.0 / 5.0)) > 1e-14:
        raise ValueError("strict runtime contract mismatch for recursive_initial_decay")
    architectures = getattr(model_config, "architectures", None)
    if architectures != ["RecursiveLlama5_10xpoisson_parcaeForCausalLM"]:
        raise ValueError(f"strict runtime contract mismatch for architectures: {architectures!r}")
    metadata = {
        "architecture_contract": MODEL_ARCHITECTURE_CONTRACT,
        "logical_depth_range": [MIN_LOGICAL_LAYER_COUNT, MAX_LOGICAL_LAYER_COUNT],
        "physical_layer_count": PHYSICAL_LAYER_COUNT,
        "sampling_contract": poisson_metadata(),
        "prelude_norm": "LlamaRMSNorm",
        "state_init": "like-init",
        "state_init_std": float(getattr(model_config, "recursive_state_init_std", getattr(model_config, "initializer_range", 0.0))),
        "injection_init": EXPECTED_INJECTION_INIT,
    }
    if metadata["state_init_std"] <= 0:
        raise ValueError("strict runtime contract requires positive recursive_state_init_std")
    return metadata


def _ddp_wrap(model: Any, device: Any) -> Any:
    import torch
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        return model
    from torch.nn.parallel import DistributedDataParallel
    return DistributedDataParallel(model, device_ids=[device.index] if getattr(device, "type", None) == "cuda" else None, find_unused_parameters=False)


def build_optimizer_param_groups(model: Any, *, weight_decay: float = 0.1) -> list[dict[str, Any]]:
    """Use no-WD groups for injection parameters and standard norm/bias."""

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        module_name = name.lower()
        # The name-based clause is the persistent contract.  Custom Parameter
        # attributes are transient and can disappear after from_pretrained.
        injection_no_decay = any(module_name.endswith(suffix) for suffix in INJECTION_NO_WEIGHT_DECAY_SUFFIXES)
        flagged = bool(getattr(parameter, "_no_weight_decay", False)) or injection_no_decay
        is_norm_or_bias = module_name.endswith(".bias") or "norm" in module_name
        (no_decay if flagged or is_norm_or_bias else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _optimizer_group_audit(model: Any, optimizer: Any) -> dict[str, Any]:
    """Expose the effective AdamW groups, including injection no-WD flags."""

    actual = model.module if hasattr(model, "module") else model
    names_by_id = {id(parameter): name for name, parameter in actual.named_parameters()}
    groups: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        names = [names_by_id.get(id(parameter), f"<unnamed:{id(parameter)}>") for parameter in group["params"]]
        groups.append({"index": index, "weight_decay": float(group.get("weight_decay", 0.0)), "count": len(names), "names": names})
    no_wd_names = [
        name
        for name, parameter in actual.named_parameters()
        if bool(getattr(parameter, "_no_weight_decay", False))
        or any(name.lower().endswith(suffix) for suffix in INJECTION_NO_WEIGHT_DECAY_SUFFIXES)
    ]
    expected_injection_no_wd = {
        name
        for name, _parameter in actual.named_parameters()
        if any(name.lower().endswith(suffix) for suffix in INJECTION_NO_WEIGHT_DECAY_SUFFIXES)
    }
    effective_no_wd = {name for group in groups if group["weight_decay"] == 0.0 for name in group["names"]}
    injection_no_wd = sorted(expected_injection_no_wd & effective_no_wd)
    return {
        "groups": groups,
        "flagged_no_weight_decay": sorted(no_wd_names),
        "expected_injection_no_weight_decay": sorted(expected_injection_no_wd),
        "flagged_parameters_effectively_no_weight_decay": injection_no_wd,
        "all_flagged_parameters_no_weight_decay": bool(expected_injection_no_wd)
        and expected_injection_no_wd.issubset(effective_no_wd)
        and set(no_wd_names).issubset(effective_no_wd),
    }


def _gradient_report(model: Any) -> dict[str, Any]:
    """Per physical-layer finite/nonzero gradient report."""

    import torch
    actual = model.module if hasattr(model, "module") else model
    recursive = getattr(actual, "model", actual)

    def report_group(prefix: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, parameter in recursive.named_parameters():
            if not name.startswith(prefix):
                continue
            gradient = parameter.grad
            valid = gradient is not None and bool(torch.isfinite(gradient).all()) and bool(torch.any(gradient != 0))
            result[name] = {"shape": list(parameter.shape), "grad_norm": None if gradient is None else float(gradient.norm().item()), "finite": bool(gradient is not None and torch.isfinite(gradient).all()), "nonzero": bool(gradient is not None and torch.any(gradient != 0)), "valid": valid}
        return result

    groups = {
        "prefix_layers": report_group("prefix_layers"),
        "middle_layers": report_group("recurrent.middle.layers"),
        "suffix_layers": report_group("suffix_layers"),
        "injection": report_group("recurrent.injection"),
    }
    per_physical_layer: dict[str, Any] = {}
    for prefix, count, offset in (("prefix_layers", PREFIX_LAYER_COUNT, 0), ("recurrent.middle.layers", MIDDLE_LAYER_COUNT, PREFIX_LAYER_COUNT), ("suffix_layers", SUFFIX_LAYER_COUNT, PREFIX_LAYER_COUNT + MIDDLE_LAYER_COUNT)):
        for layer_index in range(count):
            items = {name: item for name, item in groups["prefix_layers" if prefix == "prefix_layers" else "middle_layers" if prefix == "recurrent.middle.layers" else "suffix_layers"].items() if name.startswith(f"{prefix}.{layer_index}.")}
            if not items:
                per_physical_layer[str(offset + layer_index)] = {"finite": False, "nonzero": False, "valid": False, "parameters": {}}
            else:
                per_physical_layer[str(offset + layer_index)] = {"finite": all(item["finite"] for item in items.values()), "nonzero": all(item["nonzero"] for item in items.values()), "valid": all(item["valid"] for item in items.values()), "parameters": items}
    all_parameters: dict[str, Any] = {}
    for name, parameter in actual.named_parameters():
        gradient = parameter.grad
        all_parameters[name] = {
            "shape": list(parameter.shape),
            "parameter_finite": bool(torch.isfinite(parameter).all()),
            "grad_present": gradient is not None,
            "grad_finite": bool(gradient is not None and torch.isfinite(gradient).all()),
            "grad_nonzero": bool(gradient is not None and torch.any(gradient != 0)),
        }
    return {
        "groups": groups,
        "per_physical_layer": per_physical_layer,
        "all_physical_layers_finite_nonzero": all(item["valid"] for item in per_physical_layer.values()),
        "all_parameters": all_parameters,
        "all_parameters_finite": all(item["parameter_finite"] for item in all_parameters.values()),
        "all_gradients_finite_nonzero": all(item["grad_present"] and item["grad_finite"] and item["grad_nonzero"] for item in all_parameters.values()),
        "all_finite_nonzero": all(item["valid"] for group in groups.values() for item in group.values()) and all(item["parameter_finite"] for item in all_parameters.values()),
        "parameter_shapes": {name: list(parameter.shape) for name, parameter in actual.named_parameters()},
    }


def _compact_gradient_report(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Bound formal reports while preserving layer/injection audit outcomes."""

    if report is None:
        return None
    return {
        "all_physical_layers_finite_nonzero": bool(report.get("all_physical_layers_finite_nonzero", False)),
        "all_parameters_finite": bool(report.get("all_parameters_finite", False)),
        "all_gradients_finite_nonzero": bool(report.get("all_gradients_finite_nonzero", False)),
        "all_finite_nonzero": bool(report.get("all_finite_nonzero", False)),
        "per_physical_layer": {
            str(index): {
                "finite": bool(item.get("finite", False)),
                "nonzero": bool(item.get("nonzero", False)),
                "valid": bool(item.get("valid", False)),
            }
            for index, item in report.get("per_physical_layer", {}).items()
        },
        "parameter_count": len(report.get("all_parameters", {})),
    }


def _capture_rng_state() -> dict[str, Any]:
    import torch
    state: dict[str, Any] = {"python": __import__("random").getstate(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    import random
    import torch
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda_all") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def _gather_rank_objects(value: Any, *, rank: int, world_size: int) -> list[Any] | None:
    import torch.distributed as dist
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return [value]
    gathered: list[Any] | None = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(value, gathered, dst=0)
    return gathered


def _collective_check(ok: bool, message: str, *, rank: int, world_size: int, device: Any) -> None:
    """Broadcast rank-0 audit outcomes so failures cannot strand other ranks."""

    import torch
    import torch.distributed as dist
    value = torch.tensor(1 if ok else 0, device=device, dtype=torch.int64)
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        # Every rank contributes its local validity.  Rank 0 may additionally
        # have computed an aggregate result after gather_object; MIN makes a
        # rank-local failure impossible to hide behind a healthy rank 0.
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
        dist.broadcast(value, src=0)
    if int(value.item()) != 1:
        raise RuntimeError(message)


def _gradient_audit_passes(report: dict[str, Any] | None) -> bool:
    """Require all physical groups and the full injection group to be valid."""

    if report is None:
        return False
    injection = report.get("groups", {}).get("injection", {})
    return bool(
        report.get("all_physical_layers_finite_nonzero", False)
        and report.get("all_gradients_finite_nonzero", False)
        and report.get("all_parameters_finite", False)
        and injection
        and all(bool(item.get("valid", False)) for item in injection.values())
    )


def _collective_error_guard(local_error: str | None, *, rank: int, world_size: int, device: Any, context: str) -> None:
    errors = _gather_rank_objects(local_error, rank=rank, world_size=world_size)
    ok = rank != 0 or errors is None or all(error is None for error in errors)
    detail = ""
    if rank == 0 and errors:
        detail = "; ".join(f"rank={index}:{error}" for index, error in enumerate(errors) if error is not None)
    _collective_check(ok, f"{context} failed collectively{': ' + detail if detail else ''}", rank=rank, world_size=world_size, device=device)


def _close_process_group(*, barrier: bool = False) -> None:
    """Best-effort idempotent process-group cleanup for success and errors."""

    try:
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return
        if barrier:
            try:
                dist.barrier()
            except Exception:
                pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    except Exception:
        pass


def _write_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    *,
    output_dir: Path,
    optimizer_step: int,
    config: Stage4Config,
    metadata: dict[str, Any],
    rank: int,
    world_size: int = 1,
    tokenizer: Any | None = None,
    data_cursor: dict[str, Any] | None = None,
    cumulative_samples: int = 0,
    cumulative_valid_tokens: int = 0,
    rank_tmax_stats: dict[str, Any] | None = None,
) -> Path | None:
    """Synchronize and atomically save complete state; retain latest three."""

    import torch
    import torch.distributed as dist
    rank_payload = {"rank": rank, "data_cursor": data_cursor or {}, "rng_state": _capture_rng_state(), "tmax_stats": rank_tmax_stats or {}}
    all_rank_payloads = _gather_rank_objects(rank_payload, rank=rank, world_size=world_size)
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    checkpoint: Path | None = None
    write_error: str | None = None
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / f"checkpoint-{optimizer_step:06d}"
        staging = Path(__import__("tempfile").mkdtemp(prefix=f".{checkpoint.name}.staging-", dir=output_dir))
        try:
            actual_model = model.module if hasattr(model, "module") else model
            actual_model.save_pretrained(staging, safe_serialization=True)
            model_artifacts = [path.relative_to(staging).as_posix() for path in staging.iterdir() if path.is_file()]
            if tokenizer is None:
                raise ValueError("complete checkpoint requires tokenizer")
            tokenizer.save_pretrained(staging / "tokenizer")
            training_state = {
                "optimizer_step": int(optimizer_step),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "configuration": asdict(config),
                "metadata": metadata,
                "sampler_metadata": metadata,
                "cumulative_samples": int(cumulative_samples),
                "cumulative_valid_tokens": int(cumulative_valid_tokens),
                "data_cursors_by_rank": all_rank_payloads,
                "rng_states_by_rank": {str(item["rank"]): item["rng_state"] for item in (all_rank_payloads or [])},
                "rank_tmax_stats": {str(item["rank"]): item.get("tmax_stats", {}) for item in (all_rank_payloads or [])},
                "world_size": int(world_size),
            }
            torch.save(training_state, staging / "training_state.pt")
            weight_files = sorted(staging.glob("model*.safetensors")) + sorted(staging.glob("pytorch_model*.bin"))
            if not weight_files:
                raise ValueError("complete checkpoint is missing model weights")
            index_files = [path for path in (staging / "model.safetensors.index.json", staging / "pytorch_model.bin.index.json") if path.is_file()]
            tokenizer_files = [path.relative_to(staging).as_posix() for path in (staging / "tokenizer").rglob("*") if path.is_file()]
            tokenizer_payload = {Path(path).name for path in tokenizer_files}
            if not ({"tokenizer.json", "tokenizer.model", "vocab.json", "spiece.model"} & tokenizer_payload):
                raise ValueError("complete checkpoint tokenizer is missing a reloadable vocabulary payload")
            artifacts = sorted(set(model_artifacts + ["training_state.pt"] + tokenizer_files + [path.relative_to(staging).as_posix() for path in index_files]))
            missing = [item for item in artifacts if not (staging / item).is_file()]
            if missing:
                raise ValueError(f"complete checkpoint is missing artifacts: {missing}")
            marker = {"complete": True, "optimizer_step": int(optimizer_step), "architecture": MODEL_ARCHITECTURE_CONTRACT, "sampler_metadata": metadata, "world_size": int(world_size), "training_state": "training_state.pt", "artifacts": artifacts, "tokenizer": "tokenizer"}
            (staging / "checkpoint_complete.json").write_text(json.dumps(marker, indent=2, default=str) + "\n", encoding="utf-8")
            if checkpoint.exists():
                import shutil
                shutil.rmtree(checkpoint)
            staging.replace(checkpoint)
            complete = sorted(output_dir.glob("checkpoint-*/checkpoint_complete.json"), key=lambda path: path.parent.name)
            keep = max(1, int(config.checkpoint_retention))
            for old_marker in complete[:-keep]:
                import shutil
                shutil.rmtree(old_marker.parent)
        except Exception as exc:
            import shutil
            shutil.rmtree(staging, ignore_errors=True)
            write_error = repr(exc)
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        error_payload = [write_error]
        dist.broadcast_object_list(error_payload, src=0)
        write_error = error_payload[0]
    if write_error is not None:
        raise RuntimeError(f"checkpoint write failed on rank 0: {write_error}")
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    return checkpoint


def _validate_checkpoint_artifacts(resume_from: Path, marker: Mapping[str, Any]) -> None:
    """Validate every offline model/tokenizer file named by the marker."""

    artifacts = marker.get("artifacts")
    if marker.get("tokenizer") != "tokenizer" or marker.get("training_state") != "training_state.pt":
        raise ValueError("resume checkpoint marker tokenizer/training_state contract mismatch")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("resume checkpoint marker is missing complete artifact list")
    missing_artifacts = [str(item) for item in artifacts if not (resume_from / str(item)).is_file()]
    artifact_set = {str(item) for item in artifacts}
    tokenizer_dir = resume_from / "tokenizer"
    tokenizer_files = [path for path in tokenizer_dir.rglob("*") if path.is_file()] if tokenizer_dir.is_dir() else []
    tokenizer_payload = {path.name for path in tokenizer_files}
    if missing_artifacts or not tokenizer_dir.is_dir():
        raise ValueError(f"resume checkpoint is incomplete; missing artifacts={missing_artifacts} tokenizer_dir={tokenizer_dir.is_dir()}")
    unlisted_tokenizer_files = [path.relative_to(resume_from).as_posix() for path in tokenizer_files if path.relative_to(resume_from).as_posix() not in artifact_set]
    if unlisted_tokenizer_files:
        raise ValueError(f"resume checkpoint marker omitted tokenizer artifacts: {unlisted_tokenizer_files}")
    if not ({"tokenizer.json", "tokenizer.model", "vocab.json", "spiece.model"} & tokenizer_payload):
        raise ValueError("resume checkpoint tokenizer has no reloadable vocabulary payload")
    if "vocab.json" in tokenizer_payload and "merges.txt" not in tokenizer_payload and "tokenizer.json" not in tokenizer_payload:
        raise ValueError("resume checkpoint tokenizer vocab.json is missing required merges.txt")
    # A sharded model must include its index alongside the tensor shards.
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = resume_from / index_name
        if index_path.is_file() and index_name not in artifact_set:
            raise ValueError(f"resume checkpoint marker omitted model index artifact {index_name}")


def _normalized_data_dir(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve())


def _resume_configuration_mismatches(saved: Mapping[str, Any], current: Stage4Config) -> dict[str, Any]:
    """Return exact resume identity mismatches; missing fields fail closed."""

    expected = {
        "seed": int(current.seed),
        "world_size": int(current.world_size),
        "micro_batch_size": int(current.micro_batch_size),
        "gradient_accumulation_steps": int(current.gradient_accumulation_steps),
        "context_length": int(current.context_length),
        "learning_rate": float(current.learning_rate),
        "min_lr": float(current.min_lr),
        "data_dir": _normalized_data_dir(current.data_dir),
    }
    mismatches: dict[str, Any] = {}
    for key, value in expected.items():
        saved_value = saved.get(key)
        if key == "data_dir":
            try:
                saved_value = _normalized_data_dir(saved_value) if saved_value is not None else None
            except Exception:
                pass
        elif key in {"learning_rate", "min_lr"} and saved_value is not None:
            try:
                saved_value = float(saved_value)
            except (TypeError, ValueError):
                pass
        elif saved_value is not None:
            try:
                saved_value = int(saved_value)
            except (TypeError, ValueError):
                pass
        if saved_value != value:
            mismatches[key] = {"saved": saved_value, "current": value}
    return mismatches


def _load_resume_state(resume_from: Path, model: Any, optimizer: Any, scheduler: Any, *, rank: int, world_size: int, metadata: dict[str, Any], config: Stage4Config | None = None) -> dict[str, Any]:
    import torch
    resume_from = Path(resume_from).resolve()
    marker_path = resume_from / "checkpoint_complete.json"
    state_path = resume_from / "training_state.pt"
    if not marker_path.is_file() or not state_path.is_file():
        raise ValueError("resume checkpoint must contain checkpoint_complete.json and training_state.pt")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("complete") is not True or marker.get("architecture") != MODEL_ARCHITECTURE_CONTRACT:
        raise ValueError("resume checkpoint_complete.json contract mismatch")
    if type(marker.get("world_size")) is not int or marker.get("world_size") <= 0:
        raise ValueError("resume checkpoint marker is missing a valid world_size")
    _validate_checkpoint_artifacts(resume_from, marker)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(state.get("optimizer_step", -1)) != int(marker.get("optimizer_step", -2)):
        raise ValueError("checkpoint step mismatch between marker and training_state.pt")
    saved_world_size = state.get("world_size")
    if type(saved_world_size) is not int or saved_world_size != int(world_size):
        raise ValueError(f"checkpoint world_size mismatch or missing: saved={saved_world_size!r} current={world_size}")
    if config is not None:
        saved_configuration = state.get("configuration")
        if not isinstance(saved_configuration, Mapping):
            raise ValueError("resume checkpoint is missing configuration identity")
        configuration_mismatches = _resume_configuration_mismatches(saved_configuration, config)
        if configuration_mismatches:
            raise ValueError(f"resume checkpoint configuration identity mismatch: {configuration_mismatches}")
    validate_sampling_contract(dict(state.get("sampler_metadata", {})))
    saved_metadata = dict(state.get("sampler_metadata", {}))
    if saved_metadata != metadata:
        raise ValueError("resume sampler metadata mismatch")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    by_rank = state.get("rng_states_by_rank", {})
    rank_state = by_rank.get(str(rank)) if isinstance(by_rank, dict) else None
    if rank_state is None:
        raise ValueError(f"resume checkpoint missing RNG state for rank {rank}")
    _restore_rng_state(rank_state)
    return state


def _resume_sampler_sequence_audit(*, seed: int, rank: int, start_optimizer_step: int, batch_size: int, windows: int = 2, checkpoint_seed: int | None = None) -> dict[str, Any]:
    """Compare a resumed suffix with the uninterrupted key-derived sequence.

    Depth sampling is intentionally stateless: the tuple
    ``(seed, rank, optimizer_step, microbatch_index)`` identifies every vector.
    Rebuilding the complete reference prefix and then the resumed suffix makes
    this contract executable in Gate E/Formal resume audits without consuming
    the process-global RNG.
    """

    current_seed = int(seed)
    saved_seed = current_seed if checkpoint_seed is None else int(checkpoint_seed)
    reference: dict[str, list[int]] = {}
    for optimizer_step in range(int(start_optimizer_step) + int(windows)):
        for microbatch_index in range(DEFAULT_GRADIENT_ACCUMULATION_STEPS):
            key = f"{optimizer_step}:{microbatch_index}"
            reference[key] = sample_middle_loop_counts(saved_seed, rank, optimizer_step, microbatch_index, batch_size).tolist()
    resumed: dict[str, list[int]] = {}
    for optimizer_step in range(int(start_optimizer_step), int(start_optimizer_step) + int(windows)):
        for microbatch_index in range(DEFAULT_GRADIENT_ACCUMULATION_STEPS):
            key = f"{optimizer_step}:{microbatch_index}"
            resumed[key] = sample_middle_loop_counts(current_seed, rank, optimizer_step, microbatch_index, batch_size).tolist()
    expected = {key: reference[key] for key in resumed}
    return {
        "key": SAMPLER_KEY,
        "start_optimizer_step": int(start_optimizer_step),
        "windows": int(windows),
        "batch_size": int(batch_size),
        "checkpoint_seed": saved_seed,
        "current_seed": current_seed,
        "reference_suffix": expected,
        "resumed_suffix": resumed,
        "identical": resumed == expected,
    }


def _resume_step_hint(resume_from: Path) -> int:
    """Read the complete marker to size the data pre-audit before loading weights."""

    marker_path = Path(resume_from).resolve() / "checkpoint_complete.json"
    if not marker_path.is_file():
        raise ValueError(f"resume checkpoint is missing completeness marker: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("complete") is not True or marker.get("architecture") != MODEL_ARCHITECTURE_CONTRACT:
        raise ValueError("resume checkpoint marker is incomplete or has the wrong architecture")
    if type(marker.get("world_size")) is not int or marker.get("world_size") <= 0:
        raise ValueError("resume checkpoint marker is missing a valid world_size")
    _validate_checkpoint_artifacts(marker_path.parent, marker)
    step = marker.get("optimizer_step")
    if type(step) is not int or step < 0:
        raise ValueError("resume checkpoint marker has an invalid optimizer_step")
    return step


def _preaudit_dataset(data_dir: Path, *, tokenizer: Any, world_size: int, seed: int, context_length: int = DEFAULT_CONTEXT_LENGTH) -> dict[str, Any]:
    """Audit the immutable manifest and prove rank-local trainability.

    This deliberately does not equate raw parquet rows with trainable
    examples.  Formal capacity comes from deterministic epoch rollover; the
    preaudit only checks shard identity/schema, samples tokenization with the
    real truncation contract, and proves every rank has at least one valid
    tokenized row.
    """

    files = discover_parquet_files(data_dir)
    expected = expected_parquet_names(85)
    names = [path.name for path in files]
    if sorted(names) != expected:
        raise ValueError("dataset preaudit requires the unchanged 85-shard train-xxxxx-of-00085 manifest")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Stage 4 real-data preaudit requires pyarrow") from exc
    assignment = assign_shards(files, world_size=world_size, seed=seed)
    shard_headers: list[dict[str, Any]] = []
    for path in files:
        parquet = pq.ParquetFile(path)
        if "text" not in getattr(parquet.schema, "names", []):
            raise ValueError(f"parquet shard {path.name} is missing required text field")
        shard_headers.append({"name": path.name, "rows": int(parquet.metadata.num_rows), "has_text_field": True})

    def first_trainable(path: Path) -> tuple[bool, int]:
        parquet = pq.ParquetFile(path)
        scanned = 0
        for record_batch in parquet.iter_batches(columns=["text"], batch_size=128, use_threads=False):
            for text_value in record_batch.column("text").to_pylist():
                scanned += 1
                if text_value is None or not str(text_value).strip():
                    continue
                if _tokenize_ids(tokenizer, str(text_value), int(context_length)):
                    return True, scanned
        return False, scanned

    rank_trainable: dict[str, bool] = {}
    rank_probe: dict[str, dict[str, Any]] = {}
    for rank_key, assigned in assignment["rank_shards"].items():
        found = False
        scanned = 0
        for raw_path in assigned:
            valid, path_scanned = first_trainable(Path(raw_path))
            scanned += path_scanned
            if valid:
                found = True
                break
        rank_trainable[rank_key] = found
        rank_probe[rank_key] = {"assigned_shard_count": len(assigned), "rows_scanned_until_first_trainable": scanned, "has_trainable_data": found}
    missing = [rank for rank, found in rank_trainable.items() if not found]
    if missing:
        raise RuntimeError(f"dataset preaudit found no trainable tokenized row for ranks: {missing}")

    sample_paths = select_sample_shards(files, sample_shards=min(3, len(files)), seed=seed)
    sampled_rows = 0
    sampled_nonempty = 0
    sampled_tokenized = 0
    sampled_lengths: list[int] = []
    for path in sample_paths:
        parquet = pq.ParquetFile(path)
        for record_batch in parquet.iter_batches(columns=["text"], batch_size=128, use_threads=False):
            for text_value in record_batch.column("text").to_pylist():
                sampled_rows += 1
                if text_value is None or not str(text_value).strip():
                    continue
                sampled_nonempty += 1
                token_ids = _tokenize_ids(tokenizer, str(text_value), int(context_length))
                if token_ids:
                    sampled_tokenized += 1
                    sampled_lengths.append(len(token_ids))
                if sampled_rows >= 128:
                    break
            if sampled_rows >= 128:
                break
    return {
        "file_count": len(files),
        "expected_shards": 85,
        "manifest": assignment,
        "shards": shard_headers,
        "rank_trainable_probe": rank_probe,
        "rank_has_trainable_data": rank_trainable,
        "sample_tokenization": {
            "shards": [path.name for path in sample_paths],
            "rows_examined": sampled_rows,
            "nonempty_rows": sampled_nonempty,
            "tokenized_nonempty_rows": sampled_tokenized,
            "min_length": min(sampled_lengths) if sampled_lengths else None,
            "max_length": max(sampled_lengths) if sampled_lengths else None,
            "tokenizer_contract": f"add_special_tokens=False,truncation=True,max_length={int(context_length)}",
        },
        "capacity_policy": "deterministic_infinite_epoch_rollover; no_raw_row_capacity_claim",
        "epoch_rollover_policy": "fixed_hashed_rank_shard_order_with_pending_rows_carried_across_epochs",
        "data_cursor_policy": DistributedParquetStream.cursor_policy,
        "dataset_num_proc": 1,
        "raw_row_counts_are_not_training_capacity": True,
    }


def _histogram(values: Sequence[int]) -> dict[str, int]:
    return {str(k): int(sum(int(value) == k for value in values)) for k in POISSON_SUPPORT}


def _depth_window_summary(counts_per_microbatch: Sequence[Sequence[int]], *, tmax_per_microbatch: Sequence[int], tau_per_microbatch: Sequence[Sequence[int]]) -> dict[str, Any]:
    flattened = [int(value) for counts in counts_per_microbatch for value in counts]
    tau_flat = [int(value) for tau in tau_per_microbatch for value in tau]
    return {"microbatch_count": len(counts_per_microbatch), "microbatches": [{"counts": list(map(int, counts)), "histogram": _histogram(counts), "local_tmax": int(tmax), "tau": list(map(int, tau)), "active_updates": int(sum(counts)), "no_op_updates": int(sum(tmax - value for value in counts))} for counts, tmax, tau in zip(counts_per_microbatch, tmax_per_microbatch, tau_per_microbatch)], "histogram": _histogram(flattened), "mean": (sum(flattened) / len(flattened)) if flattened else None, "min": min(flattened) if flattened else None, "max": max(flattened) if flattened else None, "local_tmax_histogram": _histogram(tmax_per_microbatch), "tau_histogram": {str(k): tau_flat.count(k) for k in range(MAX_MIDDLE_LOOPS)}, "active_updates_total": int(sum(flattened)), "no_op_updates_total": int(sum(tmax - value for counts, tmax in zip(counts_per_microbatch, tmax_per_microbatch) for value in counts))}


def _aggregate_depth_summaries(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = [value for summary in summaries for microbatch in summary.get("microbatches", []) for value in microbatch.get("counts", [])]
    tmax = [microbatch.get("local_tmax", 0) for summary in summaries for microbatch in summary.get("microbatches", [])]
    return {"rank_count": len(summaries), "global_histogram": _histogram(counts), "global_mean": sum(counts) / len(counts) if counts else None, "global_min": min(counts) if counts else None, "global_max": max(counts) if counts else None, "local_tmax_distribution": {str(k): tmax.count(k) for k in range(MIN_MIDDLE_LOOPS, MAX_MIDDLE_LOOPS + 1)}, "global_sample_count": len(counts)}


def _aggregate_formal_depth_summaries(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    histogram = {str(k): 0 for k in POISSON_SUPPORT}
    tmax_distribution = {str(k): 0 for k in POISSON_SUPPORT}
    sample_count = active_updates = no_op_updates = 0
    for summary in summaries:
        for key, value in summary.get("histogram", {}).items():
            histogram[str(key)] = histogram.get(str(key), 0) + int(value)
        for key, value in summary.get("local_tmax_distribution", {}).items():
            tmax_distribution[str(key)] = tmax_distribution.get(str(key), 0) + int(value)
        sample_count += int(summary.get("sample_count", 0))
        active_updates += int(summary.get("active_updates_total", 0))
        no_op_updates += int(summary.get("no_op_updates_total", 0))
    total = sum(histogram.values())
    weighted = sum(int(key) * value for key, value in histogram.items())
    observed_keys = [int(key) for key, value in histogram.items() if value]
    return {"rank_count": len(summaries), "global_histogram": histogram, "global_mean": weighted / total if total else None, "global_min": min(observed_keys) if observed_keys else None, "global_max": max(observed_keys) if observed_keys else None, "local_tmax_distribution": tmax_distribution, "global_sample_count": sample_count, "active_updates_total": active_updates, "no_op_updates_total": no_op_updates}


def _synthetic_model(*, device: Any, vocab_size: int = 97) -> Any:
    from transformers import LlamaConfig
    from recursive_model_5_10xpoisson_parcae import RecursiveLlama5_10xpoisson_parcaeForCausalLM
    config = LlamaConfig(vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=MAX_LOGICAL_LAYER_COUNT, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128, initializer_range=0.02, rms_norm_eps=1e-6, pad_token_id=0)
    config.recursive_source_num_hidden_layers = 30
    config.recursive_source_layer_count = 30
    config.recursive_layer_count = PHYSICAL_LAYER_COUNT
    config.recursive_min_middle_loops = MIN_MIDDLE_LOOPS
    config.recursive_max_middle_loops = MAX_MIDDLE_LOOPS
    config.recursive_default_inference_middle_loops = DEFAULT_INFERENCE_MIDDLE_LOOPS
    config.recursive_parameter_gradient_tail_loops = 4
    config.recursive_mapping_policy = EXPECTED_MAPPING_POLICY
    config.recursive_backward_policy = EXPECTED_BACKWARD_POLICY
    config.recursive_training_loop_mode = "per_local_microbatch_per_sequence_truncated_poisson"
    config.recursive_local_tmax = True
    config.recursive_noop_left_alignment = True
    config.recursive_prefix_layer_count = PREFIX_LAYER_COUNT
    config.recursive_middle_layer_count = MIDDLE_LAYER_COUNT
    config.recursive_suffix_layer_count = SUFFIX_LAYER_COUNT
    config.recursive_min_logical_layer_count = MIN_LOGICAL_LAYER_COUNT
    config.recursive_max_logical_layer_count = MAX_LOGICAL_LAYER_COUNT
    config.recursive_sampling_policy = SAMPLING_POLICY
    config.recursive_sampler_version = SAMPLER_VERSION
    config.recursive_sampler_key = SAMPLER_KEY
    config.recursive_poisson_lambda = POISSON_LAMBDA
    config.recursive_poisson_support = list(POISSON_SUPPORT)
    config.recursive_poisson_normalization_z = POISSON_NORMALIZATION_Z
    config.recursive_poisson_Z = POISSON_NORMALIZATION_Z
    config.recursive_poisson_probabilities = list(POISSON_PROBABILITIES)
    config.recursive_ssm_decay = math.sqrt(1.0 / 5.0)
    config.recursive_target_product = -math.log(math.sqrt(1.0 / 5.0))
    config.recursive_initial_decay = math.sqrt(1.0 / 5.0)
    config.recursive_initial_dt = -math.log(math.sqrt(1.0 / 5.0))
    config.recursive_prelude_norm = "LlamaRMSNorm"
    config.recursive_state_init = "like-init"
    config.recursive_state_init_std = 0.02
    config.recursive_embedding_scale = 1.0
    config.recursive_injection_init = EXPECTED_INJECTION_INIT
    config.recursive_injection_formula = "h*decay + dt*(PN(e) @ B.T)"
    config.recursive_injection_no_weight_decay = True
    config.recursive_B_init = "identity"
    config.recursive_learned_h0 = False
    config.recursive_source_layer_indices_0based = [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29]
    config.architectures = ["RecursiveLlama5_10xpoisson_parcaeForCausalLM"]
    model = RecursiveLlama5_10xpoisson_parcaeForCausalLM(config).to(device)
    return model


def _gradient_shape_fingerprint(model: Any) -> str:
    actual = model.module if hasattr(model, "module") else model
    payload = [(name, list(parameter.shape), parameter.grad is not None) for name, parameter in actual.named_parameters()]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _synthetic_gate_a(config: Stage4Config, *, device: Any, rank: int, world_size: int) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    model = _synthetic_model(device=device)
    model = _ddp_wrap(model, device)
    model.train()
    optimizer = torch.optim.AdamW(build_optimizer_param_groups(model, weight_decay=config.weight_decay), lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8, amsgrad=False)
    optimizer_group_audit = _optimizer_group_audit(model, optimizer)
    if not optimizer_group_audit["all_flagged_parameters_no_weight_decay"]:
        raise AssertionError("Gate A injection parameters were not placed in effective weight_decay=0 group")
    optimizer.zero_grad(set_to_none=True)
    local_counts = [4] * config.micro_batch_size if rank == 0 else [4, 10, 6, 7, 8, 9, 10, 5][: config.micro_batch_size]
    local_tmax = max(local_counts)
    for microbatch_index in range(DEFAULT_GRADIENT_ACCUMULATION_STEPS):
        ids = torch.arange(config.micro_batch_size * 8, device=device, dtype=torch.long).reshape(config.micro_batch_size, 8) % 97
        counts = torch.tensor(local_counts, device=device, dtype=torch.long)
        context = model.no_sync() if hasattr(model, "no_sync") and microbatch_index < DEFAULT_GRADIENT_ACCUMULATION_STEPS - 1 else contextlib.nullcontext()
        with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext():
            outputs = model(input_ids=ids, middle_loop_counts=counts, use_cache=False)
            (outputs.logits.float().square().mean() / DEFAULT_GRADIENT_ACCUMULATION_STEPS).backward()
    gradient_report = _gradient_report(model)
    local_gradient_ok = bool(gradient_report["all_finite_nonzero"] and gradient_report["all_parameters_finite"] and gradient_report["all_gradients_finite_nonzero"])
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        gradient_flag = torch.tensor(int(local_gradient_ok), device=device, dtype=torch.int64)
        dist.all_reduce(gradient_flag, op=dist.ReduceOp.MIN)
        local_gradient_ok = bool(gradient_flag.item())
    _collective_check(local_gradient_ok, "Gate A found missing/nonfinite/zero gradients", rank=rank, world_size=world_size, device=device)
    fingerprint = _gradient_shape_fingerprint(model)
    gathered_fingerprints = _gather_rank_objects(fingerprint, rank=rank, world_size=world_size)
    gathered_tmax = _gather_rank_objects(local_tmax, rank=rank, world_size=world_size)
    fingerprint_ok = rank != 0 or not gathered_fingerprints or len(set(gathered_fingerprints)) == 1
    _collective_check(fingerprint_ok, "Gate A rank gradient shape fingerprints differ", rank=rank, world_size=world_size, device=device)
    tmax_ok = rank != 0 or world_size <= 1 or (gathered_tmax is not None and 4 in gathered_tmax and 10 in gathered_tmax)
    _collective_check(tmax_ok, f"Gate A must exercise distinct rank-local Tmax values, got={gathered_tmax}", rank=rank, world_size=world_size, device=device)
    local_depth_summary = _depth_window_summary([local_counts] * DEFAULT_GRADIENT_ACCUMULATION_STEPS, tmax_per_microbatch=[local_tmax] * DEFAULT_GRADIENT_ACCUMULATION_STEPS, tau_per_microbatch=[[(local_tmax - value) for value in local_counts]] * DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    all_depth_summaries = _gather_rank_objects(local_depth_summary, rank=rank, world_size=world_size)
    optimizer.step()
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    return {"status": "PASS", "gate": "A", "synthetic": True, "gradient_report": gradient_report, "gradient_shape_fingerprint": fingerprint, "all_rank_gradient_shape_fingerprints": gathered_fingerprints if rank == 0 else None, "rank_local_tmax": gathered_tmax if rank == 0 else [local_tmax], "local_counts": local_counts, "all_rank_depth_summaries": all_depth_summaries if rank == 0 else None, "optimizer_group_audit": optimizer_group_audit, "no_global_tmax_broadcast": True, "ddp_find_unused_parameters": False, "microbatches": DEFAULT_GRADIENT_ACCUMULATION_STEPS, "all_params_used": True, "all_parameters_finite": gradient_report["all_parameters_finite"], "no_deadlock": True}


def run_training(config: Stage4Config) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _validate_formal_runtime_configuration(config, world_size=world_size, dry_run=config.dry_run)
    schedule_audit = validate_schedule_contract()
    sampling_contract = poisson_metadata()
    validate_sampling_contract(sampling_contract)
    if config.dry_run:
        return {
            "status": "PASS",
            "dry_run": True,
            "configuration": asdict(config),
            "scheduler": scheduler_metadata(total_steps=config.scheduler_total_steps, warmup_steps=config.warmup_steps, max_lr=config.learning_rate, min_lr=config.min_lr),
            "formal_contract": {"world_size": 8, "micro_batch_size": DEFAULT_FORMAL_MICRO_BATCH_SIZE, "gradient_accumulation_steps": DEFAULT_FORMAL_GRADIENT_ACCUMULATION_STEPS, "global_effective_batch_size": DEFAULT_GLOBAL_EFFECTIVE_BATCH, "context_length": 1024, "max_optimizer_steps": 9244, "scheduler_total_steps": 9244, "warmup_steps": 463, "max_lr": DEFAULT_FORMAL_LEARNING_RATE, "min_lr": DEFAULT_FORMAL_MIN_LR, "save_every": 500, "retention": 3, "max_microbatches": None},
            "formal_optimizer_steps": DEFAULT_FORMAL_OPTIMIZER_STEPS,
            "formal_warmup_steps": DEFAULT_FORMAL_WARMUP_STEPS,
            "sampling_contract": sampling_contract,
            "schedule": schedule_audit,
            "world_size_from_environment": world_size,
            "runtime_world_size_check": "deferred_for_dry_run; production FORMAL/D requires WORLD_SIZE=8",
            "resume_supported": True,
            "checkpoint_contains_tokenizer": True,
            "data_policy": DistributedParquetStream.cursor_policy,
            "formal_report_policy": "compact_per_step_plus_first_checkpoint_final_audits",
            "no_global_tmax_broadcast": True,
        }
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if config.device.startswith("cuda") and torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size > 1 and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=config.backend if config.backend != "nccl" or device.type == "cuda" else "gloo")
    torch.manual_seed(config.seed + rank)
    __import__("random").seed(config.seed + rank)
    if config.gate in {"D", "FORMAL"} and world_size != DEFAULT_WORLD_SIZE:
        raise RuntimeError(f"{config.gate} requires torchrun WORLD_SIZE=8, got {world_size}")
    if config.gate == "A":
        try:
            result = _synthetic_gate_a(config, device=device, rank=rank, world_size=world_size)
            if rank == 0 and config.audit_report is not None:
                config.audit_report.parent.mkdir(parents=True, exist_ok=True)
                config.audit_report.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
            return result
        finally:
            _close_process_group(barrier=True)
    if config.gate == "C" and config.model_path is None and config.resume_from is None:
        config = Stage4Config(**{**asdict(config), "gate": "A"})
        try:
            result = _synthetic_gate_a(config, device=device, rank=rank, world_size=world_size)
            if rank == 0 and config.audit_report is not None:
                config.audit_report.parent.mkdir(parents=True, exist_ok=True)
                config.audit_report.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
            return result
        finally:
            _close_process_group(barrier=True)

    if config.resume_from is not None:
        resumed_step_hint = _resume_step_hint(config.resume_from)
    else:
        resumed_step_hint = 0
    model = tokenizer = None
    startup_error: str | None = None
    try:
        model, tokenizer = _load_runtime_model(config, device)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("Stage 4 tokenizer requires pad_token_id or eos_token_id")
        # The preaudit proves only actual tokenizer/schema/trainability facts;
        # epoch rollover, rather than raw row arithmetic, supplies formal data.
        manifest = _preaudit_dataset(config.data_dir, tokenizer=tokenizer, world_size=world_size, seed=config.seed, context_length=config.context_length)
    except Exception as exc:
        startup_error = repr(exc)
        pad_token_id = None
        manifest = None
    _collective_error_guard(startup_error, rank=rank, world_size=world_size, device=device, context="Stage 4 startup/model/data preaudit")
    assert model is not None and tokenizer is not None and pad_token_id is not None and manifest is not None
    model.train()
    model = _ddp_wrap(model, device)
    optimizer = torch.optim.AdamW(build_optimizer_param_groups(model, weight_decay=config.weight_decay), lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8, amsgrad=False)
    optimizer_group_audit = _optimizer_group_audit(model, optimizer)
    _collective_error_guard(None if optimizer_group_audit["all_flagged_parameters_no_weight_decay"] else "flagged injection parameters have nonzero weight decay", rank=rank, world_size=world_size, device=device, context="runtime optimizer audit")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_warmup_factor(step, config.scheduler_total_steps, config.warmup_steps))
    optimizer_step = 0
    cumulative_samples = 0
    cumulative_valid_tokens = 0
    resume_sampler_audit: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    previous_cursor: dict[str, Any] | None = None
    if config.resume_from is not None:
        state = _load_resume_state(config.resume_from, model, optimizer, scheduler, rank=rank, world_size=world_size, metadata=sampling_contract, config=config)
        optimizer_step = int(state["optimizer_step"])
        saved_configuration = state.get("configuration")
        if not isinstance(saved_configuration, Mapping) or saved_configuration.get("seed") is None:
            raise ValueError("resume checkpoint is missing the sampler seed")
        checkpoint_seed = int(saved_configuration["seed"])
        if checkpoint_seed != int(config.seed):
            raise ValueError(f"resume sampler seed mismatch: checkpoint={checkpoint_seed} current={config.seed}")
        resume_sampler_audit = _resume_sampler_sequence_audit(seed=config.seed, checkpoint_seed=checkpoint_seed, rank=rank, start_optimizer_step=optimizer_step, batch_size=config.micro_batch_size, windows=2)
        if not resume_sampler_audit["identical"]:
            raise RuntimeError("resume sampler suffix differs from key-derived uninterrupted reference")
        if config.gate == "E":
            config.max_optimizer_steps = optimizer_step + 2
            saved_configuration = state.get("configuration", {})
            if isinstance(saved_configuration, dict):
                config.scheduler_total_steps = int(saved_configuration.get("scheduler_total_steps", config.scheduler_total_steps))
                config.warmup_steps = int(saved_configuration.get("warmup_steps", config.warmup_steps))
        elif config.gate == "FORMAL":
            saved_configuration = state.get("configuration", {})
            required_saved = {"world_size": 8, "micro_batch_size": DEFAULT_FORMAL_MICRO_BATCH_SIZE, "gradient_accumulation_steps": DEFAULT_FORMAL_GRADIENT_ACCUMULATION_STEPS, "context_length": 1024, "max_optimizer_steps": 9244, "scheduler_total_steps": 9244, "warmup_steps": 463, "learning_rate": DEFAULT_FORMAL_LEARNING_RATE, "min_lr": DEFAULT_FORMAL_MIN_LR, "save_every": 500, "checkpoint_retention": 3, "max_microbatches": None}
            if not isinstance(saved_configuration, dict) or any(saved_configuration.get(key) != value for key, value in required_saved.items()):
                raise ValueError("FORMAL resume checkpoint configuration is not the exact 8-rank/9244-step contract")
        if optimizer_step > config.max_optimizer_steps:
            raise ValueError(f"resume optimizer_step={optimizer_step} exceeds target={config.max_optimizer_steps}")
        cumulative_samples = int(state.get("cumulative_samples", 0))
        cumulative_valid_tokens = int(state.get("cumulative_valid_tokens", 0))
        saved_cursors = state.get("data_cursors_by_rank", [])
        if not isinstance(saved_cursors, list):
            raise ValueError("resume checkpoint data_cursors_by_rank must be a list")
        selected_cursor = next((item for item in saved_cursors if int(item.get("rank", -1)) == rank), None)
        if selected_cursor is None or not isinstance(selected_cursor.get("data_cursor"), Mapping):
            raise ValueError(f"resume checkpoint missing exact data cursor for rank {rank}")
        previous_cursor = dict(selected_cursor["data_cursor"])
    stream = DistributedParquetStream(config.data_dir, tokenizer, rank=rank, world_size=world_size, seed=config.seed, batch_size=config.micro_batch_size, context_length=config.context_length, pad_token_id=int(pad_token_id), gradient_accumulation_steps=config.gradient_accumulation_steps)
    if previous_cursor is not None:
        stream.restore_cursor(previous_cursor)
    stream_iterator = iter(stream)

    detailed = config.gate in {"D", "E"}
    metrics: list[dict[str, Any]] = []
    formal_audits: dict[str, Any] = {}
    shape_fingerprint: str | None = None
    all_shape_fingerprints: list[Any] | None = None
    local_depth_counts = {int(k): 0 for k in POISSON_SUPPORT}
    local_tmax_counts = {int(k): 0 for k in POISSON_SUPPORT}
    local_active_updates = 0
    local_noop_updates = 0
    local_depth_samples = 0
    start = time.time()
    progress_last_time = start
    progress_last_step = optimizer_step
    progress_global_tokens = 0
    progress_global_samples = 0
    progress_log_last_tokens = 0
    progress_log_last_samples = 0
    stopping_reason = "target_reached"
    if device.type == "cuda":
        # Report peaks for the training loop itself, excluding model loading
        # and startup/preaudit allocations.
        torch.cuda.reset_peak_memory_stats(device)
    while optimizer_step < config.max_optimizer_steps:
        optimizer.zero_grad(set_to_none=True)
        window_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        window_tokens = torch.zeros((), device=device, dtype=torch.float64)
        counts_per_microbatch: list[list[int]] = []
        tmax_per_microbatch: list[int] = []
        tau_per_microbatch: list[list[int]] = []
        consumed = 0
        prefetched_batches: list[dict[str, Any]] = []
        local_window_complete = True
        for _prefetch_index in range(config.gradient_accumulation_steps):
            if config.max_microbatches is not None and stream.microbatches_seen >= config.max_microbatches:
                local_window_complete = False
                stopping_reason = "max_microbatches_reached"
                break
            try:
                prefetched_batches.append(next(stream_iterator))
            except StopIteration:
                local_window_complete = False
                stopping_reason = "data_exhausted"
                break
        if dist.is_available() and dist.is_initialized():
            availability = torch.tensor(int(local_window_complete), device=device, dtype=torch.int64)
            dist.all_reduce(availability, op=dist.ReduceOp.MIN)
            all_ranks_window_complete = bool(availability.item())
        else:
            all_ranks_window_complete = local_window_complete
        if not all_ranks_window_complete:
            if config.gate == "FORMAL":
                raise RuntimeError(f"FORMAL accumulation window unavailable: local_prefetched={len(prefetched_batches)} required={config.gradient_accumulation_steps} stop_reason={stopping_reason}")
            break
        for window_microbatch, batch in enumerate(prefetched_batches):
            input_ids = batch["input_ids"].to(device)
            valid = batch["valid_mask"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            counts = sample_middle_loop_counts(config.seed, rank, optimizer_step, window_microbatch, input_ids.shape[0]).to(device)
            local_tmax = int(counts.max().item())
            tau_i = (local_tmax - counts).tolist()
            counts_list = [int(value) for value in counts.tolist()]
            counts_per_microbatch.append(counts_list)
            tmax_per_microbatch.append(local_tmax)
            tau_per_microbatch.append([int(value) for value in tau_i])
            context = model.no_sync() if hasattr(model, "no_sync") and window_microbatch < config.gradient_accumulation_steps - 1 else contextlib.nullcontext()
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
            with context, amp:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None, use_cache=False, middle_loop_counts=counts)
                logits = outputs.logits
                shift_logits = logits[:, :-1].contiguous()
                shift_labels = batch["labels"].to(device)[:, 1:].contiguous()
                shift_valid = valid[:, 1:].contiguous().bool()
                token_losses = F.cross_entropy(shift_logits.float().reshape(-1, logits.shape[-1]), shift_labels.reshape(-1), reduction="none").reshape_as(shift_labels)
                loss_sum = token_losses.masked_select(shift_valid).sum()
                token_count = shift_valid.sum().to(dtype=torch.float64)
                (loss_sum / config.gradient_accumulation_steps).backward()
            window_loss_sum += loss_sum.detach().double()
            window_tokens += token_count
            cumulative_samples += int(input_ids.shape[0])
            cumulative_valid_tokens += int(token_count.item())
            consumed += 1
        if consumed != config.gradient_accumulation_steps:
            if config.gate == "FORMAL":
                raise RuntimeError(f"FORMAL data ended inside accumulation window: consumed={consumed} required={config.gradient_accumulation_steps}")
            optimizer.zero_grad(set_to_none=True)
            break
        if window_tokens.item() <= 0:
            raise RuntimeError("no valid tokens in accumulation window")
        global_window_tokens = window_tokens.clone()
        global_window_loss_sum = window_loss_sum.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_window_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_window_loss_sum, op=dist.ReduceOp.SUM)
        scale = token_weighted_gradient_scale(world_size=world_size, global_window_tokens=int(global_window_tokens.item()))
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale * config.gradient_accumulation_steps)
        if config.max_grad_norm > 0:
            try:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=False)
            except TypeError:  # compatibility with older pinned torch
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        else:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        total_grad_norm_finite_nonzero = bool(torch.isfinite(total_grad_norm).all() and torch.all(total_grad_norm > 0))
        _collective_check(total_grad_norm_finite_nonzero, "total_grad_norm must be finite and >0 before optimizer.step", rank=rank, world_size=world_size, device=device)
        next_step = optimizer_step + 1
        audit_point = detailed or next_step == 1 or next_step % config.save_every == 0 or next_step == config.max_optimizer_steps
        grad_report = None
        gradient_report_error: str | None = None
        if audit_point:
            try:
                grad_report = _gradient_report(model)
            except Exception as exc:
                gradient_report_error = repr(exc)
        if audit_point:
            _collective_error_guard(gradient_report_error, rank=rank, world_size=world_size, device=device, context="Stage 4 gradient report")
        gradient_audit_ok = True
        if audit_point:
            gradient_audit_ok = _gradient_audit_passes(grad_report)
            _collective_check(gradient_audit_ok, "Stage 4 gradient audit found missing/nonfinite/zero physical or injection gradients", rank=rank, world_size=world_size, device=device)
        if detailed or shape_fingerprint is None:
            candidate_fingerprint = _gradient_shape_fingerprint(model)
            candidate_all = _gather_rank_objects(candidate_fingerprint, rank=rank, world_size=world_size)
            fingerprint_ok = rank != 0 or candidate_all is None or len(set(candidate_all)) == 1
            _collective_check(fingerprint_ok, "rank gradient shape fingerprints differ", rank=rank, world_size=world_size, device=device)
            if shape_fingerprint is None:
                shape_fingerprint, all_shape_fingerprints = candidate_fingerprint, candidate_all
        optimizer.step()
        scheduler.step()
        optimizer_step = next_step
        global_window_token_count = int(global_window_tokens.item())
        global_window_sample_count = int(config.micro_batch_size * config.gradient_accumulation_steps * world_size)
        progress_global_tokens += global_window_token_count
        progress_global_samples += global_window_sample_count
        depth_summary = _depth_window_summary(counts_per_microbatch, tmax_per_microbatch=tmax_per_microbatch, tau_per_microbatch=tau_per_microbatch)
        for value in [item for row in counts_per_microbatch for item in row]:
            local_depth_counts[int(value)] += 1
        for value in tmax_per_microbatch:
            local_tmax_counts[int(value)] += 1
        local_active_updates += depth_summary["active_updates_total"]
        local_noop_updates += depth_summary["no_op_updates_total"]
        local_depth_samples += sum(len(row) for row in counts_per_microbatch)
        if detailed:
            all_depth_summaries = _gather_rank_objects(depth_summary, rank=rank, world_size=world_size)
            rank_window_tmax = _gather_rank_objects(tmax_per_microbatch, rank=rank, world_size=world_size)
            metrics.append({"optimizer_step": optimizer_step, "microbatch_index": stream.microbatches_seen, "microbatches": depth_summary, "depth_histogram": depth_summary["histogram"], "global_depth_summary": _aggregate_depth_summaries(all_depth_summaries) if rank == 0 and all_depth_summaries else None, "all_rank_depth_summaries": all_depth_summaries if rank == 0 else None, "local_tmax_stats": {"values": tmax_per_microbatch, "min": min(tmax_per_microbatch), "max": max(tmax_per_microbatch), "mean": sum(tmax_per_microbatch) / len(tmax_per_microbatch)}, "rank_local_tmax": rank_window_tmax if rank == 0 else None, "gradient_report": grad_report, "gradient_audit_passed": bool(gradient_audit_ok), "total_grad_norm_finite_nonzero": bool(total_grad_norm_finite_nonzero), "total_grad_norm": float(total_grad_norm.detach().item()), "gradient_shape_fingerprint": shape_fingerprint, "all_rank_gradient_shape_fingerprints": all_shape_fingerprints if rank == 0 else None, "optimizer_group_audit": optimizer_group_audit, "global_window_tokens": int(global_window_tokens.item()), "local_window_tokens": int(window_tokens.item()), "actual_valid_tokens": int(window_tokens.item()), "local_loss_sum": float(window_loss_sum.item()), "global_loss_sum": float(global_window_loss_sum.item()), "loss": float(global_window_loss_sum.item() / max(1, global_window_tokens.item())), "learning_rate": float(optimizer.param_groups[0]["lr"]), "live_parameter_tail_aligned_steps": 4, "no_global_tmax_broadcast": True})
        else:
            metrics.append({"optimizer_step": optimizer_step, "microbatch_index": stream.microbatches_seen, "depth_histogram": depth_summary["histogram"], "local_tmax_histogram": depth_summary["local_tmax_histogram"], "global_window_tokens": int(global_window_tokens.item()), "actual_valid_tokens": int(window_tokens.item()), "loss": float(global_window_loss_sum.item() / max(1, global_window_tokens.item())), "learning_rate": float(optimizer.param_groups[0]["lr"]), "audit_point": bool(audit_point), "gradient_audit_passed": bool(gradient_audit_ok) if audit_point else None, "grad_norm_finite": bool(total_grad_norm_finite_nonzero), "total_grad_norm_finite_nonzero": bool(total_grad_norm_finite_nonzero), "total_grad_norm": float(total_grad_norm.detach().item()), "no_global_tmax_broadcast": True})
        if config.gate == "FORMAL" and audit_point:
            formal_audits[str(optimizer_step)] = {
                "gradient_report": grad_report if optimizer_step in {1, config.max_optimizer_steps} else _compact_gradient_report(grad_report),
                "gradient_shape_fingerprint": shape_fingerprint,
                "global_window_tokens": int(global_window_tokens.item()),
                "actual_valid_tokens": int(window_tokens.item()),
                "detail_level": "full" if optimizer_step in {1, config.max_optimizer_steps} else "compact",
                "gradient_audit_passed": bool(gradient_audit_ok),
                "grad_norm_finite": bool(total_grad_norm_finite_nonzero),
                "total_grad_norm_finite_nonzero": bool(total_grad_norm_finite_nonzero),
            }
        progress_log_due = optimizer_step % 10 == 0 or optimizer_step == config.max_optimizer_steps
        if progress_log_due:
            # CUDA kernels are asynchronous; synchronize only at the requested
            # progress points so the reported interval/speed and memory are
            # meaningful.  All ranks participate in the memory reduction;
            # only rank 0 emits the human-readable line.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.time()
            memory = "memory=cpu"
            if device.type == "cuda":
                gib = float(1024 ** 3)
                memory_values = torch.tensor(
                    [
                        torch.cuda.memory_allocated(device) / gib,
                        torch.cuda.memory_reserved(device) / gib,
                        torch.cuda.max_memory_allocated(device) / gib,
                        torch.cuda.max_memory_reserved(device) / gib,
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(memory_values, op=dist.ReduceOp.MAX)
                if rank == 0:
                    memory = (
                        f"memory_max_allocated_gib={memory_values[0].item():.2f} "
                        f"memory_max_reserved_gib={memory_values[1].item():.2f} "
                        f"memory_max_peak_allocated_gib={memory_values[2].item():.2f} "
                        f"memory_max_peak_reserved_gib={memory_values[3].item():.2f}"
                    )
            if rank == 0:
                interval_seconds = max(1e-9, now - progress_last_time)
                interval_steps = max(1, optimizer_step - progress_last_step)
                interval_tokens = progress_global_tokens
                interval_samples = progress_global_samples
                if progress_last_step > 0:
                    # The counters are cumulative; retain the previous snapshot
                    # values on the function scope below for subsequent windows.
                    interval_tokens = progress_global_tokens - progress_log_last_tokens
                    interval_samples = progress_global_samples - progress_log_last_samples
                print(
                    f"[stage4][rank=0][progress] step={optimizer_step} "
                    f"loss={global_window_loss_sum.item() / max(1, global_window_token_count):.6f} "
                    f"interval_seconds={interval_seconds:.2f} "
                    f"steps_per_second={interval_steps / interval_seconds:.4f} "
                    f"global_tokens_per_second={interval_tokens / interval_seconds:.1f} "
                    f"global_samples_per_second={interval_samples / interval_seconds:.2f} "
                    f"global_valid_tokens={progress_global_tokens} "
                    f"global_samples={progress_global_samples} {memory}",
                    flush=True,
                )
                progress_last_time = now
                progress_last_step = optimizer_step
                progress_log_last_tokens = progress_global_tokens
                progress_log_last_samples = progress_global_samples
        if optimizer_step % config.save_every == 0 or optimizer_step == config.max_optimizer_steps:
            _write_checkpoint(model, optimizer, scheduler, output_dir=config.output_dir, optimizer_step=optimizer_step, config=config, metadata=sampling_contract, rank=rank, world_size=world_size, tokenizer=tokenizer, data_cursor=stream.cursor(), cumulative_samples=cumulative_samples, cumulative_valid_tokens=cumulative_valid_tokens, rank_tmax_stats={"tmax_histogram": dict(local_tmax_counts), "depth_histogram": dict(local_depth_counts), "active_updates_total": int(local_active_updates), "no_op_updates_total": int(local_noop_updates), "sample_count": int(local_depth_samples), "window_min": min(tmax_per_microbatch), "window_max": max(tmax_per_microbatch), "window_mean": sum(tmax_per_microbatch) / len(tmax_per_microbatch)})
    if config.gate == "FORMAL" and optimizer_step < config.max_optimizer_steps:
        raise RuntimeError(f"FORMAL data ended before target: optimizer_step={optimizer_step} target={config.max_optimizer_steps} stop_reason={stopping_reason}")
    local_depth_summary = {"histogram": {str(k): int(v) for k, v in local_depth_counts.items()}, "local_tmax_distribution": {str(k): int(v) for k, v in local_tmax_counts.items()}, "sample_count": int(local_depth_samples), "active_updates_total": int(local_active_updates), "no_op_updates_total": int(local_noop_updates)}
    all_formal_depth_summaries = _gather_rank_objects(local_depth_summary, rank=rank, world_size=world_size)
    elapsed = max(1e-9, time.time() - start)
    report = {"status": "PASS", "gate": config.gate, "configuration": asdict(config), "optimizer_steps": optimizer_step, "formal_optimizer_steps": DEFAULT_FORMAL_OPTIMIZER_STEPS, "formal_warmup_steps": DEFAULT_FORMAL_WARMUP_STEPS, "metrics": metrics, "formal_audits": formal_audits if config.gate == "FORMAL" and rank == 0 else None, "report_policy": "full_per_window_audit_for_gate_D_or_E; compact_scalar_per_step_and_bounded_audit_points_for_FORMAL", "sampling_contract": sampling_contract, "schedule": schedule_audit, "dataset_manifest": manifest, "world_size": world_size, "rank": rank, "cache_policy": "use_cache=False for training; cache only for scalar inference", "ddp_find_unused_parameters": False, "logical_depth_range": [MIN_LOGICAL_LAYER_COUNT, MAX_LOGICAL_LAYER_COUNT], "cumulative_samples": cumulative_samples, "cumulative_valid_tokens": cumulative_valid_tokens, "stop_reason": stopping_reason, "formal_resume": bool(config.gate == "FORMAL" and config.resume_from is not None), "resume_supported": True, "resume_sampler_audit": resume_sampler_audit, "optimizer_group_audit": optimizer_group_audit, "checkpoint_retention": int(config.checkpoint_retention), "checkpoint_contains_tokenizer": True, "data_cursor_policy": DistributedParquetStream.cursor_policy, "global_depth_summary": _aggregate_formal_depth_summaries(all_formal_depth_summaries or []) if rank == 0 else None, "elapsed_seconds": elapsed, "no_global_tmax_broadcast": True}
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        report_path = config.report_path or (config.output_dir / "stage4_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    _close_process_group(barrier=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        report = run_training(config)
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("status") == "PASS" else 1
    except Exception as exc:
        # An exception on one rank must not leave a process group behind while
        # peers wait in a later collective.  Destruction is idempotent and is
        # intentionally non-collective on this failure path.
        _close_process_group(barrier=False)
        print(f"[result] status=FAIL error={exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
