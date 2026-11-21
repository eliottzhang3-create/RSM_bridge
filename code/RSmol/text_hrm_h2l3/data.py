from __future__ import annotations
from pathlib import Path
import random
from typing import Any,Mapping,Sequence,Iterator
DEFAULT_CONTEXT_LENGTH=1024
EXPECTED_PARQUET_COUNT=85
EXPECTED_PARQUET_NAMES = tuple(
    f"train-{index:05d}-of-{EXPECTED_PARQUET_COUNT:05d}.parquet"
    for index in range(EXPECTED_PARQUET_COUNT)
)


def tupleize(value):
    return tuple(tupleize(x) for x in value) if isinstance(value, (list, tuple)) else value


class ParquetStream:
    """Original bounded shuffle; exact next-batch replay from saved shard-start RNG.

    No shuffle-buffer payload is serialized: resume re-reads the current shard
    from its original RNG state and skips the already emitted microbatches.
    This preserves row order given unchanged shards/tokenizer, but does not
    claim bitwise CUDA/DDP training reproducibility.
    """
    def __init__(self, shards, tokenizer, *, rank, seed=0, micro_batch_size=2,
                 context_length=1024, buffer_size=4096, pad_token_id=0, cursor=None):
        self.shards = [Path(p) for p in shards]
        if not self.shards or micro_batch_size <= 0 or not 0 < context_length <= 1024:
            raise ValueError('invalid stream geometry')
        self.tokenizer, self.rank = tokenizer, rank
        self.seed, self.micro = seed, micro_batch_size
        self.context, self.buffer, self.pad = context_length, buffer_size, pad_token_id
        self.rng = random.Random(seed + 1009 * rank)
        self.epoch, self.shard, self.in_shard, self.microbatches = 1, 0, 0, 0
        self.start_rng = self.rng.getstate()
        self.resume = cursor
        if cursor is not None:
            if cursor.get('version') != 1 or cursor.get('rank') != rank:
                raise ValueError('wrong data cursor version/rank')
            self.epoch, self.shard = cursor['epoch'], cursor['shard']
            self.in_shard, self.microbatches = cursor['in_shard'], cursor['microbatches']
            if not 0 <= self.shard < len(self.shards) or min(self.epoch, self.microbatches + 1) < 1 or self.in_shard < 0:
                raise ValueError('invalid cursor position')
            self.rng.setstate(tupleize(cursor['shard_start_rng']))

    def __iter__(self):
        start = self.shard
        skip = self.in_shard if self.resume else 0
        while True:
            emitted = 0
            for index in range(start, len(self.shards)):
                self.shard = index
                self.start_rng = self.rng.getstate()
                self.in_shard = skip if self.resume and index == start else 0
                to_skip = self.in_shard
                batch = []
                for text in _iter_shard_texts(self.shards[index], self.rng, self.buffer):
                    ids = _tokenize_ids(self.tokenizer, text, self.context)
                    if not ids:
                        continue
                    batch.append(ids)
                    if len(batch) == self.micro:
                        if to_skip:
                            to_skip -= 1
                        else:
                            value = collate_dynamic_padding(batch, pad_token_id=self.pad)
                            self.in_shard += 1
                            self.microbatches += 1
                            emitted += 1
                            yield value
                        batch = []
                if to_skip:
                    raise ValueError('cursor exceeds current shard contents')
                self.resume = None
            if not emitted and start == 0 and skip == 0:
                raise ValueError('rank has no complete tokenized microbatch')
            self.epoch += 1
            start, skip = 0, 0

    def cursor(self):
        return dict(version=1, rank=self.rank, epoch=self.epoch, shard=self.shard,
                    in_shard=self.in_shard, microbatches=self.microbatches,
                    shard_start_rng=self.start_rng,
                    replay='deterministic_current_shard_from_start',
                    cuda_bitwise_exact_claimed=False)

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
