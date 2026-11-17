"""Mizar text order with exact current-shard replay, without storing text buffers.

The dedicated shuffle RNG state at shard entry and the number of consumed full
microbatches reconstruct the buffer, its input position, and its RNG exactly.
Replay is CPU/I/O work and may take time on a large shard; it reads no old shard.
"""
from __future__ import annotations

import copy
from pathlib import Path
import random

CURSOR_POLICY = 'mizar_seed_shuffled_contiguous_ranks_buffer4096_exact_shard_replay_v1'


def discover_shards(data_dir):
    candidates = [Path(data_dir), Path(data_dir) / 'data']
    found = [sorted(p.resolve() for p in folder.glob('*.parquet')) for folder in candidates]
    found = [paths for paths in found if paths]
    if len(found) != 1 or len(found[0]) != 85:
        raise ValueError('Expected exactly one data directory containing 85 parquet shards')
    return found[0]


def manifest_records(paths):
    return [{'path': str(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
            for p in paths]


def assign_shards(paths, world_size, seed):
    if world_size <= 0 or len(paths) < world_size:
        raise ValueError('Each rank must receive at least one shard')
    shuffled = list(sorted(paths))
    random.Random(seed).shuffle(shuffled)
    base, remainder = divmod(len(shuffled), world_size)
    result, offset = [], 0
    for rank in range(world_size):
        count = base + (rank < remainder)
        result.append(shuffled[offset:offset+count])
        offset += count
    return result


def iter_shard_texts(path, rng, buffer_size):
    import pyarrow.parquet as parquet
    buffer = []
    for record_batch in parquet.ParquetFile(path).iter_batches(
            batch_size=min(buffer_size, 1024), columns=['text'], use_threads=False):
        for value in record_batch.column('text').to_pylist():
            if value is None or not str(value).strip():
                continue
            text = str(value)
            if len(buffer) < buffer_size:
                buffer.append(text)
            else:
                index = rng.randrange(len(buffer))
                yield buffer[index]
                buffer[index] = text
    rng.shuffle(buffer)
    yield from buffer


def collate(token_sequences, pad_token_id):
    import torch
    if not token_sequences or any(not row for row in token_sequences):
        raise ValueError('A microbatch must contain nonempty token sequences')
    length = max(map(len, token_sequences))
    ids = torch.tensor([row + [pad_token_id]*(length-len(row)) for row in token_sequences], dtype=torch.long)
    mask = torch.tensor([[1]*len(row) + [0]*(length-len(row)) for row in token_sequences], dtype=torch.long)
    labels = ids.clone().masked_fill(mask == 0, -100)
    return {'input_ids': ids, 'attention_mask': mask, 'labels': labels, 'valid_mask': mask.bool()}


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, (list, tuple)) else value


class DistributedParquetStream:
    cursor_policy = CURSOR_POLICY

    def __init__(self, paths, tokenizer, *, rank, world_size, batch_size,
                 context_length, pad_token_id, seed=0, buffer_size=4096):
        if not 0 <= rank < world_size or batch_size <= 0 or not 0 < context_length <= 1024 or buffer_size <= 0:
            raise ValueError('Invalid rank, batch, context length or shuffle buffer size')
        self.local_paths = assign_shards(paths, world_size, seed)[rank]
        self.tokenizer, self.rank, self.world_size = tokenizer, rank, world_size
        self.batch_size, self.context_length = batch_size, context_length
        self.pad_token_id, self.seed, self.buffer_size = pad_token_id, seed, buffer_size
        self.epoch = self.shard_index = self.shard_microbatches_seen = 0
        self.microbatches_seen = self.samples_seen = 0
        self._rng = random.Random(seed + 1009*rank)
        self._shard_start_rng = None

    def cursor(self):
        return copy.deepcopy({
            'policy': self.cursor_policy, 'rank': self.rank, 'world_size': self.world_size,
            'batch_size': self.batch_size, 'context_length': self.context_length,
            'pad_token_id': self.pad_token_id, 'seed': self.seed, 'buffer_size': self.buffer_size,
            'shard_order': [str(p) for p in self.local_paths], 'epoch': self.epoch,
            'shard_index': self.shard_index, 'shard_microbatches_seen': self.shard_microbatches_seen,
            'microbatches_seen': self.microbatches_seen, 'samples_seen': self.samples_seen,
            'shard_start_rng': self._shard_start_rng, 'rng': self._rng.getstate(),
        })

    def restore_cursor(self, state):
        for key in ('policy', 'rank', 'world_size', 'batch_size', 'context_length',
                    'pad_token_id', 'seed', 'buffer_size', 'shard_order'):
            if state.get(key) != self.cursor()[key]:
                raise ValueError(f'Data resume contract differs: {key}')
        for key in ('epoch', 'shard_index', 'shard_microbatches_seen', 'microbatches_seen', 'samples_seen'):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError(f'Invalid data cursor: {key}')
            setattr(self, key, state[key])
        if self.shard_index >= len(self.local_paths):
            raise ValueError('Data cursor shard index outside rank assignment')
        if self.shard_microbatches_seen and state.get('shard_start_rng') is None:
            raise ValueError('Data replay requires the RNG state at shard entry')
        self._shard_start_rng = _tuples(state['shard_start_rng']) if state['shard_start_rng'] is not None else None
        self._rng.setstate(_tuples(state['rng']))

    def __iter__(self):
        while True:
            epoch_batches = 0
            for index in range(self.shard_index, len(self.local_paths)):
                self.shard_index = index
                skip = self.shard_microbatches_seen
                if self._shard_start_rng is None:
                    self._shard_start_rng = self._rng.getstate()
                else:
                    self._rng.setstate(self._shard_start_rng)
                batch = []
                for text in iter_shard_texts(self.local_paths[index], self._rng, self.buffer_size):
                    ids = self.tokenizer(text, add_special_tokens=False, truncation=True,
                                         max_length=self.context_length)['input_ids']
                    if not ids:
                        continue
                    batch.append(list(map(int, ids)))
                    if len(batch) != self.batch_size:
                        continue
                    epoch_batches += 1
                    if skip:
                        skip -= 1
                    else:
                        self.shard_microbatches_seen += 1
                        self.microbatches_seen += 1
                        self.samples_seen += len(batch)
                        yield collate(batch, self.pad_token_id)
                    batch = []
                if skip:
                    raise ValueError('Data cursor exceeds the replayed shard')
                # As in the Mizar trainer, incomplete shard tails are dropped.
                self.shard_microbatches_seen = 0
                self._shard_start_rng = None
            self.epoch += 1
            self.shard_index = 0
            if epoch_batches == 0:
                raise ValueError('No full microbatch available in this rank data stream')
