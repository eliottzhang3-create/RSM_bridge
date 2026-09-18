"""ReasonAQA manifest dataset for the isolated MeSH audio route."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


WAVEFORM_CACHE_FORMAT = "raw_fixed_waveform_shards_v1"
UNIQUE_WAVEFORM_STORE_FORMAT = "manifest_unique_fixed_waveform_store_v1"
WAVEFORM_CACHE_GROUP_MARKERS = (
    ("/audiocaps_v2/train/", "audiocaps_train"),
    ("/clotho_aqa_audio/audio_files/", "clotho_aqa"),
    ("/clotho_v2_1/development/", "clotho_development"),
)


@dataclass(frozen=True)
class WaveformCacheLocation:
    shard_id: int
    row: int


class WaveformShardCache:
    """Read fixed float32 waveforms from immutable raw shards via mmap."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve(strict=True)
        metadata_path = self.cache_dir / "metadata.json"
        index_path = self.cache_dir / "index.jsonl"
        if (self.cache_dir / "BUILDING").exists():
            raise RuntimeError(f"waveform cache is still being built: {self.cache_dir}")
        if not metadata_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(f"waveform cache lacks metadata.json or index.jsonl: {self.cache_dir}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        index_digest = hashlib.sha256()
        with index_path.open("rb") as index_binary:
            for chunk in iter(lambda: index_binary.read(8 * 1024 * 1024), b""):
                index_digest.update(chunk)
        if index_digest.hexdigest() != self.metadata.get("index_sha256"):
            raise RuntimeError("waveform cache index SHA256 mismatch")
        expected = {
            "status": "PASS",
            "format": WAVEFORM_CACHE_FORMAT,
            "sample_rate": 32000,
            "seconds": 10,
            "samples_per_audio": 320000,
            "bytes_per_audio": 1280000,
            "dtype": "float32",
            "byte_order": "little",
        }
        mismatches = {
            key: {"expected": value, "actual": self.metadata.get(key)}
            for key, value in expected.items()
            if self.metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"waveform cache contract mismatch: {mismatches}")
        self.num_shards = int(self.metadata.get("num_shards", 0))
        if self.num_shards <= 0:
            raise RuntimeError("waveform cache metadata has no positive num_shards")
        shard_metadata = self.metadata.get("shards")
        if not isinstance(shard_metadata, list) or len(shard_metadata) != self.num_shards:
            raise RuntimeError("waveform cache shard metadata count mismatch")
        self.shard_rows: dict[int, int] = {}
        self.shard_paths: dict[int, Path] = {}
        for item in shard_metadata:
            shard_id = int(item["shard_id"])
            rows = int(item["rows"])
            path = self.cache_dir / str(item["shard"])
            expected_bytes = rows * int(self.metadata["bytes_per_audio"])
            if shard_id in self.shard_rows or not path.is_file() or path.stat().st_size != expected_bytes:
                raise RuntimeError(f"waveform cache shard contract failed: shard_id={shard_id} path={path}")
            self.shard_rows[shard_id] = rows
            self.shard_paths[shard_id] = path
        if sorted(self.shard_rows) != list(range(self.num_shards)):
            raise RuntimeError("waveform cache shard IDs are not contiguous")

        self._by_path: dict[str, WaveformCacheLocation] = {}
        self._by_group_relative: dict[tuple[str, str], WaveformCacheLocation] = {}
        observed_rows = [0 for _ in range(self.num_shards)]
        with index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                shard_id = int(item["shard_id"])
                row = int(item["row"])
                if shard_id < 0 or shard_id >= self.num_shards or row != observed_rows[shard_id]:
                    raise RuntimeError(f"waveform cache index order/row mismatch at line {line_number}")
                if row >= self.shard_rows[shard_id]:
                    raise RuntimeError(f"waveform cache index row exceeds shard at line {line_number}")
                location = WaveformCacheLocation(shard_id=shard_id, row=row)
                path_key = self._normalize_path(str(item["source_path"]))
                group_key = (str(item["source_group"]), self._normalize_relative(str(item["relative_path"])))
                if path_key in self._by_path or group_key in self._by_group_relative:
                    raise RuntimeError(f"duplicate waveform cache index key at line {line_number}")
                self._by_path[path_key] = location
                self._by_group_relative[group_key] = location
                observed_rows[shard_id] += 1
        expected_rows = [self.shard_rows[index] for index in range(self.num_shards)]
        if observed_rows != expected_rows or len(self._by_path) != int(self.metadata.get("num_unique_audio_files", -1)):
            raise RuntimeError(
                "waveform cache index cardinality mismatch: "
                f"observed_rows={observed_rows} expected_rows={expected_rows} "
                f"paths={len(self._by_path)} metadata={self.metadata.get('num_unique_audio_files')}"
            )
        self._mmap_by_shard: dict[int, np.memmap] = {}

    @staticmethod
    def _normalize_path(value: str) -> str:
        return os.path.normpath(os.path.expanduser(value)).replace("\\", "/")

    @staticmethod
    def _normalize_relative(value: str) -> str:
        return value.replace("\\", "/").lstrip("/")

    @classmethod
    def _group_relative_from_path(cls, value: str) -> tuple[str, str] | None:
        normalized = cls._normalize_path(value)
        for marker, group in WAVEFORM_CACHE_GROUP_MARKERS:
            if marker in normalized:
                return group, cls._normalize_relative(normalized.split(marker, 1)[1])
        return None

    def locate(self, source_path: str | Path) -> WaveformCacheLocation:
        value = str(source_path)
        direct = self._by_path.get(self._normalize_path(value))
        if direct is not None:
            return direct
        group_relative = self._group_relative_from_path(value)
        if group_relative is not None:
            location = self._by_group_relative.get(group_relative)
            if location is not None:
                return location
        raise KeyError(f"audio path is absent from waveform cache: {value}")

    def load(self, source_path: str | Path) -> torch.Tensor:
        return self.load_location(self.locate(source_path))

    def load_location(self, location: WaveformCacheLocation) -> torch.Tensor:
        mmap = self._mmap_by_shard.get(location.shard_id)
        if mmap is None:
            mmap = np.memmap(
                self.shard_paths[location.shard_id],
                dtype="<f4",
                mode="c",
                shape=(self.shard_rows[location.shard_id], int(self.metadata["samples_per_audio"])),
            )
            self._mmap_by_shard[location.shard_id] = mmap
        # mode="c" is a writable copy-on-write mapping, so torch can safely
        # wrap the row without copying or being able to modify the shard file.
        return torch.from_numpy(mmap[location.row]).reshape(1, int(self.metadata["samples_per_audio"]))


class UniqueWaveformStore:
    """Read one manifest-scoped, deduplicated float32 waveform store.

    This reader is deliberately independent from ``WaveformShardCache``.  The
    old cache is a historical multi-shard experiment, whereas this format has
    one fixed-stride data file that every local DDP rank can mmap and share
    through the node page cache.
    """

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir).expanduser().resolve(strict=True)
        if (self.store_dir / "BUILDING").exists():
            raise RuntimeError(f"unique waveform store is still being built: {self.store_dir}")
        metadata_path = self.store_dir / "metadata.json"
        index_path = self.store_dir / "index.jsonl"
        if not metadata_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(f"unique waveform store lacks metadata.json or index.jsonl: {self.store_dir}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "status": "PASS",
            "format": UNIQUE_WAVEFORM_STORE_FORMAT,
            "sample_rate": 32000,
            "seconds": 10,
            "samples_per_audio": 320000,
            "bytes_per_audio": 1280000,
            "dtype": "float32",
            "byte_order": "little",
            "data_file": "waveforms.f32",
        }
        mismatches = {
            key: {"expected": value, "actual": self.metadata.get(key)}
            for key, value in expected.items()
            if self.metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"unique waveform store contract mismatch: {mismatches}")
        index_digest = hashlib.sha256()
        with index_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                index_digest.update(chunk)
        if index_digest.hexdigest() != self.metadata.get("index_sha256"):
            raise RuntimeError("unique waveform store index SHA256 mismatch")

        self.num_audio = int(self.metadata.get("num_unique_audio_files", -1))
        self.samples_per_audio = int(self.metadata["samples_per_audio"])
        self.bytes_per_audio = int(self.metadata["bytes_per_audio"])
        self.data_path = self.store_dir / str(self.metadata["data_file"])
        expected_bytes = self.num_audio * self.bytes_per_audio
        if self.num_audio <= 0 or not self.data_path.is_file() or self.data_path.stat().st_size != expected_bytes:
            raise RuntimeError(
                "unique waveform store data-file contract failed: "
                f"audio={self.num_audio} path={self.data_path} expected_bytes={expected_bytes}"
            )

        self._by_path: dict[str, int] = {}
        observed = 0
        with index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                audio_id = int(item["audio_id"])
                if audio_id != observed or int(item["byte_offset"]) != audio_id * self.bytes_per_audio:
                    raise RuntimeError(f"unique waveform store index order/offset mismatch at line {line_number}")
                keys = [str(item["source_path"]), *[str(value) for value in item.get("manifest_aliases", [])]]
                for value in keys:
                    normalized = self._normalize_path(value)
                    previous = self._by_path.setdefault(normalized, audio_id)
                    if previous != audio_id:
                        raise RuntimeError(f"unique waveform store alias collision at line {line_number}: {value}")
                observed += 1
        if observed != self.num_audio:
            raise RuntimeError(f"unique waveform store index cardinality mismatch: {observed} != {self.num_audio}")
        self._mmap: np.memmap | None = None

    @staticmethod
    def _normalize_path(value: str) -> str:
        return os.path.normpath(os.path.expanduser(value)).replace("\\", "/")

    def locate(self, source_path: str | Path) -> int:
        normalized = self._normalize_path(str(source_path))
        found = self._by_path.get(normalized)
        if found is None:
            raise KeyError(f"audio path is absent from unique waveform store: {source_path}")
        return found

    def load(self, source_path: str | Path) -> torch.Tensor:
        return self.load_audio_id(self.locate(source_path))

    def load_audio_id(self, audio_id: int) -> torch.Tensor:
        audio_id = int(audio_id)
        if audio_id < 0 or audio_id >= self.num_audio:
            raise IndexError(f"unique waveform audio_id is out of range: {audio_id}")
        if self._mmap is None:
            # Copy-on-write makes the NumPy view writable for torch without
            # ever permitting an accidental mutation of the immutable file.
            # Clean mapped pages remain physically shared by all local ranks.
            self._mmap = np.memmap(
                self.data_path,
                dtype="<f4",
                mode="c",
                shape=(self.num_audio, self.samples_per_audio),
            )
        return torch.from_numpy(self._mmap[audio_id]).reshape(1, self.samples_per_audio)

    def close(self) -> None:
        """Close the mapped payload after all borrowed views have been dropped."""
        mapped, self._mmap = self._mmap, None
        if mapped is not None:
            mapped._mmap.close()


def load_waveform(path: str | Path, *, sample_rate: int = 32000, seconds: int = 10) -> torch.Tensor:
    path = Path(path)
    try:
        import torchaudio
        waveform, source_rate = torchaudio.load(str(path))
    except Exception:
        try:
            import soundfile as sf
            import numpy as np
            array, source_rate = sf.read(str(path), always_2d=True)
            waveform = torch.from_numpy(np.asarray(array, dtype="float32").T)
        except Exception:
            with wave.open(str(path), "rb") as handle:
                source_rate = handle.getframerate()
                channels = handle.getnchannels()
                width = handle.getsampwidth()
                raw = handle.readframes(handle.getnframes())
            if width != 2:
                raise RuntimeError("wave fallback supports 16-bit PCM only")
            import numpy as np
            array = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels).astype("float32") / 32768.0
            waveform = torch.from_numpy(array.T)
    waveform = waveform.float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if int(source_rate) != sample_rate:
        try:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, int(source_rate), sample_rate)
        except Exception:
            target = max(1, round(waveform.shape[-1] * sample_rate / int(source_rate)))
            waveform = F.interpolate(waveform.unsqueeze(0), size=target, mode="linear", align_corners=False).squeeze(0)
    target_samples = sample_rate * seconds
    # Project contract: long audio is always the first ten seconds.
    if waveform.shape[-1] > target_samples:
        waveform = waveform[..., :target_samples]
    elif waveform.shape[-1] < target_samples:
        waveform = F.pad(waveform, (0, target_samples - waveform.shape[-1]))
    return waveform


def _path(row: dict[str, Any], first: bool) -> str:
    keys = ("audio1_path", "filepath1") if first else ("audio2_path", "filepath2")
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return ""


class ReasonAQADataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest: str | Path, tokenizer: Any, *, max_prompt_tokens: int = 129, max_answer_tokens: int = 250, sample_rate: int = 32000, seconds: int = 10, waveform_cache_dir: str | Path | None = None, unique_waveform_store_dir: str | Path | None = None) -> None:
        self.manifest = Path(manifest)
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_answer_tokens = int(max_answer_tokens)
        self.sample_rate = int(sample_rate)
        self.seconds = int(seconds)
        if waveform_cache_dir is not None and unique_waveform_store_dir is not None:
            raise ValueError("waveform_cache_dir and unique_waveform_store_dir are mutually exclusive")
        self.waveform_cache = WaveformShardCache(waveform_cache_dir) if waveform_cache_dir is not None else None
        self.unique_waveform_store = UniqueWaveformStore(unique_waveform_store_dir) if unique_waveform_store_dir is not None else None
        if (self.waveform_cache is not None or self.unique_waveform_store is not None) and (self.sample_rate != 32000 or self.seconds != 10):
            raise ValueError("waveform cache/store requires sample_rate=32000 and seconds=10")
        if self.unique_waveform_store is not None:
            manifest_hasher = hashlib.sha256()
            with self.manifest.open("rb") as manifest_handle:
                for chunk in iter(lambda: manifest_handle.read(8 * 1024 * 1024), b""):
                    manifest_hasher.update(chunk)
            manifest_digest = manifest_hasher.hexdigest()
            if manifest_digest != self.unique_waveform_store.metadata.get("manifest_sha256"):
                raise RuntimeError(
                    "unique waveform store manifest SHA256 mismatch: "
                    f"manifest={manifest_digest} store={self.unique_waveform_store.metadata.get('manifest_sha256')}"
                )
        self.rows = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"empty ReasonAQA manifest: {self.manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def audio_paths(self, index: int) -> tuple[str, str]:
        """Return the normalized two-clip input contract without reading audio."""

        row = self.rows[int(index)]
        audio1 = _path(row, True)
        audio2 = _path(row, False)
        audio2 = audio2 or audio1
        if not audio1:
            raise ValueError(f"manifest row {index} lacks audio1")
        return audio1, audio2

    def audio_structure(self, index: int) -> tuple[bool, bool]:
        """Return (single_slot, same_waveform) from manifest structure.

        ``audio2_reused`` is the materializer's explicit marker for a missing
        second slot.  A row with two explicit identical paths remains dual;
        it must retain the second separator/prefix even though its waveform
        can be encoded once.
        """
        row = self.rows[int(index)]
        raw2 = _path(row, False)
        single = bool(row.get("audio2_reused", False)) or not bool(raw2)
        audio1, audio2 = self.audio_paths(index)
        return single, audio1 == audio2

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio1, audio2 = self.audio_paths(index)
        single_slot, same_waveform = self.audio_structure(index)
        prompt = str(row.get("prompt") or row.get("question") or row.get("input") or "")
        answer = str(row.get("answer") or row.get("target") or row.get("output") or row.get("caption1") or "")
        if not answer:
            raise ValueError(f"manifest row {index} lacks answer")
        if self.unique_waveform_store is not None:
            audio1_waveform = self.unique_waveform_store.load(audio1)
            audio2_waveform = None if single_slot else (audio1_waveform if same_waveform else self.unique_waveform_store.load(audio2))
            cache_shard_ids = None
        elif self.waveform_cache is None:
            audio1_waveform = load_waveform(audio1, sample_rate=self.sample_rate, seconds=self.seconds)
            audio2_waveform = None if single_slot else (audio1_waveform if same_waveform else load_waveform(audio2, sample_rate=self.sample_rate, seconds=self.seconds))
            cache_shard_ids = None
        else:
            first_location = self.waveform_cache.locate(audio1)
            second_location = first_location if audio2 == audio1 else self.waveform_cache.locate(audio2)
            audio1_waveform = self.waveform_cache.load_location(first_location)
            audio2_waveform = None if single_slot else (audio1_waveform if same_waveform else self.waveform_cache.load_location(second_location))
            cache_shard_ids = (first_location.shard_id, second_location.shard_id)
        item = {"audio1": audio1_waveform, "audio2": audio2_waveform, "prompt": prompt, "answer": answer, "row_index": index, "audio2_reused": same_waveform, "single_audio_slot": single_slot}
        if cache_shard_ids is not None:
            item["waveform_cache_shard_ids"] = cache_shard_ids
        return item

    def cache_shard_ids(self, index: int) -> tuple[int, int]:
        """Resolve both paths while returning audio1's primary grouping shard."""

        if self.waveform_cache is None:
            raise RuntimeError("cache_shard_ids requires a waveform shard cache")
        row = self.rows[int(index)]
        audio1 = _path(row, True)
        audio2 = _path(row, False) or audio1
        if not audio1:
            raise ValueError(f"manifest row {index} lacks audio1")
        first = self.waveform_cache.locate(audio1).shard_id
        second = first if audio2 == audio1 else self.waveform_cache.locate(audio2).shard_id
        return first, second


class ShardAwareDistributedBatchSampler(Sampler[list[int]]):
    """Assign disjoint shards to ranks, then shuffle rows within each shard."""

    def __init__(
        self,
        dataset: ReasonAQADataset,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        if dataset.waveform_cache is None:
            raise ValueError("shard-aware sampling requires waveform_cache_dir")
        if int(batch_size) <= 0 or int(num_replicas) <= 0 or not 0 <= int(rank) < int(num_replicas):
            raise ValueError("invalid shard-aware batch sampler topology")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.groups: dict[int, list[int]] = {
            shard_id: [] for shard_id in range(dataset.waveform_cache.num_shards)
        }
        secondary_cross_shard = 0
        for row_index in range(len(dataset)):
            primary, secondary = dataset.cache_shard_ids(row_index)
            self.groups[primary].append(row_index)
            secondary_cross_shard += int(primary != secondary)
        if sum(len(rows) for rows in self.groups.values()) != len(dataset):
            raise RuntimeError("shard-aware sampler did not assign every manifest row")
        self.secondary_cross_shard_records = secondary_cross_shard
        if len(self.groups) % self.num_replicas != 0:
            raise ValueError(
                "shard-aware sampler requires an equal integer number of shards per rank: "
                f"shards={len(self.groups)} ranks={self.num_replicas}"
            )
        self.shards_per_rank = len(self.groups) // self.num_replicas
        if self._epoch_batch_limit() <= 0:
            raise RuntimeError("shard-aware sampler has no complete distributed batch group")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._epoch_batch_limit()

    def _epoch_assignments(self) -> list[list[int]]:
        """Return the globally shuffled, disjoint shard allocation for this epoch."""

        rng = random.Random(self.seed + self.epoch)
        shard_order = list(self.groups)
        rng.shuffle(shard_order)
        return [
            shard_order[start:start + self.shards_per_rank]
            for start in range(0, len(shard_order), self.shards_per_rank)
        ]

    def _candidate_batches(self, shard_ids: list[int], *, rank: int) -> list[list[int]]:
        """Build one rank's batches without ever mixing full shard interiors."""

        rng = random.Random(self.seed + self.epoch + 1_000_003 * (int(rank) + 1))
        shard_order = list(shard_ids)
        rng.shuffle(shard_order)
        batches: list[list[int]] = []
        tail: list[int] = []
        for shard_id in shard_order:
            rows = list(self.groups[shard_id])
            rng.shuffle(rows)
            full_end = (len(rows) // self.batch_size) * self.batch_size
            batches.extend(
                rows[start:start + self.batch_size]
                for start in range(0, full_end, self.batch_size)
            )
            tail.extend(rows[full_end:])
        # At most one short remainder per assigned shard enters this pool.  It
        # prevents needless sample loss while keeping every interior batch on
        # one physical primary shard.
        rng.shuffle(tail)
        full_tail_end = (len(tail) // self.batch_size) * self.batch_size
        batches.extend(
            tail[start:start + self.batch_size]
            for start in range(0, full_tail_end, self.batch_size)
        )
        return batches

    def _epoch_candidate_counts(self) -> tuple[list[list[int]], list[int]]:
        assignments = self._epoch_assignments()
        counts = [
            sum(len(self.groups[shard_id]) for shard_id in shards) // self.batch_size
            for shards in assignments
        ]
        return assignments, counts

    def _epoch_batch_limit(self) -> int:
        _, counts = self._epoch_candidate_counts()
        return min(counts)

    def __iter__(self):
        assignments, counts = self._epoch_candidate_counts()
        batch_limit = min(counts)
        batches = self._candidate_batches(assignments[self.rank], rank=self.rank)
        yield from batches[:batch_limit]

    def audit(self) -> dict[str, Any]:
        counts = [len(self.groups[index]) for index in sorted(self.groups)]
        assignments, candidate_counts = self._epoch_candidate_counts()
        batch_limit = min(candidate_counts)
        return {
            "kind": "rank_owned_shard_streaming",
            "num_shards": len(counts),
            "shards_per_rank": self.shards_per_rank,
            "epoch": self.epoch,
            "epoch_shuffle": "global shard order, disjoint equal shard assignment per rank, then rank-local shard and row shuffle",
            "rank_distribution": "each rank owns exactly num_shards/world_size primary shards for the epoch",
            "assigned_shards_by_rank": assignments,
            "assigned_shards_this_rank": assignments[self.rank],
            "candidate_batches_by_rank": candidate_counts,
            "qa_records_per_primary_shard": {
                "minimum": min(counts),
                "maximum": max(counts),
                "total": sum(counts),
            },
            "secondary_cross_shard_records": self.secondary_cross_shard_records,
            "batches_per_rank_after_equalization": batch_limit,
            "dropped_samples_per_epoch": len(self.dataset) - batch_limit * self.num_replicas * self.batch_size,
        }


def _append_terminal_endoftext(
    answer_rows: list[list[int]],
    tokenizer: Any,
    *,
    max_answer_tokens: int,
) -> list[list[int]]:
    """Terminate every answer with exactly one supervised SmolLM2 EOS.

    The configured answer budget includes the terminal token.  Tokenization
    therefore reserves one position for ``<|endoftext|>`` instead of silently
    growing a 250-token answer to 251 tokens.
    """
    if int(max_answer_tokens) < 2:
        raise ValueError("max_answer_tokens must leave room for answer content and terminal EOS")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id for answer termination")
    eos_token_id = int(eos_token_id)
    try:
        literal_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    except Exception as exc:
        raise ValueError("tokenizer cannot resolve required <|endoftext|> token") from exc
    if literal_id is None or int(literal_id) != eos_token_id:
        raise ValueError(
            "tokenizer EOS contract mismatch: <|endoftext|> must equal eos_token_id, "
            f"literal={literal_id!r} eos={eos_token_id}"
        )
    terminated: list[list[int]] = []
    content_limit = int(max_answer_tokens) - 1
    for row in answer_rows:
        content = [int(token) for token in row[:content_limit]]
        while content and content[-1] == eos_token_id:
            content.pop()
        result = [*content, eos_token_id]
        if len(result) > int(max_answer_tokens) or result[-1] != eos_token_id:
            raise AssertionError("answer terminal EOS construction failed")
        terminated.append(result)
    return terminated


def collate_reasonaqa(
    items: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_prompt_tokens: int = 129,
    max_answer_tokens: int = 250,
    timing_accumulator: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not items:
        raise ValueError("empty batch")
    prompts = [item["prompt"] for item in items]
    answers = [item["answer"] for item in items]
    # Tokenize each field without padding first.  The important contract here
    # is that every sample is ``real_prompt + real_answer``; padding is added
    # only after that concatenation, across the complete text sequence in the
    # current batch.  Padding prompt and answer independently would insert
    # artificial pad tokens between the prompt and answer and would break the
    # causal next-token relationship at the answer boundary.
    phase_started = time.perf_counter()
    phase_cpu_started = time.thread_time()
    prompt = tokenizer(prompts, max_length=max_prompt_tokens, truncation=True, padding=False, return_tensors=None, add_special_tokens=True)
    if int(max_answer_tokens) < 2:
        raise ValueError("max_answer_tokens must be at least 2")
    answer = tokenizer(answers, max_length=max_answer_tokens - 1, truncation=True, padding=False, return_tensors=None, add_special_tokens=False)
    if timing_accumulator is not None:
        timing_accumulator["tokenize"] = timing_accumulator.get("tokenize", 0.0) + time.perf_counter() - phase_started
        timing_accumulator["tokenize_thread_cpu"] = timing_accumulator.get("tokenize_thread_cpu", 0.0) + time.thread_time() - phase_cpu_started

    phase_started = time.perf_counter()
    phase_cpu_started = time.thread_time()
    def _rows(value: Any) -> list[list[int]]:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return [list(row) for row in value]

    prompt_rows = _rows(prompt["input_ids"])
    answer_rows = _append_terminal_endoftext(
        _rows(answer["input_ids"]), tokenizer, max_answer_tokens=max_answer_tokens
    )
    if len(prompt_rows) != len(items) or len(answer_rows) != len(items):
        raise ValueError("tokenizer returned a batch with the wrong number of rows")
    prompt_lengths = [len(row) for row in prompt_rows]
    answer_lengths = [len(row) for row in answer_rows]
    eos_token_id = int(tokenizer.eos_token_id)
    if not answer_rows or any(not row or row[-1] != eos_token_id for row in answer_rows):
        raise AssertionError("every answer must end in supervised <|endoftext|>")
    text_rows = [p_row + a_row for p_row, a_row in zip(prompt_rows, answer_rows)]
    text_lengths = [len(row) for row in text_rows]
    max_text_length = max(text_lengths, default=0)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    text_ids = torch.full((len(items), max_text_length), int(pad_id), dtype=torch.long)
    text_attention_mask = torch.zeros((len(items), max_text_length), dtype=torch.long)
    for row_index, row in enumerate(text_rows):
        if row:
            text_ids[row_index, :len(row)] = torch.tensor(row, dtype=torch.long)
            text_attention_mask[row_index, :len(row)] = 1

    # This answer-only mask is retained for token accounting/logging.  The
    # model's actual labels are built from the exact prompt/answer boundaries
    # above, so this auxiliary mask can never introduce a loss-bearing pad.
    max_answer_length = max(answer_lengths, default=0)
    answer_attention_mask = torch.zeros((len(items), max_answer_length), dtype=torch.long)
    for row_index, length in enumerate(answer_lengths):
        answer_attention_mask[row_index, :length] = 1
    if timing_accumulator is not None:
        timing_accumulator["text_tensor_build"] = timing_accumulator.get("text_tensor_build", 0.0) + time.perf_counter() - phase_started
        timing_accumulator["text_tensor_build_thread_cpu"] = timing_accumulator.get("text_tensor_build_thread_cpu", 0.0) + time.thread_time() - phase_cpu_started

    phase_started = time.perf_counter()
    phase_cpu_started = time.thread_time()
    audio1 = torch.stack([item["audio1"] for item in items])
    single_slot_mask = torch.tensor([bool(item.get("single_audio_slot", item["audio2"] is None)) for item in items], dtype=torch.bool)
    reused_mask = torch.tensor([bool(item.get("audio2_reused", item["audio2"] is None)) for item in items], dtype=torch.bool)
    reused = bool(reused_mask.all())
    audio2 = None if reused else torch.stack([item["audio1"] if item["audio2"] is None else item["audio2"] for item in items])
    if timing_accumulator is not None:
        timing_accumulator["waveform_stack"] = timing_accumulator.get("waveform_stack", 0.0) + time.perf_counter() - phase_started
        timing_accumulator["waveform_stack_thread_cpu"] = timing_accumulator.get("waveform_stack_thread_cpu", 0.0) + time.thread_time() - phase_cpu_started

    phase_started = time.perf_counter()
    phase_cpu_started = time.thread_time()
    batch = {
        "audio1": audio1,
        "audio2": audio2,
        "audio2_reused_mask": reused_mask,
        "single_audio_slot_mask": single_slot_mask,
        "text_ids": text_ids,
        "text_attention_mask": text_attention_mask,
        "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        "answer_lengths": torch.tensor(answer_lengths, dtype=torch.long),
        "answer_attention_mask": answer_attention_mask,
        "row_indices": [item["row_index"] for item in items],
        "audio2_reused": reused,
    }
    if all("waveform_cache_shard_ids" in item for item in items):
        batch["waveform_cache_shard_ids"] = [tuple(item["waveform_cache_shard_ids"]) for item in items]
    if timing_accumulator is not None:
        timing_accumulator["batch_metadata"] = timing_accumulator.get("batch_metadata", 0.0) + time.perf_counter() - phase_started
        timing_accumulator["batch_metadata_thread_cpu"] = timing_accumulator.get("batch_metadata_thread_cpu", 0.0) + time.thread_time() - phase_cpu_started
    return batch
