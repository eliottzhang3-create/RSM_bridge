"""Mellow training data semantics over a variable-length shared waveform store."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .mellow_templates import BOTH, DETAILONLY, EMOTION, FIRST, LONGLINEONLY, SECOND, WORDONLY

MELLOW_REFERENCE_COMMIT = "c8204d8eb99b4384fd7a76ad57995731e0c0c2bf"
MELLOW_TEMPLATE_BLOB_SHA = "4d3722d904734c7b2ae1b55309002c82bf1d11bc"
MELLOW_VARIABLE_STORE_FORMAT = "mellow_faithful_variable_waveform_store_v2"
SAMPLE_RATE = 32_000
SEGMENT_SAMPLES = 320_000
PROMPT_TOKENS = 129
ANSWER_TOKENS = 250


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_path(value: str | Path) -> str:
    return os.path.normpath(os.path.expanduser(str(value))).replace("\\", "/")


def _path_value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_path_value_present(item) for item in value)
    return bool(str(value).strip())


def normalized_audio2_is_missing(row: dict[str, Any]) -> bool:
    """Recover an originally empty filepath2 from a normalized manifest.

    Stage-1 manifests materialize an empty filepath2 as audio2_path=audio1_path
    for legacy routes, while retaining explicit provenance fields.  Public
    Mellow instead samples a random filepath1 clip for that missing slot, so
    this route must consult the provenance before reading audio2_path.
    """
    if row.get("audio2_reused") is True:
        return True
    if str(row.get("audio2_source") or "") == "filepath1_duplicate":
        return True
    if "filepath2_raw" in row and not _path_value_present(row.get("filepath2_raw")):
        return True
    return False


def row_audio_path(row: dict[str, Any], first: bool) -> str:
    if not first and normalized_audio2_is_missing(row):
        return ""
    keys = ("audio1_path", "filepath1") if first else ("audio2_path", "filepath2")
    for key in keys:
        value = row.get(key)
        if _path_value_present(value):
            return str(value)
    return ""


class UniqueWaveformStore:
    """Variable-length mono 32 kHz float32 waveforms in one immutable mmap."""

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir).expanduser().resolve(strict=True)
        if (self.store_dir / "BUILDING").exists():
            raise RuntimeError(f"store is still BUILDING: {self.store_dir}")
        required = ("metadata.json", "index.jsonl", "waveforms.f32")
        missing = [name for name in required if not (self.store_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"store lacks {missing}: {self.store_dir}")
        self.metadata = json.loads((self.store_dir / "metadata.json").read_text(encoding="utf-8"))
        expected = {
            "status": "PASS",
            "format": MELLOW_VARIABLE_STORE_FORMAT,
            "sample_rate": SAMPLE_RATE,
            "dtype": "float32",
            "byte_order": "little",
            "data_file": "waveforms.f32",
            "mellow_reference_commit": MELLOW_REFERENCE_COMMIT,
        }
        mismatch = {
            key: (value, self.metadata.get(key))
            for key, value in expected.items()
            if self.metadata.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"store contract mismatch: {mismatch}")
        if sha256_file(self.store_dir / "index.jsonl") != self.metadata.get("index_sha256"):
            raise RuntimeError("store index SHA256 mismatch")
        data_path = self.store_dir / "waveforms.f32"
        if data_path.stat().st_size != int(self.metadata.get("total_waveform_bytes", -1)):
            raise RuntimeError("store waveform byte size mismatch")

        self.entries: list[dict[str, Any]] = []
        self.alias_to_id: dict[str, int] = {}
        expected_offset = 0
        with (self.store_dir / "index.jsonl").open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                audio_id = len(self.entries)
                samples = int(entry.get("num_samples", -1))
                offset = int(entry.get("byte_offset", -1))
                length = int(entry.get("byte_length", -1))
                if int(entry.get("audio_id", -1)) != audio_id or samples <= 0:
                    raise RuntimeError(f"invalid index entry at line {line_number}")
                if offset != expected_offset or length != samples * 4:
                    raise RuntimeError(f"invalid contiguous waveform span at line {line_number}")
                expected_offset += length
                self.entries.append(entry)
                for alias in [entry.get("source_path", ""), *entry.get("manifest_aliases", [])]:
                    if not alias:
                        continue
                    key = normalize_path(alias)
                    previous = self.alias_to_id.setdefault(key, audio_id)
                    if previous != audio_id:
                        raise RuntimeError(f"alias collision: {alias}")
        if len(self.entries) != int(self.metadata.get("num_unique_audio_files", -1)):
            raise RuntimeError("store index cardinality mismatch")
        if expected_offset != data_path.stat().st_size:
            raise RuntimeError("store index does not cover the complete waveform file")
        self._memmap = np.memmap(data_path, mode="r", dtype="<f4")

    def locate(self, source_path: str | Path) -> int:
        audio_id = self.alias_to_id.get(normalize_path(source_path))
        if audio_id is None:
            try:
                audio_id = self.alias_to_id.get(normalize_path(Path(source_path).expanduser().resolve()))
            except OSError:
                audio_id = None
        if audio_id is None:
            raise KeyError(f"audio path absent from store: {source_path}")
        return int(audio_id)

    def load_id(self, audio_id: int) -> torch.Tensor:
        entry = self.entries[int(audio_id)]
        start = int(entry["byte_offset"]) // 4
        count = int(entry["num_samples"])
        array = np.asarray(self._memmap[start:start + count], dtype=np.float32)
        if array.size != count:
            raise RuntimeError(f"truncated waveform: audio_id={audio_id}")
        return torch.from_numpy(array.copy()).unsqueeze(0)

    def load(self, source_path: str | Path) -> tuple[int, torch.Tensor]:
        audio_id = self.locate(source_path)
        return audio_id, self.load_id(audio_id)


class ReasonAQADataset(Dataset[dict[str, Any]]):
    """Public Mellow call order using the process-wide Python random state."""

    def __init__(self, manifest: str | Path, tokenizer: Any, *, unique_waveform_store_dir: str | Path) -> None:
        self.manifest = Path(manifest).expanduser().resolve(strict=True)
        self.rows = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError(f"empty manifest: {self.manifest}")
        self.tokenizer = tokenizer
        self.store = UniqueWaveformStore(unique_waveform_store_dir)
        manifest_sha = sha256_file(self.manifest)
        if manifest_sha != self.store.metadata.get("manifest_sha256"):
            raise RuntimeError(
                f"store manifest mismatch: manifest={manifest_sha} "
                f"store={self.store.metadata.get('manifest_sha256')}"
            )
        # Public Mellow uses list(set(nonempty filepath1)). Sorting only removes
        # historical PYTHONHASHSEED dependence; the uniform distribution is unchanged.
        pool_paths = sorted({row_audio_path(row, True) for row in self.rows if row_audio_path(row, True)})
        if not pool_paths:
            raise RuntimeError("manifest has no non-empty filepath1 random-audio pool")
        self.random_audio_pool_paths = pool_paths
        self.random_audio_pool_ids = [self.store.locate(path) for path in pool_paths]

    def __len__(self) -> int:
        return len(self.rows)

    def audio_structure(self, index: int) -> tuple[bool, bool]:
        row = self.rows[int(index)]
        first, second = row_audio_path(row, True), row_audio_path(row, False)
        return not bool(second), bool(first and second and normalize_path(first) == normalize_path(second))

    @staticmethod
    def _crop_or_pad(waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        samples = int(waveform.shape[-1])
        if samples > SEGMENT_SAMPLES:
            offset = random.randint(0, samples - SEGMENT_SAMPLES)
            result = waveform[:, offset:offset + SEGMENT_SAMPLES]
        else:
            offset = 0
            result = F.pad(waveform, (0, SEGMENT_SAMPLES - samples))
        if tuple(result.shape) != (1, SEGMENT_SAMPLES):
            raise RuntimeError(f"invalid crop/pad shape: {tuple(result.shape)}")
        return result.contiguous(), offset

    @staticmethod
    def _answer_prompt(row: dict[str, Any]) -> tuple[str, str, str]:
        prompt = str(row.get("input") or row.get("prompt") or row.get("question") or "")
        group = "generic"
        if prompt == "caption both audios":
            prompt, group = random.choice(BOTH), "BOTH"
            answer = (
                "The audio 1 is " + str(row.get("caption1", "")).lower()
                + ". The audio 2 is " + str(row.get("caption2", "")).lower() + "."
            ).replace("..", ".")
        elif prompt == "caption first audio":
            prompt, group = random.choice(FIRST), "FIRST"
            answer = ("The audio 1 is " + str(row.get("caption1", "")).lower() + ".").replace("..", ".")
        elif prompt == "caption second audio":
            prompt, group = random.choice(SECOND), "SECOND"
            answer = ("The audio 2 is " + str(row.get("caption2", "")).lower() + ".").replace("..", ".")
        elif prompt == "explain the difference in few words":
            prompt, answer, group = random.choice(WORDONLY), str(row.get("answer", "")), "WORDONLY"
        elif prompt == "explain the difference in a sentence":
            prompt, answer, group = random.choice(LONGLINEONLY), str(row.get("answer", "")), "LONGLINEONLY"
        elif prompt == "explain the difference in detail":
            prompt, answer, group = random.choice(DETAILONLY), str(row.get("answer", "")), "DETAILONLY"
        elif "emo_emo_emo" in prompt:
            prompt = prompt.replace("emo_emo_emo", random.choice(EMOTION))
            answer, group = str(row.get("answer", "")), "EMOTION"
        else:
            prompt = prompt.lower()
            answer = str(row.get("answer") or row.get("target") or row.get("output") or "").lower()
        if not prompt or not answer:
            raise ValueError("Mellow row lacks prompt or answer")
        return answer, prompt, group

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        path1, path2 = row_audio_path(row, True), row_audio_path(row, False)
        random1, random2 = not bool(path1), not bool(path2)
        if random1:
            id1 = random.choice(self.random_audio_pool_ids)
            full1 = self.store.load_id(id1)
        else:
            id1, full1 = self.store.load(path1)
        if random2:
            id2 = random.choice(self.random_audio_pool_ids)
            full2 = self.store.load_id(id2)
        else:
            id2, full2 = self.store.load(path2)
        audio1, offset1 = self._crop_or_pad(full1)
        audio2, offset2 = self._crop_or_pad(full2)
        answer, prompt, template = self._answer_prompt(row)
        return {
            "audio1": audio1,
            "audio2": audio2,
            "prompt": prompt,
            "answer": answer,
            "row_index": int(index),
            "single_audio_slot": random2,
            "audio1_random": random1,
            "audio2_random": random2,
            "audio1_id": int(id1),
            "audio2_id": int(id2),
            "audio1_crop_offset": int(offset1),
            "audio2_crop_offset": int(offset2),
            "audio1_source_samples": int(full1.shape[-1]),
            "audio2_source_samples": int(full2.shape[-1]),
            "template_group": template,
        }


def _tokenize_fixed(tokenizer: Any, values: list[str], max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        [value + " <|endoftext|>" for value in values],
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    if tuple(encoded["input_ids"].shape) != (len(values), max_length):
        raise RuntimeError("fixed tokenization shape mismatch")
    return encoded


def collate_reasonaqa(items: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    if not items or tokenizer is None:
        raise ValueError("collator requires items and tokenizer")
    prompt = _tokenize_fixed(tokenizer, [str(item["prompt"]) for item in items], PROMPT_TOKENS)
    answer = _tokenize_fixed(tokenizer, [str(item["answer"]) for item in items], ANSWER_TOKENS)
    return {
        "audio1": torch.stack([item["audio1"] for item in items]),
        "audio2": torch.stack([item["audio2"] for item in items]),
        "prompt_input_ids": prompt["input_ids"],
        "prompt_attention_mask": prompt["attention_mask"],
        "answer_input_ids": answer["input_ids"],
        "answer_attention_mask": answer["attention_mask"],
        "row_indices": [int(item["row_index"]) for item in items],
        "single_audio_slot_mask": torch.tensor([bool(item["single_audio_slot"]) for item in items]),
        "audio2_reused_mask": torch.zeros(len(items), dtype=torch.bool),
        "audio1_random_mask": torch.tensor([bool(item["audio1_random"]) for item in items]),
        "audio2_random_mask": torch.tensor([bool(item["audio2_random"]) for item in items]),
        "audio1_ids": torch.tensor([int(item["audio1_id"]) for item in items]),
        "audio2_ids": torch.tensor([int(item["audio2_id"]) for item in items]),
        "audio1_crop_offsets": torch.tensor([int(item["audio1_crop_offset"]) for item in items]),
        "audio2_crop_offsets": torch.tensor([int(item["audio2_crop_offset"]) for item in items]),
        "audio1_source_samples": torch.tensor([int(item["audio1_source_samples"]) for item in items]),
        "audio2_source_samples": torch.tensor([int(item["audio2_source_samples"]) for item in items]),
        "template_groups": [str(item["template_group"]) for item in items],
    }


__all__ = [
    "ANSWER_TOKENS",
    "MELLOW_REFERENCE_COMMIT",
    "MELLOW_TEMPLATE_BLOB_SHA",
    "MELLOW_VARIABLE_STORE_FORMAT",
    "PROMPT_TOKENS",
    "ReasonAQADataset",
    "SAMPLE_RATE",
    "SEGMENT_SAMPLES",
    "UniqueWaveformStore",
    "collate_reasonaqa",
    "normalize_path",
    "normalized_audio2_is_missing",
    "row_audio_path",
    "sha256_file",
]
