"""ReasonAQA manifest dataset for the isolated MeSH audio route."""
from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


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
    def __init__(self, manifest: str | Path, tokenizer: Any, *, max_prompt_tokens: int = 129, max_answer_tokens: int = 250, sample_rate: int = 32000, seconds: int = 10) -> None:
        self.manifest = Path(manifest)
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_answer_tokens = int(max_answer_tokens)
        self.sample_rate = int(sample_rate)
        self.seconds = int(seconds)
        self.rows = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"empty ReasonAQA manifest: {self.manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio1 = _path(row, True)
        audio2 = _path(row, False)
        audio2 = audio2 or audio1
        prompt = str(row.get("prompt") or row.get("question") or row.get("input") or "")
        answer = str(row.get("answer") or row.get("target") or row.get("output") or row.get("caption1") or "")
        if not audio1 or not answer:
            raise ValueError(f"manifest row {index} lacks audio1 or answer")
        audio1_waveform = load_waveform(audio1, sample_rate=self.sample_rate, seconds=self.seconds)
        audio2_waveform = None if audio2 == audio1 else load_waveform(audio2, sample_rate=self.sample_rate, seconds=self.seconds)
        return {"audio1": audio1_waveform, "audio2": audio2_waveform, "prompt": prompt, "answer": answer, "row_index": index, "audio2_reused": audio2 == audio1}


def collate_reasonaqa(items: list[dict[str, Any]], tokenizer: Any, *, max_prompt_tokens: int = 129, max_answer_tokens: int = 250) -> dict[str, Any]:
    if not items:
        raise ValueError("empty batch")
    prompts = [item["prompt"] for item in items]
    answers = [item["answer"] for item in items]
    # Pad only to the longest item in this batch. Per-field caps prevent
    # pathological records from exceeding the multimodal context budget.
    prompt = tokenizer(prompts, max_length=max_prompt_tokens, truncation=True, padding=True, return_tensors="pt", add_special_tokens=True)
    answer = tokenizer(answers, max_length=max_answer_tokens, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False)
    audio1 = torch.stack([item["audio1"] for item in items])
    reused_mask = torch.tensor([item["audio2"] is None for item in items], dtype=torch.bool)
    reused = bool(reused_mask.all())
    audio2 = None if reused else torch.stack([item["audio1"] if item["audio2"] is None else item["audio2"] for item in items])
    return {"audio1": audio1, "audio2": audio2, "audio2_reused_mask": reused_mask, "prompt_ids": prompt["input_ids"], "prompt_attention_mask": prompt.get("attention_mask", prompt["input_ids"].ne(int(tokenizer.pad_token_id))).long(), "answer_ids": answer["input_ids"], "answer_attention_mask": answer.get("attention_mask", answer["input_ids"].ne(int(tokenizer.pad_token_id))).long(), "row_indices": [item["row_index"] for item in items], "audio2_reused": reused}
