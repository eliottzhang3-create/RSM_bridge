"""Data boundary for the isolated runtime-silence second-slot experiment.

The immutable six-partition stores and text/EOS contract are intentionally
shared with the audited MeSH route.  This module only splits the old
``audio2_reused`` signal into two unambiguous meanings:

* a structurally single-audio row receives a runtime silence second slot;
* an explicit dual-audio row whose paths match may reuse audio1's encoding.

No silence waveform is read from or written to the partition stores.
"""
from __future__ import annotations

from typing import Any

import torch

from audio_5_10x2_5_mesh_mellow.data import (
    ReasonAQADataset,
    UniqueWaveformStore,
    WaveformShardCache,
    collate_reasonaqa as _collate_reasonaqa,
    load_waveform,
)


def collate_reasonaqa(
    items: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_prompt_tokens: int = 129,
    max_answer_tokens: int = 250,
    timing_accumulator: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Collate without allocating or transferring per-row silence tensors."""

    batch = _collate_reasonaqa(
        items,
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_answer_tokens=max_answer_tokens,
        timing_accumulator=timing_accumulator,
    )
    structural_single = batch.pop("single_audio_slot_mask").to(dtype=torch.bool)
    same_waveform = batch.pop("audio2_reused_mask").to(dtype=torch.bool)
    same_real_audio = same_waveform & ~structural_single
    if bool((structural_single & same_real_audio).any()):
        raise AssertionError("silence-slot and same-real-audio masks overlap")
    batch["silence_second_slot_mask"] = structural_single
    batch["same_real_audio_mask"] = same_real_audio
    # Kept as report-only metadata; the trainer never forwards this key.
    batch["audio2_reused"] = bool(same_real_audio.all())
    return batch


__all__ = [
    "ReasonAQADataset",
    "UniqueWaveformStore",
    "WaveformShardCache",
    "collate_reasonaqa",
    "load_waveform",
]
