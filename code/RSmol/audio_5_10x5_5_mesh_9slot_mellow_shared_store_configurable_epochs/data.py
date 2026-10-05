"""Data facade for the isolated x5/9-slot fixed-260 runtime-zero-slot route.

The canonical ReasonAQA manifest and unique waveform store are shared with the
audited shared-store route. Structurally single-audio rows are reclassified so the
model creates one exact zero waveform for their second slot at runtime.
Explicit dual-audio rows continue to use both real waveforms; identical-path
dual rows may reuse the first HTSAT embedding.
"""
from __future__ import annotations

from typing import Any

import torch

from audio_5_10x2_5_mesh_mellow_shared_store.data import (
    ReasonAQADataset,
    UniqueWaveformStore,
    collate_reasonaqa as _collate_reasonaqa,
)


def collate_reasonaqa(
    items: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_prompt_tokens: int = 129,
    max_answer_tokens: int = 250,
    timing_accumulator: dict[str, float] | None = None,
) -> dict[str, Any]:
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
        raise AssertionError("runtime-zero and same-real-audio masks overlap")
    batch["silence_second_slot_mask"] = structural_single
    batch["same_real_audio_mask"] = same_real_audio
    batch["audio2_reused"] = bool(same_real_audio.all())
    return batch


__all__ = ["ReasonAQADataset", "UniqueWaveformStore", "collate_reasonaqa"]

