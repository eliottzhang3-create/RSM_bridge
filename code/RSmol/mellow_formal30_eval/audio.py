"""Audio decoding helpers for the isolated formal30 evaluator.

The formal30 training dataset resamples multi-channel waveforms and then
averages the first two channels before cropping.  The historical MMAU
helper flattened channels, which is a different input contract.  Keep this
adapter local to the unified evaluator so older evaluation routes are not
changed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as common


def decode_training_audio(
    payload: Any,
    *,
    base_dir: Path,
    source_rate: int = common.DEFAULT_SOURCE_SAMPLE_RATE,
    target_rate: int = common.DEFAULT_SAMPLE_RATE,
) -> tuple[Any, dict[str, Any]]:
    """Decode and resample one MMAU payload with the training channel rule.

    Training's ``AudioTextDataset._cut_or_randomcrop`` keeps mono input as-is
    and, for multi-channel input, averages channels 0 and 1.  It does not
    flatten channels into one long waveform.
    """

    import torch

    waveform, actual_rate, source = common._decode_audio_payload(
        payload,
        base_dir=base_dir,
        default_rate=source_rate,
    )
    if actual_rate <= 0:
        raise RuntimeError(f"invalid source sampling rate: {actual_rate}")
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise RuntimeError(f"decoded waveform must be [channels,samples], got {tuple(waveform.shape)}")
    waveform = waveform.float()
    if not bool(torch.isfinite(waveform).all()):
        raise RuntimeError("decoded waveform contains non-finite values")
    original_channels = int(waveform.shape[0])
    original_samples = int(waveform.shape[-1])
    original_duration = original_samples / float(actual_rate)
    # Match the dataset order: resample channel-wise first, then apply the
    # channel reduction inside ``_cut_or_randomcrop``.
    waveform = common._resample_waveform(waveform, actual_rate, target_rate).contiguous()
    resampled_samples = int(waveform.shape[-1])
    if original_channels > 1:
        if original_channels < 2:
            raise RuntimeError("multi-channel waveform has fewer than two channels")
        waveform = ((waveform[0] + waveform[1]) / 2.0).unsqueeze(0)
        channel_policy = "training_first_two_channel_mean"
    else:
        channel_policy = "training_mono_unchanged"
    if waveform.shape[-1] <= 0 or not bool(torch.isfinite(waveform).all()):
        raise RuntimeError("resampled waveform is empty or non-finite")
    return waveform, {
        "audio_source": source,
        "audio_original_sample_rate": int(actual_rate),
        "audio_original_num_samples": original_samples,
        "audio_original_channels": original_channels,
        "audio_original_duration_seconds": original_duration,
        "audio_resampled": bool(actual_rate != target_rate),
        "audio_resampled_num_samples": resampled_samples,
        "audio_target_sample_rate": int(target_rate),
        "audio_target_duration_seconds": float(common.DEFAULT_AUDIO_SECONDS),
        "audio_channel_policy": channel_policy,
        "audio_segment_policy": "mellow_repeat_or_per_id_seeded_random_crop",
    }
