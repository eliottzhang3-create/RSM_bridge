"""Shared ReasonAQA audio/data implementation for the original baseline.

The implementation is imported from the current, audited audio data route so
that waveform normalization, audio2 reuse, tokenization, and answer-only
padding semantics remain literally identical.  The baseline's model and
checkpoint namespace are independent; the data contract is intentionally not
forked.
"""

from audio_5_10x2_5_mesh_mellow.data import (  # noqa: F401
    ReasonAQADataset,
    collate_reasonaqa,
    load_waveform,
)

__all__ = ["ReasonAQADataset", "collate_reasonaqa", "load_waveform"]
