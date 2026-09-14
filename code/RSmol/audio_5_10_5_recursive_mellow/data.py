"""Shared ReasonAQA data contract for the fixed 5-10-5 audio route.

Importing the audited implementation keeps waveform normalization, duplicate
audio handling, tokenization, dynamic padding, and answer boundaries exactly
aligned with the existing MeSH and original SmolLM2 audio experiments.
"""

from audio_5_10x2_5_mesh_mellow.data import (  # noqa: F401
    ReasonAQADataset,
    collate_reasonaqa,
    load_waveform,
)

__all__ = ["ReasonAQADataset", "collate_reasonaqa", "load_waveform"]

