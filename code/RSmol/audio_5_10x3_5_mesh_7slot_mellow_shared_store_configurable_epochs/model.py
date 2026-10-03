"""x3/7-slot MeSH audio model with a fixed 260-token runtime-zero slot."""
from __future__ import annotations

from dataclasses import dataclass

from audio_5_10x2_5_mesh_mellow.model import (
    AUDIO_DUAL_PREFIX_TOKENS,
    AUDIO_PREFIX_TOKENS,
    AUDIO_SINGLE_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    MAPPER_CONTRACT,
    MESH_HIDDEN_SIZE,
    _load_mellow_wrapper,
)
from audio_5_10x2_5_mesh_mellow_silence_slot.model import (
    AudioMeshSilenceSlotConfig,
    AudioMeshSilenceSlotModel,
)


ARCHITECTURE_CONTRACT = (
    "logical_40_physical_20_5_10x3_5_mesh_7slot_4router_audio_mellow_"
    "fixed260_runtime_zero_second_slot"
)
FIXED_PREFIX_TOKEN_CONTRACT = {
    "single": AUDIO_PREFIX_TOKENS,
    "dual": AUDIO_PREFIX_TOKENS,
}


@dataclass
class AudioMeshX3SevenSlotZeroConfig(AudioMeshSilenceSlotConfig):
    architecture_contract: str = ARCHITECTURE_CONTRACT
    compact_single_audio_prefix: bool = False


class AudioMeshX3SevenSlotZeroModel(AudioMeshSilenceSlotModel):
    """The audited mapper and zero-waveform slot wrapped around x3/7-slot MeSH."""


__all__ = [
    "ARCHITECTURE_CONTRACT",
    "FIXED_PREFIX_TOKEN_CONTRACT",
    "MAPPER_CONTRACT",
    "MESH_HIDDEN_SIZE",
    "AUDIO_PREFIX_TOKENS",
    "AUDIO_SINGLE_PREFIX_TOKENS",
    "AUDIO_DUAL_PREFIX_TOKENS",
    "AUDIO_TOKENS_PER_CLIP",
    "AudioMeshX3SevenSlotZeroConfig",
    "AudioMeshX3SevenSlotZeroModel",
    "_load_mellow_wrapper",
]
