"""x5/8-slot MeSH audio model with a fixed 260-token runtime-zero slot."""
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
    "logical_60_physical_20_5_10x5_5_mesh_8slot_6router_audio_mellow_"
    "fixed260_runtime_zero_second_slot"
)
FIXED_PREFIX_TOKEN_CONTRACT = {
    "single": AUDIO_PREFIX_TOKENS,
    "dual": AUDIO_PREFIX_TOKENS,
}


@dataclass
class AudioMeshX5EightSlotZeroConfig(AudioMeshSilenceSlotConfig):
    architecture_contract: str = ARCHITECTURE_CONTRACT
    compact_single_audio_prefix: bool = False


class AudioMeshX5EightSlotZeroModel(AudioMeshSilenceSlotModel):
    """The audited mapper and zero-waveform slot wrapped around x5/8-slot MeSH."""


__all__ = [
    "ARCHITECTURE_CONTRACT",
    "FIXED_PREFIX_TOKEN_CONTRACT",
    "MAPPER_CONTRACT",
    "MESH_HIDDEN_SIZE",
    "AUDIO_PREFIX_TOKENS",
    "AUDIO_SINGLE_PREFIX_TOKENS",
    "AUDIO_DUAL_PREFIX_TOKENS",
    "AUDIO_TOKENS_PER_CLIP",
    "AudioMeshX5EightSlotZeroConfig",
    "AudioMeshX5EightSlotZeroModel",
    "_load_mellow_wrapper",
]

