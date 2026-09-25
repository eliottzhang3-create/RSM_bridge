"""Audio wrapper for the isolated variable-recursive-depth MeSH route."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from audio_5_10x2_5_mesh_mellow.model import (
    AUDIO_DUAL_PREFIX_TOKENS, AUDIO_PREFIX_TOKENS, AUDIO_SINGLE_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP, MAPPER_CONTRACT, MESH_HIDDEN_SIZE,
    AudioBridge, AudioMeshConfig as FixedAudioMeshConfig,
    AudioMeshModel as FixedAudioMeshModel, build_labels, manifest_sha256, write_config,
)
from recursive_model_5_10x2to10_5_mesh import (
    MAX_RECURSIVE_DEPTH, MIN_RECURSIVE_DEPTH, MODEL_ARCHITECTURE_CONTRACT,
    RecursiveLlamaForCausalLM, validate_recursive_depth,
)

ARCHITECTURE_CONTRACT = MODEL_ARCHITECTURE_CONTRACT + "_audio_mellow"


@dataclass
class AudioMeshConfig(FixedAudioMeshConfig):
    architecture_contract: str = ARCHITECTURE_CONTRACT
    recursive_min_depth: int = MIN_RECURSIVE_DEPTH
    recursive_max_depth: int = MAX_RECURSIVE_DEPTH


class AudioMeshModel(FixedAudioMeshModel):
    """Preserve fixed260 audio semantics while selecting one depth per batch."""

    def __init__(self, mesh_model: Any, tokenizer: Any, htsat_wrapper: Any,
                 htsat_backbone: Any, config: AudioMeshConfig | None = None) -> None:
        config = config or AudioMeshConfig(projection_dim=int(mesh_model.config.hidden_size))
        if config.architecture_contract != ARCHITECTURE_CONTRACT:
            raise ValueError(
                f"variable-depth audio architecture contract mismatch: {config.architecture_contract}"
            )
        super().__init__(mesh_model, tokenizer, htsat_wrapper, htsat_backbone, config)
        self.last_recursive_depth: int | None = None

    def forward(self, *, recursive_depth: int, audio1: torch.Tensor,
                audio2: torch.Tensor | None, text_ids: torch.Tensor,
                text_attention_mask: torch.Tensor, prompt_lengths: torch.Tensor,
                answer_lengths: torch.Tensor, answer_attention_mask: torch.Tensor | None = None,
                audio2_reused_mask: torch.Tensor | None = None,
                single_audio_slot_mask: torch.Tensor | None = None) -> Any:
        depth = validate_recursive_depth(recursive_depth)
        if not hasattr(self.mesh_model, "set_recursive_depth"):
            raise TypeError("mesh_model does not implement the variable-depth contract")
        self.mesh_model.set_recursive_depth(depth)
        self.last_recursive_depth = depth
        return super().forward(
            audio1=audio1, audio2=audio2, text_ids=text_ids,
            text_attention_mask=text_attention_mask, prompt_lengths=prompt_lengths,
            answer_lengths=answer_lengths, answer_attention_mask=answer_attention_mask,
            audio2_reused_mask=audio2_reused_mask,
            single_audio_slot_mask=single_audio_slot_mask,
        )


__all__ = [
    "ARCHITECTURE_CONTRACT", "MAPPER_CONTRACT", "AUDIO_PREFIX_TOKENS",
    "AUDIO_SINGLE_PREFIX_TOKENS", "AUDIO_DUAL_PREFIX_TOKENS",
    "AUDIO_TOKENS_PER_CLIP", "MESH_HIDDEN_SIZE", "AudioBridge",
    "AudioMeshConfig", "AudioMeshModel", "RecursiveLlamaForCausalLM",
    "build_labels", "manifest_sha256", "write_config",
]
