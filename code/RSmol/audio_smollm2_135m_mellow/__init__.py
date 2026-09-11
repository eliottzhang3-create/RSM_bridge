"""Original SmolLM2-135M audio baseline route.

This package is intentionally separate from the recursive text-model audio
routes.  It reuses the already audited ReasonAQA/Mellow audio semantics while
loading the standard Hugging Face ``LlamaForCausalLM`` implementation.
"""

from .model import (
    AUDIO_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    ORIGINAL_SMOLLM2_CONTRACT,
    MAPPER_CONTRACT,
    AudioSmolLM2Config,
    AudioSmolLM2Model,
    validate_original_smollm2,
)

__all__ = [
    "AUDIO_PREFIX_TOKENS",
    "AUDIO_TOKENS_PER_CLIP",
    "ORIGINAL_SMOLLM2_CONTRACT",
    "MAPPER_CONTRACT",
    "AudioSmolLM2Config",
    "AudioSmolLM2Model",
    "validate_original_smollm2",
]
