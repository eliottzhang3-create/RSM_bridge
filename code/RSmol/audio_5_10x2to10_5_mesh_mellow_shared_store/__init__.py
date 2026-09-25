"""Isolated shared-store route for uniformly sampled 5-10xT-5 Audio MeSH."""

TRAINING_CONTRACT = (
    "node_shared_unique_store_fullshuffle_fixed260_audio_reuse_"
    "uniform_recursive_depth_2_10_answer_eos_v1"
)
DEFAULT_EPOCHS = 7
DEFAULT_WARMUP_RATIO = 0.05
RECURSIVE_DEPTHS = tuple(range(2, 11))

__all__ = [
    "DEFAULT_EPOCHS", "DEFAULT_WARMUP_RATIO", "RECURSIVE_DEPTHS", "TRAINING_CONTRACT",
]
