"""Reuse exactly the original model/capture/patch/probability implementation."""
from mizar_reasonaqa_tracing.runtime import (load, forward, controls, from_logits,
    validate_decision, decision, silent_audio, corrupt_prefix, autocast, prefix)
