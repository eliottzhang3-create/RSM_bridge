"""Explicit loader for the fixed 5+10x2+5 non-MeSH text checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from contract import TEXT_MODEL_SCHEDULE, validate_recursive_config_values


def _import_recursive_model() -> tuple[Any, Any]:
    """Use the non-MeSH implementation explicitly, independent of AutoModel registration."""
    rsmol_root = Path(__file__).resolve().parents[3]
    if str(rsmol_root) not in sys.path:
        sys.path.insert(0, str(rsmol_root))
    from models.recursive_reinject import (
        RecursiveLlamaForCausalLM,
        parameter_audit,
    )
    return RecursiveLlamaForCausalLM, parameter_audit


def validate_recursive_config(config: Any) -> None:
    validate_recursive_config_values(config.to_dict())


def validate_loaded_model(model: Any, audit: dict) -> None:
    """Check real physical modules as well as the declared execution schedule."""
    validate_recursive_config(model.config)
    expected = {
        "logical_layer_count": 30,
        "physical_layer_count": 20,
        "recursive_layer_count": 20,
        "recursive_loops": 2,
        "middle_recurrent_count": 10,
        "model_physical_module_count": 20,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError(f"loaded non-MeSH parameter audit mismatch: {audit}")
    if tuple(audit.get("schedule", ())) != TEXT_MODEL_SCHEDULE:
        raise ValueError(f"loaded non-MeSH parameter schedule mismatch: {audit}")
    layers = model.model.layers
    if len(layers) != 20 or len({id(layer) for layer in layers}) != 20:
        raise ValueError("non-MeSH text model requires 20 independent physical layers")
    if tuple(model.model.logical_to_physical) != TEXT_MODEL_SCHEDULE:
        raise ValueError("loaded non-MeSH execution schedule mismatch")
    for name, _ in (*model.named_parameters(), *model.named_buffers()):
        if any(part.startswith(("router", "mesh_", "memory_slot"))
               for part in name.split(".")):
            raise ValueError(f"non-MeSH model contains MeSH parameter/buffer: {name}")


def load_recursive_text_model(model_path: str | Path) -> Any:
    """Load only pretrained LM weights; the audio trainer owns all other state."""
    path = Path(model_path).expanduser().resolve(strict=True)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"non-MeSH text checkpoint directory/config.json missing: {path}")
    model_cls, parameter_audit = _import_recursive_model()
    model, loading_info = model_cls.from_pretrained(
        path, local_files_only=True, output_loading_info=True,
    )
    failures = {key: loading_info.get(key) for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if loading_info.get(key)}
    if failures:
        raise ValueError(f"non-MeSH text checkpoint weights do not match the model: {failures}")
    validate_loaded_model(model, parameter_audit(model))
    # No missing-key allowlist: the entire original backbone has loaded strictly.
    # Register the ONLY new state after that check, before optimizer/DDP creation.
    model.model.enable_audio_reinjection()
    model.generation_config.use_cache = False
    validate_audio_reinjection(model)
    return model


def validate_audio_reinjection(model: Any) -> dict:
    core = model.model
    width = model.config.hidden_size
    if (core.audio_reinject_tokens != 126 or tuple(core.audio_reinject_indices) != (5, 15)
            or tuple(core.audio_linear.weight.shape) != (width, width)
            or tuple(core.audio_linear.bias.shape) != (width,)
            or tuple(core.audio_gate.weight.shape) != (1, width)
            or tuple(core.audio_gate.bias.shape) != (1,)):
        raise ValueError("invalid shared audio reinjection geometry")
    names = {name for name, _ in model.named_parameters()
             if name.startswith(("model.audio_linear.", "model.audio_gate."))}
    if names != {"model.audio_linear.weight", "model.audio_linear.bias",
                 "model.audio_gate.weight", "model.audio_gate.bias"}:
        raise ValueError("expected exactly one shared audio linear and one shared scalar gate")
    return {"audio_tokens": 126, "logical_indices": [5, 15], "per_token_scalar": True,
            "parameter_count": width * width + 2 * width + 1,
            "cache": False, "shared_between_loops": True}


__all__ = [
    "load_recursive_text_model",
    "validate_recursive_config",
    "validate_loaded_model",
    "validate_audio_reinjection",
]
