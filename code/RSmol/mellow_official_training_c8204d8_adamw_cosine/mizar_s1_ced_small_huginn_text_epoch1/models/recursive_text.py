"""Strict Huginn HF weights-only loader; preserve the trained recurrent adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from contract import (TEXT_MODEL_SCHEDULE, validate_recursive_config_values,
                      validate_text_initialization)


def _import_recursive_model() -> tuple[Any, Any]:
    """Use the existing Huginn implementation, independent of AutoModel registration."""
    rsmol_root = Path(__file__).resolve().parents[3]
    if str(rsmol_root) not in sys.path:
        sys.path.insert(0, str(rsmol_root))
    from models.huginn import (
        MizarHuginnForCausalLM,
        parameter_audit,
    )
    return MizarHuginnForCausalLM, parameter_audit


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
        "adapter_parameters": 663552,
        "adapter_calls_per_forward": 2,
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
    adapter = getattr(model.model, "recurrent_adapter", None)
    if (adapter is None or adapter.bias is not None
            or tuple(adapter.weight.shape) != (576, 1152) or not adapter.weight.requires_grad):
        raise ValueError("Huginn requires its trained, trainable shared Linear(1152,576,bias=False)")
    if model.get_input_embeddings().weight is not model.lm_head.weight:
        raise ValueError("Huginn embedding/lm_head must remain tied")
    for name, _ in (*model.named_parameters(), *model.named_buffers()):
        if any(part.startswith(("router", "mesh_", "memory_slot"))
               for part in name.split(".")):
            raise ValueError(f"non-MeSH model contains MeSH parameter/buffer: {name}")


def load_recursive_text_model(model_path: str | Path) -> Any:
    """Load only pretrained LM weights; the audio trainer owns all other state."""
    path = Path(model_path).expanduser().resolve(strict=True)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"non-MeSH text checkpoint directory/config.json missing: {path}")
    validate_text_initialization(path)
    model_cls, parameter_audit = _import_recursive_model()
    model, loading_info = model_cls.from_pretrained(
        path, local_files_only=True, output_loading_info=True,
    )
    failures = {key: loading_info.get(key) for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if loading_info.get(key)}
    if failures:
        raise ValueError(f"Huginn text checkpoint weights do not match the model: {failures}")
    validate_loaded_model(model, parameter_audit(model))
    return model


__all__ = [
    "load_recursive_text_model",
    "validate_recursive_config",
    "validate_loaded_model",
]
