"""Explicit loader for the fixed 5+10x3+5 non-MeSH text checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from contract import TEXT_MODEL_SCHEDULE, validate_recursive_config_values, recursive_checkpoint_contract


def validate_loop_lora(core):
    from models.loop_lora import validate_loop_lora as validate
    return validate(core)


def _import_recursive_model() -> tuple[Any, Any]:
    """Use the non-MeSH implementation explicitly, independent of AutoModel registration."""
    rsmol_root = Path(__file__).resolve().parents[3]
    if str(rsmol_root) not in sys.path:
        sys.path.insert(0, str(rsmol_root))
    from models.recursive_layer_inject import (
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
        "logical_layer_count": 40,
        "physical_layer_count": 20,
        "recursive_layer_count": 20,
        "recursive_loops": 3,
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
    recursive_checkpoint_contract(path)
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
    # Strictly load ALL inherited weights first, then register fresh gates.
    model.model.enable_layer_injection()
    # A separate RNG fork preserves the original input-route projection initial values.
    model.model.enable_audio_reinjection()
    model.model.enable_loop_lora()
    model.generation_config.use_cache = False
    validate_serial_injection(model)
    validate_loop_lora(model.model)
    return model


def validate_layer_injection(model: Any) -> dict:
    core = model.model
    width = model.config.hidden_size
    if (core.layer_inject_tokens != 126 or core.layer_inject_indices != (5, 15)
            or set(core.audio_gates) != {"4", "8"}
            or id(core.audio_gates["4"]) == id(core.audio_gates["8"])):
        raise ValueError("invalid independent CED layer4/8 injection geometry")
    for gate in core.audio_gates.values():
        if tuple(gate.weight.shape) != (1, width) or tuple(gate.bias.shape) != (1,):
            raise ValueError("each layer injection gate must be Linear(hidden_size,1)")
    return {"ced_layers": [4, 8], "logical_indices": [5, 15], "audio_tokens": 126,
            "gate": "tanh_per_audio_token_scalar", "gate_initialization": "zero_weight_and_bias",
            "shared_between_loops": False, "cache": False,
            "gate_parameter_count": 2 * (width + 1)}


def validate_audio_reinjection(model: Any) -> dict:
    core = model.model
    width = model.config.hidden_size
    if (core.audio_reinject_tokens != 126 or tuple(core.audio_reinject_indices) != (5, 15, 25)
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
    return {"audio_tokens": 126, "logical_indices": [5, 15, 25], "per_token_scalar": True,
            "parameter_count": width * width + 2 * width + 1,
            "cache": False, "shared_between_loops": True}


def validate_serial_injection(model: Any) -> dict:
    layer = validate_layer_injection(model)
    audio = validate_audio_reinjection(model)
    if tuple(model.model.layer_inject_indices) != tuple(model.model.audio_reinject_indices[:2]):
        raise ValueError("CED injection must cover exactly the first two of three input entrances")
    return {"order": "input_then_intermediate", "input": audio, "intermediate": layer,
            "intermediate_gate_condition": "audio_hidden_after_input_residual",
            "trace": [[5, "input"], [5, "layer4"], [15, "input"], [15, "layer8"], [25, "input"]]}


__all__ = [
    "load_recursive_text_model",
    "validate_recursive_config",
    "validate_loaded_model",
    "validate_layer_injection",
    "validate_audio_reinjection",
    "validate_serial_injection",
    "validate_loop_lora",
]
