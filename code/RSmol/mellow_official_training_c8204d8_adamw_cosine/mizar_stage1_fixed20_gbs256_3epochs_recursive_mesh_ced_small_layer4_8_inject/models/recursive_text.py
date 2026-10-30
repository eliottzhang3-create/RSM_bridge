"""Explicit loader for the isolated 5-10x2-5 MeSH text checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from contract import (
    TEXT_MODEL_ARCHITECTURE,
    TEXT_MODEL_CONTRACT,
    TEXT_MODEL_SCHEDULE,
)

def _import_recursive_model() -> tuple[Any, Any]:
    """Explicitly select this route; never replace process-global AutoModel classes."""
    # decoder.py lives at code/RSmol/<family>/<route>/models; parents[3] is code/RSmol.
    rsmol_root = Path(__file__).resolve().parents[3]
    if str(rsmol_root) not in sys.path:
        sys.path.insert(0, str(rsmol_root))
    from models.recursive_layer_inject import (
        RecursiveLlamaForCausalLM,
        parameter_audit,
    )
    return RecursiveLlamaForCausalLM, parameter_audit


def validate_recursive_config(config: Any) -> None:
    hidden_size = int(getattr(config, "hidden_size", -1))
    logical_layers = int(getattr(config, "num_hidden_layers", -1))
    physical_layers = int(getattr(config, "recursive_layer_count", -1))
    loops = int(getattr(config, "recursive_loops", -1))
    memory_slots = int(getattr(config, "mesh_memory_slots", getattr(config, "memory_slots", -1)))
    router_count = int(getattr(config, "mesh_router_count", getattr(config, "router_count", -1)))
    checkpoint_contract = str(getattr(config, "mesh_architecture_contract", ""))
    architecture = str(getattr(config, "model_architecture_contract", ""))
    if hidden_size != 576:
        raise ValueError(f"mesh text hidden_size must be 576, got {hidden_size}")
    if (logical_layers, physical_layers, loops, memory_slots, router_count) != (30, 20, 2, 5, 6):
        raise ValueError(
            "mesh text geometry mismatch: "
            f"logical={logical_layers} physical={physical_layers} loops={loops} "
            f"memory_slots={memory_slots} router_count={router_count}"
        )
    if checkpoint_contract != TEXT_MODEL_CONTRACT:
        raise ValueError(
            f"unexpected mesh checkpoint contract: {checkpoint_contract!r}; "
            f"expected {TEXT_MODEL_CONTRACT!r}"
        )
    if architecture and architecture not in {TEXT_MODEL_ARCHITECTURE, TEXT_MODEL_CONTRACT}:
        raise ValueError(f"unexpected model architecture contract: {architecture!r}")
    raw_schedule = getattr(config, "logical_to_physical", None)
    if raw_schedule is None:
        raw_schedule = getattr(config, "logical_to_physical_schedule", ())
    schedule = tuple(int(value) for value in raw_schedule)
    if schedule != tuple(TEXT_MODEL_SCHEDULE):
        raise ValueError(f"unexpected logical_to_physical schedule: {schedule!r}")


def load_recursive_text_model(model_path: str | Path) -> Any:
    """Load only the full recursive LM; audio modules remain baseline-owned."""
    path = Path(model_path).expanduser().resolve(strict=True)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"mesh text checkpoint directory/config.json missing: {path}")
    model_cls, parameter_audit = _import_recursive_model()
    model, loading_info = model_cls.from_pretrained(
        path, local_files_only=True, output_loading_info=True,
    )
    failures = {key: loading_info.get(key) for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if loading_info.get(key)}
    if failures:
        raise ValueError(f"MeSH text weights did not load completely: {failures}")
    validate_loaded_model(model, parameter_audit(model))
    # All inherited tensors, including the six routers, must load before adding gates.
    model.model.enable_layer_injection()
    model.generation_config.use_cache = False
    validate_layer_injection(model)
    return model


def validate_loaded_model(model: Any, audit: dict) -> None:
    validate_recursive_config(model.config)
    expected = {
        "logical_layer_count": 30,
        "physical_layer_count": 20,
        "memory_slots": 5,
        "router_count": 6,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError(f"loaded mesh parameter audit mismatch: {audit}")
    if tuple(audit.get("schedule", ())) != tuple(TEXT_MODEL_SCHEDULE):
        raise ValueError(f"loaded mesh parameter schedule mismatch: {audit}")
    core = model.model
    routers = list(core.write_routers) + list(core.read_routers)
    if (len(core.layers) != 20 or len({id(layer) for layer in core.layers}) != 20
            or core.memory_slots != 5 or len(routers) != 6
            or len({id(router) for router in routers}) != 6
            or any(tuple(router.weight.shape) != (5, 576) for router in routers)):
        raise ValueError("MeSH requires 20 independent layers and six independent 5-slot routers")


def validate_layer_injection(model: Any) -> dict:
    core = model.model
    width = model.config.hidden_size
    if (core.layer_inject_tokens != 126 or core.layer_inject_indices != (5, 15)
            or set(core.audio_gates) != {"4", "8"}
            or id(core.audio_gates["4"]) == id(core.audio_gates["8"])):
        raise ValueError("invalid independent CED layer4/8 injection geometry")
    for gate in core.audio_gates.values():
        if tuple(gate.weight.shape) != (1, width) or tuple(gate.bias.shape) != (1,):
            raise ValueError("each injection gate must be Linear(hidden_size,1)")
    return {"ced_layers": [4, 8], "logical_indices": [5, 15], "audio_tokens": 126,
            "read_router_indices": [0, 1], "placement": "after_memory_read_before_middle_loop",
            "gate": "tanh_per_audio_token_scalar", "gate_initialization": "zero_weight_and_bias",
            "shared_between_loops": False, "cache": False,
            "gate_parameter_count": 2 * (width + 1)}


__all__ = [
    "load_recursive_text_model",
    "validate_recursive_config",
    "validate_loaded_model",
    "validate_layer_injection",
]
