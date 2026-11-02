"""Explicit loader for the isolated 5-10x4-5 MeSH text checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from contract import (
    TEXT_MODEL_ARCHITECTURE,
    TEXT_MODEL_CONTRACT,
    TEXT_MODEL_SCHEDULE,
)

_AUTO_CLASS_REGISTERED = False


def _import_recursive_model() -> tuple[Any, Any, Any]:
    """Import the shared model implementation without importing a route trainer."""
    # decoder.py lives at code/RSmol/<family>/<route>/models; parents[3] is code/RSmol.
    rsmol_root = Path(__file__).resolve().parents[3]
    if str(rsmol_root) not in sys.path:
        sys.path.insert(0, str(rsmol_root))
    from recursive_model_5_10x4_5_mesh import (  # type: ignore
        RecursiveLlamaForCausalLM,
        parameter_audit,
        register_auto_class,
    )
    return RecursiveLlamaForCausalLM, parameter_audit, register_auto_class


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
    if (logical_layers, physical_layers, loops, memory_slots, router_count) != (50, 20, 4, 7, 10):
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
    global _AUTO_CLASS_REGISTERED
    model_cls, parameter_audit, register_auto_class = _import_recursive_model()
    if not _AUTO_CLASS_REGISTERED:
        register_auto_class()
        _AUTO_CLASS_REGISTERED = True
    loaded = model_cls.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ValueError("recursive checkpoint loader did not return loading info")
    model, loading = loaded
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading.get(key):
            raise ValueError(f"recursive checkpoint did not load completely: {key}={loading[key]}")
    validate_recursive_config(model.config)
    audit = parameter_audit(model)
    expected = {
        "logical_layer_count": 50,
        "physical_layer_count": 20,
        "recursive_loops": 4,
        "logical_cache_slot_count": 50,
        "memory_slots": 7,
        "router_count": 10,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError(f"loaded mesh parameter audit mismatch: {audit}")
    if tuple(audit.get("schedule", ())) != tuple(TEXT_MODEL_SCHEDULE):
        raise ValueError(f"loaded mesh parameter schedule mismatch: {audit}")
    if audit.get("router_objects_independent") is not True:
        raise ValueError("5-10x4-5 requires ten independent router modules")
    if audit.get("memory_persistent") is not False:
        raise ValueError("MeSH memory must be transient")
    if getattr(getattr(model, "model", None), "ablate_after_two_loops", False):
        raise ValueError("5-10x4-5 training requires all four loops")
    if any(parameter.dtype != torch.float32
           for parameter in model.parameters() if parameter.is_floating_point()):
        raise ValueError("recursive text parameters must all be FP32")
    if model.get_input_embeddings().weight.shape != model.get_output_embeddings().weight.shape:
        raise ValueError("recursive embedding/lm_head shapes differ")
    return model


__all__ = [
    "load_recursive_text_model",
    "validate_recursive_config",
]
