"""Loader and contract checks for the isolated 5-10x2-5 MeSH text model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


EXPECTED_ARCHITECTURE = "logical_30_physical_20_5_10x2_5_mesh"
EXPECTED_CHECKPOINT_CONTRACT = "logical_30_physical_20_5_10x2_5"
EXPECTED_LOGICAL_TO_PHYSICAL = (
    0, 1, 2, 3, 4,
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19,
)

# ``Trainer.train`` intentionally validates the text checkpoint once before
# constructing the multimodal model.  Keep registration process-local and
# idempotent because both calls reach this loader on rank 0.
_AUTO_CLASS_REGISTERED = False


def _import_recursive_model() -> tuple[Any, Any, Any]:
    """Load the recursive implementation owned by this isolated route.

    The repository contains several independent 5-10x2-5 implementations.
    Importing by the global ``recursive_model_5_10x2_5_mesh`` name could pick
    another route when the launcher has added ``code/RSmol`` to ``sys.path``.
    Load the sibling file by absolute path instead.
    """
    module_path = Path(__file__).with_name("recursive_model_5_10x2_5_mesh.py").resolve(strict=True)
    module_name = "_mellow_v0_two_stage_recursive_model"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to load isolated recursive model: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return (
        module.RecursiveLlamaForCausalLM,
        module.parameter_audit,
        module.register_auto_class,
    )


def validate_recursive_config(config: Any) -> None:
    """Reject a checkpoint from another MeSH or ordinary-text contract."""
    hidden_size = int(getattr(config, "hidden_size", -1))
    logical_layers = int(getattr(config, "num_hidden_layers", -1))
    physical_layers = int(getattr(config, "recursive_layer_count", -1))
    loops = int(getattr(config, "recursive_loops", -1))
    memory_slots = int(getattr(config, "mesh_memory_slots", -1))
    router_count = int(getattr(config, "mesh_router_count", -1))
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
    if checkpoint_contract != EXPECTED_CHECKPOINT_CONTRACT:
        raise ValueError(
            "unexpected mesh checkpoint contract: "
            f"{checkpoint_contract!r}; expected {EXPECTED_CHECKPOINT_CONTRACT!r}"
        )
    if architecture and architecture not in {EXPECTED_ARCHITECTURE, EXPECTED_CHECKPOINT_CONTRACT}:
        raise ValueError(f"unexpected model architecture contract: {architecture!r}")
    raw_schedule = getattr(config, "logical_to_physical", None)
    if raw_schedule is None:
        raw_schedule = getattr(config, "logical_to_physical_schedule", ())
    schedule = tuple(int(value) for value in raw_schedule)
    if schedule != EXPECTED_LOGICAL_TO_PHYSICAL:
        raise ValueError(f"unexpected logical_to_physical schedule: {schedule!r}")


def load_recursive_text_model(model_path: str | Path) -> Any:
    """Load the text model only; audio modules remain owned by Mellow."""
    path = Path(model_path).expanduser().resolve(strict=True)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"mesh text checkpoint directory/config.json missing: {path}")
    global _AUTO_CLASS_REGISTERED
    model_cls, parameter_audit, register_auto_class = _import_recursive_model()
    if not _AUTO_CLASS_REGISTERED:
        register_auto_class()
        _AUTO_CLASS_REGISTERED = True
    model = model_cls.from_pretrained(path, local_files_only=True)
    validate_recursive_config(model.config)
    audit = parameter_audit(model)
    if audit.get("logical_layer_count") != 30 or audit.get("physical_layer_count") != 20:
        raise ValueError(f"loaded mesh parameter audit has invalid layer geometry: {audit}")
    # ``parameter_audit`` names this field ``memory_slots`` in the recursive
    # model implementation.  Keep the check against that canonical field so
    # a valid checkpoint is not rejected by an adapter-only alias.
    if audit.get("memory_slots") != 5 or audit.get("router_count") != 6:
        raise ValueError(f"loaded mesh parameter audit has invalid memory/router geometry: {audit}")
    return model


__all__ = [
    "EXPECTED_ARCHITECTURE",
    "EXPECTED_CHECKPOINT_CONTRACT",
    "EXPECTED_LOGICAL_TO_PHYSICAL",
    "load_recursive_text_model",
    "validate_recursive_config",
]
