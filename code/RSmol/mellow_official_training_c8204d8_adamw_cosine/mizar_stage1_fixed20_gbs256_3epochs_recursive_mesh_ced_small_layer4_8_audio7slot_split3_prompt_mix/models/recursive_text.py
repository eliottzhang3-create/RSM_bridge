"""Explicit loader for the isolated 5-10x2-5 MeSH text checkpoint."""

from __future__ import annotations

import math
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
    from models.recursive_mesh_listen import (
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
    # All inherited tensors, including the six routers, must load before copying the audio routers.
    model.model.enable_audio_listen()
    model.generation_config.use_cache = False
    validate_audio_listen(model, fresh=True)
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


def validate_audio_listen(model: Any, *, fresh: bool = False) -> dict:
    import torch
    core, width = model.model, model.config.hidden_size
    if (core.listen_audio_tokens != 126 or core.audio_memory_slots != 7
            or core.text_memory_slots != 5
            or core.audio_router_initialization != "split_slot0_three_ways"
            or hasattr(core, "audio_router_init_k")
            or hasattr(core, "audio_gates") or hasattr(core, "listen_routers")):
        raise ValueError("invalid audio7/text5 memory geometry or legacy gate modules")
    sources = list(core.write_routers) + list(core.read_routers)
    targets = list(core.audio_listen_write_routers) + list(core.audio_listen_read_routers)
    if (len(sources) != 6 or len(targets) != 6
            or len({id(r) for r in sources + targets}) != 12
            or len({p.data_ptr() for r in sources + targets for p in r.parameters()}) != 24
            or any(tuple(r.weight.shape) != (5, width) or tuple(r.bias.shape) != (5,) for r in sources)
            or any(tuple(r.weight.shape) != (7, width) or tuple(r.bias.shape) != (7,) for r in targets)):
        raise ValueError("requires six original five-way and six independent seven-way audio routers")
    if fresh:
        for source, target in zip(sources, targets):
            if (not torch.equal(source.weight[1:5], target.weight[1:5])
                    or not torch.equal(source.bias[1:5], target.bias[1:5])):
                raise ValueError("fresh split3 audio rows1..4 must preserve original rows")
            for slot in (0, 5, 6):
                if (not torch.equal(target.weight[slot], source.weight[0])
                        or not torch.equal(target.bias[slot], source.bias[0] - math.log(3.0))):
                    raise ValueError("fresh split3 rows0/5/6 must copy original slot0 weight and bias-log(3)")
    return {"audio_memory_slots": 7, "text_memory_slots": 5, "audio_tokens": 126,
            "ced_layers": [4, 8], "initialization_before_prelude": True,
            "initial_audio_slots": ["alpha_E*E", 0, 0, 0, 0, "alpha_4*A4", "alpha_8*A8"],
            "reinjection": False, "ordinary_writes_to_all_slots": True,
            "audio_router_initialization": "split_slot0_three_ways", "split_source_slot_0based": 0,
            "split_target_slots_0based": [0, 5, 6], "split_bias_offset": -math.log(3.0),
            "new_router_parameter_count": 6 * 7 * (width + 1),
            "original_routers": 6, "audio_routers": 6, "total_routers": 12, "cache": False}


__all__ = [
    "load_recursive_text_model",
    "validate_recursive_config",
    "validate_loaded_model",
    "validate_audio_listen",
]
