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

_AUTO_CLASS_REGISTERED = False


def _import_recursive_model() -> tuple[Any, Any, Any]:
    """Import the shared model implementation without importing a route trainer."""
    # decoder.py lives at code/RSmol/<family>/<route>/models; parents[3] is code/RSmol.
    rsmol_root = Path(__file__).resolve().parents[3]
    if str(rsmol_root) not in sys.path:
        sys.path.insert(0, str(rsmol_root))
    from models.recursive_mesh_listen import (  # type: ignore
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
    global _AUTO_CLASS_REGISTERED
    model_cls, parameter_audit, register_auto_class = _import_recursive_model()
    if not _AUTO_CLASS_REGISTERED:
        register_auto_class()
        _AUTO_CLASS_REGISTERED = True
    model, loading = model_cls.from_pretrained(path, local_files_only=True, output_loading_info=True)
    failures = {key: loading.get(key) for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
    if failures:
        raise ValueError(f"original MeSH text weights did not load strictly: {failures}")
    validate_recursive_config(model.config)
    audit = parameter_audit(model)
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
    # New parameters are constructed only after all inherited tensors are loaded.
    model.model.enable_audio_listen()
    model.config.use_cache = False
    model.generation_config.use_cache = False
    validate_audio_listen(model)
    return model


__all__ = [
    "load_recursive_text_model",
    "validate_recursive_config",
]


def validate_audio_listen(model):
    from contract import audio_listen_contract
    adapter = model.model.audio_listen
    if (adapter.width, adapter.audio_tokens, adapter.prompt_tokens, adapter.memory_slots,
            adapter.text_start) != (576, 126, 256, 5, 127):
        raise ValueError("audio listen geometry mismatch")
    norms = [(n, 1536) for n in adapter.audio_input_norms]
    norms += [(n, 576) for n in adapter.audio_output_norms]
    norms += [(adapter.text_norm, 576), (adapter.query_norm, 576)]
    if len(norms) != 10 or len({id(n) for n, _ in norms}) != 10:
        raise ValueError("four independent input/output norms and text/query norms required")
    if any(tuple(n.normalized_shape) != (width,) or n.eps != 1e-5 or n.elementwise_affine
           for n, width in norms):
        raise ValueError("listen LayerNorm shape/epsilon/non-affine policy mismatch")
    expected = {}
    for i in range(4):
        expected[f"projections.{i}.weight"] = (576, 1536)
        expected[f"projections.{i}.bias"] = (576,)
    for i in range(2):
        for component, rows in (("gates", 4), ("listen_routers", 5), ("amplitudes", 1)):
            expected[f"{component}.{i}.weight"] = (rows, 576)
            expected[f"{component}.{i}.bias"] = (rows,)
    parameters = dict(adapter.named_parameters())
    if (set(parameters) != set(expected)
            or any(tuple(parameters[name].shape) != shape for name, shape in expected.items())
            or len({id(p) for p in parameters.values()}) != len(expected)
            or any(not p.requires_grad for p in parameters.values())):
        raise ValueError("listen parameter inventory/sharing/trainability mismatch")
    routers = list(model.model.write_routers) + list(model.model.read_routers) + list(adapter.listen_routers)
    if len(routers) != 8 or len({id(r) for r in routers}) != 8 or len(model.model.layers) != 20:
        raise ValueError("listen route requires six original routers, two new routers and twenty physical layers")
    count = sum(p.numel() for p in parameters.values())
    if count != audio_listen_contract()["new_parameters"]:
        raise ValueError("listen parameter count mismatch")
    return {**audio_listen_contract(), "actual_parameters": count}
