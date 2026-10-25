"""Strict legacy-weight initialization, separate from full training resume."""
from pathlib import Path

import torch


NEW_PREFIXES = tuple("caption_decoder.lm.model." + name for name in
                     ("audio_memory_builder.", "audio_attention.", "perceive_routers."))
SOURCE_ROUTE = "htsat_mizar_s1_fixed20_gbs256_recursive_mesh_v1"


def load_trained_initialization(model, checkpoint):
    """Load every old tensor; missing tensors must belong to the new modules."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("route_variant") != SOURCE_ROUTE or payload.get("total_step") != 17313:
        raise ValueError("initialization requires the completed baseline MeSH step-17313 checkpoint")
    raw = payload["state_dict"]
    source = {}
    for name, tensor in raw.items():
        name = name.removeprefix("module.")
        if name in source:
            raise ValueError(f"duplicate normalized checkpoint key: {name}")
        source[name] = tensor
    target = model.state_dict()
    extra = sorted(set(source) - set(target))
    missing = sorted(set(target) - set(source))
    wrong_shapes = [name for name in source.keys() & target.keys()
                    if source[name].shape != target[name].shape]
    if extra or wrong_shapes or not missing or any(not name.startswith(NEW_PREFIXES) for name in missing):
        raise ValueError(f"legacy initialization mismatch: extra={extra}, shapes={wrong_shapes}, missing={missing}")
    if any(name.startswith(NEW_PREFIXES) for name in source):
        raise ValueError("initialization checkpoint must not contain new audio-injection parameters")
    result = model.load_state_dict(source, strict=False)
    if sorted(result.missing_keys) != missing or result.unexpected_keys:
        raise ValueError("load_state_dict did not preserve the audited legacy key contract")
    # Exact comparison of all legacy tensors; no optimizer/RNG/scheduler state
    # from the source run is used in this new one-epoch run.
    for name, old in source.items():
        if not torch.equal(model.state_dict()[name].detach().cpu(), old):
            raise ValueError(f"legacy tensor was not loaded exactly: {name}")
    return {"path": str(checkpoint), "source_route": payload["route_variant"],
            "source_step": payload["total_step"], "loaded_tensors": len(source),
            "loaded_numel": sum(t.numel() for t in source.values()),
            "new_parameter_tensors": missing, "legacy_tensors_exact": True,
            "optimizer_scheduler_rng_restarted": True}
