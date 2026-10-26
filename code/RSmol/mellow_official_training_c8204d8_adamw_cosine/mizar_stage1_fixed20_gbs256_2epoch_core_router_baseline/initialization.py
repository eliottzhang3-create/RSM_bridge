"""Strict model-only initialization from the completed recursive checkpoint."""
from pathlib import Path

import torch

from contract import SOURCE_ROUTE


def load_trained_initialization(model, checkpoint):
    checkpoint = Path(checkpoint).resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("route_variant") != SOURCE_ROUTE or payload.get("total_step") != 17313:
        raise ValueError("initialization requires the completed baseline MeSH step-17313 checkpoint")
    source = {}
    for name, tensor in payload["state_dict"].items():
        name = name.removeprefix("module.")
        if name in source:
            raise ValueError(f"duplicate normalized checkpoint key: {name}")
        source[name] = tensor
    target = model.state_dict()
    extra = sorted(set(source) - set(target))
    missing = sorted(set(target) - set(source))
    wrong_shapes = [name for name in source.keys() & target.keys()
                    if source[name].shape != target[name].shape]
    if extra or missing or wrong_shapes:
        raise ValueError(f"initialization checkpoint mismatch: extra={extra}, missing={missing}, shapes={wrong_shapes}")
    result = model.load_state_dict(source, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("strict legacy checkpoint load returned unexpected key differences")
    loaded = model.state_dict()
    for name, old in source.items():
        if not torch.equal(loaded[name].detach().cpu(), old):
            raise ValueError(f"legacy tensor was not loaded exactly: {name}")
    return {"path": str(checkpoint), "source_route": SOURCE_ROUTE, "source_step": 17313,
            "loaded_tensors": len(source), "loaded_numel": sum(t.numel() for t in source.values()),
            "legacy_tensors_exact": True, "strict_state_dict": True,
            "optimizer_scheduler_scaler_tracker_rng_restarted": True}
