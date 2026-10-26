"""Strict model-only initialization from the completed recursive checkpoint."""
from pathlib import Path

import torch

from contract import SOURCE_ROUTE, SOURCE_STEP, TEXT_MODEL_CONTRACT, TEXT_MODEL_SCHEDULE, digest
from models.recursive_text import _import_recursive_model, validate_loaded_model


def load_trained_initialization(model, checkpoint):
    checkpoint = Path(checkpoint).resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (payload.get("schema_version") != 1 or payload.get("mode") != "formal"
            or payload.get("route_variant") != SOURCE_ROUTE or payload.get("total_step") != SOURCE_STEP):
        raise ValueError("initialization requires the completed non-MeSH step-17313 audio checkpoint")
    source_contract = payload.get("contract", {})
    if (payload.get("contract_sha256") != digest(source_contract)
            or source_contract.get("route") != SOURCE_ROUTE
            or source_contract.get("text_model_contract") != TEXT_MODEL_CONTRACT
            or source_contract.get("text_model_schedule") != list(TEXT_MODEL_SCHEDULE)):
        raise ValueError("initialization source contract/architecture mismatch")
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
    wrong_dtypes = [name for name in source.keys() & target.keys()
                    if source[name].dtype != target[name].dtype]
    if extra or missing or wrong_shapes or wrong_dtypes:
        raise ValueError(f"initialization checkpoint mismatch: extra={extra}, missing={missing}, "
                         f"shapes={wrong_shapes}, dtypes={wrong_dtypes}")
    result = model.load_state_dict(source, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("strict legacy checkpoint load returned unexpected key differences")
    loaded = model.state_dict()
    for name, old in source.items():
        if not torch.equal(loaded[name].detach().cpu(), old):
            raise ValueError(f"legacy tensor was not loaded exactly: {name}")
    _, parameter_audit = _import_recursive_model()
    validate_loaded_model(model.caption_decoder.lm, parameter_audit(model.caption_decoder.lm))
    return {"path": str(checkpoint), "source_route": SOURCE_ROUTE, "source_step": SOURCE_STEP,
            "loaded_tensors": len(source), "loaded_numel": sum(t.numel() for t in source.values()),
            "legacy_tensors_exact": True, "strict_state_dict": True, "non_mesh_model_audit": True,
            "optimizer_scheduler_scaler_tracker_rng_restarted": True}
