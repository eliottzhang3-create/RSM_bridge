"""Strict, explicit 7-14x2-7 text loader; never restores text training state."""
from __future__ import annotations

import hashlib
from pathlib import Path

from contract import recursive_checkpoint_contract
from mizar_text_7_14x2_7_mesh.contract import ROUTER_KEYS, SCHEDULE, validate_model_config


def validate_recursive_config(config):
    validate_model_config(config.to_dict())


def validate_parameter_audit(audit):
    expected = {"logical_layer_count": 42, "physical_layer_count": 28,
                "recursive_loops": 2, "memory_slots": 5, "router_count": 6,
                "router_parameter_tensor_count": 12, "router_parameter_count": 17310,
                "physical_layer_objects_independent": True, "router_objects_independent": True,
                "embedding_head_tied": True, "all_parameters_trainable": True,
                "memory_persistent": False, "schedule": list(SCHEDULE)}
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError(f"loaded 7-14x2-7 parameter audit mismatch: {audit}")


def validate_loading_info(info):
    problems = {key: info[key] for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
    if problems:
        raise ValueError(f"strict 7-14x2-7 model loading failed: {problems}")


def audit_checkpoint_tensors(model, checkpoint):
    """Compare every persisted tensor, including all twelve trained routers.

    Safetensors may omit one tied embedding/head alias. The other alias must
    exist and equal both model entries. No other missing/extra key is allowed.
    """
    import torch
    from safetensors import safe_open
    state = model.state_dict()
    seen, router_hashes = set(), {}
    for path in sorted(Path(checkpoint).glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in seen or name not in state:
                    raise ValueError(f"duplicate/unexpected checkpoint tensor: {name}")
                source = handle.get_tensor(name)
                target = state[name].detach().cpu()
                if source.dtype != torch.float32 or target.dtype != torch.float32:
                    raise ValueError(f"text weights must remain FP32: {name}")
                if not torch.equal(source, target) or not bool(torch.isfinite(source).all()):
                    raise ValueError(f"source tensor inheritance/finite check failed: {name}")
                seen.add(name)
                if name in ROUTER_KEYS:
                    router_hashes[name] = hashlib.sha256(source.contiguous().numpy().tobytes()).hexdigest()
    missing = set(state) - seen
    tied = {"model.embed_tokens.weight", "lm_head.weight"}
    if missing:
        if not missing < tied or len(tied & seen) != 1:
            raise ValueError(f"checkpoint missing inherited tensors: {sorted(missing)}")
        if not torch.equal(state["model.embed_tokens.weight"], state["lm_head.weight"]):
            raise ValueError("omitted tied embedding/head alias differs")
    if set(router_hashes) != ROUTER_KEYS:
        raise ValueError("checkpoint must contain all twelve trained router tensors")
    return {"all_persisted_tensors_equal": True, "persisted_tensors": len(seen),
            "omitted_tied_aliases": sorted(missing), "router_tensors": len(router_hashes),
            "router_source_sha256": router_hashes, "parameters_finite": True}


def load_recursive_text_model(model_path):
    import torch
    from recursive_model_mizar_text_7_14x2_7_mesh import (
        MizarText7_14x2_7MeshForCausalLM, parameter_audit,
    )
    path = Path(model_path).expanduser().resolve(strict=True)
    checkpoint = recursive_checkpoint_contract(path)
    model, info = MizarText7_14x2_7MeshForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.float32, output_loading_info=True,
    )
    validate_loading_info(info)
    validate_recursive_config(model.config)
    audit = parameter_audit(model)
    validate_parameter_audit(audit)
    inherited = audit_checkpoint_tensors(model, path)
    model.audio_initialization_audit = {
        "source_checkpoint": checkpoint, "parameter_audit": audit,
        "tensor_inheritance": inherited, "audio_optimizer_start_step": 0,
        "restore_text_training_state": False, "routers": "inherited_without_reinitialization",
    }
    return model
