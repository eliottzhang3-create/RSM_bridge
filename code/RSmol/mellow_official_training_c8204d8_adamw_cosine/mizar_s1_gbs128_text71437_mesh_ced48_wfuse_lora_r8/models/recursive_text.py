"""Strict, explicit 7-14x3-7 text loader; never restores text training state."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from contract import recursive_checkpoint_contract
from mizar_text_7_14x3_7_mesh.contract import ROUTER_KEYS, SCHEDULE, validate_model_config


def validate_recursive_config(config):
    validate_model_config(config.to_dict())


def validate_parameter_audit(audit):
    expected = {"logical_layer_count": 56, "physical_layer_count": 28,
                "recursive_loops": 3, "memory_slots": 6, "router_count": 8,
                "router_parameter_tensor_count": 16, "router_parameter_count": 27696,
                "physical_layer_objects_independent": True, "router_objects_independent": True,
                "embedding_head_tied": True, "all_parameters_trainable": True,
                "memory_persistent": False, "schedule": list(SCHEDULE)}
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError(f"loaded 7-14x3-7 parameter audit mismatch: {audit}")


def validate_loading_info(info):
    problems = {key: info[key] for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
    if problems:
        raise ValueError(f"strict 7-14x3-7 model loading failed: {problems}")


def audit_checkpoint_tensors(model, checkpoint):
    """Compare every persisted tensor, including all sixteen trained routers.

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
        raise ValueError("checkpoint must contain all sixteen trained router tensors")
    return {"all_persisted_tensors_equal": True, "persisted_tensors": len(seen),
            "omitted_tied_aliases": sorted(missing), "router_tensors": len(router_hashes),
            "router_source_sha256": router_hashes, "parameters_finite": True}


def load_recursive_text_model(model_path):
    import torch
    from models.recursive_mesh_wfuse_lora import (
        MizarText7_14x3_7MeshForCausalLM, parameter_audit,
    )
    path = Path(model_path).expanduser().resolve(strict=True)
    checkpoint = recursive_checkpoint_contract(path)
    model, info = MizarText7_14x3_7MeshForCausalLM.from_pretrained(
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
    inherited_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    inherited_parameters = dict(model.named_parameters())
    model.model.enable_audio_fusion()
    from models.loop_lora import install_loop_lora
    lora = install_loop_lora(model.model)
    state = model.state_dict()
    for name, tensor in inherited_state.items():
        if not torch.equal(tensor, state[name]):
            raise ValueError(f"audio/LoRA registration changed inherited tensor: {name}")
    current_parameters = dict(model.named_parameters())
    if any(current_parameters[name] is not parameter for name, parameter in inherited_parameters.items()):
        raise ValueError("LoRA registration replaced an inherited Parameter object")
    if model.get_input_embeddings().weight is not model.lm_head.weight:
        raise ValueError("audio/LoRA registration broke embedding/head tying")
    model.audio_initialization_audit.update(
        inherited_tensors_unchanged_after_registration=True,
        inherited_parameter_objects_preserved=True,
        audio_fusion=validate_audio_fusion(model, fresh=True), loop_lora=lora)
    model.generation_config.use_cache = False
    return model


def validate_audio_fusion(model: Any, *, fresh=False) -> dict:
    import torch
    core, width = model.model, model.config.hidden_size
    if core.fusion_audio_tokens != 126 or set(core.audio_fusions) != {"4", "8"}:
        raise ValueError("invalid CED write-value fusion geometry")
    fusions = list(core.audio_fusions.values())
    if len({id(f) for f in fusions}) != 2 or len({f.weight.untyped_storage().data_ptr() for f in fusions}) != 2:
        raise ValueError("fusion matrices must be independent")
    for fusion in fusions:
        weight = fusion.weight
        if (tuple(weight.shape) != (width, 2 * width) or fusion.bias is not None
                or not weight.requires_grad or weight.dtype != model.model.embed_tokens.weight.dtype
                or weight.device != model.model.embed_tokens.weight.device or not bool(torch.isfinite(weight).all())):
            raise ValueError("fusion requires trainable bias-free Linear(2H,H) in LM dtype/device")
        if fresh and (not torch.equal(weight[:, :width], torch.eye(width, dtype=weight.dtype, device=weight.device))
                      or bool(weight[:, width:].ne(0).any())):
            raise ValueError("fresh fusion must be exactly [I,0]")
    if hasattr(core, "audio_gates") or hasattr(core, "listen_routers"):
        raise ValueError("write-value fusion must not retain gates or extra listen routers")
    return {"ced_layers": [4, 8], "logical_indices": [7, 21], "write_router_indices": [0, 1],
            "audio_tokens": 126, "placement": "original_write_value_only",
            "bias": False, "weight_initialization": "identity_hidden_zero_encoder", "independent": True,
            "parameter_count": 4 * width * width, "original_routers": 8, "total_routers": 8,
            "prefix_query": "prefix_output", "loop_query": "loop_input", "cache": False}
