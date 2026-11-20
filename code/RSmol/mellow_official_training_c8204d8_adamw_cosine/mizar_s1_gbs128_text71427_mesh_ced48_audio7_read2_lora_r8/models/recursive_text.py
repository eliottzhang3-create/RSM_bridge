"""Strict, explicit 7-14x2-7 text loader; never restores text training state."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

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
    from models.recursive_audio7_read2 import (
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
    inherited_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    inherited_parameters = dict(model.named_parameters())
    model.model.enable_audio_read2()
    model.model.enable_loop_lora()
    model.generation_config.use_cache = False
    # New wrappers keep the actual base Parameters, not copied replacements.
    current = model.state_dict()
    if any(not torch.equal(value, current[name]) for name, value in inherited_state.items()):
        raise ValueError("installing audio/LoRA modules changed inherited text tensors")
    parameters = dict(model.named_parameters())
    if any(parameters[name] is not parameter for name, parameter in inherited_parameters.items()):
        raise ValueError("installing audio/LoRA modules replaced an inherited Parameter object")
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError("installing audio/LoRA modules broke tied embedding/head")
    validate_audio_read2(model)
    validate_loop_lora(model.model)
    model.audio_initialization_audit["inherited_tensors_unchanged_after_extensions"] = True
    return model


def validate_loop_lora(core):
    from models.loop_lora import validate_loop_lora as validate
    return validate(core)

def validate_audio_read2(model: Any) -> dict:
    core = model.model
    width = model.config.hidden_size
    if core.audio_tokens != 126 or len(core.audio_read_routers) != 2:
        raise ValueError("expected exactly two seven-slot audio read routers")
    routers = list(core.write_routers) + list(core.read_routers) + list(core.audio_read_routers)
    expected = [5] * 6 + [7] * 2
    storage = set()
    for router, slots in zip(routers, expected):
        if tuple(router.weight.shape) != (slots, width) or tuple(router.bias.shape) != (slots,):
            raise ValueError("invalid audio read2/text5 router shape")
        for parameter in router.parameters():
            key = parameter.untyped_storage().data_ptr()
            if key in storage:
                raise ValueError("router parameters must have independent storage")
            storage.add(key)
    return {"audio_positions": [0, 126], "audio_read_indices": [0, 1],
            "write_slots": [5, 5, 5], "audio_read_slots": [7, 7, 5],
            "text_read_slots": [5, 5, 5], "seed_slots_read_only": [5, 6],
            "router_modules": len(routers), "router_tensors": len(storage),
            "added_router_parameters": 2 * 7 * (width + 1),
            "router_parameter_count": sum(p.numel() for r in routers for p in r.parameters())}
