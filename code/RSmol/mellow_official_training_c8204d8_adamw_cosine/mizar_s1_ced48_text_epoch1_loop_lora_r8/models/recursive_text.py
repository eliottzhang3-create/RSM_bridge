"""Explicit loader for the fixed 5+10x2+5 non-MeSH text checkpoint."""

from __future__ import annotations

import sys
import hashlib
import json
from pathlib import Path
from typing import Any

from contract import (TEXT_MODEL_SCHEDULE, validate_recursive_config_values,
                      recursive_checkpoint_contract)


def validate_loop_lora(core):
    from models.loop_lora import validate_loop_lora as validate
    return validate(core)


def _import_recursive_model() -> tuple[Any, Any]:
    """Use the non-MeSH implementation explicitly, independent of AutoModel registration."""
    rsmol_root = Path(__file__).resolve().parents[3]
    if str(rsmol_root) not in sys.path:
        sys.path.insert(0, str(rsmol_root))
    from models.recursive_layer_inject import (
        RecursiveLlamaForCausalLM,
        parameter_audit,
    )
    return RecursiveLlamaForCausalLM, parameter_audit


def validate_recursive_config(config: Any) -> None:
    validate_recursive_config_values(config.to_dict())


def validate_loaded_model(model: Any, audit: dict) -> None:
    """Check real physical modules as well as the declared execution schedule."""
    validate_recursive_config(model.config)
    expected = {
        "logical_layer_count": 30,
        "physical_layer_count": 20,
        "recursive_layer_count": 20,
        "recursive_loops": 2,
        "middle_recurrent_count": 10,
        "model_physical_module_count": 20,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError(f"loaded non-MeSH parameter audit mismatch: {audit}")
    if tuple(audit.get("schedule", ())) != TEXT_MODEL_SCHEDULE:
        raise ValueError(f"loaded non-MeSH parameter schedule mismatch: {audit}")
    layers = model.model.layers
    if len(layers) != 20 or len({id(layer) for layer in layers}) != 20:
        raise ValueError("non-MeSH text model requires 20 independent physical layers")
    if tuple(model.model.logical_to_physical) != TEXT_MODEL_SCHEDULE:
        raise ValueError("loaded non-MeSH execution schedule mismatch")
    for name, _ in (*model.named_parameters(), *model.named_buffers()):
        if any(part.startswith(("router", "mesh_", "memory_slot"))
               for part in name.split(".")):
            raise ValueError(f"non-MeSH model contains MeSH parameter/buffer: {name}")


def load_recursive_text_model(model_path: str | Path) -> Any:
    """Load only pretrained LM weights; the audio trainer owns all other state."""
    path = Path(model_path).expanduser().resolve(strict=True)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"non-MeSH text checkpoint directory/config.json missing: {path}")
    source_contract = recursive_checkpoint_contract(path)
    model_cls, parameter_audit = _import_recursive_model()
    model, loading_info = model_cls.from_pretrained(
        path, local_files_only=True, output_loading_info=True,
        torch_dtype=__import__("torch").float32,
    )
    failures = {key: loading_info.get(key) for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if loading_info.get(key)}
    if failures:
        raise ValueError(f"non-MeSH text checkpoint weights do not match the model: {failures}")
    validate_loaded_model(model, parameter_audit(model))
    validate_loop_lora(model.model)
    inheritance = audit_inherited_text_weights(path, model)
    # Strictly load ALL inherited weights AND trained A/B, then add fresh gates.
    model.model.enable_layer_injection()
    # A separate RNG fork preserves the original input-route projection initial values.
    model.model.enable_audio_reinjection()
    model.generation_config.use_cache = False
    validate_serial_injection(model)
    validate_loop_lora(model.model)
    # Registering fresh audio modules must not mutate any inherited tensor.
    after = audit_inherited_text_weights(path, model)
    if inheritance != after:
        raise ValueError("adding audio modules changed pretrained text/LoRA tensors")
    model.text_initialization_audit = {**after, "source": source_contract["initialization"],
                                     "audio_components_added_after_strict_load": True}
    return model


def checkpoint_tensors(path):
    """Read only HF model tensors; never open training_state.pt.

    Stream safetensors one tensor at a time to avoid a second full LM copy.
    Support both normal and sharded Hugging Face model files.
    """
    safe = path / "model.safetensors"
    binary = path / "pytorch_model.bin"
    index = next((path / name for name in ("model.safetensors.index.json",
                                          "pytorch_model.bin.index.json")
                  if (path / name).is_file()), None)
    if safe.is_file():
        files = [safe]
    elif binary.is_file():
        files = [binary]
    elif index is not None:
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        files = [path / name for name in sorted(set(mapping.values()))]
    else:
        raise FileNotFoundError("no supported HF model tensors")
    seen = set()
    for file in files:
        if file.parent.resolve() != path.resolve() or not file.is_file():
            raise ValueError("HF weight shard must be an existing file inside the checkpoint")
        if file.suffix == ".safetensors":
            from safetensors import safe_open
            with safe_open(str(file), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in seen:
                        raise ValueError(f"duplicate checkpoint tensor: {key}")
                    seen.add(key)
                    yield key, handle.get_tensor(key)
        elif file.name.startswith("pytorch_model") and file.suffix == ".bin":
            import torch
            state = torch.load(file, map_location="cpu", weights_only=True)
            for key, value in state.items():
                if key in seen:
                    raise ValueError(f"duplicate checkpoint tensor: {key}")
                seen.add(key)
                yield key, value
        else:
            raise ValueError("only HF model.safetensors/pytorch_model shards are supported")
    if index is not None and not safe.is_file() and not binary.is_file() and seen != set(mapping):
        raise ValueError("HF shard inventory differs from index weight_map")


def audit_inherited_text_weights(path, model):
    import torch
    state = model.state_dict()
    expected = {name for name in state if not name.startswith(
        ("model.audio_gates.", "model.audio_linear.", "model.audio_gate."))}
    actual, lora_names, lora_numel = set(), set(), 0
    base_sha, lora_sha = hashlib.sha256(), hashlib.sha256()
    for name, source in checkpoint_tensors(path):
        if name not in expected:
            raise ValueError(f"unexpected source tensor: {name}")
        loaded = state[name].detach().cpu()
        if (source.dtype != torch.float32 or source.shape != loaded.shape
                or loaded.dtype != source.dtype or not torch.isfinite(source).all()
                or not torch.equal(source, loaded)):
            raise ValueError(f"pretrained tensor changed/mismatched: {name}")
        actual.add(name)
        is_lora = ".lora_A." in name or ".lora_B." in name
        digest = lora_sha if is_lora else base_sha
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(source.contiguous().numpy().tobytes())
        if is_lora:
            lora_names.add(name)
            lora_numel += source.numel()
    # HF safetensors serializes a tied embedding/head only once.
    missing = expected - actual
    if missing in ({"lm_head.weight"}, {"model.embed_tokens.weight"}):
        if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
            raise ValueError("missing tied tensor without actual parameter tying")
        missing = set()
    if missing or len(lora_names) != 280:
        raise ValueError(f"incomplete inherited tensors: missing={sorted(missing)}, LoRA={len(lora_names)}")
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError("pretrained embedding/head tying was lost")
    return {"checkpoint": str(path), "all_source_tensors_equal": True,
            "source_tensor_count": len(actual), "lora_tensor_count": len(lora_names),
            "lora_parameter_count": lora_numel, "base_sha256": base_sha.hexdigest(),
            "lora_sha256": lora_sha.hexdigest(), "read_text_training_state": False,
            "lora_reset_after_load": False}


def validate_layer_injection(model: Any) -> dict:
    core = model.model
    width = model.config.hidden_size
    if (core.layer_inject_tokens != 126 or core.layer_inject_indices != (5, 15)
            or set(core.audio_gates) != {"4", "8"}
            or id(core.audio_gates["4"]) == id(core.audio_gates["8"])):
        raise ValueError("invalid independent CED layer4/8 injection geometry")
    for gate in core.audio_gates.values():
        if tuple(gate.weight.shape) != (1, width) or tuple(gate.bias.shape) != (1,):
            raise ValueError("each layer injection gate must be Linear(hidden_size,1)")
    return {"ced_layers": [4, 8], "logical_indices": [5, 15], "audio_tokens": 126,
            "gate": "tanh_per_audio_token_scalar", "gate_initialization": "zero_weight_and_bias",
            "shared_between_loops": False, "cache": False,
            "gate_parameter_count": 2 * (width + 1)}


def validate_audio_reinjection(model: Any) -> dict:
    core = model.model
    width = model.config.hidden_size
    if (core.audio_reinject_tokens != 126 or tuple(core.audio_reinject_indices) != (5, 15)
            or tuple(core.audio_linear.weight.shape) != (width, width)
            or tuple(core.audio_linear.bias.shape) != (width,)
            or tuple(core.audio_gate.weight.shape) != (1, width)
            or tuple(core.audio_gate.bias.shape) != (1,)):
        raise ValueError("invalid shared audio reinjection geometry")
    names = {name for name, _ in model.named_parameters()
             if name.startswith(("model.audio_linear.", "model.audio_gate."))}
    if names != {"model.audio_linear.weight", "model.audio_linear.bias",
                 "model.audio_gate.weight", "model.audio_gate.bias"}:
        raise ValueError("expected exactly one shared audio linear and one shared scalar gate")
    return {"audio_tokens": 126, "logical_indices": [5, 15], "per_token_scalar": True,
            "parameter_count": width * width + 2 * width + 1,
            "cache": False, "shared_between_loops": True}


def validate_serial_injection(model: Any) -> dict:
    layer = validate_layer_injection(model)
    audio = validate_audio_reinjection(model)
    if tuple(model.model.layer_inject_indices) != tuple(model.model.audio_reinject_indices):
        raise ValueError("serial branches must use the same two loop entrances")
    return {"order": "input_then_intermediate", "input": audio, "intermediate": layer,
            "intermediate_gate_condition": "audio_hidden_after_input_residual",
            "trace": [[5, "input"], [5, "layer4"], [15, "input"], [15, "layer8"]]}


__all__ = [
    "load_recursive_text_model",
    "validate_recursive_config",
    "validate_loaded_model",
    "validate_layer_injection",
    "validate_audio_reinjection",
    "validate_serial_injection",
    "validate_loop_lora",
]
