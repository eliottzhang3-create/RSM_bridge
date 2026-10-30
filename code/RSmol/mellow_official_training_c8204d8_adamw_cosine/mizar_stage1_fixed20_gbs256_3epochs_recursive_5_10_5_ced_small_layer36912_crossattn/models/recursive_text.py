"""Explicit loader for the fixed 5+10x2+5 non-MeSH text checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from contract import TEXT_MODEL_SCHEDULE, validate_recursive_config_values


def _import_recursive_model() -> tuple[Any, Any]:
    """Use the non-MeSH implementation explicitly, independent of AutoModel registration."""
    rsmol_root = Path(__file__).resolve().parents[3]
    if str(rsmol_root) not in sys.path:
        sys.path.insert(0, str(rsmol_root))
    from models.recursive_crossattn import (
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
    model_cls, parameter_audit = _import_recursive_model()
    model, loading_info = model_cls.from_pretrained(
        path, local_files_only=True, output_loading_info=True,
    )
    failures = {key: loading_info.get(key) for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if loading_info.get(key)}
    if failures:
        raise ValueError(f"non-MeSH text checkpoint weights do not match the model: {failures}")
    validate_loaded_model(model, parameter_audit(model))
    model.model.enable_audio_cross_attention()
    model.config.use_cache = False
    model.generation_config.use_cache = False
    validate_audio_cross_attention(model)
    return model


__all__ = [
    "load_recursive_text_model",
    "validate_recursive_config",
    "validate_loaded_model",
]


def validate_audio_cross_attention(model):
    """Audit actual modules, rather than trusting serialized architecture labels."""
    from contract import cross_attention_contract
    core = model.model
    adapter = core.audio_crossattn
    contract = cross_attention_contract()
    if (adapter.width != 576 or adapter.text_start != 127 or adapter.audio_tokens != 126
            or adapter.prompt_tokens != 256 or adapter.cross_attention.num_heads != 9
            or adapter.cross_attention.dropout != 0. or not adapter.cross_attention.batch_first):
        raise ValueError("cross attention geometry/heads/dropout contract mismatch")
    if type(adapter.residual_scale) is not float or adapter.residual_scale != 0.3:
        raise ValueError("cross attention residual scale must be the fixed float 0.3")
    if len(adapter.audio_input_norms) != 4 or len(adapter.audio_output_norms) != 4:
        raise ValueError("each of the four audio branches requires independent pre/post LayerNorm")
    norms = [(norm, 1536, True) for norm in adapter.audio_input_norms]
    norms += [(norm, 576, True) for norm in adapter.audio_output_norms]
    norms += [(adapter.text_norm, 576, True), (adapter.output_norm, 576, False)]
    if len({id(norm) for norm, _, _ in norms}) != len(norms):
        raise ValueError("audio depth norms and shared text/output norms must be independent modules")
    for norm, width, affine in norms:
        if (tuple(norm.normalized_shape) != (width,) or norm.eps != 1e-5
                or norm.elementwise_affine is not affine):
            raise ValueError("cross attention LayerNorm shape/epsilon/affine contract mismatch")
    expected = {"gate1.weight": (4, 576), "gate1.bias": (4,),
                "gate2.weight": (1, 576), "gate2.bias": (1,),
                "round_bias.0": (576,), "round_bias.1": (576,),
                "cross_attention.in_proj_weight": (1728, 576),
                "cross_attention.in_proj_bias": (1728,),
                "cross_attention.out_proj.weight": (576, 576),
                "cross_attention.out_proj.bias": (576,)}
    for index in range(4):
        expected[f"projections.{index}.weight"] = (576, 1536)
        expected[f"projections.{index}.bias"] = (576,)
        for component, width in (("audio_input_norms", 1536), ("audio_output_norms", 576)):
            expected[f"{component}.{index}.weight"] = (width,)
            expected[f"{component}.{index}.bias"] = (width,)
    expected["text_norm.weight"] = (576,)
    expected["text_norm.bias"] = (576,)
    actual = dict(adapter.named_parameters())
    if set(actual) != set(expected) or any(tuple(actual[name].shape) != shape for name, shape in expected.items()):
        raise ValueError("cross attention parameter inventory mismatch")
    if len({id(p) for p in actual.values()}) != len(actual):
        raise ValueError("the four projections and two round biases must be independent")
    count = sum(p.numel() for p in actual.values())
    if count != contract["new_parameters"] or any(not p.requires_grad for p in actual.values()):
        raise ValueError("new cross attention parameters/count/trainability mismatch")
    return {**contract, "actual_parameters": count, "actual_parameter_tensors": len(actual),
            "physical_layers": len(core.layers), "schedule": list(core.logical_to_physical)}
