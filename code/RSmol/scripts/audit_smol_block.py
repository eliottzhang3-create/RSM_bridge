#!/usr/bin/env python3
"""Deep, read-only audit of one original SmolLM2 Transformer decoder block.

The program deliberately uses the installed Hugging Face implementation rather
than reimplementing a decoder layer.  It records the exact source signature,
module/parameter structure, residual decomposition, and a complete decoder
layer hook trace for a local checkpoint.  It is a GPU audit and never writes to
the checkpoint directory.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
import time
import traceback
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn


DEFAULT_MODEL_PATH = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2")
DEFAULT_OUTPUT_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol")
REQUIRED_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
CONFIG_FIELDS = (
    "model_type",
    "architectures",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rms_norm_eps",
    "rope_theta",
    "max_position_embeddings",
    "vocab_size",
    "torch_dtype",
    "attn_implementation",
    "hidden_act",
    "attention_bias",
    "attention_dropout",
)


def package_version(name: str) -> str:
    try:
        return version(name)
    except Exception as exc:  # pragma: no cover - environment diagnostic
        return f"<unavailable:{type(exc).__name__}:{exc}>"


def json_value(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return str(value)


def canonical_dtype(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).replace("torch.", "").lower()


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def repo_roots() -> tuple[Path, Path]:
    # .../code/RSmol/scripts/audit_smol_block.py -> repository root.
    repo = Path(__file__).resolve().parents[3]
    return repo, repo / "code" / "RSmol"


def require_external_output(path: Path) -> Path:
    resolved = resolve_path(path)
    repo, transfer_dir = repo_roots()
    for forbidden in (repo, transfer_dir):
        try:
            resolved.relative_to(forbidden)
        except ValueError:
            continue
        raise ValueError(
            "Audit output must be outside the Git checkout and transfer directory: "
            f"output={resolved} forbidden={forbidden}"
        )
    if resolved.exists():
        if not resolved.is_dir():
            raise FileExistsError(f"Audit output path is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise FileExistsError(f"Audit refuses to overwrite a non-empty output directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def require_checkpoint(model_path: Path) -> Path:
    model_path = resolve_path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {model_path}")
    missing = [name for name in REQUIRED_FILES if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is incomplete: path={model_path} missing={missing}")
    return model_path


def finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"{name} contains non-finite values")


def tensor_summary(tensor: torch.Tensor, *, include_values: bool = False) -> dict[str, Any]:
    detached = tensor.detach()
    result: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "finite": bool(torch.isfinite(detached).all().item()),
    }
    if detached.numel():
        as_float = detached.float()
        result.update(
            {
                "min": float(as_float.min().item()),
                "max": float(as_float.max().item()),
                "mean": float(as_float.mean().item()),
                "abs_max": float(as_float.abs().max().item()),
            }
        )
    if include_values and detached.numel() <= 4096:
        result["values"] = detached.float().cpu().tolist()
    return result


def module_path(module: nn.Module) -> str:
    return f"{module.__class__.__module__}.{module.__class__.__name__}"


def parameter_record(name: str, parameter: nn.Parameter) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(parameter.shape),
        "dtype": str(parameter.dtype).replace("torch.", ""),
        "numel": int(parameter.numel()),
        "requires_grad": bool(parameter.requires_grad),
        "device": str(parameter.device),
    }


def module_tree(module: nn.Module) -> list[dict[str, Any]]:
    tree: list[dict[str, Any]] = []
    for name, child in module.named_modules():
        tree.append(
            {
                "path": name or "<root>",
                "class": module_path(child),
                "training": bool(child.training),
                "children": list(child._modules),
                "parameters": [
                    {
                        "name": parameter_name,
                        "shape": list(parameter.shape),
                        "dtype": str(parameter.dtype).replace("torch.", ""),
                        "numel": int(parameter.numel()),
                    }
                    for parameter_name, parameter in child.named_parameters(recurse=False)
                ],
            }
        )
    return tree


def structural_signature(module: nn.Module) -> list[tuple[Any, ...]]:
    signature: list[tuple[Any, ...]] = []
    for name, child in module.named_modules():
        signature.append((name, module_path(child), tuple(child._modules)))
        for parameter_name, parameter in child.named_parameters(recurse=False):
            signature.append(
                (
                    f"{name}.{parameter_name}" if name else parameter_name,
                    "parameter",
                    tuple(parameter.shape),
                    str(parameter.dtype),
                    bool(parameter.requires_grad),
                )
            )
    return signature


def structural_signature_digest(signature: list[tuple[Any, ...]]) -> str:
    encoded = json.dumps(signature, sort_keys=True, default=json_value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_details(cls: type[nn.Module]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "class": module_path(cls),
        "module": cls.__module__,
        "source_file": None,
        "forward_signature": None,
        "forward_source": None,
    }
    try:
        result["source_file"] = inspect.getsourcefile(cls)
        result["forward_signature"] = str(inspect.signature(cls.forward))
        result["forward_source"] = inspect.getsource(cls.forward)
    except (OSError, TypeError):
        # A source-less wheel is still auditable through its runtime module
        # tree, but this is explicitly recorded rather than fabricated.
        result["source_unavailable"] = True
    return result


def read_raw_config(model_path: Path) -> dict[str, Any]:
    with (model_path / "config.json").open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("config.json must contain a JSON object")
    return raw


def config_audit(raw: dict[str, Any], config: Any) -> dict[str, Any]:
    auto_dict = config.to_dict()
    # transformers exposes _attn_implementation as a derived runtime field;
    # a checkpoint may only contain the public attn_implementation key.
    values: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    mismatches: dict[str, Any] = {}
    for field in CONFIG_FIELDS:
        if field == "attn_implementation":
            raw_value = raw.get(field, raw.get("_attn_implementation"))
            auto_value = getattr(config, "_attn_implementation", None)
            if auto_value is None:
                auto_value = getattr(config, field, None)
        else:
            raw_value = raw.get(field)
            auto_value = getattr(config, field, None)
            if auto_value is None:
                auto_value = auto_dict.get(field)
        if field == "head_dim" and auto_value is None:
            hidden = getattr(config, "hidden_size", None)
            heads = getattr(config, "num_attention_heads", None)
            if hidden is not None and heads:
                auto_value = int(hidden) // int(heads)
                provenance[field] = "derived_from_hidden_size_div_num_attention_heads"
        if field not in provenance:
            provenance[field] = "config_attribute"
        values[field] = json_value(auto_value)
        if raw_value is not None:
            raw_normalized = canonical_dtype(raw_value) if field == "torch_dtype" else json_value(raw_value)
            auto_normalized = canonical_dtype(auto_value) if field == "torch_dtype" else json_value(auto_value)
            if raw_normalized != auto_normalized:
                mismatches[field] = {"json": raw_normalized, "auto_config": auto_normalized}
    if mismatches:
        raise ValueError(f"config.json and AutoConfig disagree: {mismatches}")
    if values["model_type"] != "llama":
        raise ValueError(f"Expected original SmolLM2 Llama config, got model_type={values['model_type']!r}")
    architectures = values["architectures"] or []
    if "LlamaForCausalLM" not in architectures:
        raise ValueError(f"Expected LlamaForCausalLM architecture, got {architectures!r}")
    required_numeric = (
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
    )
    for field in required_numeric:
        if values[field] is None or int(values[field]) <= 0:
            raise ValueError(f"Config field {field} must be a positive integer, got {values[field]!r}")
    if values["num_attention_heads"] % values["num_key_value_heads"] != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if values["head_dim"] != values["hidden_size"] // values["num_attention_heads"]:
        raise ValueError(
            "head_dim is inconsistent with hidden_size / num_attention_heads: "
            f"{values['head_dim']} vs {values['hidden_size']} / {values['num_attention_heads']}"
        )
    return {
        "raw_config": json_value(raw),
        "selected_fields": values,
        "field_provenance": provenance,
        "auto_config_class": module_path(config.__class__),
        "auto_config_dict": json_value(auto_dict),
        "cross_checked": True,
    }


def model_parameter_inventory(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        named = list(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - old torch compatibility
        named = list(model.named_parameters())
    inventory = [parameter_record(name, parameter) for name, parameter in named]
    by_id: dict[int, list[str]] = defaultdict(list)
    by_id_parameter: dict[int, nn.Parameter] = {}
    for name, parameter in named:
        by_id[id(parameter)].append(name)
        by_id_parameter[id(parameter)] = parameter
    shared = [names for names in by_id.values() if len(names) > 1]
    return inventory, {
        "parameter_count_unique": int(sum(parameter.numel() for parameter in by_id_parameter.values())),
        "parameter_count_references": int(sum(parameter.numel() for _, parameter in named)),
        "unique_parameter_object_count": len(by_id_parameter),
        "shared_parameter_groups": shared,
        "shared_parameter_group_count": len(shared),
        "model_training": bool(model.training),
    }


def checkpoint_manifest(model_path: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for path in sorted(model_path.iterdir()):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return manifest


def _module_parameter(module: nn.Module, name: str) -> dict[str, Any]:
    """Describe a projection child (the attention module owns the Linear)."""

    child = getattr(module, name, None)
    if child is None:
        return {"name": name, "present": False}
    if isinstance(child, (nn.Linear, nn.Embedding)):
        weight = child.weight
        bias = getattr(child, "bias", None)
        return {
            "name": name,
            "present": True,
            "class": module_path(child),
            "weight_shape": list(weight.shape),
            "weight_dtype": str(weight.dtype).replace("torch.", ""),
            "weight_numel": int(weight.numel()),
            "bias_present": bool(bias is not None),
            "bias_shape": list(bias.shape) if bias is not None else None,
            "bias_dtype": str(bias.dtype).replace("torch.", "") if bias is not None else None,
            "bias_numel": int(bias.numel()) if bias is not None else 0,
        }
    if isinstance(child, nn.Parameter):
        return {
            "name": name,
            "present": True,
            "class": module_path(module),
            "shape": list(child.shape),
            "dtype": str(child.dtype).replace("torch.", ""),
            "numel": int(child.numel()),
        }
    return {"name": name, "present": True, "class": module_path(child)}


def structure_audit(model: nn.Module, config_audit_payload: dict[str, Any], selected_index: int) -> dict[str, Any]:
    decoder = getattr(getattr(model, "model", None), "layers", None)
    if decoder is None or not isinstance(decoder, (nn.ModuleList, list, tuple)):
        raise RuntimeError("Loaded model does not expose model.layers decoder blocks")
    if not decoder:
        raise RuntimeError("Loaded model has no decoder layers")
    layer_count = len(decoder)
    if selected_index < 0 or selected_index >= layer_count:
        raise ValueError(f"--layer-index must be in [0,{layer_count}), got {selected_index}")
    if type(decoder[selected_index]).__name__ != "LlamaDecoderLayer":
        raise RuntimeError(
            "Original SmolLM2 audit expects the installed Hugging Face "
            f"LlamaDecoderLayer, got {module_path(decoder[selected_index])}"
        )
    signatures = [structural_signature(layer) for layer in decoder]
    signature_consistent = all(signature == signatures[0] for signature in signatures[1:])
    if not signature_consistent:
        raise RuntimeError("Decoder layer structural signatures are not identical")
    layer = decoder[selected_index]
    layer_parameter_ids: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for index, decoder_layer in enumerate(decoder):
        for name, parameter in decoder_layer.named_parameters():
            layer_parameter_ids[id(parameter)].append((index, name))
    cross_layer_shared = [
        locations for locations in layer_parameter_ids.values() if len({index for index, _ in locations}) > 1
    ]
    attention = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    norm1 = getattr(layer, "input_layernorm", None)
    norm2 = getattr(layer, "post_attention_layernorm", None)
    for name, value in (("input_layernorm", norm1), ("self_attn", attention), ("post_attention_layernorm", norm2), ("mlp", mlp)):
        if value is None:
            raise RuntimeError(f"Representative decoder layer is missing {name}")
    cfg = config_audit_payload["selected_fields"]
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    attention_details = {
        "class": module_path(attention),
        "layer_idx": json_value(getattr(attention, "layer_idx", None)),
        "q_proj": _module_parameter(attention, "q_proj"),
        "k_proj": _module_parameter(attention, "k_proj"),
        "v_proj": _module_parameter(attention, "v_proj"),
        "o_proj": _module_parameter(attention, "o_proj"),
        "num_attention_heads": json_value(getattr(attention, "num_heads", cfg["num_attention_heads"])),
        "num_key_value_heads": json_value(getattr(attention, "num_key_value_heads", cfg["num_key_value_heads"])),
        "num_key_value_groups": json_value(getattr(attention, "num_key_value_groups", None)),
        "head_dim": json_value(getattr(attention, "head_dim", cfg["head_dim"])),
        "scaling": json_value(getattr(attention, "scaling", None)),
        "is_causal": json_value(getattr(attention, "is_causal", True)),
        "attention_dropout_config": json_value(getattr(attention, "attention_dropout", None)),
        "effective_dropout_in_eval": 0.0,
        "dropout_behavior": "eval mode; attention dropout is not applied (effective p=0)",
        "rope": {
            "class": module_path(rotary) if rotary is not None else None,
            "theta": cfg["rope_theta"],
            "module_attributes": {
                name: json_value(getattr(rotary, name))
                for name in ("rope_type", "max_seq_len_cached", "original_max_seq_len", "factor")
                if rotary is not None and hasattr(rotary, name)
            },
            "applied_to": "query_states and key_states before cache/attention scores",
        },
        "causal_mask": "model-level causal mask supplied to self_attn; future key positions receive -inf",
        "forward_signature": str(inspect.signature(attention.forward)),
    }
    for field, expected in (
        ("num_attention_heads", cfg["num_attention_heads"]),
        ("num_key_value_heads", cfg["num_key_value_heads"]),
        ("head_dim", cfg["head_dim"]),
    ):
        actual = attention_details[field]
        if actual is not None and int(actual) != int(expected):
            raise RuntimeError(f"Attention runtime/config mismatch for {field}: {actual} vs {expected}")
    activation = getattr(mlp, "act_fn", None)
    mlp_details = {
        "class": module_path(mlp),
        "gate_proj": _module_parameter(mlp, "gate_proj"),
        "up_proj": _module_parameter(mlp, "up_proj"),
        "down_proj": _module_parameter(mlp, "down_proj"),
        "activation_class": module_path(activation) if isinstance(activation, nn.Module) else type(activation).__name__,
        "activation_repr": repr(activation),
        "activation_formula": "silu(x) = x * sigmoid(x)" if "silu" in repr(activation).lower() else "runtime activation recorded above",
        "forward_signature": str(inspect.signature(mlp.forward)),
    }
    norm_details = {
        "input_layernorm": {
            "class": module_path(norm1),
            "weight": tensor_summary(norm1.weight, include_values=False) if hasattr(norm1, "weight") else None,
            "eps": json_value(getattr(norm1, "variance_epsilon", getattr(norm1, "eps", cfg["rms_norm_eps"]))),
            "bias_present": bool(getattr(norm1, "bias", None) is not None),
            "position": "before self_attn (pre-norm)",
            "formula": "RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight; computation variance is float32 in LlamaRMSNorm",
        },
        "post_attention_layernorm": {
            "class": module_path(norm2),
            "weight": tensor_summary(norm2.weight, include_values=False) if hasattr(norm2, "weight") else None,
            "eps": json_value(getattr(norm2, "variance_epsilon", getattr(norm2, "eps", cfg["rms_norm_eps"]))),
            "bias_present": bool(getattr(norm2, "bias", None) is not None),
            "position": "after attention residual and before mlp (pre-norm MLP)",
            "formula": "RMSNorm(x_after_attn_residual) = (x / sqrt(mean(x^2) + eps)) * weight",
        },
    }
    source = source_details(type(layer))
    layer_parameters: list[dict[str, Any]] = []
    for name, parameter in layer.named_parameters():
        layer_parameters.append(parameter_record(name, parameter))
    return {
        "representative_layer_index": selected_index,
        "decoder_layer_count": layer_count,
        "layer_class": module_path(layer),
        "all_layer_signatures_identical": True,
        "cross_layer_parameter_sharing": {
            "shared_groups": cross_layer_shared,
            "shared_group_count": len(cross_layer_shared),
            "all_decoder_layer_storage_independent": len(cross_layer_shared) == 0,
        },
        "layer_structure_signature": [list(item) for item in signatures[0]],
        "layer_structure_signature_all_hash": [structural_signature_digest(signature) for signature in signatures],
        "module_tree": module_tree(layer),
        "parameter_inventory": layer_parameters,
        "source": source,
        "norm_details": norm_details,
        "attention_details": attention_details,
        "mlp_details": mlp_details,
        "residual_equations": {
            "attention_branch": "r = x_in; a = SelfAttention(input_layernorm(r)); x_after_attn_residual = r + a",
            "mlp_branch": "r2 = x_after_attn_residual; m = MLP(post_attention_layernorm(r2)); x_out = r2 + m",
            "final_output": "x_out = x_in + SelfAttention(RMSNorm(x_in)) + MLP(RMSNorm(x_in + SelfAttention(RMSNorm(x_in))))",
            "logical_order": ["x_in", "input_layernorm", "self_attn", "attention_residual", "post_attention_layernorm", "mlp", "mlp_residual", "x_out"],
        },
    }


def primary_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise RuntimeError(f"Expected tensor or tuple with tensor primary output, got {type(output).__name__}")


def replace_primary(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    raise RuntimeError(f"Cannot override unsupported module output type {type(output).__name__}")


def difference_metrics(lhs: torch.Tensor, rhs: torch.Tensor, *, name: str) -> dict[str, Any]:
    if tuple(lhs.shape) != tuple(rhs.shape):
        raise RuntimeError(f"{name} shape mismatch: {tuple(lhs.shape)} vs {tuple(rhs.shape)}")
    finite_tensor(lhs, f"{name}.lhs")
    finite_tensor(rhs, f"{name}.rhs")
    difference = (lhs.float() - rhs.float()).abs()
    lhs_flat = lhs.float().reshape(-1)
    rhs_flat = rhs.float().reshape(-1)
    cosine = float(torch.nn.functional.cosine_similarity(lhs_flat, rhs_flat, dim=0, eps=1e-12).item())
    return {
        "name": name,
        "shape": list(lhs.shape),
        "max_abs": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean().item()) if difference.numel() else 0.0,
        "cosine": cosine,
        "finite": True,
        "allclose_atol_1e-5_rtol_1e-4": bool(torch.allclose(lhs.float(), rhs.float(), atol=1e-5, rtol=1e-4)),
    }


def _run_inputs(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
    return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)


def decomposition_audit(model: nn.Module, *, layer_index: int, device: torch.device, seed: int, dtype: torch.dtype, seq_len: int) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    vocab_size = int(model.config.vocab_size)
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    layer = model.model.layers[layer_index]
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str, output: Any) -> None:
        captured[name] = primary_output(output).detach().clone()

    def pre_capture(name: str) -> Callable[..., None]:
        def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError(f"Could not capture tensor input for {name}")
            captured[name] = args[0].detach().clone()
        return hook

    handles = [
        layer.register_forward_pre_hook(pre_capture("x_in")),
        layer.input_layernorm.register_forward_hook(lambda _m, _i, o: capture("norm1", o)),
        layer.self_attn.register_forward_hook(lambda _m, _i, o: capture("attn_out", o)),
        layer.post_attention_layernorm.register_forward_pre_hook(pre_capture("x_after_attn_residual")),
        layer.post_attention_layernorm.register_forward_hook(lambda _m, _i, o: capture("norm2", o)),
        layer.mlp.register_forward_hook(lambda _m, _i, o: capture("mlp_out", o)),
        layer.register_forward_hook(lambda _m, _i, o: capture("x_out", o)),
    ]
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = _run_inputs(model, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"x_in", "norm1", "attn_out", "x_after_attn_residual", "norm2", "mlp_out", "x_out"}:
        raise RuntimeError(f"Incomplete block decomposition capture: got={sorted(captured)}")
    for name, value in captured.items():
        finite_tensor(value, name)
    residual_metrics = [
        difference_metrics(captured["x_after_attn_residual"], captured["x_in"] + captured["attn_out"], name="attention_residual_reconstruction"),
        difference_metrics(captured["x_out"], captured["x_after_attn_residual"] + captured["mlp_out"], name="mlp_residual_reconstruction"),
    ]
    for metric in residual_metrics:
        if not metric["allclose_atol_1e-5_rtol_1e-4"]:
            raise FloatingPointError(f"Residual reconstruction failed: {metric}")
    norm_metrics = {
        "norm1": difference_metrics(captured["norm1"], captured["x_in"], name="norm1_vs_input"),
        "norm2": difference_metrics(captured["norm2"], captured["x_after_attn_residual"], name="norm2_vs_attention_residual"),
    }
    if any(metric["max_abs"] <= 1e-8 for metric in norm_metrics.values()):
        raise RuntimeError(f"RMSNorm unexpectedly behaved as identity: {norm_metrics}")
    numeric = {
        "seed": seed,
        "dtype": str(dtype).replace("torch.", ""),
        "input_ids": input_ids.cpu().tolist(),
        "input_shape": list(input_ids.shape),
        "attention_mask_shape": list(attention_mask.shape),
        "captured": {name: tensor_summary(value) for name, value in captured.items()},
        "residual_reconstruction": residual_metrics,
        "norm_non_identity": norm_metrics,
        "model_logits": tensor_summary(output.logits),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not output.logits.shape[-1] == vocab_size:
        raise RuntimeError(f"Unexpected logits vocab dimension: {tuple(output.logits.shape)}")
    finite_tensor(output.logits, "model logits")
    return numeric, {"input_ids": input_ids, "attention_mask": attention_mask, "baseline": captured}


def controlled_ablation_audit(model: nn.Module, *, layer_index: int, device: torch.device, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    layer = model.model.layers[layer_index]
    baseline = inputs["baseline"]
    results: dict[str, Any] = {}

    def run_with_zero(target: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        captured: dict[str, torch.Tensor] = {}
        handles = []

        def save_pre(name: str) -> Callable[..., None]:
            def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
                if not args or not isinstance(args[0], torch.Tensor):
                    raise RuntimeError(f"Could not capture {name} input")
                captured[name] = args[0].detach().clone()
            return hook

        def zero_output(_module: nn.Module, module_input: tuple[Any, ...], output: Any) -> Any:
            source = primary_output(output)
            return replace_primary(output, torch.zeros_like(source))

        handles.append(layer.register_forward_pre_hook(save_pre("x_in")))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(save_pre("x_after_attn_residual")))
        handles.append(layer.self_attn.register_forward_hook(zero_output if target == "attention" else lambda *_: None))
        handles.append(layer.mlp.register_forward_hook(zero_output if target == "mlp" else lambda *_: None))
        try:
            with torch.inference_mode():
                output = _run_inputs(model, inputs["input_ids"], inputs["attention_mask"])
        finally:
            for handle in handles:
                handle.remove()
        return captured, output.logits.detach().clone()

    # With MLP output ablated, the selected block output must be its attention
    # residual.  Use a temporary block hook to capture the actual output.
    block_output: dict[str, torch.Tensor] = {}
    handle = layer.register_forward_hook(lambda _m, _i, o: block_output.setdefault("x_out", primary_output(o).detach().clone()))
    try:
        with torch.inference_mode():
            # Re-run using a dedicated MLP-zero hook while keeping the capture.
            mlp_handle = layer.mlp.register_forward_hook(lambda _m, _i, o: replace_primary(o, torch.zeros_like(primary_output(o))))
            try:
                _run_inputs(model, inputs["input_ids"], inputs["attention_mask"])
            finally:
                mlp_handle.remove()
    finally:
        handle.remove()
    results["zero_mlp"] = difference_metrics(block_output["x_out"], baseline["x_after_attn_residual"], name="zero_mlp_output_vs_attention_residual")
    if not results["zero_mlp"]["allclose_atol_1e-5_rtol_1e-4"]:
        raise FloatingPointError(f"Zero-MLP controlled check failed: {results['zero_mlp']}")
    results["zero_mlp"]["expected"] = "selected block output equals attention residual when MLP output is replaced with zero"
    zero_attn_capture, _ = run_with_zero("attention")
    results["zero_attention"] = {
        "captured_input": tensor_summary(zero_attn_capture["x_in"]),
        "captured_attention_residual_input": tensor_summary(zero_attn_capture["x_after_attn_residual"]),
        "expected_attention_residual": "x_after_attn_residual equals x_in when attention output is replaced with zero",
        "reconstruction": difference_metrics(zero_attn_capture["x_after_attn_residual"], zero_attn_capture["x_in"], name="zero_attention_residual"),
    }
    if not results["zero_attention"]["reconstruction"]["allclose_atol_1e-5_rtol_1e-4"]:
        raise FloatingPointError(f"Zero-attention controlled check failed: {results['zero_attention']}")
    results["output_shapes"] = {
        "attention_output": list(baseline["attn_out"].shape),
        "mlp_output": list(baseline["mlp_out"].shape),
        "block_input": list(baseline["x_in"].shape),
        "block_output": list(baseline["x_out"].shape),
    }
    return results


def full_forward_trace(model: nn.Module, *, device: torch.device, seed: int, seq_len: int) -> dict[str, Any]:
    torch.manual_seed(seed + 17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 17)
    input_ids = torch.randint(0, int(model.config.vocab_size), (1, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    trace: list[int] = []
    handles = []
    for index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(lambda _m, _i, _o, index=index: trace.append(index)))
    try:
        with torch.inference_mode():
            output = _run_inputs(model, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    expected = list(range(len(model.model.layers)))
    if trace != expected:
        raise RuntimeError(f"Full model decoder trace mismatch: expected={expected} got={trace}")
    finite_tensor(output.logits, "full model logits")
    return {
        "trace": trace,
        "expected_trace": expected,
        "all_layers_once": True,
        "input_shape": list(input_ids.shape),
        "logits_shape": list(output.logits.shape),
        "logits": tensor_summary(output.logits),
        "complete_embedding_to_lm_head_forward": True,
    }


def environment_audit(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in ("torch", "transformers", "tokenizers", "safetensors", "accelerate")},
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        result["gpu_index"] = device.index if device.index is not None else torch.cuda.current_device()
        result["gpu_name"] = torch.cuda.get_device_name(result["gpu_index"])
        result["gpu_capability"] = list(torch.cuda.get_device_capability(result["gpu_index"]))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seq_len < 4:
        raise ValueError(f"--seq-len must be at least 4, got {args.seq_len}")
    if args.seed < 0:
        raise ValueError(f"--seed must be non-negative, got {args.seed}")
    model_path = require_checkpoint(args.model_path)
    output_dir = require_external_output(args.output_dir or (DEFAULT_OUTPUT_ROOT / f"smol-block-audit-{time.strftime('%Y%m%d_%H%M%S')}"))
    if not torch.cuda.is_available():
        raise RuntimeError("This audit requires CUDA and must run through the pdgpu-5090 submit wrapper")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"This audit is GPU-only; got device={device}")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    selected_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print("========== SMOLLM2-135M TRANSFORMER BLOCK AUDIT ==========")
    print(f"[model] path={model_path} local_files_only=True")
    print(f"[audit] layer_index={args.layer_index} dtype={args.dtype} seq_len={args.seq_len} seed={args.seed}")
    from transformers import AutoConfig, AutoModelForCausalLM

    raw_config = read_raw_config(model_path)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config_payload = config_audit(raw_config, config)
    print(f"[config] {json.dumps(config_payload['selected_fields'], ensure_ascii=False)}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        torch_dtype=selected_dtype,
        device_map={"": str(device)},
    )
    model.eval()
    structure = structure_audit(model, config_payload, args.layer_index)
    inventory, inventory_summary = model_parameter_inventory(model)
    numeric, decomposition_inputs = decomposition_audit(
        model,
        layer_index=args.layer_index,
        device=device,
        seed=args.seed,
        dtype=selected_dtype,
        seq_len=args.seq_len,
    )
    ablation = controlled_ablation_audit(model, layer_index=args.layer_index, device=device, inputs=decomposition_inputs)
    trace = full_forward_trace(model, device=device, seed=args.seed, seq_len=args.seq_len)
    report: dict[str, Any] = {
        "status": "PASS",
        "architecture_contract": {
            "name": "original_smollm2_llama_decoder_block",
            "model_type": config_payload["selected_fields"]["model_type"],
            "logical_decoder_layer_count": config_payload["selected_fields"]["num_hidden_layers"],
            "physical_decoder_layer_count": len(model.model.layers),
            "layer_execution": "each decoder layer executes once in ascending order",
            "representative_layer_index": args.layer_index,
        },
        "model_path": str(model_path),
        "config": config_payload,
        "representative_layer_index": args.layer_index,
        "layer_structure_signature": structure["layer_structure_signature"],
        "module_tree": structure["module_tree"],
        "parameter_inventory": inventory,
        "parameter_inventory_summary": inventory_summary,
        "parameter_sharing": {
            "all_model_parameter_objects_unique": inventory_summary["shared_parameter_group_count"] == 0,
            "shared_parameter_groups": inventory_summary["shared_parameter_groups"],
            "note": "Any tied embedding/lm_head storage is recorded; decoder layers are checked structurally and by object identity.",
        },
        "layer_parameter_inventory": structure["parameter_inventory"],
        "layer_structure": {
            "class": structure["layer_class"],
            "decoder_layer_count": structure["decoder_layer_count"],
            "all_layer_signatures_identical": structure["all_layer_signatures_identical"],
            "signature_hashes": structure["layer_structure_signature_all_hash"],
            "cross_layer_parameter_sharing": structure["cross_layer_parameter_sharing"],
        },
        "source_and_api": structure["source"],
        "norm_details": structure["norm_details"],
        "attention_details": structure["attention_details"],
        "mlp_details": structure["mlp_details"],
        "residual_equations": structure["residual_equations"],
        "forward_trace": trace,
        "numeric_decomposition": numeric,
        "ablation_results": ablation,
        "environment": environment_audit(device),
        "checkpoint_manifest": checkpoint_manifest(model_path),
        "runtime": {
            "model_class": module_path(model),
            "model_eval": not model.training,
            "selected_dtype": args.dtype,
            "local_files_only": True,
            "model_writeback": False,
        },
    }
    write_report(output_dir, report)
    print(f"[result] status=PASS output_dir={output_dir}")
    return report


def write_report(output_dir: Path, report: dict[str, Any]) -> None:
    json_path = output_dir / "smol_block_audit.json"
    markdown_path = output_dir / "smol_block_audit.md"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_value) + "\n", encoding="utf-8")
    temporary.replace(json_path)
    markdown = render_markdown(report)
    markdown_tmp = markdown_path.with_suffix(".md.tmp")
    markdown_tmp.write_text(markdown, encoding="utf-8")
    markdown_tmp.replace(markdown_path)


def render_markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]["selected_fields"]
    trace = report["forward_trace"]
    numeric = report["numeric_decomposition"]
    residuals = numeric["residual_reconstruction"]
    lines = [
        "# SmolLM2-135M Transformer Block Audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Checkpoint: `{report['model_path']}`",
        f"- Representative layer: `{report['representative_layer_index']}`",
        f"- Model class: `{report['runtime']['model_class']}`",
        "",
        "## Configuration",
        "",
    ]
    for key in CONFIG_FIELDS:
        lines.append(f"- `{key}`: `{cfg.get(key)}`")
    lines += [
        "",
        "## Exact forward order",
        "",
        "```text",
        "x_in",
        "  -> input_layernorm (RMSNorm, pre-norm)",
        "  -> self_attn (Q/K/V projections, RoPE, causal attention, o_proj)",
        "  -> x_after_attn_residual = x_in + attn_out",
        "  -> post_attention_layernorm (RMSNorm, pre-norm MLP)",
        "  -> mlp (gate_proj/up_proj/down_proj and activation)",
        "  -> x_out = x_after_attn_residual + mlp_out",
        "```",
        "",
        "The final block equation is:",
        "",
        "```text",
        "x_out = x_in + SelfAttention(RMSNorm(x_in))",
        "        + MLP(RMSNorm(x_in + SelfAttention(RMSNorm(x_in))))",
        "```",
        "",
        "## Numeric decomposition",
        "",
        f"- Input shape: `{numeric['input_shape']}`; dtype: `{numeric['dtype']}`",
        f"- Attention residual max abs error: `{residuals[0]['max_abs']}`",
        f"- MLP residual max abs error: `{residuals[1]['max_abs']}`",
        f"- Full logits shape: `{trace['logits_shape']}`; finite: `{trace['logits']['finite']}`",
        "",
        "## Full model trace",
        "",
        f"- Executed trace: `{trace['trace']}`",
        f"- All layers once in ascending order: `{trace['all_layers_once']}`",
        "",
        "See `smol_block_audit.json` for complete module tree, parameter inventory, "
        "norm/attention/MLP details, source signature, cache/checkpoint manifest, "
        "and numeric tensors/statistics.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except Exception:
        traceback.print_exc()
        print("[result] status=FAIL", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
