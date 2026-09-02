#!/usr/bin/env python3
"""Stage-1 audit for the dynamic SmolLM2 5-10xr-5 checkpoint.

The audit distinguishes the complete backward path through ``r`` middle
calls from the four calls whose shared Parameters receive gradient edges.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from recursive_model_5_10xr_5 import (  # noqa: E402
    DEFAULT_INFERENCE_MIDDLE_LOOPS,
    LOGICAL_LAYER_COUNT,
    MAX_MIDDLE_LOOPS,
    MIN_MIDDLE_LOOPS,
    PARAMETER_GRADIENT_TAIL_LOOPS,
    PHYSICAL_LAYER_COUNT,
    RecursiveLlama5_10xr_5ForCausalLM,
    build_5_10xr_5_schedule,
    parameter_audit,
    register_auto_class,
)

REMOTE_CHECKOUT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM")


def ensure_external_report(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    for root in (SCRIPT_ROOT.parents[1].resolve(), REMOTE_CHECKOUT.resolve()):
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        raise ValueError(f"Stage 1 report must be outside Git checkout: {candidate}")
    return candidate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args(argv)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _layers(model: Any) -> Any:
    base = getattr(model, "model", model)
    return base.model.layers if hasattr(base, "model") else base.layers


def _trace_forward(model: Any, input_ids: torch.Tensor, r: int) -> dict[str, Any]:
    sequence: list[int] = []
    handles = [layer.register_forward_hook(lambda _m, _i, _o, index=index: sequence.append(index)) for index, layer in enumerate(_layers(model))]
    try:
        with torch.inference_mode():
            result = model(input_ids=input_ids, use_cache=False, middle_loop_count=r)
    finally:
        for handle in handles:
            handle.remove()
    expected = list(build_5_10xr_5_schedule(r))
    if sequence != expected:
        raise AssertionError(f"forward trace mismatch for r={r}: expected={expected} got={sequence}")
    return {"r": r, "trace": sequence, "expected": expected, "length": len(sequence), "ok": True, "logits_shape": list(result.logits.shape)}


def _backward_audit(model: Any, input_ids: torch.Tensor, r: int) -> dict[str, Any]:
    layers = _layers(model)
    sequence: list[int] = []
    handles = [layer.register_full_backward_hook(lambda _m, _gi, _go, index=index: sequence.append(index)) for index, layer in enumerate(layers)]
    previous_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad():
            result = model(input_ids=input_ids, use_cache=False, middle_loop_count=r)
            labels = input_ids.clone()
            logits = result.logits.float()
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), reduction="mean")
            if not torch.isfinite(loss):
                raise AssertionError("backward loss is non-finite")
            loss.backward()
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous_training)
    expected = list(reversed(build_5_10xr_5_schedule(r)))
    if sequence != expected:
        raise AssertionError(f"backward trace mismatch for r={r}: expected={expected} got={sequence}")
    audit_state = dict(getattr(getattr(model, "model", model), "_last_forward_audit", {}))
    expected_tail = list(range(r - PARAMETER_GRADIENT_TAIL_LOOPS + 1, r + 1))
    if audit_state.get("parameter_gradient_enabled_middle_loops") != expected_tail:
        raise AssertionError(f"parameter-gradient tail mismatch: {audit_state}")
    if audit_state.get("backward_traversed_middle_loops") != list(range(1, r + 1)):
        raise AssertionError(f"backward middle traversal mismatch: {audit_state}")
    gradient_norms: dict[str, float] = {}
    for index, layer in enumerate(layers):
        norms = [float(parameter.grad.detach().float().norm().item()) for parameter in layer.parameters() if parameter.grad is not None]
        if not norms or not all(torch.isfinite(torch.tensor(value)).item() and value > 0 for value in norms):
            raise AssertionError(f"layer {index} has missing/non-finite/zero gradient")
        gradient_norms[str(index)] = max(norms)
    return {
        "r": r,
        "trace": sequence,
        "expected": expected,
        "loss": float(loss.detach().item()),
        "backward_traversed_loops": list(range(1, r + 1)),
        "parameter_gradient_enabled_loops": expected_tail,
        "early_parameter_gradient_disabled_loops": list(range(1, r - PARAMETER_GRADIENT_TAIL_LOOPS + 1)),
        "gradient_norms_by_physical_layer": gradient_norms,
        "hidden_state_path_preserved": bool(result.logits.requires_grad),
        "ok": True,
    }


def _cache_slot_state(cache: Any, index: int) -> dict[str, Any]:
    length = int(cache.get_seq_length(index))
    key_state = value_state = None
    if hasattr(cache, "layers") and index < len(cache.layers):
        key_state = getattr(cache.layers[index], "keys", None)
        value_state = getattr(cache.layers[index], "values", None)
    elif hasattr(cache, "key_cache") and index < len(cache.key_cache):
        key_state = cache.key_cache[index]
        value_state = cache.value_cache[index]
    return {"sequence_length": length, "key_nonempty": bool(isinstance(key_state, torch.Tensor) and key_state.numel()), "value_nonempty": bool(isinstance(value_state, torch.Tensor) and value_state.numel())}


def _cache_audit(model: Any, input_ids: torch.Tensor, r: int) -> dict[str, Any]:
    with torch.inference_mode():
        no_cache = model(input_ids=input_ids, use_cache=False, middle_loop_count=r)
        cached = model(input_ids=input_ids, use_cache=True, middle_loop_count=r)
    if not torch.allclose(no_cache.logits, cached.logits, atol=1e-5, rtol=1e-4):
        raise AssertionError(f"cache/no-cache logits disagree for r={r}")
    cache = cached.past_key_values
    expected_slots = len(build_5_10xr_5_schedule(r))
    if len(cache) < expected_slots:
        raise AssertionError(f"cache has {len(cache)} slots; expected at least {expected_slots}")
    slots = [_cache_slot_state(cache, index) for index in range(expected_slots)]
    if not all(slot["sequence_length"] == input_ids.shape[1] and slot["key_nonempty"] and slot["value_nonempty"] for slot in slots):
        raise AssertionError(f"invalid cache slots for r={r}: {slots}")
    next_ids = input_ids[:, -1:]
    with torch.inference_mode():
        incremental = model(input_ids=next_ids, past_key_values=cache, use_cache=True, middle_loop_count=r)
        full = model(input_ids=torch.cat((input_ids, next_ids), dim=1), use_cache=False, middle_loop_count=r)
    if not torch.allclose(incremental.logits[:, -1], full.logits[:, -1], atol=1e-3, rtol=1e-3):
        raise AssertionError(f"incremental cache logits disagree for r={r}")
    mismatch_error = None
    try:
        model(input_ids=input_ids[:, :1], past_key_values=cache, use_cache=True, middle_loop_count=5 if r != 5 else 4)
    except ValueError as exc:
        mismatch_error = str(exc)
    if mismatch_error is None:
        raise AssertionError("cache r mismatch was not rejected")
    return {"r": r, "expected_slots": expected_slots, "actual_slots": len(cache), "slot_state": slots, "incremental_ok": True, "cache_r_mismatch_rejected": True, "cache_r_mismatch_error": mismatch_error, "no_cache_cache_logits_allclose": True}


def _generation_audit(model: Any, input_ids: torch.Tensor, max_new_tokens: int) -> dict[str, Any]:
    with torch.inference_mode():
        generated = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, middle_loop_count=7, eos_token_id=None, pad_token_id=0)
        default = model.generate(input_ids=input_ids, max_new_tokens=1, do_sample=False, use_cache=False, eos_token_id=None, pad_token_id=0)
    if generated.ndim != 2 or generated.shape[1] <= input_ids.shape[1]:
        raise AssertionError("generation did not append tokens")
    default_audit = dict(getattr(getattr(model, "model", model), "_last_forward_audit", {}))
    if default_audit.get("middle_loop_count") != DEFAULT_INFERENCE_MIDDLE_LOOPS:
        raise AssertionError(f"inference default r is not 7: {default_audit}")
    return {"output_shape": list(generated.shape), "default_inference_r": default_audit.get("middle_loop_count"), "default_output_shape": list(default.shape), "max_new_tokens": max_new_tokens}


def package_version(name: str) -> str:
    try:
        return version(name)
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if (MIN_MIDDLE_LOOPS, MAX_MIDDLE_LOOPS, PARAMETER_GRADIENT_TAIL_LOOPS) != (4, 7, 4):
        raise AssertionError("invalid 5-10xr-5 audit constants")
    model_path = args.model_path.expanduser().resolve()
    output_report = ensure_external_report(args.output_report)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Recursive 5-10xr-5 smoke requires a submitted CUDA job")
    register_auto_class()
    model = RecursiveLlama5_10xr_5ForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval()
    config = model.config
    if (int(getattr(config, "num_hidden_layers", -1)), int(getattr(config, "recursive_layer_count", -1))) != (80, 20):
        raise AssertionError("checkpoint must declare max logical depth 80 and 20 physical layers")
    if len(_layers(model)) != PHYSICAL_LAYER_COUNT:
        raise AssertionError("checkpoint does not own exactly 20 physical decoder modules")
    if len({id(layer) for layer in _layers(model)}) != 20:
        raise AssertionError("physical decoder modules are not unique")
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    traces = [_trace_forward(model, input_ids, r) for r in range(4, 8)]
    backwards = [_backward_audit(model, input_ids, r) for r in range(4, 8)]
    caches = [_cache_audit(model, input_ids, r) for r in range(4, 8)]
    generation = _generation_audit(model, input_ids, args.max_new_tokens)
    with tempfile.TemporaryDirectory(prefix="rsmol-5-10xr-5-reload-") as temp_dir:
        model.save_pretrained(temp_dir, safe_serialization=False)
        reloaded = RecursiveLlama5_10xr_5ForCausalLM.from_pretrained(temp_dir, local_files_only=True).to(device).eval()
        with torch.inference_mode():
            before = model(input_ids=input_ids, use_cache=False, middle_loop_count=7).logits.float()
            after = reloaded(input_ids=input_ids, use_cache=False, middle_loop_count=7).logits.float()
        if not torch.allclose(before, after, atol=1e-5, rtol=1e-4):
            raise AssertionError("save/reload logits mismatch")
    report = {
        "status": "PASS", "variant": "SmolLM2-5-10xr-5", "model_path": str(model_path),
        "logical_layer_count_max": LOGICAL_LAYER_COUNT, "physical_layer_count": PHYSICAL_LAYER_COUNT,
        "r_values_audited": [4, 5, 6, 7], "default_inference_r": 7,
        "fixed_parameter_gradient_tail_loops": 4, "physical_module_unique": True,
        "parameter_audit": parameter_audit(model), "forward_trace_audit": traces,
        "backward_trace_audit": backwards, "cache_audit": caches,
        "generation_audit": generation, "save_reload_audit": {"ok": True},
        "transformers": package_version("transformers"), "torch": torch.__version__,
    }
    atomic_json(output_report, report)
    print(f"[result] status=PASS report={output_report}", flush=True)


if __name__ == "__main__":
    main()
