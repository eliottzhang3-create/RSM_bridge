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
from collections import Counter
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
    MIDDLE_LAYER_COUNT,
    MAX_MIDDLE_LOOPS,
    MIN_MIDDLE_LOOPS,
    PARAMETER_GRADIENT_TAIL_LOOPS,
    PHYSICAL_LAYER_COUNT,
    RecursiveLlama5_10xr_5ForCausalLM,
    SAMPLER_KEY,
    SAMPLER_VERSION,
    POISSON_LAMBDA,
    POISSON_SUPPORT,
    POISSON_NORMALIZATION_Z,
    POISSON_PROBABILITIES,
    SAMPLING_POLICY,
    build_5_10xr_5_schedule,
    make_dynamic_cache,
    parameter_audit,
    register_auto_class,
)

REMOTE_CHECKOUT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM")
CACHE_ATOL = 1e-5
CACHE_RTOL = 1e-4
BF16_INCREMENTAL_MAX_ABS = 1.0
BF16_INCREMENTAL_MIN_COSINE = 0.999
FP32_INCREMENTAL_ATOL = 1e-3
FP32_INCREMENTAL_RTOL = 1e-4


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
            result = model(
                input_ids=input_ids,
                use_cache=False,
                middle_loop_count=r,
            )
    finally:
        for handle in handles:
            handle.remove()
    expected = list(build_5_10xr_5_schedule(r))
    if sequence != expected:
        raise AssertionError(f"forward trace mismatch for r={r}: expected={expected} got={sequence}")
    return {"r": r, "trace": sequence, "expected": expected, "length": len(sequence), "ok": True, "logits_shape": list(result.logits.shape)}


def _backward_audit(model: Any, input_ids: torch.Tensor, r: int) -> dict[str, Any]:
    layers = _layers(model)
    parameter_identity_before = {
        name: (id(parameter), bool(parameter.requires_grad))
        for name, parameter in model.named_parameters()
    }
    sequence: list[int] = []
    handles = [layer.register_full_backward_hook(lambda _m, _gi, _go, index=index: sequence.append(index)) for index, layer in enumerate(layers)]
    previous_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    recursive_model = getattr(model, "model", model)
    recursive_model._collect_middle_gradient_audit = True
    early_hidden_gradient_norms: dict[str, float] = {}
    early_parameter_gradient_edges_absent = True
    try:
        with torch.enable_grad():
            result = model(
                input_ids=input_ids,
                use_cache=False,
                middle_loop_counts=torch.full(
                    (input_ids.shape[0],), r, dtype=torch.long, device=input_ids.device
                ),
            )
            middle_call_audit = list(getattr(recursive_model, "_last_middle_gradient_audit", ()))
            expected_entries = r * MIDDLE_LAYER_COUNT
            if len(middle_call_audit) != expected_entries:
                raise AssertionError(
                    f"middle-call audit length mismatch for r={r}: "
                    f"expected {expected_entries} (10r) got {len(middle_call_audit)}"
                )
            for loop in range(1, r + 1):
                loop_details = [detail for detail in middle_call_audit if int(detail["loop"]) == loop]
                physical_trace = [int(detail["physical_index"]) for detail in loop_details]
                if physical_trace != list(range(5, 15)):
                    raise AssertionError(
                        f"middle loop {loop} must contain physical layers 5..14 exactly once; "
                        f"got {physical_trace}"
                    )
                for detail in loop_details:
                    physical = int(detail["physical_index"])
                    key = f"{loop}:{physical}"
                    expected_enabled = loop > r - PARAMETER_GRADIENT_TAIL_LOOPS
                    if bool(detail["parameter_grad_enabled"]) != expected_enabled:
                        raise AssertionError(f"parameter-gradient flag mismatch for {key}")
                    if expected_enabled:
                        continue
                    probe = detail["output"].float().square().mean()
                    input_grad = torch.autograd.grad(
                        probe, detail["input"], retain_graph=True, allow_unused=True
                    )[0]
                    if input_grad is None or not torch.isfinite(input_grad).all() or float(input_grad.norm().item()) <= 0:
                        raise AssertionError(
                            f"early middle loop {key} lost its hidden-state gradient path"
                        )
                    parameter_grads = torch.autograd.grad(
                        probe, tuple(detail["layer"].parameters()), retain_graph=True, allow_unused=True
                    )
                    if any(gradient is not None for gradient in parameter_grads):
                        early_parameter_gradient_edges_absent = False
                        raise AssertionError(
                            f"early middle loop {key} unexpectedly has parameter-gradient edges"
                        )
                    early_hidden_gradient_norms[key] = float(input_grad.norm().item())
            # The autograd.grad probes above intentionally execute partial
            # backward traversals and therefore fire module backward hooks.
            # They are audit traffic, not part of the subsequent loss.backward
            # trace.  Keep their count for the report, then isolate the formal
            # backward trace by clearing the hook buffer.
            probe_backward_hook_count = len(sequence)
            sequence.clear()
            labels = input_ids.clone()
            logits = result.logits.float()
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), reduction="mean")
            if not torch.isfinite(loss):
                raise AssertionError("backward loss is non-finite")
            loss.backward()
    finally:
        recursive_model._collect_middle_gradient_audit = False
        for handle in handles:
            handle.remove()
        model.train(previous_training)
    expected = list(reversed(build_5_10xr_5_schedule(r)))
    if len(sequence) != len(expected) or Counter(sequence) != Counter(expected):
        raise AssertionError(
            f"backward trace coverage mismatch for r={r}: "
            f"expected_counts={dict(Counter(expected))} got_counts={dict(Counter(sequence))}"
        )
    backward_hook_order = "exact_reverse_schedule" if sequence == expected else "functional_call_early_hook_order"
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
    parameter_identity_after = {
        name: (id(parameter), bool(parameter.requires_grad))
        for name, parameter in model.named_parameters()
    }
    if parameter_identity_after != parameter_identity_before:
        raise AssertionError("parameter identity/requires_grad state changed during selective BPTT")
    prefix_layers_with_grad = [
        index for index in range(5) if str(index) in gradient_norms and gradient_norms[str(index)] > 0
    ]
    if prefix_layers_with_grad != [0, 1, 2, 3, 4]:
        raise AssertionError(
            f"prefix layers must all receive finite nonzero gradients: {prefix_layers_with_grad}"
        )
    suffix_layers_with_grad = [
        index for index in range(15, 20) if str(index) in gradient_norms and gradient_norms[str(index)] > 0
    ]
    if suffix_layers_with_grad != [15, 16, 17, 18, 19]:
        raise AssertionError(
            f"suffix layers must all receive finite nonzero gradients: {suffix_layers_with_grad}"
        )
    return {
        "r": r,
        "trace": sequence,
        "expected": expected,
        "backward_hook_order": backward_hook_order,
        "backward_trace_multiset_ok": True,
        "probe_backward_hook_count_excluded": probe_backward_hook_count,
        "loss": float(loss.detach().item()),
        "backward_traversed_loops": list(range(1, r + 1)),
        "parameter_gradient_enabled_loops": expected_tail,
        "early_parameter_gradient_disabled_loops": list(range(1, r - PARAMETER_GRADIENT_TAIL_LOOPS + 1)),
        "gradient_norms_by_physical_layer": gradient_norms,
        "prefix_layers_with_grad": prefix_layers_with_grad,
        "prefix_all_receive_finite_nonzero_grad": True,
        "suffix_layers_with_grad": suffix_layers_with_grad,
        "suffix_all_receive_finite_nonzero_grad": True,
        "hidden_state_path_preserved": bool(result.logits.requires_grad),
        "early_hidden_gradient_norms": early_hidden_gradient_norms,
        "middle_call_count": r * MIDDLE_LAYER_COUNT,
        "each_middle_loop_has_exactly_ten_physical_calls": True,
        "middle_call_audit_keys": [
            f"{int(detail['loop'])}:{int(detail['physical_index'])}"
            for detail in middle_call_audit
        ],
        "early_parameter_gradient_edges_absent": early_parameter_gradient_edges_absent,
        "exact_parameter_gradient_tail": len(expected_tail) == PARAMETER_GRADIENT_TAIL_LOOPS,
        "parameter_identity_and_requires_grad_restored": True,
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


def _require_finite(label: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise AssertionError(f"{label} contains non-finite values")


def _cache_audit(model: Any, input_ids: torch.Tensor, r: int) -> dict[str, Any]:
    prompt_length = int(input_ids.shape[1])
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        no_cache = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, middle_loop_count=r)
        cached = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, middle_loop_count=r)
    _require_finite(f"r={r} no-cache prefill logits", no_cache.logits)
    _require_finite(f"r={r} cached prefill logits", cached.logits)
    prefill_max_diff = float((no_cache.logits.float() - cached.logits.float()).abs().max().item())
    if not torch.allclose(no_cache.logits, cached.logits, atol=CACHE_ATOL, rtol=CACHE_RTOL):
        raise AssertionError(
            f"cache/no-cache prefill logits disagree for r={r}: "
            f"max_diff={prefill_max_diff} atol={CACHE_ATOL} rtol={CACHE_RTOL}"
        )
    cache = cached.past_key_values
    expected_slots = len(build_5_10xr_5_schedule(r))
    if len(cache) < expected_slots:
        raise AssertionError(f"cache has {len(cache)} slots; expected at least {expected_slots}")
    prefill_slots = [_cache_slot_state(cache, index) for index in range(expected_slots)]
    if not all(slot["sequence_length"] == prompt_length and slot["key_nonempty"] and slot["value_nonempty"] for slot in prefill_slots):
        raise AssertionError(f"invalid prefill cache slots for r={r}: {prefill_slots}")
    precreated_cache = make_dynamic_cache()
    with torch.inference_mode():
        precreated = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=precreated_cache,
            use_cache=True,
            middle_loop_count=r,
        )
    if len(precreated_cache) < expected_slots or precreated.past_key_values is not precreated_cache:
        raise AssertionError(f"precreated lazy DynamicCache failed for r={r}")
    precreated_slots = [_cache_slot_state(precreated_cache, index) for index in range(expected_slots)]
    if not all(slot["sequence_length"] == prompt_length and slot["key_nonempty"] and slot["value_nonempty"] for slot in precreated_slots):
        raise AssertionError(f"invalid precreated cache slots for r={r}: {precreated_slots}")
    next_ids = input_ids[:, -1:]
    extended_attention_mask = torch.cat((attention_mask, torch.ones_like(attention_mask[:, :1])), dim=1)
    with torch.inference_mode():
        incremental = model(
            input_ids=next_ids,
            attention_mask=extended_attention_mask,
            past_key_values=cache,
            use_cache=True,
            middle_loop_count=r,
        )
        full = model(
            input_ids=torch.cat((input_ids, next_ids), dim=1),
            attention_mask=extended_attention_mask,
            use_cache=False,
            middle_loop_count=r,
        )
    _require_finite(f"r={r} incremental logits", incremental.logits)
    _require_finite(f"r={r} full-sequence logits", full.logits)
    incremental_logits = incremental.logits[:, -1].float()
    full_logits = full.logits[:, -1].float()
    incremental_abs_diff = (incremental_logits - full_logits).abs()
    incremental_max_diff = float(incremental_abs_diff.max().item())
    incremental_mean_diff = float(incremental_abs_diff.mean().item())
    incremental_cosine = float(
        torch.nn.functional.cosine_similarity(incremental_logits, full_logits, dim=-1).min().item()
    )
    incremental_argmax_equal = bool(
        torch.equal(incremental_logits.argmax(dim=-1), full_logits.argmax(dim=-1))
    )
    # BF16 cached single-token attention and full-sequence attention use
    # different GEMM/SDPA shapes. Judge this low-precision pass semantically;
    # the strict numerical cache comparison is repeated independently in FP32.
    incremental_semantic_ok = not (
        incremental_max_diff > BF16_INCREMENTAL_MAX_ABS
        or incremental_cosine < BF16_INCREMENTAL_MIN_COSINE
        or not incremental_argmax_equal
    )
    if not incremental_semantic_ok:
        # This is diagnostic only. A deep recursive BF16 cached decode and
        # full-sequence recomputation use different GEMM/SDPA shapes, and the
        # accumulated rounding drift can change an argmax without indicating
        # a cache-routing error. The decisive check below reruns every r in
        # FP32 with TF32 disabled; finite values and every cache-slot length
        # remain mandatory here.
        print(
            f"[cache-warning] BF16 incremental logits differ semantically for r={r}: "
            f"max_diff={incremental_max_diff} mean_diff={incremental_mean_diff} "
            f"cosine={incremental_cosine} argmax_equal={incremental_argmax_equal} "
            f"limits=(max={BF16_INCREMENTAL_MAX_ABS}, "
            f"cosine>={BF16_INCREMENTAL_MIN_COSINE}, argmax_equal=True); "
            "continuing to the decisive FP32 cache audit",
            flush=True,
        )
    incremental_slots = [_cache_slot_state(cache, index) for index in range(expected_slots)]
    if not all(slot["sequence_length"] == prompt_length + 1 and slot["key_nonempty"] and slot["value_nonempty"] for slot in incremental_slots):
        raise AssertionError(f"invalid incremental cache slots for r={r}: {incremental_slots}")
    mismatch_error = None
    try:
        mismatch_attention_mask = torch.ones(
            (input_ids.shape[0], prompt_length + 2), dtype=attention_mask.dtype, device=input_ids.device
        )
        model(
            input_ids=input_ids[:, :1],
            attention_mask=mismatch_attention_mask,
            past_key_values=cache,
            use_cache=True,
            middle_loop_count=5 if r != 5 else 4,
        )
    except ValueError as exc:
        mismatch_error = str(exc)
    if mismatch_error is None:
        raise AssertionError("cache r mismatch was not rejected")
    return {
        "r": r,
        "expected_slots": expected_slots,
        "actual_slots": len(cache),
        "prefill_slot_state": prefill_slots,
        "precreated_slot_state": precreated_slots,
        "incremental_slot_state": incremental_slots,
        "precreated_lazy_cache_ok": True,
        "incremental_semantic_ok": incremental_semantic_ok,
        "incremental_semantic_is_diagnostic_only": True,
        "incremental_max_diff": incremental_max_diff,
        "incremental_mean_diff": incremental_mean_diff,
        "incremental_cosine": incremental_cosine,
        "incremental_argmax_equal": incremental_argmax_equal,
        "bf16_incremental_max_abs_limit": BF16_INCREMENTAL_MAX_ABS,
        "bf16_incremental_min_cosine": BF16_INCREMENTAL_MIN_COSINE,
        "cache_r_mismatch_rejected": True,
        "cache_r_mismatch_error": mismatch_error,
        "no_cache_cache_logits_allclose": True,
        "prefill_max_diff": prefill_max_diff,
        "prefill_atol": CACHE_ATOL,
        "prefill_rtol": CACHE_RTOL,
    }


def _fp32_incremental_cache_audit(model: Any, input_ids: torch.Tensor, r: int) -> dict[str, Any]:
    """Strict cache routing/position audit without BF16 kernel-shape noise."""

    attention_mask = torch.ones_like(input_ids)
    next_ids = input_ids[:, -1:]
    extended_attention_mask = torch.cat((attention_mask, torch.ones_like(attention_mask[:, :1])), dim=1)
    with torch.inference_mode():
        prefill = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            middle_loop_count=r,
        )
        incremental = model(
            input_ids=next_ids,
            attention_mask=extended_attention_mask,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            middle_loop_count=r,
        )
        full = model(
            input_ids=torch.cat((input_ids, next_ids), dim=1),
            attention_mask=extended_attention_mask,
            use_cache=False,
            middle_loop_count=r,
        )
    _require_finite(f"r={r} FP32 incremental logits", incremental.logits)
    _require_finite(f"r={r} FP32 full-sequence logits", full.logits)
    incremental_logits = incremental.logits[:, -1]
    full_logits = full.logits[:, -1]
    max_diff = float((incremental_logits - full_logits).abs().max().item())
    if not torch.allclose(
        incremental_logits,
        full_logits,
        atol=FP32_INCREMENTAL_ATOL,
        rtol=FP32_INCREMENTAL_RTOL,
    ):
        raise AssertionError(
            f"FP32 incremental logits disagree for r={r}: max_diff={max_diff} "
            f"atol={FP32_INCREMENTAL_ATOL} rtol={FP32_INCREMENTAL_RTOL}"
        )
    expected_slots = len(build_5_10xr_5_schedule(r))
    slots = [_cache_slot_state(prefill.past_key_values, index) for index in range(expected_slots)]
    expected_length = int(input_ids.shape[1]) + 1
    if not all(slot["sequence_length"] == expected_length and slot["key_nonempty"] and slot["value_nonempty"] for slot in slots):
        raise AssertionError(f"invalid FP32 incremental cache slots for r={r}: {slots}")
    return {
        "r": r,
        "max_diff": max_diff,
        "atol": FP32_INCREMENTAL_ATOL,
        "rtol": FP32_INCREMENTAL_RTOL,
        "incremental_cache_slot_state": slots,
        "ok": True,
    }


def _state_dict_reload_audit(model: Any, reloaded: Any) -> dict[str, Any]:
    """Make serialization correctness independent of BF16 forward kernels."""

    original_state = model.state_dict()
    reloaded_state = reloaded.state_dict()
    original_keys = tuple(original_state)
    reloaded_keys = tuple(reloaded_state)
    if original_keys != reloaded_keys:
        missing = sorted(set(original_keys) - set(reloaded_keys))
        unexpected = sorted(set(reloaded_keys) - set(original_keys))
        raise AssertionError(
            f"save/reload state_dict keys differ: missing={missing} unexpected={unexpected}"
        )
    total_elements = 0
    for name in original_keys:
        before = original_state[name]
        after = reloaded_state[name]
        if before.shape != after.shape:
            raise AssertionError(
                f"save/reload tensor shape mismatch for {name}: "
                f"before={tuple(before.shape)} after={tuple(after.shape)}"
            )
        if before.dtype != after.dtype:
            raise AssertionError(
                f"save/reload tensor dtype mismatch for {name}: "
                f"before={before.dtype} after={after.dtype}"
            )
        if not torch.equal(before, after):
            max_diff = None
            if before.is_floating_point() or before.is_complex():
                max_diff = float((before.float() - after.float()).abs().max().item())
            raise AssertionError(
                f"save/reload tensor values differ for {name}: max_diff={max_diff}"
            )
        total_elements += int(before.numel())
    return {
        "keys_exact": True,
        "shapes_exact": True,
        "dtypes_exact": True,
        "values_bitwise_equal": True,
        "tensor_count": len(original_keys),
        "total_elements": total_elements,
    }


def _generation_audit(model: Any, input_ids: torch.Tensor, max_new_tokens: int) -> dict[str, Any]:
    explicit_runs: list[dict[str, Any]] = []
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        for r in range(MIN_MIDDLE_LOOPS, MAX_MIDDLE_LOOPS + 1):
            generated_r = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, middle_loop_count=r, eos_token_id=None, pad_token_id=0)
            audit_r = dict(getattr(getattr(model, "model", model), "_last_forward_audit", {}))
            if audit_r.get("middle_loop_count") != r:
                raise AssertionError(f"generation did not preserve explicit r={r}: {audit_r}")
            if generated_r.ndim != 2 or generated_r.shape[1] <= input_ids.shape[1]:
                raise AssertionError(f"generation did not append tokens for r={r}")
            explicit_runs.append({"r": r, "output_shape": list(generated_r.shape), "fixed": True})
        default = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1, do_sample=False, use_cache=False, eos_token_id=None, pad_token_id=0)
    default_audit = dict(getattr(getattr(model, "model", model), "_last_forward_audit", {}))
    if default_audit.get("middle_loop_count") != DEFAULT_INFERENCE_MIDDLE_LOOPS:
        raise AssertionError(f"inference default r is not 7: {default_audit}")
    return {"explicit_runs": explicit_runs, "all_supported_r_fixed": True, "explicit_r4_fixed": True, "default_inference_r": default_audit.get("middle_loop_count"), "default_output_shape": list(default.shape), "max_new_tokens": max_new_tokens}


def package_version(name: str) -> str:
    try:
        return version(name)
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if (MIN_MIDDLE_LOOPS, MAX_MIDDLE_LOOPS, PARAMETER_GRADIENT_TAIL_LOOPS) != (4, 10, 4):
        raise AssertionError("invalid 5-10xr-5 audit constants")
    model_path = args.model_path.expanduser().resolve()
    output_report = ensure_external_report(args.output_report)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Recursive 5-10xr-5 smoke requires a submitted CUDA job")
    register_auto_class()
    model = RecursiveLlama5_10xr_5ForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval()
    config = model.config
    if (int(getattr(config, "num_hidden_layers", -1)), int(getattr(config, "recursive_layer_count", -1))) != (110, 20):
        raise AssertionError("checkpoint must declare max logical depth 110 and 20 physical layers")
    config_support = tuple(int(value) for value in getattr(config, "recursive_poisson_support", ()))
    config_probabilities = tuple(float(value) for value in getattr(config, "recursive_poisson_probabilities", ()))
    if (
        str(getattr(config, "recursive_sampling_policy", "")) != SAMPLING_POLICY
        or str(getattr(config, "recursive_sampler_version", "")) != SAMPLER_VERSION
        or float(getattr(config, "recursive_poisson_lambda", -1.0)) != POISSON_LAMBDA
        or config_support != POISSON_SUPPORT
        or abs(float(getattr(config, "recursive_poisson_normalization_z", -1.0)) - POISSON_NORMALIZATION_Z) > 1e-14
        or len(config_probabilities) != len(POISSON_PROBABILITIES)
        or any(abs(a - b) > 1e-14 for a, b in zip(config_probabilities, POISSON_PROBABILITIES))
        or str(getattr(config, "recursive_sampler_key", "")) != SAMPLER_KEY
    ):
        raise AssertionError("checkpoint sampling metadata does not match the 5-10xr-5 contract")
    if len(_layers(model)) != PHYSICAL_LAYER_COUNT:
        raise AssertionError("checkpoint does not own exactly 20 physical decoder modules")
    if len({id(layer) for layer in _layers(model)}) != 20:
        raise AssertionError("physical decoder modules are not unique")
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    traces = [_trace_forward(model, input_ids, r) for r in range(4, 11)]
    backwards = [_backward_audit(model, input_ids, r) for r in range(4, 11)]
    # The last backward audit intentionally leaves real gradients on all 20
    # physical layers. They are no longer needed and would otherwise double
    # the transient memory cost of the in-place FP32 audit conversion.
    model.zero_grad(set_to_none=True)
    caches = [_cache_audit(model, input_ids, r) for r in range(4, 11)]
    original_dtype = next(model.parameters()).dtype
    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        model.float()
        fp32_incremental_caches = [
            _fp32_incremental_cache_audit(model, input_ids, r) for r in range(4, 11)
        ]
    finally:
        model.to(dtype=original_dtype)
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
    generation = _generation_audit(model, input_ids, args.max_new_tokens)
    with tempfile.TemporaryDirectory(prefix="rsmol-5-10xr-5-reload-") as temp_dir:
        model.save_pretrained(temp_dir, safe_serialization=False)
        reloaded = RecursiveLlama5_10xr_5ForCausalLM.from_pretrained(
            temp_dir, local_files_only=True, torch_dtype=original_dtype
        ).to(device).eval()
        reload_state_dict = _state_dict_reload_audit(model, reloaded)
        reload_by_r: list[dict[str, Any]] = []
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            for r in range(MIN_MIDDLE_LOOPS, MAX_MIDDLE_LOOPS + 1):
                before = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, middle_loop_count=r).logits.float()
                after = reloaded(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, middle_loop_count=r).logits.float()
                cached_after = reloaded(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, middle_loop_count=r).logits.float()
                _require_finite(f"r={r} original save/reload logits", before)
                _require_finite(f"r={r} reloaded no-cache logits", after)
                _require_finite(f"r={r} reloaded cached logits", cached_after)
                cross_instance_diff = (before - after).abs()
                cross_instance_max_diff = float(cross_instance_diff.max().item())
                cross_instance_allclose = bool(
                    torch.allclose(before, after, atol=CACHE_ATOL, rtol=CACHE_RTOL)
                )
                if not cross_instance_allclose:
                    print(
                        f"[reload-warning] BF16 logits from two bitwise-identical model "
                        f"instances differ for r={r}: max_diff={cross_instance_max_diff}; "
                        "state_dict equality is the decisive serialization audit",
                        flush=True,
                    )
                reloaded_cache_max_diff = float((after - cached_after).abs().max().item())
                if not torch.allclose(after, cached_after, atol=CACHE_ATOL, rtol=CACHE_RTOL):
                    raise AssertionError(
                        f"reloaded model cache/no-cache prefill logits mismatch for r={r}: "
                        f"max_diff={reloaded_cache_max_diff} atol={CACHE_ATOL} rtol={CACHE_RTOL}"
                    )
                reload_by_r.append(
                    {
                        "r": r,
                        "cross_instance_logits_allclose": cross_instance_allclose,
                        "cross_instance_logits_is_diagnostic_only": True,
                        "cross_instance_max_diff": cross_instance_max_diff,
                        "reloaded_cache_no_cache_logits_match": True,
                        "reloaded_cache_max_diff": reloaded_cache_max_diff,
                    }
                )
    report = {
        "status": "PASS", "variant": "SmolLM2-5-10xr-5", "model_path": str(model_path),
        "logical_layer_count_max": LOGICAL_LAYER_COUNT, "physical_layer_count": PHYSICAL_LAYER_COUNT,
        "r_values_audited": list(range(4, 11)), "default_inference_r": 7,
        "fixed_parameter_gradient_tail_loops": 4, "physical_module_unique": True,
        "parameter_audit": parameter_audit(model), "forward_trace_audit": traces,
        "backward_trace_audit": backwards, "cache_audit": caches,
        "incremental_fp32_cache_audit": fp32_incremental_caches,
        "generation_audit": generation,
        "save_reload_audit": {
            "ok": True,
            "state_dict": reload_state_dict,
            "all_supported_r": reload_by_r,
        },
        "transformers": package_version("transformers"), "torch": torch.__version__,
    }
    atomic_json(output_report, report)
    print(f"[result] status=PASS report={output_report}", flush=True)


if __name__ == "__main__":
    main()
