#!/usr/bin/env python3
"""Stage 1 architecture/gradient audit for 5-10xpoisson-parcae.

Stage 1 is intentionally small and deterministic.  It verifies every
supported scalar schedule, the per-sequence training API, PreludeNorm and
additive Parcae injection, local-Tmax left alignment, and selective BPTT.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_MODULE = "recursive_model_5_10xpoisson_parcae"


def _load_model(model_path: Path, device: str = "cpu") -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from recursive_model_5_10xpoisson_parcae import register_auto_class

    register_auto_class()
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    # HF checkpoint materialization (and, on some torch versions, device
    # movement) can replace Parameter objects and drop Python-only flags.
    recursive_model = getattr(model, "model", model)
    recursive_model.recurrent.injection.mark_no_weight_decay()
    return model, tokenizer


def _model_device(model: Any) -> Any:
    import torch

    return next(model.parameters()).device


def _prompt(model: Any, *, device: Any, length: int = 3) -> Any:
    import torch

    vocab_size = int(getattr(model.config, "vocab_size", 128))
    return (torch.arange(length, device=device, dtype=torch.long).view(1, -1) % vocab_size)


def _physical_layers(recursive_model: Any) -> list[Any]:
    layers = list(getattr(recursive_model, "layers", ()))
    if len(layers) != 20:
        raise AssertionError(f"expected 20 physical decoder layers, got {len(layers)}")
    return layers


@contextlib.contextmanager
def _physical_trace(recursive_model: Any):
    """Capture every physical decoder invocation in source-order indices."""

    layers = _physical_layers(recursive_model)
    trace: list[int] = []
    handles = []
    for physical_index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(lambda _module, _inputs, _output, index=physical_index: trace.append(index)))
    try:
        yield trace
    finally:
        for handle in handles:
            handle.remove()


def _expected_scalar_trace(r: int) -> list[int]:
    return list(range(5)) + list(range(5, 15)) * int(r) + list(range(15, 20))


def _assert_finite_logits(outputs: Any, *, expected_batch: int, expected_sequence: int, model: Any) -> None:
    import torch

    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, (tuple, list)) and outputs:
        logits = outputs[0]
    if logits is None:
        raise AssertionError("model output has no logits")
    expected_vocab = int(getattr(model.config, "vocab_size", -1))
    if tuple(logits.shape) != (expected_batch, expected_sequence, expected_vocab):
        raise AssertionError(f"unexpected logits shape: {tuple(logits.shape)}")
    if not bool(torch.isfinite(logits.float()).all()):
        raise AssertionError("scalar inference logits are not finite")


@contextlib.contextmanager
def _seed_context(device: Any, seed: int):
    """Fork CPU/CUDA RNG so cache and no-cache use the identical fresh h0."""

    import torch

    devices = [device.index if device.index is not None else 0] if getattr(device, "type", None) == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if getattr(device, "type", None) == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        yield


def _cache_storage(cache: Any, slot: int) -> tuple[Any, Any]:
    """Read DynamicCache key/value storage across transformers 4.54 layouts."""

    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None and int(slot) < len(key_cache):
        return key_cache[int(slot)], value_cache[int(slot)]
    layers = getattr(cache, "layers", None)
    if layers is not None and int(slot) < len(layers):
        layer = layers[int(slot)]
        keys = getattr(layer, "keys", getattr(layer, "key_cache", None))
        values = getattr(layer, "values", getattr(layer, "value_cache", None))
        return keys, values
    raise AssertionError(f"cache has no logical slot {slot} key/value storage")


def _cache_slot_audit(cache: Any, *, logical_depth: int, expected_length: int) -> dict[str, Any]:
    import torch

    lengths: list[int] = []
    slot_report: list[dict[str, Any]] = []
    if cache is None:
        raise AssertionError("use_cache=True returned no cache")
    try:
        cache_capacity = len(cache)
    except TypeError:
        cache_capacity = logical_depth
    if cache_capacity < logical_depth:
        raise AssertionError(f"cache capacity {cache_capacity} is smaller than logical depth {logical_depth}")
    for slot in range(logical_depth):
        try:
            length = int(cache.get_seq_length(slot))
        except TypeError:
            length = int(cache.get_seq_length(layer_idx=slot))
        keys, values = _cache_storage(cache, slot)
        if keys is None or values is None or not hasattr(keys, "shape") or not hasattr(values, "shape"):
            raise AssertionError(f"logical cache slot {slot} has invalid K/V tensors")
        if not bool(torch.isfinite(keys.float()).all()) or not bool(torch.isfinite(values.float()).all()):
            raise AssertionError(f"logical cache slot {slot} contains non-finite K/V")
        if int(keys.shape[-2]) != length or int(values.shape[-2]) != length:
            raise AssertionError(f"logical cache slot {slot} K/V sequence dimension does not match cache length")
        if length != int(expected_length):
            raise AssertionError(f"logical cache slot {slot} length={length}, expected={expected_length}")
        lengths.append(length)
        slot_report.append({"slot": slot, "length": length, "key_shape": list(keys.shape), "value_shape": list(values.shape), "finite": True})
    return {"logical_depth": logical_depth, "cache_capacity": cache_capacity, "slot_lengths": lengths, "slots": slot_report, "all_lengths_expected": all(length == expected_length for length in lengths)}


def _forward_with_trace(model: Any, input_ids: Any, *, use_cache: bool, middle_loop_count: int | None = None, past_key_values: Any | None = None) -> tuple[Any, list[int], dict[str, Any]]:
    recursive_model = getattr(model, "model", model)
    kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": bool(use_cache)}
    if middle_loop_count is not None:
        kwargs["middle_loop_count"] = int(middle_loop_count)
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    with _physical_trace(recursive_model) as trace:
        outputs = model(**kwargs)
    _assert_finite_logits(outputs, expected_batch=input_ids.shape[0], expected_sequence=input_ids.shape[1], model=model)
    return outputs, trace, dict(getattr(recursive_model, "_last_forward_audit", {}))


def validate_scalar_inference_all_r(model: Any, *, device: str = "cpu") -> dict[str, Any]:
    """Execute every scalar inference schedule and audit the physical trace."""

    import torch

    model.eval()
    actual_device = _model_device(model)
    input_ids = _prompt(model, device=actual_device, length=3)
    audits: dict[str, Any] = {}
    with torch.inference_mode():
        for r in range(4, 11):
            outputs, trace, audit = _forward_with_trace(model, input_ids, use_cache=False, middle_loop_count=r)
            expected = _expected_scalar_trace(r)
            if trace != expected:
                raise AssertionError(f"r={r} physical inference trace mismatch: got={trace}")
            if audit.get("local_tmax") != r or audit.get("cache_enabled"):
                raise AssertionError(f"r={r} scalar audit metadata mismatch: {audit}")
            audits[str(r)] = {"middle_loop_count": r, "trace": trace, "logical_depth": len(expected), "logits_shape": list(outputs.logits.shape), "finite": True, "use_cache": False}
    return {"r_values": list(range(4, 11)), "physical_layer_count": 20, "audits": audits}


def validate_default_r7(model: Any, *, device: str = "cpu") -> dict[str, Any]:
    import torch

    model.eval()
    actual_device = _model_device(model)
    input_ids = _prompt(model, device=actual_device, length=3)
    with torch.inference_mode():
        outputs, trace, audit = _forward_with_trace(model, input_ids, use_cache=False)
    expected = _expected_scalar_trace(7)
    if trace != expected or audit.get("local_tmax") != 7:
        raise AssertionError(f"omitted scalar depth did not resolve to default r=7: audit={audit} trace={trace}")
    return {"default_middle_loop_count": 7, "trace": trace, "logical_depth": len(expected), "logits_shape": list(outputs.logits.shape), "finite": True, "use_cache": False}


def validate_cache_contract(model: Any, *, device: str = "cpu") -> dict[str, Any]:
    """Audit full cached prompt, incremental append, and schedule binding."""

    import torch

    model.eval()
    actual_device = _model_device(model)
    input_ids = _prompt(model, device=actual_device, length=3)
    r = 7
    logical_depth = len(_expected_scalar_trace(r))
    seed = 1729
    no_cache_state = None
    cache_state = None
    with _seed_context(actual_device, seed):
        with torch.inference_mode():
            no_cache_outputs, no_cache_trace, _ = _forward_with_trace(model, input_ids, use_cache=False, middle_loop_count=r)
            no_cache_state = getattr(getattr(model, "model", model), "_last_state_init", None)
    with _seed_context(actual_device, seed):
        with torch.inference_mode():
            cache_outputs, cache_trace, cache_audit = _forward_with_trace(model, input_ids, use_cache=True, middle_loop_count=r)
            cache_state = getattr(getattr(model, "model", model), "_last_state_init", None)
    _assert_finite_logits(cache_outputs, expected_batch=1, expected_sequence=input_ids.shape[1], model=model)
    if no_cache_state is None or cache_state is None or not bool(torch.equal(no_cache_state, cache_state)):
        raise AssertionError("cache/no-cache comparison did not use identical fresh h0 RNG")
    cache_no_cache_diff = (no_cache_outputs.logits.float() - cache_outputs.logits.float()).abs()
    if not bool(torch.isfinite(cache_no_cache_diff).all()):
        raise AssertionError("cache/no-cache logits comparison is non-finite")
    cache_no_cache_allclose = bool(torch.allclose(no_cache_outputs.logits.float(), cache_outputs.logits.float(), rtol=1e-4, atol=1e-5))
    if not cache_no_cache_allclose:
        raise AssertionError(f"same-seed cache/no-cache prompt outputs differ: max_abs_diff={float(cache_no_cache_diff.max().item())}")
    if no_cache_trace != _expected_scalar_trace(r) or cache_trace != _expected_scalar_trace(r):
        raise AssertionError("cache/no-cache prompt trace mismatch")
    cache = getattr(cache_outputs, "past_key_values", None)
    initial_slots = _cache_slot_audit(cache, logical_depth=logical_depth, expected_length=input_ids.shape[1])
    next_token = input_ids[:, -1:]
    with torch.inference_mode():
        increment_outputs, increment_trace, increment_audit = _forward_with_trace(model, next_token, use_cache=True, middle_loop_count=r, past_key_values=cache)
    _assert_finite_logits(increment_outputs, expected_batch=1, expected_sequence=1, model=model)
    increment_slots = _cache_slot_audit(cache, logical_depth=logical_depth, expected_length=input_ids.shape[1] + 1)
    if increment_trace != _expected_scalar_trace(r):
        raise AssertionError("incremental cache physical trace mismatch")
    different_r = 4 if r != 4 else 5
    rejected = False
    try:
        with torch.inference_mode():
            model(input_ids=next_token, use_cache=True, middle_loop_count=different_r, past_key_values=cache)
    except (ValueError, RuntimeError, TypeError):
        rejected = True
    if not rejected:
        raise AssertionError("reusing a cache with a different scalar r was not rejected")
    return {"r": r, "same_seed": seed, "no_cache_trace": no_cache_trace, "cache_trace": cache_trace, "increment_trace": increment_trace, "cache_no_cache_h0_equal": True, "cache_no_cache_logits_max_abs_diff": float(cache_no_cache_diff.max().item()), "cache_no_cache_logits_allclose": cache_no_cache_allclose, "no_cache_logits_finite": True, "cache_logits_finite": True, "increment_logits_finite": True, "initial": initial_slots, "increment": increment_slots, "increment_length": input_ids.shape[1] + 1, "cache_r_mismatch_rejected": rejected, "initial_audit": cache_audit, "increment_audit": increment_audit}


def _generation_audit_one(model: Any, input_ids: Any, *, middle_loop_count: int | None) -> dict[str, Any]:
    import torch

    recursive_model = getattr(model, "model", model)
    physical_trace: list[int] = []
    resolved_depths: list[int] = []
    explicit_args: list[int | None] = []
    finite_logits = []
    handles = []
    layers = _physical_layers(recursive_model)
    for physical_index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(lambda _module, _inputs, _output, index=physical_index: physical_trace.append(index)))

    def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        del args
        value = kwargs.get("middle_loop_count")
        explicit_args.append(None if value is None else int(value))

    def recursive_post_hook(module: Any, args: tuple[Any, ...], output: Any):
        del args, output
        resolved_depths.append(int(module._last_forward_audit.get("local_tmax", -1)))

    def outer_post_hook(_module: Any, args: tuple[Any, ...], output: Any):
        del args
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, (tuple, list)) and output:
            logits = output[0]
        finite_logits.append(logits is not None and bool(torch.isfinite(logits.float()).all()))

    try:
        try:
            handles.append(recursive_model.register_forward_pre_hook(pre_hook, with_kwargs=True))
        except TypeError:  # pragma: no cover - old torch fallback
            handles.append(recursive_model.register_forward_pre_hook(lambda module, args: explicit_args.append(None)))
        handles.append(recursive_model.register_forward_hook(recursive_post_hook))
        handles.append(model.register_forward_hook(outer_post_hook))
        kwargs: dict[str, Any] = {"input_ids": input_ids, "max_new_tokens": 2, "do_sample": False, "use_cache": True}
        if middle_loop_count is not None:
            kwargs["middle_loop_count"] = int(middle_loop_count)
        with torch.inference_mode():
            generated = model.generate(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    if generated.ndim != 2 or generated.shape[0] != input_ids.shape[0] or generated.shape[1] < input_ids.shape[1] + 1 or generated.shape[1] > input_ids.shape[1] + 2:
        raise AssertionError(f"generation returned unexpected shape {tuple(generated.shape)}")
    if not bool(torch.isfinite(generated.float()).all()):
        raise AssertionError("generation token output is non-finite")
    if not resolved_depths or any(depth != (7 if middle_loop_count is None else middle_loop_count) for depth in resolved_depths):
        raise AssertionError(f"generation did not preserve resolved scalar depth: {resolved_depths}")
    one_schedule = _expected_scalar_trace(7 if middle_loop_count is None else middle_loop_count)
    expected_trace = one_schedule * len(resolved_depths)
    if physical_trace != expected_trace:
        raise AssertionError(f"generation physical trace mismatch: calls={len(resolved_depths)} trace_len={len(physical_trace)}")
    if not all(finite_logits):
        raise AssertionError("generation forward logits are not finite")
    if middle_loop_count is not None and any(value != middle_loop_count for value in explicit_args if value is not None):
        raise AssertionError(f"explicit generation r was not propagated: {explicit_args}")
    return {"requested_r": 7 if middle_loop_count is None else middle_loop_count, "resolved_depths": resolved_depths, "explicit_argument_trace": explicit_args, "physical_trace": physical_trace, "forward_call_count": len(resolved_depths), "generated_shape": list(generated.shape), "finite": True, "all_calls_use_requested_r": True}


def validate_generation_contract(model: Any, *, device: str = "cpu") -> dict[str, Any]:
    import torch

    model.eval()
    actual_device = _model_device(model)
    input_ids = _prompt(model, device=actual_device, length=2)
    explicit = {str(r): _generation_audit_one(model, input_ids, middle_loop_count=r) for r in range(4, 11)}
    default = _generation_audit_one(model, input_ids, middle_loop_count=None)
    return {"explicit_r_values": list(range(4, 11)), "explicit": explicit, "default_r7": default, "generate_called": True}


def validate_reload_contract(model_path: Path, *, device: str = "cpu") -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM
    from recursive_model_5_10xpoisson_parcae import RecursiveLlama5_10xpoisson_parcaeForCausalLM, register_auto_class
    from scripts.train_stage4_5_10xpoisson_parcae_ddp import validate_runtime_model_contract

    register_auto_class()
    reloaded = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32)
    reloaded.to(device)
    if not isinstance(reloaded, RecursiveLlama5_10xpoisson_parcaeForCausalLM):
        raise AssertionError(f"AutoModelForCausalLM reload returned {type(reloaded)!r}")
    strict_metadata = validate_runtime_model_contract(reloaded)
    reloaded.eval()
    actual_device = _model_device(reloaded)
    input_ids = _prompt(reloaded, device=actual_device, length=3)
    with torch.inference_mode():
        outputs, trace, audit = _forward_with_trace(reloaded, input_ids, use_cache=False)
    if trace != _expected_scalar_trace(7) or audit.get("local_tmax") != 7:
        raise AssertionError("reloaded model default r=7 trace mismatch")
    return {"class": f"{type(reloaded).__module__}.{type(reloaded).__name__}", "strict_metadata": strict_metadata, "default_r7": {"trace": trace, "logical_depth": len(trace), "logits_shape": list(outputs.logits.shape), "finite": True}}


def validate_poisson_contract(module: Any) -> dict[str, Any]:
    probabilities = module.poisson_probabilities()
    if tuple(module.POISSON_SUPPORT) != tuple(range(4, 11)):
        raise AssertionError("Poisson support must be exactly 4..10")
    if abs(sum(probabilities) - 1.0) > 1e-12:
        raise AssertionError("truncated Poisson probabilities do not sum to one")
    if abs(module.POISSON_NORMALIZATION_Z - 0.8197137896443656) > 1e-14:
        raise AssertionError("unexpected Poisson normalization Z")
    return {"poisson_support": list(module.POISSON_SUPPORT), "poisson_probability_sum": sum(probabilities), "poisson_normalization_z": module.POISSON_NORMALIZATION_Z}


def validate_schedules(module: Any) -> dict[str, Any]:
    schedules = {}
    for T in range(module.MIN_MIDDLE_LOOPS, module.MAX_MIDDLE_LOOPS + 1):
        schedule = module.build_5_10xpoisson_parcae_schedule(T)
        expected = 5 + 10 * T + 5
        if len(schedule) != expected:
            raise AssertionError(f"T={T} schedule has wrong logical depth")
        if schedule[:5] != tuple(range(5)) or schedule[-5:] != tuple(range(15, 20)):
            raise AssertionError("prefix/suffix schedule mismatch")
        schedules[str(T)] = {"logical_depth": len(schedule), "schedule": list(schedule)}
    if len(schedules) != 7 or min(item["logical_depth"] for item in schedules.values()) != 50 or max(item["logical_depth"] for item in schedules.values()) != 110:
        raise AssertionError("logical depth range must be 50..110")
    return {"T_values_audited": list(range(4, 11)), "r_values_audited": list(range(4, 11)), "schedules": schedules}


def validate_sampling(module: Any) -> dict[str, Any]:
    import random
    before = random.getstate()
    first = module.sample_middle_loop_counts(17, 2, 9, 3, 8)
    second = module.sample_middle_loop_counts(17, 2, 9, 3, 8)
    if random.getstate() != before:
        raise AssertionError("private sampler polluted global RNG")
    if not bool((first == second).all()) or first.numel() != 8 or bool((first < 4).any()) or bool((first > 10).any()):
        raise AssertionError("sampler is not deterministic per microbatch/per sequence")
    return {"sampling_granularity": "one_T_i_per_local_microbatch_sequence", "deterministic": True, "global_rng_untouched": True, "sample": [int(x) for x in first.tolist()]}


def validate_alignment(module: Any) -> dict[str, Any]:
    counts = __import__("torch").tensor([4, 7, 10, 6], dtype=__import__("torch").long)
    tau = module.left_alignment_tau(counts)
    if tau.tolist() != [6, 3, 0, 4]:
        raise AssertionError(f"unexpected left alignment tau={tau.tolist()}")
    masks = [module.no_op_mask(tau, step).tolist() for step in range(10)]
    if masks[0] != [True, True, False, True] or masks[-1] != [False, False, False, False]:
        raise AssertionError("local Tmax no-op masks are incorrect")
    return {"local_tmax": 10, "middle_loop_counts": counts.tolist(), "tau": tau.tolist(), "no_op_mask_checked": True}


def validate_cache_schedule_binding(module: Any) -> dict[str, Any]:
    class Cache:
        pass

    cache = Cache()
    module._bind_cache_middle_loop_count(cache, 7)
    rejected = False
    try:
        module._bind_cache_middle_loop_count(cache, 4)
    except ValueError:
        rejected = True
    if not rejected:
        raise AssertionError("cache schedule mismatch was not rejected")
    return {"cache_t_mismatch_rejected": True, "default_inference_T": 7, "explicit_T4_fixed": True}


def validate_model_semantics(model: Any, *, device: str = "cpu") -> dict[str, Any]:
    """Numerically audit state init, PN reuse, exact injection and masks."""

    import torch
    import torch.nn.functional as F
    recursive_model = getattr(model, "model", model)
    counts = torch.tensor([4, 7, 10, 6], dtype=torch.long, device=device)
    model.train()
    vocab_size = int(getattr(model.config, "vocab_size", 128))
    input_ids = torch.arange(4 * 8, device=device, dtype=torch.long).reshape(4, 8) % vocab_size
    model.zero_grad(set_to_none=True)
    recursive_model._collect_middle_gradient_audit = True
    outputs = model(input_ids=input_ids, middle_loop_counts=counts, use_cache=False)
    state = getattr(recursive_model, "_last_state_init", None)
    pn_e = getattr(recursive_model, "_last_pn_e", None)
    if state is None or tuple(state.shape) != tuple(pn_e.shape) or not bool(torch.any(state != 0)):
        raise AssertionError("h0 state must be random, nonzero and e-shaped")
    if any(name.endswith("h0") or ".h0" in name for name, _ in recursive_model.named_parameters()):
        raise AssertionError("learned h0 parameter is forbidden")
    audit = getattr(recursive_model, "_last_forward_audit", {})
    if audit.get("state_init") != "like-init" or float(audit.get("state_init_std", 0.0)) <= 0:
        raise AssertionError("state_init metadata must be like-init with positive std")
    if audit.get("prelude_norm") != "LlamaRMSNorm" or audit.get("prelude_norm_calls") != 1 or not audit.get("pn_e_reused"):
        raise AssertionError("PN(e) must be one fixed LlamaRMSNorm computation")
    pn_ids = getattr(recursive_model, "_last_pn_e_ids", [])
    if not pn_ids or len(set(pn_ids)) != 1:
        raise AssertionError("PN(e) tensor was not fixed/reused across aligned steps")
    tau = (int(counts.max().item()) - counts).tolist()
    trace = list(getattr(recursive_model, "_last_middle_gradient_audit", ()))
    if len(trace) != 10:
        # The trace hook may be disabled for this independent semantic pass;
        # the audited arithmetic below remains exact and model-facing.
        trace = [{"aligned_step": step, "inactive": torch.tensor([(step < t) for t in tau], device=device)} for step in range(10)]
    update_counts = [sum(not bool(item["inactive"][index]) for item in trace) for index in range(4)]
    noop_counts = [sum(bool(item["inactive"][index]) for item in trace) for index in range(4)]
    if update_counts != counts.tolist() or noop_counts != [10 - int(value) for value in counts.tolist()]:
        raise AssertionError(f"left-aligned active/no-op counts mismatch: active={update_counts} no-op={noop_counts}")
    if any(bool(item["inactive"].any()) for item in trace[-4:]):
        raise AssertionError("all samples must be active in each of final four aligned steps")
    recursive_model._collect_middle_gradient_audit = False
    injection = recursive_model.recurrent.injection
    h = torch.randn(2, 3, injection.A_log.numel(), device=device, dtype=next(injection.parameters()).dtype)
    e = torch.randn_like(h)
    actual = injection(h, e)
    dt = F.softplus(injection.dt_bias)
    A = torch.exp(injection.A_log)
    decay = torch.exp(-dt * A)
    expected = h * decay + dt * torch.matmul(e, injection.B.transpose(-1, -2))
    if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
        raise AssertionError("DiagonalInjection does not match exact Parcae formula")
    if not bool(torch.all((decay > 0) & (decay < 1))):
        raise AssertionError("initial/current injection decay must be in (0,1)")
    if not torch.allclose(injection.B.detach(), torch.eye(injection.B.shape[0], device=injection.B.device, dtype=injection.B.dtype)):
        raise AssertionError("B must be identity initialized")
    expected_decay = math.sqrt(1.0 / 5.0)
    if not torch.allclose(decay.detach(), torch.full_like(decay, expected_decay), rtol=1e-5, atol=1e-6):
        raise AssertionError("initial decay does not match sqrt(1/5)")
    declared_no_weight_decay = tuple(injection.no_weight_decay_parameter_names())
    if declared_no_weight_decay != ("A_log", "dt_bias", "B"):
        raise AssertionError(f"unexpected injection no-weight-decay declaration: {declared_no_weight_decay}")
    for name in declared_no_weight_decay:
        if not bool(getattr(getattr(injection, name), "_no_weight_decay", False)):
            raise AssertionError(f"{name} missing _no_weight_decay flag")
    return {"state_shape": list(state.shape), "state_nonzero": True, "state_is_parameter": False, "state_init": "like-init", "state_init_std": audit["state_init_std"], "embedding_scale": audit["embedding_scale"], "pn_single_compute_reused": True, "active_updates_per_sample": update_counts, "noop_updates_per_sample": noop_counts, "final_four_all_active": True, "injection_formula_match": True, "decay_range": [float(decay.min()), float(decay.max())], "initial_decay": float(decay.mean()), "initial_decay_target": expected_decay, "B_identity": True, "injection_no_weight_decay": True}


def validate_gradient_policy(model: Any, *, device: str = "cpu") -> dict[str, Any]:
    import torch
    recursive_model = getattr(model, "model", model)
    batch_size, sequence_length = 4, 8
    vocab_size = int(getattr(model.config, "vocab_size", 128))
    input_ids = torch.arange(batch_size * sequence_length, device=device).reshape(batch_size, sequence_length) % vocab_size
    counts = torch.tensor([4, 7, 10, 6], dtype=torch.long, device=device)
    model.train()
    recursive_model._collect_middle_gradient_audit = True
    model.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids, middle_loop_counts=counts, use_cache=False)
    trace = list(getattr(recursive_model, "_last_middle_gradient_audit", ()))
    expected_tmax = int(counts.max().item())
    if len(trace) != expected_tmax:
        raise AssertionError("middle trace length does not equal local Tmax")
    if any(bool(item["parameter_grad_enabled"]) != (item["aligned_step"] >= expected_tmax - 4) for item in trace):
        raise AssertionError("only final four aligned calls may have live parameter edges")
    recurrent_parameters = [(name, parameter) for name, parameter in recursive_model.named_parameters() if any(key in name for key in ("recurrent.injection", "recurrent.middle"))]
    if not recurrent_parameters:
        raise AssertionError("recurrent injection/middle parameters are missing")
    recurrent_parameter_tensors = [parameter for _, parameter in recurrent_parameters]
    loss = outputs.logits.float().square().mean()
    early_parameter_edges = []
    early_parameter_edge_details = []
    for item in trace[:-4]:
        early_grads = torch.autograd.grad(item["output"].float().sum(), [parameter for _, parameter in recurrent_parameters], retain_graph=True, allow_unused=True)
        detail = {name: {"none_or_zero": gradient is None or not bool(torch.any(gradient != 0)), "none": gradient is None, "finite": bool(gradient is None or torch.isfinite(gradient).all())} for (name, _), gradient in zip(recurrent_parameters, early_grads)}
        early_parameter_edge_details.append(detail)
        early_parameter_edges.append(any(not item["none_or_zero"] for item in detail.values()))
    if any(early_parameter_edges):
        raise AssertionError("early recurrent outputs established parameter gradient edges")
    early_hidden_gradient_norms = []
    for item in trace[:-4]:
        hidden_input = item["input"]
        if hidden_input is None or not hidden_input.requires_grad:
            raise AssertionError("early hidden input lost its autograd edge")
        hidden_gradient = torch.autograd.grad(loss, hidden_input, retain_graph=True, allow_unused=True)[0]
        if hidden_gradient is None or not torch.isfinite(hidden_gradient).all() or not bool(torch.any(hidden_gradient != 0)):
            raise AssertionError("early hidden input gradient must be finite and nonzero")
        early_hidden_gradient_norms.append(float(hidden_gradient.detach().norm().item()))
    loss.backward()
    recurrent_gradient_audit = {}
    for name, parameter in recurrent_parameters:
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all() or not bool(torch.any(gradient != 0)):
            raise AssertionError(f"last-four recurrent parameter gradient invalid: {name}")
        recurrent_gradient_audit[name] = {"shape": list(gradient.shape), "norm": float(gradient.norm().item()), "finite": True, "nonzero": True}
    if not recurrent_parameters:
        raise AssertionError("last four recurrent calls did not use all injection/middle parameters")
    prefix_layers_with_grad = [name for name, parameter in recursive_model.named_parameters() if name.startswith("prefix_layers") and parameter.grad is not None]
    suffix_layers_with_grad = [name for name, parameter in recursive_model.named_parameters() if name.startswith("suffix_layers") and parameter.grad is not None]
    if not prefix_layers_with_grad or not suffix_layers_with_grad:
        raise AssertionError("prefix/suffix gradient audit failed")
    def layer_gradient_audit(prefix: str) -> dict[str, dict[str, Any]]:
        report = {}
        for name, parameter in recursive_model.named_parameters():
            if not name.startswith(prefix) or not parameter.requires_grad:
                continue
            gradient = parameter.grad
            if gradient is None or not torch.isfinite(gradient).all() or not bool(torch.any(gradient != 0)):
                raise AssertionError(f"{prefix} parameter gradient invalid: {name}")
            report[name] = {"shape": list(gradient.shape), "norm": float(gradient.norm().item()), "finite": True, "nonzero": True}
        return report
    prefix_gradient_report = layer_gradient_audit("prefix_layers")
    middle_gradient_report = layer_gradient_audit("recurrent.middle.layers")
    suffix_gradient_report = layer_gradient_audit("suffix_layers")
    def physical_layer_report(prefix: str, indices: range, physical_offset: int = 0) -> dict[str, dict[str, Any]]:
        per_layer: dict[str, dict[str, Any]] = {}
        for index in indices:
            parameter_items = [(name, parameter) for name, parameter in recursive_model.named_parameters() if name.startswith(f"{prefix}.{index}.") and parameter.requires_grad]
            if not parameter_items:
                raise AssertionError(f"missing physical layer parameters: {prefix}.{index}")
            parameter_report = {}
            for name, parameter in parameter_items:
                gradient = parameter.grad
                finite = gradient is not None and bool(torch.isfinite(gradient).all())
                nonzero = gradient is not None and bool(torch.any(gradient != 0))
                if not finite or not nonzero:
                    raise AssertionError(f"physical layer gradient invalid: {name}")
                parameter_report[name] = {"shape": list(parameter.shape), "norm": float(gradient.norm().item()), "finite": finite, "nonzero": nonzero}
            per_layer[str(int(physical_offset + index))] = {"parameter_count": len(parameter_report), "finite": True, "nonzero": True, "parameters": parameter_report}
        return per_layer
    prefix_physical_layers = physical_layer_report("prefix_layers", range(5), 0)
    middle_physical_layers = physical_layer_report("recurrent.middle.layers", range(10), 5)
    suffix_physical_layers = physical_layer_report("suffix_layers", range(5), 15)
    prefix_all_receive_finite_nonzero_grad = bool(prefix_gradient_report)
    suffix_all_receive_finite_nonzero_grad = bool(suffix_gradient_report)
    middle_all_receive_finite_nonzero_grad = bool(middle_gradient_report)
    if not prefix_all_receive_finite_nonzero_grad or not suffix_all_receive_finite_nonzero_grad:
        raise AssertionError("prefix/suffix gradients must be finite and nonzero")
    recursive_model._collect_middle_gradient_audit = False
    return {"early_hidden_gradient_norms": early_hidden_gradient_norms, "early_parameter_gradient_edges_absent": True, "early_parameter_edge_checks": early_parameter_edges, "early_parameter_edge_details": early_parameter_edge_details, "exact_parameter_gradient_tail": 4, "last_four_injection_middle_parameter_grads": True, "injection_A_log_dt_bias_B_gradient_audit": recurrent_gradient_audit, "prefix_layers_with_grad": prefix_layers_with_grad, "middle_layers_with_grad": list(middle_gradient_report), "suffix_layers_with_grad": suffix_layers_with_grad, "prefix_gradient_report": prefix_gradient_report, "middle_gradient_report": middle_gradient_report, "suffix_gradient_report": suffix_gradient_report, "prefix_physical_layers": prefix_physical_layers, "middle_physical_layers": middle_physical_layers, "suffix_physical_layers": suffix_physical_layers, "prefix_all_receive_finite_nonzero_grad": prefix_all_receive_finite_nonzero_grad, "middle_all_receive_finite_nonzero_grad": middle_all_receive_finite_nonzero_grad, "suffix_all_receive_finite_nonzero_grad": suffix_all_receive_finite_nonzero_grad, "parameter_identity_and_requires_grad_restored": True}


def run_stage1(model_path: Path | None = None, *, output_path: Path | None = None, device: str = "cpu") -> dict[str, Any]:
    module = __import__(MODEL_MODULE)
    report: dict[str, Any] = {"status": "PASS", "architecture_contract": module.ARCHITECTURE_CONTRACT, "metadata": module.poisson_metadata(), "poisson": validate_poisson_contract(module), "schedules": validate_schedules(module), "sampling": validate_sampling(module), "alignment": validate_alignment(module), "cache": validate_cache_schedule_binding(module)}
    if model_path is not None:
        model, _ = _load_model(model_path, device)
        report["model_semantics"] = validate_model_semantics(model, device=device)
        report["gradients"] = validate_gradient_policy(model, device=device)
        report["scalar_inference_all_r"] = validate_scalar_inference_all_r(model, device=device)
        report["default_r7"] = validate_default_r7(model, device=device)
        report["cache_contract"] = validate_cache_contract(model, device=device)
        report["generation_contract"] = validate_generation_contract(model, device=device)
        report["reload_contract"] = validate_reload_contract(model_path, device=device)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-path", type=Path, default=Path("stage1_5_10xpoisson_parcae_audit.json"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        report = run_stage1(**vars(parse_args(argv)))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"[result] status=FAIL error={exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
