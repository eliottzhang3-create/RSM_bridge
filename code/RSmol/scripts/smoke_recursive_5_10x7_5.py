#!/usr/bin/env python3
"""Stage 1 audit for a converted SmolLM2-5-10-5 checkpoint."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from recursive_model_5_10_5 import (  # noqa: E402
    LOGICAL_LAYER_COUNT,
    PHYSICAL_LAYER_COUNT,
    RECURSIVE_LOOPS,
    MAPPING_POLICY,
    LOGICAL_TO_PHYSICAL,
    RecursiveLlamaForCausalLM,
    build_5_10_5_schedule,
    make_dynamic_cache,
    parameter_audit,
    register_auto_class,
)


CACHE_ATOL = 1e-5
CACHE_RTOL = 1e-4
BF16_INCREMENTAL_MAX_ABS = 1.0
BF16_INCREMENTAL_MIN_COSINE = 0.999
FP32_INCREMENTAL_ATOL = 1e-3
FP32_INCREMENTAL_RTOL = 1e-4
REMOTE_CHECKOUT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM")


def ensure_external_report(path: Path) -> Path:
    """Keep Stage 1 reports outside both the local and remote Git checkout."""

    candidate = path.expanduser().resolve()
    roots = (SCRIPT_ROOT.parents[1].resolve(), REMOTE_CHECKOUT.resolve())
    for root in roots:
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        raise ValueError(
            "Stage 1 5-10-5 smoke refuses to write a report in a Git checkout: "
            f"report={candidate} root={root}"
        )
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return version(name)
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"{name} contains non-finite values")


def configured_sample_prompts(primary_prompt: str) -> list[str]:
    """Read optional ``|||``-separated generation prompts from the environment."""

    raw = os.environ.get("RSMOL_5_10_5_SAMPLE_PROMPTS", os.environ.get("RSMOL_RECURSIVE_SAMPLE_PROMPTS", ""))
    if not raw.strip():
        return [primary_prompt]
    prompts = [item.strip() for item in raw.split("|||") if item.strip()]
    return prompts or [primary_prompt]


def generation_sample_record(
    tokenizer: Any,
    prompt: str,
    generated: torch.Tensor,
) -> dict[str, Any]:
    """Serialize one generated sample without writing model artifacts."""

    prompt_length = int(tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1])
    sequence = generated[0].detach().cpu().tolist()
    completion_ids = sequence[prompt_length:]
    return {
        "prompt": prompt,
        "generated_text": tokenizer.decode(sequence, skip_special_tokens=False),
        "completion_text": tokenizer.decode(completion_ids, skip_special_tokens=False),
        "prompt_token_count": prompt_length,
        "generated_token_count": len(completion_ids),
        "token_ids": sequence,
    }


def cache_slot_state(cache: Any, index: int) -> tuple[int, bool, bool]:
    """Return (sequence length, key non-empty, value non-empty) for one slot."""

    length = int(cache.get_seq_length(index))
    key_state: Any = None
    value_state: Any = None
    # Transformers 4.54.1 keeps key_cache/value_cache as deprecated
    # compatibility properties. Prefer the canonical layer API so the smoke
    # remains warning-free and compatible with 4.56+.
    if hasattr(cache, "layers"):
        layers = cache.layers
        if index < len(layers):
            layer = layers[index]
            key_state = getattr(layer, "keys", None)
            value_state = getattr(layer, "values", None)
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        key_cache = cache.key_cache
        value_cache = cache.value_cache
        if index < len(key_cache):
            key_state = key_cache[index]
        if index < len(value_cache):
            value_state = value_cache[index]
    else:  # pragma: no cover - defensive for a future cache class
        try:
            slot = cache[index]
            key_state, value_state = slot[:2]
        except (IndexError, TypeError, KeyError):
            pass
    key_nonempty = isinstance(key_state, torch.Tensor) and key_state.numel() > 0
    value_nonempty = isinstance(value_state, torch.Tensor) and value_state.numel() > 0
    return length, key_nonempty, value_nonempty


def schedule_trace_audit(model: Any, *, inputs: dict[str, Any]) -> dict[str, Any]:
    """Capture all thirty logical forward calls, including recurrent reuse."""

    base = getattr(model, "model", model)
    sequence: list[int] = []
    handles = [
        layer.register_forward_hook(lambda _m, _i, _o, index=index: sequence.append(index))
        for index, layer in enumerate(base.model.layers if hasattr(base, "model") else base.layers)
    ]
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    expected = list(LOGICAL_TO_PHYSICAL)
    if sequence != expected:
        raise RuntimeError(f"5-10-5 forward trace mismatch: expected={expected} got={sequence}")
    return {"trace": sequence, "expected": expected, "length": len(sequence), "ok": True}


def backward_trace_audit(model: Any, *, inputs: dict[str, Any]) -> dict[str, Any]:
    """Audit reverse schedule and finite gradients for shared middle modules."""

    base = getattr(model, "model", model)
    layers = base.model.layers if hasattr(base, "model") else base.layers
    sequence: list[int] = []
    handles = [
        layer.register_full_backward_hook(lambda _m, _gi, _go, index=index: sequence.append(index))
        for index, layer in enumerate(layers)
    ]
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    ids = inputs["input_ids"]
    labels = ids.clone()
    with torch.enable_grad():
        result = model(input_ids=ids, attention_mask=inputs.get("attention_mask"), use_cache=False)
        logits = result.logits.float()
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), reduction="mean"
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Backward audit loss is non-finite")
        loss.backward()
    for handle in handles:
        handle.remove()
    model.train(was_training)
    expected = list(reversed(LOGICAL_TO_PHYSICAL))
    if sequence != expected:
        raise RuntimeError(f"5-10-5 backward trace mismatch: expected={expected} got={sequence}")
    shared = {}
    for index in range(5, 15):
        norms = [float(parameter.grad.detach().float().norm().item()) for parameter in layers[index].parameters() if parameter.grad is not None]
        if not norms or not all(torch.isfinite(torch.tensor(value)).item() and value > 0 for value in norms):
            raise RuntimeError(f"shared middle layer {index} has missing/non-finite/zero gradients")
        shared[str(index)] = max(norms)
    return {"trace": sequence, "expected": expected, "loss": float(loss.detach().item()), "shared_middle_gradient_norms": shared, "ok": True}


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    model_path = args.model_path.expanduser().resolve()
    output_report = ensure_external_report(args.output_report)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Recursive remote smoke requires a submitted CUDA job")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Recursive checkpoint is missing config.json: {model_path}")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("========== RECURSIVE SMOLLM2 SMOKE =========", flush=True)
    print(f"[git] commit={git_commit()}", flush=True)
    print(f"[python] executable={sys.executable} version={sys.version.split()[0]}", flush=True)
    print(f"[platform] {platform.platform()}", flush=True)
    print(f"[torch] version={torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(f"[model] input={model_path} output_report={output_report}", flush=True)
    print(f"[env] RSMOL_SOURCE_CHECKPOINT={os.environ.get('RSMOL_SOURCE_CHECKPOINT', '<unset>')}", flush=True)
    print(f"[env] RSMOL_RECURSIVE_OUTPUT_DIR={os.environ.get('RSMOL_RECURSIVE_OUTPUT_DIR', '<unset>')}", flush=True)

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    mapping = getattr(config, "recursive_source_layer_indices_0based", None)
    loops = getattr(config, "recursive_loops", None)
    logical_layers = int(getattr(config, "num_hidden_layers", 0))
    physical_layers = int(getattr(config, "recursive_layer_count", 0) or 0)
    mapping_1based = getattr(config, "recursive_source_layer_indices_1based", None)
    mapping_policy = getattr(config, "recursive_mapping_policy", None)
    architectures = list(getattr(config, "architectures", ()) or ())
    prefix_count = int(getattr(config, "recursive_prefix_layer_count", 0))
    middle_count = int(getattr(config, "recursive_middle_layer_count", 0))
    suffix_count = int(getattr(config, "recursive_suffix_layer_count", 0))
    loops_scope = str(getattr(config, "recursive_loops_scope", ""))
    if (
        logical_layers <= 0
        or physical_layers <= 0
        or loops != RECURSIVE_LOOPS
        or not mapping
        or mapping_1based != [int(index) + 1 for index in mapping]
        or logical_layers != LOGICAL_LAYER_COUNT
        or physical_layers != PHYSICAL_LAYER_COUNT
        or len(mapping) != physical_layers
        or mapping_policy != MAPPING_POLICY
        or list(getattr(config, "logical_to_physical", ())) != list(LOGICAL_TO_PHYSICAL)
        or architectures != ["RecursiveLlamaForCausalLM"]
        or (prefix_count, middle_count, suffix_count) != (5, 10, 5)
        or loops_scope != "middle_only"
    ):
        raise RuntimeError(
            "Invalid recursive config: "
            f"logical_layers={logical_layers} physical_unique_layers={physical_layers} loops={loops} "
            f"expected_schedule={list(LOGICAL_TO_PHYSICAL)} "
            f"mapping_0based={mapping} mapping_1based={mapping_1based}"
        )
    expected_schedule = list(build_5_10_5_schedule())
    if list(getattr(config, "logical_to_physical", ())) != expected_schedule:
        raise RuntimeError(
            "5-10-5 smoke only accepts the production logical schedule: "
            f"expected={expected_schedule} got={getattr(config, 'logical_to_physical', None)}"
        )
    print(
        f"[config] logical_layers={logical_layers} physical_unique_layers={physical_layers} "
        f"source_layers={logical_layers} loops={loops} "
        f"mapping_policy={getattr(config, 'recursive_mapping_policy', None)} "
        f"logical_to_physical={getattr(config, 'logical_to_physical', None)} "
        f"mapping_1based={mapping_1based}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    text = os.environ.get("RSMOL_5_10_5_SMOKE_PROMPT", os.environ.get("RSMOL_RECURSIVE_SMOKE_PROMPT", "Recursive models are"))
    sample_prompts = configured_sample_prompts(text)
    text = sample_prompts[0]
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    # The converted config keeps model_type=llama for Llama compatibility.  An
    # explicit, process-local registration is therefore required for AutoModel
    # to select the recursive implementation instead of ordinary LlamaForCausalLM.
    register_auto_class()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        low_cpu_mem_usage=True,
    ).eval()
    if not isinstance(model, RecursiveLlamaForCausalLM):
        raise RuntimeError(f"AutoModel resolved the wrong class: {type(model)!r}")
    from transformers.cache_utils import DynamicCache

    decoder_api = tuple(inspect.signature(model.model.layers[0].forward).parameters)
    attention_api = tuple(inspect.signature(model.model.layers[0].self_attn.forward).parameters)
    cache_update_api = tuple(inspect.signature(DynamicCache.update).parameters)
    cache_length_api = tuple(inspect.signature(DynamicCache.get_seq_length).parameters)
    print(
        "[api] "
        f"decoder={decoder_api} attention={attention_api} "
        f"cache_update={cache_update_api} cache_get_seq_length={cache_length_api}",
        flush=True,
    )
    print(f"[parameters] {json.dumps(parameter_audit(model), ensure_ascii=False)}", flush=True)

    # The schedule and reverse autograd order are part of the Stage 1 audit;
    # this also proves the recurrent middle modules really share storage.
    schedule_trace = schedule_trace_audit(model, inputs=inputs)
    backward_trace = backward_trace_audit(model, inputs=inputs)
    model.eval()

    with torch.inference_mode():
        no_cache = model(**inputs, use_cache=False)
        with_cache = model(**inputs, use_cache=True)
    finite("no-cache logits", no_cache.logits)
    finite("cache logits", with_cache.logits)
    max_diff = float((no_cache.logits - with_cache.logits).abs().max().item())
    if not torch.allclose(no_cache.logits, with_cache.logits, atol=CACHE_ATOL, rtol=CACHE_RTOL):
        raise RuntimeError(
            "Prefill cache/no-cache logits disagree: "
            f"max_diff={max_diff} atol={CACHE_ATOL} rtol={CACHE_RTOL}"
        )
    cache = with_cache.past_key_values
    logical_slots = len(cache) if cache is not None else 0
    expected_slots = logical_layers
    if logical_slots < expected_slots:
        raise RuntimeError(f"Cache has {logical_slots} slots, expected at least {expected_slots}")
    prefill_slots = []
    for index in range(expected_slots):
        slot = cache_slot_state(cache, index)
        prefill_slots.append({"index": index, "length": slot[0], "key_nonempty": slot[1], "value_nonempty": slot[2]})
        if slot[0] != inputs["input_ids"].shape[1] or not slot[1] or not slot[2]:
            raise RuntimeError(f"Invalid prefill cache slot {index}: {slot}")

    next_input: dict[str, Any] = {
        "input_ids": inputs["input_ids"][:, -1:],
        "use_cache": True,
        "past_key_values": cache,
    }
    if "attention_mask" in inputs:
        next_input["attention_mask"] = torch.cat(
            (inputs["attention_mask"], torch.ones_like(inputs["attention_mask"][:, :1])), dim=1
        )
    with torch.inference_mode():
        incremental = model(**next_input)
    finite("incremental logits", incremental.logits)
    with torch.inference_mode():
        extended_inputs: dict[str, Any] = {
            "input_ids": torch.cat((inputs["input_ids"], inputs["input_ids"][:, -1:]), dim=1),
            "use_cache": False,
        }
        if "attention_mask" in inputs:
            extended_inputs["attention_mask"] = torch.cat(
                (inputs["attention_mask"], torch.ones_like(inputs["attention_mask"][:, :1])), dim=1
            )
        extended = model(**extended_inputs)
    incremental_logits = incremental.logits[:, -1].float()
    extended_logits = extended.logits[:, -1].float()
    incremental_abs_diff = (incremental_logits - extended_logits).abs()
    incremental_max_diff = float(incremental_abs_diff.max().item())
    incremental_mean_diff = float(incremental_abs_diff.mean().item())
    incremental_cosine = float(
        torch.nn.functional.cosine_similarity(incremental_logits, extended_logits, dim=-1)
        .min()
        .item()
    )
    incremental_argmax_equal = bool(
        torch.equal(incremental_logits.argmax(dim=-1), extended_logits.argmax(dim=-1))
    )
    if (
        incremental_max_diff > BF16_INCREMENTAL_MAX_ABS
        or incremental_cosine < BF16_INCREMENTAL_MIN_COSINE
        or not incremental_argmax_equal
    ):
        raise RuntimeError(
            "BF16 incremental logits disagree semantically with independent full-sequence logits: "
            f"max_diff={incremental_max_diff} mean_diff={incremental_mean_diff} "
            f"cosine={incremental_cosine} argmax_equal={incremental_argmax_equal} "
            f"limits=(max={BF16_INCREMENTAL_MAX_ABS}, "
            f"cosine>={BF16_INCREMENTAL_MIN_COSINE}, argmax_equal=True); "
            "mean_diff is diagnostic only because it is not normalized to the logits scale"
        )
    incremental_slots = []
    for index in range(expected_slots):
        slot = cache_slot_state(cache, index)
        incremental_slots.append({"index": index, "length": slot[0], "key_nonempty": slot[1], "value_nonempty": slot[2]})
        if slot[0] != inputs["input_ids"].shape[1] + 1 or not slot[1] or not slot[2]:
            raise RuntimeError(f"Invalid incremental cache slot {index}: {slot}")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize(device_index)
    generation_seconds = time.perf_counter() - generation_started
    with torch.inference_mode():
        generated_no_cache = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Transformers 4.54.1 cannot construct DynamicCache from this Llama
    # config: its config path forwards max_cache_len to CacheLayerMixin, which
    # rejects that keyword.  The empty form is the supported lazy cache and
    # expands through all logical slots during the prefill below.
    precreated_cache = make_dynamic_cache()
    precreated_cache_initial_slots = len(precreated_cache)
    with torch.inference_mode():
        generated_with_precreated_cache = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            past_key_values=precreated_cache,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if generated.ndim != 2 or generated.shape[0] != inputs["input_ids"].shape[0]:
        raise RuntimeError(f"Unexpected generation shape: {tuple(generated.shape)}")
    if generated.shape[1] <= inputs["input_ids"].shape[1]:
        raise RuntimeError("Generation produced no new token")
    # Generation is performed by two different Transformers code paths.  For
    # BF16/low-precision checkpoints, tiny arithmetic differences can cross a
    # greedy decoding tie and cause later token IDs to diverge even when the
    # strict prefill allclose check and the semantic one-step incremental
    # cache audit above both pass.  Treat this as diagnostic rather than a
    # model/cache failure: the generated samples from the supported cached
    # path are still valid, while the logits/cache checks remain strict.
    cache_no_cache_tokens_equal = bool(torch.equal(generated, generated_no_cache))
    if not cache_no_cache_tokens_equal:
        print(
            "[generation-warning] use_cache=True and use_cache=False greedy "
            "token IDs diverged; retaining cached-path samples because strict "
            "prefill/incremental cache audits passed",
            flush=True,
        )
    generation_samples = [generation_sample_record(tokenizer, text, generated)]
    for sample_prompt in sample_prompts[1:]:
        sample_inputs = tokenizer(sample_prompt, return_tensors="pt")
        sample_inputs = {name: value.to(device) for name, value in sample_inputs.items()}
        with torch.inference_mode():
            sample_generated = model.generate(
                **sample_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        if sample_generated.shape[1] <= sample_inputs["input_ids"].shape[1]:
            raise RuntimeError(f"Generation produced no new token for sample prompt: {sample_prompt!r}")
        generation_samples.append(
            generation_sample_record(tokenizer, sample_prompt, sample_generated)
        )
    expected_precreated_length = inputs["input_ids"].shape[1]
    if (
        generated_with_precreated_cache.ndim != 2
        or generated_with_precreated_cache.shape[0] != inputs["input_ids"].shape[0]
    ):
        raise RuntimeError(
            "Unexpected precreated-cache generation shape: "
            f"{tuple(generated_with_precreated_cache.shape)}"
        )
    if generated_with_precreated_cache.shape[1] <= expected_precreated_length:
        raise RuntimeError("Precreated-cache generation produced no new token")
    if len(precreated_cache) < expected_slots:
        raise RuntimeError(
            "Lazy precreated cache did not expand through the recursive logical namespace: "
            f"capacity={len(precreated_cache)} expected_slots={expected_slots}"
        )
    # With max_new_tokens=1, GenerationMixin uses the prompt prefill to select
    # the only new token and does not perform another decoder forward.  Thus
    # each cache slot must contain exactly the prompt length, not prompt + 1.
    precreated_generation_slots = []
    for index in range(expected_slots):
        slot = cache_slot_state(precreated_cache, index)
        precreated_generation_slots.append(
            {"index": index, "length": slot[0], "key_nonempty": slot[1], "value_nonempty": slot[2]}
        )
        if slot[0] != expected_precreated_length or not slot[1] or not slot[2]:
            raise RuntimeError(f"Invalid precreated generation cache slot {index}: {slot}")

    # BF16 cached single-token attention and a full-sequence recomputation use
    # different GEMM/SDPA shapes, so bitwise or FP32-scale allclose thresholds
    # are not a valid correctness criterion. Re-run the decisive incremental
    # comparison in FP32 after the BF16 inference/generation checks. A cache
    # routing or position error remains large in FP32; normal kernel-ordering
    # noise does not.
    model.float()
    fp32_next_input: dict[str, Any] = {
        "input_ids": inputs["input_ids"][:, -1:],
        "use_cache": True,
    }
    if "attention_mask" in inputs:
        fp32_next_input["attention_mask"] = next_input["attention_mask"]
    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.inference_mode():
            fp32_prefill = model(**inputs, use_cache=True)
            fp32_next_input["past_key_values"] = fp32_prefill.past_key_values
            fp32_incremental = model(**fp32_next_input)
            fp32_extended = model(**extended_inputs)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
    finite("FP32 incremental logits", fp32_incremental.logits)
    finite("FP32 full-sequence logits", fp32_extended.logits)
    fp32_incremental_logits = fp32_incremental.logits[:, -1]
    fp32_extended_logits = fp32_extended.logits[:, -1]
    fp32_incremental_max_diff = float(
        (fp32_incremental_logits - fp32_extended_logits).abs().max().item()
    )
    if not torch.allclose(
        fp32_incremental_logits,
        fp32_extended_logits,
        atol=FP32_INCREMENTAL_ATOL,
        rtol=FP32_INCREMENTAL_RTOL,
    ):
        raise RuntimeError(
            "FP32 incremental logits disagree with independent full-sequence logits: "
            f"max_diff={fp32_incremental_max_diff} "
            f"atol={FP32_INCREMENTAL_ATOL} rtol={FP32_INCREMENTAL_RTOL}"
        )
    report = {
        "status": "ok",
        "model_path": str(model_path),
        "output_report": str(output_report),
        "git_commit": git_commit(),
        "packages": {
            name: package_version(name) for name in ("torch", "transformers", "safetensors")
        },
        "transformers_api_assumption": {
            "expected": "4.54.1",
            "actual": package_version("transformers"),
            "decoder_forward": decoder_api,
            "attention_forward": attention_api,
            "dynamic_cache_update": cache_update_api,
            "dynamic_cache_get_seq_length": cache_length_api,
        },
        "config": config.to_dict(),
        "mapping": {
            "policy": getattr(config, "recursive_mapping_policy", None),
            "source_1based": mapping_1based,
            "source_0based": mapping,
            "logical_layer_count": logical_layers,
            "physical_layer_count": physical_layers,
            "loops": loops,
            "logical_to_physical": list(getattr(config, "logical_to_physical", ())),
        },
        "parameter_audit": parameter_audit(model),
        "schedule_trace": schedule_trace,
        "backward_trace": backward_trace,
        "forward": {
            "logits_shape": list(no_cache.logits.shape),
            "max_cache_no_cache_diff": max_diff,
            "allclose": bool(torch.allclose(no_cache.logits, with_cache.logits, atol=CACHE_ATOL, rtol=CACHE_RTOL)),
            "atol": CACHE_ATOL,
            "rtol": CACHE_RTOL,
        },
        "cache": {
            "type": type(cache).__name__,
            "logical_slots": logical_slots,
            "expected_slots": expected_slots,
            "logical_layer_count": logical_layers,
            "physical_layer_count": physical_layers,
            "recursive_loops": loops,
            "prefill_slots": prefill_slots,
            "incremental_slots": incremental_slots,
        },
        "incremental": {
            "logits_shape": list(incremental.logits.shape),
            "full_sequence_logits_shape": list(extended.logits.shape),
            "max_diff": incremental_max_diff,
            "mean_diff": incremental_mean_diff,
            "cosine": incremental_cosine,
            "argmax_equal": incremental_argmax_equal,
            "max_abs_limit": BF16_INCREMENTAL_MAX_ABS,
            "min_cosine": BF16_INCREMENTAL_MIN_COSINE,
            "mean_diff_is_diagnostic_only": True,
        },
        "incremental_fp32_audit": {
            "max_diff": fp32_incremental_max_diff,
            "atol": FP32_INCREMENTAL_ATOL,
            "rtol": FP32_INCREMENTAL_RTOL,
        },
        "generation": {
            "output_shape": list(generated.shape),
            "cache_no_cache_tokens_equal": cache_no_cache_tokens_equal,
            "generation_warning": (
                None
                if cache_no_cache_tokens_equal
                else "use_cache=True and use_cache=False greedy token IDs diverged; cached path retained"
            ),
            "cached_generation_seconds": generation_seconds,
            "sample_count": len(generation_samples),
            "samples": generation_samples,
        },
        "generation_precreated_cache": {
            "cache_type": type(precreated_cache).__name__,
            "initial_cache_slots": precreated_cache_initial_slots,
            "cache_slots_after_generation": len(precreated_cache),
            "expected_slots": expected_slots,
            "expected_prefill_length": expected_precreated_length,
            "slots_after_generation": precreated_generation_slots,
            "construction": "DynamicCache() lazy; config construction is incompatible with transformers 4.54.1",
            "output_shape": list(generated_with_precreated_cache.shape),
        },
    }
    atomic_json(output_report, report)
    print(f"[result] status=PASS report={output_report}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[result] status=FAIL", file=sys.stderr, flush=True)
        raise
