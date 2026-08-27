#!/usr/bin/env python3
"""Offline smoke test for a converted recursive Llama checkpoint."""

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

from recursive_model import (  # noqa: E402
    DEFAULT_LOOPS,
    MAPPING_POLICY,
    PROJECT_SOURCE_LAYERS,
    RecursiveLlamaForCausalLM,
    build_stepwise_mapping,
    parameter_audit,
    register_auto_class,
)


CACHE_ATOL = 1e-5
CACHE_RTOL = 1e-4


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


def cache_slot_state(cache: Any, index: int) -> tuple[int, bool, bool]:
    """Return (sequence length, key non-empty, value non-empty) for one slot."""

    length = int(cache.get_seq_length(index))
    key_state: Any = None
    value_state: Any = None
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        key_cache = cache.key_cache
        value_cache = cache.value_cache
        if index < len(key_cache):
            key_state = key_cache[index]
        if index < len(value_cache):
            value_state = value_cache[index]
    elif hasattr(cache, "layers"):
        layers = cache.layers
        if index < len(layers):
            layer = layers[index]
            key_state = getattr(layer, "keys", None)
            value_state = getattr(layer, "values", None)
    else:  # pragma: no cover - defensive for a future cache class
        try:
            slot = cache[index]
            key_state, value_state = slot[:2]
        except (IndexError, TypeError, KeyError):
            pass
    key_nonempty = isinstance(key_state, torch.Tensor) and key_state.numel() > 0
    value_nonempty = isinstance(value_state, torch.Tensor) and value_state.numel() > 0
    return length, key_nonempty, value_nonempty


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    model_path = args.model_path.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
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
    if (
        logical_layers <= 0
        or physical_layers <= 0
        or loops != DEFAULT_LOOPS
        or not mapping
        or mapping_1based != [int(index) + 1 for index in mapping]
        or logical_layers != physical_layers * int(loops)
        or len(mapping) != physical_layers
        or mapping_policy != MAPPING_POLICY
    ):
        raise RuntimeError(
            "Invalid recursive config: "
            f"logical_layers={logical_layers} physical_unique_layers={physical_layers} loops={loops} "
            f"expected_product={physical_layers * int(loops) if loops is not None else None} "
            f"mapping_0based={mapping} mapping_1based={mapping_1based}"
        )
    expected_production_mapping = list(build_stepwise_mapping(PROJECT_SOURCE_LAYERS))
    if list(mapping) != expected_production_mapping:
        raise RuntimeError(
            "Recursive smoke only accepts the production Stage 1 mapping: "
            f"expected={expected_production_mapping} got={list(mapping)}"
        )
    print(
        f"[config] logical_layers={logical_layers} physical_unique_layers={physical_layers} "
        f"source_layers={logical_layers} loops={loops} "
        f"mapping_policy={getattr(config, 'recursive_mapping_policy', None)} "
        f"mapping_1based={mapping_1based}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    text = os.environ.get("RSMOL_RECURSIVE_SMOKE_PROMPT", "Recursive models are")
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
    incremental_max_diff = float((incremental.logits[:, -1] - extended.logits[:, -1]).abs().max().item())
    if not torch.allclose(
        incremental.logits[:, -1],
        extended.logits[:, -1],
        atol=CACHE_ATOL,
        rtol=CACHE_RTOL,
    ):
        raise RuntimeError(
            "Incremental logits disagree with independent full-sequence logits: "
            f"max_diff={incremental_max_diff} atol={CACHE_ATOL} rtol={CACHE_RTOL}"
        )
    incremental_slots = []
    for index in range(expected_slots):
        slot = cache_slot_state(cache, index)
        incremental_slots.append({"index": index, "length": slot[0], "key_nonempty": slot[1], "value_nonempty": slot[2]})
        if slot[0] != inputs["input_ids"].shape[1] + 1 or not slot[1] or not slot[2]:
            raise RuntimeError(f"Invalid incremental cache slot {index}: {slot}")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    precreated_cache = DynamicCache(config=model.config)
    if len(precreated_cache) < expected_slots:
        raise RuntimeError(
            "DynamicCache(config=model.config) did not pre-create the logical namespace: "
            f"capacity={len(precreated_cache)} expected={expected_slots}"
        )
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
    torch.cuda.synchronize(device_index)
    if generated.ndim != 2 or generated.shape[0] != inputs["input_ids"].shape[0]:
        raise RuntimeError(f"Unexpected generation shape: {tuple(generated.shape)}")
    if generated.shape[1] <= inputs["input_ids"].shape[1]:
        raise RuntimeError("Generation produced no new token")
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
        },
        "parameter_audit": parameter_audit(model),
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
            "atol": CACHE_ATOL,
            "rtol": CACHE_RTOL,
        },
        "generation": {"output_shape": list(generated.shape), "seconds": time.perf_counter() - started},
        "generation_precreated_cache": {
            "cache_type": type(precreated_cache).__name__,
            "cache_slots": len(precreated_cache),
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
