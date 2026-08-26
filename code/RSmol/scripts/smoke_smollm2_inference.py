#!/usr/bin/env python3
"""Run a submitted, offline SmolLM2-135M load/forward/generation smoke test."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch


EXPECTED_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "num_hidden_layers": 30,
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "vocab_size": 49152,
    "max_position_embeddings": 8192,
}

REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="Gravity is")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("outputs/RSmol/smollm2_inference_smoke.json"),
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{__import__('os').getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def package_version(name: str) -> str:
    try:
        return version(name)
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def require_model_files(model_path: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"SmolLM2 snapshot is incomplete: missing={missing} path={model_path}"
        )


def config_summary(config: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in EXPECTED_CONFIG:
        value = getattr(config, key, None)
        if isinstance(value, tuple):
            value = list(value)
        summary[key] = value
    dtype = getattr(config, "torch_dtype", None)
    summary["torch_dtype"] = str(dtype).replace("torch.", "")
    return summary


def validate_config(config: Any) -> dict[str, Any]:
    summary = config_summary(config)
    mismatches = {}
    for key, expected in EXPECTED_CONFIG.items():
        if summary.get(key) != expected:
            mismatches[key] = {
                "expected": expected,
                "actual": summary.get(key),
            }
    if mismatches:
        raise RuntimeError(f"Unexpected SmolLM2 config: {mismatches}")
    if summary.get("torch_dtype") != "bfloat16":
        raise RuntimeError(
            f"Expected BF16 checkpoint config, actual torch_dtype={summary.get('torch_dtype')}"
        )
    return summary


def finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"{name} contains non-finite values")


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError(f"--max-new-tokens must be positive, got {args.max_new_tokens}")

    model_path = args.model_path.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    require_model_files(model_path)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "SmolLM2 inference smoke requires CUDA and must run inside a submitted GPU job"
        )

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"This smoke test is GPU-only; got device={device}")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("========== SMOLLM2-135M INFERENCE SMOKE ==========", flush=True)
    print(
        f"[python] version={sys.version.split()[0]} executable={sys.executable}",
        flush=True,
    )
    print(f"[platform] {platform.platform()}", flush=True)
    print(
        f"[torch] version={torch.__version__} cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name(device_index)}",
        flush=True,
    )
    print(f"[model] path={model_path}", flush=True)
    print("[model] local_files_only=True", flush=True)
    print(f"[model] transformers={package_version('transformers')}", flush=True)

    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model_config = validate_config(config)
    print(f"[config] {json.dumps(model_config, ensure_ascii=False)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
        print(
            f"[tokenizer] pad_token_id=None; generation will use eos_token_id={pad_token_id} "
            "and this does not modify the checkpoint",
            flush=True,
        )

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_length = int(inputs["input_ids"].shape[1])
    if prompt_length <= 0:
        raise RuntimeError("Tokenizer produced an empty prompt")
    print(
        f"[tokenizer] class={type(tokenizer).__name__} vocab_size={len(tokenizer)} "
        f"prompt_tokens={prompt_length}",
        flush=True,
    )

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_dtypes: dict[str, int] = {}
    for parameter in model.parameters():
        key = str(parameter.dtype).replace("torch.", "")
        parameter_dtypes[key] = parameter_dtypes.get(key, 0) + int(parameter.numel())
    print(
        f"[load] class={model.__class__.__module__}.{model.__class__.__name__} "
        f"parameters={parameter_count} dtypes={parameter_dtypes} seconds={load_seconds:.3f}",
        flush=True,
    )

    model_device = next(model.parameters()).device
    if model_device.type != "cuda":
        raise RuntimeError(f"Model was not placed on CUDA: model_device={model_device}")

    torch.cuda.synchronize(device_index)
    forward_started = time.perf_counter()
    with torch.inference_mode():
        forward = model(**inputs, use_cache=True)
    torch.cuda.synchronize(device_index)
    forward_seconds = time.perf_counter() - forward_started

    logits = forward.logits
    expected_logits_shape = (1, prompt_length, int(config.vocab_size))
    if tuple(logits.shape) != expected_logits_shape:
        raise RuntimeError(
            f"Unexpected forward logits shape: expected={expected_logits_shape} "
            f"actual={tuple(logits.shape)}"
        )
    finite_tensor(logits, "forward logits")

    cache = getattr(forward, "past_key_values", None)
    if cache is None:
        raise RuntimeError("Forward did not return past_key_values with use_cache=True")
    cache_length = None
    if hasattr(cache, "get_seq_length"):
        cache_length = int(cache.get_seq_length())
        if cache_length != prompt_length:
            raise RuntimeError(
                f"Unexpected cache length: expected={prompt_length} actual={cache_length}"
            )
    print(
        f"[forward] logits_shape={tuple(logits.shape)} cache_type={type(cache).__name__} "
        f"cache_length={cache_length} seconds={forward_seconds:.3f}",
        flush=True,
    )

    torch.cuda.reset_peak_memory_stats(device_index)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize(device_index)
    generation_seconds = time.perf_counter() - generation_started

    if generated.ndim != 2 or generated.shape[0] != 1:
        raise RuntimeError(f"Unexpected generated shape: {tuple(generated.shape)}")
    generated_length = int(generated.shape[1])
    if generated_length <= prompt_length:
        raise RuntimeError(
            f"Generation produced no new tokens: prompt={prompt_length} output={generated_length}"
        )
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
    new_token_ids = generated[0, prompt_length:].tolist()
    print(
        f"[generate] output_tokens={generated_length} new_tokens={len(new_token_ids)} "
        f"seconds={generation_seconds:.3f}",
        flush=True,
    )
    print(f"[generate] text={generated_text!r}", flush=True)

    report = {
        "status": "ok",
        "model_path": str(model_path),
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "tokenizers": package_version("tokenizers"),
            "safetensors": package_version("safetensors"),
            "accelerate": package_version("accelerate"),
        },
        "device": {
            "index": device_index,
            "name": torch.cuda.get_device_name(device_index),
            "torch_cuda": torch.version.cuda,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
        },
        "config": model_config,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "generation_pad_token_id": pad_token_id,
        },
        "parameter_count": parameter_count,
        "parameter_dtypes": parameter_dtypes,
        "prompt": args.prompt,
        "prompt_length": prompt_length,
        "forward": {
            "logits_shape": list(logits.shape),
            "cache_type": type(cache).__name__,
            "cache_length": cache_length,
            "seconds": forward_seconds,
        },
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "output_length": generated_length,
            "new_token_ids": new_token_ids,
            "text": generated_text,
            "seconds": generation_seconds,
        },
        "load_seconds": load_seconds,
    }
    atomic_write_json(output_report, report)
    print(f"[result] status=OK output_report={output_report}", flush=True)


if __name__ == "__main__":
    main()
