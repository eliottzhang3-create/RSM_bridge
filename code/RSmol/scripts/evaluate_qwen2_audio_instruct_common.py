#!/usr/bin/env python3
"""Shared Qwen2-Audio-Instruct runtime for isolated MMAU/MMAR evaluation."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MODEL_PATH = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen2Audio-Instruct"
)
MODEL_CONTRACT = "qwen2_audio_instruct_local_huggingface_bf16_v1"
CHAT_CONTRACT = "single_user_audio_then_benchmark_prompt_no_system_v1"
SOURCE_AUDIO_SAMPLE_RATE = 32_000
NOT_APPLICABLE_PATH = Path("QWEN2_AUDIO_NOT_APPLICABLE")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _chat_template(processor: Any) -> str:
    template = getattr(processor, "chat_template", None)
    if template is None:
        template = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
    return str(template or "")


def audit_model_artifact(
    model_path: Path,
    *,
    load_processor: bool = False,
) -> dict[str, Any]:
    """Audit the local Hugging Face artifact without reading remote resources."""

    model_path = Path(model_path)
    required = (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
    )
    if not model_path.is_dir():
        raise FileNotFoundError(f"Qwen2-Audio model directory not found: {model_path}")
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise RuntimeError(f"Qwen2-Audio artifact missing required files: {missing}")
    empty = [name for name in required if (model_path / name).stat().st_size <= 0]
    if empty:
        raise RuntimeError(f"Qwen2-Audio artifact contains empty files: {empty}")

    config = _read_json(model_path / "config.json")
    model_type = str(config.get("model_type", ""))
    architectures = [str(value) for value in config.get("architectures", [])]
    if model_type not in {"qwen2_audio", "qwen2-audio"}:
        raise RuntimeError(f"unexpected Qwen2-Audio model_type: {model_type!r}")
    if "Qwen2AudioForConditionalGeneration" not in architectures:
        raise RuntimeError(
            "config architectures do not identify Qwen2AudioForConditionalGeneration: "
            f"{architectures}"
        )

    index = _read_json(model_path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise RuntimeError("model.safetensors.index.json has no non-empty weight_map")
    referenced_shards = sorted({str(value) for value in weight_map.values()})
    actual_shards = sorted(path.name for path in model_path.glob("model-*.safetensors"))
    if referenced_shards != actual_shards:
        raise RuntimeError(
            "Qwen2-Audio shard inventory mismatch: "
            f"referenced={referenced_shards} actual={actual_shards}"
        )
    if len(actual_shards) != 5:
        raise RuntimeError(f"expected five Qwen2-Audio shards, got {actual_shards}")
    shard_inventory = []
    for name in actual_shards:
        path = model_path / name
        if path.stat().st_size <= 0:
            raise RuntimeError(f"Qwen2-Audio shard is empty: {path}")
        shard_inventory.append({"name": name, "bytes": path.stat().st_size})

    report: dict[str, Any] = {
        "status": "PASS",
        "contract": MODEL_CONTRACT,
        "path": str(model_path),
        "model_type": model_type,
        "architectures": architectures,
        "required_files": list(required),
        "config_sha256": _sha256(model_path / "config.json"),
        "generation_config_sha256": _sha256(model_path / "generation_config.json"),
        "preprocessor_config_sha256": _sha256(model_path / "preprocessor_config.json"),
        "tokenizer_config_sha256": _sha256(model_path / "tokenizer_config.json"),
        "model_index_sha256": _sha256(model_path / "model.safetensors.index.json"),
        "weight_tensors": len(weight_map),
        "shards": shard_inventory,
        "total_shard_bytes": sum(item["bytes"] for item in shard_inventory),
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if load_processor:
        import transformers
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        sampling_rate = int(processor.feature_extractor.sampling_rate)
        template = _chat_template(processor)
        if sampling_rate <= 0:
            raise RuntimeError(f"invalid Qwen2-Audio processor sampling rate: {sampling_rate}")
        if not template:
            raise RuntimeError("Qwen2-Audio processor/tokenizer has no chat template")
        tokenizer = processor.tokenizer
        report["processor"] = {
            "class": type(processor).__name__,
            "model_class_available": Qwen2AudioForConditionalGeneration.__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "feature_extractor_class": type(processor.feature_extractor).__name__,
            "sampling_rate": sampling_rate,
            "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        report["transformers_version"] = transformers.__version__
    return report


def apply_qwen_compatibility_args(args: Any) -> Any:
    """Populate legacy common-runner fields that Qwen does not consume."""

    args.htsat_checkpoint = NOT_APPLICABLE_PATH
    args.mellow_root = NOT_APPLICABLE_PATH
    return args


def load_runtime_model(args: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load the local Qwen2-Audio-Instruct artifact on one GPU in BF16."""

    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    if args.dtype != "bf16":
        raise RuntimeError(f"Qwen2-Audio evaluation is fixed to BF16, got {args.dtype!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen2-Audio evaluation requires one CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    artifact_audit = audit_model_artifact(args.checkpoint, load_processor=False)
    processor = AutoProcessor.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()
    if model.training:
        raise RuntimeError("Qwen2-Audio model remained in training mode")
    parameter_devices = {str(parameter.device) for parameter in model.parameters()}
    if parameter_devices != {str(device)}:
        raise RuntimeError(
            f"Qwen2-Audio parameters are not confined to cuda:0: {sorted(parameter_devices)}"
        )
    floating_dtypes = {
        str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()
    }
    if floating_dtypes != {str(torch.bfloat16)}:
        raise RuntimeError(
            f"Qwen2-Audio floating parameter dtype mismatch: {sorted(floating_dtypes)}"
        )
    template = _chat_template(processor)
    if not template:
        raise RuntimeError("Qwen2-Audio processor/tokenizer has no chat template")
    sampling_rate = int(processor.feature_extractor.sampling_rate)
    runtime = {
        "status": "PASS",
        "contract": MODEL_CONTRACT,
        "chat_contract": CHAT_CONTRACT,
        "artifact_audit": artifact_audit,
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(processor.tokenizer).__name__,
        "device": str(device),
        "parameter_dtype": "bfloat16",
        "processor_sampling_rate": sampling_rate,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "local_files_only": True,
        "trust_remote_code": False,
        "system_prompt": None,
    }
    return model, processor, device, runtime


def _token_rows(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(token) for token in value]


def prepare_prompt_text(
    processor: Any,
    prompt: str,
    *,
    max_prompt_tokens: int,
    truncate: bool,
) -> tuple[str, dict[str, Any]]:
    """Apply the benchmark text-token limit before adding Qwen ChatML tokens."""

    tokenizer = processor.tokenizer
    untruncated = tokenizer(
        prompt,
        truncation=False,
        padding=False,
        add_special_tokens=True,
        return_tensors=None,
    )
    original_ids = _token_rows(untruncated["input_ids"])
    if not original_ids:
        from evaluate_mmau_test_mini_5_10x2_5_mesh_mellow import RowSkip

        raise RowSkip("prompt", "empty_prompt_tokens", "Qwen tokenizer returned an empty prompt")
    if not truncate and len(original_ids) > max_prompt_tokens:
        from evaluate_mmau_test_mini_5_10x2_5_mesh_mellow import RowSkip

        raise RowSkip(
            "prompt",
            "prompt_exceeds_max_tokens",
            f"prompt has {len(original_ids)} Qwen tokens; contract limit is {max_prompt_tokens}",
            prompt_token_count=len(original_ids),
            max_prompt_tokens=max_prompt_tokens,
        )
    if truncate:
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=max_prompt_tokens,
            padding=False,
            add_special_tokens=True,
            return_tensors=None,
        )
        effective_ids = _token_rows(encoded["input_ids"])
    else:
        effective_ids = original_ids
    if not effective_ids:
        raise RuntimeError("Qwen prompt truncation produced zero tokens")
    effective_text = tokenizer.decode(
        effective_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return effective_text, {
        "benchmark_prompt": str(prompt),
        "effective_prompt": effective_text,
        "prompt_original_token_count": len(original_ids),
        "prompt_token_count": len(effective_ids),
        "prompt_truncated": len(effective_ids) < len(original_ids),
        "max_prompt_tokens": int(max_prompt_tokens),
        "prompt_tokenizer": type(tokenizer).__name__,
    }


def build_audio_analysis_conversation(prompt: str) -> list[dict[str, Any]]:
    """Build one Qwen audio-analysis turn without extra semantic instructions."""

    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "rsmol://in-memory-audio.wav"},
                {"type": "text", "text": str(prompt)},
            ],
        }
    ]


def _resample_for_processor(waveform: Any, target_rate: int) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise RuntimeError(f"Qwen input waveform must be [1,samples], got {tuple(waveform.shape)}")
    waveform = waveform.detach().cpu().float().contiguous()
    source_samples = int(waveform.shape[-1])
    if target_rate != SOURCE_AUDIO_SAMPLE_RATE:
        try:
            import torchaudio

            waveform = torchaudio.functional.resample(
                waveform, SOURCE_AUDIO_SAMPLE_RATE, target_rate
            )
            backend = "torchaudio.functional.resample"
        except Exception:
            target_samples = max(
                1, round(source_samples * float(target_rate) / SOURCE_AUDIO_SAMPLE_RATE)
            )
            waveform = F.interpolate(
                waveform.unsqueeze(0),
                size=target_samples,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
            backend = "torch_linear_interpolate_fallback"
    else:
        backend = "identity"
    if not bool(torch.isfinite(waveform).all()):
        raise RuntimeError("Qwen processor waveform contains non-finite values")
    array = waveform.squeeze(0).numpy()
    return array, {
        "qwen_audio_source_sample_rate": SOURCE_AUDIO_SAMPLE_RATE,
        "qwen_audio_source_num_samples": source_samples,
        "qwen_audio_processor_sample_rate": int(target_rate),
        "qwen_audio_processor_num_samples": int(array.shape[0]),
        "qwen_audio_resample_backend": backend,
    }


def _prepare_inputs(
    processor: Any,
    device: Any,
    prompt: str,
    waveform: Any,
    *,
    max_prompt_tokens: int,
    truncate_prompt: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    effective_prompt, prompt_audit = prepare_prompt_text(
        processor,
        prompt,
        max_prompt_tokens=max_prompt_tokens,
        truncate=truncate_prompt,
    )
    conversation = build_audio_analysis_conversation(effective_prompt)
    rendered = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    audio, audio_audit = _resample_for_processor(
        waveform,
        int(processor.feature_extractor.sampling_rate),
    )
    batch = processor(
        text=rendered,
        audios=[audio],
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }
    input_ids = inputs.get("input_ids")
    if input_ids is None or tuple(input_ids.shape[:1]) != (1,):
        raise RuntimeError(f"unexpected Qwen input_ids shape: {getattr(input_ids, 'shape', None)}")
    audit = {
        **prompt_audit,
        **audio_audit,
        "qwen_chat_contract": CHAT_CONTRACT,
        "qwen_conversation": conversation,
        "qwen_rendered_chat": rendered,
        "qwen_input_token_count": int(input_ids.shape[1]),
        "qwen_input_keys": sorted(inputs),
        "system_prompt": None,
    }
    return inputs, audit


def _eos_ids(model: Any, processor: Any) -> set[int]:
    values: list[Any] = [
        getattr(model.generation_config, "eos_token_id", None),
        getattr(processor.tokenizer, "eos_token_id", None),
    ]
    result: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            result.update(int(item) for item in value)
        else:
            result.add(int(value))
    if not result:
        raise RuntimeError("Qwen2-Audio has no EOS token ID")
    return result


def _decode(processor: Any, generated_ids: Any) -> tuple[str, str, list[int]]:
    token_ids = _token_rows(generated_ids)
    raw = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    clean = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return str(clean), str(raw), token_ids


def top_p_filter_logits(logits: Any, *, top_p: float, temperature: float) -> Any:
    """Apply the exact existing Mellow top-p boundary rule."""

    import torch

    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    filtered = logits.float() / (float(temperature) if temperature > 0 else 1.0)
    sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > float(top_p)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    for batch_index in range(sorted_indices_to_remove.shape[0]):
        indices_to_remove = sorted_indices[batch_index][sorted_indices_to_remove[batch_index]]
        filtered[batch_index, indices_to_remove] = -float("inf")
    return filtered


def generate_top_p_argmax(
    model: Any,
    processor: Any,
    device: Any,
    prompt: str,
    waveform: Any,
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
    top_p: float = 0.8,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Run BF16 full-recompute top-p filtering followed by argmax."""

    import torch

    inputs, audit = _prepare_inputs(
        processor,
        device,
        prompt,
        waveform,
        max_prompt_tokens=max_prompt_tokens,
        truncate_prompt=True,
    )
    eos_ids = _eos_ids(model, processor)
    generated: list[int] = []
    stop_reason = "max_new_tokens"
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(int(max_new_tokens)):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(**inputs, use_cache=False, return_dict=True)
            logits = output.logits[:, -1, :]
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Qwen2-Audio generation logits contain non-finite values")
            filtered = top_p_filter_logits(
                logits,
                top_p=top_p,
                temperature=temperature,
            )
            next_token = int(torch.argmax(filtered, dim=-1).item())
            generated.append(next_token)
            token = torch.tensor([[next_token]], dtype=torch.long, device=device)
            inputs["input_ids"] = torch.cat((inputs["input_ids"], token), dim=1)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat(
                    (
                        inputs["attention_mask"],
                        torch.ones((1, 1), dtype=inputs["attention_mask"].dtype, device=device),
                    ),
                    dim=1,
                )
            if next_token in eos_ids:
                stop_reason = "eos_token"
                break
    elapsed = time.perf_counter() - started
    generated_tensor = torch.tensor([generated], dtype=torch.long)
    clean, raw, token_ids = _decode(processor, generated_tensor)
    return {
        "generated_token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "generated_text": clean,
        "generated_text_raw": raw,
        "stop_reason": stop_reason,
        "eos_token_ids": sorted(eos_ids),
        "requested_max_new_tokens": int(max_new_tokens),
        "generation_seconds": elapsed,
        "tokens_per_second": len(token_ids) / max(elapsed, 1e-9),
        "decoder": "qwen2_audio_top_p_filter_then_argmax_full_recompute",
        "do_sample": False,
        "use_cache": False,
        "top_p": float(top_p),
        "temperature": float(temperature),
        "inference_dtype": "bfloat16",
        **audit,
    }


def generate_greedy(
    model: Any,
    processor: Any,
    device: Any,
    prompt: str,
    waveform: Any,
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run deterministic BF16 Qwen generation for MMAR."""

    import torch

    inputs, audit = _prepare_inputs(
        processor,
        device,
        prompt,
        waveform,
        max_prompt_tokens=max_prompt_tokens,
        truncate_prompt=False,
    )
    eos_ids = _eos_ids(model, processor)
    generated: list[int] = []
    stop_reason = "max_new_tokens"
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(int(max_new_tokens)):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(**inputs, use_cache=False, return_dict=True)
            logits = output.logits[:, -1, :]
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Qwen2-Audio generation logits contain non-finite values")
            next_token = int(torch.argmax(logits.float(), dim=-1).item())
            generated.append(next_token)
            token = torch.tensor([[next_token]], dtype=torch.long, device=device)
            inputs["input_ids"] = torch.cat((inputs["input_ids"], token), dim=1)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat(
                    (
                        inputs["attention_mask"],
                        torch.ones(
                            (1, 1),
                            dtype=inputs["attention_mask"].dtype,
                            device=device,
                        ),
                    ),
                    dim=1,
                )
            if next_token in eos_ids:
                stop_reason = "eos_token"
                break
    elapsed = time.perf_counter() - started
    generated_ids = torch.tensor([generated], dtype=torch.long)
    clean, raw, token_ids = _decode(processor, generated_ids)
    return {
        "generated_token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "generated_text": clean,
        "generated_text_raw": raw,
        "stop_reason": stop_reason,
        "eos_token_ids": sorted(eos_ids),
        "requested_max_new_tokens": int(max_new_tokens),
        "generation_seconds": elapsed,
        "tokens_per_second": len(token_ids) / max(elapsed, 1e-9),
        "decoder": "qwen2_audio_greedy_argmax_full_recompute",
        "do_sample": False,
        "use_cache": False,
        "top_p": None,
        "temperature": None,
        "inference_dtype": "bfloat16",
        **audit,
    }
