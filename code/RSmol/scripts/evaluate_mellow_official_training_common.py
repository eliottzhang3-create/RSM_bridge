#!/usr/bin/env python3
"""Shared runtime adapter for official Mellow training checkpoints.

This module intentionally loads only the official training-route Mellow model
and the model state_dict stored inside its schema-v2 training checkpoint.  It
shares dataset traversal and scoring with the established MMAU/MMAR evaluators.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

CHECKPOINT_ADAMW_COSINE = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "mellow_official_reasonaqa_adamw_cosine_5090/formal_30epochs_20261001_003838/"
    "checkpoints/mellow_adamw_cosine_reasonaqa_formal_20_20260930_163845508069938/"
    "model--epo-30.ckpt"
)
CHECKPOINT_C8204D8 = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "mellow_official_c8204d8/formal_30epochs_20260930_132438/"
    "checkpoints/mellow_official_reasonaqa_formal_20_20260930_052445889901842/"
    "model--epo-30.ckpt"
)
RUNTIME_ADAMW_COSINE = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "mellow_official_reasonaqa_adamw_cosine_5090/formal_30epochs_20261001_003838/"
    "runtime_30epochs.yaml"
)
RUNTIME_C8204D8 = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "mellow_official_c8204d8/formal_30epochs_20260930_132438/"
    "runtime_30epochs.yaml"
)
ROUTE_ROOT_ADAMW_COSINE = ROOT / "mellow_official_training_c8204d8_adamw_cosine"
ROUTE_ROOT_C8204D8 = ROOT / "mellow_official_training_c8204d8"
OUTPUT_ROOT_ADAMW_COSINE = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "mellow_official_reasonaqa_adamw_cosine_5090/formal_30epochs_20261001_003838/eval"
)
OUTPUT_ROOT_C8204D8 = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "mellow_official_c8204d8/formal_30epochs_20260930_132438/eval"
)

ROUTE_SPECS: dict[str, dict[str, Any]] = {
    "c8204d8": {
        "route": "c8204d8",
        "training_branch": "mellow_official_training_c8204d8",
        "route_root": ROUTE_ROOT_C8204D8,
        "checkpoint": CHECKPOINT_C8204D8,
        "runtime_config": RUNTIME_C8204D8,
        "output_root": OUTPUT_ROOT_C8204D8,
    },
    "adamw_cosine": {
        "route": "adamw_cosine",
        "training_branch": "mellow_official_training_c8204d8_adamw_cosine",
        "route_root": ROUTE_ROOT_ADAMW_COSINE,
        "checkpoint": CHECKPOINT_ADAMW_COSINE,
        "runtime_config": RUNTIME_ADAMW_COSINE,
        "output_root": OUTPUT_ROOT_ADAMW_COSINE,
    },
}

PROMPT_TOKENS = 129
AUDIO_TOKENS_PER_SLOT = 129
AUDIO_PREFIX_TOKENS = 260
TOTAL_PREFIX_TOKENS = 389
HIDDEN_SIZE = 576
SAMPLE_RATE = 32000
AUDIO_SECONDS = 10
PAD_TOKEN = "!"
EOS_TOKEN = "<|endoftext|>"


def route_spec(route: str) -> dict[str, Any]:
    try:
        return dict(ROUTE_SPECS[route])
    except KeyError as exc:
        raise ValueError(f"unsupported official Mellow training route: {route!r}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise FileNotFoundError(f"official Mellow runtime config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, Mapping):
        raise RuntimeError(f"runtime config is not a mapping: {path}")
    return dict(config)


def _import_route_model(route_root: Path):
    route_root = route_root.resolve()
    if not route_root.is_dir():
        raise FileNotFoundError(f"official Mellow source root not found: {route_root}")
    text = str(route_root)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)
    importlib.invalidate_caches()
    model_module = importlib.import_module("models.model")
    return model_module.get_model_class("Mellow")


def _config_model_args(config: Mapping[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model") or {}
    encoder_cfg = model_cfg.get("encoder") or {}
    decoder_cfg = model_cfg.get("decoder") or {}
    required = {
        "audioenc_name": encoder_cfg.get("audioenc_name"),
        "d_in": encoder_cfg.get("out_emb"),
        "text_decoder": decoder_cfg.get("text_decoder"),
        "prefix_length": decoder_cfg.get("prefix_length"),
        "freeze_text_decoder_weights": decoder_cfg.get("freeze_gpt_weights"),
        "d_out": encoder_cfg.get("d_proj"),
        "use_pretrained_audioencoder": encoder_cfg.get("use_pretrained_audioencoder"),
        "freeze_audio_encoder_weights": encoder_cfg.get("freeze_audio_encoder_weights"),
        "pretrained_audioencoder_path": encoder_cfg.get("pretrained_audioencoder_path"),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise RuntimeError(f"runtime config lacks official Mellow model fields: {missing}")
    return required


def _make_tokenizer(config: Mapping[str, Any]):
    from transformers import AutoTokenizer

    tokenizer_name = str((config.get("data") or {}).get("tokenizer_type", ""))
    if not tokenizer_name:
        raise RuntimeError("runtime config has no data.tokenizer_type")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise RuntimeError("official Mellow tokenizer has no pad_token_id")
    if tokenizer.eos_token_id is None:
        raise RuntimeError("official Mellow tokenizer has no eos_token_id")
    return tokenizer


def load_runtime_model(args: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("official Mellow evaluation requires one CUDA GPU")
    route_root = Path(args.route_root).resolve()
    runtime_config_path = Path(args.runtime_config).resolve()
    checkpoint_path = Path(args.training_checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"official training checkpoint not found: {checkpoint_path}")
    config = _load_yaml(runtime_config_path)
    model_args = _config_model_args(config)
    route_model = _import_route_model(route_root)
    model = route_model(**model_args)
    tokenizer = _make_tokenizer(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("state_dict"), Mapping):
        raise RuntimeError(
            "expected a schema-v2 official training checkpoint with checkpoint['state_dict']; "
            "standalone Mellow or SmolLM2 checkpoints are not accepted"
        )
    state_dict = checkpoint["state_dict"]
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"strict official Mellow state load failed: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model.to(device)
    model.eval()
    modes = {"model": bool(model.training)}
    modes.update({name: bool(module.training) for name, module in model.named_modules() if name})
    if any(modes.values()):
        raise RuntimeError(f"official Mellow inference requires eval mode for all modules: {modes}")
    parameter_devices = {str(parameter.device) for parameter in model.parameters()}
    if parameter_devices != {str(device)}:
        raise RuntimeError(f"official Mellow parameters are not all on cuda:0: {parameter_devices}")
    buffer_devices = {str(buffer.device) for buffer in model.buffers()}
    if buffer_devices and buffer_devices != {str(device)}:
        raise RuntimeError(f"official Mellow buffers are not all on cuda:0: {buffer_devices}")
    model_cfg = config.get("model") or {}
    decoder_cfg = model_cfg.get("decoder") or {}
    if int(decoder_cfg.get("total_prefix_length", TOTAL_PREFIX_TOKENS)) != TOTAL_PREFIX_TOKENS:
        raise RuntimeError("official Mellow runtime config does not declare total_prefix_length=389")
    if int((config.get("data") or {}).get("ip_text_len", PROMPT_TOKENS)) != PROMPT_TOKENS:
        raise RuntimeError("official Mellow runtime config does not declare ip_text_len=129")
    runtime = {
        "model_family": "official_mellow_training",
        "training_branch": str(args.training_branch),
        "route_root": str(route_root),
        "checkpoint_file": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "runtime_config": str(runtime_config_path),
        "runtime_config_sha256": sha256_file(runtime_config_path),
        "checkpoint_schema_version": checkpoint.get("schema_version"),
        "checkpoint_epoch_completed": checkpoint.get("epoch_completed"),
        "checkpoint_num_epochs": checkpoint.get("num_epochs"),
        "standalone_smollm2_checkpoint": False,
        "legacy_mellow_v0_checkpoint": False,
        "model_loader": "official_training_native_mellow_state_dict",
        "model_args": _jsonable(model_args),
        "runtime_eval_modes": modes,
        "runtime_parameter_devices": sorted(parameter_devices),
        "runtime_buffer_devices": sorted(buffer_devices),
        "text_decoder_role": "internal_component_of_official_mellow",
    }
    return model, tokenizer, device, runtime


def tokenize_training_prompt(tokenizer: Any, prompt: str, max_prompt_tokens: int, device: Any):
    import torch

    if max_prompt_tokens != PROMPT_TOKENS:
        raise ValueError(f"official Mellow prompt length is fixed at {PROMPT_TOKENS}")
    training_prompt = str(prompt) + " " + EOS_TOKEN
    unpadded = tokenizer(
        training_prompt,
        truncation=False,
        padding=False,
        add_special_tokens=True,
        return_tensors="pt",
    )
    encoded = tokenizer(
        training_prompt,
        truncation=True,
        padding="max_length",
        max_length=max_prompt_tokens,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    if tuple(input_ids.shape) != (1, PROMPT_TOKENS):
        raise RuntimeError(f"official Mellow prompt shape mismatch: {tuple(input_ids.shape)}")
    if tokenizer.pad_token_id is None:
        raise RuntimeError("official Mellow tokenizer has no pad_token_id")
    pad_id = int(tokenizer.pad_token_id)
    bang_id = int(tokenizer.convert_tokens_to_ids(PAD_TOKEN))
    if pad_id != bang_id:
        raise RuntimeError(f"official Mellow tokenizer pad token is not literal !: pad_id={pad_id}, bang_id={bang_id}")
    padding_positions = attention_mask[0].eq(0)
    if bool(padding_positions.any()) and not bool(input_ids[0][padding_positions].eq(pad_id).all()):
        raise RuntimeError("official Mellow prompt is not right-padded with tokenizer pad token")
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }, {
        "prompt_training_text_appended_eos": True,
        "prompt_eos_token": EOS_TOKEN,
        "prompt_padding_token": PAD_TOKEN,
        "prompt_padding_token_id": pad_id,
        "prompt_slot_tokens": PROMPT_TOKENS,
        "prompt_padding_side": "right",
        "prompt_original_token_count": int(unpadded["input_ids"].shape[-1]),
        "prompt_truncated": bool(unpadded["input_ids"].shape[-1] > PROMPT_TOKENS),
    }


def _same_waveform_segment(common: Any, waveform: Any):
    segment, audit = common.mellow_author_reply_audio_segment(waveform)
    return segment, dict(audit)


def build_native_prefix(model: Any, common: Any, waveform: Any, text_input: Mapping[str, Any], device: Any, segmenter: Any | None = None):
    import torch

    if segmenter is None:
        segment, segment_audit = _same_waveform_segment(common, waveform)
    else:
        segment, segment_audit = segmenter(waveform)
    audio1 = segment.to(device, non_blocking=True)
    audio2 = audio1.clone()
    prefix, _, _ = model.generate_prefix_inference({
        "audio1": audio1,
        "audio2": audio2,
        "input": dict(text_input),
    })
    if tuple(prefix.shape) != (1, TOTAL_PREFIX_TOKENS, HIDDEN_SIZE):
        raise RuntimeError(f"official Mellow prefix shape mismatch: {tuple(prefix.shape)}")
    if not bool(torch.isfinite(prefix).all()):
        raise RuntimeError("official Mellow prefix contains non-finite values")
    first = prefix[:, :AUDIO_TOKENS_PER_SLOT, :]
    second_start = AUDIO_TOKENS_PER_SLOT + 1
    second = prefix[:, second_start:second_start + AUDIO_TOKENS_PER_SLOT, :]
    return prefix, {
        "audio1_prefix_shape": list(first.shape),
        "audio2_prefix_shape": list(second.shape),
        "combined_prefix_shape": list(prefix.shape),
        "prefix_token_count": TOTAL_PREFIX_TOKENS,
        "prefix_layout": "native_audio1_129 + separator + native_audio2_129 + separator + eos_pad_prompt_129",
        "single_audio_slot": False,
        "audio2_same_source": True,
        "audio2_reused": False,
        "audio1_audio2_same_segment": True,
        "audio1_encoded_separately": True,
        "audio2_encoded_separately": True,
        "native_audio_encoder_invocations": 2,
        "compact_single_audio_prefix_used": False,
        "audio1_segment": segment_audit,
        "audio2_segment": {**segment_audit, "same_segment_as_audio1": True},
    }


def greedy_decode_native(model: Any, tokenizer: Any, prefix: Any, *, max_new_tokens: int, top_p: float = 0.8, temperature: float = 1.0):
    import time
    import torch

    eos_id = int(tokenizer.eos_token_id)
    generated_ids: list[int] = []
    generated = prefix
    started = time.perf_counter()
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        output = model.caption_decoder.lm(inputs_embeds=generated)
        logits = output.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        for batch_index in range(remove.shape[0]):
            logits[batch_index, sorted_indices[batch_index][remove[batch_index]]] = -float("inf")
        next_token = int(torch.argmax(logits, dim=-1).item())
        generated_ids.append(next_token)
        token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=generated.device)
        token_embed = model.caption_decoder.lm.model.embed_tokens(token_tensor)
        generated = torch.cat((generated, token_embed), dim=1)
        if next_token == eos_id:
            stop_reason = "eos_token"
            break
    elapsed = time.perf_counter() - started
    raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
    text = raw.split(tokenizer.eos_token or EOS_TOKEN)[0]
    return {
        "generated_token_ids": generated_ids,
        "generated_text_raw": raw,
        "generated_text": text,
        "stop_reason": stop_reason,
        "eos_token_ids": [eos_id],
        "requested_max_new_tokens": int(max_new_tokens),
        "generated_token_count": len(generated_ids),
        "generation_seconds": elapsed,
        "generation_decoder": "official_mellow_top_p_filter_then_argmax_full_recompute",
        "generation_top_p": float(top_p),
        "generation_temperature": float(temperature),
        "generation_do_sample": False,
        "generation_use_cache": False,
        "inference_dtype": "float32",
    }


def run_generation(model: Any, tokenizer: Any, device: Any, sample: Mapping[str, Any], *, max_prompt_tokens: int, max_new_tokens: int, common: Any, segmenter: Any | None = None) -> dict[str, Any]:
    import torch

    text_input, prompt_audit = tokenize_training_prompt(
        tokenizer, str(sample["prompt"]), max_prompt_tokens, device
    )
    with torch.inference_mode():
        prefix, prefix_audit = build_native_prefix(
            model, common, sample["waveform"], text_input, device, segmenter=segmenter
        )
        generated = greedy_decode_native(
            model, tokenizer, prefix, max_new_tokens=max_new_tokens
        )
    generated.update(prompt_audit)
    generated.update(prefix_audit)
    generated["answer_start_position"] = TOTAL_PREFIX_TOKENS
    return generated


def attach_model_identity(report: dict[str, Any], args: Any, runtime: Mapping[str, Any]) -> dict[str, Any]:
    report["model_identity"] = {
        "model_family": "official_mellow_training",
        "training_branch": str(args.training_branch),
        "checkpoint_file": str(args.training_checkpoint),
        "checkpoint_sha256": runtime.get("checkpoint_sha256"),
        "runtime_config": str(args.runtime_config),
        "runtime_config_sha256": runtime.get("runtime_config_sha256"),
        "model_loader": "official_training_native_mellow_state_dict",
        "legacy_mellow_v0_checkpoint": False,
        "standalone_smollm2_checkpoint": False,
        "text_decoder_role": "internal_component_of_official_mellow",
    }
    report["protocol"] = dict(report.get("protocol") or {})
    report["protocol"].update({
        "prompt_tokens": PROMPT_TOKENS,
        "prompt_eos_appended": True,
        "prompt_padding_token": PAD_TOKEN,
        "prompt_padding_side": "right",
        "audio_prefix_tokens": AUDIO_PREFIX_TOKENS,
        "total_prefix_tokens": TOTAL_PREFIX_TOKENS,
        "audio2_policy": "same_waveform_native_audio1_audio2_dual_encoding",
        "inference_dtype": "float32",
        "top_p": 0.8,
        "temperature": 1.0,
        "do_sample": False,
        "use_cache": False,
    })
    return report


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
