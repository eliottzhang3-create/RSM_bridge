"""Compact lowercase MMSU inputs with the existing strict CED runtimes."""
from __future__ import annotations

import copy
import importlib
import io
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import mizar_s1_ced_eval_runtime as base  # noqa: E402
from mizar_s1_ced_mmsu_eval import dataset

from . import runtime as selected_runtime
VARIANTS = tuple(selected_runtime.ROUTES)
PROTOCOL = copy.deepcopy(base.PROTOCOL)
PROTOCOL.update(version=dataset.PROMPT_VERSION,
                prompt_policy=dataset.PROMPT_POLICY,
                scoring="mmsu_answer_text_contains_v2_then_pinned_external_official_scorer")


def protocol_for(variant):
    return selected_runtime.protocol_for(variant,PROTOCOL)


def adapter(variant):
    return selected_runtime, 'multilayer', selected_runtime.sources.route_root(variant)



def load_runtime(args):
    selected, _, _ = adapter(args.eval_variant)
    return selected.load_runtime(args)


def read_audio(payload, base_dir):
    """Decode the real codec header; embedded bytes win over a stale HF path."""
    import soundfile as sf

    if not isinstance(payload, dict):
        raise ValueError("MMSU audio must be an HF bytes/path struct")
    embedded = payload.get("bytes")
    if embedded is not None:
        if not isinstance(embedded, (bytes, bytearray, memoryview)) or not embedded:
            raise ValueError("invalid/empty embedded MMSU audio bytes")
        source = io.BytesIO(bytes(embedded))
        source_label = "parquet_embedded_bytes"
    else:
        raw_path = payload.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("MMSU audio has neither embedded bytes nor a path")
        path = Path(raw_path)
        path = path if path.is_absolute() else Path(base_dir) / path
        source = str(path.resolve(strict=True))
        source_label = f"path_fallback:{source}"
    samples, rate = sf.read(source, dtype="float32", always_2d=True)
    if rate <= 0 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("empty/invalid decoded MMSU audio")
    return samples, int(rate), source_label


def decode_audio(payload, base_dir):
    """SoundFile FP32 decoding, then training's mono/resample ordering."""
    import numpy as np
    import torch
    import torchaudio

    samples, rate, source_label = read_audio(payload, base_dir)
    waveform = torch.from_numpy(np.ascontiguousarray(samples.T))
    waveform = waveform.mean(dim=0, keepdim=True)
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("nonfinite MMSU audio")
    if rate != 32000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=rate, new_freq=32000)
    waveform = waveform[0].contiguous()
    if waveform.numel() == 0 or not bool(torch.isfinite(waveform).all()):
        raise ValueError("invalid MMSU resampled waveform")
    length = waveform.numel()
    return waveform, {
        "audio_source": source_label, "audio_original_sample_rate": int(rate),
        "audio_original_channels": int(samples.shape[1]),
        "audio_original_samples": int(samples.shape[0]),
        "audio_original_duration_seconds": samples.shape[0] / rate,
        "audio_target_sample_rate": 32000, "audio_full_waveform_samples": length,
        "audio_center_crop_start": max(0, (length - 640000) // 2),
        "audio_cropped_samples": max(0, length - 640000),
        "audio_right_pad_samples": max(0, 640000 - length),
        "audio_crop_policy": "center_crop_before_split_or_right_pad_to20s",
    }


def prefix_and_decode(selected, family, variant, loaded, inputs):
    """Keep the real route's per-sample context on all decoding forwards."""
    model, tokenizer, _, _, _, audit, _ = loaded
    if family == "multilayer":
        prefix, audio, context = selected.generate_prefix_inference(
            model, inputs, eval_variant=variant, audit=audit)
        context = selected.sample_context(variant, audio, context, inputs["input"])
        result = selected.greedy_decode(model, tokenizer, prefix, eval_variant=variant, context=context)
    else:
        prefix, _, context = model.generate_prefix_inference(inputs)
        if family == "recursive":
            result = selected.greedy_decode(model, tokenizer, prefix, eval_variant=variant,
                                             audio_information=context)
        else:
            result = selected.greedy_decode(model, tokenizer, prefix)
    return prefix, result


def generate(loaded, sample, *, variant):
    import torch

    selected, family, _ = adapter(variant)
    _, tokenizer, device, text_encoder, fixed20, _, _ = loaded
    # Production dataset.prompt supplies the compact lowercase input. Keep the
    # supplied text unchanged here so diagnostic prompt ablations remain valid.
    text = sample["prompt"]
    tokens = text_encoder.tokenize(text, 256)
    count = len(tokenizer.encode(text + " <|endoftext|>", add_special_tokens=True))
    whole = fixed20.fixed_len(sample["waveform"], 640000)
    first, second = whole[:320000].contiguous(), whole[320000:].contiguous()
    if first.numel() != 320000 or second.numel() != 320000:
        raise ValueError("MMSU fixed20 must produce two 320000-sample halves")
    started = time.monotonic()
    with torch.inference_mode():
        inputs = {"audio1": first.unsqueeze(0).to(device), "audio2": second.unsqueeze(0).to(device),
                  "input": {key: value.to(device) for key, value in tokens.items()}}
        prefix, result = prefix_and_decode(selected, family, variant, loaded, inputs)
    from mizar_s1_ced_reasonaqa5000_eval.storage import object_sha
    from .reasonaqa_runtime import waveform_sha
    actual_input = dict(prompt_token_ids=tokens['input_ids'][0].tolist(),
        prompt_attention_mask=tokens['attention_mask'][0].tolist(),
        audio_half_sha256=[waveform_sha(first),waveform_sha(second)])
    result.update(**actual_input,input_fingerprint_sha256=object_sha(actual_input),prompt_used=text)
    result.update(prefix_shape=list(prefix.shape), prompt_original_token_count=count,
                  prompt_truncated=count > 256, prompt_tokens=256,
                  answer_start_position=383, first_answer_logit_position=382,
                  generation_seconds=time.monotonic() - started, **sample["audio_audit"])
    return result
