"""Preserve official MMSU prompts while reusing existing strict CED runtimes."""
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
import mizar_s1_ced_recursive_mesh_eval_runtime as mesh  # noqa: E402
import mizar_s1_ced_recursive_eval_runtime as recursive  # noqa: E402
import mizar_s1_ced_multilayer_eval_runtime as multilayer  # noqa: E402

VARIANTS = ("base", "recursive_mesh", *recursive.ROUTE_SPECS, *multilayer.ROUTE_SPECS,
            "official_mizar")
PROTOCOL = copy.deepcopy(base.PROTOCOL)
PROTOCOL.update(version="mmsu_official_prompt_ced_fixed20_raw_greedy_v1",
                prompt_policy="MMSU_official_instruction_ABCD_original_case_and_null_positions",
                scoring="pinned_external_MMSU_mmsu_evaluation.py_positional_jsonl")


def adapter(variant):
    if variant == "base":
        return base, "base", base.ROUTE_ROOT
    if variant == "recursive_mesh":
        return mesh, "mesh", mesh.ROUTE_ROOT
    if variant in recursive.ROUTE_SPECS:
        return recursive, "recursive", recursive.variant_spec(variant)["route_root"]
    if variant in multilayer.ROUTE_SPECS:
        return multilayer, "multilayer", multilayer.variant_spec(variant)["route_root"]
    if variant == "official_mizar":
        official = importlib.import_module("mizar_official_eval.runtime")
        return official, "official", official.ROUTE_ROOT
    raise ValueError(f"unsupported existing CED runtime: {variant}")


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
    # The existing MMAU/MMAR runners lowercase here. MMSU uses official case.
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
    result.update(prefix_shape=list(prefix.shape), prompt_original_token_count=count,
                  prompt_truncated=count > 256, prompt_tokens=256,
                  answer_start_position=383, first_answer_logit_position=382,
                  generation_seconds=time.monotonic() - started, **sample["audio_audit"])
    return result
