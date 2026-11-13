"""Read-only calls into existing CED training loaders and greedy decoders."""
from __future__ import annotations

import copy
import hashlib
import sys
import time
from pathlib import Path

from . import config, dataset, scoring, storage

SCRIPTS = config.ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import mizar_s1_ced_eval_runtime as base
import mizar_s1_ced_recursive_mesh_eval_runtime as mesh
import mizar_s1_ced_recursive_eval_runtime as recursive
import mizar_s1_ced_multilayer_eval_runtime as multilayer

PROTOCOL = copy.deepcopy(base.PROTOCOL)
PROTOCOL.update(version='reasonaqa_close_ended_ced_fixed20_full_answer_exact_v1',
                seed=1234,
                prompt_policy='original_reasonaqa_input_lowercase_then_training_tokenize_no_added_instruction',
                audio_policy='training_fixed20_single_or_independent_pair_no_random_audio_fallback',
                prediction_preprocessing='raw_preserved_separate_scoring_copy_only',
                scoring=scoring.POLICY, selection=dataset.SELECTION_POLICY)


def adapter(variant):
    if variant not in config.VARIANTS:
        raise ValueError(f'unsupported ReasonAQA CED variant: {variant}')
    if variant == 'base':
        return base, 'base'
    if variant == 'recursive_mesh':
        return mesh, 'mesh'
    if variant == 'recursive_5_10_5':
        return recursive, 'recursive'
    return multilayer, 'multilayer'


def load_runtime(args):
    return adapter(args.eval_variant)[0].load_runtime(args)


def prefix_and_decode(selected, family, variant, loaded, inputs):
    model, tokenizer, _, _, _, audit, _ = loaded
    if family == 'multilayer':
        prefix, audio, context = selected.generate_prefix_inference(
            model, inputs, eval_variant=variant, audit=audit)
        context = selected.sample_context(variant, audio, context, inputs['input'])
        result = selected.greedy_decode(model, tokenizer, prefix, eval_variant=variant, context=context)
    else:
        prefix, _, context = model.generate_prefix_inference(inputs)
        if family == 'recursive':
            result = selected.greedy_decode(model, tokenizer, prefix, eval_variant=variant,
                                            audio_information=context)
        else:
            result = selected.greedy_decode(model, tokenizer, prefix)
    return prefix, result


def waveform_sha(waveform):
    return hashlib.sha256(waveform.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def fixed20_pair(fixed20, record):
    """Execute the same mono/resample/fixed_len functions used in training."""
    sources = [record['audio1']] + ([record['audio2']] if record['audio2'] else [])
    waves, audits = [], []
    target = 320000 if record['audio2'] else 640000
    for source in sources:
        path = Path(source['path'])
        stat = path.stat()
        if stat.st_size != source['bytes'] or stat.st_mtime_ns != source['mtime_ns']:
            raise ValueError(f'audio changed since metadata resolution: {path}')
        waveform = fixed20.load_mono(str(path))
        length = waveform.numel()
        if length == 0:
            raise ValueError(f'empty decoded audio: {path}')
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (source['bytes'], source['mtime_ns']):
            raise ValueError(f'audio changed while decoding: {path}')
        audits.append({**source, 'resampled_samples': length, 'resampled_waveform_sha256': waveform_sha(waveform),
                       'target_samples': target, 'center_crop_start': max(0, (length - target) // 2),
                       'cropped_samples': max(0, length - target), 'right_pad_samples': max(0, target - length)})
        waves.append(fixed20.fixed_len(waveform, target))
    if record['audio2']:
        first, second = waves
    else:
        first, second = waves[0][:320000], waves[0][320000:]
    if first.numel() != 320000 or second.numel() != 320000:
        raise ValueError('fixed20 requires two 320000-sample halves')
    return first.contiguous(), second.contiguous(), audits


def generate(loaded, record, *, variant):
    import torch

    selected, family = adapter(variant)
    _, tokenizer, device, text_encoder, fixed20, _, _ = loaded
    prompt = record['original_record']['input'].lower()
    tokens = text_encoder.tokenize(prompt, 256)
    prompt_count = len(tokenizer.encode(prompt + ' <|endoftext|>', add_special_tokens=True))
    first, second, audio_audit = fixed20_pair(fixed20, record)
    token_ids = tokens['input_ids'][0].tolist()
    input_identity = {'prompt_token_ids': token_ids,
                      'audio_half_sha256': [waveform_sha(first), waveform_sha(second)]}
    started = time.monotonic()
    with torch.inference_mode():
        inputs = {'audio1': first.unsqueeze(0).to(device), 'audio2': second.unsqueeze(0).to(device),
                  'input': {key: value.to(device) for key, value in tokens.items()}}
        prefix, result = prefix_and_decode(selected, family, variant, loaded, inputs)
    result.update(prefix_shape=list(prefix.shape), prompt_used=prompt,
                  prompt_original_token_count=prompt_count, prompt_truncated=prompt_count > 256,
                  prompt_tokens=256, answer_start_position=383, first_answer_logit_position=382,
                  generation_seconds=time.monotonic() - started, audio_audit=audio_audit,
                  audio_slot_policy='two_actual_files' if record['audio2'] else 'single_fixed20_split10',
                  input_fingerprint_sha256=storage.object_sha(input_identity), **input_identity)
    return result
