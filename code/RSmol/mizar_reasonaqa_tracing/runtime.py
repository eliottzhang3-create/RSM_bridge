"""Fresh full forwards; exact post-block text capture and single-layer replacement."""
import math
from mizar_layer_analysis_common import runtime as reference
from mizar_layer_pair_search import audio

autocast = reference.autocast
prefix = reference.prefix


def load(args, bundle, binding):
    loaded = reference.load(args, bundle, binding)
    loaded['controller'].set_drop(None)
    loaded['audit'].update(experiment='reasonaqa_silence_text_patching', all_30_blocks_executed=True)
    return loaded


def decision(record, logprobs):
    if (len(logprobs) != len(record['candidate_labels']) or len(logprobs) < 2
            or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x > 0
                   for x in logprobs)):
        raise ValueError('invalid candidate full-vocabulary log probabilities')
    predicted = max(range(len(logprobs)), key=lambda i: logprobs[i])
    target = logprobs[record['gold_index']]
    return {'candidate_logprobs': [float(x) for x in logprobs],
        'candidate_probabilities': [math.exp(x) for x in logprobs],
        'prediction_index': predicted, 'prediction_label': record['candidate_labels'][predicted],
        'correct': predicted == record['gold_index'], 'target_logprob': float(target),
        'target_probability': math.exp(target), 'tie': logprobs.count(logprobs[predicted]) > 1}


def from_logits(record, logits):
    import torch
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError('expected finite first-answer vocabulary logits')
    logprobs = logits.float().log_softmax(-1)
    ids = torch.tensor(record['candidate_token_ids'], dtype=torch.long, device=logprobs.device)
    return decision(record, logprobs.index_select(0, ids).tolist())


def validate_decision(record, value):
    if value != decision(record, value['candidate_logprobs']):
        raise ValueError('saved prediction/probability disagrees with candidate logprobs')


def forward(loaded, current_prefix, *, capture=False, layer=None, clean_states=None, patch_input=None):
    """Read raw block output, not HF's final normalized hidden_states[-1]."""
    import torch
    lm = loaded['lm']
    start = loaded.get('text_start', 127)
    expected_shape = (30, current_prefix.shape[1]-start, current_prefix.shape[2])
    if (layer is None) != (clean_states is None) or layer is not None and not 1 <= layer <= 30:
        raise ValueError('patch requires exactly one layer1..30 and its full clean tensor')
    if capture and layer is not None:
        raise ValueError('capture and replacement must be separate forwards')
    if layer is not None and tuple(clean_states.shape) != expected_shape:
        raise ValueError('incorrect raw clean states layout')
    current = current_prefix
    if patch_input is not None:
        if layer is not None or patch_input.shape != current_prefix[:, start:].shape:
            raise ValueError('invalid layer0 control')
        current = current_prefix.clone()
        current[:, start:] = patch_input.to(current.device)
    trace, collected, edits = [], [], []
    def hook(index):
        def after(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape != current.shape or not bool(torch.isfinite(hidden).all()):
                raise ValueError('post-block shape/nonfinite error')
            trace.append(index)
            if capture:
                collected.append(hidden[0, start:].detach().clone().cpu().contiguous())
            if layer == index:
                saved = clean_states[index-1]
                if saved.dtype != hidden.dtype:
                    raise ValueError('patch dtype differs; never silently downcast cached hidden states')
                replacement = hidden.clone()
                replacement[:, start:] = saved.to(hidden.device).unsqueeze(0)
                if not torch.equal(replacement[:, :start], hidden[:, :start]):
                    raise ValueError('patch touched audio/SEP')
                edits.append(index)
                return (replacement, *output[1:]) if isinstance(output, tuple) else replacement
        return after
    handles = [block.register_forward_hook(hook(i)) for i, block in enumerate(lm.model.layers, 1)]
    try:
        with torch.inference_mode(), autocast(loaded['device']):
            out = lm(inputs_embeds=current, attention_mask=None, labels=None, use_cache=False)
            logits = out.logits[0, -1].detach().float().cpu().clone()
        if trace != list(range(1, 31)) or edits != ([] if layer is None else [layer]) or getattr(out, 'past_key_values', None) is not None:
            raise ValueError('expected full thirty-layer fresh no-cache forward')
        return logits, torch.stack(collected) if capture else None
    finally:
        for handle in handles:
            handle.remove()


def silent_audio(loaded, record):
    import torch
    # The zero waveform goes through the unchanged frontend/encoder/mapper.
    full = prefix(loaded, record, torch.zeros(640000, dtype=torch.float32))
    return full[:, :126].detach().clone()


def corrupt_prefix(clean_prefix, silence):
    import torch
    if silence.shape != (1, 126, clean_prefix.shape[2]) or silence.dtype != clean_prefix.dtype:
        raise ValueError('invalid encoded silence layout/dtype')
    value = clean_prefix.clone()
    value[:, :126] = silence.to(value.device)
    if not torch.equal(value[:, 126:], clean_prefix[:, 126:]):
        raise ValueError('corruption changed SEP/text')
    return value


def controls(loaded, clean_prefix, corrupted_prefix, clean_states, record, clean_result, corrupt_result):
    import torch
    clean_logits, captured = forward(loaded, clean_prefix, capture=True)
    native, _ = forward(loaded, clean_prefix)
    corrupt_logits, _ = forward(loaded, corrupted_prefix)
    layer0, _ = forward(loaded, corrupted_prefix, patch_input=clean_prefix[:, loaded.get('text_start', 127):])
    last, _ = forward(loaded, corrupted_prefix, layer=30, clean_states=clean_states)
    for actual, expected in ((native, clean_logits), (layer0, corrupt_logits), (last, clean_logits)):
        if not torch.equal(actual, expected):
            raise ValueError('no-patch/layer0/layer30 control failed (full vocabulary logits)')
    if not torch.equal(captured, clean_states):
        raise ValueError('cached clean states differ from repeated clean capture')
    if from_logits(record, clean_logits) != clean_result or from_logits(record, corrupt_logits) != corrupt_result:
        raise ValueError('screening probabilities differ after cache reload/rank reassignment')
    return {'status': 'PASS', 'capture_vs_native_logits_bitwise_equal': True,
        'layer0_equals_corrupted_logits_bitwise': True, 'layer30_equals_clean_logits_bitwise': True,
        'cache_equals_repeated_clean_states_bitwise': True, 'state_shape': list(clean_states.shape),
        'state_dtype': str(clean_states.dtype), 'text_start': loaded.get('text_start', 127)}
