"""Strict final-release loading with isolated BF16 inputs and mask routing."""
import copy
import os
import random
from contextlib import nullcontext

from . import audio, config, storage


def autocast(device):
    import torch
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


class MaskedLM:
    """Route retained original modules directly; no parameter writes or zeroing."""
    def __init__(self, lm, hidden_size=576, prefix_tokens=383):
        if len(lm.model.layers) != 30:
            raise ValueError('expected thirty original layers')
        self.lm, self.original = lm, tuple(lm.model.layers)
        self.pair, self.trace, self.forwards = (), [], 0
        self.hidden_size, self.prefix_tokens = hidden_size, prefix_tokens
        lm.register_forward_pre_hook(self._begin, with_kwargs=True)
        for index, layer in enumerate(self.original):
            layer.register_forward_pre_hook(self._observe(index))
        lm.register_forward_hook(self._finish)

    def _observe(self, index):
        def hook(_module, _args):
            self.trace.append(index)
        return hook

    def _begin(self, _module, _args, kwargs):
        self.trace.clear()
        x = kwargs.get('inputs_embeds')
        if (kwargs.get('use_cache') is not False or kwargs.get('attention_mask') is not None
                or kwargs.get('past_key_values') is not None or x is None
                or x.ndim != 3 or x.shape[0] != 1 or x.shape[2] != self.hidden_size
                or not self.prefix_tokens <= x.shape[1] <= 1024):
            raise ValueError('masked forward violates fixed-prefix/no-cache protocol')

    def _finish(self, _module, _args, output):
        expected = [i for i in range(30) if i not in self.pair]
        if self.trace != expected or getattr(output, 'past_key_values', None) is not None:
            raise ValueError(f'mask execution/cache mismatch: expected={expected}, actual={self.trace}')
        self.forwards += 1

    def set_pair(self, pair):
        import torch
        config.drop_mask(pair)
        self.pair = tuple(pair)
        self.lm.model.layers = torch.nn.ModuleList([m for i, m in enumerate(self.original) if i not in self.pair])
        self.lm.config.num_hidden_layers = len(self.lm.model.layers)

    def report(self):
        return {'drop_pair_0based': list(self.pair), 'drop_mask': config.drop_mask(self.pair),
                'retained_original_layers': [i for i in range(30) if i not in self.pair],
                'last_trace': list(self.trace), 'checked_forwards': self.forwards}


def load(args, bundle_audit, binding):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from mizar_official_eval import runtime as official
    from .manifest import load_tokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('worker must see exactly one assigned CUDA GPU')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('assigned GPU does not support BF16; no implicit precision fallback')
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    official.verify_bundle_unchanged(bundle_audit, args.model_dir, args.checkpoint)
    modules = official.base.import_training_route(config.ROUTE)
    model = modules['models.mellow'].Mellow(
        audioenc_name='CEDSmall', d_in=384, d_out=576,
        text_decoder=str(args.model_dir / 'assets/smollm2-135m'), prefix_length=383,
        freeze_text_decoder_weights=False, use_pretrained_audioencoder=True,
        freeze_audio_encoder_weights=True, pretrained_audioencoder_path=str(args.model_dir / 'assets/ced-small'))
    model.float()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state_audit = official.validate_state_dict(state, torch)
    model.load_state_dict(state, strict=True)
    del state
    structure = official.audit_model(model)
    model.to(device).eval().requires_grad_(False)
    tokenizer, _ = load_tokenizer(args.model_dir)
    if any(m.training for m in model.modules()) or {p.dtype for p in model.parameters()} != {torch.float32}:
        raise ValueError('source weights must remain FP32 and all inference modules eval')
    controller = MaskedLM(model.caption_decoder.lm)
    audit = {'status': 'PASS', **state_audit, **structure, 'binding': binding,
             'strict_state_dict': True,
             'checkpoint_sha256': bundle_audit['verified_files'][0]['sha256'],
             'weights_dtype': 'float32', 'forward_dtype': 'bfloat16_autocast',
             'audio_frontend_dtype': 'float32', 'prefix_dtype': 'bfloat16', 'tf32': False,
             'all_modules_eval': True, 'bf16_supported': True, 'torch': torch.__version__,
             'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device),
             'attention_implementation': model.caption_decoder.lm.config._attn_implementation,
             'sdpa_backends': {'flash': False, 'mem_efficient': False, 'math': True}}
    return {'model': model, 'lm': model.caption_decoder.lm, 'tokenizer': tokenizer,
            'device': device, 'controller': controller, 'audit': audit}


def prefix(runtime, record, waveform):
    """Same CED and trained mapper, with the audio frontend outside autocast."""
    import torch
    model, device = runtime['model'], runtime['device']
    encoder = model.audio_encoder.base
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=False):
            mel = encoder._waveform_to_mel(waveform.unsqueeze(0).to(device=device, dtype=torch.float32))
        with autocast(device):
            hidden = encoder._forward_hidden(mel)
            grid = encoder._hidden_to_temporal_grid(hidden)
            mapped = model.audio_encoder.projection(encoder.mapper(grid))
        embed = runtime['lm'].get_input_embeddings()
        ids = torch.tensor([record['prompt_token_ids']], device=device, dtype=torch.long)
        sep = torch.tensor([[0]], device=device, dtype=torch.long)
        result = torch.cat([mapped.to(torch.bfloat16), embed(sep).to(torch.bfloat16),
                            embed(ids).to(torch.bfloat16)], dim=1).contiguous()
    if result.shape != (1, 383, 576) or not bool(torch.isfinite(result).all()):
        raise ValueError('invalid native-Mizar BF16 prefix')
    return result


def option_logits(lm, prefix, ids, device):
    import torch
    if not ids or prefix.shape[1] + len(ids) > 1024:
        raise ValueError('candidate body empty or exceeds context; never truncate')
    # Last candidate token is a target only. This also works for one-token options.
    suffix = torch.tensor([ids[:-1]], device=device, dtype=torch.long)
    with torch.inference_mode(), autocast(device):
        full = torch.cat([prefix, lm.get_input_embeddings()(suffix).to(prefix.dtype)], dim=1)
        out = lm(inputs_embeds=full, attention_mask=None, labels=None, use_cache=False)
        logits = out.logits[:, prefix.shape[1]-1:prefix.shape[1]-1+len(ids), :].float()
    if logits.shape[1] != len(ids) or not bool(torch.isfinite(logits).all()):
        raise ValueError('nonfinite or incorrectly shifted candidate logits')
    return logits


def option_score(lm, prefix, ids, device):
    import torch
    logits = option_logits(lm, prefix, ids, device)
    targets = torch.tensor(ids, device=device, dtype=torch.long).view(1, -1, 1)
    scores = logits.log_softmax(dim=-1).gather(-1, targets).squeeze(-1)
    return float(scores.sum(dtype=torch.float32).item())


def score(runtime, prefix, record):
    from .results import choose
    scores = [option_score(runtime['lm'], prefix, ids, runtime['device']) for ids in record['choice_token_ids']]
    predicted = choose(scores)
    return {'option_loglikelihood': scores,
            'option_token_counts': [len(ids) for ids in record['choice_token_ids']],
            'option_mean_loglikelihood': [s/len(ids) for s, ids in zip(scores, record['choice_token_ids'])],
            'prediction_index': predicted, 'correct': predicted == record['gold_index'],
            'option_score_tie': scores.count(scores[predicted]) > 1}


def generate(runtime, prefix):
    import torch
    lm, device, tokenizer = runtime['lm'], runtime['device'], runtime['tokenizer']
    tokens, current, stop = [], prefix, 'length'
    with torch.inference_mode(), autocast(device):
        for _ in range(300):
            out = lm(inputs_embeds=current, attention_mask=None, labels=None, use_cache=False)
            logits = out.logits[:, -1, :].float()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError('nonfinite free-generation logits')
            token = int(logits.argmax(dim=-1).item())
            tokens.append(token)
            if token == 0:
                stop = 'eos'
                break
            continuation = lm.get_input_embeddings()(torch.tensor([[token]], device=device)).to(prefix.dtype)
            current = torch.cat([current, continuation], dim=1)
    return {'generated_token_ids': tokens, 'generated_token_count': len(tokens), 'stop_reason': stop,
            'generated_text_raw': tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            'generated_output': tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)}


def check_physical_equivalence(runtime, prefix, record):
    """One real BF16 retained-stack/deepcopy/reindex comparison inside the same job."""
    import torch
    controller = runtime['controller']
    controller.set_pair(config.PAIRS[0])
    lm = runtime['lm']
    # Use a temporary unhooked LM: no state writes to any source tensor.
    pruned = copy.deepcopy(lm)
    for module in pruned.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
    for index, layer in enumerate(pruned.model.layers):
        layer.self_attn.layer_idx = index
    try:
        ids = record['choice_token_ids'][0]
        expected = option_logits(lm, prefix, ids, runtime['device'])
        actual = option_logits(pruned, prefix, ids, runtime['device'])
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        a = option_score(lm, prefix, ids, runtime['device'])
        b = option_score(pruned, prefix, ids, runtime['device'])
        if abs(a-b) > 0.02 * len(ids):
            raise ValueError('physical pruning and mask likelihood differ')
        return {'status': 'PASS', 'drop_pair_0based': list(config.PAIRS[0]),
                'physical_layers': len(pruned.model.layers), 'rtol': 0.02, 'atol': 0.02,
                'logits_max_abs_error': float((actual-expected).abs().max().item()),
                'loglikelihood_abs_error': abs(a-b)}
    finally:
        del pruned
        controller.set_pair(())
