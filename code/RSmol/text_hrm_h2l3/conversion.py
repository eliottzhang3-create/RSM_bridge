"""Offline CPU conversion of original 30-layer SmolLM2 to 20-layer HRM."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import traceback

from .contract import (ROUTE, ARCHITECTURE, SOURCE_LAYERS, SCHEDULE, DEFAULT_SOURCE,
    DEFAULT_MODEL, target_config, validate_source_config, require_external,
    file_inventory, verify_inventory, canonical_hash, write_json, validate_config)
from .publication import atomic_publish


def read_weights(path, *, require_float32=False):
    """Load one unambiguous HF weight format, validating every shard and tensor."""
    import torch
    path = Path(path).resolve()
    candidates = [p for p in ('model.safetensors', 'model.safetensors.index.json',
                              'pytorch_model.bin', 'pytorch_model.bin.index.json') if (path / p).is_file()]
    if len(candidates) != 1:
        raise ValueError(f'expected one HF weight entry point: {candidates}')
    entry = candidates[0]
    weight_map = None
    if entry.endswith('.index.json'):
        weight_map = json.loads((path / entry).read_text(encoding='utf-8'))['weight_map']
        names = sorted(set(weight_map.values()))
    else:
        names = [entry]
    state = {}
    for name in names:
        shard = (path / name).resolve()
        if path not in shard.parents or not shard.is_file():
            raise ValueError('weight shard outside artifact or missing')
        if shard.suffix == '.safetensors':
            from safetensors.torch import load_file
            values = load_file(str(shard), device='cpu')
        else:
            values = torch.load(shard, map_location='cpu', weights_only=True)
        for key, value in values.items():
            if key in state or not torch.is_tensor(value) or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f'duplicate/nonfinite/non-floating tensor: {key}')
            if require_float32 and value.dtype != torch.float32:
                raise ValueError(f'HRM checkpoint tensor must be FP32: {key}')
            if weight_map is not None and weight_map.get(key) != name:
                raise ValueError(f'weight index contradicts shard: {key}')
            state[key] = value.float()
    if weight_map is not None and set(state) != set(weight_map):
        raise ValueError('weight index inventory mismatch')
    embed, head = state.get('model.embed_tokens.weight'), state.get('lm_head.weight')
    if embed is None and head is None:
        raise ValueError('missing tied embedding/head')
    if embed is not None and head is not None and not torch.equal(embed, head):
        raise ValueError('conflicting tied source embedding/head')
    state.setdefault('model.embed_tokens.weight', head)
    state.setdefault('lm_head.weight', embed)
    return state


def forward_audit(model):
    import torch
    model.eval()
    ids = (torch.arange(12).reshape(2, 6) + 3) % model.config.vocab_size
    mask = torch.ones_like(ids)
    mask[1, -2:] = 0
    counts = {'entry': 0, 'L': 0, 'H': 0}
    def hook(name):
        def record(*_args):
            counts[name] += 1
        return record
    handles = [model.model.entry_norm.register_forward_hook(hook('entry')),
               model.model.l_exit_norm.register_forward_hook(hook('L')),
               model.model.h_exit_norm.register_forward_hook(hook('H'))]
    try:
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            changed = ids.clone()
            changed[:, -1] = (changed[:, -1] + 1) % model.config.vocab_size
            before = dict(counts)
            other = model(input_ids=changed, attention_mask=mask, use_cache=False).logits
        if before != {'entry': 1, 'L': 6, 'H': 2} or model.model.last_trace != list(SCHEDULE):
            raise ValueError('wrong trace/MagicNorm counts')
        if not torch.isfinite(logits).all() or not torch.equal(logits[:, :-1], other[:, :-1]):
            raise ValueError('nonfinite logits or future-token leakage')
    finally:
        for handle in handles:
            handle.remove()
    return dict(trace=list(SCHEDULE), magicnorm_calls=before,
                causal_suffix_perturbation_pass=True, finite_logits=True), logits


def _verify_layer_transform(source, target, *, middle):
    """Exact tensor certificate, independent of any forward tolerance."""
    import torch
    from .model import ParameterlessRMSNorm
    expected = dict(source.state_dict())
    if middle:
        for norm in ('input_layernorm', 'post_attention_layernorm'):
            module = getattr(target, norm)
            if not isinstance(module, ParameterlessRMSNorm) or module.eps != getattr(source, norm).variance_epsilon:
                raise ValueError(f'converted {norm} type/epsilon mismatch')
            gamma = expected.pop(norm + '.weight')
            names = ('self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight') if norm == 'input_layernorm' else ('mlp.gate_proj.weight', 'mlp.up_proj.weight')
            for name in names:
                expected[name] = expected[name] * gamma.unsqueeze(0)
        expected['self_attn.o_proj.weight'] = expected['self_attn.o_proj.weight'] * 2.0
        expected['self_attn.gate_proj.weight'] = torch.zeros_like(source.self_attn.q_proj.weight)
    actual = target.state_dict()
    if set(actual) != set(expected):
        raise ValueError('converted layer tensor inventory mismatch')
    for name, value in expected.items():
        if actual[name].dtype != torch.float32 or not torch.equal(value, actual[name]):
            raise ValueError(f'converted layer tensor mismatch: {name}')


def _parity_metrics(actual, reference, *, atol=3e-6, rtol=3e-5):
    """Per-token scale handles cancellation; RMS and elementwise limits both apply."""
    import torch
    a, b = actual.double(), reference.double()
    if a.shape != b.shape or not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise ValueError('nonfinite/shape mismatch in conversion parity')
    error = (a - b).abs()
    scale = b.square().mean(-1, keepdim=True).sqrt()
    # Fixed small-value floor plus per-token RMS and per-element relative terms.
    # Using a per-token scale avoids hiding one bad token behind another large token.
    envelope = atol + rtol * (scale + b.abs())
    element_ratio = float((error / envelope).max())
    rms_ratio = float((error.square().mean(-1, keepdim=True).sqrt() / (atol + rtol * scale)).max())
    return dict(passed=element_ratio <= 1.0 and rms_ratio <= 1.0,
        max_abs=float(error.max()), max_element_budget_ratio=element_ratio,
        max_token_rms_budget_ratio=rms_ratio)


def _layer_reference_fp64(layer, x, *, positions, cos_sin):
    """Independent CPU FP64 block math; never casts the production module itself.

    Merely calling layer.double() would still use FP32 RMSNorm math in Llama and
    ParameterlessRMSNorm. This functional reference keeps ALL block math FP64,
    using the same precomputed RoPE constants as the actual FP32 comparison.
    """
    import torch
    import torch.nn.functional as F
    from .model import ParameterlessRMSNorm

    def norm(value, module):
        eps = module.eps if isinstance(module, ParameterlessRMSNorm) else module.variance_epsilon
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        return value if isinstance(module, ParameterlessRMSNorm) else value * module.weight.double()

    def linear(value, module):
        return F.linear(value, module.weight.double(), None if module.bias is None else module.bias.double())

    def rotate(value):
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    x = x.double()
    y = norm(x, layer.input_layernorm)
    attention = layer.self_attn
    shape = (*y.shape[:-1], -1, attention.head_dim)
    q, k, v = [linear(y, module).view(shape).transpose(1, 2)
               for module in (attention.q_proj, attention.k_proj, attention.v_proj)]
    cos, sin = [value.double().unsqueeze(1) for value in cos_sin]
    q, k = q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
    groups = q.shape[1] // k.shape[1]
    k, v = [value.repeat_interleave(groups, dim=1) for value in (k, v)]
    scores = (q @ k.transpose(-2, -1)) * attention.scaling
    causal = positions[:, None, :, None] >= positions[:, None, None, :]
    scores = scores.masked_fill(~causal, float('-inf'))
    a = (scores.softmax(-1) @ v).transpose(1, 2).reshape(*y.shape[:-1], -1)
    if hasattr(attention, 'gate_proj'):
        a = a * torch.sigmoid(linear(y, attention.gate_proj))
    x = x + linear(a, attention.o_proj)
    y = norm(x, layer.post_attention_layernorm)
    mlp = layer.mlp
    return x + linear(F.silu(linear(y, mlp.gate_proj)) * linear(y, mlp.up_proj), mlp.down_proj)


def conversion_parity(source, target):
    """Exact folding certificate + scale-aware FP32 audit, FP64 on cancellation."""
    import torch
    from .llama_compat import _causal_mask, _call_decoder_layer
    generator = torch.Generator(device='cpu').manual_seed(19)
    x = torch.randn((2, 5, target.config.hidden_size), generator=generator)
    positions = torch.arange(5).unsqueeze(0).expand(2, -1)
    mask = _causal_mask(attention_mask=None, batch_size=2, query_length=5,
                       past_length=0, dtype=x.dtype, device=x.device)
    cos_sin = target.model.rotary_emb(x, positions)
    errors, reports = [], []
    source.eval(); target.eval()
    with torch.no_grad():
        for i, source_index in enumerate(SOURCE_LAYERS):
            original, converted = source.model.layers[source_index], target.model.layers[i]
            try:
                _verify_layer_transform(original, converted, middle=5 <= i < 15)
            except ValueError as exc:
                raise ValueError(f'fold/gate tensor check failed at target layer {i}: {exc}') from exc
            args = dict(attention_mask=mask, position_ids=positions, cache=None,
                        use_cache=False, cache_position=positions[0],
                        position_embeddings=cos_sin, output_attentions=False)
            a = _call_decoder_layer(original, x, **args)[0]
            b = _call_decoder_layer(converted, x, **args)[0]
            metrics = _parity_metrics(b, a)
            report = dict(target_layer=i, source_layer=source_index, fp32=metrics,
                old_elementwise_allclose=bool(torch.allclose(a, b, atol=3e-6, rtol=3e-5)))
            if not report['old_elementwise_allclose']:
                a64 = _layer_reference_fp64(original, x, positions=positions, cos_sin=cos_sin)
                b64 = _layer_reference_fp64(converted, x, positions=positions, cos_sin=cos_sin)
                report['fp64'] = dict(source_fp32_vs_reference=_parity_metrics(a, a64),
                    target_fp32_vs_reference=_parity_metrics(b, b64),
                    stored_folded_vs_original=_parity_metrics(b64, a64, atol=3e-7, rtol=3e-6))
            if not metrics['passed'] or any(not value['passed'] for value in report.get('fp64', {}).values()):
                raise ValueError(f'fold/gate parity failed at target layer {i}: {json.dumps(report)}')
            if 'fp64' in report:
                print(f'[convert parity] layer={i} max_abs={metrics["max_abs"]:.8g}; exact tensors and FP64 references PASS', flush=True)
            reports.append(report)
            errors.append(metrics['max_abs'])
    return dict(layer_max_abs_errors=errors, atol=3e-6, rtol=3e-5,
                policy='exact_tensors_per_token_rms_and_element_budget_fp64_on_cancellation_v2',
                exact_layer_tensors=True, layers=reports,
                whole_model_parity_claimed=False)


def convert(source, output, *, threads=8, production=True):
    import torch
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
    from .model import SmolHRMForCausalLM, convert_state, strict_load, load_model
    source = Path(source).resolve()
    output = require_external(output)
    if output.exists() or output.is_symlink() or source == output or source in output.parents or output in source.parents:
        raise ValueError('output must be new, external and disjoint from source')
    if threads <= 0:
        raise ValueError('threads must be positive')
    if (source / 'training_state.pt').exists() or (source / 'checkpoint_complete.json').exists():
        raise ValueError('requires original SmolLM2, not a training checkpoint')
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    validate_source_config(config.to_dict(), production=production)
    config._attn_implementation = 'sdpa'
    inventory = file_inventory(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name('.' + output.name + '.conversion.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    staging = None
    try:
        os.write(fd, f'{os.getpid()}\n'.encode())
        print('[convert 1/5] Loading and validating original 30-layer source', flush=True)
        state = read_weights(source)
        original = LlamaForCausalLM(config).float()
        expected = original.state_dict()
        if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
            raise ValueError('source tensor inventory/shape differs from original Llama')
        original.load_state_dict(state, strict=True)
        original.tie_weights()
        print('[convert 2/5] Selecting 20 layers, folding norms and compensating gates', flush=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = SmolHRMForCausalLM(target_config(config)).float()
        converted = convert_state(state, model)
        strict_load(model, converted)
        parity = conversion_parity(original, model)
        audit, logits = forward_audit(model)
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.staging-', dir=output.parent))
        print('[convert 3/5] Saving FP32 model and tokenizer', flush=True)
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        print('[convert 4/5] Strict reload, tensor/logit/tokenizer verification', flush=True)
        reloaded = load_model(staging, production=production)
        if any(not torch.equal(v, reloaded.state_dict()[k]) for k, v in model.state_dict().items()):
            raise ValueError('reload tensor mismatch')
        _, other_logits = forward_audit(reloaded)
        if not torch.equal(logits, other_logits):
            raise ValueError('reload logits differ')
        tokenizer2 = AutoTokenizer.from_pretrained(staging, local_files_only=True)
        probes = ['The sound of rain.', 'Hello world!', '中文测试\n123']
        if tokenizer.get_vocab() != tokenizer2.get_vocab() or tokenizer.special_tokens_map != tokenizer2.special_tokens_map or tokenizer(probes, add_special_tokens=False) != tokenizer2(probes, add_special_tokens=False):
            raise ValueError('tokenizer changed')
        verify_inventory(source, inventory)
        metadata = dict(route=ROUTE, architecture=ARCHITECTURE, production=production,
            source_path=str(source), source_files=inventory, source_identity=canonical_hash(inventory),
            source_layers=list(SOURCE_LAYERS), physical_layers=20, execution_layers=50,
            unique_parameters=sum(p.numel() for p in model.parameters()), parity=parity,
            forward=audit, reload_tensors_logits_tokenizer_equal=True,
            dtype='float32', initial_state_shape=list(model.model.zL_init.shape),
            norm_folding='qkv/gate_up columns times source gamma; o_proj times2',
            new_gate_weights_zero=True, audio_constructed=False)
        write_json(staging / 'conversion_audit.json', metadata)
        write_json(staging / 'conversion_complete.json', dict(complete=True, route=ROUTE,
            architecture=ARCHITECTURE, production=production, source_identity=metadata['source_identity'],
            files=file_inventory(staging)))
        print('[convert 5/5] Atomic publication without overwrite', flush=True)
        atomic_publish(staging, output)
        return metadata | {'output': str(output)}
    except BaseException as exc:
        if staging is not None and staging.exists():
            write_json(staging / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        os.close(fd)
        lock.unlink()


def inspect_artifact(path, *, resume=False, production=True):
    root = Path(path).resolve()
    raw = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    validate_config(raw, production=production)
    name = 'checkpoint_complete.json' if (root / 'checkpoint_complete.json').exists() else 'conversion_complete.json'
    marker = json.loads((root / name).read_text(encoding='utf-8'))
    if marker.get('complete') is not True or marker.get('route') != ROUTE or marker.get('architecture') != ARCHITECTURE or marker.get('production') is not production:
        raise ValueError('artifact identity/completion mismatch')
    if name == 'checkpoint_complete.json':
        from .contract import FORMAT, RECIPE
        step = marker.get('step')
        if marker.get('format') != FORMAT or type(step) is not int or not 1 <= step <= 9244 or root.name != f'checkpoint-step-{step:06d}':
            raise ValueError('checkpoint step/name mismatch')
        if not resume and step != RECIPE['total_steps']:
            raise ValueError('fresh weights-only training requires a final text checkpoint')
        if not (root / 'training_state.pt').is_file():
            raise ValueError('missing full training state')
    elif resume:
        raise ValueError('conversion is not a full training resume checkpoint')
    verify_inventory(root, marker.get('files'))
    return marker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    print(json.dumps(convert(args.source_dir, args.output_dir, threads=args.threads), indent=2))


if __name__ == '__main__':
    main()
