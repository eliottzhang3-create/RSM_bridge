"""CPU-only extraction of final Mizar LM weights and audited recurrent conversion."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import traceback

from .contract import (ROUTE, SOURCE_LAYERS, SCHEDULE, LM_PREFIX, DEFAULT_SOURCE,
                       DEFAULT_MODEL, build_target_config, release_manifest,
                       sha256_file, source_identity, validate_model_config)
from .publication import atomic_publish

REPO_ROOT = Path(__file__).resolve().parents[3]
REMOTE_CHECKOUT = Path('/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def validate_output(source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing existing conversion output: {output}')
    output = output.resolve()
    if (output == source or source in output.parents or output in source.parents
            or any(output == root or root in output.parents
                   for root in (REPO_ROOT.resolve(), REMOTE_CHECKOUT.resolve()))):
        raise ValueError('Conversion output must be external to the source and Git checkouts')
    return source, output


def verify_source_files(source):
    """Check only final checkpoint and text configuration/tokenizer assets."""
    release = release_manifest()
    records = [release['checkpoint']] + [item for item in release['files']
               if item['path'].startswith('assets/smollm2-135m/')
               and not item['path'].endswith('model.safetensors')]
    for item in records:
        path = source / item['path']
        if path.stat().st_size != item['bytes'] or sha256_file(path) != item['sha256']:
            raise ValueError(f'Official Mizar release file mismatch: {path}')
    return records


def extract_lm_state(state):
    import torch
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v)
                                                        for k, v in state.items()):
        raise ValueError('Official checkpoint must be a plain tensor state_dict')
    result = {key[len(LM_PREFIX):]: value for key, value in state.items() if key.startswith(LM_PREFIX)}
    if not result:
        raise ValueError(f'Official checkpoint contains no {LM_PREFIX} tensors')
    for key, value in result.items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f'Expected finite FP32 text weights: {key}')
    # A tied load can overwrite the embedding with lm_head without reporting a key error.
    embedding, head = result.get('model.embed_tokens.weight'), result.get('lm_head.weight')
    if embedding is None or head is None or not torch.equal(embedding, head):
        raise ValueError('Official tied embedding/lm_head values differ or are missing')
    return result


def strict_load(model, state):
    expected = model.state_dict()
    if set(expected) != set(state):
        raise ValueError(f'Text state keys differ: missing={sorted(set(expected)-set(state))}; '
                         f'extra={sorted(set(state)-set(expected))}')
    for name, value in expected.items():
        if value.shape != state[name].shape or value.dtype != state[name].dtype:
            raise ValueError(f'Text tensor shape/dtype mismatch: {name}')
    model.load_state_dict(state, strict=True)


def source_key(target_key):
    match = re.match(r'^model\.layers\.(\d+)\.(.+)$', target_key)
    return (f'model.layers.{SOURCE_LAYERS[int(match[1])]}.{match[2]}' if match else target_key)


def is_lora_tensor(name):
    return '.lora_A.' in name or '.lora_B.' in name


def inherited_state(source_state, target):
    result = {}
    inherited_count = 0
    for name, value in target.state_dict().items():
        if is_lora_tensor(name):
            result[name] = value
            continue
        original = source_key(name)
        if original not in source_state:
            raise ValueError(f'Missing inherited text tensor: {original}')
        result[name] = source_state[original]
        inherited_count += 1
    strict_load(target, result)
    target.tie_weights()
    assert_inherited(source_state, target)
    return inherited_count


def assert_inherited(source_state, target):
    import torch
    for name, value in target.state_dict().items():
        if is_lora_tensor(name):
            continue
        original = source_state[source_key(name)]
        if value.dtype != original.dtype or not torch.equal(value, original):
            raise AssertionError(f'Inherited tensor changed: {name}')
    if target.model.embed_tokens.weight is not target.lm_head.weight:
        raise AssertionError('Embedding/lm_head must share the same Parameter')


def audit_forward(target, *, require_lora=True):
    import torch
    from code.RSmol.mizar_text_5_10x2_5_loop_lora_r8.loop_lora import validate_loop_lora
    target.eval()
    ids = (torch.arange(12).reshape(2, 6) + 3) % target.config.vocab_size
    mask = torch.ones_like(ids)
    mask[1, -2:] = 0
    trace = []
    handles = [layer.register_forward_pre_hook(lambda _m, _a, i=i: trace.append(i))
               for i, layer in enumerate(target.model.layers)]
    try:
        with torch.no_grad():
            logits = target(input_ids=ids, attention_mask=mask, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()
    if trace != list(SCHEDULE) or not torch.isfinite(logits).all():
        raise AssertionError(f'Invalid recurrent forward: {trace}')
    audit = {'physical_execution_trace': trace, 'finite_logits': True}
    if require_lora:
        audit['loop_lora'] = validate_loop_lora(target.model)
        expected = [(i, p, 0 if i < 15 else 1) for i, p in enumerate(SCHEDULE) if 5 <= p < 15]
        if list(target.model.last_loop_lora_trace) != expected:
            raise AssertionError('Middle-loop LoRA execution mismatch')
        audit['loop_lora_trace'] = expected
    return audit, logits


def assert_zero_lora_parity(source_state, target, logits):
    """Compare to the original recurrent model using the SAME inherited weights."""
    import torch
    from code.RSmol.recursive_model_5_10_5 import RecursiveLlamaForCausalLM
    lora = {name: value for name, value in target.state_dict().items() if is_lora_tensor(name)}
    if len(lora) != 280 or any(value.ne(0).any() for name, value in lora.items() if '.lora_B.' in name):
        raise AssertionError('Fresh conversion requires 280 LoRA matrices and zero B')
    reference = RecursiveLlamaForCausalLM(copy.deepcopy(target.config)).float().eval()
    inherited_state(source_state, reference)
    _, reference_logits = audit_forward(reference, require_lora=False)
    torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)
    return True


def assert_reload_equal(target, reloaded):
    import torch
    expected, actual = target.state_dict(), reloaded.state_dict()
    if set(expected) != set(actual):
        raise AssertionError('Reloaded model tensor inventory changed')
    for name, value in expected.items():
        if value.dtype != actual[name].dtype or not torch.equal(value, actual[name]):
            raise AssertionError(f'Reloaded tensor changed: {name}')


def convert(source, output, *, threads=8):
    source, output = validate_output(source, output)
    if threads <= 0:
        raise ValueError('threads must be positive')
    print('[convert 1/5] Verifying official final checkpoint and text assets', flush=True)
    records = verify_source_files(source)
    import torch
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
    from code.RSmol.recursive_model_mizar_text_5_10x2_5_loop_lora_r8 import MizarTextLoopLoRAR8ForCausalLM
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    assets = source / 'assets/smollm2-135m'
    config = AutoConfig.from_pretrained(assets, local_files_only=True)
    if (config.model_type != 'llama' or config.num_hidden_layers != 30
            or config.hidden_size != 576 or config.vocab_size != 49152
            or config.tie_word_embeddings is not True):
        raise ValueError('Unexpected official Mizar text configuration')
    state = torch.load(source / source_identity()['checkpoint_relative_path'],
                       map_location='cpu', weights_only=True)
    release = release_manifest()
    if (len(state) != release['tensor_count']
            or sum(v.numel() for v in state.values()) != release['tensor_elements']):
        raise ValueError('Official full checkpoint tensor inventory mismatch')
    text_state = extract_lm_state(state)
    print('[convert 2/5] Strictly loading final LM, then inheriting selected 20 layers', flush=True)
    source_model = LlamaForCausalLM(config).float()
    strict_load(source_model, text_state)
    target = MizarTextLoopLoRAR8ForCausalLM(build_target_config(source_model.config)).float()
    inherited_count = inherited_state(text_state, target)
    validate_model_config(target.config.to_dict())
    if len(target.model.layers) != 20 or not all(p.requires_grad for p in target.parameters()):
        raise AssertionError('Expected 20 fully trainable decoder modules')
    audit, logits = audit_forward(target)
    zero_lora_parity = assert_zero_lora_parity(text_state, target, logits)
    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True, use_fast=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.staging-', dir=output.parent))
    try:
        print(f'[convert 3/5] Saving FP32 HF weights/tokenizer to {staging}', flush=True)
        target.save_pretrained(staging, safe_serialization=True)
        copied = []
        for item in records:
            path = Path(item['path'])
            if path.parent.as_posix() == 'assets/smollm2-135m' and path.name != 'config.json':
                shutil.copy2(source / path, staging / path.name)
                copied.append(path.name)
        print('[convert 4/5] Reloading weights, checking inherited tensors, logits and tokenizer', flush=True)
        reloaded, loading = MizarTextLoopLoRAR8ForCausalLM.from_pretrained(
            staging, local_files_only=True, torch_dtype=torch.float32,
            output_loading_info=True, attn_implementation=target.config._attn_implementation)
        if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError(f'Converted strict reload failed: {loading}')
        assert_inherited(text_state, reloaded)
        assert_reload_equal(target, reloaded)
        reload_audit, reloaded_logits = audit_forward(reloaded)
        torch.testing.assert_close(reloaded_logits, logits, rtol=0, atol=0)
        tokenizer2 = AutoTokenizer.from_pretrained(staging, local_files_only=True, use_fast=True)
        probes = ['The sound of rain.', 'Hello world!', '中文测试\n123']
        if (tokenizer.get_vocab() != tokenizer2.get_vocab()
                or tokenizer.special_tokens_map != tokenizer2.special_tokens_map
                or tokenizer(probes, add_special_tokens=False) != tokenizer2(probes, add_special_tokens=False)):
            raise AssertionError('Saved tokenizer changed')
        metadata = {'status': 'PASS', 'text_route': ROUTE, 'source': source_identity(),
                    'source_root': str(source), 'source_files_verified': records,
                    'source_layer_indices_0based': list(SOURCE_LAYERS),
                    'physical_layers': 20, 'logical_layers': 30,
                    'equal_inherited_tensors': inherited_count, 'parameter_dtype': 'float32',
                    'final_lm_strict_load': True, 'audio_model_constructed': False,
                    'zero_lora_matches_original_recursive_logits': zero_lora_parity,
                    'loop_lora': audit['loop_lora'],
                    'forward': audit, 'reload_forward': reload_audit,
                    'reload_tensors_logits_tokenizer_equal': True, 'copied_text_assets': copied,
                    'original_30_layer_logits_parity_claimed': False}
        write_json(staging / 'recursive_conversion_metadata.json', metadata)
        write_json(staging / 'conversion_complete.json', {
            'complete': True, 'text_route': ROUTE, 'source': source_identity(),
            'physical_layers': 20, 'logical_layers': 30})
        print('[convert 5/5] Publishing without overwriting existing outputs', flush=True)
        method = atomic_publish(staging, output)
        return {**metadata, 'output': str(output), 'publication_method': method}
    except BaseException as error:
        if staging.exists():
            write_json(staging / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        print(f'[convert failed] Staging preserved: {staging}', flush=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    print(json.dumps(convert(args.source_root, args.output_dir, threads=args.threads), indent=2))


if __name__ == '__main__':
    main()
