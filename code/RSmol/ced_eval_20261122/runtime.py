"""Strict model-only inference, importing each unchanged training implementation."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import random
import sys
from pathlib import Path

from .registry import (ROOT, ROUTES, PACKAGE, SCHEDULE, TEXT_CONTRACT,
                       TEXT_ARCHITECTURE, TRAINING_GEOMETRY, variant_spec)
from .preflight import validate_header
from . import sources

if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
import mizar_s1_ced_eval_runtime as reference

PROMPT_TOKENS, MAX_NEW_TOKENS, AUDIO_TOKENS = 256, 300, 126
PREFIX_TOKENS, HIDDEN_SIZE = 383, 576
PROTOCOL = copy.deepcopy(reference.PROTOCOL)
sha256_file = reference.sha256_file
parse_training_geometry = reference.parse_training_geometry
BRANCH_KEYS = {'layer_injection', 'audio_listen', 'audio_read2', 'audio_fusion',
               'loop_lora', 'audio_cross_attention', 'audio_reinjection', 'prompt_mix',
               'serial_audio_injection', 'serial_injection', 'text_memory'}


def read_checkpoint(args):
    """Resolve epoch-file steps once; keep the CPU payload explicit on this invocation."""
    sources.prepare(args.eval_variant,args.sources_cache)
    if hasattr(args, '_checkpoint_payload'):
        current = args.checkpoint.stat()
        if (current.st_size, current.st_mtime_ns) != args._checkpoint_signature:
            raise ValueError('checkpoint changed after identity resolution')
        return args._checkpoint_payload
    import torch
    before = args.checkpoint.stat()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    after = args.checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('checkpoint changed while it was being read')
    step = saved.get('total_step') if isinstance(saved, dict) else None
    if type(step) is not int or step <= 0 or (args.expected_step is not None and step != args.expected_step):
        raise ValueError('checkpoint total_step disagrees with requested/filename step')
    args.expected_step = step
    args._checkpoint_signature = (after.st_size, after.st_mtime_ns)
    args._checkpoint_payload = saved
    return saved


def evaluation_sources():
    """Include all reused helpers, without changing any old runner's identity glob."""
    files = set(PACKAGE.glob('*.py')) | set(PACKAGE.glob('*.sh')) | set(PACKAGE.glob('*.json'))
    files.add(ROOT / 'run_ced_eval_20261122_4090.sh')
    files.add(ROOT / 'text71427_ced_eval/observers.py')
    for name in ('mizar_s1_ced_reasonaqa5000_eval', 'mizar_s1_ced_mmsu_eval'):
        files.update((ROOT / name).glob('*.py'))
        files.update((ROOT / name).glob('*.sh'))
    for name in ('mizar_s1_ced_eval_runtime.py', 'mizar_s1_ced_multilayer_eval_runtime.py',
                 'evaluate_mizar_s1_ced_multilayer_benchmark.py',
                 'evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.py',
                 'evaluate_mmar_5_10x2_5_mesh_mellow.py'):
        files.add(ROOT / 'scripts' / name)
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in sorted(files)}


def validate_checkpoint(saved, runtime, route_contract, expected_step, expected_geometry=None, *, eval_variant):
    spec = variant_spec(eval_variant)
    validate_header(saved, runtime, route_contract, target=eval_variant)
    route, contract = spec['route'], saved.get('contract', {})
    identity = route_contract.digest(contract)
    if (saved.get('contract_sha256') != identity or runtime.get('contract_sha256') != identity
            or runtime.get('contract') != contract or runtime.get('mode') != 'formal'
            or runtime.get('route') != route):
        raise ValueError('checkpoint/runtime contract identity mismatch')
    _, pinned = sources.profile(eval_variant)
    if (contract.get('code_sha256') != route_contract.code_digest() or
            contract.get('code_sha256') != pinned['training_code_sha256'][eval_variant]):
        raise ValueError('training source differs from the checkpoint: '+json.dumps({
            'target':eval_variant,'source_revision':spec['revision'],
            'checkpoint_code_sha256':contract.get('code_sha256'),
            'loaded_code_sha256':route_contract.code_digest(),
            'pinned_code_sha256':pinned['training_code_sha256'][eval_variant]}))
    if not spec['plain'] and contract.get('recursive_model_source_sha256') != route_contract.recursive_model_source_sha256():
        raise ValueError('shared text source differs from the checkpoint')
    expected = {
        'route': route,
        'sample_rate': 32000, 'audio_policy': 'mono_resample_center_crop_right_pad_fixed20_split10',
        'prefix': [126, 1, 256], 'answer_tokens': 300, 'attention_mask': None,
        'pad_token': '!', 'eos_suffix': ' <|endoftext|>', 'forward_dtype': 'float32', 'autocast': False,
    }
    if not spec['plain']:
        expected.update(text_model_contract=TEXT_CONTRACT,text_model_architecture=TEXT_ARCHITECTURE,
                        text_model_schedule=list(SCHEDULE))
    elif any(k in contract for k in ('text_model_contract','text_model_architecture','text_model_schedule','recursive_text_checkpoint')):
        raise ValueError('plain 30-layer SmolLM2 must not declare a recursive text architecture')
    for key, value in expected.items():
        if key not in contract or contract[key] != value:
            raise ValueError(f'text71427 contract mismatch: {key}')
    if not spec['plain'] and (route_contract.TEXT_MODEL_CONTRACT != TEXT_CONTRACT
            or route_contract.TEXT_MODEL_ARCHITECTURE != TEXT_ARCHITECTURE
            or tuple(route_contract.TEXT_MODEL_SCHEDULE) != SCHEDULE
            or not contract.get('recursive_text_checkpoint')):
        raise ValueError('missing or incorrect 7-14x2-7 text provenance')
    if (type(expected_step) is not int or type(saved.get('total_step')) is not int
            or saved['total_step'] != expected_step or saved.get('num_epochs') != 3
            or saved.get('loss_reduction') != reference.TRAINING_LOSS_REDUCTION):
        raise ValueError('checkpoint step / epoch horizon / loss reduction mismatch')
    geometry = contract.get('geometry', {})
    actual = tuple(geometry.get(k) for k in ('world_size', 'microbatch', 'gradient_accumulation_steps'))
    if (actual != TRAINING_GEOMETRY or parse_training_geometry(expected_geometry or actual) != TRAINING_GEOMETRY
            or geometry.get('total_steps') != 34626 or not 0 < expected_step <= 34626):
        raise ValueError('expected training geometry 8,4,4 and the 34626-step horizon')
    if saved.get('batch_geometry') != {'world_size': 8, 'microbatch_size': 4,
            'gradient_accumulation_steps': 4, 'global_batch_size': 128}:
        raise ValueError('checkpoint batch geometry mismatch')
    for section, wanted in {
        'audio_encoder': {'name': 'CEDSmall', 'hidden_cache': False, 'frozen': True, 'mode': 'eval'},
        'mapper': {'name': 'freq_merge_126', 'hidden_dim': 384, 'frequency_bins': 4,
                   'output_dim': 576, 'resample': False, 'dropout': 0.05,
                   'max_temporal_tokens': 512, 'external_projection': 'Identity'},
    }.items():
        if any(contract.get(section, {}).get(k) != v for k, v in wanted.items()):
            raise ValueError(f'unsupported {section} contract')
    branches = {}
    if spec['branch_key']:
        branches[spec['branch_key']] = dict(getattr(route_contract, spec['branch_constant']))
        branches['loop_lora'] = dict(route_contract.LOOP_LORA_CONTRACT)
    for key in BRANCH_KEYS:
        if key in branches:
            if contract.get(key) != branches[key]:
                raise ValueError(f'trained branch contract mismatch: {key}')
        elif key in contract:
            raise ValueError(f'unexpected branch contract: {key}')
    if eval_variant == 'mesh_reinit':
        init = contract.get('audio_initialization',{})
        from .targets import target_records
        original = target_records()['mesh_original']['checkpoint']
        if (init.get('mode')!='audio_checkpoint_weights_only' or init.get('checkpoint')!=original
                or init.get('source_checkpoint_step')!=34626 or init.get('start_optimizer_step')!=0
                or init.get('restore_optimizer_scheduler_rng_cursor') is not False
                or not init.get('source_checkpoint_sha256') or not init.get('source_contract_sha256')):
            raise ValueError('retrained MeSH checkpoint lacks its original audio weights-only provenance')
        if any(contract.get('adamw',{}).get(k)!=v for k,v in
               {'decoder_lr':2e-4,'map_lr':2e-4,'min_lr':2e-5}.items()):
            raise ValueError('retrained MeSH training recipe identity differs')
    elif 'audio_initialization' in contract:
        raise ValueError('unexpected weights-only retraining provenance for this target')
    return contract


def normalize_state(saved, model):
    """Reject silent dtype casts, mixed namespaces and contradictory tied aliases."""
    import torch
    state = saved['state_dict']
    keys = list(state)
    ddp = bool(keys and all(str(k).startswith('module.') for k in keys))
    if ddp:
        state = {str(k)[7:]: v for k, v in state.items()}
    elif any(str(k).startswith('module.') for k in keys):
        raise ValueError('mixed DDP state namespace')
    expected = model.state_dict()
    if state.keys() != expected.keys():
        raise ValueError(f'complete audio state inventory differs: missing={sorted(expected.keys()-state.keys())}, extra={sorted(state.keys()-expected.keys())}')
    for name, value in state.items():
        target = expected[name]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f'state shape/dtype mismatch: {name}')
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f'nonfinite checkpoint tensor: {name}')
    embedding, head = 'caption_decoder.lm.model.embed_tokens.weight', 'caption_decoder.lm.lm_head.weight'
    if not torch.equal(state[embedding], state[head]):
        raise ValueError('tied embedding/head values disagree in checkpoint')
    return state, ddp


def audit_model(model, route_contract, contract, variant, route_root=None):
    """Training validators run on trained values, before the inference-only freeze."""
    import torch
    spec = variant_spec(variant)
    root = Path(route_root or spec['route_root']).resolve()
    if spec['plain']:
        from .plain import audit_model as audit_plain
        return audit_plain(model,root)
    variant = spec['model_variant']
    loader = importlib.import_module('models.recursive_text')
    implementation = importlib.import_module(spec['implementation'])
    expected_impl = ((root if spec['implementation'].startswith('models.') else root.parents[1])
                     / (spec['implementation'].replace('.', '/') + '.py'))
    if (Path(loader.__file__).resolve() != root / 'models/recursive_text.py'
            or Path(implementation.__file__).resolve() != expected_impl.resolve()
            or not isinstance(model.caption_decoder.lm, implementation.MizarText7_14x2_7MeshForCausalLM)):
        raise ValueError('loader/model implementation escaped the selected route')
    lm, core = model.caption_decoder.lm, model.caption_decoder.lm.model
    loader.validate_recursive_config(lm.config)
    if route_contract.recursive_checkpoint_contract(Path(contract['text_model_dir'])) != contract['recursive_text_checkpoint']:
        raise ValueError('source text checkpoint changed since training')
    audit = implementation.parameter_audit(lm)
    for key, value in {'logical_layer_count': 42, 'physical_layer_count': 28,
            'logical_cache_slot_count': 42, 'recursive_loops': 2, 'schedule': list(SCHEDULE),
            'memory_persistent': False, 'embedding_head_tied': True,
            'physical_layer_objects_independent': True}.items():
        if audit.get(key) != value:
            raise ValueError(f'loaded architecture mismatch: {key}')
    if (lm.config.hidden_size != HIDDEN_SIZE or len(core.layers) != 28
            or len({id(x) for x in core.layers}) != 28 or core.gradient_checkpointing
            or model.caption_decoder.text_model_contract != TEXT_CONTRACT
            or not model.caption_decoder.is_recursive_mesh):
        raise ValueError('expected all 28 independent physical layers without gradient checkpointing')
    if any(getattr(core, k, False) for k in ('audit_mode', 'gradient_audit_mode', 'routing_stats_mode',
            'collect_injection_stats', 'collect_listen_stats', 'collect_fusion_stats')):
        raise ValueError('tensor-retaining training diagnostics must be disabled during inference')
    routers = list(core.write_routers) + list(core.read_routers)
    if (len(routers) != 6 or any(tuple(r.weight.shape) != (5, HIDDEN_SIZE)
            or tuple(r.bias.shape) != (5,) for r in routers)):
        raise ValueError('original six five-slot routers differ')
    extras, branch = [], {}
    if variant != 'base':
        mlps = model.audio_encoder.base.layer_mlps
        if set(mlps) != {'4', '8'} or mlps['4'] is mlps['8']:
            raise ValueError('expected independent aligned CED4/8 MLPs')
        extras.append(mlps)
        if variant == 'loop_lora':
            branch = loader.validate_layer_injection(lm)
            extras.append(core.audio_gates)
        elif variant == 'listen':
            branch = loader.validate_audio_listen(lm)
            extras += [core.audio_gates, core.listen_routers]
            routers += list(core.listen_routers.values())
        elif variant == 'wfuse':
            branch = loader.validate_audio_fusion(lm, fresh=False)
            extras.append(core.audio_fusions)
        else:
            branch = loader.validate_audio_read2(lm, fresh=False)
            extras.append(core.audio_read_routers)
            routers += list(core.audio_read_routers)
            if variant == 'audio7_read2_ar6':
                extras.append(core.audio_write_routers)
                routers += list(core.audio_write_routers)
        lora_module = importlib.import_module('models.loop_lora')
        if Path(lora_module.__file__).resolve() != root / 'models/loop_lora.py':
            raise ValueError('LoRA implementation escaped selected route')
        lora = lora_module.validate_loop_lora(core)
        if any(lora.get(k) != v for k, v in {'rank': 8, 'alpha': 16, 'scale': 2.,
                'loops': 2, 'matrix_count': 392, 'parameter_count': 2279424, 'dropout': 0.}.items()):
            raise ValueError('expected the trained 392-matrix two-loop LoRA')
        count = sum(p.numel() for m in extras for p in m.parameters()) + lora['parameter_count']
        if count != spec['new_parameters']:
            raise ValueError('new branch parameter inventory differs')
        audit.update(loop_lora=lora, branch=branch, new_parameters=count)
    if len(routers) != spec['router_modules'] or len({id(r) for r in routers}) != len(routers):
        raise ValueError('router module count or independence differs')
    parameters = [p for r in routers for p in r.parameters()]
    if len({(p.device, p.untyped_storage().data_ptr()) for p in parameters}) != len(parameters):
        raise ValueError('router tensors share storage')
    if any('memory_slot' in n or n.endswith('.memory') for n in lm.state_dict()):
        raise ValueError('memory must be rebuilt per forward, not persisted in the checkpoint')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('inference weights must remain FP32')
    audit.update(router_modules=len(routers), trained_values_preserved=True)
    return audit


def load_runtime(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError('text71427 CED evaluation requires one CUDA GPU')
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    spec = variant_spec(args.eval_variant)
    if Path(args.route_root).resolve() != sources.route_root(args.eval_variant,args.sources_cache).resolve():
        raise ValueError('route implementation is fixed by the selected model')
    sources.prepare(args.eval_variant,args.sources_cache)
    modules = sources.import_training(args.eval_variant,args.sources_cache)
    route_contract = modules['contract']
    saved = read_checkpoint(args)
    del args._checkpoint_payload
    runtime = json.loads(args.runtime_config.read_text(encoding='utf-8'))
    contract = validate_checkpoint(saved, runtime, route_contract, args.expected_step,
                                   args.expected_geometry, eval_variant=args.eval_variant)
    assets = reference.audit_assets(contract)
    model = modules['models.mellow'].Mellow(
        audioenc_name='CEDSmall', d_in=384, d_out=HIDDEN_SIZE,
        text_decoder=contract['text_model_dir'], prefix_length=PREFIX_TOKENS,
        freeze_text_decoder_weights=False, use_pretrained_audioencoder=True,
        freeze_audio_encoder_weights=True, pretrained_audioencoder_path=contract['ced_model_dir'])
    model.float()
    state, ddp = normalize_state(saved, model)
    for name,value in state.items():
        if name.startswith('audio_encoder.base.ced.') and not torch.equal(value,model.state_dict()[name]):
            raise ValueError(f'checkpoint frozen CED differs from pretrained asset: {name}')
    model.load_state_dict(state, strict=True)
    structure = audit_model(model, route_contract, contract, args.eval_variant,args.route_root)
    model.to(device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(contract['text_model_dir'], local_files_only=True)
    source_tokens = (route_contract.validate_checkpoint_tokenizer(tokenizer, model.caption_decoder.lm.config.vocab_size)
                     if not spec['plain'] else {'source':'standard SmolLM2; training PAD override audited below'})
    tokenizer.add_special_tokens({'pad_token': '!'})
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'
    tokens = route_contract.validate_tokenizer_contract(tokenizer, tokenizer.encode(tokenizer.pad_token)[0],
        model.caption_decoder.lm.get_input_embeddings().num_embeddings)
    tokens['checkpoint_source'] = source_tokens
    if any(m.training for m in model.modules()):
        raise RuntimeError('every inference module must be eval')
    text_encoder = modules['data.audiotext_dataset'].AudioTextDataset.__new__(modules['data.audiotext_dataset'].AudioTextDataset)
    text_encoder.tokenizer = tokenizer
    if spec['plain']:
        from .plain import install_trace_audit
        trace = install_trace_audit(model.caption_decoder.lm)
    else:
        from text71427_ced_eval.observers import install_trace_audit
        trace = install_trace_audit(model.caption_decoder.lm, spec['model_variant'])
    audit = {'status': 'PASS', 'eval_variant': args.eval_variant, 'checkpoint_step': saved['total_step'],
        'checkpoint_epoch': saved.get('epoch'), 'contract_sha256': saved['contract_sha256'],
        'training_code_sha256': contract['code_sha256'], 'strict_state_dict': True,
        'source_revision':spec['revision'],'training_source_root':str(args.route_root),
        'architecture':'smollm2_standard_30' if spec['plain'] else 'mesh_7_14x2_7',
        'audio_initialization':contract.get('audio_initialization'),
        'ddp_state_dict_prefix_normalized': ddp, 'state_dict_source': 'training_checkpoint_state_dict',
        'training_route': saved['route_variant'], 'training_geometry': contract['geometry'],
        'training_loss_reduction': saved['loss_reduction'], 'expected_geometry': list(TRAINING_GEOMETRY),
        'tokenizer': tokens, 'assets': assets, 'all_modules_eval': True,
        'ced_all_eval': all(not m.training for m in model.audio_encoder.base.ced.modules()),
        'ced_parameters_frozen': all(not p.requires_grad for p in model.audio_encoder.base.ced.parameters()),
        'dtype': 'float32', 'deterministic_greedy': True, 'tf32': False,
        'recursive_audit': structure, 'recursive_forward_audit': trace,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device)}
    del saved, state
    return model, tokenizer, device, text_encoder, modules['data.fixed20'], audit, modules


def generate_prefix_inference(model, input_dict, *, eval_variant, audit):
    return model.generate_prefix_inference(input_dict)


def sample_context(eval_variant, audio_output, context, text_input):
    if variant_spec(eval_variant)['context_kind'] == 'none':
        if context is not None:
            raise ValueError('base route must not supply intermediate audio context')
        return {}
    return {'audio_information': context}


def validate_context(eval_variant, context, prefix):
    import torch
    if not isinstance(context, dict):
        raise ValueError('explicit per-sample context is required')
    if variant_spec(eval_variant)['context_kind'] == 'none':
        if context:
            raise ValueError('base route has no additional context')
    else:
        pair = context.get('audio_information')
        if (set(context) != {'audio_information'} or not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(not isinstance(x, torch.Tensor) or tuple(x.shape) != (1, AUDIO_TOKENS, HIDDEN_SIZE)
                    or x.device != prefix.device or x.dtype != prefix.dtype
                    or not bool(torch.isfinite(x).all()) for x in pair)):
            raise ValueError('expected this sample\'s aligned finite FP32 (A4,A8) tensors')
    return context


def greedy_decode(model, tokenizer, prefix, max_new_tokens=MAX_NEW_TOKENS, *, eval_variant, context):
    import torch

    if max_new_tokens != MAX_NEW_TOKENS:
        raise ValueError("both benchmarks use exactly max_new_tokens=300")
    if tuple(prefix.shape) != (1, PREFIX_TOKENS, HIDDEN_SIZE) or prefix.dtype != torch.float32:
        raise ValueError("CED prefix shape/dtype differs from the FP32 training layout")
    forward_kwargs = validate_context(eval_variant, context, prefix)
    generated_ids, generated = [], prefix
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        output = model.caption_decoder.lm(
            inputs_embeds=generated, attention_mask=None, labels=None, use_cache=False,
            **forward_kwargs)
        logits = output.logits[:, -1, :]
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite CED generation logits")
        next_id = int(torch.argmax(logits, dim=-1).item())
        generated_ids.append(next_id)
        del output, logits
        if next_id == int(tokenizer.eos_token_id):
            stop_reason = "eos_token"
            break
        ids = torch.tensor([[next_id]], device=prefix.device, dtype=torch.long)
        generated = torch.cat((generated, model.caption_decoder.lm.get_input_embeddings()(ids)), dim=1)
    text = tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {"generated_token_ids": generated_ids, "generated_text": text,
            "generated_text_raw": text, "generated_token_count": len(generated_ids),
            "stop_reason": stop_reason, "max_new_tokens": max_new_tokens}
