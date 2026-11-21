"""Atomic full-state checkpoints, rank RNG/cursors, strict optimizer and EMA restore."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import shutil
import tempfile

from .contract import (ROUTE, ARCHITECTURE, FORMAT, RECIPE, write_json,
                       file_inventory, canonical_hash)
from .publication import atomic_publish


def capture_rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA RNG cannot be restored without CUDA')
        torch.cuda.set_rng_state(state['cuda'])


def optimizer_names(model, optimizer):
    by_id = {id(p): n for n, p in model.named_parameters()}
    names = [[by_id[id(p)] for p in group['params']] for group in optimizer.param_groups]
    flat = [n for group in names for n in group]
    if len(flat) != len(set(flat)) or set(flat) != set(by_id.values()):
        raise ValueError('optimizer must contain every unique trainable parameter exactly once')
    return names


def validate_optimizer(model, optimizer, saved, names, step):
    import torch
    from .contract import learning_rate
    if names != optimizer_names(model, optimizer) or len(saved['param_groups']) != 1:
        raise ValueError('optimizer named inventory/order mismatch')
    group = saved['param_groups'][0]
    lr = float(group['lr'])
    if tuple(group['betas']) != (0.9, 0.95) or group['weight_decay'] != 0.1 or group['ema'] != 0.9999 or lr != learning_rate(step):
        raise ValueError('optimizer recipe/LR mismatch')
    if set(saved['state']) != set(group['params']):
        raise ValueError('optimizer state inventory mismatch')
    for param, key in zip(optimizer.param_groups[0]['params'], group['params']):
        state = saved['state'][key]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq', 'param_ema'}:
            raise ValueError('incomplete Adam-atan2/EMA state')
        if not torch.is_tensor(state['step']) or state['step'].numel() != 1 or float(state['step']) != step:
            raise ValueError('optimizer parameter step mismatch')
        for name in ('exp_avg', 'exp_avg_sq', 'param_ema'):
            value = state[name]
            if value.shape != param.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f'invalid optimizer {name}')
        if (state['exp_avg_sq'] < 0).any():
            raise ValueError('negative second moment')


def ema_state_dict(model, optimizer):
    tensors = {id(p): optimizer.state[p]['param_ema'].detach().cpu()
               for p in model.parameters()}
    aliases = dict(model.named_parameters(remove_duplicate=False))
    return {name: tensors[id(aliases[name])] if name in aliases else value.detach().cpu()
            for name, value in model.state_dict().items()}


def save_checkpoint(run, model, optimizer, tokenizer, *, step, binding, rank_states,
                    production=True, retain=True):
    import torch
    from .contract import learning_rate, bp_steps
    run = Path(run)
    if not 1 <= step <= RECIPE['total_steps']:
        raise ValueError('invalid save step')
    if set(rank_states) != {str(i) for i in range(binding['world_size'])}:
        raise ValueError('missing rank state')
    for rank, state in rank_states.items():
        if state['cursor']['rank'] != int(rank) or state['cursor']['microbatches'] != step * binding['accumulation_steps']:
            raise ValueError('rank data cursor does not match optimizer boundary')
    names = optimizer_names(model, optimizer)
    optim_state = optimizer.state_dict()
    validate_optimizer(model, optimizer, optim_state, names, step)
    final = run / f'checkpoint-step-{step:06d}'
    staging = Path(tempfile.mkdtemp(prefix='.' + final.name + '.staging-', dir=run))
    # Preserve RNG against any present/future serialization-side consumption.
    rng = capture_rng()
    try:
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        ema_dir = staging / 'ema'
        model.save_pretrained(ema_dir, state_dict=ema_state_dict(model, optimizer), safe_serialization=True)
        tokenizer.save_pretrained(ema_dir)
        write_json(ema_dir / 'weights_identity.json', dict(route=ROUTE, weights_kind='ema',
            decay=0.9999, step=step, full_resume=False, binding_sha256=canonical_hash(binding)))
        payload = dict(format=FORMAT, route=ROUTE, architecture=ARCHITECTURE, step=step,
            binding=binding, binding_sha256=canonical_hash(binding), optimizer=optim_state,
            optimizer_names=names, rank_states=rank_states,
            schedules=dict(last_lr=learning_rate(step), last_bp=bp_steps(step),
                next_lr=learning_rate(step + 1) if step < 9244 else None,
                next_bp=bp_steps(step + 1) if step < 9244 else None),
            raw_weights_kind='raw', ema_weights_kind='ema')
        write_json(staging / 'binding.json', binding)
        torch.save(payload, staging / 'training_state.pt')
        write_json(staging / 'checkpoint_complete.json', dict(complete=True, route=ROUTE,
            architecture=ARCHITECTURE, format=FORMAT, production=production,
            step=step, binding_sha256=canonical_hash(binding), world_size=binding['world_size'],
            rank_states=sorted(rank_states), files=file_inventory(staging, include_state=True)))
        atomic_publish(staging, final)
        tmp = run / '.latest_complete_checkpoint.tmp'
        write_json(tmp, dict(step=step, path=str(final), complete=True))
        os.replace(tmp, run / 'latest_complete_checkpoint.json')
        if retain:
            retain_checkpoints(run)
    finally:
        restore_rng(rng)
    return final


def retain_checkpoints(run):
    """Only remove verified complete older children of this exact run directory."""
    from .conversion import inspect_artifact
    root = Path(run).resolve()
    checkpoints = []
    for path in root.glob('checkpoint-step-*'):
        if path.is_symlink() or not path.is_dir() or path.resolve().parent != root:
            raise ValueError('unsafe checkpoint retention path')
        raw = json.loads((path / 'checkpoint_complete.json').read_text(encoding='utf-8'))
        marker = inspect_artifact(path, resume=True, production=raw.get('production'))
        checkpoints.append((marker['step'], path))
    for _, path in sorted(checkpoints)[:-RECIPE['retention']]:
        shutil.rmtree(path)


def restore_checkpoint(path, model, optimizer, *, binding, rank, production=True):
    import torch
    from .conversion import inspect_artifact
    from .contract import learning_rate, bp_steps
    marker = inspect_artifact(path, resume=True, production=production)
    # Inventory SHA verified before deserializing this same-route private training state.
    payload = torch.load(Path(path) / 'training_state.pt', map_location='cpu', weights_only=False)
    step = marker['step']
    if marker.get('world_size') != binding['world_size']:
        raise ValueError('checkpoint world-size mismatch')
    expected_sha = canonical_hash(binding)
    if payload.get('format') != FORMAT or payload.get('route') != ROUTE or payload.get('architecture') != ARCHITECTURE or payload.get('step') != step:
        raise ValueError('training payload identity mismatch')
    if payload.get('binding') != binding or payload.get('binding_sha256') != expected_sha or marker.get('binding_sha256') != expected_sha:
        raise ValueError('resume contract/code/data/environment/initialization changed')
    ranks = payload.get('rank_states', {})
    if set(ranks) != {str(i) for i in range(binding['world_size'])} or marker.get('rank_states') != sorted(ranks):
        raise ValueError('incomplete rank RNG/data state')
    for key, state in ranks.items():
        if set(state.get('rng', {})) != {'python', 'numpy', 'cpu', 'cuda'} or state.get('cursor', {}).get('rank') != int(key) or state['cursor'].get('microbatches') != step * binding['accumulation_steps']:
            raise ValueError('invalid rank RNG/data cursor')
    expected_schedules = dict(last_lr=learning_rate(step), last_bp=bp_steps(step),
        next_lr=learning_rate(step + 1) if step < 9244 else None,
        next_bp=bp_steps(step + 1) if step < 9244 else None)
    if payload.get('schedules') != expected_schedules:
        raise ValueError('LR/BP schedule cursor mismatch')
    validate_optimizer(model, optimizer, payload['optimizer'], payload['optimizer_names'], step)
    optimizer.load_state_dict(payload['optimizer'])
    if step == 9244:
        raise ValueError('training already complete; use model weights for inference/initialization')
    restore_rng(ranks[str(rank)]['rng'])
    return step, ranks[str(rank)]['cursor']
