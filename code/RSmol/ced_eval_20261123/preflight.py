"""Strict per-target headers and CPU contracts; never construct GPU models."""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping

from .registry import variant_spec
from . import sources


def validate_header(saved, runtime, route_contract, *, target):
    spec = variant_spec(target)
    if not isinstance(saved, Mapping):
        raise ValueError('checkpoint must be a mapping')
    problems = []
    for field,actual,wanted in (
        ('implementation.ROUTE',route_contract.ROUTE,spec['route']),
        ('checkpoint.route_variant',saved.get('route_variant'),spec['route']),
        ('checkpoint.schema_version',saved.get('schema_version'),1),
        ('checkpoint.mode',saved.get('mode'),'formal'),
        ('runtime.route',runtime.get('route'),spec['route']),
        ('runtime.mode',runtime.get('mode'),'formal'),
    ):
        if actual!=wanted: problems.append({'field':field,'expected':wanted,'actual':actual})
    if not isinstance(saved.get('state_dict'),Mapping) or not saved['state_dict']:
        problems.append({'field':'checkpoint.state_dict','expected':'nonempty complete mapping'})
    if problems:
        raise ValueError('target binding mismatch: '+json.dumps({'target':target,'problems':problems}))


def check(args):
    from . import runtime
    for path in (args.checkpoint,args.runtime_config):
        if not path.is_file(): raise FileNotFoundError(path)
    before=[(p.stat().st_size,p.stat().st_mtime_ns) for p in (args.checkpoint,args.runtime_config)]
    root=sources.prepare(args.eval_variant,args.sources_cache)
    if root!=args.route_root:raise ValueError('source root differs from pinned target')
    from .interfaces import audit_interfaces
    interfaces=audit_interfaces(args.eval_variant,root)
    contract_module=sources.load_contract(args.eval_variant,args.sources_cache)
    config=json.loads(args.runtime_config.read_text(encoding='utf-8'))
    if config.get('route')!=variant_spec(args.eval_variant)['route'] or config.get('mode')!='formal':
        raise ValueError(f'runtime route/mode does not match requested target {args.eval_variant}')
    import torch
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    step=saved.get('total_step') if isinstance(saved,Mapping) else None
    expected=args.expected_step if args.expected_step is not None else step
    runtime.validate_checkpoint(saved,config,contract_module,expected,args.expected_geometry,eval_variant=args.eval_variant)
    if before!=[(p.stat().st_size,p.stat().st_mtime_ns) for p in (args.checkpoint,args.runtime_config)]:
        raise ValueError('checkpoint/runtime changed during CPU preflight')
    return {'status':'READY_FOR_GPU','target':args.eval_variant,'checkpoint':str(args.checkpoint),
        'checkpoint_step':step,'training_route':saved['route_variant'],'contract_sha256':saved['contract_sha256'],
        'training_code_sha256':saved['contract']['code_sha256'],'source_revision':variant_spec(args.eval_variant)['revision'],
        'training_source_root':str(root),'state_tensor_count':len(saved['state_dict']),
        'implementation_interfaces':interfaces,
        'scope':'CPU header/contract/pinned source; tensor values/assets/logits pending GPU'}
