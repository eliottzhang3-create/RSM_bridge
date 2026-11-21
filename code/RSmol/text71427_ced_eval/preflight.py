"""CPU checkpoint binding checks before requesting a GPU; no model construction."""
from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path

from .registry import ROUTES, variant_spec


class CheckpointBindingError(ValueError):
    def __init__(self, problems, **context):
        self.diagnostic = {'problems': problems, **context}
        super().__init__('checkpoint binding rejected: ' + json.dumps(self.diagnostic, ensure_ascii=False))


def route_model(route):
    matches = [alias for alias, spec in ROUTES.items() if spec['route'] == route]
    return matches[0] if len(matches) == 1 else None


def load_contract(alias):
    """Load only the dependency-light contract, without the global 'contract' name."""
    path = variant_spec(alias)['route_root'] / 'contract.py'
    spec = importlib.util.spec_from_file_location('_text71427_eval_contract_' + alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_runtime(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'runtime.json must contain an object: {path}')
    return value


def validate_runtime(config, alias, contract_module, *, runtime_path=None):
    expected_route = variant_spec(alias)['route']
    contract = config.get('contract')
    problems = []
    for field, actual, expected in (
        ('implementation.ROUTE', contract_module.ROUTE, expected_route),
        ('runtime.route', config.get('route'), expected_route),
        ('runtime.mode', config.get('mode'), 'formal'),
        ('runtime.contract.route', contract.get('route') if isinstance(contract, dict) else None, expected_route),
    ):
        if actual != expected:
            problems.append({'field': field, 'expected': expected, 'actual': actual})
    if not isinstance(contract, dict) or not contract:
        problems.append({'field': 'runtime.contract', 'expected': 'nonempty object', 'actual': type(contract).__name__})
    elif config.get('contract_sha256') != contract_module.digest(contract):
        problems.append({'field': 'runtime.contract_sha256', 'expected': contract_module.digest(contract),
                         'actual': config.get('contract_sha256')})
    if problems:
        raise CheckpointBindingError(problems, requested_model=alias,
            runtime_model=route_model(config.get('route')), runtime_config=str(runtime_path) if runtime_path else None)
    return contract


def infer_model(runtime_path):
    """Exact registered route + intact formal contract, never a directory-name guess."""
    config = read_runtime(runtime_path)
    alias = route_model(config.get('route'))
    if alias is None:
        raise CheckpointBindingError([{'field': 'runtime.route', 'expected': 'one of the six registered routes',
                                      'actual': config.get('route')}], runtime_config=str(runtime_path))
    validate_runtime(config, alias, load_contract(alias), runtime_path=runtime_path)
    return alias


def validate_header(saved, config, contract_module, alias):
    expected_route = variant_spec(alias)['route']
    if not isinstance(saved, Mapping):
        raise CheckpointBindingError([{'field': 'checkpoint', 'expected': 'mapping', 'actual': type(saved).__name__}],
                                     requested_model=alias)
    problems = []
    for field, actual, expected in (
        ('implementation.ROUTE', contract_module.ROUTE, expected_route),
        ('checkpoint.schema_version', saved.get('schema_version'), 1),
        ('checkpoint.route_variant', saved.get('route_variant'), expected_route),
        ('checkpoint.mode', saved.get('mode'), 'formal'),
        ('checkpoint.contract.route', saved.get('contract', {}).get('route')
             if isinstance(saved.get('contract'), dict) else None, expected_route),
    ):
        if actual != expected:
            problems.append({'field': field, 'expected': expected, 'actual': actual})
    state = saved.get('state_dict')
    if not isinstance(state, Mapping) or not state:
        problems.append({'field': 'checkpoint.state_dict', 'expected': 'nonempty mapping', 'actual': type(state).__name__})
    if problems:
        raise CheckpointBindingError(problems, requested_model=alias,
            checkpoint_model=route_model(saved.get('route_variant')), runtime_model=route_model(config.get('route')),
            checkpoint_step=saved.get('total_step'), state_tensor_count=len(state) if isinstance(state, Mapping) else None)
    validate_runtime(config, alias, contract_module)


def check(args):
    """Metadata/contract checks only. Tensor values and inference are audited on GPU."""
    from . import runtime

    for path in (args.checkpoint, args.runtime_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in (args.checkpoint, args.runtime_config)]
    config = read_runtime(args.runtime_config)
    contract_module = load_contract(args.eval_variant)
    # Reject a wrong explicit alias before importing Torch or mapping any weights.
    validate_runtime(config, args.eval_variant, contract_module, runtime_path=args.runtime_config)
    import torch
    # These routes publish torch.save ZIP files atomically. mmap keeps optimizer
    # tensors off the login process's resident memory; no tensor values are read.
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    step = saved.get('total_step') if isinstance(saved, Mapping) else None
    validate_header(saved, config, contract_module, args.eval_variant)
    expected_step = args.expected_step if args.expected_step is not None else step
    runtime.validate_checkpoint(saved, config, contract_module, expected_step,
                                args.expected_geometry, eval_variant=args.eval_variant)
    after = [(p.stat().st_size, p.stat().st_mtime_ns) for p in (args.checkpoint, args.runtime_config)]
    if before != after:
        raise ValueError('checkpoint/runtime changed during CPU preflight')
    return {'status': 'READY_FOR_GPU', 'model': args.eval_variant,
            'training_route': saved['route_variant'], 'checkpoint': str(args.checkpoint),
            'runtime_config': str(args.runtime_config), 'checkpoint_step': step,
            'contract_sha256': saved['contract_sha256'], 'state_tensor_count': len(saved['state_dict']),
            'scope': 'CPU metadata/contracts/source identity; tensor values, assets and inference pending GPU'}
