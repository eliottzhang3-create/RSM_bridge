"""Verify the user's tentative mesh_inject48 binding on CPU before submission."""
from __future__ import annotations

import json

from . import config, runtime, storage


def inspect(args):
    import torch

    selected, family = runtime.adapter(args.eval_variant)
    before = (storage.file_sha(args.checkpoint), storage.file_sha(args.runtime_config))
    modules = selected.import_training_route(args.route_root)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    runtime_config = json.loads(args.runtime_config.read_text(encoding='utf-8'))
    kwargs = {'eval_variant': args.eval_variant} if family in ('recursive', 'multilayer') else {}
    contract = selected.validate_checkpoint(saved, runtime_config, modules['contract'],
                                             args.expected_step, args.expected_geometry, **kwargs)
    if before != (storage.file_sha(args.checkpoint), storage.file_sha(args.runtime_config)):
        raise ValueError('checkpoint/runtime changed during CPU preflight')
    if not isinstance(saved.get('state_dict'), dict) or not saved['state_dict']:
        raise ValueError('candidate checkpoint has no full state_dict')
    return {'status': 'PASS_CPU_CONTRACT_ONLY', 'target': args.target, 'eval_variant': args.eval_variant,
            'checkpoint': str(args.checkpoint), 'runtime_config': str(args.runtime_config),
            'checkpoint_sha256': before[0], 'runtime_sha256': before[1],
            'step': saved['total_step'], 'contract_sha256': saved['contract_sha256'],
            'training_geometry': contract['geometry'], 'gpu_model_load_verified': False}


def main(argv=None):
    from .evaluate import parse_args
    args = parse_args(argv)
    print(json.dumps(inspect(args), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
