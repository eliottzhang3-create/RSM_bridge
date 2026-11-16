"""CPU contract preflight for the previously tentative MeSH4to8 binding."""
import json

from mizar_s1_ced_mmsu_eval import runtime, storage


def inspect(args):
    import torch

    selected, _, _ = runtime.adapter(args.eval_variant)
    before = (storage.file_sha(args.checkpoint), storage.file_sha(args.runtime_config))
    modules = selected.import_training_route(args.route_root)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = json.loads(args.runtime_config.read_text(encoding='utf-8'))
    contract = selected.validate_checkpoint(saved, config, modules['contract'], args.expected_step,
                                             args.expected_geometry, eval_variant=args.eval_variant)
    if before != (storage.file_sha(args.checkpoint), storage.file_sha(args.runtime_config)):
        raise ValueError('candidate checkpoint/runtime changed during CPU inspection')
    return {'status': 'PASS_CPU_CONTRACT_ONLY', 'checkpoint': str(args.checkpoint),
            'checkpoint_sha256': before[0], 'runtime_sha256': before[1],
            'contract_sha256': saved['contract_sha256'], 'training_geometry': contract['geometry'],
            'gpu_model_load_verified': False}
