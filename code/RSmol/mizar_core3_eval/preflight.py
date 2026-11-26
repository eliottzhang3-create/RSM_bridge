"""CPU-only admission, once per explicit checkpoint; no model construction."""
from . import runtime,sources


def check(args):
    runtime.read_checkpoint(args)
    if args.eval_variant == 'official_mizar':
        return dict(status='READY_FOR_GPU',target=args.eval_variant,checkpoint=str(args.checkpoint),
            official_release=args._bundle,scope='CPU complete release SHA; CUDA logits pending')
    saved = args._checkpoint_payload
    c = saved['contract']
    sources.verify(args.eval_variant,args.route_root)
    result = dict(status='READY_FOR_GPU',target=args.eval_variant,checkpoint=str(args.checkpoint),
        checkpoint_step=saved['total_step'],total_steps=c['geometry']['total_steps'],
        training_code_sha256=c['code_sha256'],contract_sha256=saved['contract_sha256'],
        scope='CPU header/runtime/source; tensor values and CUDA pending')
    vars(args).pop('_checkpoint_payload',None)
    return result
