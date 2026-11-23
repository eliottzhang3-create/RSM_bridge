import argparse
from pathlib import Path
from . import config


def parse_args(task, argv=None):
    if task not in ('cka','ablation'): raise ValueError('unknown task')
    parser=argparse.ArgumentParser(description=f'Isolated Mizar {task}: one eight-card 3090 job')
    parser.add_argument('--model-dir',type=Path,default=config.MODEL_DIR)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output-dir',type=Path,default=config.REMOTE/'outputs/RSmol'/f'mizar_{task}_bf16_8gpu3090_v1')
    default=config.FROZEN100 if task=='cka' else config.DATA/'mizar_mmau1000_likelihood_v1/manifest.json'
    parser.add_argument('--manifest',type=Path,default=default)
    if task=='cka':
        parser.add_argument('--permutation-repeats',type=int,default=100)
    else:
        parser.add_argument('--mmau-metadata',type=Path,default=config.DATA/'MMAU_test_mini/mmau-test-mini.json')
        parser.add_argument('--mmau-audio-root',type=Path,default=config.DATA/'MMAU_test_mini/test-mini-audios')
        parser.add_argument('--mmau-parquet',type=Path,default=config.DATA/'MMAU_test_mini/test_mini.parquet')
    parser.add_argument('--check-args',action='store_true')
    args=parser.parse_args(argv)
    args.checkpoint=args.checkpoint or args.model_dir/config.CHECKPOINT
    for key,value in vars(args).items():
        if isinstance(value,Path): setattr(args,key,value.expanduser().resolve())
    if args.manifest.is_relative_to(args.output_dir): parser.error('manifest must be outside owned output')
    if task=='cka' and not 1<=args.permutation_repeats<=1000: parser.error('permutation repeats must be 1..1000')
    return args


def forwarded(args):
    return [item for key,value in vars(args).items() if isinstance(value,Path) or type(value) is int
            for item in ('--'+key.replace('_','-'),str(value))]
