"""Only consume an existing frozen manifest; no sampling or data overrides."""
import argparse
from pathlib import Path
from mizar_layer_pair_search import config as reference
from . import config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Full 30-layer Mizar: baseline then audio+SEP cutoff at layers 3..20')
    parser.add_argument('--world-size', type=int, choices=(1, 8), default=1)
    parser.add_argument('--manifest', type=Path,
                        default=reference.DATA / 'mizar_prune_calibration100_seed1234_v1/manifest.json')
    parser.add_argument('--model-dir', type=Path, default=reference.MODEL_DIR)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-args', action='store_true')
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint or args.model_dir / reference.CHECKPOINT
    suffix = '1gpu4090' if args.world_size == 1 else '8gpu3090'
    args.output_dir = args.output_dir or reference.REMOTE / 'outputs/RSmol' / f'{config.SCHEMA}_{suffix}'
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.manifest.is_relative_to(args.output_dir):
        parser.error('manifest must be outside evaluation output')
    return args


def forwarded(args):
    return ['--world-size', str(args.world_size), *[item for key, value in vars(args).items()
        if isinstance(value, Path) for item in ('--'+key.replace('_', '-'), str(value))]]
