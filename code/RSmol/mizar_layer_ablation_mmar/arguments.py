"""The same resolved inputs reach manifest creation, coordinator and workers."""
import argparse
from pathlib import Path
from . import config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Full Mizar: MMAR1000 baseline + single-layer ablation 2..29, 8x5090')
    parser.add_argument('--model-dir', type=Path, default=config.MODEL_DIR)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--manifest', type=Path, default=config.DATA/'mizar_mmar1000_likelihood_v1/manifest.json')
    parser.add_argument('--output-dir', type=Path, default=config.REMOTE/'outputs/RSmol'/config.SCHEMA)
    parser.add_argument('--mmar-metadata', type=Path, default=config.DATA/'MMAR/MMAR-meta.json')
    parser.add_argument('--mmar-audio-root', type=Path, default=config.DATA/'MMAR/mmar-audio')
    parser.add_argument('--evaluation-script', type=Path, default=config.DATA/'MMAR/code/evaluation.py')
    parser.add_argument('--check-args', action='store_true', help='parse only; no production assets, CUDA or vc')
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint or args.model_dir/config.CHECKPOINT
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    if args.manifest == args.output_dir or args.manifest.is_relative_to(args.output_dir):
        parser.error('manifest must be outside evaluation output')
    return args


def forwarded(args):
    return [item for key, value in vars(args).items() if isinstance(value, Path)
            for item in ('--'+key.replace('_', '-'), str(value))]
