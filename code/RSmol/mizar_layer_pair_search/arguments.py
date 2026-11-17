"""Shared paths for standalone manifest creation and one-job evaluation."""
import argparse
from pathlib import Path

from . import config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Isolated Mizar 378-mask BF16 likelihood search')
    parser.add_argument('--model-dir', type=Path, default=config.MODEL_DIR)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--manifest', type=Path,
                        default=config.DATA / 'mizar_prune_calibration100_seed1234_v1/manifest.json')
    parser.add_argument('--output-dir', type=Path, default=config.DEFAULT_OUTPUT)
    parser.add_argument('--mmau-metadata', type=Path, default=config.DATA / 'MMAU_test_mini/mmau-test-mini.json')
    parser.add_argument('--mmau-parquet', type=Path, default=config.DATA / 'MMAU_test_mini/test_mini.parquet')
    parser.add_argument('--mmau-audio-root', type=Path, default=config.DATA / 'MMAU_test_mini/test-mini-audios')
    parser.add_argument('--mmar-metadata', type=Path, default=config.DATA / 'MMAR/MMAR-meta.json')
    parser.add_argument('--mmar-audio-root', type=Path, default=config.DATA / 'MMAR/mmar-audio')
    parser.add_argument('--mmsu-dir', type=Path, default=config.DATA / 'MMSU/MMSU data')
    parser.add_argument('--reasonaqa-test', type=Path, default=config.DATA / 'reasonaqa/test.json')
    parser.add_argument('--audiocaps-root', type=Path, default=config.DATA / 'audiocaps_v2')
    parser.add_argument('--clotho-root', type=Path, default=config.DATA / 'clotho_v2_1')
    parser.add_argument('--check-args', action='store_true', help='parse only, no assets, CUDA or submission')
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint or args.model_dir / config.CHECKPOINT
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    if args.manifest == args.output_dir or args.manifest.is_relative_to(args.output_dir):
        parser.error('--manifest must be outside the owned evaluation output directory')
    return args


def forwarded(args):
    result = []
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result.extend(['--' + key.replace('_', '-'), str(value)])
    return result
