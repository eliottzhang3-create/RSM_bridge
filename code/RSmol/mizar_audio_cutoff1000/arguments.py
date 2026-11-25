"""CPU manifest creation and one-job evaluation share resolved source paths."""
import argparse
from pathlib import Path
from . import config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Full Mizar BF16: balanced1000, baseline + cutoff layers 3..28')
    parser.add_argument('--world-size', type=int, choices=(1, 8), default=1)
    parser.add_argument('--manifest', type=Path, default=config.DEFAULT_MANIFEST)
    parser.add_argument('--model-dir', type=Path, default=config.MODEL_DIR)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--mmau-metadata', type=Path, default=config.DATA / 'MMAU_test_mini/mmau-test-mini.json')
    parser.add_argument('--mmau-parquet', type=Path, default=config.DATA / 'MMAU_test_mini/test_mini.parquet')
    parser.add_argument('--mmau-audio-root', type=Path, default=config.DATA / 'MMAU_test_mini/test-mini-audios')
    parser.add_argument('--mmar-metadata', type=Path, default=config.DATA / 'MMAR/MMAR-meta.json')
    parser.add_argument('--mmar-audio-root', type=Path, default=config.DATA / 'MMAR/mmar-audio')
    parser.add_argument('--mmsu-dir', type=Path, default=config.DATA / 'MMSU/MMSU data')
    parser.add_argument('--reasonaqa-test', type=Path, default=config.DATA / 'reasonaqa/test.json')
    parser.add_argument('--audiocaps-root', type=Path, default=config.DATA / 'audiocaps_v2')
    parser.add_argument('--clotho-root', type=Path, default=config.DATA / 'clotho_v2_1')
    parser.add_argument('--compare-manifest', type=Path,
                        help='optional old manifest for overlap audit only; never affects sampling')
    parser.add_argument('--check-args', action='store_true', help='parse only, no assets/CUDA/submission')
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint or args.model_dir / config.CHECKPOINT
    suffix = '1gpu4090' if args.world_size == 1 else '8gpu3090'
    args.output_dir = args.output_dir or config.REMOTE / 'outputs/RSmol' / f'{config.SCHEMA}_{suffix}'
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    if args.manifest == args.output_dir or args.manifest.is_relative_to(args.output_dir):
        parser.error('manifest must be outside evaluation output')
    if args.compare_manifest == args.manifest:
        parser.error('compare-manifest must differ from new manifest')
    return args


def forwarded(args):
    return ['--world-size', str(args.world_size), *[item for key, value in vars(args).items()
        if isinstance(value, Path) for item in ('--'+key.replace('_', '-'), str(value))]]
