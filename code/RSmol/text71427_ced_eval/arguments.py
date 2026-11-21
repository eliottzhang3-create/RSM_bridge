"""Pure argument resolution: --check-args never imports Torch or reads assets."""
import argparse
import re
from pathlib import Path

from .registry import ROOT, REMOTE, ROUTES, TRAINING_GEOMETRY


def parse_args(benchmark, argv=None):
    parser = argparse.ArgumentParser(description='Explicit checkpoint evaluation with the original CED protocol')
    if benchmark == 'benchmark':
        parser.add_argument('--benchmark', required=True, choices=('mmau', 'mmar'))
    parser.add_argument('--eval-variant', required=True, choices=tuple(ROUTES))
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--runtime-config', type=Path)
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-args', action='store_true')
    parser.add_argument('--parquet-batch-size', type=int, default=8)
    if benchmark == 'reasonaqa5000':
        parser.add_argument('--manifest', type=Path, default=REMOTE / 'data/reasonaqa/test_close_ended_sample5000_seed1234_v1/manifest.json')
        parser.add_argument('--test-json', type=Path)
        parser.add_argument('--audiocaps-root', type=Path, default=REMOTE / 'data/audiocaps_v2')
        parser.add_argument('--clotho-root', type=Path, default=REMOTE / 'data/clotho_v2_1')
        parser.add_argument('--clotho-aqa-root', type=Path, default=REMOTE / 'data/clotho_aqa_audio/audio_files')
    else:
        parser.add_argument('--dataset-dir', type=Path)
        parser.add_argument('--evaluation-script', type=Path)
        if benchmark == 'mmsu':
            parser.add_argument('--mmsu-root', type=Path, default=REMOTE / 'data/MMSU')
        else:
            for name in ('metadata-json', 'parquet', 'audio-root'):
                parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if benchmark != 'benchmark':
        args.benchmark = benchmark
    args.expected_geometry = TRAINING_GEOMETRY
    args.route_root = ROUTES[args.eval_variant]['route_root']
    if args.checkpoint.suffix != '.ckpt':
        parser.error('--checkpoint must name a published .ckpt file (temporary files are rejected)')
    match = re.fullmatch(r'model--step-(\d+)\.ckpt', args.checkpoint.name)
    if match:
        filename_step = int(match[1])
        if args.expected_step is not None and args.expected_step != filename_step:
            parser.error('--expected-step disagrees with the checkpoint filename')
        args.expected_step = filename_step
    if args.expected_step is not None and args.expected_step <= 0:
        parser.error('--expected-step must be positive')
    canonical = args.checkpoint.parent.name == 'checkpoints' and args.checkpoint.parent.parent.name == 'formal'
    args.run_root = args.checkpoint.parent.parent.parent if canonical else None
    if not canonical and (args.runtime_config is None or args.output_dir is None):
        parser.error('noncanonical checkpoint requires explicit --runtime-config and --output-dir')
    args.runtime_config = args.runtime_config or args.run_root / 'formal/runtime.json'
    args.output_dir = args.output_dir or args.run_root / 'eval/text71427_ced_v2' / args.benchmark / args.checkpoint.stem
    if args.parquet_batch_size < 1:
        parser.error('--parquet-batch-size must be positive')
    if args.benchmark == 'mmsu':
        args.dataset_dir = args.dataset_dir or args.mmsu_root / 'MMSU data'
        args.evaluation_script = args.evaluation_script or args.mmsu_root / 'MMSU code/mmsu_evaluation.py'
    elif args.benchmark in ('mmau', 'mmar'):
        mmau = args.benchmark == 'mmau'
        args.dataset_dir = args.dataset_dir or REMOTE / ('data/MMAU_test_mini' if mmau else 'data/MMAR')
        args.metadata_json = args.metadata_json or args.dataset_dir / ('mmau-test-mini.json' if mmau else 'MMAR-meta.json')
        args.evaluation_script = args.evaluation_script or args.dataset_dir / ('evaluation.py' if mmau else 'code/evaluation.py')
        args.parquet = (args.parquet or args.dataset_dir / 'test_mini.parquet') if mmau else None
        args.audio_root = args.audio_root or args.dataset_dir / ('test-mini-audios' if mmau else 'mmar-audio')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    return args


def printable(args):
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if not key.startswith('_')}
