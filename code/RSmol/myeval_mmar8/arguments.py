"""Parameter previews are asset-free; checkpoint identities cannot be overridden."""
import argparse
from pathlib import Path

from .registry import (REMOTE, SOURCE_CACHE, OUTPUT_NAMESPACE, ANNOTATIONS,
                       TRAINING_GEOMETRY, ROUTES, variant_spec)
from .sources import route_root


def parse_args(task, argv=None):
    parser = argparse.ArgumentParser(description='Three fixed models on MyEval8; zero48 on full MMAR')
    parser.add_argument('--eval-variant', required=True, choices=tuple(ROUTES))
    parser.add_argument('--sources-cache', type=Path, default=SOURCE_CACHE)
    parser.add_argument('--annotations', type=Path, default=ANNOTATIONS)
    parser.add_argument('--dataset-dir', type=Path, default=REMOTE / 'data/MMAR')
    parser.add_argument('--metadata-json', type=Path)
    parser.add_argument('--audio-root', type=Path)
    parser.add_argument('--evaluation-script', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-args', action='store_true')
    if task == 'benchmark':
        parser.add_argument('--benchmark', choices=('mmar',), default='mmar')
    args = parser.parse_args(argv)
    if task == 'benchmark' and args.eval_variant != 'read2_zero48':
        parser.error('full MMAR is requested only for read2_zero48')
    spec = variant_spec(args.eval_variant)
    args.benchmark = 'mmar' if task == 'benchmark' else 'myeval8'
    args.checkpoint, args.runtime_config, args.run_root = spec['checkpoint'], spec['runtime_config'], spec['run_root']
    args.expected_step, args.expected_geometry = 34626, TRAINING_GEOMETRY
    args.route_root = route_root(args.eval_variant, args.sources_cache)
    args.metadata_json = args.metadata_json or args.dataset_dir / 'MMAR-meta.json'
    args.audio_root = args.audio_root or args.dataset_dir / 'mmar-audio'
    args.evaluation_script = args.evaluation_script or args.dataset_dir / 'code/evaluation.py'
    args.output_dir = args.output_dir or args.run_root / 'eval' / OUTPUT_NAMESPACE / args.eval_variant / args.benchmark / args.checkpoint.stem
    args.parquet, args.parquet_batch_size = None, 8
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    return args


def printable(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith('_')}
