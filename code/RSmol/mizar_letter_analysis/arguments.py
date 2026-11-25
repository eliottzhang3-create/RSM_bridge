"""Resolve separate source/letter manifests and outputs for the three reruns."""
import argparse
from pathlib import Path
from . import config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Mizar lowercase-letter likelihood + EOT, one 8x5090 job per task')
    parser.add_argument('--task', required=True, choices=config.TASKS)
    parser.add_argument('--model-dir', type=Path, default=config.MODEL_DIR)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output-dir', type=Path)
    defaults = {
        'mmau-metadata': 'MMAU_test_mini/mmau-test-mini.json', 'mmau-parquet': 'MMAU_test_mini/test_mini.parquet',
        'mmau-audio-root': 'MMAU_test_mini/test-mini-audios', 'mmar-metadata': 'MMAR/MMAR-meta.json',
        'mmar-audio-root': 'MMAR/mmar-audio', 'mmsu-dir': 'MMSU/MMSU data',
        'reasonaqa-test': 'reasonaqa/test.json', 'audiocaps-root': 'audiocaps_v2', 'clotho-root': 'clotho_v2_1',
        'evaluation-script': 'MMAR/code/evaluation.py',
    }
    for name, path in defaults.items():
        parser.add_argument('--'+name, type=Path, default=config.DATA/path,
                            help='source creation only when the old manifest is absent')
    parser.add_argument('--check-args', action='store_true', help='parse only; no model/assets/CUDA/vc')
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint or args.model_dir/config.CHECKPOINT
    args.source_manifest = args.source_manifest or config.SOURCE_MANIFESTS[args.task]
    args.manifest = args.manifest or config.DATA/f'mizar_{args.task}1000_letters_eot_v1/manifest.json'
    args.output_dir = args.output_dir or config.REMOTE/'outputs/RSmol'/config.RUN_NAMES[args.task]
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.source_manifest == args.manifest:
        parser.error('new letter manifest must differ from the original body-score manifest')
    if any(p == args.output_dir or p.is_relative_to(args.output_dir) for p in (args.source_manifest, args.manifest)):
        parser.error('both manifests must be outside evaluation output')
    if args.manifest.parent == args.source_manifest.parent:
        parser.error('letter manifest requires its own directory')
    return args


def forwarded(args):
    return ['--task', args.task, *[item for key, value in vars(args).items() if isinstance(value, Path)
                                  for item in ('--'+key.replace('_', '-'), str(value))]]
