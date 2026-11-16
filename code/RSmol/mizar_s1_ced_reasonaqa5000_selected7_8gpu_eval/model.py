"""Frozen target arguments and identity; no model or Torch import at parse time."""
from __future__ import annotations

import argparse
from pathlib import Path

from mizar_s1_ced_reasonaqa5000_eval import storage
from mizar_s1_ced_reasonaqa5000_eval.evaluate import file_identity, summarize
from . import config, runtime


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=config.target_name)
    parser.add_argument('--manifest', type=Path, default=config.DEFAULT_MANIFEST)
    parser.add_argument('--test-json', type=Path, help='relocated source test.json must have the same SHA')
    parser.add_argument('--audiocaps-root', type=Path, default=config.DATA / 'audiocaps_v2')
    parser.add_argument('--clotho-root', type=Path, default=config.DATA / 'clotho_v2_1')
    parser.add_argument('--clotho-aqa-root', type=Path, default=config.DATA / 'clotho_aqa_audio/audio_files')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--list-targets', action='store_true')
    parser.add_argument('--check-args', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    if args.list_targets:
        return args
    if not args.target:
        parser.error('--target is required')
    spec = config.TARGETS[args.target]
    for key in ('eval_variant', 'route_root', 'run_root', 'expected_step', 'expected_geometry'):
        setattr(args, key, spec[key])
    args.checkpoint = args.run_root / f'formal/checkpoints/model--step-{args.expected_step}.ckpt'
    args.runtime_config = args.run_root / 'formal/runtime.json'
    args.output_dir = args.output_dir or args.run_root / 'eval' / config.OUTPUT_NAME.format(step=args.expected_step)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    return args


def identity(args, records, manifest):
    sources = set(config.PACKAGE.glob('*.py'))
    sources.update(config.PACKAGE.glob('*.sh'))
    sources.add(config.PACKAGE / 'targets.json')
    sources.add(config.ROOT / 'run_reasonaqa5000_selected7_8gpu_3090.sh')
    sources.update((config.ROOT / 'mizar_s1_ced_reasonaqa5000_eval').glob('*.py'))
    sources.add(config.ROOT / 'mizar_s1_ced_reasonaqa5000_eval/targets.json')
    sources.update((config.ROOT / 'mizar_s1_ced_reasonaqa5000_8gpu_eval').glob('*.py'))
    sources.update((config.ROOT / 'scripts').glob('mizar_s1_ced_*eval_runtime.py'))
    sources.update(config.ROOT.glob('recursive_model*.py'))
    route_sources = {path.relative_to(args.route_root).as_posix(): storage.text_sha(path)
                     for path in sorted(args.route_root.rglob('*.py'))}
    if not route_sources:
        raise FileNotFoundError(f'missing training implementation: {args.route_root}')
    result = {'benchmark': 'reasonaqa_close_ended_sample5000', 'mode': 'sample5000',
              'expected_rows': len(records), 'eval_variant': args.eval_variant,
              'runtime_family': runtime.adapter(args.eval_variant)[1],
              'checkpoint': file_identity(args.checkpoint), 'runtime_config': file_identity(args.runtime_config),
              'expected_step': args.expected_step, 'expected_geometry': list(args.expected_geometry),
              'route_root': str(args.route_root), 'model_sources': route_sources,
              'evaluation_code': {path.relative_to(config.ROOT).as_posix(): storage.text_sha(path) for path in sorted(sources)},
              'manifest': file_identity(args.manifest), 'manifest_sha256': manifest['manifest_sha256'],
              'test_json': manifest['source_test_json'], 'selection_sha256': manifest['sample_records_sha256'],
              'resolved_records_sha256': storage.object_sha(records), 'protocol': runtime.PROTOCOL}
    result['identity_sha256'] = storage.object_sha(result)
    return result
