"""Evaluate exactly the immutable 5000-row manifest using the remaining trained targets."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from . import config, dataset, runtime, scoring, storage, sample


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=config.target_name)
    parser.add_argument('--manifest', type=Path, default=config.DEFAULT_MANIFEST)
    parser.add_argument('--test-json', type=Path, help='optional relocated original source; SHA must agree')
    parser.add_argument('--audiocaps-root', type=Path, default=config.DATA / 'audiocaps_v2')
    parser.add_argument('--clotho-root', type=Path, default=config.DATA / 'clotho_v2_1')
    parser.add_argument('--clotho-aqa-root', type=Path, default=config.DATA / 'clotho_aqa_audio/audio_files')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--run-root', type=Path, help='only mesh_inject48 may override its tentative run')
    parser.add_argument('--checkpoint', type=Path, help='only mesh_inject48 may override its tentative checkpoint')
    parser.add_argument('--runtime-config', type=Path, help='only mesh_inject48 may override its tentative runtime')
    parser.add_argument('--list-targets', action='store_true')
    parser.add_argument('--check-args', action='store_true')
    parser.add_argument('--preflight-only', action='store_true', help='manifest/audio resolution and candidate CPU contract; no inference')
    args = parser.parse_args(argv)
    if args.list_targets:
        return args
    if not args.target:
        parser.error('--target is required')
    spec = config.TARGETS[args.target]
    if args.target != 'mesh_inject48' and any((args.run_root, args.checkpoint, args.runtime_config)):
        parser.error('fixed target identity cannot be overridden; only mesh_inject48 accepts candidate corrections')
    run_root = args.run_root
    if args.checkpoint and not run_root:
        path = args.checkpoint
        if path.parent.name == 'checkpoints' and path.parent.parent.name == 'formal':
            run_root = path.parent.parent.parent
        elif not args.runtime_config or not args.output_dir:
            parser.error('noncanonical mesh_inject48 checkpoint needs explicit runtime-config and output-dir')
    args.run_root = run_root or spec['run_root']
    args.eval_variant = spec['eval_variant']
    args.route_root = spec['route_root']
    args.expected_step = spec['expected_step']
    args.expected_geometry = spec['expected_geometry']
    args.checkpoint = args.checkpoint or args.run_root / f'formal/checkpoints/model--step-{args.expected_step}.ckpt'
    args.runtime_config = args.runtime_config or args.run_root / 'formal/runtime.json'
    args.output_dir = args.output_dir or args.run_root / 'eval' / config.OUTPUT_NAME.format(step=args.expected_step)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    return args


def file_identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': storage.file_sha(path)}


def identity(args, records, audit):
    sources = set(Path(__file__).parent.glob('*.py'))
    sources.update(Path(__file__).parent.glob('*.sh'))
    sources.update(config.ROOT.glob('run_reasonaqa5000*5090.sh'))
    sources.add(config.PACKAGE / 'targets.json')
    for name in ('mizar_official_recursive_5_20x2_5', 'mizar_official_recursive_5_20x2_5_input_only'):
        sources.update((config.ROOT / name).rglob('*.py'))
    sources.update((config.ROOT / 'scripts').glob('mizar_s1_ced_*eval_runtime.py'))
    sources.update(config.ROOT.glob('recursive_model*.py'))
    route_sources = {p.relative_to(args.route_root).as_posix(): storage.text_sha(p)
                     for p in sorted(args.route_root.rglob('*.py'))}
    if not route_sources:
        raise FileNotFoundError(f'missing training implementation: {args.route_root}')
    result = {'benchmark': 'reasonaqa_close_ended_sample5000', 'mode': 'sample5000', 'expected_rows': len(records),
              'eval_variant': args.eval_variant, 'runtime_family': runtime.adapter(args.eval_variant)[1],
              'checkpoint': file_identity(args.checkpoint), 'runtime_config': file_identity(args.runtime_config),
              'expected_step': args.expected_step, 'expected_geometry': list(args.expected_geometry),
              'route_root': str(args.route_root), 'model_sources': route_sources,
              'evaluation_code': {p.relative_to(config.ROOT).as_posix(): storage.text_sha(p) for p in sorted(sources)},
              'manifest': file_identity(args.manifest), 'manifest_sha256': audit['manifest_sha256'],
              'test_json': audit['source_test_json'], 'selection_sha256': audit['sample_records_sha256'],
              'resolved_records_sha256': storage.object_sha(records), 'protocol': runtime.PROTOCOL}
    result['identity_sha256'] = storage.object_sha(result)
    return result


def summarize(saved):
    generations = [item['generation'] for item in saved]
    return {'generated': len(saved), 'skipped': 0,
            'prompt_truncated_rows': sum(bool(g.get('prompt_truncated')) for g in generations),
            'audio_cropped_rows': sum(any(a['cropped_samples'] > 0 for a in g.get('audio_audit', [])) for g in generations),
            'audio_padded_rows': sum(any(a['right_pad_samples'] > 0 for a in g.get('audio_audit', [])) for g in generations)}


def execute(args, report):
    audit = sample.load_manifest(args.manifest, args.test_json)
    records = dataset.prepare_records(audit['records'], args)
    run_identity = identity(args, records, audit)
    storage.claim_output(args.output_dir, run_identity)
    report.update(output_claimed=True, run_identity=run_identity, dataset_audit=audit)
    storage.write_json(args.output_dir / 'dataset_audit.json', {k: v for k, v in audit.items() if k != 'records'})
    storage.write_jsonl(args.output_dir / 'selected_records.jsonl', records)
    store = storage.GenerationStore(args.output_dir, records)
    report['resumed_rows'] = len(store.state)
    try:
        if len(store.state) < len(records):
            loaded = runtime.load_runtime(args)
            report['model_audit'] = loaded[5]
            storage.write_json(args.output_dir / 'model_audit.json', loaded[5])
            try:
                for index in range(len(store.state), len(records)):
                    generation = runtime.generate(loaded, records[index], variant=args.eval_variant)
                    store.append(index, records[index], generation)
                    if len(store.state) % 25 == 0 or len(store.state) == len(records):
                        print(json.dumps({'benchmark': 'reasonaqa_close_ended', 'generated': len(store.state),
                                          'total': len(records), 'last_id': records[index]['id']}), flush=True)
            finally:
                storage.write_json(args.output_dir / 'model_audit.json', loaded[5])
        else:
            report['model_audit'] = json.loads((args.output_dir / 'model_audit.json').read_text(encoding='utf-8'))
    finally:
        report['records'] = summarize(store.state)
        report['inference_coverage'] = {'status': 'PASS' if len(store.state) == len(records) else 'FAILED',
                                         'generated': len(store.state), 'expected_rows': len(records), 'skipped': 0}
    report['accuracy'] = scoring.score(args.output_dir, store.state, len(records))
    report['status'] = 'PASS'


def run(args):
    started = time.monotonic()
    report = {'status': 'FAILED', 'benchmark': 'reasonaqa_close_ended_sample5000', 'mode': 'sample5000',
              'protocol': runtime.PROTOCOL, 'output_claimed': False, 'records': {},
              'inference_coverage': {'status': 'FAILED'}, 'accuracy': None, 'fatal_error': None}
    try:
        with storage.output_lock(args.output_dir):
            try:
                execute(args, report)
            except Exception as exc:
                report['fatal_error'] = {'error': repr(exc), 'traceback': traceback.format_exc()}
                print(json.dumps({'event': 'REASONAQA_EVALUATION_FAILURE', **report['fatal_error']}),
                      file=sys.stderr, flush=True)
            finally:
                report['elapsed_seconds'] = time.monotonic() - started
                if report['output_claimed']:
                    storage.write_json(args.output_dir / 'evaluation_report.json', report)
    except Exception as exc:
        report['status'] = 'FAILED'
        report['fatal_error'] = {'error': repr(exc), 'traceback': traceback.format_exc()}
    return report


def main(argv=None):
    args = parse_args(argv)
    if args.list_targets:
        print(json.dumps(config.TARGETS, default=str, ensure_ascii=False, indent=2))
        return 0
    if args.preflight_only:
        manifest = sample.load_manifest(args.manifest, args.test_json)
        dataset.prepare_records(manifest['records'], args)
        candidate = None
        if args.target == 'mesh_inject48':
            from .inspect_checkpoint import inspect
            candidate = inspect(args)
        print(json.dumps({'status': 'PASS_PREFLIGHT_ONLY', 'sample_rows': config.SAMPLE_SIZE,
                          'manifest_sha256': manifest['manifest_sha256'], 'candidate': candidate}))
        return 0
    if args.check_args:
        print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}))
        return 0
    report = run(args)
    print(json.dumps({'status': report['status'], 'records': report['records'], 'accuracy': report['accuracy'],
                      'fatal_error': report['fatal_error'],
                      'report': str(args.output_dir / 'evaluation_report.json') if report['output_claimed'] else None}), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
