"""Evaluate exactly the immutable 5000-row manifest using the remaining trained targets."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from mizar_s1_ced_reasonaqa5000_eval import config, dataset, scoring, storage, sample
from . import reasonaqa_runtime as runtime
from . import arguments, runtime as model_runtime


def parse_args(argv=None):
    return arguments.parse_args('reasonaqa5000', argv)



def file_identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': storage.file_sha(path)}


def identity(args, records, audit):
    model_runtime.read_checkpoint(args)
    route_sources = {p.relative_to(args.route_root).as_posix(): storage.text_sha(p)
                     for p in sorted(args.route_root.rglob('*.py'))}
    if not route_sources:
        raise FileNotFoundError(f'missing training implementation: {args.route_root}')
    result = {'benchmark': 'reasonaqa_close_ended_sample5000', 'mode': 'sample5000', 'expected_rows': len(records),
              'eval_variant': args.eval_variant, 'runtime_family': runtime.adapter(args.eval_variant)[1],
              'training_route': model_runtime.variant_spec(args.eval_variant)['route'],
              'source_revision': model_runtime.variant_spec(args.eval_variant)['revision'],
              'model_dir': str(args.model_dir), 'checkpoint': file_identity(args.checkpoint), 'runtime_config': file_identity(args.runtime_config) if args.runtime_config else None,
              'expected_step': args.expected_step, 'expected_geometry': list(args.expected_geometry),
              'route_root': str(args.route_root), 'model_sources': route_sources,
              'evaluation_code': model_runtime.evaluation_sources(),
              'manifest': file_identity(args.manifest), 'manifest_sha256': audit['manifest_sha256'],
              'test_json': audit['source_test_json'], 'selection_sha256': audit['sample_records_sha256'],
              'resolved_records_sha256': storage.object_sha(records), 'protocol': runtime.protocol_for(args.eval_variant)}
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
              'protocol': runtime.protocol_for(args.eval_variant), 'output_claimed': False, 'records': {},
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
    if args.check_args:
        print(json.dumps(arguments.printable(args), default=str))
        return 0
    try:
        report = run(args)
        print(json.dumps({'status': report['status'], 'records': report['records'],
                          'accuracy': report['accuracy'], 'fatal_error': report['fatal_error'],
                          'report': str(args.output_dir / 'evaluation_report.json')
                          if report['output_claimed'] else None}), flush=True)
        return 0 if report['status'] == 'PASS' else 1
    finally:
        vars(args).pop('_checkpoint_payload', None)



if __name__ == '__main__':
    raise SystemExit(main())
