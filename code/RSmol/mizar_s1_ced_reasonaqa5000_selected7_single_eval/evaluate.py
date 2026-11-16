"""Evaluate all 5000 rows in one process with the original strict model loaders."""
from __future__ import annotations

import json
import sys
import time
import traceback

from . import config
from mizar_s1_ced_reasonaqa5000_selected7_8gpu_eval import model, runtime
from mizar_s1_ced_reasonaqa5000_eval import dataset, sample, scoring, storage


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = model.parse_args(argv)
    if not args.list_targets and not any(value == '--output-dir' or value.startswith('--output-dir=') for value in argv):
        args.output_dir = args.run_root / 'eval' / config.OUTPUT_NAME.format(step=args.expected_step)
    return args


def identity(args, records, manifest):
    result = model.identity(args, records, manifest)
    sources = [*config.PACKAGE.glob('*.py'), *config.PACKAGE.glob('*.sh'),
               config.ROOT / 'run_reasonaqa5000_selected7_1gpu_4090.sh']
    result['evaluation_code'].update({path.relative_to(config.ROOT).as_posix(): storage.text_sha(path)
                                      for path in sources})
    result['execution'] = config.EXECUTION
    result['identity_sha256'] = storage.object_sha({key: value for key, value in result.items() if key != 'identity_sha256'})
    return result


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
        report['records'] = model.summarize(store.state)
        report['inference_coverage'] = {'status': 'PASS' if len(store.state) == len(records) else 'FAILED',
                                         'generated': len(store.state), 'expected_rows': len(records), 'skipped': 0}
    report['accuracy'] = scoring.score(args.output_dir, store.state, len(records))
    report['status'] = 'PASS'


def run(args):
    started = time.monotonic()
    report = {'status': 'FAILED', 'benchmark': 'reasonaqa_close_ended_sample5000', 'mode': 'sample5000',
              'protocol': runtime.PROTOCOL, 'execution': config.EXECUTION,
              'output_claimed': False, 'records': {},
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
    if args.check_args:
        print(json.dumps({**vars(args), 'execution': config.EXECUTION}, default=str, ensure_ascii=False))
        return 0
    if args.preflight_only:
        manifest = sample.load_manifest(args.manifest, args.test_json)
        dataset.prepare_records(manifest['records'], args)
        print(json.dumps({'status': 'PASS_PREFLIGHT_ONLY', 'sample_rows': config.SAMPLE_SIZE,
                          'manifest_sha256': manifest['manifest_sha256'], 'execution': config.EXECUTION}))
        return 0
    report = run(args)
    print(json.dumps({'status': report['status'], 'records': report['records'], 'accuracy': report['accuracy'],
                      'fatal_error': report['fatal_error'], 'output_dir': str(args.output_dir)}), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
