"""Run eight independent replicas, durably merge exactly 5000 original samples."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from mizar_s1_ced_reasonaqa5000_eval import (
    dataset, sample, scoring, storage,
)
from . import config, runtime, model as single


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = single.parse_args(argv)
    if not args.list_targets and not any(value == '--output-dir' or value.startswith('--output-dir=') for value in argv):
        args.output_dir = args.run_root / 'eval' / config.OUTPUT_NAME.format(step=args.expected_step)
    return args


def identity(args, records, manifest):
    result = single.identity(args, records, manifest)
    sources = [*config.PACKAGE.glob('*.py'), *config.PACKAGE.glob('*.sh'),
               config.ROOT / 'run_reasonaqa5000_selected7_8gpu_3090.sh']
    result['evaluation_code'].update({path.relative_to(config.ROOT).as_posix(): storage.text_sha(path)
                                      for path in sources})
    result['parallel_evaluation'] = config.PARALLEL
    result['identity_sha256'] = storage.object_sha({key: value for key, value in result.items() if key != 'identity_sha256'})
    return result


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path):
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError(f'incomplete JSONL: {path}')
    return [json.loads(line) for line in raw.splitlines()]


def verify_sources(args, expected):
    body = {key: value for key, value in expected.items() if key != 'identity_sha256'}
    if (expected.get('identity_sha256') != storage.object_sha(body)
            or expected.get('parallel_evaluation') != config.PARALLEL
            or expected.get('expected_rows') != config.SAMPLE_SIZE
            or expected.get('eval_variant') != args.eval_variant
            or expected.get('route_root') != str(args.route_root)
            or expected.get('expected_step') != args.expected_step
            or expected.get('expected_geometry') != list(args.expected_geometry)):
        raise ValueError('eight-GPU evaluation identity/target mismatch')
    for key in ('checkpoint', 'runtime_config', 'manifest'):
        if single.file_identity(getattr(args, key)) != expected[key]:
            raise ValueError(f'{key} changed since coordinator preflight')
    source = args.test_json or Path(expected['test_json']['path'])
    if (source.stat().st_size != expected['test_json']['bytes']
            or storage.file_sha(source) != expected['test_json']['sha256']):
        raise ValueError('source test.json changed since coordinator preflight')
    for name, sha in expected['evaluation_code'].items():
        if storage.text_sha(config.ROOT / name) != sha:
            raise ValueError(f'evaluation source changed: {name}')
    for name, sha in expected['model_sources'].items():
        if storage.text_sha(args.route_root / name) != sha:
            raise ValueError(f'model source changed: {name}')


def verify_records(records, expected):
    originals = [{key: value for key, value in row.items() if key not in ('audio1', 'audio2')} for row in records]
    if (len(records) != config.SAMPLE_SIZE or len({row['id'] for row in records}) != config.SAMPLE_SIZE
            or storage.object_sha(records) != expected['resolved_records_sha256']
            or storage.object_sha(originals) != expected['selection_sha256']):
        raise ValueError('coordinator selected records differ from the fixed 5000-row identity')
    for record in records:
        for source in (record['audio1'], record['audio2']):
            if source:
                stat = Path(source['path']).stat()
                if (stat.st_size, stat.st_mtime_ns) != (source['bytes'], source['mtime_ns']):
                    raise ValueError(f'audio changed since coordinator preflight: {source["path"]}')


def rank_identity(expected, rank, records):
    indices = config.rank_indices(rank)
    return {'benchmark': 'reasonaqa_sample5000_rank', 'root_identity_sha256': expected['identity_sha256'],
            'rank': rank, 'world_size': config.WORLD_SIZE, 'global_indices': indices,
            'expected_rows': len(indices), 'records_sha256': storage.object_sha([records[index] for index in indices])}


def run_rank(args, rank, binding):
    directory = config.rank_dir(args.output_dir, rank)
    report = {'status': 'FAILED', 'rank': rank, 'output_claimed': False, 'fatal_error': None,
              'binding': binding, 'inference_coverage': {'status': 'FAILED', 'generated': 0,
                                                       'expected_rows': config.ROWS_PER_RANK, 'skipped': 0}}
    started = time.monotonic()
    try:
        expected = read_json(args.output_dir / 'run_config.json')
        verify_sources(args, expected)
        records = read_jsonl(args.output_dir / 'selected_records.jsonl')
        verify_records(records, expected)
        if (binding.get('rank') != rank or binding.get('local_rank') != rank
                or binding.get('world_size') != config.WORLD_SIZE or binding.get('logical_device') != 'cuda:0'):
            raise ValueError('rank GPU binding mismatch')
        selected = [records[index] for index in config.rank_indices(rank)]
        selected_identity = rank_identity(expected, rank, records)
        with storage.output_lock(directory):
            storage.claim_output(directory, selected_identity)
            report.update(output_claimed=True, run_identity=selected_identity)
            try:
                store = storage.GenerationStore(directory, selected)
                report['resumed_rows'] = len(store.state)
                try:
                    if len(store.state) < len(selected):
                        loaded = runtime.load_runtime(args)
                        report['model_audit'] = loaded[5]
                        storage.write_json(directory / 'model_audit.json', loaded[5])
                        try:
                            for index in range(len(store.state), len(selected)):
                                generated = runtime.generate(loaded, selected[index], variant=args.eval_variant)
                                store.append(index, selected[index], generated)
                                if len(store.state) % 25 == 0 or len(store.state) == len(selected):
                                    print(json.dumps({'rank': rank, 'generated': len(store.state), 'total': len(selected),
                                                      'global_index': config.rank_indices(rank)[index]}), flush=True)
                        finally:
                            storage.write_json(directory / 'model_audit.json', loaded[5])
                    else:
                        report['model_audit'] = read_json(directory / 'model_audit.json')
                finally:
                    report['inference_coverage'].update(generated=len(store.state),
                        status='PASS' if len(store.state) == len(selected) else 'FAILED')
                report['status'] = 'PASS'
            except Exception as exc:
                report['fatal_error'] = {'error': repr(exc), 'traceback': traceback.format_exc()}
            finally:
                report['elapsed_seconds'] = time.monotonic() - started
                storage.write_json(directory / 'rank_report.json', report)
    except Exception as exc:
        report['status'] = 'FAILED'
        report['fatal_error'] = {'error': repr(exc), 'traceback': traceback.format_exc()}
    print(json.dumps({'rank': rank, 'status': report['status'], 'coverage': report['inference_coverage'],
                      'fatal_error': report['fatal_error']}), flush=True)
    return report


def read_rank(output, rank, records, expected):
    directory = config.rank_dir(output, rank)
    selected_identity = rank_identity(expected, rank, records)
    report = read_json(directory / 'rank_report.json')
    coverage = {'status': 'PASS', 'generated': config.ROWS_PER_RANK,
                'expected_rows': config.ROWS_PER_RANK, 'skipped': 0}
    if (read_json(directory / 'run_config.json') != selected_identity or report.get('run_identity') != selected_identity
            or report.get('rank') != rank or report.get('status') != 'PASS'
            or report.get('inference_coverage') != coverage
            or read_json(directory / 'model_audit.json') != report.get('model_audit')):
        raise ValueError(f'incomplete/inconsistent rank report: {rank}')
    binding = report.get('binding', {})
    if (binding.get('rank') != rank or binding.get('local_rank') != rank
            or binding.get('world_size') != config.WORLD_SIZE or binding.get('logical_device') != 'cuda:0'
            or not binding.get('worker_cuda_visible_devices')):
        raise ValueError(f'invalid rank GPU audit: {rank}')
    saved = read_jsonl(directory / 'raw_generations.jsonl')
    if len(saved) != config.ROWS_PER_RANK:
        raise ValueError(f'rank {rank} requires exactly 625 raw generations')
    for local_index, item in enumerate(saved):
        record = records[selected_identity['global_indices'][local_index]]
        if (item.get('row_index') != local_index or item.get('record') != record
                or item.get('record_sha256') != storage.object_sha(record)
                or item.get('generation_sha256') != storage.object_sha(item.get('generation'))):
            raise ValueError(f'rank {rank} raw identity/hash mismatch at {local_index}')
        storage.validate_generation(item['generation'], record)
    return saved, report


def all_ready(output, records, expected):
    for rank in range(config.WORLD_SIZE):
        directory = config.rank_dir(output, rank)
        if not (directory / 'rank_report.json').is_file():
            return False
        if read_json(directory / 'rank_report.json').get('status') != 'PASS':
            return False
        read_rank(output, rank, records, expected)
    return True


def merge(output, records, expected):
    merged, rank_reports = [None] * config.SAMPLE_SIZE, []
    for rank in range(config.WORLD_SIZE):
        saved, report = read_rank(output, rank, records, expected)
        for global_index, item in zip(config.rank_indices(rank), saved):
            if merged[global_index] is not None:
                raise ValueError(f'duplicate global sample index: {global_index}')
            merged[global_index] = {**item, 'row_index': global_index}
        rank_reports.append(report)
    if (any(item is None for item in merged) or len({item['record']['id'] for item in merged}) != config.SAMPLE_SIZE
            or [item['record'] for item in merged] != records):
        raise ValueError('merged output must cover the exact 5000 manifest records once each')
    devices = [report['binding']['worker_cuda_visible_devices'] for report in rank_reports]
    inherited = [report['binding']['inherited_cuda_visible_devices'] for report in rank_reports]
    if len(set(devices)) != config.WORLD_SIZE or len(set(inherited)) != 1:
        raise ValueError('eight ranks did not use distinct GPUs from one inherited visibility map')
    storage.write_jsonl(output / 'raw_generations.jsonl', merged)
    storage.write_jsonl(output / 'progress.jsonl',
        ({'row_index': item['row_index'], 'id': item['record']['id'], 'status': 'generated'} for item in merged))
    model_audit = {'status': 'PASS', 'parallel_evaluation': config.PARALLEL,
                   'ranks': [{'rank': report['rank'], 'binding': report['binding'], 'audit': report['model_audit']}
                             for report in rank_reports]}
    storage.write_json(output / 'model_audit.json', model_audit)
    return merged, rank_reports, model_audit


def worker_argv(args):
    result = ['--target', args.target]
    for key in ('manifest', 'test_json', 'audiocaps_root', 'clotho_root', 'clotho_aqa_root', 'output_dir'):
        value = getattr(args, key)
        if value is not None:
            result.extend(['--' + key.replace('_', '-'), str(value)])
    return result


def launch_workers(args):
    command = [sys.executable, '-u', '-B', '-m', 'torch.distributed.run', '--standalone',
               '--nnodes=1', '--nproc-per-node=8', '--max-restarts=0', '--module',
               'mizar_s1_ced_reasonaqa5000_selected7_8gpu_eval.worker', *worker_argv(args)]
    print(json.dumps({'event': 'LAUNCH_EIGHT_GPU_WORKERS', 'argv': command}), flush=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    result = subprocess.run(command, cwd=config.ROOT, env=environment)
    if result.returncode:
        raise RuntimeError(f'eight-GPU workers failed: torchrun exit {result.returncode}; see rank reports')


def execute(args, report):
    manifest = sample.load_manifest(args.manifest, args.test_json)
    records = dataset.prepare_records(manifest['records'], args)
    expected = identity(args, records, manifest)
    storage.claim_output(args.output_dir, expected)
    report.update(output_claimed=True, run_identity=expected,
                  dataset_audit={key: value for key, value in manifest.items() if key != 'records'})
    storage.write_json(args.output_dir / 'dataset_audit.json', report['dataset_audit'])
    storage.write_jsonl(args.output_dir / 'selected_records.jsonl', records)
    ready = all_ready(args.output_dir, records, expected)
    report['workers_launched'] = not ready
    if not ready:
        launch_workers(args)
    verify_sources(args, expected)
    verify_records(records, expected)
    saved, rank_reports, model_audit = merge(args.output_dir, records, expected)
    report.update(records=single.summarize(saved), model_audit=model_audit,
                  rank_coverage=[{'rank': row['rank'], 'coverage': row['inference_coverage'],
                                  'resumed_rows': row['resumed_rows']} for row in rank_reports],
                  inference_coverage={'status': 'PASS', 'generated': config.SAMPLE_SIZE,
                                      'expected_rows': config.SAMPLE_SIZE, 'skipped': 0})
    report['accuracy'] = scoring.score(args.output_dir, saved, config.SAMPLE_SIZE)
    report['status'] = 'PASS'


def run(args):
    report = {'status': 'FAILED', 'benchmark': 'reasonaqa_close_ended_sample5000', 'mode': 'sample5000',
              'parallel_evaluation': config.PARALLEL, 'protocol': runtime.PROTOCOL,
              'output_claimed': False, 'inference_coverage': {'status': 'FAILED'},
              'records': {}, 'accuracy': None, 'fatal_error': None}
    started = time.monotonic()
    try:
        with storage.output_lock(args.output_dir):
            try:
                execute(args, report)
            except Exception as exc:
                report['status'] = 'FAILED'
                report['fatal_error'] = {'error': repr(exc), 'traceback': traceback.format_exc()}
                print(json.dumps({'event': 'EIGHT_GPU_EVALUATION_FAILED', **report['fatal_error']}),
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
        print(json.dumps({**vars(args), 'parallel_evaluation': config.PARALLEL}, default=str, ensure_ascii=False))
        return 0
    if args.preflight_only:
        manifest = sample.load_manifest(args.manifest, args.test_json)
        dataset.prepare_records(manifest['records'], args)
        print(json.dumps({'status': 'PASS_PREFLIGHT_ONLY', 'sample_rows': config.SAMPLE_SIZE,
                          'parallel_evaluation': config.PARALLEL}))
        return 0
    report = run(args)
    print(json.dumps({'status': report['status'], 'accuracy': report['accuracy'],
                      'fatal_error': report['fatal_error'], 'output_dir': str(args.output_dir)}), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
