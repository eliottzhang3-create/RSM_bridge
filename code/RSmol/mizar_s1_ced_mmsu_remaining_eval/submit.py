"""Submit only the remaining 27 fixed MMSU targets, all on pdgpu-4090."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from mizar_s1_ced_mmsu_eval import evaluate, rescore, storage
from . import checkpoint, config, inspection


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', nargs='?', default='all', help='all, alias or complete eval_variant')
    parser.add_argument('--mode', choices=('submit', 'inspect'), default='submit')
    parser.add_argument('--list-targets', action='store_true')
    parser.add_argument('--check-args', action='store_true', help='no assets, output writes or vc calls')
    parser.add_argument('--mmsu-root', type=Path, default=evaluate.DEFAULT_MMSU)
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--evaluation-script', type=Path)
    parser.add_argument('--parquet-batch-size', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, help='single target only')
    parser.add_argument('--run-root', type=Path, help='single mesh_inject48 candidate correction only')
    parser.add_argument('--checkpoint', type=Path, help='single mesh_inject48 candidate correction only')
    parser.add_argument('--runtime-config', type=Path, help='single mesh_inject48 candidate correction only')
    parser.add_argument('--mesh-inject48-run-root', type=Path, help='candidate correction in all only')
    parser.add_argument('--submission-report', type=Path)
    parser.add_argument('--ledger-dir', type=Path, default=config.DEFAULT_LEDGER)
    parser.add_argument('--retry-submitted', action='store_true',
                        help='single target only: after canceling/confirming the old job stopped')
    args = parser.parse_args(argv)
    try:
        if args.target != 'all':
            args.target = config.target_name(args.target)
    except ValueError as exc:
        parser.error(str(exc))
    if args.parquet_batch_size < 1:
        parser.error('--parquet-batch-size must be positive; model inference remains batch1')
    if args.target == 'all' and any((args.output_dir, args.run_root, args.checkpoint,
                                    args.runtime_config, args.retry_submitted)):
        parser.error('all forbids shared output/model overrides and retry-submitted')
    if args.target != 'all' and args.mesh_inject48_run_root:
        parser.error('--mesh-inject48-run-root is for all only')
    if args.target not in ('all', 'mesh_inject48') and any((args.run_root, args.checkpoint, args.runtime_config)):
        parser.error('only the mesh_inject48 candidate allows explicit model path correction')
    return args


def planned(args):
    result = []
    names = tuple(config.TARGETS) if args.target == 'all' else (args.target,)
    for name in names:
        spec = config.TARGETS[name]
        run = (args.mesh_inject48_run_root if name == 'mesh_inject48' and args.mesh_inject48_run_root
               else args.run_root or spec['run_root'])
        if args.checkpoint and not args.run_root:
            if args.checkpoint.parent.name == 'checkpoints' and args.checkpoint.parent.parent.name == 'formal':
                run = args.checkpoint.parent.parent.parent
            elif not all((args.runtime_config, args.output_dir)):
                raise ValueError('noncanonical candidate checkpoint needs runtime-config and output-dir')
        argv = ['--eval-variant', spec['eval_variant'], '--run-root', str(run),
                '--route-root', str(spec['route_root']), '--expected-step', str(spec['expected_step']),
                '--expected-geometry', ','.join(map(str, spec['expected_geometry'])),
                '--mmsu-root', str(args.mmsu_root), '--parquet-batch-size', str(args.parquet_batch_size)]
        for key in ('dataset_dir', 'evaluation_script', 'output_dir', 'checkpoint', 'runtime_config'):
            value = getattr(args, key)
            if value is not None:
                argv.extend(['--' + key.replace('_', '-'), str(value)])
        resolved = evaluate.parse_args(argv)
        result.append((name, argv, resolved))
    if len({str(item[2].output_dir) for item in result}) != len(result):
        raise ValueError('MMSU batch outputs must be independent')
    return result


def job_name(name, tag):
    digest = hashlib.sha256(f'{tag}:{name}'.encode()).hexdigest()[:12]
    return f'mmsu27-{name.replace("_", "-")[:40]}-{digest}'


def command(name, argv, tag):
    # Execute the original compute shell, so generation and resume identity
    # are exactly the same as the existing single-target MMSU implementation.
    compute = shlex.join(['bash', 'mizar_s1_ced_mmsu_eval/run.sh', *argv])
    return ['submit', '-p', 'pdgpu-4090', '-i', 'docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1',
            '-c', '8', '-m', '32G', '-g', '1', '-n', '1', '-j', job_name(name, tag),
            '-d', str(config.ROOT), 'JOB=1:1', str(config.ROOT / 'log' / f'mmsu27.{tag}.{name}.JOB.log'),
            '--cmd', compute]


def admission_dir(args, resolved):
    binding = {key: str(getattr(resolved, key)) for key in
               ('eval_variant', 'checkpoint', 'runtime_config', 'output_dir', 'dataset_dir', 'evaluation_script')}
    return args.ledger_dir.resolve() / storage.object_sha(binding)


def dispatch(args, name, argv, resolved, tag, cache):
    item = {'target': name, 'eval_variant': resolved.eval_variant, 'checkpoint': str(resolved.checkpoint),
            'output_dir': str(resolved.output_dir), 'status': 'FAILED', 'vc_attempted': False}
    try:
        if args.mode == 'inspect':
            state = inspection.inspect(resolved, cache)
            item.update(status=state['status'], existing_output=state)
            record = admission_dir(args, resolved) / 'admission.json'
            if record.exists():
                item['admission'] = inspection.read_json(record)
            return item
        # Serialize dispatcher decisions, including the queued-job reservation.
        # This lock is in the new ledger, not in an existing evaluation directory.
        ledger = admission_dir(args, resolved)
        with storage.output_lock(ledger):
            state = inspection.inspect(resolved, cache)
            item['existing_output'] = state
            if state['status'] == 'COMPLETE':
                item['status'] = 'SKIPPED_COMPLETE'
                return item
            if state['status'] == 'BUSY':
                item['status'] = 'SKIPPED_BUSY'
                return item
            if state['status'] == 'NEEDS_RESCORE':
                report = rescore.run(rescore.parse_args(['--source-output-dir', str(resolved.output_dir),
                                                       '--evaluation-script', str(resolved.evaluation_script)]))
                item['cpu_rescore'] = {'status': report['status'], 'report': report['report']}
                if report['status'] != 'PASS':
                    raise ValueError(f'CPU rescore failed: {report.get("fatal_error")}')
                state = inspection.inspect(resolved, cache)
                if state['status'] != 'COMPLETE':
                    raise ValueError('CPU rescore did not produce a verified complete MMSU result')
                item.update(status='RESCORED_CPU', existing_output=state)
                return item
            record = ledger / 'admission.json'
            if record.exists() and not args.retry_submitted:
                old = inspection.read_json(record)
                if old['status'] in ('SUBMITTING', 'SUBMITTED'):
                    item.update(status='SKIPPED_SUBMITTED', admission=old)
                    return item
            if name == 'mesh_inject48':
                item['candidate_cpu_contract'] = checkpoint.inspect(resolved)
            vc_args = command(name, argv, tag)
            item['vc_args'] = vc_args
            admission = {'status': 'SUBMITTING', 'target': name, 'tag': tag,
                         'checkpoint': str(resolved.checkpoint), 'output_dir': str(resolved.output_dir),
                         'vc_args': vc_args}
            storage.write_json(record, admission)
            item['vc_attempted'] = True
            try:
                result = subprocess.run(['bash', '-c', 'vc "$@"', 'mmsu27-vc', *vc_args],
                    cwd=config.ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
                item.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
                admission.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                                 status='SUBMITTED' if result.returncode == 0 else 'FAILED')
                storage.write_json(record, admission)
                if result.returncode:
                    raise RuntimeError(f'vc returned {result.returncode}: {result.stderr.strip()}')
                item['status'] = 'SUBMITTED'
            except OSError:
                admission['status'] = 'FAILED'
                storage.write_json(record, admission)
                raise
    except Exception as exc:
        item['error'] = repr(exc)
    return item


def main(argv=None):
    args = parse_args(argv)
    if args.list_targets:
        print(json.dumps({'excluded': config.EXCLUDED, 'targets': config.TARGETS}, default=str, ensure_ascii=False, indent=2))
        return 0
    items = planned(args)
    if args.check_args:
        for name, forwarded, resolved in items:
            print(json.dumps({'target': name, 'partition': 'pdgpu-4090', **vars(resolved),
                              'vc_args': command(name, forwarded, 'CHECK_ONLY')}, default=str, ensure_ascii=False))
        return 0
    tag = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + str(os.getpid())
    path = (args.submission_report or config.ROOT / 'log' / f'mmsu27.{tag}.{args.mode}.json').resolve()
    if path.exists():
        raise FileExistsError(f'refusing to replace previous batch report: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'RUNNING', 'partition': 'pdgpu-4090', 'mode': args.mode,
              'excluded': config.EXCLUDED, 'targets': []}
    storage.write_json(path, report)
    cache = inspection.FileCache()
    for name, forwarded, resolved in items:
        item = dispatch(args, name, forwarded, resolved, tag, cache)
        report['targets'].append(item)
        storage.write_json(path, report)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    report.update(status='FAILED' if any(i['status'] == 'FAILED' for i in report['targets']) else 'PASS',
                  submitted=sum(i['status'] == 'SUBMITTED' for i in report['targets']),
                  failed=sum(i['status'] == 'FAILED' for i in report['targets']))
    storage.write_json(path, report)
    print(json.dumps({'status': report['status'], 'submitted': report['submitted'], 'failed': report['failed'],
                      'batch_report': str(path)}, ensure_ascii=False))
    return int(report['status'] == 'FAILED')


if __name__ == '__main__':
    raise SystemExit(main())
