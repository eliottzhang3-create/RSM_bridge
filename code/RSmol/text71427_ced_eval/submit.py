"""One explicitly bound checkpoint per target; one independent job per benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from .registry import ROOT, ROUTES, BENCHMARKS
from .arguments import printable
from . import preflight


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=(*ROUTES, 'auto', 'batch'))
    parser.add_argument('benchmark', choices=(*BENCHMARKS, 'all'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--runtime-config', type=Path)
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--targets-json', type=Path, help='batch: JSON list of model/checkpoint/optional runtime_config/expected_step')
    parser.add_argument('--output-dir', type=Path, help='one model and one benchmark only')
    parser.add_argument('--output-root', type=Path, help='independent model/benchmark/checkpoint directories under this root')
    parser.add_argument('--queue', choices=('pdgpu-3090', 'pdgpu-4090', 'pdgpu-5090'))
    parser.add_argument('--submission-report', type=Path)
    parser.add_argument('--check-args', action='store_true')
    parser.add_argument('--preflight-only', action='store_true', help='CPU checkpoint/contract checks, no GPU submission')
    args, args.forwarded = parser.parse_known_args(argv)
    if args.check_args and args.preflight_only:
        parser.error('--check-args and --preflight-only are mutually exclusive')
    if args.output_dir and (args.model == 'batch' or args.benchmark == 'all' or args.output_root):
        parser.error('--output-dir requires one model/benchmark and no --output-root')
    if args.model == 'batch':
        if not args.targets_json or any((args.checkpoint, args.runtime_config, args.expected_step)):
            parser.error('batch requires --targets-json; checkpoints are specified in its records')
    elif not args.checkpoint or args.targets_json:
        parser.error('one model requires explicit --checkpoint and no --targets-json')
    # Identity/policy flags must be supplied through the validated arguments above.
    if any(x.split('=', 1)[0] in {'--eval-variant', '--benchmark', '--route-root', '--run-root',
            '--expected-geometry', '--check-args'} for x in args.forwarded):
        parser.error('model, benchmark and training geometry cannot be overridden')
    return args


def runner(benchmark):
    if benchmark == 'reasonaqa5000':
        from . import reasonaqa
        return 'reasonaqa', reasonaqa
    if benchmark == 'mmsu':
        from . import mmsu
        return 'mmsu', mmsu
    from . import benchmark as paired
    return 'benchmark', paired


def targets(args):
    if args.model != 'batch':
        return [{'model': args.model, 'checkpoint': str(args.checkpoint),
                 'runtime_config': str(args.runtime_config) if args.runtime_config else None,
                 'expected_step': args.expected_step}]
    records = json.loads(args.targets_json.read_text(encoding='utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError('targets JSON must be a nonempty list')
    for record in records:
        if (not isinstance(record, dict) or not {'model', 'checkpoint'} <= record.keys()
                or set(record) - {'model', 'checkpoint', 'runtime_config', 'expected_step'}
                or record['model'] not in (*ROUTES, 'auto') or not isinstance(record['checkpoint'], str)
                or not record['checkpoint']):
            raise ValueError('each batch target needs a supported model and explicit checkpoint')
    return records


def planned(args):
    jobs = []
    for record in targets(args):
        model = record['model']
        if model == 'auto':
            checkpoint = Path(record['checkpoint']).expanduser()
            runtime_config = record.get('runtime_config')
            if not runtime_config:
                if checkpoint.parent.name != 'checkpoints' or checkpoint.parent.parent.name != 'formal':
                    raise ValueError('auto with noncanonical checkpoint requires explicit --runtime-config')
                runtime_config = checkpoint.parent.parent / 'runtime.json'
            model = preflight.infer_model(Path(runtime_config).expanduser().resolve())
        for benchmark in BENCHMARKS if args.benchmark == 'all' else (args.benchmark,):
            module, selected = runner(benchmark)
            argv = ['--eval-variant', model, '--checkpoint', record['checkpoint']]
            if module == 'benchmark':
                argv += ['--benchmark', benchmark]
            for key in ('runtime_config', 'expected_step'):
                if record.get(key) is not None:
                    argv += ['--' + key.replace('_', '-'), str(record[key])]
            output = args.output_dir
            if args.output_root:
                output = args.output_root / model / benchmark / Path(record['checkpoint']).stem
            if output:
                argv += ['--output-dir', str(output)]
            argv += args.forwarded
            resolved = selected.parse_args([*argv, '--check-args'])
            jobs.append({'model': model, 'requested_model': record['model'], 'benchmark': benchmark, 'module': module,
                         'argv': argv, 'resolved': resolved,
                         'queue': args.queue or 'pdgpu-4090'})
    outputs = [str(x['resolved'].output_dir) for x in jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError('batch contains repeated output directories')
    return jobs


def command(job, tag):
    digest = hashlib.sha256(f"{tag}:{job['model']}:{job['benchmark']}:{job['resolved'].checkpoint}".encode()).hexdigest()[:12]
    name = f"t71427-{job['model'].replace('_', '-')[:20]}-{job['benchmark'][:5]}-{digest}"
    compute = shlex.join(['bash', 'text71427_ced_eval/run.sh', job['module'], *job['argv']])
    return ['submit', '-p', job['queue'], '-i', 'docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1',
            '-c', '8', '-m', '32G', '-g', '1', '-n', '1', '-j', name,
            '-d', str(ROOT), 'JOB=1:1', str(ROOT / 'log/text71427_ced_eval' / (name + '.JOB.log')),
            '--cmd', compute]


def main(argv=None):
    args = parse_args(argv)
    jobs = planned(args)
    tag = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + str(os.getpid())
    if args.check_args:
        for job in jobs:
            print(json.dumps({'model': job['model'], 'requested_model': job['requested_model'], 'benchmark': job['benchmark'],
                              'binding': printable(job['resolved']), 'vc_args': command(job, tag)}, default=str))
        return 0
    log = ROOT / 'log/text71427_ced_eval'
    log.mkdir(parents=True, exist_ok=True)
    report_path = (args.submission_report or log / (tag + '.submission.json')).resolve()
    if report_path.exists():
        raise FileExistsError(f'refusing to overwrite submission report: {report_path}')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    from mizar_s1_ced_reasonaqa5000_eval.storage import write_json
    report = {'status': 'RUNNING', 'preflight_only': args.preflight_only, 'preflights': [], 'jobs': []}
    checks = {}
    for job in jobs:
        resolved = job['resolved']
        key = (job['model'], str(resolved.checkpoint), str(resolved.runtime_config), resolved.expected_step)
        job['preflight_key'] = key
        if key in checks:
            continue
        try:
            result = preflight.check(resolved)
        except Exception as exc:
            result = {'status': 'FAILED', 'model': job['model'], 'checkpoint': str(resolved.checkpoint),
                      'runtime_config': str(resolved.runtime_config), 'error': repr(exc)}
            if isinstance(exc, preflight.CheckpointBindingError):
                result['diagnostic'] = exc.diagnostic
        checks[key] = result
        report['preflights'].append(result)
        write_json(report_path, report)
        print(json.dumps({'event': 'CHECKPOINT_PREFLIGHT', **result}, ensure_ascii=False), flush=True)
    for job in jobs:
        item = {'model': job['model'], 'benchmark': job['benchmark'], 'status': 'FAILED',
                'requested_model': job['requested_model'],
                'checkpoint': str(job['resolved'].checkpoint), 'output_dir': str(job['resolved'].output_dir),
                'queue': job['queue'], 'vc_attempted': False}
        try:
            checked = checks[job['preflight_key']]
            item['preflight'] = checked
            if checked['status'] != 'READY_FOR_GPU':
                item['error'] = checked['error']
            elif args.preflight_only:
                item['status'] = 'READY_FOR_GPU'
            else:
                item['vc_args'] = command(job, tag)
                item['vc_attempted'] = True
                result = subprocess.run(['vc', *item['vc_args']], capture_output=True, text=True, check=False)
                item.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                            status='SUBMITTED' if result.returncode == 0 else 'FAILED')
        except Exception as exc:
            item['error'] = repr(exc)
        report['jobs'].append(item)
        write_json(report_path, report)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    success = 'READY_FOR_GPU' if args.preflight_only else 'SUBMITTED'
    report['status'] = success if all(x['status'] == success for x in report['jobs']) else 'FAILED'
    write_json(report_path, report)
    print(json.dumps({'status': report['status'], 'submission_report': str(report_path)}))
    return 0 if report['status'] == success else 1


if __name__ == '__main__':
    raise SystemExit(main())
