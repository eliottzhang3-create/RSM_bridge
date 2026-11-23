"""Four independent 4090 jobs with immutable checkpoint and intervention binding."""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from . import arguments, benchmark, dataset, evaluate, preflight, scoring, storage
from .registry import ROOT, ROUTES, SOURCE_CACHE, ANNOTATIONS, intervention


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', nargs='?', default='all', choices=('all', *ROUTES))
    parser.add_argument('task', nargs='?', default='requested', choices=('requested', 'myeval8', 'mmar'))
    parser.add_argument('--check-args', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--sources-cache', type=Path, default=SOURCE_CACHE)
    parser.add_argument('--annotations', type=Path, default=ANNOTATIONS)
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--metadata-json', type=Path)
    parser.add_argument('--audio-root', type=Path)
    parser.add_argument('--evaluation-script', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--submission-report', type=Path)
    args = parser.parse_args(argv)
    if args.check_args and args.preflight_only:
        parser.error('choose --check-args or --preflight-only')
    return args


def planned(args):
    jobs = []
    for target in ROUTES if args.target == 'all' else (args.target,):
        requested = ('myeval8', 'mmar') if target == 'read2_zero48' else ('myeval8',)
        tasks = requested if args.task == 'requested' else (args.task,)
        for task in tasks:
            if task not in requested:
                if args.target != 'all':
                    raise ValueError('full MMAR is requested only for read2_zero48')
                continue
            argv = ['--eval-variant', target, '--sources-cache', str(args.sources_cache),
                    '--annotations', str(args.annotations)]
            for key in ('dataset_dir', 'metadata_json', 'audio_root', 'evaluation_script'):
                if getattr(args, key) is not None:
                    argv += ['--' + key.replace('_', '-'), str(getattr(args, key))]
            if args.output_root:
                argv += ['--output-dir', str(args.output_root / target / task / 'model--step-34626')]
            module, runner = ('evaluate', evaluate) if task == 'myeval8' else ('benchmark', benchmark)
            resolved = runner.parse_args([*argv, '--check-args'])
            jobs.append({'target': target, 'task': task, 'module': module, 'argv': argv, 'resolved': resolved})
    if not jobs or len({str(j['resolved'].output_dir) for j in jobs}) != len(jobs):
        raise ValueError('empty job selection or duplicate outputs')
    return jobs


def command(job, tag):
    digest = hashlib.sha256(f'{tag}:{job["target"]}:{job["task"]}'.encode()).hexdigest()[:12]
    name = f'mye8-{job["target"].replace("_", "-")}-{job["task"]}-{digest}'
    compute = shlex.join(['bash', 'myeval_mmar8/run.sh', job['module'], *job['argv']])
    return ['submit', '-p', 'pdgpu-4090', '-i', 'docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1',
            '-c', '8', '-m', '32G', '-g', '1', '-n', '1', '-j', name, '-d', str(ROOT),
            'JOB=1:1', str(ROOT / 'log/myeval_mmar8' / (name + '.JOB.log')), '--cmd', compute]


def dataset_preflight(args):
    rows = dataset.load_annotations(args.annotations)
    official = benchmark.mmar.load_mmar_records(args.metadata_json)
    records = dataset.expand(rows, official, args.audio_root)
    for path in {Path(r['audio_path_resolved']) for r in records}:
        if not path.is_file():
            raise FileNotFoundError(path)
    _, scorer = scoring.load_scorer(args.evaluation_script)
    return {'status': 'READY', 'rows': len(rows), 'generations_per_model': len(records), 'scorer': scorer}


def main(argv=None):
    args = parse_args(argv); jobs = planned(args)
    tag = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + str(os.getpid())
    if args.check_args:
        for job in jobs:
            print(json.dumps({'target': job['target'], 'task': job['task'],
                'binding': arguments.printable(job['resolved']), 'intervention': intervention(job['target']),
                'vc_args': command(job, tag)}, default=str))
        return 0
    log = ROOT / 'log/myeval_mmar8'; log.mkdir(parents=True, exist_ok=True)
    path = (args.submission_report or log / (tag + '.submission.json')).resolve()
    if path.exists():
        raise FileExistsError('refusing existing submission report')
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'RUNNING', 'jobs': [], 'preflights': [], 'queue': 'pdgpu-4090'}
    if any(j['task'] == 'myeval8' for j in jobs):
        try:
            report['dataset_preflight'] = dataset_preflight(next(j['resolved'] for j in jobs if j['task'] == 'myeval8'))
        except Exception as exc:
            report.update(status='FAILED', dataset_preflight={'status': 'FAILED', 'error': repr(exc)})
            storage.write_json(path, report)
            print(json.dumps(report, ensure_ascii=False)); return 1
    checks = {}
    for job in jobs:
        resolved = job['resolved']
        key = (str(resolved.checkpoint), str(resolved.runtime_config), str(resolved.route_root))
        if key not in checks:
            try:
                checks[key] = preflight.check(resolved)
            except Exception as exc:
                checks[key] = {'status': 'FAILED', 'error': repr(exc)}
            report['preflights'].append(checks[key]); storage.write_json(path, report)
        item = {'target': job['target'], 'task': job['task'], 'status': 'FAILED', 'vc_attempted': False,
                'checkpoint': str(resolved.checkpoint), 'output_dir': str(resolved.output_dir),
                'intervention': intervention(job['target']), 'preflight': checks[key]}
        try:
            if checks[key]['status'] != 'READY_FOR_GPU':
                item['error'] = checks[key]['error']
            elif args.preflight_only:
                item['status'] = 'READY_FOR_GPU'
            else:
                item['vc_args'] = command(job, tag); item['vc_attempted'] = True
                result = subprocess.run(['vc', *item['vc_args']], capture_output=True, text=True, check=False)
                item.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                            status='SUBMITTED' if result.returncode == 0 else 'FAILED')
        except Exception as exc:
            item['error'] = repr(exc)
        report['jobs'].append(item); storage.write_json(path, report)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    success = 'READY_FOR_GPU' if args.preflight_only else 'SUBMITTED'
    report['status'] = success if all(x['status'] == success for x in report['jobs']) else 'FAILED'
    storage.write_json(path, report)
    print(json.dumps({'status': report['status'], 'submission_report': str(path)}))
    return int(report['status'] != success)


if __name__ == '__main__':
    raise SystemExit(main())
