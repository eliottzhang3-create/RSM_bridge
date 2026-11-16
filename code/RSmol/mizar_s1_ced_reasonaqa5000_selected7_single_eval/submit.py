"""Submit only the supplemental seven fixed-5000 single-GPU jobs on pdgpu-4090."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from . import config, evaluate
from mizar_s1_ced_reasonaqa5000_eval import dataset, sample, storage


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', nargs='?', default='all', help='all, target alias or full eval_variant; use --list-targets')
    parser.add_argument('--list-targets', action='store_true')
    parser.add_argument('--check-args', action='store_true', help='print all effective bindings without assets or submission')
    parser.add_argument('--manifest', type=Path, default=config.DEFAULT_MANIFEST)
    parser.add_argument('--test-json', type=Path)
    parser.add_argument('--audiocaps-root', type=Path)
    parser.add_argument('--clotho-root', type=Path)
    parser.add_argument('--clotho-aqa-root', type=Path)
    parser.add_argument('--output-dir', type=Path, help='single target only')
    parser.add_argument('--submission-report', type=Path)
    args = parser.parse_args(argv)
    if args.target != 'all':
        try:
            args.target = config.target_name(args.target)
        except ValueError as exc:
            parser.error(str(exc))
    if args.target == 'all' and args.output_dir:
        parser.error('all requires seven independent default output directories')
    return args


def planned(args):
    names = tuple(config.TARGETS) if args.target == 'all' else (args.target,)
    result = []
    for name in names:
        argv = ['--target', name, '--manifest', str(args.manifest)]
        for key in ('test_json', 'audiocaps_root', 'clotho_root', 'clotho_aqa_root',
                    'output_dir'):
            value = getattr(args, key)
            if value is not None:
                argv.extend(['--' + key.replace('_', '-'), str(value)])
        resolved = evaluate.parse_args(argv)
        result.append((name, argv, resolved))
    outputs = [str(item[2].output_dir) for item in result]
    if len(outputs) != len(set(outputs)):
        raise ValueError('batch output directories must be distinct')
    return result


def job_name(name, tag):
    """Fit vc's 60-character limit, retaining identity beyond the readable prefix."""
    digest = hashlib.sha256(f'{tag}:{name}'.encode('utf-8')).hexdigest()[:12]
    return f'raqa7s-{name.replace("_", "-")[:40]}-{digest}'


def command(name, argv, tag):
    compute = shlex.join(['bash', 'mizar_s1_ced_reasonaqa5000_selected7_single_eval/run.sh', 'evaluate', *argv])
    return ['submit', '-p', 'pdgpu-4090', '-i', 'docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1',
            '-c', '8', '-m', '32G', '-g', '1', '-n', '1', '-j', job_name(name, tag),
            '-d', str(config.ROOT), 'JOB=1:1', str(config.ROOT / 'log' / f'reasonaqa5000.selected7.1gpu4090.{tag}.{name}.JOB.log'),
            '--cmd', compute]


def main(argv=None):
    args = parse_args(argv)
    if args.list_targets:
        print(json.dumps({'scope': 'supplemental_selected7_only', 'excluded': config.EXCLUDED, 'targets': config.TARGETS}, default=str, ensure_ascii=False, indent=2))
        return 0
    items = planned(args)
    if args.check_args:
        for name, forwarded, resolved in items:
            print(json.dumps({**vars(resolved), 'execution': config.EXECUTION,
                              'vc_args': command(name, forwarded, 'check_args')}, default=str, ensure_ascii=False))
        return 0
    first = items[0][2]
    manifest = sample.load_manifest(first.manifest, first.test_json)
    dataset.prepare_records(manifest['records'], first)
    tag = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + str(os.getpid())
    (config.ROOT / 'log').mkdir(exist_ok=True)
    report_path = args.submission_report or config.ROOT / 'log' / f'reasonaqa5000.selected7.1gpu4090.{tag}.submission.json'
    report_path = report_path.resolve()
    if report_path.exists():
        raise FileExistsError(f'refusing to overwrite a prior submission report: {report_path}')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'RUNNING', 'manifest_sha256': manifest['manifest_sha256'],
              'sample_rows': config.SAMPLE_SIZE, 'execution': config.EXECUTION, 'excluded': config.EXCLUDED, 'targets': []}
    storage.write_json(report_path, report)
    for name, forwarded, resolved in items:
        item = {'target': name, 'checkpoint': str(resolved.checkpoint), 'output_dir': str(resolved.output_dir),
                'status': 'FAILED', 'vc_attempted': False}
        try:
            vc_args = command(name, forwarded, tag)
            item['vc_args'] = vc_args
            item['vc_attempted'] = True
            result = subprocess.run(['bash', '-c', 'vc "$@"', 'reasonaqa5000-vc', *vc_args],
                                    cwd=config.ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
            item.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            if result.returncode:
                raise RuntimeError(f'vc returned {result.returncode}: {result.stderr.strip()}')
            item['status'] = 'SUBMITTED'
        except Exception as exc:
            item['error'] = repr(exc)
        report['targets'].append(item)
        storage.write_json(report_path, report)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    failed = sum(item['status'] != 'SUBMITTED' for item in report['targets'])
    report.update(status='FAILED' if failed else 'SUBMITTED', submitted=len(items) - failed, failed=failed)
    storage.write_json(report_path, report)
    print(json.dumps({'status': report['status'], 'submitted': report['submitted'], 'failed': failed,
                      'submission_report': str(report_path)}), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
