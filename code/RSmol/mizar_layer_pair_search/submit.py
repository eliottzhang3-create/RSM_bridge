"""Exactly one vc submit call for the complete mask-search pipeline."""
import json
import os
import shlex
import subprocess
import uuid

from . import arguments, config, execution, storage


def command(args, tag, world_size=8):
    execution.validate_world(world_size)
    partition, cpus, memory = ('pdgpu-4090', '8', '32G') if world_size == 1 else ('pdgpu-3090', '64', '256G')
    mode = 'pipeline_single_gpu' if world_size == 1 else 'pipeline'
    compute = shlex.join(['bash', 'mizar_layer_pair_search/run.sh', mode, *arguments.forwarded(args)])
    return ['submit', '-p', partition, '-i', 'docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1',
            '-c', cpus, '-m', memory, '-g', str(world_size), '-n', '1', '-j', f'mizar-prune28-{world_size}g-{tag}',
            '-d', str(config.ROOT), 'JOB=1:1', str(config.ROOT / 'log' / f'mizar-prune28.{tag}.JOB.log'),
            '--cmd', compute]


def main(argv=None, *, world_size=8):
    args = execution.parse_args(argv, world_size)
    tag = uuid.uuid4().hex[:12]
    planned = command(args, tag, world_size)
    if args.check_args:
        print(json.dumps({'vc_args': planned, 'submitted_jobs': 1, 'protocol': execution.protocol(world_size),
                          'args': vars(args)}, default=str, ensure_ascii=False, indent=2))
        return 0
    (config.ROOT / 'log').mkdir(exist_ok=True)
    report_path = config.ROOT / 'log' / f'mizar-prune28.{tag}.submission.json'
    result = subprocess.run(['bash', '-c', 'vc "$@"', 'mizar-prune28-vc', *planned],
                            cwd=config.ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    report = {'status': 'SUBMITTED' if result.returncode == 0 else 'FAILED', 'submitted_jobs': int(result.returncode == 0),
              'vc_args': planned, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
              'world_size': world_size, 'output_dir': str(args.output_dir), 'manifest': str(args.manifest)}
    storage.write_json(report_path, report)
    print(json.dumps({**report, 'submission_report': str(report_path)}, ensure_ascii=False), flush=True)
    return 0 if result.returncode == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
