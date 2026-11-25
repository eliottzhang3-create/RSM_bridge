"""Exactly one vc job runs all 29 conditions on eight 5090 GPUs."""
import json
import shlex
import subprocess
import uuid
from mizar_layer_pair_search import storage
from . import arguments, config


def command(args, tag):
    compute = shlex.join(['bash', 'mizar_layer_ablation_mmar/run.sh', 'evaluate', *arguments.forwarded(args)])
    return ['submit', '-p', 'pdgpu-5090', '-i', 'docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1',
            '-c', '64', '-m', '256G', '-g', '8', '-n', '1', '-j', f'mizar-mmar-ablation-8g-{tag}',
            '-d', str(config.ROOT), 'JOB=1:1',
            str(config.ROOT/'log'/f'mizar-layer-ablation-mmar.{tag}.JOB.log'), '--cmd', compute]


def main(argv=None):
    args = arguments.parse_args(argv)
    tag = uuid.uuid4().hex[:12]
    planned = command(args, tag)
    if args.check_args:
        print(json.dumps({'task': config.SCHEMA, 'submitted_jobs': 1, 'vc_args': planned, 'args': vars(args)},
                         default=str, indent=2))
        return 0
    (config.ROOT/'log').mkdir(exist_ok=True)
    path = config.ROOT/'log'/f'mizar-layer-ablation-mmar.{tag}.submission.json'
    result = subprocess.run(['bash', '-c', 'vc "$@"', 'mizar-mmar-ablation-vc', *planned], cwd=config.ROOT,
                            capture_output=True, text=True, encoding='utf-8', errors='replace')
    report = {'status': 'SUBMITTED' if result.returncode == 0 else 'FAILED', 'task': config.SCHEMA,
              'submitted_jobs': int(result.returncode == 0), 'vc_args': planned,
              'stdout': result.stdout, 'stderr': result.stderr, 'returncode': result.returncode,
              'output_dir': str(args.output_dir), 'manifest': str(args.manifest),
              'log_path': str(config.ROOT/'log'/f'mizar-layer-ablation-mmar.{tag}.1.log')}
    storage.write_json(path, report)
    print(json.dumps({**report, 'submission_report': str(path)}, ensure_ascii=False), flush=True)
    return 0 if result.returncode == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
