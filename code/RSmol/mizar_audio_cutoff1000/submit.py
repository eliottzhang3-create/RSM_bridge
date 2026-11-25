"""One job completes baseline and 26 cutoff conditions."""
import argparse
import json
import shlex
import subprocess
import sys
import uuid
from mizar_layer_pair_search import storage
from . import arguments, config


def parse_args(argv=None):
    # Placement belongs to submission only. Keep arguments.py and the manifest
    # generator's pinned sources unchanged so existing 1000-row manifests work.
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--partition', choices=('pdgpu-4090', 'pdgpu-3090', 'pdgpu-5090'))
    placement, remaining = parser.parse_known_args(argv)
    args = arguments.parse_args(remaining)
    args.partition = placement.partition or ('pdgpu-4090' if args.world_size == 1 else 'pdgpu-3090')
    if (args.world_size == 1) != (args.partition == 'pdgpu-4090'):
        parser.error('single GPU requires pdgpu-4090; eight GPUs require pdgpu-3090 or pdgpu-5090')
    explicit_output = any(value == '--output-dir' or value.startswith('--output-dir=') for value in remaining)
    if args.partition == 'pdgpu-5090' and not explicit_output:
        args.output_dir = (config.REMOTE / 'outputs/RSmol' / f'{config.SCHEMA}_8gpu5090').resolve()
    if args.manifest == args.output_dir or args.manifest.is_relative_to(args.output_dir):
        parser.error('manifest must be outside evaluation output')
    return args


def command(args, tag):
    partition = getattr(args, 'partition', None) or ('pdgpu-4090' if args.world_size == 1 else 'pdgpu-3090')
    cpus, memory = ('8', '32G') if args.world_size == 1 else ('64', '256G')
    compute = shlex.join(['bash','mizar_audio_cutoff1000/run.sh','evaluate',*arguments.forwarded(args)])
    return ['submit','-p',partition,'-i','docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1',
        '-c',cpus,'-m',memory,'-g',str(args.world_size),'-n','1','-j',f'mizar-acut1000-{args.world_size}g-{tag}',
        '-d',str(config.ROOT),'JOB=1:1',str(config.ROOT/'log'/f'mizar-audio-cutoff1000.{tag}.JOB.log'),'--cmd',compute]


def main(argv=None):
    args = parse_args(argv)
    tag = uuid.uuid4().hex[:12]
    planned = command(args, tag)
    if args.check_args:
        print(json.dumps({'vc_args':planned, 'submitted_jobs':1, 'args':vars(args), 'protocol':config.PROTOCOL},
                         default=str, ensure_ascii=False, indent=2))
        return 0
    (config.ROOT/'log').mkdir(exist_ok=True)
    report_path = config.ROOT/'log'/f'mizar-audio-cutoff1000.{tag}.submission.json'
    run = subprocess.run(['bash','-c','vc "$@"','mizar-audio-cutoff1000-vc',*planned], cwd=config.ROOT,
                          capture_output=True, text=True, encoding='utf-8', errors='replace')
    report = {'status':'SUBMITTED' if run.returncode == 0 else 'FAILED', 'submitted_jobs':int(run.returncode == 0),
              'vc_args':planned, 'returncode':run.returncode, 'stdout':run.stdout, 'stderr':run.stderr,
              'manifest':str(args.manifest), 'output_dir':str(args.output_dir), 'world_size':args.world_size,
              'partition':args.partition}
    storage.write_json(report_path, report)
    print(json.dumps({**report,'submission_report':str(report_path)},ensure_ascii=False),flush=True)
    return 0 if run.returncode == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
