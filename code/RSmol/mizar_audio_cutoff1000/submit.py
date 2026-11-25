"""One job completes baseline and 26 cutoff conditions."""
import json
import shlex
import subprocess
import uuid
from mizar_layer_pair_search import storage
from . import arguments, config


def command(args, tag):
    partition, cpus, memory = ('pdgpu-4090','8','32G') if args.world_size == 1 else ('pdgpu-3090','64','256G')
    compute = shlex.join(['bash','mizar_audio_cutoff1000/run.sh','evaluate',*arguments.forwarded(args)])
    return ['submit','-p',partition,'-i','docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1',
        '-c',cpus,'-m',memory,'-g',str(args.world_size),'-n','1','-j',f'mizar-acut1000-{args.world_size}g-{tag}',
        '-d',str(config.ROOT),'JOB=1:1',str(config.ROOT/'log'/f'mizar-audio-cutoff1000.{tag}.JOB.log'),'--cmd',compute]


def main(argv=None):
    args = arguments.parse_args(argv)
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
              'manifest':str(args.manifest), 'output_dir':str(args.output_dir), 'world_size':args.world_size}
    storage.write_json(report_path, report)
    print(json.dumps({**report,'submission_report':str(report_path)},ensure_ascii=False),flush=True)
    return 0 if run.returncode == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
