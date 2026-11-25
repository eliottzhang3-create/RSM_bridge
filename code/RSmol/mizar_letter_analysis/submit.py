"""Exactly one eight-card 5090 job per requested experiment."""
import json
import shlex
import subprocess
import uuid
from mizar_layer_pair_search import storage
from . import arguments, config


def command(args, tag):
    forwarded=arguments.forwarded(args)
    # run.sh adds --task; do not duplicate its positional selection.
    forwarded=forwarded[2:]
    compute=shlex.join(['bash','mizar_letter_analysis/run.sh',args.task,'evaluate',*forwarded])
    return ['submit','-p','pdgpu-5090','-i','docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1',
        '-c','64','-m','256G','-g','8','-n','1','-j',f'mizar-letters-{args.task}-{tag}',
        '-d',str(config.ROOT),'JOB=1:1',str(config.ROOT/'log'/f'mizar-letters-{args.task}.{tag}.JOB.log'),'--cmd',compute]


def main(argv=None):
    args=arguments.parse_args(argv);tag=uuid.uuid4().hex[:12];planned=command(args,tag)
    if args.check_args:
        print(json.dumps({'task':args.task,'submitted_jobs':1,'vc_args':planned,'args':vars(args),
                          'protocol':config.protocol(args.task)},default=str,indent=2));return 0
    (config.ROOT/'log').mkdir(exist_ok=True)
    path=config.ROOT/'log'/f'mizar-letters-{args.task}.{tag}.submission.json'
    completed=subprocess.run(['bash','-c','vc "$@"','mizar-letters-vc',*planned],cwd=config.ROOT,
        capture_output=True,text=True,encoding='utf-8',errors='replace')
    report={'status':'SUBMITTED' if completed.returncode==0 else 'FAILED','task':args.task,
        'submitted_jobs':int(completed.returncode==0),'vc_args':planned,'returncode':completed.returncode,
        'stdout':completed.stdout,'stderr':completed.stderr,'source_manifest':str(args.source_manifest),
        'manifest':str(args.manifest),'output_dir':str(args.output_dir),
        'log_path':str(config.ROOT/'log'/f'mizar-letters-{args.task}.{tag}.1.log')}
    storage.write_json(path,report)
    print(json.dumps({**report,'submission_report':str(path)},ensure_ascii=False),flush=True)
    return 0 if completed.returncode==0 else 1


if __name__=='__main__':raise SystemExit(main())
