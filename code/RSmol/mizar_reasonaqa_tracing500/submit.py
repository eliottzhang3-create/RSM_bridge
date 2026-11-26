import json
import shlex
import subprocess
import uuid
from mizar_layer_pair_search import storage
from . import arguments, config


def command(args,tag):
    compute=shlex.join(['bash','mizar_reasonaqa_tracing500/run.sh','evaluate',*arguments.forwarded(args)])
    return ['submit','-p','pdgpu-5090','-i','docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1',
        '-c','64','-m','256G','-g','8','-n','1','-j',f'mizar-reasonaqa-tracing500-{tag}',
        '-d',str(config.ROOT),'JOB=1:1',str(config.ROOT/'log'/f'mizar-reasonaqa-tracing500.{tag}.JOB.log'),'--cmd',compute]


def main(argv=None):
    args=arguments.parse_args(argv);tag=uuid.uuid4().hex[:12];planned=command(args,tag)
    if args.check_args:
        print(json.dumps({'submitted_jobs':1,'vc_args':planned,'args':vars(args),'protocol':config.PROTOCOL},default=str,indent=2));return 0
    (config.ROOT/'log').mkdir(exist_ok=True)
    completed=subprocess.run(['bash','-c','vc "$@"','mizar-tracing500-vc',*planned],cwd=config.ROOT,
        capture_output=True,text=True,encoding='utf-8',errors='replace')
    report={'status':'SUBMITTED' if completed.returncode==0 else 'FAILED','submitted_jobs':int(completed.returncode==0),
        'vc_args':planned,'returncode':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr,
        'source_output_dir':str(args.source_output_dir),'output_dir':str(args.output_dir),
        'selected_manifest':str(args.output_dir/'selected/manifest.json'),
        'log_path':str(config.ROOT/'log'/f'mizar-reasonaqa-tracing500.{tag}.1.log')}
    path=config.ROOT/'log'/f'mizar-reasonaqa-tracing500.{tag}.submission.json'
    storage.write_json(path,report)
    print(json.dumps({**report,'submission_report':str(path)},ensure_ascii=False),flush=True)
    return 0 if completed.returncode==0 else 1


if __name__=='__main__':
    raise SystemExit(main())
