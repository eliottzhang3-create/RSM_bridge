import json
import shlex
import subprocess
import uuid
from mizar_layer_pair_search import storage
from . import arguments,config


def command(args,task,tag):
    compute=shlex.join(['bash','mizar_layer_analysis_common/run.sh',task,'evaluate',*arguments.forwarded(args)])
    return ['submit','-p','pdgpu-3090','-i','docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1',
        '-c','64','-m','256G','-g','8','-n','1','-j',f'mizar-{task}-8g-{tag}','-d',str(config.ROOT),
        'JOB=1:1',str(config.ROOT/'log'/f'mizar-layer-{task}.{tag}.JOB.log'),'--cmd',compute]


def main(task,argv=None):
    args=arguments.parse_args(task,argv); tag=uuid.uuid4().hex[:12]; planned=command(args,task,tag)
    if args.check_args:
        print(json.dumps({'task':task,'submitted_jobs':1,'vc_args':planned,'args':vars(args)},default=str,indent=2)); return 0
    (config.ROOT/'log').mkdir(exist_ok=True)
    path=config.ROOT/'log'/f'mizar-layer-{task}.{tag}.submission.json'
    result=subprocess.run(['bash','-c','vc "$@"','mizar-layer-analysis-vc',*planned],cwd=config.ROOT,
        capture_output=True,text=True,encoding='utf-8',errors='replace')
    report={'status':'SUBMITTED' if result.returncode==0 else 'FAILED','task':task,'submitted_jobs':int(result.returncode==0),
        'vc_args':planned,'stdout':result.stdout,'stderr':result.stderr,'returncode':result.returncode,
        'output_dir':str(args.output_dir),'manifest':str(args.manifest)}
    storage.write_json(path,report); print(json.dumps({**report,'submission_report':str(path)},ensure_ascii=False),flush=True)
    return 0 if result.returncode==0 else 1
