import json
import shlex
import subprocess
import uuid
from mizar_layer_pair_search import storage
from . import arguments,config


def command(a,tag):
    compute=shlex.join(['bash','mizar_benchmark_tracing/run.sh','evaluate',*arguments.forwarded(a)])
    return ['submit','-p','pdgpu-5090','-i','docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1','-c','64','-m','256G',
        '-g','8','-n','1','-j',f'mizar-tracing-{a.task}-{tag}','-d',str(config.ROOT),'JOB=1:1',
        str(config.ROOT/'log'/f'mizar-tracing-{a.task}.{tag}.JOB.log'),'--cmd',compute]


def main(argv=None):
    a=arguments.parse_args(argv);tag=uuid.uuid4().hex[:12];planned=command(a,tag)
    if a.check_args:
        print(json.dumps({'task':a.task,'submitted_jobs':1,'vc_args':planned,'args':vars(a),'protocol':config.protocol(a.task)},default=str,indent=2));return 0
    (config.ROOT/'log').mkdir(exist_ok=True)
    done=subprocess.run(['bash','-c','vc "$@"','mizar-benchmark-tracing-vc',*planned],cwd=config.ROOT,
        capture_output=True,text=True,encoding='utf-8',errors='replace')
    report={'status':'SUBMITTED' if done.returncode==0 else 'FAILED','task':a.task,'submitted_jobs':int(done.returncode==0),
        'vc_args':planned,'returncode':done.returncode,'stdout':done.stdout,'stderr':done.stderr,'input_manifest':str(a.manifest),
        'output_dir':str(a.output_dir),'selected_manifest':str(a.output_dir/'selected/manifest.json'),
        'log_path':str(config.ROOT/'log'/f'mizar-tracing-{a.task}.{tag}.1.log')}
    path=config.ROOT/'log'/f'mizar-tracing-{a.task}.{tag}.submission.json';storage.write_json(path,report)
    print(json.dumps({**report,'submission_report':str(path)},ensure_ascii=False),flush=True);return 0 if done.returncode==0 else 1


if __name__=='__main__':raise SystemExit(main())
