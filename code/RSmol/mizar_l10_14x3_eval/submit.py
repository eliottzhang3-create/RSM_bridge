"""Submit AU/AR for the fixed L10-14x3 gated checkpoint, with CPU admission."""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from .registry import ROOT,BENCHMARKS,SOURCE_CACHE,TARGET
from . import arguments,preflight
from mizar_s1_ced_reasonaqa5000_eval import storage


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('benchmark',nargs='?',default='all',choices=('all',*BENCHMARKS))
    for name in ('checkpoint','runtime-config','model-dir','output-root','sources-cache','submission-report'):
        p.add_argument('--'+name,type=Path,default=SOURCE_CACHE if name=='sources-cache' else None)
    p.add_argument('--source-revision')
    p.add_argument('--check-args',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args(argv)
    if a.check_args and a.preflight_only:p.error('choose check-args or preflight-only')
    return a


def planned(args):
    jobs=[]
    for benchmark in BENCHMARKS if args.benchmark=='all' else (args.benchmark,):
        argv=['--benchmark',benchmark,'--sources-cache',str(args.sources_cache)]
        for name in ('checkpoint','runtime_config','model_dir','source_revision'):
            value=getattr(args,name)
            if value is not None:argv+=['--'+name.replace('_','-'),str(value)]
        if args.output_root:
            argv+=['--output-dir',str(args.output_root/benchmark/'model--step-2542')]
        resolved=arguments.parse_args('benchmark',[*argv,'--check-args'])
        jobs.append(dict(target=TARGET,benchmark=benchmark,argv=argv,resolved=resolved))
    if len({str(j['resolved'].output_dir) for j in jobs})!=len(jobs):raise ValueError('conflicting outputs')
    return jobs


def command(job,tag):
    suffix=hashlib.sha256(f"{tag}:{job['benchmark']}:{job['resolved'].checkpoint}".encode()).hexdigest()[:12]
    name=f"ml1014-gb3-{job['benchmark']}-{suffix}"
    if len(name)>=60:raise ValueError('job name exceeds cluster limit')
    compute=shlex.join(['bash','mizar_l10_14x3_eval/run.sh','benchmark',*job['argv']])
    log=ROOT/'log/mizar_l10_14x3_eval'/(name+'.JOB.log')
    return ['submit','-p','pdgpu-4090','-i','docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1',
        '-c','8','-m','32G','-g','1','-n','1','-j',name,'-d',str(ROOT),'JOB=1:1',str(log),'--cmd',compute]


def main(argv=None):
    args=parse_args(argv);jobs=planned(args)
    tag=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+str(os.getpid())
    if args.check_args:
        for job in jobs:
            print(json.dumps(dict(target=TARGET,benchmark=job['benchmark'],binding=arguments.printable(job['resolved']),vc_args=command(job,tag))))
        return 0
    log=ROOT/'log/mizar_l10_14x3_eval';log.mkdir(parents=True,exist_ok=True)
    path=(args.submission_report or log/(tag+'.submission.json')).resolve()
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='RUNNING',jobs=[],queue='pdgpu-4090')
    storage.write_json(path,report)
    try:checked=preflight.check(jobs[0]['resolved'])
    except Exception as exc:checked=dict(status='FAILED',error=repr(exc))
    report['preflight']=checked;storage.write_json(path,report)
    for job in jobs:
        item=dict(benchmark=job['benchmark'],checkpoint=str(job['resolved'].checkpoint),output_dir=str(job['resolved'].output_dir),status='FAILED',vc_attempted=False)
        try:
            if checked['status']!='READY_FOR_GPU':item['error']=checked.get('error','CPU preflight failed')
            elif args.preflight_only:item['status']='READY_FOR_GPU'
            else:
                item.update(vc_args=command(job,tag),vc_attempted=True)
                result=subprocess.run(['vc',*item['vc_args']],capture_output=True,text=True,check=False)
                item.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
                    status='SUBMITTED' if result.returncode==0 else 'FAILED')
        except Exception as exc:item['error']=repr(exc)
        report['jobs'].append(item);storage.write_json(path,report);print(json.dumps(item),flush=True)
    expected='READY_FOR_GPU' if args.preflight_only else 'SUBMITTED'
    report['status']=expected if all(j['status']==expected for j in report['jobs']) else 'FAILED'
    storage.write_json(path,report);print(json.dumps(dict(status=report['status'],submission_report=str(path))),flush=True)
    return int(report['status']!=expected)


if __name__=='__main__':raise SystemExit(main())
