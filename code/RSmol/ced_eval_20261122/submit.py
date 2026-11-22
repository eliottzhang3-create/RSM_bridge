"""Submit the 19 requested jobs; fixed checkpoint/source binding and 4090 only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from .registry import ROOT, ROUTES, BENCHMARKS, SOURCE_CACHE
from .targets import target_records
from .arguments import printable
from . import preflight
from mizar_s1_ced_reasonaqa5000_eval import storage


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target',nargs='?',default='all',choices=('all',*ROUTES))
    parser.add_argument('benchmark',nargs='?',default='requested',choices=('requested',*BENCHMARKS))
    parser.add_argument('--check-args',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--sources-cache',type=Path,default=SOURCE_CACHE)
    parser.add_argument('--submission-report',type=Path)
    args=parser.parse_args(argv)
    if args.check_args and args.preflight_only:parser.error('choose check-args or preflight-only')
    return args


def runner(benchmark):
    if benchmark=='reasonaqa5000':
        from . import reasonaqa
        return 'reasonaqa',reasonaqa
    if benchmark=='mmsu':
        from . import mmsu
        return 'mmsu',mmsu
    from . import benchmark as paired
    return 'benchmark',paired


def planned(args):
    records=target_records()
    jobs=[]
    names=records if args.target=='all' else (args.target,)
    for name in names:
        record=records[name]
        benchmarks=record['benchmarks'] if args.benchmark=='requested' else [args.benchmark]
        for benchmark in benchmarks:
            if benchmark not in record['benchmarks']:
                if args.target!='all':raise ValueError(f'{name} was not requested for {benchmark}')
                continue
            module,selected=runner(benchmark)
            argv=['--eval-variant',name,'--checkpoint',record['checkpoint'],
                  '--sources-cache',str(args.sources_cache)]
            if module=='benchmark':argv+=['--benchmark',benchmark]
            resolved=selected.parse_args([*argv,'--check-args'])
            jobs.append({'target':name,'benchmark':benchmark,'module':module,'argv':argv,'resolved':resolved})
    if not jobs or len({str(j['resolved'].output_dir) for j in jobs})!=len(jobs):
        raise ValueError('no jobs or conflicting output directories')
    return jobs


def command(job,tag):
    digest=hashlib.sha256(f"{tag}:{job['target']}:{job['benchmark']}:{job['resolved'].checkpoint}".encode()).hexdigest()[:12]
    name=f"ced22-{job['target'].replace('_','-')}-{job['benchmark'][:5]}-{digest}"
    compute=shlex.join(['bash','ced_eval_20261122/run.sh',job['module'],*job['argv']])
    logfile=ROOT/'log/ced_eval_20261122'/(name+'.JOB.log')
    return ['submit','-p','pdgpu-4090','-i','docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1',
        '-c','8','-m','32G','-g','1','-n','1','-j',name,'-d',str(ROOT),'JOB=1:1',str(logfile),'--cmd',compute]


def main(argv=None):
    args=parse_args(argv);jobs=planned(args)
    tag=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+str(os.getpid())
    if args.check_args:
        for job in jobs:print(json.dumps({'target':job['target'],'benchmark':job['benchmark'],
            'binding':printable(job['resolved']),'source_revision':ROUTES[job['target']]['revision'],
            'vc_args':command(job,tag)},default=str))
        return 0
    log=ROOT/'log/ced_eval_20261122';log.mkdir(parents=True,exist_ok=True)
    path=(args.submission_report or log/(tag+'.submission.json')).resolve()
    if path.exists():raise FileExistsError(f'refusing existing submission report: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'RUNNING','preflights':[],'jobs':[],'queue':'pdgpu-4090','preflight_only':args.preflight_only}
    checks={}
    for job in jobs:
        name=job['target']
        if name in checks:continue
        try:result=preflight.check(job['resolved'])
        except Exception as exc:result={'status':'FAILED','target':name,'error':repr(exc)}
        checks[name]=result;report['preflights'].append(result)
        storage.write_json(path,report)
        print(json.dumps({'event':'CHECKPOINT_PREFLIGHT',**result}),flush=True)
    for job in jobs:
        item={'target':job['target'],'benchmark':job['benchmark'],'status':'FAILED','vc_attempted':False,
              'checkpoint':str(job['resolved'].checkpoint),'output_dir':str(job['resolved'].output_dir),
              'preflight':checks[job['target']]}
        try:
            if item['preflight']['status']!='READY_FOR_GPU':item['error']=item['preflight']['error']
            elif args.preflight_only:item['status']='READY_FOR_GPU'
            else:
                item['vc_args']=command(job,tag);item['vc_attempted']=True
                result=subprocess.run(['vc',*item['vc_args']],capture_output=True,text=True,check=False)
                item.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
                            status='SUBMITTED' if result.returncode==0 else 'FAILED')
        except Exception as exc:item['error']=repr(exc)
        report['jobs'].append(item);storage.write_json(path,report)
        print(json.dumps(item),flush=True)
    success='READY_FOR_GPU' if args.preflight_only else 'SUBMITTED'
    report['status']=success if all(j['status']==success for j in report['jobs']) else 'FAILED'
    storage.write_json(path,report)
    print(json.dumps({'status':report['status'],'submission_report':str(path)}))
    return int(report['status']!=success)


if __name__=='__main__':raise SystemExit(main())
