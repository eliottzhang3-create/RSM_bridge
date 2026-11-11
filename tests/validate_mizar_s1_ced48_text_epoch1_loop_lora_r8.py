"""Local isolated-route checks; no remote assets and no real vc submission."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
ROUTE=ROOT/'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced48_text_epoch1_loop_lora_r8'
SUITES=('identity','route','behavior','shared_start','tensor')


def run_suite(name, result_path):
    sys.path.insert(0,str(ROOT/'tests'))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=f'test_mizar_s1_ced48_text_epoch1_loop_lora_r8_{name}.py')
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    record={'run':result.testsRun,'passed':result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),
            'success':result.wasSuccessful(),
            'skipped':[{'test':t.id(),'reason':r} for t,r in result.skipped],
            'failures':[{'test':t.id(),'traceback':r} for t,r in result.failures],
            'errors':[{'test':t.id(),'traceback':r} for t,r in result.errors]}
    result_path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    return 0 if result.wasSuccessful() else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-tensors',action='store_true')
    parser.add_argument('--suite',choices=SUITES,help=argparse.SUPPRESS)
    parser.add_argument('--result',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    os.chdir(ROOT)
    sys.dont_write_bytecode=True
    if os.name=='nt':
        os.environ['PATH']='C:/Program Files/Git/bin'+os.pathsep+os.environ['PATH']
    if args.suite:
        return run_suite(args.suite,args.result)
    files=[p for p in ROUTE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files+=[ROOT/'code/RSmol/run_mizar_s1_ced48_text_epoch1_loop_lora_r8_5090.sh']
    files+=list((ROOT/'tests').glob('*mizar_s1_ced48_text_epoch1_loop_lora_r8*.py'))
    for path in files:
        if path.suffix=='.py':
            compile(ast.parse(path.read_text(encoding='utf-8')),str(path),'exec')
    # The route uses generic module names (contract/models/data); each suite gets a fresh interpreter.
    import tempfile
    suites={}
    with tempfile.TemporaryDirectory(prefix='mizar-ced-e1-validation-') as directory:
        for name in SUITES:
            output=Path(directory)/(name+'.json')
            result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),
                                   '--suite',name,'--result',str(output)],cwd=ROOT)
            if output.is_file():
                suites[name]=json.loads(output.read_text(encoding='utf-8'))
            else:
                suites[name]={'run':0,'passed':0,'success':False,'skipped':[],
                              'failures':[],'errors':[{'error':f'suite exited {result.returncode} without a report'}]}
    skipped=sum(len(s['skipped']) for s in suites.values())
    passed=sum(s['passed'] for s in suites.values())
    success=all(s['success'] for s in suites.values()) and not (args.require_tensors and skipped)
    report={'status':('PASS_LOCAL_WITH_SKIPS' if skipped else 'PASS_LOCAL') if success else 'FAIL',
            'time_utc':datetime.now(timezone.utc).isoformat(),'passed':passed,'skipped':skipped,'suites':suites,
            'initial_text_checkpoint':'/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_loop_lora_r8_gbs1024/epoch1/checkpoint-step-009244',
            'production_assets_read':False,'gpu_training_verified':False,
            'real_vc_job_submitted':False,'eight_rank_resume_verified':False,
            'syntax_checked':[p.relative_to(ROOT).as_posix() for p in files if p.suffix=='.py'],
            'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    target=ROOT/'audit_reports/mizar_s1_ced48_text_epoch1_loop_lora_r8_validation.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f"{report['status']}: passed={passed}, skipped={skipped}; {target}")
    return 0 if success else 1


if __name__=='__main__':
    raise SystemExit(main())
