"""Validate isolated audio cutoff using real CPU Llama and explicit simulations."""
import argparse
import ast
from datetime import datetime, timezone
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
sys.path.insert(0,str(ROOT/'code/RSmol'))
from mizar_layer_pair_search import storage
from mizar_audio_cutoff import config


def shell_checks():
    bash='C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else 'bash'
    scripts=['code/RSmol/mizar_audio_cutoff/run.sh','code/RSmol/run_mizar_audio_cutoff_4090.sh',
             'code/RSmol/run_mizar_audio_cutoff_8gpu_3090.sh']
    for script in scripts:
        run=subprocess.run([bash,'-n',script],cwd=ROOT,capture_output=True,text=True)
        if run.returncode: raise RuntimeError(run.stderr)
    plans=[]
    with tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP')) as directory:
        fake=Path(directory)/'conda/etc/profile.d/conda.sh'; fake.parent.mkdir(parents=True)
        fake.write_text('conda() { test "$1" = activate && test "$2" = mellow_c8204d8; }\n',encoding='utf-8')
        env=dict(os.environ,RSMOL_CONDA_BASE=fake.parents[2].as_posix(),PYTHONUTF8='1')
        for script,world in zip(scripts[1:],(1,8)):
            argv=[bash,script,'--check-args','--output-dir',str(Path(directory)/"output's space")]
            run=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
            if run.returncode: raise RuntimeError(run.stderr)
            plan=json.loads(run.stdout); args=plan['vc_args']
            if args[args.index('-g')+1]!=str(world) or plan['submitted_jobs']!=1: raise ValueError('wrong shell topology')
            plans.append({'world':world,'partition':args[args.index('-p')+1],'actual_vc_called':False})
    return {'status':'PASS','syntax':scripts,'real_shell_parse_only':plans,'conda':'explicit fake activation'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-tensors',action='store_true')
    parser.add_argument('--report',type=Path,default=ROOT/'audit_reports/mizar_audio_cutoff_validation.json')
    args=parser.parse_args()
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'FAILED',
            'production_GPU_run':False,'production_weights_read':False,'remote_jobs_submitted':False,
            'simulations':'19x100 finalization and 1/8-worker control flow use explicit synthetic scores and runtime artifacts'}
    try:
        paths=[*config.PACKAGE.glob('*.py'),ROOT/'tests/test_mizar_audio_cutoff.py',Path(__file__)]
        for path in paths: ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,10))
        report['python310_AST_files']=len(paths)
        versions={}
        for name in ('torch','transformers','matplotlib'):
            try: versions[name]=importlib.import_module(name).__version__
            except ImportError: versions[name]='unavailable'
        report['versions']=versions
        suite=unittest.defaultTestLoader.loadTestsFromName('test_mizar_audio_cutoff')
        stream=io.StringIO()
        outcome=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        report['tests']={'run':outcome.testsRun,'passed':outcome.testsRun-len(outcome.errors)-len(outcome.failures)-len(outcome.skipped),
                        'skipped':[{'test':str(t),'reason':why} for t,why in outcome.skipped],
                        'errors':len(outcome.errors),'failures':len(outcome.failures),'log':stream.getvalue()}
        print(stream.getvalue(),flush=True)
        if not outcome.wasSuccessful() or (args.require_tensors and outcome.skipped): raise RuntimeError('tests failed or tensors skipped')
        report['shell']=shell_checks()
        report['status']='PASS_LOCAL_CPU' if not outcome.skipped else 'PASS_LOCAL_WITH_SKIPS'
    except Exception as exc:
        import traceback
        report.update(error=repr(exc),traceback=traceback.format_exc())
    storage.write_json(args.report,report)
    print(json.dumps({k:v for k,v in report.items() if k!='tests'},ensure_ascii=False,indent=2))
    print('Report:',args.report)
    return 0 if report['status'].startswith('PASS') else 1


if __name__=='__main__': raise SystemExit(main())
