"""Local CPU validation, existing experiment remains pinned and unchanged."""
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
sys.path.insert(0,str(ROOT/'code/RSmol'));sys.path.insert(0,str(ROOT/'tests'))
from mizar_layer_pair_search import storage
from mizar_reasonaqa_tracing500 import config


def shell_checks():
    bash='C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else 'bash'
    scripts=['code/RSmol/mizar_reasonaqa_tracing500/run.sh','code/RSmol/run_mizar_reasonaqa_tracing500_8gpu_5090.sh']
    for script in scripts:
        r=subprocess.run([bash,'-n',script],cwd=ROOT,capture_output=True,text=True)
        if r.returncode:raise RuntimeError(r.stderr)
    with tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP')) as temp:
        folder=Path(temp);fake=folder/'conda/etc/profile.d/conda.sh';fake.parent.mkdir(parents=True)
        fake.write_text('conda() { test "$1" = activate && test "$2" = mellow_c8204d8; }\n',encoding='utf-8')
        env=dict(os.environ,RSMOL_CONDA_BASE=fake.parents[2].as_posix(),PYTHONUTF8='1')
        r=subprocess.run([bash,scripts[1],'--check-args','--output-dir',str(folder/"a's output")],cwd=ROOT,
            env=env,capture_output=True,text=True,encoding='utf-8')
        if r.returncode:raise RuntimeError(r.stderr)
        plan=json.loads(r.stdout);argv=plan['vc_args']
        for flag,val in (('-p','pdgpu-5090'),('-g','8'),('-c','64'),('-m','256G'),('-n','1')):
            if argv[argv.index(flag)+1]!=val:raise ValueError('resource mismatch')
        if plan['submitted_jobs']!=1:raise ValueError('not one job')
    return {'status':'PASS','actual_Bash_parse_only':True,'actual_vc_called':False,'conda':'activation fixture','scripts':scripts}


def main():
    target=ROOT/'audit_reports/mizar_reasonaqa_tracing500_validation.json'
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'FAILED',
        'remote_GPU_run':False,'jobs_submitted':False,'production_weights_read':False,
        'scope':'Synthetic original5000/730 accepted source import and full15000 patch results; real BF16 lossless cache. Original real30-layer CPU tests run as regressions.'}
    if target.exists():
        old=storage.read_json(target);report['previous_runs']=[*old.get('previous_runs',[]),{k:v for k,v in old.items() if k!='previous_runs'}]
    try:
        paths=[*config.PACKAGE.glob('*.py'),ROOT/'tests/test_mizar_reasonaqa_tracing500.py',Path(__file__)]
        for path in paths:ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,10))
        report['python310_AST_files']=len(paths)
        report['versions']={n:importlib.import_module(n).__version__ for n in ('torch','transformers','numpy','matplotlib')}
        stream=io.StringIO();suite=unittest.TestSuite()
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_mizar_reasonaqa_tracing500'))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_mizar_reasonaqa_tracing.RealNumerics'))
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        report['tests']={'run':result.testsRun,'passed':result.testsRun-len(result.errors)-len(result.failures)-len(result.skipped),
            'errors':len(result.errors),'failures':len(result.failures),'skipped':len(result.skipped),'log':stream.getvalue()}
        print(stream.getvalue(),flush=True)
        if not result.wasSuccessful() or result.skipped:raise RuntimeError('tests failed or skipped')
        report['shell']=shell_checks();report['status']='PASS_LOCAL_CPU'
    except Exception as exc:
        import traceback
        report.update(error=repr(exc),traceback=traceback.format_exc())
    storage.write_json(target,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('tests','previous_runs')},ensure_ascii=False,indent=2))
    return 0 if report['status']=='PASS_LOCAL_CPU' else 1


if __name__=='__main__':raise SystemExit(main())
