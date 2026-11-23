"""Real CPU numerical verification plus explicitly synthetic full-size task flows."""
import argparse
import ast
from datetime import datetime,timezone
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
sys.path.insert(0,str(ROOT/'tests')); sys.path.insert(0,str(ROOT/'code/RSmol'))
from mizar_layer_pair_search import storage


def shell_checks():
    bash='C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else 'bash'
    scripts=['code/RSmol/mizar_layer_analysis_common/run.sh','code/RSmol/run_mizar_layer_cka_8gpu_3090.sh',
             'code/RSmol/run_mizar_layer_ablation_8gpu_3090.sh']
    for script in scripts:
        run=subprocess.run([bash,'-n',script],cwd=ROOT,capture_output=True,text=True)
        if run.returncode: raise RuntimeError(run.stderr)
    plans=[]
    with tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP')) as directory:
        fake=Path(directory)/'conda/etc/profile.d/conda.sh'; fake.parent.mkdir(parents=True)
        fake.write_text('conda() { test "$1" = activate && test "$2" = mellow_c8204d8; }\n',encoding='utf-8')
        env=dict(os.environ,RSMOL_CONDA_BASE=fake.parents[2].as_posix(),PYTHONUTF8='1')
        for script,task in zip(scripts[1:],('cka','ablation')):
            run=subprocess.run([bash,script,'--check-args','--output-dir',str(Path(directory)/(task+"'s output"))],
                               cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
            if run.returncode: raise RuntimeError(run.stderr)
            plan=json.loads(run.stdout); argv=plan['vc_args']
            if argv[argv.index('-g')+1]!='8' or plan['submitted_jobs']!=1 or plan['task']!=task:
                raise ValueError('wrong actual shell/CLI resource plan')
            plans.append({'task':task,'partition':argv[argv.index('-p')+1],'GPU_count':8,'actual_vc_called':False})
    return {'status':'PASS','syntax':scripts,'real_shell_parameter_chain':plans,'conda':'explicit fixture activation'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-tensors',action='store_true')
    parser.add_argument('--report',type=Path,default=ROOT/'audit_reports/mizar_layer_analysis_validation.json')
    args=parser.parse_args()
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'FAILED',
            'production_weights_read':False,'remote_GPU_run':False,'jobs_submitted':False,
            'scope':'real CPU CKA and small 30-layer Llama; full 100-row CKA/29000-decision ablation use explicit synthetic artifacts'}
    try:
        paths=[p for folder in ('mizar_layer_analysis_common','mizar_layer_cka','mizar_layer_ablation')
               for p in (ROOT/'code/RSmol'/folder).glob('*.py')]
        paths.extend([ROOT/'tests/test_mizar_layer_analysis.py',Path(__file__),ROOT/'tests/fixtures/cka_official_reference.py'])
        for path in paths: ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,10))
        report['python310_AST_files']=len(paths)
        report['versions']={}
        for name in ('numpy','torch','transformers','matplotlib'):
            try: report['versions'][name]=importlib.import_module(name).__version__
            except ImportError: report['versions'][name]='unavailable'
        stream=io.StringIO(); suite=unittest.defaultTestLoader.loadTestsFromName('test_mizar_layer_analysis')
        outcome=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        report['tests']={'run':outcome.testsRun,'passed':outcome.testsRun-len(outcome.skipped)-len(outcome.errors)-len(outcome.failures),
            'skipped':[{'test':str(test),'reason':why} for test,why in outcome.skipped],
            'failures':len(outcome.failures),'errors':len(outcome.errors),'log':stream.getvalue()}
        print(stream.getvalue(),flush=True)
        from test_mizar_layer_analysis import ML
        if not outcome.wasSuccessful() or (args.require_tensors and not ML): raise RuntimeError('tests failed or required real tensor dependencies unavailable')
        report['shell']=shell_checks()
        report['status']='PASS_LOCAL_CPU' if not outcome.skipped else 'PASS_LOCAL_WITH_OPTIONAL_SKIPS'
    except Exception as exc:
        import traceback
        report.update(error=repr(exc),traceback=traceback.format_exc())
    storage.write_json(args.report,report)
    print(json.dumps({k:v for k,v in report.items() if k!='tests'},ensure_ascii=False,indent=2))
    print('Report:',args.report)
    return 0 if report['status'].startswith('PASS') else 1


if __name__=='__main__': raise SystemExit(main())
