"""Validate three lowercase-letter/EOT reruns without production assets or jobs."""
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
sys.path.insert(0,str(ROOT/'tests'));sys.path.insert(0,str(ROOT/'code/RSmol'))
from mizar_layer_pair_search import storage
from mizar_letter_analysis import config


def shell_checks():
    bash='C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else 'bash'
    scripts=['code/RSmol/mizar_letter_analysis/run.sh',
        'code/RSmol/run_mizar_audio_cutoff1000_letters_eot_8gpu_5090.sh',
        'code/RSmol/run_mizar_layer_ablation_mmau_letters_eot_8gpu_5090.sh',
        'code/RSmol/run_mizar_layer_ablation_mmar_letters_eot_8gpu_5090.sh']
    for script in scripts:
        run=subprocess.run([bash,'-n',script],cwd=ROOT,capture_output=True,text=True)
        if run.returncode:raise RuntimeError(run.stderr)
    plans=[]
    with tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP')) as directory:
        fake=Path(directory)/'conda/etc/profile.d/conda.sh';fake.parent.mkdir(parents=True)
        fake.write_text('conda() { test "$1" = activate && test "$2" = mellow_c8204d8; }\n',encoding='utf-8')
        env=dict(os.environ,RSMOL_CONDA_BASE=fake.parents[2].as_posix(),PYTHONUTF8='1')
        for script,task in zip(scripts[1:],config.TASKS):
            run=subprocess.run([bash,script,'--check-args','--output-dir',str(Path(directory)/(task+"'s output"))],
                cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
            if run.returncode:raise RuntimeError(run.stderr)
            plan=json.loads(run.stdout);argv=plan['vc_args']
            for flag,value in (('-p','pdgpu-5090'),('-g','8'),('-c','64'),('-m','256G'),('-n','1')):
                if argv[argv.index(flag)+1]!=value:raise ValueError('wrong queue/resources')
            if plan['submitted_jobs']!=1 or plan['task']!=task:raise ValueError('wrong task count or routing')
            plans.append({'task':task,'partition':'pdgpu-5090','GPU_count':8,'actual_vc_called':False})
    return {'status':'PASS','scripts':scripts,'real_shell_parse_only':plans,'conda':'explicit activation fixture'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-tensors',action='store_true')
    parser.add_argument('--report',type=Path,default=ROOT/'audit_reports/mizar_letter_analysis_validation.json')
    args=parser.parse_args()
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'FAILED',
        'production_weights_read':False,'remote_GPU_run':False,'jobs_submitted':False,
        'scope':'real small 30-layer CPU Llama/BF16/one-forward letter gather, frozen input conversion and cache serialization; three full 85000-decision runs and eight-rank control use explicit synthetic fixtures'}
    if args.report.exists():
        prior=storage.read_json(args.report)
        report['previous_runs']=[*prior.get('previous_runs',[]),{k:v for k,v in prior.items() if k!='previous_runs'}]
    try:
        paths=[*config.PACKAGE.glob('*.py'),ROOT/'tests/test_mizar_letter_analysis.py',Path(__file__)]
        for path in paths:ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,10))
        report['python310_AST_files']=len(paths)
        report['versions']={name:importlib.import_module(name).__version__ for name in ('torch','transformers','numpy','matplotlib')}
        stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('test_mizar_letter_analysis')
        outcome=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        report['tests']={'run':outcome.testsRun,'passed':outcome.testsRun-len(outcome.skipped)-len(outcome.errors)-len(outcome.failures),
            'skipped':[{'test':str(t),'reason':why} for t,why in outcome.skipped],
            'failures':len(outcome.failures),'errors':len(outcome.errors),'log':stream.getvalue()}
        print(stream.getvalue(),flush=True)
        if not outcome.wasSuccessful() or args.require_tensors and outcome.skipped:raise RuntimeError('tests failed or required checks skipped')
        report['shell']=shell_checks()
        report['status']='PASS_LOCAL_CPU' if not outcome.skipped else 'PASS_WITH_SKIPS'
    except Exception as exc:
        import traceback
        report.update(error=repr(exc),traceback=traceback.format_exc())
    storage.write_json(args.report,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('tests','previous_runs')},ensure_ascii=False,indent=2))
    print('Report:',args.report)
    return 0 if report['status'].startswith('PASS') else 1


if __name__=='__main__':raise SystemExit(main())
