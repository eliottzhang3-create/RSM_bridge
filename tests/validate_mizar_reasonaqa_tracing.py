"""Run local causal-tracing validation without submitting jobs or reading model weights."""
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
from mizar_reasonaqa_tracing import config


def shell_checks():
    bash='C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else 'bash'
    scripts=['code/RSmol/mizar_reasonaqa_tracing/run.sh','code/RSmol/run_mizar_reasonaqa_tracing_8gpu_5090.sh']
    for script in scripts:
        r=subprocess.run([bash,'-n',script],cwd=ROOT,capture_output=True,text=True)
        if r.returncode:raise RuntimeError(r.stderr)
    with tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP')) as temp:
        directory=Path(temp);fake=directory/'conda/etc/profile.d/conda.sh';fake.parent.mkdir(parents=True)
        fake.write_text('conda() { test "$1" = activate && test "$2" = mellow_c8204d8; }\n',encoding='utf-8')
        env=dict(os.environ,RSMOL_CONDA_BASE=fake.parents[2].as_posix(),PYTHONUTF8='1')
        r=subprocess.run([bash,scripts[1],'--check-args','--output-dir',str(directory/"a's result")],cwd=ROOT,
                         env=env,capture_output=True,text=True,encoding='utf-8')
        if r.returncode:raise RuntimeError(r.stderr)
        plan=json.loads(r.stdout);argv=plan['vc_args']
        for flag,val in (('-p','pdgpu-5090'),('-g','8'),('-c','64'),('-m','256G'),('-n','1')):
            if argv[argv.index(flag)+1]!=val:raise ValueError('incorrect resources')
        if plan['submitted_jobs']!=1:raise ValueError('must be exactly one submitted job')
        # Execute the real submit Python/Bash argument chain with a recording
        # vc function. No cluster service is contacted.
        fake.write_text('conda() { test "$1" = activate && test "$2" = mellow_c8204d8; }\n'
                        'vc() { python -c \'import sys,json; print(json.dumps(sys.argv[1:]))\' "$@"; }\n'
                        'export -f vc\n',encoding='utf-8')
        py="from pathlib import Path;from unittest.mock import patch;from mizar_reasonaqa_tracing import submit;import sys;\nwith patch.object(submit.config,'ROOT',Path(sys.argv[1])): raise SystemExit(submit.main(['--output-dir',sys.argv[2]]))"
        bootstrap=directory/'submit_check.py';bootstrap.write_text(py,encoding='utf-8')
        env['PYTHONPATH']=os.pathsep.join([str(config.ROOT),os.environ.get('PYTHONPATH','')])
        # source exports the recording function into the child's bash -c.
        cmd='source "$1"; python -X utf8 -B "$2" "$3" "$4"'
        r=subprocess.run([bash,'-c',cmd,'tracing-shell-fixture',fake.as_posix(),bootstrap.as_posix(),
                          directory.as_posix(),(directory/"a's result").as_posix()],cwd=ROOT,
                          env=env,capture_output=True,text=True,encoding='utf-8')
        if r.returncode:raise RuntimeError(r.stderr)
        submitted=json.loads(r.stdout);sent=json.loads(submitted['stdout'])
        if sent!=submitted['vc_args'] or submitted['submitted_jobs']!=1:
            raise ValueError('Bash vc arguments differ from planned single submission')
    return {'status':'PASS','scripts':scripts,'actual_shell_parse_only':True,'conda':'explicit activation fixture','actual_vc_called':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,default=ROOT/'audit_reports/mizar_reasonaqa_tracing_validation.json')
    a=p.parse_args()
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'FAILED',
        'production_weights_read':False,'remote_GPU_run':False,'jobs_submitted':False,
        'scope':'real CPU 30-layer Llama eager/SDPA/BF16, exact hidden state capture/patch/cache, real release tokenizer and existing5000 metadata; audio encoder, distributed eight-rank scheduling and full30000 results explicitly synthetic'}
    if a.report.exists():
        old=storage.read_json(a.report)
        report['previous_runs']=[*old.get('previous_runs',[]),{k:v for k,v in old.items() if k!='previous_runs'}]
    try:
        paths=[*config.PACKAGE.glob('*.py'),ROOT/'tests/test_mizar_reasonaqa_tracing.py',Path(__file__)]
        for path in paths:ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,10))
        report['python310_AST_files']=len(paths)
        report['versions']={n:importlib.import_module(n).__version__ for n in ('torch','transformers','numpy','matplotlib','torchaudio','soundfile')}
        stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('test_mizar_reasonaqa_tracing')
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        report['tests']={'run':result.testsRun,'passed':result.testsRun-len(result.errors)-len(result.failures)-len(result.skipped),
                         'errors':len(result.errors),'failures':len(result.failures),'skipped':len(result.skipped),'log':stream.getvalue()}
        print(stream.getvalue(),flush=True)
        if not result.wasSuccessful() or result.skipped:raise RuntimeError('tests failed or skipped')
        report['shell']=shell_checks();report['status']='PASS_LOCAL_CPU'
    except Exception as exc:
        import traceback
        report.update(error=repr(exc),traceback=traceback.format_exc())
    storage.write_json(a.report,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('tests','previous_runs')},ensure_ascii=False,indent=2),flush=True)
    return 0 if report['status']=='PASS_LOCAL_CPU' else 1


if __name__=='__main__':raise SystemExit(main())
