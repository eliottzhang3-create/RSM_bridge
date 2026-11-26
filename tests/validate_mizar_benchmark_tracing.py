"""CPU numerical verification and explicit full1000 synthetic orchestration."""
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
sys.path.insert(0,str(ROOT/'code/RSmol'));sys.path.insert(0,str(ROOT/'tests'))
from mizar_layer_pair_search import storage
from mizar_benchmark_tracing import config


def shell_checks():
    bash='C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else 'bash'
    scripts=['code/RSmol/mizar_benchmark_tracing/run.sh','code/RSmol/run_mizar_mmau_tracing_8gpu_5090.sh',
        'code/RSmol/run_mizar_mmar_tracing_8gpu_5090.sh']
    for script in scripts:
        result=subprocess.run([bash,'-n',script],cwd=ROOT,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
    plans=[]
    with tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP')) as temp:
        folder=Path(temp);fake=folder/'conda/etc/profile.d/conda.sh';fake.parent.mkdir(parents=True)
        fake.write_text('conda() { test "$1" = activate && test "$2" = mellow_c8204d8; }\n',encoding='utf-8')
        env=dict(os.environ,RSMOL_CONDA_BASE=fake.parents[2].as_posix(),PYTHONUTF8='1')
        for script,task in zip(scripts[1:],config.TASKS):
            result=subprocess.run([bash,script,'--check-args','--output-dir',str(folder/(task+"'s output"))],cwd=ROOT,
                env=env,capture_output=True,text=True,encoding='utf-8')
            if result.returncode:raise RuntimeError(result.stderr)
            plan=json.loads(result.stdout);argv=plan['vc_args']
            if plan['task']!=task or plan['submitted_jobs']!=1:raise ValueError('wrong task or submission count')
            for flag,value in (('-p','pdgpu-5090'),('-g','8'),('-c','64'),('-m','256G'),('-n','1')):
                if argv[argv.index(flag)+1]!=value:raise ValueError('wrong queue/resources')
            plans.append({'task':task,'GPU_count':8,'actual_vc_called':False})
        # Execute actual Python -> bash -> vc argument passing with a recording
        # shell fixture. The cluster service and GPU compute command are unused.
        fake.write_text('conda() { test "$1" = activate && test "$2" = mellow_c8204d8; }\n'
            'vc() { python -c \'import sys,json; print(json.dumps(sys.argv[1:]))\' "$@"; }\nexport -f vc\n',encoding='utf-8')
        helper=folder/'submit_check.py'
        helper.write_text("from pathlib import Path\nfrom unittest.mock import patch\nfrom mizar_benchmark_tracing import submit\nimport sys\nwith patch.object(submit.config,'ROOT',Path(sys.argv[1])):\n    raise SystemExit(submit.main(['--task',sys.argv[2],'--output-dir',sys.argv[3]]))\n",encoding='utf-8')
        env['PYTHONPATH']=os.pathsep.join([str(config.ROOT),os.environ.get('PYTHONPATH','')])
        for task in config.TASKS:
            result=subprocess.run([bash,'-c','source "$1"; python -X utf8 -B "$2" "$3" "$4" "$5"','tracing-vc-fixture',
                fake.as_posix(),helper.as_posix(),folder.as_posix(),task,(folder/(task+"'s output")).as_posix()],cwd=ROOT,
                env=env,capture_output=True,text=True,encoding='utf-8')
            if result.returncode:raise RuntimeError(result.stderr)
            report=json.loads(result.stdout)
            if report['submitted_jobs']!=1 or json.loads(report['stdout'])!=report['vc_args']:raise ValueError('vc argument chain changed')
    return {'status':'PASS','scripts':scripts,'plans':plans,'actual_Bash_and_mock_vc_chain':True,'conda':'activation fixture'}


class Tee(io.StringIO):
    def write(self,text):
        sys.stdout.write(text);sys.stdout.flush();return super().write(text)


def main():
    target=ROOT/'audit_reports/mizar_benchmark_tracing_validation.json'
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'FAILED','remote_GPU_run':False,
        'production_weights_read':False,'jobs_submitted':False,
        'scope':'Real release tokenizer/local2000 metadata, real30-layer Llama BF16/eager/SDPA capture/patch and real tensor cache. Full1000 screening/distributed orchestration/model outputs/bulk result finalizers use explicit synthetic fixtures.'}
    if target.exists():
        old=storage.read_json(target);report['previous_runs']=[*old.get('previous_runs',[]),{k:v for k,v in old.items() if k!='previous_runs'}]
    try:
        paths=[*config.PACKAGE.glob('*.py'),ROOT/'tests/test_mizar_benchmark_tracing.py',Path(__file__)]
        for p in paths:ast.parse(p.read_text(encoding='utf-8'),filename=str(p),feature_version=(3,10))
        report['python310_AST_files']=len(paths)
        report['versions']={n:importlib.import_module(n).__version__ for n in ('torch','transformers','numpy','matplotlib','soundfile')}
        stream=Tee();suite=unittest.TestSuite()
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_mizar_benchmark_tracing'))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_mizar_reasonaqa_tracing.RealNumerics'))
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        report['tests']={'run':result.testsRun,'passed':result.testsRun-len(result.errors)-len(result.failures)-len(result.skipped),
            'errors':len(result.errors),'failures':len(result.failures),'skipped':len(result.skipped),'log':stream.getvalue()}
        if not result.wasSuccessful() or result.skipped:raise RuntimeError('tests failed or skipped')
        report['shell']=shell_checks();report['status']='PASS_LOCAL_CPU'
    except Exception as exc:
        import traceback
        report.update(error=repr(exc),traceback=traceback.format_exc())
    storage.write_json(target,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('tests','previous_runs')},ensure_ascii=False,indent=2),flush=True)
    return 0 if report['status']=='PASS_LOCAL_CPU' else 1


if __name__=='__main__':raise SystemExit(main())
