"""Verify CPU-only PSR regrouping, real plotting, and the conda shell entry."""
import argparse
import ast
from datetime import datetime,timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
sys.path.insert(0,str(ROOT/'code/RSmol'))
from mizar_layer_pair_search import storage


def shell_check():
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else 'bash'
    script = 'code/RSmol/run_mizar_ablation_regroup_cpu.sh'
    run = subprocess.run([bash,'-n',script],cwd=ROOT,capture_output=True,text=True)
    if run.returncode: raise RuntimeError(run.stderr)
    with tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP')) as directory:
        fake = Path(directory)/'conda/etc/profile.d/conda.sh'
        fake.parent.mkdir(parents=True)
        marker = Path(directory)/'conda_activation.txt'
        fake.write_text('conda() { test "$1" = activate && test "$2" = mellow_c8204d8 && printf "%s" "$2" > "$PSR_CONDA_MARKER"; }\n',encoding='utf-8')
        env = dict(os.environ,RSMOL_CONDA_BASE=fake.parents[2].as_posix(),PSR_CONDA_MARKER=marker.as_posix(),PYTHONUTF8='1')
        run = subprocess.run([bash,script,'--check-args','--source-dir',str(Path(directory)/"source's space")],
            cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
        if run.returncode: raise RuntimeError(run.stderr)
        parsed = json.loads(run.stdout)
        if "source's space" not in parsed['source_dir'] or marker.read_text() != 'mellow_c8204d8':
            raise ValueError('shell activation/parameter forwarding failed')
    return {'status':'PASS','bash_syntax':True,'real_shell_parameter_chain':True,
            'conda':'explicit fake activation of mellow_c8204d8','vc_called':False,'GPU_used':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'audit_reports/mizar_ablation_regroup_validation.json')
    args = parser.parse_args()
    report = {'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'FAILED',
              'scope':'Synthetic complete 29-condition summaries; actual CPU aggregation and PNG rendering',
              'remote_results_read':False,'jobs_submitted':False,'GPU_used':False}
    try:
        paths = [*Path(ROOT/'code/RSmol/mizar_ablation_regroup').glob('*.py'),ROOT/'tests/test_mizar_ablation_regroup.py',Path(__file__)]
        for path in paths: ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,10))
        report['python310_AST_files'] = len(paths)
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_mizar_ablation_regroup'))
        report['tests'] = {'run':result.testsRun,'passed':result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),
            'skipped':[{'test':str(t),'reason':why} for t,why in result.skipped],
            'failures':len(result.failures),'errors':len(result.errors),'log':stream.getvalue()}
        print(stream.getvalue(),flush=True)
        if not result.wasSuccessful() or result.skipped: raise RuntimeError('regression failed or real plotting unavailable')
        report['shell'] = shell_check()
        import matplotlib
        report['matplotlib'] = matplotlib.__version__
        report['status'] = 'PASS_LOCAL_CPU'
    except Exception as exc:
        import traceback
        report.update(error=repr(exc),traceback=traceback.format_exc())
    storage.write_json(args.report,report)
    print(json.dumps({k:v for k,v in report.items() if k!='tests'},ensure_ascii=False,indent=2))
    print('Report:',args.report)
    return 0 if report['status']=='PASS_LOCAL_CPU' else 1


if __name__ == '__main__':
    raise SystemExit(main())
