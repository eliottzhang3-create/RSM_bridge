"""Run portable HRM regression and record actual dependencies/skips, never GPU PASS."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-tensors',action='store_true')
    parser.add_argument('--report',type=Path,default=ROOT/'audit_reports/text_hrm_h2l3_validation.json')
    args=parser.parse_args()
    from test_text_hrm_h2l3 import ML
    if args.require_tensors and not ML: raise SystemExit('real tensor dependencies required')
    suite=unittest.defaultTestLoader.loadTestsFromName('test_text_hrm_h2l3')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    versions={}
    for name in ('torch','transformers','numpy','safetensors','pyarrow','tokenizers'):
        if importlib.util.find_spec(name):
            module=__import__(name); versions[name]=getattr(module,'__version__','unknown')
    report=dict(status=('PASS_LOCAL_WITH_SKIPS' if result.skipped else 'PASS_LOCAL_CPU') if result.wasSuccessful() else 'FAILED',
        tests_run=result.testsRun,passed=result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),
        skipped=[{'test':str(t),'reason':r} for t,r in result.skipped],
        failures=[{'test':str(t),'traceback':r} for t,r in result.failures+result.errors],
        python=platform.python_version(),versions=versions,real_tensors_available=ML,
        gpu_training_run=False,production_weights_read=False,remote_jobs_submitted=False)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    raise SystemExit(0 if result.wasSuccessful() and (not args.require_tensors or not result.skipped) else 1)


if __name__=='__main__': main()
