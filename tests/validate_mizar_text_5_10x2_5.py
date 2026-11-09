"""Validate the isolated Mizar text route locally, without production assets."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-tensors',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT/'tests'))
    if os.name=='nt':
        os.environ['PATH']='C:/Program Files/Git/bin'+os.pathsep+os.environ['PATH']
    source_files=list((ROOT/'code/RSmol/mizar_text_5_10x2_5').glob('*.py'))
    source_files+=list((ROOT/'code/RSmol/scripts').glob('*mizar_text_5_10x2_5*'))
    source_files+=list((ROOT/'code/RSmol').glob('run_stage4_mizar_text_5_10x2_5*'))
    source_files+=list((ROOT/'tests').glob('*mizar_text_5_10x2_5*.py'))
    source_files+=[ROOT/'code/RSmol/mizar_text_5_10x2_5'/name for name in ('README.md','source_guard.json')]
    for path in source_files:
        if path.suffix=='.py':
            compile(ast.parse(path.read_text(encoding='utf-8')),str(path),'exec')
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mizar_text_5_10x2_5*.py')
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    success=result.wasSuccessful() and not (args.require_tensors and result.skipped)
    report={'status':('PASS_LOCAL_WITH_SKIPS' if result.skipped else 'PASS_LOCAL') if success else 'FAIL',
            'time_utc':datetime.now(timezone.utc).isoformat(),'run':result.testsRun,
            'passed':result.testsRun-len(result.skipped)-len(result.errors)-len(result.failures),
            'skipped':[{'test':test.id(),'reason':reason} for test,reason in result.skipped],
            'failures':[{'test':test.id(),'traceback':trace} for test,trace in result.failures],
            'errors':[{'test':test.id(),'traceback':trace} for test,trace in result.errors],
            'production_assets_read':False,'remote_conversion_verified':False,
            'gpu_training_verified':False,'eight_rank_resume_verified':False,
            'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in source_files if p.is_file()}}
    out=ROOT/'audit_reports/mizar_text_5_10x2_5_validation.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f"{report['status']}: passed={report['passed']}, skipped={len(result.skipped)}; {out}")
    return 0 if success else 1


if __name__=='__main__':
    raise SystemExit(main())
