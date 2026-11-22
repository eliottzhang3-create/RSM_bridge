"""Persist local offline evidence; never contacts a server or invokes real vc."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'code/RSmol')]

def main():
    report_path=ROOT/'audit_reports/ced_eval_20261122_validation.json'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    start=time.monotonic()
    suite=unittest.TestLoader().discover(str(ROOT/'tests'),pattern='test_ced_eval_20261122*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    package=ROOT/'code/RSmol/ced_eval_20261122'
    for path in package.glob('*.py'):ast.parse(path.read_text(),feature_version=(3,10))
    status='FAILED' if not result.wasSuccessful() else 'PASS_LOCAL_WITH_SKIPS' if result.skipped else 'PASS_LOCAL'
    run={'status':status,'tests_run':result.testsRun,
        'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        'skipped':[{'test':str(t),'reason':r} for t,r in result.skipped],
        'failures':[{'test':str(t),'traceback':trace} for t,trace in result.failures+result.errors],
        'elapsed_seconds':time.monotonic()-start}
    report.setdefault('validation_runs',[]).append(run)
    report.update(status=status,latest=run,
        sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in sorted(package.iterdir()) if p.is_file()},
        full_size_synthetic={'unique_jobs':19,'reasonaqa5000':4,'mmau1000':4,'mmar1000':4,'mmsu5000':7,
            'inference':'explicit model/audio/parquet/CUDA fixtures','bulk_fsync':'mocked; production storage unchanged'},
        source_snapshots='real Git archive and actual training contract code_digest for all seven',
        remote_gpu='NOT_RUN',production_checkpoint_loading='NOT_RUN',job_submission='MOCK_ONLY',
        prior_test_iterations=[{'tests':15,'passed':13,'errors':2,
            'detail':'test fixture initially used choices instead of choice_a..d and referenced a nonexistent helper; corrected tests, production generation unchanged'},
            {'tests':27,'passed':26,'failures':1,
             'detail':'sources.py was edited to remove cached-reader lock contention during the long workflow test; the existing identity guard correctly rejected rescore after that edit. Re-run with frozen production sources.'}])
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:run[k] for k in ('status','tests_run','passed','elapsed_seconds')}))
    return int(not result.wasSuccessful())

if __name__=='__main__':raise SystemExit(main())
