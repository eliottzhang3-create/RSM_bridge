"""Local offline acceptance only; no production weights, remote access or real vc."""
import argparse
import ast
import hashlib
import json
import pathlib
import sys
import time
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'code/RSmol')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=pathlib.Path,default=ROOT/'audit_reports/text71427_ced_eval_validation.json')
    parser.add_argument('--require-tensors',action='store_true')
    args=parser.parse_args()
    started=time.monotonic()
    suite=unittest.TestLoader().discover(str(ROOT/'tests'),pattern='test_text71427_ced_eval*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    package=ROOT/'code/RSmol/text71427_ced_eval'
    sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
             for p in sorted(package.glob('*.py'))}
    for path in package.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,10))
    failed=not result.wasSuccessful() or (args.require_tensors and bool(result.skipped))
    report={'status':'FAILED' if failed else 'PASS_LOCAL_WITH_SKIPS' if result.skipped else 'PASS_LOCAL',
            'tests_run':result.testsRun,'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
            'skipped':[{'test':str(test),'reason':reason} for test,reason in result.skipped],
            'failures':[{'test':str(test),'traceback':trace} for test,trace in result.failures+result.errors],
            'elapsed_seconds':time.monotonic()-started,'sources':sources,
            'full_sized_fixture_coverage':{'models':6,'reasonaqa5000':6,'mmau':6,'mmar':6,'mmsu':6,
                                         'inference':'explicit synthetic model/audio/parquet fixtures',
                                         'bulk_fsync':'mocked; existing durable storage implementation unchanged'},
            'remote_gpu':'NOT_RUN','production_checkpoint_loading':'NOT_RUN',
            'job_submission':'MOCK_ONLY','original_sources_protected':len(json.loads((package/'protected_sources.json').read_text())['sources'])}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','tests_run','passed','elapsed_seconds','original_sources_protected')},ensure_ascii=False))
    return 1 if failed else 0


if __name__=='__main__':raise SystemExit(main())
