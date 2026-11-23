"""Record offline acceptance; production weights, remote GPU and vc are not run."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'code/RSmol')]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-tensors',action='store_true')
    args=parser.parse_args()
    path=ROOT/'audit_reports/ced_eval_20261123_validation.json'
    report=json.loads(path.read_text(encoding='utf-8'));start=time.monotonic()
    suite=unittest.TestLoader().discover(str(ROOT/'tests'),pattern='test_ced_eval_20261123*.py')
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_mmsu_compact_lowercase'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    failed=not result.wasSuccessful() or (args.require_tensors and bool(result.skipped))
    status='FAILED' if failed else 'PASS_LOCAL_WITH_SKIPS' if result.skipped else 'PASS_LOCAL'
    run={'status':status,'tests_run':result.testsRun,'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        'skipped':[{'test':str(t),'reason':reason} for t,reason in result.skipped],
        'failures':[{'test':str(t),'traceback':trace} for t,trace in result.failures+result.errors],
        'elapsed_seconds':time.monotonic()-start}
    package=ROOT/'code/RSmol/ced_eval_20261123'
    for p in [*package.glob('*.py'),*(ROOT/'tests').glob('*ced_eval_20261123*.py')]:ast.parse(p.read_text(),feature_version=(3,10))
    report.setdefault('validation_runs',[]).append(run)
    report.update(status=status,latest=run,python=platform.python_version(),
        dependencies={n:bool(importlib.util.find_spec(n)) for n in ('torch','transformers','safetensors','numpy','pyarrow','torchaudio')},
        sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in sorted(package.iterdir()) if p.is_file()},
        full_size_synthetic={'unique_jobs':8,'reasonaqa5000':2,'mmau1000':2,'mmar1000':2,'mmsu5000':2,
            'inference':'explicit model/audio/parquet/CUDA fixtures','bulk_fsync':'mocked; original durable storage unchanged'},
        actual_audit_scope='actual audit_model for both targets and both older read2 entrypoints; real pinned validator/parameter_audit/LoRA validator source on metadata module fixtures, not mocked audit results',
        source_snapshots='real local Git archive and actual code_digest for the two targets',
        job_submission='MOCK_ONLY',production_checkpoint_loading='NOT_RUN',remote_gpu='NOT_RUN')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:run[k] for k in ('status','tests_run','passed','elapsed_seconds')}))
    return int(failed)

if __name__=='__main__':raise SystemExit(main())
