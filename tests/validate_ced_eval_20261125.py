"""Offline acceptance only: native CPU tensors, synthetic datasets, mock vc."""
import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import sysconfig
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-dir',type=Path,action='append',default=[])
    parser.add_argument('--require-tensors',action='store_true')
    parser.add_argument('--pattern',default='test_ced_eval_20261125*.py')
    args=parser.parse_args()
    sys.path[:0]=[str(p.resolve(strict=True)) for p in args.dependency_dir]
    sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'code/RSmol')]
    # Preserve stdlib code.InteractiveConsole when Torch imports pdb.
    spec=importlib.util.spec_from_file_location('code',Path(sysconfig.get_path('stdlib'))/'code.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules['code']=module
    path=ROOT/'audit_reports/ced_eval_20261125_validation.json'
    report=json.loads(path.read_text(encoding='utf-8'));start=time.monotonic()
    suite=unittest.TestLoader().discover(str(ROOT/'tests'),pattern=args.pattern)
    if args.pattern=='test_ced_eval_20261125*.py':
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_mmsu_compact_lowercase'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    failed=not result.wasSuccessful() or args.require_tensors and bool(result.skipped)
    status='FAILED' if failed else 'PASS_LOCAL_WITH_SKIPS' if result.skipped else 'PASS_LOCAL_CPU'
    run={'status':status,'pattern':args.pattern,'tests_run':result.testsRun,
        'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        'skipped':[{'test':str(t),'reason':reason} for t,reason in result.skipped],
        'failures':[{'test':str(t),'traceback':trace} for t,trace in result.failures+result.errors],
        'elapsed_seconds':time.monotonic()-start}
    package=ROOT/'code/RSmol/ced_eval_20261125'
    for p in [*package.glob('*.py'),*(ROOT/'tests').glob('*ced_eval_20261125*.py')]:ast.parse(p.read_text(),feature_version=(3,10))
    versions={}
    for name in ('torch','transformers','numpy','safetensors','torchaudio','soundfile','pyarrow'):
        try:versions[name]=getattr(importlib.import_module(name),'__version__','unknown')
        except ImportError:versions[name]='unavailable'
    report.setdefault('validation_runs',[]).append(run)
    report.update(status=status,latest=run,python=platform.python_version(),versions=versions,
        sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in sorted(package.iterdir()) if p.is_file()},
        full_size_synthetic={'unique_jobs':8,'reasonaqa5000':2,'mmau1000':2,'mmar1000':2,'mmsu5000':2,
            'scope':'native CPU model tests separate; full-sized runners use explicit model/audio/parquet/CUDA fixtures',
            'bulk_fsync':'mocked; original durable storage implementation unchanged'},
        source_snapshots='real Git archive; fixed training/shared hashes; no branch switch',
        production_checkpoint_loading='NOT_RUN',remote_gpu='NOT_RUN',job_submission='MOCK_ONLY')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:run[k] for k in ('status','pattern','tests_run','passed','elapsed_seconds')}))
    return int(bool(failed))

if __name__=='__main__':raise SystemExit(main())
