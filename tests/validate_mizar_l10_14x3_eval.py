"""Offline CPU acceptance; synthetic full-size flows are explicitly labeled."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import sysconfig
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dependency-dir',type=Path,action='append',default=[])
    p.add_argument('--pattern',default='test_mizar_l10_14x3_eval*.py')
    a=p.parse_args()
    sys.path[:0]=[str(x.resolve()) for x in a.dependency_dir]+[str(ROOT/'tests'),str(ROOT/'code/RSmol')]
    definition=importlib.util.spec_from_file_location('code',Path(sysconfig.get_path('stdlib'))/'code.py')
    module=importlib.util.module_from_spec(definition);definition.loader.exec_module(module);sys.modules['code']=module
    # Existing Windows torchaudio 2.9 extension is not ABI-compatible with Torch2.15.
    # Only disable that unused extension in this test process; real Python transforms/SoundFile stay active.
    original=importlib.util.find_spec
    def find(name,*args,**kwargs):
        return None if name=='torchaudio.lib._torchaudio' else original(name,*args,**kwargs)
    with patch('importlib.util.find_spec',find):
        import torchaudio
    import torch,transformers
    torch.set_num_threads(1)
    path=ROOT/'audit_reports/mizar_l10_14x3_eval_validation.json'
    report=json.loads(path.read_text()) if path.exists() else {'validation_runs':[]}
    start=time.monotonic()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=a.pattern)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    run=dict(status='PASS_LOCAL_CPU' if result.wasSuccessful() and not result.skipped else 'FAILED',
        tests=result.testsRun,passed=result.testsRun-len(result.errors)-len(result.failures)-len(result.skipped),
        skipped=[str(t) for t,_ in result.skipped],
        failures=[dict(test=str(t),traceback=s) for t,s in result.errors+result.failures],
        seconds=time.monotonic()-start)
    report['validation_runs'].append(run)
    report.update(latest=run,status=run['status'],versions=dict(python=sys.version,torch=torch.__version__,
        transformers=transformers.__version__,torchaudio=torchaudio.__version__),
        remote_checkpoint_loading='NOT_RUN',cuda='NOT_RUN',job_submission='MOCK_ONLY',
        fixtures='Full-size runners use explicit model/audio/parquet fixtures; native Llama and SoundFile tests are real CPU.')
    for f in [*(ROOT/'code/RSmol/mizar_l10_14x3_eval').glob('*.py'),*(ROOT/'tests').glob('*mizar_l10_14x3_eval*.py')]:
        ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,10))
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(run))
    return int(run['status']!='PASS_LOCAL_CPU')


if __name__=='__main__':raise SystemExit(main())
