"""Run the new route and unchanged original HRM CPU regressions."""
from __future__ import annotations
import argparse
import importlib
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
    parser.add_argument('--report',type=Path,default=ROOT/'audit_reports/text_hrm_reinit3600_bp5_validation.json')
    args=parser.parse_args()
    from test_text_hrm_reinit3600_bp5 import ML
    if args.require_tensors and not ML: raise SystemExit('real CPU dependencies are required')
    # The original repository-wide historical inventory predates later approved
    # MMSU edits. Replace only that inventory assertion with this task's complete
    # 3280-file start snapshot; all original numerical/operational tests still run.
    superseded='test_text_hrm_h2l3.ContractTests.test_original_tracked_sources_unchanged'
    def flatten(suite):
        for item in suite:
            if isinstance(item,unittest.TestSuite): yield from flatten(item)
            else: yield item
    suite=unittest.TestSuite(test for name in ('test_text_hrm_reinit3600_bp5','test_text_hrm_h2l3')
        for test in flatten(unittest.defaultTestLoader.loadTestsFromName(name)) if test.id()!=superseded)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    versions={}
    for name in ('torch','transformers','numpy','pyarrow','safetensors','tokenizers'):
        try: versions[name]=getattr(importlib.import_module(name),'__version__','unknown')
        except ImportError: versions[name]='unavailable'
    report=dict(status=('PASS_LOCAL_WITH_SKIPS' if result.skipped else 'PASS_LOCAL_CPU') if result.wasSuccessful() else 'FAILED',
        tests_run=result.testsRun,passed=result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),
        skipped=[dict(test=str(t),reason=r) for t,r in result.skipped],
        failures=[dict(test=str(t),traceback=r) for t,r in result.failures+result.errors],
        python=platform.python_version(),versions=versions,real_tensors_available=ML,
        source_checkpoint='user_specified_remote_step3600_NOT_READ_LOCALLY',gpu_training_run=False,
        remote_jobs_submitted=False,production_weights_read=False)
    report['superseded_inventory_test']=dict(test=superseded,
        replacement='test_isolated_recipe_and_existing_sources_unchanged',
        reason='Original inventory predates approved MMSU edits; protect the full task-start checkout instead.')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    raise SystemExit(0 if result.wasSuccessful() and (not args.require_tensors or not result.skipped) else 1)


if __name__=='__main__': main()
