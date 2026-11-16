"""Validate the independent 4090 single-GPU route and preservation of old sources."""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code/RSmol'), str(ROOT / 'tests')]
from mizar_s1_ced_reasonaqa5000_selected7_single_eval import config
from mizar_s1_ced_reasonaqa5000_eval import storage


def main():
    git = Path('C:/Program Files/Git/bin')
    if os.name == 'nt' and (git / 'bash.exe').is_file():
        os.environ['PATH'] = str(git) + os.pathsep + os.environ.get('PATH', '')
    started = time.monotonic()
    previous = config.ROOT / 'mizar_s1_ced_reasonaqa5000_selected7_8gpu_eval'
    frozen = json.loads((previous / 'reference_audit.json').read_text())['files']
    protected = {str(path.relative_to(ROOT).as_posix()): storage.file_sha(path)
                 for path in [*previous.glob('*.*'), config.ROOT / 'run_reasonaqa5000_selected7_8gpu_3090.sh',
                              *ROOT.glob('tests/*reasonaqa5000_selected7_8gpu*.py')]}
    modules = ['test_reasonaqa5000_selected7_single_eval', 'test_reasonaqa5000_selected7_single_shell',
               'test_reasonaqa5000_selected7_8gpu_eval.PartitionAndDeviceTests',
               'test_mizar_s1_ced_reasonaqa_eval.DataTests', 'test_mizar_s1_ced_reasonaqa_eval.ScoreTests']
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromName(name) for name in modules))
    report = {'status': 'PASS_LOCAL_WITH_SKIPS' if result.skipped else 'PASS_LOCAL',
              'scope': 'synthetic data and fake models, real Bash and mocked vc; no production GPU or submissions',
              'utc': datetime.now(timezone.utc).isoformat(), 'modules': modules, 'tests_run': result.testsRun,
              'passed': result.testsRun - len(result.skipped) - len(result.errors) - len(result.failures),
              'skipped': [{'test': str(test), 'reason': why} for test, why in result.skipped],
              'failures': [{'test': str(test), 'traceback': why} for test, why in result.failures],
              'errors': [{'test': str(test), 'traceback': why} for test, why in result.errors]}
    hashes = {**frozen, **protected}
    changed = [path for path, sha in hashes.items() if storage.file_sha(ROOT / path) != sha]
    report['original_sources'] = {'checked': len(hashes), 'changed': changed, 'previous_seven_source_hashes': protected}
    report['python_syntax'] = []
    for path in [*config.PACKAGE.glob('*.py'), *ROOT.glob('tests/*reasonaqa5000_selected7_single*.py')]:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 10))
        report['python_syntax'].append(path.relative_to(ROOT).as_posix())
    report['bash_syntax'] = []
    for path in (config.PACKAGE / 'run.sh', config.ROOT / 'run_reasonaqa5000_selected7_1gpu_4090.sh'):
        checked = subprocess.run([shutil.which('bash'), '-n', str(path)], capture_output=True, text=True)
        report['bash_syntax'].append({'path': path.relative_to(ROOT).as_posix(),
                                      'returncode': checked.returncode, 'stderr': checked.stderr})
    if not result.wasSuccessful() or changed or any(item['returncode'] for item in report['bash_syntax']):
        report['status'] = 'FAILED'
    report['elapsed_seconds'] = time.monotonic() - started
    output = ROOT / 'audit_reports/reasonaqa5000_selected7_single_eval_validation.json'
    storage.write_json(output, report)
    print(json.dumps({'status': report['status'], 'tests_run': result.testsRun, 'passed': report['passed'],
                      'skipped': len(result.skipped), 'original_sources': len(hashes), 'report': str(output)}))
    return 1 if report['status'] == 'FAILED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
