"""Local ReasonAQA fixtures plus existing benchmark regressions; no production inference."""
from __future__ import annotations

import argparse
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
from mizar_s1_ced_reasonaqa_eval import storage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'audit_reports/mizar_s1_ced_reasonaqa_eval_validation.json')
    args = parser.parse_args()
    git_bin = Path('C:/Program Files/Git/bin')
    if os.name == 'nt' and (git_bin / 'bash.exe').is_file():
        os.environ['PATH'] = str(git_bin) + os.pathsep + os.environ.get('PATH', '')
    started = time.monotonic()
    modules = ['test_mizar_s1_ced_reasonaqa_eval', 'test_mizar_s1_ced_reasonaqa_shell',
               'test_mizar_s1_ced_mmsu_eval', 'test_mizar_s1_ced_mmsu_dependencies', 'test_mizar_s1_ced_mmsu_shell',
               'test_mizar_mmsu_selected5', 'test_mizar_selected4_eval', 'test_mizar_selected4_eval_tensor',
               'test_mizar_s1_ced_benchmark_eval', 'test_mizar_s1_ced_recursive_eval',
               'test_mizar_s1_ced_recursive_mesh_eval', 'test_mizar_s1_ced_multilayer_eval',
               'test_mizar_s1_ced_eval_extensions', 'test_mizar_s1_ced_input_ca_eval']
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'scope': 'local synthetic fixtures and existing regressions; no remote assets or GPU inference',
              'status': 'PASS' if result.wasSuccessful() else 'FAILED', 'utc': datetime.now(timezone.utc).isoformat(),
              'modules': modules, 'tests_run': result.testsRun,
              'passed': result.testsRun - len(result.skipped) - len(result.errors) - len(result.failures),
              'skipped': [{'test': str(test), 'reason': reason} for test, reason in result.skipped],
              'failures': [{'test': str(test), 'traceback': trace} for test, trace in result.failures],
              'errors': [{'test': str(test), 'traceback': trace} for test, trace in result.errors]}
    baseline = json.loads((ROOT / 'code/RSmol/mizar_s1_ced_reasonaqa_eval/reference_audit.json').read_text())
    report['existing_benchmark_sources'] = {
        'checked': len(baseline['files']), 'changed': [path for path, digest in baseline['files'].items()
                                                     if storage.file_sha(ROOT / path) != digest]}
    report['python_syntax'] = []
    sources = list((ROOT / 'code/RSmol/mizar_s1_ced_reasonaqa_eval').glob('*.py'))
    sources.extend(ROOT.glob('tests/*mizar_s1_ced_reasonaqa*.py'))
    for path in sources:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        report['python_syntax'].append(path.relative_to(ROOT).as_posix())
    report['bash_syntax'] = []
    bash = shutil.which('bash')
    for relative in ('code/RSmol/mizar_s1_ced_reasonaqa_eval/run.sh',
                     'code/RSmol/run_mizar_s1_ced_small_reasonaqa_eval_5090.sh',
                     'code/RSmol/run_mizar_reasonaqa_selected7_5090.sh'):
        checked = subprocess.run([bash, '-n', str(ROOT / relative)], capture_output=True, text=True) if bash else None
        report['bash_syntax'].append({'path': relative, 'status': 'PASS' if checked and checked.returncode == 0 else 'FAILED',
                                      'stderr': checked.stderr if checked else 'Bash unavailable'})
    if (report['existing_benchmark_sources']['changed']
            or any(item['status'] != 'PASS' for item in report['bash_syntax'])):
        report['status'] = 'FAILED'
    report['elapsed_seconds'] = time.monotonic() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    storage.write_json(args.report, report)
    print(json.dumps({'status': report['status'], 'tests_run': report['tests_run'], 'passed': report['passed'],
                      'skipped': len(report['skipped']), 'report': str(args.report)}))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
