"""Validate isolated ReasonAQA5000 and existing benchmark regressions without production assets."""
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
from mizar_s1_ced_reasonaqa5000_eval import config, storage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'audit_reports/reasonaqa5000_eval_validation.json')
    args = parser.parse_args(argv)
    git = Path('C:/Program Files/Git/bin')
    if os.name == 'nt' and (git / 'bash.exe').is_file():
        os.environ['PATH'] = str(git) + os.pathsep + os.environ.get('PATH', '')
    started = time.monotonic()
    modules = ['test_reasonaqa5000_eval', 'test_reasonaqa5000_shell',
        'test_mizar_s1_ced_reasonaqa_eval.DataTests', 'test_mizar_s1_ced_reasonaqa_eval.ScoreTests',
        'test_mizar_s1_ced_reasonaqa_eval.NumericalFixtureTests',
        'test_mizar_s1_ced_reasonaqa_eval.RuntimeTests.test_real_torch_audio_and_prompt_to_generation',
        'test_mizar_s1_ced_mmsu_eval', 'test_mizar_s1_ced_mmsu_dependencies', 'test_mizar_s1_ced_mmsu_shell',
        'test_mizar_init0904_lora_eval', 'test_mizar_init0904_lora_eval_tensor', 'test_mizar_mmsu_rescore',
        'test_mizar_mmsu_selected5', 'test_mizar_selected4_eval', 'test_mizar_selected4_eval_tensor',
        'test_mizar_s1_ced_benchmark_eval', 'test_mizar_s1_ced_recursive_eval',
        'test_mizar_s1_ced_recursive_mesh_eval', 'test_mizar_s1_ced_multilayer_eval',
        'test_mizar_s1_ced_eval_extensions', 'test_mizar_s1_ced_input_ca_eval']
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromName(name) for name in modules))
    report = {'status': 'PASS_LOCAL_WITH_SKIPS' if result.skipped else 'PASS_LOCAL',
              'scope': 'synthetic local data, mocked GPU model/vc, real Bash; no production inference or real job submissions',
              'utc': datetime.now(timezone.utc).isoformat(), 'modules': modules, 'tests_run': result.testsRun,
              'passed': result.testsRun - len(result.skipped) - len(result.errors) - len(result.failures),
              'skipped': [{'test': str(t), 'reason': why} for t, why in result.skipped],
              'failures': [{'test': str(t), 'traceback': why} for t, why in result.failures],
              'errors': [{'test': str(t), 'traceback': why} for t, why in result.errors]}
    audit = json.loads((config.PACKAGE / 'reference_audit.json').read_text())
    changed = [path for path, sha in audit['files'].items() if storage.file_sha(ROOT / path) != sha]
    report['original_sources'] = {'checked': len(audit['files']), 'changed': changed}
    report['python_syntax'] = []
    for path in [*config.PACKAGE.glob('*.py'), *ROOT.glob('tests/*reasonaqa5000*.py')]:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3,10))
        report['python_syntax'].append(path.relative_to(ROOT).as_posix())
    report['bash_syntax'] = []
    for relative in ('code/RSmol/generate_reasonaqa_test5000.sh', 'code/RSmol/run_reasonaqa5000_eval_5090.sh',
                     'code/RSmol/mizar_s1_ced_reasonaqa5000_eval/run.sh'):
        checked = subprocess.run([shutil.which('bash'), '-n', str(ROOT / relative)], capture_output=True, text=True)
        report['bash_syntax'].append({'path': relative, 'returncode': checked.returncode, 'stderr': checked.stderr})
    if not result.wasSuccessful() or changed or any(item['returncode'] for item in report['bash_syntax']):
        report['status'] = 'FAILED'
    report['elapsed_seconds'] = time.monotonic() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    storage.write_json(args.report, report)
    print(json.dumps({'status': report['status'], 'tests_run': result.testsRun, 'passed': report['passed'],
                      'skipped': len(result.skipped), 'report': str(args.report)}))
    return 1 if report['status'] == 'FAILED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
