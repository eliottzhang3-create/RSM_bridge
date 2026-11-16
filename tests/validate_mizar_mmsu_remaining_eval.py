"""Validate the new 27-target dispatcher and unchanged MMSU protocol locally."""
from __future__ import annotations

import argparse
import ast
import hashlib
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
from mizar_s1_ced_mmsu_eval import storage

PACKAGE = ROOT / 'code/RSmol/mizar_s1_ced_mmsu_remaining_eval'
MODULES = (
    'test_mizar_mmsu_remaining_eval', 'test_mizar_mmsu_remaining_shell',
    'test_mizar_s1_ced_mmsu_eval', 'test_mizar_s1_ced_mmsu_dependencies',
    'test_mizar_s1_ced_mmsu_shell', 'test_mizar_mmsu_rescore', 'test_mizar_mmsu_selected5',
    'test_mizar_init0904_lora_eval', 'test_mizar_init0904_lora_eval_tensor',
    'test_mizar_selected4_eval', 'test_mizar_selected4_eval_tensor',
    'test_mizar_s1_ced_benchmark_eval', 'test_mizar_s1_ced_recursive_eval',
    'test_mizar_s1_ced_recursive_mesh_eval', 'test_mizar_s1_ced_multilayer_eval',
    'test_mizar_s1_ced_eval_extensions', 'test_mizar_s1_ced_input_ca_eval',
)


def source_audit():
    reference = json.loads((PACKAGE / 'reference_audit.json').read_text(encoding='utf-8'))
    raw_changed, lf_changed = [], []
    for name, expected in reference['files'].items():
        path = ROOT / name
        if not path.is_file() or storage.file_sha(path) != expected['raw_sha256']:
            raw_changed.append(name)
        if not path.is_file() or storage.text_sha(path) != expected['lf_sha256']:
            lf_changed.append(name)
    history = reference['rm_history']
    current = (ROOT / 'RM.md').read_bytes()
    history_ok = (len(current) >= history['bytes'] and
                  hashlib.sha256(current[-history['bytes']:]).hexdigest() == history['raw_sha256'])
    return {'files_checked': len(reference['files']), 'raw_bytes_preserved': not raw_changed,
            'raw_changed': raw_changed, 'lf_content_preserved': not lf_changed, 'lf_changed': lf_changed,
            'rm_history_bytes_preserved': history_ok,
            # CRLF checkout conversion is permitted only if canonical content is identical.
            'status': 'PASS' if not lf_changed and history_ok else 'FAILED'}


def syntax_audit():
    files = [*PACKAGE.glob('*.py'), Path(__file__), ROOT / 'tests/test_mizar_mmsu_remaining_eval.py',
             ROOT / 'tests/test_mizar_mmsu_remaining_shell.py']
    for path in files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 10))
    shell_files = [PACKAGE / 'run.sh', ROOT / 'code/RSmol/run_mizar_mmsu_remaining_4090.sh']
    bash = shutil.which('bash')
    if not bash:
        raise RuntimeError('Bash required for shell syntax and end-to-end fixture validation')
    for path in shell_files:
        subprocess.run([bash, '-n', str(path)], check=True, capture_output=True, text=True)
    return {'status': 'PASS', 'python310_files': len(files), 'bash_files': len(shell_files)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'audit_reports/mizar_mmsu_remaining_eval_validation.json')
    args = parser.parse_args(argv)
    git_bin = Path('C:/Program Files/Git/bin')
    if os.name == 'nt' and (git_bin / 'bash.exe').is_file():
        os.environ['PATH'] = str(git_bin) + os.pathsep + os.environ.get('PATH', '')
    started = time.monotonic()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in MODULES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'status': 'PASS_LOCAL_WITH_SKIPS' if result.skipped else 'PASS_LOCAL',
              'scope': 'local fixtures only; vc/weights/GPU inference simulated; no production jobs or assets',
              'utc': datetime.now(timezone.utc).isoformat(), 'modules': MODULES, 'tests_run': result.testsRun,
              'passed': result.testsRun - len(result.skipped) - len(result.errors) - len(result.failures),
              'skipped': [{'test': str(test), 'reason': reason} for test, reason in result.skipped],
              'errors': [{'test': str(test), 'traceback': trace} for test, trace in result.errors],
              'failures': [{'test': str(test), 'traceback': trace} for test, trace in result.failures]}
    if not result.wasSuccessful():
        report['status'] = 'FAILED'
    for name, check in (('preservation', source_audit), ('syntax', syntax_audit)):
        try:
            report[name] = check()
        except Exception as exc:
            report[name] = {'status': 'FAILED', 'error': repr(exc)}
        if report[name]['status'] != 'PASS':
            report['status'] = 'FAILED'
    report['elapsed_seconds'] = time.monotonic() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    storage.write_json(args.report, report)
    print(json.dumps({'status': report['status'], 'passed': report['passed'], 'skipped': len(result.skipped),
                      'report': str(args.report)}))
    return int(report['status'] == 'FAILED')


if __name__ == '__main__':
    raise SystemExit(main())
