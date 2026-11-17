"""Local CPU verification; synthetic results must never be called GPU validation."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/RSmol'))
from mizar_layer_pair_search import config, data, identity, storage


def shell_checks():
    bash = (Path(os.environ.get('ProgramFiles', 'C:/Program Files'))/'Git/bin/bash.exe'
            if os.name == 'nt' else Path(shutil.which('bash') or '/usr/bin/bash'))
    scripts = ['code/RSmol/mizar_layer_pair_search/run.sh', 'code/RSmol/run_mizar_layer_pair_search_8gpu_3090.sh']
    checks = []
    for script in scripts:
        run = subprocess.run([str(bash), '-n', script], cwd=ROOT, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f'Bash syntax failed: {script}: {run.stderr}')
        checks.append(script)
    # Real shell entry points, a fake conda activation, and a real parse-only
    # Python invocation. This cannot call vc, load assets, or launch CUDA.
    with tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP')) as directory:
        fake = Path(directory)/'conda/etc/profile.d/conda.sh'
        fake.parent.mkdir(parents=True)
        fake.write_text('conda() { test "$1" = activate && test "$2" = mellow_c8204d8; }\n', encoding='utf-8')
        args = ['--check-args', '--manifest', str(Path(directory)/'input with spaces/manifest.json'),
                '--output-dir', str(Path(directory)/'output with spaces')]
        env = dict(os.environ, RSMOL_CONDA_BASE=fake.parents[2].as_posix(), PYTHONUTF8='1')
        run = subprocess.run([str(bash), scripts[1], *args], cwd=ROOT, env=env,
                             capture_output=True, text=True, encoding='utf-8')
        if run.returncode:
            raise RuntimeError(f'Parse-only shell integration failed: {run.stderr}')
        parsed = json.loads(run.stdout)
        if parsed['submitted_jobs'] != 1 or parsed['vc_args'].count('submit') != 1:
            raise ValueError('single-job shell plan changed')
    return {'status': 'PASS', 'syntax': checks, 'real_shell_parse_only': True,
            'conda_activation': 'explicit fake fixture', 'vc_called': False, 'CUDA_started': False}


def reasonaqa_metadata(path):
    rows = storage.read_json(path)
    accepted, excluded = Counter(), Counter()
    for index, row in enumerate(rows):
        try:
            value = data.normalize_record('reasonaqa', index, row)
            accepted[value['category']] += 1
        except data.Ineligible as exc:
            excluded[str(exc)] += 1
    if sum(accepted.values()) < 25:
        raise ValueError('real ReasonAQA has fewer than 25 eligible MCQ rows')
    return {'source': storage.file_identity(path), 'source_rows': len(rows),
            'eligible_by_subtype': dict(accepted), 'eligible_rows': sum(accepted.values()),
            'excluded': dict(excluded), 'scope': 'real metadata parsing only; no tokenizer/audio/model'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'audit_reports/mizar_layer_pair_search_validation.json')
    parser.add_argument('--reasonaqa-json', type=Path)
    parser.add_argument('--require-tensors', action='store_true', help='fail if real Torch tests are skipped')
    args = parser.parse_args()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'FAILED',
              'scope': 'LOCAL_CPU_ONLY', 'production_gpu_run': 'NOT_RUN',
              'simulated379rounds': 'synthetic rows and runtime artifacts; tests actual finalization code',
              'dependencies': {name: importlib.util.find_spec(name) is not None for name in
                               ('torch', 'transformers', 'pyarrow', 'soundfile')}}
    try:
        sources = [*config.PACKAGE.glob('*.py'), ROOT/'tests/test_mizar_layer_pair_search.py', Path(__file__)]
        for source in sources:
            ast.parse(source.read_text(encoding='utf-8'), filename=str(source), feature_version=(3, 10))
        guard = identity.verify_guard()
        paths = ['code/RSmol/'+name for name in guard['sources_lf_sha256']]
        unchanged = subprocess.run(['git', 'diff', '--exit-code', '--quiet', 'HEAD', '--', *paths], cwd=ROOT)
        if unchanged.returncode:
            raise ValueError('reused reference sources differ from committed HEAD')
        report['static_checks'] = {'python310_AST_files': len(sources), 'reference_sources': len(paths),
                                   'reference_sources_equal_git_HEAD': True, 'LF_hashes': 'PASS'}
        stream = io.StringIO()
        suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_mizar_layer_pair_search.py')
        outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        report['tests'] = {'run': outcome.testsRun, 'passed': outcome.testsRun-len(outcome.skipped)-len(outcome.errors)-len(outcome.failures),
                           'skipped': [{'test': str(t), 'reason': reason} for t, reason in outcome.skipped],
                           'failures': len(outcome.failures), 'errors': len(outcome.errors), 'log': stream.getvalue()}
        if not outcome.wasSuccessful() or (args.require_tensors and outcome.skipped):
            raise RuntimeError('tests failed or required real tensor tests were skipped')
        report['shell'] = shell_checks()
        if args.reasonaqa_json:
            report['reasonaqa_metadata'] = reasonaqa_metadata(args.reasonaqa_json)
        report['status'] = 'PASS_LOCAL_CPU'
    except Exception as exc:
        report['error'] = repr(exc)
        import traceback
        report['traceback'] = traceback.format_exc()
    storage.write_json(args.report, report)
    print(json.dumps({k:v for k,v in report.items() if k != 'tests'}, ensure_ascii=False, indent=2))
    if 'tests' in report:
        print(report['tests']['log'])
    print('Report:', args.report)
    return 0 if report['status'] == 'PASS_LOCAL_CPU' else 1


if __name__ == '__main__':
    raise SystemExit(main())
