"""Offline acceptance; does not read production weights, run CUDA, or submit jobs."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'code/RSmol')]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--require-tensors', action='store_true')
    parser.add_argument('--include-reference-regressions', action='store_true')
    args = parser.parse_args(argv)
    if bool(args.annotations) != bool(args.metadata): parser.error('supply --annotations and --metadata together')
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_myeval_mmar8*.py')
    if args.include_reference_regressions:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_ced_eval_20261123_loading'))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_text71427_ced_eval_tensors.ObserverTests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failed = not result.wasSuccessful() or (args.require_tensors and bool(result.skipped))
    from myeval_mmar8 import dataset, benchmark, scoring, storage
    package = ROOT / 'code/RSmol/myeval_mmar8'
    assets = {'status': 'NOT_REQUESTED', 'audio_decoding': 'NOT_RUN', 'production_inference': 'NOT_RUN'}
    if args.annotations:
        rows = dataset.load_annotations(args.annotations)
        official = benchmark.mmar.load_mmar_records(args.metadata)
        records = dataset.expand(rows, official, args.metadata.parent / 'mmar-audio')
        groups = {r['question_key']: r for r in records if r['type'] == 'close-ended'}
        collisions = []
        for key, r in groups.items():
            tokens = lambda text: set(re.findall(r'\b\w+\b', text.lower()))
            same = [c for c in r['choices'] if c != r['answer'] and tokens(c) == tokens(r['answer'])]
            if same: collisions.append({'question_key': key, 'answer': r['answer'], 'same_token_set_incorrect_options': same})
        assets.update(status='ANNOTATION_AND_METADATA_BINDING_VERIFIED', audio_rows=len(rows),
            original_questions=67, closed_questions=53, closed_permutations=734, open_questions=14,
            generations_per_model=len(records), normalized_lf_sha256=dataset.normalized_sha(args.annotations),
            official_token_set_collisions=collisions)
    syntax = []
    for path in [*package.glob('*.py'), *(ROOT / 'tests').glob('*myeval_mmar8*.py')]:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 10))
        syntax.append(path.relative_to(ROOT).as_posix())
    status = 'FAILED' if failed else 'PASS_LOCAL_WITH_SKIPS' if result.skipped else 'PASS_LOCAL'
    run = {'status': status, 'tests_run': result.testsRun,
           'passed': result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
           'skipped': [{'test': str(t), 'reason': reason} for t, reason in result.skipped],
           'failures': [{'test': str(t), 'traceback': trace} for t, trace in result.failures + result.errors],
           'elapsed_seconds': time.monotonic() - started}
    path = ROOT / 'audit_reports/myeval_mmar8_validation.json'
    report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'validation_runs': []}
    report['validation_runs'].append(run)
    report.update(status=status, latest=run, python=platform.python_version(),
        dependencies={n: bool(importlib.util.find_spec(n)) for n in ('torch', 'transformers', 'safetensors', 'numpy', 'torchaudio')},
        python310_syntax=syntax, annotations=assets,
        source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                       for p in sorted(package.iterdir()) if p.is_file()},
        full_size_synthetic={'custom_models': 3, 'generations_each': 748, 'extra_mmar': 1000,
                             'model_audio_cuda': 'explicit synthetic fixtures', 'bulk_fsync': 'mocked in bulk fixtures'},
        cpu_numerical='NumPy executing actual production _mixed_interaction/_read/_write; not Torch autograd',
        tensor_test='real tiny CPU model when dependencies are present, otherwise explicitly skipped',
        remote_weights='NOT_READ', gpu='NOT_RUN', real_vc_submission='NOT_RUN', commit_push='NOT_RUN')
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': status, 'tests': result.testsRun, 'passed': run['passed'],
                      'skipped': len(result.skipped), 'report': str(path)}))
    return int(failed)


if __name__ == '__main__': raise SystemExit(main())
