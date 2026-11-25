"""CPU submission regression only; no production assets or GPU jobs."""
import ast
import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
sys.path.insert(0, str(ROOT/'code/RSmol'))
from mizar_audio_cutoff1000 import config, manifest, submit
from mizar_layer_pair_search import storage
from validate_mizar_audio_cutoff1000 import shell_checks


def main():
    path = ROOT/'audit_reports/mizar_audio_cutoff1000_5090_validation.json'
    report = storage.read_json(path)
    previous = {k:v for k,v in report.items() if k != 'previous_runs'}
    if previous.get('status') == 'FAILED':
        report['previous_runs'] = [*report.get('previous_runs', []), previous]
    try:
        assert manifest.generator_sources() == report['manifest_generator_sources_before']
        report['manifest_generator_sources_unchanged'] = True
        report['manifest_source_count'] = len(manifest.generator_sources())
        for p in [*config.PACKAGE.glob('*.py'), Path(__file__), ROOT/'tests/test_mizar_audio_cutoff1000.py',
                  ROOT/'tests/validate_mizar_audio_cutoff1000.py']:
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p), feature_version=(3,10))
        suite = unittest.defaultTestLoader.loadTestsFromName('test_mizar_audio_cutoff1000.Contracts')
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        report['tests'] = {'run': result.testsRun, 'failures': len(result.failures),
                           'errors': len(result.errors), 'log': stream.getvalue()}
        print(stream.getvalue(), flush=True)
        assert result.wasSuccessful()
        for flags in (['--world-size','1','--partition','pdgpu-5090'],
                      ['--world-size','8','--partition','pdgpu-4090']):
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    submit.parse_args(flags)
                except SystemExit as exc:
                    assert exc.code == 2
                else:
                    raise AssertionError('invalid queue topology accepted')
        report['invalid_queue_topologies_rejected'] = 2
        report['shell'] = shell_checks()
        report['status'] = 'PASS_LOCAL_SUBMISSION_CHECKS'
        report.pop('error', None)
    except Exception as exc:
        report.update(status='FAILED', error=repr(exc))
        raise
    finally:
        storage.write_json(path, report)
    print(report['status'], path)


if __name__ == '__main__':
    main()
