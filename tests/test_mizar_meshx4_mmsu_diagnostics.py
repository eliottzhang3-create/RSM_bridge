"""Diagnose runner/prompt differences without changing production evaluation."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code/RSmol'), str(ROOT / 'tests')]
from mizar_mmsu_meshx4_diagnostics import probe
from mizar_s1_ced_mmsu_eval import dataset
import test_mizar_s1_ced_mmsu_eval as fixture
import evaluate_mizar_s1_ced_multilayer_benchmark as benchmark


class DiagnosticTests(unittest.TestCase):
    def args(self, path):
        return probe.resolved_args(probe.parse_args(['--output-dir', str(path)]))

    def test_precise_checkpoint_binding_diagnostic_directory_and_original_protocol(self):
        args = self.args('/diagnostic/new')
        self.assertTrue(str(args.checkpoint).replace('\\', '/').endswith(
            'formal_20261103_093224843011222/formal/checkpoints/model--step-34626.ckpt'))
        self.assertEqual(args.eval_variant, 'recursive_mesh_5_10x4_5')
        self.assertEqual(args.expected_geometry, (8, 4, 4))
        self.assertEqual(args.expected_step, 34626)
        row = fixture.record()
        variants = probe.prompts(row, benchmark)
        self.assertEqual(variants['mmsu_original'], dataset.prompt(row))
        self.assertEqual(variants['mmsu_lowercase'], dataset.prompt(row).lower())
        self.assertEqual(variants['mmau_format_lowercase'],
                         'which country? a) india b) none c) china d) japan')

    def test_raw_prompt_runs_are_unmodified_and_runner_comparison_detects_difference(self):
        import numpy as np
        class Wave(np.ndarray):
            def __new__(cls): return np.ones(10, dtype=np.float32).view(cls)
            def numpy(self): return np.asarray(self)
            def numel(self): return self.size
            def abs(self): return np.abs(self)
            def square(self): return np.square(self)
            def mean(self, *args, **kwargs): return np.mean(np.asarray(self), *args, **kwargs)
        # RMS result must expose Tensor.sqrt, while no torch is installed locally.
        class Mean:
            def sqrt(self): return 1.
        wave = Wave()
        wave.mean = lambda *args, **kwargs: Mean()
        class Squared:
            def mean(self): return Mean()
        wave.square = lambda: Squared()
        encoder = SimpleNamespace(tokenize=lambda *args: {
            'input_ids': np.ones((1, 256), dtype=np.int64), 'attention_mask': np.ones((1, 256), dtype=np.int64)})
        loaded = (None, None, None, encoder, None, {}, {})
        row = fixture.record()
        expected = {'generated_token_ids': [3, 0], 'generated_text': 'naive, grade <|endoftext|>'}
        other = SimpleNamespace(mmau=benchmark.mmau, generate=Mock(return_value=expected))
        with patch.object(probe.runtime, 'generate', return_value=expected) as generate:
            first = probe.probe_sample(loaded, row, wave, {}, other)
            self.assertTrue(first['same_input_runner_token_ids_equal'])
            self.assertEqual([call.args[1]['prompt'] for call in generate.call_args_list],
                             list(probe.prompts(row, benchmark).values()))
            other.generate.assert_called_once()
            self.assertEqual(other.generate.call_args.args[1]['prompt'], dataset.prompt(row).lower())
            self.assertTrue(all(call.args[1]['waveform'] is wave for call in generate.call_args_list))
            self.assertEqual(first['cases']['mmsu_original']['generation']['generated_text'], expected['generated_text'])
            other.generate.return_value = {'generated_token_ids': [4, 0], 'generated_text': 'other'}
            second = probe.probe_sample(loaded, row, wave, {}, other)
            self.assertFalse(second['same_input_runner_token_ids_equal'])

    def fixture(self, path):
        args = self.args(path)
        records = [{**fixture.record(i), 'id': key} for i, key in enumerate(probe.DEFAULT_IDS)]
        rows = [(i, {**record, 'audio': {'bytes': b'fake-audio'}}) for i, record in enumerate(records)]
        patches = [patch.object(probe.dataset, 'shard_paths', return_value=[]),
                   patch.object(probe.dataset, 'preflight', return_value=(records, {'synthetic': True})),
                   patch.object(probe.dataset, 'iter_rows', return_value=iter(rows)),
                   patch.object(probe.evaluate, 'identity', return_value={'synthetic_checkpoint': True}),
                   patch.object(probe.runtime, 'load_runtime', return_value=(None,) * 5 + ({'fixture': True}, None)),
                   patch.object(probe.runtime, 'decode_audio', return_value=(object(), {})),
                   patch.object(probe, 'probe_sample', side_effect=lambda loaded, row, *rest: {
                       'same_input_runner_token_ids_equal': True, 'cases': {}, 'id': row['id']}),
                   patch.object(probe, 'compare_audio', return_value={'mmsu_vs_mmau_bytes_reader_equal': True})]
        return args, patches

    def test_actual_orchestration_only_selected_three_and_existing_directory_refused(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'new_diagnostic'
            args, patches = self.fixture(path)
            with ExitStack() as stack, redirect_stdout(io.StringIO()):
                mocks = [stack.enter_context(item) for item in patches]
                result = probe.run(args)
                self.assertEqual(mocks[6].call_count, 3)
                self.assertEqual(result['status'], 'PASS')
                self.assertIsNone(result['benchmark_accuracy'])
                self.assertEqual(len(result['samples']), 3)
                self.assertEqual([sample['id'] for sample in result['samples']], list(probe.DEFAULT_IDS))
                self.assertIn('mizar_mmsu_meshx4_diagnostics/probe.py', result['diagnostic_sources'])
                before = (path / 'diagnostic_report.json').read_bytes()
                with self.assertRaises(FileExistsError): probe.run(args)
                self.assertEqual((path / 'diagnostic_report.json').read_bytes(), before)

    def test_runner_mismatch_is_reported_without_fabricating_benchmark_score(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory:
            args, patches = self.fixture(Path(directory) / 'new')
            with ExitStack() as stack, redirect_stdout(io.StringIO()):
                mocks = [stack.enter_context(item) for item in patches]
                mocks[6].side_effect = None
                mocks[6].return_value = {'same_input_runner_token_ids_equal': False, 'cases': {}, 'id': 'fixture'}
                result = probe.run(args)
                self.assertEqual(result['status'], 'DIFFERENCE_FOUND')
                self.assertIsNone(result['benchmark_accuracy'])

    def test_wrong_dataset_fails_before_model_load_and_saves_diagnostic_error(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'new'
            args, patches = self.fixture(path)
            with ExitStack() as stack, redirect_stdout(io.StringIO()):
                mocks = [stack.enter_context(item) for item in patches]
                mocks[1].return_value = ([], {'synthetic_wrong_dataset': True})
                with self.assertRaisesRegex(ValueError, 'exact three'):
                    probe.run(args)
                mocks[4].assert_not_called()
            report = json.loads((path / 'diagnostic_report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'FAILED')
            self.assertIn('exact three', report['error'])

    def test_real_cli_preview_needs_no_assets_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'not_created'
            result = subprocess.run([sys.executable, '-B', '-m', 'mizar_mmsu_meshx4_diagnostics.probe',
                '--output-dir', str(path), '--check-args'], cwd=ROOT / 'code/RSmol',
                capture_output=True, text=True, check=True)
            args = json.loads(result.stdout)
            self.assertEqual(args['expected_step'], 34626)
            self.assertEqual(args['expected_geometry'], [8, 4, 4])
            self.assertEqual(args['selected_ids'], list(probe.DEFAULT_IDS))
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
