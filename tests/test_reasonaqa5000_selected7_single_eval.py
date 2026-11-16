"""Single-GPU submission and complete fixed-5000 flow with explicit fake models."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code/RSmol'), str(ROOT / 'tests')]
from mizar_s1_ced_reasonaqa5000_selected7_single_eval import config, evaluate, compare, submit
from mizar_s1_ced_reasonaqa5000_selected7_8gpu_eval import config as eight, model, runtime
from mizar_s1_ced_reasonaqa5000_eval import storage
from test_reasonaqa5000_eval import Fixture as SampleFixture, generation


class RegistryTests(unittest.TestCase):
    def test_exact_seven_same_models_manifest_protocol_and_isolated_single_outputs(self):
        items = submit.planned(submit.parse_args(['all']))
        self.assertEqual(len(items), 7)
        self.assertEqual(config.EXCLUDED, ('official_mizar',))
        self.assertEqual(config.DEFAULT_MANIFEST, eight.DEFAULT_MANIFEST)
        self.assertIs(evaluate.runtime, runtime)
        names, outputs = [], []
        for name, argv, args in items:
            original = model.parse_args(['--target', name])
            for key in ('checkpoint', 'runtime_config', 'run_root', 'route_root',
                        'eval_variant', 'expected_geometry', 'expected_step'):
                self.assertEqual(getattr(args, key), getattr(original, key), (name, key))
            self.assertEqual(args.expected_geometry, (8, 8, 4))
            self.assertEqual(args.expected_step, 17313)
            self.assertEqual(submit.parse_args([args.eval_variant]).target, name)
            self.assertTrue(args.output_dir.name.endswith('_1gpu4090_v1'))
            self.assertNotEqual(args.output_dir, original.output_dir)
            outputs.append(args.output_dir)
            command = submit.command(name, argv, 'very_long_batch_tag_' * 20)
            for flag, value in (('-p', 'pdgpu-4090'), ('-g', '1'), ('-c', '8'), ('-m', '32G'), ('-n', '1')):
                self.assertEqual(command[command.index(flag) + 1], value)
            names.append(command[command.index('-j') + 1])
            self.assertLessEqual(len(names[-1]), 60)
            self.assertTrue(names[-1].isascii())
            self.assertNotIn('torch.distributed.run', ' '.join(command))
        self.assertEqual(len(set(names)), 7)
        self.assertEqual(len(set(outputs)), 7)

    def test_reject_official_wrong_scope_overrides_and_shared_output(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (['official_mizar'], ['init0904'], ['all', '--output-dir', '/shared']):
                with self.assertRaises(SystemExit):
                    submit.parse_args(argv)
            for flag in ('--checkpoint', '--run-root', '--runtime-config', '--expected-step'):
                with self.assertRaises(SystemExit):
                    evaluate.parse_args(['--target', 'base', flag, 'wrong'])
        args = evaluate.parse_args(['--target', 'base', '--output-dir=/a path/eval'])
        self.assertEqual(args.output_dir, Path('/a path/eval').resolve())

    def test_single_process_loop_preserves_old_inference_calls_and_resume(self):
        def body(path):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
            # The only implementation change inside the loop is the helper's module qualifier.
            return ast.dump(ast.parse(ast.unparse(node).replace('model.summarize(', 'summarize(')), include_attributes=False)
        self.assertEqual(body(config.PACKAGE / 'evaluate.py'),
                         body(config.ROOT / 'mizar_s1_ced_reasonaqa5000_eval/evaluate.py'))
        self.assertNotIn('torch.distributed', (config.PACKAGE / 'evaluate.py').read_text())


class WorkflowTests(SampleFixture):
    def args(self, target='base', output='eval'):
        argv = ['--target', target, '--manifest', str(self.bundle / 'manifest.json'),
                '--output-dir', str(self.directory / output)]
        for flag, value in self.roots.items():
            argv.extend(['--' + flag, str(value)])
        args = evaluate.parse_args(argv)
        args.checkpoint = self.directory / 'fixture.ckpt'
        args.checkpoint.write_bytes(b'not a real checkpoint')
        args.runtime_config = self.directory / 'fixture_runtime.json'
        storage.write_json(args.runtime_config, {'fixture': True})
        return args

    def run_fixture(self, args, callback=None, forbid_loader=False):
        loaded = (None, None, None, None, None, {'status': 'PASS', 'mock_model': True}, None)
        with mock.patch.object(runtime, 'load_runtime',
                               side_effect=AssertionError('model load forbidden') if forbid_loader else None,
                               return_value=loaded) as loader, \
             mock.patch.object(runtime, 'generate', side_effect=callback or (lambda _, row, **kw: generation(row))), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            report = evaluate.run(args)
        return report, loader.call_count

    def test_full5000_exact_match_compare_and_rescore_no_model(self):
        args = self.args()
        seen = []
        def generate(_, row, **kwargs):
            seen.append(row['id'])
            return generation(row, 'B' if len(seen) == 1 else None)
        report, calls = self.run_fixture(args, generate)
        self.assertEqual(report['status'], 'PASS', report['fatal_error'])
        self.assertEqual(calls, 1)
        self.assertEqual(seen, self.manifest['sample_ids'])
        self.assertEqual(report['accuracy']['overall'], {'correct': 4999, 'total': 5000, 'accuracy': 4999/5000})
        self.assertEqual(report['execution'], config.EXECUTION)
        self.assertNotIn('parallel_evaluation', report['run_identity'])
        self.assertFalse((args.output_dir / 'ranks').exists())
        raw = (args.output_dir / 'raw_generations.jsonl').read_bytes()
        again, calls = self.run_fixture(args, forbid_loader=True)
        self.assertEqual((again['status'], calls), ('PASS', 0), again['fatal_error'])
        self.assertEqual((args.output_dir / 'raw_generations.jsonl').read_bytes(), raw)
        identity, accuracy, common, inputs = compare.read_run(args.output_dir)
        self.assertEqual(identity['execution'], config.EXECUTION)
        self.assertEqual(common['manifest_sha256'], self.manifest['manifest_sha256'])
        self.assertEqual(len(inputs), 5000)
        rows = compare.compare_runs([('base', args.output_dir)], self.directory / 'compare.csv')
        self.assertEqual(rows[0]['overall_correct'], 4999)
        self.assertEqual(rows[0]['overall_total'], 5000)
        accuracy['overall']['correct'] = 5000
        storage.write_json(args.output_dir / 'accuracy.json', accuracy)
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            compare.read_run(args.output_dir)

    def test_interrupted13_resume4987_and_refuse_identity_change(self):
        args = self.args('mesh48_lora_r8')
        seen = []
        def interrupt(_, row, **kwargs):
            if len(seen) == 13:
                raise RuntimeError('explicit interruption')
            seen.append(row['id'])
            return generation(row)
        report, calls = self.run_fixture(args, interrupt)
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(report['inference_coverage']['generated'], 13)
        self.assertFalse((args.output_dir / 'accuracy.json').exists())
        with (args.output_dir / 'raw_generations.jsonl').open('ab') as stream:
            stream.write(b'{"partial":')
        resumed = []
        def finish(_, row, **kwargs):
            resumed.append(row['id'])
            return generation(row)
        report, _ = self.run_fixture(args, finish)
        self.assertEqual(report['status'], 'PASS', report['fatal_error'])
        self.assertEqual(resumed, self.manifest['sample_ids'][13:])
        self.assertEqual(len(resumed), 4987)
        before = (args.output_dir / 'evaluation_report.json').read_bytes()
        args.checkpoint.write_bytes(b'changed checkpoint')
        report, calls = self.run_fixture(args, forbid_loader=True)
        self.assertEqual((report['status'], report['output_claimed'], calls), ('FAILED', False, 0))
        self.assertEqual((args.output_dir / 'evaluation_report.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
