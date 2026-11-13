"""ReasonAQA policy, durable orchestration and decoder dispatch with explicit fixtures."""
from __future__ import annotations

import ast
import copy
from contextlib import nullcontext
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/RSmol'))
from mizar_s1_ced_reasonaqa_eval import config, dataset, evaluate, runtime, scoring, storage, compare


def row(subtype='AudioCaps-MCQ.json', answer='B) A person snoring', **changes):
    return {'taskname': 'audiocaps', 'filepath1': 'AudioCapsLarger/test/example.wav', 'filepath2': '',
            'caption1': 'DO NOT SEND CAPTION', 'caption2': '', 'input': 'Sound? A) talking B) snoring',
            'answer': answer, 'subtype': subtype, **changes}


def generation(record, text=None):
    inputs = {'prompt_token_ids': [7] * 256, 'audio_half_sha256': ['a' * 64, 'b' * 64]}
    text = record['original_record']['answer'] + '<|endoftext|>' if text is None else text
    return {'generated_text': text, 'generated_text_raw': text, 'generated_token_ids': [12, 0],
            'generated_token_count': 2, 'stop_reason': 'eos_token', 'max_new_tokens': 300,
            'prefix_shape': [1, 383, 576], 'prompt_used': record['original_record']['input'].lower(),
            'input_fingerprint_sha256': storage.object_sha(inputs), **inputs}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = SimpleNamespace(audiocaps_root=self.root / 'audiocaps', clotho_root=self.root / 'clotho',
                                    clotho_aqa_root=self.root / 'aqa')
        for name in ('audiocaps', 'clotho', 'aqa'):
            (self.root / name).mkdir()

    def write_rows(self, rows):
        path = self.root / 'test.json'
        storage.write_json(path, rows)
        return path

    def audio(self, root, name):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'explicit audio path fixture, not real audio')
        return path

    def run_args(self, name='output'):
        self.audio(self.args.audiocaps_root, 'test/example.wav')
        self.audio(self.args.clotho_root, 'evaluation/Freezing Rain.wav')
        self.audio(self.args.clotho_aqa_root, 'dripping.wav')
        test_json = self.write_rows([row(), row('Clotho-MCQ.json', 'B) Light drizzle',
            filepath1='ClothoV21/evaluation/Freezing Rain.wav'), row('ClothoAQA.json', 'no',
            filepath1='ClothoAQA\\audio_files\\dripping.wav'), row('Other.json')])
        route = self.root / 'training_route'
        route.mkdir(exist_ok=True)
        (route / 'model.py').write_text('# explicit local fixture\n')
        checkpoint = self.root / 'model.ckpt'
        checkpoint.write_bytes(b'fixture checkpoint; never passed to real loader')
        runtime_json = self.root / 'runtime.json'
        storage.write_json(runtime_json, {'fixture': True})
        return evaluate.parse_args(['--eval-variant', 'base', '--checkpoint', str(checkpoint),
            '--runtime-config', str(runtime_json), '--output-dir', str(self.root / name),
            '--route-root', str(route), '--expected-step', '17313', '--expected-geometry', '8,8,4',
            '--test-json', str(test_json), '--audiocaps-root', str(self.args.audiocaps_root),
            '--clotho-root', str(self.args.clotho_root), '--clotho-aqa-root', str(self.args.clotho_aqa_root)])

    def run_fixture(self, args, callback=None):
        loaded = (None, None, None, None, None, {'status': 'PASS', 'fixture': True}, None)
        with mock.patch.object(runtime, 'load_runtime', return_value=loaded) as loader, \
             mock.patch.object(runtime, 'generate', side_effect=callback or (lambda _, r, **kw: generation(r))):
            result = evaluate.run(args)
        return result, loader.call_count


class DataTests(Fixture):
    def test_selection_order_duplicates_and_excluded_counts(self):
        rows = [row(), row('ClothoAQA.json', ' Yes '), row('ClothoAQA.json', 'two'),
                row('Clotho-MCQ.json'), row('ClothoAQA.json', 'NO'), row('AudioCaps-Open.json'), row()]
        selected, audit = dataset.select_records(self.write_rows(rows))
        self.assertEqual([r['source_row_index'] for r in selected], [0, 1, 3, 4, 6])
        self.assertEqual(selected[-1]['id'], 'reasonaqa_test:6')
        self.assertEqual(audit['selected_category_counts'],
                         {'audiocaps_mcq': 2, 'clotho_mcq': 1, 'clotho_aqa_yesno': 2})
        self.assertEqual(audit['excluded_counts'], {'AudioCaps-Open.json': 1, 'ClothoAQA_non_yesno': 1})
        self.assertEqual(selected[0]['original_record'], rows[0])

    def test_selected_bad_rows_fail_and_empty_selection_fails(self):
        for rows in ([row(answer='')], [row(filepath1='')], [row(filepath2=None)],
                     [row(input=None)], [row('Other.json')], {}):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                dataset.select_records(self.write_rows(rows))

    def test_paths_backslashes_spaces_and_original_rows(self):
        paths = [self.audio(self.args.audiocaps_root, 'test/example.wav'),
                 self.audio(self.args.clotho_root, 'evaluation/Freezing Rain.wav'),
                 self.audio(self.args.clotho_aqa_root, 'dripping.wav')]
        raw = [row(), row('Clotho-MCQ.json', filepath1='ClothoV21/evaluation/Freezing Rain.wav'),
               row('ClothoAQA.json', 'no', filepath1='ClothoAQA\\audio_files\\dripping.wav')]
        selected, _ = dataset.select_records(self.write_rows(raw))
        prepared = dataset.prepare_records(selected, self.args)
        self.assertEqual([r['audio1']['path'] for r in prepared], [str(p.resolve()) for p in paths])
        self.assertEqual([r['original_record'] for r in prepared], raw)
        self.assertTrue(all(r['audio2'] is None for r in prepared))

    def test_fallback_unique_then_ambiguous_and_missing(self):
        self.audio(self.args.clotho_aqa_root, 'Froge__cafe!.wav')
        resolver = dataset.AudioResolver(self.args)
        self.assertTrue(resolver.resolve('ClothoAQA/Frogé café.wav', 'clotho_aqa_yesno')['path'].endswith('Froge__cafe!.wav'))
        self.audio(self.args.clotho_aqa_root, 'Froge-cafe.flac')
        with self.assertRaisesRegex(FileNotFoundError, 'ambiguous'):
            dataset.AudioResolver(self.args).resolve('ClothoAQA/Frogé café.wav', 'clotho_aqa_yesno')
        with self.assertRaises(FileNotFoundError):
            resolver.resolve('AudioCapsLarger/test/missing.wav', 'audiocaps_mcq')

    def test_unsafe_paths_outside_root_and_multiple_matches_fail(self):
        resolver = dataset.AudioResolver(self.args)
        for path in ('../escape.wav', 'C:\\audio.wav', ''):
            with self.subTest(path=path), self.assertRaises(ValueError):
                resolver.resolve(path, 'audiocaps_mcq')
        outside = self.audio(self.root, 'outside.wav')
        with self.assertRaises(ValueError):
            resolver.resolve(outside.as_posix(), 'audiocaps_mcq')
        self.audio(self.args.clotho_aqa_root, 'drip.wav')
        self.audio(self.args.clotho_aqa_root, 'audio_files/drip.wav')
        with self.assertRaisesRegex(FileNotFoundError, 'ambiguous'):
            resolver.resolve('ClothoAQA/audio_files/drip.wav', 'clotho_aqa_yesno')


class ScoreTests(Fixture):
    def test_only_full_answer_exact_match(self):
        target = 'B) A person snoring'
        for text, expected in [(' b) A  person\nSNORING <|endoftext|> ', True), ('B', False),
                               ('A person snoring', False), ('Answer: B) A person snoring', False),
                               ('B) A person snoring.', False), ('', False),
                               ('B) A person snoring<|endoftext|><|endoftext|>', False)]:
            with self.subTest(text=text):
                self.assertEqual(scoring.normalize(text) == scoring.normalize(target), expected)
        self.assertNotEqual(scoring.normalize('no.'), scoring.normalize('no'))
        self.assertEqual(scoring.normalize(' No<|endoftext|> '), 'no')

    def test_micro_denominator_keeps_empty_and_incorrect(self):
        selected, _ = dataset.select_records(self.write_rows([row(), row(), row('ClothoAQA.json', 'no')]))
        saved = [{'record': r, 'generation': generation(r, text)} for r, text in
                 zip(selected, ['b) a person snoring', 'B', ''])]
        result, details = scoring.build_results(saved, 3)
        self.assertEqual(result['overall'], {'correct': 1, 'total': 3, 'accuracy': 1 / 3})
        self.assertEqual(result['by_category']['clotho_aqa_yesno']['total'], 1)
        self.assertIsNone(result['by_category']['clotho_mcq']['accuracy'])
        self.assertEqual(details[1]['generated_text_raw'], 'B')
        with self.assertRaises(ValueError):
            scoring.build_results(saved[:2], 3)


class PersistenceTests(Fixture):
    def test_full_run_then_rescore_never_loads_model(self):
        args = self.run_args()
        report, calls = self.run_fixture(args)
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual(calls, 1)
        self.assertEqual(report['accuracy']['overall']['total'], 3)
        self.assertEqual(report['accuracy']['overall']['accuracy'], 1)
        before = (args.output_dir / 'raw_generations.jsonl').read_bytes()
        again, calls = self.run_fixture(args)
        self.assertEqual(again['status'], 'PASS', again)
        self.assertEqual(calls, 0)
        self.assertEqual(before, (args.output_dir / 'raw_generations.jsonl').read_bytes())

    def test_interrupted_resume_and_partial_tail_progress_repair(self):
        args = self.run_args()
        seen = []
        def fail(_, record, **kwargs):
            seen.append(record['id'])
            if len(seen) == 2:
                raise RuntimeError('intentional fixture interruption')
            return generation(record)
        report, _ = self.run_fixture(args, fail)
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(report['inference_coverage']['generated'], 1)
        with (args.output_dir / 'raw_generations.jsonl').open('ab') as stream:
            stream.write(b'{"crash_tail":')
        (args.output_dir / 'progress.jsonl').write_bytes(b'stale progress')
        resumed = []
        def finish(_, record, **kwargs):
            resumed.append(record['source_row_index'])
            return generation(record)
        report, _ = self.run_fixture(args, finish)
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual(resumed, [1, 2])
        self.assertEqual(len((args.output_dir / 'progress.jsonl').read_text().splitlines()), 3)

    def test_complete_corruption_never_truncated_or_silently_skipped(self):
        args = self.run_args()
        self.run_fixture(args)
        raw = args.output_dir / 'raw_generations.jsonl'
        raw.write_bytes(raw.read_bytes() + b'broken complete record\n')
        before = raw.read_bytes()
        report, calls = self.run_fixture(args)
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(calls, 0)
        self.assertEqual(raw.read_bytes(), before)

    def test_identity_changes_do_not_overwrite_existing_report(self):
        for field in ('checkpoint', 'runtime_config', 'test_json'):
            with self.subTest(field=field):
                args = self.run_args(field)
                self.run_fixture(args)
                before = (args.output_dir / 'evaluation_report.json').read_bytes()
                path = getattr(args, field)
                path.write_bytes(path.read_bytes() + b' ')
                report, calls = self.run_fixture(args)
                self.assertEqual(report['status'], 'FAILED')
                self.assertFalse(report['output_claimed'])
                self.assertEqual(calls, 0)
                self.assertEqual((args.output_dir / 'evaluation_report.json').read_bytes(), before)

    def test_model_input_and_generation_tampering_rejected(self):
        args = self.run_args()
        self.run_fixture(args)
        raw = args.output_dir / 'raw_generations.jsonl'
        saved = [json.loads(line) for line in raw.read_text().splitlines()]
        saved[0]['generation']['prompt_token_ids'][0] += 1
        saved[0]['generation_sha256'] = storage.object_sha(saved[0]['generation'])
        storage.write_jsonl(raw, saved)
        report, calls = self.run_fixture(args)
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(calls, 0)

    def test_unowned_output_and_process_lock(self):
        args = self.run_args()
        args.output_dir.mkdir()
        (args.output_dir / 'old.txt').write_text('keep')
        report, calls = self.run_fixture(args)
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(calls, 0)
        with storage.output_lock(args.output_dir):
            with self.assertRaises(RuntimeError):
                with storage.output_lock(args.output_dir):
                    self.fail('second writer acquired lock')

    def test_comparison_checks_raw_scores_and_inputs(self):
        args = self.run_args()
        self.run_fixture(args)
        other = copy.copy(args)
        other.output_dir = self.root / 'other'
        self.run_fixture(other)
        output = self.root / 'comparison.csv'
        rows = compare.compare_runs([('a', args.output_dir), ('b', other.output_dir)], output)
        self.assertEqual(len(rows), 2)
        self.assertIn('overall_accuracy', output.read_text(encoding='utf-8-sig'))
        score_path = other.output_dir / 'accuracy.json'
        score = json.loads(score_path.read_text())
        score['overall']['correct'] = 0
        storage.write_json(score_path, score)
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            compare.compare_runs([('b', other.output_dir)], self.root / 'bad.csv')
        self.assertFalse((self.root / 'bad.csv').exists())
        # Build a valid but different model input, with internally consistent hashes.
        self.run_fixture(other)
        raw = other.output_dir / 'raw_generations.jsonl'
        saved = [json.loads(line) for line in raw.read_text().splitlines()]
        g = saved[0]['generation']
        g['audio_half_sha256'][0] = 'c' * 64
        g['input_fingerprint_sha256'] = storage.object_sha({key: g[key] for key in ('prompt_token_ids', 'audio_half_sha256')})
        saved[0]['generation_sha256'] = storage.object_sha(g)
        storage.write_jsonl(raw, saved)
        with self.assertRaisesRegex(ValueError, 'model inputs differ'):
            compare.compare_runs([('a', args.output_dir), ('b', other.output_dir)], self.root / 'bad.csv')


class RuntimeTests(unittest.TestCase):
    def test_all_seven_dispatch_strict_loader_and_per_sample_context(self):
        for alias, spec in config.TARGETS.items():
            variant = spec['eval_variant']
            selected, family = runtime.adapter(variant)
            args = SimpleNamespace(eval_variant=variant)
            with self.subTest(target=alias), mock.patch.object(selected, 'load_runtime', return_value='loaded') as loader:
                self.assertEqual(runtime.load_runtime(args), 'loaded')
                loader.assert_called_once_with(args)
            model, prefix, audio, context, tokens = mock.Mock(), object(), object(), object(), object()
            model.generate_prefix_inference.return_value = prefix, audio, context
            loaded = (model, 'tokenizer', None, None, None, {}, None)
            inputs = {'input': tokens}
            with mock.patch.object(selected, 'greedy_decode', return_value={'generated_text': 'raw'}) as decoder:
                if family == 'multilayer':
                    with mock.patch.object(selected, 'generate_prefix_inference', return_value=(prefix, audio, context)), \
                         mock.patch.object(selected, 'sample_context', return_value={'sample': context}) as build:
                        runtime.prefix_and_decode(selected, family, variant, loaded, inputs)
                        build.assert_called_once_with(variant, audio, context, tokens)
                    self.assertEqual(decoder.call_args.kwargs, {'eval_variant': variant, 'context': {'sample': context}})
                else:
                    runtime.prefix_and_decode(selected, family, variant, loaded, inputs)
                    expected = {'eval_variant': variant, 'audio_information': context} if family == 'recursive' else {}
                    self.assertEqual(decoder.call_args.kwargs, expected)
                self.assertIs(decoder.call_args.args[2], prefix)

    def test_exact_target_paths_and_routes_match_existing_registry(self):
        expected = {'base': ('mizar_s1_ced_small_gbs256_5090', '20261026_193830617147305'),
            'recursive_5_10_5': ('mizar_s1_ced_small_gbs256_recursive_5_10_5_5090', '20261028_160823559722647'),
            'serial84': ('mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_5090', '20261102_112121702891801'),
            'serial48_lora_r8': ('mizar_s1_ced_small_gbs256_recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8_5090', '20261104_131843931946322'),
            'serial84_lora_r8': ('mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8_5090', '20261103_165410438279511'),
            'recursive_mesh': ('mizar_s1_ced_small_gbs256_recursive_mesh_5090', '20261027_081123150934256'),
            'mesh48_lora_r8': ('mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_loop_lora_r8_5090', '20261104_161727489909082')}
        self.assertEqual(set(config.TARGETS), set(expected))
        for alias, (directory, stamp) in expected.items():
            args = evaluate.parse_args(['--target', alias, '--check-args'])
            self.assertTrue(args.checkpoint.as_posix().endswith(f'{directory}/formal_{stamp}/formal/checkpoints/model--step-17313.ckpt'))
            self.assertEqual(args.expected_geometry, (8, 8, 4))
            self.assertTrue((args.route_root / 'models/mellow.py').is_file())
            selected, family = runtime.adapter(args.eval_variant)
            route = selected.variant_spec(args.eval_variant)['route_root'] if family in ('recursive', 'multilayer') else selected.ROUTE_ROOT
            self.assertEqual(args.route_root, route.resolve())

    def test_target_rejects_model_overrides_and_other_benchmarks_unchanged(self):
        for option in (['--checkpoint', '/wrong'], ['--eval-variant', 'base'], ['--expected-step', '1']):
            with self.assertRaises(SystemExit):
                evaluate.parse_args(['--target', 'base', *option])
        audit = json.loads((config.ROOT / 'mizar_s1_ced_reasonaqa_eval/reference_audit.json').read_text())
        self.assertEqual(len(audit['files']), 132)
        for path, digest in audit['files'].items():
            self.assertEqual(storage.file_sha(ROOT / path), digest, path)
        self.assertNotEqual(runtime.PROTOCOL['version'], runtime.base.PROTOCOL['version'])
        self.assertEqual(runtime.base.PROTOCOL['scoring'], 'benchmark_official_evaluation.py_only')

    def test_real_torch_audio_and_prompt_to_generation(self):
        try:
            import torch
            import torchaudio
        except (ImportError, OSError) as exc:
            self.skipTest(f'real Torch/torchaudio unavailable: {exc}')
        # Execute actual training preprocessing functions with explicit decoded-wave fixture.
        path = config.TARGETS['base']['route_root'] / 'data/fixed20.py'
        functions = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)]
        waves = {'single': (torch.arange(800000, dtype=torch.float32).reshape(1, -1), 32000),
                 'pair1': (torch.ones(2, 16000), 16000), 'pair2': (torch.full((1, 400000), 2.), 32000)}
        namespace = {'torch': torch, 'torchaudio': torchaudio, 'SAMPLE_RATE': 32000, 'HALF_SAMPLES': 320000,
                     'load_audio': lambda name, **kw: waves[name]}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        first, second = namespace['load_fixed20']('single')
        self.assertTrue(torch.equal(first, waves['single'][0][0, 80000:400000]))
        self.assertTrue(torch.equal(second, waves['single'][0][0, 400000:720000]))
        a, b = namespace['load_fixed20']('pair1', 'pair2')
        self.assertEqual(a.shape, (320000,))
        self.assertEqual(torch.count_nonzero(a[32000:]).item(), 0)
        self.assertTrue(torch.equal(b, torch.full((320000,), 2.)))


class NumericalFixtureTests(Fixture):
    def test_training_fixed20_and_reasonaqa_generation_with_numpy_tensor_fixture(self):
        import numpy as np

        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float32)
            def numel(self):
                return self.value.size
            def contiguous(self):
                return self
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.value
            def to(self, device):
                return self
            def unsqueeze(self, axis):
                return Tensor(np.expand_dims(self.value, axis))
            def __getitem__(self, key):
                return Tensor(self.value[key])

        # This is an explicit NumPy tensor fixture, not a claim of Torch execution.
        pad = lambda tensor, widths: Tensor(np.pad(tensor.value, widths))
        fake_torch = SimpleNamespace(nn=SimpleNamespace(functional=SimpleNamespace(pad=pad)), inference_mode=nullcontext)
        path = config.TARGETS['base']['route_root'] / 'data/fixed20.py'
        function = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'fixed_len')
        namespace = {'torch': fake_torch}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        source1 = self.audio(self.root, 'one.wav')
        source2 = self.audio(self.root, 'two.wav')
        def source(path):
            stat = path.stat()
            return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        record = {'original_record': row(), 'audio1': source(source1), 'audio2': None}
        wave = Tensor(np.arange(800001, dtype=np.float32))
        fixed20 = SimpleNamespace(load_mono=lambda _: wave, fixed_len=namespace['fixed_len'])
        a, b, audit = runtime.fixed20_pair(fixed20, record)
        np.testing.assert_array_equal(a.value, np.arange(80000, 400000, dtype=np.float32))
        np.testing.assert_array_equal(b.value, np.arange(400000, 720000, dtype=np.float32))
        self.assertEqual(audit[0]['cropped_samples'], 160001)
        short, long = Tensor(np.ones(100000)), Tensor(np.full(400000, 2))
        fixed20.load_mono = lambda name: short if name == str(source1) else long
        record['audio2'] = source(source2)
        a, b, audit = runtime.fixed20_pair(fixed20, record)
        np.testing.assert_array_equal(a.value[:100000], 1)
        np.testing.assert_array_equal(a.value[100000:], 0)
        np.testing.assert_array_equal(b.value, 2)
        self.assertEqual(audit[1]['center_crop_start'], 40000)
        tokens = {'input_ids': mock.Mock(), 'attention_mask': mock.Mock()}
        tokens['input_ids'].__getitem__ = mock.Mock(return_value=SimpleNamespace(tolist=lambda: [7] * 256))
        encoder = mock.Mock()
        encoder.tokenize.return_value = tokens
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = list(range(300))
        loaded = (None, tokenizer, 'fixture_device', encoder, fixed20, {}, None)
        prefix = SimpleNamespace(shape=(1, 383, 576))
        with mock.patch.dict(sys.modules, torch=fake_torch), \
             mock.patch.object(runtime, 'prefix_and_decode', return_value=(prefix, generation(record))) as decode:
            result = runtime.generate(loaded, record, variant='base')
        encoder.tokenize.assert_called_once_with(record['original_record']['input'].lower(), 256)
        self.assertEqual(set(decode.call_args.args[-1]), {'audio1', 'audio2', 'input'})
        self.assertTrue(result['prompt_truncated'])
        self.assertEqual(result['audio_slot_policy'], 'two_actual_files')
        storage.validate_generation(result, record)
        source1.write_bytes(b'changed fixture')
        with self.assertRaisesRegex(ValueError, 'changed'):
            runtime.fixed20_pair(fixed20, record)


if __name__ == '__main__':
    unittest.main()
