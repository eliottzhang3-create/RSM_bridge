"""Fixed-5000 selection and real isolated control flow with explicit model fixtures."""
from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import random
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code/RSmol'), str(ROOT / 'tests')]
from mizar_s1_ced_reasonaqa5000_eval import config, sample, dataset, storage, scoring, runtime, evaluate, compare, submit, inspect_checkpoint
from test_mizar_s1_ced_reasonaqa_eval import row as original_row, generation as old_generation


def generation(record, text=None):
    result = old_generation(record, text)
    result['prompt_attention_mask'] = [1] * 256
    fields = ('prompt_token_ids', 'prompt_attention_mask', 'audio_half_sha256')
    result['input_fingerprint_sha256'] = storage.object_sha({key: result[key] for key in fields})
    return result


def write_pool(directory):
    rows = []
    for subtype, count, fields in (
        ('AudioCaps-MCQ.json', 23672, {'filepath1': 'AudioCapsLarger/test/example.wav'}),
        ('Clotho-MCQ.json', 26031, {'filepath1': 'ClothoV21/evaluation/Rain fall.wav'}),
        ('ClothoAQA.json', 5676, {'filepath1': 'ClothoAQA\\audio_files\\dripping.wav', 'answer': 'no'})):
        rows.extend(original_row(subtype, **fields) for _ in range(count))
    # Explicit excluded rows demonstrate that sampling uses original indices after filtering.
    rows.insert(20, original_row('ClothoAQA.json', 'two'))
    rows.insert(40, original_row('Other-Open.json', 'free text'))
    source = directory / "source test's.json"
    storage.write_json(source, rows)
    return source


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.shared = Path(cls.temp.name)
        cls.source = write_pool(cls.shared)
        cls.bundle = cls.shared / 'sample bundle'
        cls.manifest = sample.generate_bundle(cls.source, cls.bundle)
        cls.roots = {}
        for flag, name in [('audiocaps-root', 'caps/test/example.wav'), ('clotho-root', 'clotho/evaluation/Rain fall.wav'),
                           ('clotho-aqa-root', 'aqa/dripping.wav')]:
            path = cls.shared / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'explicit path fixture; GPU/audio is mocked')
            cls.roots[flag] = cls.shared / name.split('/')[0]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.output_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.output_temp.cleanup)
        self.directory = Path(self.output_temp.name)

    def args(self, target='r515_inject84', output='eval'):
        argv = ['--target', target, '--manifest', str(self.bundle / 'manifest.json'), '--output-dir', str(self.directory / output)]
        for flag, value in self.roots.items():
            argv.extend(['--' + flag, str(value)])
        args = evaluate.parse_args(argv)
        # Only checkpoint/runtime bytes are fixtures; use real registered training source hashes.
        args.checkpoint = self.directory / 'fixture.ckpt'
        args.checkpoint.write_bytes(b'not a real checkpoint')
        args.runtime_config = self.directory / 'fixture_runtime.json'
        storage.write_json(args.runtime_config, {'fixture': True})
        return args

    def run_fixture(self, args, callback=None, forbid_loader=False):
        loaded = (None, None, None, None, None, {'status': 'PASS', 'fixture': True}, None)
        with mock.patch.object(runtime, 'load_runtime', side_effect=AssertionError('model load forbidden') if forbid_loader else None,
                               return_value=loaded) as loader, \
             mock.patch.object(runtime, 'generate', side_effect=callback or (lambda _, r, **kw: generation(r))), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            report = evaluate.run(args)
        return report, loader.call_count


class SamplingTests(Fixture):
    def test_exact_size_fixed_seed_original_ids_and_no_global_rng_change(self):
        state = random.getstate()
        pool, audit = dataset.select_records(self.source)
        chosen = sample.select_sample(pool)
        self.assertEqual(random.getstate(), state)
        indices = sorted(random.Random(1234).sample(range(55379), 5000))
        self.assertEqual(chosen, [pool[index] for index in indices])
        self.assertEqual(self.manifest['records'], chosen)
        self.assertEqual(len({r['id'] for r in chosen}), 5000)
        self.assertEqual(sum(self.manifest['sample_category_counts'].values()), 5000)
        self.assertEqual(audit['selected_category_counts'], config.POOL_COUNTS)
        self.assertEqual(sample.build_manifest(self.source), self.manifest)
        self.assertEqual(sample.load_manifest(self.bundle / 'manifest.json'), self.manifest)

    def test_original_json_order_content_export_and_idempotent_generation(self):
        rows = json.loads((self.bundle / 'sampled_test.json').read_text(encoding='utf-8'))
        self.assertEqual(rows, [r['original_record'] for r in self.manifest['records']])
        before = (self.bundle / 'manifest.json').read_bytes()
        self.assertEqual(sample.generate_bundle(self.source, self.bundle), self.manifest)
        self.assertEqual((self.bundle / 'manifest.json').read_bytes(), before)
        indices = [r['source_row_index'] for r in self.manifest['records']]
        self.assertEqual(indices, sorted(indices))
        self.assertFalse(set(indices) & {20, 40})

    def test_wrong_population_tampered_source_manifest_and_replacement_rejected(self):
        small = self.directory / 'small.json'
        storage.write_json(small, [original_row()])
        with self.assertRaisesRegex(ValueError, '55379'):
            sample.build_manifest(small)
        with self.assertRaises(ValueError):
            sample.select_sample([1] * 4999)
        wrong = copy.deepcopy(self.manifest)
        wrong['records'][0]['original_record']['answer'] = 'forged'
        forged = self.directory / 'forged.json'
        storage.write_json(forged, wrong)
        with self.assertRaises(ValueError):
            sample.load_manifest(forged)
        wrong['sample_records_sha256'] = storage.object_sha(wrong['records'])
        wrong['manifest_sha256'] = storage.object_sha({k: v for k, v in wrong.items() if k != 'manifest_sha256'})
        storage.write_json(forged, wrong)
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            sample.load_manifest(forged)
        changed = self.directory / 'source.json'
        changed.write_bytes(self.source.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'source test.json differs'):
            sample.load_manifest(self.bundle / 'manifest.json', changed)
        unowned = self.directory / 'unowned'
        unowned.mkdir()
        (unowned / 'keep').write_text('keep')
        with self.assertRaises(FileExistsError):
            sample.generate_bundle(self.source, unowned)


class WorkflowTests(Fixture):
    def test_complete_5000_flow_exact_scoring_comparison_and_rescore_without_model(self):
        args = self.args()
        seen = []
        def generate(_, record, **kwargs):
            seen.append(record['id'])
            # First response matches only the label: it must be wrong.
            return generation(record, 'B' if len(seen) == 1 else None)
        report, calls = self.run_fixture(args, generate)
        self.assertEqual(report['status'], 'PASS', report['fatal_error'])
        self.assertEqual(calls, 1)
        self.assertEqual(seen, self.manifest['sample_ids'])
        self.assertEqual(report['accuracy']['overall'], {'correct': 4999, 'total': 5000, 'accuracy': 4999 / 5000})
        self.assertEqual(report['inference_coverage'], {'status': 'PASS', 'generated': 5000, 'expected_rows': 5000, 'skipped': 0})
        before = (args.output_dir / 'raw_generations.jsonl').read_bytes()
        again, calls = self.run_fixture(args, forbid_loader=True)
        self.assertEqual(again['status'], 'PASS', again['fatal_error'])
        self.assertEqual(calls, 0)
        self.assertEqual(before, (args.output_dir / 'raw_generations.jsonl').read_bytes())
        identity, accuracy, common, inputs = compare.read_run(args.output_dir)
        self.assertEqual(accuracy, report['accuracy'])
        self.assertEqual(common['manifest_sha256'], self.manifest['manifest_sha256'])
        self.assertEqual(len(inputs), 5000)
        rows = compare.compare_runs([('fixture', args.output_dir)], self.directory / 'comparison.csv')
        self.assertEqual(rows[0]['overall_total'], 5000)
        score = args.output_dir / 'accuracy.json'
        tampered = json.loads(score.read_text())
        tampered['overall']['correct'] = 5000
        storage.write_json(score, tampered)
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            compare.read_run(args.output_dir)
        self.run_fixture(args, forbid_loader=True)
        report_path = args.output_dir / 'evaluation_report.json'
        invalid = json.loads(report_path.read_text())
        invalid['status'] = 'FAILED'
        storage.write_json(report_path, invalid)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            compare.read_run(args.output_dir)

    def test_interruption_durable_resume_only_unfinished_rows(self):
        args = self.args()
        completed = []
        def interrupt(_, record, **kwargs):
            if len(completed) == 13:
                raise RuntimeError('explicit fixture interruption')
            completed.append(record['id'])
            return generation(record)
        report, _ = self.run_fixture(args, interrupt)
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(report['inference_coverage']['generated'], 13)
        self.assertFalse((args.output_dir / 'accuracy.json').exists())
        with (args.output_dir / 'raw_generations.jsonl').open('ab') as stream:
            stream.write(b'{"partial":')
        (args.output_dir / 'progress.jsonl').write_bytes(b'stale')
        resumed = []
        def finish(_, record, **kwargs):
            resumed.append(record['id'])
            return generation(record)
        report, _ = self.run_fixture(args, finish)
        self.assertEqual(report['status'], 'PASS', report['fatal_error'])
        self.assertEqual(resumed, self.manifest['sample_ids'][13:])
        self.assertEqual(len((args.output_dir / 'progress.jsonl').read_text().splitlines()), 5000)

    def test_missing_audio_changed_identity_and_complete_corruption_fail(self):
        args = self.args()
        def interrupt(*args, **kwargs):
            raise RuntimeError('fixture')
        self.run_fixture(args, interrupt)
        old_report = (args.output_dir / 'evaluation_report.json').read_bytes()
        args.checkpoint.write_bytes(b'changed checkpoint')
        report, calls = self.run_fixture(args)
        self.assertFalse(report['output_claimed'])
        self.assertEqual(calls, 0)
        self.assertEqual((args.output_dir / 'evaluation_report.json').read_bytes(), old_report)
        args.checkpoint.write_bytes(b'not a real checkpoint')
        (args.output_dir / 'raw_generations.jsonl').write_bytes(b'broken complete row\n')
        report, calls = self.run_fixture(args)
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(calls, 0)
        args.audiocaps_root = self.directory / 'missing_audio'
        args.output_dir = self.directory / 'new_output'
        report, calls = self.run_fixture(args)
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(calls, 0)
        self.assertFalse(report['output_claimed'])

    def test_fingerprints_include_real_prompt_mask_and_hash_tampering_fails(self):
        record = self.manifest['records'][0]
        valid = generation(record)
        storage.validate_generation(valid, record)
        tampered = copy.deepcopy(valid)
        tampered['prompt_attention_mask'][0] = 0
        with self.assertRaises(ValueError):
            storage.validate_generation(tampered, record)
        for key, value in [('generated_token_ids', []), ('prefix_shape', [1, 1, 576]),
                           ('prompt_used', 'contains ground truth'), ('generated_text_raw', 'edited')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                storage.validate_generation({**valid, key: value}, record)


class RegistryTests(unittest.TestCase):
    def test_scope_equals_current_mmau_minus_exact_eight_exclusions(self):
        text = (config.ROOT / 'run_mizar_s1_ced_small_mmau_eval_5090.sh').read_text()
        variants = set()
        for line in text.splitlines():
            if re.fullmatch(r'  [a-z0-9_]+(?:\|[a-z0-9_]+)*\)', line):
                variants.update(line.strip()[:-1].split('|'))
        self.assertEqual(len(variants), 38)
        self.assertEqual(len(config.TARGETS), 30)
        self.assertEqual(set(config.VARIANT_TARGETS), variants - set(config.EXCLUDED))
        self.assertFalse(set(config.VARIANT_TARGETS) & set(config.EXCLUDED))

    def test_each_frozen_binding_loader_step_geometry_route_and_outputs(self):
        outputs = []
        for name, spec in config.TARGETS.items():
            args = evaluate.parse_args(['--target', name, '--check-args'])
            selected, family = runtime.adapter(args.eval_variant)
            registered = selected.variant_spec(args.eval_variant)
            self.assertEqual(spec['runtime_family'], family)
            self.assertEqual(args.route_root, registered['route_root'].resolve())
            self.assertEqual(args.expected_step, registered.get('default_step', 17313))
            self.assertEqual(args.expected_geometry, registered.get('default_geometry', (8, 8, 4)))
            if name != 'mesh_inject48':
                self.assertEqual(args.run_root, registered['run_root'].resolve())
            else:
                self.assertTrue(args.checkpoint.as_posix().endswith('formal_20261030_172359752928500/formal/checkpoints/model--step-17313.ckpt'))
            outputs.append(str(args.output_dir))
            with mock.patch.object(selected, 'load_runtime', return_value='loaded') as loader:
                self.assertEqual(runtime.load_runtime(args), 'loaded')
                loader.assert_called_once_with(args)
        self.assertEqual(len(set(outputs)), 30)
        step34626 = {name for name, spec in config.TARGETS.items() if spec['expected_step'] == 34626}
        self.assertEqual(step34626, {'mesh_x4', 'official_serial84_r8', 'official_input_only'})

    def test_excluded_targets_overrides_and_candidate_corrections(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for variant in config.EXCLUDED:
                with self.subTest(variant=variant), self.assertRaises(SystemExit):
                    evaluate.parse_args(['--target', variant])
                with self.assertRaises(SystemExit):
                    submit.parse_args([variant, '--check-args'])
            for flags in (['--checkpoint', '/wrong'], ['--run-root', '/wrong'], ['--runtime-config', '/wrong'],
                          ['--expected-step', '1'], ['--eval-variant', 'base']):
                with self.assertRaises(SystemExit):
                    evaluate.parse_args(['--target', 'r515_inject84', *flags])
        args = evaluate.parse_args(['--target', 'mesh_inject48', '--checkpoint', '/correct run/formal/checkpoints/model--step-17313.ckpt'])
        self.assertTrue(args.runtime_config.as_posix().endswith('/correct run/formal/runtime.json'))
        items = submit.planned(submit.parse_args(['all', '--mesh-inject48-run-root', '/correct run']))
        corrected = next(args for name, _, args in items if name == 'mesh_inject48')
        self.assertTrue(corrected.checkpoint.as_posix().endswith('/correct run/formal/checkpoints/model--step-17313.ckpt'))

    def test_original_sources_and_original_full_identity_glob_untouched(self):
        audit = json.loads((config.PACKAGE / 'reference_audit.json').read_text())
        self.assertGreater(len(audit['files']), 1000)
        for path, digest in audit['files'].items():
            self.assertEqual(storage.file_sha(ROOT / path), digest, path)
        original_glob = {p.name for p in config.ROOT.glob('run_mizar*reasonaqa*5090.sh')}
        self.assertEqual(original_glob, {'run_mizar_s1_ced_small_reasonaqa_eval_5090.sh', 'run_mizar_reasonaqa_selected7_5090.sh'})

    def test_thirty_per_sample_context_paths_are_preserved(self):
        for name, spec in config.TARGETS.items():
            selected, family = runtime.adapter(spec['eval_variant'])
            model = mock.Mock()
            prefix, audio, ctx, tokens = object(), object(), object(), object()
            model.generate_prefix_inference.return_value = prefix, audio, ctx
            loaded = (model, 'tokenizer', None, None, None, {}, None)
            with self.subTest(name=name), mock.patch.object(selected, 'greedy_decode', return_value={'generated_text': 'raw'}) as decoder:
                if family == 'multilayer':
                    with mock.patch.object(selected, 'generate_prefix_inference', return_value=(prefix, audio, ctx)), \
                         mock.patch.object(selected, 'sample_context', return_value={'sample': ctx}) as build:
                        runtime.prefix_and_decode(selected, family, spec['eval_variant'], loaded, {'input': tokens})
                        build.assert_called_once_with(spec['eval_variant'], audio, ctx, tokens)
                    expected = {'eval_variant': spec['eval_variant'], 'context': {'sample': ctx}}
                else:
                    runtime.prefix_and_decode(selected, family, spec['eval_variant'], loaded, {'input': tokens})
                    expected = {'eval_variant': spec['eval_variant'], 'audio_information': ctx}
                self.assertEqual(decoder.call_args.kwargs, expected)
                self.assertIs(decoder.call_args.args[2], prefix)

    def test_real_numeric_inference_policy_is_an_exact_copy_of_previous_version(self):
        def functions(path):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
        old = functions(config.ROOT / 'mizar_s1_ced_reasonaqa_eval/runtime.py')
        new = functions(config.PACKAGE / 'runtime.py')
        for name in ('fixed20_pair', 'waveform_sha', 'prefix_and_decode'):
            self.assertEqual(old[name], new[name], name)
        for name in ('scoring.py', 'dataset.py'):
            self.assertEqual((config.PACKAGE / name).read_bytes(), (config.ROOT / 'mizar_s1_ced_reasonaqa_eval' / name).read_bytes())


class CandidateTests(unittest.TestCase):
    def test_candidate_cpu_contract_uses_actual_validator_before_marking_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            args = evaluate.parse_args(['--target', 'mesh_inject48'])
            args.checkpoint = Path(directory) / 'model.ckpt'
            args.checkpoint.write_bytes(b'explicit checkpoint fixture')
            args.runtime_config = Path(directory) / 'runtime.json'
            storage.write_json(args.runtime_config, {'fixture': True})
            route = object()
            saved = {'state_dict': {'some_tensor': 'fixture'}, 'total_step': 17313, 'contract_sha256': 'fixture'}
            fake_torch = SimpleNamespace(load=mock.Mock(return_value=saved))
            selected, _ = runtime.adapter(args.eval_variant)
            with mock.patch.dict(sys.modules, torch=fake_torch), \
                 mock.patch.object(selected, 'import_training_route', return_value={'contract': route}), \
                 mock.patch.object(selected, 'validate_checkpoint', return_value={'geometry': {'world_size': 8}}) as validate:
                result = inspect_checkpoint.inspect(args)
                validate.assert_called_once_with(saved, {'fixture': True}, route, 17313, (8, 8, 4), eval_variant=args.eval_variant)
                self.assertEqual(result['status'], 'PASS_CPU_CONTRACT_ONLY')
                self.assertFalse(result['gpu_model_load_verified'])
                self.assertEqual(fake_torch.load.call_args.kwargs['map_location'], 'cpu')
                validate.side_effect = ValueError('wrong route contract')
                with self.assertRaisesRegex(ValueError, 'wrong route'):
                    inspect_checkpoint.inspect(args)


class NumericalTests(unittest.TestCase):
    def test_generate_uses_original_question_audio_and_mask_only(self):
        import numpy as np

        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)
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
            def tolist(self):
                return self.value.tolist()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audio.wav'
            path.write_bytes(b'explicit numeric fixture')
            stat = path.stat()
            record = {'id': 'reasonaqa_test:9', 'original_record': original_row(), 'audio2': None,
                      'audio1': {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}}
            source = config.TARGETS['r515_inject84']['route_root'] / 'data/fixed20.py'
            node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'fixed_len')
            fake_torch = SimpleNamespace(inference_mode=contextlib.nullcontext,
                nn=SimpleNamespace(functional=SimpleNamespace(pad=lambda t, pad: Tensor(np.pad(t.value, pad)))))
            namespace = {'torch': fake_torch}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
            fixed20 = SimpleNamespace(load_mono=lambda _: Tensor(np.arange(800001, dtype=np.float32)), fixed_len=namespace['fixed_len'])
            tokens = {'input_ids': Tensor([[7] * 256]), 'attention_mask': Tensor([[1] * 200 + [0] * 56])}
            encoder, tokenizer = mock.Mock(), mock.Mock()
            encoder.tokenize.return_value = tokens
            tokenizer.encode.return_value = list(range(300))
            loaded = (None, tokenizer, 'fixture', encoder, fixed20, {}, None)
            result = generation(record)
            with mock.patch.dict(sys.modules, torch=fake_torch), \
                 mock.patch.object(runtime, 'prefix_and_decode', return_value=(SimpleNamespace(shape=(1,383,576)), result)) as decode:
                result = runtime.generate(loaded, record, variant='recursive_5_10_5_layer8_4_inject')
            encoder.tokenize.assert_called_once_with(record['original_record']['input'].lower(), 256)
            model_inputs = decode.call_args.args[-1]
            self.assertEqual(set(model_inputs), {'audio1', 'audio2', 'input'})
            self.assertEqual(set(model_inputs['input']), {'input_ids', 'attention_mask'})
            np.testing.assert_array_equal(model_inputs['audio1'].value[0], np.arange(80000,400000,dtype=np.float32))
            np.testing.assert_array_equal(model_inputs['audio2'].value[0], np.arange(400000,720000,dtype=np.float32))
            self.assertEqual(result['prompt_attention_mask'], [1] * 200 + [0] * 56)
            self.assertTrue(result['prompt_truncated'])
            storage.validate_generation(result, record)


if __name__ == '__main__':
    unittest.main()
