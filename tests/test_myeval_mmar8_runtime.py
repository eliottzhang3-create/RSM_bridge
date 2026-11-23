"""Strict real Git contracts, production seed math, full MMAR, and public launch chain."""
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_myeval_mmar8 import (ROOT, SCORER, fixture_args, fixture_files, fixture_rows,
                              fixture_metadata, fixture_generation)
from myeval_mmar8 import arguments, benchmark, preflight, registry, runtime, sources, storage, submit


class HeaderTests(unittest.TestCase):
    def fixture(self, target, cache):
        contract_module = sources.load_contract(target, cache)
        prepared = dict(rows=1477492, rows_canonical_sha256='synthetic', mapping_sha256='synthetic',
                        source_rows=dict(contract_module.SOURCE_COUNTS))
        contract = contract_module.semantic_contract(prepared, '/synthetic/text', '/synthetic/ced')
        contract['recursive_text_checkpoint'] = {'explicit_fixture': True}
        saved = dict(schema_version=1, route_variant=contract_module.ROUTE, mode='formal',
            contract=contract, contract_sha256=contract_module.digest(contract), state_dict={'synthetic': True},
            total_step=34626, num_epochs=3, loss_reduction=runtime.reference.TRAINING_LOSS_REDUCTION,
            batch_geometry=dict(world_size=8, microbatch_size=4, gradient_accumulation_steps=4, global_batch_size=128))
        config = dict(mode='formal', route=contract_module.ROUTE, contract=contract,
                      contract_sha256=contract_module.digest(contract))
        return contract_module, saved, config

    def test_three_actual_contracts_and_trained_read2_validator_signature(self):
        import inspect
        with tempfile.TemporaryDirectory() as directory:
            for target in registry.ROUTES:
                sources.prepare(target, directory)
                module, saved, config = self.fixture(target, directory)
                runtime.validate_checkpoint(saved, config, module, 34626, (8, 4, 4), eval_variant=target)
                if target != 'base':
                    root = sources.route_root(target, directory)
                    tree = ast.parse((root / 'models/recursive_text.py').read_text(encoding='utf-8'))
                    validator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_audio_read2')
                    self.assertEqual([a.arg for a in validator.args.args], ['model'])

    def test_wrong_architecture_source_precision_and_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sources.prepare('base', directory); sources.prepare('read2', directory)
            for target in ('base', 'read2_zero48'):
                module, saved, config = self.fixture(target, directory)
                for key, value in [('code_sha256', 'bad'), ('forward_dtype', 'bfloat16'),
                                   ('text_model_schedule', list(range(42))), ('audio_read2', {'bad': True})]:
                    bad = copy.deepcopy(saved); bad['contract'][key] = value
                    bad['contract_sha256'] = module.digest(bad['contract'])
                    cfg = {**config, 'contract': bad['contract'], 'contract_sha256': bad['contract_sha256']}
                    with self.subTest(target=target, key=key), self.assertRaises(ValueError):
                        runtime.validate_checkpoint(bad, cfg, module, 34626, (8, 4, 4), eval_variant=target)
                with self.assertRaises(ValueError):
                    runtime.validate_checkpoint(saved, config, module, 34626, (8, 8, 4), eval_variant=target)

    def test_cpu_preflight_reads_only_cpu_mmap_and_never_needs_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for target in registry.ROUTES:
                args = fixture_args(root, target); sources.prepare(target, args.sources_cache)
                _, saved, config = self.fixture(target, args.sources_cache)
                args.checkpoint.write_bytes(b'synthetic-checkpoint')
                args.runtime_config.write_text(json.dumps(config))
                loader = Mock(return_value=saved)
                with patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=loader)}):
                    result = preflight.check(args)
                self.assertEqual(result['status'], 'READY_FOR_GPU')
                self.assertTrue(loader.call_args.kwargs['mmap'])
                self.assertEqual(loader.call_args.kwargs['map_location'], 'cpu')


class NumpySeedTests(unittest.TestCase):
    def test_zero_context_replaces_only_pair_and_guards_each_forward_and_sample(self):
        callbacks = []
        lm = types.SimpleNamespace(register_forward_pre_hook=lambda cb, **kw: callbacks.append(cb))
        prefix, output = object(), object()
        pair = (np.ones((1, 126, 3), np.float32), np.full((1, 126, 3), 2, np.float32))
        model = types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=lm),
            generate_prefix_inference=lambda inputs: (prefix, output, pair))
        audit = {'intervention': registry.intervention('read2_zero48')}
        runtime.install_zero48(model, audit)
        with patch.dict(sys.modules, {'torch': types.SimpleNamespace(zeros_like=np.zeros_like)}):
            p, o, zero = runtime.generate_prefix_inference(model, {}, eval_variant='read2_zero48', audit=audit)
            self.assertIs(p, prefix); self.assertIs(o, output)
            self.assertTrue(all(not np.count_nonzero(x) for x in zero))
            self.assertTrue(all(np.count_nonzero(x) for x in pair))
            for _ in range(3): callbacks[0](lm, (), {'audio_information': zero})
            self.assertEqual(audit['intervention']['forwards_checked'], 3)
            with self.assertRaises(RuntimeError): callbacks[0](lm, (), {'audio_information': pair})
            runtime.generate_prefix_inference(model, {}, eval_variant='read2_zero48', audit=audit)
            with self.assertRaises(RuntimeError): callbacks[0](lm, (), {'audio_information': zero})
            self.assertIs(runtime.generate_prefix_inference(model, {}, eval_variant='read2', audit={})[2], pair)

    def test_actual_mixed_interaction_reads_zero_seeds_without_softmax_renormalization(self):
        # Execute the ACTUAL production methods with a NumPy tensor API, not a rewritten model formula.
        class Tensor(np.ndarray):
            def unsqueeze(self, dim): return np.expand_dims(self, dim).view(Tensor)
            def transpose(self, a, b): return self.swapaxes(a, b).view(Tensor)
            def sum(self, dim): return np.asarray(self).sum(axis=dim).view(Tensor)
        def tensor(x): return np.asarray(x, np.float32).view(Tensor)
        path = ROOT / 'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_lora_r8/models/recursive_audio7_read2.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MeshLlamaModel')
        methods = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name in ('_mixed_interaction', '_read', '_write')]
        for n in methods: n.decorator_list = []
        namespace = {'torch': types.SimpleNamespace(Tensor=Tensor,
            cat=lambda values, dim: tensor(np.concatenate(values, axis=dim)))}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
        n, length, width = 126, 132, 3
        core = types.SimpleNamespace(audio_tokens=n, audit_mode=False,
            write_routers=[5, 5, 5], read_routers=[5, 5, 5], audio_read_routers=[7, 7],
            _write=namespace['_write'], _read=namespace['_read'])
        core._route = lambda slots, query, name: tensor(np.full((*query.shape[:-1], slots), 1 / slots))
        memory = (tensor(np.ones((1, 7, n, width))), tensor(np.ones((1, 5, length - n, width))))
        zero_memory = (memory[0].copy(), memory[1].copy()); zero_memory[0][:, 5:] = 0
        query = tensor(np.ones((1, length, width))); value = tensor(np.full(query.shape, .7))
        for index in (0, 1, 2):
            updated, out = namespace['_mixed_interaction'](core, zero_memory, query, value, index, str(index))
            np.testing.assert_array_equal(updated[0][:, 5:], 0)
            expected = (1 + .7 / 5) * (5 / 7 if index < 2 else 1)
            np.testing.assert_allclose(out[:, :n], expected, rtol=1e-6)
            self.assertEqual(updated[0].shape[1], 7)
            np.testing.assert_array_equal(memory[0], 1)


class FullMmarTests(unittest.TestCase):
    def execute(self, args, records, generation):
        identity = {'benchmark': 'mmar', 'eval_variant': 'read2_zero48', 'checkpoint': 'fixture',
                    'prediction_key': 'model_prediction', 'intervention': registry.intervention('read2_zero48')}
        loaded = (None, None, None, None, None, {'checkpoint_step': 34626}, {})
        # The pinned scorer imports tqdm but never calls it. Supply only this
        # unused display dependency for the offline subprocess, without changing
        # the real official scorer or replacing any scoring/CLI implementation.
        display_dependency = args.output_dir.parent / 'unused_display_dependency'
        display_dependency.mkdir(exist_ok=True)
        (display_dependency / 'tqdm.py').write_text(
            'def tqdm(*args, **kwargs):\n    raise AssertionError("unexpected tqdm call")\n', encoding='utf-8')
        subprocess_path = str(display_dependency) + os.pathsep + os.environ.get('PYTHONPATH', '')
        original_run = subprocess.run
        def scorer_subprocess(*args, **kwargs):
            # Do not patch os.environ: on Windows restoring an empty env value
            # deletes it, which can invalidate injected GIT_CONFIG_VALUE_*.
            kwargs['env'] = dict(os.environ, PYTHONPATH=subprocess_path)
            return original_run(*args, **kwargs)
        with patch.object(benchmark, '_validate_input_paths'), patch.object(benchmark, '_identity', return_value=identity), \
             patch.object(benchmark, '_metadata_records', return_value=(records, {'status': 'PASS'}, 'model_prediction')), \
             patch.object(benchmark, '_call_model_loader', return_value=loaded), \
             patch.object(benchmark, 'generate', generation), patch('os.fsync'), \
             patch.object(benchmark.subprocess, 'run', side_effect=scorer_subprocess), \
             contextlib.redirect_stdout(io.StringIO()):
            return benchmark.run(args)

    def test_full_thousand_uses_original_questions_raw_predictions_and_real_official_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args = fixture_args(root, 'read2_zero48', 'benchmark')
            records = fixture_metadata(fixture_rows())
            generator = Mock(return_value={'generated_text': '  a) correct <|endoftext|>  ',
                                          'generated_token_ids': [1, 0], 'generated_token_count': 2})
            report = self.execute(args, records, generator)
            self.assertEqual(report['status'], 'PASS', report)
            self.assertEqual(generator.call_count, 1000)
            self.assertEqual(report['official_evaluation']['reported_rows'], 1000)
            predictions = json.loads((args.output_dir / 'predictions_official.json').read_text())
            self.assertTrue(all(x['model_prediction'] == '  a) correct <|endoftext|>  ' for x in predictions))
            self.assertEqual(generator.call_args_list[0].args[1]['question'], 'official question')
            self.assertEqual(report['official_evaluation']['total_accuracy_percent'], 100.)

    def test_full_mmar_interruption_never_scores_partial_and_resumes_remaining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args = fixture_args(root, 'read2_zero48', 'benchmark')
            records = fixture_metadata(fixture_rows())
            raw = {'generated_text': 'correct', 'generated_token_ids': [1, 0], 'generated_token_count': 2}
            generator = Mock(side_effect=[raw] * 13 + [RuntimeError('interrupted')])
            failed = self.execute(args, records, generator)
            self.assertEqual(failed['status'], 'FAILED')
            self.assertFalse((args.output_dir / 'official_evaluation.txt').exists())
            generator = Mock(return_value=raw)
            passed = self.execute(args, records, generator)
            self.assertEqual(passed['status'], 'PASS', passed)
            self.assertEqual(generator.call_count, 987)


class SubmissionTests(unittest.TestCase):
    def test_submission_failure_continues_and_deduplicates_checkpoint_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'submission.json'
            commands = []
            def fake_vc(argv, **kwargs):
                commands.append(argv)
                return types.SimpleNamespace(returncode=1 if len(commands) == 2 else 0, stdout='fixture', stderr='')
            with patch.object(submit, 'dataset_preflight', return_value={'status': 'READY'}), \
                 patch.object(preflight, 'check', return_value={'status': 'READY_FOR_GPU'}) as check, \
                 patch.object(submit.subprocess, 'run', side_effect=fake_vc), \
                 contextlib.redirect_stdout(io.StringIO()):
                status = submit.main(['all', '--submission-report', str(report)])
            self.assertEqual(status, 1); self.assertEqual(len(commands), 4); self.assertEqual(check.call_count, 2)
            result = json.loads(report.read_text())
            self.assertEqual([j['status'] for j in result['jobs']], ['SUBMITTED', 'FAILED', 'SUBMITTED', 'SUBMITTED'])

    def test_dataset_preflight_failure_submits_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(submit, 'dataset_preflight', side_effect=ValueError('bad labels')), \
                 patch.object(submit.subprocess, 'run') as vc, contextlib.redirect_stdout(io.StringIO()):
                status = submit.main(['all', '--submission-report', str(Path(directory) / 'report.json')])
            self.assertEqual(status, 1); vc.assert_not_called()

    def test_real_bash_syntax_and_public_wrapper_preview(self):
        bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else Path('/bin/bash')
        if not bash.exists(): self.skipTest('Bash not installed')
        for script in (ROOT / 'code/RSmol/run_myeval_mmar8_4090.sh', registry.PACKAGE / 'run.sh'):
            result = subprocess.run([str(bash), '-n', str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory(prefix='myeval shell ') as directory:
            base = Path(directory); hook = base / 'etc/profile.d/conda.sh'; hook.parent.mkdir(parents=True)
            hook.write_text('conda() { return 0; }\n', encoding='utf-8')
            env = dict(os.environ, RSMOL_CONDA_BASE=base.as_posix(), PYTHONPATH=str(ROOT / 'code/RSmol'))
            result = subprocess.run([str(bash), str(ROOT / 'code/RSmol/run_myeval_mmar8_4090.sh'), 'all', '--check-args'],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len([json.loads(x) for x in result.stdout.splitlines()]), 4)


if __name__ == '__main__': unittest.main()
