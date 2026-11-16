"""Supplemental seven fixed checkpoints on the same immutable 5000 rows, eight-GPU mode."""
from __future__ import annotations

import contextlib
import importlib.abc
import importlib.util
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code/RSmol'), str(ROOT / 'tests')]
from mizar_s1_ced_reasonaqa5000_selected7_8gpu_eval import config, evaluate, submit, compare, runtime
from mizar_s1_ced_reasonaqa5000_8gpu_eval import device, config as previous_config
from mizar_s1_ced_reasonaqa5000_eval import storage, runtime as previous_runtime
from mizar_s1_ced_reasonaqa_eval import config as original_config, runtime as original_runtime
from test_reasonaqa5000_eval import Fixture as SampleFixture, generation


def binding(rank):
    return device.device_binding({'RANK': str(rank), 'LOCAL_RANK': str(rank),
        'WORLD_SIZE': '8', 'LOCAL_WORLD_SIZE': '8', 'CUDA_VISIBLE_DEVICES': '7,6,5,4,3,2,1,0'})


def fixture_worker(args_file):
    """Run the actual worker entry in a child, with an explicitly fake Torch/model."""
    from mizar_s1_ced_reasonaqa5000_selected7_8gpu_eval import worker
    values = json.loads(Path(args_file).read_text(encoding='utf-8'))
    for name in ('manifest', 'test_json', 'audiocaps_root', 'clotho_root', 'clotho_aqa_root',
                 'output_dir', 'run_root', 'checkpoint', 'runtime_config', 'route_root'):
        if values.get(name) is not None:
            values[name] = Path(values[name])
    args = SimpleNamespace(**values)

    class FakeTorch(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            return importlib.util.spec_from_loader(fullname, self) if fullname == 'torch' else None
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            # This runs only after the real worker.main has masked its GPU.
            assert os.environ['CUDA_VISIBLE_DEVICES'] == str(7 - int(os.environ['RANK']))
            assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            module.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)

    sys.meta_path.insert(0, FakeTorch())
    first_id = evaluate.read_jsonl(args.output_dir / 'selected_records.jsonl')[0]['id']
    loaded = (None, None, 'fixture', None, None, {'status': 'PASS', 'mock_gpu_model': True}, None)
    with mock.patch.object(evaluate, 'parse_args', return_value=args), \
         mock.patch.object(runtime, 'load_runtime', return_value=loaded), \
         mock.patch.object(runtime, 'generate', side_effect=lambda _, record, **kw:
                           generation(record, 'B' if record['id'] == first_id else None)):
        return worker.main([])


class PartitionAndDeviceTests(unittest.TestCase):
    def test_eight_disjoint_625_row_shards_cover_exact_5000_without_padding(self):
        shards = [config.rank_indices(rank) for rank in range(8)]
        self.assertEqual([len(shard) for shard in shards], [625] * 8)
        self.assertEqual(sorted(index for shard in shards for index in shard), list(range(5000)))
        for wrong in (-1, 8, True, '0'):
            with self.assertRaises(ValueError):
                config.rank_indices(wrong)

    def test_reverse_numbered_uuid_and_unset_visibility_bind_each_rank_once(self):
        self.assertEqual([binding(rank)['worker_cuda_visible_devices'] for rank in range(8)], list('76543210'))
        environment = {'RANK': '3', 'LOCAL_RANK': '3', 'WORLD_SIZE': '8', 'LOCAL_WORLD_SIZE': '8'}
        self.assertEqual(device.device_binding(environment)['worker_cuda_visible_devices'], '3')
        environment['CUDA_VISIBLE_DEVICES'] = ','.join(f'GPU-uuid-{index}' for index in range(8))
        self.assertEqual(device.device_binding(environment)['worker_cuda_visible_devices'], 'GPU-uuid-3')

    def test_invalid_world_duplicate_missing_or_short_visibility_rejected(self):
        base = {'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '8', 'LOCAL_WORLD_SIZE': '8'}
        for changed in ({'WORLD_SIZE': '16'}, {'LOCAL_RANK': '1'}, {'RANK': '8'},
                        {'CUDA_VISIBLE_DEVICES': '0,1'}, {'CUDA_VISIBLE_DEVICES': '0,0,2,3,4,5,6,7'},
                        {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,'}, {'CUDA_VISIBLE_DEVICES': '-1'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                device.device_binding({**base, **changed})
        with self.assertRaises(ValueError):
            device.device_binding({})

    def test_binding_before_torch_import_is_enforced(self):
        with mock.patch.dict(sys.modules, {'torch': object()}), self.assertRaisesRegex(RuntimeError, 'before importing'):
            device.bind_worker()

    def test_all_seven_exact_bindings_3090_resources_short_names_and_new_outputs(self):
        items = submit.planned(submit.parse_args(['all']))
        self.assertEqual(len(items), 7)
        jobs, outputs = [], []
        for name, argv, args in items:
            self.assertEqual(submit.parse_args([name]).target, name)
            self.assertEqual(submit.parse_args([args.eval_variant]).target, name)
            command = submit.command(name, argv, '20261116_120000_123456_1234567890123456789')
            for flag, value in (('-p', 'pdgpu-3090'), ('-g', '8'), ('-c', '64'), ('-m', '256G'), ('-n', '1')):
                self.assertEqual(command[command.index(flag) + 1], value)
            jobs.append(command[command.index('-j') + 1])
            self.assertLessEqual(len(jobs[-1]), 60)
            self.assertTrue(args.output_dir.name.endswith('_8gpu_v1'))
            self.assertFalse(set(config.EXCLUDED) & {args.eval_variant})
            outputs.append(str(args.output_dir))
        self.assertEqual(len(set(jobs)), 7)
        self.assertEqual(len(set(outputs)), 7)
        for name, _, args in items:
            original = original_config.TARGETS[name]
            self.assertEqual(args.run_root, original['run_root'].resolve())
            self.assertEqual(args.route_root, original['route_root'].resolve())
            self.assertEqual(args.eval_variant, original['eval_variant'])
            self.assertEqual(args.expected_step, 17313)
            self.assertEqual(args.expected_geometry, (8, 8, 4))
            self.assertTrue(args.checkpoint.as_posix().endswith('/formal/checkpoints/model--step-17313.ckpt'))
            self.assertTrue(args.runtime_config.as_posix().endswith('/formal/runtime.json'))
            selected, family = runtime.adapter(args.eval_variant)
            self.assertEqual((selected, family), original_runtime.adapter(args.eval_variant))
            with mock.patch.object(selected, 'load_runtime', return_value='strict_loader_fixture') as loader:
                self.assertEqual(runtime.load_runtime(args), 'strict_loader_fixture')
                loader.assert_called_once_with(args)
        self.assertEqual(len(set(config.VARIANT_TARGETS) | {s['eval_variant'] for s in previous_config.TARGETS.values()}), 37)
        self.assertFalse(set(config.VARIANT_TARGETS) & {s['eval_variant'] for s in previous_config.TARGETS.values()})
        self.assertEqual(config.EXCLUDED, ('official_mizar',))
        self.assertEqual(runtime.PROTOCOL, previous_runtime.PROTOCOL)
        self.assertEqual(config.PARALLEL, previous_config.PARALLEL)

    def test_torchrun_command_has_eight_workers_no_restarts_and_streamed_output(self):
        args = evaluate.parse_args(['--target', 'base'])
        with mock.patch.object(evaluate.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as launch, \
             contextlib.redirect_stdout(io.StringIO()):
            evaluate.launch_workers(args)
        command = launch.call_args.args[0]
        self.assertIn('torch.distributed.run', command)
        self.assertIn('--nproc-per-node=8', command)
        self.assertIn('--nnodes=1', command)
        self.assertIn('--standalone', command)
        self.assertIn('--max-restarts=0', command)
        self.assertNotIn('capture_output', launch.call_args.kwargs)
        self.assertEqual(launch.call_args.kwargs['env']['PYTHONDONTWRITEBYTECODE'], '1')
        forwarded = command[command.index('mizar_s1_ced_reasonaqa5000_selected7_8gpu_eval.worker') + 1:]
        self.assertEqual(evaluate.parse_args(forwarded).output_dir, args.output_dir)
        with mock.patch.object(evaluate.subprocess, 'run', return_value=subprocess.CompletedProcess([], 3)), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'exit 3'):
            evaluate.launch_workers(args)

    def test_all_existing_source_bytes_and_old_identity_globs_remain_unchanged(self):
        audit = json.loads((config.PACKAGE / 'reference_audit.json').read_text())
        self.assertGreater(len(audit['files']), 1100)
        for name, sha in audit['files'].items():
            self.assertEqual(storage.file_sha(ROOT / name), sha, name)
        self.assertEqual({path.name for path in config.ROOT.glob('run_reasonaqa5000*5090.sh')},
                         {'run_reasonaqa5000_eval_5090.sh'})

    def test_official_mizar_wrong_scope_and_model_overrides_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for name in ('official_mizar', 'init0904', 'r515_inject84'):
                with self.subTest(name=name), self.assertRaises(SystemExit):
                    submit.parse_args([name])
                with self.assertRaises(SystemExit):
                    evaluate.parse_args(['--target', name])
            for flag in ('--checkpoint', '--run-root', '--runtime-config', '--route-root', '--expected-step'):
                with self.subTest(flag=flag), self.assertRaises(SystemExit):
                    evaluate.parse_args(['--target', 'base', flag, '/wrong'])
            with self.assertRaises(SystemExit):
                submit.parse_args(['all', '--output-dir', '/shared'])

    def test_actual_generation_numeric_functions_and_seven_context_dispatches_match_original(self):
        import ast
        def functions(path):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(node, ast.FunctionDef)}
        old = functions(previous_runtime.__file__ and Path(previous_runtime.__file__))
        current = functions(Path(runtime.__file__))
        for name in ('fixed20_pair', 'waveform_sha', 'prefix_and_decode', 'generate'):
            self.assertEqual(current[name], old[name], name)
        for spec in config.TARGETS.values():
            variant = spec['eval_variant']
            selected, family = runtime.adapter(variant)
            model = mock.Mock()
            prefix, audio, ctx, tokens = object(), object(), object(), object()
            model.generate_prefix_inference.return_value = prefix, audio, ctx
            loaded = (model, 'tokenizer', None, None, None, {}, None)
            with self.subTest(variant=variant), mock.patch.object(selected, 'greedy_decode', return_value={}) as decoder:
                if family == 'multilayer':
                    with mock.patch.object(selected, 'generate_prefix_inference', return_value=(prefix, audio, ctx)), \
                         mock.patch.object(selected, 'sample_context', return_value={'sample': ctx}) as build:
                        runtime.prefix_and_decode(selected, family, variant, loaded, {'input': tokens})
                        build.assert_called_once_with(variant, audio, ctx, tokens)
                    expected = {'eval_variant': variant, 'context': {'sample': ctx}}
                elif family == 'recursive':
                    runtime.prefix_and_decode(selected, family, variant, loaded, {'input': tokens})
                    expected = {'eval_variant': variant, 'audio_information': ctx}
                else:
                    runtime.prefix_and_decode(selected, family, variant, loaded, {'input': tokens})
                    expected = {}
                self.assertEqual(decoder.call_args.kwargs, expected)
                self.assertIs(decoder.call_args.args[2], prefix)


class Fixture(SampleFixture):
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


class WorkflowTests(Fixture):
    def sequential(self, args, fail_rank=None, fail_after=13, seen=None):
        loaded = (None, None, 'fixture', None, None, {'status': 'PASS', 'mock_gpu_model': True}, None)
        reports = []
        for rank in range(8):
            count = 0
            def generate(_, record, **kw):
                nonlocal count
                if rank == fail_rank and count == fail_after:
                    raise RuntimeError('explicit rank interruption')
                count += 1
                if seen is not None:
                    seen.append(record['id'])
                return generation(record)
            with mock.patch.object(runtime, 'load_runtime', return_value=loaded), \
                 mock.patch.object(runtime, 'generate', side_effect=generate):
                reports.append(evaluate.run_rank(args, rank, binding(rank)))
        if any(report['status'] != 'PASS' for report in reports):
            raise RuntimeError('explicit worker group failure')

    def run_root(self, args, launch):
        with mock.patch.object(evaluate, 'launch_workers', side_effect=launch) as launcher, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            report = evaluate.run(args)
        return report, launcher.call_count

    def test_eight_concurrent_processes_merge_5000_and_rescore_without_models(self):
        args = self.args(output='eight-gpu-output')
        arguments = self.directory / 'worker_args.json'
        arguments.write_text(json.dumps(vars(args), default=str), encoding='utf-8')
        def launch(_):
            processes = []
            for rank in range(8):
                env = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='8', LOCAL_WORLD_SIZE='8',
                           CUDA_VISIBLE_DEVICES='7,6,5,4,3,2,1,0', PYTHONDONTWRITEBYTECODE='1')
                processes.append(subprocess.Popen([sys.executable, '-B', str(Path(__file__)), '--fixture-worker', str(arguments)],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
            try:
                for process in processes:
                    stdout, stderr = process.communicate(timeout=60)
                    self.assertEqual(process.returncode, 0, stdout[-2000:] + stderr)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
        report, count = self.run_root(args, launch)
        self.assertEqual(report['status'], 'PASS', report['fatal_error'])
        self.assertEqual(count, 1)
        self.assertEqual(report['accuracy']['overall'], {'correct': 4999, 'total': 5000, 'accuracy': 4999 / 5000})
        self.assertEqual([row['coverage']['generated'] for row in report['rank_coverage']], [625] * 8)
        raw = (args.output_dir / 'raw_generations.jsonl').read_bytes()
        saved = evaluate.read_jsonl(args.output_dir / 'raw_generations.jsonl')
        self.assertEqual([item['record']['id'] for item in saved], self.manifest['sample_ids'])
        compare.read_run(args.output_dir)
        compared = compare.compare_runs([('fixture', args.output_dir)], self.directory / 'eight-gpu.csv')
        self.assertEqual(compared[0]['overall_total'], 5000)
        with mock.patch.object(runtime, 'load_runtime', side_effect=AssertionError('forbidden model load')):
            again, count = self.run_root(args, lambda _: self.fail('completed shards must not launch workers'))
        self.assertEqual(again['status'], 'PASS', again['fatal_error'])
        self.assertEqual(count, 0)
        self.assertEqual(raw, (args.output_dir / 'raw_generations.jsonl').read_bytes())

    def test_interrupted_rank_recovers_partial_tail_and_only_generates_remaining_612(self):
        args = self.args()
        report, _ = self.run_root(args, lambda a: self.sequential(a, fail_rank=3))
        self.assertEqual(report['status'], 'FAILED')
        self.assertIsNone(report['accuracy'])
        self.assertFalse((args.output_dir / 'raw_generations.jsonl').exists())
        self.assertFalse((args.output_dir / 'accuracy.json').exists())
        directory = config.rank_dir(args.output_dir, 3)
        with (directory / 'raw_generations.jsonl').open('ab') as stream:
            stream.write(b'{"partial":')
        (directory / 'progress.jsonl').write_bytes(b'stale progress')
        seen = []
        with mock.patch.object(runtime, 'load_runtime', side_effect=AssertionError('outer loader must be patched')):
            report, _ = self.run_root(args, lambda a: self.sequential(a, seen=seen))
        self.assertEqual(report['status'], 'PASS', report['fatal_error'])
        self.assertEqual(seen, [self.manifest['records'][index]['id'] for index in config.rank_indices(3)[13:]])
        self.assertEqual(len(seen), 612)
        self.assertEqual(len(evaluate.read_jsonl(directory / 'progress.jsonl')), 625)
        self.assertEqual(report['accuracy']['overall']['total'], 5000)

    def test_bad_rank_raw_coverage_binding_and_changed_identity_refuse_merge(self):
        args = self.args()
        report, _ = self.run_root(args, self.sequential)
        self.assertEqual(report['status'], 'PASS', report['fatal_error'])
        expected = evaluate.read_json(args.output_dir / 'run_config.json')
        records = evaluate.read_jsonl(args.output_dir / 'selected_records.jsonl')
        directory = config.rank_dir(args.output_dir, 2)
        raw_path, report_path = directory / 'raw_generations.jsonl', directory / 'rank_report.json'
        raw_bytes, report_bytes = raw_path.read_bytes(), report_path.read_bytes()
        bad = evaluate.read_jsonl(raw_path)
        bad[0]['record'] = records[0]
        bad[0]['record_sha256'] = storage.object_sha(records[0])
        storage.write_jsonl(raw_path, bad)
        with self.assertRaisesRegex(ValueError, 'identity/hash'):
            evaluate.merge(args.output_dir, records, expected)
        raw_path.write_bytes(raw_bytes)
        wrong = json.loads(report_bytes)
        wrong['inference_coverage']['generated'] = 624
        storage.write_json(report_path, wrong)
        with self.assertRaisesRegex(ValueError, 'rank report'):
            evaluate.merge(args.output_dir, records, expected)
        wrong = json.loads(report_bytes)
        wrong['binding']['worker_cuda_visible_devices'] = binding(1)['worker_cuda_visible_devices']
        storage.write_json(report_path, wrong)
        with self.assertRaisesRegex(ValueError, 'distinct GPUs'):
            evaluate.merge(args.output_dir, records, expected)
        report_path.write_bytes(report_bytes)
        original_report = (args.output_dir / 'evaluation_report.json').read_bytes()
        args.checkpoint.write_bytes(b'changed checkpoint fixture')
        result, calls = self.run_root(args, lambda _: self.fail('changed identity cannot launch'))
        self.assertFalse(result['output_claimed'])
        self.assertEqual(calls, 0)
        self.assertEqual(original_report, (args.output_dir / 'evaluation_report.json').read_bytes())


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--fixture-worker':
        raise SystemExit(fixture_worker(sys.argv[2]))
    unittest.main()
