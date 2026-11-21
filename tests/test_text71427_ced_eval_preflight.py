"""Reproduce the reported base/listen misbinding before vc is allowed to run."""
import copy
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code/RSmol'), str(ROOT / 'tests')]
from text71427_ced_eval import preflight, registry, runtime, submit
from test_text71427_ced_eval import checkpoint_fixture


def target(root, alias, step=34626):
    checkpoint = root / alias / 'formal/checkpoints' / f'model--step-{step}.ckpt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'synthetic checkpoint; only torch.load is mocked')
    module, saved, config = checkpoint_fixture(alias, step)
    path = checkpoint.parent.parent / 'runtime.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    return checkpoint, path, module, saved, config


class PreflightTests(unittest.TestCase):
    def test_reported_listen_checkpoint_bound_as_base_rejected_before_torch_and_vc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            checkpoint, _, _, _, _ = target(root, 'listen')
            argv = ['base', 'all', '--checkpoint', str(checkpoint), '--submission-report', str(root/'submission.json')]
            with patch.object(submit, 'ROOT', root), redirect_stdout(io.StringIO()), \
                 patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=Mock(side_effect=AssertionError('weights read')))}), \
                 patch.object(submit.subprocess, 'run', side_effect=AssertionError('GPU submitted')) as vc:
                self.assertEqual(submit.main(argv), 1)
                vc.assert_not_called()
            report = json.loads((root/'submission.json').read_text())
            self.assertEqual(len(report['preflights']), 1)
            diagnostic = report['preflights'][0]['diagnostic']
            self.assertEqual(diagnostic['requested_model'], 'base')
            self.assertEqual(diagnostic['runtime_model'], 'listen')
            self.assertIn('runtime.route', [p['field'] for p in diagnostic['problems']])
            self.assertEqual(len(report['jobs']), 4)
            self.assertTrue(all(j['status'] == 'FAILED' and not j['vc_attempted'] for j in report['jobs']))

    def test_gpu_validator_reports_exact_checkpoint_field_and_correct_alias(self):
        module, saved, config = checkpoint_fixture('listen')
        with self.assertRaises(preflight.CheckpointBindingError) as context:
            runtime.validate_checkpoint(saved, config, module, 5771, (8,4,4), eval_variant='base')
        diagnostic = context.exception.diagnostic
        self.assertEqual(diagnostic['checkpoint_model'], 'listen')
        mismatch = next(p for p in diagnostic['problems'] if p['field'] == 'checkpoint.route_variant')
        self.assertEqual(mismatch['expected'], registry.ROUTES['base']['route'])
        self.assertEqual(mismatch['actual'], registry.ROUTES['listen']['route'])

    def test_real_cpu_preflight_control_flow_all_six_no_model_construction(self):
        for alias in registry.ROUTES:
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as directory:
                checkpoint, _, _, saved, _ = target(pathlib.Path(directory), alias)
                args = submit.planned(submit.parse_args([alias, 'all', '--checkpoint', str(checkpoint)]))[0]['resolved']
                load = Mock(return_value=saved)
                with patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=load)}), \
                     patch.object(runtime.reference, 'import_training_route', side_effect=AssertionError('model imported')):
                    result = preflight.check(args)
                self.assertEqual(result['status'], 'READY_FOR_GPU')
                self.assertEqual(result['checkpoint_step'], 34626)
                self.assertEqual(result['model'], alias)
                load.assert_called_once_with(checkpoint, map_location='cpu', weights_only=False, mmap=True)

    def test_auto_matches_exact_formal_runtime_route_for_all_six_and_pins_gpu_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            for alias in registry.ROUTES:
                checkpoint, _, _, _, _ = target(pathlib.Path(directory), alias)
                jobs = submit.planned(submit.parse_args(['auto', 'all', '--checkpoint', str(checkpoint), '--check-args']))
                self.assertEqual(len(jobs), 4)
                for job in jobs:
                    self.assertEqual(job['requested_model'], 'auto')
                    self.assertEqual(job['model'], alias)
                    self.assertEqual(job['argv'][job['argv'].index('--eval-variant')+1], alias)
                    self.assertEqual(job['resolved'].route_root, registry.ROUTES[alias]['route_root'])
                    self.assertIn('text71427_ced_v2', job['resolved'].output_dir.parts)

    def test_auto_rejects_unknown_route_bad_hash_and_inconsistent_embedded_route(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path, _, _, original = target(pathlib.Path(directory), 'listen')
            for mutation in ('unknown', 'hash', 'embedded', 'mode'):
                config = copy.deepcopy(original)
                if mutation == 'unknown': config['route'] = 'not_a_registered_route'
                elif mutation == 'hash': config['contract_sha256'] = 'corrupted'
                elif mutation == 'embedded': config['contract']['route'] = registry.ROUTES['base']['route']
                else: config['mode'] = 'smoke'
                path.write_text(json.dumps(config))
                with self.subTest(mutation=mutation), self.assertRaises(preflight.CheckpointBindingError):
                    preflight.infer_model(path)

    def test_missing_schema_mode_state_and_wrong_checkpoint_route_still_rejected(self):
        for alias in registry.ROUTES:
            module, original, config = checkpoint_fixture(alias)
            for key, value in [('schema_version', 2), ('mode', 'smoke'), ('state_dict', {}),
                               ('route_variant', 'foreign'), ('contract', None)]:
                saved = {**original, key: value}
                with self.subTest(alias=alias, key=key), self.assertRaises(preflight.CheckpointBindingError) as context:
                    runtime.validate_checkpoint(saved, config, module, 5771, (8,4,4), eval_variant=alias)
                self.assertTrue(any(p['field'].startswith('checkpoint.') for p in context.exception.diagnostic['problems']))

    def test_cpu_checks_full_contract_and_detects_checkpoint_changed_while_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, path, module, saved, config = target(pathlib.Path(directory), 'base')
            args = submit.planned(submit.parse_args(['base', 'mmar', '--checkpoint', str(checkpoint)]))[0]['resolved']
            bad = copy.deepcopy(saved)
            bad['contract']['code_sha256'] = 'changed training source'
            bad['contract_sha256'] = module.digest(bad['contract'])
            config.update(contract=bad['contract'], contract_sha256=bad['contract_sha256'])
            path.write_text(json.dumps(config))
            with patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=lambda *a,**kw: bad)}), \
                 self.assertRaisesRegex(ValueError, 'training source differs'):
                preflight.check(args)
            _, saved, config = checkpoint_fixture('base', 34626)
            path.write_text(json.dumps(config))
            def changed(*a, **kw):
                checkpoint.write_bytes(b'replaced while CPU was reading the checkpoint')
                return saved
            with patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=changed)}), \
                 self.assertRaisesRegex(ValueError, 'changed during CPU preflight'):
                preflight.check(args)

    def test_contract_import_ignores_global_contract_module_and_preserves_it(self):
        foreign = types.SimpleNamespace(ROUTE='foreign route')
        with patch.dict(sys.modules, {'contract': foreign}):
            for alias in registry.ROUTES:
                loaded = preflight.load_contract(alias)
                self.assertEqual(loaded.ROUTE, registry.ROUTES[alias]['route'])
                self.assertIs(sys.modules['contract'], foreign)

    def test_batch_16_jobs_checks_each_checkpoint_once_and_continues_after_vc_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            entries, payloads = [], {}
            for alias in ('base', 'audio7_read2', 'listen', 'loop_lora'):
                checkpoint, _, _, saved, _ = target(root, alias)
                entries.append({'model': alias, 'checkpoint': str(checkpoint)})
                payloads[checkpoint] = saved
            manifest = root/'targets.json'
            manifest.write_text(json.dumps(entries))
            load = Mock(side_effect=lambda p, **kw: payloads[p])
            good = types.SimpleNamespace(returncode=0, stdout='mock submit', stderr='')
            bad = types.SimpleNamespace(returncode=1, stdout='', stderr='mock vc failure')
            argv = ['batch', 'all', '--targets-json', str(manifest), '--submission-report', str(root/'submit.json')]
            with patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=load)}), \
                 patch.object(submit, 'ROOT', root), redirect_stdout(io.StringIO()), \
                 patch.object(submit.subprocess, 'run', side_effect=[bad]+[good]*15) as vc:
                self.assertEqual(submit.main(argv), 1)
            self.assertEqual(load.call_count, 4)
            self.assertEqual(vc.call_count, 16)
            report = json.loads((root/'submit.json').read_text())
            self.assertEqual(sum(j['status']=='SUBMITTED' for j in report['jobs']), 15)
            self.assertEqual(len(report['preflights']), 4)
            self.assertTrue(all(j['queue']=='pdgpu-4090' for j in report['jobs']))

    def test_preflight_only_never_calls_vc_and_epoch_step_comes_from_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            checkpoint, _, _, saved, _ = target(root, 'listen')
            epoch = checkpoint.with_name('model--epo-3.ckpt')
            checkpoint.rename(epoch)
            argv = ['auto', 'all', '--checkpoint', str(epoch), '--preflight-only',
                    '--submission-report', str(root/'submit.json')]
            with patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=Mock(return_value=saved))}), \
                 patch.object(submit, 'ROOT', root), redirect_stdout(io.StringIO()), \
                 patch.object(submit.subprocess, 'run', side_effect=AssertionError('GPU submitted')):
                self.assertEqual(submit.main(argv), 0)
            report = json.loads((root/'submit.json').read_text())
            self.assertEqual(report['status'], 'READY_FOR_GPU')
            self.assertTrue(all(j['preflight']['checkpoint_step']==34626 for j in report['jobs']))
            self.assertTrue(all(not j['vc_attempted'] for j in report['jobs']))

    def test_valid_runtime_with_wrong_checkpoint_header_blocks_only_its_four_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            entries, payloads = [], {}
            for alias in ('base', 'listen'):
                checkpoint, _, _, saved, _ = target(root, alias)
                entries.append({'model': alias, 'checkpoint': str(checkpoint)})
                payloads[checkpoint] = saved
                if alias == 'listen':
                    saved['route_variant'] = registry.ROUTES['base']['route']
            manifest = root/'targets.json'
            manifest.write_text(json.dumps(entries))
            argv = ['batch','all','--targets-json',str(manifest),'--submission-report',str(root/'submit.json')]
            good = types.SimpleNamespace(returncode=0, stdout='mock submitted', stderr='')
            with patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=lambda p,**kw: payloads[p])}), \
                 patch.object(submit, 'ROOT', root), redirect_stdout(io.StringIO()), \
                 patch.object(submit.subprocess, 'run', return_value=good) as vc:
                self.assertEqual(submit.main(argv), 1)
            self.assertEqual(vc.call_count, 4)
            report = json.loads((root/'submit.json').read_text())
            self.assertTrue(all(j['status']=='SUBMITTED' for j in report['jobs'][:4]))
            self.assertTrue(all(j['status']=='FAILED' and not j['vc_attempted'] for j in report['jobs'][4:]))

    def test_user_four_checkpoint_manifest_has_exact_alias_binding_v2_outputs_and_4090(self):
        manifest = registry.PACKAGE/'rerun_20261121.json'
        jobs = submit.planned(submit.parse_args(['batch', 'all', '--targets-json', str(manifest), '--check-args']))
        self.assertEqual(len(jobs), 16)
        self.assertEqual([j['model'] for j in jobs[::4]], ['base','audio7_read2','listen','loop_lora'])
        expected_runs = ['formal_20261119_141531507533438','formal_20261120_111448563391250',
                         'formal_20261119_141724655549596','formal_20261119_141712678007968']
        for job, run in zip(jobs[::4], expected_runs):
            self.assertEqual(job['resolved'].run_root.name, run)
        for job in jobs:
            self.assertEqual(job['resolved'].expected_step, 34626)
            self.assertEqual(job['queue'], 'pdgpu-4090')
            self.assertIn('text71427_ced_v2', job['resolved'].output_dir.parts)


if __name__ == '__main__':
    unittest.main()
