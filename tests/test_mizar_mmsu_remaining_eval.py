"""27 fixed targets, unchanged full MMSU execution, reuse, admission and scoring."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code/RSmol'), str(ROOT / 'tests')]
from mizar_s1_ced_mmsu_remaining_eval import checkpoint, compare, config, inspection, submit
from mizar_s1_ced_mmsu_eval import dataset, evaluate, parsing, runtime, scoring, storage
import test_mizar_s1_ced_mmsu_eval as legacy


def resolve(name='r515_inject84', extra=()):
    return submit.planned(submit.parse_args([name, *extra]))[0][2]


def fixture(directory, count=0, *, score=False):
    root = Path(directory)
    data = root / 'dataset/data'
    data.mkdir(parents=True)
    for filename in dataset.SHARD_NAMES:
        (data / filename).write_bytes(filename.encode())
    args = resolve(extra=['--dataset-dir', str(data.parent), '--output-dir', str(root / 'eval'),
                          '--evaluation-script', str(legacy.SCORER)])
    args.checkpoint = root / 'final.ckpt'
    args.runtime_config = root / 'runtime.json'
    args.checkpoint.write_bytes(b'synthetic final checkpoint, never torch loaded')
    args.runtime_config.write_text('{}')
    identity = evaluate.identity(args, dataset.shard_paths(args.dataset_dir))
    args.output_dir.mkdir()
    storage.claim_output(args.output_dir, identity)
    model = {'status': 'PASS', 'checkpoint_step': args.expected_step, 'strict_state_dict': True}
    storage.write_json(args.output_dir / 'model_audit.json', model)
    records = [legacy.record(i) for i in range(5000)]
    store = storage.GenerationStore(args.output_dir, records)
    with patch.object(storage.os, 'fsync'):
        for index, row in enumerate(records[:count]):
            store.append(index, row, legacy.generation('b. INDIA <|endoftext|>' if index % 2 == 0 else 'A'))
    if score:
        scoring.write_predictions(args.output_dir / 'predictions_official.jsonl', store.state)
        parsed, official = parsing.score_records(
            parsing.read_complete_jsonl(args.output_dir / 'predictions_official.jsonl'), legacy.SCORER, args.output_dir)
        storage.write_json(args.output_dir / 'evaluation_report.json', {
            'status': 'PASS', 'mode': 'full', 'run_identity': identity, 'response_parsing': parsed,
            'official_evaluation': official, 'inference_coverage': {'status': 'PASS', 'rows': 5000, 'skipped': 0}})
    return args, identity, store


class ScopeTests(unittest.TestCase):
    def test_exact_38_minus_confirmed11_is27_without_reasonaqa_exclusion_leak(self):
        self.assertEqual(len(runtime.VARIANTS), 38)
        self.assertEqual(len(config.EXCLUDED), 11)
        self.assertEqual(len(config.TARGETS), 27)
        self.assertEqual(set(config.VARIANT_TARGETS), set(runtime.VARIANTS) - set(config.EXCLUDED))
        for name in ('serial84', 'serial48_lora_r8'):
            self.assertIn(name, config.TARGETS)
        for name in ('official_mizar', 'base', 'recursive_5_10_5', 'recursive_mesh', 'mesh48_lora_r8',
                     'init0904', 'text_epoch1', 'init0904_serial84_r8'):
            with self.assertRaises(ValueError): config.target_name(name)

    def test_all_explicit_paths_match_previous_checkpoint_evidence_and_training_routes(self):
        previous = {}
        for package in ('mizar_s1_ced_reasonaqa5000_eval', 'mizar_s1_ced_reasonaqa5000_selected7_8gpu_eval'):
            previous.update(json.loads((config.ROOT / package / 'targets.json').read_text()))
        items = submit.planned(submit.parse_args(['all']))
        self.assertEqual(sum(a.expected_step == 34626 for _, _, a in items), 3)
        for name, argv, args in items:
            old = previous[name]
            self.assertEqual(args.eval_variant, old['eval_variant'])
            self.assertEqual(args.checkpoint, (Path(old['run_root']) / f'formal/checkpoints/model--step-{old["expected_step"]}.ckpt').resolve())
            self.assertEqual(args.runtime_config, (Path(old['run_root']) / 'formal/runtime.json').resolve())
            self.assertEqual(args.route_root, (config.ROOT / old['route_root']).resolve())
            self.assertEqual(args.expected_geometry, (8, 4, 4) if args.expected_step == 34626 else (8, 8, 4))
            self.assertEqual(args.output_dir, (Path(old['run_root']) / 'eval' / dataset.OUTPUT_NAMESPACE / f'step{args.expected_step}').resolve())
            selected, family, default_route = runtime.adapter(args.eval_variant)
            self.assertEqual(family, config.TARGETS[name]['runtime_family'])
            self.assertEqual(args.route_root, default_route.resolve())
            spec = importlib.util.spec_from_file_location('remaining_contract_' + name, args.route_root / 'contract.py')
            contract = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(contract)
            self.assertEqual(contract.ROUTE, selected.variant_spec(args.eval_variant)['route'])
            self.assertEqual(contract.PREFIX_TOKENS, 383)
            self.assertEqual(contract.SEED, 1234)

    def test_all_commands_are_single_4090_jobs_and_use_original_compute_runner(self):
        jobs = []
        for name, argv, resolved in submit.planned(submit.parse_args(['all'])):
            command = submit.command(name, argv, 'batch-123')
            for key, value in (('-p', 'pdgpu-4090'), ('-g', '1'), ('-c', '8'), ('-m', '32G'), ('-n', '1')):
                self.assertEqual(command[command.index(key) + 1], value)
            job = command[command.index('-j') + 1]
            self.assertLessEqual(len(job), 52)
            self.assertLess(len(job), 60)
            jobs.append(job)
            compute = shlex.split(command[command.index('--cmd') + 1])
            self.assertEqual(compute[:2], ['bash', 'mizar_s1_ced_mmsu_eval/run.sh'])
            actual = evaluate.parse_args(compute[2:])
            self.assertEqual(vars(actual), vars(resolved))
        self.assertEqual(len(set(jobs)), 27)

    def test_complete_job_names_stay_short_for_all_targets_and_long_batch_suffixes(self):
        batches = ('CHECK_ONLY', '20261116_235959_999999_1',
                   '20261116_235959_999999_12345678901234567890', 'batch-' + '9' * 1000)
        all_jobs = []
        for tag in batches:
            jobs = []
            for name, argv, _ in submit.planned(submit.parse_args(['all'])):
                command = submit.command(name, argv, tag)
                job = command[command.index('-j') + 1]
                self.assertLessEqual(len(job.encode('ascii')), 52)
                self.assertRegex(job, r'^mmsu27-[a-z0-9-]+-[0-9a-f]{12}$')
                jobs.append(job)
            self.assertEqual(len(set(jobs)), 27)
            all_jobs.extend(jobs)
        self.assertEqual(len(set(all_jobs)), len(all_jobs))

    def test_short_alias_collision_keeps_digest_of_complete_target_and_batch(self):
        prefix = 'shared-long-target-prefix-' + 'x' * 80
        first = submit.job_name(prefix + '-a', 'batch1')
        other_target = submit.job_name(prefix + '-b', 'batch1')
        other_batch = submit.job_name(prefix + '-a', 'batch2')
        self.assertEqual(first.rsplit('-', 1)[0], other_target.rsplit('-', 1)[0])
        self.assertEqual(len({first, other_target, other_batch}), 3)
        self.assertTrue(all(len(job) == 52 for job in (first, other_target, other_batch)))

    def test_rejects_excluded_targets_shared_outputs_and_model_overrides(self):
        for args in (['base'], ['official_mizar'], ['all', '--output-dir', '/shared'],
                     ['all', '--retry-submitted'], ['serial84', '--run-root', '/wrong'],
                     ['serial84', '--expected-step', '1'], ['serial84', '--parquet-batch-size', '0']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                submit.parse_args(args)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'formal/checkpoints/model--step-17313.ckpt'
            actual = resolve('mesh_inject48', ['--checkpoint', str(path)])
            self.assertEqual(actual.runtime_config, Path(directory).resolve() / 'formal/runtime.json')


class InspectionTests(unittest.TestCase):
    def test_complete_v2_is_verified_from_all5000_raw_and_official_score_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = fixture(directory, 5000, score=True)
            before = {p.name: storage.file_sha(p) for p in args.output_dir.iterdir()}
            result = inspection.inspect(args)
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertEqual(result['score']['correct'], 2500)
            self.assertEqual(result['score']['total'], 5000)
            self.assertEqual(result['score']['accuracy'], .5)
            self.assertEqual(before, {p.name: storage.file_sha(p) for p in args.output_dir.iterdir()})

    def test_partial_identity_exact_resume_and_unfinished_tail_readonly(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = fixture(directory, 13)
            raw = args.output_dir / 'raw_generations.jsonl'
            with raw.open('ab') as stream: stream.write(b'{unfinished')
            before = raw.read_bytes()
            result = inspection.inspect(args)
            self.assertEqual(result, {'status': 'RESUME', 'rows': 13, 'unfinished_tail': True})
            self.assertEqual(raw.read_bytes(), before)
            with patch.object(evaluate, 'identity', return_value={'changed_code': True}), self.assertRaises(ValueError):
                inspection.inspect(args)

    def test_completed_without_scoring_is_cpu_only_rescore_and_retains_raw_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = fixture(directory, 5000)
            before = (args.output_dir / 'raw_generations.jsonl').read_bytes()
            self.assertEqual(inspection.inspect(args)['status'], 'NEEDS_RESCORE')
            options = submit.parse_args(['r515_inject84', '--ledger-dir', str(Path(directory) / 'ledger')])
            with patch.object(submit.subprocess, 'run', wraps=subprocess.run) as calls:
                result = submit.dispatch(options, 'r515_inject84', [], args, 'tag', inspection.FileCache())
            self.assertEqual(result['status'], 'RESCORED_CPU', result)
            self.assertFalse(result['vc_attempted'])
            self.assertTrue(all('vc' not in str(call.args[0]) for call in calls.call_args_list))
            self.assertEqual(inspection.inspect(args)['status'], 'COMPLETE')
            self.assertEqual((args.output_dir / 'raw_generations.jsonl').read_bytes(), before)

    def test_wrong_model_dataset_protocol_and_corrupt_raw_are_never_silently_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            args, identity, _ = fixture(directory, 13)
            for field, value in (('eval_variant', 'base'), ('expected_step', 1), ('checkpoint', {'path': '/wrong'}),
                                 ('protocol', {'prompt_policy': 'lowercase'}), ('model_sources', {})):
                changed = {**identity, field: value}
                changed['identity_sha256'] = storage.object_sha({k: v for k, v in changed.items() if k != 'identity_sha256'})
                storage.write_json(args.output_dir / 'run_config.json', changed)
                with self.subTest(field=field), self.assertRaises(ValueError): inspection.inspect(args)
            storage.write_json(args.output_dir / 'run_config.json', identity)
            raw = args.output_dir / 'raw_generations.jsonl'
            raw.write_bytes(raw.read_bytes().replace(b'INDIA', b'CHINA', 1))
            before = raw.read_bytes()
            with self.assertRaises(ValueError): inspection.inspect(args)
            self.assertEqual(raw.read_bytes(), before)

    def test_busy_output_returns_skip_without_reading_or_mutating_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = fixture(directory, 13)
            before = (args.output_dir / 'raw_generations.jsonl').read_bytes()
            with storage.output_lock(args.output_dir):
                self.assertEqual(inspection.inspect(args)['status'], 'BUSY')
            self.assertEqual((args.output_dir / 'raw_generations.jsonl').read_bytes(), before)

    def test_lock_alone_already_blocks_duplicate_submission_during_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            args = resolve(extra=['--output-dir', directory])
            with storage.output_lock(args.output_dir):
                self.assertEqual(inspection.inspect(args)['status'], 'BUSY')
            self.assertEqual(inspection.inspect(args)['status'], 'NEW')

    def test_tampered_complete_score_is_withheld(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = fixture(directory, 5000, score=True)
            path = args.output_dir / 'evaluation_report.json'
            report = inspection.read_json(path)
            report['official_evaluation']['overall_accuracy'] = 1.
            storage.write_json(path, report)
            self.assertEqual(inspection.inspect(args)['status'], 'NEEDS_RESCORE')


class DispatchTests(unittest.TestCase):
    def test_all27_submitted_once_then_queued_reservations_prevent_duplicate_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with (patch.object(config, 'DEFAULT_LEDGER', base / 'ledger'),
                  patch.object(inspection, 'inspect', return_value={'status': 'NEW', 'rows': 0}),
                  patch.object(checkpoint, 'inspect', return_value={'status': 'PASS_CPU_CONTRACT_ONLY'}) as candidate,
                  patch.object(submit.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'job accepted', '')) as vc,
                  contextlib.redirect_stdout(io.StringIO())):
                first = submit.main(['all', '--submission-report', str(base / 'first.json')])
                second = submit.main(['all', '--submission-report', str(base / 'second.json')])
            self.assertEqual((first, second), (0, 0))
            self.assertEqual(vc.call_count, 27)
            candidate.assert_called_once()
            self.assertTrue(all(item['status'] == 'SKIPPED_SUBMITTED' for item in inspection.read_json(base / 'second.json')['targets']))

    def test_candidate_and_submission_failures_continue_other_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            failed = Mock(side_effect=[subprocess.CompletedProcess([], 1, '', 'queue rejected')] +
                          [subprocess.CompletedProcess([], 0, 'accepted', '')] * 25)
            with (patch.object(config, 'DEFAULT_LEDGER', base / 'ledger'),
                  patch.object(inspection, 'inspect', return_value={'status': 'NEW', 'rows': 0}),
                  patch.object(checkpoint, 'inspect', side_effect=ValueError('wrong actual route')),
                  patch.object(submit.subprocess, 'run', failed), contextlib.redirect_stdout(io.StringIO())):
                code = submit.main(['all', '--submission-report', str(base / 'batch.json')])
            report = inspection.read_json(base / 'batch.json')
            self.assertEqual(code, 1)
            self.assertEqual((report['submitted'], report['failed']), (25, 2))
            self.assertEqual(failed.call_count, 26)

    def test_busy_and_complete_outputs_never_submit_and_single_retry_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            options = submit.parse_args(['serial84', '--ledger-dir', directory])
            name, argv, args = submit.planned(options)[0]
            with patch.object(submit.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')) as vc:
                for state in ('COMPLETE', 'BUSY'):
                    with patch.object(inspection, 'inspect', return_value={'status': state}):
                        result = submit.dispatch(options, name, argv, args, 'tag', inspection.FileCache())
                    self.assertTrue(result['status'].startswith('SKIPPED'))
                vc.assert_not_called()
                with patch.object(inspection, 'inspect', return_value={'status': 'NEW', 'rows': 0}):
                    submit.dispatch(options, name, argv, args, 'tag', inspection.FileCache())
                    options.retry_submitted = True
                    result = submit.dispatch(options, name, argv, args, 'tag2', inspection.FileCache())
                self.assertEqual(result['status'], 'SUBMITTED')
                self.assertEqual(vc.call_count, 2)

    def test_actual_cpu_candidate_validator_rejects_other_route_and_does_not_build_model(self):
        import importlib.util
        from types import SimpleNamespace
        import test_mizar_selected4_eval as helper
        tool = helper.RuntimeTests()
        tool.select('recursive_mesh_layer4_8_inject')
        _, saved, current = tool.fixture()
        with tempfile.TemporaryDirectory() as directory:
            args = resolve('mesh_inject48')
            args.checkpoint, args.runtime_config = Path(directory) / 'model.ckpt', Path(directory) / 'runtime.json'
            args.checkpoint.write_bytes(b'fixture')
            args.runtime_config.write_text(json.dumps(current))
            fake = SimpleNamespace(load=Mock(return_value=saved))
            with (patch.dict(sys.modules, {'torch': fake}),
                  patch.object(tool.runtime, 'import_training_route', return_value={'contract': tool.route})):
                self.assertEqual(checkpoint.inspect(args)['status'], 'PASS_CPU_CONTRACT_ONLY')
                fake.load.assert_called_with(args.checkpoint, map_location='cpu', weights_only=False)
                saved['route_variant'] = 'wrong'
                with self.assertRaises(ValueError): checkpoint.inspect(args)


class ComparisonTests(unittest.TestCase):
    def test_27_actual_paths_produce_verified_summary_and_missing_scores_stay_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'summary.csv'
            common = {'shards': [{'sha256': 'same_data'}], 'protocol': runtime.PROTOCOL,
                      'evaluation_code': {'shared.py': 'same_source'}, 'scoring_policy': parsing.POLICY}
            state = {'status': 'COMPLETE', 'rows': 5000, 'common': common, 'metadata_sha256': 'same_rows',
                     'score': {'correct': 2500, 'total': 5000, 'accuracy': .5, 'score_dir': '/scored'}}
            with patch.object(inspection, 'inspect', return_value=state), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compare.main(['--output', str(output)]), 0)
            result = inspection.read_json(output.with_suffix('.json'))
            self.assertEqual(result['complete_targets'], 27)
            self.assertEqual({r['target'] for r in result['rows']}, set(config.TARGETS))
            with (patch.object(inspection, 'inspect', side_effect=[state] * 26 + [{'status': 'NEW'}]),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(compare.main(['--output', str(output)]), 1)
            self.assertEqual(inspection.read_json(output.with_suffix('.json'))['rows'][-1]['accuracy'], '')

    def test_real_complete_raw_to_csv_and_data_or_source_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = fixture(directory, 5000, score=True)
            output = Path(directory) / 'scores.csv'
            with patch.object(submit, 'planned', return_value=[('r515_inject84', [], args)]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compare.main(['r515_inject84', '--output', str(output)]), 0)
            self.assertEqual(inspection.read_json(output.with_suffix('.json'))['rows'][0]['accuracy'], .5)
            good = inspection.inspect(args)
            for field in ('shards', 'evaluation_code'):
                bad = copy.deepcopy(good)
                bad['common'][field] = [{'sha256': 'different'}] if field == 'shards' else {next(iter(good['common'][field])): 'changed'}
                with (patch.object(inspection, 'inspect', side_effect=[good, bad]),
                      patch.object(submit, 'planned', return_value=[('a', [], args), ('b', [], args)]),
                      contextlib.redirect_stdout(io.StringIO())):
                    self.assertEqual(compare.main(['--output', str(output)]), 1)


if __name__ == '__main__':
    unittest.main()
