"""CPU contracts/control flow plus optional real tensor checks; no production GPU."""
import contextlib
import copy
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/RSmol'))
from mizar_layer_pair_search import (arguments, config, data, device, evaluate, identity,
                                     manifest, results, runtime, storage, submit, worker)


class TokenizerFixture:
    """Explicit tiny deterministic tokenizer, never used with real weights."""
    def encode(self, text, add_special_tokens=True):
        return [1 + ord(c) % 19 for c in text]

    def __call__(self, text, **kwargs):
        ids = self.encode(text)[:256]
        return {'input_ids': ids + [17]*(256-len(ids)), 'attention_mask': [1]*len(ids)+[0]*(256-len(ids))}


def source_row(dataset, index):
    base = {'id': str(index), 'question': f'Question {index}?', 'choices': ['Sound one', 'Sound two'],
            'answer': 'Sound one'}
    if dataset == 'mmsu':
        return {'id': str(index), 'question': base['question'], 'choice_a': 'Sound one', 'choice_b': None,
                'choice_c': 'Sound two', 'choice_d': None, 'answer_gt': 'Sound one', 'task_name': 'fixture'}
    if dataset == 'reasonaqa':
        return {'input': f'Question {index}? A) Sound one B) Sound two C) Sound three D) Sound four',
                'answer': 'A) Sound one', 'subtype': 'AudioCaps-MCQ.json',
                'filepath1': 'caps/test/example.wav', 'filepath2': '', 'caption1': 'SECRET_CAPTION'}
    return base


def records_fixture():
    records, _ = data.select_records({d: [(i, source_row(d, i)) for i in range(40)] for d in config.DATASETS}, TokenizerFixture())
    for row in records:
        row['choice_token_ids'] = [[i+1]*(i+1) for i in range(len(row['choice_bodies']))]
        row['audio_ref'] = {'kind': 'file', 'path': '/explicit-fixture/audio.wav'}
        row['audio_audit'] = {'waveform_sha256': 'explicit-fixture-not-production'}
    return records


def score_fixture(record, correct=True):
    scores = [-float(i+3) for i in range(len(record['choice_bodies']))]
    predicted = record['gold_index'] if correct else (record['gold_index']+1) % len(scores)
    scores[predicted] = -1.0
    counts = [len(ids) for ids in record['choice_token_ids']]
    return {'option_loglikelihood': scores, 'option_token_counts': counts,
            'option_mean_loglikelihood': [s/n for s, n in zip(scores, counts)],
            'prediction_index': predicted, 'correct': correct, 'option_score_tie': False}


def generation_fixture():
    return {'generated_token_ids': [2, 0], 'generated_token_count': 2, 'stop_reason': 'eos',
            'generated_text_raw': 'INDEPENDENT_FREE_OUTPUT<|endoftext|>',
            'generated_output': 'INDEPENDENT_FREE_OUTPUT'}


def row_fixture(record, index, correct=True, identity_sha='fixture-identity'):
    pair = config.PAIRS[index] if index < 378 else ()
    return storage.seal({'identity_sha256': identity_sha, 'mask_index': index,
        'drop_pair_0based': list(pair), 'global_index': record['global_index'], 'key': record['key'],
        'record_sha256': storage.object_sha(record), 'generation': generation_fixture() if index == 0 else None,
        'executed_original_layers': [i for i in range(30) if i not in pair], **score_fixture(record, correct)})


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP'))
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)


class DataContracts(TempCase):
    def test_378_protected_pairs_and_explicit_mask(self):
        self.assertEqual(len(set(config.PAIRS)), 378)
        for pair in config.PAIRS:
            mask = config.drop_mask(pair)
            self.assertEqual((len(mask), sum(mask), mask[0], mask[29]), (30, 2, 0, 0))
        for bad in [(0, 2), (1, 29), (2, 1), (4, 4), (1,)]:
            with self.assertRaises(ValueError):
                config.drop_mask(bad)

    def test_labelled_mmau_option_body_and_no_double_labels(self):
        row = source_row('mmau', 1)
        row.update(choices=['(A) Man', '(B) Woman'], answer='(B) Woman')
        value = data.normalize_record('mmau', 0, row)
        self.assertEqual(value['choice_bodies'], ['man', 'woman'])
        self.assertEqual(value['prompt'], 'question 1? a) man b) woman')
        self.assertEqual(value['gold_index'], 1)

    def test_mmsu_nulls_keep_valid_order_and_gt_mapping(self):
        value = data.normalize_record('mmsu', 0, source_row('mmsu', 0))
        self.assertEqual(value['valid_original_slots'], [0, 2])
        self.assertNotIn('none', value['prompt'])
        self.assertEqual(len(value['choice_bodies']), 2)

    def test_reason_parser_does_not_use_gt_or_captions_as_candidates(self):
        row = source_row('reasonaqa', 0)
        value = data.normalize_record('reasonaqa', 0, row)
        row.update(answer='D) Sound four', caption1='NEW_SECRET')
        changed = data.normalize_record('reasonaqa', 0, row)
        for field in ('prompt', 'choice_bodies', 'question_original'):
            self.assertEqual(value[field], changed[field])
        self.assertNotIn('secret', changed['prompt'])
        self.assertEqual(changed['gold_index'], 3)

    def test_reason_open_yesno_two_recordings_and_ambiguous_delimiters_rejected(self):
        for edit in ({'subtype': 'ClothoAQA.json'}, {'filepath2': 'other.wav'},
                     {'input': 'Q A) first B) second B) repeated C) third D) fourth'}):
            with self.assertRaises(data.Ineligible):
                data.normalize_record('reasonaqa', 0, {**source_row('reasonaqa', 0), **edit})

    def test_duplicate_normalized_choices_and_wrong_gt_rejected(self):
        for edit in ({'choices': ['Rain', 'RAIN']}, {'answer': 'not an option'}, {'choices': ['', 'Rain']}):
            with self.assertRaises(data.Ineligible):
                data.normalize_record('mmau', 0, {**source_row('mmau', 0), **edit})

    def test_prompt_no_eot_and_body_tokenization_has_no_specials(self):
        class Spy(TokenizerFixture):
            def __init__(self):
                self.calls = []
            def encode(self, text, add_special_tokens=True):
                self.calls.append((text, add_special_tokens))
                return super().encode(text, add_special_tokens)
        tok = Spy()
        value = data.tokenize_record(data.normalize_record('mmau', 0, source_row('mmau', 0)), tok)
        self.assertEqual(len(value['prompt_token_ids']), 256)
        self.assertTrue(all(not special for text, special in tok.calls if text in value['choice_bodies']))
        self.assertNotIn('<|endoftext|>', value['prompt'])

    def test_full_options_are_never_truncated(self):
        row = data.normalize_record('mmau', 0, source_row('mmau', 0))
        row['choice_bodies'][0] = 'x'*642
        with self.assertRaises(data.Ineligible):
            data.tokenize_record(row, TokenizerFixture())

    def test_sampling_fixed_exact25_each_and_stable(self):
        pools = {d: [(i, source_row(d, i)) for i in range(50)] for d in config.DATASETS}
        a, audit = data.select_records(pools, TokenizerFixture())
        b, _ = data.select_records(pools, TokenizerFixture())
        self.assertEqual(a, b)
        self.assertEqual(len({r['key'] for r in a}), 100)
        self.assertEqual([r['global_index'] for r in a], list(range(100)))
        self.assertTrue(all(sum(r['dataset'] == d for r in a) == 25 for d in config.DATASETS))
        self.assertEqual(len(audit), 4)

    def test_missing_candidates_fail_instead_of_changing_denominator(self):
        with self.assertRaises(ValueError):
            data.select_records({d: [(i, source_row(d, i)) for i in range(24)] for d in config.DATASETS}, TokenizerFixture())

    def test_path_escape_rejected(self):
        for path in ('../secret.wav', '/absolute.wav', 'C:/outside.wav'):
            with self.assertRaises(ValueError):
                data.contained(self.path, path)

    def test_manifest_tampering_and_wrong_coverage_rejected(self):
        value = storage.seal({'schema': config.SCHEMA, 'seed': 1234, 'protocol': config.PROTOCOL,
                              'records': records_fixture()}, 'manifest_sha256')
        data.validate_manifest(value)
        broken = copy.deepcopy(value)
        broken['records'][0]['gold_index'] = 1
        with self.assertRaises(ValueError):
            data.validate_manifest(broken)
        broken = {**value, 'records': value['records'][:-1]}
        broken = storage.seal({k:v for k,v in broken.items() if k != 'manifest_sha256'}, 'manifest_sha256')
        with self.assertRaises(ValueError):
            data.validate_manifest(broken)


class PersistenceAndResults(TempCase):
    def test_frozen_manifest_rejects_changed_source_arguments(self):
        args = arguments.parse_args(['--manifest', str(self.path/'input/manifest.json'),
                                     '--output-dir', str(self.path/'output')])
        args.manifest.parent.mkdir()
        args.manifest.write_text('{}', encoding='utf-8')
        frozen = {'source_arguments': {**manifest.source_arguments(args), 'mmau_metadata': '/other.json'}}
        with patch.object(manifest, 'load', return_value=frozen), self.assertRaises(ValueError):
            manifest.ensure(args)

    def test_reused_sources_are_unchanged_and_pinned(self):
        guard = identity.verify_guard()
        self.assertGreaterEqual(len(guard['sources_lf_sha256']), 31)
        self.assertIn('mizar_layer_pair_search/runtime.py', identity.sources())

    def test_partial_final_write_recovered_but_complete_corruption_fails(self):
        path = self.path / 'rows.jsonl'
        storage.append_jsonl(path, {'row': 1})
        with path.open('ab') as stream:
            stream.write(b'{"partial":')
        self.assertEqual(len(storage.recover_jsonl(path)), 1)
        self.assertTrue(path.read_bytes().endswith(b'\n'))
        with path.open('ab') as stream:
            stream.write(b'{broken}\n')
        with self.assertRaises(ValueError):
            storage.recover_jsonl(path)

    def test_output_identity_prevents_mixed_runs(self):
        with storage.output_lock(self.path):
            storage.claim(self.path, {'identity': 1})
            storage.claim(self.path, {'identity': 1})
            with self.assertRaises(ValueError):
                storage.claim(self.path, {'identity': 2})

    def test_choose_sum_ties_and_nonfinite(self):
        self.assertEqual(results.choose([-2.0, -2.0]), 0)
        # Raw sum chooses the first, even though its per-token mean could lose.
        self.assertEqual(results.choose([-2.0, -3.0]), 0)
        for scores in ([math.nan, 1], [1, math.inf], [1]):
            with self.assertRaises(ValueError):
                results.choose(scores)

    def test_13_12_coverage_and_count_weighted_accuracy(self):
        records = records_fixture()
        parts = [[row_fixture(records[i], 1, correct=(rank == 0)) for i in config.rank_indices(rank)] for rank in range(8)]
        rows = results.merge_rank_rows(parts, records, 1, 'fixture-identity')
        summary = results.summarize(rows, records, 1)
        self.assertEqual([len(p) for p in parts], [13]*4+[12]*4)
        self.assertEqual(summary['accuracy'], 0.13)
        self.assertNotEqual(summary['accuracy'], 1/8)
        self.assertAlmostEqual(summary['accuracy'], sum(s['accuracy'] for s in summary['subsets'].values())/4)

    def test_duplicate_missing_wrong_mask_and_wrong_gt_results_fail(self):
        records = records_fixture()
        for edit in ({'mask_index': 3}, {'prediction_index': 1}, {'correct': False}, {'executed_original_layers': list(range(30))}):
            good = row_fixture(records[0], 1)
            bad = storage.seal({**{k:v for k,v in good.items() if k != 'sha256'}, **edit})
            with self.assertRaises(ValueError):
                results.validate_row(bad, records[0], 1, 'fixture-identity')
        parts = [[row_fixture(records[i], 1) for i in config.rank_indices(rank)] for rank in range(8)]
        parts[0] = parts[0][:-1]
        with self.assertRaises(ValueError):
            results.merge_rank_rows(parts, records, 1, 'fixture-identity')

    def test_first_mask_needs100_generations_and_later_masks_forbid_them(self):
        record = records_fixture()[0]
        value = row_fixture(record, 0)
        value['generation'] = None
        with self.assertRaises(ValueError):
            results.validate_row(storage.seal({k:v for k,v in value.items() if k != 'sha256'}), record, 0, 'fixture-identity')
        value = row_fixture(record, 1)
        value['generation'] = generation_fixture()
        with self.assertRaises(ValueError):
            results.validate_row(storage.seal({k:v for k,v in value.items() if k != 'sha256'}), record, 1, 'fixture-identity')

    def test_ranking_requires_all378_and_keeps_shared_ranks(self):
        summaries = [{'mask_index': i, 'correct': 70 if i < 2 else 60, 'accuracy': .7 if i < 2 else .6,
                      'drop_pair_0based': list(pair), 'total': 100,
                      'subsets': {d:{'accuracy': .5} for d in config.DATASETS}} for i, pair in enumerate(config.PAIRS)]
        with self.assertRaises(ValueError):
            results.save_ranking(self.path, summaries[:-1])
        ordered = results.save_ranking(self.path, summaries)
        self.assertEqual([r['rank'] for r in ordered[:3]], [1, 1, 3])

    def test_resume_checks_each_record_and_only_keeps_completed_prefix(self):
        records = records_fixture()
        directory = self.path / 'ranks/rank00'
        storage.write_jsonl(directory / 'mask_001.jsonl', [row_fixture(records[i], 1) for i in config.rank_indices(0)[:3]])
        saved = results.read_rank(self.path, 0, records, 1, 'fixture-identity')
        self.assertEqual(len(saved), 3)
        with self.assertRaises(ValueError):
            results.read_rank(self.path, 0, records, 1, 'other-identity')


class LaunchAndSynchronization(TempCase):
    def test_exact_one_job_resources_and_shell_quoting(self):
        args = arguments.parse_args(['--output-dir', str(self.path / "output's folder"), '--manifest', str(self.path/'inputs/m.json')])
        command = submit.command(args, 'test123')
        self.assertEqual(command.count('submit'), 1)
        for flag, expected in [('-p','pdgpu-3090'),('-g','8'),('-n','1'),('-c','64'),('-m','256G')]:
            self.assertEqual(command[command.index(flag)+1], expected)
        import shlex
        parsed = shlex.split(command[command.index('--cmd')+1])
        self.assertEqual(parsed[:3], ['bash', 'mizar_layer_pair_search/run.sh', 'pipeline'])
        self.assertIn(str(args.output_dir), parsed)

    def test_device_visibility_handles_noncontiguous_and_UUIDs(self):
        env = {'RANK':'5','LOCAL_RANK':'5','WORLD_SIZE':'8','LOCAL_WORLD_SIZE':'8',
               'CUDA_VISIBLE_DEVICES':'GPU-a,GPU-b,GPU-c,GPU-d,GPU-e,GPU-f,GPU-g,GPU-h'}
        self.assertEqual(device.device_binding(env)['worker_cuda_visible_devices'], 'GPU-f')
        for edit in ({'WORLD_SIZE':'7'}, {'CUDA_VISIBLE_DEVICES':'0,0,1,2,3,4,5,6'}, {'LOCAL_RANK':'2'}):
            with self.assertRaises(ValueError):
                device.device_binding({**env, **edit})

    def test_torchrun_has8_workers_no_restarts_and_no_378_submissions(self):
        args = arguments.parse_args(['--output-dir', str(self.path/'run'), '--manifest', str(self.path/'in/m.json')])
        with patch.object(evaluate.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as mocked, contextlib.redirect_stdout(io.StringIO()):
            evaluate.launch(args)
        self.assertEqual(mocked.call_count, 1)
        argv = mocked.call_args.args[0]
        self.assertIn('--nproc-per-node=8', argv)
        self.assertIn('--max-restarts=0', argv)

    def test_real_worker_mask_loop_simulated8ranks_generation_and_resume(self):
        records = records_fixture()
        args = SimpleNamespace(output_dir=self.path)
        expected = {'identity_sha256':'fixture-identity'}
        gathered, events = {}, []
        class Controller:
            def set_pair(self, pair):
                self.trace = [i for i in range(30) if i not in pair]
        class Dist:
            def __init__(self, rank, index):
                self.rank, self.index = rank, index
            def broadcast_object_list(self, values, src):
                pair = config.PAIRS[self.index] if self.index < 378 else ()
                values[0] = (self.index, list(pair), 'fixture-identity')
                events.append(('broadcast', self.rank))
            def gather_object(self, rows, destination, dst):
                gathered[self.rank] = rows
                if self.rank == 0:
                    destination[:] = [gathered[r] for r in range(8)]
            def barrier(self):
                events.append(('barrier', self.rank))
        output = io.StringIO()
        with patch.object(runtime, 'generate', return_value=generation_fixture()) as generate, \
             patch.object(runtime, 'score', side_effect=lambda loaded, p, r: score_fixture(r, False)) as score, \
             contextlib.redirect_stdout(output):
            for index in (0, 1, 378):
                for rank in [1,2,3,4,5,6,7,0]:
                    worker.evaluate_mask(args, {'controller':Controller()}, {i:'prefix-only' for i in range(100)},
                                         records, expected, index, rank, Dist(rank, index))
            self.assertEqual(generate.call_count, 100)
            self.assertEqual(score.call_count, 300)
            for rank in [1,2,3,4,5,6,7,0]:
                worker.evaluate_mask(args, {'controller':Controller()}, {}, records, expected, 0, rank, Dist(rank, 0))
            self.assertEqual(generate.call_count, 100)
            self.assertEqual(score.call_count, 300)
        lines = [json.loads(line) for line in output.getvalue().splitlines()]
        samples = [r for r in lines if r['event'] == 'FIRST_MASK_SAMPLE']
        self.assertEqual(len(samples), 200)  # 100 originals and 100 honest replay logs.
        self.assertTrue(all(r['generated_output'] == 'INDEPENDENT_FREE_OUTPUT' for r in samples))
        self.assertTrue(all(r['likelihood_prediction'] != r['generated_output'] for r in samples))
        saved = [json.loads(line) for line in (self.path/'first_mask_generations.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(saved), 100)
        for row in saved:
            storage.verify_seal(row)
        self.assertEqual(events[-1], ('barrier', 0))


class FullFinalizationSimulation(TempCase):
    """Explicit synthetic CPU artifacts; never evidence of actual BF16/GPU PASS."""
    def make_runtime_fixture(self):
        expected = {'identity_sha256': 'fixture-identity'}
        audits = []
        for rank in range(8):
            audit = {'binding': {'rank': rank, 'worker_cuda_visible_devices': str(rank)},
                     'fixture_only': 'CPU_SYNTHETIC_NOT_GPU'}
            audits.append(audit)
            directory = self.path / 'ranks' / f'rank{rank:02d}'
            storage.write_json(directory/'model_audit.json', storage.seal(audit))
            storage.write_json(directory/'rank_report.json', storage.seal({
                'status': 'PASS', 'identity_sha256': 'fixture-identity', 'binding': audit['binding'],
                'masks_completed': 379, 'rows_per_mask': len(config.rank_indices(rank))}))
        entries = []
        for index in range(100):
            directory = self.path/'cache'
            directory.mkdir(exist_ok=True)
            path = directory/f'prefix_{index:03d}.pt'
            path.write_bytes(b'EXPLICIT_CPU_FIXTURE_NOT_A_REAL_TENSOR')
            meta = storage.seal({'file_sha256': storage.file_sha(path)})
            storage.write_json(directory/f'prefix_{index:03d}.json', meta)
            entries.append(meta['sha256'])
        cache = storage.seal({'rows': 100, 'cache_identity': {'run_identity_sha256': 'fixture-identity'},
                              'entry_meta_sha256': entries})
        storage.write_json(self.path/'cache/cache_manifest.json', cache)
        checks = storage.seal({'identity_sha256': 'fixture-identity', 'status': 'PASS',
            'physical_pruning_equivalence': {'status': 'PASS', 'fixture_only': True},
            'online_cache_check': {'status': 'PASS', 'fixture_only': True},
            'rank_audits': audits, 'cache_manifest_sha256': cache['sha256']})
        storage.write_json(self.path/'runtime_checks.json', checks)
        return expected

    def test_all379_rounds_finalize_exact_counts_shared_ranks_and_integrity(self):
        expected, records = self.make_runtime_fixture(), records_fixture()
        def read_fixture(directory, rank, unused_records, index, identity_sha):
            correct_count = 80 if index == 378 else 70 if index < 2 else 60
            return [row_fixture(records[i], index, i < correct_count) for i in config.rank_indices(rank)]
        with patch.object(results, 'read_rank', side_effect=read_fixture) as read:
            report = evaluate.finalize(self.path, {'records': records}, expected)
        self.assertEqual(read.call_count, 379*8)
        self.assertEqual(report['candidate_decisions']+report['baseline_decisions'], 37900)
        self.assertEqual(report['first_mask_free_generations'], 100)
        self.assertEqual(report['top_accuracy'], .7)
        self.assertEqual(report['baseline_accuracy'], .8)
        ranked = storage.read_json(self.path/'ranking.json')
        self.assertEqual(len(ranked), 378)
        self.assertEqual([r['rank'] for r in ranked[:3]], [1, 1, 3])
        self.assertEqual(len((self.path/'mask_scores.jsonl').read_text(encoding='utf-8').splitlines()), 378)
        for line in (self.path/'first_mask_generations.jsonl').read_text(encoding='utf-8').splitlines():
            storage.verify_seal(json.loads(line))
        # Tampering with persisted cache bytes is detected independently of scores.
        (self.path/'cache/prefix_099.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            evaluate.verify_runtime_artifacts(self.path, expected)

    def test_completion_rejects_missing_failed_and_duplicate_runtime_evidence(self):
        expected = self.make_runtime_fixture()
        original = storage.read_json(self.path/'runtime_checks.json')
        edits = [{'status': 'FAILED'}, {'identity_sha256': 'other'},
                 {'rank_audits': original['rank_audits'][:-1]},
                 {'rank_audits': [original['rank_audits'][0]]*8},
                 {'online_cache_check': {'status': 'FAILED'}}]
        for edit in edits:
            storage.write_json(self.path/'runtime_checks.json', storage.seal({
                **{k:v for k,v in original.items() if k != 'sha256'}, **edit}))
            with self.assertRaises(ValueError):
                evaluate.verify_runtime_artifacts(self.path, expected)


HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'requires real torch; run in remote mellow_c8204d8 for tensor checks')
class TensorLikelihood(unittest.TestCase):
    def setUp(self):
        import torch
        torch.manual_seed(1)
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.delta = torch.nn.Parameter(torch.randn(4)*.01)
            def forward(self, x):
                return x+self.delta
        class LM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([Layer() for _ in range(30)])
                self.config = SimpleNamespace(num_hidden_layers=30)
                self.embed = torch.nn.Embedding(20, 4)
                self.head = torch.nn.Linear(4, 20)
            def get_input_embeddings(self):
                return self.embed
            def forward(self, inputs_embeds, **kwargs):
                hidden = inputs_embeds
                for layer in self.model.layers:
                    hidden = layer(hidden)
                return SimpleNamespace(logits=self.head(hidden), past_key_values=None)
        self.torch, self.lm = torch, LM().eval()
        self.prefix = self.lm.embed(torch.tensor([[3,4,5]]))
        self.controller = runtime.MaskedLM(self.lm, hidden_size=4, prefix_tokens=3)
        self.device = torch.device('cpu')

    def test_causal_shift_sum_matches_manual_token_predictions(self):
        torch = self.torch
        expected = 0
        history = self.prefix
        for token in [6,7,8]:
            out = self.lm(inputs_embeds=history, attention_mask=None, use_cache=False)
            expected += float(out.logits[0,-1].log_softmax(-1)[token])
            history = torch.cat([history, self.lm.embed(torch.tensor([[token]]))], dim=1)
        actual = runtime.option_score(self.lm, self.prefix, [6,7,8], self.device)
        self.assertAlmostEqual(actual, expected, places=5)

    def test_one_token_option_uses_prefix_last_position(self):
        out = self.lm(inputs_embeds=self.prefix, attention_mask=None, use_cache=False)
        expected = float(out.logits[0,-1].log_softmax(-1)[7])
        self.assertAlmostEqual(runtime.option_score(self.lm, self.prefix, [7], self.device), expected, places=5)

    def test_mask_skips_whole_modules_and_restores_originals(self):
        original = tuple(self.lm.model.layers)
        self.controller.set_pair((1, 28))
        runtime.option_score(self.lm, self.prefix, [7,8], self.device)
        self.assertEqual(self.controller.trace, [i for i in range(30) if i not in (1,28)])
        self.assertEqual(len(self.lm.model.layers), 28)
        self.controller.set_pair(())
        self.assertEqual(tuple(self.lm.model.layers), original)


if __name__ == '__main__':
    unittest.main()
