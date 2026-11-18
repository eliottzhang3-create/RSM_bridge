"""Exercise complete offline reporting with the actual pinned official scorer."""
from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / 'code/RSmol'
sys.path.insert(0, str(RSMOL))
from mizar_mmsu_detailed_report import report
from mizar_s1_ced_mmsu_eval import dataset, parsing, scoring, storage

SCORER = ROOT / 'tests/fixtures/mmsu/mmsu_evaluation.py'


def records():
    result = []
    for index in range(5000):
        task = index % 47
        subtype = task % 4
        linguistic = subtype < 2
        third = ('Semantics', 'Phonology', 'Speaker Traits', 'Speaking Style')[subtype]
        result.append({'id': f'sample-{index}', 'task_name': f'task-{task:02d}',
                       'question': 'Which country?', 'choice_a': 'India', 'choice_b': 'China',
                       'choice_c': None, 'choice_d': 'Japan', 'answer_gt': 'India',
                       'category': 'Perception' if task % 8 < 4 else 'Reasoning',
                       'sub-category': 'Linguistics' if linguistic else 'Paralinguistics',
                       'sub-sub-category': third,
                       'linguistics_sub_discipline': ('Pragmatics' if subtype == 0 else 'Prosody') if linguistic else 'N/A',
                       'response': 'a. INDIA <|endoftext|>' if index % 3 == 0 else ('A' if index % 3 == 1 else '')})
    return result


def fixture(directory, *, raw=True, history=None):
    source = Path(directory) / "original MMSU's output"
    source.mkdir(parents=True)
    rows = records()
    saved = []
    for index, row in enumerate(rows):
        original = {key: row[key] for key in dataset.METADATA}
        generation = {'generated_text': row['response']}
        saved.append({'row_index': index, 'id': row['id'], 'official_record': original,
                      'record_sha256': storage.object_sha(original), 'generation': generation,
                      'generation_sha256': storage.object_sha(generation)})
    if raw:
        with (source / 'raw_generations.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            stream.writelines(json.dumps(row) + '\n' for row in saved)
    scoring.write_predictions(source / 'predictions_official.jsonl', saved)
    storage.write_json(source / 'run_config.json', {'fixture': 'no model assets', 'eval_variant': 'any architecture'})
    if history is not None:
        storage.write_json(source / 'evaluation_report.json', history)
    with storage.output_lock(source):
        pass
    return source, source / report.SUBDIRECTORY


def snapshot(source):
    return {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}


def historical_score(correct=1667, version=parsing.VERSION):
    return {'status': 'PASS', 'official_evaluation': {'status': 'PASS', 'response_policy': version,
            'actual_denominator': 5000, 'overall_accuracy': correct / 5000},
            'response_parsing': {'matched_rows': correct}}


class GroupingTests(unittest.TestCase):
    def test_actual_parser_47_tasks_six_domains_and_sample_weighted_counts(self):
        rows = records()
        actual = report.aggregate(rows)
        self.assertEqual(actual['overall']['correct'], 1667)
        self.assertEqual(actual['overall']['total'], 5000)
        self.assertEqual(actual['taxonomy_audit']['task_count'], 47)
        self.assertEqual(len(actual['groups']['category_subcategory']), 4)
        self.assertEqual(len(actual['groups']['leaderboard_domain']), 6)
        expected = Counter(row['task_name'] for row in rows)
        expected_correct = Counter(row['task_name'] for i, row in enumerate(rows) if i % 3 == 0)
        for group in actual['groups']['task']:
            task = group['labels']['task_name']
            self.assertEqual((group['correct'], group['total']), (expected_correct[task], expected[task]))
        for coverage in actual['group_coverage'].values():
            self.assertEqual((coverage['correct'], coverage['total']), (1667, 5000))
        self.assertNotEqual(sum(g['accuracy'] for g in actual['groups']['task']) / 47,
                            actual['overall']['accuracy'])
        self.assertFalse(actual['warnings'])

    def test_empty_unknown_and_same_named_tasks_are_not_dropped_or_guessed(self):
        rows = records()
        rows[0].update({'sub-sub-category': '', 'linguistics_sub_discipline': ''})
        rows[1]['task_name'] = rows[0]['task_name']
        actual = report.aggregate(rows)
        self.assertEqual(actual['overall']['total'], 5000)
        self.assertEqual(actual['taxonomy_audit']['empty_label_rows']['sub-sub-category'], 1)
        self.assertEqual(actual['taxonomy_audit']['unmapped_leaderboard_rows'][0]['total'], 1)
        self.assertTrue(any(g['labels']['sub-sub-category'] == '' for g in actual['groups']['taxonomy']))
        self.assertEqual(sum(g['labels']['task_name'] == 'task-00' for g in actual['groups']['task']), 1)
        self.assertGreater(sum(g['labels']['task_name'] == 'task-00' for g in actual['groups']['task_taxonomy']), 1)
        self.assertTrue(actual['warnings'])

    def test_current_text_matching_policy_retains_declared_edge_cases(self):
        rows = records()
        for row in rows:
            row['response'] = 'a. China'
        rows[0]['response'] = 'D. india <|endoftext|>'
        rows[1]['response'] = 'It is not India; maybe China.'
        rows[2]['response'] = 'A'
        rows[3]['response'] = 'Indian'
        rows[4]['response'] = 'The answer is India.'
        self.assertEqual(report.aggregate(rows)['overall']['correct'], 3)


class WorkflowTests(unittest.TestCase):
    def test_full5000_raw_real_official_scoring_csv_markdown_history_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = fixture(directory, history=historical_score())
            before = snapshot(source)
            result = report.run(source, output, SCORER)
            self.assertEqual(result['status'], 'PASS', result['fatal_error'])
            self.assertEqual(result['overall']['correct'], 1667)
            self.assertFalse(result['gpu_inference'])
            self.assertEqual(result['historical_score_comparison'][0]['status'], 'AGREES')
            self.assertEqual(result['official_evaluation']['actual_denominator'], 5000)
            self.assertEqual(snapshot(source), before)
            with (output / 'detailed_scores.csv').open(encoding='utf-8-sig', newline='') as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len([r for r in csv_rows if r['view'] == 'task']), 47)
            self.assertEqual(len([r for r in csv_rows if r['view'] == 'leaderboard_domain']), 6)
            for view in report.VIEWS:
                self.assertEqual(sum(int(r['total']) for r in csv_rows if r['view'] == view), 5000)
            self.assertIn('Linguistics sub-discipline', (output / 'detailed_scores.md').read_text(encoding='utf-8'))
            self.assertEqual(json.loads((output / 'detailed_report.json').read_text())['status'], 'PASS')
            self.assertEqual(report.run(source, output, SCORER)['status'], 'PASS')
            self.assertEqual(snapshot(source), before)

    def test_prediction_only_and_old_protocol_history_are_explicitly_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = fixture(directory, raw=False, history=historical_score(0, 'old_letter_policy'))
            before = snapshot(source)
            result = report.run(source, output, SCORER)
            self.assertEqual(result['status'], 'PASS', result['fatal_error'])
            self.assertEqual(result['source']['kind'], 'raw_predictions_only')
            self.assertEqual(result['historical_score_comparison'][0]['status'], 'NOT_COMPARED')
            self.assertEqual(snapshot(source), before)

    def test_broken_or_incomplete_input_never_overwritten_or_scored(self):
        for kind in ('raw_hash', 'tail', 'short', 'duplicate', 'metadata'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                source, output = fixture(directory, raw=kind == 'raw_hash')
                path = source / ('raw_generations.jsonl' if kind == 'raw_hash' else 'predictions_official.jsonl')
                if kind == 'raw_hash':
                    path.write_bytes(path.read_bytes().replace(b'INDIA', b'CHINA', 1))
                elif kind in ('tail', 'short'):
                    path.write_bytes(path.read_bytes().split(b'\n')[0] + (b'\n' if kind == 'short' else b''))
                else:
                    rows = parsing.read_complete_jsonl(path)
                    if kind == 'duplicate': rows[1]['id'] = rows[0]['id']
                    else: del rows[0]['sub-sub-category']
                    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
                before = snapshot(source)
                with patch.object(report.parsing, 'score_records') as score:
                    result = report.run(source, output, SCORER)
                    score.assert_not_called()
                self.assertEqual(result['status'], 'FAILED')
                self.assertEqual(snapshot(source), before)
                self.assertFalse(output.exists())

    def test_historical_v2_score_disagreement_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = fixture(directory, history=historical_score(10))
            before = snapshot(source)
            result = report.run(source, output, SCORER)
            self.assertEqual(result['status'], 'FAILED')
            self.assertIn('historical successful v2 totals disagree', result['fatal_error']['error'])
            self.assertEqual(snapshot(source), before)
            self.assertFalse(output.exists())

    def test_source_identity_change_refuses_to_replace_previous_report(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = fixture(directory)
            self.assertEqual(report.run(source, output, SCORER)['status'], 'PASS')
            before = snapshot(output)
            storage.write_json(source / 'run_config.json', {'changed': True})
            result = report.run(source, output, SCORER)
            self.assertEqual(result['status'], 'FAILED')
            self.assertEqual(snapshot(output), before)

    def test_lock_collision_and_unsafe_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = fixture(directory)
            before = snapshot(source)
            with storage.output_lock(source):
                result = report.run(source, output, SCORER)
                self.assertIn('in use', result['fatal_error']['error'])
            self.assertFalse(output.exists())
            self.assertEqual(report.run(source, source, SCORER)['status'], 'FAILED')
            self.assertEqual(report.run(source, source.parent, SCORER)['status'], 'FAILED')
            self.assertEqual(snapshot(source), before)

    def test_source_mutation_during_score_withholds_successful_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = fixture(directory)
            original = parsing.score_records
            def changed(*args):
                result = original(*args)
                storage.write_json(source / 'run_config.json', {'external_mutation': True})
                return result
            with patch.object(report.parsing, 'score_records', side_effect=changed):
                result = report.run(source, output, SCORER)
            self.assertEqual(result['status'], 'FAILED')
            self.assertIn('artifacts changed', result['fatal_error']['error'])
            self.assertFalse((output / 'detailed_scores.csv').exists())
            self.assertEqual(json.loads((output / 'detailed_report.json').read_text())['status'], 'FAILED')

    def test_official_four_group_disagreement_withholds_table(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = fixture(directory)
            original = parsing.score_records
            def changed(*args):
                parsed, official = original(*args)
                official['category_subcategory_accuracy']['Perception']['Linguistics'] = -1
                return parsed, official
            with patch.object(report.parsing, 'score_records', side_effect=changed):
                result = report.run(source, output, SCORER)
            self.assertEqual(result['status'], 'FAILED')
            self.assertIn('subcategory accuracy disagrees', result['fatal_error']['error'])
            self.assertFalse((output / 'detailed_scores.csv').exists())


class EntryTests(unittest.TestCase):
    def test_cpu_import_and_real_cli_preview_never_load_inference_dependencies(self):
        code = ('import sys; from mizar_mmsu_detailed_report import report; '
                'assert not any(n in sys.modules for n in '
                '("torch","transformers","pyarrow","soundfile","mizar_s1_ced_mmsu_eval.runtime"))')
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=RSMOL, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'missing source'
            result = subprocess.run([sys.executable, '-B', '-m', 'mizar_mmsu_detailed_report.report',
                '--source-output-dir', str(source), '--check-args'], cwd=RSMOL, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)['gpu_inference'])
            self.assertFalse(source.exists())

    def test_invalid_argument_combinations_fail_without_source_writes(self):
        for argv in (['--source-output-dir', '/one', '--source-output-dir', '/one'],
                     ['--source-output-dir', '/one', '--source-output-dir', '/two', '--output-dir', '/three'],
                     ['--source-output-dir', '/one', '--output-dir', '/one']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                report.parse_args(argv)

    def test_multi_source_continues_after_failure_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = fixture(directory, raw=False)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                status = report.main(['--source-output-dir', str(Path(directory) / 'absent'),
                    '--source-output-dir', str(source), '--evaluation-script', str(SCORER)])
            actual = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(status, 1)
            self.assertEqual([r['status'] for r in actual], ['FAILED', 'PASS'])
            self.assertTrue((output / 'detailed_scores.csv').is_file())

    def test_actual_bash_entry_scores_complete_data_without_vc_or_gpu(self):
        import test_mizar_s1_ced_mmsu_shell as shell
        helper = shell.ShellTests()
        helper.setUp()
        with tempfile.TemporaryDirectory() as directory:
            base, rs, env = helper.fixture(directory)
            shutil.copytree(RSMOL / 'mizar_mmsu_detailed_report', rs / 'mizar_mmsu_detailed_report')
            if os.name == 'nt': env['PATH'] = str(Path(helper.bash).parent) + os.pathsep + env['PATH']
            source, output = fixture(base / 'data', raw=False)
            result = subprocess.run([helper.bash, str(rs / 'mizar_mmsu_detailed_report/run.sh'),
                '--source-output-dir', str(source), '--evaluation-script', str(SCORER)],
                cwd=rs, env=env, capture_output=True, text=True, encoding='utf-8', timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)['overall']['correct'], 1667)
            self.assertFalse((base / 'submission.bin').exists())
            self.assertTrue((output / 'detailed_scores.csv').is_file())


if __name__ == '__main__':
    unittest.main()
