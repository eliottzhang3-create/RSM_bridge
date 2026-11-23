"""Offline MyEval acceptance. CUDA, audio decoding and production weights are not exercised."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code/RSmol'), str(ROOT / 'tests')]
from myeval_mmar8 import arguments, benchmark, compare, dataset, evaluate, registry, runtime, scoring, sources, storage, submit

SCORER = ROOT / 'tests/fixtures/myeval_mmar8_official_scorer.py'


def fixture_rows():
    """Synthetic 8/67/53/14 inventory, with unmistakably artificial answers."""
    rows = []
    for i in range(8):
        questions = [{'n': 1, 'type': 'open-ended', 'dim': 'C', 'text': f'Describe synthetic audio {i}.'}]
        if i < 6:
            questions.append({'n': 2, 'type': 'open-ended', 'dim': 'P', 'text': f'Name synthetic event {i}.'})
        rows.append({'benchmark': 'MMAR', 'item_id': f'synthetic_audio_{i}', 'row_index': i,
                     'item_no': i + 1, 'layer': 'Perception Layer', 'sub_category': 'Synthetic',
                     'modality': 'sound', 'audio': {'path': f'audio/{i}.wav', 'timestamp': '00:00:00,00:00:30'},
                     'original': {'question': 'OLD_QUESTION_MUST_NOT_ENTER_MODEL'},
                     'models': {'old': {'raw': 'OLD_ANSWER_MUST_NOT_ENTER_MODEL'}}, 'questions': questions})
    dims = ['P'] * 29 + ['S'] * 11 + ['W'] * 7 + ['R'] * 6
    counts = [2] * 28 + [3] + [4] * 23 + [5]
    for i, (dim, n) in enumerate(zip(dims, counts)):
        q = {'n': len(rows[i % 8]['questions']) + 1, 'type': 'close-ended', 'dim': dim,
             'valid': True, 'text': f'Which synthetic event is number {i}?', 'difficulty': i % 3 + 1,
             'options': [{'label': chr(97 + j), 'text': f'event_{i}_{j}', 'correct': j == i % n}
                         for j in range(n)], 'answer': [f'event_{i}_{i % n}']}
        rows[i % 8]['questions'].append(q)
    for row in rows:
        row['question_count'] = len(row['questions'])
    return rows


def fixture_metadata(rows):
    official = [{'id': f'other_{i}', 'audio_path': f'audio/{i}.wav', 'question': 'official question',
                 'choices': ['correct', 'incorrect'], 'answer': 'correct', 'modality': 'sound',
                 'category': 'Perception Layer', 'sub-category': 'Synthetic'} for i in range(1000)]
    for row in rows:
        official[row['row_index']].update(id=row['item_id'], audio_path=row['audio']['path'])
    return official


def fixture_generation(record, target='base', answer=None):
    text = record.get('answer') or 'synthetic open response' if answer is None else answer
    tokens = [int(hashlib.sha256(record['prompt'].encode()).hexdigest()[:6], 16) % 100 + 1] * 256
    mask, audio = [1] * 256, ['1' * 64, '2' * 64]
    return {'generated_text': text, 'generated_text_raw': text, 'generated_token_ids': [0],
            'generated_token_count': 1, 'max_new_tokens': 300, 'stop_reason': 'eos_token',
            'prefix_shape': [1, 383, 576], 'first_answer_logit_position': 382,
            'prompt_used': record['prompt'], 'prompt_token_ids': tokens, 'prompt_attention_mask': mask,
            'audio_half_sha256': audio, 'prompt_truncated': False, 'prompt_original_token_count': 50,
            'input_fingerprint_sha256': storage.object_sha({'prompt_token_ids': tokens,
                'prompt_attention_mask': mask, 'audio_half_sha256': audio}),
            'intervention': registry.intervention(target)}


def fixture_state(records, target='base'):
    return [{'row_index': i, 'record': r, 'record_sha256': storage.object_sha(r),
             'generation': fixture_generation(r, target),
             'generation_sha256': storage.object_sha(fixture_generation(r, target))}
            for i, r in enumerate(records)]


def fixture_args(root, target='base', task='evaluate'):
    args = arguments.parse_args(task, ['--eval-variant', target, '--output-dir', str(root / target)])
    args.annotations = root / 'annotations.jsonl'
    args.metadata_json = root / 'metadata.json'
    args.audio_root = root / 'mmar-audio'
    args.checkpoint = root / ('base.ckpt' if target == 'base' else 'read2.ckpt')
    args.runtime_config = root / ('base-runtime.json' if target == 'base' else 'read2-runtime.json')
    args.evaluation_script = SCORER
    args.sources_cache = root / 'sources'
    args.route_root = sources.route_root(target, args.sources_cache)
    return args


def fixture_files(root, rows):
    for filename in ('base.ckpt', 'read2.ckpt', 'base-runtime.json', 'read2-runtime.json'):
        (root / filename).write_text('explicit synthetic checkpoint/config', encoding='utf-8')
    (root / 'annotations.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    (root / 'metadata.json').write_text(json.dumps(fixture_metadata(rows)), encoding='utf-8')
    for row in rows:
        path = root / 'mmar-audio' / row['audio']['path']; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'not real audio; decoder is mocked explicitly')


def execute_fixture(args, rows, callback=None):
    loader = Mock(return_value=(None, None, None, None, None,
        {'status': 'PASS', 'eval_variant': args.eval_variant, 'explicit_fixture': True}, None))
    generator = Mock(side_effect=callback or (lambda live, record: fixture_generation(record, args.eval_variant)))
    with patch.object(dataset, 'load_annotations', return_value=rows), \
         patch.object(benchmark.mmar, 'audit_mmar_artifacts', return_value={'status': 'PASS', 'explicit_fixture': True}), \
         patch.object(runtime.sources, 'prepare'), patch.object(runtime, 'load_runtime', loader), \
         patch.object(benchmark, 'generate', generator), patch('os.fsync'), contextlib.redirect_stdout(io.StringIO()):
        report = evaluate.run(args)
    return report, loader, generator


class DatasetAndScoreTests(unittest.TestCase):
    def setUp(self):
        self.rows = fixture_rows()
        self.records = dataset.expand(self.rows, fixture_metadata(self.rows), ROOT / 'synthetic-audio')
        self.matcher, _ = scoring.load_scorer(SCORER)

    def test_inventory_and_all_index_permutations(self):
        self.assertEqual(len(self.records), 748)
        closed = [r for r in self.records if r['type'] == 'close-ended']
        self.assertEqual(len(closed), 734)
        for row in self.rows:
            for q in row['questions']:
                if q['type'] == 'open-ended': continue
                group = [r for r in closed if r['item_id'] == row['item_id'] and r['question_number'] == q['n']]
                n = len(q['options'])
                self.assertEqual([tuple(r['permutation']) for r in group], list(itertools.permutations(range(n))))
                for record in group:
                    self.assertEqual(record['choices'][record['correct_display_index']], q['answer'][0])
                    self.assertTrue(record['prompt'].endswith(f'{chr(96+n)}) {record["choices"][-1]}'.lower()))

    def test_questions_have_no_old_question_answer_or_reference_leak(self):
        for record in self.records:
            self.assertNotIn('old_question', record['prompt'])
            self.assertNotIn('old_answer', record['prompt'])
            if record['type'] == 'open-ended': self.assertEqual(record['choices'], [])

    def test_bad_label_answer_multi_correct_dimension_and_source_rejected(self):
        for change in ('answer', 'correct', 'dim', 'path', 'id'):
            rows = copy.deepcopy(self.rows)
            q = next(q for r in rows for q in r['questions'] if q['type'] == 'close-ended')
            if change == 'answer': q['answer'] = ['wrong']
            if change == 'correct': q['options'][0]['correct'] = not q['options'][0]['correct']
            if change == 'dim': q['dim'] = 'unknown'
            if change == 'path': rows[0]['audio']['path'] = '../escape.wav'
            if change == 'id': rows[0]['item_id'] = rows[1]['item_id']
            with self.subTest(change=change), self.assertRaises(ValueError):
                dataset.expand(rows, fixture_metadata(self.rows), ROOT)

    def test_all_four_threshold_boundaries(self):
        for n, cutoff in ((2, 2), (3, 6), (4, 21), (5, 115)):
            self.assertTrue(scoring.question_pass(n, cutoff, math.factorial(n)))
            self.assertFalse(scoring.question_pass(n, cutoff - 1, math.factorial(n)))
            with self.assertRaises(ValueError): scoring.question_pass(n, cutoff, math.factorial(n) - 1)

    def test_original_question_weighting_and_all_four_denominators(self):
        state = fixture_state(self.records)
        result = scoring.summarize(self.records, state, self.matcher)
        self.assertEqual(result['total']['correct'], 53)
        self.assertEqual({k: v['total'] for k, v in result['dimensions'].items()}, {'P': 29, 'S': 11, 'W': 7, 'R': 6})
        self.assertEqual(result['total']['accuracy'], 1)
        self.assertEqual(result['open_ended_count'], 14)
        self.assertFalse(result['open_ended_scored'])
        five = next(r['question_key'] for r in self.records if len(r['choices']) == 5)
        for saved in state:
            if saved['record']['question_key'] == five:
                saved['generation']['generated_text'] = 'incorrect_unrelated'
        result = scoring.summarize(self.records, state, self.matcher)
        self.assertEqual(result['total']['correct'], 52)
        self.assertEqual(result['total']['total'], 53)

    def test_incomplete_or_duplicate_permutations_rejected(self):
        state = fixture_state(self.records)
        with self.assertRaises(ValueError): scoring.summarize(self.records, state[:-1], self.matcher)
        records = copy.deepcopy(self.records); records[1] = records[0]
        state = fixture_state(records)
        with self.assertRaises(ValueError): scoring.summarize(records, state, self.matcher)

    def test_exact_official_matcher_and_pinned_integrity(self):
        self.assertTrue(self.matcher('Yes', 'a) yes<|endoftext|>', ['Yes', 'No']))
        self.assertFalse(self.matcher('Yes', 'a)', ['Yes', 'No']))
        self.assertFalse(self.matcher('Yes', 'yes and no', ['Yes', 'No']))
        # Preserve, explicitly expose, rather than silently "fix" the official set semantics.
        self.assertTrue(self.matcher('[a], [o]', '[o], [a]', ['[a], [o]', '[o], [a]']))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scorer.py'; path.write_bytes(SCORER.read_bytes() + b'\n')
            with self.assertRaises(ValueError): scoring.load_scorer(path)

    def test_annotation_identity_is_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.jsonl'; path.write_text('{}\n')
            with self.assertRaises(ValueError): dataset.load_annotations(path)


class WorkflowTests(unittest.TestCase):
    def test_three_748_runs_compare_and_cpu_rescore_preserve_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows = fixture_rows(); fixture_files(root, rows)
            directories = []
            for target in registry.ROUTES:
                args = fixture_args(root, target)
                report, loader, generator = execute_fixture(args, rows)
                self.assertEqual(report['status'], 'PASS', report.get('traceback'))
                self.assertEqual(generator.call_count, 748); loader.assert_called_once()
                self.assertEqual(report['total']['total'], 53)
                report, loader, generator = execute_fixture(args, rows)
                self.assertEqual(report['status'], 'PASS', report.get('traceback'))
                loader.assert_not_called(); generator.assert_not_called()
                directories.append(args.output_dir)
            before = {str(p): p.read_bytes() for d in directories for p in d.iterdir() if p.is_file()}
            with patch('os.fsync'):
                result = compare.compare(directories, root / 'comparison')
                for d in directories: compare.read_result(d)
            self.assertEqual(set(result), set(registry.ROUTES))
            self.assertEqual(before, {str(p): p.read_bytes() for d in directories for p in d.iterdir() if p.is_file()})
            opened = (root / 'comparison/open_answers_comparison.md').read_text(encoding='utf-8')
            for target in registry.ROUTES: self.assertIn('### ' + target, opened)
            self.assertEqual(len((root / 'comparison/open_answers_comparison.jsonl').read_text().splitlines()), 14)

    def test_interrupt_thirteen_tail_recovery_and_remaining_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows = fixture_rows(); fixture_files(root, rows)
            args = fixture_args(root)
            count = 0
            def interrupt(live, record):
                nonlocal count
                count += 1
                if count == 14: raise RuntimeError('explicit interruption fixture')
                return fixture_generation(record)
            report, _, generator = execute_fixture(args, rows, interrupt)
            self.assertEqual(report['status'], 'FAILED'); self.assertEqual(report['generated'], 13)
            self.assertFalse((args.output_dir / 'scores.json').exists())
            raw = args.output_dir / 'raw_generations.jsonl'
            with raw.open('ab') as stream: stream.write(b'{"incomplete":')
            report, _, generator = execute_fixture(args, rows)
            self.assertEqual(report['status'], 'PASS', report.get('traceback'))
            self.assertEqual(generator.call_count, 735)
            self.assertEqual(len(raw.read_text().splitlines()), 748)

    def test_corrupt_complete_raw_and_wrong_mask_output_reuse_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows = fixture_rows(); fixture_files(root, rows)
            args = fixture_args(root, 'read2')
            report, _, _ = execute_fixture(args, rows)
            self.assertEqual(report['status'], 'PASS')
            masked = fixture_args(root, 'read2_zero48'); masked.output_dir = args.output_dir
            before = (args.output_dir / 'evaluation_report.json').read_bytes()
            report, loader, _ = execute_fixture(masked, rows)
            self.assertEqual(report['status'], 'FAILED'); loader.assert_not_called()
            self.assertEqual(before, (args.output_dir / 'evaluation_report.json').read_bytes())
            raw = args.output_dir / 'raw_generations.jsonl'
            value = raw.read_bytes(); raw.write_bytes(value.replace(b'event_', b'evxnt_', 1))
            with self.assertRaises(ValueError): compare.read_result(args.output_dir)

    def test_source_or_model_inputs_changed_before_comparison_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows = fixture_rows(); fixture_files(root, rows)
            args = fixture_args(root)
            report, _, _ = execute_fixture(args, rows); self.assertEqual(report['status'], 'PASS')
            path = args.output_dir / 'run_config.json'; content = json.loads(path.read_text())
            content['evaluation_sources'] = {}; path.write_text(json.dumps(content))
            with self.assertRaises(ValueError): compare.read_result(args.output_dir)


class BindingAndIsolationTests(unittest.TestCase):
    def test_exact_four_jobs_resources_and_short_names(self):
        jobs = submit.planned(submit.parse_args(['all', '--check-args']))
        self.assertEqual([(j['target'], j['task']) for j in jobs], [
            ('base', 'myeval8'), ('read2', 'myeval8'), ('read2_zero48', 'myeval8'), ('read2_zero48', 'mmar')])
        self.assertEqual(jobs[1]['resolved'].checkpoint, jobs[2]['resolved'].checkpoint)
        self.assertEqual(len({str(j['resolved'].output_dir) for j in jobs}), 4)
        for j in jobs:
            command = submit.command(j, 'long-tag' * 100)
            for flag, value in [('-p', 'pdgpu-4090'), ('-c', '8'), ('-m', '32G'), ('-g', '1'), ('-n', '1')]:
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertLess(len(command[command.index('-j') + 1]), 60)
        with self.assertRaises(ValueError): submit.planned(submit.parse_args(['base', 'mmar']))

    def test_public_cli_preview_reads_no_checkpoint_or_audio(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'code/RSmol'))
        result = subprocess.run([sys.executable, '-B', '-m', 'myeval_mmar8.submit', 'all', '--check-args'],
                                cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len([json.loads(x) for x in result.stdout.splitlines()]), 4)

    def test_real_git_snapshot_contains_both_original_training_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            for target in ('base', 'read2', 'read2_zero48'):
                root = sources.prepare(target, directory)
                self.assertTrue((root / 'contract.py').is_file())
                module = sources.load_contract(target, directory)
                _, profile = sources.profile(target)
                self.assertEqual(module.code_digest(), profile['training_code_sha256'][target])

    def test_copied_inference_and_official_scoring_arithmetic_preserved(self):
        def function(path, name):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        root = ROOT / 'code/RSmol'
        for filename, names in [('runtime.py', ['greedy_decode', 'normalize_state']),
                                ('benchmark.py', ['_fixed20_pair', '_prepare_mmar', 'run_official_scorer', '_run'])]:
            for name in names:
                old = function(root / 'ced_eval_20261122' / filename, name)
                new = function(root / 'myeval_mmar8' / filename, name)
                self.assertEqual(ast.dump(old), ast.dump(new), name)
        old = function(root / 'ced_eval_20261122/benchmark.py', 'generate')
        new = function(root / 'myeval_mmar8/benchmark.py', 'generate')
        boundary = next(i for i, n in enumerate(new.body) if isinstance(n, ast.ImportFrom)
                        and n.module == 'mizar_s1_ced_reasonaqa5000_eval.storage')
        new.body = new.body[:boundary] + [new.body[-1]]
        self.assertEqual(ast.dump(old), ast.dump(new))

    def test_existing_tracked_sources_remain_byte_equivalent_after_lf_normalization(self):
        guard = json.loads((registry.PACKAGE / 'source_guard.json').read_text())
        changed = [name for name, wanted in guard.items()
                   if hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != wanted]
        self.assertEqual(changed, [])


MISSING = [name for name in ('torch', 'transformers', 'safetensors') if importlib.util.find_spec(name) is None]


@unittest.skipIf(MISSING, 'Real CPU tensor dependencies missing: ' + ', '.join(MISSING))
class RealZero48Tests(unittest.TestCase):
    def test_actual_read2_zero_seeds_keep_weights_lora_prefix_and_42_call_trace(self):
        import torch
        import importlib
        from transformers import LlamaConfig
        from mizar_text_7_14x2_7_mesh.contract import build_target_config
        from myeval_mmar8 import observers
        oldpath = list(sys.path)
        oldmodules = {k: v for k, v in sys.modules.items() if k == 'models' or k.startswith('models.')}
        try:
            for k in oldmodules: del sys.modules[k]
            sys.path.insert(0, str(ROOT / 'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_lora_r8'))
            implementation = importlib.import_module('models.recursive_audio7_read2')
            torch.set_num_threads(1); torch.manual_seed(1234)
            config = LlamaConfig(num_hidden_layers=30, hidden_size=16, intermediate_size=32,
                num_attention_heads=4, num_key_value_heads=2, vocab_size=43, tie_word_embeddings=True,
                pad_token_id=17, eos_token_id=0, max_position_embeddings=1024, use_cache=False,
                attention_dropout=0., _attn_implementation='eager')
            lm = implementation.MizarText7_14x2_7MeshForCausalLM(build_target_config(config)).float()
            lm.model.enable_audio_read2(); lm.model.enable_loop_lora(); lm.eval()
            with torch.no_grad():
                for name, value in lm.named_parameters():
                    if 'lora_B' in name: value.add_(.03)
            before = {k: v.clone() for k, v in lm.state_dict().items()}
            prefix = torch.randn(1, 383, 16); pair = (torch.randn(1, 126, 16), torch.randn(1, 126, 16))
            model = types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=lm),
                generate_prefix_inference=lambda inputs: (prefix, {'explicit': True}, pair))
            trace = observers.install_trace_audit(lm, 'read2_zero48')
            audit = {'intervention': registry.intervention('read2_zero48')}
            runtime.install_zero48(model, audit)
            p, _, zero = runtime.generate_prefix_inference(model, {}, eval_variant='read2_zero48', audit=audit)
            self.assertIs(p, prefix); self.assertTrue(all(torch.count_nonzero(x) == 0 for x in zero))
            context = {'audio_information': zero}
            with torch.inference_mode(), patch.object(runtime, 'HIDDEN_SIZE', 16):
                runtime.validate_context('read2_zero48', context, prefix)
                logits = lm(inputs_embeds=prefix, use_cache=False, **context).logits
                lm(inputs_embeds=torch.cat((prefix, prefix[:, -1:]), dim=1), use_cache=False, **context)
                with self.assertRaises(RuntimeError):
                    lm(inputs_embeds=prefix, use_cache=False, audio_information=pair)
            self.assertEqual(trace['checked_forwards'], 2)
            self.assertEqual(trace['loop_lora_calls_per_loop'], [98, 98])
            self.assertEqual(trace['audio_memory_slots'], 7)
            for k, v in before.items(): self.assertTrue(torch.equal(v, lm.state_dict()[k]), k)
            # Fresh sample must replace the old zero pair; cross-sample contexts are rejected.
            runtime.generate_prefix_inference(model, {}, eval_variant='read2_zero48', audit=audit)
            with self.assertRaises(RuntimeError), torch.inference_mode():
                lm(inputs_embeds=prefix, use_cache=False, audio_information=zero)
        finally:
            for k in list(sys.modules):
                if k == 'models' or k.startswith('models.'): del sys.modules[k]
            sys.modules.update(oldmodules); sys.path[:] = oldpath


if __name__ == '__main__':
    unittest.main()
