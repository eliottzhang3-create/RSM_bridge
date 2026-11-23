"""Use the exact pinned official MMAR string_match, then aggregate original questions."""
import ast
from collections import defaultdict
import hashlib
import re
from pathlib import Path

from .registry import DIMENSIONS, REQUIRED_CORRECT
from .storage import write_json, write_jsonl, object_sha

SCORER_HASHES = {
    'a3c57b829e40e67e3f7f0bbe7d54a112ea06ed7a781534241ecc5d3751486198',
    'fb046e16b9c0d2482b5b0698bc242e75a34d24a8165ba2a8f344572e56ee7918',
}


def load_scorer(path):
    raw = Path(path).read_bytes().replace(b'\r\n', b'\n')
    digest = hashlib.sha256(raw).hexdigest()
    if digest not in SCORER_HASHES:
        raise ValueError('MMAR official evaluation.py content is not a pinned supported release')
    tree = ast.parse(raw.decode('utf-8'), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'string_match']
    if len(functions) != 1:
        raise ValueError('expected one official string_match function')
    scope = {'re': re}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)
    return scope['string_match'], {'name': 'official_mmar_string_match', 'normalized_lf_sha256': digest,
                                  'raw_generation_unchanged': True, 'letter_only_parser': False}


def question_pass(option_count, correct_count, total_count):
    import math
    if (option_count not in REQUIRED_CORRECT or total_count != math.factorial(option_count)
            or type(correct_count) is not int or not 0 <= correct_count <= total_count):
        raise ValueError('invalid or incomplete permutation coverage')
    return correct_count >= REQUIRED_CORRECT[option_count]


def _metric(correct, total):
    return {'correct': correct, 'total': total, 'accuracy': correct / total if total else None,
            'accuracy_percent': 100 * correct / total if total else None}


def summarize(records, state, matcher):
    if (len(records) != 748 or len(state) != len(records)
            or len({r['id'] for r in records}) != len(records)):
        raise ValueError('cannot score a partial MyEval run')
    groups, predictions, opened = defaultdict(list), [], []
    for index, (record, saved) in enumerate(zip(records, state)):
        if saved['record'] != record or saved['row_index'] != index:
            raise ValueError('record order/content changed before scoring')
        text = saved['generation']['generated_text']
        if record['type'] == 'open-ended':
            opened.append({'item_id': record['item_id'], 'item_no': record['item_no'],
                           'question_number': record['question_number'], 'dim': record['dim'],
                           'question': record['question'], 'generation': saved['generation']})
            continue
        matched = bool(matcher(record['answer'], text, record['choices']))
        row = {'id': record['id'], 'question_key': record['question_key'],
               'permutation_index': record['permutation_index'], 'correct': matched,
               'answer': record['answer'], 'choices': record['choices'],
               'correct_display_index': record['correct_display_index'], 'generated_text': text}
        predictions.append(row); groups[record['question_key']].append((record, row))
    questions, collisions = [], []
    for key, rows in groups.items():
        sample = rows[0][0]
        expected = sample['permutation_count']
        if (len(rows) != expected or {r['permutation_index'] for r, _ in rows} != set(range(expected))
                or len({tuple(r['permutation']) for r, _ in rows}) != expected):
            raise ValueError('missing/duplicate question permutations')
        correct = sum(int(p['correct']) for _, p in rows)
        n = len(sample['choices'])
        item = {'question_key': key, 'item_id': sample['item_id'], 'item_no': sample['item_no'],
                'question_number': sample['question_number'], 'question': sample['question'],
                'dim': sample['dim'], 'difficulty': sample['difficulty'],
                'source_category': sample['source_category'], 'source_sub_category': sample['source_sub_category'],
                'correct_permutations': correct, 'total_permutations': expected,
                'required_correct': REQUIRED_CORRECT[n], 'passed': question_pass(n, correct, expected)}
        questions.append(item)
        answer_tokens = set(re.findall(r'\b\w+\b', sample['answer'].lower()))
        same = [c for c in sample['choices'] if c != sample['answer']
                and set(re.findall(r'\b\w+\b', c.lower())) == answer_tokens]
        if same:
            collisions.append({'question_key': key, 'answer': sample['answer'],
                               'incorrect_options_with_same_token_set': same})
    if len(questions) != 53 or len(predictions) != 734 or len(opened) != 14:
        raise ValueError('scoring denominator changed')
    metrics = {dim: _metric(sum(int(q['passed']) for q in questions if q['dim'] == dim),
                           sum(q['dim'] == dim for q in questions)) for dim in DIMENSIONS}
    total = _metric(sum(int(q['passed']) for q in questions), len(questions))
    return {'status': 'PASS', 'scoring_unit': 'original_closed_question', 'dimensions': metrics, 'total': total,
            'permutation_diagnostic': _metric(sum(int(p['correct']) for p in predictions), len(predictions)),
            'open_ended_count': len(opened), 'open_ended_scored': False,
            'official_token_set_collisions': collisions,
            'question_scores': questions, 'permutation_scores': predictions, 'open_answers': opened}


def publish(directory, result):
    import csv
    directory = Path(directory)
    write_json(directory / 'scores.json', {k: v for k, v in result.items()
                                           if k not in ('permutation_scores', 'open_answers')})
    write_jsonl(directory / 'permutation_scores.jsonl', result['permutation_scores'])
    write_jsonl(directory / 'open_answers.jsonl', result['open_answers'])
    with (directory / 'accuracy.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['dimension', 'correct', 'total', 'accuracy_percent'])
        for dim, metric in [*result['dimensions'].items(), ('total', result['total'])]:
            writer.writerow([dim, metric['correct'], metric['total'], metric['accuracy_percent']])
