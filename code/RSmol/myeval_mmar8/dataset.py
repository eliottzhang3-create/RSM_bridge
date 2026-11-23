"""Read the user's annotations without using original MMAR questions in inference."""
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath

from .registry import ANNOTATIONS_SHA256, DIMENSIONS, REQUIRED_CORRECT
from mizar_s1_ced_reasonaqa5000_eval.storage import object_sha


def normalized_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def load_annotations(path):
    digest = normalized_sha(path)
    if digest != ANNOTATIONS_SHA256:
        raise ValueError('annotations differ from the approved eight-audio file; do not silently replace labels')
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    validate(rows)
    return rows


def validate(rows):
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError('expected eight annotated MMAR audio rows')
    ids, positions = set(), set()
    counts, options, opened = Counter(), Counter(), 0
    for row in rows:
        if (row.get('benchmark') != 'MMAR' or not isinstance(row.get('item_id'), str)
                or not row['item_id'] or row['item_id'] in ids
                or type(row.get('row_index')) is not int or row['row_index'] < 0
                or row['row_index'] in positions):
            raise ValueError('invalid/duplicate MMAR source identity')
        ids.add(row['item_id']); positions.add(row['row_index'])
        questions = row.get('questions')
        if not isinstance(questions, list) or not questions or row.get('question_count') != len(questions):
            raise ValueError('annotation question_count mismatch')
        numbers = set()
        for q in questions:
            if (type(q.get('n')) is not int or q['n'] <= 0 or q['n'] in numbers
                    or not isinstance(q.get('text'), str) or not q['text'].strip()):
                raise ValueError('invalid/duplicate question number or missing question')
            numbers.add(q['n'])
            if q.get('type') == 'open-ended':
                if q.get('dim') not in (*DIMENSIONS, 'C') or q.get('options'):
                    raise ValueError('invalid open-ended dimension/options')
                opened += 1
                continue
            if q.get('type') != 'close-ended' or q.get('dim') not in DIMENSIONS or q.get('valid') is not True:
                raise ValueError('invalid closed-ended type/dimension/valid flag')
            choices = q.get('options')
            if not isinstance(choices, list) or len(choices) not in REQUIRED_CORRECT:
                raise ValueError('supported option counts are 2,3,4,5')
            labels = [o.get('label') for o in choices]
            texts = [o.get('text') for o in choices]
            if (any(not isinstance(t, str) or not t.strip() for t in texts)
                    or len(set(t.strip().casefold() for t in texts)) != len(texts)
                    or labels != [chr(97 + i) for i in range(len(choices))]
                    or any(type(o.get('correct')) is not bool for o in choices)
                    or sum(o['correct'] for o in choices) != 1
                    or q.get('answer') != [o['text'] for o in choices if o['correct']]):
                raise ValueError('inconsistent options/correct/answer fields')
            counts[q['dim']] += 1; options[len(choices)] += 1
    if (counts != Counter(P=29, S=11, W=7, R=6)
            or options != Counter({2: 28, 3: 1, 4: 23, 5: 1}) or opened != 14):
        raise ValueError('approved MyEval inventory is 53 closed + 14 open questions')


def _relative_audio(value):
    path = PurePosixPath(str(value).replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('audio path must stay inside the original MMAR audio root')
    return path.as_posix()


def expand(rows, official, audio_root):
    """Fresh contexts; lexicographic permutations of option INDICES, not old labels."""
    validate(rows)
    if len(official) != 1000 or len({r.get('id') for r in official}) != 1000:
        raise ValueError('expected the original complete 1000-row MMAR metadata')
    root = Path(audio_root).resolve()
    records = []
    for row in rows:
        index = row['row_index']
        if index >= len(official):
            raise ValueError('MMAR source row index out of range')
        source = official[index]
        if (source.get('id') != row['item_id']
                or _relative_audio(source.get('audio_path')) != _relative_audio(row['audio']['path'])
                or source.get('category') != row['layer']
                or source.get('sub-category') != row['sub_category']
                or source.get('modality') != row['modality']):
            raise ValueError(f'annotation/audio/official metadata binding mismatch: {row["item_id"]}')
        path = (root / _relative_audio(source['audio_path'])).resolve()
        if not path.is_relative_to(root):
            raise ValueError('MMAR audio escaped its root')
        for q in row['questions']:
            closed = q['type'] == 'close-ended'
            original = [o['text'] for o in q['options']] if closed else []
            correct = next(i for i, o in enumerate(q['options']) if o['correct']) if closed else None
            permutations = itertools.permutations(range(len(original))) if closed else [()]
            key = f'{row["item_id"]}::q{q["n"]}'
            for permutation_index, permutation in enumerate(permutations):
                choices = [original[i] for i in permutation]
                # The same compact builder as the existing CED evaluator; no new instructions.
                from .benchmark import mmau
                prompt = mmau.build_fixed_order_prompt(q['text'], choices) if closed else q['text'].strip()
                record = {
                    'id': f'{key}::p{permutation_index:03d}' if closed else f'{key}::open',
                    'question_key': key, 'item_id': row['item_id'], 'item_no': row['item_no'],
                    'source_row_index': index, 'question_number': q['n'],
                    'type': q['type'], 'dim': q['dim'], 'difficulty': q.get('difficulty'),
                    'source_category': row['layer'], 'source_sub_category': row['sub_category'],
                    'modality': row['modality'], 'question': q['text'],
                    'choices': choices, 'permutation': list(permutation),
                    'permutation_index': permutation_index if closed else None,
                    'permutation_count': math.factorial(len(original)) if closed else 1,
                    'required_correct': REQUIRED_CORRECT[len(original)] if closed else None,
                    'answer': original[correct] if closed else None,
                    'correct_display_index': permutation.index(correct) if closed else None,
                    'audio_path_resolved': str(path), 'audio_timestamp': row['audio']['timestamp'],
                    'prompt': prompt.lower(),
                }
                record['record_sha256'] = object_sha(record)
                records.append(record)
    if len(records) != 748 or len({r['id'] for r in records}) != 748:
        raise ValueError('expected exactly 748 unique generation tasks')
    return records
