"""One metric only: normalized full-answer exact match, including MCQ labels."""
from .dataset import CATEGORIES
from . import storage

EOS = '<|endoftext|>'
POLICY = {'version': 'full_answer_exact_match_v1',
          'normalization': ['strip_outer_whitespace', 'remove_one_terminal_eos',
                            'unicode_casefold', 'collapse_unicode_whitespace'],
          'preserve': ['choice_label', 'parentheses', 'answer_body', 'punctuation'],
          'empty_response': 'incorrect_in_full_selected_denominator'}


def normalize(text):
    if not isinstance(text, str):
        raise ValueError('answer must be text')
    text = text.strip()
    if text.endswith(EOS):
        text = text[:-len(EOS)]
    return ' '.join(text.casefold().split())


def build_results(saved, expected_rows):
    if len(saved) != expected_rows or expected_rows <= 0:
        raise ValueError('full selected coverage required before scoring')
    counts = {key: {'correct': 0, 'total': 0} for key in CATEGORIES}
    details = []
    for item in saved:
        record, generation = item['record'], item['generation']
        row = record['original_record']
        prediction, target = normalize(generation['generated_text']), normalize(row['answer'])
        correct = bool(prediction) and prediction == target
        group = counts[record['category']]
        group['total'] += 1
        group['correct'] += int(correct)
        details.append({'id': record['id'], 'source_row_index': record['source_row_index'],
                        'category': record['category'], 'input': row['input'],
                        'answer': row['answer'], 'generated_text_raw': generation['generated_text'],
                        'normalized_prediction': prediction, 'normalized_answer': target, 'correct': correct})
    for group in counts.values():
        group['accuracy'] = group['correct'] / group['total'] if group['total'] else None
    correct = sum(group['correct'] for group in counts.values())
    result = {'status': 'PASS', 'metric': 'normalized_full_answer_exact_match', 'policy': POLICY,
              'overall': {'correct': correct, 'total': expected_rows, 'accuracy': correct / expected_rows},
              'by_category': counts}
    return result, details


def score(directory, saved, expected_rows):
    result, details = build_results(saved, expected_rows)
    storage.write_jsonl(directory / 'per_sample_results.jsonl', details)
    storage.write_json(directory / 'accuracy.json', result)
    return result
