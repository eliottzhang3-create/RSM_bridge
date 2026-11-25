"""Strict primary decisions plus unchanged official MMAR diagnostic matcher.

string_match copied from MMAR/code/evaluation.py; whitespace normalized. Its AST
is checked against the provided official evaluation script before use.
"""
import ast
import math
import re
from pathlib import Path

def string_match(answer, prediction, choices):
    # Function to normalize and tokenize text
    def tokenize(text):
        # Convert to lowercase and find all word tokens
        return set(re.findall(r'\b\w+\b', text.lower()))

    # Tokenize prediction and answer
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)

    if not prediction_tokens:
        return False

    # Tokenize incorrect choices and exclude tokens present in the answer
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    # Condition 1: All tokens of the answer are in the prediction
    cond1 = answer_tokens.issubset(prediction_tokens)

    # Condition 2: Prediction does not contain any tokens from incorrect choices (excluding shared words)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)

    return cond1 and cond2


def verify_reference(path):
    """Compare function AST, without importing or executing the external script."""
    def function_tree(source):
        matches = [node for node in ast.parse(source).body
                   if isinstance(node, ast.FunctionDef) and node.name == 'string_match']
        if len(matches) != 1:
            raise ValueError('official scorer must contain exactly one string_match function')
        return ast.dump(matches[0], include_attributes=False)
    if function_tree(Path(path).read_text(encoding='utf-8-sig')) != function_tree(Path(__file__).read_text(encoding='utf-8')):
        raise ValueError('official MMAR string_match differs from the audited reference')


def decision(record, scores):
    """No empty score is zero or -inf in JSON; null slots are ineligible."""
    valid = record['choice_valid_mask']
    if len(scores) != len(valid) or len(record['choice_token_ids']) != len(valid):
        raise ValueError('option slot count mismatch')
    candidates = []
    for index, (enabled, score) in enumerate(zip(valid, scores)):
        if type(enabled) is not bool:
            raise ValueError('choice validity must be boolean')
        if enabled:
            if (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
                    or not record['choice_token_ids'][index]):
                raise ValueError('valid choice must have tokens and a finite likelihood')
            candidates.append(index)
        elif score is not None or record['choice_token_ids'][index]:
            raise ValueError('empty option must retain null likelihood and zero tokens')
    if len(candidates) < 2:
        raise ValueError('need at least two nonempty options')
    prediction = max(candidates, key=lambda index: scores[index])
    counts = [len(ids) for ids in record['choice_token_ids']]
    return {
        'prediction_index': prediction, 'prediction_text': record['choice_bodies'][prediction],
        'correct': record['gold_index'] is not None and prediction == record['gold_index'],
        'gold_mapped': record['gold_index'] is not None,
        'official_correct': record['official_correct_by_slot'][prediction],
        'option_loglikelihood': scores, 'option_token_counts': counts,
        'option_mean_loglikelihood': [score/n if n else None for score, n in zip(scores, counts)],
        'choice_valid_mask': valid,
        'option_score_tie': sum(scores[i] == scores[prediction] for i in candidates) > 1,
    }
