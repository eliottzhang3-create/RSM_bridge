"""Keep stored evidence exact; allow only a few FP64 exp rounding ULPs."""
import math
from mizar_layer_pair_search import storage
from mizar_reasonaqa_tracing import runtime as original_runtime, selection as original_selection

EXP_MAX_ULPS = 8


def probability_error(saved, expected, field):
    if (type(saved) is not float or not math.isfinite(saved) or not 0 <= saved <= 1
            or type(expected) is not float or not math.isfinite(expected) or not 0 <= expected <= 1):
        raise ValueError(f'{field}: expected a finite FP64 probability in [0,1]; saved={saved!r}, recomputed={expected!r}')
    delta = abs(saved-expected)
    # Exact endpoints: zero/one must not hide underflow or saturation changes.
    allowed = 0.0 if saved in (0.0,1.0) or expected in (0.0,1.0) else EXP_MAX_ULPS*max(math.ulp(saved),math.ulp(expected))
    if delta > allowed:
        raise ValueError(f'{field}: probability differs beyond {EXP_MAX_ULPS} FP64 ULPs; '
            f'saved={saved!r} ({saved.hex()}), recomputed={expected!r} ({expected.hex()}), '
            f'abs_error={delta!r}, allowed={allowed!r}')
    return delta


def compare_decisions(record, saved, expected):
    """Only derived exp values may round; classification/logprobs stay exact."""
    if not isinstance(saved,dict) or not isinstance(expected,dict) or set(saved) != set(expected):
        raise ValueError('decision fields differ from the original scoring schema')
    derived = {'candidate_probabilities','target_probability'}
    for key in expected.keys()-derived:
        if saved[key] != expected[key]:
            raise ValueError(f'{key}: saved={saved[key]!r}, recomputed={expected[key]!r}')
    # Avoid Python treating 0/1 as equivalent to bool predictions.
    if (type(saved['prediction_index']) is not int or type(saved['correct']) is not bool
            or type(saved['tie']) is not bool or type(saved['target_logprob']) is not float
            or not isinstance(saved['candidate_logprobs'],list)
            or any(type(p) is not float for p in saved['candidate_logprobs'])):
        raise ValueError('saved prediction/logprob types differ from the original scorer')
    actual = saved['candidate_probabilities']; reference = expected['candidate_probabilities']
    if not isinstance(actual,list) or len(actual) != len(reference):
        raise ValueError('candidate probability count differs')
    errors = [probability_error(p,q,f'candidate_probabilities[{i}]') for i,(p,q) in enumerate(zip(actual,reference))]
    errors.append(probability_error(saved['target_probability'],expected['target_probability'],'target_probability'))
    if saved['target_probability'] != actual[record['gold_index']]:
        raise ValueError('target_probability differs from its own saved gold candidate probability')
    return max(errors,default=0.0)


def validate_decision(record, value):
    try:
        expected = original_runtime.decision(record,value['candidate_logprobs'])
        return compare_decisions(record,value,expected)
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError(f'saved decision invalid for {record.get("key",record.get("id","?"))}: {exc}') from exc


def assert_same_decision(record, actual, saved):
    validate_decision(record,actual);validate_decision(record,saved)
    return compare_decisions(record,actual,saved)


def validate_outcome(value, record, expected):
    """Original qualification/identity/seal checks with narrow exp verification."""
    storage.verify_seal(value)
    if (value['identity_sha256'] != expected['identity_sha256'] or value['key'] != record['key']
            or value['record_sha256'] != storage.object_sha(record)
            or value['source_manifest_index'] != record['source_manifest_index']):
        raise ValueError('screening outcome belongs to another question/run')
    validate_decision(record,value['clean'])
    if value['clean']['correct']:
        validate_decision(record,value['corrupted'])
    elif value['corrupted'] is not None:
        raise ValueError('clean-wrong candidate should not have a corrupted forward')
    # Use original stored probabilities, not recalculated exp values, for both
    # membership and RR baselines. Source bytes and selections remain unchanged.
    accepted = original_selection.qualifies(value['clean'],value['corrupted'])
    gap = None if value['corrupted'] is None else value['clean']['target_probability']-value['corrupted']['target_probability']
    if (value['accepted'] is not accepted or value['reason'] != original_selection.reason(value['clean'],value['corrupted'])
            or value['probability_gap'] != gap or bool(value['cache_meta_sha256']) != accepted
            or not value['audio_audit'].get('audio_source_sha256')):
        raise ValueError('incorrect screening qualification/cache/audio evidence')
    return value
