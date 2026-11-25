"""Strict ordered stopping and immutable accepted manifest, no replacement pool."""
import copy
import json
from collections import Counter
from mizar_layer_pair_search import storage
from . import config, runtime


def outcome_path(directory, index):
    return directory/'screening'/f'source_{index:04d}.json'


def qualifies(clean, corrupted):
    return (clean['correct'] and corrupted is not None and not corrupted['correct']
            and clean['target_probability'] > corrupted['target_probability'])


def reason(clean, corrupted):
    if not clean['correct']:
        return 'clean_wrong'
    if corrupted['correct']:
        return 'silence_correct'
    if clean['target_probability'] <= corrupted['target_probability']:
        return 'nonpositive_probability_gap'
    return 'accepted'


def validate_outcome(value, record, expected):
    storage.verify_seal(value)
    if (value['identity_sha256'] != expected['identity_sha256'] or value['key'] != record['key']
            or value['record_sha256'] != storage.object_sha(record)
            or value['source_manifest_index'] != record['source_manifest_index']):
        raise ValueError('screening outcome belongs to another question/run')
    runtime.validate_decision(record, value['clean'])
    if value['clean']['correct']:
        runtime.validate_decision(record, value['corrupted'])
    elif value['corrupted'] is not None:
        raise ValueError('clean-wrong candidate should not have a corrupted forward')
    accepted = qualifies(value['clean'], value['corrupted'])
    gap = None if value['corrupted'] is None else value['clean']['target_probability']-value['corrupted']['target_probability']
    if (value['accepted'] is not accepted or value['reason'] != reason(value['clean'], value['corrupted'])
            or value['probability_gap'] != gap or bool(value['cache_meta_sha256']) != accepted
            or not value['audio_audit'].get('audio_source_sha256')):
        raise ValueError('incorrect screening qualification/cache/audio evidence')
    return value


def next_window(history, order, target=None, world=None):
    target = config.TARGET if target is None else target
    world = config.WORLD if world is None else world
    accepted = sum(r['accepted'] for r in history)
    if accepted > target:
        raise ValueError('selection exceeded target')
    width = min(world, target-accepted, len(order)-len(history))
    return order[len(history):len(history)+width]


def recover(directory, pool, expected):
    history = storage.recover_jsonl(directory/'screening/journal.jsonl')
    accepted = 0
    for pos, row in enumerate(history):
        if pos >= len(pool['shuffle_order']):
            raise ValueError('journal exceeds fixed5000')
        record = pool['records'][pool['shuffle_order'][pos]]
        validate_outcome(row, record, expected)
        if (row['shuffle_position'] != pos or row['acceptance_order'] != (accepted if row['accepted'] else None)
                or accepted >= config.TARGET):
            raise ValueError('screening journal reordered or forwarded past stopping point')
        accepted += int(row['accepted'])
    return history


def commit(directory, history, outcomes, pool, expected):
    window = next_window(history, pool['shuffle_order'])
    if len(outcomes) != len(window):
        raise ValueError('incomplete screening window')
    accepted = sum(r['accepted'] for r in history)
    for index, outcome in zip(window, outcomes):
        record = pool['records'][index]
        validate_outcome(outcome, record, expected)
        row = {k: v for k, v in outcome.items() if k != 'sha256'}
        row.update(shuffle_position=len(history), acceptance_order=accepted if row['accepted'] else None)
        storage.append_jsonl(directory/'screening/journal.jsonl', row)
        history.append(storage.seal(row))
        accepted += int(row['accepted'])
        print(json.dumps({'event': 'SCREENED', 'scanned': len(history), 'accepted': accepted, 'target': config.TARGET,
            'source_manifest_index': index, 'source_row_index': record['source_row_index'], 'key': record['key'],
            'reason': row['reason'], 'gold_label': record['gold_label'],
            'clean_prediction': row['clean']['prediction_label'],
            'corrupted_prediction': None if row['corrupted'] is None else row['corrupted']['prediction_label'],
            'Pclean': row['clean']['target_probability'],
            'Pcorrupted': None if row['corrupted'] is None else row['corrupted']['target_probability']}, ensure_ascii=False), flush=True)
    return history


def finish(directory, history, pool, expected):
    selected = [row for row in history if row['accepted']]
    audit = {'status': 'SELECTED' if len(selected) == config.TARGET else 'INSUFFICIENT_QUALIFYING',
        'identity_sha256': expected['identity_sha256'], 'pool_size': len(pool['records']),
        'scanned': len(history), 'accepted': len(selected), 'required': config.TARGET,
        'reasons': dict(Counter(r['reason'] for r in history)),
        'clean_correct': sum(r['clean']['correct'] for r in history),
        'corrupted_forwards': sum(r['corrupted'] is not None for r in history),
        'journal_sha256': storage.object_sha(history), 'shuffle_sha256': pool['shuffle_sha256'],
        'note': 'conditional subset; clean100%/silence0% are selection criteria, not full5000 accuracy'}
    storage.write_json(directory/'selection_report.json', storage.seal(audit))
    if len(selected) != config.TARGET:
        return None
    sorted_rows = sorted(selected, key=lambda r: pool['records'][r['source_manifest_index']]['source_row_index'])
    records = []
    for index, row in enumerate(sorted_rows):
        r = copy.deepcopy(pool['records'][row['source_manifest_index']])
        r.update(selected_index=index, shuffle_position=row['shuffle_position'], acceptance_order=row['acceptance_order'],
                 screening={k: row[k] for k in ('clean', 'corrupted', 'probability_gap', 'cache_meta_sha256', 'audio_audit')})
        records.append(r)
    frozen = storage.seal({'schema': config.SCHEMA, 'identity_sha256': expected['identity_sha256'],
        'pool_sha256': pool['manifest_sha256'], 'shuffle_sha256': pool['shuffle_sha256'],
        'protocol': config.PROTOCOL, 'selection_audit': audit, 'records': records,
        'category_counts': dict(Counter(r['category'] for r in records))}, 'manifest_sha256')
    path = directory/'selected/manifest.json'
    if path.exists() and storage.read_json(path) != frozen:
        raise ValueError('existing selected1000 manifest differs')
    storage.write_json(path, frozen)
    storage.write_jsonl(directory/'selected/selected_records.jsonl', records)
    storage.write_json(directory/'selected/sampled_test.json', [r['original_record'] for r in records])
    return frozen


def load_selected(directory, pool, expected):
    frozen = storage.verify_seal(storage.read_json(directory/'selected/manifest.json'), 'manifest_sha256')
    history = recover(directory, pool, expected)
    # Independently reconstruct every accepted row and the exact stop point.
    recomputed = finish(directory, history, pool, expected)
    if frozen != recomputed:
        raise ValueError('selected manifest differs from ordered screening journal')
    return frozen
