"""Exact 1000-row primary scores and separate official-matcher diagnostics."""
from collections import Counter
from mizar_layer_pair_search import storage
from . import config, scoring

CONDITIONS = config.CONDITIONS


def name(layer):
    if layer is not None and (type(layer) is not int or not 2 <= layer <= 29):
        raise ValueError('invalid original ablation layer; expected 2..29 or baseline')
    return 'baseline' if layer is None else f'ablate_{layer:02d}'


def validate(row, record, layer, expected):
    storage.verify_seal(row)
    if (row.get('identity_sha256') != expected['identity_sha256'] or row.get('condition') != name(layer)
            or row.get('global_index') != record['global_index'] or row.get('key') != record['key']
            or row.get('record_sha256') != storage.object_sha(record) or row.get('generation') is not None):
        raise ValueError('wrong MMAR ablation/input/identity')
    computed = scoring.decision(record, row.get('option_loglikelihood', []))
    for key, value in computed.items():
        if row.get(key) != value or (isinstance(value, bool) and type(row.get(key)) is not bool):
            raise ValueError(f'MMAR decision does not follow raw sums: {key}')
    trace = row.get('execution', {})
    if (trace.get('drop_layer_1based') != layer
            or trace.get('executed_original_layers_1based') != [i for i in range(1, 31) if i != layer]
            or trace.get('effective_layers') != (30 if layer is None else 29)
            or trace.get('checked_forwards', 0) < 1):
        raise ValueError('wrong actual original-layer route')


def read_rank(directory, rank, records, layer, expected):
    rows = storage.recover_jsonl(directory/'ranks'/f'rank{rank:02d}'/(name(layer)+'.jsonl'))
    assigned = config.indices(rank)
    if len(rows) > len(assigned):
        raise ValueError('extra MMAR journal rows')
    for row, index in zip(rows, assigned):
        validate(row, records[index], layer, expected)
    return rows


def merge(parts, records, layer, expected):
    if len(parts) != config.WORLD:
        raise ValueError('need all eight ranks')
    rows = []
    for rank, part in enumerate(parts):
        if [row['global_index'] for row in part] != config.indices(rank):
            raise ValueError('need 125 distinct assigned rows per GPU')
        for row in part:
            validate(row, records[row['global_index']], layer, expected)
        rows.extend(part)
    rows.sort(key=lambda row: row['global_index'])
    if [row['global_index'] for row in rows] != list(range(config.ROWS)):
        raise ValueError('not exact 1000-row denominator')
    return rows


def group_score(indices, rows, baseline, key='correct'):
    n = len(indices)
    correct = sum(rows[i][key] for i in indices)
    old = sum(baseline[i][key] for i in indices)
    return {
        'correct': correct, 'total': n, 'accuracy': correct/n,
        'baseline_correct': old, 'baseline_accuracy': old/n, 'delta_vs_baseline': (correct-old)/n,
        'newly_correct': sum(rows[i][key] and not baseline[i][key] for i in indices),
        'newly_wrong': sum(baseline[i][key] and not rows[i][key] for i in indices),
        'prediction_changed': sum(rows[i]['prediction_index'] != baseline[i]['prediction_index'] for i in indices),
    }


def summarize(rows, baseline, records, layer):
    if (len(rows) != config.ROWS or len(baseline) != config.ROWS or len(records) != config.ROWS
            or Counter(r['category'] for r in records) != Counter(config.CATEGORY_COUNTS)
            or [r['global_index'] for r in rows] != list(range(config.ROWS))
            or [r['global_index'] for r in baseline] != list(range(config.ROWS))):
        raise ValueError('incomplete MMAR condition/baseline or changed category denominator')
    groups, official_groups = {}, {}
    for field in ('category', 'sub_category', 'modality', 'language'):
        values = config.CATEGORIES if field == 'category' else sorted({r[field] for r in records})
        groups[field], official_groups[field] = {}, {}
        for value in values:
            indices = [i for i, r in enumerate(records) if r[field] == value]
            groups[field][value] = group_score(indices, rows, baseline)
            official_groups[field][value] = group_score(indices, rows, baseline, 'official_correct')
    improved = [i for i in range(config.ROWS) if rows[i]['correct'] and not baseline[i]['correct']]
    harmed = [i for i in range(config.ROWS) if baseline[i]['correct'] and not rows[i]['correct']]
    return {
        'condition': name(layer), 'drop_layer_1based': layer, 'drop_layer_0based': None if layer is None else layer-1,
        **group_score(list(range(config.ROWS)), rows, baseline),
        'newly_correct_indices': improved, 'newly_wrong_indices': harmed, 'groups': groups,
        'unmapped_gold_rows': sum(r['gold_index'] is None for r in records),
        'empty_choice_rows': sum(not all(r['choice_valid_mask']) for r in records),
        'option_score_ties': sum(row['option_score_tie'] for row in rows),
        'official_diagnostic': {**group_score(list(range(config.ROWS)), rows, baseline, 'official_correct'),
                                'groups': official_groups, 'primary_metric': False},
        'raw_rows_sha256': storage.object_sha(rows), 'baseline_rows_sha256': storage.object_sha(baseline),
    }


def save_condition(directory, rows, baseline, records, layer, expected, elapsed):
    summary = {**summarize(rows, baseline, records, layer),
               'identity_sha256': expected['identity_sha256'], 'elapsed_seconds': elapsed}
    folder = directory/'conditions'/name(layer)
    storage.write_jsonl(folder/'likelihood.jsonl', rows)
    storage.write_json(folder/'summary.json', storage.seal(summary))
    changes = []
    for i in summary['newly_correct_indices']+summary['newly_wrong_indices']:
        record = records[i]
        changes.append({
            'index': i, 'key': record['key'], 'category': record['category'], 'prompt': record['prompt'],
            'choice_bodies': record['choice_bodies'], 'gold_index': record['gold_index'],
            'gold_original': record['gold_original'], 'baseline_prediction': baseline[i]['prediction_index'],
            'prediction': rows[i]['prediction_index'], 'baseline_scores': baseline[i]['option_loglikelihood'],
            'scores': rows[i]['option_loglikelihood'], 'change': 'newly_correct' if rows[i]['correct'] else 'newly_wrong',
        })
    storage.write_jsonl(folder/'paired_changes.jsonl', changes)
    return summary
