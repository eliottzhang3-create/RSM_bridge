"""Probability-space RR, paired question records, conditional aggregation."""
import csv
import math
from mizar_layer_pair_search import storage
from . import config, runtime, selection


def recovery(patched, clean, corrupted):
    # Python floats are IEEE FP64. Do not add epsilon or clamp the result.
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (patched, clean, corrupted)) or clean <= corrupted:
        raise ValueError('RR requires valid probabilities and a strictly positive denominator')
    rr = (patched-corrupted)/(clean-corrupted)
    if not math.isfinite(rr):
        raise ValueError('nonfinite RR')
    return rr


def make_row(record, expected, frozen, layer, patched):
    s = record['screening']
    return {'identity_sha256': expected['identity_sha256'], 'manifest_sha256': frozen['manifest_sha256'],
        'selected_index': record['selected_index'], 'source_manifest_index': record['source_manifest_index'],
        'source_row_index': record['source_row_index'], 'key': record['key'], 'category': record['category'],
        'record_sha256': storage.object_sha(record), 'layer_1based': layer, 'gold_label': record['gold_label'],
        'patched': patched, 'Pclean': s['clean']['target_probability'], 'Pcorrupted': s['corrupted']['target_probability'],
        'Ppatched': patched['target_probability'], 'probability_gap': s['probability_gap'],
        'RR': recovery(patched['target_probability'], s['clean']['target_probability'], s['corrupted']['target_probability']),
        'small_gap': s['probability_gap'] < config.PROTOCOL['small_gap_threshold'],
        'executed_layers_1based': list(range(1, 31)), 'patched_text_slice_0based': [127, 383]}


def validate(value, record, expected, frozen, layer):
    storage.verify_seal(value)
    runtime.validate_decision(record, value['patched'])
    if not selection.qualifies(record['screening']['clean'], record['screening']['corrupted']):
        raise ValueError('nonqualifying question reached patch stage')
    recomputed = storage.seal(make_row(record, expected, frozen, layer, value['patched']))
    if value != recomputed or not 1 <= layer <= 30:
        raise ValueError('patch result identity/probability/RR/route mismatch')
    if layer == 30:
        # Exact logits/logprobs are the positive control; derived FP64 exp may
        # differ in its final ULP across nodes. Keep the actual RR unclipped.
        runtime.assert_same_decision(record,value['patched'],record['screening']['clean'])
    return value


def rank_path(directory, rank, layer):
    return directory/'ranks'/f'rank{rank:02d}'/f'patch_{layer:02d}.jsonl'


def read_rank(directory, rank, frozen, expected, layer):
    rows = storage.recover_jsonl(rank_path(directory, rank, layer))
    indices = config.indices(rank)
    if len(rows) > len(indices):
        raise ValueError('duplicate patch rows')
    for row, index in zip(rows, indices):
        validate(row, frozen['records'][index], expected, frozen, layer)
    return rows


def merge(parts, frozen, expected, layer):
    if len(parts) != 8 or any(len(p) != len(config.indices(rank)) for rank, p in enumerate(parts)):
        raise ValueError('incomplete eight-rank patch layer')
    rows = sorted([r for part in parts for r in part], key=lambda r: r['selected_index'])
    if [r['selected_index'] for r in rows] != list(range(config.TARGET)):
        raise ValueError('patch coverage is not exactly the selected500')
    for row, record in zip(rows, frozen['records']):
        validate(row, record, expected, frozen, layer)
    return rows


def stats(rows):
    import numpy as np
    if not rows:
        return {'total': 0, 'mean_RR': None, 'median_RR': None, 'mean_RR_CI95': None, 'patched_accuracy': None}
    rr = np.asarray([r['RR'] for r in rows], dtype=np.float64)
    rng = np.random.default_rng(config.SEED)
    boot = rr[rng.integers(0, len(rr), size=(2000, len(rr)))].mean(axis=1)
    gap = np.asarray([r['probability_gap'] for r in rows], dtype=np.float64)
    return {'total': len(rows), 'mean_RR': float(rr.mean()), 'median_RR': float(np.median(rr)),
        'mean_RR_CI95': [float(x) for x in np.quantile(boot, [.025, .975])],
        'RR_quantiles': {str(p): float(np.quantile(rr, p)) for p in (0, .05, .25, .5, .75, .95, 1)},
        'negative_RR': int((rr < 0).sum()), 'above_one_RR': int((rr > 1).sum()),
        'small_gap_rows': int((gap < config.PROTOCOL['small_gap_threshold']).sum()),
        'gap_quantiles': {str(p): float(np.quantile(gap, p)) for p in (0, .05, .5, .95, 1)},
        'sensitivity_mean_RR_gap_at_least': {str(t): {
            'total': int((gap >= t).sum()), 'mean_RR': float(rr[gap >= t].mean()) if (gap >= t).any() else None}
            for t in (1e-6, 1e-5, 1e-4)},
        'patched_correct': sum(r['patched']['correct'] for r in rows),
        'patched_accuracy': sum(r['patched']['correct'] for r in rows)/len(rows),
        'mean_Pclean': sum(r['Pclean'] for r in rows)/len(rows),
        'mean_Pcorrupted': sum(r['Pcorrupted'] for r in rows)/len(rows),
        'mean_Ppatched': sum(r['Ppatched'] for r in rows)/len(rows)}


def save_layer(directory, rows, frozen, expected, layer, elapsed):
    value = storage.seal({'identity_sha256': expected['identity_sha256'], 'manifest_sha256': frozen['manifest_sha256'],
        'layer_1based': layer, **stats(rows), 'groups': {c: stats([r for r in rows if r['category'] == c]) for c in config.CATEGORIES},
        'raw_rows_sha256': storage.object_sha(rows), 'elapsed_seconds': elapsed})
    folder = directory/'layers'/f'patch_{layer:02d}'
    storage.write_jsonl(folder/'per_question.jsonl', rows)
    storage.write_json(folder/'summary.json', value)
    return value


def write_tables(directory, summaries):
    storage.write_json(directory/'layer_scores.json', summaries)
    fields = ('layer_1based', 'group', 'total', 'mean_RR', 'median_RR', 'CI95_low', 'CI95_high',
              'patched_accuracy', 'mean_Pclean', 'mean_Pcorrupted', 'mean_Ppatched', 'small_gap_rows', 'negative_RR', 'above_one_RR')
    with (directory/'layer_scores.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for summary in summaries:
            for group, data in [('overall', summary), *summary['groups'].items()]:
                ci = data['mean_RR_CI95'] or [None, None]
                writer.writerow({'layer_1based': summary['layer_1based'], 'group': group,
                    **{k: data.get(k) for k in fields if k not in ('layer_1based', 'group', 'CI95_low', 'CI95_high')},
                    'CI95_low': ci[0], 'CI95_high': ci[1]})
