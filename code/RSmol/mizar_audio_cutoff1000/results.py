"""Recompute all decisions and paired changes from the saved likelihood sums."""
import csv
from mizar_layer_pair_search import storage
from mizar_layer_pair_search.results import choose
from . import config


def validate_row(row, record, cutoff, identity_sha):
    storage.verify_seal(row)
    config.condition_name(cutoff)
    scores, counts = row.get('option_loglikelihood', []), [len(ids) for ids in record['choice_token_ids']]
    if (row.get('identity_sha256') != identity_sha or row.get('condition') != config.condition_name(cutoff)
            or row.get('cutoff_layer_1based') != cutoff or row.get('global_index') != record['global_index']
            or row.get('key') != record['key'] or row.get('record_sha256') != storage.object_sha(record)
            or len(scores) != len(counts) or row.get('generation') is not None):
        raise ValueError('wrong condition/input/identity or unexpected generation')
    prediction = choose(scores)
    if (row.get('option_token_counts') != counts or row.get('prediction_index') != prediction
            or row.get('correct') is not (prediction == record['gold_index'])
            or row.get('option_mean_loglikelihood') != [s/n for s, n in zip(scores, counts)]
            or row.get('option_score_tie') is not (scores.count(scores[prediction]) > 1)):
        raise ValueError('saved decision cannot be reproduced from raw sums')
    trace = row.get('execution', {})
    if (trace.get('cutoff_layer_1based') != cutoff or trace.get('executed_layers_1based') != list(range(1, 31))
            or trace.get('masked_layers_1based') != config.masked_layers(cutoff)
            or trace.get('blocked_audio_plus_sep_tokens') != 127 or trace.get('checked_forwards', 0) < 1):
        raise ValueError('wrong actual layer/mask trace')


def read_rank(directory, rank, records, cutoff, identity_sha, world):
    rows = storage.recover_jsonl(directory/'ranks'/f'rank{rank:02d}'/(config.condition_name(cutoff)+'.jsonl'))
    indices = config.indices(rank, world)
    if len(rows) > len(indices):
        raise ValueError('extra journal rows')
    for row, index in zip(rows, indices):
        validate_row(row, records[index], cutoff, identity_sha)
    return rows


def merge(parts, records, cutoff, identity_sha, world):
    if len(parts) != world:
        raise ValueError('missing worker')
    combined = []
    for rank, rows in enumerate(parts):
        if [row['global_index'] for row in rows] != config.indices(rank, world):
            raise ValueError('wrong rank coverage or order')
        for row in rows:
            validate_row(row, records[row['global_index']], cutoff, identity_sha)
        combined.extend(rows)
    combined.sort(key=lambda row: row['global_index'])
    if [row['global_index'] for row in combined] != list(range(config.ROWS)):
        raise ValueError('must have exactly 1000 decisions')
    return combined


def summarize(rows, records, cutoff, baseline):
    if len(rows) != config.ROWS or len(baseline) != config.ROWS:
        raise ValueError('incomplete condition or baseline')
    subsets = {}
    for name in config.DATASETS:
        selected = [r for r in rows if records[r['global_index']]['dataset'] == name]
        if len(selected) != config.PER_DATASET:
            raise ValueError('changed subset denominator')
        n = sum(r['correct'] for r in selected)
        old = sum(r['correct'] for r in baseline if records[r['global_index']]['dataset'] == name)
        subsets[name] = {'correct': n, 'total': config.PER_DATASET, 'accuracy': n/config.PER_DATASET, 'delta_vs_baseline': (n-old)/config.PER_DATASET}
    improved = [r['global_index'] for r, b in zip(rows, baseline) if r['correct'] and not b['correct']]
    harmed = [r['global_index'] for r, b in zip(rows, baseline) if b['correct'] and not r['correct']]
    correct, old = sum(r['correct'] for r in rows), sum(r['correct'] for r in baseline)
    return {'condition': config.condition_name(cutoff), 'cutoff_layer_1based': cutoff,
            'audio_visible_layers': 30 if cutoff is None else cutoff-1, 'physical_layers': 30,
            'correct': correct, 'total': config.ROWS, 'accuracy': correct/config.ROWS, 'baseline_accuracy': old/config.ROWS,
            'delta_vs_baseline': (correct-old)/config.ROWS, 'subsets': subsets,
            'newly_correct_indices': improved, 'newly_wrong_indices': harmed,
            'newly_correct': len(improved), 'newly_wrong': len(harmed),
            'prediction_changed': sum(r['prediction_index'] != b['prediction_index'] for r, b in zip(rows, baseline)),
            'raw_rows_sha256': storage.object_sha(rows), 'baseline_rows_sha256': storage.object_sha(baseline)}


def save_condition(directory, records, cutoff, rows, baseline, identity_sha, elapsed):
    summary = {**summarize(rows, records, cutoff, baseline), 'identity_sha256': identity_sha,
               'elapsed_seconds': elapsed}
    folder = directory/'conditions'/config.condition_name(cutoff)
    storage.write_jsonl(folder/'likelihood.jsonl', rows)
    storage.write_json(folder/'summary.json', storage.seal(summary))
    changed = []
    for index in summary['newly_correct_indices']+summary['newly_wrong_indices']:
        record, row, base = records[index], rows[index], baseline[index]
        changed.append({'global_index': index, 'key': record['key'], 'dataset': record['dataset'],
            'prompt': record['prompt'], 'choice_bodies': record['choice_bodies'], 'gold_index': record['gold_index'],
            'baseline_prediction_index': base['prediction_index'], 'prediction_index': row['prediction_index'],
            'baseline_option_loglikelihood': base['option_loglikelihood'], 'option_loglikelihood': row['option_loglikelihood'],
            'change': 'newly_correct' if row['correct'] else 'newly_wrong'})
    storage.write_jsonl(folder/'paired_changes.jsonl', changed)
    return summary


def save_analysis(directory, summaries):
    if [s['cutoff_layer_1based'] for s in summaries] != list(config.CONDITIONS):
        raise ValueError('need all 27 conditions in baseline-first order')
    storage.write_json(directory/'condition_scores.json', summaries)
    storage.write_json(directory/'baseline.json', summaries[0])
    path = directory/'accuracy_by_cutoff.csv'
    temporary = path.with_suffix('.csv.tmp')
    fields = ['condition', 'cutoff_layer_1based', 'audio_visible_layers', 'correct', 'total', 'accuracy',
              'delta_vs_baseline', 'newly_correct', 'newly_wrong', *config.DATASETS]
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({**{k:row[k] for k in fields if k not in config.DATASETS},
                             **{k:row['subsets'][k]['accuracy'] for k in config.DATASETS}})
        stream.flush()
        __import__('os').fsync(stream.fileno())
    __import__('os').replace(temporary, path)
    # Rank by correct count, retain shared ranks; ordered layer curve remains separate.
    ordered = sorted(summaries[1:], key=lambda row: (-row['correct'], row['cutoff_layer_1based']))
    ranked, last, rank = [], None, None
    for position, row in enumerate(ordered, 1):
        if row['correct'] != last:
            rank, last = position, row['correct']
        ranked.append({**row, 'rank': rank})
    storage.write_json(directory/'ranking.json', ranked)
    return ranked
