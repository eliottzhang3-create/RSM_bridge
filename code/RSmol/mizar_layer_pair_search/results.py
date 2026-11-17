"""Recompute accuracy from raw likelihoods; never score generated text."""
import csv
import math

from . import config, storage


def choose(scores):
    if len(scores) < 2 or any(isinstance(s, bool) or not isinstance(s, (float, int)) or not math.isfinite(s) for s in scores):
        raise ValueError('need finite likelihoods for every valid option')
    return max(range(len(scores)), key=lambda i: scores[i])


def validate_row(row, record, mask_index, identity_sha):
    storage.verify_seal(row)
    pair = config.PAIRS[mask_index] if mask_index < len(config.PAIRS) else ()
    scores = row.get('option_loglikelihood', [])
    if (row.get('global_index') != record['global_index'] or row.get('key') != record['key']
            or row.get('record_sha256') != storage.object_sha(record)
            or row.get('identity_sha256') != identity_sha or row.get('mask_index') != mask_index
            or row.get('drop_pair_0based') != list(pair) or len(scores) != len(record['choice_bodies'])
            or row.get('executed_original_layers') != [i for i in range(30) if i not in pair]):
        raise ValueError('journal row has wrong input/mask/identity/executed layers')
    prediction = choose(scores)
    counts = [len(ids) for ids in record['choice_token_ids']]
    if (row.get('prediction_index') != prediction or row.get('correct') is not (prediction == record['gold_index'])
            or row.get('option_token_counts') != counts
            or row.get('option_mean_loglikelihood') != [s/n for s, n in zip(scores, counts)]):
        raise ValueError('saved prediction/accuracy/lengths cannot be reproduced from likelihoods')
    generation = row.get('generation')
    if mask_index == 0:
        if not isinstance(generation, dict):
            raise ValueError('first mask must have a real free generation for every record')
        ids = generation.get('generated_token_ids', [])
        stop = generation.get('stop_reason')
        if (not 1 <= len(ids) <= 300 or generation.get('generated_token_count') != len(ids)
                or any(type(i) is not int or not 0 <= i < 49152 for i in ids)
                or 0 in ids[:-1] or stop not in ('eos', 'length')
                or (stop == 'eos' and ids[-1] != 0) or (stop == 'length' and (len(ids) != 300 or ids[-1] == 0))
                or not isinstance(generation.get('generated_text_raw'), str)
                or not isinstance(generation.get('generated_output'), str)):
            raise ValueError('invalid first-mask free generation')
    elif generation is not None:
        raise ValueError('later masks/baseline must not generate answers')


def merge_rank_rows(rank_rows, records, mask_index, identity_sha):
    if len(rank_rows) != 8:
        raise ValueError('must gather all eight ranks')
    combined = []
    for rank, rows in enumerate(rank_rows):
        if [r['global_index'] for r in rows] != config.rank_indices(rank):
            raise ValueError(f'wrong 13/12 sample coverage on rank {rank}')
        for row in rows:
            validate_row(row, records[row['global_index']], mask_index, identity_sha)
        combined.extend(rows)
    combined.sort(key=lambda r: r['global_index'])
    if [r['global_index'] for r in combined] != list(range(100)):
        raise ValueError('mask lacks exact 100-row coverage')
    return combined


def summarize(rows, records, index):
    pair = config.PAIRS[index] if index < len(config.PAIRS) else ()
    correct = sum(row['correct'] for row in rows)
    subsets = {}
    for name in config.DATASETS:
        subset = [r for r in rows if records[r['global_index']]['dataset'] == name]
        if len(subset) != 25:
            raise ValueError('dataset denominator changed')
        count = sum(r['correct'] for r in subset)
        subsets[name] = {'correct': count, 'total': 25, 'accuracy': count/25}
    return {'mask_index': index, 'drop_pair_0based': list(pair), 'drop_pair_1based': [i+1 for i in pair],
            'drop_mask': config.drop_mask(pair), 'correct': correct, 'total': 100, 'accuracy': correct/100,
            'subsets': subsets, 'raw_rows_sha256': storage.object_sha(rows),
            'free_generations': sum(r.get('generation') is not None for r in rows)}


def read_rank(directory, rank, records, index, identity_sha):
    path = directory / 'ranks' / f'rank{rank:02d}' / (config.mask_name(index) + '.jsonl')
    rows = storage.recover_jsonl(path)
    indices = config.rank_indices(rank)
    if len(rows) > len(indices):
        raise ValueError('extra journal rows')
    for local, row in enumerate(rows):
        validate_row(row, records[indices[local]], index, identity_sha)
    return rows


def save_mask(directory, records, index, rows, identity_sha, elapsed):
    summary = summarize(rows, records, index)
    summary.update(identity_sha256=identity_sha, elapsed_seconds=elapsed)
    storage.write_jsonl(directory / 'masks' / config.mask_name(index) / 'likelihood.jsonl', rows)
    storage.write_json(directory / 'scores' / (config.mask_name(index)+'.json'), storage.seal(summary))
    if index == 0:
        storage.write_jsonl(directory / 'first_mask_generations.jsonl',
            (storage.seal({**{k: v for k, v in row.items() if k != 'sha256'},
              'prompt': records[row['global_index']]['prompt'],
              'choice_bodies': records[row['global_index']]['choice_bodies'],
              'gold_index': records[row['global_index']]['gold_index'],
              'gold_original': records[row['global_index']]['gold_original']}) for row in rows))
    return summary


def save_ranking(directory, summaries):
    if len(summaries) != len(config.PAIRS) or {s['mask_index'] for s in summaries} != set(range(378)):
        raise ValueError('ranking requires all 378 unique candidate masks')
    ordered = sorted(summaries, key=lambda s: (-s['correct'], s['mask_index']))
    last, rank = None, None
    for position, value in enumerate(ordered, 1):
        if value['correct'] != last:
            rank, last = position, value['correct']
        value['rank'] = rank
    storage.write_json(directory / 'ranking.json', ordered)
    path = directory / 'ranking.csv'
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        fields = ['rank', 'mask_index', 'drop_pair_0based', 'correct', 'total', 'accuracy', *config.DATASETS]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in ordered:
            writer.writerow({**{k: row[k] for k in fields if k not in config.DATASETS},
                             **{k: row['subsets'][k]['accuracy'] for k in config.DATASETS}})
        stream.flush()
        __import__('os').fsync(stream.fileno())
    __import__('os').replace(temporary, path)
    return ordered
