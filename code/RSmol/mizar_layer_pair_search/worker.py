"""One synchronized mask at a time, with independent 13/12-row GPU shards."""
import json
import time
import traceback
from datetime import timedelta

from . import arguments, config, device, identity, results, storage


def print_first(rows, records):
    for row in rows:
        record = records[row['global_index']]
        print(json.dumps({'event': 'FIRST_MASK_SAMPLE', 'index': row['global_index'], 'key': record['key'],
            'dataset': record['dataset'], 'id': record['id'], 'rank': row['global_index'] % 8,
            'drop_pair_0based': row['drop_pair_0based'], 'drop_pair_1based': [i+1 for i in row['drop_pair_0based']],
            'prompt': record['prompt'], 'choice_bodies': record['choice_bodies'],
            'generated_output': row['generation']['generated_output'],
            'generated_text_raw': row['generation']['generated_text_raw'],
            'generated_token_ids': row['generation']['generated_token_ids'],
            'generated_token_count': row['generation']['generated_token_count'],
            'stop_reason': row['generation']['stop_reason'], 'gold_index': record['gold_index'],
            'gold_text': record['choice_bodies'][record['gold_index']], 'gold_original': record['gold_original'],
            'likelihood_prediction': record['choice_bodies'][row['prediction_index']],
            'likelihood_prediction_index': row['prediction_index'],
            'option_loglikelihood': row['option_loglikelihood'], 'likelihood_correct': row['correct']},
            ensure_ascii=False, allow_nan=False), flush=True)


def evaluate_mask(args, loaded, prefixes, records, expected, index, rank, dist):
    from . import runtime
    started = time.monotonic()
    pair = config.PAIRS[index] if index < len(config.PAIRS) else ()
    control = [(index, list(pair), expected['identity_sha256']) if rank == 0 else None]
    dist.broadcast_object_list(control, src=0)
    if tuple(control[0]) != (index, list(pair), expected['identity_sha256']):
        raise ValueError('ranks disagree on active mask or run identity')
    loaded['controller'].set_pair(pair)
    rows = results.read_rank(args.output_dir, rank, records, index, expected['identity_sha256'])
    path = args.output_dir / 'ranks' / f'rank{rank:02d}' / (config.mask_name(index)+'.jsonl')
    for global_index in config.rank_indices(rank)[len(rows):]:
        record, prefix = records[global_index], prefixes[global_index]
        # Free generation has no candidate or reference continuation as input.
        generation = runtime.generate(loaded, prefix) if index == 0 else None
        scored = runtime.score(loaded, prefix, record)
        value = {'identity_sha256': expected['identity_sha256'], 'mask_index': index,
                 'drop_pair_0based': list(pair), 'global_index': global_index, 'key': record['key'],
                 'record_sha256': storage.object_sha(record), 'generation': generation,
                 'executed_original_layers': list(loaded['controller'].trace), **scored}
        results.validate_row(storage.seal(value), record, index, expected['identity_sha256'])
        storage.append_jsonl(path, value)
        rows.append(storage.seal(value))
        if index == 0:
            print(json.dumps({'event': 'FIRST_MASK_WORKER_PROGRESS', 'rank': rank,
                              'global_index': global_index, 'finished': len(rows),
                              'total': len(config.rank_indices(rank))}), flush=True)
    gathered = [None]*8 if rank == 0 else None
    dist.gather_object(rows, gathered, dst=0)
    if rank == 0:
        combined = results.merge_rank_rows(gathered, records, index, expected['identity_sha256'])
        if index == 0:
            print_first(combined, records)
        summary = results.save_mask(args.output_dir, records, index, combined,
                                    expected['identity_sha256'], time.monotonic()-started)
        print(json.dumps({'event': 'MASK_DONE' if index < 378 else 'BASELINE_DONE',
                          'completed': index+1 if index < 378 else 378, 'candidate_masks': 378,
                          **summary}, ensure_ascii=False, allow_nan=False), flush=True)
    # Includes durable merge and printed score: nobody switches mask early.
    dist.barrier()


def main(argv=None):
    binding = device.bind_worker()
    args = arguments.parse_args(argv)
    expected = storage.read_json(args.output_dir / 'run_config.json')
    frozen = identity.verify(args, expected)
    records = frozen['records']
    import torch.distributed as dist
    from . import cache, runtime
    initialized = False
    rank = binding['rank']
    rank_dir = args.output_dir / 'ranks' / f'rank{rank:02d}'
    try:
        loaded = runtime.load(args, expected['official_bundle'], binding)
        storage.write_json(rank_dir / 'model_audit.json', storage.seal(loaded['audit']))
        dist.init_process_group('gloo', timeout=timedelta(hours=1))
        initialized = True
        audits = [None]*8
        dist.all_gather_object(audits, loaded['audit'])
        if (len({a['binding']['worker_cuda_visible_devices'] for a in audits}) != 8
                or any(cache.precision_identity(a) != cache.precision_identity(audits[0]) for a in audits)):
            raise ValueError('GPU bindings or runtime precision/backend differ across ranks')
        if rank == 0:
            cache.prepare(loaded, records, args.output_dir / 'cache', expected)
        dist.barrier()
        prefixes, cache_meta = cache.load_all(loaded, records, args.output_dir / 'cache', expected)
        if rank == 0:
            check = runtime.check_physical_equivalence(loaded, prefixes[0], records[0])
            storage.write_json(args.output_dir / 'runtime_checks.json', storage.seal({
                'identity_sha256': expected['identity_sha256'], 'status': 'PASS',
                'cache_manifest_sha256': cache_meta['sha256'], 'physical_pruning_equivalence': check,
                'online_cache_check': cache_meta['online_serialized_check'], 'rank_audits': audits}))
        dist.barrier()
        for index in range(len(config.PAIRS)+1):
            evaluate_mask(args, loaded, prefixes, records, expected, index, rank, dist)
        identity.verify(args, expected)
        storage.write_json(rank_dir / 'rank_report.json', storage.seal({
            'status': 'PASS', 'identity_sha256': expected['identity_sha256'], 'binding': binding,
            'masks_completed': 379, 'rows_per_mask': len(config.rank_indices(rank)),
            'execution_audit': loaded['controller'].report()}))
        dist.barrier()
        return 0
    except Exception as exc:
        storage.write_json(rank_dir / 'failure.json', {'status': 'FAILED', 'error': repr(exc),
                                                     'traceback': traceback.format_exc()})
        raise
    finally:
        if initialized:
            dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
