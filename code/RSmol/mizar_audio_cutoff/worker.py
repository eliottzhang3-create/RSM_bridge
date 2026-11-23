"""One or eight ranks finish each condition before advancing to the next."""
import json
import time
import traceback
from datetime import timedelta
from mizar_layer_pair_search import device, storage
from . import arguments, config, identity, results


def evaluate_condition(args, loaded, prefixes, records, expected, cutoff, rank, dist):
    from . import runtime
    started, world = time.monotonic(), args.world_size
    control = [(config.condition_name(cutoff), cutoff, expected['identity_sha256']) if rank == 0 else None]
    dist.broadcast_object_list(control, src=0)
    if tuple(control[0]) != (config.condition_name(cutoff), cutoff, expected['identity_sha256']):
        raise ValueError('ranks disagree on condition')
    loaded['cutoff_controller'].set_cutoff(cutoff)
    rows = results.read_rank(args.output_dir, rank, records, cutoff, expected['identity_sha256'], world)
    path = args.output_dir/'ranks'/f'rank{rank:02d}'/(config.condition_name(cutoff)+'.jsonl')
    for index in config.indices(rank, world)[len(rows):]:
        record = records[index]
        value = {'identity_sha256': expected['identity_sha256'], 'condition': config.condition_name(cutoff),
            'cutoff_layer_1based': cutoff, 'global_index': index, 'key': record['key'],
            'record_sha256': storage.object_sha(record), **runtime.score(loaded, prefixes[index], record)}
        sealed = storage.seal(value)
        results.validate_row(sealed, record, cutoff, expected['identity_sha256'])
        storage.append_jsonl(path, value)
        rows.append(sealed)
    gathered = [None]*world if rank == 0 else None
    dist.gather_object(rows, gathered, dst=0)
    if rank == 0:
        combined = results.merge(gathered, records, cutoff, expected['identity_sha256'], world)
        if cutoff is None:
            baseline = combined
        else:
            parts = [results.read_rank(args.output_dir, r, records, None, expected['identity_sha256'], world)
                     for r in range(world)]
            baseline = results.merge(parts, records, None, expected['identity_sha256'], world)
        summary = results.save_condition(args.output_dir, records, cutoff, combined, baseline,
                                         expected['identity_sha256'], time.monotonic()-started)
        print(json.dumps({'event': 'BASELINE_DONE' if cutoff is None else 'CUTOFF_DONE',
            'completed': config.CONDITIONS.index(cutoff)+1, 'total_conditions': 19,
            **summary}, ensure_ascii=False, allow_nan=False), flush=True)
    dist.barrier()


def main(argv=None):
    args = arguments.parse_args(argv)
    binding = device.bind_worker(args.world_size)
    expected = storage.read_json(args.output_dir/'run_config.json')
    rank, initialized = binding['rank'], False
    rank_dir = args.output_dir/'ranks'/f'rank{rank:02d}'
    try:
        frozen = identity.verify(args, expected)
        import torch.distributed as dist
        from . import cache, runtime
        records = frozen['records']
        loaded = runtime.load(args, expected['official_bundle'], binding)
        storage.write_json(rank_dir/'model_audit.json', storage.seal(loaded['audit']))
        dist.init_process_group('gloo', timeout=timedelta(hours=1))
        initialized = True
        audits = [None]*args.world_size
        dist.all_gather_object(audits, loaded['audit'])
        if (len({a['binding']['worker_cuda_visible_devices'] for a in audits}) != args.world_size
                or any(cache.precision_identity(a) != cache.precision_identity(audits[0]) for a in audits)):
            raise ValueError('worker device/precision/backend mismatch')
        if rank == 0:
            cache.prepare(loaded, records, args.output_dir/'cache', expected)
        dist.barrier()
        prefixes, cache_meta = cache.load_all(loaded, records, args.output_dir/'cache', expected)
        # Run native/no-mask equality and downstream isolation on every GPU.
        check = runtime.runtime_checks(loaded, prefixes[0], records[0])
        check.update(identity_sha256=expected['identity_sha256'], rank=rank)
        storage.write_json(rank_dir/'runtime_check.json', storage.seal(check))
        checks = [None]*args.world_size
        dist.all_gather_object(checks, check)
        if rank == 0:
            storage.write_json(args.output_dir/'runtime_checks.json', storage.seal({
                'status': 'PASS', 'identity_sha256': expected['identity_sha256'], 'world_size': args.world_size,
                'rank_audits': audits, 'rank_checks': checks, 'cache_manifest_sha256': cache_meta['sha256']}))
        dist.barrier()
        for cutoff in config.CONDITIONS:
            evaluate_condition(args, loaded, prefixes, records, expected, cutoff, rank, dist)
        identity.verify(args, expected)
        storage.write_json(rank_dir/'rank_report.json', storage.seal({'status': 'PASS',
            'identity_sha256': expected['identity_sha256'], 'binding': binding, 'conditions_completed': 19,
            'rows_per_condition': len(config.indices(rank, args.world_size)),
            'execution': loaded['cutoff_controller'].evidence()}))
        dist.barrier()
        return 0
    except Exception as exc:
        storage.write_json(rank_dir/'failure.json', {'status': 'FAILED', 'error': repr(exc),
                                                     'traceback': traceback.format_exc()})
        raise
    finally:
        if initialized:
            __import__('torch.distributed', fromlist=['destroy_process_group']).destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
