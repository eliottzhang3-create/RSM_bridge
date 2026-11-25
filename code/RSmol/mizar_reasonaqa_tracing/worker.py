"""One eight-GPU process group: ordered screening, then common-layer patches."""
import json
import time
import traceback
from datetime import timedelta
from mizar_layer_pair_search import device, storage
from . import arguments, config, identity, selection


def screen_one(args, loaded, record, expected, silence):
    from mizar_layer_pair_search import audio
    from . import cache, runtime
    path = selection.outcome_path(args.output_dir, record['source_manifest_index'])
    if path.is_file():
        value = selection.validate_outcome(storage.read_json(path), record, expected)
        # Pending outcomes survive an interruption before coordinator commit.
        if value['accepted']:
            cache.load(args.output_dir, record, expected, digest=value['cache_meta_sha256'],
                       precision=cache.precision(loaded['audit']), tensors=False)
        if storage.file_sha(record['audio_ref']['path']) != value['audio_audit']['audio_source_sha256']:
            raise ValueError('audio content changed since pending screening forward')
        return value
    wave, audio_audit = audio.load_fixed20(record['audio_ref'])
    clean_prefix = runtime.prefix(loaded, record, wave)
    logits, states = runtime.forward(loaded, clean_prefix, capture=True)
    clean = runtime.from_logits(record, logits)
    corrupted, meta, corrupted_prefix = None, None, None
    if clean['correct']:
        corrupted_prefix = runtime.corrupt_prefix(clean_prefix, silence)
        logits, _ = runtime.forward(loaded, corrupted_prefix)
        corrupted = runtime.from_logits(record, logits)
    accepted = selection.qualifies(clean, corrupted)
    if accepted:
        meta = cache.save(args.output_dir, record, expected, clean_prefix, corrupted_prefix, states,
                          audio_audit, cache.precision(loaded['audit']))
    value = storage.seal({'identity_sha256': expected['identity_sha256'], 'key': record['key'],
        'source_manifest_index': record['source_manifest_index'], 'record_sha256': storage.object_sha(record),
        'clean': clean, 'corrupted': corrupted, 'accepted': accepted, 'reason': selection.reason(clean, corrupted),
        'probability_gap': None if corrupted is None else clean['target_probability']-corrupted['target_probability'],
        'cache_meta_sha256': None if meta is None else meta['sha256'], 'audio_audit': audio_audit})
    selection.validate_outcome(value, record, expected)
    storage.write_json(path, value)
    return value


def select(args, loaded, pool, expected, silence, rank, dist):
    history = selection.recover(args.output_dir, pool, expected) if rank == 0 else None
    while True:
        request = [selection.next_window(history, pool['shuffle_order']) if rank == 0 else None]
        dist.broadcast_object_list(request, src=0)
        window = request[0]
        if not window:
            break
        local = screen_one(args, loaded, pool['records'][window[rank]], expected, silence) if rank < len(window) else None
        gathered = [None]*config.WORLD if rank == 0 else None
        dist.gather_object(local, gathered, dst=0)
        if rank == 0:
            selection.commit(args.output_dir, history, gathered[:len(window)], pool, expected)
    if rank == 0:
        frozen = selection.finish(args.output_dir, history, pool, expected)
        status = {'success': frozen is not None, 'scanned': len(history), 'accepted': sum(x['accepted'] for x in history)}
    else:
        status = None
    control = [status]; dist.broadcast_object_list(control, src=0)
    if not control[0]['success']:
        raise RuntimeError(f'fixed5000 exhausted: {control[0]}; see selection_report.json; no patch stage started')
    dist.barrier()
    # Only rank0 reconstructs/writes; other ranks read the published sealed file.
    return storage.verify_seal(storage.read_json(args.output_dir/'selected/manifest.json'), 'manifest_sha256')


def evaluate_layer(args, loaded, tensors, frozen, expected, layer, rank, dist):
    from . import runtime, results
    records = frozen['records']; started = time.monotonic()
    control = [(layer, frozen['manifest_sha256']) if rank == 0 else None]
    dist.broadcast_object_list(control, src=0)
    if tuple(control[0]) != (layer, frozen['manifest_sha256']):
        raise ValueError('ranks disagree on patched layer')
    rows = results.read_rank(args.output_dir, rank, frozen, expected, layer)
    path = results.rank_path(args.output_dir, rank, layer)
    for index in config.indices(rank)[len(rows):]:
        r, data = records[index], tensors[index]
        logits, _ = runtime.forward(loaded, data['corrupted_prefix'], layer=layer, clean_states=data['clean_states'])
        value = results.make_row(r, expected, frozen, layer, runtime.from_logits(r, logits))
        results.validate(storage.seal(value), r, expected, frozen, layer)
        storage.append_jsonl(path, value); rows.append(storage.seal(value))
    parts = [None]*8 if rank == 0 else None
    dist.gather_object(rows, parts, dst=0)
    if rank == 0:
        merged = results.merge(parts, frozen, expected, layer)
        summary = results.save_layer(args.output_dir, merged, frozen, expected, layer, time.monotonic()-started)
        print(json.dumps({'event': 'PATCH_LAYER_DONE', **{k: summary[k] for k in
            ('layer_1based', 'total', 'mean_RR', 'median_RR', 'patched_accuracy', 'mean_Ppatched', 'groups', 'elapsed_seconds')}},
            ensure_ascii=False), flush=True)
    dist.barrier()


def main(argv=None):
    args = arguments.parse_args(argv)
    binding = device.bind_worker(config.WORLD)
    rank, initialized = binding['rank'], False
    folder = args.output_dir/'ranks'/f'rank{rank:02d}'
    try:
        expected = storage.read_json(args.output_dir/'run_config.json')
        pool = identity.verify(args, expected)
        import torch
        import torch.distributed as dist
        from . import cache, runtime
        loaded = runtime.load(args, expected['official_bundle'], binding)
        storage.write_json(folder/'model_audit.json', storage.seal(loaded['audit']))
        print(json.dumps({'event': 'TRACING_MODEL_READY', 'rank': rank,
            'gpu': loaded['audit']['gpu'], 'source_weights': 'float32', 'forward': 'bfloat16'}), flush=True)
        dist.init_process_group('gloo', timeout=timedelta(hours=6)); initialized = True
        audits = [None]*8; dist.all_gather_object(audits, loaded['audit'])
        if (len({a['binding']['worker_cuda_visible_devices'] for a in audits}) != 8
                or any(cache.precision(a) != cache.precision(audits[0]) for a in audits)):
            raise ValueError('duplicate GPU or precision/backend mismatch')
        # One genuine silent waveform encoding per rank/device, independent of text.
        silence = runtime.silent_audio(loaded, pool['records'][0])
        silence_sha = __import__('mizar_layer_pair_search.audio', fromlist=['tensor_sha']).tensor_sha(silence)
        silence_hashes = [None]*8; dist.all_gather_object(silence_hashes, silence_sha)
        if len(set(silence_hashes)) != 1:
            raise ValueError('encoded silence differs across GPUs')
        if rank == 0:
            print(json.dumps({'event': 'SILENCE_ENCODED', 'rank_count': 8, 'audio_tokens': 126,
                'silence_sha256': silence_sha, 'nonzero_elements': int(silence.count_nonzero().item())}), flush=True)
        frozen = select(args, loaded, pool, expected, silence, rank, dist)
        if args.stage == 'select':
            return 0
        tensors, inventory = {}, []
        for index in config.indices(rank):
            record = frozen['records'][index]
            original = pool['records'][record['source_manifest_index']]
            data, meta = cache.load(args.output_dir, original, expected,
                digest=record['screening']['cache_meta_sha256'], precision=cache.precision(loaded['audit']))
            for key in ('clean_prefix', 'corrupted_prefix'):
                data[key] = data[key].to(loaded['device'])
            if not torch.equal(data['corrupted_prefix'][:, :126], silence):
                raise ValueError('cached corruption differs from genuine encoded silence')
            tensors[index] = data
            inventory.append({'selected_index': index, 'meta_sha256': meta['sha256']})
            if len(inventory) % 25 == 0:
                print(json.dumps({'event': 'TRACING_CACHE_LOADED', 'rank': rank,
                    'loaded': len(inventory), 'total': len(config.indices(rank))}), flush=True)
        probe = config.indices(rank)[0]; r = frozen['records'][probe]; data = tensors[probe]
        check = runtime.controls(loaded, data['clean_prefix'], data['corrupted_prefix'], data['clean_states'], r,
                                 r['screening']['clean'], r['screening']['corrupted'])
        check.update(rank=rank, identity_sha256=expected['identity_sha256'], silence_sha256=silence_sha)
        storage.write_json(folder/'runtime_check.json', storage.seal(check))
        storage.write_json(folder/'cache_inventory.json', storage.seal({'identity_sha256': expected['identity_sha256'], 'entries': inventory}))
        checks = [None]*8; dist.all_gather_object(checks, check)
        if rank == 0:
            storage.write_json(args.output_dir/'runtime_checks.json', storage.seal({'status': 'PASS',
                'identity_sha256': expected['identity_sha256'], 'rank_audits': audits, 'rank_checks': checks}))
            print(json.dumps({'event': 'TRACING_CONTROLS_PASSED', 'ranks': 8,
                'controls': ['capture_native', 'layer0_corrupted', 'layer30_clean', 'cache_recapture']}), flush=True)
        dist.barrier()
        for layer in range(1, 31):
            evaluate_layer(args, loaded, tensors, frozen, expected, layer, rank, dist)
        identity.verify(args, expected)
        storage.write_json(folder/'rank_report.json', storage.seal({'status': 'PASS',
            'identity_sha256': expected['identity_sha256'], 'manifest_sha256': frozen['manifest_sha256'],
            'indices': config.indices(rank), 'layers': list(range(1, 31))}))
        dist.barrier()
        return 0
    except Exception as exc:
        storage.write_json(folder/'failure.json', {'status': 'FAILED', 'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        if initialized:
            __import__('torch.distributed', fromlist=['destroy_process_group']).destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
