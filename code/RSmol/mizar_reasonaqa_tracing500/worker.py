"""Patch only: reuse source states, no screening or per-question audio encoding."""
import json
import time
import traceback
from datetime import timedelta
from mizar_layer_pair_search import device, storage
from . import arguments, config, identity


def evaluate_layer(args, loaded, tensors, frozen, expected, layer, rank, dist):
    from . import runtime, results
    records = frozen['records']; started = time.monotonic()
    control = [(layer, frozen['manifest_sha256']) if rank == 0 else None]
    dist.broadcast_object_list(control, src=0)
    if tuple(control[0]) != (layer, frozen['manifest_sha256']):
        raise ValueError('ranks disagree on patched layer')
    rows = results.read_rank(args.output_dir, rank, frozen, expected, layer)
    for index in config.indices(rank)[len(rows):]:
        r, data = records[index], tensors[index]
        logits, _ = runtime.forward(loaded, data['corrupted_prefix'], layer=layer, clean_states=data['clean_states'])
        value = results.make_row(r, expected, frozen, layer, runtime.from_logits(r, logits))
        results.validate(storage.seal(value), r, expected, frozen, layer)
        storage.append_jsonl(results.rank_path(args.output_dir, rank, layer), value)
        rows.append(storage.seal(value))
    parts = [None]*8 if rank == 0 else None
    dist.gather_object(rows, parts, dst=0)
    if rank == 0:
        merged = results.merge(parts, frozen, expected, layer)
        summary = results.save_layer(args.output_dir, merged, frozen, expected, layer, time.monotonic()-started)
        print(json.dumps({'event': 'PATCH_LAYER_DONE', **{k: summary[k] for k in
            ('layer_1based','total','mean_RR','median_RR','patched_accuracy','mean_Ppatched','groups','elapsed_seconds')}},
            ensure_ascii=False), flush=True)
    dist.barrier()


def main(argv=None):
    args = arguments.parse_args(argv); binding = device.bind_worker(8)
    rank, initialized, phase = binding['rank'], False, 'verify_source'
    folder = args.output_dir/'ranks'/f'rank{rank:02d}'
    try:
        expected = storage.read_json(args.output_dir/'run_config.json')
        context, frozen = identity.verify(args, expected)
        import torch
        import torch.distributed as dist
        from mizar_layer_pair_search import audio
        from mizar_reasonaqa_tracing import cache
        from . import runtime, source
        phase = 'load_model'
        loaded = runtime.load(source.original_arguments(context['expected']), expected['official_bundle'], binding)
        loaded['audit'].update(experiment=config.SCHEMA, reused_source_identity_sha256=context['expected']['identity_sha256'])
        storage.write_json(folder/'model_audit.json', storage.seal(loaded['audit']))
        print(json.dumps({'event':'TRACING500_MODEL_READY','rank':rank,'gpu':loaded['audit']['gpu']}), flush=True)
        dist.init_process_group('gloo', timeout=timedelta(hours=6)); initialized = True
        audits = [None]*8; dist.all_gather_object(audits, loaded['audit'])
        if (len({a['binding']['worker_cuda_visible_devices'] for a in audits}) != 8
                or any(cache.precision(a) != cache.precision(audits[0]) for a in audits)):
            raise ValueError('duplicate GPU or precision/backend mismatch')
        phase = 'load_original_caches'
        tensors, inventory, silence = {}, [], None
        for index in config.indices(rank):
            r = frozen['records'][index]; original = context['pool']['records'][r['source_manifest_index']]
            data, meta = cache.load(args.source_output_dir, original, context['expected'],
                digest=r['screening']['cache_meta_sha256'], precision=cache.precision(loaded['audit']))
            for key in ('clean_prefix','corrupted_prefix'):
                data[key] = data[key].to(loaded['device'])
            if silence is None:
                # One genuine silent waveform control per device, not per question.
                silence = runtime.silent_audio(loaded, r)
            if not torch.equal(data['corrupted_prefix'][:, :126], silence):
                raise ValueError('cached corruption differs from encoded silence')
            tensors[index] = data
            inventory.append({'selected_index':index,'source_manifest_index':r['source_manifest_index'],'meta_sha256':meta['sha256']})
            if len(inventory) % 20 == 0 or len(inventory) == len(config.indices(rank)):
                print(json.dumps({'event':'TRACING500_CACHE_LOADED','rank':rank,'loaded':len(inventory),
                    'total':len(config.indices(rank))}), flush=True)
        silence_hash = audio.tensor_sha(silence)
        phase = 'runtime_controls'
        probe = config.indices(rank)[0]; r, data = frozen['records'][probe], tensors[probe]
        check = runtime.controls(loaded,data['clean_prefix'],data['corrupted_prefix'],data['clean_states'],r,
            r['screening']['clean'],r['screening']['corrupted'])
        check.update(rank=rank,identity_sha256=expected['identity_sha256'],silence_sha256=silence_hash)
        storage.write_json(folder/'runtime_check.json',storage.seal(check))
        storage.write_json(folder/'cache_inventory.json',storage.seal({'identity_sha256':expected['identity_sha256'],'entries':inventory}))
        checks=[None]*8;dist.all_gather_object(checks,check)
        if len({c['silence_sha256'] for c in checks}) != 1:
            raise ValueError('encoded silence differs across GPUs')
        if rank == 0:
            storage.write_json(args.output_dir/'runtime_checks.json',storage.seal({'status':'PASS',
                'identity_sha256':expected['identity_sha256'],'rank_audits':audits,'rank_checks':checks}))
            print(json.dumps({'event':'TRACING500_CONTROLS_PASSED','selected':500,'ranks':8}),flush=True)
        dist.barrier()
        for layer in range(1,31):
            phase=f'patch_layer_{layer:02d}'
            evaluate_layer(args,loaded,tensors,frozen,expected,layer,rank,dist)
        phase='final_verify'
        identity.verify(args,expected)
        storage.write_json(folder/'rank_report.json',storage.seal({'status':'PASS','identity_sha256':expected['identity_sha256'],
            'manifest_sha256':frozen['manifest_sha256'],'indices':config.indices(rank),'layers':list(range(1,31))}))
        dist.barrier()
        return 0
    except Exception as exc:
        failure={'status':'FAILED','rank':rank,'phase':phase,'error':repr(exc),'traceback':traceback.format_exc()}
        storage.write_json(folder/'failure.json',failure)
        print(json.dumps({'event':'TRACING500_WORKER_FAILED',**failure}),flush=True)
        raise
    finally:
        if initialized:
            __import__('torch.distributed',fromlist=['destroy_process_group']).destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
