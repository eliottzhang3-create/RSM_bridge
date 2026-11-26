"""Eight ranks screen all1000, then finish each common patch layer together."""
import json
import time
import traceback
from datetime import timedelta
from mizar_layer_pair_search import device,storage
from . import arguments,config,identity,screening


def screen_one(a,loaded,record,expected,position,silence):
    from mizar_layer_pair_search import audio
    from mizar_reasonaqa_tracing import cache
    from . import runtime,scoring
    wave,audit=audio.load_fixed20(record['audio_ref'])
    if audit!=record['audio_audit']:raise ValueError(f'audio differs from frozen input: {record["key"]}')
    cp=runtime.prefix(loaded,record,wave)
    clean_logits,states=runtime.forward(loaded,cp,capture=True)
    clean=scoring.from_logits(record,clean_logits)
    sp=runtime.corrupt_prefix(cp,silence)
    corrupt_logits,_=runtime.forward(loaded,sp)
    corrupt=scoring.from_logits(record,corrupt_logits)
    accepted=scoring.reason(record,clean,corrupt)=='accepted'
    meta=cache.save(a.output_dir,record,expected,cp,sp,states,audit,cache.precision(loaded['audit'])) if accepted else None
    return screening.make_row(record,expected,position,clean,corrupt,audit,None if meta is None else meta['sha256'])


def screen(a,loaded,inputs,expected,silence,rank,dist):
    from mizar_layer_pair_search import audio
    from mizar_reasonaqa_tracing import cache
    rows=screening.read_rank(a.output_dir,rank,inputs,expected)
    positions=config.indices(rank,1000)
    for row in rows:
        if row['accepted']:
            record=inputs['records'][row['source_manifest_index']]
            cache.load(a.output_dir,record,expected,digest=row['cache_meta_sha256'],precision=cache.precision(loaded['audit']),tensors=False)
    remaining=positions[len(rows):]
    audio.prepare_selected_rows([inputs['records'][inputs['shuffle_order'][p]] for p in remaining])
    try:
        for pos in remaining:
            record=inputs['records'][inputs['shuffle_order'][pos]]
            value=screen_one(a,loaded,record,expected,pos,silence)
            sealed=storage.seal(value);screening.validate(sealed,record,expected,pos)
            storage.append_jsonl(screening.rank_path(a.output_dir,rank),value);rows.append(sealed)
            print(json.dumps({'event':'SCREENED','task':a.task,'rank':rank,'local_completed':len(rows),'local_total':125,
                'source_index':record['source_manifest_index'],'key':record['key'],'group':record['analysis_group'],
                'reason':value['reason'],'clean_prediction':value['clean']['prediction_label'],
                'silence_prediction':value['corrupted']['prediction_label'],'Pclean':value['clean']['target_probability'],
                'Pcorrupted':value['corrupted']['target_probability']}),flush=True)
    finally:audio.clear_selected_rows()
    parts=[None]*8 if rank==0 else None
    dist.gather_object(rows,parts,dst=0)
    if rank==0:screening.publish(a.output_dir,screening.merge(parts,inputs,expected),inputs,expected)
    dist.barrier()
    return storage.verify_seal(storage.read_json(a.output_dir/'selected/manifest.json'),'manifest_sha256')


def evaluate_layer(a,loaded,tensors,frozen,expected,layer,rank,dist):
    from . import runtime,scoring,results
    started=time.monotonic();control=[(layer,frozen['manifest_sha256']) if rank==0 else None]
    dist.broadcast_object_list(control,src=0)
    if tuple(control[0])!=(layer,frozen['manifest_sha256']):raise ValueError('ranks disagree on patch layer')
    rows=results.read_rank(a.output_dir,rank,frozen,expected,layer)
    for index in config.indices(rank,frozen['selected'])[len(rows):]:
        r,data=frozen['records'][index],tensors[index]
        logits,_=runtime.forward(loaded,data['corrupted_prefix'],layer=layer,clean_states=data['clean_states'])
        value=results.make_row(r,expected,frozen,layer,scoring.from_logits(r,logits))
        sealed=storage.seal(value);results.validate(sealed,r,expected,frozen,layer)
        storage.append_jsonl(results.rank_path(a.output_dir,rank,layer),value);rows.append(sealed)
    parts=[None]*8 if rank==0 else None
    dist.gather_object(rows,parts,dst=0)
    if rank==0:
        merged=results.merge(parts,frozen,expected,layer)
        summary=results.save_layer(a.output_dir,merged,frozen,expected,layer,time.monotonic()-started)
        print(json.dumps({'event':'PATCH_LAYER_DONE','task':a.task,**{k:summary[k] for k in
            ('layer_1based','total','mean_RR','median_RR','patched_accuracy','groups','elapsed_seconds')}},ensure_ascii=False),flush=True)
    dist.barrier()


def local_caches(a,loaded,inputs,frozen,expected,rank,silence):
    from mizar_reasonaqa_tracing import cache
    import torch
    tensors,entries={},[]
    own=config.indices(rank,frozen['selected'])
    for index in own:
        r=frozen['records'][index];base=inputs['records'][r['source_manifest_index']]
        data,meta=cache.load(a.output_dir,base,expected,digest=r['screening']['cache_meta_sha256'],precision=cache.precision(loaded['audit']))
        for k in ('clean_prefix','corrupted_prefix'):data[k]=data[k].to(loaded['device'])
        if not torch.equal(data['corrupted_prefix'][:,:126],silence):raise ValueError('cached silence differs from native encoding')
        tensors[index]=data;entries.append({'selected_index':index,'source_manifest_index':r['source_manifest_index'],'meta_sha256':meta['sha256']})
    return tensors,entries


def check_rank(a,loaded,inputs,frozen,expected,rank,silence,tensors):
    from mizar_layer_pair_search import audio
    from mizar_reasonaqa_tracing import cache
    from . import runtime
    n=frozen['selected'];own=config.indices(rank,n)
    if not n:
        check={'status':'SKIPPED_NO_QUALIFYING','probe_selected_index':None}
    else:
        probe=own[0] if own else 0;r=frozen['records'][probe]
        if probe in tensors:data=tensors[probe]
        else:
            # An idle rank still executes the same correctness controls using
            # a shared selected probe, while its patch result shard stays empty.
            data,_=cache.load(a.output_dir,inputs['records'][r['source_manifest_index']],expected,
                digest=r['screening']['cache_meta_sha256'],precision=cache.precision(loaded['audit']))
            for k in ('clean_prefix','corrupted_prefix'):data[k]=data[k].to(loaded['device'])
        check=runtime.controls(loaded,data['clean_prefix'],data['corrupted_prefix'],data['clean_states'],r,
            r['screening']['clean'],r['screening']['corrupted'])
        check['probe_selected_index']=probe
    check.update(rank=rank,identity_sha256=expected['identity_sha256'],silence_sha256=audio.tensor_sha(silence))
    return check


def main(argv=None):
    a=arguments.parse_args(argv);binding=device.bind_worker(8);rank=binding['rank'];initialized=False;phase='verify_identity'
    folder=a.output_dir/'ranks'/f'rank{rank:02d}'
    try:
        expected=storage.read_json(a.output_dir/'run_config.json');inputs=identity.verify(a,expected)
        import torch.distributed as dist
        from mizar_reasonaqa_tracing import cache
        from . import runtime
        phase='load_model';loaded=runtime.load(a,expected['official_bundle'],binding)
        storage.write_json(folder/'model_audit.json',storage.seal(loaded['audit']))
        print(json.dumps({'event':'TRACING_MODEL_READY','task':a.task,'rank':rank,'gpu':loaded['audit']['gpu']}),flush=True)
        dist.init_process_group('gloo',timeout=timedelta(hours=6));initialized=True
        audits=[None]*8;dist.all_gather_object(audits,loaded['audit'])
        if (len({x['binding']['worker_cuda_visible_devices'] for x in audits})!=8
                or any(cache.precision(x)!=cache.precision(audits[0]) for x in audits)):raise ValueError('GPU/precision mismatch')
        phase='encode_silence';silence=runtime.silent_audio(loaded,inputs['records'][0])
        phase='screen_full1000';frozen=screen(a,loaded,inputs,expected,silence,rank,dist)
        if a.stage=='select':return 0
        phase='load_selected_cache';tensors,entries=local_caches(a,loaded,inputs,frozen,expected,rank,silence)
        print(json.dumps({'event':'TRACING_CACHE_READY','task':a.task,'rank':rank,'local_selected':len(entries),'selected':frozen['selected']}),flush=True)
        phase='runtime_controls';check=check_rank(a,loaded,inputs,frozen,expected,rank,silence,tensors)
        storage.write_json(folder/'runtime_check.json',storage.seal(check))
        storage.write_json(folder/'cache_inventory.json',storage.seal({'identity_sha256':expected['identity_sha256'],'entries':entries}))
        checks=[None]*8;dist.all_gather_object(checks,check)
        if len({c['silence_sha256'] for c in checks})!=1:raise ValueError('silence differs across ranks')
        if rank==0:
            storage.write_json(a.output_dir/'runtime_checks.json',storage.seal({'status':'PASS','identity_sha256':expected['identity_sha256'],
                'rank_audits':audits,'rank_checks':checks}))
            print(json.dumps({'event':'TRACING_CONTROLS_DONE','task':a.task,'selected':frozen['selected']}),flush=True)
        dist.barrier()
        layers=list(range(1,31)) if frozen['selected'] else []
        for layer in layers:
            phase=f'patch_layer_{layer:02d}';evaluate_layer(a,loaded,tensors,frozen,expected,layer,rank,dist)
        phase='final_verify';identity.verify(a,expected)
        storage.write_json(folder/'rank_report.json',storage.seal({'status':'PASS','task':a.task,'identity_sha256':expected['identity_sha256'],
            'manifest_sha256':frozen['manifest_sha256'],'indices':config.indices(rank,frozen['selected']),'layers':layers}))
        dist.barrier();return 0
    except Exception as exc:
        failure={'status':'FAILED','task':a.task,'rank':rank,'phase':phase,'error':repr(exc),'traceback':traceback.format_exc()}
        storage.write_json(folder/'failure.json',failure);print(json.dumps({'event':'WORKER_FAILED',**failure}),flush=True);raise
    finally:
        if initialized:__import__('torch.distributed',fromlist=['destroy_process_group']).destroy_process_group()


if __name__=='__main__':raise SystemExit(main())
