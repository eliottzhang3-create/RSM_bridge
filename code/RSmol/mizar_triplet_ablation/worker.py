"""Eight GPUs finish each common condition before moving to the next."""
import json
from datetime import timedelta
import time
import traceback
from mizar_layer_pair_search import device, storage
from . import arguments, config, identity, results


def evaluate_condition(args, loaded, prefixes, records, expected, layer, rank, dist):
    from . import runtime
    started=time.monotonic()
    control=[(args.task,layer,expected['identity_sha256']) if rank==0 else None]
    dist.broadcast_object_list(control,src=0)
    if tuple(control[0])!=(args.task,layer,expected['identity_sha256']):raise ValueError('ranks disagree on active condition')
    loaded['controller'].set_drop(layer)
    rows=results.read_rank(args.output_dir,rank,records,args.task,layer,expected)
    path=args.output_dir/'ranks'/f'rank{rank:02d}'/(config.condition_name(args.task,layer)+'.jsonl')
    for index in config.indices(rank)[len(rows):]:
        record=records[index]
        value={'identity_sha256':expected['identity_sha256'],'task':args.task,
            'condition':config.condition_name(args.task,layer),'condition_layer_1based':layer,
            'global_index':index,'key':record['key'],'record_sha256':storage.object_sha(record),
            **runtime.score(loaded,prefixes[index],record)}
        sealed=storage.seal(value);results.validate(sealed,record,args.task,layer,expected)
        storage.append_jsonl(path,value);rows.append(sealed)
    parts=[None]*8 if rank==0 else None
    dist.gather_object(rows,parts,dst=0)
    if rank==0:
        combined=results.merge(parts,records,args.task,layer,expected)
        baseline=combined if layer is None else results.merge(
            [results.read_rank(args.output_dir,r,records,args.task,None,expected) for r in range(8)],records,args.task,None,expected)
        summary=results.save_condition(args.output_dir,combined,baseline,records,args.task,layer,expected,time.monotonic()-started)
        print(json.dumps({'event':'BASELINE_DONE' if layer is None else 'TRIPLET_DONE',
            'task':args.task,'drop_layers_1based':config.drop_layers(layer),'completed':config.conditions(args.task).index(layer)+1,'total_conditions':len(config.conditions(args.task)),
            **{k:summary[k] for k in ('condition','condition_layer_1based','correct','total','accuracy','delta_vs_baseline','elapsed_seconds')},
            'groups':summary['groups']},ensure_ascii=False),flush=True)
    dist.barrier()


def work(args, loaded, prefixes, records, expected, rank, dist):
    for layer in config.conditions(args.task):evaluate_condition(args,loaded,prefixes,records,expected,layer,rank,dist)


def main(argv=None):
    args=arguments.parse_args(argv)
    binding=device.bind_worker(config.WORLD)
    rank=binding['rank'];initialized=False
    folder=args.output_dir/'ranks'/f'rank{rank:02d}'
    try:
        expected=storage.read_json(args.output_dir/'run_config.json')
        frozen=identity.verify(args,expected);records=frozen['records']
        import torch.distributed as dist
        from . import cache, runtime
        loaded=runtime.load(args,expected['official_bundle'],binding)
        storage.write_json(folder/'model_audit.json',storage.seal(loaded['audit']))
        print(json.dumps({'event':'TRIPLET_MODEL_READY','rank':rank,'task':args.task,'gpu':loaded['audit']['gpu']}),flush=True)
        dist.init_process_group('gloo',timeout=timedelta(hours=6));initialized=True
        audits=[None]*8;dist.all_gather_object(audits,loaded['audit'])
        if (len({a['binding']['worker_cuda_visible_devices'] for a in audits})!=8
                or any(cache.precision(a)!=cache.precision(audits[0]) for a in audits)):
            raise ValueError('GPU binding/precision/backend mismatch')
        prefixes,cache_meta=cache.prepare_local(loaded,records,args.output_dir,expected,rank)
        probe=config.indices(rank)[0]
        check=runtime.checks(loaded,prefixes[probe],records[probe])
        check.update(rank=rank,identity_sha256=expected['identity_sha256'])
        storage.write_json(folder/'runtime_check.json',storage.seal(check))
        checks,caches=[None]*8,[None]*8
        dist.all_gather_object(checks,check);dist.all_gather_object(caches,cache_meta)
        if rank==0:
            storage.write_json(args.output_dir/'runtime_checks.json',storage.seal({'status':'PASS',
                'identity_sha256':expected['identity_sha256'],'rank_audits':audits,'rank_checks':checks,
                'cache_manifests':caches}))
        dist.barrier()
        work(args,loaded,prefixes,records,expected,rank,dist)
        identity.verify(args,expected)
        storage.write_json(folder/'rank_report.json',storage.seal({'status':'PASS','task':args.task,
            'identity_sha256':expected['identity_sha256'],'indices':config.indices(rank),
            'conditions_completed':len(config.conditions(args.task))}))
        dist.barrier()
        return 0
    except Exception as exc:
        failure={'status':'FAILED','rank':rank,'task':args.task,'error':repr(exc),'traceback':traceback.format_exc()}
        storage.write_json(folder/'failure.json',failure)
        print(json.dumps({'event':'TRIPLET_WORKER_FAILED',**failure}),flush=True)
        raise
    finally:
        if initialized:__import__('torch.distributed',fromlist=['destroy_process_group']).destroy_process_group()


if __name__=='__main__':raise SystemExit(main())
