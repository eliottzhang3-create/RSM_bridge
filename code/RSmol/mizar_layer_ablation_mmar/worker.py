import traceback
from datetime import timedelta
from mizar_layer_pair_search import device,storage
from . import arguments,cache,config,identity,runtime,task as task_api


def main(argv=None):
    args=arguments.parse_args(argv)
    binding=device.bind_worker(8); rank=binding['rank']; initialized=False
    folder=args.output_dir/'ranks'/f'rank{rank:02d}'
    try:
        expected=storage.read_json(args.output_dir/'run_config.json')
        frozen=identity.verify(args,expected,task_api.protocol(args)); records=frozen['records']
        import torch.distributed as dist
        loaded=runtime.load(args,expected['official_bundle'],binding)
        storage.write_json(folder/'model_audit.json',storage.seal(loaded['audit']))
        dist.init_process_group('gloo',timeout=timedelta(hours=6)); initialized=True
        audits=[None]*8; dist.all_gather_object(audits,loaded['audit'])
        if (len({a['binding']['worker_cuda_visible_devices'] for a in audits})!=8
                or any(cache.precision(a)!=cache.precision(audits[0]) for a in audits)):
            raise ValueError('GPU binding/precision/backend mismatch')
        prefixes,cache_meta=cache.prepare_local(loaded,records,args.output_dir,expected,rank)
        rank_checks=task_api.runtime_check(loaded,prefixes[config.indices(rank,len(records))[0]],records[config.indices(rank,len(records))[0]])
        rank_checks.update(rank=rank,identity_sha256=expected['identity_sha256'])
        storage.write_json(folder/'runtime_check.json',storage.seal(rank_checks))
        checks,caches=[None]*8,[None]*8
        dist.all_gather_object(checks,rank_checks); dist.all_gather_object(caches,cache_meta)
        if rank==0:
            storage.write_json(args.output_dir/'runtime_checks.json',storage.seal({'status':'PASS',
                'identity_sha256':expected['identity_sha256'],'rank_audits':audits,'rank_checks':checks,'rank_caches':caches}))
        dist.barrier()
        task_api.work(args,loaded,prefixes,records,expected,rank,dist)
        identity.verify(args,expected,task_api.protocol(args))
        storage.write_json(folder/'rank_report.json',storage.seal({'status':'PASS','task':config.SCHEMA,
            'identity_sha256':expected['identity_sha256'],'indices':config.indices(rank,len(records))}))
        dist.barrier()
        return 0
    except Exception as exc:
        storage.write_json(folder/'failure.json',{'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        if initialized: __import__('torch.distributed',fromlist=['destroy_process_group']).destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
