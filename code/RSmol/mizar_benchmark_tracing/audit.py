from mizar_layer_pair_search import storage
from mizar_reasonaqa_tracing import cache
from . import config


def verify(directory,inputs,frozen,expected):
    checks=storage.verify_seal(storage.read_json(directory/'runtime_checks.json'))
    if (checks['status']!='PASS' or checks['identity_sha256']!=expected['identity_sha256']
            or len(checks['rank_audits'])!=8 or len(checks['rank_checks'])!=8
            or len({m['binding']['worker_cuda_visible_devices'] for m in checks['rank_audits']})!=8):
        raise ValueError('incomplete/duplicate-GPU runtime evidence')
    n=frozen['selected'];layers=list(range(1,31)) if n else []
    for rank,(model,check) in enumerate(zip(checks['rank_audits'],checks['rank_checks'])):
        folder=directory/'ranks'/f'rank{rank:02d}'
        for name,value in [('model_audit.json',model),('runtime_check.json',check)]:
            if storage.read_json(folder/name)!=storage.seal(value):raise ValueError('rank evidence changed')
        if (check['rank']!=rank or model['binding']['rank']!=rank or check['identity_sha256']!=expected['identity_sha256']
                or cache.precision(model)!=cache.precision(checks['rank_audits'][0])):raise ValueError('wrong rank or precision evidence')
        own=config.indices(rank,n)
        if n:
            if (check['status']!='PASS' or check['probe_selected_index']!=(own[0] if own else 0)
                    or check['state_shape']!=[30,256,576] or check['exp_max_ULPs']!=8
                    or any(check[k] is not True for k in ('capture_native_bitwise','layer0_corrupted_bitwise',
                        'layer30_clean_bitwise','cache_recapture_bitwise','saved_logprobs_exact'))):
                raise ValueError('missing native/capture/layer0/layer30 controls')
        elif check['status']!='SKIPPED_NO_QUALIFYING' or check['probe_selected_index'] is not None:
            raise ValueError('wrong no-qualifying control status')
        report=storage.verify_seal(storage.read_json(folder/'rank_report.json'))
        if (report['status']!='PASS' or report['task']!=frozen['task'] or report['identity_sha256']!=expected['identity_sha256']
                or report['manifest_sha256']!=frozen['manifest_sha256'] or report['indices']!=own or report['layers']!=layers):
            raise ValueError('rank did not finish exact dynamic questions/layers')
        inventory=storage.verify_seal(storage.read_json(folder/'cache_inventory.json'));entries=[]
        for index in own:
            r=frozen['records'][index];base=inputs['records'][r['source_manifest_index']]
            _,meta=cache.load(directory,base,expected,digest=r['screening']['cache_meta_sha256'],precision=cache.precision(model),tensors=False)
            if meta['audio_audit']!=r['screening']['audio_audit']:raise ValueError('cache audio audit changed')
            entries.append({'selected_index':index,'source_manifest_index':r['source_manifest_index'],'meta_sha256':meta['sha256']})
        if inventory['identity_sha256']!=expected['identity_sha256'] or inventory['entries']!=entries:raise ValueError('rank cache inventory changed')
    if len({c['silence_sha256'] for c in checks['rank_checks']})!=1:raise ValueError('rank silence differs')
    return checks
