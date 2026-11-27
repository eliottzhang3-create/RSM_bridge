from mizar_layer_pair_search import storage
from . import config,cache


def verify(directory,expected,records):
    checks=storage.verify_seal(storage.read_json(directory/'runtime_checks.json'))
    if (checks['status']!='PASS' or checks['identity_sha256']!=expected['identity_sha256']
            or len(checks['rank_audits'])!=8 or len(checks['rank_checks'])!=8 or len(checks['cache_manifests'])!=8
            or len({a['binding']['worker_cuda_visible_devices'] for a in checks['rank_audits']})!=8):
        raise ValueError('incomplete or duplicate-GPU evidence')
    task=expected['task']
    for rank,(model,check,inventory) in enumerate(zip(checks['rank_audits'],checks['rank_checks'],checks['cache_manifests'])):
        folder=directory/'ranks'/f'rank{rank:02d}'
        for name,value in [('model_audit.json',model),('runtime_check.json',check)]:
            if storage.read_json(folder/name)!=storage.seal(value):raise ValueError('rank evidence changed')
        if (model['binding']['rank']!=rank or check['rank']!=rank or check['status']!='PASS'
                or check['identity_sha256']!=expected['identity_sha256'] or check['baseline_restored_bitwise'] is not True
                or cache.precision(model)!=cache.precision(checks['rank_audits'][0])):raise ValueError('rank/native/precision check failed')
        probes=check['triplet_identity_probes']
        if len(probes)!=3 or [p['start'] for p in probes]!=[6,15,25]:raise ValueError('wrong triplet probes')
        for probe in probes:
            drop=config.drop_layers(probe['start']);trace=probe['execution']
            if (probe['status']!='PASS' or probe['max_abs_error']!=0 or probe['drop_layers_1based']!=drop
                    or trace['drop_layers_1based']!=drop or trace['effective_layers']!=27
                    or trace['executed_original_layers_1based']!=[i for i in range(1,31) if i not in drop]):
                raise ValueError('triplet identity vs bypass check failed')
        report=storage.verify_seal(storage.read_json(folder/'rank_report.json'))
        if (report['status']!='PASS' or report['task']!=task or report['identity_sha256']!=expected['identity_sha256']
                or report['indices']!=config.indices(rank) or report['conditions_completed']!=21):raise ValueError('rank did not finish125x21')
        directory_cache=directory/'cache'/f'rank{rank:02d}'
        meta=storage.verify_seal(storage.read_json(directory_cache/'cache_manifest.json'))
        if (meta!=inventory or meta['cache_identity']!={'run_identity_sha256':expected['identity_sha256'],
                'precision':cache.precision(model),'rank':rank} or meta['indices']!=config.indices(rank)
                or len(meta['entry_meta_sha256'])!=125):raise ValueError('prefix cache inventory changed')
        for index,digest in zip(config.indices(rank),meta['entry_meta_sha256']):
            entry=storage.verify_seal(storage.read_json(directory_cache/f'prefix_{index:04d}.json'))
            if (entry['sha256']!=digest or entry['cache_identity']!=meta['cache_identity']
                    or entry['record_sha256']!=storage.object_sha(records[index])
                    or entry['audio_audit']!=records[index]['audio_audit']
                    or storage.file_sha(directory_cache/f'prefix_{index:04d}.pt')!=entry['file_sha256']):
                raise ValueError('prefix/audio content changed')
    return checks
