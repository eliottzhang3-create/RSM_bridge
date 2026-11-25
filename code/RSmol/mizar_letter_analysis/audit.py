"""Verify route checks, durable completion and new EOT prefix inventories."""
from mizar_layer_pair_search import storage
from . import config


def ranks_complete(directory):
    return all((directory/'ranks'/f'rank{rank:02d}/rank_report.json').is_file() for rank in range(config.WORLD))


def verify(directory, expected, records):
    task=expected['task']
    checks=storage.verify_seal(storage.read_json(directory/'runtime_checks.json'))
    if (checks['identity_sha256']!=expected['identity_sha256'] or checks['status']!='PASS'
            or len(checks['rank_audits'])!=8 or len(checks['rank_checks'])!=8
            or len({r['binding']['worker_cuda_visible_devices'] for r in checks['rank_audits']})!=8):
        raise ValueError('incomplete or duplicate-GPU runtime evidence')
    for rank,(model,check) in enumerate(zip(checks['rank_audits'],checks['rank_checks'])):
        folder=directory/'ranks'/f'rank{rank:02d}'
        for name,value in (('model_audit.json',model),('runtime_check.json',check)):
            saved=storage.verify_seal(storage.read_json(folder/name))
            if {k:v for k,v in saved.items() if k!='sha256'}!=value:raise ValueError('rank evidence changed')
        if (check['status']!='PASS' or check['rank']!=rank or model['binding']['rank']!=rank
                or check['identity_sha256']!=expected['identity_sha256']):raise ValueError('wrong runtime rank identity')
        report=storage.verify_seal(storage.read_json(folder/'rank_report.json'))
        if (report['status']!='PASS' or report['task']!=task or report['identity_sha256']!=expected['identity_sha256']
                or report['indices']!=config.indices(rank) or report['conditions_completed']!=len(config.conditions(task))):
            raise ValueError('rank did not finish its exact 125 rows for every condition')
        if task=='cutoff':
            probes=check['cutoff_isolation']
            if check['baseline_native_bitwise_equal'] is not True or [p['cutoff_layer_1based'] for p in probes]!=[3,28]:
                raise ValueError('cutoff native/isolation checks missing')
            if any(p['status']!='PASS' or p['audio_sep_intervention_max_abs_error']!=0 for p in probes):
                raise ValueError('audio/SEP isolation failed')
        else:
            probes=check['bypass_vs_identity_output']
            if [p['layer_1based'] for p in probes]!=[2,29] or any(p['status']!='PASS' or p['max_abs_error']!=0 for p in probes):
                raise ValueError('ablation bypass/identity equivalence failed')
    if task=='cutoff':
        inventories=[(directory/'cache',list(range(1000)),checks['cache_manifests'][0],3)]
    else:
        if len(checks['cache_manifests'])!=8:raise ValueError('missing rank caches')
        inventories=[(directory/'cache'/f'rank{rank:02d}',config.indices(rank),checks['cache_manifests'][rank],4) for rank in range(8)]
    for folder,indices,reference,width in inventories:
        meta=storage.verify_seal(storage.read_json(folder/'cache_manifest.json'))
        key=meta['cache_identity']
        run_sha=key.get('run_identity_sha256')
        if (meta['sha256']!=reference['sha256'] or run_sha!=expected['identity_sha256']
                or len(meta['entry_meta_sha256'])!=len(indices)
                or task!='cutoff' and meta['indices']!=indices):raise ValueError('prefix cache inventory changed')
        for index,digest in zip(indices,meta['entry_meta_sha256']):
            name=f'prefix_{index:0{width}d}'
            entry=storage.verify_seal(storage.read_json(folder/(name+'.json')))
            if (entry['sha256']!=digest or entry['record_sha256']!=storage.object_sha(records[index])
                    or storage.file_sha(folder/(name+'.pt'))!=entry['file_sha256']):
                raise ValueError('cached EOT prefix/input changed')
    return checks
