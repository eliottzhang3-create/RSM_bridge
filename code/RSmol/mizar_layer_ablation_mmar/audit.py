from mizar_layer_pair_search import storage
from . import config


def verify(directory,expected,records):
    checks=storage.verify_seal(storage.read_json(directory/'runtime_checks.json'))
    if (checks['identity_sha256']!=expected['identity_sha256'] or checks['status']!='PASS'
            or len(checks['rank_audits'])!=8 or len(checks['rank_checks'])!=8 or len(checks['rank_caches'])!=8):
        raise ValueError('incomplete eight-worker runtime evidence')
    if len({a['binding']['worker_cuda_visible_devices'] for a in checks['rank_audits']})!=8:
        raise ValueError('duplicate GPU binding')
    for rank in range(8):
        folder=directory/'ranks'/f'rank{rank:02d}'
        for key,name in (('rank_audits','model_audit.json'),('rank_checks','runtime_check.json')):
            value=storage.verify_seal(storage.read_json(folder/name))
            if {k:v for k,v in value.items() if k!='sha256'}!=checks[key][rank]: raise ValueError('changed per-rank evidence')
        check=checks['rank_checks'][rank]; model=checks['rank_audits'][rank]
        if (check['status']!='PASS' or check['rank']!=rank or check['identity_sha256']!=expected['identity_sha256']
                or model['binding']['rank']!=rank): raise ValueError('wrong rank/runtime check identity')
        report=storage.verify_seal(storage.read_json(folder/'rank_report.json'))
        if (report['status']!='PASS' or report['identity_sha256']!=expected['identity_sha256']
                or report.get('task')!=config.SCHEMA
                or report['indices']!=config.indices(rank,len(records))): raise ValueError('rank did not finish exact input shard')
        cache_folder=directory/'cache'/f'rank{rank:02d}'
        cache=storage.verify_seal(storage.read_json(cache_folder/'cache_manifest.json'))
        if (cache['sha256']!=checks['rank_caches'][rank]['sha256']
                or cache['cache_identity']['run_identity_sha256']!=expected['identity_sha256']
                or cache['indices']!=config.indices(rank,len(records)) or len(cache['entry_meta_sha256'])!=len(cache['indices'])):
            raise ValueError('changed cache inventory')
        for index,digest in zip(cache['indices'],cache['entry_meta_sha256']):
            meta=storage.verify_seal(storage.read_json(cache_folder/f'prefix_{index:04d}.json'))
            if (meta['sha256']!=digest or meta['record_sha256']!=storage.object_sha(records[index])
                    or storage.file_sha(cache_folder/f'prefix_{index:04d}.pt')!=meta['file_sha256']):
                raise ValueError('changed prefix file/input')
    return checks


def ranks_complete(directory):
    return all((directory/'ranks'/f'rank{rank:02d}'/'rank_report.json').exists() for rank in range(8))
