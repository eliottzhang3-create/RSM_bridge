"""Final evidence checks including cached tensors and original screened audio."""
from mizar_layer_pair_search import storage
from . import cache, config, selection


def verify(directory, pool, frozen, expected):
    checks = storage.verify_seal(storage.read_json(directory/'runtime_checks.json'))
    if (checks['status'] != 'PASS' or checks['identity_sha256'] != expected['identity_sha256']
            or len(checks['rank_audits']) != 8 or len(checks['rank_checks']) != 8
            or len({x['binding']['worker_cuda_visible_devices'] for x in checks['rank_audits']}) != 8):
        raise ValueError('incomplete/duplicate-GPU runtime evidence')
    history = selection.recover(directory, pool, expected)
    if sum(r['accepted'] for r in history) != config.TARGET or not history[-1]['accepted']:
        raise ValueError('screening did not stop exactly on the 1000th accepted question')
    audio_paths = {}
    for row in history:
        record = pool['records'][row['source_manifest_index']]
        path = record['audio_ref']['path']; digest = row['audio_audit']['audio_source_sha256']
        if path in audio_paths and audio_paths[path] != digest:
            raise ValueError('same audio has conflicting content hashes')
        audio_paths[path] = digest
    for path, digest in audio_paths.items():
        if storage.file_sha(path) != digest:
            raise ValueError(f'screened audio content changed: {path}')
    for rank, (model, check) in enumerate(zip(checks['rank_audits'], checks['rank_checks'])):
        folder = directory/'ranks'/f'rank{rank:02d}'
        for name, value in [('model_audit.json', model), ('runtime_check.json', check)]:
            if storage.read_json(folder/name) != storage.seal(value):
                raise ValueError('rank runtime evidence changed')
        if (check['status'] != 'PASS' or check['rank'] != rank or model['binding']['rank'] != rank
                or check['identity_sha256'] != expected['identity_sha256']
                or any(check[k] is not True for k in ('capture_vs_native_logits_bitwise_equal',
                    'layer0_equals_corrupted_logits_bitwise', 'layer30_equals_clean_logits_bitwise',
                    'cache_equals_repeated_clean_states_bitwise'))
                or check['state_shape'] != [30, 256, 576] or check['text_start'] != 127):
            raise ValueError('missing native/capture/layer0/layer30 controls')
        report = storage.verify_seal(storage.read_json(folder/'rank_report.json'))
        if (report['status'] != 'PASS' or report['identity_sha256'] != expected['identity_sha256']
                or report['manifest_sha256'] != frozen['manifest_sha256'] or report['indices'] != config.indices(rank)
                or report['layers'] != list(range(1, 31))):
            raise ValueError('rank did not finish its exact questions/layers')
        inventory = storage.verify_seal(storage.read_json(folder/'cache_inventory.json'))
        entries = []
        for index in config.indices(rank):
            record = frozen['records'][index]
            _, meta = cache.load(directory, pool['records'][record['source_manifest_index']], expected,
                digest=record['screening']['cache_meta_sha256'], precision=cache.precision(model), tensors=False)
            entries.append({'selected_index': index, 'meta_sha256': meta['sha256']})
        if inventory['identity_sha256'] != expected['identity_sha256'] or inventory['entries'] != entries:
            raise ValueError('rank cache inventory differs')
    if len({x['silence_sha256'] for x in checks['rank_checks']}) != 1:
        raise ValueError('encoded silence differs by rank')
    return checks
