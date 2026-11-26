from mizar_layer_pair_search import storage
from mizar_reasonaqa_tracing import cache
from . import config, source


def verify(args, frozen, expected, context):
    checks = storage.verify_seal(storage.read_json(args.output_dir/'runtime_checks.json'))
    if (checks['status'] != 'PASS' or checks['identity_sha256'] != expected['identity_sha256']
            or len(checks['rank_audits']) != 8 or len(checks['rank_checks']) != 8
            or len({a['binding']['worker_cuda_visible_devices'] for a in checks['rank_audits']}) != 8):
        raise ValueError('incomplete or duplicate-GPU runtime evidence')
    if frozen != source.manifest(context, args.source_output_dir):
        raise ValueError('selected500 is not the first500 qualifying source rows')
    for rank, (model, check) in enumerate(zip(checks['rank_audits'], checks['rank_checks'])):
        folder = args.output_dir/'ranks'/f'rank{rank:02d}'
        for name, value in [('model_audit.json',model),('runtime_check.json',check)]:
            if storage.read_json(folder/name) != storage.seal(value):
                raise ValueError('rank evidence changed')
        if (check['status'] != 'PASS' or check['rank'] != rank or model['binding']['rank'] != rank
                or check['identity_sha256'] != expected['identity_sha256']
                or any(check[k] is not True for k in ('capture_vs_native_logits_bitwise_equal',
                    'layer0_equals_corrupted_logits_bitwise','layer30_equals_clean_logits_bitwise',
                    'cache_equals_repeated_clean_states_bitwise'))
                or check['state_shape'] != [30,256,576] or check['text_start'] != 127):
            raise ValueError('missing capture/layer0/layer30/cache controls')
        report = storage.verify_seal(storage.read_json(folder/'rank_report.json'))
        if (report['status'] != 'PASS' or report['identity_sha256'] != expected['identity_sha256']
                or report['manifest_sha256'] != frozen['manifest_sha256'] or report['indices'] != config.indices(rank)
                or report['layers'] != list(range(1,31))):
            raise ValueError('rank did not finish exact62/63 questions and all30 layers')
        inventory = storage.verify_seal(storage.read_json(folder/'cache_inventory.json'))
        entries = []
        for index in config.indices(rank):
            r = frozen['records'][index]; original = context['pool']['records'][r['source_manifest_index']]
            if storage.file_sha(original['audio_ref']['path']) != r['screening']['audio_audit']['audio_source_sha256']:
                raise ValueError('selected original audio content changed')
            _, meta = cache.load(args.source_output_dir,original,context['expected'],
                digest=r['screening']['cache_meta_sha256'],precision=cache.precision(model),tensors=False)
            entries.append({'selected_index':index,'source_manifest_index':r['source_manifest_index'],'meta_sha256':meta['sha256']})
        if inventory['identity_sha256'] != expected['identity_sha256'] or inventory['entries'] != entries:
            raise ValueError('rank source cache inventory changed')
    if len({c['silence_sha256'] for c in checks['rank_checks']}) != 1:
        raise ValueError('encoded silence differs across GPUs')
    return checks
