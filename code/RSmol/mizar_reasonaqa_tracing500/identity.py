from mizar_layer_pair_search import storage
from mizar_reasonaqa_tracing import identity as original
from . import config, source


def sources():
    guard = storage.read_json(config.PACKAGE/'reference_guard.json')['sources_lf_sha256']
    for name, digest in guard.items():
        if storage.text_sha(config.ROOT/name) != digest:
            raise ValueError(f'original tracing source changed: {name}; cached source identity must remain intact')
    return {**original.reused_sources(), **guard, **{p.relative_to(config.ROOT).as_posix(): storage.text_sha(p)
        for p in [*config.PACKAGE.glob('*.py'), *config.PACKAGE.glob('*.sh'), config.PACKAGE/'reference_guard.json',
                  config.ROOT/'run_mizar_reasonaqa_tracing500_8gpu_5090.sh']}}


def build(args, frozen, context):
    return storage.seal({'schema': config.SCHEMA, 'protocol': config.PROTOCOL,
        'source_output_dir': str(args.source_output_dir), 'manifest_sha256': frozen['manifest_sha256'],
        'source_identity_sha256': context['expected']['identity_sha256'], 'source_snapshots': context['snapshots'],
        'source_expected': context['expected'], 'official_bundle': context['expected']['official_bundle'],
        'sources': sources(), 'environment': original.environment()}, 'identity_sha256')


def verify(args, expected, full=False):
    storage.verify_seal(expected, 'identity_sha256')
    if (expected['schema'] != config.SCHEMA or expected['protocol'] != config.PROTOCOL
            or expected['source_output_dir'] != str(args.source_output_dir)
            or expected['sources'] != sources() or expected['environment'] != original.environment()):
        raise ValueError('500-question continuation identity changed')
    context = source.load(args.source_output_dir, full=full, verify_caches=full)
    frozen = source.manifest(context, args.source_output_dir)
    if (context['expected'] != expected['source_expected'] or context['snapshots'] != expected['source_snapshots']
            or frozen['manifest_sha256'] != expected['manifest_sha256']
            or expected['official_bundle'] != context['expected']['official_bundle']
            or storage.read_json(args.output_dir/'selected/manifest.json') != frozen):
        raise ValueError('source screening/first500 manifest changed')
    return context, frozen
