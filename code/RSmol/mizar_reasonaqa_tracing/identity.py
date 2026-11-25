"""Bind this experiment to its model, pool, reused source and runtime versions."""
from importlib.metadata import version
from mizar_layer_pair_search import storage
from . import config, prepare


def reused_sources():
    guard = storage.read_json(config.PACKAGE/'reference_guard.json')['sources_lf_sha256']
    for name, digest in guard.items():
        if storage.text_sha(config.ROOT/name) != digest:
            raise ValueError(f'reused implementation changed: {name}')
    return guard


def sources():
    return {**reused_sources(), **{p.relative_to(config.ROOT).as_posix(): storage.text_sha(p)
        for p in [*config.PACKAGE.glob('*.py'), *config.PACKAGE.glob('*.sh'),
                  config.PACKAGE/'reference_guard.json', config.ROOT/'run_mizar_reasonaqa_tracing_8gpu_5090.sh']}}


def environment():
    return {name: version(name) for name in ('torch', 'torchaudio', 'transformers', 'numpy', 'soundfile')}


def build(args, pool):
    from mizar_official_eval import runtime as official
    return storage.seal({'schema': config.SCHEMA, 'protocol': config.PROTOCOL,
        'model_dir': str(args.model_dir), 'checkpoint': str(args.checkpoint),
        'pool_manifest': str(args.pool_manifest), 'pool_sha256': pool['manifest_sha256'],
        'source_arguments': prepare.source_arguments(args), 'sources': sources(), 'environment': environment(),
        'official_bundle': official.audit_bundle(args.model_dir, args.checkpoint)}, 'identity_sha256')


def verify(args, expected, full=False):
    storage.verify_seal(expected, 'identity_sha256')
    if (expected['schema'] != config.SCHEMA or expected['protocol'] != config.PROTOCOL
            or expected['sources'] != sources() or expected['environment'] != environment()
            or expected['source_arguments'] != prepare.source_arguments(args)
            or expected['pool_manifest'] != str(args.pool_manifest)
            or expected['checkpoint'] != str(args.checkpoint) or expected['model_dir'] != str(args.model_dir)):
        raise ValueError('tracing data/model/source/environment identity differs')
    pool = prepare.load(args.pool_manifest, full=full)
    if pool['manifest_sha256'] != expected['pool_sha256']:
        raise ValueError('fixed5000 pool changed')
    from mizar_official_eval import runtime as official
    official.verify_bundle_unchanged(expected['official_bundle'], args.model_dir, args.checkpoint)
    if full and official.audit_bundle(args.model_dir, args.checkpoint) != expected['official_bundle']:
        raise ValueError('model assets changed during experiment')
    return pool
