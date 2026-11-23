"""Pin reused files without changing the earlier experiment's identity rules."""
from importlib.metadata import version
from mizar_layer_pair_search import storage
from . import config, frozen_manifest


def sources():
    guard = storage.read_json(config.PACKAGE/'reference_guard.json')
    for name, digest in guard['sources_lf_sha256'].items():
        if storage.text_sha(config.ROOT/name) != digest:
            raise ValueError(f'reused source changed: {name}')
    paths = [*config.PACKAGE.glob('*.py'), *config.PACKAGE.glob('*.sh'), config.PACKAGE/'reference_guard.json',
             config.ROOT/'run_mizar_audio_cutoff_4090.sh', config.ROOT/'run_mizar_audio_cutoff_8gpu_3090.sh']
    paths.extend(config.ROOT/name for name in guard['sources_lf_sha256'])
    return {p.relative_to(config.ROOT).as_posix(): storage.text_sha(p) for p in sorted(set(paths))}


def environment():
    return {name: version(name) for name in ('torch', 'torchaudio', 'transformers', 'numpy', 'soundfile', 'pyarrow')}


def build(args, frozen):
    from mizar_official_eval import runtime as official
    return storage.seal({'schema': config.SCHEMA, 'protocol': config.PROTOCOL, 'world_size': args.world_size,
        'manifest_path': str(args.manifest), 'manifest_sha256': frozen['manifest_sha256'],
        'model_dir': str(args.model_dir), 'checkpoint': str(args.checkpoint),
        'official_bundle': official.audit_bundle(args.model_dir, args.checkpoint), 'sources': sources(),
        'environment': environment(), 'manifest_source_audit': frozen_manifest.source_audit(frozen)}, 'identity_sha256')


def verify(args, expected, *, full=False):
    storage.verify_seal(expected, 'identity_sha256')
    if (expected['schema'] != config.SCHEMA or expected['protocol'] != config.PROTOCOL
            or expected['world_size'] != args.world_size or expected['sources'] != sources()
            or expected['manifest_path'] != str(args.manifest) or expected['model_dir'] != str(args.model_dir)
            or expected['checkpoint'] != str(args.checkpoint)):
        raise ValueError('run topology/code/model/protocol/path identity mismatch')
    frozen = frozen_manifest.load(args.manifest, full=full)
    if frozen['manifest_sha256'] != expected['manifest_sha256']:
        raise ValueError('frozen manifest changed')
    if expected['manifest_source_audit'] != frozen_manifest.source_audit(frozen):
        raise ValueError('manifest compatibility decision changed')
    if expected['environment'] != environment():
        raise ValueError('runtime dependency versions changed')
    from mizar_official_eval import runtime as official
    official.verify_bundle_unchanged(expected['official_bundle'], args.model_dir, args.checkpoint)
    if full and official.audit_bundle(args.model_dir, args.checkpoint) != expected['official_bundle']:
        raise ValueError('release changed during evaluation')
    return frozen
