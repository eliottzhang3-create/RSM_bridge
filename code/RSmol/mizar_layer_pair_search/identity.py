"""Pin code/data/release separately from mutable progress and summaries."""
from . import config, execution, manifest, storage


def verify_guard():
    reference = storage.read_json(config.PACKAGE / 'reference_guard.json')
    for path, digest in reference['sources_lf_sha256'].items():
        if storage.text_sha(config.ROOT / path) != digest:
            raise ValueError(f'reused source differs from isolated reference snapshot: {path}')
    return reference


def sources():
    paths = [*config.PACKAGE.glob('*.py'), *config.PACKAGE.glob('*.sh'),
             config.PACKAGE / 'reference_guard.json', config.ROOT / 'run_mizar_layer_pair_search_8gpu_3090.sh',
             config.ROOT / 'run_mizar_layer_pair_search_1gpu_4090.sh']
    paths.extend(config.ROOT / name for name in verify_guard()['sources_lf_sha256'])
    return {p.relative_to(config.ROOT).as_posix(): storage.text_sha(p) for p in sorted(set(paths))}


def build(args, frozen, world_size=8):
    from mizar_official_eval import runtime as official
    bundle = official.audit_bundle(args.model_dir, args.checkpoint)
    return storage.seal({'schema': config.SCHEMA, 'protocol': execution.protocol(world_size),
                         'manifest_path': str(args.manifest), 'manifest_sha256': frozen['manifest_sha256'],
                         'model_dir': str(args.model_dir), 'checkpoint': str(args.checkpoint),
                         'official_bundle': bundle, 'sources': sources()}, 'identity_sha256')


def verify(args, expected, *, full_manifest=False, world_size=8):
    storage.verify_seal(expected, 'identity_sha256')
    if (expected['schema'] != config.SCHEMA or expected['protocol'] != execution.protocol(world_size)
            or expected['manifest_path'] != str(args.manifest) or expected['model_dir'] != str(args.model_dir)
            or expected['checkpoint'] != str(args.checkpoint) or expected['sources'] != sources()):
        raise ValueError('evaluation code/protocol/path identity mismatch')
    frozen = manifest.load(args.manifest, full=full_manifest)
    if expected['manifest_sha256'] != frozen['manifest_sha256']:
        raise ValueError('manifest changed after run identity was created')
    from mizar_official_eval import runtime as official
    official.verify_bundle_unchanged(expected['official_bundle'], args.model_dir, args.checkpoint)
    if full_manifest and official.audit_bundle(args.model_dir, args.checkpoint) != expected['official_bundle']:
        raise ValueError('official release changed during evaluation')
    return frozen
