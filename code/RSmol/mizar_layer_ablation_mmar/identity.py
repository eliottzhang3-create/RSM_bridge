"""Model/data/code/environment identity isolated from all prior experiments."""
from importlib.metadata import version
from mizar_layer_pair_search import storage
from . import config, dataset


def sources():
    guard = storage.read_json(config.PACKAGE/'reference_guard.json')['sources_lf_sha256']
    for name, digest in guard.items():
        if storage.text_sha(config.ROOT/name) != digest:
            raise ValueError(f'reused source changed: {name}')
    paths = [*config.PACKAGE.glob('*.py'), *config.PACKAGE.glob('*.sh'),
             config.PACKAGE/'reference_guard.json', config.ROOT/'run_mizar_layer_ablation_mmar_8gpu_5090.sh',
             *(config.ROOT/name for name in guard)]
    return {p.relative_to(config.ROOT).as_posix(): storage.text_sha(p) for p in sorted(set(paths))}


def environment():
    return {name: version(name) for name in ('torch', 'torchaudio', 'transformers', 'numpy', 'soundfile')}


def build(args, frozen, protocol):
    from mizar_official_eval import runtime as official
    return storage.seal({
        'task': config.SCHEMA, 'protocol': protocol, 'manifest_path': str(args.manifest),
        'manifest_sha256': frozen['manifest_sha256'], 'model_dir': str(args.model_dir),
        'checkpoint': str(args.checkpoint), 'sources': sources(), 'environment': environment(),
        'official_bundle': official.audit_bundle(args.model_dir, args.checkpoint),
        'manifest_source_audit': frozen['generator_sources'],
    }, 'identity_sha256')


def verify(args, expected, protocol, full=False):
    storage.verify_seal(expected, 'identity_sha256')
    if (expected['task'] != config.SCHEMA or expected['protocol'] != protocol or expected['sources'] != sources()
            or expected['environment'] != environment() or expected['manifest_path'] != str(args.manifest)
            or expected['model_dir'] != str(args.model_dir) or expected['checkpoint'] != str(args.checkpoint)):
        raise ValueError('MMAR task/code/environment/model/input identity mismatch')
    frozen = dataset.load(args.manifest, full=full)
    if (frozen['manifest_sha256'] != expected['manifest_sha256']
            or frozen['generator_sources'] != expected['manifest_source_audit']):
        raise ValueError('frozen MMAR input changed')
    from mizar_official_eval import runtime as official
    official.verify_bundle_unchanged(expected['official_bundle'], args.model_dir, args.checkpoint)
    if full and official.audit_bundle(args.model_dir, args.checkpoint) != expected['official_bundle']:
        raise ValueError('official model assets changed during evaluation')
    return frozen
