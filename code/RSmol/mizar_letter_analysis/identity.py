"""New experiment identities pin unchanged reused implementations and inputs."""
from importlib.metadata import version
from mizar_layer_pair_search import storage
from . import config, manifest


def sources():
    guard=storage.read_json(config.PACKAGE/'reference_guard.json')['sources_lf_sha256']
    for name,digest in guard.items():
        if storage.text_sha(config.ROOT/name)!=digest:raise ValueError(f'reused source changed: {name}')
    paths=[*config.PACKAGE.glob('*.py'),*config.PACKAGE.glob('*.sh'),config.PACKAGE/'mmau_mapping.json',
           config.PACKAGE/'reference_guard.json',*(config.ROOT/name for name in guard)]
    for folder in ('mizar_audio_cutoff1000','mizar_layer_analysis_common','mizar_layer_ablation','mizar_layer_ablation_mmar'):
        paths.extend((config.ROOT/folder).glob('*.py'))
    paths.extend(config.ROOT/name for name in ('run_mizar_audio_cutoff1000_letters_eot_8gpu_5090.sh',
        'run_mizar_layer_ablation_mmau_letters_eot_8gpu_5090.sh','run_mizar_layer_ablation_mmar_letters_eot_8gpu_5090.sh'))
    return {p.relative_to(config.ROOT).as_posix():storage.text_sha(p) for p in sorted(set(paths))}


def environment():
    return {name:version(name) for name in ('torch','torchaudio','transformers','numpy','soundfile','pyarrow')}


def build(args, frozen):
    from mizar_official_eval import runtime as official
    return storage.seal({'schema':config.SCHEMA,'task':args.task,'protocol':config.protocol(args.task),
        'world_size':config.WORLD,'manifest_path':str(args.manifest),'manifest_sha256':frozen['manifest_sha256'],
        'source_manifest_path':str(args.source_manifest),'source_manifest_sha256':frozen['source_manifest_sha256'],
        'model_dir':str(args.model_dir),'checkpoint':str(args.checkpoint),'sources':sources(),
        'environment':environment(),'official_bundle':official.audit_bundle(args.model_dir,args.checkpoint)},'identity_sha256')


def verify(args, expected, full=False):
    storage.verify_seal(expected,'identity_sha256')
    if (expected['schema']!=config.SCHEMA or expected['task']!=args.task or expected['world_size']!=config.WORLD
            or expected['protocol']!=config.protocol(args.task) or expected['sources']!=sources()
            or expected['environment']!=environment() or expected['manifest_path']!=str(args.manifest)
            or expected['source_manifest_path']!=str(args.source_manifest) or expected['model_dir']!=str(args.model_dir)
            or expected['checkpoint']!=str(args.checkpoint)):
        raise ValueError('letter experiment/model/code/environment/input identity mismatch')
    frozen=manifest.load(args.manifest,full=full)
    if (frozen['manifest_sha256']!=expected['manifest_sha256'] or frozen['task']!=args.task
            or frozen['source_manifest_sha256']!=expected['source_manifest_sha256']):
        raise ValueError('letter/source manifest changed')
    from mizar_official_eval import runtime as official
    official.verify_bundle_unchanged(expected['official_bundle'],args.model_dir,args.checkpoint)
    if full and official.audit_bundle(args.model_dir,args.checkpoint)!=expected['official_bundle']:
        raise ValueError('official model assets changed during the experiment')
    return frozen
