"""Independent task identities; changing one task never changes the other's sources."""
from importlib.metadata import version
from pathlib import Path
from mizar_layer_pair_search import storage
from . import config


def sources(task):
    guard=storage.read_json(config.PACKAGE/'reference_guard.json')['sources_lf_sha256']
    for name,digest in guard.items():
        if storage.text_sha(config.ROOT/name)!=digest: raise ValueError(f'reused source changed: {name}')
    task_folder=config.ROOT/('mizar_layer_cka' if task=='cka' else 'mizar_layer_ablation')
    paths=[*config.PACKAGE.glob('*.py'),*config.PACKAGE.glob('*.sh'),config.PACKAGE/'reference_guard.json',
           *task_folder.glob('*.py'),*task_folder.glob('*.sh'),config.ROOT/f'run_mizar_layer_{task}_8gpu_3090.sh',
           *(config.ROOT/name for name in guard)]
    return {p.relative_to(config.ROOT).as_posix():storage.text_sha(p) for p in sorted(set(paths))}


def environment():
    return {name:version(name) for name in ('torch','torchaudio','transformers','numpy','soundfile','pyarrow')}


def build(args,frozen,task,protocol):
    from mizar_official_eval import runtime as official
    from mizar_audio_cutoff.frozen_manifest import source_audit
    return storage.seal({'task':task,'protocol':protocol,'manifest_path':str(args.manifest),
        'manifest_sha256':frozen['manifest_sha256'],'model_dir':str(args.model_dir),'checkpoint':str(args.checkpoint),
        'sources':sources(task),'environment':environment(),'official_bundle':official.audit_bundle(args.model_dir,args.checkpoint),
        'manifest_source_audit':source_audit(frozen) if task=='cka' else frozen['generator_sources']},'identity_sha256')


def verify(args,expected,task,protocol,full=False):
    storage.verify_seal(expected,'identity_sha256')
    if (expected['task']!=task or expected['protocol']!=protocol or expected['sources']!=sources(task)
            or expected['environment']!=environment() or expected['manifest_path']!=str(args.manifest)
            or expected['model_dir']!=str(args.model_dir) or expected['checkpoint']!=str(args.checkpoint)):
        raise ValueError('task/code/environment/model/input identity mismatch')
    if task=='cka':
        from mizar_audio_cutoff.frozen_manifest import load,source_audit
    else:
        from mizar_layer_ablation.dataset import load
    frozen=load(args.manifest,full=full)
    if frozen['manifest_sha256']!=expected['manifest_sha256']: raise ValueError('manifest changed')
    if task=='cka' and source_audit(frozen)!=expected['manifest_source_audit']: raise ValueError('compatibility decision changed')
    from mizar_official_eval import runtime as official
    official.verify_bundle_unchanged(expected['official_bundle'],args.model_dir,args.checkpoint)
    if full and official.audit_bundle(args.model_dir,args.checkpoint)!=expected['official_bundle']:
        raise ValueError('official assets changed during evaluation')
    return frozen
