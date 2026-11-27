from importlib.metadata import version
from mizar_layer_pair_search import storage
from . import config,manifest


def sources():
    guard=storage.read_json(config.PACKAGE/'reference_guard.json')['sources_lf_sha256']
    for name,digest in guard.items():
        if storage.text_sha(config.ROOT/name)!=digest:raise ValueError(f'reused source changed: {name}')
    return {**guard,**{p.relative_to(config.ROOT).as_posix():storage.text_sha(p) for p in
        [*config.PACKAGE.glob('*.py'),*config.PACKAGE.glob('*.sh'),*config.PACKAGE.glob('*.json'),
         config.ROOT/'run_mizar_mmau_triplet_ablation_8gpu_3090.sh',config.ROOT/'run_mizar_mmar_triplet_ablation_8gpu_3090.sh']}}


def environment():return {k:version(k) for k in ('torch','torchaudio','transformers','numpy','soundfile','pyarrow')}


def build(a,frozen):
    from mizar_official_eval import runtime as official
    return storage.seal({'schema':config.SCHEMA,'task':a.task,'protocol':config.protocol(a.task),'manifest':str(a.manifest),
        'manifest_sha256':frozen['manifest_sha256'],'model_dir':str(a.model_dir),'checkpoint':str(a.checkpoint),
        'sources':sources(),'environment':environment(),'official_bundle':official.audit_bundle(a.model_dir,a.checkpoint)},'identity_sha256')


def verify(a,expected,full=False):
    storage.verify_seal(expected,'identity_sha256')
    if (expected['schema']!=config.SCHEMA or expected['task']!=a.task or expected['protocol']!=config.protocol(a.task)
            or expected['sources']!=sources() or expected['environment']!=environment() or expected['manifest']!=str(a.manifest)
            or expected['model_dir']!=str(a.model_dir) or expected['checkpoint']!=str(a.checkpoint)):raise ValueError('triplet identity changed')
    frozen=manifest.load(a.manifest,full=full)
    if frozen['manifest_sha256']!=expected['manifest_sha256']:raise ValueError('full1000 input changed')
    from mizar_official_eval import runtime as official
    official.verify_bundle_unchanged(expected['official_bundle'],a.model_dir,a.checkpoint)
    if full and official.audit_bundle(a.model_dir,a.checkpoint)!=expected['official_bundle']:raise ValueError('model assets changed')
    return frozen
