import argparse
from pathlib import Path
from . import config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='ReasonAQA silence causal tracing, original Mizar, 8x5090')
    for name, default in {
        'model-dir': config.MODEL_DIR, 'checkpoint': None, 'source-manifest': config.SOURCE,
        'pool-manifest': config.POOL, 'test-json': config.DATA/'reasonaqa/test.json',
        'audiocaps-root': config.DATA/'audiocaps_v2', 'clotho-root': config.DATA/'clotho_v2_1',
        'clotho-aqa-root': config.DATA/'clotho_aqa_audio/audio_files', 'output-dir': config.OUTPUT,
    }.items():
        p.add_argument('--'+name, type=Path, default=default)
    p.add_argument('--stage', choices=('all', 'select'), default='all', help='select: stop after qualifying manifest')
    p.add_argument('--check-args', action='store_true', help='no files/model/CUDA/vc accessed')
    a = p.parse_args(argv)
    a.checkpoint = a.checkpoint or a.model_dir/config.CHECKPOINT
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.expanduser().resolve())
    if a.source_manifest == a.pool_manifest or a.source_manifest.parent == a.pool_manifest.parent:
        p.error('pool manifest requires its own directory; original 5000 remains read-only')
    if any(x == a.output_dir or x.is_relative_to(a.output_dir) for x in
           (a.source_manifest, a.pool_manifest, a.test_json, a.model_dir)):
        p.error('inputs must be outside the output directory')
    return a


def forwarded(a):
    return [*[s for k, v in vars(a).items() if isinstance(v, Path)
              for s in ('--'+k.replace('_', '-'), str(v))], '--stage', a.stage]
