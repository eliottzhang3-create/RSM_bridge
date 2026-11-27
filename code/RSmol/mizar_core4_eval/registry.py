"""The three requested final AudioMCQ x4 models; only three benchmarks."""
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
REMOTE = Path('/hpc_stor03/sjtu_home/jinwei.zhang')
MODEL_DIR = REMOTE/'models/Mizar159M'
BENCHMARKS = ('mmau', 'mmar', 'mmsu')
ROWS = dict(zip(BENCHMARKS, (1000, 1000, 5000)))
TRAINING_GEOMETRY = (8, 4, 4)
EXPECTED_STEP = 16890
SOURCE_CACHE = ROOT/'log/mizar_core4_eval_sources'
OUTPUT_NAMESPACE = 'mizar_core4_audiomcq_eval_20261127_v1'
ROUTES = {}
for name, start in (('l9_12x4', 8), ('l14_17x4', 13), ('l19_22x4', 18)):
    stem = name[:-2]
    ROUTES[name] = dict(
        route=f'mizar_final_{name}_audiomcq_gbs128_3epochs_v1',
        source_relative=f'mizar_{stem}x3_strong_reason',
        core=tuple(range(start, start+4)),
        schedule=tuple(range(start))+tuple(range(start, start+4))*4+tuple(range(start+4, 30)),
        state_count=454, gate=False, prompt_padding_visible=True)


def variant_spec(variant):
    if variant not in ROUTES:
        raise ValueError(f'unsupported AudioMCQ x4 target: {variant!r}')
    result = dict(ROUTES[variant])
    result['revision'] = json.loads((PACKAGE/'source_manifest.json').read_text(encoding='utf-8'))[variant]['revision']
    return result


def targets():
    records = json.loads((PACKAGE/'targets.json').read_text(encoding='utf-8'))
    if (set(records) != set(ROUTES) or any(not isinstance(p, str) or not p.startswith('/')
            or not p.endswith('/formal/checkpoints/model--step-16890.ckpt') for p in records.values())):
        raise ValueError('fixed three-checkpoint target list differs')
    return {name:Path(path) for name,path in records.items()}
