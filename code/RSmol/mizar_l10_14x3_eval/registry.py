"""One explicitly requested checkpoint and the original AU/AR protocols."""
import json
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parent
REMOTE=Path('/hpc_stor03/sjtu_home/jinwei.zhang')
MODEL_DIR=REMOTE/'models/Mizar159M'
TARGET='l10_14x3_a9gate_bias3_l15'
BENCHMARKS=('mmau','mmar')
EXPECTED_STEP=2542
TRAINING_GEOMETRY=(8,4,4)
SOURCE_CACHE=ROOT/'log/mizar_l10_14x3_eval_sources'
OUTPUT_NAMESPACE='mizar_l10_14x3_a9gate_bias3_l15_eval_20261127_v1'
ROUTES={TARGET:dict(route='mizar_final_l10_14x3_a9_gate_bias3_l15_strong_gbs128_v1',
    source_relative='mizar_l9_12x3_strong_reason',state_count=457,
    schedule=tuple(range(9))+tuple(range(9,14))*3+tuple(range(14,30)),
    core=tuple(range(9,14)),trainable=tuple(range(9,15)),prompt_padding_visible=True)}


def variant_spec(variant):
    if variant not in ROUTES:
        raise ValueError('unsupported L10-14 evaluation model: '+str(variant))
    result=dict(ROUTES[variant])
    result['revision']=json.loads((PACKAGE/'source_manifest.json').read_text(encoding='utf-8'))[variant]['revision']
    return result


def checkpoint_path():
    targets=json.loads((PACKAGE/'target.json').read_text(encoding='utf-8'))
    if set(targets)!={TARGET} or not targets[TARGET].endswith('/formal/checkpoints/model--step-2542.ckpt'):
        raise ValueError('explicit checkpoint binding differs')
    return Path(targets[TARGET])
