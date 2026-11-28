"""One native S3 model and two original generation benchmarks."""
import json
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parent
REMOTE=Path('/hpc_stor03/sjtu_home/jinwei.zhang')
MODEL_DIR=REMOTE/'models/Mizar159M'
TARGET='s3_full_halfepoch'
BENCHMARKS=('mmau','mmar')
EXPECTED_STEP=1066
TRAINING_GEOMETRY=(8,4,4)
SOURCE_CACHE=ROOT/'log/mizar_s3_full_finetune_eval_sources'
OUTPUT_NAMESPACE='mizar_s3_full_finetune_eval_20261128_v1'
ROUTES={TARGET:dict(route='mizar_final_s3_full_projector_lm_halfepoch_gbs128_v1',
    source_relative='mizar_s3_full_finetune',state_count=454,schedule=tuple(range(30)),prompt_padding_visible=True)}


def variant_spec(variant):
    if variant not in ROUTES:raise ValueError('unsupported S3 evaluation model: '+str(variant))
    result=dict(ROUTES[variant])
    result['revision']=json.loads((PACKAGE/'source_manifest.json').read_text(encoding='utf-8'))[variant]['revision']
    return result


def checkpoint_path():
    records=json.loads((PACKAGE/'target.json').read_text(encoding='utf-8'))
    if set(records)!={TARGET} or not records[TARGET].startswith('/') or not records[TARGET].endswith('/formal/checkpoints/model--step-1066.ckpt'):
        raise ValueError('selected S3 checkpoint binding differs')
    return Path(records[TARGET])
