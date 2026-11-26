"""Four families; training checkpoint paths are always explicit."""
from pathlib import Path
PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
REMOTE = Path('/hpc_stor03/sjtu_home/jinwei.zhang')
MODEL_DIR = REMOTE/'models/Mizar159M'
BENCHMARKS = ('reasonaqa5000','mmau','mmar','mmsu')
ROWS = dict(zip(BENCHMARKS,(5000,1000,1000,5000)))
TRAINING_GEOMETRY = (8,4,4)
SOURCE_CACHE = ROOT/'log/mizar_core3_eval_sources'
OUTPUT_NAMESPACE = 'mizar_core3_eval_v1'
ROUTES = {
 'official_mizar': dict(route='official_mizar_stage3_q4_seed20260905',source_relative='mizar_official_eval',schedule=tuple(range(30)),state_count=454,gate=False,core=()),
 'l9_12x3': dict(route='mizar_final_l9_12x3_strong_reason_gbs128_v2_previous_training',source_relative='mizar_l9_12x3_strong_reason',schedule=tuple(range(8))+tuple(range(8,12))*3+tuple(range(12,30)),state_count=456,gate=True,core=tuple(range(8,12))),
 'l14_17x3': dict(route='mizar_final_l14_17x3_strong_reason_gbs128_v2_previous_training',source_relative='mizar_l14_17x3_strong_reason',schedule=tuple(range(13))+tuple(range(13,17))*3+tuple(range(17,30)),state_count=454,gate=False,core=tuple(range(13,17))),
 'l19_22x3': dict(route='mizar_final_l19_22x3_strong_reason_gbs128_no_injection_v1',source_relative='mizar_l19_22x3_strong_reason',schedule=tuple(range(18))+tuple(range(18,22))*3+tuple(range(22,30)),state_count=454,gate=False,core=tuple(range(18,22))),
}


def variant_spec(variant):
    if variant not in ROUTES:
        raise ValueError(f'unsupported Mizar target: {variant!r}')
    result = dict(ROUTES[variant])
    import json
    result['prompt_padding_visible'] = True
    result['revision'] = ('official_release' if variant == 'official_mizar' else json.loads((PACKAGE/'source_manifest.json').read_text())[variant]['code_sha256'])
    return result
