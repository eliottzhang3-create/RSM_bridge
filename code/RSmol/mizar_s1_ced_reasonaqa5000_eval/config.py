"""Frozen scope: MMAU's 38 routes minus official_mizar and the previous seven."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
REMOTE = Path('/hpc_stor03/sjtu_home/jinwei.zhang')
DATA = REMOTE / 'data'
SAMPLE_SIZE, SAMPLE_SEED = 5000, 1234
POOL_COUNTS = {'audiocaps_mcq': 23672, 'clotho_mcq': 26031, 'clotho_aqa_yesno': 5676}
DEFAULT_BUNDLE = DATA / 'reasonaqa/test_close_ended_sample5000_seed1234_v1'
DEFAULT_MANIFEST = DEFAULT_BUNDLE / 'manifest.json'
EXCLUDED = ('official_mizar', 'base', 'recursive_5_10_5', 'recursive_mesh',
            'recursive_5_10_5_layer8_4_inject_audio_reinject',
            'recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8',
            'recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8',
            'recursive_mesh_layer4_8_inject_loop_lora_r8')
TARGETS = json.loads((PACKAGE / 'targets.json').read_text(encoding='utf-8'))
for _spec in TARGETS.values():
    _spec['route_root'] = ROOT / _spec['route_root']
    _spec['run_root'] = Path(_spec['run_root']) if _spec['run_root'] else None
    _spec['expected_geometry'] = tuple(_spec['expected_geometry'])
VARIANT_TARGETS = {spec['eval_variant']: name for name, spec in TARGETS.items()}
OUTPUT_NAME = 'reasonaqa_close_ended_sample5000_seed1234_step{step}_v1'


def target_name(value):
    if value in TARGETS:
        return value
    if value in VARIANT_TARGETS:
        return VARIANT_TARGETS[value]
    raise ValueError(f'unsupported/excluded ReasonAQA5000 target: {value}')
