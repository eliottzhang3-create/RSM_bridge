"""Explicit user-selected checkpoints; no latest-checkpoint discovery."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path('/hpc_stor03/sjtu_home/jinwei.zhang')
TRAINING = ROOT / 'mellow_official_training_c8204d8_adamw_cosine'
DATA = REMOTE / 'data'
TARGETS = {}
for alias, variant, stamp, directory in (
    ('base', 'base', '20261026_193830617147305',
     'mizar_stage1_fixed20_gbs256_3epochs_ced_small'),
    ('recursive_5_10_5', 'recursive_5_10_5', '20261028_160823559722647',
     'mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small'),
    ('serial84', 'recursive_5_10_5_layer8_4_inject_audio_reinject', '20261102_112121702891801',
     'mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject'),
    ('serial48_lora_r8', 'recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8', '20261104_131843931946322',
     'mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8'),
    ('serial84_lora_r8', 'recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8', '20261103_165410438279511',
     'mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8'),
    ('recursive_mesh', 'recursive_mesh', '20261027_081123150934256',
     'mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small'),
    ('mesh48_lora_r8', 'recursive_mesh_layer4_8_inject_loop_lora_r8', '20261104_161727489909082',
     'mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject_loop_lora_r8'),
):
    output = 'mizar_s1_ced_small_gbs256' + ('' if variant == 'base' else '_' + variant) + '_5090'
    TARGETS[alias] = {'eval_variant': variant, 'run_root': REMOTE / 'outputs/RSmol' / output / ('formal_' + stamp),
                      'route_root': TRAINING / directory}

VARIANTS = tuple(spec['eval_variant'] for spec in TARGETS.values())
OUTPUT_NAME = 'reasonaqa_close_ended_step{step}_v1'
