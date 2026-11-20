"""Six fixed implementations; checkpoints and training runs have no defaults."""
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
REMOTE = Path('/hpc_stor03/sjtu_home/jinwei.zhang')
TRAINING_ROOT = ROOT / 'mellow_official_training_c8204d8_adamw_cosine'
SCHEDULE = tuple(range(7)) + tuple(range(7, 21)) * 2 + tuple(range(21, 28))
TEXT_CONTRACT = 'mizar_final_text_7_14x2_7_mesh_5slot_v1'
TEXT_ARCHITECTURE = 'logical_42_physical_28_7_14x2_7_mesh_5slot_3router_v1'
BENCHMARKS = ('reasonaqa5000', 'mmau', 'mmar', 'mmsu')
ROWS = dict(zip(BENCHMARKS, (5000, 1000, 1000, 5000)))
TRAINING_GEOMETRY = (8, 4, 4)
ROUTES = {}
for alias, suffix, implementation, key, constant, count, routers in (
    ('base', 'ced_small', 'recursive_model_mizar_text_7_14x2_7_mesh', None, None, 0, 6),
    ('loop_lora', 'ced48_loop_lora_r8', 'models.recursive_layer_inject', 'layer_injection', 'INJECTION_CONTRACT', 4724354, 6),
    ('listen', 'ced48_listen_lora_r8', 'models.recursive_mesh_listen_lora', 'audio_listen', 'LISTEN_CONTRACT', 4730124, 8),
    ('audio7_read2', 'ced48_audio7_read2_lora_r8', 'models.recursive_audio7_read2', 'audio_read2', 'AUDIO_READ2_CONTRACT', 4731278, 8),
    ('wfuse', 'ced48_wfuse_lora_r8', 'models.recursive_mesh_wfuse_lora', 'audio_fusion', 'FUSION_CONTRACT', 6050304, 6),
    ('audio7_read2_ar6', 'ced48_audio7_read2_ar6_lora_r8', 'models.recursive_audio7_read2_ar6', 'audio_read2', 'AUDIO_READ2_CONTRACT', 4742818, 12),
):
    ROUTES[alias] = {
        'route_root': TRAINING_ROOT / ('mizar_s1_gbs128_text71427_mesh_' + suffix),
        'route': 'ced_small_mizar_s1_fixed20_gbs128_text71427_mesh' + ('' if alias == 'base' else '_' + suffix) + '_v1',
        'implementation': implementation, 'branch_key': key, 'branch_constant': constant,
        'new_parameters': count, 'router_modules': routers,
        'context_kind': 'none' if alias == 'base' else 'pair',
    }


def variant_spec(variant):
    try:
        return ROUTES[variant]
    except KeyError:
        raise ValueError(f'unsupported text71427 CED model: {variant!r}') from None
