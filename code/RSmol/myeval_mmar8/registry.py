"""Fixed user-approved models, permutations and isolated output identities."""
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
REMOTE = Path('/hpc_stor03/sjtu_home/jinwei.zhang')
REVISION = '8a9f1f76722e77cba4be429a578c63fdcf6ee527'
SOURCE_CACHE = ROOT / 'log/myeval_mmar8_sources'
OUTPUT_NAMESPACE = 'myeval_mmar8_v1'
ANNOTATIONS = REMOTE / 'data/MyEval/mmar8selected.jsonl'
ANNOTATIONS_SHA256 = 'a670c97eef24859c6dc015ba137e723fa2e7205d198a0b4d234ced0e642b7815'
SCHEDULE = tuple(range(7)) + tuple(range(7, 21)) * 2 + tuple(range(21, 28))
TEXT_CONTRACT = 'mizar_final_text_7_14x2_7_mesh_5slot_v1'
TEXT_ARCHITECTURE = 'logical_42_physical_28_7_14x2_7_mesh_5slot_3router_v1'
TRAINING_GEOMETRY = (8, 4, 4)
REQUIRED_CORRECT = {2: 2, 3: 6, 4: 21, 5: 115}
DIMENSIONS = {'P': 'Perception', 'S': 'Semantic Understanding',
              'W': 'World Knowledge', 'R': 'Reasoning'}
BASE_RUN = REMOTE / 'outputs/RSmol/mizar_s1_ced_small_gbs128_text71427_mesh_3090/formal_20261119_141531507533438'
READ2_RUN = REMOTE / 'outputs/RSmol/mizar_s1_ced48_audio7_read2_lora_r8_gbs128_text71427_3090/formal_20261120_111448563391250'
ROUTES = {}
for alias in ('base', 'read2', 'read2_zero48'):
    base = alias == 'base'
    suffix = 'ced_small' if base else 'ced48_audio7_read2_lora_r8'
    relative = Path('code/RSmol/mellow_official_training_c8204d8_adamw_cosine') / ('mizar_s1_gbs128_text71427_mesh_' + suffix)
    run = BASE_RUN if base else READ2_RUN
    ROUTES[alias] = {
        'revision': REVISION, 'source_relative': relative.as_posix(),
        'route_root': SOURCE_CACHE / REVISION / relative,
        'route': 'ced_small_mizar_s1_fixed20_gbs128_text71427_mesh' + ('' if base else '_' + suffix) + '_v1',
        'model_variant': 'base' if base else 'audio7_read2', 'plain': False,
        'implementation': 'recursive_model_mizar_text_7_14x2_7_mesh' if base else 'models.recursive_audio7_read2',
        'branch_key': None if base else 'audio_read2',
        'branch_constant': None if base else 'AUDIO_READ2_CONTRACT',
        'new_parameters': 0 if base else 4731278, 'router_modules': 6 if base else 8,
        'context_kind': 'none' if base else 'pair', 'zero48': alias == 'read2_zero48',
        'checkpoint': run / 'formal/checkpoints/model--step-34626.ckpt',
        'runtime_config': run / 'formal/runtime.json', 'run_root': run,
    }


def variant_spec(variant):
    try:
        return ROUTES[variant]
    except KeyError:
        raise ValueError(f'unsupported MyEval model: {variant!r}') from None


def intervention(variant):
    return {'name': 'zero_ced4_ced8_readonly_seeds' if variant_spec(variant)['zero48'] else 'none',
            'seed_slots': [5, 6] if variant_spec(variant)['zero48'] else [],
            'weights_modified': False, 'lora_enabled': variant != 'base',
            'seven_way_softmax_preserved': variant != 'base', 'renormalize': False}
