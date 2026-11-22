"""The seven explicit user targets; architecture and source revision are separate."""
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
REMOTE = Path('/hpc_stor03/sjtu_home/jinwei.zhang')
BENCHMARKS = ('reasonaqa5000', 'mmau', 'mmar', 'mmsu')
ROWS = dict(zip(BENCHMARKS, (5000,1000,1000,5000)))
TRAINING_GEOMETRY = (8,4,4)
SCHEDULE = tuple(range(7)) + tuple(range(7,21))*2 + tuple(range(21,28))
TEXT_CONTRACT = 'mizar_final_text_7_14x2_7_mesh_5slot_v1'
TEXT_ARCHITECTURE = 'logical_42_physical_28_7_14x2_7_mesh_5slot_3router_v1'
OLD_REVISION = '8a9f1f76722e77cba4be429a578c63fdcf6ee527'
NEW_REVISION = '75dcda3a7bcd8765044b14ecd4e408fe33dfdda6'
SOURCE_CACHE = ROOT / 'log/ced_eval_20261122_sources'
OUTPUT_NAMESPACE = 'ced_eval_20261122_v1'
ROUTES = {}
for target, variant, suffix, implementation, key, constant, count, routers, revision in (
    ('mesh_original','base','ced_small','recursive_model_mizar_text_7_14x2_7_mesh',None,None,0,6,OLD_REVISION),
    ('audio7_read2','audio7_read2','ced48_audio7_read2_lora_r8','models.recursive_audio7_read2','audio_read2','AUDIO_READ2_CONTRACT',4731278,8,OLD_REVISION),
    ('listen','listen','ced48_listen_lora_r8','models.recursive_mesh_listen_lora','audio_listen','LISTEN_CONTRACT',4730124,8,OLD_REVISION),
    ('loop_lora','loop_lora','ced48_loop_lora_r8','models.recursive_layer_inject','layer_injection','INJECTION_CONTRACT',4724354,6,OLD_REVISION),
    ('wfuse','wfuse','ced48_wfuse_lora_r8','models.recursive_mesh_wfuse_lora','audio_fusion','FUSION_CONTRACT',6050304,6,OLD_REVISION),
    ('mesh_reinit','base','ced_small','recursive_model_mizar_text_7_14x2_7_mesh',None,None,0,6,NEW_REVISION),
    ('smol_gbs128','plain',None,None,None,None,0,0,NEW_REVISION),
):
    plain = variant == 'plain'
    folder = ('mizar_stage1_fixed20_gbs128_3epochs_ced_small' if plain else
              'mizar_s1_gbs128_text71427_mesh_' + suffix)
    relative = Path('code/RSmol/mellow_official_training_c8204d8_adamw_cosine') / folder
    ROUTES[target] = {
        'model_variant': variant, 'plain':plain, 'revision':revision,
        'source_relative':relative.as_posix(),
        'route_root':SOURCE_CACHE/revision/relative,
        'route':'ced_small_mizar_s1_fixed20_gbs128_v1' if plain else
            'ced_small_mizar_s1_fixed20_gbs128_text71427_mesh'+('' if variant=='base' else '_'+suffix)+'_v1',
        'implementation':implementation,'branch_key':key,'branch_constant':constant,
        'new_parameters':count,'router_modules':routers,
        'context_kind':'none' if variant in ('plain','base') else 'pair',
    }


def variant_spec(variant):
    try:
        return ROUTES[variant]
    except KeyError:
        raise ValueError(f'unsupported 20261122 target: {variant!r}') from None
