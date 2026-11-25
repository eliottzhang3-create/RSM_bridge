"""The two explicit step34626 checkpoints; each native attention policy is pinned."""
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parent
REMOTE=Path('/hpc_stor03/sjtu_home/jinwei.zhang')
BENCHMARKS=('reasonaqa5000','mmau','mmar','mmsu')
ROWS=dict(zip(BENCHMARKS,(5000,1000,1000,5000)))
TRAINING_GEOMETRY=(8,4,4)
SOURCE_CACHE=ROOT/'log/ced_eval_20261125_sources'
OUTPUT_NAMESPACE='ced_eval_20261125_v1'
PREFIXLM_POLICY='prefixlm383_bidirectional_prefix_causal_answer_padding_keys_v1'
MESH_SCHEDULE=tuple(range(7))+tuple(range(7,21))*3+tuple(range(21,28))
HRM_SCHEDULE=tuple(range(5))+(tuple(range(5,10))*3+tuple(range(10,15)))*2+tuple(range(15,20))
ROUTES={
    'audio8_read3':dict(
        revision='2c7d4485979f30f23c96196533c6de5914c47888',
        source_relative='code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71437_mesh_ced48_audio8_read3_lora_r8',
        route='ced_small_mizar_s1_fixed20_gbs128_text71437_mesh_ced48_audio8_read3_lora_r8_v1',
        family='mesh',implementation='models.recursive_audio8_read3',model_class='MizarText7_14x3_7MeshForCausalLM',
        shared_packages=('mizar_text_7_14x3_7_mesh',),
        text_contract='mizar_final_text_7_14x3_7_mesh_6slot_v1',
        text_architecture='logical_56_physical_28_7_14x3_7_mesh_6slot_4router_v1',
        architecture='mesh_7_14x3_7_audio8_read3_lora_r8',schedule=MESH_SCHEDULE,physical_layers=28,
        slots=6,router_groups=4,router_modules=11,context_kind='pair',
        branch_key='audio_read3',branch_constant='AUDIO_READ3_CONTRACT',new_parameters=5876760,
        attention_mask=None,prompt_padding_visible=True),
    'hrm_prefixlm':dict(
        revision='ee6f54474139fa5b86a9189617a2d2c0b9c84492',
        source_relative='code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small',
        route='ced_small_mizar_s1_gbs128_hrm_bp5_prefixlm_v1',
        family='hrm',implementation='models.hrm_prefix',model_class='AudioPrefixHRMForCausalLM',
        shared_packages=('text_hrm_h2l3','text_hrm_reinit3600_bp5'),
        text_contract='hrm_text_reinit3600_bp5_seed1_cosine_v1',
        text_architecture='physical20_execution50_hrm_h2l3_magicnorm_gated_gqa_v1',
        architecture='hrm_h2l3_bp5_prefixlm383',schedule=HRM_SCHEDULE,physical_layers=20,
        router_modules=0,context_kind='prefixlm',branch_key=None,branch_constant=None,new_parameters=0,
        attention_mask=PREFIXLM_POLICY,prompt_padding_visible=False),
}
for spec in ROUTES.values():
    spec['route_root']=SOURCE_CACHE/spec['revision']/spec['source_relative']

def variant_spec(variant):
    try:return ROUTES[variant]
    except KeyError:raise ValueError(f'unsupported 20261125 target: {variant!r}') from None
