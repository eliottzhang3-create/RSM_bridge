"""Two explicit checkpoints; original training code and architecture stay pinned."""
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parent
REMOTE=Path('/hpc_stor03/sjtu_home/jinwei.zhang')
BENCHMARKS=('reasonaqa5000','mmau','mmar','mmsu')
ROWS=dict(zip(BENCHMARKS,(5000,1000,1000,5000)))
TRAINING_GEOMETRY=(8,4,4)
SOURCE_CACHE=ROOT/'log/ced_eval_20261123_sources'
OUTPUT_NAMESPACE='ced_eval_20261123_v1'
ROUTES={}
for target,loops,slots,groups,revision,folder in (
    ('mesh_71437',3,6,4,'75dcda3a7bcd8765044b14ecd4e408fe33dfdda6','mizar_s1_gbs128_text71437_mesh_ced_small'),
    ('audio7_read2',2,5,3,'8a9f1f76722e77cba4be429a578c63fdcf6ee527','mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_lora_r8'),
):
    plain=target=='mesh_71437'
    stem=f'7_14x{loops}_7'
    relative=Path('code/RSmol/mellow_official_training_c8204d8_adamw_cosine')/folder
    ROUTES[target]={
        'revision':revision,'source_relative':relative.as_posix(),
        'route_root':SOURCE_CACHE/revision/relative,'plain':False,
        'route':('ced_small_mizar_s1_fixed20_gbs128_text71437_mesh_v1' if plain else
                 'ced_small_mizar_s1_fixed20_gbs128_text71427_mesh_ced48_audio7_read2_lora_r8_v1'),
        'model_variant':'base' if plain else 'audio7_read2',
        'implementation':f'recursive_model_mizar_text_{stem}_mesh' if plain else 'models.recursive_audio7_read2',
        'model_class':f'MizarText7_14x{loops}_7MeshForCausalLM',
        'text_package':f'mizar_text_{stem}_mesh',
        'text_contract':f'mizar_final_text_{stem}_mesh_{slots}slot_v1',
        'text_architecture':f'logical_{14+14*loops}_physical_28_{stem}_mesh_{slots}slot_{groups}router_v1',
        'architecture':f'mesh_{stem}','schedule':tuple(range(7))+tuple(range(7,21))*loops+tuple(range(21,28)),
        'loops':loops,'slots':slots,'router_groups':groups,
        'branch_key':None if plain else 'audio_read2',
        'branch_constant':None if plain else 'AUDIO_READ2_CONTRACT',
        'new_parameters':0 if plain else 4731278,'router_modules':8,
        'context_kind':'none' if plain else 'pair',
    }

def variant_spec(variant):
    try:return ROUTES[variant]
    except KeyError:raise ValueError(f'unsupported 20261123 target: {variant!r}') from None
