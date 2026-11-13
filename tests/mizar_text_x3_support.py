"""Synthetic final text-checkpoint JSON fixture, never real production weights."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NAME='mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8'
ROUTE=ROOT/'code/RSmol/mellow_official_training_c8204d8_adamw_cosine'/NAME
SCHEDULE=[*range(5),*list(range(5,15))*3,*range(15,20)]
ARCH='logical_40_physical_20_5_10x3_5_pure'


def final_text_metadata(path):
    path=Path(path)
    metadata=dict(architecture_contract=ARCH,training_contract='text_pure_x3_bf16_full_epoch_v1',
                  optimizer_step=9244,memory_slots=0,router_groups=0,router_module_count=0,logical_to_physical=SCHEDULE)
    (path/'recursive_checkpoint_metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
    files=[p.name for p in path.iterdir() if p.is_file()]+['checkpoint_manifest.json','checkpoint_complete.json']
    marker=dict(status='complete',complete_marker=True,architecture_contract=ARCH,optimizer_step=9244,files=files)
    for name in ('checkpoint_manifest.json','checkpoint_complete.json'):
        (path/name).write_text(json.dumps(marker),encoding='utf-8')
