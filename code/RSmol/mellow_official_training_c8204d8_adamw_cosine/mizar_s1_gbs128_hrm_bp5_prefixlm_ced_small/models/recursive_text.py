"""Strict raw HRM text inheritance into the isolated PrefixLM class."""
from __future__ import annotations
import json
from pathlib import Path

from contract import recursive_checkpoint_contract


def load_recursive_text_model(model_path, *, production=True):
    import torch
    import transformers
    from transformers import AutoConfig
    from text_hrm_h2l3.conversion import read_weights
    from text_hrm_h2l3.model import strict_load
    from text_hrm_h2l3.contract import verify_inventory
    from models.hrm_prefix import AudioPrefixHRMForCausalLM
    if transformers.__version__!='4.54.1':
        raise RuntimeError('HRM requires transformers==4.54.1')
    path=Path(model_path).resolve(strict=True)
    metadata=recursive_checkpoint_contract(path,production=production)
    marker=json.loads((path/'checkpoint_complete.json').read_text(encoding='utf-8'))
    # Verify the root raw files; EMA and text training state are not initialized
    # into the audio model. No pickle/training_state deserialization occurs.
    inventory={n:v for n,v in marker['files'].items() if '/' not in n and n!='training_state.pt'}
    verify_inventory(path,inventory)
    config=AutoConfig.from_pretrained(path,local_files_only=True)
    config._attn_implementation='sdpa'
    model=AudioPrefixHRMForCausalLM(config).float()
    state=read_weights(path,require_float32=True)
    strict_load(model,state)
    if any(not torch.equal(value,model.state_dict()[name]) for name,value in state.items()):
        raise ValueError('raw text tensor inheritance differs after strict load')
    if len({id(layer) for layer in model.model.layers})!=20:
        raise ValueError('HRM physical layers unexpectedly aliased')
    count=sum(p.numel() for p in model.parameters())
    if production and count!=102420288:
        raise ValueError(f'HRM unique parameter count differs: {count}')
    model.audio_initialization_audit=dict(source_checkpoint=metadata,raw_files_verified=True,
        all_persisted_tensors_equal=True,unique_parameters=count,physical_layers=20,logical_calls=50,
        trained_gates_preserved=True,fixed_initial_state_preserved=True,embedding_head_tied=True,
        audio_optimizer_start_step=0,restore_text_training_state=False,mapper_initialization='new_random')
    return model
