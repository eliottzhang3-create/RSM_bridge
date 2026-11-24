"""Named optimizer/model inventory and strict same-route full-state restore checks."""
from __future__ import annotations
import math
import torch
from contract import (ROUTE,EPOCHS,WORLD_SIZE,BATCH_SIZE,ACCUMULATION_STEPS,
    RANK_UPDATE_SAMPLES,ATTENTION_POLICY,digest,lr_factor)


def optimizer_parameter_inventory(model,optimizer):
    by_id={id(p):name for name,p in model.named_parameters() if p.requires_grad}
    result=[[dict(name=by_id[id(p)],shape=list(p.shape),dtype=str(p.dtype)) for p in g['params']]
        for g in optimizer.param_groups]
    flat=[item['name'] for g in result for item in g]
    if len(flat)!=len(set(flat)) or set(flat)!=set(by_id.values()):
        raise ValueError('optimizer must contain every unique trainable parameter exactly once')
    return result


def validate_resume_model_state(saved,model,optimizer,config):
    required={'state_dict','optimizer','optimizer_parameter_inventory','scheduler','grad_scaler','grad_norm_tracker',
        'random_state_by_rank','contract','contract_sha256','total_step','epoch','next_optimizer_step',
        'schema_version','route_variant','mode','num_epochs','loss_reduction','batch_geometry'}
    if not required<=saved.keys(): raise ValueError('incomplete full audio checkpoint')
    if (saved['schema_version']!=1 or saved['route_variant']!=ROUTE or saved['mode']!='formal'
        or saved['num_epochs']!=EPOCHS or saved['contract']!=config['contract']
        or saved['contract_sha256']!=config['contract_sha256'] or digest(saved['contract'])!=config['contract_sha256']):
        raise ValueError('resume requires same HRM PrefixLM source/data/assets/recipe/dependencies')
    contract=saved['contract'];geo=contract['geometry']
    if contract.get('attention_mask')!=ATTENTION_POLICY or contract.get('bp_steps')!=5:
        raise ValueError('resume attention/BP policy mismatch')
    expected_geometry=dict(world_size=WORLD_SIZE,microbatch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUMULATION_STEPS,global_batch_size=WORLD_SIZE*RANK_UPDATE_SAMPLES)
    if saved['batch_geometry']!=expected_geometry or saved['loss_reduction']!='mean_of_rank_local_microbatch_token_means_then_DDP_mean':
        raise ValueError('resume batch/loss geometry mismatch')
    step=saved['total_step'];epoch=saved['epoch'];cursor=saved['next_optimizer_step']
    if (type(step) is not int or not 1<=step<=geo['total_steps'] or type(epoch) is not int
        or type(cursor) is not int or step!=epoch*geo['steps_per_epoch']+cursor
        or not 0<=cursor<geo['steps_per_epoch'] or not 0<=epoch<=EPOCHS
        or (epoch==EPOCHS and step!=geo['total_steps'])):
        raise ValueError('resume data/optimizer cursor mismatch')
    state=saved['state_dict'];expected=model.state_dict()
    if set(state)!=set(expected): raise ValueError('audio model tensor inventory mismatch')
    for name,value in state.items():
        other=expected[name]
        if (not torch.is_tensor(value) or value.shape!=other.shape or value.dtype!=other.dtype
            or (value.is_floating_point() and not bool(torch.isfinite(value).all()))):
            raise ValueError(f'invalid audio model tensor:{name}')
        if name.startswith('audio_encoder.base.ced.') and not torch.equal(value.cpu(),other.detach().cpu()):
            raise ValueError(f'frozen CED assets differ:{name}')
    if not torch.equal(state['caption_decoder.lm.model.embed_tokens.weight'],state['caption_decoder.lm.lm_head.weight']):
        raise ValueError('tied embedding/head values conflict')
    if not torch.equal(state['caption_decoder.lm.model.zL_init'].cpu(),expected['caption_decoder.lm.model.zL_init'].cpu()):
        raise ValueError('fixed HRM initial-state buffer changed')
    names=optimizer_parameter_inventory(model,optimizer)
    if saved['optimizer_parameter_inventory']!=names:
        raise ValueError('optimizer parameter name/order/shape mismatch')
    original_groups=optimizer.state_dict()['param_groups'];groups=saved['optimizer']['param_groups']
    if len(groups)!=len(original_groups): raise ValueError('optimizer group count mismatch')
    all_ids=[]
    for group,original,parameters in zip(groups,original_groups,optimizer.param_groups):
        for key in ('params','role','initial_lr','weight_decay','betas','eps','amsgrad','maximize'):
            if group.get(key)!=original.get(key): raise ValueError(f'optimizer group field differs:{key}')
        lr=group['initial_lr']*lr_factor(step,geo['total_steps'],group['role'])
        if not math.isclose(group['lr'],lr,rel_tol=1e-12,abs_tol=1e-15): raise ValueError('optimizer LR cursor differs')
        all_ids.extend(group['params'])
        for key,param in zip(group['params'],parameters['params']):
            entry=saved['optimizer']['state'].get(key,{})
            if set(entry)!={'step','exp_avg','exp_avg_sq'}: raise ValueError('missing AdamW moments')
            if not torch.is_tensor(entry['step']) or entry['step'].numel()!=1 or float(entry['step'])!=step:
                raise ValueError('per-parameter optimizer step differs')
            for name in ('exp_avg','exp_avg_sq'):
                value=entry[name]
                if value.shape!=param.shape or value.dtype!=torch.float32 or not bool(torch.isfinite(value).all()):
                    raise ValueError('invalid AdamW moment')
            if bool((entry['exp_avg_sq']<0).any()): raise ValueError('negative AdamW second moment')
    if len(set(all_ids))!=len(all_ids) or set(all_ids)!=set(saved['optimizer']['state']):
        raise ValueError('AdamW serialized parameter inventory differs')
    scheduler=saved['scheduler']
    if (scheduler.get('last_epoch')!=step or scheduler.get('base_lrs')!=[g['initial_lr'] for g in groups]
        or scheduler.get('_last_lr')!=[g['lr'] for g in groups] or scheduler.get('_step_count')!=step+1):
        raise ValueError('scheduler LR/step state differs')
    states=saved['random_state_by_rank']
    if len(states)!=8 or any(set(s)!={'python','numpy','torch','cuda'} for s in states):
        raise ValueError('requires complete eight-rank RNG states')
    trainable={name for name,p in model.named_parameters() if p.requires_grad}
    if set(saved['grad_norm_tracker'])!=trainable:
        raise ValueError('GradNorm parameter inventory differs')
    if any(len(v)!=2 or any(not math.isfinite(float(x)) or x<0 for x in v) for v in saved['grad_norm_tracker'].values()):
        raise ValueError('invalid GradNorm state')
    if not isinstance(saved['grad_scaler'],dict): raise ValueError('invalid GradScaler state')
    if contract.get('grad_scaler') is True:
        scaler=saved['grad_scaler']
        if (set(scaler)!={'scale','growth_factor','backoff_factor','growth_interval','_growth_tracker'}
            or not math.isfinite(float(scaler['scale'])) or scaler['scale']<=0
            or scaler['growth_factor']<=1 or not 0<scaler['backoff_factor']<1
            or scaler['growth_interval']<1 or scaler['_growth_tracker']<0):
            raise ValueError('incomplete/invalid enabled GradScaler state')
    return dict(passed=True,step=step,parameter_tensors=len(state),optimizer_tensors=len(all_ids))
