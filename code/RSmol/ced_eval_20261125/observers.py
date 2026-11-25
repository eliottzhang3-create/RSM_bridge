"""Read-only architecture/attention observers. Never patch a model forward."""
from .registry import variant_spec

TARGETS=('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
         'mlp.gate_proj','mlp.up_proj','mlp.down_proj')

def install_trace_audit(lm,variant):
    if variant_spec(variant)['family']=='hrm':return install_hrm(lm)
    return install_mesh(lm)

def install_mesh(lm):
    spec=variant_spec('audio8_read3');core=lm.model
    trace=[];routers=[];projections=[];modules=[];width=lm.config.hidden_size
    audit={'status':'PENDING','checked_forwards':0,'expected_schedule':list(spec['schedule']),
        'memory_policy':'fresh_per_full_sequence_forward','use_cache':False,
        'audio_memory_slots':8,'text_memory_slots':6,'router_modules':11}
    def idle():
        if core.gradient_checkpointing or any(m.loop_index is not None for _,_,m in modules):
            raise RuntimeError('LoRA scope not reset or gradient checkpointing enabled')
    def begin(_module,_inputs,kwargs):
        trace.clear();routers.clear();projections.clear();idle()
        if (kwargs.get('use_cache') is not False or kwargs.get('past_key_values') is not None
                or kwargs.get('attention_mask') is not None or kwargs.get('labels') is not None
                or kwargs.get('input_ids') is not None or kwargs.get('inputs_embeds') is None
                or not isinstance(kwargs.get('audio_information'),(tuple,list))):
            raise RuntimeError('audio8 read3 requires explicit full prefix and A4/A8, without cache/mask/labels')
    def layer(index):
        def hook(_module,_inputs,_output):trace.append(index)
        return hook
    def router(name):
        def hook(_module,inputs,output):routers.append((len(trace),name,tuple(inputs[0].shape),tuple(output.shape)))
        return hook
    def projection(index,target):
        def hook(module,inputs):
            logical=len(trace);loop=(logical-7)//14 if 7<=logical<49 else None
            if loop is None or index!=spec['schedule'][logical] or module.loop_index!=loop:
                raise RuntimeError('wrong three-loop LoRA selection')
            projections.append((logical,index,target,loop,tuple(inputs[0].shape)))
        return hook
    def finish(_module,_inputs,output):
        idle()
        if trace!=list(spec['schedule']) or output.past_key_values is not None:
            raise RuntimeError('audio8 read3 must execute exactly56 calls without cache')
        if [x['physical_index'] for x in core.last_forward_trace]!=trace:raise RuntimeError('internal/observed traces differ')
        length=int(output.logits.shape[1]);expected=[]
        for i,boundary in enumerate((7,21,35,49)):
            expected.append((boundary,f'write{i}',(1,length,width),(1,length,6)))
            if i<3:
                expected.append((boundary,f'audio_read{i}',(1,126,width),(1,126,8)))
                expected.append((boundary,f'read{i}',(1,length-126,width),(1,length-126,6)))
            else:expected.append((boundary,f'read{i}',(1,length,width),(1,length,6)))
        if routers!=expected:raise RuntimeError('audio8 read3 router order/shape differs')
        expected=[(logical,spec['schedule'][logical],target,(logical-7)//14,(1,length,module.in_features))
            for logical in range(7,49) for index,target,module in modules if index==spec['schedule'][logical]]
        if sorted(projections)!=sorted(expected) or len(projections)!=294:raise RuntimeError('expected98 LoRA calls per loop')
        if list(core.last_loop_lora_trace)!=[(i,spec['schedule'][i],(i-7)//14) for i in range(7,49)]:
            raise RuntimeError('internal/observed LoRA traces differ')
        audit.update(status='PASS',checked_forwards=audit['checked_forwards']+1,last_sequence_length=length,
            last_schedule=list(trace),last_router_events=[list(x[:2])+[list(x[2]),list(x[3])] for x in routers],
            loop_lora_projection_calls=294,loop_lora_calls_per_loop=[98]*3,loop_lora_indices_reset=True)
    lm.register_forward_pre_hook(begin,with_kwargs=True)
    for i,module in enumerate(core.layers):module.register_forward_hook(layer(i))
    for kind in ('write','read','audio_read'):
        for i,module in enumerate(getattr(core,kind+'_routers')):module.register_forward_hook(router(f'{kind}{i}'))
    for i in range(7,21):
        for target in TARGETS:
            module=core.layers[i].get_submodule(target);modules.append((i,target,module))
            module.register_forward_pre_hook(projection(i,target))
    lm.register_forward_hook(finish)
    return audit

def install_hrm(lm):
    import torch
    spec=variant_spec('hrm_prefixlm');core=lm.model
    trace=[];state={}
    audit={'status':'PENDING','checked_forwards':0,'expected_schedule':list(spec['schedule']),
        'attention_mask':spec['attention_mask'],'prefix_length':383,'bp_steps':5,'use_cache':False,
        'prompt_padding_visible':False,'query382_retained':True}
    def begin(_module,_inputs,kwargs):
        trace.clear();state.clear()
        x=kwargs.get('inputs_embeds');valid=kwargs.get('attention_mask')
        if (x is None or x.shape[0]!=1 or x.shape[1]<383 or valid is None or tuple(valid.shape)!=tuple(x.shape[:2])
                or kwargs.get('prefix_length')!=383 or kwargs.get('bp_steps')!=5
                or kwargs.get('use_cache') is not False or kwargs.get('past_key_values') is not None
                or kwargs.get('labels') is not None or core.gradient_checkpointing or core.audit_gradients
                or not bool(((valid==0)|(valid==1)).all()) or not bool(valid[:,:127].eq(1).all())):
            raise RuntimeError('HRM requires native PrefixLM383/BP5/full validity and no cache/training audits')
        state.update(length=int(x.shape[1]),valid=valid,mask=None)
    def layer(index):
        def hook(_module,_inputs,kwargs):
            mask=kwargs.get('attention_mask');length=state['length']
            if mask is None or tuple(mask.shape)!=(1,1,length,length):raise RuntimeError('HRM missing full PrefixLM mask')
            if state['mask'] is None:
                q=torch.arange(length,device=mask.device).view(1,1,length,1)
                k=torch.arange(length,device=mask.device).view(1,1,1,length)
                allowed=((k<383)|((q>=383)&(k<=q))) & state['valid'][:,None,None,:].bool()
                if not torch.equal(mask.eq(0),allowed) or not bool(mask.masked_select(~allowed).eq(torch.finfo(mask.dtype).min).all()):
                    raise RuntimeError('HRM PrefixLM/padding mask values differ')
                state['mask']=mask
            elif mask is not state['mask']:raise RuntimeError('all50 HRM layer calls must share the same mask')
            trace.append(index)
        return hook
    def finish(_module,_inputs,output):
        if trace!=list(spec['schedule']) or core.last_trace!=trace or output.past_key_values is not None:
            raise RuntimeError('actual HRM50 trace or cache differs')
        if core.last_calls!=[(label,False) for label in ('L1','L2','L3','H1','L4','L5','L6','H2')] or core.audit_refs:
            raise RuntimeError('HRM inference must disable all task gradients/retained activations')
        if core.last_mask!={'policy':spec['attention_mask'],'prefix_length':383,
                'shape':[1,1,state['length'],state['length']],'padding_keys_masked':True}:
            raise RuntimeError('internal HRM mask metadata differs')
        audit.update(status='PASS',checked_forwards=audit['checked_forwards']+1,
            last_sequence_length=state['length'],last_schedule=list(trace),same_mask_all50=True)
        state.clear()
    lm.register_forward_pre_hook(begin,with_kwargs=True)
    for i,module in enumerate(core.layers):module.register_forward_pre_hook(layer(i),with_kwargs=True)
    lm.register_forward_hook(finish)
    return audit
