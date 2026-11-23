"""Read-only per-forward verification for 56-call MeSH and the original read2."""
from .registry import variant_spec
from text71427_ced_eval.observers import install_trace_audit as install_read2

def install_trace_audit(lm,variant):
    spec=variant_spec(variant)
    if spec['model_variant']=='audio7_read2':
        return install_read2(lm,'audio7_read2')
    core=lm.model
    trace=[];routers=[]
    width=lm.config.hidden_size
    audit={'status':'PENDING','checked_forwards':0,'expected_schedule':list(spec['schedule']),
        'memory_policy':'fresh_per_full_sequence_forward','use_cache':False,
        'audio_memory_slots':spec['slots'],'text_memory_slots':spec['slots'],
        'router_modules':spec['router_modules']}
    def begin(_module,_inputs,kwargs):
        trace.clear();routers.clear()
        if (core.gradient_checkpointing or kwargs.get('use_cache') is not False
                or kwargs.get('past_key_values') is not None or kwargs.get('attention_mask') is not None
                or kwargs.get('labels') is not None or kwargs.get('input_ids') is not None
                or kwargs.get('inputs_embeds') is None):
            raise RuntimeError('expected full inputs_embeds without checkpointing/cache/mask/labels')
    def layer(index):
        def hook(_module,_inputs,_output):trace.append(index)
        return hook
    def router(name):
        def hook(_module,inputs,output):
            routers.append((len(trace),name,tuple(inputs[0].shape),tuple(output.shape)))
        return hook
    def finish(_module,_inputs,output):
        if trace!=list(spec['schedule']) or output.past_key_values is not None:
            raise RuntimeError('actual 56-call physical schedule or cache policy differs')
        if [x['physical_index'] for x in core.last_forward_trace]!=trace:
            raise RuntimeError('internal and observed physical traces disagree')
        length=int(output.logits.shape[1]);expected=[]
        for index,boundary in enumerate((7,21,35,49)):
            for kind in ('write','read'):
                expected.append((boundary,f'{kind}{index}',(1,length,width),(1,length,spec['slots'])))
        if routers!=expected:
            raise RuntimeError(f'three-loop router order/shape differs: {routers}')
        audit.update(status='PASS',checked_forwards=audit['checked_forwards']+1,
            last_sequence_length=length,last_schedule=list(trace),
            last_router_events=[list(x[:2])+[list(x[2]),list(x[3])] for x in routers])
    lm.register_forward_pre_hook(begin,with_kwargs=True)
    for index,module in enumerate(core.layers):module.register_forward_hook(layer(index))
    for kind in ('write','read'):
        for index,module in enumerate(getattr(core,kind+'_routers')):
            module.register_forward_hook(router(f'{kind}{index}'))
    lm.register_forward_hook(finish)
    return audit
