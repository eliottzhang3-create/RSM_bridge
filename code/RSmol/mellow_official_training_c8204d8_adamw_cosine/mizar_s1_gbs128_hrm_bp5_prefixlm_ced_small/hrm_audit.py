"""Actual PrefixLM trace and task gradients, before clipping/optimizer decay."""
from contract import TEXT_MODEL_SCHEDULE,ATTENTION_POLICY


def recursive_forward_audit(model,batch_size,require_memory=False):
    core=model.caption_decoder.lm.model
    if core.last_trace!=list(TEXT_MODEL_SCHEDULE) or len(core.layers)!=20:
        raise ValueError('HRM must execute P5 + (L5x3 H5)x2 + C5 =50 calls')
    mask=core.last_mask
    if (mask['policy']!=ATTENTION_POLICY or mask['prefix_length']!=383
        or mask['shape']!=[batch_size,1,683,683]):
        raise ValueError('PrefixLM must cover the complete683 sequence at every call')
    return dict(physical_layers=20,logical_calls=50,schedule=core.last_trace,bp_steps=5,mask=mask)


def prelude_gradient_audit(model):
    from text_hrm_reinit3600_bp5.gradients import prelude_parameters
    return prelude_parameters(model.caption_decoder.lm)


def recursive_backward_audit(model,batch_size):
    from text_hrm_reinit3600_bp5.gradients import gradient_audit
    core=model.caption_decoder.lm.model
    try:
        result=recursive_forward_audit(model,batch_size)
        result['task_gradients']=gradient_audit(model.caption_decoder.lm,5)
        result['passed']=result['task_gradients']['passed']
        return result
    finally:
        core.audit_refs.clear()
        core.audit_gradients=False
