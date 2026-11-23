"""Read-only post-block capture; no answers and no final LM norm contamination."""
from mizar_layer_analysis_common import runtime

VIEWS=('audio_mean','valid_text_mean','answer_readout')


def extract(loaded,prefix,valid_prompt_mask):
    import torch
    if prefix.shape[1:]!=(383,576) or len(valid_prompt_mask)!=256 or any(type(x) is not int or x not in (0,1) for x in valid_prompt_mask):
        raise ValueError('expected original prefix383 and tokenizer validity mask256')
    return extract_layout(loaded,prefix,valid_prompt_mask,audio_tokens=126,prompt_start=127)


def extract_layout(loaded,prefix,valid_prompt_mask,*,audio_tokens,prompt_start):
    """Layout parameters expose the same computation to small real-Llama tests."""
    import torch
    lm=loaded['lm']; ctl=loaded['controller']
    ctl.set_drop(None)
    mask=torch.tensor(valid_prompt_mask,device=prefix.device,dtype=torch.bool)
    if not bool(mask.any()) or prefix.shape[1]!=prompt_start+len(mask): raise ValueError('invalid effective prompt positions')
    collected,trace=[],[]
    def observer(index):
        def after(_module,_args,output):
            hidden=output[0] if isinstance(output,tuple) else output
            if hidden.shape!=prefix.shape: raise ValueError('block output layout changed')
            h=hidden.float()
            collected.append(torch.stack([h[0,:audio_tokens].mean(0),h[0,prompt_start:][mask].mean(0),h[0,-1]],dim=0))
            trace.append(index)
            # Returning None makes this hook strictly read-only.
        return after
    handles=[layer.register_forward_hook(observer(i)) for i,layer in enumerate(ctl.original,1)]
    try:
        with torch.inference_mode(),runtime.autocast(loaded['device']):
            output=lm(inputs_embeds=prefix,attention_mask=None,use_cache=False)
        if trace!=list(range(1,31)) or len(collected)!=30: raise ValueError('not exactly thirty captured block outputs')
        value=torch.stack(collected,dim=1).detach().cpu().contiguous()
        if not bool(torch.isfinite(value).all()): raise ValueError('nonfinite pooled representation')
        return value,output.logits[:,-1].detach().float().cpu()
    finally:
        for handle in handles: handle.remove()


def runtime_check(loaded,prefix,record):
    import torch
    ctl=loaded['controller']; ctl.set_drop(None)
    with torch.inference_mode(),runtime.autocast(loaded['device']):
        native=loaded['lm'](inputs_embeds=prefix,attention_mask=None,use_cache=False).logits[:,-1].float().cpu()
    features,observed=extract(loaded,prefix,record['prompt_attention_mask_audit_only'])
    if not torch.equal(native,observed): raise ValueError('capture changed native baseline logits')
    return {'status':'PASS','capture_native_logits_bitwise_equal':True,'feature_shape':list(features.shape),
            'capture_point':'raw_post_block_before_final_LM_norm','execution':ctl.evidence()}
