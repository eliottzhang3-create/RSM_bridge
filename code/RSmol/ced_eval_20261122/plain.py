"""Standard 30 physical layers: no recursive class, MeSH state, or extra LoRA."""
from pathlib import Path


def audit_model(model, route_root):
    import torch
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    lm = model.caption_decoder.lm
    core = lm.model
    if type(lm) is not LlamaForCausalLM:
        raise ValueError('plain target requires the standard Hugging Face LlamaForCausalLM')
    if (len(core.layers)!=30 or len({id(layer) for layer in core.layers})!=30
            or lm.config.num_hidden_layers!=30 or lm.config.hidden_size!=576
            or core.gradient_checkpointing or lm.model.embed_tokens.weight is not lm.lm_head.weight):
        raise ValueError('plain target must retain 30 independent layers and tied embedding/head')
    if any(any(fragment in name for fragment in ('router','lora_','audio_gates','audio_fusions','layer_mlps'))
           for name in model.state_dict()):
        raise ValueError('plain target unexpectedly contains MeSH/LoRA/extra audio branches')
    if any(p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('plain model weights must remain FP32')
    import sys
    for name in ('models.mellow','models.decoder','models.ced'):
        path=Path(sys.modules[name].__file__).resolve()
        if path != Path(route_root)/(name.replace('.','/')+'.py'):
            raise ValueError('plain training implementation escaped pinned source')
    return {'logical_layer_count':30,'physical_layer_count':30,'schedule':list(range(30)),
            'router_modules':0,'recursive_loops':0,'embedding_head_tied':True,'trained_values_preserved':True}


def install_trace_audit(lm):
    trace=[]
    audit={'status':'PENDING','checked_forwards':0,'expected_schedule':list(range(30)),
           'use_cache':False,'router_modules':0}
    def begin(_module,_inputs,kwargs):
        trace.clear()
        if (kwargs.get('use_cache') is not False or kwargs.get('attention_mask') is not None
                or kwargs.get('past_key_values') is not None or kwargs.get('inputs_embeds') is None
                or kwargs.get('labels') is not None or kwargs.get('input_ids') is not None):
            raise ValueError('plain inference requires full inputs_embeds without cache/mask/labels')
    def layer_hook(index):
        def hook(_module,_inputs,_output):trace.append(index)
        return hook
    def finish(_module,_inputs,output):
        if trace!=list(range(30)) or output.past_key_values is not None:
            raise ValueError('plain inference did not execute exactly 0..29 without cache')
        audit.update(status='PASS',checked_forwards=audit['checked_forwards']+1,
                     last_schedule=list(trace),last_sequence_length=int(output.logits.shape[1]))
    lm.register_forward_pre_hook(begin,with_kwargs=True)
    for index,layer in enumerate(lm.model.layers):layer.register_forward_hook(layer_hook(index))
    lm.register_forward_hook(finish)
    return audit
