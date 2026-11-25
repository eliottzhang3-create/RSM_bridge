"""Native PrefixLM inference; reuse the exact trained Decoder.inference_logits."""
import importlib
from pathlib import Path

def audit_model(model,route_contract,contract,root,*,hidden_size=576):
    import torch
    from .registry import variant_spec
    spec=variant_spec('hrm_prefixlm')
    implementation=importlib.import_module('models.hrm_prefix')
    shared=importlib.import_module('text_hrm_h2l3.model')
    definition=importlib.import_module('text_hrm_h2l3.contract')
    lm=model.caption_decoder.lm;core=lm.model
    for module,path in ((implementation,root/'models/hrm_prefix.py'),
                        (shared,root.parents[1]/'text_hrm_h2l3/model.py')):
        if Path(module.__file__).resolve()!=path.resolve():raise ValueError('HRM implementation escaped snapshot')
    if type(lm) is not implementation.AudioPrefixHRMForCausalLM or type(core) is not implementation.AudioPrefixHRMModel:
        raise ValueError('expected native AudioPrefixHRM model, not causal text HRM')
    definition.validate_config(lm.config.to_dict(),production=hidden_size==576)
    if (len(core.layers)!=20 or len({id(x) for x in core.layers})!=20 or core.gradient_checkpointing
            or core.audit_gradients or core.audit_refs or lm.config.hidden_size!=hidden_size
            or model.caption_decoder.text_model_contract!=spec['text_contract']
            or not model.caption_decoder.is_hrm_prefixlm
            or lm.get_input_embeddings().weight is not lm.lm_head.weight):
        raise ValueError('HRM physical/config/diagnostic/tied-weight contract differs')
    if route_contract.recursive_checkpoint_contract(Path(contract['text_model_dir']))!=contract['recursive_text_checkpoint']:
        raise ValueError('HRM raw text source changed since audio training')
    if any(any(x in name for x in ('router','lora_','audio_fusions','audio_gates')) for name in lm.state_dict()):
        raise ValueError('HRM must not contain MeSH/LoRA/audio branches')
    for i,layer in enumerate(core.layers):
        if 5<=i<15:
            if (type(layer.self_attn) is not shared.GatedLlamaAttention
                    or tuple(layer.self_attn.gate_proj.weight.shape)!=(hidden_size,hidden_size)):
                raise ValueError('HRM learned attention gate missing')
            for norm in (layer.input_layernorm,layer.post_attention_layernorm):
                if type(norm) is not shared.ParameterlessRMSNorm or list(norm.parameters()) or norm.eps!=lm.config.rms_norm_eps:
                    raise ValueError('HRM middle MagicNorm differs')
    for norm,eps in ((core.entry_norm,lm.config.hrm_entry_eps),(core.l_exit_norm,lm.config.hrm_exit_eps),
                     (core.h_exit_norm,lm.config.hrm_exit_eps)):
        if type(norm) is not shared.ParameterlessRMSNorm or list(norm.parameters()) or norm.eps!=eps:
            raise ValueError('HRM entry/exit Norm differs')
    if (tuple(core.zL_init.shape)!=(hidden_size,) or core.zL_init.dtype!=torch.float32
            or core.zL_init.requires_grad or not bool(torch.isfinite(core.zL_init).all())
            or 'zL_init' not in dict(core.named_buffers())):
        raise ValueError('HRM fixed initial state differs')
    count=sum(p.numel() for p in lm.parameters())
    if hidden_size==576 and count!=102420288:raise ValueError('HRM parameter inventory differs')
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('HRM inference weights must be FP32')
    return {'physical_layer_count':20,'logical_layer_count':50,'schedule':list(spec['schedule']),
        'unique_lm_parameters':count,'embedding_head_tied':True,'router_modules':0,
        'trained_gates_preserved':True,'fixed_initial_state_preserved':True,'bp_steps':5,
        'attention_mask':spec['attention_mask'],'trained_values_preserved':True}


def greedy_decode(model,tokenizer,prefix,texts_enc,*,max_new_tokens=300,hidden_size=576):
    import torch
    if max_new_tokens!=300:raise ValueError('all benchmarks require max_new_tokens=300')
    if tuple(prefix.shape)!=(1,383,hidden_size) or prefix.dtype!=torch.float32:
        raise ValueError('HRM requires the original FP32 prefix383')
    generated_ids=[];stop_reason='max_new_tokens'
    for _ in range(max_new_tokens):
        ids=(torch.tensor([generated_ids],dtype=torch.long,device=prefix.device) if generated_ids else None)
        output=model.caption_decoder.inference_logits(prefix,texts_enc,ids)
        logits=output[:,-1,:]
        if not bool(torch.isfinite(logits).all()):raise ValueError('nonfinite HRM generation logits')
        token=int(torch.argmax(logits,dim=-1).item());generated_ids.append(token)
        del output,logits
        if token==int(tokenizer.eos_token_id):stop_reason='eos_token';break
    text=tokenizer.decode(generated_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    return {'generated_token_ids':generated_ids,'generated_text':text,'generated_text_raw':text,
        'generated_token_count':len(generated_ids),'stop_reason':stop_reason,'max_new_tokens':max_new_tokens}
