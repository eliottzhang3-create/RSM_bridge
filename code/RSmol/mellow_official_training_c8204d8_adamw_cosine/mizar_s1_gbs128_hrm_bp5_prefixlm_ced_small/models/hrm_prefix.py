"""Isolated HRM audio forward: original recurrence/BP5, PrefixLM at every layer."""
from __future__ import annotations
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from text_hrm_h2l3.model import SmolHRMModel, SmolHRMForCausalLM
from text_hrm_h2l3.contract import SCHEDULE
from contract import ATTENTION_POLICY


def prefixlm_mask(valid_keys,prefix_length,*,batch,length,dtype,device):
    if type(prefix_length) is not int or not 1<=prefix_length<=length:
        raise ValueError('explicit prefix boundary must be in1..sequence_length')
    if valid_keys is None or tuple(valid_keys.shape)!=(batch,length) or valid_keys.ndim!=2:
        raise ValueError('PrefixLM requires a2-D full-sequence validity mask')
    valid_keys=valid_keys.to(device=device)
    if not bool(((valid_keys==0)|(valid_keys==1)).all()):
        raise ValueError('validity mask must contain only0/1')
    valid_keys=valid_keys.bool()
    if not bool(valid_keys[:,:prefix_length].any(dim=-1).all()):
        raise ValueError('each sample must have a valid prefix key')
    q=torch.arange(length,device=device).view(1,1,length,1)
    k=torch.arange(length,device=device).view(1,1,1,length)
    allowed=((k<prefix_length)|((q>=prefix_length)&(k<=q))) & valid_keys[:,None,None,:]
    # Mask K/V only. In particular, query382 remains the first-answer predictor
    # even when its prompt slot is PAD. Each row has a valid prefix key.
    return torch.zeros((batch,1,length,length),dtype=dtype,device=device).masked_fill(~allowed,torch.finfo(dtype).min)


class AudioPrefixHRMModel(SmolHRMModel):
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                position_ids=None, use_cache=False, past_key_values=None, prefix_length=None,
                bp_steps=5, return_dict=True, **kwargs):
        if kwargs or use_cache or past_key_values is not None or self.gradient_checkpointing:
            raise ValueError(f'unsupported HRM cache/checkpointing/arguments: {sorted(kwargs)}')
        if type(bp_steps) is not int or bp_steps != 5:
            raise ValueError('audio HRM requires fixed BP5')
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('specify exactly one of input_ids / inputs_embeds')
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length = x.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(length, device=x.device).unsqueeze(0).expand(batch, -1)
        if tuple(position_ids.shape) != (batch, length):
            raise ValueError('position_ids shape mismatch')
        mask = prefixlm_mask(attention_mask, prefix_length, batch=batch, length=length,
                             dtype=x.dtype, device=x.device)
        self.last_mask = dict(policy=ATTENTION_POLICY,prefix_length=prefix_length,
                              shape=list(mask.shape),padding_keys_masked=True)
        cos_sin = self.rotary_emb(x, position_ids=position_ids)
        self.last_trace, self.last_calls, self.audit_refs = [], [], {}
        x = self._stack(x, range(5), mask=mask, positions=position_ids, cos_sin=cos_sin)
        self._retain('prelude', x)
        z_h = self.entry_norm(x)
        self._retain('entry', z_h)
        z_l = self.zL_init.to(device=x.device, dtype=x.dtype)
        h_bp = min(2, bp_steps - 1)
        l_bp = bp_steps - h_bp
        outer_grad = torch.is_grad_enabled()
        for i in range(2):
            for k in range(i * 3, (i + 1) * 3):
                enabled = outer_grad and k >= 6 - l_bp
                with torch.set_grad_enabled(enabled):
                    z_l = self.l_exit_norm(self._stack(z_l + z_h, range(5, 10),
                        mask=mask, positions=position_ids, cos_sin=cos_sin))
                self.last_calls.append((f'L{k + 1}', enabled))
                self._retain(f'L{k + 1}', z_l)
            enabled = outer_grad and i >= 2 - h_bp
            with torch.set_grad_enabled(enabled):
                z_h = self.h_exit_norm(self._stack(z_h + z_l, range(10, 15),
                    mask=mask, positions=position_ids, cos_sin=cos_sin))
            self.last_calls.append((f'H{i + 1}', enabled))
            self._retain(f'H{i + 1}', z_h)
        x = self._stack(z_h, range(15, 20), mask=mask, positions=position_ids, cos_sin=cos_sin)
        x = self.norm(x)
        if self.last_trace != list(SCHEDULE):
            raise RuntimeError('HRM execution trace mismatch')
        result = BaseModelOutputWithPast(last_hidden_state=x)
        return result if return_dict else result.to_tuple()


class AudioPrefixHRMForCausalLM(SmolHRMForCausalLM):
    def __init__(self,config):
        LlamaPreTrainedModel.__init__(self,config)
        self.model=AudioPrefixHRMModel(config)
        self.vocab_size=config.vocab_size
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        self.post_init()
        # Only construction defaults. Strict checkpoint load overwrites every
        # parameter, including learned nonzero gates; never reconvert weights.
        for layer in self.model.layers[5:15]:
            nn.init.zeros_(layer.self_attn.gate_proj.weight)
