"""Explicit 20-physical-layer / 50-execution HRM; no global AutoModel registration."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM, LlamaPreTrainedModel,
    LlamaRMSNorm, LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv,
)

from .contract import SCHEDULE, SOURCE_LAYERS, validate_config
from .llama_compat import _causal_mask, _call_decoder_layer


class ParameterlessRMSNorm(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x):
        # Preserve FP32 residual states under BF16 autocast, with FP32 norm math.
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype)


class GatedLlamaAttention(LlamaAttention):
    """Original GQA/RoPE, followed by sigmoid-per-channel gating before o_proj."""
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, past_key_values=None, cache_position=None, **kwargs):
        if past_key_value is not None or past_key_values is not None:
            raise ValueError('HRM first version does not support KV cache')
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        k, v = repeat_kv(k, groups), repeat_kv(v, groups)
        a = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                                          attn_mask=attention_mask, dropout_p=0.0,
                                          is_causal=False, scale=self.scaling)
        a = a.transpose(1, 2).contiguous().reshape(*hidden_states.shape[:-1], -1)
        return self.o_proj(torch.sigmoid(self.gate_proj(hidden_states)) * a), None


class SmolHRMModel(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        validate_config(config.to_dict(), production=False)
        if config.attention_dropout != 0 or config.attention_bias or getattr(config, 'mlp_bias', False):
            raise ValueError('conversion requires bias-free, dropout-free SmolLM2 blocks')
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, i) for i in range(20)])
        for i in range(5, 15):
            layer = self.layers[i]
            layer.self_attn = GatedLlamaAttention(config, i)
            layer.input_layernorm = ParameterlessRMSNorm(config.rms_norm_eps)
            layer.post_attention_layernorm = ParameterlessRMSNorm(config.rms_norm_eps)
        self.entry_norm = ParameterlessRMSNorm(config.hrm_entry_eps)
        self.l_exit_norm = ParameterlessRMSNorm(config.hrm_exit_eps)
        self.h_exit_norm = ParameterlessRMSNorm(config.hrm_exit_eps)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.hrm_initial_state_seed)
            c = torch.empty(config.hidden_size, dtype=torch.float32, device='cpu')
            # Exact official approximate truncation formula; persistent FP32 buffer.
            c.normal_().fmod_(3.0).mul_(1.014762601732121)
        self.register_buffer('zL_init', c, persistent=True)
        self.gradient_checkpointing = False
        self.audit_gradients = False
        self.audit_refs = {}
        self.last_trace = []
        self.last_calls = []
        self.post_init()

    def _stack(self, hidden, indices, *, mask, positions, cos_sin):
        for index in indices:
            self.last_trace.append(index)
            hidden = _call_decoder_layer(self.layers[index], hidden,
                attention_mask=mask, position_ids=positions, cache=None, use_cache=False,
                cache_position=positions[0], position_embeddings=cos_sin,
                output_attentions=False)[0]
        return hidden

    def _retain(self, name, tensor):
        if self.audit_gradients and tensor.requires_grad:
            tensor.retain_grad()
            self.audit_refs[name] = tensor

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                position_ids=None, use_cache=False, past_key_values=None,
                bp_steps=5, return_dict=True, **kwargs):
        if kwargs or use_cache or past_key_values is not None or self.gradient_checkpointing:
            raise ValueError(f'unsupported HRM cache/checkpointing/arguments: {sorted(kwargs)}')
        if type(bp_steps) is not int or not 3 <= bp_steps <= 8:
            raise ValueError('bp_steps must be 3..8 (production schedule: 3..5)')
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('specify exactly one of input_ids / inputs_embeds')
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length = x.shape[:2]
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError('this route accepts padding-only 2-D masks, never PrefixLM masks')
        if position_ids is None:
            position_ids = torch.arange(length, device=x.device).unsqueeze(0).expand(batch, -1)
        if tuple(position_ids.shape) != (batch, length):
            raise ValueError('position_ids shape mismatch')
        mask = _causal_mask(attention_mask=attention_mask, batch_size=batch,
                            query_length=length, past_length=0, dtype=x.dtype, device=x.device)
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


class SmolHRMForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = SmolHRMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        for layer in self.model.layers[5:15]:
            nn.init.zeros_(layer.self_attn.gate_proj.weight)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        raise ValueError('decoder gradient checkpointing is not supported by this route')

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                inputs_embeds=None, labels=None, use_cache=False, bp_steps=5,
                return_dict=True, **kwargs):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, inputs_embeds=inputs_embeds,
                            use_cache=use_cache, bp_steps=bp_steps, **kwargs)
        logits = self.lm_head(output.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, self.vocab_size),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        result = CausalLMOutputWithPast(loss=loss, logits=logits)
        return result if return_dict else result.to_tuple()


def convert_state(source_state, target):
    """Fold gamma and compensate zero sigmoid gate; never called on trained HRM."""
    result = {}
    for name, value in target.state_dict().items():
        if name == 'model.zL_init' or name.endswith('self_attn.gate_proj.weight'):
            result[name] = value.detach().clone()
            continue
        parts = name.split('.')
        is_layer = len(parts) > 3 and parts[1] == 'layers'
        source_name = '.'.join(parts[:2] + [str(SOURCE_LAYERS[int(parts[2])])] + parts[3:]) if is_layer else name
        source = source_state[source_name].detach().float()
        middle = is_layer and 5 <= int(parts[2]) < 15
        if middle:
            prefix = f'model.layers.{SOURCE_LAYERS[int(parts[2])]}'
            if any(name.endswith(f'self_attn.{kind}_proj.weight') for kind in ('q', 'k', 'v')):
                source = source * source_state[prefix + '.input_layernorm.weight'].float().unsqueeze(0)
            elif name.endswith(('mlp.gate_proj.weight', 'mlp.up_proj.weight')):
                source = source * source_state[prefix + '.post_attention_layernorm.weight'].float().unsqueeze(0)
            elif name.endswith('self_attn.o_proj.weight'):
                source = 2.0 * source
        result[name] = source.clone()
    return result


def validate_tensors(model, state):
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError(f'tensor inventory mismatch: missing={sorted(set(expected)-set(state))}; extra={sorted(set(state)-set(expected))}')
    for name, value in state.items():
        if value.shape != expected[name].shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f'invalid tensor: {name}')
    if not torch.equal(state['model.embed_tokens.weight'], state['lm_head.weight']):
        raise ValueError('tied embedding/head values conflict')


def strict_load(model, state):
    validate_tensors(model, state)
    model.load_state_dict(state, strict=True)
    model.tie_weights()
    if model.model.embed_tokens.weight is not model.lm_head.weight:
        raise ValueError('embedding/head Parameter binding lost')


def load_model(path, *, production=True):
    """Load raw checkpoint model only; model init never reads training_state.pt."""
    from transformers import AutoConfig
    from .conversion import read_weights
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    validate_config(config.to_dict(), production=production)
    config._attn_implementation = 'sdpa'
    # Inspect serialized dtype/tied aliases before HF could cast or overwrite them.
    state = read_weights(path, require_float32=True)
    model = SmolHRMForCausalLM(config).float()
    strict_load(model, state)
    if model.model.embed_tokens.weight is not model.lm_head.weight:
        raise ValueError('tied weights lost after HF load')
    return model
