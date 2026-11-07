"""Thirty inherited layers; two shared middle passes with only gated audio input."""
from __future__ import annotations

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaForCausalLM, LlamaPreTrainedModel, LlamaRMSNorm, LlamaRotaryEmbedding,
)

from .contract import (AUDIO_TOKENS, INJECTION_INDICES, SCHEDULE,
                       SINGLE_PASS_SCHEDULE, parameter_role, validate_text_config)
from .llama_compat import _assert_supported_llama_api, _call_decoder_layer, _causal_mask


class RecursiveModel(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Small dimensions are allowed for real CPU unit tests; the artifact
        # loader separately requires the complete production SmolLM2 geometry.
        validate_text_config(config.to_dict(), production=False)
        self.layers = nn.ModuleList(LlamaDecoderLayer(config, layer_idx=i) for i in range(30))
        _assert_supported_llama_api(self.layers[0])
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.logical_to_physical = SCHEDULE
        self.gradient_checkpointing = False
        self.last_layer_trace = ()
        self.last_input_injection_trace = ()
        self.last_input_projection_calls = 0
        self.post_init()

    def enable_new_modules(self):
        if hasattr(self, "audio_linear") or hasattr(self, "audio_gate"):
            raise ValueError("new modules are already registered; refusing initialization reset")
        width, weight = self.config.hidden_size, self.embed_tokens.weight
        # Keep the same first two allocations/order as the serial84 version.
        # Unused branches in that route were created only AFTER this pair.
        with torch.random.fork_rng(devices=[]):
            linear = nn.Linear(width, width, dtype=weight.dtype, device="cpu")
            gate = nn.Linear(width, 1, dtype=weight.dtype, device="cpu")
            nn.init.zeros_(linear.bias)
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.audio_linear = linear.to(weight.device)
        self.audio_gate = gate.to(weight.device)

    def _execute(self, inputs_embeds, *, single_pass_audit=False,
                 position_ids=None):
        if self.gradient_checkpointing:
            raise ValueError("input-only route requires gradient_checkpointing=False")
        if not hasattr(self, "audio_linear") or not hasattr(self, "audio_gate"):
            raise RuntimeError("strict-load all official weights, then enable new modules")
        if (inputs_embeds.ndim != 3 or inputs_embeds.shape[1] < AUDIO_TOKENS
                or inputs_embeds.shape[-1] != self.config.hidden_size):
            raise ValueError("requires full-prefix [B,L>=126,hidden] inputs_embeds")
        if single_pass_audit:
            if self.training or torch.is_grad_enabled():
                raise ValueError("single-pass comparison is restricted to eval/no_grad audit")
            schedule = SINGLE_PASS_SCHEDULE
        else:
            schedule = SCHEDULE
        hidden = inputs_embeds
        batch, length = hidden.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(length, device=hidden.device).unsqueeze(0).expand(batch, -1)
        if tuple(position_ids.shape) != (batch, length):
            raise ValueError("position_ids must match batch and sequence length")
        causal = _causal_mask(attention_mask=None, batch_size=batch, query_length=length,
                              past_length=0, dtype=hidden.dtype, device=hidden.device)
        rotary = self.rotary_emb(hidden, position_ids=position_ids)
        self.last_layer_trace = ()
        self.last_input_injection_trace = ()
        self.last_input_projection_calls = 0
        if not single_pass_audit:
            original_input = self.audio_linear(inputs_embeds[:, :AUDIO_TOKENS])
            self.last_input_projection_calls += 1
        for logical, physical in enumerate(schedule):
            if not single_pass_audit and logical in INJECTION_INDICES:
                audio, text = hidden[:, :AUDIO_TOKENS], hidden[:, AUDIO_TOKENS:]
                audio = audio + torch.tanh(self.audio_gate(audio)) * original_input
                self.last_input_injection_trace += (logical,)
                hidden = torch.cat((audio, text), dim=1)
            kwargs = dict(attention_mask=causal, position_ids=position_ids, cache=None,
                          use_cache=False, cache_position=position_ids[0],
                          position_embeddings=rotary, output_attentions=False)
            layer = self.layers[physical]
            output = _call_decoder_layer(layer, hidden, **kwargs)
            hidden = output[0]
            self.last_layer_trace += (physical,)
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden), past_key_values=None)

    def forward(self, *, inputs_embeds, attention_mask=None,
                use_cache=False, past_key_values=None, position_ids=None, **kwargs):
        if attention_mask is not None or use_cache or past_key_values is not None or kwargs:
            raise ValueError("requires attention_mask=None, use_cache=False and full recomputation")
        return self._execute(inputs_embeds, position_ids=position_ids)


class MizarInputOnlyRecursiveForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = RecursiveModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, *, inputs_embeds, attention_mask=None,
                use_cache=False, past_key_values=None, labels=None, return_dict=True, **kwargs):
        result = self.model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask, use_cache=use_cache,
                            past_key_values=past_key_values, **kwargs)
        logits = self.lm_head(result.last_hidden_state)
        loss = None if labels is None else self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.vocab_size)
        value = CausalLMOutputWithPast(logits=logits, loss=loss, past_key_values=None)
        return value if return_dict else value.to_tuple()


class AudioEncoder(nn.Module):
    def __init__(self, ced_dir):
        super().__init__()
        from .ced import CEDSmallWrapper
        self.base = CEDSmallWrapper({"pretrained_audioencoder_path": str(ced_dir)}, d_out=576)
        self.projection = nn.Identity()

    def forward(self, waveform):
        return self.base(waveform)


class CaptionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lm = MizarInputOnlyRecursiveForCausalLM(config)
        self.prefix_length = 383


class MizarInputOnlyRecursive(nn.Module):
    """Namespaces intentionally match every tensor in the official plain state_dict."""
    def __init__(self, config, ced_dir):
        super().__init__()
        validate_text_config(config.to_dict())
        self.audio_encoder = AudioEncoder(ced_dir)
        self.caption_decoder = CaptionDecoder(config)

    def enable_new_modules(self):
        self.caption_decoder.lm.model.enable_new_modules()

    def generate_prefix_inference(self, input_dict):
        waveform = torch.cat((input_dict["audio1"], input_dict["audio2"]), dim=-1)
        output = self.audio_encoder(waveform)
        embed = self.caption_decoder.lm.get_input_embeddings()
        prompt = input_dict["input"]["input_ids"]
        if prompt.ndim != 2 or prompt.shape[1] != 256:
            raise ValueError("requires exactly 256 prompt tokens")
        sep = torch.zeros((prompt.shape[0], 1), dtype=torch.long, device=prompt.device)
        prefix = torch.cat((output["embedding"], embed(sep), embed(prompt)), dim=1)
        return prefix, output

    def forward(self, input_dict):
        prefix, _ = self.generate_prefix_inference(input_dict)
        answer = input_dict["answer"]["input_ids"]
        if answer.ndim != 2 or answer.shape[1] != 300:
            raise ValueError("requires exactly 300 answer tokens")
        embeddings = torch.cat((prefix, self.caption_decoder.lm.get_input_embeddings()(answer)), dim=1)
        return self.caption_decoder.lm(inputs_embeds=embeddings,
                                       attention_mask=None, use_cache=False)


def trainability_audit(model):
    """Unique parameters, including tied embeddings, appear in exactly one LR group."""
    groups, seen = {}, set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            raise ValueError("duplicate parameter in optimizer identity")
        seen.add(id(parameter))
        role, lr = parameter_role(name)
        if parameter.requires_grad != (lr is not None):
            raise ValueError(f"unexpected requires_grad for {name}")
        decay = 0. if parameter.ndim <= 1 or name.endswith(".bias") else .01
        key = f"{role}:lr={lr}:decay={decay if lr is not None else None}"
        item = groups.setdefault(key, {"role": role, "max_lr": lr,
                                      "weight_decay": decay if lr is not None else None,
                                      "parameters": 0, "names": []})
        item["parameters"] += parameter.numel()
        item["names"].append(name)
    return list(groups.values())
