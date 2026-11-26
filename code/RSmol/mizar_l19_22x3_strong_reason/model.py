"""Official 30 physical layers; L19-22 shared three times, no injection."""
from __future__ import annotations
from contextlib import nullcontext
import json
from pathlib import Path
import torch
from torch import nn
from transformers import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaForCausalLM, LlamaPreTrainedModel, LlamaRMSNorm, LlamaRotaryEmbedding)
from contract import (AUDIO_TOKENS, CORE_LAYERS, SCHEDULE, parameter_role)
from llama_compat import _assert_supported_llama_api, _call_decoder_layer, _causal_mask


class RecurrentModel(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if config.num_hidden_layers != 30:
            raise ValueError("keep every original physical layer")
        self.layers = nn.ModuleList(LlamaDecoderLayer(config, layer_idx=i) for i in range(30))
        _assert_supported_llama_api(self.layers[0])
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.last_layer_trace = ()
        self.last_injection_trace = ()
        self.post_init()

    def _execute(self, inputs_embeds, *, single_pass=False, position_ids=None):
        if self.gradient_checkpointing:
            raise ValueError("decoder gradient checkpointing is disabled for repeated physical layers")
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[1] < AUDIO_TOKENS:
            raise ValueError("requires a full audio/text sequence")
        if single_pass and (self.training or torch.is_grad_enabled()):
            raise ValueError("native single-pass comparison is eval/no_grad only")
        hidden = inputs_embeds
        batch, length = hidden.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(length, device=hidden.device).unsqueeze(0).expand(batch, -1)
        if tuple(position_ids.shape) != (batch, length):
            raise ValueError("position IDs must match the full sequence")
        mask = _causal_mask(attention_mask=None, batch_size=batch, query_length=length,
                            past_length=0, dtype=hidden.dtype, device=hidden.device)
        rotary = self.rotary_emb(hidden, position_ids=position_ids)
        self.last_layer_trace, self.last_injection_trace = (), ()
        for logical, physical in enumerate(range(30) if single_pass else SCHEDULE):
            # Prefix is entirely frozen. Suffix MUST retain the input-gradient
            # graph even though none of its own parameters are trainable.
            context = torch.no_grad() if logical < CORE_LAYERS[0] else nullcontext()
            with context:
                hidden = _call_decoder_layer(self.layers[physical], hidden, attention_mask=mask,
                    position_ids=position_ids, cache=None, use_cache=False,
                    cache_position=position_ids[0], position_embeddings=rotary,
                    output_attentions=False)[0]
            self.last_layer_trace += (physical,)
        # The one original final norm is applied AFTER layer 30, never per loop.
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden), past_key_values=None)

    def forward(self, *, inputs_embeds, attention_mask=None, use_cache=False,
                past_key_values=None, position_ids=None, **kwargs):
        if attention_mask is not None or use_cache or past_key_values is not None or kwargs:
            raise ValueError("full recomputation, original causal attention, and no KV cache required")
        return self._execute(inputs_embeds, position_ids=position_ids)


class RecurrentForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = RecurrentModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, *, inputs_embeds, attention_mask=None, use_cache=False,
                past_key_values=None, labels=None, return_dict=True, **kwargs):
        if labels is not None:
            raise ValueError("use the unchanged external answer-only shifted CE")
        result = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                            use_cache=use_cache, past_key_values=past_key_values, **kwargs)
        output = CausalLMOutputWithPast(logits=self.lm_head(result.last_hidden_state), past_key_values=None)
        return output if return_dict else output.to_tuple()


class AudioEncoder(nn.Module):
    def __init__(self, ced_dir):
        super().__init__()
        from models.ced import CEDSmallWrapper
        self.base = CEDSmallWrapper({"pretrained_audioencoder_path": str(ced_dir)}, d_out=576)
        self.projection = nn.Identity()

    def forward(self, waveform):
        return self.base(waveform)


class CaptionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lm = RecurrentForCausalLM(config)
        self.prefix_length = 383
        self.lm_embedding_size = config.hidden_size
        self.last_layout = None


class MizarCoreTrain(nn.Module):
    """Retain all 454 original state keys, with no added model parameters."""
    def __init__(self, config, ced_dir=None, *, audio_encoder=None):
        super().__init__()
        self.audio_encoder = AudioEncoder(ced_dir) if audio_encoder is None else audio_encoder
        self.caption_decoder = CaptionDecoder(config)

    def enable_training(self):
        self.requires_grad_(False)
        for name, p in self.named_parameters():
            p.requires_grad_(parameter_role(name) != "frozen")
        return self.train()

    def train(self, mode=True):
        super().train(False)
        if mode:
            self.training = self.caption_decoder.training = self.caption_decoder.lm.training = True
            core = self.caption_decoder.lm.model
            core.training = True
            for index in CORE_LAYERS:
                core.layers[index].train(True)
        return self

    def generate_prefix_inference(self, inputs):
        embed = self.caption_decoder.lm.get_input_embeddings()
        with torch.no_grad():
            whole = torch.cat((inputs["audio1"], inputs["audio2"]), dim=-1)
            encoded = self.audio_encoder(whole)
            prompt = inputs["input"]["input_ids"]
            if prompt.ndim != 2 or prompt.shape[1] != 256:
                raise ValueError("prompt must have 256 positions")
            sep = torch.zeros((prompt.shape[0], 1), dtype=torch.long, device=prompt.device)
            prefix = torch.cat((encoded["embedding"], embed(sep), embed(prompt)), dim=1)
        if prefix.shape[1] != 383:
            raise ValueError("official audio126 + SEP1 + prompt256 layout changed")
        return prefix, encoded

    def forward(self, inputs):
        prefix, _ = self.generate_prefix_inference(inputs)
        answer = inputs["answer"]["input_ids"]
        if answer.ndim != 2 or answer.shape[1] != 300:
            raise ValueError("answer must have 300 positions")
        with torch.no_grad():
            answer_embeddings = self.caption_decoder.lm.get_input_embeddings()(answer)
        self.caption_decoder.last_layout = {"audio": 126, "separators": 1, "prompt": 256,
            "prefix": 383, "answer": 300, "total": 683, "attention_mask": None}
        return self.caption_decoder.lm(inputs_embeds=torch.cat((prefix, answer_embeddings), dim=1),
                                       attention_mask=None, use_cache=False)


def validate_trainability(model, *, production=True):
    counts = {}
    for name, p in model.named_parameters():
        role = parameter_role(name)
        if p.requires_grad != (role != "frozen") or p.dtype != torch.float32:
            raise ValueError(f"incorrect trainability/dtype: {name}")
        counts[role] = counts.get(role, 0) + p.numel()
    if set(counts) != {"frozen", "layer19", "layer20", "layer21", "layer22"}:
        raise ValueError(f"unexpected parameter roles: {counts}")
    lm = model.caption_decoder.lm
    if lm.get_input_embeddings().weight is not lm.get_output_embeddings().weight:
        raise ValueError("embedding/head tied Parameter identity lost")
    for i, layer in enumerate(lm.model.layers):
        if layer.training != (model.training and i in CORE_LAYERS):
            raise ValueError(f"layer mode changed: {i}")
    if any(m.training for m in model.audio_encoder.modules()):
        raise ValueError("frozen encoder/projector must remain eval")
    if production:
        if counts != {"frozen":145139456,**{f"layer{i}":3540096 for i in (19,20,21,22)}}:
            raise ValueError(f"production parameter inventory differs: {counts}")
        if len(list(model.state_dict())) != 454:
            raise ValueError("expected exactly the original454 state entries")
    return counts


def load_official(model_dir):
    model_dir = Path(model_dir)
    config = LlamaConfig.from_dict(json.loads((model_dir / "assets/smollm2-135m/config.json").read_text()))
    expected = {"hidden_size":576,"intermediate_size":1536,"num_hidden_layers":30,
                "num_attention_heads":9,"num_key_value_heads":3,"vocab_size":49152}
    if any(getattr(config, k) != v for k, v in expected.items()) or not config.tie_word_embeddings:
        raise ValueError("wrong official text geometry")
    config.use_cache = False
    config._attn_implementation = "sdpa"
    model = MizarCoreTrain(config, model_dir / "assets/ced-small").float()
    checkpoint = model_dir / "checkpoints/final/stage3_q4_seed20260905.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if (not isinstance(state, dict) or len(state) != 454
            or any(not torch.is_tensor(v) or (v.is_floating_point() and v.dtype != torch.float32)
                   or not bool(torch.isfinite(v).all()) for v in state.values())
            or sum(v.numel() for v in state.values()) != 187628509):
        raise ValueError("invalid official454 FP32 finite plain state_dict")
    a, b = "caption_decoder.lm.model.embed_tokens.weight", "caption_decoder.lm.lm_head.weight"
    if not torch.equal(state[a], state[b]):
        raise ValueError("conflicting tied source tensors")
    model.load_state_dict(state, strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value, state[name]):
            raise ValueError(f"official inheritance changed: {name}")
    del state
    model.enable_training()
    validate_trainability(model)
    return model
