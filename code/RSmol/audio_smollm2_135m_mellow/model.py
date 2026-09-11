"""HTSAT/Mellow audio adapter with the standard original SmolLM2-135M.

Only the text backbone differs from the current recursive audio route.  The
audio path deliberately reuses the existing Mellow loader, waveform adapter,
projection bridge, and answer-only label helper.  The public model and
architecture contract here are baseline-specific and never claim recursive,
router, or shared-layer behavior.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from audio_5_10x2_5_mesh_mellow.model import (  # noqa: F401
    AudioBridge,
    _audio_forward,
    _load_mellow_wrapper,
    build_labels,
)


SMOLLM2_HIDDEN_SIZE = 576
SMOLLM2_LAYER_COUNT = 30
AUDIO_TOKENS_PER_CLIP = 129
AUDIO_PREFIX_TOKENS = 260
MAPPER_CONTRACT = (
    "mellow_c2l_527x768__concat_cls_frames__projection_768x576x576_"
    "biasfree_dropout0.5__cls_preserving_avgpool8"
)
ORIGINAL_SMOLLM2_CONTRACT = (
    "original_smollm2_135m_standard_llama_30_layers_hidden576_audio_mellow"
)


@dataclass
class AudioSmolLM2Config:
    sample_rate: int = 32000
    audio_seconds: int = 10
    framewise_classes: int = 527
    encoder_dim: int = 768
    projection_dim: int = SMOLLM2_HIDDEN_SIZE
    downsample_kernel: int = 8
    projection_dropout: float = 0.5
    max_prompt_tokens: int = 129
    max_answer_tokens: int = 250
    max_context_length: int = 768
    separator_token_id: int | None = None
    architecture_contract: str = ORIGINAL_SMOLLM2_CONTRACT


def validate_original_smollm2(model: nn.Module) -> dict[str, Any]:
    """Validate the standard local SmolLM2 architecture before training.

    This check intentionally rejects converted/custom text models.  It also
    checks that all thirty decoder-layer objects are distinct and that no
    router/shared-loop parameters are present.
    """
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    if not isinstance(model, LlamaForCausalLM):
        raise TypeError(
            "original baseline requires transformers LlamaForCausalLM, "
            f"got {type(model).__module__}.{type(model).__name__}"
        )
    config = model.config
    model_type = str(getattr(config, "model_type", ""))
    if model_type != "llama":
        raise ValueError(f"original baseline requires model_type='llama', got {model_type!r}")
    architecture_names = [str(value) for value in (getattr(config, "architectures", None) or [])]
    if architecture_names and "LlamaForCausalLM" not in architecture_names:
        raise ValueError(f"original baseline requires LlamaForCausalLM architecture, got {architecture_names}")
    hidden_size = int(getattr(config, "hidden_size", -1))
    layer_count = int(getattr(config, "num_hidden_layers", -1))
    if hidden_size != SMOLLM2_HIDDEN_SIZE:
        raise ValueError(f"original SmolLM2 hidden_size must be {SMOLLM2_HIDDEN_SIZE}, got {hidden_size}")
    if layer_count != SMOLLM2_LAYER_COUNT:
        raise ValueError(f"original SmolLM2 num_hidden_layers must be {SMOLLM2_LAYER_COUNT}, got {layer_count}")
    decoder = getattr(getattr(model, "model", None), "layers", None)
    if decoder is None or len(decoder) != SMOLLM2_LAYER_COUNT:
        raise ValueError("standard Llama model must expose exactly 30 model.layers")
    layer_object_ids = [id(layer) for layer in decoder]
    if len(set(layer_object_ids)) != SMOLLM2_LAYER_COUNT:
        raise ValueError("decoder layers must be thirty independent module objects")
    parameter_names = [name.lower() for name, _ in model.named_parameters(remove_duplicate=False)]
    forbidden = [name for name in parameter_names if "router" in name or "memory" in name or "recursive" in name or "shared_loop" in name]
    if forbidden:
        raise ValueError(f"standard baseline contains forbidden custom text parameters: {forbidden[:8]}")
    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    if input_embedding is None or output_embedding is None:
        raise ValueError("standard baseline must expose input and output embeddings")
    input_weight = getattr(input_embedding, "weight", None)
    output_weight = getattr(output_embedding, "weight", None)
    if input_weight is None or output_weight is None:
        raise ValueError("standard baseline embeddings must expose weights")
    tied = bool(input_weight.data_ptr() == output_weight.data_ptr())
    return {
        "model_type": model_type,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "architectures": architecture_names,
        "hidden_size": hidden_size,
        "num_hidden_layers": layer_count,
        "physical_decoder_layer_count": len(decoder),
        "independent_decoder_layers": True,
        "forbidden_custom_parameter_names": [],
        "embedding_lm_head_tied": tied,
        "vocab_size": int(getattr(config, "vocab_size", 0)),
    }


def _find_embedding(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    return model.get_input_embeddings()(ids)


class AudioSmolLM2Model(nn.Module):
    """Composite baseline model: frozen HTSAT plus trainable audio/text path."""

    def __init__(
        self,
        text_model: nn.Module,
        tokenizer: Any,
        htsat_wrapper: nn.Module,
        htsat_backbone: nn.Module,
        config: AudioSmolLM2Config | None = None,
    ) -> None:
        super().__init__()
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.htsat_wrapper = htsat_wrapper
        self.htsat_backbone = htsat_backbone
        self.text_contract = validate_original_smollm2(text_model)
        for parameter in self.text_model.parameters():
            parameter.requires_grad_(True)
        self.config_audio = config or AudioSmolLM2Config()
        if int(self.config_audio.encoder_dim) != 768:
            raise ValueError("audio baseline requires a 768-dimensional Mellow embedding")
        if int(self.config_audio.framewise_classes) != 527:
            raise ValueError("audio baseline requires Mellow c2l(527, 768)")
        if int(self.config_audio.downsample_kernel) != 8:
            raise ValueError("audio baseline requires 8x temporal pooling")
        if float(self.config_audio.projection_dropout) != 0.5:
            raise ValueError("audio baseline requires projection dropout=0.5")
        if int(self.text_contract["hidden_size"]) != int(self.config_audio.projection_dim):
            raise ValueError("audio bridge output dimension does not match text hidden size")
        self.bridge = AudioBridge(
            768,
            int(self.text_contract["hidden_size"]),
            self.config_audio.downsample_kernel,
            self.config_audio.projection_dropout,
        )
        self.last_labels: torch.Tensor | None = None
        self.last_prefix_length: int | None = None
        self.last_audio_tokens_per_clip: tuple[int, int] | None = None
        self._freeze_audio_except_c2l()
        self.separator_token_id = self._resolve_separator()

    def _freeze_audio_except_c2l(self) -> None:
        c2l = getattr(self.htsat_wrapper, "c2l", None)
        if not isinstance(c2l, nn.Linear) or int(c2l.in_features) != 527 or int(c2l.out_features) != 768:
            raise ValueError(f"audio baseline requires wrapper.c2l = Linear(527, 768), got {c2l}")
        for parameter in self.htsat_wrapper.parameters():
            parameter.requires_grad_(False)
        for parameter in self.htsat_backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in c2l.parameters():
            parameter.requires_grad_(True)
        self.htsat_wrapper.eval()
        self.htsat_backbone.eval()

    def _resolve_separator(self) -> int:
        # The audio contract uses the literal vocabulary token ``!`` between
        # the two 129-token audio prefixes.  Keep generic separator/eos/bos
        # IDs only as compatibility fallbacks for a tokenizer that cannot
        # resolve ``!``.
        candidates: list[int | None] = []
        try:
            token_id = self.tokenizer.convert_tokens_to_ids("!")
            if token_id is not None and int(token_id) >= 0:
                candidates.append(int(token_id))
        except Exception:
            pass
        candidates.extend([
            getattr(self.tokenizer, "sep_token_id", None),
            getattr(getattr(self.text_model, "config", None), "sep_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "bos_token_id", None),
        ])
        vocab_size = int(getattr(getattr(self.text_model, "config", None), "vocab_size", 0))
        for value in candidates:
            if value is not None and int(value) >= 0 and (not vocab_size or int(value) < vocab_size):
                return int(value)
        raise ValueError("unable to resolve a valid separator token id")

    def train(self, mode: bool = True) -> "AudioSmolLM2Model":
        super().train(mode)
        self.text_model.train(mode)
        self.bridge.train(mode)
        self.htsat_wrapper.eval()
        self.htsat_backbone.eval()
        c2l = getattr(self.htsat_wrapper, "c2l", None)
        if c2l is not None:
            c2l.train(mode)
        self._freeze_audio_except_c2l()
        if mode and c2l is not None:
            c2l.train(True)
            self._assert_training_contract()
        return self

    def _assert_training_contract(self) -> None:
        audit = self.trainable_parameter_audit()
        if not audit["training_mode_contract"]:
            raise RuntimeError(f"audio baseline training mode contract failed: {audit}")
        if audit["unexpected_audio_trainable_names"]:
            raise RuntimeError(f"unexpected trainable audio parameters: {audit}")

    def _waveform_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        embedding = _audio_forward(self.htsat_wrapper, waveform)
        if embedding.shape[-1] == 527:
            embedding = self.htsat_wrapper.c2l(embedding)
        return embedding

    def encode_audio(
        self,
        audio1: torch.Tensor,
        audio2: torch.Tensor | None = None,
        audio2_reused_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if audio1.ndim == 2:
            audio1 = audio1.unsqueeze(1)
        if audio2 is None:
            audio2 = audio1
        if audio2.ndim == 2:
            audio2 = audio2.unsqueeze(1)
        first = self._waveform_embedding(audio1)
        if audio2.data_ptr() == audio1.data_ptr() or (audio2_reused_mask is not None and bool(audio2_reused_mask.all())):
            second = first
        elif audio2_reused_mask is not None and bool(audio2_reused_mask.any()):
            mask = audio2_reused_mask.to(first.device).bool()
            second = first.clone()
            unique = ~mask
            if bool(unique.any()):
                second[unique] = self._waveform_embedding(audio2[unique])
        else:
            second = self._waveform_embedding(audio2)
        return self.bridge(first), self.bridge(second)

    def forward(
        self,
        *,
        audio1: torch.Tensor,
        audio2: torch.Tensor | None,
        text_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
        answer_lengths: torch.Tensor,
        answer_attention_mask: torch.Tensor | None = None,
        audio2_reused_mask: torch.Tensor | None = None,
    ) -> Any:
        audio_prefix1, audio_prefix2 = self.encode_audio(audio1, audio2, audio2_reused_mask)
        if tuple(audio_prefix1.shape[1:]) != (AUDIO_TOKENS_PER_CLIP, SMOLLM2_HIDDEN_SIZE) or tuple(audio_prefix2.shape[1:]) != (AUDIO_TOKENS_PER_CLIP, SMOLLM2_HIDDEN_SIZE):
            raise RuntimeError(f"audio baseline requires two [B,129,576] prefixes, got {tuple(audio_prefix1.shape)} and {tuple(audio_prefix2.shape)}")
        text_embeds = _find_embedding(self.text_model, text_ids)
        separator_ids = torch.full((text_ids.shape[0], 1), self.separator_token_id, dtype=torch.long, device=text_ids.device)
        separator = _find_embedding(self.text_model, separator_ids)
        inputs_embeds = torch.cat((audio_prefix1, separator, audio_prefix2, separator, text_embeds), dim=1)
        prefix_length = int(audio_prefix1.shape[1] + 1 + audio_prefix2.shape[1] + 1)
        if prefix_length != AUDIO_PREFIX_TOKENS:
            raise RuntimeError(f"audio baseline requires total prefix length {AUDIO_PREFIX_TOKENS}, got {prefix_length}")
        labels = build_labels(
            text_ids=text_ids,
            prompt_lengths=prompt_lengths,
            answer_lengths=answer_lengths,
            prefix_length=prefix_length,
        )
        self.last_audio_tokens_per_clip = (int(audio_prefix1.shape[1]), int(audio_prefix2.shape[1]))
        self.last_labels = labels.detach()
        self.last_prefix_length = prefix_length
        prefix_mask = torch.ones((inputs_embeds.shape[0], prefix_length), dtype=torch.long, device=inputs_embeds.device)
        attention_mask = torch.cat((prefix_mask, text_attention_mask), dim=1)
        if tuple(attention_mask.shape) != tuple(inputs_embeds.shape[:2]):
            raise AssertionError(f"attention mask/embedding shape mismatch: mask={tuple(attention_mask.shape)} embeds={tuple(inputs_embeds.shape[:2])}")
        if labels.shape[1] != inputs_embeds.shape[1]:
            raise AssertionError("labels and multimodal embeddings have different lengths")
        if inputs_embeds.shape[1] > int(self.config_audio.max_context_length):
            raise AssertionError(f"multimodal sequence length {inputs_embeds.shape[1]} exceeds max_context_length={self.config_audio.max_context_length}")
        return self.text_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, use_cache=False, return_dict=True)

    def trainable_parameter_audit(self) -> dict[str, Any]:
        c2l = getattr(self.htsat_wrapper, "c2l", None)
        trainable = [name for name, parameter in self.named_parameters() if parameter.requires_grad]
        audio_trainable = [(name, parameter) for name, parameter in self.htsat_wrapper.named_parameters() if parameter.requires_grad]
        unexpected_audio = [name for name, _ in audio_trainable if not name.startswith("c2l.")]
        layers = list(getattr(getattr(self.text_model, "model", None), "layers", []))
        text_parameters = list(self.text_model.parameters())
        text_trainable = any(name.startswith("text_model.") for name in trainable)
        all_text_trainable = bool(text_parameters) and all(parameter.requires_grad for parameter in text_parameters)
        bridge_trainable = any(name.startswith("bridge.") for name in trainable)
        c2l_trainable = isinstance(c2l, nn.Module) and any(parameter.requires_grad for parameter in c2l.parameters())
        modes = {
            "model_training": bool(self.training),
            "text_training": bool(self.text_model.training),
            "bridge_training": bool(self.bridge.training),
            "c2l_training": bool(c2l.training) if isinstance(c2l, nn.Module) else False,
            "htsat_wrapper_training": bool(self.htsat_wrapper.training),
            "htsat_backbone_training": bool(self.htsat_backbone.training),
        }
        return {
            "trainable_parameter_count": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "text_trainable": text_trainable,
            "all_text_trainable": all_text_trainable,
            "bridge_trainable": bridge_trainable,
            "c2l_trainable": c2l_trainable,
            "htsat_frozen": all(not parameter.requires_grad for parameter in self.htsat_backbone.parameters()),
            "unexpected_audio_trainable_names": unexpected_audio,
            "audio_trainable_names": [name for name, _ in audio_trainable],
            "standard_text_contract": self.text_contract,
            "decoder_layer_count": len(layers),
            "independent_decoder_layers": len({id(layer) for layer in layers}) == len(layers) == SMOLLM2_LAYER_COUNT,
            "has_router_parameters": any("router" in name.lower() for name in trainable),
            "training_modes": modes,
            "training_mode_contract": bool(
                self.training
                and self.text_model.training
                and self.bridge.training
                and modes["c2l_training"]
                and not self.htsat_wrapper.training
                and not self.htsat_backbone.training
                and text_trainable
                and all_text_trainable
                and bridge_trainable
                and c2l_trainable
                and not unexpected_audio
                and not any("router" in name.lower() for name in trainable)
            ),
            "trainable_names_sample": trainable[:20],
        }

    def runtime_gradient_audit(self) -> dict[str, Any]:
        layers = list(self.text_model.model.layers)
        layer_gradients = []
        for index, layer in enumerate(layers):
            finite = any(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in layer.parameters())
            layer_gradients.append({"layer": index, "finite_gradient": finite})
        input_embedding = self.text_model.get_input_embeddings()
        output_embedding = self.text_model.get_output_embeddings()
        embedding_parameters = list(input_embedding.parameters()) if input_embedding is not None else []
        output_parameters = list(output_embedding.parameters()) if output_embedding is not None else []
        c2l = getattr(self.htsat_wrapper, "c2l", None)
        bridge_gradients = {
            name: bool(parameter.grad is not None and torch.isfinite(parameter.grad).all())
            for name, parameter in self.bridge.named_parameters()
        }
        c2l_gradients = {
            name: bool(parameter.grad is not None and torch.isfinite(parameter.grad).all())
            for name, parameter in c2l.named_parameters()
        } if c2l is not None else {}
        embedding_gradients = bool(embedding_parameters) and all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in embedding_parameters)
        lm_head_gradients = bool(output_parameters) and all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in output_parameters)
        result = {
            "standard_text_model": self.text_contract,
            "decoder_layers": layer_gradients,
            "all_decoder_layers_have_finite_gradient": all(item["finite_gradient"] for item in layer_gradients),
            "embedding_has_finite_gradient": embedding_gradients,
            "lm_head_has_finite_gradient": lm_head_gradients,
            "embedding_lm_head_tied": bool(self.text_contract["embedding_lm_head_tied"]),
            "bridge_gradients": bridge_gradients,
            "c2l_gradients": c2l_gradients,
            "htsat_frozen_and_gradient_free": all(not parameter.requires_grad and parameter.grad is None for parameter in self.htsat_backbone.parameters()),
            "has_router_parameters": False,
        }
        required = (
            result["all_decoder_layers_have_finite_gradient"],
            result["embedding_has_finite_gradient"],
            result["lm_head_has_finite_gradient"],
            bool(bridge_gradients) and all(bridge_gradients.values()),
            bool(c2l_gradients) and all(c2l_gradients.values()),
            result["htsat_frozen_and_gradient_free"],
        )
        if not all(required):
            raise RuntimeError(f"original baseline runtime gradient audit failed: {result}")
        return result


def manifest_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_config(path: Path, config: AudioSmolLM2Config, extra: dict[str, Any] | None = None) -> None:
    payload = asdict(config)
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


__all__ = [
    "AUDIO_PREFIX_TOKENS",
    "AUDIO_TOKENS_PER_CLIP",
    "MAPPER_CONTRACT",
    "ORIGINAL_SMOLLM2_CONTRACT",
    "SMOLLM2_HIDDEN_SIZE",
    "SMOLLM2_LAYER_COUNT",
    "AudioSmolLM2Config",
    "AudioSmolLM2Model",
    "_load_mellow_wrapper",
    "manifest_sha256",
    "validate_original_smollm2",
    "write_config",
]
