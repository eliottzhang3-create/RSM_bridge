"""Mellow official-runtime-faithful v2 fixed-layout SmolLM2 model."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_smollm2_135m_mellow.model import (
    AUDIO_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    MAPPER_CONTRACT,
    ORIGINAL_SMOLLM2_CONTRACT,
    SMOLLM2_HIDDEN_SIZE,
    AudioSmolLM2Model as BaseAudioSmolLM2Model,
)
from .data import ANSWER_TOKENS, PROMPT_TOKENS

MELLOW_TEXT_PREFIX_TOKENS = AUDIO_PREFIX_TOKENS + PROMPT_TOKENS
MELLOW_SEQUENCE_TOKENS = MELLOW_TEXT_PREFIX_TOKENS + ANSWER_TOKENS
OFFICIAL_SEPARATOR_TOKEN_ID = 0


class MellowOfficialFaithfulV2AudioSmolLM2Model(BaseAudioSmolLM2Model):
    """Match Mellow's train-mode frozen HTSAT and fixed separator ID 0."""

    def _freeze_audio_except_c2l(self) -> None:
        c2l = getattr(self.htsat_wrapper, "c2l", None)
        if not isinstance(c2l, nn.Linear) or int(c2l.in_features) != 527 or int(c2l.out_features) != 768:
            raise ValueError(f"official-faithful v2 requires wrapper.c2l = Linear(527, 768), got {c2l}")
        for parameter in self.htsat_wrapper.parameters():
            parameter.requires_grad_(False)
        for parameter in self.htsat_backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in c2l.parameters():
            parameter.requires_grad_(True)

    def _resolve_separator(self) -> int:
        vocab_size = int(getattr(getattr(self.text_model, "config", None), "vocab_size", 0))
        if vocab_size <= OFFICIAL_SEPARATOR_TOKEN_ID:
            raise ValueError(f"token ID 0 is outside text vocabulary size {vocab_size}")
        return OFFICIAL_SEPARATOR_TOKEN_ID

    def train(self, mode: bool = True) -> "MellowOfficialFaithfulV2AudioSmolLM2Model":
        # The parent implementation forces HTSAT back to eval. Calling the
        # nn.Module implementation directly reproduces Mellow's whole-model
        # train mode while requires_grad remains independently frozen.
        nn.Module.train(self, mode)
        self._freeze_audio_except_c2l()
        if mode:
            self._assert_training_contract()
        return self

    def trainable_parameter_audit(self) -> dict[str, Any]:
        audit = super().trainable_parameter_audit()
        modes = audit["training_modes"]
        audit["official_runtime_faithful_v2"] = True
        audit["separator_token_id"] = int(self.separator_token_id)
        audit["separator_is_official_zero"] = int(self.separator_token_id) == OFFICIAL_SEPARATOR_TOKEN_ID
        audit["training_mode_contract"] = bool(
            self.training
            and self.text_model.training
            and self.bridge.training
            and modes["c2l_training"]
            and modes["htsat_wrapper_training"]
            and modes["htsat_backbone_training"]
            and audit["text_trainable"]
            and audit["all_text_trainable"]
            and audit["bridge_trainable"]
            and audit["c2l_trainable"]
            and audit["htsat_frozen"]
            and not audit["unexpected_audio_trainable_names"]
            and not audit["has_router_parameters"]
            and audit["separator_is_official_zero"]
        )
        return audit

    def htsat_buffer_state(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.htsat_wrapper.named_buffers()
        }

    def load_htsat_buffer_state(self, state: dict[str, torch.Tensor]) -> None:
        current = dict(self.htsat_wrapper.named_buffers())
        if set(state) != set(current):
            missing = sorted(set(current) - set(state))
            unexpected = sorted(set(state) - set(current))
            raise RuntimeError(f"HTSAT buffer contract mismatch: missing={missing}, unexpected={unexpected}")
        with torch.no_grad():
            for name, target in current.items():
                source = state[name]
                if tuple(source.shape) != tuple(target.shape) or source.dtype != target.dtype:
                    raise RuntimeError(
                        f"HTSAT buffer shape/dtype mismatch for {name}: "
                        f"checkpoint={tuple(source.shape)}/{source.dtype}, model={tuple(target.shape)}/{target.dtype}"
                    )
                target.copy_(source.to(device=target.device))

    def separator_contract(self) -> dict[str, Any]:
        bang_id = self.tokenizer.convert_tokens_to_ids("!")
        return {
            "separator_token_id": OFFICIAL_SEPARATOR_TOKEN_ID,
            "separator_token_text": self.tokenizer.convert_ids_to_tokens(OFFICIAL_SEPARATOR_TOKEN_ID),
            "bang_token_id": None if bang_id is None else int(bang_id),
            "pad_token_id": int(self.tokenizer.pad_token_id),
            "pad_token_text": self.tokenizer.convert_ids_to_tokens(int(self.tokenizer.pad_token_id)),
            "separator_equals_pad": int(self.tokenizer.pad_token_id) == OFFICIAL_SEPARATOR_TOKEN_ID,
        }

    def forward(
        self,
        *,
        audio1: torch.Tensor,
        audio2: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        answer_input_ids: torch.Tensor,
        answer_attention_mask: torch.Tensor,
        audio2_reused_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> Any:
        if audio2 is None:
            raise ValueError("Mellow training always supplies audio2")
        if tuple(prompt_input_ids.shape[1:]) != (PROMPT_TOKENS,):
            raise RuntimeError("Mellow prompt must contain 129 tokens")
        if tuple(answer_input_ids.shape[1:]) != (ANSWER_TOKENS,):
            raise RuntimeError("Mellow answer must contain 250 tokens")
        if audio2_reused_mask is not None and bool(audio2_reused_mask.any()):
            raise RuntimeError("official-faithful v2 forbids audio embedding reuse")

        prefix1, prefix2 = self.encode_audio(audio1, audio2, audio2_reused_mask=None)
        expected = (AUDIO_TOKENS_PER_CLIP, SMOLLM2_HIDDEN_SIZE)
        if tuple(prefix1.shape[1:]) != expected or tuple(prefix2.shape[1:]) != expected:
            raise RuntimeError("Mellow audio prefix shape mismatch")
        embedding = self.text_model.get_input_embeddings()
        separator_ids = torch.full(
            (prompt_input_ids.shape[0], 1), OFFICIAL_SEPARATOR_TOKEN_ID,
            dtype=torch.long, device=prompt_input_ids.device,
        )
        inputs_embeds = torch.cat((
            prefix1, embedding(separator_ids), prefix2, embedding(separator_ids),
            embedding(prompt_input_ids), embedding(answer_input_ids),
        ), dim=1)
        if int(inputs_embeds.shape[1]) != MELLOW_SEQUENCE_TOKENS:
            raise RuntimeError(f"Mellow sequence length must be {MELLOW_SEQUENCE_TOKENS}")

        output = self.text_model(inputs_embeds=inputs_embeds, use_cache=False, return_dict=True)
        answer_logits = output.logits[:, MELLOW_TEXT_PREFIX_TOKENS - 1:-1]
        if tuple(answer_logits.shape[:2]) != tuple(answer_input_ids.shape):
            raise RuntimeError("Mellow answer-logit shift is incorrect")
        pad_id = int(self.tokenizer.pad_token_id)
        loss = F.cross_entropy(
            answer_logits.reshape(-1, answer_logits.shape[-1]),
            answer_input_ids.reshape(-1), ignore_index=pad_id,
        )

        labels = torch.full(
            (answer_input_ids.shape[0], MELLOW_SEQUENCE_TOKENS), -100,
            dtype=torch.long, device=answer_input_ids.device,
        )
        labels[:, MELLOW_TEXT_PREFIX_TOKENS:] = answer_input_ids.masked_fill(answer_input_ids.eq(pad_id), -100)
        self.last_labels = labels.detach()
        self.last_prefix_length = AUDIO_PREFIX_TOKENS
        self.last_prefix_lengths = None
        self.last_audio_tokens_per_clip = (int(prefix1.shape[1]), int(prefix2.shape[1]))
        self.last_prompt_input_ids = prompt_input_ids.detach()
        self.last_answer_input_ids = answer_input_ids.detach()
        self.last_prompt_attention_mask = prompt_attention_mask.detach()
        self.last_answer_attention_mask = answer_attention_mask.detach()
        self.last_separator_ids = separator_ids.detach()
        self.last_separator_pair_ids = torch.cat((separator_ids, separator_ids), dim=1).detach()
        self.last_multimodal_sequence_length = int(inputs_embeds.shape[1])
        return SimpleNamespace(loss=loss, logits=output.logits)


AudioSmolLM2Model = MellowOfficialFaithfulV2AudioSmolLM2Model

__all__ = [
    "AUDIO_PREFIX_TOKENS",
    "AUDIO_TOKENS_PER_CLIP",
    "MAPPER_CONTRACT",
    "OFFICIAL_SEPARATOR_TOKEN_ID",
    "ORIGINAL_SMOLLM2_CONTRACT",
    "SMOLLM2_HIDDEN_SIZE",
    "AudioSmolLM2Model",
    "MellowOfficialFaithfulV2AudioSmolLM2Model",
]
