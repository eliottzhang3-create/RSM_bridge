"""Public-Mellow fixed-layout loss on the standard SmolLM2 audio model."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from audio_smollm2_135m_mellow.model import AudioSmolLM2Model as BaseAudioSmolLM2Model
from audio_smollm2_135m_mellow.model import AUDIO_PREFIX_TOKENS, AUDIO_TOKENS_PER_CLIP, SMOLLM2_HIDDEN_SIZE
from .data import ANSWER_TOKENS, PROMPT_TOKENS

MELLOW_TEXT_PREFIX_TOKENS = AUDIO_PREFIX_TOKENS + PROMPT_TOKENS
MELLOW_SEQUENCE_TOKENS = MELLOW_TEXT_PREFIX_TOKENS + ANSWER_TOKENS


class MellowFaithfulAudioSmolLM2Model(BaseAudioSmolLM2Model):
    """Use Mellow's fixed 129-token prompt and 250-token answer layout."""

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
            raise RuntimeError("Mellow-faithful training forbids audio embedding reuse")

        prefix1, prefix2 = self.encode_audio(audio1, audio2, audio2_reused_mask=None)
        expected = (AUDIO_TOKENS_PER_CLIP, SMOLLM2_HIDDEN_SIZE)
        if tuple(prefix1.shape[1:]) != expected or tuple(prefix2.shape[1:]) != expected:
            raise RuntimeError("Mellow audio prefix shape mismatch")
        embedding = self.text_model.get_input_embeddings()
        separator_ids = torch.full(
            (prompt_input_ids.shape[0], 1), int(self.separator_token_id),
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
        self.last_multimodal_sequence_length = int(inputs_embeds.shape[1])
        return SimpleNamespace(loss=loss, logits=output.logits)
