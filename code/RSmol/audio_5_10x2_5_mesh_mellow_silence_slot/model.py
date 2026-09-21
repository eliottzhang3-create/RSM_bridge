"""Fixed-two-slot variant of the Mellow + 5-10x2-5 MeSH audio model.

Every row owns exactly two 129-token audio prefixes and two separators.  For
a structurally single-audio row, the second prefix is produced from an exact
all-zero waveform created on the current GPU.  One zero waveform is encoded
per microbatch containing any single-audio rows; its embedding is expanded
before the trainable bridge so bridge dropout remains independent per row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.profiler import record_function

from audio_5_10x2_5_mesh_mellow.model import (
    AUDIO_DUAL_PREFIX_TOKENS,
    AUDIO_PREFIX_TOKENS,
    AUDIO_SINGLE_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    ARCHITECTURE_CONTRACT,
    MAPPER_CONTRACT,
    MESH_HIDDEN_SIZE,
    AudioMeshConfig,
    AudioMeshModel,
    _find_embedding,
    _load_mellow_wrapper,
    build_labels,
)


SILENCE_SLOT_ARCHITECTURE_CONTRACT = (
    "logical_30_physical_20_5_10x2_5_mesh_audio_mellow_"
    "fixed260_runtime_silence_second_slot"
)
FIXED_PREFIX_TOKEN_CONTRACT = {"single": AUDIO_PREFIX_TOKENS, "dual": AUDIO_PREFIX_TOKENS}


@dataclass
class AudioMeshSilenceSlotConfig(AudioMeshConfig):
    architecture_contract: str = SILENCE_SLOT_ARCHITECTURE_CONTRACT
    compact_single_audio_prefix: bool = False


class AudioMeshSilenceSlotModel(AudioMeshModel):
    """MeSH audio model with an always-materialized 260-token prefix."""

    def __init__(self, *args: Any, config: AudioMeshSilenceSlotConfig | None = None, **kwargs: Any) -> None:
        selected = config or AudioMeshSilenceSlotConfig()
        if selected.compact_single_audio_prefix:
            raise ValueError("silence-slot route forbids compact single-audio prefixes")
        super().__init__(*args, config=selected, **kwargs)
        self.last_audio_slot_audit: dict[str, Any] = {}

    @staticmethod
    def _cpu_mask(mask: torch.Tensor | None, *, batch: int, name: str) -> torch.Tensor:
        if mask is None:
            raise ValueError(f"fixed silence-slot route requires {name}")
        value = mask.detach().to(device="cpu", dtype=torch.bool)
        if value.ndim != 1 or int(value.shape[0]) != batch:
            raise ValueError(f"{name} must have shape [{batch}], got {tuple(value.shape)}")
        return value

    def encode_audio(
        self,
        audio1: torch.Tensor,
        audio2: torch.Tensor | None,
        silence_second_slot_mask: torch.Tensor,
        same_real_audio_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if audio1.ndim == 2:
            audio1 = audio1.unsqueeze(1)
        if audio1.ndim != 3:
            raise ValueError(f"audio1 must be [B,C,T], got {tuple(audio1.shape)}")
        batch = int(audio1.shape[0])
        silence_cpu = self._cpu_mask(
            silence_second_slot_mask, batch=batch, name="silence_second_slot_mask"
        )
        same_cpu = self._cpu_mask(
            same_real_audio_mask, batch=batch, name="same_real_audio_mask"
        )
        if bool((silence_cpu & same_cpu).any()):
            raise ValueError("a second slot cannot be both runtime silence and real-audio reuse")
        distinct_cpu = ~(silence_cpu | same_cpu)
        silence_count = int(silence_cpu.sum().item())
        same_count = int(same_cpu.sum().item())
        distinct_count = int(distinct_cpu.sum().item())

        if distinct_count:
            if audio2 is None:
                raise ValueError("distinct real second-audio rows require audio2")
            if audio2.ndim == 2:
                audio2 = audio2.unsqueeze(1)
            if audio2.ndim != 3 or int(audio2.shape[0]) != batch:
                raise ValueError(f"audio2 must be [B,C,T], got {tuple(audio2.shape)}")

        with record_function("audio/waveform_embedding_audio1"):
            first = self._waveform_embedding(audio1)

        second_inputs: list[torch.Tensor] = []
        silence_offset: int | None = None
        if silence_count:
            # Runtime-only: no CPU tensor, data-file member, mmap row, or H2D
            # transfer is created for silence.  The waveform is exact zeros.
            silence_offset = 0
            second_inputs.append(torch.zeros_like(audio1[:1]))
        distinct_indices_cpu = torch.nonzero(distinct_cpu, as_tuple=False).flatten()
        if distinct_count:
            assert audio2 is not None
            distinct_indices_device = distinct_indices_cpu.to(audio2.device)
            second_inputs.append(audio2.index_select(0, distinct_indices_device))

        encoded_second_inputs: torch.Tensor | None = None
        if second_inputs:
            with record_function("audio/waveform_embedding_audio2_runtime_silence_and_distinct"):
                encoded_second_inputs = self._waveform_embedding(torch.cat(second_inputs, dim=0))

        # Start from audio1 embeddings: these are exactly correct for explicit
        # same-real-audio dual rows.  Replace only silence and distinct rows.
        second = first
        cursor = 0
        if silence_count:
            assert encoded_second_inputs is not None and silence_offset == 0
            silence_embedding = encoded_second_inputs[0:1]
            silence_device = silence_cpu.to(first.device).view(batch, 1, 1)
            second = torch.where(silence_device, silence_embedding.expand(batch, -1, -1), second)
            cursor = 1
        if distinct_count:
            assert encoded_second_inputs is not None
            distinct_indices_device = distinct_indices_cpu.to(first.device)
            distinct_embeddings = encoded_second_inputs[cursor:cursor + distinct_count]
            second = second.index_copy(0, distinct_indices_device, distinct_embeddings)

        projected_first = self.bridge(first)
        # Expansion happens before this call.  Dropout in the trainable bridge
        # therefore samples independently for every silence-bearing row.
        projected_second = self.bridge(second)
        self.last_audio_slot_audit = {
            "batch_size": batch,
            "silence_second_slot_rows": silence_count,
            "same_real_audio_rows": same_count,
            "distinct_real_audio_rows": distinct_count,
            "all_rows_classified": silence_count + same_count + distinct_count == batch,
            "runtime_silence_waveforms_created": 1 if silence_count else 0,
            "runtime_silence_shape": [1, int(audio1.shape[1]), int(audio1.shape[2])] if silence_count else None,
            "runtime_silence_constructor": "torch.zeros_like(audio1[:1])" if silence_count else None,
            "second_encoder_input_batch_size": (1 if silence_count else 0) + distinct_count,
            "same_real_audio_reused_first_embedding": True,
            "fixed_prefix_tokens": AUDIO_PREFIX_TOKENS,
        }
        return projected_first, projected_second

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
        silence_second_slot_mask: torch.Tensor | None = None,
        same_real_audio_mask: torch.Tensor | None = None,
    ) -> Any:
        audio_prefix1, audio_prefix2 = self.encode_audio(
            audio1,
            audio2,
            silence_second_slot_mask,
            same_real_audio_mask,
        )
        expected_shape = (AUDIO_TOKENS_PER_CLIP, MESH_HIDDEN_SIZE)
        for name, prefix in (("audio1", audio_prefix1), ("audio2", audio_prefix2)):
            if tuple(prefix.shape[1:]) != expected_shape:
                raise RuntimeError(
                    f"{name} prefix must be [B,{expected_shape[0]},{expected_shape[1]}], "
                    f"got {tuple(prefix.shape)}"
                )

        with record_function("mesh/text_and_fixed260_prefix"):
            text_embeds = _find_embedding(self.mesh_model, text_ids)
            separator = _find_embedding(
                self.mesh_model,
                torch.full(
                    (text_ids.shape[0], 1),
                    self.separator_token_id,
                    dtype=torch.long,
                    device=text_ids.device,
                ),
            )
            inputs_embeds = torch.cat(
                (audio_prefix1, separator, audio_prefix2, separator, text_embeds), dim=1
            )
        prefix_length = int(audio_prefix1.shape[1] + 1 + audio_prefix2.shape[1] + 1)
        if prefix_length != AUDIO_PREFIX_TOKENS:
            raise RuntimeError(f"fixed silence-slot prefix must be 260 tokens, got {prefix_length}")
        labels = build_labels(
            text_ids=text_ids,
            prompt_lengths=prompt_lengths,
            answer_lengths=answer_lengths,
            prefix_length=prefix_length,
        )
        prefix_mask = torch.ones(
            (inputs_embeds.shape[0], prefix_length),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        attention_mask = torch.cat(
            (prefix_mask, text_attention_mask.to(device=inputs_embeds.device)), dim=1
        )
        if tuple(attention_mask.shape) != tuple(inputs_embeds.shape[:2]):
            raise AssertionError("attention mask and fixed-prefix embeddings have different shapes")
        if tuple(labels.shape) != tuple(attention_mask.shape):
            raise AssertionError("labels and fixed-prefix attention mask have different shapes")
        if int(inputs_embeds.shape[1]) > int(self.config_audio.max_context_length):
            raise AssertionError(
                f"multimodal sequence length {inputs_embeds.shape[1]} exceeds "
                f"max_context_length={self.config_audio.max_context_length}"
            )
        self.last_labels = labels.detach()
        self.last_prefix_length = prefix_length
        self.last_prefix_lengths = torch.full(
            (text_ids.shape[0],), prefix_length, dtype=torch.long, device="cpu"
        )
        self.last_audio_tokens_per_clip = (
            int(audio_prefix1.shape[1]), int(audio_prefix2.shape[1])
        )
        return self.mesh_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )


__all__ = [
    "AUDIO_DUAL_PREFIX_TOKENS",
    "AUDIO_PREFIX_TOKENS",
    "AUDIO_SINGLE_PREFIX_TOKENS",
    "AUDIO_TOKENS_PER_CLIP",
    "ARCHITECTURE_CONTRACT",
    "FIXED_PREFIX_TOKEN_CONTRACT",
    "MAPPER_CONTRACT",
    "MESH_HIDDEN_SIZE",
    "SILENCE_SLOT_ARCHITECTURE_CONTRACT",
    "AudioMeshSilenceSlotConfig",
    "AudioMeshSilenceSlotModel",
    "_load_mellow_wrapper",
]
