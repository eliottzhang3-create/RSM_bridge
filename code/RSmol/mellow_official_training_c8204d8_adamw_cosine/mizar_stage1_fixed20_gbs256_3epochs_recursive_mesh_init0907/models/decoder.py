"""Mizar S1 decoder with the recursive MeSH text initialization."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as nnf

from contract import TEXT_MODEL_CONTRACT
from models.recursive_text import load_recursive_text_model


def get_decoder(name: str):
    if name == "Decoder":
        return DecoderModel
    raise Exception(f"the decoder model {name} is incorrect or not supported")


def downsample(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 32:
        return x
    clip_latent = x[:, 0, :].unsqueeze(1)
    pooled = nnf.avg_pool2d(x[:, 1:, :], kernel_size=(8, 1))
    return torch.concat((clip_latent, pooled), axis=1)


class Downsampler(nn.Module):
    def __init__(self, din, dout):
        super().__init__()

    def forward(self, x):
        return downsample(x)


class DecoderModel(nn.Module):
    """Keep the baseline 815-token interface around a recursive LM."""

    def __init__(self, text_decoder: str, prefix_length: int, freeze_decoder_weights: bool = True):
        super().__init__()
        self.prefix_length = prefix_length
        self.text_decoder = str(text_decoder)
        self.text_model_contract = TEXT_MODEL_CONTRACT
        self.is_recursive_mesh = True
        self.lm = load_recursive_text_model(self.text_decoder)
        embedding = self.lm.get_input_embeddings()
        self.lm_embedding_size = int(embedding.weight.shape[1])
        if self.lm_embedding_size != 576:
            raise ValueError(f"recursive text hidden size must be 576, got {self.lm_embedding_size}")
        self.separator_token_id = 0
        self.lora = False
        if freeze_decoder_weights:
            for parameter in self.lm.parameters():
                parameter.requires_grad = False

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(input_ids)

    def _separator(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        token = torch.full(
            (batch_size, 1), self.separator_token_id, dtype=torch.long, device=device
        )
        return self._embed_tokens(token).to(dtype=dtype)

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def generate_prefix_inference(self, daudio1, daudio2, texts_enc):
        audio_projections1 = downsample(daudio1).contiguous()
        audio_projections2 = downsample(daudio2).contiguous()
        dtext = self._embed_tokens(texts_enc["input_ids"]).contiguous()
        sep_embed = self._separator(dtext.shape[0], dtext.device, dtext.dtype)
        return torch.cat((audio_projections1, audio_projections2, sep_embed, dtext), dim=1)

    def forward(
        self,
        daudio1: torch.Tensor,
        daudio2: torch.Tensor,
        texts_enc: dict[str, torch.Tensor],
        tokens: dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        if mask is not None or labels is not None:
            raise ValueError("Mizar S1 uses attention_mask=None and external answer-only CE")

        audio_projections1 = downsample(daudio1).contiguous()
        audio_projections2 = downsample(daudio2).contiguous()
        dtext = self._embed_tokens(texts_enc["input_ids"]).contiguous()
        embedding_text = self._embed_tokens(tokens["input_ids"]).contiguous()
        if audio_projections1.shape[1] != 129 or audio_projections2.shape[1] != 129:
            raise RuntimeError("HTSAT must produce 129 pooled tokens per 10-second half")
        if dtext.shape[1] != 256 or embedding_text.shape[1] != 300:
            raise RuntimeError("Mizar S1 requires fixed prompt=256 and answer=300")
        sep_embed = self._separator(dtext.shape[0], dtext.device, dtext.dtype)
        prefix = torch.cat((audio_projections1, audio_projections2, sep_embed, dtext), dim=1)
        embedding_cat = torch.cat((prefix, embedding_text), dim=1)
        if prefix.shape[1] != 515 or embedding_cat.shape[1] != 815:
            raise RuntimeError(
                f"Mizar S1 requires prefix=515 and total=815, got {prefix.shape[1]} and {embedding_cat.shape[1]}"
            )
        self.last_layout = {
            "audio1": 129,
            "audio2": 129,
            "separators": 1,
            "prompt": 256,
            "prefix": int(prefix.shape[1]),
            "answer": 300,
            "total": int(embedding_cat.shape[1]),
            "attention_mask": None,
            "input_dtype": str(embedding_cat.dtype),
        }
        return self.lm(
            inputs_embeds=embedding_cat,
            labels=None,
            attention_mask=None,
            use_cache=False,
        )
