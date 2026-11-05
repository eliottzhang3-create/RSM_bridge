"""Mizar S1 single_audio layout: audio126 + SEP1 + prompt256 + answer300."""
import torch
from torch import nn
from models.recursive_text import load_recursive_text_model

from contract import (ANSWER_TOKENS, AUDIO_TOKENS, PREFIX_TOKENS, PROMPT_TOKENS,
                      SEPARATOR_TOKEN_ID, TEXT_MODEL_CONTRACT)


def get_decoder(name):
    if name == "Decoder":
        return DecoderModel
    raise ValueError(f"unsupported decoder {name!r}")


class DecoderModel(nn.Module):
    def __init__(self, text_decoder, prefix_length, freeze_decoder_weights=True):
        super().__init__()
        if prefix_length != PREFIX_TOKENS:
            raise ValueError("CED Mizar S1 requires prefix_length=383")
        self.prefix_length = prefix_length
        self.text_decoder = text_decoder.lower()
        self.text_model_contract = TEXT_MODEL_CONTRACT
        self.is_recursive_mesh = True
        self.lm = load_recursive_text_model(text_decoder)
        self.lm_embedding_size = self.lm.get_input_embeddings().weight.shape[1]
        self.lora = False
        if self.lm_embedding_size != 576:
            raise ValueError("recursive MeSH hidden size must remain 576")
        if freeze_decoder_weights:
            for param in self.lm.parameters():
                param.requires_grad = False
        self.last_layout = None

    def _make_prefix(self, daudio1, daudio2, texts_enc):
        if daudio2 is not None:
            raise ValueError("single_audio CED prefix has one already-mapped audio sequence")
        if daudio1.shape[1:] != (AUDIO_TOKENS, self.lm_embedding_size):
            raise ValueError(f"expected audio126 in LM dimensions, got {tuple(daudio1.shape)}")
        if texts_enc["input_ids"].shape[1] != PROMPT_TOKENS:
            raise ValueError("Mizar S1 requires fixed prompt=256")
        embed = self.lm.get_input_embeddings()
        dtext = embed(texts_enc["input_ids"]).contiguous()
        sep_ids = torch.full((dtext.shape[0], 1), SEPARATOR_TOKEN_ID,
                             dtype=torch.long, device=dtext.device)
        return torch.cat((daudio1.contiguous(), embed(sep_ids), dtext), dim=1)

    def generate_prefix_inference(self, daudio1, daudio2, texts_enc):
        return self._make_prefix(daudio1, daudio2, texts_enc)

    def forward(self, daudio1, daudio2, texts_enc, tokens, mask=None, labels=None):
        if mask is not None or labels is not None:
            raise ValueError("Mizar S1 uses attention_mask=None and external answer-only CE")
        if tokens["input_ids"].shape[1] != ANSWER_TOKENS:
            raise ValueError("Mizar S1 requires fixed answer=300")
        prefix = self._make_prefix(daudio1, daudio2, texts_enc)
        answer = self.lm.get_input_embeddings()(tokens["input_ids"])
        embedding_cat = torch.cat((prefix, answer), dim=1)
        if prefix.shape[1] != PREFIX_TOKENS or embedding_cat.shape[1] != 683:
            raise RuntimeError("invalid CED Mizar prefix/sequence length")
        self.last_layout = {"audio": AUDIO_TOKENS, "separators": 1, "prompt": PROMPT_TOKENS,
                            "prefix": int(prefix.shape[1]), "answer": ANSWER_TOKENS,
                            "total": int(embedding_cat.shape[1]), "attention_mask": None,
                            "input_dtype": str(embedding_cat.dtype)}
        return self.lm(inputs_embeds=embedding_cat, labels=None, attention_mask=None, use_cache=False)
