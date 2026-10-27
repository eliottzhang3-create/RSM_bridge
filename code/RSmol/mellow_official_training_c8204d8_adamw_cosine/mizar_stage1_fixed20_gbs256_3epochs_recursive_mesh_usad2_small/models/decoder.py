import torch
import torch.nn as nn
from models.recursive_text import load_recursive_text_model
from contract import TEXT_MODEL_CONTRACT

from contract import ANSWER_TOKENS, AUDIO_TOKENS, PREFIX_TOKENS, PROMPT_TOKENS


def get_decoder(name: str):
    if name == "Decoder":
        return DecoderModel
    raise ValueError(f"unsupported decoder: {name}")


class DecoderModel(nn.Module):
    def __init__(self, text_decoder: str, prefix_length: int, freeze_decoder_weights: bool = True):
        super().__init__()
        if prefix_length != PREFIX_TOKENS:
            raise ValueError(f"USAD2-Small prefix must be {PREFIX_TOKENS}, got {prefix_length}")
        self.prefix_length = prefix_length
        self.text_decoder = str(text_decoder)
        self.text_model_contract = TEXT_MODEL_CONTRACT
        self.is_recursive_mesh = True
        self.lm = load_recursive_text_model(text_decoder)
        self.lm_embedding_size = int(self.lm.get_input_embeddings().weight.shape[1])
        if self.lm_embedding_size != 576:
            raise ValueError("MeSH hidden size must remain 576")
        self.lora = False
        if freeze_decoder_weights:
            for parameter in self.lm.parameters():
                parameter.requires_grad_(False)

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def _embeddings(self, texts_enc, tokens):
        embed = self.lm.get_input_embeddings()
        sep = torch.tensor([0], device=texts_enc["input_ids"].device)
        prompt = embed(texts_enc["input_ids"]).contiguous()
        answer = embed(tokens["input_ids"]).contiguous()
        separator = embed(sep).unsqueeze(0).expand(prompt.shape[0], -1, -1)
        return prompt, answer, separator

    def generate_prefix_inference(self, audio: torch.Tensor, texts_enc):
        prompt, _, separator = self._embeddings(texts_enc, texts_enc)
        if audio.shape[1] != AUDIO_TOKENS or prompt.shape[1] != PROMPT_TOKENS:
            raise RuntimeError("USAD2-Small requires audio=251 and prompt=256 tokens")
        return torch.cat((audio, separator, prompt), dim=1).contiguous()

    def forward(self, audio: torch.Tensor, texts_enc, tokens, mask=None, labels=None):
        prompt, answer, separator = self._embeddings(texts_enc, tokens)
        if (audio.shape[1] != AUDIO_TOKENS or prompt.shape[1] != PROMPT_TOKENS
                or answer.shape[1] != ANSWER_TOKENS):
            raise RuntimeError("USAD2-Small requires audio=251, prompt=256, answer=300 tokens")
        if mask is not None or labels is not None:
            raise ValueError("Mizar S1 uses attention_mask=None and external answer-only CE")
        prefix = torch.cat((audio, separator, prompt), dim=1)
        embedding_cat = torch.cat((prefix, answer), dim=1)
        self.last_layout = {"audio": 251, "separators": 1, "prompt": 256,
                            "prefix": int(prefix.shape[1]), "answer": 300,
                            "total": int(embedding_cat.shape[1]),
                            "attention_mask": None, "input_dtype": str(embedding_cat.dtype)}
        return self.lm(inputs_embeds=embedding_cat, labels=None,
                       attention_mask=None, use_cache=False)
