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
        # Register after strict text loading; preserve all inherited RNG state.
        self._init_prompt_mix_mlp()
        self.last_layout = None

    def _init_prompt_mix_mlp(self):
        if hasattr(self, "prompt_mix_mlp"):
            raise RuntimeError("prompt mixture MLP is already initialized")
        weight = self.lm.get_input_embeddings().weight
        with torch.random.fork_rng(devices=[]):
            mlp = nn.Sequential(
                nn.Linear(self.lm_embedding_size, self.lm_embedding_size, device="cpu", dtype=weight.dtype),
                nn.GELU(),
                nn.Linear(self.lm_embedding_size, 3, device="cpu", dtype=weight.dtype))
            with torch.no_grad():
                mlp[0].bias.zero_()
                mlp[2].bias.zero_()
        self.prompt_mix_mlp = mlp.to(weight.device)
        self.last_prompt_mix_stats = None

    def _mix_audio(self, audio, information, prompt_embeddings, prompt_mask):
        if (prompt_mask is None or prompt_mask.shape != prompt_embeddings.shape[:2]
                or prompt_mask.device != prompt_embeddings.device):
            raise ValueError("prompt attention_mask must match prompt embeddings")
        if not bool(((prompt_mask == 0) | (prompt_mask == 1)).all().item()):
            raise ValueError("prompt attention_mask must be binary")
        valid = prompt_mask.to(dtype=prompt_embeddings.dtype)
        counts = valid.sum(dim=1, keepdim=True)
        if bool((counts == 0).any().item()):
            raise ValueError("every prompt must contain at least one valid token")
        if not isinstance(information, (tuple, list)) or len(information) != 2:
            raise ValueError("explicit (A4,A8) audio_information is required")
        for value in (audio, *information):
            if (not isinstance(value, torch.Tensor)
                    or tuple(value.shape) != (prompt_embeddings.shape[0], AUDIO_TOKENS, self.lm_embedding_size)
                    or value.dtype != prompt_embeddings.dtype or value.device != prompt_embeddings.device):
                raise ValueError("E/A4/A8 shape, dtype and device must match prompt embeddings")
        pooled = (prompt_embeddings * valid.unsqueeze(-1)).sum(dim=1) / counts
        alpha = torch.softmax(self.prompt_mix_mlp(pooled), dim=-1)
        # These exact weighted branches feed both the mixture and memory seeds.
        seeds = tuple(alpha[:, i, None, None] * value for i, value in enumerate((audio, *information)))
        mixed = seeds[0] + seeds[1] + seeds[2]
        self.last_prompt_mix_stats = None
        if self.lm.model.collect_listen_stats:
            with torch.no_grad():
                p = alpha.detach().float()
                rms = lambda x: x.detach().float().square().mean().sqrt()
                self.last_prompt_mix_stats = torch.cat((
                    p.mean(0), p.min(0).values, p.max(0).values,
                    torch.stack((-(p * p.clamp_min(1e-12).log()).sum(-1).mean(),
                                 *(rms(x) for x in (audio, *information, *seeds, mixed))))))
        return mixed, seeds

    def prompt_mix_diagnostics(self, scope="rank0_last_microbatch"):
        keys = tuple(f"alpha_{branch}_{stat}" for stat in ("mean", "min", "max")
                     for branch in ("E", "A4", "A8")) + (
            "entropy", "E_rms", "A4_rms", "A8_rms", "weighted_E_rms",
            "weighted_A4_rms", "weighted_A8_rms", "E_mix_rms")
        values = self.last_prompt_mix_stats
        return None if values is None else {"scope": scope, **dict(zip(keys, values.cpu().tolist()))}

    def _make_prefix(self, daudio1, daudio2, texts_enc, *, audio_information=None):
        if daudio2 is not None:
            raise ValueError("single_audio CED prefix has one already-mapped audio sequence")
        if daudio1.shape[1:] != (AUDIO_TOKENS, self.lm_embedding_size):
            raise ValueError(f"expected audio126 in LM dimensions, got {tuple(daudio1.shape)}")
        if texts_enc["input_ids"].ndim != 2 or texts_enc["input_ids"].shape[1] != PROMPT_TOKENS:
            raise ValueError("Mizar S1 requires fixed prompt=256")
        embed = self.lm.get_input_embeddings()
        dtext = embed(texts_enc["input_ids"]).contiguous()
        mixed, seeds = self._mix_audio(daudio1, audio_information, dtext, texts_enc.get("attention_mask"))
        sep_ids = torch.full((dtext.shape[0], 1), SEPARATOR_TOKEN_ID,
                             dtype=torch.long, device=dtext.device)
        return torch.cat((mixed.contiguous(), embed(sep_ids), dtext), dim=1), seeds

    def generate_prefix_inference(self, daudio1, daudio2, texts_enc, *, audio_information=None):
        return self._make_prefix(daudio1, daudio2, texts_enc, audio_information=audio_information)

    def forward(self, daudio1, daudio2, texts_enc, tokens, mask=None, labels=None, *, audio_information=None):
        if mask is not None or labels is not None:
            raise ValueError("Mizar S1 uses attention_mask=None and external answer-only CE")
        if tokens["input_ids"].shape[1] != ANSWER_TOKENS:
            raise ValueError("Mizar S1 requires fixed answer=300")
        prefix, seeds = self._make_prefix(daudio1, daudio2, texts_enc, audio_information=audio_information)
        answer = self.lm.get_input_embeddings()(tokens["input_ids"])
        embedding_cat = torch.cat((prefix, answer), dim=1)
        if prefix.shape[1] != PREFIX_TOKENS or embedding_cat.shape[1] != 683:
            raise RuntimeError("invalid CED Mizar prefix/sequence length")
        self.last_layout = {"audio": AUDIO_TOKENS, "separators": 1, "prompt": PROMPT_TOKENS,
                            "prefix": int(prefix.shape[1]), "answer": ANSWER_TOKENS,
                            "total": int(embedding_cat.shape[1]), "attention_mask": None,
                            "input_dtype": str(embedding_cat.dtype)}
        return self.lm(inputs_embeds=embedding_cat, labels=None, attention_mask=None, use_cache=False,
                       audio_memory_seeds=seeds)
