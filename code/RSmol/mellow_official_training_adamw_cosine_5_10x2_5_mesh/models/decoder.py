
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from enum import Enum
import json
from pathlib import Path
# from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM
from typing import Tuple, Optional, Union
from models.recursive_text import load_recursive_text_model
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print("Please install the 'peft' library to use this module.")

def get_decoder(name: str):
    if name == "Decoder":
        return DecoderModel
    else:
        raise Exception('The decoder model {} is incorrect or not supported'.format(name))
    
def downsample(x):
    if x.shape[1] == 32:
        return x
    clip_latent = x[:,0,:].unsqueeze(1)
    pooled = nnf.avg_pool2d(x[:,1:,:], kernel_size=(8,1))
    x = torch.concat((clip_latent,pooled),axis=1)
    return x


def _is_recursive_mesh_checkpoint(text_decoder: str) -> bool:
    """Detect the isolated mesh checkpoint from its config, never by model size."""
    path = Path(text_decoder).expanduser()
    config_path = path / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return config.get("mesh_architecture_contract") == "logical_30_physical_20_5_10x2_5"

class Downsampler(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        # self.fc = nn.Linear(din, dout)

    def forward(self, x):
        # x = self.fc(x)
        clip_latent = x[:,0,:].unsqueeze(1)
        # downsample the timesteps by 4. Downsample only the frame-level info
        pooled = nnf.avg_pool2d(x[:,1:,:], kernel_size=(8,1))
        # add clip-level latent back to audio
        x = torch.concat((clip_latent,pooled),axis=1)
        return x

class DecoderModel(nn.Module):
    def __init__(self, text_decoder: str, prefix_length: int, freeze_decoder_weights: bool = True,):
        super(DecoderModel, self).__init__()
        self.prefix_length = prefix_length
        self.text_decoder = str(text_decoder).lower()
        self.is_recursive_mesh = _is_recursive_mesh_checkpoint(text_decoder)
        # self.gpt = GPT2LMHeadModel.from_pretrained(text_decoder)
        if self.is_recursive_mesh:
            self.lm = load_recursive_text_model(text_decoder)
            self.lm_embedding_size = self.lm.get_input_embeddings().weight.shape[1]
            self.separator_token_id = 0
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(text_decoder)
        if self.is_recursive_mesh:
            pass
        elif "gpt2" in self.text_decoder:
            self.lm_embedding_size = self.lm.transformer.wte.weight.shape[1]
        elif "smollm2" in self.text_decoder:
            self.lm_embedding_size = self.lm.model.embed_tokens.weight.shape[1]
        else:
            raise ValueError(f"text decoder {self.text_decoder} not supported")

        # Keep the upstream route's disabled LoRA branch explicit. The mesh
        # route trains the full recursive text model together with Mellow.
        self.lora = False
        if self.lora:
            lora_config = LoraConfig(
                r=256,
                lora_alpha=256,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.lm = get_peft_model(self.lm, lora_config)
            self.lm.print_trainable_parameters()

        if freeze_decoder_weights:
            for parameter in self.lm.parameters():
                parameter.requires_grad = False

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.is_recursive_mesh:
            return self.lm.get_input_embeddings()(token_ids)
        if "gpt2" in self.text_decoder:
            return self.lm.transformer.wte(token_ids)
        if self.lora:
            return self.lm.base_model.model.model.embed_tokens(token_ids)
        return self.lm.model.embed_tokens(token_ids)

    def _separator_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        separator_id = self.separator_token_id if self.is_recursive_mesh else None
        if separator_id is None:
            if "gpt2" in self.text_decoder:
                separator_id = 50256
            elif "smollm2" in self.text_decoder:
                separator_id = 0
            else:
                raise ValueError(f"text decoder {self.text_decoder} not supported")
        separator = torch.tensor([separator_id], dtype=torch.long, device=device)
        return self._embed_tokens(separator).unsqueeze(0).repeat(batch_size, 1, 1)

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)
    
    def generate_prefix_inference(self, daudio1, daudio2, texts_enc):
        audio_projections1 = downsample(daudio1).contiguous()
        audio_projections2 = downsample(daudio2).contiguous()

        # separate token between two audios'
        if self.is_recursive_mesh:
            dtext = self._embed_tokens(texts_enc["input_ids"])
            dtext = dtext.contiguous()
            sep_embed = self._separator_embedding(dtext.shape[0], dtext.device)
        elif "gpt" in self.text_decoder:
            dtext = self.lm.transformer.wte(texts_enc['input_ids'])
            dtext = dtext.contiguous()
            sep_embed = self._separator_embedding(dtext.shape[0], dtext.device)
        elif "smollm2" in self.text_decoder:
            dtext = self._embed_tokens(texts_enc["input_ids"])
            dtext = dtext.contiguous()
            sep_embed = self._separator_embedding(dtext.shape[0], dtext.device)
        else:
            raise ValueError(f"text decoder {self.text_decoder} not supported")
        # Match the audio bridge dtype at the multimodal boundary.
        audio_projections1 = audio_projections1.to(dtype=dtext.dtype)
        audio_projections2 = audio_projections2.to(dtype=dtext.dtype)
        
        prefix = torch.cat((audio_projections1, sep_embed, audio_projections2, sep_embed, dtext), dim=1)
        return prefix

    def forward(self, daudio1: torch.Tensor, daudio2: torch.Tensor, texts_enc: torch.Tensor, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):

        if self.is_recursive_mesh:
            dtext = self._embed_tokens(texts_enc["input_ids"])
            dtext = dtext.contiguous()
            embedding_text = self._embed_tokens(tokens["input_ids"])
            sep_embed = self._separator_embedding(dtext.shape[0], dtext.device)
        elif "gpt2" in self.text_decoder:
            # input prompt
            dtext = self.lm.transformer.wte(texts_enc['input_ids'])
            dtext = dtext.contiguous()
            # output labels
            embedding_text = self.lm.transformer.wte(tokens['input_ids'])
            sep_embed = self._separator_embedding(dtext.shape[0], dtext.device)
        elif "smollm2" in self.text_decoder:
            # input prompt
            dtext = self._embed_tokens(texts_enc["input_ids"])
            dtext = dtext.contiguous()
            # output labels
            embedding_text = self._embed_tokens(tokens["input_ids"])
            sep_embed = self._separator_embedding(dtext.shape[0], dtext.device)
        else:
            raise ValueError(f"text decoder {self.text_decoder} not supported")
        
        audio_projections1 = downsample(daudio1).contiguous()
        audio_projections2 = downsample(daudio2).contiguous()
        # The converted MeSH checkpoint may retain BF16 weights while the
        # official audio bridge is initialized in FP32.  Match the text
        # embedding dtype before concatenating multimodal inputs.
        audio_projections1 = audio_projections1.to(dtype=embedding_text.dtype)
        audio_projections2 = audio_projections2.to(dtype=embedding_text.dtype)
        
        prefix = torch.cat((audio_projections1, sep_embed, audio_projections2, sep_embed, dtext), dim=1)
        embedding_cat = torch.cat((prefix, embedding_text), dim=1)
        # The trainer owns the answer-only CE reduction.  Do not ask the
        # language model to compute a second loss over the audio prefix and
        # prompt; logits are all that the Mellow loss path consumes.
        out = self.lm(
            inputs_embeds=embedding_cat,
            attention_mask=mask,
            use_cache=False,
            return_dict=True,
        )
        return out
