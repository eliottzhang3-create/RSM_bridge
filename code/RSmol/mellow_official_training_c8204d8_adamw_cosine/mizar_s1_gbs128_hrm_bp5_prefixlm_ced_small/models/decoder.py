"""Audio126 + SEP1 + prompt256 + answer300, PrefixLM383 and answer-only loss."""
import torch
from torch import nn
from models.recursive_text import load_recursive_text_model
from contract import (ANSWER_TOKENS,AUDIO_TOKENS,PREFIX_TOKENS,PROMPT_TOKENS,
    SEPARATOR_TOKEN_ID,TEXT_MODEL_CONTRACT,ATTENTION_POLICY,BP_STEPS)


def get_decoder(name):
    if name!='Decoder': raise ValueError(f'unsupported decoder{name!r}')
    return DecoderModel


class DecoderModel(nn.Module):
    def __init__(self,text_decoder,prefix_length,freeze_decoder_weights=True):
        super().__init__()
        if prefix_length!=PREFIX_TOKENS or freeze_decoder_weights:
            raise ValueError('HRM audio requires prefix383 and full LM training')
        self.prefix_length=prefix_length
        self.text_model_contract=TEXT_MODEL_CONTRACT
        self.is_hrm_prefixlm=True
        self.lm=load_recursive_text_model(text_decoder)
        self.lm_embedding_size=self.lm.config.hidden_size
        if self.lm_embedding_size!=576: raise ValueError('production HRM hidden must remain576')
        self.lora=False
        self.last_layout=None

    def _make_prefix(self,daudio1,daudio2,texts_enc):
        if daudio2 is not None or daudio1.shape[1:]!=(AUDIO_TOKENS,self.lm_embedding_size):
            raise ValueError('requires one mapped126-token audio sequence')
        if texts_enc['input_ids'].shape!=(daudio1.shape[0],PROMPT_TOKENS):
            raise ValueError('requires fixed256 prompt tokens')
        embed=self.lm.get_input_embeddings()
        prompt=embed(texts_enc['input_ids'])
        sep=torch.full((prompt.shape[0],1),SEPARATOR_TOKEN_ID,dtype=torch.long,device=prompt.device)
        return torch.cat((daudio1.contiguous(),embed(sep),prompt),dim=1)

    def prefix_validity(self,texts_enc):
        prompt=texts_enc['attention_mask']
        if prompt.ndim!=2 or prompt.shape[1]!=PROMPT_TOKENS:
            raise ValueError('prompt validity must be[B,256]')
        return torch.cat((torch.ones((prompt.shape[0],AUDIO_TOKENS+1),dtype=prompt.dtype,device=prompt.device),prompt),dim=1)

    def generate_prefix_inference(self,daudio1,daudio2,texts_enc):
        return self._make_prefix(daudio1,daudio2,texts_enc)

    def inference_logits(self,prefix,texts_enc,generated_ids=None):
        """No-cache inference matching training positions and prefix boundary."""
        validity=self.prefix_validity(texts_enc)
        if generated_ids is not None:
            if generated_ids.ndim!=2 or generated_ids.shape[0]!=prefix.shape[0]:
                raise ValueError('generated IDs must be[B,n]')
            prefix=torch.cat((prefix,self.lm.get_input_embeddings()(generated_ids)),dim=1)
            validity=torch.cat((validity,torch.ones_like(generated_ids)),dim=1)
        return self.lm(inputs_embeds=prefix,attention_mask=validity,prefix_length=PREFIX_TOKENS,
                       bp_steps=BP_STEPS,use_cache=False).logits

    def forward(self,daudio1,daudio2,texts_enc,tokens,mask=None,labels=None):
        if mask is not None or labels is not None:
            raise ValueError('use explicit PrefixLM validity and external answer-only loss')
        prefix=self._make_prefix(daudio1,daudio2,texts_enc)
        if tokens['input_ids'].shape!=(prefix.shape[0],ANSWER_TOKENS):
            raise ValueError('requires fixed300 answer tokens')
        if tokens['attention_mask'].shape!=tokens['input_ids'].shape:
            raise ValueError('answer validity shape mismatch')
        answer=self.lm.get_input_embeddings()(tokens['input_ids'])
        sequence=torch.cat((prefix,answer),dim=1)
        validity=torch.cat((self.prefix_validity(texts_enc),tokens['attention_mask']),dim=1)
        self.last_layout=dict(audio=126,separators=1,prompt=256,prefix=383,answer=300,total=683,
            attention_mask=ATTENTION_POLICY,padding_keys_masked=True,answer_start_query=382,
            bp_steps=5,input_dtype=str(sequence.dtype))
        return self.lm(inputs_embeds=sequence,attention_mask=validity,prefix_length=PREFIX_TOKENS,
                       bp_steps=BP_STEPS,labels=None,use_cache=False)
