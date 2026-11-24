"""Real two-process CPU DDP; audio-free setup uses exact production mapper AST."""
from __future__ import annotations
import ast
import copy
import datetime
import importlib.util
import json
from pathlib import Path
import sys
import sysconfig
import tempfile
import unittest
from contextlib import nullcontext

ROOT=Path(__file__).resolve().parents[1]
ROUTE=ROOT/'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small'
spec=importlib.util.spec_from_file_location('code',Path(sysconfig.get_path('stdlib'))/'code.py')
stdlib=importlib.util.module_from_spec(spec);spec.loader.exec_module(stdlib);sys.modules['code']=stdlib
sys.path[:0]=[str(ROUTE),str(ROOT/'code/RSmol')]
import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlamaConfig
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import contract as C
from text_hrm_h2l3.contract import target_config
from models.hrm_prefix import AudioPrefixHRMForCausalLM
from models.decoder import DecoderModel
from hrm_audit import prelude_gradient_audit
torch.set_num_threads(1)

scope=dict(torch=torch,nn=nn,F=F,PREFIX_TOKENS=383,ANSWER_TOKENS=300)
nodes=[n for n in ast.parse((ROUTE/'models/ced.py').read_text()).body
    if isinstance(n,ast.ClassDef) and n.name in ('CEDFrequencyMerger','CEDTemporalMapper')]
nodes += [n for n in ast.parse((ROUTE/'train.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='answer_loss']
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<unchanged production mapper and loss AST>','exec'),scope)


class Model(nn.Module):
    def __init__(self):
        super().__init__();torch.manual_seed(123)
        self.audio_encoder=nn.Module();self.audio_encoder.base=nn.Module()
        self.audio_encoder.base.mapper=scope['CEDTemporalMapper'](384,4,16,output_tokens=126,resample=False,dropout=0.)
        decoder=DecoderModel.__new__(DecoderModel);nn.Module.__init__(decoder)
        cfg=target_config(LlamaConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=30,num_attention_heads=2,
            num_key_value_heads=1,vocab_size=43,tie_word_embeddings=True,pad_token_id=17,eos_token_id=0,attention_dropout=0.,attention_bias=False))
        cfg._attn_implementation='sdpa';decoder.lm=AudioPrefixHRMForCausalLM(cfg)
        decoder.lm_embedding_size=16;decoder.last_layout=None
        self.caption_decoder=decoder
    def forward(self,batch):
        audio=self.audio_encoder.base.mapper(batch['grid'])
        return self.caption_decoder(audio,None,batch['input'],batch['answer'])


def batch(tokens):
    prompt=torch.full((1,256),17,dtype=torch.long);prompt[:,:3]=torch.tensor([3,4,0]);pm=torch.zeros_like(prompt);pm[:,:3]=1
    answer=torch.full((1,300),17,dtype=torch.long);am=torch.zeros_like(answer)
    answer[:,:tokens]=torch.tensor([5,6,7,0][:tokens]);am[:,:tokens]=1
    return dict(grid=torch.randn(1,126,4,384),input=dict(input_ids=prompt,attention_mask=pm),answer=dict(input_ids=answer,attention_mask=am))


def worker(rank,init_method,output):
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method=init_method,timeout=datetime.timedelta(seconds=90))
    try:
        model=Model();ddp=DistributedDataParallel(model,broadcast_buffers=False,find_unused_parameters=False)
        reference=copy.deepcopy(model)
        windows=[[batch(4),batch(3)],[batch(2),batch(4)]]
        for i,value in enumerate(windows[rank]):
            with ddp.no_sync() if i==0 else nullcontext():
                (scope['answer_loss'](ddp(value),value,17)/2).backward()
        for window in windows:
            for value in window:(scope['answer_loss'](reference(value),value,17)/4).backward()
        maximum=0.
        for p,q in zip(model.parameters(),reference.parameters()):
            if p.grad is None or not torch.allclose(p.grad,q.grad,atol=5e-6,rtol=5e-4):
                raise AssertionError('DDP shared HRM/mapper gradient differs from rank/micro mean reference')
            maximum=max(maximum,float((p.grad-q.grad).abs().max()))
        result=dict(passed=True,prelude_passed=prelude_gradient_audit(model)['passed'],max_abs_gradient_error=maximum,
            backend='gloo',world_size=2,torch=torch.__version__,cuda=False)
        (Path(output)/f'rank{rank}.json').write_text(json.dumps(result),encoding='utf-8')
    finally:dist.destroy_process_group()


class DDPTests(unittest.TestCase):
    def test_real_two_rank_prefixlm_mapper_shared_weights_and_prelude(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory(prefix='audio_hrm_ddp_') as tmp:
            root=Path(tmp)
            mp.spawn(worker,args=((root/'rendezvous').as_uri(),str(root)),nprocs=2,join=True)
            for rank in range(2):
                result=json.loads((root/f'rank{rank}.json').read_text())
                self.assertTrue(result['passed']);self.assertTrue(result['prelude_passed'])
                print(json.dumps(result),flush=True)


if __name__=='__main__':unittest.main(verbosity=2)
