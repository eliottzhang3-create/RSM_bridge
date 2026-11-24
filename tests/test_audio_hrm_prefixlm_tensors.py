"""Real CPU HRM/SDPA/mapper/autograd/audio tests; no production weights or CUDA."""
from __future__ import annotations
import ast
import copy
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import sysconfig
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
ROUTE=ROOT/'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small'
spec=importlib.util.spec_from_file_location('code',Path(sysconfig.get_path('stdlib'))/'code.py')
stdlib_code=importlib.util.module_from_spec(spec);spec.loader.exec_module(stdlib_code);sys.modules['code']=stdlib_code
sys.path[:0]=[str(ROUTE),str(ROOT/'code/RSmol')]
MISSING=[n for n in ('torch','transformers','safetensors','torchaudio','soundfile','numpy') if not importlib.util.find_spec(n)]
if not MISSING:
    import torch
    import numpy as np
    import soundfile as sf
    from transformers import LlamaConfig,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    import contract as C
    import train as T
    from models.hrm_prefix import AudioPrefixHRMForCausalLM,prefixlm_mask
    from models.recursive_text import load_recursive_text_model
    from models.decoder import DecoderModel
    from models.ced import CEDSmallWrapper,CEDTemporalMapper
    from models.mellow import Mellow
    from optim import build_optimizer,GradNormTracker
    from checkpoint_state import validate_resume_model_state,optimizer_parameter_inventory
    from text_hrm_h2l3.contract import target_config,canonical_hash,file_inventory
    from text_hrm_h2l3.model import SmolHRMForCausalLM
    from text_hrm_reinit3600_bp5.train import build_binding
    from hrm_audit import recursive_backward_audit,prelude_gradient_audit
    torch.set_num_threads(1)


def lm():
    torch.manual_seed(123)
    cfg=LlamaConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=30,num_attention_heads=2,
        num_key_value_heads=1,vocab_size=43,tie_word_embeddings=True,pad_token_id=17,eos_token_id=0,
        rms_norm_eps=1e-5,rope_theta=100000.,attention_dropout=0.,attention_bias=False)
    cfg=target_config(cfg);cfg._attn_implementation='sdpa'
    model=AudioPrefixHRMForCausalLM(cfg).float()
    with torch.no_grad():
        for i,layer in enumerate(model.model.layers[5:15]): layer.self_attn.gate_proj.weight.fill_(.003*(i+1))
    return model


def fixture():
    # Actual Mellow/Decoder/HRM and actual mapper, with a small frozen CPU encoder
    # fixture replacing production CED asset loading and waveform->grid only.
    nn=torch.nn
    wrapper=CEDSmallWrapper.__new__(CEDSmallWrapper);nn.Module.__init__(wrapper)
    wrapper.ced=nn.Linear(1,1,bias=False).eval()
    for p in wrapper.ced.parameters():p.requires_grad=False
    wrapper.mapper=CEDTemporalMapper(384,4,16,output_tokens=126,resample=False,dropout=.05)
    wrapper.mapper_type='freq_merge_126'
    wrapper.geometry={'fixture':True};wrapper.last_audio_layout=None
    wrapper._waveform_to_mel=lambda wave:wave
    wrapper._forward_hidden=lambda value:torch.ones(value.shape[0],504,384)*value.mean(dim=1).view(-1,1,1)
    wrapper._hidden_to_temporal_grid=lambda hidden:hidden.reshape(hidden.shape[0],2,4,63,384).permute(0,1,3,2,4).reshape(hidden.shape[0],126,4,384)
    # The original wrapper enforces production576 output. Its identical mapper
    # path below permits hidden16 for the small-model task-gradient tests.
    def audio_forward(waveform):
        with torch.no_grad():grid=wrapper._hidden_to_temporal_grid(wrapper._forward_hidden(waveform))
        embedding=wrapper.mapper(grid)
        return {'embedding':embedding,'clipwise_output':embedding.new_zeros(embedding.shape[0],527)}
    wrapper.forward=audio_forward
    model=Mellow.__new__(Mellow);nn.Module.__init__(model)
    from models.mellow import AudioEncoder
    encoder=AudioEncoder.__new__(AudioEncoder);nn.Module.__init__(encoder)
    encoder.base=wrapper;encoder.projection=nn.Identity();model.audio_encoder=encoder
    decoder=DecoderModel.__new__(DecoderModel);nn.Module.__init__(decoder)
    decoder.lm=lm();decoder.prefix_length=383;decoder.lm_embedding_size=16;decoder.lora=False
    decoder.text_model_contract=C.TEXT_MODEL_CONTRACT;decoder.is_hrm_prefixlm=True;decoder.last_layout=None
    model.caption_decoder=decoder
    return model


def inputs(batch=1):
    prompt=torch.full((batch,256),17,dtype=torch.long);prompt[:,:4]=torch.tensor([3,4,5,0])
    pm=torch.zeros_like(prompt);pm[:,:4]=1
    answer=torch.full((batch,300),17,dtype=torch.long);answer[:,:4]=torch.tensor([6,7,8,0])
    am=torch.zeros_like(answer);am[:,:4]=1
    return dict(audio1=torch.ones(batch,16),audio2=torch.ones(batch,16),
        input=dict(input_ids=prompt,attention_mask=pm),answer=dict(input_ids=answer,attention_mask=am))


def tokenizer():
    vocab={'<|endoftext|>':0,'[UNK]':1,'hello':2,'!':17}|{f'w{i}':i for i in range(3,43) if i!=17}
    backend=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));backend.pre_tokenizer=Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend,eos_token='<|endoftext|>',pad_token='!',unk_token='[UNK]')


def source_checkpoint(root):
    path=Path(root)/'checkpoint-step-009244';path.mkdir()
    model=lm();model.config.architectures=['SmolHRMForCausalLM']
    # HF would write the audio class name; save raw tensors with original text class metadata.
    from safetensors.torch import save_file
    model.config.to_json_file(path/'config.json')
    state={n:v.detach().clone() for n,v in model.state_dict().items() if n!='lm_head.weight'}
    save_file(state,str(path/'model.safetensors'));tokenizer().save_pretrained(path)
    binding=build_binding(initialization={'source_step':3600},data={'fixture':True},environment={'fixture':True},production=False)
    C.write_json(path/'binding.json',binding)
    (path/'training_state.pt').write_bytes(b'never deserialize this fixture')
    C.write_json(path/'checkpoint_complete.json',dict(complete=True,step=9244,route=C.TEXT_MODEL_CONTRACT,
        architecture=C.TEXT_MODEL_ARCHITECTURE,production=False,format='hrm_text_reinit3600_full_checkpoint_v1',
        world_size=8,binding_sha256=canonical_hash(binding),files=file_inventory(path,include_state=True)))
    return path,model


def config():
    contract=dict(route=C.ROUTE,attention_mask=C.ATTENTION_POLICY,bp_steps=5,
        geometry=C.geometry(1477492),fixture=True)
    return dict(mode='formal',contract=contract,contract_sha256=C.digest(contract))


def update(model,opt,scheduler,tracker,batch=None):
    opt.zero_grad(set_to_none=True)
    output=model(batch or inputs());loss=T.answer_loss(output,batch or inputs(),17)
    loss.backward();tracker.track_and_clip_(model.named_parameters());opt.step();scheduler.step()


@unittest.skipIf(MISSING,'Real tensor/audio dependencies missing:'+','.join(MISSING))
class TensorTests(unittest.TestCase):
    def test_actual_training_three_epoch_loop_and_full_checkpoint_cpu_harness(self):
        """Production loop with real HRM/autograd; CUDA/DDP/data source simulated."""
        from contextlib import nullcontext
        from data.audiotext_dataset import collate_fn
        model=fixture();model.caption_decoder.lm.audio_initialization_audit={'fixture':True}
        initial_state={name:value.detach().clone() for name,value in model.state_dict().items()}
        class Dataset:
            pad_id=17
            audio_paths=NS(local=True,read_files={})
            def __init__(self,*args,**kwargs):pass
            def __len__(self):return 128
            def __getitem__(self,index):
                b=inputs()
                return dict(audio1=b['audio1'][0],audio2=b['audio2'][0],input=b['input'],answer=b['answer'],row_index=index,source='fixture')
        class DDP:
            def __init__(self,module,**kwargs):self.module=module
            def __call__(self,value):return self.module(value)
            def no_sync(self):return nullcontext()
        class Gate:
            info=None;complete=True;wait_seconds=0.
            def __init__(self,*args,**kwargs):pass
            def wait_for_batch(self,*args):pass
            def wait_complete(self):pass
        class Scaler:
            def __init__(self,**kwargs):pass
            def scale(self,loss):return loss
            def unscale_(self,opt):pass
            def step(self,opt):opt.step()
            def update(self):pass
            def is_enabled(self):return True
            def get_scale(self):return 1.
            def state_dict(self):return {'scale':1.}
            def load_state_dict(self,state):
                if state!={'scale':1.}:raise ValueError('fixture Scaler state differs')
        def gather(states,state):states[:]=[copy.deepcopy(state) for _ in range(8)]
        rng=lambda device:dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.get_rng_state())
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);prepared=run/'prepared.json'
            C.write_json(prepared,dict(mode='formal',metadata='fixture',stage_root='fixture',read_policy='shared_until_complete',
                probe_rows={},expected_first8_rows={str(i):list(range(i*16,i*16+16)) for i in range(8)}))
            cfg=config();cfg['contract'].update(geometry=C.geometry(128),code_sha256=C.code_digest(),recursive_model_source_sha256=C.recursive_model_source_sha256(),
                text_model_contract=C.TEXT_MODEL_CONTRACT,text_model_dir='fixture',recursive_text_checkpoint={'config':{'vocab_size':43}},ced_model_dir='fixture',prefix=[126,1,256],packages={'fixture':True})
            cfg.update(contract_sha256=C.digest(cfg['contract']),run_dir=str(run),prepared_report=str(prepared),max_optimizer_steps=3,
                resume_checkpoint='',checkpoint_steps=[1,2,3])
            # Sampler row-identity logic remains exercised using known contiguous batches.
            class Sampler:
                start_index=0
                def __init__(self,*args,**kwargs):pass
                def set_epoch(self,epoch):pass
                def __iter__(self):return iter(range(self.start_index,16))
                def __len__(self):return 16-self.start_index
            with ExitStack() as stack:
                for name,value in [('AudioTextDataset',Dataset),('Mellow',lambda **kw:model),('StagingGate',Gate),
                    ('DistributedDataParallel',DDP),('CustomDistributedSampler',Sampler),('rng_state',rng),
                    ('audit_checkpoint_tokenizer',lambda *a:{}),('inspect_tokenizer',lambda *a:dict(pad_id=17,sep_id=0,eos_id=0,ignore_index=17,tokenizer_vocab_rows=43,embedding_rows=43)),
                    ('validate_ced_assets',lambda *a:{'temporal_tokens':126})]:
                    stack.enter_context(patch.object(T,name,value))
                stack.enter_context(patch('torch.cuda.amp.GradScaler',Scaler))
                for name in ('manual_seed','set_rng_state'):stack.enter_context(patch('torch.cuda.'+name))
                stack.enter_context(patch('torch.cuda.get_device_name',return_value='CPU_SIMULATION'))
                stack.enter_context(patch('torch.cuda.max_memory_allocated',return_value=0))
                stack.enter_context(patch.object(T.dist,'all_reduce'))
                stack.enter_context(patch.object(T.dist,'barrier'))
                stack.enter_context(patch.object(T.dist,'all_gather_object',side_effect=gather))
                stack.enter_context(patch.object(T.dist,'get_rank',return_value=0))
                stack.enter_context(patch.object(T.dist,'get_world_size',return_value=8))
                T.train(cfg,0,torch.device('cpu'))
                uninterrupted={name:value.detach().clone() for name,value in model.state_dict().items()}
                # The synthetic encoder has a local Python forward closure;
                # deepcopy would keep that closure bound to the old mapper.
                model=fixture();model.load_state_dict(initial_state,strict=True)
                model.caption_decoder.lm.audio_initialization_audit={'fixture':True}
                cfg['resume_checkpoint']=str(run/'checkpoints/model--step-1.ckpt')
                T.train(cfg,0,torch.device('cpu'))
                for name,value in model.state_dict().items():
                    torch.testing.assert_close(value,uninterrupted[name],rtol=0,atol=0)
            result=json.loads((run/'training_report.json').read_text())
            self.assertEqual(result['status'],'PASS');self.assertEqual(result['optimizer_steps'],3)
            self.assertTrue(json.loads((run/'first_gradient_rank0.json').read_text())['prelude']['passed'])
            self.assertTrue(json.loads((run/'first_recursive_backward_rank0.json').read_text())['passed'])
            saved=torch.load(run/'checkpoints/model--epo-3.ckpt',map_location='cpu',weights_only=False)
            self.assertEqual(saved['epoch'],3);self.assertEqual(saved['next_optimizer_step'],0)
            validate_resume_model_state(saved,model,build_optimizer(model,3)[0],cfg)
            metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([m['step'] for m in metrics],[1,2,3,2,3])

    def test_mask_all_four_blocks_padding_and_query382(self):
        valid=torch.ones(2,683,dtype=torch.long);valid[0,131:383]=0;valid[:,387:]=0
        mask=prefixlm_mask(valid,383,batch=2,length=683,dtype=torch.float32,device='cpu')
        for b in range(2):
            for q in (0,126,127,130,131,382,383,384,682):
                for k in (0,126,127,130,131,382,383,384,386,387,682):
                    expected=bool(valid[b,k]) and (k<383 or (q>=383 and k<=q))
                    self.assertEqual(float(mask[b,0,q,k])==0.,expected,(b,q,k))
        self.assertTrue(bool((mask[0,0,382,:131]==0).all()))
        self.assertTrue(bool((mask[0,0,:383,383:]<0).all()))
        for kw in ({'prefix_length':None},{'prefix_length':684},{'prefix_length':True}):
            with self.assertRaises(ValueError):prefixlm_mask(valid,**kw,batch=2,length=683,dtype=torch.float32,device='cpu')

    def test_prefix_future_answer_padding_and_inference_numerical_parity(self):
        model=fixture().eval();batch=inputs();decoder=model.caption_decoder
        with torch.no_grad():
            baseline=model(batch).logits
            self.assertTrue(T.prefixlm_inference_audit(model,batch,baseline)['passed'])
            changed=copy.deepcopy(batch);changed['answer']['input_ids'][:,:4]=torch.tensor([20,21,22,0])
            other=model(changed).logits
            torch.testing.assert_close(baseline[:,:383],other[:,:383],rtol=0,atol=0)
            changed=copy.deepcopy(batch);changed['answer']['input_ids'][:,2]=24
            other=model(changed).logits
            torch.testing.assert_close(baseline[:,:385],other[:,:385],rtol=0,atol=0)
            changed=copy.deepcopy(batch)
            changed['input']['input_ids'][:,4:255]=22 # Excludes query382's token.
            changed['answer']['input_ids'][:,4:]=23
            other=model(changed).logits
            torch.testing.assert_close(baseline[:,382:387],other[:,382:387],rtol=0,atol=0)
            audio=model.audio_encoder(torch.cat((batch['audio1'],batch['audio2']),dim=-1))[0]
            prefix=decoder.generate_prefix_inference(audio,None,batch['input'])
            first=decoder.inference_logits(prefix,batch['input'])
            second=decoder.inference_logits(prefix,batch['input'],batch['answer']['input_ids'][:,:1])
            torch.testing.assert_close(first[:,-1],baseline[:,382],atol=3e-6,rtol=3e-5)
            torch.testing.assert_close(second[:,-1],baseline[:,383],atol=3e-6,rtol=3e-5)

    def test_bidirectional_prompt_to_audio_and_same_mask_all50_layers(self):
        model=lm().eval();x=torch.randn(1,683,16);valid=torch.ones(1,683,dtype=torch.long)
        seen=[];handles=[]
        for i,layer in enumerate(model.model.layers):
            def record(module,args,kwargs,index=i):seen.append((index,kwargs['attention_mask']))
            handles.append(layer.register_forward_pre_hook(record,with_kwargs=True))
        try:
            with torch.no_grad():
                a=model(inputs_embeds=x,attention_mask=valid,prefix_length=383).logits
                y=x.clone();y[:,200]+=torch.linspace(-2,2,16)
                b=model(inputs_embeds=y,attention_mask=valid,prefix_length=383).logits
            self.assertFalse(torch.allclose(a[:,0],b[:,0],atol=1e-7,rtol=1e-7))
            self.assertEqual([i for i,_ in seen[:50]],list(C.TEXT_MODEL_SCHEDULE))
            self.assertEqual(len({id(m) for _,m in seen[:50]}),1)
            self.assertEqual(model.model.last_trace,list(C.TEXT_MODEL_SCHEDULE))
            self.assertEqual(len(model.model.layers),20)
        finally:
            for h in handles:h.remove()

    def test_bp5_prelude_mapper_gradients_frozen_encoder_and_loss_shift(self):
        model=fixture().train();batch=inputs();model.caption_decoder.lm.model.audit_gradients=True
        output=model(batch);loss=T.answer_loss(output,batch,17)
        expected=torch.nn.functional.cross_entropy(output.logits[:,382:386].reshape(-1,43),torch.tensor([6,7,8,0]))
        torch.testing.assert_close(loss,expected,rtol=1e-7,atol=5e-7)
        loss.backward()
        report=recursive_backward_audit(model,1)
        self.assertTrue(report['passed']);self.assertTrue(prelude_gradient_audit(model)['passed'])
        self.assertEqual({name for name,on in model.caption_decoder.lm.model.last_calls if on},{'H1','H2','L4','L5','L6'})
        self.assertTrue(T.first_gradient_audit(model)['passed'])
        self.assertTrue(all(p.grad is None for p in model.audio_encoder.base.ced.parameters()))
        self.assertFalse(model.audio_encoder.base.ced.training)
        self.assertFalse(model.caption_decoder.lm.model.audit_refs)
        self.assertFalse(model.caption_decoder.lm.model.audit_gradients)

    def test_raw_text_inheritance_exact_nonzero_gates_and_bad_sources_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,original=source_checkpoint(tmp)
            with patch('torch.load',side_effect=AssertionError('must not deserialize text optimizer')):
                loaded=load_recursive_text_model(source,production=False)
            for name,value in original.state_dict().items():torch.testing.assert_close(loaded.state_dict()[name],value,rtol=0,atol=0)
            self.assertGreater(float(loaded.model.layers[5].self_attn.gate_proj.weight.abs().sum().detach()),0)
            self.assertIs(loaded.model.embed_tokens.weight,loaded.lm_head.weight)
            mark=json.loads((source/'checkpoint_complete.json').read_text());mark['step']=3600
            C.write_json(source/'checkpoint_complete.json',mark)
            with self.assertRaises(ValueError):load_recursive_text_model(source,production=False)

    def test_source_tensor_inventory_dtype_nonfinite_and_tied_conflicts_rejected(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as tmp:
            source,model=source_checkpoint(tmp)
            marker=json.loads((source/'checkpoint_complete.json').read_text())
            for fault in ('missing','extra','dtype','nonfinite','tied'):
                values={name:value.detach().clone() for name,value in model.state_dict().items()}
                key='model.layers.5.self_attn.gate_proj.weight'
                if fault=='missing':del values[key]
                if fault=='extra':values['unexpected.weight']=torch.ones(1)
                if fault=='dtype':values[key]=values[key].bfloat16()
                if fault=='nonfinite':values[key].flatten()[0]=float('nan')
                if fault=='tied':values['lm_head.weight'].flatten()[0]+=1
                save_file(values,str(source/'model.safetensors'))
                marker['files']=file_inventory(source,include_state=True)
                C.write_json(source/'checkpoint_complete.json',marker)
                with self.assertRaises(ValueError):load_recursive_text_model(source,production=False)

    def test_role_optimizer_unique_tied_parameter_inventory_and_full_lr_curve(self):
        model=fixture();opt,scheduler,audit=build_optimizer(model,34626)
        inventory=optimizer_parameter_inventory(model,opt)
        self.assertEqual(len({id(p) for g in opt.param_groups for p in g['params']}),len(list(p for p in model.parameters() if p.requires_grad)))
        self.assertEqual(set(g['role'] for g in opt.param_groups),{'map','default'})
        for step in range(34627):
            warmup=1731
            for role,maximum,minimum in [('default',6e-4,3e-5),('map',8e-4,6e-5)]:
                expected=maximum*max(.1,(step+1)/warmup) if step<warmup else minimum+(maximum-minimum)*.5*(1+math.cos(math.pi*min(1.,(step-warmup)/(34625-warmup))))
                self.assertAlmostEqual(maximum*C.lr_factor(step,34626,role),expected,places=17)
        for cursor in (0,1730,1731,34625,34626):
            opt.step();scheduler.last_epoch=cursor-1;scheduler.step()
            for g in opt.param_groups:
                self.assertAlmostEqual(g['lr'],g['initial_lr']*C.lr_factor(cursor,34626,g['role']),places=17)
        for g in opt.param_groups:self.assertAlmostEqual(g['lr'],6e-5 if g['role']=='map' else 3e-5,places=17)

    def test_full_model_optimizer_scheduler_gradnorm_rng_restore_next_update(self):
        model=fixture().train();opt,sched,_=build_optimizer(model,34626);tracker=GradNormTracker()
        update(model,opt,sched,tracker)
        cfg=config();rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.get_rng_state())
        def gather(states,state):states[:]=[copy.deepcopy(state) for _ in range(8)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'step1.ckpt';scaler=torch.amp.GradScaler('cpu',enabled=False)
            with patch.object(T,'rng_state',return_value=rng),patch.object(T.dist,'all_gather_object',side_effect=gather),patch.object(T.dist,'get_rank',return_value=0),patch.object(T.dist,'barrier'),patch.object(T.dist,'all_reduce'):
                T.save_checkpoint(path,model,opt,sched,scaler,tracker,cfg,1,0,1,'cpu')
            saved=torch.load(path,map_location='cpu',weights_only=False)
            other=fixture().train();opt2,sched2,_=build_optimizer(other,34626);tracker2=GradNormTracker()
            validate_resume_model_state(saved,other,opt2,cfg)
            other.load_state_dict(saved['state_dict'],strict=True);opt2.load_state_dict(saved['optimizer']);sched2.load_state_dict(saved['scheduler']);tracker2.load_state_dict(saved['grad_norm_tracker'])
            torch.set_rng_state(rng['torch']);update(model,opt,sched,tracker)
            torch.set_rng_state(rng['torch']);update(other,opt2,sched2,tracker2)
            for name,value in model.state_dict().items():torch.testing.assert_close(value,other.state_dict()[name],rtol=0,atol=0)
            self.assertEqual(sched.state_dict(),sched2.state_dict());self.assertEqual(tracker.state_dict(),tracker2.state_dict())
            for a,b in zip(opt.state.values(),opt2.state.values()):
                for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            for fault in ('dtype','tied','nan_moment','parameter_order','lr','prefix','rng'):
                bad=copy.deepcopy(saved)
                key='caption_decoder.lm.model.layers.5.self_attn.gate_proj.weight'
                if fault=='dtype':bad['state_dict'][key]=bad['state_dict'][key].double()
                if fault=='tied':bad['state_dict']['caption_decoder.lm.lm_head.weight']=bad['state_dict']['caption_decoder.lm.lm_head.weight'].clone()+1
                if fault=='nan_moment':next(iter(bad['optimizer']['state'].values()))['exp_avg'].fill_(float('nan'))
                if fault=='parameter_order':bad['optimizer_parameter_inventory'][0].reverse()
                if fault=='lr':bad['optimizer']['param_groups'][0]['lr']=2e-4
                if fault=='prefix':bad['contract']['attention_mask']='causal'
                if fault=='rng':bad['random_state_by_rank'].pop()
                with self.assertRaises(ValueError):validate_resume_model_state(bad,other,opt2,cfg)

    def test_real_waveform_fixed20_mel_and_temporal_grid(self):
        from data.fixed20 import load_fixed20
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'synthetic.wav'
            sf.write(path,np.sin(np.arange(32000*2)*.03).astype('float32'),32000,subtype='FLOAT')
            a,b=load_fixed20(str(path));self.assertEqual(a.shape,(320000,));self.assertEqual(b.shape,(320000,))
            self.assertEqual(int(torch.count_nonzero(b)),0)
            wrapper=CEDSmallWrapper.__new__(CEDSmallWrapper);torch.nn.Module.__init__(wrapper)
            import torchaudio
            wrapper.resample=torchaudio.transforms.Resample(32000,16000)
            wrapper.mel_spectrogram=torchaudio.transforms.MelSpectrogram(sample_rate=16000,win_length=512,n_fft=512,hop_length=160,n_mels=64,f_min=0,f_max=8000,center=True)
            wrapper.amplitude_to_db=torchaudio.transforms.AmplitudeToDB(top_db=120)
            whole=torch.cat((a,b)).unsqueeze(0);mel=wrapper._waveform_to_mel(whole)
            self.assertEqual(mel.shape,(1,64,2001));self.assertTrue(bool(torch.isfinite(mel).all()))
            hidden=torch.arange(504*384).reshape(1,504,384).float()
            grid=wrapper._hidden_to_temporal_grid(hidden)
            for split in (0,1):
                for freq in range(4):
                    for time in (0,31,62):
                        torch.testing.assert_close(grid[0,split*63+time,freq],hidden[0,split*252+freq*63+time])

    def test_reject_cache_checkpointing_bad_bp_or_missing_prefix_validity(self):
        model=lm();x=torch.randn(1,8,16);mask=torch.ones(1,8,dtype=torch.long)
        for kwargs in ({'use_cache':True},{'bp_steps':4},{'prefix_length':None},{'attention_mask':None}):
            args=dict(inputs_embeds=x,attention_mask=mask,prefix_length=4);args.update(kwargs)
            with self.assertRaises(ValueError):model(**args)
        with self.assertRaises(ValueError):model.gradient_checkpointing_enable()


if __name__=='__main__':unittest.main(verbosity=2)
