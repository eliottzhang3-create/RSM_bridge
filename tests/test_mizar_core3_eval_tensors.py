"""Real 30-layer Llama inference and trained gate/state preservation on CPU."""
import copy
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from torch import nn
from transformers import LlamaConfig,LlamaForCausalLM
from mizar_core3_eval import registry,sources,runtime,reasonaqa_runtime,mmsu_runtime,benchmark
from test_mizar_core3_eval import cache


class Audio(nn.Module):
    def __init__(self,width):
        super().__init__();self.map=nn.Linear(1,width);self.calls=0
    def forward(self,wave):
        self.calls+=1
        return {'embedding':self.map(wave[:,:126,None])}


def config(attention='sdpa'):
    c=LlamaConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=30,num_attention_heads=4,
        num_key_value_heads=2,vocab_size=64,tie_word_embeddings=True,pad_token_id=17,eos_token_id=0,
        attention_dropout=0.,rope_theta=100000.,use_cache=False)
    c._attn_implementation=attention
    return c


def native(target,attention='sdpa'):
    modules=sources.import_training(target,cache())
    model=modules['model'].MizarCoreTrain(config(attention),audio_encoder=Audio(16)).float()
    model.enable_training()
    return model,modules


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_three_native_schedules_state_reload_and_independent_full_recurrence(self):
        for target in ('l9_12x3','l14_17x3','l19_22x3'):
            for attention in ('eager','sdpa'):
                model,modules=native(target,attention)
                lm=model.caption_decoder.lm
                if registry.ROUTES[target]['gate']:
                    with torch.no_grad():lm.model.audio_gate.weight.fill_(.013);lm.model.audio_gate.bias.fill_(.07)
                before={k:v.clone() for k,v in model.state_dict().items()}
                model.load_state_dict(before,strict=True)
                model.eval().requires_grad_(False)
                audit=runtime.install_forward_audit(lm,target)
                x=torch.randn(1,383,16)
                with torch.inference_mode():
                    actual=lm(inputs_embeds=x,attention_mask=None,use_cache=False).logits
                    lm(inputs_embeds=x+.1,attention_mask=None,use_cache=False)
                    again=lm(inputs_embeds=x,attention_mask=None,use_cache=False).logits
                    torch.testing.assert_close(actual,again,rtol=0,atol=0)
                    # Independently execute the raw decoder blocks and gate formula.
                    positions=torch.arange(383)[None,:]
                    mask=modules['model']._causal_mask(attention_mask=None,batch_size=1,query_length=383,
                        past_length=0,dtype=x.dtype,device=x.device)
                    rotary=lm.model.rotary_emb(x,position_ids=positions)
                    h=x;anchor=None
                    for logical,physical in enumerate(registry.ROUTES[target]['schedule']):
                        if target=='l9_12x3' and logical in (12,16):
                            audio=h[:,:126]
                            h=torch.cat((audio+torch.tanh(lm.model.audio_gate(audio))*anchor,h[:,126:]),dim=1)
                        h=modules['model']._call_decoder_layer(lm.model.layers[physical],h,attention_mask=mask,
                            position_ids=positions,cache=None,use_cache=False,cache_position=positions[0],
                            position_embeddings=rotary,output_attentions=False)[0]
                        if target=='l9_12x3' and logical==7:anchor=h[:,:126].clone()
                    manual=lm.lm_head(lm.model.norm(h))
                    torch.testing.assert_close(actual,manual,rtol=0,atol=0)
                self.assertEqual(audit['checked_forwards'],3)
                self.assertEqual(audit['last_layers'],list(registry.ROUTES[target]['schedule']))
                self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))

    def test_prefix383_matches_training683_and_future_answer_causality(self):
        for target in ('l9_12x3','l14_17x3','l19_22x3'):
            model,_=native(target);model.eval().requires_grad_(False)
            inputs={'audio1':torch.randn(1,320000),'audio2':torch.randn(1,320000),
                'input':{'input_ids':torch.full((1,256),17,dtype=torch.long)},
                'answer':{'input_ids':torch.randint(0,64,(1,300))}}
            with torch.inference_mode():
                full=model(inputs).logits
                prefix,_=model.generate_prefix_inference(inputs)
                first=model.caption_decoder.lm(inputs_embeds=prefix,use_cache=False).logits
                second_input=torch.cat((prefix,model.caption_decoder.lm.get_input_embeddings()(inputs['answer']['input_ids'][:,:1])),dim=1)
                second=model.caption_decoder.lm(inputs_embeds=second_input,use_cache=False).logits
                torch.testing.assert_close(first[:,-1],full[:,382],rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(second[:,-1],full[:,383],rtol=1e-5,atol=1e-6)
                inputs['answer']['input_ids']=(inputs['answer']['input_ids']+1)%64
                changed=model(inputs).logits
                torch.testing.assert_close(full[:,:383],changed[:,:383],rtol=0,atol=0)

    def test_strict_model_state_trained_values_frozen_drift_and_gate_rejected(self):
        for target in ('l9_12x3','l14_17x3','l19_22x3'):
            model,modules=native(target);c=modules['contract']
            frozen=modules['training_helpers'].tensor_digest([(n,t) for n,t in model.state_dict().items() if c.parameter_role(n)=='frozen'])
            state={k:v.clone() for k,v in model.state_dict().items()}
            for name,p in model.named_parameters():
                if c.parameter_role(name)!='frozen':state[name].add_(.03)
            saved={'state_dict':state,'frozen_sha256':frozen}
            checked=runtime.validate_model_state(model,saved,c)
            model.load_state_dict(checked,strict=True)
            self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()))
            for bad in ('missing','dtype','nonfinite','frozen','digest'):
                copy_saved=copy.deepcopy(saved)
                name=next(iter(copy_saved['state_dict']))
                if bad=='missing':copy_saved['state_dict'].pop(name)
                elif bad=='dtype':copy_saved['state_dict'][name]=copy_saved['state_dict'][name].double()
                elif bad=='nonfinite':copy_saved['state_dict'][name].view(-1)[0]=float('nan')
                elif bad=='frozen':copy_saved['state_dict'][name].add_(1)
                else:copy_saved['frozen_sha256']='bad'
                with self.assertRaises(ValueError):runtime.validate_model_state(model,copy_saved,c)

    def test_baseline_native30_and_original_loader_delegation(self):
        lm=LlamaForCausalLM(config()).float().eval().requires_grad_(False)
        model=nn.Module();model.caption_decoder=types.SimpleNamespace(lm=lm)
        audit=runtime.install_forward_audit(lm,'official_mizar')
        x=torch.randn(1,383,16)
        with torch.inference_mode():
            expected=lm(inputs_embeds=x,use_cache=False).logits[:,-1]
            eos=int(expected.argmax(-1))
            tok=types.SimpleNamespace(eos_token_id=eos,decode=lambda ids,**kw:str(ids))
            result=runtime.greedy_decode(model,tok,x,eval_variant='official_mizar',context={})
        self.assertEqual(result['generated_token_ids'],[eos]);self.assertEqual(audit['checked_forwards'],2)
        self.assertEqual(audit['last_layers'],list(range(30)))
        args=types.SimpleNamespace(eval_variant='official_mizar',_bundle={'fixture':True})
        loaded=(model,tok,None,None,None,{},None)
        with patch.object(torch.cuda,'is_available',return_value=True),patch.object(runtime,'read_checkpoint'), \
             patch.object(runtime.official,'load_runtime',return_value=loaded) as loader:
            actual=runtime.load_runtime(args)
        loader.assert_called_once_with(args,bundle_audit=args._bundle)
        self.assertEqual(actual[5]['checkpoint_step'],None)

    def test_actual_model_loader_control_flow_preserves_all_trained_values(self):
        import sys
        import transformers
        for target in ('l9_12x3','l14_17x3','l19_22x3'):
            model,modules=native(target);c=modules['contract']
            frozen=modules['training_helpers'].tensor_digest([(n,t) for n,t in model.state_dict().items() if c.parameter_role(n)=='frozen'])
            state={n:t.clone() for n,t in model.state_dict().items()}
            for name,p in model.named_parameters():
                if c.parameter_role(name)!='frozen':state[name].add_(.019)
            args=types.SimpleNamespace(eval_variant=target,sources_cache=cache(),source_revision=None,model_dir=Path('/fixture').resolve())
            saved=dict(contract=dict(model_dir=str(args.model_dir),official_bundle={'fixture':True},geometry={'fixture':True},code_sha256='fixture'),
                state_dict=state,frozen_sha256=frozen,total_step=5,route_variant=c.ROUTE,contract_sha256='fixture')
            proxy=types.ModuleType('torch');proxy.__dict__.update(torch.__dict__)
            proxy.device=lambda *a,**kw:torch.device('cpu')
            proxy.manual_seed=lambda seed:torch.random.default_generator.manual_seed(seed)
            proxy.cuda=types.SimpleNamespace(is_available=lambda:True,manual_seed_all=lambda *a:None,
                set_device=lambda *a:None,get_device_name=lambda *a:'explicit CPU CUDA fixture')
            tokenizer=types.SimpleNamespace(pad_token_id=17,add_special_tokens=lambda *a:0)
            original_validate=modules['model'].validate_trainability
            with patch.dict(sys.modules,{'torch':proxy}),patch.object(runtime,'read_checkpoint',return_value=saved), \
                 patch.object(sources,'import_training',return_value=modules), \
                 patch.object(modules['assets'],'audit_bundle',return_value={'fixture':True}), \
                 patch.object(modules['model'],'load_official',return_value=model), \
                 patch.object(modules['model'],'validate_trainability',side_effect=lambda m:original_validate(m,production=False)), \
                 patch.object(c,'validate_tokenizer_contract',return_value={'fixture':True}), \
                 patch.object(transformers.AutoTokenizer,'from_pretrained',return_value=tokenizer):
                loaded=runtime.load_runtime(args)
            self.assertTrue(all(torch.equal(t,loaded[0].state_dict()[n]) for n,t in state.items()))
            self.assertTrue(all(not m.training for m in loaded[0].modules()))
            self.assertTrue(all(not p.requires_grad for p in loaded[0].parameters()))
            self.assertEqual(loaded[5]['checkpoint_step'],5)


    def test_real_audio_and_all_four_actual_generation_adapters(self):
        import numpy as np
        import soundfile as sf
        from mizar_core3_eval import reasonaqa_runtime as rr
        for target in registry.ROUTES:
            if target == 'official_mizar':
                modules = runtime.reference.import_training_route(runtime.official.ROUTE_ROOT)
                mellow_class = modules['models.mellow'].Mellow
                import importlib
                decoder_class = importlib.import_module('models.decoder').DecoderModel
                model = mellow_class.__new__(mellow_class);nn.Module.__init__(model)
                decoder = decoder_class.__new__(decoder_class);nn.Module.__init__(decoder)
                decoder.lm = LlamaForCausalLM(config()).float()
                decoder.lm_embedding_size = 16;decoder.prefix_length = 383
                class TupleAudio(Audio):
                    def forward(self,wave):
                        out = super().forward(wave)
                        return out['embedding'],torch.zeros(1,527),out
                model.caption_decoder = decoder;model.audio_encoder = TupleAudio(16)
            else:
                model,modules=native(target)
            model.eval().requires_grad_(False)
            fixed=modules['data.fixed20']
            prompt=torch.full((1,256),17,dtype=torch.long);prompt[:,:3]=torch.tensor([3,4,0])
            mask=torch.zeros_like(prompt);mask[:,:3]=1
            tok=types.SimpleNamespace(eos_token_id=0,pad_token_id=17,encode=lambda *a,**kw:[3,4,0],decode=lambda ids,**kw:str(ids))
            enc=types.SimpleNamespace(tokenize=lambda *a:{'input_ids':prompt.clone(),'attention_mask':mask.clone()})
            with tempfile.TemporaryDirectory() as temp:
                wav=Path(temp)/'stereo.wav';sf.write(wav,np.ones((1600,2),dtype='float32')*.1,16000,subtype='FLOAT')
                wave=fixed.load_mono(wav);decoded,aa=mmsu_runtime.decode_audio({'bytes':wav.read_bytes(),'path':'stale'},temp)
                torch.testing.assert_close(wave,decoded,atol=0,rtol=0)
                halves=fixed.load_fixed20(wav)
                with torch.inference_mode():
                    prefix,_,_=runtime.generate_prefix_inference(model,{'audio1':halves[0][None],'audio2':halves[1][None],'input':{'input_ids':prompt}},eval_variant=target,audit={})
                    tok.eos_token_id=int(model.caption_decoder.lm(inputs_embeds=prefix,use_cache=False).logits[:,-1].argmax(-1))
                loaded=(model,tok,torch.device('cpu'),enc,fixed,{'eval_variant':target},modules)
                stat=wav.stat();record=dict(original_record={'input':'Question?'},audio1=dict(path=str(wav),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),audio2=None)
                for task in registry.BENCHMARKS:
                    model.audio_encoder.calls=0
                    if task=='reasonaqa5000':result=rr.generate(loaded,record,variant=target)
                    elif task=='mmsu':result=mmsu_runtime.generate(loaded,dict(prompt='question?',waveform=wave,audio_audit=aa),variant=target)
                    else:result=benchmark.generate(loaded,dict(prompt='question?',waveform=wave))
                    self.assertEqual(result['generated_token_ids'],[tok.eos_token_id])
                    self.assertEqual(model.audio_encoder.calls,1)
                    self.assertEqual(result['first_answer_logit_position'],382)
                    self.assertEqual(len(result['input_fingerprint_sha256']),64)


if __name__=='__main__':unittest.main()
