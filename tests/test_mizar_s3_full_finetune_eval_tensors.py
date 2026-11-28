"""Real native Llama/model methods and explicit CPU fixtures for release/CUDA."""
import ast
import copy
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from torch import nn
from transformers import AutoTokenizer,LlamaConfig,LlamaForCausalLM
from mizar_s3_full_finetune_eval import registry,sources,runtime,benchmark,eval_audio
from mizar_s3_full_finetune_eval.eval_text import TextEncoder
ROOT=Path(__file__).resolve().parents[1]
TOKENIZER=Path('C:/Xlance/assets/Mizar/Mizar159M/assets/smollm2-135m')


class Audio(nn.Module):
    def __init__(self):
        super().__init__();self.base=nn.Module();self.base.ced=nn.Linear(1,16)
        self.base.mapper=nn.Linear(16,16);self.projection=nn.Identity();self.calls=0
    def forward(self,wave):
        self.calls+=1
        out=self.base.mapper(self.base.ced(wave[:,:126,None]))
        return out,torch.zeros(1,527),{'embedding':out}


def native(attention='sdpa'):
    modules=sources.import_training(registry.TARGET)
    c=LlamaConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=30,num_attention_heads=4,
        num_key_value_heads=2,vocab_size=64,tie_word_embeddings=True,pad_token_id=17,eos_token_id=0,
        attention_dropout=0.,rope_theta=100000.,use_cache=False)
    c._attn_implementation=attention
    cls=modules['model'].Mellow;decoder_cls=importlib.import_module('models.decoder').DecoderModel
    model=cls.__new__(cls);nn.Module.__init__(model);model.audio_encoder=Audio()
    decoder=decoder_cls.__new__(decoder_cls);nn.Module.__init__(decoder)
    decoder.lm=LlamaForCausalLM(c).float();decoder.prefix_length=383;decoder.lm_embedding_size=16;decoder.last_layout=None
    model.caption_decoder=decoder;modules['model'].configure_training(model)
    modules['model'].validate_trainability(model,production=False)
    return model,modules


def trained_state(model,c):
    # Include both tied state aliases and every LM/mapper tensor; CED remains identical.
    return {name:t.clone()+(.017 if c.parameter_role(name)!='frozen' else 0.) for name,t in model.state_dict().items()}


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_native30_eager_sdpa_single_norm_and_no_carry(self):
        for attention in ('eager','sdpa'):
            model,modules=native(attention);model.eval().requires_grad_(False);lm=model.caption_decoder.lm
            baseline=LlamaForCausalLM(copy.deepcopy(lm.config)).float().eval();baseline.load_state_dict(lm.state_dict())
            audit=runtime.install_forward_audit(lm,registry.TARGET);x=torch.randn(1,383,16)
            with torch.inference_mode():
                a=lm(inputs_embeds=x,use_cache=False).logits
                expected=baseline(inputs_embeds=x,use_cache=False).logits
                lm(inputs_embeds=x+.2,use_cache=False)
                again=lm(inputs_embeds=x,use_cache=False).logits
                torch.testing.assert_close(a,expected,rtol=0,atol=0)
                torch.testing.assert_close(a,again,rtol=0,atol=0)
            self.assertEqual(audit['checked_forwards'],3)
            self.assertEqual(audit['last_layers'],list(range(30)))
            self.assertEqual(audit['last_final_norm_calls'],1)
            self.assertFalse(any('audio_gate' in n or 'loop_bias' in n for n in model.state_dict()))

    def test_real_s3_479_and_383_384_same_input_logits_argmax_causality(self):
        model,_=native();model.eval().requires_grad_(False);lm=model.caption_decoder.lm
        inputs=dict(audio1=torch.randn(1,320000),audio2=torch.randn(1,320000),
            input={'input_ids':torch.full((1,256),17,dtype=torch.long)},answer={'input_ids':torch.randint(0,64,(1,96))})
        with torch.inference_mode():
            full=model(inputs).logits;prefix,_,_=model.generate_prefix_inference(inputs)
            self.assertEqual(full.shape,(1,479,64))
            first=lm(inputs_embeds=prefix,use_cache=False).logits[:,-1]
            second=lm(inputs_embeds=torch.cat((prefix,lm.get_input_embeddings()(inputs['answer']['input_ids'][:,:1])),1),use_cache=False).logits[:,-1]
            for expected,actual in ((full[:,382],first),(full[:,383],second)):
                torch.testing.assert_close(expected,actual,rtol=1e-4,atol=1e-4)
                self.assertTrue(torch.equal(expected.argmax(-1),actual.argmax(-1)))
            inputs['answer']['input_ids']=(inputs['answer']['input_ids']+1)%64
            torch.testing.assert_close(model(inputs).logits[:,:383],full[:,:383],rtol=0,atol=0)

    def test_trained_full_mapper_lm_embedding_norm_preserved_only_ced_frozen(self):
        model,modules=native();c=modules['contract'];helper=modules['training_helpers']
        before={n:t.clone() for n,t in model.state_dict().items()};state=trained_state(model,c)
        frozen=helper.tensor_digest([(n,t) for n,t in before.items() if c.parameter_role(n)=='frozen'])
        saved=dict(state_dict=state,frozen_sha256=frozen)
        runtime.validate_model_state(model,saved,c);model.load_state_dict(state,strict=True)
        self.assertTrue(all(torch.equal(t,model.state_dict()[n]) for n,t in state.items()))
        for name in ('audio_encoder.base.mapper.weight','caption_decoder.lm.model.embed_tokens.weight',
                'caption_decoder.lm.model.norm.weight','caption_decoder.lm.model.layers.29.self_attn.q_proj.weight'):
            self.assertFalse(torch.equal(before[name],state[name]),name)
        self.assertTrue(torch.equal(before['audio_encoder.base.ced.weight'],state['audio_encoder.base.ced.weight']))
        for kind in ('missing','extra','shape','dtype','nan','ced','digest','tied'):
            bad=copy.deepcopy(saved);name='audio_encoder.base.ced.weight'
            if kind=='missing':bad['state_dict'].pop('audio_encoder.base.mapper.weight')
            elif kind=='extra':bad['state_dict']['audio_gate.weight']=torch.zeros(1)
            elif kind=='shape':bad['state_dict']['audio_encoder.base.mapper.weight']=torch.zeros(3)
            elif kind=='dtype':bad['state_dict'][name]=bad['state_dict'][name].double()
            elif kind=='nan':bad['state_dict'][name].view(-1)[0]=float('nan')
            elif kind=='ced':bad['state_dict'][name].add_(1)
            elif kind=='digest':bad['frozen_sha256']='bad'
            else:bad['state_dict']['caption_decoder.lm.lm_head.weight'].add_(1)
            with self.assertRaises(ValueError):runtime.validate_model_state(model,bad,c)

    def test_actual_loader_preserves_complete_trained_values_and_uses_eval_preprocessing(self):
        import transformers
        model,modules=native();c=modules['contract'];state=trained_state(model,c)
        digest=modules['training_helpers'].tensor_digest([(n,t) for n,t in model.state_dict().items() if c.parameter_role(n)=='frozen'])
        args=types.SimpleNamespace(eval_variant=registry.TARGET,sources_cache=registry.SOURCE_CACHE,source_revision=None,
            model_dir=Path('/fixture').resolve(),route_root=sources.route_root(registry.TARGET))
        saved=dict(contract=dict(model_dir=str(args.model_dir),official_bundle={'fixture':True},geometry={'fixture':True},code_sha256='fixture'),
            state_dict=state,frozen_sha256=digest,total_step=1066,route_variant=c.ROUTE,contract_sha256='fixture')
        proxy=types.ModuleType('torch');proxy.__dict__.update(torch.__dict__);proxy.device=lambda *a,**kw:torch.device('cpu')
        proxy.manual_seed=lambda seed:torch.random.default_generator.manual_seed(seed)
        proxy.cuda=types.SimpleNamespace(is_available=lambda:True,manual_seed_all=lambda *a:None,set_device=lambda *a:None,get_device_name=lambda *a:'CPU CUDA fixture')
        validate=modules['model'].validate_trainability
        with patch.dict(sys.modules,{'torch':proxy}),patch.object(runtime,'read_checkpoint',return_value=saved), \
             patch.object(sources,'import_training',return_value=modules),patch.object(modules['assets'],'audit_bundle',return_value={'fixture':True}), \
             patch.object(modules['model'],'load_official',return_value=model), \
             patch.object(modules['model'],'validate_trainability',side_effect=lambda m:validate(m,production=False)), \
             patch.object(c,'validate_tokenizer_contract',return_value={'fixture':True}), \
             patch.object(transformers.AutoTokenizer,'from_pretrained',return_value=types.SimpleNamespace(pad_token_id=17,add_special_tokens=lambda *a:0)):
            loaded=runtime.load_runtime(args)
        self.assertTrue(all(torch.equal(t,model.state_dict()[n]) for n,t in state.items()))
        self.assertTrue(all(not m.training for m in model.modules()));self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertIsInstance(loaded[3],TextEncoder);self.assertIs(loaded[4],eval_audio)
        self.assertIs(loaded[6]['data.fixed20'],eval_audio)
        self.assertIsNot(modules['data.fixed20'],eval_audio)

    def test_real_tokenizer_common_eos_truncation_matches_previous_not_s3_training(self):
        modules=sources.import_training(registry.TARGET)
        tok=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True);tok.add_special_tokens({'pad_token':'!'})
        tok.padding_side=tok.truncation_side='right';c=modules['contract'];c.validate_tokenizer_contract(tok,17,49152)
        evaluation=TextEncoder(tok);training=modules['data.audiotext_dataset'].AudioTextDataset.__new__(modules['data.audiotext_dataset'].AudioTextDataset)
        training.tokenizer=tok
        for text in ('what is heard? a) car b) train c) bird d) rain','sound '*700):
            actual=evaluation.tokenize(text,256)
            expected=tok.encode_plus(text=text+' <|endoftext|>',add_special_tokens=True,max_length=256,padding='max_length',truncation=True,return_tensors='pt')
            for key in actual:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
        short='what is heard? a) car b) train c) bird d) rain'
        self.assertFalse(torch.equal(evaluation.tokenize(short,256)['input_ids'],training.tokenize(short,256)['input_ids']))
        self.assertEqual(tok.eos_token_id,0);self.assertEqual(tok.pad_token_id,17)

    def test_real_long_stereo_audio_center20_and_generation_encoder_once(self):
        import numpy as np
        import soundfile as sf
        import torchaudio
        model,modules=native();model.eval().requires_grad_(False)
        ids=torch.full((1,256),17,dtype=torch.long);ids[:,:3]=torch.tensor([3,4,0]);mask=torch.zeros_like(ids);mask[:,:3]=1
        tok=types.SimpleNamespace(eos_token_id=0,pad_token_id=17,encode=lambda *a,**kw:[3,4,0],decode=lambda ids,**kw:str(ids))
        enc=types.SimpleNamespace(tokenize=lambda *a:dict(input_ids=ids.clone(),attention_mask=mask.clone()))
        with tempfile.TemporaryDirectory() as temp:
            wav=Path(temp)/'long_stereo.wav';samples=np.linspace(-.4,.6,24*16000,dtype='float32')
            sf.write(wav,np.stack((samples,samples*.5),axis=1),16000,subtype='FLOAT')
            wave=eval_audio.load_mono(wav);halves=eval_audio.load_fixed20(wav)
            independent=torchaudio.functional.resample(torch.from_numpy(np.stack((samples,samples*.5))).mean(0,keepdim=True),16000,32000)[0]
            expected=independent[64000:704000]
            torch.testing.assert_close(torch.cat(halves),expected,rtol=0,atol=0)
            training=modules['data.fixed20'].load_fixed20(wav)
            self.assertFalse(torch.equal(torch.cat(halves),torch.cat(training)))
            with torch.inference_mode():
                prefix,_,_=model.generate_prefix_inference(dict(audio1=halves[0][None],audio2=halves[1][None],input={'input_ids':ids}))
                tok.eos_token_id=int(model.caption_decoder.lm(inputs_embeds=prefix,use_cache=False).logits[:,-1].argmax(-1))
            loaded=(model,tok,torch.device('cpu'),enc,eval_audio,{'eval_variant':registry.TARGET},{**modules,'data.fixed20':eval_audio})
            results=[]
            for task,sample in (('mmar',dict(audio_path_resolved=str(wav))),('mmau',dict(waveform=wave))):
                model.audio_encoder.calls=0
                result=benchmark.generate(loaded,dict(prompt='Question? a) One b) Two',**sample));results.append(result)
                self.assertEqual(model.audio_encoder.calls,1);self.assertEqual(result['generated_token_ids'],[tok.eos_token_id])
                self.assertEqual(result['first_answer_logit_position'],382)
            self.assertEqual(results[0]['input_fingerprint_sha256'],results[1]['input_fingerprint_sha256'])


if __name__=='__main__':unittest.main()
