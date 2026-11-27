"""Real small Llama, nonzero trained adapters and independent recurrence on CPU."""
import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from torch import nn
from transformers import LlamaConfig
from mizar_l10_14x3_eval import registry,sources,runtime,benchmark


class Audio(nn.Module):
    def __init__(self):super().__init__();self.map=nn.Linear(1,16);self.calls=0
    def forward(self,wave):
        self.calls+=1
        return {'embedding':self.map(wave[:,:126,None])}


def native(attention='sdpa'):
    modules=sources.import_training(registry.TARGET)
    c=LlamaConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=30,num_attention_heads=4,
        num_key_value_heads=2,vocab_size=64,tie_word_embeddings=True,pad_token_id=17,eos_token_id=0,
        attention_dropout=0.,rope_theta=100000.,use_cache=False)
    c._attn_implementation=attention
    model=modules['model'].MizarCoreTrain(c,audio_encoder=Audio()).float();model.enable_training()
    with torch.no_grad():
        core=model.caption_decoder.lm.model
        core.audio_gate.weight.copy_(torch.linspace(-.15,.2,16)[None]);core.audio_gate.bias.fill_(.12)
        core.loop_bias.copy_(torch.arange(48).reshape(3,16)*.001)
    return model,modules


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_nonzero_gate_bias_independent_40_execution_eager_sdpa_no_carry(self):
        for attention in ('eager','sdpa'):
            model,modules=native(attention);model.eval().requires_grad_(False);lm=model.caption_decoder.lm
            audit=runtime.install_forward_audit(lm,registry.TARGET);x=torch.randn(1,383,16)
            with torch.inference_mode():
                actual=lm(inputs_embeds=x,use_cache=False).logits
                lm(inputs_embeds=x+.13,use_cache=False)
                again=lm(inputs_embeds=x,use_cache=False).logits
                torch.testing.assert_close(actual,again,rtol=0,atol=0)
            self.assertEqual(audit['checked_forwards'],3)
            self.assertEqual(audit['last_layers'],list(registry.ROUTES[registry.TARGET]['schedule']))
            self.assertEqual((audit['last_gate_calls'],audit['last_final_norm_calls'],audit['last_loop_entries']),(2,1,[0,1,2]))
            # Independent scalar gate equation and explicit bias positions; no production loop helper.
            with torch.inference_mode():
                position=torch.arange(383)[None,:]
                mask=modules['model']._causal_mask(attention_mask=None,batch_size=1,query_length=383,past_length=0,dtype=x.dtype,device=x.device)
                rotary=lm.model.rotary_emb(x,position_ids=position);h=x
                for logical,physical in enumerate(registry.ROUTES[registry.TARGET]['schedule']):
                    if logical in (9,14,19):
                        loop=(logical-9)//5
                        if loop:
                            prev=h[:,:126]
                            weights=((prev*lm.model.audio_gate.weight[0]).sum(-1,keepdim=True)+lm.model.audio_gate.bias[0]).tanh()
                            h=torch.cat((anchor+weights*prev,h[:,126:]),dim=1)
                        h=h+lm.model.loop_bias[loop]
                    h=modules['model']._call_decoder_layer(lm.model.layers[physical],h,attention_mask=mask,
                        position_ids=position,cache=None,use_cache=False,cache_position=position[0],position_embeddings=rotary,output_attentions=False)[0]
                    if logical==8:anchor=h[:,:126].clone()
                manual=lm.lm_head(lm.model.norm(h))
                torch.testing.assert_close(actual,manual,rtol=1e-5,atol=1e-6)

    def test_training683_prefix383_second_answer_and_future_causality(self):
        model,_=native();model.eval().requires_grad_(False);lm=model.caption_decoder.lm
        inputs=dict(audio1=torch.randn(1,320000),audio2=torch.randn(1,320000),
            input={'input_ids':torch.full((1,256),17,dtype=torch.long)},answer={'input_ids':torch.randint(0,64,(1,300))})
        with torch.inference_mode():
            full=model(inputs).logits;prefix,_=model.generate_prefix_inference(inputs)
            first=lm(inputs_embeds=prefix,use_cache=False).logits[:,-1]
            second=lm(inputs_embeds=torch.cat((prefix,lm.get_input_embeddings()(inputs['answer']['input_ids'][:,:1])),1),use_cache=False).logits[:,-1]
            for expected,actual in ((full[:,382],first),(full[:,383],second)):
                torch.testing.assert_close(expected,actual,rtol=1e-5,atol=1e-6)
                self.assertEqual(expected.argmax(-1),actual.argmax(-1))
            inputs['answer']['input_ids']=(inputs['answer']['input_ids']+1)%64
            torch.testing.assert_close(model(inputs).logits[:,:383],full[:,:383],rtol=0,atol=0)

    def test_strict_trained_layers_gate_bias_preserved_frozen_and_bad_state_refused(self):
        model,modules=native();c=modules['contract'];helper=modules['training_helpers']
        state={n:t.clone() for n,t in model.state_dict().items()}
        for name,p in model.named_parameters():
            if c.parameter_role(name)!='frozen':state[name].add_(.017)
        frozen=helper.tensor_digest([(n,t) for n,t in model.state_dict().items() if c.parameter_role(n)=='frozen'])
        saved=dict(state_dict=state,frozen_sha256=frozen)
        checked=runtime.validate_model_state(model,saved,c);model.load_state_dict(checked,strict=True)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()))
        for problem in ('missing','extra','shape','dtype','nan','frozen','digest','tied'):
            bad=copy.deepcopy(saved);name=next(iter(state))
            if problem=='missing':bad['state_dict'].pop('caption_decoder.lm.model.loop_bias')
            elif problem=='extra':bad['state_dict']['extra']=torch.zeros(1)
            elif problem=='shape':bad['state_dict']['caption_decoder.lm.model.loop_bias']=torch.zeros(4,16)
            elif problem=='dtype':bad['state_dict'][name]=state[name].double()
            elif problem=='nan':bad['state_dict'][name].view(-1)[0]=float('nan')
            elif problem=='frozen':bad['state_dict'][name].add_(1)
            elif problem=='digest':bad['frozen_sha256']='bad'
            else:bad['state_dict']['caption_decoder.lm.lm_head.weight'].add_(.1)
            with self.assertRaises(ValueError):runtime.validate_model_state(model,bad,c)

    def test_real_loader_control_flow_does_not_reset_trained_adapters(self):
        import transformers
        model,modules=native();c=modules['contract'];state={n:t.clone() for n,t in model.state_dict().items()}
        for name in state:
            if c.parameter_role(name)!='frozen':state[name].add_(.019)
        with torch.no_grad():
            model.caption_decoder.lm.model.audio_gate.weight.zero_()
            model.caption_decoder.lm.model.audio_gate.bias.zero_()
            model.caption_decoder.lm.model.loop_bias.zero_()
        frozen=modules['training_helpers'].tensor_digest([(n,t) for n,t in model.state_dict().items() if c.parameter_role(n)=='frozen'])
        args=types.SimpleNamespace(eval_variant=registry.TARGET,sources_cache=registry.SOURCE_CACHE,source_revision=None,
            model_dir=Path('/fixture').resolve(),route_root=sources.route_root(registry.TARGET))
        saved=dict(contract=dict(model_dir=str(args.model_dir),official_bundle={'fixture':True},geometry={'fixture':True},code_sha256='fixture'),
            state_dict=state,frozen_sha256=frozen,total_step=2542,route_variant=c.ROUTE,contract_sha256='fixture')
        proxy=types.ModuleType('torch');proxy.__dict__.update(torch.__dict__)
        proxy.device=lambda *a,**kw:torch.device('cpu')
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
        self.assertTrue(all(torch.equal(t,loaded[0].state_dict()[n]) for n,t in state.items()))
        self.assertTrue(all(not m.training for m in loaded[0].modules()))
        self.assertTrue(all(not p.requires_grad for p in loaded[0].parameters()))
        self.assertEqual(loaded[5]['checkpoint_step'],2542)

    def test_real_stereo16k_audio_fixed20_and_two_tasks_encode_once(self):
        import numpy as np
        import soundfile as sf
        model,modules=native();model.eval().requires_grad_(False);fixed=modules['data.fixed20']
        ids=torch.full((1,256),17,dtype=torch.long);ids[:,:3]=torch.tensor([3,4,0])
        mask=torch.zeros_like(ids);mask[:,:3]=1
        tok=types.SimpleNamespace(eos_token_id=0,pad_token_id=17,encode=lambda *a,**kw:[3,4,0],decode=lambda ids,**kw:str(ids))
        enc=types.SimpleNamespace(tokenize=lambda *a:dict(input_ids=ids.clone(),attention_mask=mask.clone()))
        with tempfile.TemporaryDirectory() as temp:
            wav=Path(temp)/'stereo.wav';sf.write(wav,np.ones((1600,2),dtype='float32')*.1,16000,subtype='FLOAT')
            wave=fixed.load_mono(wav);halves=fixed.load_fixed20(wav)
            self.assertEqual(wave.numel(),3200)
            with torch.inference_mode():
                prefix,_=model.generate_prefix_inference(dict(audio1=halves[0][None],audio2=halves[1][None],input={'input_ids':ids}))
                tok.eos_token_id=int(model.caption_decoder.lm(inputs_embeds=prefix,use_cache=False).logits[:,-1].argmax(-1))
            loaded=(model,tok,torch.device('cpu'),enc,fixed,{'eval_variant':registry.TARGET},modules)
            for task in registry.BENCHMARKS:
                model.audio_encoder.calls=0
                result=benchmark.generate(loaded,dict(prompt='Question? a) One b) Two',audio_path_resolved=str(wav)))
                self.assertEqual(model.audio_encoder.calls,1)
                self.assertEqual(result['generated_token_ids'],[tok.eos_token_id])
                self.assertEqual(result['first_answer_logit_position'],382)
                self.assertEqual(len(result['input_fingerprint_sha256']),64)


if __name__=='__main__':unittest.main()
