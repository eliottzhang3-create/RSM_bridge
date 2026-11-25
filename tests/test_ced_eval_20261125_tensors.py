"""Real CPU native models, masks, strict audits and generation; no remote assets."""
import gc
import importlib
import importlib.util
from pathlib import Path
import sys
import sysconfig
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/RSmol'),str(ROOT/'tests')]
# The repository's code/ package otherwise shadows the Python stdlib in pdb.
definition=importlib.util.spec_from_file_location('code',Path(sysconfig.get_path('stdlib'))/'code.py')
stdlib_code=importlib.util.module_from_spec(definition);definition.loader.exec_module(stdlib_code);sys.modules['code']=stdlib_code
from ced_eval_20261125 import registry,sources,runtime,observers,hrm
from test_ced_eval_20261125 import cache

MISSING=[n for n in ('torch','transformers','safetensors','torchaudio','soundfile') if importlib.util.find_spec(n) is None]
if not MISSING:
    import torch
    from torch import nn
    from transformers import LlamaConfig

def native(target,width=16):
    root=sources.activate(target,cache());c=importlib.import_module('contract');spec=registry.variant_spec(target)
    config=LlamaConfig(num_hidden_layers=30,hidden_size=width,intermediate_size=1536 if width==576 else 32,
        num_attention_heads=9 if width==576 else 2,num_key_value_heads=3 if width==576 else 1,
        vocab_size=49152 if width==576 else 43,tie_word_embeddings=True,pad_token_id=17,eos_token_id=0,
        max_position_embeddings=1024,rope_theta=100000.,rms_norm_eps=1e-5,attention_dropout=0.,attention_bias=False,
        use_cache=False,_attn_implementation='sdpa')
    implementation=importlib.import_module(spec['implementation'])
    text=importlib.import_module(spec['shared_packages'][0]+'.contract')
    config=(text.build_target_config(config) if target=='audio8_read3' else text.target_config(config))
    lm=getattr(implementation,spec['model_class'])(config).float()
    if target=='audio8_read3':lm.model.enable_audio_read3();lm.model.enable_loop_lora()
    decoder_class=importlib.import_module('models.decoder').DecoderModel
    decoder=decoder_class.__new__(decoder_class);nn.Module.__init__(decoder)
    decoder.lm=lm;decoder.prefix_length=383;decoder.lm_embedding_size=width
    decoder.text_model_contract=spec['text_contract'];decoder.is_hrm_prefixlm=target=='hrm_prefixlm'
    decoder.is_recursive_mesh=target=='audio8_read3';decoder.last_layout=None
    model=nn.Module();model.caption_decoder=decoder
    if target=='audio8_read3':
        merger=importlib.import_module('models.ced').CEDFrequencyMerger
        model.audio_encoder=nn.Module();model.audio_encoder.base=nn.Module()
        model.audio_encoder.base.layer_mlps=nn.ModuleDict({str(i):merger(384,4,width,dropout=0.) for i in (4,8)})
    with torch.no_grad():
        for name,p in lm.named_parameters():
            if any(k in name for k in ('lora_B','audio_read_routers','write_routers','read_routers','self_attn.gate_proj')):p.add_(.013)
    contract={'text_model_dir':'/fixture','recursive_text_checkpoint':{'fixture':True}}
    c.recursive_checkpoint_contract=lambda path:contract['recursive_text_checkpoint']
    return model,c,contract,root

@unittest.skipIf(MISSING,'Real CPU dependencies missing: '+', '.join(MISSING))
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_real_production_dimension_audit_for_both_native_architectures(self):
        for target in registry.ROUTES:
            model,c,contract,root=native(target,576)
            lm=model.caption_decoder.lm
            gate=lm.model.layers[5].self_attn.gate_proj.weight.clone() if target=='hrm_prefixlm' else lm.model.audio_read_routers[2].weight.clone()
            audit=runtime.audit_model(model,c,contract,target,root)
            self.assertEqual(audit['physical_layer_count'],registry.variant_spec(target)['physical_layers'])
            self.assertEqual(audit['router_modules'],11 if target=='audio8_read3' else 0)
            if target=='audio8_read3':
                self.assertEqual(audit['new_parameters'],5876760);self.assertEqual(audit['loop_lora']['matrix_count'],588)
                self.assertTrue(torch.equal(gate,lm.model.audio_read_routers[2].weight))
                lm.model.audio_read_routers[2].bias=lm.model.audio_read_routers[1].bias
            else:
                self.assertEqual(audit['unique_lm_parameters'],102420288)
                self.assertTrue(torch.equal(gate,lm.model.layers[5].self_attn.gate_proj.weight))
                lm.model.layers[5].input_layernorm.eps=0.1
            with self.assertRaises(ValueError):runtime.audit_model(model,c,contract,target,root)
            del gate,lm,model,c;gc.collect()
    def test_real_audio8_three_loops_reload_hooks_and_native_validator(self):
        model,c,contract,root=native('audio8_read3');lm=model.caption_decoder.lm
        loader=importlib.import_module('models.recursive_text')
        before={k:v.clone() for k,v in lm.state_dict().items()}
        self.assertEqual(loader.validate_audio_read3(lm)['router_modules'],11)
        with self.assertRaises(TypeError):loader.validate_audio_read3(lm,fresh=False)
        lm.load_state_dict(before,strict=True);self.assertTrue(all(torch.equal(v,lm.state_dict()[k]) for k,v in before.items()))
        model.eval();model.requires_grad_(False);audit=observers.install_trace_audit(lm,'audio8_read3')
        x=torch.randn(1,383,16);pair=tuple(torch.randn(1,126,16) for _ in range(2))
        with torch.inference_mode():
            first=lm(inputs_embeds=x,use_cache=False,audio_information=pair).logits
            lm(inputs_embeds=torch.cat((x,torch.randn(1,1,16)),dim=1),use_cache=False,audio_information=pair)
            again=lm(inputs_embeds=x,use_cache=False,audio_information=pair).logits
            torch.testing.assert_close(first,again,rtol=0,atol=0)
            eos=int(first[:,-1].argmax(-1).item());tok=types.SimpleNamespace(eos_token_id=eos,decode=lambda ids,**kw:str(ids))
            with patch.object(runtime,'HIDDEN_SIZE',16):
                result=runtime.greedy_decode(model,tok,x,eval_variant='audio8_read3',context={'audio_information':pair})
            self.assertEqual(result['generated_token_ids'],[eos])
        self.assertEqual(audit['checked_forwards'],4);self.assertEqual(audit['loop_lora_calls_per_loop'],[98]*3)
        self.assertEqual(audit['last_schedule'],list(registry.MESH_SCHEDULE))
        self.assertTrue(all(torch.equal(v,lm.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(module.loop_index is None for layer in lm.model.layers[7:21] for module in [layer.get_submodule(t) for t in observers.TARGETS]))
    def test_hrm_native_training_prefix_first_second_logits_and_future_answer_isolation(self):
        model,c,contract,root=native('hrm_prefixlm');decoder=model.caption_decoder;lm=decoder.lm
        model.eval();model.requires_grad_(False);audit=observers.install_trace_audit(lm,'hrm_prefixlm')
        prompt=torch.full((1,256),17,dtype=torch.long);prompt[:,:4]=torch.tensor([3,4,5,0])
        pm=torch.zeros_like(prompt);pm[:,:4]=1;text={'input_ids':prompt,'attention_mask':pm}
        answer=torch.full((1,300),17,dtype=torch.long);answer[:,:4]=torch.tensor([6,7,8,0]);am=torch.zeros_like(answer);am[:,:4]=1
        audio=torch.randn(1,126,16)
        with torch.inference_mode():
            prefix=decoder.generate_prefix_inference(audio,None,text)
            full=decoder(audio,None,text,{'input_ids':answer,'attention_mask':am}).logits
            first=decoder.inference_logits(prefix,text)
            second=decoder.inference_logits(prefix,text,answer[:,:1])
            torch.testing.assert_close(first[:,-1],full[:,382],atol=1e-4,rtol=1e-4)
            torch.testing.assert_close(second[:,-1],full[:,383],atol=1e-4,rtol=1e-4)
            self.assertTrue(torch.equal(first[:,-1].argmax(-1),full[:,382].argmax(-1)))
            changed=decoder(audio,None,text,{'input_ids':(answer+1)%43,'attention_mask':am}).logits
            torch.testing.assert_close(full[:,:383],changed[:,:383],rtol=0,atol=0)
            changed_prompt=prompt.clone();changed_prompt[:,4:255]=22
            # Query382 stays !; changing only invalid prompt keys cannot change it.
            other={'input_ids':changed_prompt,'attention_mask':pm}
            masked=decoder.inference_logits(decoder.generate_prefix_inference(audio,None,other),other)
            torch.testing.assert_close(first[:,382],masked[:,382],atol=0,rtol=0)
        self.assertTrue(audit['same_mask_all50']);self.assertEqual(audit['last_schedule'],list(registry.HRM_SCHEDULE))
    def test_hrm_greedy_matches_independent_native_recompute_and_no_state_carry(self):
        model,c,contract,root=native('hrm_prefixlm');model.eval();model.requires_grad_(False)
        decoder=model.caption_decoder
        ids=torch.full((1,256),17,dtype=torch.long);mask=torch.zeros_like(ids);mask[:,:3]=1;ids[:,:3]=torch.tensor([3,4,0])
        text={'input_ids':ids,'attention_mask':mask};prefix=decoder.generate_prefix_inference(torch.randn(1,126,16),None,text)
        before={k:v.clone() for k,v in decoder.lm.state_dict().items()}
        with torch.inference_mode():
            manual=[]
            for i in range(3):
                previous=torch.tensor([manual],dtype=torch.long) if manual else None
                manual.append(int(decoder.inference_logits(prefix,text,previous)[:,-1].argmax(-1)))
            # Tokenizer EOS is set to the independently computed third token;
            # earlier EOS equality is handled by the same first-EOS semantics.
            tok=types.SimpleNamespace(eos_token_id=manual[-1],decode=lambda ids,**kw:str(ids))
            with patch.object(runtime,'HIDDEN_SIZE',16):
                generated=runtime.greedy_decode(model,tok,prefix,eval_variant='hrm_prefixlm',context={'texts_enc':text})
            stop=manual.index(manual[-1])+1
            self.assertEqual(generated['generated_token_ids'],manual[:stop])
            other=prefix+0.2;decoder.inference_logits(other,text)
            again=decoder.inference_logits(prefix,text)
            self.assertEqual(int(again[:,-1].argmax(-1)),manual[0])
        self.assertTrue(all(torch.equal(v,decoder.lm.state_dict()[k]) for k,v in before.items()))
    def test_hrm_mask_all50_layers_and_bad_validity_or_cache_refused(self):
        model,c,contract,root=native('hrm_prefixlm');model.eval();model.requires_grad_(False)
        lm=model.caption_decoder.lm;audit=observers.install_trace_audit(lm,'hrm_prefixlm')
        x=torch.randn(1,384,16);valid=torch.ones(1,384,dtype=torch.long);valid[:,132:383]=0
        with torch.inference_mode():
            lm(inputs_embeds=x,attention_mask=valid,prefix_length=383,bp_steps=5,use_cache=False)
            for change in ({'attention_mask':None},{'prefix_length':382},{'bp_steps':4},{'use_cache':True}):
                kwargs=dict(inputs_embeds=x,attention_mask=valid,prefix_length=383,bp_steps=5,use_cache=False);kwargs.update(change)
                with self.assertRaises((RuntimeError,ValueError)):lm(**kwargs)
        self.assertEqual(audit['checked_forwards'],1)
    def test_real_training_import_graph_and_four_actual_generator_adapters(self):
        import numpy as np
        import soundfile as sf
        from ced_eval_20261125 import reasonaqa_runtime,mmsu_runtime,benchmark
        for target in registry.ROUTES:
            # Load the real complete import graph from the pinned snapshot;
            # no constructor/asset loader is mocked by sources.import_training.
            modules=sources.import_training(target,cache())
            self.assertTrue(Path(modules['models.mellow'].__file__).resolve().is_relative_to(cache()))
            model,c,contract,root=native(target);decoder=model.caption_decoder
            mellow_class=importlib.import_module('models.mellow').Mellow
            class Encoder(nn.Module):
                def __init__(self):
                    super().__init__();self.calls=0;self.register_buffer('audio',torch.randn(1,126,16))
                    self.register_buffer('a4',torch.randn(1,126,16));self.register_buffer('a8',torch.randn(1,126,16))
                def forward(self,wave):
                    self.calls+=1
                    self_outer.assertEqual(tuple(wave.shape),(1,640000))
                    return self.audio,torch.zeros(1,527),{'embedding':self.audio,'layer4_audio':self.a4,'layer8_audio':self.a8}
            self_outer=self
            mellow=mellow_class.__new__(mellow_class);nn.Module.__init__(mellow)
            mellow.caption_decoder=decoder;mellow.audio_encoder=Encoder();mellow.eval();mellow.requires_grad_(False)
            prompt=torch.full((1,256),17,dtype=torch.long);prompt[:,:3]=torch.tensor([3,4,0])
            mask=torch.zeros_like(prompt);mask[:,:3]=1;text={'input_ids':prompt,'attention_mask':mask}
            class Tokenizer:
                eos_token_id=0;pad_token_id=17
                def encode(self,*args,**kw):return [3,4,0]
                def decode(self,ids,**kw):return str(ids)
            tok=Tokenizer();text_encoder=types.SimpleNamespace(tokenize=lambda *a: {k:v.clone() for k,v in text.items()})
            fixed20=importlib.import_module('data.fixed20')
            with tempfile.TemporaryDirectory() as directory:
                wav=Path(directory)/'stereo.wav'
                samples=np.stack([np.sin(np.arange(1600)*.01),np.cos(np.arange(1600)*.02)],axis=1).astype('float32')
                sf.write(wav,samples,16000,subtype='FLOAT')
                waveform=fixed20.load_mono(str(wav))
                payload={'bytes':wav.read_bytes(),'path':'stale.wav'}
                decoded,audio_audit=mmsu_runtime.decode_audio(payload,directory)
                torch.testing.assert_close(waveform,decoded,atol=0,rtol=0)
                halves=fixed20.load_fixed20(str(wav));inputs={'audio1':halves[0].unsqueeze(0),'audio2':halves[1].unsqueeze(0),'input':text}
                audit={'eval_variant':target}
                loaded=(mellow,tok,torch.device('cpu'),text_encoder,fixed20,audit,{'data.fixed20':fixed20})
                with torch.inference_mode():
                    prefix,output,ctx=mellow.generate_prefix_inference(inputs)
                    if target=='hrm_prefixlm':first=decoder.inference_logits(prefix,text)
                    else:first=decoder.lm(inputs_embeds=prefix,use_cache=False,audio_information=ctx).logits
                    tok.eos_token_id=int(first[:,-1].argmax(-1))
                trace=observers.install_trace_audit(decoder.lm,target)
                stat=wav.stat();record={'original_record':{'input':'Which country?'},'audio1':{'path':str(wav),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns},'audio2':None}
                with patch.object(runtime,'HIDDEN_SIZE',16):
                    for task in ('reasonaqa5000','mmau','mmar','mmsu'):
                        mellow.audio_encoder.calls=0
                        if task=='reasonaqa5000':result=reasonaqa_runtime.generate(loaded,record,variant=target)
                        elif task=='mmsu':result=mmsu_runtime.generate(loaded,{'prompt':'which country? a) india b) none c) china d) japan','waveform':decoded,'audio_audit':audio_audit},variant=target)
                        else:result=benchmark.generate(loaded,{'prompt':'which country? a) india b) none c) china d) japan','waveform':waveform})
                        self.assertEqual(result['generated_token_ids'],[tok.eos_token_id]);self.assertEqual(mellow.audio_encoder.calls,1)
                        self.assertEqual(result['first_answer_logit_position'],382)
                        if task in ('mmau','mmar'):self.assertEqual(result['prompt_padding_visible'],target=='audio8_read3')
                self.assertEqual(trace['checked_forwards'],4)

if __name__=='__main__':unittest.main()
