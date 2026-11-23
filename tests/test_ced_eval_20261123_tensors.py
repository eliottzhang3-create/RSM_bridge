"""Optional real CPU forward/reload tests; never production assets or CUDA."""
import importlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/RSmol'),str(ROOT/'tests')]
from ced_eval_20261123 import registry,sources,runtime,observers
from test_ced_eval_20261123 import cache

MISSING=[name for name in ('torch','transformers','safetensors') if importlib.util.find_spec(name) is None]

@unittest.skipIf(MISSING,'Real CPU tensor dependencies missing: '+', '.join(MISSING))
class RealTensorTests(unittest.TestCase):
    def check_model(self,target):
        import torch
        from transformers import LlamaConfig
        torch.set_num_threads(1);torch.manual_seed(1234)
        spec=registry.variant_spec(target);old_path=list(sys.path);old_modules=dict(sys.modules)
        try:
            sources.activate(target,cache())
            text=importlib.import_module(spec['text_package']+'.contract')
            implementation=importlib.import_module(spec['implementation'])
            config=LlamaConfig(num_hidden_layers=30,hidden_size=16,intermediate_size=32,
                num_attention_heads=4,num_key_value_heads=2,vocab_size=43,tie_word_embeddings=True,
                pad_token_id=17,eos_token_id=0,max_position_embeddings=1024,use_cache=False,
                attention_dropout=0.,_attn_implementation='eager')
            def model():
                lm=getattr(implementation,spec['model_class'])(text.build_target_config(config)).float()
                if target=='audio7_read2':lm.model.enable_audio_read2();lm.model.enable_loop_lora()
                return lm
            lm=model()
            with torch.no_grad():
                for name,parameter in lm.named_parameters():
                    if any(k in name for k in ('lora_B','audio_read_routers','write_routers','read_routers')):parameter.add_(.013)
            if target=='audio7_read2':
                loader=importlib.import_module('models.recursive_text')
                before={k:v.clone() for k,v in lm.state_dict().items()}
                loader.validate_audio_read2(lm)
                self.assertTrue(all(torch.equal(value,lm.state_dict()[key]) for key,value in before.items()))
            state={k:v.clone() for k,v in lm.state_dict().items()}
            restored=model();restored.load_state_dict(state,strict=True)
            self.assertTrue(all(torch.equal(value,restored.state_dict()[key]) for key,value in state.items()))
            self.assertIs(restored.model.embed_tokens.weight,restored.lm_head.weight)
            lm.eval();restored.eval();audit=observers.install_trace_audit(restored,target)
            prefix=torch.randn(1,383,16)
            context={} if target=='mesh_71437' else {'audio_information':tuple(torch.randn(1,126,16) for _ in range(2))}
            with torch.inference_mode():
                expected=lm(inputs_embeds=prefix,use_cache=False,**context).logits
                actual=restored(inputs_embeds=prefix,use_cache=False,**context).logits
                torch.testing.assert_close(expected,actual,rtol=0,atol=0)
                restored(inputs_embeds=torch.cat((prefix,torch.randn(1,1,16)),dim=1),use_cache=False,**context)
                restored(inputs_embeds=prefix,use_cache=False,**context)
                eos=int(actual[:,-1].argmax(-1).item())
                tokenizer=types.SimpleNamespace(eos_token_id=eos,decode=lambda ids,**kw:str(ids))
                with patch.object(runtime,'HIDDEN_SIZE',16):
                    generation=runtime.greedy_decode(types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=restored)),
                        tokenizer,prefix,eval_variant=target,context=context)
                self.assertEqual(generation['generated_token_ids'],[eos])
            self.assertEqual(audit['checked_forwards'],4)
            self.assertEqual(audit['last_schedule'],list(spec['schedule']))
        finally:
            prefixes=('models','data','utils','contract','mizar_text_7_14x2_7_mesh','mizar_text_7_14x3_7_mesh',
                      'recursive_model_mizar_text_7_14x2_7_mesh','recursive_model_mizar_text_7_14x3_7_mesh')
            for name in list(sys.modules):
                if any(name==p or name.startswith(p+'.') for p in prefixes):del sys.modules[name]
            sys.modules.update({n:m for n,m in old_modules.items() if any(n==p or n.startswith(p+'.') for p in prefixes)})
            sys.path[:]=old_path
    def test_mesh_71437(self):self.check_model('mesh_71437')
    def test_audio7_read2(self):self.check_model('audio7_read2')

if __name__=='__main__':unittest.main()
