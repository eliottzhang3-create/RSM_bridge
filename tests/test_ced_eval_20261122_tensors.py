"""Optional real CPU tensor checks; no production checkpoint or CUDA needed."""
import importlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/RSmol'),str(ROOT/'tests')]
from ced_eval_20261122 import runtime, registry, sources, plain
from text71427_ced_eval import observers
from test_ced_eval_20261122 import cache

MISSING=[n for n in ('torch','transformers','safetensors') if importlib.util.find_spec(n) is None]

@unittest.skipIf(MISSING,'Real CPU tensor dependencies missing: '+', '.join(MISSING))
class RealTensorTests(unittest.TestCase):
    def check_model(self,target):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.set_num_threads(1);torch.manual_seed(123)
        spec=registry.ROUTES[target];variant=spec['model_variant']
        old_path=list(sys.path);old_modules=dict(sys.modules)
        try:
            sources.activate(target,cache())
            config=LlamaConfig(num_hidden_layers=30,hidden_size=16,intermediate_size=32,
                num_attention_heads=4,num_key_value_heads=2,vocab_size=43,tie_word_embeddings=True,
                pad_token_id=17,eos_token_id=0,max_position_embeddings=1024,use_cache=False,
                attention_dropout=0.,_attn_implementation='eager')
            if spec['plain']:
                factory=lambda:LlamaForCausalLM(config).float().eval()
            else:
                from mizar_text_7_14x2_7_mesh.contract import build_target_config
                implementation=importlib.import_module(spec['implementation'])
                def factory():
                    lm=implementation.MizarText7_14x2_7MeshForCausalLM(build_target_config(config)).float()
                    core=lm.model
                    if variant=='loop_lora':core.enable_layer_injection();core.enable_loop_lora()
                    elif variant=='listen':
                        core.enable_audio_listen();importlib.import_module('models.loop_lora').install_loop_lora(core)
                    elif variant=='audio7_read2':core.enable_audio_read2();core.enable_loop_lora()
                    elif variant=='wfuse':
                        core.enable_audio_fusion();importlib.import_module('models.loop_lora').install_loop_lora(core)
                    return lm.eval()
            lm=factory()
            with torch.no_grad():
                for name,value in lm.named_parameters():
                    if any(k in name for k in ('lora_B','audio_gates','audio_fusions','audio_read_routers')):value.add_(.013)
            state={k:v.clone() for k,v in lm.state_dict().items()}
            restored=factory();restored.load_state_dict(state,strict=True)
            self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in state.items()))
            self.assertIs(restored.model.embed_tokens.weight,restored.lm_head.weight)
            trace=plain.install_trace_audit(restored) if spec['plain'] else observers.install_trace_audit(restored,variant)
            prefix=torch.randn(1,383,16)
            context={} if spec['context_kind']=='none' else {'audio_information':tuple(torch.randn(1,126,16) for _ in range(2))}
            with torch.inference_mode():
                expected=lm(inputs_embeds=prefix,use_cache=False,**context).logits
                actual=restored(inputs_embeds=prefix,use_cache=False,**context).logits
                torch.testing.assert_close(expected,actual,rtol=0,atol=0)
                restored(inputs_embeds=torch.cat((prefix,torch.randn(1,1,16)),dim=1),use_cache=False,**context)
                restored(inputs_embeds=prefix,use_cache=False,**context)
                eos=int(actual[:,-1].argmax(-1).item())
                tok=types.SimpleNamespace(eos_token_id=eos,decode=lambda ids,**kw:str(ids))
                with patch.object(runtime,'HIDDEN_SIZE',16):
                    result=runtime.greedy_decode(types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=restored)),
                        tok,prefix,eval_variant=target,context=context)
                self.assertEqual(result['generated_token_ids'],[eos])
            self.assertEqual(trace['checked_forwards'],4)
            self.assertEqual(trace['last_schedule'],list(range(30)) if spec['plain'] else list(registry.SCHEDULE))
        finally:
            prefixes=('models','data','utils','contract','mizar_text_7_14x2_7_mesh','recursive_model_mizar_text_7_14x2_7_mesh')
            for name in list(sys.modules):
                if any(name==p or name.startswith(p+'.') for p in prefixes):del sys.modules[name]
            sys.modules.update({n:m for n,m in old_modules.items() if any(n==p or n.startswith(p+'.') for p in prefixes)})
            sys.path[:]=old_path
    def test_mesh_original(self):self.check_model('mesh_original')
    def test_audio7_read2(self):self.check_model('audio7_read2')
    def test_listen(self):self.check_model('listen')
    def test_loop_lora(self):self.check_model('loop_lora')
    def test_wfuse(self):self.check_model('wfuse')
    def test_mesh_reinit(self):self.check_model('mesh_reinit')
    def test_plain_smollm2(self):self.check_model('smol_gbs128')

if __name__=='__main__':unittest.main()
