"""Observer fixtures plus optional real CPU models; no production assets or CUDA."""
import importlib
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/RSmol'))
from text71427_ced_eval import registry, runtime, observers


class HookModule:
    def __init__(self,inputs=16,outputs=16):
        self.pre=[];self.post=[];self.loop_index=None;self.in_features=inputs;self.out_features=outputs
    def register_forward_pre_hook(self,callback,**kw):self.pre.append((callback,kw.get('with_kwargs',False)))
    def register_forward_hook(self,callback):self.post.append(callback)
    def __call__(self,value):
        for callback,_ in self.pre:callback(self,(value,))
        output=np.zeros((*value.shape[:-1],self.out_features),np.float32)
        for callback in self.post:callback(self,(value,),output)
        return output


def observed_fixture(alias,length=383,fault=None):
    width=16
    class Layer(HookModule):
        def __init__(self):
            super().__init__();self.projections={t:HookModule() for t in observers.TARGETS}
        def get_submodule(self,target):return self.projections[target]
    core=types.SimpleNamespace(layers=[Layer() for _ in range(28)],gradient_checkpointing=False,
        write_routers=[HookModule(outputs=5) for _ in range(3)],
        read_routers=[HookModule(outputs=5) for _ in range(3)],last_forward_trace=[],last_loop_lora_trace=[])
    if alias.startswith('audio7'):
        core.audio_read_routers=[HookModule(outputs=n) for n in ([7,7,5] if alias.endswith('ar6') else [7,7])]
    if alias.endswith('ar6'):core.audio_write_routers=[HookModule(outputs=5) for _ in range(3)]
    if alias in ('loop_lora','listen'):core.audio_gates={str(i):HookModule(outputs=1) for i in (4,8)}
    if alias=='listen':core.listen_routers={str(i):HookModule(outputs=5) for i in (4,8)}
    if alias=='wfuse':core.audio_fusions={str(i):HookModule(inputs=32) for i in (4,8)}
    lm=HookModule();lm.model=core;lm.config=types.SimpleNamespace(hidden_size=width)
    audit=observers.install_trace_audit(lm,alias)
    value=np.zeros((1,length,width),np.float32);audio=value[:,:126];text=value[:,126:]
    for cb,_ in lm.pre:cb(lm,(),{'inputs_embeds':value,'use_cache':False})
    def interaction(index):
        if alias.endswith('ar6'):
            core.audio_write_routers[index](audio);core.write_routers[index](text)
            core.audio_read_routers[index](audio);core.read_routers[index](text)
        elif alias=='audio7_read2' and index<2:
            core.write_routers[index](value);core.audio_read_routers[index](audio);core.read_routers[index](text)
        else:
            core.write_routers[index](value);core.read_routers[index](value)
        if index<2:
            layer=str((4,8)[index])
            if alias in ('listen','loop_lora'):core.audio_gates[layer](audio)
            if alias=='listen':core.listen_routers[layer](audio)
            if alias=='wfuse':core.audio_fusions[layer](np.zeros((1,126,32),np.float32))
        if fault=='extra_router' and index==0:core.read_routers[0](value)
    for logical,physical in enumerate(registry.SCHEDULE):
        if fault=='missing_layer' and logical==41:continue
        if alias!='base' and 7<=logical<35:
            loop=(logical-7)//14
            for target in observers.TARGETS:
                module=core.layers[physical].get_submodule(target)
                module.loop_index=1-loop if fault=='wrong_lora' else loop
                try:module(value)
                finally:module.loop_index=None
            core.last_loop_lora_trace.append((logical,physical,loop))
        core.last_forward_trace.append({'physical_index':physical})
        for cb in core.layers[physical].post:cb(core.layers[physical],(value,),value)
        if logical in (6,20,34):interaction((logical-6)//14)
    output=types.SimpleNamespace(logits=np.zeros((1,length,4)),past_key_values=None)
    for cb in lm.post:cb(lm,(),output)
    return audit


class ObserverTests(unittest.TestCase):
    def test_hooks_accept_all_six_real_geometry_event_sequences(self):
        for alias in registry.ROUTES:
            for length in (383,384,682):
                audit=observed_fixture(alias,length)
                self.assertEqual(audit['status'],'PASS');self.assertEqual(audit['checked_forwards'],1)
                self.assertEqual(audit['last_schedule'],list(registry.SCHEDULE))
                if alias!='base':self.assertEqual(audit['loop_lora_calls_per_loop'],[98,98])

    def test_hooks_reject_skipped_layer_wrong_loop_or_extra_router(self):
        for fault in ('missing_layer','wrong_lora','extra_router'):
            for alias in set(registry.ROUTES)-{'base'}:
                with self.subTest(alias=alias,fault=fault),self.assertRaises(RuntimeError):
                    observed_fixture(alias,fault=fault)


MISSING=[name for name in ('torch','transformers','safetensors') if importlib.util.find_spec(name) is None]


@unittest.skipIf(MISSING,'Real CPU tensor dependencies missing: '+', '.join(MISSING))
class RealTensorTests(unittest.TestCase):
    def check_model(self,alias):
        import torch
        from transformers import LlamaConfig
        from mizar_text_7_14x2_7_mesh.contract import build_target_config
        torch.set_num_threads(1);torch.manual_seed(123)
        spec=registry.ROUTES[alias]
        # Each route has a different namespace package named models. Match the public loader's isolation.
        old_path=list(sys.path)
        old_modules={name:module for name,module in sys.modules.items() if name=='models' or name.startswith('models.') or name=='contract'}
        for name in old_modules:del sys.modules[name]
        sys.path.insert(0,str(spec['route_root']))
        try:
            implementation=importlib.import_module(spec['implementation'])
            def model():
                config=LlamaConfig(num_hidden_layers=30,hidden_size=16,intermediate_size=32,
                    num_attention_heads=4,num_key_value_heads=2,vocab_size=43,tie_word_embeddings=True,
                    pad_token_id=17,eos_token_id=0,max_position_embeddings=1024,use_cache=False,
                    attention_dropout=0.,_attn_implementation='eager')
                lm=implementation.MizarText7_14x2_7MeshForCausalLM(build_target_config(config)).float()
                core=lm.model
                if alias=='loop_lora':core.enable_layer_injection();core.enable_loop_lora()
                elif alias=='listen':
                    core.enable_audio_listen();importlib.import_module('models.loop_lora').install_loop_lora(core)
                elif alias.startswith('audio7'):core.enable_audio_read2();core.enable_loop_lora()
                elif alias=='wfuse':
                    core.enable_audio_fusion();importlib.import_module('models.loop_lora').install_loop_lora(core)
                return lm.eval()
            lm=model()
            with torch.no_grad():
                for name,value in lm.named_parameters():
                    if any(key in name for key in ('lora_B','audio_gates','audio_fusions','audio_read_routers','audio_write_routers')):
                        value.add_(.013)
            state={k:v.clone() for k,v in lm.state_dict().items()}
            restored=model();restored.load_state_dict(state,strict=True)
            for name,value in state.items():self.assertTrue(torch.equal(value,restored.state_dict()[name]),name)
            self.assertIs(restored.model.embed_tokens.weight,restored.lm_head.weight)
            trace=observers.install_trace_audit(restored,alias)
            prefix=torch.randn(1,383,16)
            pair=tuple(torch.randn(1,126,16) for _ in range(2))
            kwargs={} if alias=='base' else {'audio_information':pair}
            with torch.inference_mode():
                before=lm(inputs_embeds=prefix,use_cache=False,**kwargs).logits
                after=restored(inputs_embeds=prefix,use_cache=False,**kwargs).logits
                torch.testing.assert_close(before,after,rtol=0,atol=0)
                restored(inputs_embeds=torch.cat((prefix,torch.randn(1,1,16)),dim=1),use_cache=False,**kwargs)
                restored(inputs_embeds=prefix,use_cache=False,**kwargs)
            self.assertEqual(trace['checked_forwards'],3)
            self.assertEqual(trace['last_sequence_length'],383)
            # The exact copied greedy loop's first token is predicted at prefix position 382.
            eos=int(after[:,-1].argmax(-1).item())
            tokenizer=types.SimpleNamespace(eos_token_id=eos,decode=lambda ids,**kw:str(ids))
            with patch.object(runtime,'HIDDEN_SIZE',16),torch.inference_mode():
                generated=runtime.greedy_decode(types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=restored)),
                    tokenizer,prefix,eval_variant=alias,context=kwargs)
            self.assertEqual(generated['generated_token_ids'],[eos])
        finally:
            for name in list(sys.modules):
                if name=='models' or name.startswith('models.') or name=='contract':del sys.modules[name]
            sys.modules.update(old_modules);sys.path[:]=old_path

    def test_base_model(self):self.check_model('base')
    def test_loop_lora_model(self):self.check_model('loop_lora')
    def test_listen_model(self):self.check_model('listen')
    def test_audio7_read2_model(self):self.check_model('audio7_read2')
    def test_wfuse_model(self):self.check_model('wfuse')
    def test_ar6_model(self):self.check_model('audio7_read2_ar6')


if __name__=='__main__':unittest.main()
