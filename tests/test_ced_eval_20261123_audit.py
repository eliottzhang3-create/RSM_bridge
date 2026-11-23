"""Execute real audit_model + pinned validators on explicit metadata-only modules.

No audit function is mocked. Tensor/module objects below stand in for absent
Torch; this exercises production geometry/storage/control flow, not numerics.
"""
import ast
from collections import defaultdict
import copy
import importlib.util
import itertools
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/RSmol'),str(ROOT/'tests')]
from ced_eval_20261123 import registry,runtime,sources,observers
from test_ced_eval_20261123 import cache,checkpoint_fixture

_pointers=itertools.count(1)
class Parameter:
    def __init__(self,*shape):
        self.shape=shape;self.requires_grad=True;self.device='cpu';self.dtype='float32';self.ptr=next(_pointers)
    def numel(self):return math.prod(self.shape)
    def untyped_storage(self):return types.SimpleNamespace(data_ptr=lambda:self.ptr)

class Linear:
    def __init__(self,inputs,outputs,bias=True):
        self.in_features=inputs;self.out_features=outputs
        self.weight=Parameter(outputs,inputs);self.bias=Parameter(outputs) if bias else None
    def named_parameters(self):
        yield 'weight',self.weight
        if self.bias is not None:yield 'bias',self.bias
    def parameters(self):return (p for _,p in self.named_parameters())

class LoopLoRALinear(Linear):
    def __init__(self,inputs,outputs):
        super().__init__(inputs,outputs,bias=False)
        self.rank=8;self.alpha=16;self.scaling=2.;self.loop_index=None
        self.lora_A=[Parameter(8,inputs) for _ in range(2)]
        self.lora_B=[Parameter(outputs,8) for _ in range(2)]
    def named_parameters(self):
        yield from super().named_parameters()
        for name in ('lora_A','lora_B'):
            for i,p in enumerate(getattr(self,name)):yield f'{name}.{i}',p

class Modules(list):
    def parameters(self):return (p for module in self for p in module.parameters())

TARGETS=('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
         'mlp.gate_proj','mlp.up_proj','mlp.down_proj')

class Layer:
    def __init__(self,index,read2):
        dims=[(576,576),(576,192),(576,192),(576,576),(576,1536),(576,1536),(1536,576)]
        self.projections={name:(LoopLoRALinear if read2 and 7<=index<21 else
            lambda a,b:Linear(a,b,bias=False))(a,b) for name,(a,b) in zip(TARGETS,dims)}
    def get_submodule(self,target):return self.projections[target]

class Core:
    def __init__(self,spec):
        read2=spec['context_kind']=='pair'
        self.layers=[Layer(i,read2) for i in range(28)];self.gradient_checkpointing=False
        self.write_routers=Modules(Linear(576,spec['slots']) for _ in range(spec['router_groups']))
        self.read_routers=Modules(Linear(576,spec['slots']) for _ in range(spec['router_groups']))
        if read2:self.audio_read_routers=Modules(Linear(576,7) for _ in range(2))
        self.audio_tokens=126;self.loop_lora_enabled=read2
        self.embed_tokens=types.SimpleNamespace(weight=Parameter(49152,576))
    def named_parameters(self):
        yield 'embed_tokens.weight',self.embed_tokens.weight
        for kind in ('write','read','audio_read'):
            for i,module in enumerate(getattr(self,kind+'_routers',[])):
                for name,p in module.named_parameters():yield f'{kind}_routers.{i}.{name}',p
        for i,layer in enumerate(self.layers):
            for target,module in layer.projections.items():
                for name,p in module.named_parameters():yield f'layers.{i}.{target}.{name}',p

class LM:
    def __init__(self,spec,config):
        self.model=Core(spec);self.config=types.SimpleNamespace(**config)
        self.config.to_dict=lambda:config
        self.lm_head=types.SimpleNamespace(weight=self.model.embed_tokens.weight)
    def named_parameters(self,remove_duplicate=False):
        for name,p in self.model.named_parameters():yield 'model.'+name,p
        yield 'lm_head.weight',self.lm_head.weight
    def parameters(self):return (p for _,p in self.named_parameters())
    def state_dict(self):return dict(self.named_parameters())

class MLPs(dict):
    def parameters(self):return (p for module in self.values() for p in module.parameters())

def extract(path,names,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    body += [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    code=ast.fix_missing_locations(ast.Module(body=body,type_ignores=[]))
    exec(compile(code,str(path),'exec'),namespace)

def fixture(target):
    c,s,_=checkpoint_fixture(target);spec=registry.variant_spec(target)
    root=sources.route_root(target,cache()).resolve()
    text=__import__(spec['text_package']+'.contract',fromlist=['contract'])
    config={**text.architecture_fields(),'hidden_size':576,'vocab_size':49152,
            'tie_word_embeddings':True,'model_type':'llama'}
    lm=LM(spec,config)
    model=types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=lm,text_model_contract=spec['text_contract'],is_recursive_mesh=True),
        state_dict=lm.state_dict,parameters=lm.parameters)
    if target=='audio7_read2':
        # Exact trained CEDFrequencyMerger parameter totals; no fake audio inference.
        model.audio_encoder=types.SimpleNamespace(base=types.SimpleNamespace(layer_mlps=MLPs({
            '4':types.SimpleNamespace(parameters=lambda:iter([Parameter(1221888)])),
            '8':types.SimpleNamespace(parameters=lambda:iter([Parameter(1221888)]))})))
    definition=importlib.util.spec_from_file_location('models.recursive_text',root/'models/recursive_text.py')
    loader=importlib.util.module_from_spec(definition)
    with patch.dict(sys.modules,{'contract':c}):definition.loader.exec_module(loader)
    implementation_path=(root if spec['implementation'].startswith('models.') else root.parents[1])/ (spec['implementation'].replace('.','/')+'.py')
    implementation=types.ModuleType(spec['implementation']);implementation.__file__=str(implementation_path)
    setattr(implementation,spec['model_class'],LM)
    extract(implementation_path,['parameter_audit'],implementation.__dict__)
    implementation.__dict__.update(defaultdict=defaultdict,SCHEDULE=spec['schedule'],LOGICAL_TO_PHYSICAL=spec['schedule'],
        RECURSIVE_LOOPS=spec['loops'],MEMORY_SLOT_COUNT=spec['slots'],ROUTER_COUNT=spec['router_groups'],
        SOURCE_LAYER_INDICES_0BASED=text.SOURCE_LAYERS)
    modules={'models.recursive_text':loader,spec['implementation']:implementation,
        'torch':types.SimpleNamespace(float32='float32')}
    if target=='audio7_read2':
        loop=types.ModuleType('models.loop_lora');loop.__file__=str(root/'models/loop_lora.py')
        loop.__dict__.update(nn=types.SimpleNamespace(Linear=Linear),LoopLoRALinear=LoopLoRALinear,
            TARGETS=TARGETS,MIDDLE_LAYERS=tuple(range(7,21)),RANK=8,ALPHA=16,LOOPS=2)
        extract(root/'models/loop_lora.py',['projections','validate_loop_lora'],loop.__dict__)
        modules['models.loop_lora']=loop
    # Its provenance reader is real too; replace only external filesystem assets.
    c.recursive_checkpoint_contract=lambda path:s['contract']['recursive_text_checkpoint']
    return model,c,s['contract'],root,modules

class ActualAuditTests(unittest.TestCase):
    def test_exact_fresh_error_reproduced_and_real_validator_passes_trained_geometry(self):
        model,c,contract,root,modules=fixture('audio7_read2');loader=modules['models.recursive_text']
        with self.assertRaisesRegex(TypeError,"unexpected keyword argument 'fresh'"):
            loader.validate_audio_read2(model.caption_decoder.lm,fresh=False)
        self.assertEqual(loader.validate_audio_read2(model.caption_decoder.lm)['router_modules'],8)
        with patch.dict(sys.modules,modules):audit=runtime.audit_model(model,c,contract,'audio7_read2',root)
        self.assertEqual(audit['new_parameters'],4731278);self.assertEqual(audit['loop_lora']['matrix_count'],392)
        self.assertTrue(audit['trained_values_preserved'])
    def test_actual_architecture_audit_both_targets_and_real_old_entrypoint_regression(self):
        for target in registry.ROUTES:
            model,c,contract,root,modules=fixture(target)
            with patch.dict(sys.modules,modules):audit=runtime.audit_model(model,c,contract,target,root)
            self.assertEqual(audit['logical_layer_count'],56 if target=='mesh_71437' else 42)
            self.assertEqual(audit['router_modules'],8)
            if target=='audio7_read2':
                from ced_eval_20261122 import runtime as old
                from text71427_ced_eval import runtime as oldest
                with patch.dict(sys.modules,modules):
                    self.assertEqual(old.audit_model(model,c,contract,target,root)['new_parameters'],4731278)
                    with patch.object(oldest,'variant_spec',return_value={**oldest.variant_spec(target),'route_root':root}):
                        self.assertEqual(oldest.audit_model(model,c,contract,target)['new_parameters'],4731278)
    def test_wrong_layer_shape_alias_router_lora_dtype_and_provenance_rejected(self):
        for target in registry.ROUTES:
            for fault in ('layer','router_shape','router_storage','dtype','config','text_contract','provenance'):
                model,c,contract,root,modules=fixture(target);lm=model.caption_decoder.lm
                if fault=='layer':lm.model.layers[-1]=lm.model.layers[0]
                elif fault=='router_shape':lm.model.read_routers[0].weight.shape=(5 if target=='mesh_71437' else 6,576)
                elif fault=='router_storage':lm.model.read_routers[0].weight.ptr=lm.model.write_routers[0].weight.ptr
                elif fault=='dtype':lm.model.embed_tokens.weight.dtype='float64'
                elif fault=='config':lm.config.to_dict=lambda:{}
                elif fault=='text_contract':model.caption_decoder.text_model_contract='wrong'
                elif fault=='provenance':c.recursive_checkpoint_contract=lambda path:{'changed':True}
                with patch.dict(sys.modules,modules),self.subTest(target=target,fault=fault),self.assertRaises(ValueError):
                    runtime.audit_model(model,c,contract,target,root)
        for fault in ('router_shape','lora_shared','lora_selected','mlp_shared'):
            model,c,contract,root,modules=fixture('audio7_read2');core=model.caption_decoder.lm.model
            if fault=='router_shape':core.audio_read_routers[0].weight.shape=(5,576)
            elif fault=='lora_shared':
                projection=core.layers[7].get_submodule(TARGETS[0]);projection.lora_A[1]=projection.lora_A[0]
            elif fault=='lora_selected':core.layers[7].get_submodule(TARGETS[0]).loop_index=0
            elif fault=='mlp_shared':model.audio_encoder.base.layer_mlps['8']=model.audio_encoder.base.layer_mlps['4']
            with patch.dict(sys.modules,modules),self.subTest(fault=fault),self.assertRaises(ValueError):runtime.audit_model(model,c,contract,'audio7_read2',root)
    def test_trained_validator_does_not_change_tensor_metadata_or_parameter_ids(self):
        for target in registry.ROUTES:
            model,c,contract,root,modules=fixture(target)
            def state():return [(n,id(p),p.shape,p.dtype,p.ptr,p.requires_grad) for n,p in model.caption_decoder.lm.named_parameters()]
            before=state()
            with patch.dict(sys.modules,modules):runtime.audit_model(model,c,contract,target,root)
            self.assertEqual(state(),before)

class ForwardObserverTests(unittest.TestCase):
    def test_actual56_event_schedule_and_wrong_router_layer_cache_rejected(self):
        import numpy as np
        from test_text71427_ced_eval_tensors import HookModule
        for length in (383,384,682):
            for fault in (None,'layer','router','shape','mask','cache'):
                core=types.SimpleNamespace(layers=[HookModule() for _ in range(28)],gradient_checkpointing=False,
                    write_routers=[HookModule(outputs=6) for _ in range(4)],read_routers=[HookModule(outputs=6) for _ in range(4)],last_forward_trace=[])
                lm=HookModule();lm.model=core;lm.config=types.SimpleNamespace(hidden_size=16)
                audit=observers.install_trace_audit(lm,'mesh_71437')
                value=np.zeros((1,length,16),np.float32)
                def forward():
                    for cb,_ in lm.pre:cb(lm,(),{'inputs_embeds':value,'use_cache':False,'attention_mask':True if fault=='mask' else None})
                    for logical,physical in enumerate(registry.variant_spec('mesh_71437')['schedule']):
                        if fault=='layer' and logical==55:continue
                        core.last_forward_trace.append({'physical_index':physical})
                        for cb in core.layers[physical].post:cb(core.layers[physical],(value,),value)
                        if logical in (6,20,34,48):
                            index=(logical-6)//14;core.write_routers[index](value)
                            if fault=='router' and index==3:continue
                            core.read_routers[index](value[:,:126] if fault=='shape' else value)
                    for cb in lm.post:cb(lm,(),types.SimpleNamespace(logits=np.zeros((1,length,4)),past_key_values=() if fault=='cache' else None))
                if fault:
                    with self.assertRaises(RuntimeError):forward()
                else:
                    forward();self.assertEqual(audit['status'],'PASS');self.assertEqual(len(audit['last_router_events']),8)
                    self.assertEqual(audit['last_schedule'],list(registry.variant_spec('mesh_71437')['schedule']))

if __name__=='__main__':unittest.main()
