"""Execute production routing/forward source with NumPy, plus optional real CPU autograd.

NumPy fixtures cover the three interactions, not Transformer/autograd correctness.
The real tensor class never downloads weights and explicitly skips missing dependencies.
"""
import ast
import copy
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager, nullcontext

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT/'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_lora_r8'
sys.path[:0] = [str(ROUTE), str(ROOT/'code/RSmol')]
import audit_reference as A
import contract as C


def function(file, name, scope):
    fn = copy.deepcopy(A.find_function(A.tree(ROUTE/file), name))
    fn.decorator_list, fn.returns = [], None
    for arg in fn.args.args + fn.args.kwonlyargs:
        arg.annotation = None
    A.execute([fn], scope)
    return scope[name]


class Tensor(np.ndarray):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        if np.isscalar(value) and isinstance(key, int):
            index = key % self.shape[0]
            return super().__getitem__(slice(index,index+1)).reshape(()).view(Tensor)
        return value
    def detach(self): return self
    def cpu(self): return self
    def clone(self): return self.copy()
    def unsqueeze(self, dim): return np.expand_dims(self, dim).view(Tensor)
    def transpose(self, a, b): return self.swapaxes(a, b).view(Tensor)
    def new_zeros(self, *shape): return np.zeros(shape, dtype=self.dtype).view(Tensor)
    def float(self): return self.astype(np.float32).view(Tensor)
    def to(self, *args, **kwargs): return self.astype(kwargs.get('dtype', self.dtype)).view(Tensor)
    def copy_(self, value): np.copyto(self, value); return self
    def ne(self, value): return self != value
    def sum(self, axis=None, keepdims=False, dim=None, **kwargs):
        return super().sum(axis=dim if dim is not None else axis,keepdims=keepdims,**kwargs)
    @property
    def device(self): return 'cpu'


def T(value): return np.asarray(value, dtype=np.float64).view(Tensor)


class Linear:
    def __init__(self, width, slots, **kwargs):
        self.weight = T(np.random.randn(slots, width))
        self.bias = T(np.random.randn(slots))
    def __call__(self, q): return T(q @ self.weight.T + self.bias)


class Modules(list):
    def to(self, device): return self


@contextmanager
def fork_rng(**kwargs):
    state = np.random.get_state()
    try: yield
    finally: np.random.set_state(state)


def softmax(z, dim=-1):
    value = np.exp(z - z.max(axis=dim, keepdims=True))
    return T(value / value.sum(axis=dim, keepdims=True))


class RoutingSourceTests(unittest.TestCase):
    def fixture(self):
        np.random.seed(13)
        obj = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=3),
            embed_tokens=types.SimpleNamespace(weight=T(np.zeros((10, 3)))),
            write_routers=[Linear(3, 5) for _ in range(3)],
            read_routers=[Linear(3, 5) for _ in range(3)], audit_mode=True,
            gradient_audit_mode=False, routing_stats_mode=False)
        scope = dict(torch=types.SimpleNamespace(Tensor=Tensor,random=types.SimpleNamespace(fork_rng=fork_rng),
            no_grad=nullcontext, cat=lambda xs, dim: T(np.concatenate(xs, axis=dim))),
            nn=types.SimpleNamespace(ModuleList=Modules, Linear=Linear), math=math,
            F=types.SimpleNamespace(softmax=softmax), record_function=lambda name:nullcontext())
        source = 'models/recursive_audio7_read2.py'
        for name in ('enable_audio_read2', '_mixed_interaction', '_route', '_write', '_read'):
            fn = function(source, name, scope)
            setattr(obj, name, fn if name in ('_write', '_read') else types.MethodType(fn, obj))
        before = np.random.get_state()
        obj.enable_audio_read2()
        after = np.random.get_state()
        self.assertTrue(all(np.array_equal(a,b) for a,b in zip(before,after)))
        for i in range(2):
            old, new = obj.read_routers[i], obj.audio_read_routers[i]
            np.testing.assert_array_equal(new.weight[:5], old.weight)
            np.testing.assert_allclose(new.bias[:5], old.bias-math.log(2), atol=0, rtol=0)
            for row in (5,6):
                np.testing.assert_array_equal(new.weight[row], old.weight[0])
                self.assertEqual(new.bias[row], old.bias[0])
                self.assertFalse(np.shares_memory(new.weight, old.weight))
        def stack(h, indices, start, **kwargs):
            obj.events.append((start, h.copy()))
            for offset, physical in enumerate(indices):
                obj.last_forward_trace.append(dict(physical_index=physical, logical_index=start+offset))
                h = h + (physical+1)*.003
                if 7 <= physical < 21:
                    loop = kwargs['loop_index']
                    obj.last_loop_lora_trace += ((start+offset,physical,loop),)
                    h = h + (loop+1)*.002
            return h
        obj._run_stack, obj.norm = stack, lambda x:x
        x = T(np.random.default_rng(42).normal(size=(2,131,3)))
        pair = tuple(T(np.random.default_rng(k).normal(size=(2,126,3))) for k in (4,8))
        return obj, x, pair, scope

    def forward(self, obj, x, pair, scope):
        module = A.tree(ROUTE/'models/recursive_audio7_read2.py')
        core = next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='MeshLlamaModel')
        fn = A.find_function(core, 'forward')
        begin = next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
            and ast.unparse(n.targets[0]) == 'self.last_forward_trace')
        end = next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
            and ast.unparse(n.value) == 'self.norm(hidden)')
        obj.events = []
        scope = dict(scope, self=obj, hidden=x, audio_information=pair, batch_size=x.shape[0],
            query_length=x.shape[1], output_hidden_states=False, mask=None, position_ids=None,
            cache=None, use_cache=False, cache_position=None, position_embeddings=None,
            output_attentions=False)
        A.execute(copy.deepcopy(fn.body[begin:end+1]), scope)
        return scope['hidden'], scope['memory']

    def test_exact_initializer_rng_and_no_reinitialization(self):
        obj, _, _, _ = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'reset'): obj.enable_audio_read2()
        self.assertEqual(len(obj.audio_read_routers), 2)

    def test_query_dependent_probability_formula(self):
        obj,x,_,_=self.fixture()
        for old,new in zip(obj.read_routers,obj.audio_read_routers):
            z=old(x); p=softmax(z); new_p=softmax(new(x))
            stable=np.exp(z-z.max(axis=-1,keepdims=True))
            alpha=stable.sum(axis=-1,keepdims=True)/(stable.sum(axis=-1,keepdims=True)+4*stable[...,:1])
            np.testing.assert_allclose(new_p[...,:5],alpha*p,atol=1e-14)
            np.testing.assert_allclose(new_p[...,5:6],(1-alpha)/2,atol=1e-14)
            np.testing.assert_array_equal(new_p[...,5],new_p[...,6])
            self.assertGreater(np.ptp(alpha), .01)

    def test_three_actual_interactions_match_independent_reference(self):
        obj,x,pair,scope=self.fixture(); out,mem=self.forward(obj,x,pair,scope)
        audio,text=obj.last_initial_memory
        np.testing.assert_array_equal(audio[:,0],x[:,:126]); np.testing.assert_array_equal(audio[:,1:5],0)
        np.testing.assert_array_equal(audio[:,5],pair[0]); np.testing.assert_array_equal(audio[:,6],pair[1])
        np.testing.assert_array_equal(text[:,0],x[:,126:]); np.testing.assert_array_equal(text[:,1:],0)
        ma,mt=audio.copy(),text.copy(); h=x.copy()
        for stage,(indices,suffix) in enumerate(((range(7),'pre'),(range(7,21),'0'),(range(7,21),'1'))):
            v=h+sum((i+1)*.003 for i in indices)+(0 if stage==0 else 14*stage*.002)
            q=v if stage==0 else h
            w=softmax(obj.write_routers[stage](q))
            ma[:,:5] += v[:,:126,None,:].transpose(1,2)*np.asarray(w[:,:126]).swapaxes(1,2)[...,None]
            mt += v[:,126:,None,:].transpose(1,2)*np.asarray(w[:,126:]).swapaxes(1,2)[...,None]
            ar=softmax((obj.audio_read_routers[stage] if stage<2 else obj.read_routers[stage])(q[:,:126]))
            tr=softmax(obj.read_routers[stage](q[:,126:]))
            ah=(ma[:,:ar.shape[-1]]*np.asarray(ar).swapaxes(1,2)[...,None]).sum(axis=1)
            th=(mt*np.asarray(tr).swapaxes(1,2)[...,None]).sum(axis=1)
            h=T(np.concatenate((ah,th),axis=1))
            # Production softmax deliberately computes float32, then restores query dtype.
            np.testing.assert_array_equal(mem[0][:,5:],np.stack(pair,axis=1))
        h += sum((i+1)*.003 for i in range(21,28))
        np.testing.assert_allclose(out,h,rtol=5e-6,atol=5e-6)
        np.testing.assert_allclose(mem[0],ma,rtol=5e-6,atol=5e-6)
        self.assertEqual([p['physical_index'] for p in obj.last_forward_trace], list(C.TEXT_MODEL_SCHEDULE))
        self.assertEqual([v[0] for v in obj.events],[0,7,21,35])
        self.assertEqual(len(obj.last_loop_lora_trace),28)

    def test_seed_slots_unchanged_after_each_write_and_last_read_is_five(self):
        obj,x,pair,scope=self.fixture(); self.forward(obj,x,pair,scope)
        self.assertEqual(len(obj.last_memory_write_history),3)
        for audio,_ in obj.last_memory_write_history:
            np.testing.assert_array_equal(audio[:,5],pair[0]); np.testing.assert_array_equal(audio[:,6],pair[1])
        expected={'write_pre':5,'write_0':5,'write_1':5,'read_pre':5,'read_0':5,'read_1':5,
                  'audio_read_pre':7,'audio_read_0':7}
        self.assertEqual({k:v.shape[-1] for k,v in obj.last_router_weights.items()}, expected)
        np.testing.assert_allclose(obj.last_router_queries['write_0'],obj.last_core_inputs[0])
        np.testing.assert_allclose(obj.last_router_queries['audio_read_0'],obj.last_core_inputs[0][:,:126])

    def test_each_seed_affects_output_and_every_forward_rebuilds_memory(self):
        obj,x,pair,scope=self.fixture(); original,_=self.forward(obj,x,pair,scope)
        for layer in (0,1):
            changed=[v.copy() for v in pair]; changed[layer] += .1
            output,_=self.forward(obj,x,changed,scope)
            self.assertGreater(float(np.max(abs(output-original))), 1e-8)
        again,_=self.forward(obj,x,pair,scope)
        np.testing.assert_array_equal(original,again)


def valid_resume(step=2):
    prepared=dict(rows=1477492,source_rows=C.SOURCE_COUNTS,rows_canonical_sha256='r',mapping_sha256='m')
    recipe=C.semantic_contract(prepared,'text','ced'); g=recipe['geometry']
    config=dict(mode='formal',contract=recipe,contract_sha256=C.digest(recipe))
    roles=['map','default','layer4_mlp','layer8_mlp','audio_read_0','audio_read_1',
           'loop1_lora_A','loop1_lora_B','loop2_lora_A','loop2_lora_B']
    inventory=[]; groups=[]
    for i,role in enumerate(roles):
        lr=3e-4 if role=='default' else 6e-4
        inventory.append(dict(role=role,base_lr=lr,weight_decay=.01,names=[role+'.weight'],numel=1))
        groups.append(dict(params=[i],initial_lr=lr,lr=lr*C.lr_factor(step,g['total_steps']),
                           betas=(.9,.999),eps=1e-8,weight_decay=.01))
    saved=dict(schema_version=1,route_variant=C.ROUTE,mode='formal',contract=recipe,contract_sha256=config['contract_sha256'],
        state_dict={},optimizer=dict(param_groups=groups,state={i:{'step':step} for i in range(len(roles))}),
        optimizer_inventory=inventory,scheduler={'last_epoch':step},grad_scaler={},grad_norm_tracker={},
        random_state_by_rank=[dict(python=1,numpy=2,torch=3,cuda=4) for _ in range(8)],
        total_step=step,epoch=step//g['steps_per_epoch'],next_optimizer_step=step%g['steps_per_epoch'],num_epochs=3,
        loss_reduction='mean_of_rank_local_microbatch_token_means_then_DDP_mean',
        batch_geometry=dict(world_size=8,microbatch_size=4,gradient_accumulation_steps=4,global_batch_size=128),
        audio_gradient_state=dict(checks=min(step,2)+step//25,last_checked_update=max(min(step,2),(step//25)*25)))
    return saved,config,g


class ContractTests(unittest.TestCase):
    def test_parameter_counts_and_explicit_read_scope(self):
        c=C.AUDIO_READ2_CONTRACT
        self.assertEqual(c['audio_read_slots'],[7,7,5]); self.assertEqual(c['write_slots'],[5,5,5])
        self.assertEqual(c['new_bias_source_row'],0)
        self.assertEqual(2*7*577,c['added_router_parameters'])
        self.assertEqual(2443776+8078+C.LOOP_LORA_CONTRACT['new_parameters'],4731278)
        self.assertEqual(c['total_router_parameters'],6*5*577+8078)

    def test_strict_loading_precedes_extensions_and_checks_every_source_tensor(self):
        text=ast.unparse(A.find_function(A.tree(ROUTE/'models/recursive_text.py'),'load_recursive_text_model'))
        self.assertLess(text.index('audit_checkpoint_tensors'),text.index('enable_audio_read2'))
        self.assertLess(text.index('audit_checkpoint_tensors'),text.index('enable_loop_lora'))
        self.assertIn('inherited_tensors_unchanged_after_extensions',text)
        self.assertNotIn('torch.load',text)

    def test_resume_accepts_valid_and_rejects_corrupt_payloads(self):
        saved,config,g=valid_resume(); C.validate_resume_payload(saved,config,g)
        variants=[]
        for key,value in [('route_variant','old'),('mode','smoke'),('total_step',3),('next_optimizer_step',3),
                          ('random_state_by_rank',[]),('audio_gradient_state',{'checks':0,'last_checked_update':0})]:
            item=copy.deepcopy(saved); item[key]=value; variants.append(item)
        item=copy.deepcopy(saved);item['optimizer']['param_groups'][0]['lr']=6e-4;variants.append(item)
        item=copy.deepcopy(saved);item['optimizer']['state'][0]['step']=1;variants.append(item)
        item=copy.deepcopy(saved);item['optimizer_inventory'][0]['role']='layer4_gate';variants.append(item)
        item=copy.deepcopy(saved);item['optimizer']['param_groups'][0]['params']=[1];variants.append(item)
        item=copy.deepcopy(saved);item['batch_geometry']['global_batch_size']=256;variants.append(item)
        for key in ('state_dict','grad_scaler','optimizer_inventory'):
            item=copy.deepcopy(saved);item.pop(key);variants.append(item)
        for i,item in enumerate(variants):
            with self.subTest(case=i), self.assertRaises(ValueError): C.validate_resume_payload(item,config,g)

    def test_completed_run_reload_only_and_gradient_audit_cursor(self):
        saved,config,g=valid_resume(34626)
        with self.assertRaises(ValueError):C.validate_resume_payload(saved,config,g)
        C.validate_resume_payload(saved,config,g,allow_completed=True)
        for step in (0,1,25,26,5771,11542):
            saved,config,g=valid_resume(step)
            if not step:saved['optimizer']['state']={}
            C.validate_resume_payload(saved,config,g)

    def test_optimizer_roles_exclude_new_writes_and_gates(self):
        scope={}; role=function('optim.py','parameter_role',scope)
        for i in (0,1):
            self.assertEqual(role(f'caption_decoder.lm.model.audio_read_routers.{i}.weight'),f'audio_read_{i}')
        self.assertEqual(role('caption_decoder.lm.model.write_routers.0.weight'),'default')
        self.assertEqual(role('caption_decoder.lm.model.read_routers.2.weight'),'default')
        for i in (0,1):
            self.assertEqual(role(f'caption_decoder.lm.model.layers.7.self_attn.q_proj.lora_A.{i}'),f'loop{i+1}_lora_A')
        with self.assertRaises(ValueError):role('caption_decoder.lm.model.layers.6.self_attn.q_proj.lora_A.0')

    def test_actual_full_checkpoint_writer_preserves_all_state_and_eight_rngs(self):
        saved,config,g=valid_resume()
        model=types.SimpleNamespace(state_dict=lambda:{'synthetic_weight':1},
            optimizer_inventory=saved['optimizer_inventory'],caption_decoder=types.SimpleNamespace(
                lm=types.SimpleNamespace(model=types.SimpleNamespace(audio_gradient_state=saved['audio_gradient_state']))))
        state=lambda name:types.SimpleNamespace(state_dict=lambda:saved[name])
        captured={}
        def save(payload,path):
            captured['payload']=copy.deepcopy(payload)
            Path(path).write_bytes(b'explicit CPU checkpoint-writer fixture')
        def gather(output,value):output[:]=[copy.deepcopy(value) for _ in range(8)]
        scope=dict(torch=types.SimpleNamespace(save=save),WORLD_SIZE=8,EPOCHS=3,BATCH_SIZE=4,
            RANK_UPDATE_SAMPLES=16,ACCUMULATION_STEPS=4,ROUTE=C.ROUTE,os=os,
            rng_state=lambda device:dict(python=1,numpy=2,torch=3,cuda=4),
            dist=types.SimpleNamespace(all_gather_object=gather,get_rank=lambda:0,barrier=lambda:None))
        writer=function('train.py','save_checkpoint',scope)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'full.ckpt'
            writer(path,model,state('optimizer'),state('scheduler'),state('grad_scaler'),
                   state('grad_norm_tracker'),config,2,0,2,'explicit-CPU-fixture')
            self.assertTrue(path.is_file());self.assertFalse(path.with_suffix('.tmp').exists())
        C.validate_resume_payload(captured['payload'],config,g)
        self.assertEqual(len(captured['payload']['random_state_by_rank']),8)

    def test_actual_restore_guard_rejects_reordered_named_inventory(self):
        fn=A.find_function(A.tree(ROUTE/'train.py'),'train')
        guard=next(n for n in ast.walk(fn) if isinstance(n,ast.If)
            and ast.unparse(n.test)=="saved['optimizer_inventory'] != groups")
        scope=dict(saved={'optimizer_inventory':[{'names':['A','B']}]},groups=[{'names':['A','B']}])
        A.execute([guard],scope)
        scope['groups']=[{'names':['B','A']}]
        with self.assertRaisesRegex(ValueError,'inventory'):A.execute([guard],scope)


class GradientAuditSourceTests(unittest.TestCase):
    def fixture(self):
        scope=dict(torch=types.SimpleNamespace(isfinite=np.isfinite),TEXT_MODEL_SCHEDULE=C.TEXT_MODEL_SCHEDULE)
        role=function('optim.py','parameter_role',scope)
        roles=('map','default','layer4_mlp','layer8_mlp','audio_read_0','audio_read_1',
               'loop1_lora_A','loop1_lora_B','loop2_lora_A','loop2_lora_B')
        scope.update(parameter_role=role,TRAINABLE_ROLES=roles)
        prefixes=['audio_encoder.base.mapper','caption_decoder.lm.model.layers.0',
                  'audio_encoder.base.layer_mlps.4','audio_encoder.base.layer_mlps.8']
        names=[p+'.weight' for p in prefixes]
        names += [f'caption_decoder.lm.model.{kind}_routers.{i}.{part}'
                  for kind in ('write','read') for i in range(3) for part in ('weight','bias')]
        names += [f'caption_decoder.lm.model.audio_read_routers.{i}.{part}' for i in (0,1) for part in ('weight','bias')]
        names += [f'caption_decoder.lm.model.layers.7.self_attn.q_proj.lora_{kind}.{i}' for kind in ('A','B') for i in (0,1)]
        params={}
        for n in names:
            p=T(np.ones((2,2)));p.requires_grad=True
            if '.lora_B.' in n:p.fill(0)
            p.grad=T(np.zeros((2,2)) if '.lora_A.' in n else np.ones((2,2)))
            params[n]=p
        B=[params[f'caption_decoder.lm.model.layers.7.self_attn.q_proj.lora_B.{i}'] for i in (0,1)]
        core=types.SimpleNamespace(modules=lambda:[types.SimpleNamespace(lora_B=B)],
            last_loop_lora_trace=[(i,p,0 if i<21 else 1) for i,p in enumerate(C.TEXT_MODEL_SCHEDULE) if 7<=p<21])
        lm=types.SimpleNamespace(model=core,named_parameters=lambda:[(n.removeprefix('caption_decoder.lm.'),p)
                                for n,p in params.items() if n.startswith('caption_decoder.lm.')])
        model=types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=lm),named_parameters=lambda:params.items())
        audit=function('train.py','first_gradient_audit',scope)
        routers=function('train.py','router_gradient_audit',scope)
        return model,params,audit,routers

    def test_zero_A_expected_and_zero_real_sample_roles_diagnostic(self):
        model,params,audit,routers=self.fixture()
        self.assertTrue(audit(model)['passed']);self.assertTrue(routers(model)['passed'])
        self.assertEqual(len(routers(model)['tensors']),16)
        for n,p in params.items():
            if '.lora_A.' not in n:p.grad.fill(0)
        self.assertTrue(audit(model)['passed']);self.assertTrue(routers(model)['passed'])

    def test_missing_nonfinite_wrong_scope_and_nonzero_A_while_B_closed_rejected(self):
        for case in ('missing','nonfinite','scope','closed_A'):
            model,params,audit,_=self.fixture()
            if case=='missing':params['audio_encoder.base.layer_mlps.4.weight'].grad=None
            if case=='nonfinite':params['caption_decoder.lm.model.audio_read_routers.0.bias'].grad.fill(np.nan)
            if case=='scope':model.caption_decoder.lm.model.last_loop_lora_trace=[]
            if case=='closed_A':params['caption_decoder.lm.model.layers.7.self_attn.q_proj.lora_A.0'].grad.fill(1)
            self.assertFalse(audit(model)['passed'],case)


HAS_TORCH=all(importlib.util.find_spec(n) for n in ('torch','transformers','safetensors'))


@unittest.skipUnless(HAS_TORCH,'requires real torch/transformers/safetensors; no downloads')
class RealTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LlamaConfig
        from models.recursive_audio7_read2 import MizarText7_14x2_7MeshForCausalLM
        from mizar_text_7_14x2_7_mesh.contract import build_target_config
        cls.torch=torch; torch.set_num_threads(1)
        cls.model_class=MizarText7_14x2_7MeshForCausalLM
        cls.cfg=build_target_config(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=24,
            num_hidden_layers=30,num_attention_heads=4,num_key_value_heads=2,
            max_position_embeddings=1024,tie_word_embeddings=True,eos_token_id=0,pad_token_id=0,
            attention_dropout=0.,_attn_implementation='eager'))

    def model(self):
        model=self.model_class(copy.deepcopy(self.cfg)).float()
        model.model.enable_audio_read2(); model.model.enable_loop_lora()
        return model

    def inputs(self,length=131):
        t=self.torch
        return t.randn(1,length,16),tuple(t.randn(1,126,16,requires_grad=True) for _ in range(2))

    def call(self,m,x,pair):return m(inputs_embeds=x,audio_information=pair,use_cache=False).logits

    def test_rng_inherited_parameter_identity_and_initialization(self):
        t=self.torch;m=self.model_class(copy.deepcopy(self.cfg)).float()
        before={n:(p,p.detach().clone()) for n,p in m.named_parameters()}
        state=t.get_rng_state().clone();m.model.enable_audio_read2();m.model.enable_loop_lora()
        self.assertTrue(t.equal(state,t.get_rng_state()))
        for n,(original,value) in before.items():
            self.assertIs(dict(m.named_parameters())[n],original);t.testing.assert_close(original,value,rtol=0,atol=0)
        for old,new in zip(m.model.read_routers,m.model.audio_read_routers):
            t.testing.assert_close(new.weight[:5],old.weight,rtol=0,atol=0)
            t.testing.assert_close(new.bias[:5],old.bias-math.log(2),rtol=0,atol=0)
            for i in (5,6):
                t.testing.assert_close(new.weight[i],old.weight[0],rtol=0,atol=0)
                t.testing.assert_close(new.bias[i],old.bias[0],rtol=0,atol=0)

    def test_real_forward_memory_scope_and_two_seed_gradients(self):
        t=self.torch;m=self.model();x,p=self.inputs(683);m.model.audit_mode=True
        out=self.call(m,x,p);out[:,-4:].square().mean().backward()
        self.assertEqual(m.model.last_memory_shape,((1,7,126,16),(1,5,557,16)))
        for a,_ in m.model.last_memory_write_history:
            t.testing.assert_close(a[:,5],p[0].detach());t.testing.assert_close(a[:,6],p[1].detach())
        for seed in p:
            self.assertTrue(bool(t.isfinite(seed.grad).all()));self.assertGreater(seed.grad.abs().sum().item(),0)
        self.assertEqual(len(m.model.last_forward_trace),42)
        self.assertEqual(len(m.model.last_loop_lora_trace),28)
        for n,param in m.named_parameters():
            if '.lora_A.' in n:self.assertEqual(param.grad.abs().sum().item(),0)
            if '.lora_B.' in n:self.assertGreater(param.grad.abs().sum().item(),0)

    def test_zero_B_matches_same_audio7_model_with_adapter_delta_removed(self):
        t=self.torch;m=self.model().eval();x,p=self.inputs()
        expected=self.call(m,x,p)
        modules=[v for v in m.modules() if hasattr(v,'lora_B')]
        for v in modules:v.scaling=0.
        t.testing.assert_close(self.call(m,x,p),expected,rtol=0,atol=0)

    def test_both_independent_loops_receive_A_gradient_after_opening_B(self):
        t=self.torch;m=self.model();x,p=self.inputs()
        with t.no_grad():
            for v in m.modules():
                if hasattr(v,'lora_B'):
                    for b in v.lora_B:b.normal_(std=.01)
        self.call(m,x,p)[:,-3:].square().mean().backward()
        from models.loop_lora import validate_loop_lora
        self.assertEqual(validate_loop_lora(m.model)['matrix_count'],392)
        for n,v in m.named_parameters():
            if '.lora_A.' in n or '.lora_B.' in n:
                self.assertIsNotNone(v.grad,n);self.assertGreater(v.grad.abs().sum().item(),0,n)

    def test_wrong_features_cache_checkpointing_and_reinitialization_rejected(self):
        m=self.model();x,p=self.inputs()
        for bad in (None,(p[0],),(p[0][:,:125],p[1])):
            with self.assertRaises(ValueError):self.call(m,x,bad)
        with self.assertRaises(ValueError):m(inputs_embeds=x,audio_information=p,use_cache=True)
        m.model.gradient_checkpointing=True
        with self.assertRaises(ValueError):self.call(m,x,p)
        with self.assertRaises(RuntimeError):m.model.enable_audio_read2()

    def test_strict_model_optimizer_scheduler_rng_next_update_resume(self):
        t=self.torch;m=self.model();x,p=self.inputs()
        opt=t.optim.AdamW(m.parameters(),lr=6e-4);sched=t.optim.lr_scheduler.LambdaLR(opt,lambda s:C.lr_factor(s,34626))
        def update(model,optimizer,scheduler):
            optimizer.zero_grad(set_to_none=True)
            noise=t.rand_like(x)*.01
            self.call(model,x+noise,p)[:,-3:].square().mean().backward()
            optimizer.step();scheduler.step()
        update(m,opt,sched)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'full.pt'
            t.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=t.get_rng_state()),path)
            update(m,opt,sched);expected={n:v.clone() for n,v in m.state_dict().items()}
            saved=t.load(path,weights_only=False);resumed=self.model()
            resumed.load_state_dict(saved['model'],strict=True)
            opt2=t.optim.AdamW(resumed.parameters(),lr=6e-4);sched2=t.optim.lr_scheduler.LambdaLR(opt2,lambda s:C.lr_factor(s,34626))
            opt2.load_state_dict(saved['optimizer']);sched2.load_state_dict(saved['scheduler']);t.set_rng_state(saved['rng'])
            update(resumed,opt2,sched2)
            for n,v in resumed.state_dict().items():t.testing.assert_close(v,expected[n],rtol=0,atol=0)
            self.assertEqual(opt.param_groups[0]['lr'],opt2.param_groups[0]['lr'])


if __name__=='__main__':unittest.main()
