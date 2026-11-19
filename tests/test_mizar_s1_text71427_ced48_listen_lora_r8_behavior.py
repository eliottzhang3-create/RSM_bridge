"""Production listen/LoRA source executed with NumPy; not autograd/GPU evidence."""
import ast
import copy
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, SimpleNamespace as NS
import types
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced48_listen_lora_r8"
sys.path.insert(0, str(ROUTE))
import contract
import audit_reference as audit


class ListenMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np=np
        class Tensor(np.ndarray):
            @property
            def device(self): return 'cpu'
            def unsqueeze(self,dim): return np.expand_dims(self,dim)
            def transpose(self,*axes):
                return np.swapaxes(self,*axes) if len(axes)==2 and self.ndim!=2 else super().transpose(*axes)
        cls.Tensor=Tensor
        scope=dict(torch=SimpleNamespace(tanh=np.tanh,cat=lambda xs,dim:np.concatenate(xs,axis=dim)))
        module=audit.tree(ROUTE/'models/recursive_mesh_listen_lora.py')
        audit.execute([audit.find_function(module,'listen_audio')],scope)
        cls.listen=staticmethod(scope['listen_audio'])
        forward=next(n for n in next(n for n in module.body if isinstance(n,ast.ClassDef)
            and n.name=='MeshLlamaModel').body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        begin=next(i for i,n in enumerate(forward.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='memory')
        end=next(i for i,n in enumerate(forward.body) if isinstance(n,ast.Assign) and ast.unparse(n.value)=='self.norm(hidden)')
        cls.body=forward.body[begin:end+1]
        scope={}
        audit.execute([audit.find_function(audit.tree(ROUTE/'models/ced.py'),'_hidden_to_temporal_grid')],scope)
        cls.grid=staticmethod(scope['_hidden_to_temporal_grid'])

    def fixture(self,closed=False):
        np=self.np
        class Linear:
            def __init__(self,out,value):
                self.weight=np.full((out,3),value)*np.arange(1,out+1)[:,None]
                self.bias=np.zeros(out)
                self.inputs=[]
            def __call__(self,x):
                self.inputs.append(x.copy())
                return x@self.weight.T+self.bias
        obj=SimpleNamespace(listen_audio_tokens=126,collect_listen_stats=False,last_listen_trace=(),
            audit_mode=False,gradient_audit_mode=False,write_routers=list(range(3)),read_routers=list(range(3)),
            audio_gates={str(l):Linear(1,0. if closed else (.03 if l==4 else -.02)) for l in (4,8)},
            listen_routers={str(l):Linear(5,.01*l) for l in (4,8)},norm=lambda x:x,events=[],queries={},last_forward_trace=[])
        obj.listen_audio=types.MethodType(self.listen,obj)
        def stack(h,indices,start,**kw):
            obj.events.append(('stack',start,h.copy()))
            for physical in indices:
                obj.last_forward_trace.append(dict(physical_index=physical))
                h=h+.01*(physical+1)
            return h
        def route(router,q,name):
            obj.queries[name]=q.copy()
            scores=router(q) if name.startswith('listen') else q.mean(axis=-1,keepdims=True)*np.linspace(-.1,.2,5)+router*.02*np.arange(5)
            p=np.exp(scores-scores.max(axis=-1,keepdims=True))
            return p/p.sum(axis=-1,keepdims=True)
        def write(m,v,p):
            out=m+v[:,None]*p.transpose(0,2,1)[...,None]
            obj.events.append(('write',len([e for e in obj.events if e[0]=='write']),out.copy()))
            return out
        def read(m,p):
            obj.events.append(('read',len([e for e in obj.events if e[0]=='read']),m.copy()))
            return (m*p.transpose(0,2,1)[...,None]).sum(axis=1)
        obj._run_stack,obj._route,obj._write,obj._read=stack,route,write,read
        x=np.random.default_rng(42).normal(size=(2,131,3)).view(self.Tensor)
        info=(np.ones((2,126,3)).view(self.Tensor),np.full((2,126,3),2.).view(self.Tensor))
        return obj,x,info

    def run_forward(self,obj,x,info):
        np=self.np
        scope=dict(self=obj,hidden=x.copy(),audio_information=info,batch_size=x.shape[0],query_length=x.shape[1],
            MEMORY_SLOT_COUNT=5,torch=SimpleNamespace(zeros=lambda shape,**kw:np.zeros(shape).view(self.Tensor)),
            record_function=lambda name:nullcontext(),output_hidden_states=False,mask=None,position_ids=None,
            cache=None,use_cache=False,cache_position=None,position_embeddings=None,output_attentions=False,attentions=[])
        audit.execute(copy.deepcopy(self.body),scope)
        return scope['hidden'],scope['memory']

    def test_two_real_listen_writes_between_original_write_and_read_with_output_queries(self):
        np=self.np
        obj,x,info=self.fixture()
        result,memory=self.run_forward(obj,x,info)
        self.assertEqual([(e[0],e[1]) for e in obj.events], [('stack',0),('write',0),('read',0),
            ('stack',7),('write',1),('read',1),('stack',21),('write',2),('read',2),('stack',35)])
        stacks=[e[2] for e in obj.events if e[0]=='stack']
        writes=[e[2] for e in obj.events if e[0]=='write']
        reads=[e[2] for e in obj.events if e[0]=='read']
        for i,layer in enumerate((4,8)):
            # Recover module OUTPUT from the known synthetic stacks (not the memory read).
            output=stacks[i].copy()
            for physical in (range(7) if i==0 else range(7,21)): output=output+.01*(physical+1)
            q=output[:,:126]
            np.testing.assert_array_equal(obj.audio_gates[str(layer)].inputs[0],q)
            np.testing.assert_array_equal(obj.listen_routers[str(layer)].inputs[0],q)
            g=np.tanh(q@obj.audio_gates[str(layer)].weight.T)
            raw=q@obj.listen_routers[str(layer)].weight.T
            p=np.exp(raw-raw.max(axis=-1,keepdims=True));p=p/p.sum(axis=-1,keepdims=True)
            expected=g[:,None]*info[i][:,None]*p.transpose(0,2,1)[...,None]
            np.testing.assert_allclose(reads[i][:,:,:126]-writes[i][:,:,:126],expected,atol=1e-15)
            np.testing.assert_array_equal(reads[i][:,:,126:],writes[i][:,:,126:])
        np.testing.assert_array_equal(reads[2],writes[2])
        # Both original loop routers continue to query the corresponding loop INPUT.
        for i in (0,1):
            np.testing.assert_array_equal(obj.queries[f'write_{i}'],stacks[i+1])
            np.testing.assert_array_equal(obj.queries[f'read_{i}'],stacks[i+1])
        self.assertEqual(obj.last_listen_trace,((7,4),(21,8)))
        self.assertEqual([e['physical_index'] for e in obj.last_forward_trace],list(contract.TEXT_MODEL_SCHEDULE))
        self.assertEqual(memory.shape,(2,5,131,3))
        self.assertEqual(result.shape,x.shape)

    def test_zero_gates_equal_original_memory_and_hidden_exactly(self):
        obj,x,info=self.fixture(closed=True)
        actual,memory=self.run_forward(obj,x,info)
        base,_,_=self.fixture()
        base.listen_audio=lambda m,*args:m
        expected,old_memory=self.run_forward(base,x,info)
        self.np.testing.assert_array_equal(actual,expected)
        self.np.testing.assert_array_equal(memory,old_memory)

    def test_listen_is_out_of_place_and_preserves_sep_text_memory(self):
        obj,x,info=self.fixture()
        np=self.np
        m=np.random.default_rng(7).normal(size=(2,5,131,3)).view(self.Tensor)
        original=m.copy()
        out=obj.listen_audio(m,x,info[0],4,5)
        np.testing.assert_array_equal(m,original)
        np.testing.assert_array_equal(out[:,:,126:],original[:,:,126:])
        self.assertFalse(np.array_equal(out[:,:,:126],original[:,:,:126]))

    def test_zero_start_blocks_mlp_and_router_derivatives_then_gate_opens_both(self):
        obj,x,info=self.fixture(closed=True)
        np=self.np
        memory=np.zeros((2,5,131,3)).view(self.Tensor)
        target=np.random.default_rng(2).normal(size=memory.shape)
        def loss():return float((obj.listen_audio(memory,x,info[0],4,5)*target).sum())
        def derivative(array,index):
            old,eps=array[index],1e-5
            array[index]=old+eps;plus=loss()
            array[index]=old-eps;minus=loss()
            array[index]=old
            return (plus-minus)/(2*eps)
        self.assertEqual(derivative(info[0],(0,0,0)),0.)
        self.assertEqual(derivative(obj.listen_routers['4'].weight,(0,0)),0.)
        self.assertGreater(abs(derivative(obj.audio_gates['4'].weight,(0,0))),1e-5)
        obj.audio_gates['4'].bias[0]=.1
        self.assertGreater(abs(derivative(info[0],(0,0,0))),1e-5)
        self.assertGreater(abs(derivative(obj.listen_routers['4'].weight,(0,0))),1e-5)

    def test_batch_segment_frequency_time_mapping(self):
        np=self.np
        class GridTensor(np.ndarray):
            def permute(self,*axes):return super().transpose(*axes)
        h=np.arange(2*504*384).reshape(2,504,384).view(GridTensor)
        actual=self.grid(None,h)
        original=h.reshape(2,2,4,63,384)
        for b in range(2):
            for s in range(2):
                for t in (0,1,31,62):
                    for f in range(4):np.testing.assert_array_equal(actual[b,s*63+t,f],original[b,s,f,t])

def source_functions(file, names, scope):
    module = audit.tree(ROUTE / file)
    audit.execute([audit.find_function(module, name) for name in names], scope)
    return scope

class LoRAMathTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        module = audit.tree(ROUTE / 'models/loop_lora.py')
        linear = next((n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'LoopLoRALinear'))
        forward = next((copy.deepcopy(n) for n in linear.body if isinstance(n, ast.FunctionDef) and n.name == 'forward'))
        scope = dict(F=NS(linear=lambda x, w, bias=None: x @ w.T + (0 if bias is None else bias)))
        audit.execute([forward], scope)
        cls.projection_forward = staticmethod(scope['forward'])
        scope = source_functions('models/loop_lora.py', ['projections', 'call_with_loop_lora'], dict(TARGETS=contract.LOOP_LORA_CONTRACT['targets']))
        cls.call_scope = staticmethod(scope['call_with_loop_lora'])

    def projection(self, seed=0, closed=False):
        rng = np.random.default_rng(seed)
        return NS(weight=np.eye(3) * 0.99, bias=np.zeros(3), scaling=2.0, loop_index=None, lora_A=[rng.normal(size=(8, 3)) * 0.1 for _ in range(2)], lora_B=[rng.normal(size=(3, 8)) * (0.0 if closed else 0.04) for _ in range(2)])

    def test_nonzero_adapters_scale_two_and_select_distinct_rounds(self):
        p, x = (self.projection(), np.ones((2, 131, 3)))
        outputs = []
        for loop in (0, 1):
            p.loop_index = loop
            y = self.projection_forward(p, x)
            np.testing.assert_allclose(y, x @ p.weight.T + 2 * (x @ p.lora_A[loop].T @ p.lora_B[loop].T))
            outputs.append(y)
        self.assertFalse(np.allclose(*outputs))
        self.assertFalse(np.allclose(outputs[0][:, 126:], x[:, 126:] @ p.weight.T))
        p.loop_index = None
        with self.assertRaises(RuntimeError):
            self.projection_forward(p, x)

    def test_initial_B_zero_A_zero_task_gradient_and_independent_branch_gradient(self):
        p, x = (self.projection(closed=True), np.random.default_rng(5).normal(size=(2, 3)))
        p.loop_index = 0

        def loss():
            return (self.projection_forward(p, x) ** 2).sum()

        def diff(matrix, index):
            old, eps = (matrix[index], 1e-05)
            matrix[index] = old + eps
            plus = loss()
            matrix[index] = old - eps
            minus = loss()
            matrix[index] = old
            return (plus - minus) / (2 * eps)
        self.assertEqual(diff(p.lora_A[0], (0, 0)), 0.0)
        self.assertNotEqual(diff(p.lora_B[0], (0, 0)), 0.0)
        self.assertEqual(diff(p.lora_B[1], (0, 0)), 0.0)
        self.assertNotEqual(diff(p.weight, (0, 0)), 0.0)
        p.lora_B[0][0, 0] = 0.1
        self.assertNotEqual(diff(p.lora_A[0], (0, 0)), 0.0)

    def test_scope_resets_selection_on_success_exception_and_backward_independent_tensors(self):
        modules = {name: self.projection(i) for i, name in enumerate(contract.LOOP_LORA_CONTRACT['targets'])}
        layer = NS(get_submodule=lambda name: modules[name])
        selected = []

        def callback(layer, x):
            selected.append([(p.lora_A[p.loop_index], p.lora_B[p.loop_index]) for p in modules.values()])
            return x + 1
        for loop in (0, 1):
            self.assertEqual(self.call_scope(layer, loop, callback, 2), 3)
            self.assertTrue(all((p.loop_index is None for p in modules.values())))
        self.assertIsNot(selected[0][0][0], selected[1][0][0])

        def failure(*a):
            raise ValueError('synthetic decoder failure')
        with self.assertRaises(ValueError):
            self.call_scope(layer, 0, failure)
        self.assertTrue(all((p.loop_index is None for p in modules.values())))

class ResumeTests(unittest.TestCase):
    def payload(self, step=1):
        recipe = {'synthetic': True}
        geometry = contract.geometry(1477492)
        role_names = {f'layer{l}_{k}': True for l in (4,8) for k in ('mlp','gate','listen')}
        role_names.update({f'loop{loop}_lora_{kind}':True for loop in (1,2) for kind in ('A','B')})
        state = {'complete': False, 'seen': {k: k.endswith('gate') for k in role_names}, 'deadline': 32}
        if step>=2: state = dict(complete=True, seen=role_names, deadline=32, opened_at_update=2)
        payload = dict(schema_version=1, route_variant=contract.ROUTE, mode='formal', contract=recipe,
            contract_sha256=contract.digest(recipe), state_dict={}, grad_scaler={}, grad_norm_tracker={},
            random_state_by_rank=[dict(python=1,numpy=2,torch=3,cuda=4) for _ in range(8)],
            total_step=step, epoch=step//geometry['steps_per_epoch'], next_optimizer_step=step%geometry['steps_per_epoch'],
            scheduler={'last_epoch':step}, num_epochs=3,
            loss_reduction='mean_of_rank_local_microbatch_token_means_then_DDP_mean',
            batch_geometry=dict(world_size=8,microbatch_size=4,gradient_accumulation_steps=4,global_batch_size=128),
            listen_gradient_state=state,
            optimizer={'state': {0:{'step':float(step)}}, 'param_groups':[dict(params=[0],betas=(.9,.999),eps=1e-8,
                weight_decay=.01,initial_lr=6e-4,lr=6e-4*contract.lr_factor(step,geometry['total_steps']))]})
        return payload, dict(contract=recipe, contract_sha256=contract.digest(recipe)), geometry

    def test_complete_states_before_after_unlock_and_final_reload(self):
        for step in (1,2,5771,34626):
            data,cfg,geom=self.payload(step)
            contract.validate_resume_payload(data,cfg,geom,allow_completed=True)
            if step==34626:
                with self.assertRaisesRegex(ValueError,'already-completed'): contract.validate_resume_payload(data,cfg,geom)
            else: contract.validate_resume_payload(data,cfg,geom)

    def test_reject_wrong_route_optimizer_rng_cursor_and_unlock_state(self):
        changes = [lambda d:d.update(route_variant='old_route'), lambda d:d.update(total_step=3),
            lambda d:d['optimizer']['state'][0].update(step=2), lambda d:d['optimizer']['state'].clear(),
            lambda d:d['optimizer']['param_groups'][0].update(lr=1.), lambda d:d['random_state_by_rank'].pop(),
            lambda d:d['listen_gradient_state'].update(complete=True), lambda d:d['listen_gradient_state'].update(deadline=0),
            lambda d:d['listen_gradient_state']['seen'].update(unknown=True), lambda d:d.pop('listen_gradient_state'),
            lambda d:d['contract'].update(synthetic=False)]
        for mutate in changes:
            d,c,g=self.payload()
            mutate(d)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError): contract.validate_resume_payload(d,c,g)

class ArchitectureTests(unittest.TestCase):
    def test_exact_lora_gqa_count_and_listen_counts(self):
        c=contract.LOOP_LORA_CONTRACT
        self.assertEqual(c['physical_layers_0based'], list(range(7,21)))
        self.assertEqual(c['logical_ranges_0based_inclusive'], [[7,20],[21,34]])
        self.assertEqual((c['rank'],c['alpha'],c['scale'],c['dropout']), (8,16,2.,0.))
        count=14*2*8*(2*(576+576)+2*(576+192)+3*(576+1536))
        self.assertEqual(count, 2279424)
        self.assertEqual(c['new_parameters'],count)
        self.assertEqual(14*7*2*2,392)
        self.assertEqual(contract.LISTEN_CONTRACT['new_parameter_count'],2450700)
        self.assertEqual(count+2450700,4730124)
        self.assertEqual(contract.LISTEN_CONTRACT['logical_indices_0based'],[7,21])

    def test_all_392_optimizer_matrices_and_twelve_roles(self):
        class Parameter:
            requires_grad=True
            def __init__(self,ndim): self.ndim=ndim
            def numel(self): return 12
        prefixes=['audio_encoder.base.mapper','caption_decoder.lm.model.layers.0']
        for layer in (4,8):
            prefixes += [f'audio_encoder.base.layer_mlps.{layer}',f'caption_decoder.lm.model.audio_gates.{layer}',
                         f'caption_decoder.lm.model.listen_routers.{layer}']
        pairs=[(p+suffix,Parameter(ndim)) for p in prefixes for suffix,ndim in (('.weight',2),('.bias',1))]
        names=[f'caption_decoder.lm.model.layers.{i}.{target}.lora_{kind}.{loop}'
            for i in range(7,21) for target in contract.LOOP_LORA_CONTRACT['targets'] for kind in ('A','B') for loop in (0,1)]
        pairs += [(n,Parameter(2)) for n in names]
        class Optimizer:
            def __init__(self,groups,**kw): self.param_groups,self.kw=groups,kw
        scope=dict(torch=NS(optim=NS(AdamW=Optimizer,lr_scheduler=NS(LambdaLR=lambda o,f:f))),lr_factor=contract.lr_factor)
        module=audit.tree(ROUTE/'optim.py')
        for n in module.body:
            if isinstance(n,ast.Assign): audit.execute([n],scope)
        source_functions('optim.py',['parameter_role','build_optimizer'],scope)
        opt,schedule,groups=scope['build_optimizer'](NS(named_parameters=lambda:pairs),34626)
        self.assertEqual(len({g['role'] for g in groups}),12)
        self.assertEqual(sum(len(g['names']) for g in groups if 'lora' in g['role']),392)
        for g in groups:
            self.assertEqual(g['base_lr'],3e-4 if g['role']=='default' else 6e-4)
            for n in g['names']:
                self.assertEqual(g['weight_decay'],0. if n.endswith('.bias') else .01)
        for step in range(34627): self.assertEqual(schedule(step),contract.lr_factor(step,34626))
        for wrong in (names[0].replace('layers.7.','layers.6.'), names[0].replace('layers.7.','layers.21.'),
                      names[0].replace('lora_A.0','lora_A.2'),names[0].replace('q_proj','bad_proj')):
            with self.assertRaises(ValueError): scope['parameter_role'](wrong)

    def test_initializers_match_reference_and_never_reinitialize_originals(self):
        parent=ROUTE.with_name('mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_listen')
        new=copy.deepcopy(audit.find_function(audit.tree(ROUTE/'models/recursive_mesh_listen_lora.py'),'_init_listen_router'))
        new.name='_init_router'
        self.assertEqual(ast.dump(new),ast.dump(audit.find_function(audit.tree(parent/'models/recursive_mesh_listen.py'),'_init_router')))
        gate=audit.find_function(audit.tree(ROUTE/'models/recursive_mesh_listen_lora.py'),'enable_audio_listen')
        calls=[ast.unparse(n) for n in ast.walk(gate) if isinstance(n,ast.Call)]
        self.assertIn('_init_listen_router(router)',calls)
        self.assertIn('nn.init.zeros_(gate.weight)',calls)
        self.assertFalse(any('_init_router(' in n for n in calls))
        loader=ast.unparse(audit.find_function(audit.tree(ROUTE/'models/recursive_text.py'),'load_recursive_text_model'))
        self.assertLess(loader.index('audit_checkpoint_tensors'),loader.index('enable_audio_listen'))
        self.assertLess(loader.index('enable_audio_listen'),loader.index('lora = install_loop_lora'))
        self.assertIn('inherited_parameter_objects_preserved=True',loader)

    def test_loop_stack_selects_only_middle_and_rejects_mismatched_scope(self):
        from typing import Sequence, Any
        fn=copy.deepcopy(audit.find_function(audit.tree(ROUTE/'models/recursive_mesh_listen_lora.py'),'_run_stack'))
        for arg in fn.args.args+fn.args.kwonlyargs: arg.annotation=None
        fn.returns=None
        calls=[]
        fake=types.ModuleType('models.loop_lora')
        def lora(layer,loop,callback,*args,**kw):
            calls.append((layer,loop));return callback(layer,*args,**kw)
        fake.call_with_loop_lora=lora
        obj=NS(layers=list(range(28)),last_forward_trace=[],last_loop_lora_trace=())
        scope=dict(_call_layer=lambda layer,hidden,**kw:(hidden+layer+1,),LogicalSlotCacheView=None)
        audit.execute([fn],scope)
        kw=dict(attention_mask=None,position_ids=None,cache=None,use_cache=False,cache_position=None,
                position_embeddings=None,output_attentions=False,all_attentions=[])
        from unittest.mock import patch
        with patch.dict(sys.modules,{'models.loop_lora':fake}):
            scope['_run_stack'](obj,0,range(7),0,**kw)
            self.assertEqual(calls,[])
            for loop,start in ((0,7),(1,21)):
                scope['_run_stack'](obj,0,range(7,21),start,loop_index=loop,**kw)
            self.assertEqual(calls,[(i,k) for k in (0,1) for i in range(7,21)])
            for indices,start,loop in ((range(7,21),7,1),(range(7,21),21,None),(range(7),0,0)):
                with self.assertRaises(ValueError): scope['_run_stack'](obj,0,indices,start,loop_index=loop,**kw)


class GradientAuditTests(unittest.TestCase):

    def fixture(self, opened=False):

        class Tensor(np.ndarray):

            def detach(self):
                return self

            def ne(self, value):
                return self != value

            def numel(self):
                return self.size

            def cpu(self):
                return self

            def clone(self):
                return self.copy()

            def abs(self):
                return abs(self)
        prefixes = {'map': 'audio_encoder.base.mapper', 'default': 'caption_decoder.lm.model.layers.0', 'layer4_mlp': 'audio_encoder.base.layer_mlps.4', 'layer8_mlp': 'audio_encoder.base.layer_mlps.8', 'layer4_gate': 'caption_decoder.lm.model.audio_gates.4', 'layer8_gate': 'caption_decoder.lm.model.audio_gates.8'}
        for layer in (4,8):
            prefixes[f'layer{layer}_listen']=f'caption_decoder.lm.model.listen_routers.{layer}'
        for loop in (1, 2):
            for kind in ('A', 'B'):
                prefixes[f'loop{loop}_lora_{kind}'] = f'caption_decoder.lm.model.layers.7.self_attn.q_proj.lora_{kind}.{loop - 1}'
        roles, pairs = ({}, [])
        for role, prefix in prefixes.items():
            closed = role.endswith('gate') or role.endswith('lora_B')
            value = np.full((1, 2), 0.0 if closed and (not opened) else 1.0).view(Tensor)
            value.requires_grad = True
            zero_grad = role in {'layer4_mlp', 'layer8_mlp', 'layer4_listen', 'layer8_listen', 'loop1_lora_A', 'loop2_lora_A'} and (not opened)
            value.grad = np.full((1, 2), 0.0 if zero_grad else 1.0).view(Tensor)
            roles[role] = value
            pairs.append((prefix if 'lora_' in role else prefix + '.weight', value))
        gate = lambda role: NS(parameters=lambda: [roles[role]])
        lora = NS(lora_B=[roles['loop1_lora_B'], roles['loop2_lora_B']])
        core = NS(audio_gates={str(k): gate(f'layer{k}_gate') for k in (4, 8)}, modules=lambda: [lora], last_listen_trace=((7,4),(21,8)), last_loop_lora_trace=[(i, p, 0 if i < 21 else 1) for i, p in enumerate(contract.TEXT_MODEL_SCHEDULE) if 7 <= p < 21])
        core.last_forward_trace = [dict(physical_index=p) for p in contract.TEXT_MODEL_SCHEDULE]
        core.write_routers, core.read_routers = [], []
        for kind in ("write", "read"):
            for index in range(3):
                value = np.ones((1,2)).view(Tensor)
                value.requires_grad = True
                value.grad = np.ones((1,2)).view(Tensor)
                getattr(core, kind+"_routers").append(NS(parameters=lambda value=value:[value]))
                pairs.append((f"caption_decoder.lm.model.{kind}_routers.{index}.weight", value))
        model = NS(caption_decoder=NS(lm=NS(model=core)), named_parameters=lambda: pairs)
        scope = dict(torch=NS(isfinite=np.isfinite, equal=np.array_equal), TEXT_MODEL_SCHEDULE=contract.TEXT_MODEL_SCHEDULE)
        source_functions('optim.py', ['parameter_role'], scope)
        source = audit.tree(ROUTE / 'train.py')
        for n in source.body:
            if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id in {'TRAINABLE_ROLES', 'NEW_ROLES'} for t in n.targets)):
                audit.execute([n], scope)
        source_functions('train.py', ['first_gradient_audit', 'role_snapshots', 'weights_changed'], scope)
        for n in audit.tree(ROUTE/'optim.py').body:
            if isinstance(n,ast.Assign): audit.execute([n],scope)
        return (model, roles, scope)

    def test_closed_B_expects_zero_A_but_requires_nonzero_B_in_both_loops(self):
        model, roles, scope = self.fixture()
        result = scope['first_gradient_audit'](model)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['closed_lora_B'], {'1': True, '2': True})

        for role in ('loop1_lora_B', 'loop2_lora_B'):
            roles[role].grad.fill(0)
            self.assertFalse(scope['first_gradient_audit'](model)['passed'])
            roles[role].grad.fill(1)

    def test_opened_B_requires_task_gradient_in_A_and_each_audio_upstream(self):
        model, roles, scope = self.fixture(opened=True)
        self.assertTrue(scope['first_gradient_audit'](model)['passed'])
        for role in ('loop1_lora_A', 'loop2_lora_A', 'layer4_mlp', 'layer8_mlp'):
            roles[role].grad.fill(0)
            self.assertFalse(scope['first_gradient_audit'](model)['passed'], role)
            roles[role].grad.fill(1)

    def test_missing_nonfinite_gradients_and_wrong_loop_trace_fail(self):
        model, roles, scope = self.fixture(opened=True)
        for bad in (None, np.full((1, 2), np.nan)):
            old = roles['loop2_lora_A'].grad
            roles['loop2_lora_A'].grad = bad.view(type(old)) if isinstance(bad, np.ndarray) else bad
            self.assertFalse(scope['first_gradient_audit'](model)['passed'])
            roles['loop2_lora_A'].grad = old
        model.caption_decoder.lm.model.last_loop_lora_trace = []
        self.assertFalse(scope['first_gradient_audit'](model)['passed'])

    def test_all_twelve_roles_have_separate_snapshot_and_change_detection(self):
        model, roles, scope = self.fixture()
        snapshots = scope['role_snapshots'](model)
        self.assertEqual(set(snapshots), set(roles))
        self.assertTrue(all((not item['changed'] for item in scope['weights_changed'](model, snapshots).values())))
        roles['loop2_lora_B'] += 0.5
        changed = scope['weights_changed'](model, snapshots)
        self.assertTrue(changed['loop2_lora_B']['changed'])
        self.assertFalse(changed['loop1_lora_B']['changed'])

    def test_six_inherited_routers_keep_gradient_presence_and_finiteness_checks(self):
        model, roles, scope = self.fixture(opened=True)
        core = model.caption_decoder.lm.model
        audit_result = scope['first_gradient_audit'](model)
        self.assertTrue(audit_result['passed'])
        self.assertEqual(len(audit_result['mesh_router_gradients']), 6)
        for kind in ('write', 'read'):
            for router in getattr(core, kind+'_routers'):
                param = list(router.parameters())[0]
                old = param.grad
                for bad in (None, np.full_like(old, np.nan)):
                    param.grad = bad
                    self.assertFalse(scope['first_gradient_audit'](model)['passed'])
                param.grad = old
        # Preserve the parent's diagnostic-only rule for nonzero router gradients.
        for router in core.write_routers + core.read_routers:
            list(router.parameters())[0].grad.fill(0)
        self.assertTrue(scope['first_gradient_audit'](model)['passed'])

if __name__=='__main__': unittest.main()
