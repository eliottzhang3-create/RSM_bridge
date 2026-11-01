"""Execute actual memory code with NumPy; not a substitute for Torch autograd tests."""
import ast
import copy
import math
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT/'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_audio7slot_split3_inject'
sys.path.insert(0, str(ROUTE))
import audit_reference as audit
import contract


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np
        class Tensor(np.ndarray):
            def unsqueeze(self, dim): return np.expand_dims(self, dim)
            def transpose(self, *axes): return np.swapaxes(self, *axes)
            def new_zeros(self, *shape): return np.zeros(shape).view(Tensor)
            def detach(self): return self
            def cpu(self): return self.copy()
            def sum(self, dim=None, **kw): return super().sum(axis=kw.pop('axis',dim), **kw)
        cls.Tensor = Tensor
        scope = dict(torch=types.SimpleNamespace(cat=lambda xs, dim:np.concatenate(xs, axis=dim).view(Tensor),
                                                tanh=np.tanh))
        module = audit.tree(ROUTE/'models/recursive_mesh_listen.py')
        for name in ('_write', '_read', '_mixed_interaction', 'inject_audio'):
            function = copy.deepcopy(audit.find_function(module, name))
            function.decorator_list = []
            function.returns = None
            for arg in function.args.args: arg.annotation = None
            audit.execute([function], scope)
            setattr(cls, name, staticmethod(scope[name]))
        core = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'MeshLlamaModel')
        forward = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        begin = next(i for i,n in enumerate(forward.body) if isinstance(n,ast.Assign)
                     and ast.unparse(n.targets[0]) == 'n')
        end = next(i for i,n in enumerate(forward.body) if isinstance(n,ast.Assign)
                   and ast.unparse(n.value) == 'self.norm(hidden)')
        cls.body = forward.body[begin:end+1]

    def fixture(self):
        np, T = self.np, self.Tensor
        obj = types.SimpleNamespace(listen_audio_tokens=126, audit_mode=True, gradient_audit_mode=False,
            collect_listen_stats=False, write_routers=list(range(3)), read_routers=list(range(3,6)),
            audio_listen_write_routers=list(range(6,9)), audio_listen_read_routers=list(range(9,12)),
            last_router_weights={}, queries={}, last_memory_write_history=[],
            last_core_inputs=[], last_core_outputs=[], events=[], norm=lambda x:x, last_forward_trace=[],
            last_injection_trace=(), last_injection_audit=[],
            audio_gates={str(k):lambda h:np.zeros((*h.shape[:2],1)).view(T) for k in (4,8)})
        obj._write, obj._read = self._write, self._read
        obj._mixed_interaction = types.MethodType(self._mixed_interaction, obj)
        obj.inject_audio = types.MethodType(self.inject_audio, obj)
        def stack(h, indices, start, **kw):
            obj.events.append((start,h.copy()))
            for physical in indices:
                obj.last_forward_trace.append(dict(physical_index=physical))
                h = h + (physical+1)*.01
            return h
        def route(router, q, name):
            count=7 if name.startswith('audio_') else 5
            scores=q.mean(axis=-1,keepdims=True)*np.linspace(-.1,.2,count)+router*.01*np.arange(count)
            p=np.exp(scores-scores.max(axis=-1,keepdims=True)); p=p/p.sum(axis=-1,keepdims=True)
            obj.queries[name]=q.copy(); obj.last_router_weights[name]=p.copy()
            return p.view(T)
        obj._run_stack, obj._route = stack, route
        x=np.random.default_rng(42).normal(size=(2,131,3)).view(T)
        pair=tuple(np.random.default_rng(k).normal(size=(2,126,3)).view(T) for k in (4,8))
        return obj,x,pair

    def forward(self,obj,x,pair):
        np = self.np
        scope = dict(self=obj,hidden=x,audio_information=pair,batch_size=x.shape[0],query_length=x.shape[1],
            torch=types.SimpleNamespace(cat=lambda xs,dim:np.concatenate(xs,axis=dim).view(self.Tensor)),
            record_function=lambda name:nullcontext(),output_hidden_states=False,mask=None,position_ids=None,
            cache=None,use_cache=False,cache_position=None,position_embeddings=None,output_attentions=False,attentions=[])
        audit.execute(copy.deepcopy(self.body),scope)
        return scope['hidden'],scope['memory']

    def test_initial_seeds_and_exact_reference_three_interactions(self):
        np = self.np
        obj,x,pair=self.fixture()
        out,mem=self.forward(obj,x,pair)
        audio,text=obj.last_initial_memory
        np.testing.assert_array_equal(audio[:,0],x[:,:126])
        np.testing.assert_array_equal(audio[:,1:5],0)
        np.testing.assert_array_equal(audio[:,5],pair[0]); np.testing.assert_array_equal(audio[:,6],pair[1])
        np.testing.assert_array_equal(text[:,0],x[:,126:]); np.testing.assert_array_equal(text[:,1:],0)
        m=[audio.copy(),text.copy()]
        h=x.copy()
        for stage,(indices,suffix) in enumerate(((range(5),'pre'),(range(5,15),'0'),(range(5,15),'1'))):
            value=h.copy()
            for physical in indices: value=value+(physical+1)*.01
            query=value if stage==0 else h
            for branch,region in enumerate((slice(None,126),slice(126,None))):
                stem='audio_' if branch==0 else ''
                np.testing.assert_array_equal(obj.queries[f'{stem}write_{suffix}'],query[:,region])
                np.testing.assert_array_equal(obj.queries[f'{stem}read_{suffix}'],query[:,region])
                w=obj.last_router_weights[f'{stem}write_{suffix}']
                p=obj.last_router_weights[f'{stem}read_{suffix}']
                m[branch]=m[branch]+value[:,None,region,:]*np.swapaxes(w,1,2)[...,None]
                actual=obj.last_memory_write_history[stage][branch]
                np.testing.assert_allclose(actual,m[branch],rtol=0,atol=0)
            h=np.concatenate([(m[b]*np.swapaxes(obj.last_router_weights[f'{stem}read_{suffix}'],1,2)[...,None]).sum(axis=1)
                              for b,stem in ((0,'audio_'),(1,''))],axis=1).view(self.Tensor)
        for physical in range(15,20): h=h+(physical+1)*.01
        np.testing.assert_array_equal(out,h)
        self.assertEqual(len(obj.queries),12)
        self.assertEqual([e['physical_index'] for e in obj.last_forward_trace],list(contract.TEXT_MODEL_SCHEDULE))
        self.assertEqual([m.shape for m in mem],[(2,7,126,3),(2,5,5,3)])

    def test_all_seven_slots_receive_ordinary_writes_and_no_reinjection(self):
        np = self.np
        obj,x,pair=self.fixture(); _,mem=self.forward(obj,x,pair)
        # Subtract each ordinary write from final memory: only the original seeds remain.
        recovered=mem[0].copy()
        for stage,suffix in enumerate(('pre','0','1')):
            h=obj.events[stage][1].copy()
            for i in (range(5) if stage==0 else range(5,15)): h=h+(i+1)*.01
            w=obj.last_router_weights[f'audio_write_{suffix}']
            recovered-=h[:,:126][:,None]*np.swapaxes(w,1,2)[...,None]
        np.testing.assert_allclose(recovered,obj.last_initial_memory[0],atol=2e-15)
        self.assertFalse(np.array_equal(mem[0][:,5:],obj.last_initial_memory[0][:,5:]))

    def test_three_way_probability_mass_and_other_four_slots_preserved(self):
        np=self.np
        logits=np.concatenate((np.random.default_rng(5).normal(size=(100,5))*50,
            [[1000,-1000,-1000,-1000,-1000],[-1000,-999,-998,-997,-996],
             [-1000,1000,1000,1000,1000],[0,0,0,0,0]]))
        def softmax(z):
            p=np.exp(z-z.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)
        expanded=logits[:,[0,1,2,3,4,0,0]].copy()
        expanded[:,[0,5,6]]-=math.log(3.)
        old,seven=softmax(logits),softmax(expanded)
        np.testing.assert_allclose(seven[:,1:5],old[:,1:5],rtol=1e-12,atol=1e-15)
        for i in (0,5,6):np.testing.assert_allclose(seven[:,i],old[:,0]/3,rtol=1e-12,atol=1e-15)
        np.testing.assert_allclose(seven[:,[0,5,6]].sum(-1),old[:,0],rtol=1e-12,atol=1e-15)

    def initialized_router(self,seed=0):
        np=self.np
        class Parameter(np.ndarray):
            def detach(self):return self
            def cpu(self):return self
            def copy_(self,value):np.copyto(self,value);return self
            def __getitem__(self,key):
                value=super().__getitem__(key)
                if isinstance(value,np.generic) and isinstance(key,int):
                    return super().__getitem__(slice(key,key+1)).reshape(())
                return value
        scope=dict(torch=types.SimpleNamespace(no_grad=nullcontext),math=math)
        audit.execute([audit.find_function(audit.tree(ROUTE/'models/recursive_mesh_listen.py'),'_init_audio_router_split3')],scope)
        rng=np.random.default_rng(seed)
        source=types.SimpleNamespace(weight=rng.normal(size=(5,8)).view(Parameter),
            bias=(rng.normal(size=5)+2.).view(Parameter))
        target=types.SimpleNamespace(weight=np.full((7,8),np.nan).view(Parameter),
                                    bias=np.full(7,np.nan).view(Parameter))
        scope['_init_audio_router_split3'](source,target)
        return source,target

    def test_actual_initializer_copies_corresponding_rows_adjusts_first_bias_and_does_not_alias(self):
        np=self.np
        for router_index in range(6):
            source,target=self.initialized_router(router_index)
            np.testing.assert_array_equal(target.weight,source.weight[[0,1,2,3,4,0,0]])
            np.testing.assert_array_equal(target.bias[1:5],source.bias[1:5])
            for i in (0,5,6):self.assertEqual(target.bias[i],source.bias[0]-math.log(3.))
            source_before=source.weight.copy();other_rows=target.weight[[0,5]].copy()
            target.weight[6,0]+=1.
            np.testing.assert_array_equal(source.weight,source_before)
            np.testing.assert_array_equal(target.weight[[0,5]],other_rows)

    def test_actual_initializer_real_linear_queries_follow_logit_formula(self):
        np=self.np
        for router_index in range(6):
            source,target=self.initialized_router(router_index)
            query=np.random.default_rng(router_index+20).normal(size=(2,126,8))
            z=query@source.weight.T+source.bias
            actual=query@target.weight.T+target.bias
            expected=z[...,[0,1,2,3,4,0,0]].copy();expected[...,[0,5,6]]-=math.log(3.)
            np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-14)

    def test_split_write_and_read_contribution_is_not_artificially_rescaled(self):
        np=self.np
        obj,_,_=self.fixture()
        e=np.array([1.,2.,3.]);a4=np.array([4.,5.,6.]);a8=np.array([7.,8.,9.]);v=np.array([2.,4.,6.])
        pwrite,pread=.3,.6
        memory=np.stack((e,a4,a8)).reshape(1,3,1,3).view(self.Tensor)
        pw=np.full((1,1,3),pwrite/3).view(self.Tensor)
        pr=np.full((1,1,3),pread/3).view(self.Tensor)
        actual=obj._read(obj._write(memory,v.reshape(1,1,3).view(self.Tensor),pw),pr)
        np.testing.assert_allclose(actual[0,0],(pread/3)*(e+a4+a8)+(pwrite*pread/3)*v,rtol=1e-15)

    def test_grid_order_preserved(self):
        scope={};audit.execute([audit.find_function(audit.tree(ROUTE/'models/ced.py'),'_hidden_to_temporal_grid')],scope)
        class Grid(self.np.ndarray):
            def permute(self,*axes):return super().transpose(*axes)
        h=self.np.arange(2*504*384).reshape(2,504,384).view(Grid)
        grid=scope['_hidden_to_temporal_grid'](None,h)
        for b in range(2):
            for s in range(2):
                for t in (0,31,62):
                    for f in range(4):self.np.testing.assert_array_equal(grid[b,s*63+t,f],h[b,s*252+f*63+t])

    def test_zero_gates_match_actual_parent_forward_and_memories(self):
        obj,x,pair=self.fixture()
        actual,memory=self.forward(obj,x,pair)
        parent=next(n for n in audit.tree(audit.PARENT/'models/recursive_mesh_listen.py').body
                    if isinstance(n,ast.ClassDef) and n.name=='MeshLlamaModel')
        function=next(n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        begin=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='n')
        end=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign) and ast.unparse(n.value)=='self.norm(hidden)')
        old=self.body
        try:
            self.body=function.body[begin:end+1]
            other,y,information=self.fixture()
            expected,expected_memory=self.forward(other,y,information)
        finally:
            self.body=old
        self.np.testing.assert_array_equal(actual,expected)
        for a,b in zip(memory,expected_memory):self.np.testing.assert_array_equal(a,b)
        self.assertEqual(obj.last_injection_trace,((5,4),(15,8)))
        self.assertEqual(len(obj.last_injection_audit),2)

    def test_nonzero_gates_use_read_hidden_only_at_two_correct_boundaries(self):
        np,T=self.np,self.Tensor
        obj,x,pair=self.fixture()
        seen={}
        weights={'4':np.array([.12,-.07,.04]),'8':np.array([-.09,.03,.11])}
        biases={'4':.02,'8':-.05}
        def make_gate(layer):
            def gate(h):
                seen[layer]=h.copy()
                return (h@weights[layer]+biases[layer])[...,None].view(T)
            return gate
        obj.audio_gates={k:make_gate(k) for k in ('4','8')}
        _,memory=self.forward(obj,x,pair)
        for stage,layer in enumerate(('4','8')):
            record=obj.last_injection_audit[stage]
            suffix='pre' if stage==0 else '0'
            read=np.concatenate([obj._read(obj.last_memory_write_history[stage][b],
                obj.last_router_weights[f'{stem}read_{suffix}']) for b,stem in ((0,'audio_'),(1,''))],axis=1)
            np.testing.assert_array_equal(record['read_hidden'],read)
            np.testing.assert_array_equal(seen[layer],read[:,:126])
            self.assertFalse(np.array_equal(seen[layer],obj.queries[f'audio_read_{suffix}']))
            gate=np.tanh(read[:,:126]@weights[layer]+biases[layer])[...,None]
            np.testing.assert_array_equal(record['gate'],gate)
            expected=read.copy();expected[:,:126]+=gate*pair[stage]
            np.testing.assert_array_equal(record['injected_hidden'],expected)
            np.testing.assert_array_equal(obj.events[stage+1][1],expected)
            np.testing.assert_array_equal(record['injected_hidden'][:,126:],read[:,126:])
            np.testing.assert_array_equal(obj.queries[f'audio_read_{stage}'],expected[:,:126])
        # Direct residuals never add an extra write to any of the seven slots.
        recovered=memory[0].copy()
        for stage,suffix in enumerate(('pre','0','1')):
            value=obj.events[stage][1].copy()
            for physical in (range(5) if stage==0 else range(5,15)):value+=(physical+1)*.01
            recovered-=value[:,:126][:,None]*np.swapaxes(obj.last_router_weights[f'audio_write_{suffix}'],1,2)[...,None]
        np.testing.assert_allclose(recovered,obj.last_initial_memory[0],atol=3e-15)
        self.assertEqual(obj.last_injection_trace,((5,4),(15,8)))


if __name__=='__main__':unittest.main()
