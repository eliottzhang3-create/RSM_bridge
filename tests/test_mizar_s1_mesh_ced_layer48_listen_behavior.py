"""Run real source using stdlib/NumPy; these checks do not claim Torch autograd."""
import ast
import copy
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT/'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_listen'
sys.path.insert(0, str(ROUTE))
import audit_reference as audit
import contract
from models import recursive_text


class ContractTests(unittest.TestCase):
    def test_complete_mother_training_and_backbone_parity(self):
        result = audit.audit()
        self.assertEqual(len(result['parent_parity']['identical_files_normalized_lf']), 14)
        self.assertTrue(result['backbone_parity']['entire_backbone_ast_equal_after_explicit_additions'])

    def test_exact_ced_branch_matches_original_non_mesh_route(self):
        source = ROUTE.with_name('mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject')
        for name in ('models/ced.py', 'models/mellow.py'):
            self.assertEqual((ROUTE/name).read_bytes().replace(b'\r\n', b'\n'),
                             (source/name).read_bytes().replace(b'\r\n', b'\n'))
        self.assertEqual(contract.LISTEN_CONTRACT['placement'], 'after_original_write_before_original_read')
        self.assertEqual(contract.LISTEN_CONTRACT['gate_and_listen_query'], 'module_output_audio_positions')
        self.assertEqual(contract.LISTEN_CONTRACT['new_parameter_count'], 2443776 + 2*577 + 2*2885)

    def test_protected_parent_files_unchanged(self):
        baseline = json.loads((ROOT/'audit_reports/mizar_s1_mesh_ced_layer48_listen_parent_baseline.json').read_text())
        for name, expected in baseline.items():
            # This sibling is not an ancestor: another chat edits it concurrently.
            # Retain the original snapshot as provenance, without gating our isolated route on it.
            if '/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer36912_listen/' in name:
                continue
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest(), expected, name)

    def test_identity_includes_listen_math_and_is_independent_of_staging_root(self):
        prepared = dict(rows=1477492, source_rows=contract.SOURCE_COUNTS,
                        rows_canonical_sha256='rows', mapping_sha256='mapping', stage_root='a')
        one = contract.semantic_contract(prepared, 'text', 'ced')
        prepared['stage_root'] = 'b'
        self.assertEqual(one, contract.semantic_contract(prepared, 'text', 'ced'))
        self.assertEqual(one['audio_listen'], contract.LISTEN_CONTRACT)
        changed = copy.deepcopy(one)
        changed['audio_listen']['placement'] = 'after_read'
        self.assertNotEqual(contract.digest(one), contract.digest(changed))
        raw = (ROUTE/'models/recursive_mesh_listen.py').read_bytes().replace(b'\r\n', b'\n')
        self.assertEqual(contract.recursive_model_source_sha256(), hashlib.sha256(raw).hexdigest())

    def test_strict_loading_finishes_before_new_modules_registration(self):
        for failure in (None, 'missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
            events = []
            model = SimpleNamespace(model=SimpleNamespace(enable_audio_listen=lambda: events.append('register')),
                                    generation_config=SimpleNamespace(use_cache=True))
            cls = Mock()
            cls.from_pretrained.side_effect = lambda *a, **k: (events.append('load') or model,
                                                              {} if failure is None else {failure: ['bad']})
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path/'config.json').write_text('{}')
                with patch.object(recursive_text, '_import_recursive_model', return_value=(cls, lambda m: {})), \
                     patch.object(recursive_text, 'validate_loaded_model', side_effect=lambda *a: events.append('validate')), \
                     patch.object(recursive_text, 'validate_audio_listen'):
                    if failure:
                        with self.assertRaises(ValueError): recursive_text.load_recursive_text_model(path)
                        self.assertEqual(events, ['load'])
                    else:
                        recursive_text.load_recursive_text_model(path)
                        self.assertEqual(events, ['load', 'validate', 'register'])
                        self.assertFalse(model.generation_config.use_cache)

    def test_optimizer_covers_eight_roles_and_preserves_original_lr_decay_schedule(self):
        class Parameter:
            requires_grad = True
            def __init__(self, ndim): self.ndim = ndim
            def numel(self): return 12
        prefixes = ['audio_encoder.base.mapper', 'caption_decoder.lm.model.layers.5',
                    'caption_decoder.lm.model.write_routers.0', 'caption_decoder.lm.model.read_routers.2']
        for layer in (4, 8):
            prefixes += [f'audio_encoder.base.layer_mlps.{layer}', f'caption_decoder.lm.model.audio_gates.{layer}',
                         f'caption_decoder.lm.model.listen_routers.{layer}']
        pairs = [(p+suffix, Parameter(ndim)) for p in prefixes
                 for suffix, ndim in (('.weight', 2), ('.bias', 1), ('.norm.weight', 1))]
        class Optimizer:
            def __init__(self, groups, **kwargs): self.param_groups, self.kwargs = groups, kwargs
        roles = ('map', 'default', *(f'layer{l}_{k}' for l in (4,8) for k in ('mlp','gate','listen')))
        scope = dict(torch=SimpleNamespace(optim=SimpleNamespace(AdamW=Optimizer,
            lr_scheduler=SimpleNamespace(LambdaLR=lambda opt, fn: fn))), lr_factor=contract.lr_factor,
            TRAINABLE_ROLES=roles)
        module = audit.tree(ROUTE/'optim.py')
        audit.execute([audit.find_function(module, n) for n in ('parameter_role', 'build_optimizer')], scope)
        opt, schedule, groups = scope['build_optimizer'](SimpleNamespace(named_parameters=lambda: iter(pairs)), 17313)
        self.assertEqual({g['role'] for g in groups}, set(roles))
        for group in groups:
            self.assertEqual(group['base_lr'], 3e-4 if group['role']=='default' else 6e-4)
            for name in group['names']:
                self.assertEqual(group['weight_decay'], 0. if name.endswith('bias') or '.norm.' in name else .01)
        self.assertEqual(opt.kwargs['betas'], (.9,.999))
        for step in range(17314): self.assertEqual(schedule(step), contract.lr_factor(step, 17313))

    def test_formal_namespace_and_cache_rejection(self):
        launcher = (ROOT/'code/RSmol/run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_listen_5090.sh').read_text()
        runner = (ROUTE/'run.sh').read_text()
        self.assertIn('recursive_mesh_layer4_8_listen_5090/index_cache', launcher)
        self.assertIn('/dev/shm/mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_listen_', runner)
        self.assertNotIn('python runtime_checks.py', runner)
        self.assertNotIn('smoke-report', runner+launcher)


class ResumeTests(unittest.TestCase):
    def payload(self, step=1):
        recipe = {'synthetic': True}
        geometry = contract.geometry(1477492)
        role_names = {f'layer{l}_{k}': True for l in (4,8) for k in ('mlp','gate','listen')}
        state = {'complete': False, 'seen': {k: k.endswith('gate') for k in role_names}, 'deadline': 32}
        if step>=2: state = dict(complete=True, seen=role_names, deadline=32, opened_at_update=2)
        payload = dict(schema_version=1, route_variant=contract.ROUTE, mode='formal', contract=recipe,
            contract_sha256=contract.digest(recipe), state_dict={}, grad_scaler={}, grad_norm_tracker={},
            random_state_by_rank=[dict(python=1,numpy=2,torch=3,cuda=4) for _ in range(8)],
            total_step=step, epoch=step//geometry['steps_per_epoch'], next_optimizer_step=step%geometry['steps_per_epoch'],
            scheduler={'last_epoch':step}, num_epochs=3,
            loss_reduction='mean_of_rank_local_microbatch_token_means_then_DDP_mean',
            batch_geometry=dict(world_size=8,microbatch_size=8,gradient_accumulation_steps=4,global_batch_size=256),
            listen_gradient_state=state,
            optimizer={'state': {0:{'step':float(step)}}, 'param_groups':[dict(params=[0],betas=(.9,.999),eps=1e-8,
                weight_decay=.01,initial_lr=6e-4,lr=6e-4*contract.lr_factor(step,geometry['total_steps']))]})
        return payload, dict(contract=recipe, contract_sha256=contract.digest(recipe)), geometry

    def test_complete_states_before_after_unlock_and_final_reload(self):
        for step in (1,2,5771,17313):
            data,cfg,geom=self.payload(step)
            contract.validate_resume_payload(data,cfg,geom,allow_completed=True)
            if step==17313:
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


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'requires NumPy')
class NumericTests(unittest.TestCase):
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
        module=audit.tree(ROUTE/'models/recursive_mesh_listen.py')
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
            ('stack',5),('write',1),('read',1),('stack',15),('write',2),('read',2),('stack',25)])
        stacks=[e[2] for e in obj.events if e[0]=='stack']
        writes=[e[2] for e in obj.events if e[0]=='write']
        reads=[e[2] for e in obj.events if e[0]=='read']
        for i,layer in enumerate((4,8)):
            # Recover module OUTPUT from the known synthetic stacks (not the memory read).
            output=stacks[i].copy()
            for physical in (range(5) if i==0 else range(5,15)): output=output+.01*(physical+1)
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
        self.assertEqual(obj.last_listen_trace,((5,4),(15,8)))
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


if __name__=='__main__': unittest.main()
