"""Actual production forward/functions with NumPy fixtures; never GPU/autograd evidence."""
import ast
import copy
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / 'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71437_mesh_ced48_wfuse_lora_r8'
sys.path.insert(0, str(ROUTE))
import contract
import audit_reference as audit


def source_functions(file, names, scope):
    module = audit.tree(ROUTE / file)
    audit.execute([audit.find_function(module, name) for name in names], scope)
    return scope


class Tensor(np.ndarray):
    @property
    def device(self): return 'cpu'
    def detach(self): return self
    def ne(self, value): return self != value
    def numel(self): return self.size
    def cpu(self): return self
    def clone(self): return self.copy()
    def abs(self): return abs(self)
    def zero_(self): self.fill(0); return self
    def copy_(self, value): self[...] = value; return self
    def unsqueeze(self, dim): return np.expand_dims(self, dim)
    def transpose(self, *axes):
        return np.swapaxes(self, *axes) if len(axes) == 2 and self.ndim != 2 else super().transpose(*axes)
    def permute(self, *axes): return super().transpose(*axes)


class FusionMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = audit.tree(ROUTE / 'models/recursive_mesh_wfuse_lora.py')
        scope = dict(torch=NS(cat=lambda xs, dim: np.concatenate(xs, axis=dim)))
        audit.execute([audit.find_function(module, 'fuse_write_value')], scope)
        cls.fuse = staticmethod(scope['fuse_write_value'])
        forward = next(n for c in module.body if isinstance(c, ast.ClassDef) and c.name == 'MeshLlamaModel'
                       for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        begin = next(i for i, n in enumerate(forward.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'memory')
        end = next(i for i, n in enumerate(forward.body) if isinstance(n, ast.Assign) and ast.unparse(n.value) == 'self.norm(hidden)')
        resets = [n for n in forward.body[:begin] if isinstance(n, ast.Assign)
                  and ast.unparse(n.targets[0]) in ('self.last_fusion_trace', 'self.last_fusion_stats')]
        cls.body = resets + forward.body[begin:end+1]

    def fixture(self, closed=False):
        class Linear:
            def __init__(self, layer):
                self.weight = np.concatenate((np.eye(3), np.eye(3) * (0 if closed else .1 * layer)), axis=1)
                self.inputs = []
            def __call__(self, x): self.inputs.append(x.copy()); return x @ self.weight.T
        obj = NS(fusion_audio_tokens=126, collect_fusion_stats=False, last_fusion_trace=(),
                 audit_mode=False, gradient_audit_mode=False, write_routers=list(range(4)), read_routers=list(range(4)),
                 audio_fusions={str(k): Linear(k) for k in (4, 8)}, norm=lambda x: x,
                 events=[], queries={}, values=[], memories=[], last_forward_trace=[])
        obj.fuse_write_value = types.MethodType(self.fuse, obj)
        def stack(h, indices, start, **kw):
            obj.events.append((start, h.copy()))
            for physical in indices:
                obj.last_forward_trace.append(dict(physical_index=physical))
                h = h + .01 * (physical + 1)
            return h
        def route(router, q, name):
            obj.queries[name] = q.copy()
            return self.probabilities(q, router)
        def write(m, v, p):
            obj.values.append(v.copy())
            out = m + v[:, None] * p.transpose(0, 2, 1)[..., None]
            obj.memories.append(out.copy())
            return out
        obj._run_stack, obj._route, obj._write = stack, route, write
        obj._read = lambda m, p: (m * p.transpose(0, 2, 1)[..., None]).sum(axis=1)
        x = np.random.default_rng(42).normal(size=(2, 131, 3)).view(Tensor)
        pair = tuple(np.random.default_rng(k).normal(size=(2, 126, 3)).view(Tensor) for k in (4, 8))
        return obj, x, pair

    @staticmethod
    def probabilities(q, index):
        score = q.mean(axis=-1, keepdims=True) * np.linspace(-.1, .2, 6) + index * .02 * np.arange(6)
        p = np.exp(score - score.max(axis=-1, keepdims=True))
        return p / p.sum(axis=-1, keepdims=True)

    def run_forward(self, obj, x, pair):
        obj.events, obj.queries, obj.values, obj.memories, obj.last_forward_trace = [], {}, [], [], []
        scope = dict(self=obj, hidden=x.copy(), audio_information=pair, batch_size=x.shape[0], query_length=x.shape[1],
            MEMORY_SLOT_COUNT=6, RECURSIVE_LOOPS=3, PREFIX_LAYER_COUNT=7, MIDDLE_LAYER_COUNT=14, PHYSICAL_LAYER_COUNT=28, torch=NS(zeros=lambda shape, **kw: np.zeros(shape).view(Tensor)),
            record_function=lambda name: nullcontext(), output_hidden_states=False, mask=None, position_ids=None,
            cache=None, use_cache=False, cache_position=None, position_embeddings=None, output_attentions=False, attentions=[])
        audit.execute(copy.deepcopy(self.body), scope)
        return scope['hidden'], scope['memory']

    def test_actual_forward_matches_independent_value_only_recurrence(self):
        obj, x, pair = self.fixture()
        actual, memory = self.run_forward(obj, x, pair)
        expected_memory = np.zeros((2, 6, 131, 3)); expected_memory[:, 0] = x
        prefix = x + .01 * sum(range(1, 8))
        query = prefix
        for index in range(4):
            value = prefix if index == 0 else hidden + .01 * sum(range(8, 22))
            expected_value = value.copy()
            if index < 2:
                weight = obj.audio_fusions[str((4, 8)[index])].weight
                expected_value[:, :126] = value[:, :126] @ weight[:, :3].T + pair[index] @ weight[:, 3:].T
            probabilities = self.probabilities(query, index)
            expected_memory += expected_value[:, None] * probabilities.transpose(0, 2, 1)[..., None]
            hidden = (expected_memory * probabilities.transpose(0, 2, 1)[..., None]).sum(axis=1)
            query = hidden.copy()
        expected = hidden + .01 * sum(range(22, 29))
        np.testing.assert_allclose(actual, expected, atol=1e-13)
        np.testing.assert_allclose(memory, expected_memory, atol=1e-13)
        self.assertEqual(obj.last_fusion_trace, ((7, 4), (21, 8)))
        self.assertEqual([n['physical_index'] for n in obj.last_forward_trace], list(contract.TEXT_MODEL_SCHEDULE))

    def test_original_queries_and_non_audio_values_are_preserved(self):
        obj, x, pair = self.fixture(); self.run_forward(obj, x, pair)
        prefix = obj.events[0][1].copy()
        for p in range(7): prefix += .01 * (p+1)
        np.testing.assert_array_equal(obj.queries['write_pre'], prefix)
        np.testing.assert_array_equal(obj.queries['read_pre'], prefix)
        np.testing.assert_array_equal(obj.values[0][:, 126:], prefix[:, 126:])
        for i in (0, 1, 2):
            entrance = obj.events[i+1][1]
            np.testing.assert_array_equal(obj.queries[f'write_{i}'], entrance)
            np.testing.assert_array_equal(obj.queries[f'read_{i}'], entrance)
            output = entrance.copy()
            for p in range(7, 21): output += .01 * (p+1)
            np.testing.assert_array_equal(obj.values[i+1][:, 126:], output[:, 126:])
            if i >= 1: np.testing.assert_array_equal(obj.values[i+1], output)

    def test_identity_fusions_equal_original_full_forward_and_memory(self):
        obj, x, pair = self.fixture(closed=True)
        actual, memory = self.run_forward(obj, x, pair)
        original, _, _ = self.fixture()
        original.fuse_write_value = lambda h, *a: h
        expected, expected_memory = self.run_forward(original, x, pair)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(memory, expected_memory)

    def test_fusion_is_out_of_place_and_each_matrix_is_independent(self):
        obj, x, pair = self.fixture(); before = x.copy()
        out = obj.fuse_write_value(x, pair[0], 4, 7)
        np.testing.assert_array_equal(x, before)
        np.testing.assert_array_equal(out[:, 126:], x[:, 126:])
        self.assertFalse(np.array_equal(out[:, :126], x[:, :126]))
        second = obj.audio_fusions['8'].weight.copy()
        obj.audio_fusions['4'].weight[0, 0] += 1
        np.testing.assert_array_equal(second, obj.audio_fusions['8'].weight)

    def test_memory_and_trace_rebuilt_each_complete_forward(self):
        obj, x, pair = self.fixture()
        y, memory = self.run_forward(obj, x, pair)
        changed = tuple(a * 2 for a in pair)
        other, _ = self.run_forward(obj, x, changed)
        self.assertFalse(np.allclose(y, other))
        again, rebuilt = self.run_forward(obj, x, pair)
        np.testing.assert_array_equal(y, again); np.testing.assert_array_equal(memory, rebuilt)
        self.assertEqual(obj.last_fusion_trace, ((7, 4), (21, 8)))

    def test_zero_encoder_half_blocks_feature_gradient_but_can_learn_first(self):
        obj, x, pair = self.fixture(closed=True)
        target = np.random.default_rng(10).normal(size=x.shape)
        def loss(): return float((obj.fuse_write_value(x, pair[0], 4, 7) * target).sum())
        def diff(value, index):
            old, eps = value[index], 1e-5
            value[index] = old + eps; plus = loss()
            value[index] = old - eps; minus = loss()
            value[index] = old
            return (plus-minus)/(2*eps)
        self.assertEqual(diff(pair[0], (0, 0, 0)), 0.)
        self.assertGreater(abs(diff(obj.audio_fusions['4'].weight, (0, 3))), 1e-5)
        self.assertGreater(abs(diff(obj.audio_fusions['4'].weight, (0, 0))), 1e-5)
        obj.audio_fusions['4'].weight[0, 3] = .2
        self.assertGreater(abs(diff(pair[0], (0, 0, 0))), 1e-5)

    def test_fusion_initialization_executes_real_function_preserving_rng(self):
        class ModuleDict(dict):
            def to(self, device): return self
        class Linear:
            def __init__(self, n, m, bias, **kw):
                self.weight = np.random.normal(size=(m, n)).view(Tensor); self.bias = None
        class Fork:
            def __enter__(self): self.state = np.random.get_state()
            def __exit__(self, *args): np.random.set_state(self.state)
        scope = dict(nn=NS(ModuleDict=ModuleDict, Linear=Linear), torch=NS(random=NS(fork_rng=lambda **kw: Fork()),
            no_grad=nullcontext, eye=lambda n, **kw: np.eye(n)))
        source_functions('models/recursive_mesh_wfuse_lora.py', ['enable_audio_fusion'], scope)
        obj = NS(config=NS(hidden_size=3), embed_tokens=NS(weight=np.zeros((7, 3)).view(Tensor)))
        state = copy.deepcopy(np.random.get_state())
        scope['enable_audio_fusion'](obj)
        after = np.random.get_state()
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(state, after)))
        for layer in (4, 8):
            np.testing.assert_array_equal(obj.audio_fusions[str(layer)].weight, np.concatenate((np.eye(3), np.zeros((3,3))), axis=1))
        self.assertIsNot(obj.audio_fusions['4'].weight, obj.audio_fusions['8'].weight)
        with self.assertRaises(RuntimeError): scope['enable_audio_fusion'](obj)

    def test_ced_grid_keeps_segment_time_frequency_alignment(self):
        scope = source_functions('models/ced.py', ['_hidden_to_temporal_grid'], {})
        hidden = np.arange(2*504*384).reshape(2,504,384).view(Tensor)
        actual = scope['_hidden_to_temporal_grid'](None, hidden)
        source = hidden.reshape(2,2,4,63,384)
        for b in range(2):
            for segment in range(2):
                for t in (0, 31, 62):
                    for f in range(4): np.testing.assert_array_equal(actual[b,segment*63+t,f],source[b,segment,f,t])


class ArchitectureTests(unittest.TestCase):
    def test_parameter_counts_and_eight_original_routers(self):
        self.assertEqual(2*576*1152, contract.FUSION_CONTRACT['fusion_parameters'])
        self.assertEqual(2443776+1327104+3419136, 7190016)
        self.assertEqual(contract.FUSION_CONTRACT['router_modules'], 8)
        self.assertEqual(contract.FUSION_CONTRACT['memory_slots'], 6)
        self.assertEqual(contract.FUSION_CONTRACT['write_router_indices_0based'], [0,1])
        self.assertEqual(contract.LOOP_LORA_CONTRACT['new_parameters'], 3419136)

    def test_actual_optimizer_has_twelve_roles_and_all_588_lora_tensors(self):
        class Parameter:
            requires_grad=True
            def __init__(self, ndim): self.ndim=ndim
            def numel(self): return 12
        pairs=[]
        for prefix in ('audio_encoder.base.mapper','caption_decoder.lm.model.layers.0'):
            pairs.extend((prefix+suffix,Parameter(ndim)) for suffix,ndim in (('.weight',2),('.bias',1)))
        for layer in (4,8):
            pairs.extend([(f'audio_encoder.base.layer_mlps.{layer}.net.1.weight',Parameter(2)),
                          (f'audio_encoder.base.layer_mlps.{layer}.net.0.weight',Parameter(1)),
                          (f'caption_decoder.lm.model.audio_fusions.{layer}.weight',Parameter(2))])
        pairs.extend((f'caption_decoder.lm.model.layers.{i}.{target}.lora_{kind}.{loop}',Parameter(2))
            for i in range(7,21) for target in contract.LOOP_LORA_CONTRACT['targets'] for kind in ('A','B') for loop in (0,1,2))
        class Optimizer:
            def __init__(self, groups, **kw): self.param_groups=groups
        scope=dict(torch=NS(optim=NS(AdamW=Optimizer,lr_scheduler=NS(LambdaLR=lambda o,f:f))),lr_factor=contract.lr_factor)
        for node in audit.tree(ROUTE/'optim.py').body:
            if isinstance(node,ast.Assign): audit.execute([node],scope)
        source_functions('optim.py',['parameter_role','build_optimizer'],scope)
        _, schedule, groups = scope['build_optimizer'](NS(named_parameters=lambda:pairs),34626)
        self.assertEqual(len(scope['TRAINABLE_ROLES']),12)
        self.assertEqual(sum(len(g['names']) for g in groups if 'lora' in g['role']),588)
        for g in groups:
            self.assertEqual(g['base_lr'],3e-4 if g['role']=='default' else 6e-4)
            if g['role'].endswith('_fusion'): self.assertEqual(g['weight_decay'],.01)
        for step in range(34627): self.assertEqual(schedule(step),contract.lr_factor(step,34626))

    def test_loader_strict_inheritance_precedes_identity_registration_and_lora(self):
        source=ast.unparse(audit.find_function(audit.tree(ROUTE/'models/recursive_text.py'),'load_recursive_text_model'))
        self.assertLess(source.index('audit_checkpoint_tensors'),source.index('enable_audio_fusion'))
        self.assertLess(source.index('enable_audio_fusion'),source.index('install_loop_lora(model.model)'))
        self.assertIn('inherited_parameter_objects_preserved=True',source)
        self.assertIn('fresh=True',source)



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
        return NS(weight=np.eye(3) * 0.99, bias=np.zeros(3), scaling=2.0, loop_index=None, lora_A=[rng.normal(size=(8, 3)) * 0.1 for _ in range(3)], lora_B=[rng.normal(size=(3, 8)) * (0.0 if closed else 0.04) for _ in range(3)])

    def test_nonzero_adapters_scale_two_and_select_distinct_rounds(self):
        p, x = (self.projection(), np.ones((2, 131, 3)))
        outputs = []
        for loop in (0, 1, 2):
            p.loop_index = loop
            y = self.projection_forward(p, x)
            np.testing.assert_allclose(y, x @ p.weight.T + 2 * (x @ p.lora_A[loop].T @ p.lora_B[loop].T))
            outputs.append(y)
        self.assertFalse(np.allclose(outputs[0], outputs[1])); self.assertFalse(np.allclose(outputs[1], outputs[2]))
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
        for loop in (0, 1, 2):
            self.assertEqual(self.call_scope(layer, loop, callback, 2), 3)
            self.assertTrue(all((p.loop_index is None for p in modules.values())))
        self.assertIsNot(selected[0][0][0], selected[1][0][0])

        def failure(*a):
            raise ValueError('synthetic decoder failure')
        with self.assertRaises(ValueError):
            self.call_scope(layer, 0, failure)
        self.assertTrue(all((p.loop_index is None for p in modules.values())))


class ResumeTests(unittest.TestCase):
    def test_actual_checkpoint_writer_saves_eight_rngs_fusion_state_and_named_optimizer(self):
        parameter=NS(shape=(576,1152),dtype='torch.float32')
        name='caption_decoder.lm.model.audio_fusions.4.weight'
        roles={f'layer{layer}_{kind}':True for layer in (4,8) for kind in ('mlp','fusion')}
        roles.update({f'loop{loop}_lora_{kind}':True for loop in (1,2,3) for kind in ('A','B')})
        unlock=dict(complete=True,seen=roles,deadline=32,opened_at_update=2)
        marker=np.eye(3)
        model=NS(named_parameters=lambda:[(name,parameter)],state_dict=lambda:{name:marker.copy()},
                 caption_decoder=NS(lm=NS(model=NS(fusion_gradient_state=unlock))))
        steps=contract.geometry(1477492)
        group=dict(params=[0],betas=(.9,.999),eps=1e-8,weight_decay=.01,initial_lr=6e-4,
                   lr=6e-4*contract.lr_factor(2,34626))
        optimizer=NS(param_groups=[dict(params=[parameter])],state_dict=lambda:dict(state={0:dict(step=2.)},param_groups=[group]))
        state=lambda **value:NS(state_dict=lambda:value)
        cfg=dict(mode='formal',contract=dict(geometry=steps),contract_sha256=contract.digest(dict(geometry=steps)))
        captured={}
        def save(payload,path):
            captured.update(copy.deepcopy(payload));Path(path).write_bytes(b'explicit torch.save fixture')
        def gather(values, local):
            values[:]=[dict(python=f'p{rank}',numpy=f'n{rank}',torch=f't{rank}',cuda=f'c{rank}') for rank in range(8)]
        scope=dict(WORLD_SIZE=8,ROUTE=contract.ROUTE,torch=NS(save=save),os=__import__('os'),
                   RANK_UPDATE_SAMPLES=16,BATCH_SIZE=4,ACCUMULATION_STEPS=4,EPOCHS=3,
                   parameter_role=lambda value:'layer4_fusion',rng_state=lambda device:{},
                   dist=NS(all_gather_object=gather,get_rank=lambda:0,barrier=lambda:None))
        source_functions('train.py',['optimizer_parameter_inventory','save_checkpoint'],scope)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'step2.ckpt'
            scope['save_checkpoint'](path,model,optimizer,state(last_epoch=2),state(scale=65536),state(running_norm=1),cfg,2,0,2,'fixture')
            self.assertTrue(path.is_file());self.assertFalse(path.with_suffix('.tmp').exists())
        contract.validate_resume_payload(captured,cfg,steps)
        self.assertEqual(captured['random_state_by_rank'][3]['torch'],'t3')
        np.testing.assert_array_equal(captured['state_dict'][name],marker)
        self.assertEqual(captured['fusion_gradient_state'],unlock)
        self.assertEqual(captured['optimizer_parameter_inventory'][0][0]['name'],name)

    def payload(self, step=1):
        recipe = {'synthetic': True}
        geometry = contract.geometry(1477492)
        role_names = {f'layer{l}_{k}': True for l in (4,8) for k in ('mlp','fusion')}
        role_names.update({f'loop{loop}_lora_{kind}':True for loop in (1,2,3) for kind in ('A','B')})
        state = {'complete': False, 'seen': {k: k.endswith('fusion') for k in role_names}, 'deadline': 32}
        if step>=2: state = dict(complete=True, seen=role_names, deadline=32, opened_at_update=2)
        payload = dict(schema_version=1, route_variant=contract.ROUTE, mode='formal', contract=recipe,
            contract_sha256=contract.digest(recipe), state_dict={}, grad_scaler={}, grad_norm_tracker={},
            random_state_by_rank=[dict(python=1,numpy=2,torch=3,cuda=4) for _ in range(8)],
            total_step=step, epoch=step//geometry['steps_per_epoch'], next_optimizer_step=step%geometry['steps_per_epoch'],
            scheduler={'last_epoch':step}, num_epochs=3,
            loss_reduction='mean_of_rank_local_microbatch_token_means_then_DDP_mean',
            batch_geometry=dict(world_size=8,microbatch_size=4,gradient_accumulation_steps=4,global_batch_size=128),
            fusion_gradient_state=state,
            optimizer_parameter_inventory=[[dict(name='caption_decoder.lm.model.audio_fusions.4.weight',shape=[576,1152],dtype='torch.float32',role='layer4_fusion')]],
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
            lambda d:d['fusion_gradient_state'].update(complete=True), lambda d:d['fusion_gradient_state'].update(deadline=0),
            lambda d:d['fusion_gradient_state']['seen'].update(unknown=True), lambda d:d.pop('fusion_gradient_state'),
            lambda d:d.pop('optimizer_parameter_inventory'), lambda d:d['optimizer_parameter_inventory'][0][0].update(role='default'),
            lambda d:d['contract'].update(synthetic=False),
            lambda d:d['fusion_gradient_state']['seen'].pop('loop3_lora_A')]
        for mutate in changes:
            d,c,g=self.payload()
            mutate(d)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError): contract.validate_resume_payload(d,c,g)


class GradientAuditTests(unittest.TestCase):
    def fixture(self, opened=False):
        prefixes = {'map':'audio_encoder.base.mapper', 'default':'caption_decoder.lm.model.layers.0'}
        for layer in (4,8):
            prefixes[f'layer{layer}_mlp']=f'audio_encoder.base.layer_mlps.{layer}'
            prefixes[f'layer{layer}_fusion']=f'caption_decoder.lm.model.audio_fusions.{layer}'
        for loop in (1,2,3):
            for kind in ('A','B'):
                prefixes[f'loop{loop}_lora_{kind}']=f'caption_decoder.lm.model.layers.7.self_attn.q_proj.lora_{kind}.{loop-1}'
        roles, pairs={},[]
        for role,prefix in prefixes.items():
            if role.endswith('_fusion'):
                value=np.concatenate((np.eye(2),np.eye(2)*(1 if opened else 0)),axis=1).view(Tensor)
            else:
                value=np.full((1,2),0. if role.endswith('lora_B') and not opened else 1.).view(Tensor)
            value.requires_grad=True
            zero=not opened and (role.endswith('_mlp') or role.endswith('lora_A'))
            value.grad=np.full_like(value,0. if zero else 1.).view(Tensor)
            roles[role]=value
            pairs.append((prefix if 'lora_' in role else prefix+'.weight',value))
        lora=NS(lora_B=[roles[f'loop{i}_lora_B'] for i in (1,2,3)])
        core=NS(config=NS(hidden_size=2),audio_fusions={str(k):NS(weight=roles[f'layer{k}_fusion']) for k in (4,8)},
            modules=lambda:[lora],last_fusion_trace=((7,4),(21,8)),
            last_loop_lora_trace=[(i,p,(i-7)//14) for i,p in enumerate(contract.TEXT_MODEL_SCHEDULE) if 7<=p<21],
            last_forward_trace=[dict(physical_index=p) for p in contract.TEXT_MODEL_SCHEDULE],write_routers=[],read_routers=[])
        for kind in ('write','read'):
            for index in range(4):
                value=np.ones((1,2)).view(Tensor);value.requires_grad=True;value.grad=np.ones_like(value)
                getattr(core,kind+'_routers').append(NS(parameters=lambda value=value:[value]))
                pairs.append((f'caption_decoder.lm.model.{kind}_routers.{index}.weight',value))
        model=NS(caption_decoder=NS(lm=NS(model=core)),named_parameters=lambda:pairs)
        scope=dict(torch=NS(isfinite=np.isfinite,equal=np.array_equal,tensor=lambda a,**kw:np.array(a).view(Tensor)),
                   TEXT_MODEL_SCHEDULE=contract.TEXT_MODEL_SCHEDULE)
        for node in audit.tree(ROUTE/'optim.py').body:
            if isinstance(node,ast.Assign):audit.execute([node],scope)
        source_functions('optim.py',['parameter_role'],scope)
        source_functions('train.py',['first_gradient_audit','fusion_gradient_audit','role_snapshots','weights_changed'],scope)
        return model,roles,scope

    def test_initial_zero_encoder_and_lora_B_expect_zero_upstream_task_gradients(self):
        model,roles,scope=self.fixture()
        result=scope['first_gradient_audit'](model)
        self.assertTrue(result['passed'],result)
        self.assertEqual(result['closed_encoder_weights'],{'4':True,'8':True})
        self.assertEqual(result['closed_lora_B'],{'1':True,'2':True,'3':True})
        roles['layer4_mlp'].grad.fill(1)
        self.assertFalse(scope['first_gradient_audit'](model)['passed'])

    def test_encoder_half_requires_task_gradient_even_when_hidden_half_has_gradient(self):
        model,roles,scope=self.fixture()
        roles['layer8_fusion'].grad[:,2:]=0
        result=scope['first_gradient_audit'](model)
        self.assertTrue(result['nonzero_by_role']['layer8_fusion'])
        self.assertFalse(result['fusion_blocks']['8']['encoder_nonzero'])
        self.assertFalse(result['passed'])

    def test_opened_fusion_and_lora_require_upstream_gradients(self):
        model,roles,scope=self.fixture(opened=True)
        self.assertTrue(scope['first_gradient_audit'](model)['passed'])
        for role in ('layer4_mlp','layer8_mlp','loop1_lora_A','loop2_lora_A','loop3_lora_A'):
            roles[role].grad.fill(0)
            self.assertFalse(scope['first_gradient_audit'](model)['passed'],role)
            roles[role].grad.fill(1)

    def test_missing_nonfinite_and_wrong_trace_are_rejected(self):
        model,roles,scope=self.fixture(opened=True)
        for bad in (None,np.full_like(roles['layer8_fusion'],np.nan)):
            old=roles['layer8_fusion'].grad;roles['layer8_fusion'].grad=bad
            self.assertFalse(scope['first_gradient_audit'](model)['passed'])
            roles['layer8_fusion'].grad=old
        model.caption_decoder.lm.model.last_fusion_trace=((7,8),(21,4))
        self.assertFalse(scope['first_gradient_audit'](model)['passed'])

    def test_all_eight_original_router_gradients_checked(self):
        model,roles,scope=self.fixture(opened=True);core=model.caption_decoder.lm.model
        for router in core.write_routers+core.read_routers:
            parameter=list(router.parameters())[0];old=parameter.grad
            parameter.grad=None
            self.assertFalse(scope['first_gradient_audit'](model)['passed'])
            parameter.grad=old
        self.assertTrue(scope['first_gradient_audit'](model)['passed'])

    def test_each_new_role_has_snapshot_and_changes_are_independent(self):
        model,roles,scope=self.fixture()
        before=scope['role_snapshots'](model)
        self.assertEqual(set(before),set(roles))
        roles['layer4_fusion'] += .25
        changes=scope['weights_changed'](model,before)
        self.assertTrue(changes['layer4_fusion']['changed'])
        self.assertFalse(changes['layer8_fusion']['changed'])

    def test_actual_collective_unlock_state_first_then_opened_and_deadline(self):
        model,roles,scope=self.fixture();reports=[]
        def require(condition,message,device):
            if not condition:raise RuntimeError(message)
        scope.update(collective_require=require,write_json=lambda path,value:reports.append(copy.deepcopy(value)),
                     dist=NS(all_reduce=lambda *a,**kw:None,ReduceOp=NS(MIN=0)))
        source_functions('train.py',['audit_fusion_update'],scope)
        state=dict(complete=False,seen={r:False for r in scope['NEW_ROLES']},deadline=32)
        scope['audit_fusion_update'](model,state,1,Path('.'),0,'cpu')
        self.assertFalse(state['complete']);self.assertTrue(state['seen']['layer4_fusion'])
        opened,_,_=self.fixture(opened=True)
        scope['audit_fusion_update'](opened,state,2,Path('.'),0,'cpu')
        self.assertTrue(state['complete']);self.assertEqual(reports[0]['opened_at_update'],2)
        state=dict(complete=False,seen={},deadline=32)
        with self.assertRaises(RuntimeError):scope['audit_fusion_update'](model,state,32,Path('.'),0,'cpu')


if __name__ == '__main__': unittest.main()

