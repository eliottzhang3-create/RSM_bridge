"""Actual MeSH and LoRA source execution using stdlib/NumPy; not PyTorch autograd."""
import ast
import copy
import os
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_textmesh_ced_small_layer8_4_inject_audio_reinject_loop_modality_lora_r8"
sys.path.insert(0, str(ROUTE))
import audit_reference as audit
import contract
from models import recursive_text


def source_functions(file, names, scope):
    module = audit.tree(ROUTE / file)
    audit.execute([audit.find_function(module, name) for name in names], scope)
    return scope


class ContractTests(unittest.TestCase):

    def test_runtime_contract_rejects_changed_memory_or_serial_semantics(self):
        fn = audit.find_function(audit.tree(ROUTE / 'train.py'), 'train')
        guard = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                     and "contract.get('audio_reinjection')" in ast.unparse(n.test))
        recipe = dict(audio_reinjection=contract.AUDIO_REINJECTION,
                      serial_injection=contract.SERIAL_INJECTION_CONTRACT,
                      text_memory=contract.TEXT_MEMORY_CONTRACT)
        scope = dict(contract=recipe, AUDIO_REINJECTION=contract.AUDIO_REINJECTION,
                     SERIAL_INJECTION_CONTRACT=contract.SERIAL_INJECTION_CONTRACT,
                     TEXT_MEMORY_CONTRACT=contract.TEXT_MEMORY_CONTRACT)
        audit.execute([guard], scope)
        for key, field, value in (('text_memory', 'audio_memory', True),
                                  ('text_memory', 'text_start', 127),
                                  ('serial_injection', 'ced_layers_1based', [4, 8]),
                                  ('serial_injection', 'intermediate_gate_condition', 'old_hidden'),
                                  ('audio_reinjection', 'shared_gate', False)):
            changed = copy.deepcopy(recipe)
            changed[key][field] = value
            with self.subTest(key=key, field=field), self.assertRaises(ValueError):
                audit.execute([guard], dict(scope, contract=changed))

    def test_input_projection_is_once_outside_loops_and_ced_is_unchanged(self):
        tree = audit.tree(ROUTE/'models/recursive_layer_inject.py')
        core = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MeshLlamaModel')
        forward = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        calls = [n for n in ast.walk(forward) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == 'self.audio_linear']
        self.assertEqual(len(calls), 1)
        for loop in (n for n in ast.walk(forward) if isinstance(n, (ast.For, ast.While))):
            self.assertFalse(any(n is calls[0] for n in ast.walk(loop)))
        self.assertEqual((ROUTE/'models/ced.py').read_bytes(), (audit.PARENT/'models/ced.py').read_bytes())

    def test_full_source_parity_and_protected_originals(self):
        result = audit.audit()
        self.assertTrue(result['protected_originals']['all_unchanged'])
        self.assertEqual(len(result['identical_source_files_except_route_names']), 19)
        self.assertTrue(result['training_parity']['entire_train_function_ast_equal_after_explicit_audits'])

    def test_exact_lora_recipe_and_actual_gqa_parameter_count(self):
        lora = contract.LOOP_LORA_CONTRACT
        self.assertEqual((lora['rank'], lora['alpha'], lora['scale'], lora['max_lr']), (8, 16, 2.0, 0.0006))
        self.assertEqual(lora['physical_layers_0based'], list(range(5, 15)))
        self.assertEqual(len(lora['targets']), 7)
        actual = 10 * 4 * 8 * (2 * (576 + 576) + 2 * (576 + 192) + 3 * (576 + 1536))
        self.assertEqual(actual, 3256320)
        self.assertEqual(actual, lora['new_parameters'])
        self.assertTrue(lora['base_trainable'] and lora['shared_base_between_loops'])
        self.assertFalse(lora['prefix_suffix_lora'])

    def test_strict_loader_never_installs_lora_on_failed_pretrained_load(self):
        for failure in (None, 'missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
            events = []
            core = NS(enable_layer_injection=lambda: events.append('layers'), enable_audio_reinjection=lambda: events.append('input'), enable_loop_lora=lambda: events.append('lora'))
            model = NS(model=core, generation_config=NS(use_cache=True))
            cls = Mock()
            cls.from_pretrained.return_value = (model, {} if failure is None else {failure: ['bad']})
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                (Path(directory) / 'config.json').write_text('{}')
                with patch.object(recursive_text, '_import_recursive_model', return_value=(cls, lambda m: {})), patch.object(recursive_text, 'validate_loaded_model', side_effect=lambda *a: events.append('validate')), patch.object(recursive_text, 'validate_serial_injection'), patch.object(recursive_text, 'validate_text_memory'), patch.object(recursive_text, 'validate_loop_lora'):
                    if failure:
                        with self.assertRaises(ValueError):
                            recursive_text.load_recursive_text_model(directory)
                        self.assertEqual(events, [])
                    else:
                        recursive_text.load_recursive_text_model(directory)
                        self.assertEqual(events, ['validate', 'layers', 'input', 'lora'])
                        self.assertFalse(model.generation_config.use_cache)
                    self.assertTrue(cls.from_pretrained.call_args.kwargs['local_files_only'])

    def test_resume_rejects_other_route_and_changed_lora_configuration(self):
        prepared = dict(rows=1477492, source_rows=contract.SOURCE_COUNTS, rows_canonical_sha256='r', mapping_sha256='m')
        recipe = contract.semantic_contract(prepared, 'text', 'ced')
        fn = audit.find_function(audit.tree(ROUTE / 'train.py'), 'train')
        guard = next((n for n in ast.walk(fn) if isinstance(n, ast.If) and "saved.get('schema_version')" in ast.unparse(n.test)))
        valid = dict(schema_version=1, route_variant=contract.ROUTE, mode='formal', contract_sha256=contract.digest(recipe))
        scope = dict(saved=valid, config={'contract_sha256': valid['contract_sha256']}, ROUTE=contract.ROUTE)
        audit.execute([guard], scope)
        for key, wrong in (('route_variant', contract.ROUTE.replace('layer8_4', 'other').replace('layer4_8', 'other')), ('route_variant', contract.ROUTE.replace('_loop_modality_lora_r8', '')), ('contract_sha256', 'changed'), ('mode', 'smoke')):
            with self.subTest(key=key, wrong=wrong), self.assertRaises(ValueError):
                audit.execute([guard], dict(scope, saved=dict(valid, **{key: wrong})))
        mother = dict(valid, route_variant=contract.ROUTE.replace('loop_modality_lora_r8', 'loop_lora_r16'))
        with self.assertRaises(ValueError):
            audit.execute([guard], dict(scope, saved=mother))
        for key, wrong in (('alpha', 8), ('max_lr', 0.0003), ('rank', 4),
                           ('physical_layers_0based', list(range(20))), ('audio_positions', [0, 127]),
                           ('text_start', 127), ('modalities', ['text', 'audio']), ('scale', 1.),
                           ('adapter_sets', 2)):
            changed = copy.deepcopy(recipe)
            changed['loop_lora'][key] = wrong
            self.assertNotEqual(contract.digest(recipe), contract.digest(changed))

    def test_optimizer_four_lora_sets_are_6e4_all_groups_keep_original_schedule(self):

        class Parameter:
            requires_grad = True

            def __init__(self, ndim):
                self.ndim = ndim

            def numel(self):
                return 12
        prefixes = ['audio_encoder.base.mapper', 'caption_decoder.lm.model.layers.0', 'audio_encoder.base.layer_mlps.4', 'audio_encoder.base.layer_mlps.8', 'caption_decoder.lm.model.audio_gates.4', 'caption_decoder.lm.model.audio_gates.8', 'caption_decoder.lm.model.audio_linear', 'caption_decoder.lm.model.audio_gate']
        pairs = [(p + suffix, Parameter(ndim)) for p in prefixes for suffix, ndim in (('.weight', 2), ('.bias', 1))]
        lora_names = [f'caption_decoder.lm.model.layers.{layer}.{target}.{kind}.{modality}.{loop}' for layer in range(5, 15) for target in contract.LOOP_LORA_CONTRACT['targets'] for kind in ('lora_A', 'lora_B') for modality in ('audio', 'text') for loop in (0, 1)]
        pairs += [(name, Parameter(2)) for name in lora_names]

        class Optimizer:

            def __init__(self, groups, **kw):
                self.param_groups, self.kw = (groups, kw)
        fake = NS(optim=NS(AdamW=Optimizer, lr_scheduler=NS(LambdaLR=lambda opt, fn: fn)))
        scope = source_functions('optim.py', ['parameter_role', 'build_optimizer'], dict(torch=fake, lr_factor=contract.lr_factor))
        optimizer, schedule, groups = scope['build_optimizer'](NS(named_parameters=lambda: pairs), 17313)
        lora_groups = [g for g in groups if 'lora' in g['role']]
        self.assertEqual(len(lora_groups), 8)
        self.assertEqual(sum((len(g['names']) for g in lora_groups)), 560)
        for group in groups:
            self.assertEqual(group['base_lr'], 0.0003 if group['role'] == 'default' else 0.0006)
            if 'lora' in group['role']:
                self.assertEqual(group['weight_decay'], 0.01)
        self.assertEqual(optimizer.kw['betas'], (0.9, 0.999))
        self.assertEqual(optimizer.kw['eps'], 1e-08)
        for step in range(17314):
            self.assertEqual(schedule(step), contract.lr_factor(step, 17313))
        for wrong in (lora_names[0].replace('layers.5.', 'layers.4.'), lora_names[0].replace('lora_A.audio.0', 'lora_A.audio.2'), lora_names[0].replace('q_proj', 'bad_proj'), lora_names[0].replace('.audio.', '.unknown.')):
            with self.assertRaises(ValueError):
                scope['parameter_role'](wrong)


class NumericalTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        module = audit.tree(ROUTE / 'models/loop_lora.py')
        linear = next((n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'LoopLoRALinear'))
        forward = next((copy.deepcopy(n) for n in linear.body if isinstance(n, ast.FunctionDef) and n.name == 'forward'))
        scope = dict(F=NS(linear=lambda x, w, bias=None: x @ w.T + (0 if bias is None else bias)),
                     torch=NS(cat=lambda xs,dim:np.concatenate(xs,axis=dim)))
        audit.execute([forward], scope)
        cls.projection_forward = staticmethod(scope['forward'])
        scope = source_functions('models/loop_lora.py', ['projections', 'call_with_loop_lora'], dict(TARGETS=contract.LOOP_LORA_CONTRACT['targets']))
        cls.call_scope = staticmethod(scope['call_with_loop_lora'])

    def projection(self, seed=0, closed=False):
        rng = np.random.default_rng(seed)
        return NS(weight=np.eye(3)*.99,bias=np.zeros(3),scaling=2.,loop_index=None,audio_tokens=126,
                  forward_counts=[0,0],last_token_shapes=[None,None],
                  lora_A={m:[rng.normal(size=(8,3))*.1 for _ in range(2)] for m in ('audio','text')},
                  lora_B={m:[rng.normal(size=(3,8))*(0. if closed else .04) for _ in range(2)] for m in ('audio','text')})

    def test_nonzero_adapters_scale_two_and_select_distinct_rounds(self):
        p, x = (self.projection(), np.ones((2, 131, 3)))
        outputs = []
        for loop in (0, 1):
            p.loop_index = loop
            y = self.projection_forward(p, x)
            np.testing.assert_allclose(y, x @ p.weight.T + 2 * (np.concatenate((x[:,:126] @ p.lora_A['audio'][loop].T @ p.lora_B['audio'][loop].T, x[:,126:] @ p.lora_A['text'][loop].T @ p.lora_B['text'][loop].T),axis=1)))
            outputs.append(y)
        self.assertFalse(np.allclose(*outputs))
        self.assertFalse(np.allclose(outputs[0][:, 126:], x[:, 126:] @ p.weight.T))
        p.loop_index = None
        with self.assertRaises(RuntimeError):
            self.projection_forward(p, x)

    def test_initial_B_zero_A_zero_task_gradient_and_independent_branch_gradient(self):
        p, x = (self.projection(closed=True), np.random.default_rng(5).normal(size=(2, 131, 3)))
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
        self.assertEqual(diff(p.lora_A['audio'][0], (0, 0)), 0.0)
        self.assertNotEqual(diff(p.lora_B['audio'][0], (0, 0)), 0.0)
        self.assertEqual(diff(p.lora_B['audio'][1], (0, 0)), 0.0)
        self.assertNotEqual(diff(p.weight, (0, 0)), 0.0)
        p.lora_B['audio'][0][0, 0] = 0.1
        self.assertNotEqual(diff(p.lora_A['audio'][0], (0, 0)), 0.0)

    def test_each_adapter_changes_only_its_direct_modality_and_round_including_separator(self):
        x = np.random.default_rng(6).normal(size=(2, 131, 3))
        for loop in (0, 1):
            for modality in ('audio', 'text'):
                with self.subTest(loop=loop, modality=modality):
                    p = self.projection()
                    p.loop_index = loop
                    before = self.projection_forward(p, x)
                    p.lora_B[modality][loop] += .3
                    after = self.projection_forward(p, x)
                    changed = slice(None, 126) if modality == 'audio' else slice(126, None)
                    unchanged = slice(126, None) if modality == 'audio' else slice(None, 126)
                    np.testing.assert_array_equal(after[:, unchanged], before[:, unchanged])
                    self.assertFalse(np.allclose(after[:, changed], before[:, changed]))
                    for boundary, owner in ((125, 'audio'), (126, 'text')):
                        self.assertEqual(np.allclose(after[:, boundary], before[:, boundary]), owner != modality)
                    p.loop_index = 1 - loop
                    untouched_round = self.projection_forward(p, x)
                    original = self.projection()
                    original.loop_index = 1 - loop
                    np.testing.assert_array_equal(untouched_round, self.projection_forward(original, x))

    def test_modality_only_loss_routes_finite_difference_gradients_to_own_adapter(self):
        x = np.random.default_rng(7).normal(size=(1, 130, 3))
        for loop in (0, 1):
            for modality, tokens in (('audio', slice(None, 126)), ('text', slice(126, None))):
                p = self.projection()
                p.loop_index = loop
                for branch in ('audio', 'text'):
                    for other_loop in (0, 1):
                        for kind in ('lora_A', 'lora_B'):
                            matrix = getattr(p, kind)[branch][other_loop]
                            old, eps = matrix[0, 0], 1e-5
                            matrix[0, 0] = old + eps
                            plus = np.square(self.projection_forward(p, x)[:, tokens]).sum()
                            matrix[0, 0] = old - eps
                            minus = np.square(self.projection_forward(p, x)[:, tokens]).sum()
                            matrix[0, 0] = old
                            gradient = (plus - minus) / (2 * eps)
                            with self.subTest(loop=loop, modality=modality, branch=branch, other_loop=other_loop, kind=kind):
                                if (branch, other_loop) == (modality, loop):
                                    self.assertNotEqual(gradient, 0.)
                                else:
                                    self.assertEqual(gradient, 0.)

    def test_projection_rejects_missing_text_or_flattened_token_axis(self):
        p = self.projection()
        p.loop_index = 0
        for shape in ((126, 3), (1, 126, 3), (1, 125, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.projection_forward(p, np.ones(shape))
        self.assertEqual(p.forward_counts, [0, 0])

    def test_scope_resets_selection_on_success_exception_and_backward_independent_tensors(self):
        modules = {name: self.projection(i) for i, name in enumerate(contract.LOOP_LORA_CONTRACT['targets'])}
        layer = NS(get_submodule=lambda name: modules[name])
        selected = []

        def callback(layer, x):
            selected.append([(p.lora_A['audio'][p.loop_index], p.lora_B['audio'][p.loop_index]) for p in modules.values()])
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
        prefixes = {'map': 'audio_encoder.base.mapper', 'default': 'caption_decoder.lm.model.layers.0', 'layer4_mlp': 'audio_encoder.base.layer_mlps.4', 'layer8_mlp': 'audio_encoder.base.layer_mlps.8', 'layer4_gate': 'caption_decoder.lm.model.audio_gates.4', 'layer8_gate': 'caption_decoder.lm.model.audio_gates.8', 'reinject_linear': 'caption_decoder.lm.model.audio_linear', 'reinject_gate': 'caption_decoder.lm.model.audio_gate'}
        for loop in (1, 2):
            for modality in ('audio','text'):
                for kind in ('A', 'B'):
                    prefixes[f'loop{loop}_{modality}_lora_{kind}'] = f'caption_decoder.lm.model.layers.5.self_attn.q_proj.lora_{kind}.{modality}.{loop - 1}'
        roles, pairs = ({}, [])
        for role, prefix in prefixes.items():
            closed = role.endswith('gate') or role.endswith('lora_B')
            value = np.full((1, 2), 0.0 if closed and (not opened) else 1.0).view(Tensor)
            value.requires_grad = True
            zero_grad = (role in {'layer4_mlp','layer8_mlp','reinject_linear'} or role.endswith('lora_A')) and not opened
            value.grad = np.full((1, 2), 0.0 if zero_grad else 1.0).view(Tensor)
            roles[role] = value
            pairs.append((prefix if 'lora_' in role else prefix + '.weight', value))
        gate = lambda role: NS(parameters=lambda: [roles[role]])
        lora = NS(lora_B={m:[roles[f'loop{i}_{m}_lora_B'] for i in (1,2)] for m in ('audio','text')})
        core = NS(audio_gates={str(k): gate(f'layer{k}_gate') for k in (4, 8)}, modules=lambda: [lora], last_injection_trace=((5,8),(15,4)), last_loop_lora_trace=[(i, p, 0 if i < 15 else 1) for i, p in enumerate(contract.TEXT_MODEL_SCHEDULE) if 5 <= p < 15])
        core.audio_gate = gate('reinject_gate')
        core.last_reinject_trace = (5,15)
        core.last_serial_injection_trace = tuple(tuple(v) for v in contract.SERIAL_INJECTION_CONTRACT['trace'])
        core.config = NS(hidden_size=2)
        core.last_memory_shape = (1,5,557,2)
        core.last_sequence_length = 683
        core.last_router_query_shapes = {name:(1,557,2) for name in ('write_pre','read_pre','write_0','read_0','write_1','read_1')}
        projection=lambda:NS(loop_index=None,forward_counts=[1,1],last_token_shapes=[(1,126,557)]*2)
        core.layers=[NS(ps={name:projection() for name in contract.LOOP_LORA_CONTRACT['targets']}) for _ in range(20)]
        for layer in core.layers:layer.get_submodule=lambda name,layer=layer:layer.ps[name]
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
        scope = dict(torch=NS(isfinite=np.isfinite, equal=np.array_equal), TEXT_MODEL_SCHEDULE=contract.TEXT_MODEL_SCHEDULE, SERIAL_INJECTION_CONTRACT=contract.SERIAL_INJECTION_CONTRACT)
        source_functions('optim.py', ['parameter_role'], scope)
        scope.update(MIDDLE_LAYERS=tuple(range(5,15)), LOOPS=2, MODALITIES=('audio','text'),
                     AUDIO_TOKENS=126, TARGETS=contract.LOOP_LORA_CONTRACT['targets'])
        source_functions('models/loop_lora.py', ['projections', 'validate_loop_lora_execution', 'reset_loop_lora_trace'], scope)
        source = audit.tree(ROUTE / 'train.py')
        for n in source.body:
            if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id in {'TRAINABLE_ROLES', 'NEW_ROLES'} for t in n.targets)):
                audit.execute([n], scope)
        source_functions('train.py', ['first_gradient_audit', 'role_snapshots', 'weights_changed'], scope)
        return (model, roles, scope)

    def test_closed_B_expects_zero_A_but_requires_nonzero_B_in_both_loops(self):
        model, roles, scope = self.fixture()
        result = scope['first_gradient_audit'](model)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['closed_lora_B'], {f'loop{i}_{m}':True for i in (1,2) for m in ('audio','text')})
        self.assertIn('loop1_audio_lora_A', result['zero_task_gradient_expected'])
        for role in (f'loop{i}_{m}_lora_B' for i in (1,2) for m in ('audio','text')):
            roles[role].grad.fill(0)
            self.assertFalse(scope['first_gradient_audit'](model)['passed'])
            roles[role].grad.fill(1)

    def test_opened_B_requires_task_gradient_in_A_and_each_audio_upstream(self):
        model, roles, scope = self.fixture(opened=True)
        self.assertTrue(scope['first_gradient_audit'](model)['passed'])
        for role in [*(f'loop{i}_{m}_lora_A' for i in (1,2) for m in ('audio','text')),'layer4_mlp','layer8_mlp','reinject_linear']:
            roles[role].grad.fill(0)
            self.assertFalse(scope['first_gradient_audit'](model)['passed'], role)
            roles[role].grad.fill(1)

    def test_missing_nonfinite_gradients_and_wrong_loop_trace_fail(self):
        model, roles, scope = self.fixture(opened=True)
        for bad in (None, np.full((1, 2), np.nan)):
            old = roles['loop2_audio_lora_A'].grad
            roles['loop2_audio_lora_A'].grad = bad.view(type(old)) if isinstance(bad, np.ndarray) else bad
            self.assertFalse(scope['first_gradient_audit'](model)['passed'])
            roles['loop2_audio_lora_A'].grad = old
        model.caption_decoder.lm.model.last_loop_lora_trace = []
        self.assertFalse(scope['first_gradient_audit'](model)['passed'])

    def test_all_sixteen_roles_have_separate_snapshot_and_change_detection(self):
        model, roles, scope = self.fixture()
        snapshots = scope['role_snapshots'](model)
        self.assertEqual(set(snapshots), set(roles))
        self.assertTrue(all((not item['changed'] for item in scope['weights_changed'](model, snapshots).values())))
        roles['loop2_audio_lora_B'] += 0.5
        changed = scope['weights_changed'](model, snapshots)
        self.assertTrue(changed['loop2_audio_lora_B']['changed'])
        self.assertFalse(changed['loop1_audio_lora_B']['changed'])

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


    def test_audio_memory_and_wrong_router_region_are_rejected(self):
        model,roles,scope=self.fixture(opened=True);core=model.caption_decoder.lm.model
        saved=core.last_memory_shape
        core.last_memory_shape=(1,5,683,2)
        self.assertFalse(scope['first_gradient_audit'](model)['passed'])
        core.last_memory_shape=saved
        core.last_router_query_shapes['write_0']=(1,683,2)
        self.assertFalse(scope['first_gradient_audit'](model)['passed'])
        core.last_router_query_shapes['write_0']=(1,557,2)
        core.last_serial_injection_trace=((5,'layer8'),(5,'input'),(15,'input'),(15,'layer4'))
        self.assertFalse(scope['first_gradient_audit'](model)['passed'])

    def test_execution_validator_rejects_missing_duplicate_boundary_and_stale_selection(self):
        model, roles, scope = self.fixture(opened=True)
        core = model.caption_decoder.lm.model
        validate = scope['validate_loop_lora_execution']
        result = validate(core)
        self.assertEqual(result['adapter_calls'], 280)
        self.assertEqual(result['calls_per_set'], {f'loop{i}_{m}': 70 for i in (1, 2) for m in ('audio', 'text')})
        for index in range(5, 15):
            for target in contract.LOOP_LORA_CONTRACT['targets']:
                module = core.layers[index].ps[target]
                for field, bad in (('forward_counts', [0, 1]), ('forward_counts', [1, 2]),
                                   ('last_token_shapes', [(1, 127, 556), (1, 126, 557)]),
                                   ('last_token_shapes', [(1, 126, 557), (2, 126, 557)]),
                                   ('loop_index', 1)):
                    old = getattr(module, field)
                    setattr(module, field, bad)
                    with self.subTest(index=index, target=target, field=field), self.assertRaises(ValueError):
                        validate(core)
                    self.assertFalse(scope['first_gradient_audit'](model)['passed'])
                    setattr(module, field, old)
        scope['reset_loop_lora_trace'](core)
        for index in range(5, 15):
            for module in core.layers[index].ps.values():
                self.assertEqual(module.forward_counts, [0, 0])
                self.assertEqual(module.last_token_shapes, [None, None])
        with self.assertRaises(ValueError):
            validate(core)
        core.layers[5].ps[contract.LOOP_LORA_CONTRACT['targets'][0]].loop_index = 0
        with self.assertRaises(RuntimeError):
            scope['reset_loop_lora_trace'](core)



class MeshExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        NumericalTests.setUpClass()
        cls.projection_forward = staticmethod(NumericalTests.projection_forward)
        cls.call_scope = staticmethod(NumericalTests.call_scope)
        module = audit.tree(ROUTE/'models/recursive_layer_inject.py')
        model = next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='MeshLlamaModel')
        forward = next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        start = next(i for i,n in enumerate(forward.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='memory')
        end = next(i for i,n in enumerate(forward.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='hidden' and ast.unparse(n.value)=='self.norm(hidden)')
        cls.body = forward.body[start:end+2]
        stack = copy.deepcopy(audit.find_function(module,'_run_stack'))
        for arg in [*stack.args.args,*stack.args.kwonlyargs]: arg.annotation=None
        stack.returns=None
        scope=dict(call_with_loop_lora=cls.call_scope, _call_layer=lambda layer,h,**kw:(layer(h),))
        audit.execute([stack],scope)
        cls.stack = staticmethod(scope['_run_stack'])
        scope=dict(torch=NS(tanh=np.tanh,cat=lambda seq,dim:np.concatenate(seq,axis=dim)))
        source_functions('models/recursive_layer_inject.py',['inject_audio','inject_input_audio'],scope)
        cls.inject = staticmethod(scope['inject_audio'])
        cls.inject_input = staticmethod(scope['inject_input_audio'])

    def fixture(self, closed=False, close_gates=False):
        class Tensor(np.ndarray):
            @property
            def device(self):return 'cpu'
        class Gate:
            def __init__(slf,value):slf.weight,slf.inputs=np.full((3,1),value),[]
            def __call__(slf,h):slf.inputs.append(h.copy());return h@slf.weight
        class Layer:
            def __init__(slf,index):
                slf.index=index
                slf.ps={name:NumericalTests.projection(self,index*10+i,closed)
                        for i,name in enumerate(contract.LOOP_LORA_CONTRACT['targets'])} if 5<=index<15 else {}
            def get_submodule(slf,name):return slf.ps[name]
            def __call__(slf,h):
                if slf.ps:
                    for p in slf.ps.values(): h=self.projection_forward(p,h)
                return h+.001*(slf.index+1)
        obj=NS(layers=[Layer(i) for i in range(20)],layer_inject_tokens=126,collect_injection_stats=False,
               audio_reinject_tokens=126,collect_reinject_stats=False,audio_gate=Gate(0. if close_gates else .04),
               audio_linear=lambda x:x*.7+.1,
               audio_gates={'4':Gate(0. if close_gates else .03),'8':Gate(0. if close_gates else -.02)},
               write_routers=list(range(3)),read_routers=list(range(3)),audit_mode=False,gradient_audit_mode=False,
               norm=lambda h:h,events=[],queries={},writes=[])
        obj._run_stack=types.MethodType(self.stack,obj)
        obj.inject_audio=types.MethodType(self.inject,obj)
        obj.inject_input_audio=types.MethodType(self.inject_input,obj)
        def route(router,query,name):
            obj.queries[name]=query.copy()
            scores=query.mean(axis=-1,keepdims=True)*np.linspace(-.1,.2,5)+router*.02*np.arange(5)
            weights=np.exp(scores-scores.max(axis=-1,keepdims=True))
            return weights/weights.sum(axis=-1,keepdims=True)
        def write(memory,value,weights):
            obj.writes.append((memory.copy(),value.copy()))
            return memory+value[:,None]*weights.transpose(0,2,1)[...,None]
        def read(memory,weights):
            hidden=(memory*weights.transpose(0,2,1)[...,None]).sum(axis=1)
            obj.events.append(hidden.copy())
            return hidden
        obj._route,obj._write,obj._read=route,write,read
        x=np.random.default_rng(42).normal(size=(2,131,3)).view(Tensor)
        info=(np.ones((2,126,3)),np.full((2,126,3),2.))
        return obj,x,info

    def run_forward(self,obj,x,info):
        obj.last_forward_trace=[];obj.last_loop_lora_trace=();obj.last_injection_trace=()
        obj.events=[];obj.queries={};obj.writes=[];obj.last_reinject_trace=();obj.last_serial_injection_trace=()
        obj.audio_gate.inputs=[]
        for gate in obj.audio_gates.values():gate.inputs=[]
        scope=dict(self=obj,hidden=x.copy(),audio_information=info,batch_size=x.shape[0],query_length=x.shape[1],
                   audio_input_information=obj.audio_linear(x[:,:126]),
                   MEMORY_SLOT_COUNT=5,torch=NS(zeros=lambda shape,**kw:np.zeros(shape),cat=lambda xs,dim:np.concatenate(xs,axis=dim)),
                   record_function=lambda name:nullcontext(),output_hidden_states=False,
                   mask=None,position_ids=None,cache=None,use_cache=False,cache_position=None,
                   position_embeddings=None,output_attentions=False,attentions=[])
        audit.execute(cls_nodes(self.body),scope)
        return scope['hidden'],scope['memory']

    def reference(self,obj,x,info,with_lora=True):
        def stack(h,indices,loop=None):
            for physical in indices:
                for p in obj.layers[physical].ps.values():
                    base=h@p.weight.T+p.bias
                    h=base+2*np.concatenate((h[:,:126]@p.lora_A['audio'][loop].T@p.lora_B['audio'][loop].T,h[:,126:]@p.lora_A['text'][loop].T@p.lora_B['text'][loop].T),axis=1) if with_lora else base
                h=h+.001*(physical+1)
            return h
        def update(memory,value,query,index):
            scores=query.mean(axis=-1,keepdims=True)*np.linspace(-.1,.2,5)+index*.02*np.arange(5)
            w=np.exp(scores-scores.max(axis=-1,keepdims=True));w=w/w.sum(axis=-1,keepdims=True)
            memory=memory+value[:,None]*w.transpose(0,2,1)[...,None]
            return memory,(memory*w.transpose(0,2,1)[...,None]).sum(axis=1)
        projected=x[:,:126]*.7+.1
        def inject(audio,layer,branch):
            u=audio+np.tanh(audio@obj.audio_gate.weight)*projected
            return u+np.tanh(u@obj.audio_gates[str(layer)].weight)*info[branch]
        memory=np.zeros((x.shape[0],5,x.shape[1]-126,3));memory[:,0]=x[:,126:]
        prefix=stack(x.copy(),range(5))
        memory,text=update(memory,prefix[:,126:],prefix[:,126:],0)
        h=np.concatenate((inject(prefix[:,:126],8,1),text),axis=1);inputs=[];outputs=[]
        for loop in range(2):
            inputs.append(h.copy());core=stack(h,range(5,15),loop);outputs.append(core.copy())
            memory,text=update(memory,core[:,126:],h[:,126:],loop+1)
            audio=inject(core[:,:126],4,0) if loop==0 else core[:,:126]
            h=np.concatenate((audio,text),axis=1)
        return stack(h,range(15,20)),memory,inputs,outputs

    def test_actual_hybrid_forward_memory_queries_serial_gates_and_final_audio_bypass(self):
        obj,x,info=self.fixture()
        actual,memory=self.run_forward(obj,x,info)
        expected,expected_memory,inputs,outputs=self.reference(obj,x,info)
        np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(memory,expected_memory,rtol=1e-13,atol=1e-13)
        self.assertEqual(memory.shape,(2,5,5,3))
        self.assertEqual(obj.last_injection_trace,((5,8),(15,4)))
        self.assertEqual(obj.last_serial_injection_trace,tuple(tuple(v) for v in contract.SERIAL_INJECTION_CONTRACT['trace']))
        self.assertEqual([n['physical_index'] for n in obj.last_forward_trace],list(contract.TEXT_MODEL_SCHEDULE))
        self.assertEqual(obj.last_loop_lora_trace,tuple((i,p,0 if i<15 else 1) for i,p in enumerate(contract.TEXT_MODEL_SCHEDULE) if 5<=p<15))
        np.testing.assert_array_equal(obj.writes[0][0][:,0],x[:,126:])
        np.testing.assert_array_equal(obj.writes[0][0][:,1:],0)
        for loop,layer in enumerate((8,4)):
            before=obj.audio_gate.inputs[loop]
            post=before+np.tanh(before@obj.audio_gate.weight)*(x[:,:126]*.7+.1)
            np.testing.assert_array_equal(obj.audio_gates[str(layer)].inputs[0],post)
            self.assertFalse(np.allclose(before,post))
            for kind in ('write','read'):
                np.testing.assert_allclose(obj.queries[f'{kind}_{loop}'],inputs[loop][:,126:],rtol=1e-13,atol=1e-13)
            np.testing.assert_allclose(obj.writes[loop+1][1],outputs[loop][:,126:],rtol=1e-13,atol=1e-13)
        self.assertTrue(all(v.shape[1]==5 for v in obj.queries.values()))
        self.assertEqual(len(obj.events),3)
        self.assertTrue(all(p.loop_index is None for layer in obj.layers for p in layer.ps.values()))

    def test_zero_lora_matches_unadapted_mesh_with_open_ced_gates(self):
        obj,x,info=self.fixture(closed=True)
        actual,memory=self.run_forward(obj,x,info)
        expected,mem,_,_=self.reference(obj,x,info,with_lora=False)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(memory,mem)

    def test_every_forward_rebuilds_memory_and_accepts_new_audio_features(self):
        obj,x,info=self.fixture()
        first,mem=self.run_forward(obj,x,info)
        again,mem_again=self.run_forward(obj,x,info)
        np.testing.assert_array_equal(first,again);np.testing.assert_array_equal(mem,mem_again)
        changed,_=self.run_forward(obj,x,(info[0]*2,info[1]*3))
        self.assertFalse(np.allclose(changed,first))

    def test_middle_stack_rejects_missing_or_wrong_round_and_scope_resets_on_exception(self):
        obj,x,info=self.fixture()
        kwargs=dict(attention_mask=None,position_ids=None,cache=None,use_cache=False,cache_position=None,
                    position_embeddings=None,output_attentions=False,all_attentions=[])
        for indices,start,loop in ((range(5,15),5,None),(range(5,15),5,1),(range(0,5),0,0)):
            with self.subTest(start=start,loop=loop),self.assertRaises(ValueError):
                obj._run_stack(x,indices,start,loop_index=loop,**kwargs)
        obj.last_forward_trace=[];obj.last_loop_lora_trace=()
        module=obj.layers[5].ps[contract.LOOP_LORA_CONTRACT['targets'][0]]
        with patch.object(obj.layers[5], 'get_submodule', side_effect=RuntimeError('synthetic failure')):
            with self.assertRaises(RuntimeError):obj._run_stack(x,range(5,15),5,loop_index=0,**kwargs)
        self.assertIsNone(module.loop_index)


def cls_nodes(nodes):
    return copy.deepcopy(nodes)


if __name__ == '__main__':
    unittest.main()
