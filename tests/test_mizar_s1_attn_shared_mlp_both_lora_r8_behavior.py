"""Execute actual route source with NumPy/fixtures; this is not PyTorch autograd."""
import ast
import copy
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORDER = "84"
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_recursive_5_10_5_ced_small_in84_loop_attn_shared_mlp_both_lora_r8"
sys.path.insert(0, str(ROUTE))
import audit_reference as audit
import contract
from models import recursive_text


def source_functions(file, names, scope):
    module = audit.tree(ROUTE / file)
    audit.execute([audit.find_function(module, name) for name in names], scope)
    return scope


class ContractTests(unittest.TestCase):
    def test_full_source_parity_and_protected_originals(self):
        result = audit.audit()
        self.assertTrue(result["protected_originals"]["all_unchanged"])
        self.assertEqual(len(result["identical_source_files_except_route_names"]), 20)
        self.assertTrue(result["training_parity"]["entire_train_function_ast_equal_after_explicit_audits"])

    def test_exact_lora_recipe_and_actual_gqa_parameter_count(self):
        lora = contract.LOOP_LORA_CONTRACT
        self.assertEqual((lora["rank"], lora["max_lr"]), (8, 6e-4))
        self.assertEqual(lora["alpha_by_family"], dict(attention_shared=16,mlp_shared=8,mlp_audio=8,mlp_text=8))
        self.assertEqual(lora["scale_by_family"], dict(attention_shared=2.,mlp_shared=1.,mlp_audio=1.,mlp_text=1.))
        self.assertEqual(lora["physical_layers_0based"], list(range(5, 15)))
        self.assertEqual(len(lora["targets"]), 7)
        # q/o 576x576, k/v 192x576 (GQA), gate/up/down 1536x576.
        actual = 10 * 2 * 8 * (2 * (576 + 576) + 2 * (576 + 192) + 3 * 3 * (576 + 1536))
        self.assertEqual(actual, 3655680)
        self.assertEqual(actual, lora["new_parameters"])
        self.assertTrue(lora["base_trainable"] and lora["shared_base_between_loops"])
        self.assertFalse(lora["prefix_suffix_lora"])

    def test_non_lora_recipe_is_exactly_inherited(self):
        result = audit.audit()
        self.assertTrue(result['common_recipe_equal'])
        self.assertTrue(result['backbone_parity']['entire_backbone_ast_equal_after_metadata'])
        self.assertEqual(contract.LOOP_LORA_CONTRACT['parameters_by_family'],
                         dict(attention_shared=614400, mlp_shared=1013760, mlp_audio=1013760, mlp_text=1013760))
        self.assertEqual(contract.LOOP_LORA_CONTRACT['matrix_count'], 520)
        self.assertEqual(contract.TEXT_MODEL_CONTRACT, 'logical_30_physical_20_5_10_5_loops_2_no_mesh')

    def test_strict_loader_never_installs_lora_on_failed_pretrained_load(self):
        for failure in (None, "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            events = []
            core = NS(enable_layer_injection=lambda: events.append("layers"),
                      enable_audio_reinjection=lambda: events.append("input"),
                      enable_loop_lora=lambda: events.append("lora"))
            model = NS(model=core, generation_config=NS(use_cache=True))
            cls = Mock()
            cls.from_pretrained.return_value = (model, {} if failure is None else {failure: ["bad"]})
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "config.json").write_text("{}")
                with patch.object(recursive_text, "_import_recursive_model", return_value=(cls, lambda m: {})), \
                     patch.object(recursive_text, "validate_loaded_model", side_effect=lambda *a: events.append("validate")), \
                     patch.object(recursive_text, "validate_serial_injection"), patch.object(recursive_text, "validate_loop_lora"):
                    if failure:
                        with self.assertRaises(ValueError):
                            recursive_text.load_recursive_text_model(directory)
                        self.assertEqual(events, [])
                    else:
                        recursive_text.load_recursive_text_model(directory)
                        self.assertEqual(events, ["validate", "layers", "input", "lora"])
                        self.assertFalse(model.generation_config.use_cache)
                    self.assertTrue(cls.from_pretrained.call_args.kwargs["local_files_only"])

    def test_resume_rejects_other_route_and_changed_lora_configuration(self):
        prepared = dict(rows=1477492, source_rows=contract.SOURCE_COUNTS,
                        rows_canonical_sha256="r", mapping_sha256="m")
        recipe = contract.semantic_contract(prepared, "text", "ced")
        fn = audit.find_function(audit.tree(ROUTE / "train.py"), "train")
        guard = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and "saved.get('schema_version')" in ast.unparse(n.test))
        valid = dict(schema_version=1, route_variant=contract.ROUTE, mode="formal", contract_sha256=contract.digest(recipe))
        scope = dict(saved=valid, config={"contract_sha256": valid["contract_sha256"]}, ROUTE=contract.ROUTE)
        audit.execute([guard], scope)
        for key, wrong in (("route_variant", contract.ROUTE.replace("layer8_4", "other").replace("layer4_8", "other")),
                           ("route_variant", contract.ROUTE.replace("_loop_attn_shared_mlp_shared_modality_lora_r8", "_loop_lora_r8")),
                           ("contract_sha256", "changed"), ("mode", "smoke")):
            with self.subTest(key=key, wrong=wrong), self.assertRaises(ValueError):
                audit.execute([guard], dict(scope, saved=dict(valid, **{key: wrong})))
        for key, wrong in (("alpha_by_family", {"attention_shared":16,"mlp_shared":16,"mlp_audio":8,"mlp_text":8}), ("max_lr", 3e-4), ("rank", 4), ("physical_layers_0based", list(range(20))), ("audio_positions", [0,127]),
                           ("text_start",127), ("attention_sharing","shared_modality"), ("mlp_sharing","shared")):
            changed = copy.deepcopy(recipe)
            changed["loop_lora"][key] = wrong
            self.assertNotEqual(contract.digest(recipe), contract.digest(changed))

    def test_optimizer_all_lora_families_are_6e4_and_keep_original_schedule(self):
        class Parameter:
            requires_grad = True
            def __init__(self, ndim): self.ndim = ndim
            def numel(self): return 12
        prefixes = ["audio_encoder.base.mapper", "caption_decoder.lm.model.layers.0",
                    "audio_encoder.base.layer_mlps.4", "audio_encoder.base.layer_mlps.8",
                    "caption_decoder.lm.model.audio_gates.4", "caption_decoder.lm.model.audio_gates.8",
                    "caption_decoder.lm.model.audio_linear", "caption_decoder.lm.model.audio_gate"]
        pairs = [(p + suffix, Parameter(ndim)) for p in prefixes for suffix, ndim in ((".weight", 2), (".bias", 1))]
        lora_names = [f'caption_decoder.lm.model.layers.{layer}.{target}.{kind}.{branch}.{loop}'
                      for layer in range(5,15) for target in contract.LOOP_LORA_CONTRACT['targets']
                      for kind in ('lora_A','lora_B')
                      for branch in (('shared',) if target.startswith('self_attn.') else ('shared','audio','text'))
                      for loop in (0,1)]
        pairs += [(name, Parameter(2)) for name in lora_names]
        class Optimizer:
            def __init__(self, groups, **kw): self.param_groups, self.kw = groups, kw
        fake = NS(optim=NS(AdamW=Optimizer, lr_scheduler=NS(LambdaLR=lambda opt, fn: fn)))
        scope = source_functions("optim.py", ["parameter_role", "build_optimizer"], dict(torch=fake, lr_factor=contract.lr_factor))
        optimizer, schedule, groups = scope["build_optimizer"](NS(named_parameters=lambda: pairs), 17313)
        lora_groups = [g for g in groups if "lora" in g["role"]]
        self.assertEqual(len(lora_groups), 16)
        self.assertEqual(sum(len(g["names"]) for g in lora_groups), 520)
        for group in groups:
            self.assertEqual(group["base_lr"], 3e-4 if group["role"] == "default" else 6e-4)
            if "lora" in group["role"]: self.assertEqual(group["weight_decay"], .01)
        self.assertEqual(optimizer.kw["betas"], (.9, .999))
        self.assertEqual(optimizer.kw["eps"], 1e-8)
        for step in range(17314): self.assertEqual(schedule(step), contract.lr_factor(step, 17313))
        for wrong in (lora_names[0].replace("layers.5.", "layers.4."),
                      lora_names[0].replace("lora_A.shared.0", "lora_A.shared.2"),
                      lora_names[0].replace("q_proj", "bad_proj"),
                      lora_names[0].replace(".shared.", ".audio."),
                      lora_names[-1].replace(".text.", ".bad.")):
            with self.assertRaises(ValueError): scope["parameter_role"](wrong)


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = audit.tree(ROUTE / "models/loop_lora.py")
        linear = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "LoopLoRALinear")
        forward = next(copy.deepcopy(n) for n in linear.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        scope = dict(F=NS(linear=lambda x, w, bias=None: x @ w.T + (0 if bias is None else bias)),
                     torch=NS(cat=lambda xs, dim: np.concatenate(xs,axis=dim)))
        audit.execute([forward], scope)
        cls.projection_forward = staticmethod(scope["forward"])
        scope = source_functions("models/loop_lora.py", ["projections", "call_with_loop_lora"],
                                 dict(TARGETS=contract.LOOP_LORA_CONTRACT["targets"]))
        cls.call_scope = staticmethod(scope["call_with_loop_lora"])
        scope = source_functions("models/recursive_layer_inject.py", ["inject_input_audio", "inject_audio"],
                                 dict(torch=NS(tanh=np.tanh, cat=lambda seq, dim: np.concatenate(seq, axis=dim))))
        cls.input_inject, cls.layer_inject = staticmethod(scope["inject_input_audio"]), staticmethod(scope["inject_audio"])
        model = next(n for n in audit.tree(ROUTE / "models/recursive_layer_inject.py").body
                     if isinstance(n, ast.ClassDef) and n.name == "RecursiveLlama5_10_5Model")
        fn = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        cls.loop = next(n for n in fn.body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple))

    def projection(self, seed=0, closed=False, mode='shared'):
        rng = np.random.default_rng(seed)
        branches = ('shared',) if mode == 'shared' else ('shared','audio','text')
        return NS(weight=np.eye(3)*.99, bias=np.zeros(3), scaling=2. if mode=="shared" else 1., loop_index=None,
                  mode=mode, audio_tokens=126, forward_counts=[0,0], last_token_shapes=[None,None],
                  branch_counts=[{m:0 for m in branches} for _ in range(2)],
                  lora_A={m:[rng.normal(size=(8,3))*.1 for _ in range(2)] for m in branches},
                  lora_B={m:[rng.normal(size=(3,8))*(0. if closed else .04) for _ in range(2)] for m in branches})

    def test_nonzero_adapters_scale_two_and_select_distinct_rounds(self):
        p, x = self.projection(), np.ones((2, 131, 3))
        outputs = []
        for loop in (0, 1):
            p.loop_index = loop
            y = self.projection_forward(p, x)
            np.testing.assert_allclose(y, x @ p.weight.T + 2*(x @ p.lora_A["shared"][loop].T @ p.lora_B["shared"][loop].T))
            outputs.append(y)
        self.assertFalse(np.allclose(*outputs))
        self.assertFalse(np.allclose(outputs[0][:,126:], x[:,126:] @ p.weight.T))
        p.loop_index = None
        with self.assertRaises(RuntimeError): self.projection_forward(p, x)

    def test_initial_B_zero_A_zero_task_gradient_and_independent_branch_gradient(self):
        p, x = self.projection(closed=True), np.random.default_rng(5).normal(size=(2, 131, 3))
        p.loop_index = 0
        def loss(): return (self.projection_forward(p, x)**2).sum()
        def diff(matrix, index):
            old, eps = matrix[index], 1e-5
            matrix[index] = old + eps; plus = loss()
            matrix[index] = old - eps; minus = loss()
            matrix[index] = old
            return (plus-minus)/(2*eps)
        self.assertEqual(diff(p.lora_A["shared"][0], (0,0)), 0.)
        self.assertNotEqual(diff(p.lora_B["shared"][0], (0,0)), 0.)
        self.assertEqual(diff(p.lora_B["shared"][1], (0,0)), 0.)
        self.assertNotEqual(diff(p.weight, (0,0)), 0.)
        p.lora_B["shared"][0][0,0] = .1
        self.assertNotEqual(diff(p.lora_A["shared"][0], (0,0)), 0.)

    def test_scope_resets_selection_on_success_exception_and_backward_independent_tensors(self):
        modules = {name: self.projection(i) for i, name in enumerate(contract.LOOP_LORA_CONTRACT["targets"])}
        layer = NS(get_submodule=lambda name: modules[name])
        selected = []
        def callback(layer, x):
            selected.append([(p.lora_A["shared"][p.loop_index], p.lora_B["shared"][p.loop_index]) for p in modules.values()])
            return x + 1
        for loop in (0, 1):
            self.assertEqual(self.call_scope(layer, loop, callback, 2), 3)
            self.assertTrue(all(p.loop_index is None for p in modules.values()))
        self.assertIsNot(selected[0][0][0], selected[1][0][0])
        def failure(*a): raise ValueError("synthetic decoder failure")
        with self.assertRaises(ValueError): self.call_scope(layer, 0, failure)
        self.assertTrue(all(p.loop_index is None for p in modules.values()))

    def fixture(self, closed=False):
        forward = self.projection_forward
        targets = contract.LOOP_LORA_CONTRACT["targets"]
        class Layer:
            def __init__(slf, index):
                slf.index = index
                slf.ps = {name: self.projection(index*10+i, closed, "shared" if name.startswith("self_attn.") else "shared_modality") for i, name in enumerate(targets)} if 5 <= index < 15 else {}
            def get_submodule(slf, name): return slf.ps[name]
            def __call__(slf, h):
                if slf.ps:
                    for p in slf.ps.values(): h = forward(p, h)
                return h + .001*(slf.index+1)
        class Gate:
            def __init__(slf, value): slf.weight, slf.inputs = np.full((3,1),value), []
            def __call__(slf, h): slf.inputs.append(h.copy()); return h @ slf.weight
        obj = NS(layers=[Layer(i) for i in range(20)], logical_to_physical=contract.TEXT_MODEL_SCHEDULE,
                 layer_inject_indices=(5,15), audio_reinject_tokens=126, layer_inject_tokens=126,
                 collect_injection_stats=False, collect_reinject_stats=False,
                 last_reinject_trace=(), last_injection_trace=(), last_serial_injection_trace=(), last_loop_lora_trace=(),
                 audio_gate=Gate(.05), audio_gates={"4":Gate(.03), "8":Gate(-.02)})
        obj.inject_input_audio = types.MethodType(self.input_inject, obj)
        obj.inject_audio = types.MethodType(self.layer_inject, obj)
        x = np.random.default_rng(42).normal(size=(2,131,3))
        return obj, x, (np.ones((2,126,3)), np.full((2,126,3),2.))

    def test_actual_logical_loop_nonzero_lora_serial_order_and_post_input_gate(self):
        obj, x, information = self.fixture()
        scope = dict(self=obj, hidden_states=x.copy(), audio_information=information, audio_input_information=x[:,:126]*.7,
                     output_hidden_states=False, output_attentions=False, cache=None, causal_mask=None,
                     position_ids=None, use_cache=False, cache_position=None, position_embeddings=None,
                     call_with_loop_lora=self.call_scope, _call_decoder_layer=lambda layer, h, **kw:(layer(h),))
        audit.execute([self.loop], scope)
        actual = scope["hidden_states"]
        expected, posts = x.copy(), []
        for logical, physical in enumerate(contract.TEXT_MODEL_SCHEDULE):
            if logical in (5,15):
                branch = ([8,4] if ORDER == "84" else [4,8])[(5,15).index(logical)]
                audio = expected[:,:126]
                updated = audio + np.tanh(audio @ obj.audio_gate.weight) * x[:,:126]*.7
                posts.append(updated.copy())
                final = updated + np.tanh(updated @ obj.audio_gates[str(branch)].weight) * information[0 if branch==4 else 1]
                expected = np.concatenate((final, expected[:,126:]), axis=1)
            if 5 <= physical < 15:
                loop = 0 if logical < 15 else 1
                for p in obj.layers[physical].ps.values():
                    if p.mode == 'shared':
                        delta = expected @ p.lora_A['shared'][loop].T @ p.lora_B['shared'][loop].T
                    else:
                        delta = np.concatenate((expected[:,:126] @ p.lora_A['audio'][loop].T @ p.lora_B['audio'][loop].T,
                                                expected[:,126:] @ p.lora_A['text'][loop].T @ p.lora_B['text'][loop].T),axis=1)
                    if p.mode != 'shared':
                        delta = expected @ p.lora_A['shared'][loop].T @ p.lora_B['shared'][loop].T + delta
                    expected = expected @ p.weight.T + (2. if p.mode == 'shared' else 1.)*delta
            expected = expected + .001*(physical+1)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        self.assertEqual([list(x) for x in obj.last_serial_injection_trace], contract.SERIAL_INJECTION_CONTRACT["trace"])
        self.assertEqual(obj.last_loop_lora_trace, tuple((i,p,0 if i<15 else 1) for i,p in enumerate(contract.TEXT_MODEL_SCHEDULE) if 5<=p<15))
        for branch, post in zip(([8,4] if ORDER=="84" else [4,8]), posts):
            np.testing.assert_allclose(obj.audio_gates[str(branch)].inputs[0], post, rtol=1e-13, atol=1e-13)
        self.assertTrue(all(p.loop_index is None for layer in obj.layers for p in layer.ps.values()))

    def test_each_adapter_changes_only_its_direct_modality_and_round_including_separator(self):
        x = np.random.default_rng(6).normal(size=(2, 131, 3))
        for loop in (0, 1):
            for modality in ('audio', 'text'):
                with self.subTest(loop=loop, modality=modality):
                    p = self.projection(mode='shared_modality')
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
                    original = self.projection(mode='shared_modality')
                    original.loop_index = 1 - loop
                    np.testing.assert_array_equal(untouched_round, self.projection_forward(original, x))

    def test_modality_only_loss_routes_finite_difference_gradients_to_own_adapter(self):
        x = np.random.default_rng(7).normal(size=(1, 130, 3))
        for loop in (0, 1):
            for modality, tokens in (('audio', slice(None, 126)), ('text', slice(126, None))):
                p = self.projection(mode='shared_modality')
                p.loop_index = loop
                for branch in ('shared', 'audio', 'text'):
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
                                if other_loop == loop and branch in ('shared', modality):
                                    self.assertNotEqual(gradient, 0.)
                                else:
                                    self.assertEqual(gradient, 0.)

    def test_projection_rejects_missing_text_or_flattened_token_axis(self):
        p = self.projection(mode='shared_modality')
        p.loop_index = 0
        for shape in ((126, 3), (1, 126, 3), (1, 125, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.projection_forward(p, np.ones(shape))
        self.assertEqual(p.forward_counts, [0, 0])

    def test_shared_attention_uses_same_matrices_for_both_regions_and_adds_gradients(self):
        p = self.projection(mode='shared')
        x = np.random.default_rng(9).normal(size=(2, 131, 3))
        for loop in (0, 1):
            p.loop_index = loop
            before = self.projection_forward(p, x)
            matrix = p.lora_B['shared'][loop]
            original = matrix.copy()
            matrix += .2
            after = self.projection_forward(p, x)
            for tokens in (slice(None, 126), slice(126, None)):
                self.assertFalse(np.allclose(before[:, tokens], after[:, tokens]))
            matrix[:] = original
            def gradient(tokens):
                old, eps = matrix[0, 0], 1e-5
                matrix[0, 0] = old + eps
                plus = np.square(self.projection_forward(p, x)[:, tokens]).sum()
                matrix[0, 0] = old - eps
                minus = np.square(self.projection_forward(p, x)[:, tokens]).sum()
                matrix[0, 0] = old
                return (plus - minus) / (2 * eps)
            audio, text, full = [gradient(tokens) for tokens in (slice(None, 126), slice(126, None), slice(None))]
            self.assertNotEqual(audio, 0.)
            self.assertNotEqual(text, 0.)
            self.assertAlmostEqual(full, audio + text, places=7)

    def test_mlp_parallel_shared_and_private_scale_one_base_once(self):
        x = np.random.default_rng(31).normal(size=(2, 131, 3))
        for loop in (0, 1):
            p = self.projection(mode='shared_modality')
            p.loop_index = loop
            linear = self.projection_forward.__globals__['F']
            original, base_calls = linear.linear, []
            def counted(inputs, weight, bias=None):
                if weight is p.weight:
                    base_calls.append(inputs.shape)
                return original(inputs, weight, bias)
            with patch.object(linear, 'linear', counted):
                actual = self.projection_forward(p, x)
            shared = x @ p.lora_A['shared'][loop].T @ p.lora_B['shared'][loop].T
            private = np.concatenate([tokens @ p.lora_A[branch][loop].T @ p.lora_B[branch][loop].T
                                      for branch, tokens in (('audio', x[:, :126]), ('text', x[:, 126:]))], axis=1)
            expected = x @ p.weight.T + p.bias + shared + private
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
            self.assertEqual(p.scaling, 1.)
            self.assertEqual(base_calls, [x.shape])
            self.assertEqual(p.branch_counts[loop], dict(shared=1, audio=1, text=1))
            self.assertEqual(p.branch_counts[1-loop], dict(shared=0, audio=0, text=0))

    def test_shared_mlp_affects_both_regions_only_selected_round(self):
        x = np.random.default_rng(32).normal(size=(2, 131, 3))
        for loop in (0, 1):
            p = self.projection(mode='shared_modality')
            p.loop_index = loop
            before = self.projection_forward(p, x)
            p.lora_B['shared'][loop] += .3
            after = self.projection_forward(p, x)
            for tokens in (slice(None,126), slice(126,None)):
                self.assertFalse(np.allclose(before[:,tokens], after[:,tokens]))
            p.loop_index = 1-loop
            reference = self.projection(mode='shared_modality')
            reference.loop_index = 1-loop
            np.testing.assert_array_equal(self.projection_forward(p,x), self.projection_forward(reference,x))

    def test_shared_mlp_finite_difference_receives_sum_of_both_regions(self):
        x = np.random.default_rng(33).normal(size=(2, 131, 3))
        p = self.projection(mode='shared_modality')
        for loop in (0,1):
            p.loop_index = loop
            matrix = p.lora_B['shared'][loop]
            def gradient(tokens):
                old, eps = matrix[0,0], 1e-5
                matrix[0,0] = old+eps
                plus = np.square(self.projection_forward(p,x)[:,tokens]).sum()
                matrix[0,0] = old-eps
                minus = np.square(self.projection_forward(p,x)[:,tokens]).sum()
                matrix[0,0] = old
                return (plus-minus)/(2*eps)
            audio,text,full = [gradient(tokens) for tokens in (slice(None,126),slice(126,None),slice(None))]
            self.assertNotEqual(audio,0.)
            self.assertNotEqual(text,0.)
            self.assertAlmostEqual(full,audio+text,places=7)



class GradientAuditTests(unittest.TestCase):
    def fixture(self, opened=False):
        class Tensor(np.ndarray):
            def detach(self): return self
            def ne(self, value): return self != value
            def numel(self): return self.size
            def cpu(self): return self
            def clone(self): return self.copy()
            def abs(self): return abs(self)
        prefixes = {"map":"audio_encoder.base.mapper", "default":"caption_decoder.lm.model.layers.0",
                    "layer4_mlp":"audio_encoder.base.layer_mlps.4", "layer8_mlp":"audio_encoder.base.layer_mlps.8",
                    "layer4_gate":"caption_decoder.lm.model.audio_gates.4", "layer8_gate":"caption_decoder.lm.model.audio_gates.8",
                    "reinject_linear":"caption_decoder.lm.model.audio_linear", "reinject_gate":"caption_decoder.lm.model.audio_gate"}
        for loop in (1,2):
            for family, target, branch in (('attention_shared','self_attn.q_proj','shared'), ('mlp_shared','mlp.gate_proj','shared'),
                                           ('mlp_audio','mlp.gate_proj','audio'), ('mlp_text','mlp.gate_proj','text')):
                for kind in ('A','B'):
                    prefixes[f'loop{loop}_{family}_lora_{kind}'] = f'caption_decoder.lm.model.layers.5.{target}.lora_{kind}.{branch}.{loop-1}'
        roles, pairs = {}, []
        for role, prefix in prefixes.items():
            closed = role.endswith("gate") or role.endswith("lora_B")
            value = np.full((1,2), 0. if closed and not opened else 1.).view(Tensor)
            value.requires_grad = True
            zero_grad = (role in {"layer4_mlp","layer8_mlp","reinject_linear"} or role.endswith("lora_A")) and not opened
            value.grad = np.full((1,2),0. if zero_grad else 1.).view(Tensor)
            roles[role] = value
            pairs.append((prefix if "lora_" in role else prefix+".weight",value))
        gate=lambda role: NS(parameters=lambda:[roles[role]])
        shared=NS(mode='shared',lora_B={'shared':[roles[f'loop{i}_attention_shared_lora_B'] for i in (1,2)]})
        modality=NS(mode='shared_modality',lora_B={m:[roles[f'loop{i}_mlp_{m}_lora_B'] for i in (1,2)] for m in ('shared','audio','text')})
        core=NS(audio_gate=gate('reinject_gate'), audio_gates={str(k):gate(f'layer{k}_gate') for k in (4,8)},
                modules=lambda:[shared,modality], last_reinject_trace=(5,15),
                last_injection_trace=((5,8),(15,4)),last_serial_injection_trace=contract.SERIAL_INJECTION_CONTRACT['trace'],
                last_loop_lora_trace=[(i,p,0 if i<15 else 1) for i,p in enumerate(contract.TEXT_MODEL_SCHEDULE) if 5<=p<15],
                last_batch_size=1,last_sequence_length=683)
        core.layers=[]
        for index in range(20):
            ps={name:NS(mode='shared' if name.startswith('self_attn.') else 'shared_modality',loop_index=None,
                        forward_counts=[1,1],last_token_shapes=[(1,126,557)]*2,
                        rank=8,alpha=16 if name.startswith('self_attn.') else 8,scaling=2. if name.startswith('self_attn.') else 1.,
                        lora_A={m:None for m in (('shared',) if name.startswith('self_attn.') else ('shared','audio','text'))},
                        branch_counts=[{m:1 for m in (('shared',) if name.startswith('self_attn.') else ('shared','audio','text'))} for _ in range(2)]) for name in contract.LOOP_LORA_CONTRACT['targets']}
            core.layers.append(NS(ps=ps,get_submodule=lambda name,ps=ps:ps[name]))
        model=NS(caption_decoder=NS(lm=NS(model=core)),named_parameters=lambda:pairs)
        scope=dict(torch=NS(isfinite=np.isfinite,equal=np.array_equal), TEXT_MODEL_SCHEDULE=contract.TEXT_MODEL_SCHEDULE,
                   SERIAL_INJECTION_CONTRACT=contract.SERIAL_INJECTION_CONTRACT)
        source_functions("optim.py",["parameter_role"],scope)
        scope.update(TARGETS=contract.LOOP_LORA_CONTRACT['targets'],ATTENTION_TARGETS=contract.LOOP_LORA_CONTRACT['attention_targets'],
                     MIDDLE_LAYERS=tuple(range(5,15)),LOOPS=2,AUDIO_TOKENS=126,RANK=8,ATTENTION_ALPHA=16,MLP_ALPHA=8)
        source_functions('models/loop_lora.py',['projections','validate_loop_lora_execution','reset_loop_lora_trace'],scope)
        source=audit.tree(ROUTE/"train.py")
        for n in source.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {"TRAINABLE_ROLES","NEW_ROLES"} for t in n.targets):
                audit.execute([n],scope)
        source_functions("train.py",["first_gradient_audit","role_snapshots","weights_changed"],scope)
        return model,roles,scope

    def test_closed_B_expects_zero_A_but_requires_nonzero_B_in_both_loops(self):
        model,roles,scope=self.fixture()
        result=scope["first_gradient_audit"](model)
        self.assertTrue(result["passed"],result)
        self.assertEqual(result["closed_lora_B"],{f"loop{i}_{f}":True for i in (1,2) for f in ("attention_shared","mlp_shared","mlp_audio","mlp_text")})
        self.assertIn("loop1_mlp_audio_lora_A",result["zero_task_gradient_expected"])
        for role in (f"loop{i}_{f}_lora_B" for i in (1,2) for f in ("attention_shared","mlp_shared","mlp_audio","mlp_text")):
            roles[role].grad.fill(0)
            self.assertFalse(scope["first_gradient_audit"](model)["passed"])
            roles[role].grad.fill(1)

    def test_opened_B_requires_task_gradient_in_A_and_each_audio_upstream(self):
        model,roles,scope=self.fixture(opened=True)
        self.assertTrue(scope["first_gradient_audit"](model)["passed"])
        for role in ([f"loop{i}_{f}_lora_A" for i in (1,2) for f in ("attention_shared","mlp_shared","mlp_audio","mlp_text")] + ["layer4_mlp","layer8_mlp","reinject_linear"]):
            roles[role].grad.fill(0)
            self.assertFalse(scope["first_gradient_audit"](model)["passed"],role)
            roles[role].grad.fill(1)

    def test_missing_nonfinite_gradients_and_wrong_loop_trace_fail(self):
        model,roles,scope=self.fixture(opened=True)
        for bad in (None,np.full((1,2),np.nan)):
            old=roles["loop2_mlp_audio_lora_A"].grad
            roles["loop2_mlp_audio_lora_A"].grad=bad.view(type(old)) if isinstance(bad,np.ndarray) else bad
            self.assertFalse(scope["first_gradient_audit"](model)["passed"])
            roles["loop2_mlp_audio_lora_A"].grad=old
        model.caption_decoder.lm.model.last_loop_lora_trace=[]
        self.assertFalse(scope["first_gradient_audit"](model)["passed"])

    def test_all_twenty_four_roles_have_separate_snapshot_and_change_detection(self):
        model,roles,scope=self.fixture()
        snapshots=scope["role_snapshots"](model)
        self.assertEqual(set(snapshots),set(roles))
        self.assertTrue(all(not item["changed"] for item in scope["weights_changed"](model,snapshots).values()))
        roles["loop2_mlp_audio_lora_B"] += .5
        changed=scope["weights_changed"](model,snapshots)
        self.assertTrue(changed["loop2_mlp_audio_lora_B"]["changed"])
        self.assertFalse(changed["loop1_mlp_audio_lora_B"]["changed"])

    def test_call_audit_rejects_wrong_mode_missing_duplicate_boundary_and_stale_scope(self):
        model, roles, scope = self.fixture(opened=True)
        core = model.caption_decoder.lm.model
        validate = scope['validate_loop_lora_execution']
        self.assertEqual(validate(core)['adapter_calls'], 260)
        for index in range(5, 15):
            for target in contract.LOOP_LORA_CONTRACT['targets']:
                module = core.layers[index].ps[target]
                for field, bad in (('mode', 'bad'), ('alpha',99), ('scaling',0.), ('branch_counts',[{'shared':0}]*2), ('forward_counts', [0, 1]), ('forward_counts', [1, 2]),
                                   ('last_token_shapes', [(1, 127, 556), (1, 126, 557)]), ('loop_index', 1)):
                    old = getattr(module, field)
                    setattr(module, field, bad)
                    with self.subTest(index=index, target=target, field=field), self.assertRaises(ValueError):
                        validate(core)
                    self.assertFalse(scope['first_gradient_audit'](model)['passed'])
                    setattr(module, field, old)
        scope['reset_loop_lora_trace'](core)
        with self.assertRaises(ValueError): validate(core)
        core.layers[5].ps['self_attn.q_proj'].loop_index = 0
        with self.assertRaises(RuntimeError): scope['reset_loop_lora_trace'](core)


if __name__ == "__main__":
    unittest.main()
