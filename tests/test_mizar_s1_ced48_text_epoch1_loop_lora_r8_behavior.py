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
ORDER = "48"
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced48_text_epoch1_loop_lora_r8"
sys.path.insert(0, str(ROUTE))
import audit_reference as audit
import contract
from models import recursive_text


def source_functions(file, names, scope):
    module = audit.tree(ROUTE / file)
    audit.execute([audit.find_function(module, name) for name in names], scope)
    return scope


class ContractTests(unittest.TestCase):

    def test_exact_lora_recipe_and_actual_gqa_parameter_count(self):
        lora = contract.LOOP_LORA_CONTRACT
        self.assertEqual((lora["rank"], lora["alpha"], lora["scale"], lora["max_lr"]), (8, 16, 2., 6e-4))
        self.assertEqual(lora["physical_layers_0based"], list(range(5, 15)))
        self.assertEqual(len(lora["targets"]), 7)
        # q/o 576x576, k/v 192x576 (GQA), gate/up/down 1536x576.
        actual = 10 * 2 * 8 * (2 * (576 + 576) + 2 * (576 + 192) + 3 * (576 + 1536))
        self.assertEqual(actual, 1628160)
        self.assertEqual(actual, lora["new_parameters"])
        self.assertTrue(lora["base_trainable"] and lora["shared_base_between_loops"])
        self.assertFalse(lora["prefix_suffix_lora"])



    def test_resume_rejects_other_route_and_changed_lora_configuration(self):
        prepared = dict(rows=1477492, source_rows=contract.SOURCE_COUNTS,
                        rows_canonical_sha256="r", mapping_sha256="m")
        recipe = contract.semantic_contract(prepared, "text", "ced")
        fn = audit.find_function(audit.tree(ROUTE / "train.py"), "train")
        guard = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and "saved.get('schema_version')" in ast.unparse(n.test))
        valid = dict(schema_version=1, route_variant=contract.ROUTE, mode="formal", contract_sha256=contract.digest(recipe))
        scope = dict(saved=valid, config={"contract_sha256": valid["contract_sha256"]}, ROUTE=contract.ROUTE)
        audit.execute([guard], scope)
        for key, wrong in (("route_variant", "ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8_v1"),
                           ("route_variant", contract.ROUTE.replace("_loop_lora_r8", "")),
                           ("contract_sha256", "changed"), ("mode", "smoke")):
            with self.subTest(key=key, wrong=wrong), self.assertRaises(ValueError):
                audit.execute([guard], dict(scope, saved=dict(valid, **{key: wrong})))
        for key, wrong in (("alpha", 8), ("max_lr", 3e-4), ("rank", 4), ("physical_layers_0based", list(range(20)))):
            changed = copy.deepcopy(recipe)
            changed["loop_lora"][key] = wrong
            self.assertNotEqual(contract.digest(recipe), contract.digest(changed))

    def test_optimizer_both_lora_sets_are_6e4_all_groups_keep_original_schedule(self):
        class Parameter:
            requires_grad = True
            def __init__(self, ndim): self.ndim = ndim
            def numel(self): return 12
        prefixes = ["audio_encoder.base.mapper", "caption_decoder.lm.model.layers.0",
                    "audio_encoder.base.layer_mlps.4", "audio_encoder.base.layer_mlps.8",
                    "caption_decoder.lm.model.audio_gates.4", "caption_decoder.lm.model.audio_gates.8",
                    "caption_decoder.lm.model.audio_linear", "caption_decoder.lm.model.audio_gate"]
        pairs = [(p + suffix, Parameter(ndim)) for p in prefixes for suffix, ndim in ((".weight", 2), (".bias", 1))]
        lora_names = [f"caption_decoder.lm.model.layers.{layer}.{target}.{kind}.{loop}"
                      for layer in range(5, 15) for target in contract.LOOP_LORA_CONTRACT["targets"]
                      for kind in ("lora_A", "lora_B") for loop in (0, 1)]
        pairs += [(name, Parameter(2)) for name in lora_names]
        class Optimizer:
            def __init__(self, groups, **kw): self.param_groups, self.kw = groups, kw
        fake = NS(optim=NS(AdamW=Optimizer, lr_scheduler=NS(LambdaLR=lambda opt, fn: fn)))
        scope = source_functions("optim.py", ["parameter_role", "build_optimizer"], dict(torch=fake, lr_factor=contract.lr_factor))
        optimizer, schedule, groups = scope["build_optimizer"](NS(named_parameters=lambda: pairs), 17313)
        lora_groups = [g for g in groups if "lora" in g["role"]]
        self.assertEqual(len(lora_groups), 4)
        self.assertEqual(sum(len(g["names"]) for g in lora_groups), 280)
        for group in groups:
            self.assertEqual(group["base_lr"], 3e-4 if group["role"] == "default" else 6e-4)
            if "lora" in group["role"]: self.assertEqual(group["weight_decay"], .01)
        self.assertEqual(optimizer.kw["betas"], (.9, .999))
        self.assertEqual(optimizer.kw["eps"], 1e-8)
        for step in range(17314): self.assertEqual(schedule(step), contract.lr_factor(step, 17313))
        for wrong in (lora_names[0].replace("layers.5.", "layers.4."),
                      lora_names[0].replace("lora_A.0", "lora_A.2"),
                      lora_names[0].replace("q_proj", "bad_proj")):
            with self.assertRaises(ValueError): scope["parameter_role"](wrong)


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = audit.tree(ROUTE / "models/loop_lora.py")
        linear = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "LoopLoRALinear")
        forward = next(copy.deepcopy(n) for n in linear.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        scope = dict(F=NS(linear=lambda x, w, bias=None: x @ w.T + (0 if bias is None else bias)))
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

    def projection(self, seed=0, closed=False):
        rng = np.random.default_rng(seed)
        return NS(weight=np.eye(3)*.99, bias=np.zeros(3), scaling=2., loop_index=None,
                  lora_A=[rng.normal(size=(8, 3))*.1 for _ in range(2)],
                  lora_B=[rng.normal(size=(3, 8))*(0. if closed else .04) for _ in range(2)])

    def test_nonzero_adapters_scale_two_and_select_distinct_rounds(self):
        p, x = self.projection(), np.ones((2, 131, 3))
        outputs = []
        for loop in (0, 1):
            p.loop_index = loop
            y = self.projection_forward(p, x)
            np.testing.assert_allclose(y, x @ p.weight.T + 2*(x @ p.lora_A[loop].T @ p.lora_B[loop].T))
            outputs.append(y)
        self.assertFalse(np.allclose(*outputs))
        self.assertFalse(np.allclose(outputs[0][:,126:], x[:,126:] @ p.weight.T))
        p.loop_index = None
        with self.assertRaises(RuntimeError): self.projection_forward(p, x)

    def test_initial_B_zero_A_zero_task_gradient_and_independent_branch_gradient(self):
        p, x = self.projection(closed=True), np.random.default_rng(5).normal(size=(2, 3))
        p.loop_index = 0
        def loss(): return (self.projection_forward(p, x)**2).sum()
        def diff(matrix, index):
            old, eps = matrix[index], 1e-5
            matrix[index] = old + eps; plus = loss()
            matrix[index] = old - eps; minus = loss()
            matrix[index] = old
            return (plus-minus)/(2*eps)
        self.assertEqual(diff(p.lora_A[0], (0,0)), 0.)
        self.assertNotEqual(diff(p.lora_B[0], (0,0)), 0.)
        self.assertEqual(diff(p.lora_B[1], (0,0)), 0.)
        self.assertNotEqual(diff(p.weight, (0,0)), 0.)
        p.lora_B[0][0,0] = .1
        self.assertNotEqual(diff(p.lora_A[0], (0,0)), 0.)

    def test_scope_resets_selection_on_success_exception_and_backward_independent_tensors(self):
        modules = {name: self.projection(i) for i, name in enumerate(contract.LOOP_LORA_CONTRACT["targets"])}
        layer = NS(get_submodule=lambda name: modules[name])
        selected = []
        def callback(layer, x):
            selected.append([(p.lora_A[p.loop_index], p.lora_B[p.loop_index]) for p in modules.values()])
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
                slf.ps = {name: self.projection(index*10+i, closed) for i, name in enumerate(targets)} if 5 <= index < 15 else {}
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
                    expected = expected @ p.weight.T + 2*(expected @ p.lora_A[loop].T @ p.lora_B[loop].T)
            expected = expected + .001*(physical+1)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        self.assertEqual([list(x) for x in obj.last_serial_injection_trace], contract.SERIAL_INJECTION_CONTRACT["trace"])
        self.assertEqual(obj.last_loop_lora_trace, tuple((i,p,0 if i<15 else 1) for i,p in enumerate(contract.TEXT_MODEL_SCHEDULE) if 5<=p<15))
        for branch, post in zip(([8,4] if ORDER=="84" else [4,8]), posts):
            np.testing.assert_allclose(obj.audio_gates[str(branch)].inputs[0], post, rtol=1e-13, atol=1e-13)
        self.assertTrue(all(p.loop_index is None for layer in obj.layers for p in layer.ps.values()))


class GradientAuditTests(unittest.TestCase):
    def test_trained_lora_first_audio_step_accepts_nonzero_b_and_requires_a_b_gradients(self):
        model,roles,scope=self.fixture(opened=False)
        for loop in (1,2):
            roles[f"loop{loop}_lora_B"].fill(.1)
            roles[f"loop{loop}_lora_A"].grad.fill(1.)
        result=scope['first_gradient_audit'](model)
        self.assertTrue(result['passed'],result)
        self.assertEqual(result['closed_lora_B'],{'1':False,'2':False})
        self.assertTrue(all(result['closed_gates'].values()))
        # Execute the EXACT actual first-step training verdict with this audit.
        train=audit.find_function(audit.tree(ROUTE/'train.py'),'train')
        verdict=next(n for n in ast.walk(train) if isinstance(n,ast.Call)
                     and isinstance(n.func,ast.Name) and n.func.id=='collective_require'
                     and n.args and 'segment_start_step != 0' in ast.unparse(n.args[0]))
        def require(condition,*args): self.assertTrue(condition)
        audit.execute([ast.Expr(value=copy.deepcopy(verdict))],dict(
            collective_require=require,gradient_audit=result,segment_start_step=0,device='cpu'))
        for role in ('loop1_lora_A','loop1_lora_B','loop2_lora_A','loop2_lora_B'):
            roles[role].grad.fill(0)
            self.assertFalse(scope['first_gradient_audit'](model)['passed'],role)
            roles[role].grad.fill(1)

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
            for kind in ("A","B"):
                prefixes[f"loop{loop}_lora_{kind}"] = f"caption_decoder.lm.model.layers.5.self_attn.q_proj.lora_{kind}.{loop-1}"
        roles, pairs = {}, []
        for role, prefix in prefixes.items():
            closed = role.endswith("gate") or role.endswith("lora_B")
            value = np.full((1,2), 0. if closed and not opened else 1.).view(Tensor)
            value.requires_grad = True
            zero_grad = role in {"layer4_mlp","layer8_mlp","reinject_linear","loop1_lora_A","loop2_lora_A"} and not opened
            value.grad = np.full((1,2),0. if zero_grad else 1.).view(Tensor)
            roles[role] = value
            pairs.append((prefix if "lora_" in role else prefix+".weight",value))
        gate=lambda role: NS(parameters=lambda:[roles[role]])
        lora=NS(lora_B=[roles["loop1_lora_B"],roles["loop2_lora_B"]])
        core=NS(audio_gate=gate("reinject_gate"), audio_gates={str(k):gate(f"layer{k}_gate") for k in (4,8)},
                modules=lambda:[lora], last_reinject_trace=(5,15),
                last_injection_trace=tuple((5 if i==0 else 15,k) for i,k in enumerate([8,4] if ORDER=="84" else [4,8])),
                last_serial_injection_trace=contract.SERIAL_INJECTION_CONTRACT["trace"],
                last_loop_lora_trace=[(i,p,0 if i<15 else 1) for i,p in enumerate(contract.TEXT_MODEL_SCHEDULE) if 5<=p<15])
        model=NS(caption_decoder=NS(lm=NS(model=core)),named_parameters=lambda:pairs)
        scope=dict(torch=NS(isfinite=np.isfinite,equal=np.array_equal), TEXT_MODEL_SCHEDULE=contract.TEXT_MODEL_SCHEDULE,
                   SERIAL_INJECTION_CONTRACT=contract.SERIAL_INJECTION_CONTRACT)
        source_functions("optim.py",["parameter_role"],scope)
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
        self.assertEqual(result["closed_lora_B"],{"1":True,"2":True})
        self.assertIn("loop1_lora_A",result["zero_task_gradient_expected"])
        for role in ("loop1_lora_B","loop2_lora_B"):
            roles[role].grad.fill(0)
            self.assertFalse(scope["first_gradient_audit"](model)["passed"])
            roles[role].grad.fill(1)

    def test_opened_B_requires_task_gradient_in_A_and_each_audio_upstream(self):
        model,roles,scope=self.fixture(opened=True)
        self.assertTrue(scope["first_gradient_audit"](model)["passed"])
        for role in ("loop1_lora_A","loop2_lora_A","layer4_mlp","layer8_mlp","reinject_linear"):
            roles[role].grad.fill(0)
            self.assertFalse(scope["first_gradient_audit"](model)["passed"],role)
            roles[role].grad.fill(1)

    def test_missing_nonfinite_gradients_and_wrong_loop_trace_fail(self):
        model,roles,scope=self.fixture(opened=True)
        for bad in (None,np.full((1,2),np.nan)):
            old=roles["loop2_lora_A"].grad
            roles["loop2_lora_A"].grad=bad.view(type(old)) if isinstance(bad,np.ndarray) else bad
            self.assertFalse(scope["first_gradient_audit"](model)["passed"])
            roles["loop2_lora_A"].grad=old
        model.caption_decoder.lm.model.last_loop_lora_trace=[]
        self.assertFalse(scope["first_gradient_audit"](model)["passed"])

    def test_all_twelve_roles_have_separate_snapshot_and_change_detection(self):
        model,roles,scope=self.fixture()
        snapshots=scope["role_snapshots"](model)
        self.assertEqual(set(snapshots),set(roles))
        self.assertTrue(all(not item["changed"] for item in scope["weights_changed"](model,snapshots).values()))
        roles["loop2_lora_B"] += .5
        changed=scope["weights_changed"](model,snapshots)
        self.assertTrue(changed["loop2_lora_B"]["changed"])
        self.assertFalse(changed["loop1_lora_B"]["changed"])


if __name__ == "__main__":
    unittest.main()
