"""Execute real source with stdlib/NumPy; no claim about PyTorch autograd."""
import ast
import copy
import hashlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT/"code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject"
sys.path.insert(0, str(ROUTE))
import audit_reference as audit
import contract
from models import recursive_text


class ContractTests(unittest.TestCase):
    def test_all_parent_training_and_backbone_semantics(self):
        result = audit.audit()
        self.assertTrue(result["protected_originals"]["all_unchanged"])
        self.assertEqual(result["protected_originals"]["files"], 137)
        self.assertEqual(len(result["parent_parity"]["identical_source_files"]), 18)
        self.assertTrue(result["backbone_parity"]["entire_parent_backbone_ast_equal"])
        self.assertTrue(result["backbone_parity"]["original_input_branch_methods_equal"])
        self.assertTrue(result["training_parity"]["complete_train_function_ast_equal_after_explicit_audits"])


    def test_identity_and_complete_schedule(self):
        prepared = dict(rows=1477492, source_rows=contract.SOURCE_COUNTS,
                        rows_canonical_sha256="rows", mapping_sha256="mapping", stage_root="a")
        one = contract.semantic_contract(prepared, "text", "ced")
        prepared["stage_root"] = "b"
        self.assertEqual(one, contract.semantic_contract(prepared, "text", "ced"))
        self.assertEqual(one["layer_injection"], contract.INJECTION_CONTRACT)
        self.assertEqual(one["layer_injection"]["ced_layers_1based"], [8, 4])
        self.assertEqual(one["layer_injection"]["logical_indices_0based"], [5, 15])
        other = copy.deepcopy(one)
        other["layer_injection"]["ced_layers_1based"] = [3, 7]
        self.assertNotEqual(contract.digest(one), contract.digest(other))
        self.assertEqual(one["geometry"]["total_steps"], 17313)
        self.assertEqual(one["adamw"]["new_parameters_lr"], 6e-4)
        raw = (ROUTE/"models/recursive_layer_inject.py").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(contract.recursive_model_source_sha256(), hashlib.sha256(raw).hexdigest())

    def test_strict_loading_finishes_before_gate_registration(self):
        for failure in (None, "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            events = []
            model = SimpleNamespace(model=SimpleNamespace(enable_layer_injection=lambda: events.append("register_layers"),
                                        enable_audio_reinjection=lambda: events.append("register_input")),
                                    generation_config=SimpleNamespace(use_cache=True))
            cls = Mock()
            cls.from_pretrained.side_effect = lambda *a, **k: (events.append("load") or model,
                                                              {} if failure is None else {failure: ["bad"]})
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/"config.json").write_text("{}")
                with patch.object(recursive_text, "_import_recursive_model", return_value=(cls, lambda m: {})), \
                     patch.object(recursive_text, "validate_loaded_model", side_effect=lambda *a: events.append("validate")), \
                     patch.object(recursive_text, "validate_serial_injection"):
                    if failure:
                        with self.assertRaises(ValueError):
                            recursive_text.load_recursive_text_model(root)
                        self.assertEqual(events, ["load"])
                    else:
                        recursive_text.load_recursive_text_model(root)
                        self.assertEqual(events, ["load", "validate", "register_layers", "register_input"])
                        self.assertFalse(model.generation_config.use_cache)
                    self.assertTrue(cls.from_pretrained.call_args.kwargs["local_files_only"])

    def test_resume_guard_accepts_only_this_route_and_matching_contract(self):
        train = audit.find_function(audit.tree(ROUTE/"train.py"), "train")
        guard = next(n for n in ast.walk(train) if isinstance(n, ast.If)
                     and "saved.get('schema_version')" in ast.unparse(n.test))
        saved = dict(schema_version=1, route_variant=contract.ROUTE,
                     mode="formal", contract_sha256="new-8-4-contract")
        scope = dict(saved=saved, config={"contract_sha256": "new-8-4-contract"}, ROUTE=contract.ROUTE)
        audit.execute([copy.deepcopy(guard)], scope)
        for key, wrong in (("route_variant", contract.ROUTE.replace("layer8_4", "layer4_8")),
                           ("contract_sha256", "old-4-8-contract"), ("mode", "smoke")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identical route"):
                audit.execute([copy.deepcopy(guard)], dict(scope, saved=dict(saved, **{key: wrong})))

    def test_optimizer_exact_roles_lrs_decay_and_schedule(self):
        class Parameter:
            requires_grad = True
            def __init__(self, ndim): self.ndim = ndim
            def numel(self): return 12
        prefixes = ["audio_encoder.base.mapper", "caption_decoder.lm.model.layers.5",
                    "audio_encoder.base.layer_mlps.4", "audio_encoder.base.layer_mlps.8",
                    "caption_decoder.lm.model.audio_gates.4", "caption_decoder.lm.model.audio_gates.8",
                    "caption_decoder.lm.model.audio_linear", "caption_decoder.lm.model.audio_gate"]
        pairs = [(p+suffix, Parameter(ndim)) for p in prefixes
                 for suffix, ndim in ((".weight", 2), (".bias", 1), (".norm.weight", 1))]
        class Optimizer:
            def __init__(self, groups, **kwargs): self.param_groups, self.kwargs = groups, kwargs
        scope = dict(torch=SimpleNamespace(optim=SimpleNamespace(AdamW=Optimizer,
                     lr_scheduler=SimpleNamespace(LambdaLR=lambda opt, fn: fn))), lr_factor=contract.lr_factor)
        module = audit.tree(ROUTE/"optim.py")
        audit.execute([audit.find_function(module, n) for n in ("parameter_role", "build_optimizer")], scope)
        optimizer, schedule, groups = scope["build_optimizer"](SimpleNamespace(named_parameters=lambda: iter(pairs)), 17313)
        self.assertEqual(len({g["role"] for g in groups}), 8)
        for group in groups:
            self.assertEqual(group["base_lr"], 3e-4 if group["role"] == "default" else 6e-4)
            for name in group["names"]:
                self.assertEqual(group["weight_decay"], 0. if name.endswith("bias") or ".norm." in name else .01)
        self.assertEqual(optimizer.kwargs["betas"], (.9, .999))
        for step in range(17314):
            self.assertEqual(schedule(step), contract.lr_factor(step, 17313))
        with self.assertRaises(ValueError):
            scope["parameter_role"]("audio_encoder.base.ced.blocks.3.weight")

    def test_formal_only_and_isolated_namespace(self):
        launcher = (ROOT/"code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_5090.sh").read_text()
        runner = (ROUTE/"run.sh").read_text()
        self.assertIn("layer8_4_inject_audio_reinject_5090/index_cache", launcher)
        self.assertIn("serial audio-input then layer8/layer4 injection", launcher)
        self.assertIn("/dev/shm/mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_", runner)
        self.assertIn("--nproc_per_node=8", runner)
        self.assertNotIn("smoke-report", launcher+runner)
        self.assertNotIn("python runtime_checks.py", runner)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires NumPy")
class NumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np
        scope = dict(torch=SimpleNamespace(tanh=np.tanh, cat=lambda xs, dim: np.concatenate(xs, axis=dim)))
        core = next(n for n in audit.tree(ROUTE/"models/recursive_layer_inject.py").body
                    if isinstance(n, ast.ClassDef) and n.name == "RecursiveLlama5_10_5Model")
        audit.execute([next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "inject_audio")], scope)
        cls.inject = staticmethod(scope["inject_audio"])
        audit.execute([next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "inject_input_audio")], scope)
        cls.inject_input = staticmethod(scope["inject_input_audio"])
        forward = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        cls.project = next(n for n in forward.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "audio_input_information")
        cls.loop = next(n for n in next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "forward").body
                        if isinstance(n, ast.For) and ast.unparse(n.iter).startswith("enumerate("))
        scope = {}
        audit.execute([audit.find_function(audit.tree(ROUTE/"models/ced.py"), "_hidden_to_temporal_grid")], scope)
        cls.grid = staticmethod(scope["_hidden_to_temporal_grid"])

    def fixture(self, closed=False):
        np = self.np
        class Linear:
            def __init__(self, value): self.weight, self.bias, self.inputs = np.full((1, 3), value), np.zeros(1), []
            def __call__(self, x):
                self.inputs.append(x.copy())
                return x @ self.weight.T + self.bias
        obj = SimpleNamespace(layer_inject_tokens=126, layer_inject_indices=(5, 15),
                              last_injection_trace=(), collect_injection_stats=False,
                              audio_gates={"4": Linear(0. if closed else .03), "8": Linear(0. if closed else -.02)},
                              logical_to_physical=contract.TEXT_MODEL_SCHEDULE)
        class Projection:
            def __init__(self): self.weight, self.bias, self.inputs = np.eye(3)*.7, np.zeros(3), []
            def __call__(self, x):
                self.inputs.append(x.copy())
                return x @ self.weight.T + self.bias
        obj.audio_reinject_tokens, obj.audio_reinject_indices = 126, (5,15)
        obj.audio_gate, obj.audio_linear = Linear(0. if closed else .05), Projection()
        obj.last_reinject_trace, obj.last_serial_injection_trace = (), ()
        obj.collect_reinject_stats = False
        obj.inject_input_audio = types.MethodType(self.inject_input, obj)
        obj.layers = [lambda h, i=i: h + .01*(i+1) for i in range(20)]
        obj.inject_audio = types.MethodType(self.inject, obj)
        x = np.random.default_rng(42).normal(size=(2, 131, 3))
        return obj, x, (np.ones((2,126,3)), np.full((2,126,3), 2.))

    def run_loop(self, obj, x, information, loop=None):
        scope = dict(self=obj, hidden_states=x.copy(), inputs_embeds=x, audio_information=information,
                     output_hidden_states=False, output_attentions=False, cache=None, causal_mask=None,
                     position_ids=None, use_cache=False, cache_position=None, position_embeddings=None,
                     _call_decoder_layer=lambda layer, h, **kw: (layer(h),))
        audit.execute([copy.deepcopy(self.project), copy.deepcopy(self.loop if loop is None else loop)], scope)
        return scope["hidden_states"]

    def test_two_independent_branches_at_exact_loop_entrances(self):
        obj, x, information = self.fixture()
        result = self.run_loop(obj, x, information)
        expected = x.copy()
        projected = x[:,:126] @ obj.audio_linear.weight.T
        posts = []
        for index, physical in enumerate(contract.TEXT_MODEL_SCHEDULE):
            if index in (5,15):
                audio = expected[:,:126]
                updated = audio + self.np.tanh(audio @ obj.audio_gate.weight.T) * projected
                posts.append(updated.copy())
                branch = 1 - (5,15).index(index)
                gate = obj.audio_gates[str((4,8)[branch])]
                final = updated + self.np.tanh(updated @ gate.weight.T) * information[branch]
                expected = self.np.concatenate((final, expected[:,126:]), axis=1)
            expected = obj.layers[physical](expected)
        self.np.testing.assert_allclose(result, expected)
        self.assertEqual(obj.last_injection_trace, ((5,8),(15,4)))
        self.assertEqual(obj.last_reinject_trace, (5,15))
        self.assertEqual(obj.last_serial_injection_trace, ((5,"input"),(5,"layer8"),(15,"input"),(15,"layer4")))
        for layer, post in zip((8,4), posts):
            self.assertEqual(len(obj.audio_gates[str(layer)].inputs),1)
            self.np.testing.assert_allclose(obj.audio_gates[str(layer)].inputs[0], post)
        self.assertEqual(len(obj.audio_linear.inputs),1)
        self.assertEqual(len(obj.audio_gate.inputs),2)
        self.np.testing.assert_array_equal(obj.audio_linear.inputs[0], x[:,:126])


    def test_nonzero_gates_distinguish_serial_from_parallel_and_reverse(self):
        obj,x,information = self.fixture()
        actual = self.run_loop(obj,x,information)
        for mode in ("parallel", "reverse"):
            other,_,_ = self.fixture()
            expected=x.copy()
            projected=x[:,:126] @ other.audio_linear.weight.T
            for index,physical in enumerate(contract.TEXT_MODEL_SCHEDULE):
                if index in (5,15):
                    branch=1-(5,15).index(index)
                    audio=expected[:,:126]
                    gate=other.audio_gates[str((4,8)[branch])]
                    layer_residual=self.np.tanh(audio@gate.weight.T)*information[branch]
                    input_hidden=audio if mode=="parallel" else audio+layer_residual
                    input_residual=self.np.tanh(input_hidden@other.audio_gate.weight.T)*projected
                    expected=self.np.concatenate((audio+input_residual+layer_residual,expected[:,126:]),axis=1)
                expected=other.layers[physical](expected)
            self.assertFalse(self.np.allclose(actual[:,:126],expected[:,:126]), mode)
            self.np.testing.assert_array_equal(actual[:,126:],expected[:,126:])


    def test_zero_gate_exact_baseline_and_non_audio_positions(self):
        obj, x, information = self.fixture(closed=True)
        expected = x.copy()
        for physical in contract.TEXT_MODEL_SCHEDULE:
            expected = obj.layers[physical](expected)
        self.np.testing.assert_array_equal(self.run_loop(obj,x,information), expected)
        obj, _, _ = self.fixture()
        self.np.testing.assert_array_equal(self.run_loop(obj,x,information)[:,126:],expected[:,126:])

    def test_gate_scalar_broadcast_and_no_input_mutation(self):
        obj,x,information = self.fixture()
        before = x.copy()
        out = obj.inject_audio(x,information[0],4,5)
        self.np.testing.assert_array_equal(x,before)
        gate = self.np.tanh(x[:,:126]@obj.audio_gates["4"].weight.T)
        self.assertEqual(gate.shape,(2,126,1))
        self.assertTrue((gate<0).any() and (gate>0).any())
        self.np.testing.assert_allclose(out[:,:126]-x[:,:126],gate*information[0],atol=1e-15)
        self.np.testing.assert_array_equal(out[:,126:],x[:,126:])

    def test_grid_preserves_each_batch_segment_frequency_and_time(self):
        np = self.np
        class Tensor(np.ndarray):
            def permute(self,*axes): return self.transpose(axes)
        values = np.arange(2*2*4*63*384).reshape(2,504,384).view(Tensor)
        result = self.grid(None,values)
        source = values.reshape(2,2,4,63,384)
        self.assertEqual(result.shape,(2,126,4,384))
        for batch in range(2):
            for segment in range(2):
                for time in (0,1,31,62):
                    for frequency in range(4):
                        np.testing.assert_array_equal(result[batch,segment*63+time,frequency],source[batch,segment,frequency,time])

    def test_closed_gate_gradient_then_open_gate_gradient_finite_difference(self):
        obj,x,information = self.fixture(closed=True)
        for layer, branch in ((4,0),(8,1)):
            gate = obj.audio_gates[str(layer)]
            a = information[branch].copy()
            def loss(): return float((obj.inject_audio(x,a,layer,5)*x).sum())
            def diff(value,index):
                old,eps = value[index],1e-5
                value[index]=old+eps
                plus=loss()
                value[index]=old-eps
                minus=loss()
                value[index]=old
                return (plus-minus)/(2*eps)
            self.assertEqual(diff(a,(0,0,0)),0.)
            self.assertGreater(abs(diff(gate.weight,(0,0))),1e-5)
            gate.bias[0]=.1
            self.assertGreater(abs(diff(a,(0,0,0))),1e-5)



    def test_input_branch_gradient_opens_after_zero_gate(self):
        obj,x,information=self.fixture(closed=True)
        def loss():
            a=obj.audio_linear(x[:,:126])
            first=obj.inject_input_audio(x,a,5)
            return float((obj.inject_audio(first,information[1],8,5)*x).sum())
        def diff(value,index):
            before,eps=value[index],1e-5
            value[index]=before+eps;plus=loss()
            value[index]=before-eps;minus=loss()
            value[index]=before
            return (plus-minus)/(2*eps)
        self.assertEqual(diff(obj.audio_linear.weight,(0,0)),0.)
        self.assertGreater(abs(diff(obj.audio_gate.weight,(0,0))),1e-5)
        obj.audio_gate.bias[0]=.1
        self.assertGreater(abs(diff(obj.audio_linear.weight,(0,0))),1e-5)

    def test_disabling_each_branch_reproduces_parent_loop(self):
        np=self.np
        for disabled in ("input","intermediate"):
            obj,x,information=self.fixture()
            if disabled=="input": obj.audio_gate.weight.fill(0)
            else:
                for gate in obj.audio_gates.values(): gate.weight.fill(0)
            result=self.run_loop(obj,x,information)
            expected=x.copy();projected=x[:,:126]@obj.audio_linear.weight.T
            for index,physical in enumerate(contract.TEXT_MODEL_SCHEDULE):
                if index in (5,15):
                    audio=expected[:,:126]
                    if disabled=="input":
                        branch=1-(5,15).index(index)
                        audio=audio+np.tanh(audio@obj.audio_gates[str((4,8)[branch])].weight.T)*information[branch]
                    else: audio=audio+np.tanh(audio@obj.audio_gate.weight.T)*projected
                    expected=np.concatenate((audio,expected[:,126:]),axis=1)
                expected=obj.layers[physical](expected)
            np.testing.assert_array_equal(result,expected)


class SerialContractTests(unittest.TestCase):
    def test_serial_order_and_gate_condition_are_resume_identity(self):
        prepared=dict(rows=1477492,source_rows=contract.SOURCE_COUNTS,rows_canonical_sha256="r",mapping_sha256="m")
        recipe=contract.semantic_contract(prepared,"text","ced")
        self.assertEqual(recipe["audio_reinjection"],contract.AUDIO_REINJECTION)
        self.assertEqual(recipe["serial_audio_injection"]["new_parameters"],2777859)
        self.assertEqual(recipe["serial_audio_injection"]["intermediate_gate_condition"],"audio_hidden_after_input_residual")
        for key,value in (("order",["ced_intermediate","audio_input_embeddings"]),
                          ("intermediate_gate_condition","audio_hidden_before_input_residual")):
            wrong=copy.deepcopy(recipe)
            wrong["serial_audio_injection"][key]=value
            self.assertNotEqual(contract.digest(recipe),contract.digest(wrong))
        self.assertTrue(recipe["audio_reinjection"]["shared_gate"])
        self.assertTrue(recipe["layer_injection"]["independent_gates"])

    def test_single_route_full_resume_rejects_both_original_routes(self):
        function=audit.find_function(audit.tree(ROUTE/"train.py"),"train")
        guard=next(n for n in ast.walk(function) if isinstance(n,ast.If) and "saved.get('schema_version')" in ast.unparse(n.test))
        config={"contract_sha256":"same"}
        for suffix in ("layer8_4_inject", "audio_reinject", "layer4_8_inject"):
            saved=dict(schema_version=1,route_variant=f"ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_{suffix}_v1",mode="formal",contract_sha256="same")
            with self.subTest(suffix=suffix),self.assertRaises(ValueError):
                audit.execute([guard],dict(saved=saved,config=config,ROUTE=contract.ROUTE))

    def test_three_closed_gates_are_audited_and_all_new_roles_have_snapshots(self):
        module=audit.tree(ROUTE/"train.py")
        names={}
        for n in module.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {"TRAINABLE_ROLES","NEW_ROLES"} for t in n.targets):
                audit.execute([n],names)
        self.assertEqual(len(names["TRAINABLE_ROLES"]),8)
        self.assertEqual(set(names["NEW_ROLES"]),{"layer4_mlp","layer8_mlp","layer4_gate","layer8_gate","reinject_linear","reinject_gate"})
        self.assertEqual(contract.SERIAL_INJECTION_CONTRACT["trace"],[[5,"input"],[5,"layer8"],[15,"input"],[15,"layer4"]])



@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires NumPy")
class GradientAuditTests(unittest.TestCase):
    def fixture(self, opened=False):
        import numpy as np
        class Tensor(np.ndarray):
            def detach(self): return self
            def ne(self,value): return self!=value
        prefixes={"map":"audio_encoder.base.mapper", "default":"caption_decoder.lm.model.layers.0",
                  "layer4_mlp":"audio_encoder.base.layer_mlps.4", "layer8_mlp":"audio_encoder.base.layer_mlps.8",
                  "layer4_gate":"caption_decoder.lm.model.audio_gates.4", "layer8_gate":"caption_decoder.lm.model.audio_gates.8",
                  "reinject_linear":"caption_decoder.lm.model.audio_linear", "reinject_gate":"caption_decoder.lm.model.audio_gate"}
        roles={}
        for role in prefixes:
            value=np.full((1,2),0. if role.endswith("gate") and not opened else 1.).view(Tensor)
            value.requires_grad=True
            zero_gradient=not opened and role in {"layer4_mlp","layer8_mlp","reinject_linear"}
            value.grad=np.full((1,2),0. if zero_gradient else .2).view(Tensor)
            roles[role]=value
        core=SimpleNamespace(audio_gates={str(k):SimpleNamespace(parameters=lambda k=k:[roles[f"layer{k}_gate"]]) for k in (4,8)},
             audio_gate=SimpleNamespace(parameters=lambda:[roles["reinject_gate"]]),
             last_injection_trace=((5,8),(15,4)),last_reinject_trace=(5,15),
             last_serial_injection_trace=((5,"input"),(5,"layer8"),(15,"input"),(15,"layer4")))
        model=SimpleNamespace(caption_decoder=SimpleNamespace(lm=SimpleNamespace(model=core)),
                              named_parameters=lambda:iter((prefixes[k]+".weight",v) for k,v in roles.items()))
        scope=dict(torch=SimpleNamespace(isfinite=np.isfinite),SERIAL_INJECTION_CONTRACT=contract.SERIAL_INJECTION_CONTRACT,
                   TRAINABLE_ROLES=tuple(roles))
        audit.execute([audit.find_function(audit.tree(ROUTE/"optim.py"),"parameter_role"),
                       audit.find_function(audit.tree(ROUTE/"train.py"),"first_gradient_audit")],scope)
        return scope["first_gradient_audit"],model,roles,core

    def test_closed_and_opened_gradient_audits_include_all_three_paths(self):
        for opened in (False,True):
            fn,model,roles,core=self.fixture(opened)
            result=fn(model)
            self.assertTrue(result["passed"],result)
            self.assertEqual(set(result["closed_gates"]),{"4","8","input"})
            self.assertEqual(set(result["zero_task_gradient_expected"]),set() if opened else {"layer4_mlp","layer8_mlp","reinject_linear"})

    def test_missing_input_gate_gradient_and_premature_projection_gradient_fail(self):
        for role,value in (("reinject_gate",0.),("reinject_linear",.5)):
            fn,model,roles,core=self.fixture()
            roles[role].grad.fill(value)
            self.assertFalse(fn(model)["passed"])

    def test_wrong_serial_trace_and_nonfinite_branch_gradient_fail(self):
        fn,model,roles,core=self.fixture(True)
        core.last_serial_injection_trace=((5,"layer8"),(5,"input"),(15,"layer4"),(15,"input"))
        self.assertFalse(fn(model)["passed"])
        fn,model,roles,core=self.fixture(True)
        roles["reinject_linear"].grad.fill(float("nan"))
        result=fn(model)
        self.assertFalse(result["passed"])
        self.assertTrue(result["nonfinite"])


if __name__ == "__main__":
    unittest.main()
