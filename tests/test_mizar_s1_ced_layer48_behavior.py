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
ROUTE = ROOT/"code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject"
sys.path.insert(0, str(ROUTE))
import audit_reference as audit
import contract
from models import recursive_text


class ContractTests(unittest.TestCase):
    def test_all_parent_training_and_backbone_semantics(self):
        result = audit.audit()
        self.assertEqual(len(result["parent_parity"]["identical_files_normalized_lf"]), 14)
        self.assertTrue(result["backbone_parity"]["entire_backbone_ast_equal_after_explicit_additions"])

    def test_identity_and_complete_schedule(self):
        prepared = dict(rows=1477492, source_rows=contract.SOURCE_COUNTS,
                        rows_canonical_sha256="rows", mapping_sha256="mapping", stage_root="a")
        one = contract.semantic_contract(prepared, "text", "ced")
        prepared["stage_root"] = "b"
        self.assertEqual(one, contract.semantic_contract(prepared, "text", "ced"))
        self.assertEqual(one["layer_injection"], contract.INJECTION_CONTRACT)
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
            model = SimpleNamespace(model=SimpleNamespace(enable_layer_injection=lambda: events.append("register")),
                                    generation_config=SimpleNamespace(use_cache=True))
            cls = Mock()
            cls.from_pretrained.side_effect = lambda *a, **k: (events.append("load") or model,
                                                              {} if failure is None else {failure: ["bad"]})
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/"config.json").write_text("{}")
                with patch.object(recursive_text, "_import_recursive_model", return_value=(cls, lambda m: {})), \
                     patch.object(recursive_text, "validate_loaded_model", side_effect=lambda *a: events.append("validate")), \
                     patch.object(recursive_text, "validate_layer_injection"):
                    if failure:
                        with self.assertRaises(ValueError):
                            recursive_text.load_recursive_text_model(root)
                        self.assertEqual(events, ["load"])
                    else:
                        recursive_text.load_recursive_text_model(root)
                        self.assertEqual(events, ["load", "validate", "register"])
                        self.assertFalse(model.generation_config.use_cache)
                    self.assertTrue(cls.from_pretrained.call_args.kwargs["local_files_only"])

    def test_optimizer_exact_roles_lrs_decay_and_schedule(self):
        class Parameter:
            requires_grad = True
            def __init__(self, ndim): self.ndim = ndim
            def numel(self): return 12
        prefixes = ["audio_encoder.base.mapper", "caption_decoder.lm.model.layers.5",
                    "audio_encoder.base.layer_mlps.4", "audio_encoder.base.layer_mlps.8",
                    "caption_decoder.lm.model.audio_gates.4", "caption_decoder.lm.model.audio_gates.8"]
        pairs = [(p+suffix, Parameter(ndim)) for p in prefixes
                 for suffix, ndim in ((".weight", 2), (".bias", 1), (".norm.weight", 1))]
        class Optimizer:
            def __init__(self, groups, **kwargs): self.param_groups, self.kwargs = groups, kwargs
        scope = dict(torch=SimpleNamespace(optim=SimpleNamespace(AdamW=Optimizer,
                     lr_scheduler=SimpleNamespace(LambdaLR=lambda opt, fn: fn))), lr_factor=contract.lr_factor)
        module = audit.tree(ROUTE/"optim.py")
        audit.execute([audit.find_function(module, n) for n in ("parameter_role", "build_optimizer")], scope)
        optimizer, schedule, groups = scope["build_optimizer"](SimpleNamespace(named_parameters=lambda: iter(pairs)), 17313)
        self.assertEqual(len({g["role"] for g in groups}), 6)
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
        launcher = (ROOT/"code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4_8_inject_5090.sh").read_text()
        runner = (ROUTE/"run.sh").read_text()
        self.assertIn("layer4_8_inject_5090/index_cache", launcher)
        self.assertIn("/dev/shm/mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4_8_inject_", runner)
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
        obj.layers = [lambda h, i=i: h + .01*(i+1) for i in range(20)]
        obj.inject_audio = types.MethodType(self.inject, obj)
        x = np.random.default_rng(42).normal(size=(2, 131, 3))
        return obj, x, (np.ones((2,126,3)), np.full((2,126,3), 2.))

    def run_loop(self, obj, x, information):
        scope = dict(self=obj, hidden_states=x.copy(), audio_information=information,
                     output_hidden_states=False, output_attentions=False, cache=None, causal_mask=None,
                     position_ids=None, use_cache=False, cache_position=None, position_embeddings=None,
                     _call_decoder_layer=lambda layer, h, **kw: (layer(h),))
        audit.execute([copy.deepcopy(self.loop)], scope)
        return scope["hidden_states"]

    def test_two_independent_branches_at_exact_loop_entrances(self):
        obj, x, information = self.fixture()
        result = self.run_loop(obj, x, information)
        expected = x.copy()
        for index, physical in enumerate(contract.TEXT_MODEL_SCHEDULE):
            if index in (5,15):
                branch = (5,15).index(index)
                audio = expected[:,:126]
                gate = obj.audio_gates[str((4,8)[branch])]
                expected = self.np.concatenate((audio+self.np.tanh(audio@gate.weight.T)*information[branch], expected[:,126:]), axis=1)
            expected = obj.layers[physical](expected)
        self.np.testing.assert_allclose(result, expected)
        self.assertEqual(obj.last_injection_trace, ((5,4),(15,8)))
        self.assertEqual(len(obj.audio_gates["4"].inputs), 1)
        self.assertEqual(len(obj.audio_gates["8"].inputs), 1)
        self.assertFalse(self.np.array_equal(obj.audio_gates["4"].inputs[0],obj.audio_gates["8"].inputs[0]))

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


if __name__ == "__main__":
    unittest.main()
