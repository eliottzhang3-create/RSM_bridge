"""Dependency-light behavior/parity checks. Run each route test in its own process."""
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_audio_reinject"
sys.path.insert(0, str(ROUTE))
import audit_reference
import contract
from models import recursive_text


def config():
    return dict(model_type="llama", architectures=["RecursiveLlamaForCausalLM"], hidden_size=576,
                num_hidden_layers=30, recursive_layer_count=20, recursive_loops=2,
                recursive_prefix_layer_count=5, recursive_middle_layer_count=10,
                recursive_suffix_layer_count=5, recursive_loops_scope="middle_only",
                logical_to_physical=list(contract.TEXT_MODEL_SCHEDULE), vocab_size=49152,
                recursive_source_layer_indices_0based=list(contract.TEXT_MODEL_SOURCE_MAPPING), eos_token_id=0)


class ReinjectionContractTests(unittest.TestCase):
    def test_parent_and_entire_backbone_parity_and_report_identity(self):
        result = audit_reference.audit()
        self.assertEqual(len(result["parent_parity"]["identical_files_normalized_lf"]), 17)
        self.assertTrue(result["backbone_parity"]["entire_backbone_ast_equal_after_explicit_additions"])
        saved = json.loads((ROUTE / "reference_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["route_code_sha256"], contract.code_digest())
        self.assertIn("not tensor", saved["scope"])

    def test_config_matches_parent_except_explicit_architecture_and_identity(self):
        prepared = dict(rows=1477492, source_rows=contract.SOURCE_COUNTS,
                        rows_canonical_sha256="rows", mapping_sha256="mapping", stage_root="job1")
        one = contract.semantic_contract(prepared, "text", "ced")
        prepared["stage_root"] = "job2"
        self.assertEqual(one, contract.semantic_contract(prepared, "text", "ced"))
        parent_scope = {"__file__": str(audit_reference.PARENT/"contract.py")}
        audit_reference.execute(audit_reference.tree(audit_reference.PARENT/"contract.py").body, parent_scope)
        parent = parent_scope["semantic_contract"](prepared, "text", "ced")
        self.assertNotEqual(contract.digest(parent), contract.digest(one))
        self.assertNotEqual(parent["route"], one["route"])
        self.assertEqual(one["audio_reinjection"]["logical_indices_0based"], [5, 15])
        self.assertEqual(one["audio_reinjection"]["new_parameters"], 332929)

    def test_source_digest_tracks_local_variant(self):
        source = (ROUTE/"models/recursive_reinject.py").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(contract.recursive_model_source_sha256(), hashlib.sha256(source).hexdigest())

    def test_strict_backbone_load_finishes_before_registration_and_no_failure_is_allowed(self):
        audit = dict(logical_layer_count=30, physical_layer_count=20, recursive_layer_count=20,
                     recursive_loops=2, middle_recurrent_count=10, model_physical_module_count=20,
                     schedule=list(contract.TEXT_MODEL_SCHEDULE))
        for failure in (None, "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory)
                (checkpoint/"config.json").write_text("{}")
                events = []
                model = SimpleNamespace(config=SimpleNamespace(to_dict=config), generation_config=SimpleNamespace(use_cache=True))
                model.model = SimpleNamespace(layers=[object() for _ in range(20)],
                    logical_to_physical=contract.TEXT_MODEL_SCHEDULE,
                    enable_audio_reinjection=lambda: events.append("register"))
                model.named_parameters = lambda: iter([])
                model.named_buffers = lambda: iter([])
                cls = Mock()
                cls.from_pretrained.return_value = model, {} if failure is None else {failure: ["bad"]}
                def audit_model(candidate):
                    events.append("strict_audit")
                    return audit
                with patch.object(recursive_text, "_import_recursive_model", return_value=(cls, audit_model)), \
                        patch.object(recursive_text, "validate_audio_reinjection", return_value={"fixture": True}):
                    if failure:
                        with self.assertRaisesRegex(ValueError, "weights do not match"):
                            recursive_text.load_recursive_text_model(checkpoint)
                        self.assertEqual(events, [])
                    else:
                        self.assertIs(recursive_text.load_recursive_text_model(checkpoint), model)
                        self.assertEqual(events, ["strict_audit", "register"])
                        self.assertFalse(model.generation_config.use_cache)
                    self.assertEqual(cls.from_pretrained.call_args.kwargs,
                                     dict(local_files_only=True, output_loading_info=True))

    def test_new_parameters_have_explicit_lr_and_decay_and_are_not_lm_grouped(self):
        class Param:
            requires_grad = True
            def __init__(self, shape):
                self.ndim, self.size = len(shape), __import__("math").prod(shape)
            def numel(self):
                return self.size
        pairs = [("audio_encoder.base.mapper.weight", Param((576,384))),
                 ("caption_decoder.lm.model.layers.5.weight", Param((576,576)))]
        prefix = "caption_decoder.lm.model."
        for name, shape in (("audio_linear.weight", (576,576)), ("audio_linear.bias", (576,)),
                            ("audio_gate.weight", (1,576)), ("audio_gate.bias", (1,))):
            pairs.append((prefix+name, Param(shape)))
        class Optimizer:
            def __init__(self, groups, **kwargs):
                self.param_groups, self.kwargs = groups, kwargs
        stub = SimpleNamespace(optim=SimpleNamespace(AdamW=Optimizer,
            lr_scheduler=SimpleNamespace(LambdaLR=lambda opt, fn: (opt, fn))))
        scope = dict(torch=stub, lr_factor=contract.lr_factor)
        nodes = [audit_reference.find_function(audit_reference.tree(ROUTE/"optim.py"), n)
                 for n in ("parameter_role", "build_optimizer")]
        audit_reference.execute(nodes, scope)
        optimizer, _, audit = scope["build_optimizer"](SimpleNamespace(named_parameters=lambda: iter(pairs)), 17313)
        self.assertEqual(optimizer.kwargs["betas"], (.9,.999))
        self.assertEqual(optimizer.kwargs["eps"], 1e-8)
        self.assertEqual(sum(g["numel"] for g in audit if g["role"].startswith("reinject")), 332929)
        for group in audit:
            self.assertEqual(group["base_lr"], 3e-4 if group["role"] == "default" else 6e-4)
            for name in group["names"]:
                self.assertEqual(group["weight_decay"], 0 if name.endswith("bias") else .01)

    def test_no_runtime_smoke_prerequisite_and_isolated_namespaces(self):
        launcher = (ROOT/"code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_audio_reinject_5090.sh").read_text()
        runner = (ROUTE/"run.sh").read_text()
        self.assertNotIn("--mode smoke", launcher+runner)
        self.assertNotIn("smoke-report", launcher+runner)
        self.assertIn("mizar_s1_ced_small_gbs256_recursive_5_10_5_audio_reinject_5090/index_cache", launcher)
        self.assertIn("log/mizar_s1_gbs256_recursive_5_10_5_ced_small_audio_reinject.${RUN_TAG}.JOB.log", launcher)
        self.assertIn("/dev/shm/mizar_s1_gbs256_recursive_5_10_5_ced_small_audio_reinject_", runner)
        self.assertIn("--nproc_per_node=8", runner)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy required for dependency-light numeric checks")
class ActualInjectionNumpyTests(unittest.TestCase):
    """Execute actual injection/loop AST with NumPy; not a PyTorch/autograd claim."""
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np
        module = audit_reference.tree(ROUTE/"models/recursive_reinject.py")
        core = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "RecursiveLlama5_10_5Model")
        scope = dict(torch=SimpleNamespace(tanh=np.tanh, cat=lambda xs, dim: np.concatenate(xs, axis=dim)))
        audit_reference.execute([next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "inject_audio")], scope)
        cls.inject = staticmethod(scope["inject_audio"])
        forward = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        cls.actual_nodes = [n for n in forward.body if isinstance(n, ast.Assign) and any(
            ast.unparse(t) == "audio_information" for t in n.targets)] + [next(n for n in forward.body if isinstance(n, ast.For))]

    def fixture(self, closed=False):
        np = self.np
        rng = np.random.default_rng(1234)
        class Linear:
            def __init__(self, weight, bias):
                self.weight, self.bias, self.inputs = weight, bias, []
            def __call__(self, x):
                self.inputs.append(x.copy())
                return x @ self.weight.T + self.bias
        obj = SimpleNamespace(audio_reinject_tokens=126, audio_reinject_indices=(5,15),
                              last_reinject_trace=(), collect_reinject_stats=False)
        obj.audio_linear = Linear(rng.normal(size=(3,3)), np.zeros(3))
        obj.audio_gate = Linear(np.zeros((1,3)) if closed else np.array([[.01,-.02,.03]]), np.zeros(1))
        obj.inject_audio = types.MethodType(self.inject, obj)
        obj.logical_to_physical = contract.TEXT_MODEL_SCHEDULE
        obj.layers = [lambda h, i=i: h+(i+1)*.01 for i in range(20)]
        return obj, rng.normal(size=(1,130,3))

    def run_actual_loop(self, obj, x):
        scope = dict(self=obj, inputs_embeds=x, hidden_states=x.copy(), output_hidden_states=False,
                     output_attentions=False, cache=None, causal_mask=None, position_ids=None,
                     use_cache=False, cache_position=None, position_embeddings=None,
                     _call_decoder_layer=lambda layer,h,**kw: (layer(h),))
        audit_reference.execute(copy.deepcopy(self.actual_nodes), scope)
        return scope["hidden_states"]

    def test_A_once_gate_twice_current_states_and_unchanged_text_positions(self):
        obj, x = self.fixture()
        actual = self.run_actual_loop(obj,x)
        self.assertEqual(len(obj.audio_linear.inputs), 1)
        self.assertEqual(len(obj.audio_gate.inputs), 2)
        self.assertEqual(obj.last_reinject_trace, (5,15))
        self.np.testing.assert_array_equal(obj.audio_linear.inputs[0], x[:,:126])
        self.assertFalse(self.np.array_equal(*obj.audio_gate.inputs))
        baseline = x.copy()
        for i in contract.TEXT_MODEL_SCHEDULE:
            baseline = obj.layers[i](baseline)
        self.np.testing.assert_array_equal(actual[:,126:], baseline[:,126:])
        self.assertFalse(self.np.array_equal(actual[:,:126], baseline[:,:126]))

    def test_zero_gate_forward_exactly_matches_original_loop(self):
        obj, x = self.fixture(closed=True)
        expected = x.copy()
        for i in contract.TEXT_MODEL_SCHEDULE:
            expected = obj.layers[i](expected)
        self.np.testing.assert_array_equal(self.run_actual_loop(obj,x), expected)

    def test_scalar_broadcast_signed_gate_and_no_in_place_mutation(self):
        obj,x = self.fixture()
        original = x.copy()
        a = obj.audio_linear(x[:,:126])
        g = self.np.tanh(obj.audio_gate(x[:,:126]))
        self.assertEqual(g.shape, (1,126,1))
        self.assertTrue((g < 0).any() and (g > 0).any())
        out = obj.inject_audio(x,a,5)
        self.np.testing.assert_array_equal(x,original)
        self.np.testing.assert_allclose(out[:,:126]-x[:,:126], g*a, atol=1e-15)
        self.np.testing.assert_array_equal(out[:,126:],x[:,126:])

    def test_finite_difference_closed_and_open_gate_gradient_behavior(self):
        obj,x = self.fixture(closed=True)
        target = self.np.random.default_rng(4).normal(size=x.shape)
        def loss():
            a = obj.audio_linear(x[:,:126])
            return float((obj.inject_audio(x,a,5)*target).sum())
        def derivative(weight,index):
            old,eps = weight[index],1e-5
            weight[index] = old+eps
            plus = loss()
            weight[index] = old-eps
            minus = loss()
            weight[index] = old
            return (plus-minus)/(2*eps)
        self.assertEqual(derivative(obj.audio_linear.weight,(0,0)),0.)
        self.assertGreater(abs(derivative(obj.audio_gate.weight,(0,0))),1e-4)
        obj.audio_gate.bias[0] = .2
        self.assertGreater(abs(derivative(obj.audio_linear.weight,(0,0))),1e-4)


if __name__ == "__main__":
    unittest.main()
