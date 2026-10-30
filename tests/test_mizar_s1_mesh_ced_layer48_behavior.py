"""Execute real source with stdlib/NumPy; no claim about PyTorch autograd."""
import ast
import copy
from contextlib import nullcontext
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
ROUTE = ROOT/"code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject"
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
                    "caption_decoder.lm.model.write_routers.0", "caption_decoder.lm.model.read_routers.2",
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
        launcher = (ROOT/"code/RSmol/run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_inject_5090.sh").read_text()
        runner = (ROUTE/"run.sh").read_text()
        self.assertIn("layer4_8_inject_5090/index_cache", launcher)
        self.assertIn("/dev/shm/mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_inject_", runner)
        self.assertIn("--nproc_per_node=8", runner)
        self.assertNotIn("smoke-report", launcher+runner)
        self.assertNotIn("python runtime_checks.py", runner)

    def test_protected_parents_unchanged(self):
        saved = ROOT/"audit_reports/mizar_s1_mesh_ced_layer48_parent_baseline.json"
        import json
        for name, expected in json.loads(saved.read_text()).items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), expected, name)

    def test_audio_and_residual_methods_are_exact_source_implementations(self):
        source = ROUTE.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject")
        self.assertEqual((ROUTE/"models/ced.py").read_bytes().replace(b"\r\n", b"\n"),
                         (source/"models/ced.py").read_bytes().replace(b"\r\n", b"\n"))
        for name in ("enable_layer_injection", "inject_audio", "injection_diagnostics"):
            self.assertEqual(ast.dump(audit.find_function(audit.tree(ROUTE/"models/recursive_layer_inject.py"), name)),
                             ast.dump(audit.find_function(audit.tree(source/"models/recursive_layer_inject.py"), name)))
        self.assertEqual(contract.INJECTION_CONTRACT["read_router_indices_0based"], [0, 1])
        self.assertEqual(contract.INJECTION_CONTRACT["placement"], "after_memory_read_before_middle_loop")


@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires NumPy")
class NumericTests(unittest.TestCase):
    """Execute the actual MeSH forward body with NumPy stand-ins, not autograd."""
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np
        scope = dict(torch=SimpleNamespace(tanh=np.tanh, cat=lambda xs, dim: np.concatenate(xs, axis=dim)))
        module = audit.tree(ROUTE/"models/recursive_layer_inject.py")
        core = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "MeshLlamaModel")
        injection = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "inject_audio")
        audit.execute([injection], scope)
        cls.inject = staticmethod(scope["inject_audio"])
        forward = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        begin = next(i for i,n in enumerate(forward.body) if isinstance(n,ast.Assign)
                     and ast.unparse(n.targets[0]) == "memory")
        end = next(i for i,n in enumerate(forward.body) if isinstance(n,ast.Assign)
                   and ast.unparse(n.targets[0]) == "hidden" and ast.unparse(n.value) == "self.norm(hidden)")
        cls.forward_body = forward.body[begin:end+1]
        scope = {}
        audit.execute([audit.find_function(audit.tree(ROUTE/"models/ced.py"), "_hidden_to_temporal_grid")], scope)
        cls.grid = staticmethod(scope["_hidden_to_temporal_grid"])

    def fixture(self, closed=False):
        np = self.np
        class Tensor(np.ndarray):
            @property
            def device(self): return "cpu"
        class Linear:
            def __init__(self, value):
                self.weight, self.bias, self.inputs = np.full((1,3), value), np.zeros(1), []
            def __call__(self, x):
                self.inputs.append(x.copy())
                return x @ self.weight.T + self.bias
        obj = SimpleNamespace(layer_inject_tokens=126, last_injection_trace=(),
                              collect_injection_stats=False, audit_mode=False, gradient_audit_mode=False,
                              write_routers=list(range(3)), read_routers=list(range(3)),
                              audio_gates={"4": Linear(0. if closed else .03), "8": Linear(0. if closed else -.02)},
                              norm=lambda h:h, events=[], queries={})
        obj.inject_audio = types.MethodType(self.inject, obj)
        def stack(h, indices, start, **kwargs):
            obj.events.append(("stack", start, h.copy()))
            for physical in indices:
                h = h + .01*(physical+1)
            return h
        def route(router, query, name):
            obj.queries[name] = query.copy()
            scores = query.mean(axis=-1, keepdims=True) * np.linspace(-.1,.2,5) + router*.02*np.arange(5)
            weights = np.exp(scores-scores.max(axis=-1,keepdims=True))
            return weights/weights.sum(axis=-1,keepdims=True)
        obj._run_stack, obj._route = stack, route
        obj._write = lambda memory,value,w: memory+value[:,None]*w.transpose(0,2,1)[...,None]
        def read(memory,w):
            h = (memory*w.transpose(0,2,1)[...,None]).sum(axis=1)
            obj.events.append(("read",len([e for e in obj.events if e[0]=="read"]),h.copy()))
            return h
        obj._read = read
        x = np.random.default_rng(42).normal(size=(2,131,3)).view(Tensor)
        information = (np.ones((2,126,3)), np.full((2,126,3),2.))
        return obj, x, information

    def run_forward(self,obj,x,information):
        np=self.np
        scope=dict(self=obj, hidden=x.copy(), audio_information=information,
                   batch_size=x.shape[0], query_length=x.shape[1], MEMORY_SLOT_COUNT=5,
                   torch=SimpleNamespace(zeros=lambda shape,**kw:np.zeros(shape)),
                   record_function=lambda name:nullcontext(), output_hidden_states=False,
                   mask=None, position_ids=None, cache=None, use_cache=False, cache_position=None,
                   position_embeddings=None, output_attentions=False, attentions=[])
        audit.execute(copy.deepcopy(self.forward_body),scope)
        return scope["hidden"],scope["memory"]

    def test_actual_memory_reads_feed_the_two_gates_and_middle_loops(self):
        obj,x,information=self.fixture()
        result,memory=self.run_forward(obj,x,information)
        self.assertEqual([(e[0],e[1]) for e in obj.events],
                         [("stack",0),("read",0),("stack",5),("read",1),("stack",15),("read",2),("stack",25)])
        reads=[e[2] for e in obj.events if e[0]=="read"]
        stacks=[e[2] for e in obj.events if e[0]=="stack"]
        for branch,layer in enumerate((4,8)):
            self.np.testing.assert_array_equal(obj.audio_gates[str(layer)].inputs[0],reads[branch][:,:126])
            expected=obj.inject_audio(reads[branch],information[branch],layer,(5,15)[branch])
            self.np.testing.assert_array_equal(stacks[branch+1],expected)
            # Preserve the mother's router query: the input of the preceding loop.
            self.np.testing.assert_array_equal(obj.queries[f"write_{branch}"],stacks[branch+1])
            self.np.testing.assert_array_equal(obj.queries[f"read_{branch}"],stacks[branch+1])
        self.np.testing.assert_array_equal(stacks[3],reads[2])
        self.assertEqual(obj.last_injection_trace[:2],((5,4),(15,8)))
        self.assertEqual(memory.shape,(2,5,131,3))
        self.assertEqual(result.shape,x.shape)

    def test_zero_gates_equal_original_mesh_computation_and_non_audio_unchanged(self):
        obj,x,information=self.fixture(closed=True)
        actual,actual_memory=self.run_forward(obj,x,information)
        base,_,_=self.fixture()
        base.inject_audio=lambda hidden,*args:hidden
        expected,expected_memory=self.run_forward(base,x,information)
        self.np.testing.assert_array_equal(actual,expected)
        self.np.testing.assert_array_equal(actual_memory,expected_memory)
        opened,_,_=self.fixture()
        changed,_=self.run_forward(opened,x,information)
        self.np.testing.assert_array_equal(changed[:,126:],expected[:,126:])
        self.assertFalse(self.np.array_equal(changed[:,:126],expected[:,:126]))

    def test_scalar_gate_no_mutation_and_explicit_per_forward_information(self):
        obj,x,information=self.fixture()
        before=x.copy()
        out=obj.inject_audio(x,information[0],4,5)
        gate=self.np.tanh(x[:,:126]@obj.audio_gates["4"].weight.T)
        self.assertEqual(gate.shape,(2,126,1))
        self.np.testing.assert_allclose(out[:,:126]-x[:,:126],gate*information[0],atol=1e-15)
        self.np.testing.assert_array_equal(x,before)
        self.np.testing.assert_array_equal(out[:,126:],x[:,126:])
        old=self.run_forward(self.fixture()[0],x,information)[0]
        new=self.run_forward(self.fixture()[0],x,(information[0]*2,information[1]*3))[0]
        self.assertFalse(self.np.array_equal(old,new))

    def test_grid_preserves_batch_segment_frequency_and_time(self):
        np=self.np
        class Tensor(np.ndarray):
            def permute(self,*axes):return self.transpose(axes)
        values=np.arange(2*2*4*63*384).reshape(2,504,384).view(Tensor)
        result=self.grid(None,values)
        source=values.reshape(2,2,4,63,384)
        self.assertEqual(result.shape,(2,126,4,384))
        for batch in range(2):
            for segment in range(2):
                for time in (0,1,31,62):
                    for frequency in range(4):
                        np.testing.assert_array_equal(result[batch,segment*63+time,frequency],source[batch,segment,frequency,time])

    def test_zero_gate_then_open_gate_finite_difference(self):
        obj,x,information=self.fixture(closed=True)
        a=information[0].copy()
        def loss():return float((obj.inject_audio(x,a,4,5)*x).sum())
        def diff(value,index):
            old,eps=value[index],1e-5
            value[index]=old+eps; plus=loss()
            value[index]=old-eps; minus=loss()
            value[index]=old
            return (plus-minus)/(2*eps)
        self.assertEqual(diff(a,(0,0,0)),0.)
        self.assertGreater(abs(diff(obj.audio_gates["4"].weight,(0,0))),1e-5)
        obj.audio_gates["4"].bias[0]=.1
        self.assertGreater(abs(diff(a,(0,0,0))),1e-5)


if __name__ == "__main__":
    unittest.main()


