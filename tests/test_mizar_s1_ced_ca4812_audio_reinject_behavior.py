"""Execute actual adapter/encoder/schedule methods with NumPy, not autograd."""
import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_recursive_5_10_5_ced_small_in_ca4812"
sys.path.insert(0, str(ROUTE))
import audit_parent
import contract


def execute(nodes, scope):
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<actual-route-methods>", "exec"), scope)


def method(path, cls, name):
    return audit_parent.function(audit_parent.function(audit_parent.tree(path), cls), name)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy unavailable")
class ActualMethodsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np
        class Tensor(np.ndarray):
            @property
            def device(self):
                return "cpu"
            def to(self, device=None, dtype=None):
                return np.asarray(self, dtype=dtype).view(Tensor)
            def sum(self, dim=None, keepdim=False):
                return np.asarray(self).sum(axis=dim, keepdims=keepdim).view(Tensor)
            def unsqueeze(self, dim):
                return np.expand_dims(self, dim).view(Tensor)
            def permute(self, *axes):
                return self.transpose(axes).view(Tensor)
            def split(self, length, dim):
                return tuple(np.split(self, range(length, self.shape[dim], length), axis=dim))
            def new_zeros(self, *shape):
                return np.zeros(shape, dtype=self.dtype).view(Tensor)
            def ne(self, value):
                return (self != value).view(Tensor)
            def detach(self):
                return self
            def cpu(self):
                return self
            def square(self):
                return np.square(np.asarray(self)).view(Tensor)
            def abs(self):
                return np.abs(np.asarray(self)).view(Tensor)
            def mean(self, dim=None, axis=None, keepdims=False):
                return np.asarray(np.asarray(self).mean(
                    axis=dim if axis is None else axis, keepdims=keepdims)).view(Tensor)
            def sqrt(self):
                return np.sqrt(np.asarray(self)).view(Tensor)
            def max(self):
                return np.asarray(np.asarray(self).max()).view(Tensor)
        cls.Tensor = Tensor
        def softmax(x, dim):
            x = np.asarray(x)
            e = np.exp(x - x.max(axis=dim, keepdims=True))
            return (e / e.sum(axis=dim, keepdims=True)).view(Tensor)
        cls.torch = SimpleNamespace(Tensor=Tensor, softmax=softmax, tanh=np.tanh,
            stack=lambda xs, dim=0: np.stack(xs, axis=dim).view(Tensor),
            cat=lambda xs, dim=0: np.concatenate(xs, axis=dim).view(Tensor))
        scope = dict(torch=cls.torch)
        for name in ("project_audio", "project_input", "forward", "diagnostics"):
            execute([method(ROUTE/"models/multilayer_audio.py", "MultilayerAudioCrossAttention", name)], scope)
        cls.actual = scope
        scope = dict(torch=cls.torch)
        for name in ("_forward_hidden", "_hidden_to_temporal_grid"):
            execute([method(ROUTE/"models/ced.py", "CEDSmallWrapper", name)], scope)
        cls.ced_methods = scope

    def tensor(self, x):
        return self.np.asarray(x, dtype=float).view(self.Tensor)

    def fixture(self, closed=False):
        np, tensor = self.np, self.tensor
        rng = np.random.default_rng(49)
        class Linear:
            def __init__(self, d_in, d_out):
                self.weight = tensor(rng.normal(size=(d_in, d_out)) * .1)
                self.bias = tensor(np.zeros(d_out))
                self.calls = 0
            def __call__(self, x):
                self.calls += 1
                self.last = x.copy()
                return (x @ self.weight + self.bias).view(type(x))
        class Attention:
            output_scale = 1.
            def __call__(self, query, key, value, need_weights):
                self.query, self.key, self.value = query.copy(), key.copy(), value.copy()
                scores = query @ key.transpose(0, 2, 1) / 2.
                weights = self_softmax(scores, dim=-1)
                return (self.output_scale * (weights @ value)).view(type(query)), None
        class LayerNorm:
            def __init__(self, width):
                self.width, self.calls = width, 0
            def __call__(self, x):
                self.calls += 1
                self.last = x.copy()
                values = np.asarray(x)
                return tensor((values - values.mean(axis=-1, keepdims=True)) /
                              np.sqrt(values.var(axis=-1, keepdims=True) + 1e-5))
        self_softmax = self.torch.softmax
        class Adapter(SimpleNamespace):
            def __call__(candidate, *args):
                return self.actual["forward"](candidate, *args)
        adapter = Adapter(width=4, audio_tokens=2, prompt_tokens=3, text_start=3,
                          collect_diagnostics=False, last_diagnostics=[], last_injection_trace=())
        adapter.project_audio = types.MethodType(self.actual["project_audio"], adapter)
        adapter.projections = [Linear(1536, 4) for _ in range(3)]
        adapter.audio_input_norms = [LayerNorm(1536) for _ in range(3)]
        adapter.audio_output_norms = [LayerNorm(4) for _ in range(3)]
        adapter.query_norm = LayerNorm(4)
        adapter.input_projection, adapter.input_gate = Linear(4, 4), Linear(4, 1)
        adapter.input_gate.bias[:] = 0 if closed else .2
        if closed: adapter.input_gate.weight[:] = 0
        adapter.gate1, adapter.gate2 = Linear(4, 3), Linear(4, 1)
        adapter.gate2.weight[:] = 0
        adapter.gate2.bias[:] = 0 if closed else .4
        adapter.cross_attention = Attention()
        h = tensor(rng.normal(size=(2, 11, 4)))
        bank = tensor(rng.normal(size=(2, 3, 2, 4)))
        mask = adapter.input_projection(h[:, :2])
        return adapter, h, bank, mask

    def test_both_closed_gates_are_exact_identity(self):
        a, h, bank, mask = self.fixture(closed=True)
        for r in (0, 1):
            self.np.testing.assert_array_equal(a(h, bank, mask, r), h)

    def test_raw_gate_inputs_tokenwise_mixing_and_audio_only_writes(self):
        a,h,bank,r=self.fixture()
        output=a(h,bank,r,0)
        pre=h[:,:2]
        self.np.testing.assert_array_equal(a.input_gate.last,pre)
        injected=pre+self.np.tanh(pre@a.input_gate.weight+a.input_gate.bias)*r
        self.np.testing.assert_array_equal(a.gate1.last,injected)
        self.np.testing.assert_array_equal(a.gate2.last,injected)
        self.np.testing.assert_array_equal(a.query_norm.last,injected)
        w=self.torch.softmax(a.gate1(injected),-1)
        self.assertEqual(w.shape,(2,2,3))
        self.assertFalse(self.np.allclose(w[:,0],w[:,1]))
        mixed=(bank*w.permute(0,2,1).unsqueeze(-1)).sum(dim=1)
        self.np.testing.assert_allclose(a.cross_attention.key,mixed)
        self.np.testing.assert_array_equal(a.cross_attention.key,a.cross_attention.value)
        self.np.testing.assert_allclose(a.cross_attention.query,a.query_norm(injected))
        self.np.testing.assert_array_equal(output[:,2:],h[:,2:])
        self.assertFalse(self.np.allclose(output[:,:2],h[:,:2]))
        self.assertEqual(a.last_injection_trace,((0,'input','cross_attention'),))

    def test_prompt_answer_separator_do_not_drive_audio_routing(self):
        a,h,bank,r=self.fixture()
        out=a(h,bank,r,0)
        k=a.cross_attention.key.copy()
        changed=h.copy();changed[:,2:]+=1e10
        actual=a(changed,bank,r,1)
        self.np.testing.assert_array_equal(out[:,:2],actual[:,:2])
        self.np.testing.assert_array_equal(k,a.cross_attention.key)
        self.np.testing.assert_array_equal(actual[:,2:],changed[:,2:])

    def test_bad_round_or_bank_shape_is_rejected(self):
        a,h,bank,r=self.fixture()
        for b,res,round_index in ((bank,r,2),(bank[:,:2],r,0),(bank,r[:,:1],0)):
            with self.assertRaises(ValueError): a(h,b,res,round_index)

    def test_three_pre_post_norms_and_input_projection_execute_once(self):
        a,h,_,_=self.fixture()
        grids=self.tensor(self.np.random.default_rng(5).normal(size=(2,3,2,1536)))
        bank=a.project_audio(grids,h)
        r=self.actual['project_input'](a,h)
        for index in (0,1): a(h,bank,r,index)
        self.assertEqual(bank.shape,(2,3,2,4))
        self.assertEqual([p.calls for p in a.projections],[1,1,1])
        self.assertEqual([n.calls for n in a.audio_input_norms+a.audio_output_norms],[1]*6)
        self.assertEqual(a.input_projection.calls,2) # fixture plus this complete forward
        self.assertEqual(a.query_norm.calls,2)
        for i in range(3):
            self.np.testing.assert_array_equal(a.projections[i].last,a.audio_input_norms[i](grids[:,i]))
        for bad in (None,grids[:,:2],grids[:,:,:1],grids[...,:384]):
            with self.assertRaises(ValueError): a.project_audio(bad,h)

    def test_ca_amplitude_is_preserved_and_diagnostics_match_actual_residual(self):
        a,h,bank,r=self.fixture()
        a.input_gate.weight[:]=0;a.input_gate.bias[:]=0
        a.gate2.bias[:]=100;a.collect_diagnostics=True
        small=a(h,bank,r,0)
        a.cross_attention.output_scale=1e6
        large=a(h,bank,r,1)
        self.np.testing.assert_allclose(large[:,:2]-h[:,:2],1e6*(small[:,:2]-h[:,:2]),rtol=1e-10)
        self.assertGreater(float(self.np.abs(large[:,:2]-h[:,:2]).max()),1000)
        self.np.testing.assert_array_equal(large[:,2:],h[:,2:])
        residual=large[:,:2]-h[:,:2]
        self.assertAlmostEqual(float(a.last_diagnostics[-1][1][9]),float(self.np.sqrt(self.np.square(self.np.asarray(residual)).mean())),places=8)
        a.diagnostics=types.MethodType(self.actual['diagnostics'],a)
        self.assertEqual(len(a.diagnostics()),2)
        for forbidden in ('output_norm','round_bias','residual_scale','text_norm'):
            self.assertFalse(hasattr(a,forbidden))

    def test_raw_residual_base_and_input_then_ca_are_observable(self):
        a,h,bank,r=self.fixture()
        h=self.tensor(self.np.asarray(h)*100+500)
        out=a(h,bank,r,0)
        post=h[:,:2]+self.np.tanh(h[:,:2]@a.input_gate.weight+a.input_gate.bias)*r
        self.np.testing.assert_array_equal(a.gate2.last,post)
        self.assertGreater(abs(float(a.gate2.last.mean())),100)
        a.cross_attention.output_scale=0
        only_input=a(h,bank,r,1)
        self.np.testing.assert_allclose(only_input[:,:2],post)
        self.assertFalse(self.np.allclose(out[:,:2],only_input[:,:2]))

    def run_actual_schedule(self, a, h, bank, mask):
        calls = []
        def call(layer, hidden_states, **kwargs):
            calls.append(layer)
            n = hidden_states.shape[1]
            # A deterministic causal layer; execute the actual route's scheduling AST.
            context = self.np.cumsum(hidden_states, axis=1) / self.np.arange(1,n+1)[None,:,None]
            return (self.tensor(hidden_states + context * .01),)
        core = SimpleNamespace(logical_to_physical=contract.TEXT_MODEL_SCHEDULE,
                               layers=list(range(20)), audio_crossattn=a, last_crossattn_trace=())
        forward = method(ROUTE/"models/recursive_crossattn.py", "RecursiveLlama5_10_5Model", "forward")
        loop = next(n for n in forward.body if isinstance(n, ast.For))
        scope = dict(self=core, hidden_states=h, audio_bank=bank, input_residual=mask,
                     output_hidden_states=False, cache=None, causal_mask=None, position_ids=None,
                     use_cache=False, cache_position=None, position_embeddings=None,
                     output_attentions=False, _call_decoder_layer=call)
        execute([loop], scope)
        self.assertEqual(calls, list(contract.TEXT_MODEL_SCHEDULE))
        self.assertEqual(core.last_crossattn_trace, (5, 15))
        return scope["hidden_states"]

    def test_open_gates_future_answers_do_not_change_prior_predictions(self):
        a, h, bank, mask = self.fixture()
        expected = self.run_actual_schedule(a, h.copy(), bank, mask)
        changed = h.copy()
        changed[:, 8:] += 10000
        actual = self.run_actual_schedule(a, changed, bank, mask)
        self.np.testing.assert_array_equal(expected[:, :8], actual[:, :8])
        self.assertFalse(self.np.allclose(expected[:, 8:], actual[:, 8:]))

    def test_full_sequence_and_incremental_full_recomputation_match(self):
        a, h, bank, mask = self.fixture()
        full = self.run_actual_schedule(a, h.copy(), bank, mask)
        for end in range(6, h.shape[1] + 1):
            prefix = self.run_actual_schedule(a, h[:, :end].copy(), bank, mask)
            self.np.testing.assert_allclose(full[:, :end], prefix, rtol=1e-12, atol=1e-12)

    def test_temporal_grid_preserves_split_frequency_time_order(self):
        hidden = self.tensor(self.np.arange(2*504*384).reshape(2,504,384))
        grid = self.ced_methods["_hidden_to_temporal_grid"](None, hidden)
        self.assertEqual(grid.shape, (2,126,4,384))
        original = hidden.reshape(2,2,4,63,384)
        for b, split, frequency, time, channel in ((0,0,0,0,0),(0,1,2,19,8),(1,1,3,62,383)):
            self.assertEqual(grid[b,split*63+time,frequency,channel], original[b,split,frequency,time,channel])

    def toy_ced(self, fail=False):
        tensor, np = self.tensor, self.np
        class Block:
            def __init__(self, index):
                self.index, self.hooks = index, {}
            def register_forward_hook(self, hook):
                key = len(self.hooks)
                self.hooks[key] = hook
                return SimpleNamespace(remove=lambda: self.hooks.pop(key, None))
            def __call__(self, x):
                if fail and self.index == 7:
                    raise RuntimeError("synthetic CED failure")
                out = x + self.index + 1
                for hook in list(self.hooks.values()):
                    self.assert_no_replacement(hook(self, (x,), out))
                return out
            @staticmethod
            def assert_no_replacement(value):
                if value is not None:
                    raise AssertionError("read-only hook replaced the encoder output")
        blocks = [Block(i) for i in range(12)]
        ced = SimpleNamespace(maximal_allowed_length=1012, blocks=blocks, calls=0)
        def features(flat):
            ced.calls += 1
            x = tensor(np.zeros((flat.shape[0],252,384)))
            for block in blocks:
                x = block(x)
            return x * 100  # distinguish raw block12 from final normalization path
        ced.forward_features = features
        wrapper = SimpleNamespace(ced=ced, _prepare_ced_input=lambda mel: tensor(np.zeros((mel.shape[0],1,64,2001))))
        return wrapper

    def test_original_ced_called_once_raw_three_depths_and_hooks_removed(self):
        wrapper = self.toy_ced()
        final, selected = self.ced_methods["_forward_hidden"](wrapper, self.tensor(self.np.zeros((2,64,2001))))
        self.assertEqual(wrapper.ced.calls, 1)
        self.assertEqual(final.shape, (2,504,384))
        self.assertEqual([x[0,0,0] for x in selected], [10,36,78])
        self.assertEqual(final[0,0,0], 7800)
        self.assertTrue(all(not block.hooks for block in wrapper.ced.blocks))

    def test_ced_exception_also_removes_every_hook(self):
        wrapper = self.toy_ced(fail=True)
        with self.assertRaisesRegex(RuntimeError, "synthetic CED failure"):
            self.ced_methods["_forward_hidden"](wrapper, self.tensor(self.np.zeros((1,64,2001))))
        self.assertTrue(all(not block.hooks for block in wrapper.ced.blocks))

    def test_real_gradient_audit_distinguishes_closed_open_and_disconnected_branches(self):
        scope=dict(torch=SimpleNamespace(isfinite=self.np.isfinite))
        execute([audit_parent.function(audit_parent.tree(ROUTE/"optim.py"),"parameter_role")],scope)
        execute([audit_parent.function(audit_parent.tree(ROUTE/"train.py"),"first_gradient_audit")],scope)
        prefix="caption_decoder.lm.model.audio_crossattn."
        parameters={name:SimpleNamespace(requires_grad=True,grad=self.tensor([1.,2.])) for name in (
            "audio_encoder.base.mapper.weight", "caption_decoder.lm.model.layers.5.weight",
            *(prefix+f"projections.{i}.weight" for i in range(3)),
            *(prefix+f"audio_input_norms.{i}.weight" for i in range(3)),
            *(prefix+f"audio_output_norms.{i}.weight" for i in range(3)),
            *(prefix+name for name in ("gate1.weight","gate2.weight","cross_attention.in_proj_weight",
                                      "input_projection.weight","input_gate.weight","query_norm.weight")))}
        model=SimpleNamespace(named_parameters=lambda:iter(parameters.items()))
        actual=scope["first_gradient_audit"]
        self.assertTrue(actual(model)["all_branches_nonzero"])
        for name,param in parameters.items():
            if name.startswith(prefix) and "gate2" not in name and "input_gate" not in name:
                param.grad=self.tensor([0.,0.])
        audit=actual(model)
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["all_branches_nonzero"])
        for param in parameters.values():param.grad=self.tensor([1.,2.])
        parameters[prefix+"projections.2.weight"].grad=self.tensor([0.,0.])
        audit=actual(model)
        self.assertTrue(audit["nonzero_by_role"]["crossattn_projections"])
        self.assertFalse(audit["all_branches_nonzero"])
        parameters[prefix+"projections.2.weight"].grad=None
        self.assertFalse(actual(model)["passed"])


if __name__ == "__main__":
    unittest.main()
