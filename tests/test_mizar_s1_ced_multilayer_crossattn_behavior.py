"""Execute actual adapter/encoder/schedule methods with NumPy, not autograd."""
import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_crossattn"
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
        cls.Tensor = Tensor
        def softmax(x, dim):
            x = np.asarray(x)
            e = np.exp(x - x.max(axis=dim, keepdims=True))
            return (e / e.sum(axis=dim, keepdims=True)).view(Tensor)
        cls.torch = SimpleNamespace(Tensor=Tensor, softmax=softmax, tanh=np.tanh,
            stack=lambda xs, dim: np.stack(xs, axis=dim).view(Tensor),
            cat=lambda xs, dim: np.concatenate(xs, axis=dim).view(Tensor))
        scope = dict(torch=cls.torch)
        for name in ("project_audio", "pool_prompt", "forward"):
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
            def __call__(self, query, key, value, need_weights):
                self.query, self.key, self.value = query.copy(), key.copy(), value.copy()
                scores = query @ key.transpose(0, 2, 1) / 2.
                weights = self_softmax(scores, dim=-1)
                return (weights @ value).view(type(query)), None
        self_softmax = self.torch.softmax
        class Adapter(SimpleNamespace):
            def __call__(candidate, *args):
                return self.actual["forward"](candidate, *args)
        adapter = Adapter(width=4, audio_tokens=2, prompt_tokens=3, text_start=3,
                          collect_diagnostics=False, last_diagnostics=[])
        adapter.pool_prompt = types.MethodType(self.actual["pool_prompt"], adapter)
        adapter.project_audio = types.MethodType(self.actual["project_audio"], adapter)
        adapter.projections = [Linear(1536, 4) for _ in range(4)]
        adapter.gate1, adapter.gate2 = Linear(4, 4), Linear(4, 1)
        adapter.gate2.weight[:] = 0
        adapter.gate2.bias[:] = 0 if closed else .4
        adapter.cross_attention = Attention()
        adapter.round_bias = [tensor([.2, -.3, .1, .5]), tensor([-.5, .1, .3, -.2])]
        h = tensor(rng.normal(size=(2, 11, 4)))
        bank = tensor(rng.normal(size=(2, 4, 2, 4)))
        mask = self.tensor([[1, 1, 0], [1, 0, 0]])
        return adapter, h, bank, mask

    def test_closed_gate_is_exact_identity_with_nonzero_round_bias(self):
        a, h, bank, mask = self.fixture(closed=True)
        for r in (0, 1):
            self.np.testing.assert_array_equal(a(h, bank, mask, r), h)

    def test_round_bias_enters_only_query_and_preserves_audio_separator(self):
        a, h, bank, mask = self.fixture()
        output = a(h, bank, mask, 0)
        expected_summary = self.np.stack([h[0, 3:5].mean(axis=0), h[1, 3]])
        self.np.testing.assert_allclose(a.gate1.last, expected_summary)
        self.np.testing.assert_array_equal(a.gate2.last, h[:, 3:])
        self.np.testing.assert_allclose(a.cross_attention.query, h[:, 3:] + a.round_bias[0])
        weights = self.torch.softmax(a.gate1(self.tensor(expected_summary)), -1)
        expected_audio = (bank * weights[:, :, None, None]).sum(dim=1)
        self.np.testing.assert_allclose(a.cross_attention.key, expected_audio)
        self.np.testing.assert_array_equal(a.cross_attention.key, a.cross_attention.value)
        self.np.testing.assert_array_equal(output[:, :3], h[:, :3])
        self.assertFalse(self.np.allclose(output[:, 3:], h[:, 3:]))

    def test_prompt_pool_excludes_answer_and_actual_pad_but_includes_valid_positions(self):
        a, h, bank, mask = self.fixture()
        expected = a.pool_prompt(h, mask)
        changed = h.copy()
        changed[:, 6:] = 1e10  # answer
        changed[0, 5] = 1e9   # actual prompt PAD
        changed[1, 4:6] = -1e9
        self.np.testing.assert_array_equal(a.pool_prompt(changed, mask), expected)
        changed[0, 3] += 1
        self.assertFalse(self.np.array_equal(a.pool_prompt(changed, mask), expected))

    def test_prompt_mask_rejects_empty_wrong_shape_and_nonbinary(self):
        a, h, bank, mask = self.fixture()
        for bad in (None, self.tensor([[1]]), mask * 0, mask * .5):
            with self.subTest(mask=bad), self.assertRaises(ValueError):
                a(h, bank, bad, 0)
        with self.assertRaises(ValueError):
            a(h, bank, mask, 2)

    def test_four_projections_execute_once_and_bad_grid_is_rejected(self):
        a, h, _, mask = self.fixture()
        grids = self.tensor(self.np.arange(2*4*2*1536).reshape(2,4,2,1536) / 10000)
        bank = a.project_audio(grids, h)
        self.assertEqual(bank.shape, (2,4,2,4))
        a(h, bank, mask, 0)
        a(h, bank, mask, 1)
        self.assertEqual([p.calls for p in a.projections], [1,1,1,1])
        for bad in (None, grids[:, :3], grids[:, :, :1], grids[..., :384]):
            with self.assertRaises(ValueError):
                a.project_audio(bad, h)

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
        scope = dict(self=core, hidden_states=h, audio_bank=bank, prompt_mask=mask,
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

    def test_original_ced_called_once_raw_four_depths_and_hooks_removed(self):
        wrapper = self.toy_ced()
        final, selected = self.ced_methods["_forward_hidden"](wrapper, self.tensor(self.np.zeros((2,64,2001))))
        self.assertEqual(wrapper.ced.calls, 1)
        self.assertEqual(final.shape, (2,504,384))
        self.assertEqual([x[0,0,0] for x in selected], [6,21,45,78])
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
            *(prefix+f"projections.{i}.weight" for i in range(4)),
            *(prefix+name for name in ("gate1.weight","gate2.weight","cross_attention.in_proj_weight",
                                      "round_bias.0","round_bias.1")))}
        model=SimpleNamespace(named_parameters=lambda:iter(parameters.items()))
        actual=scope["first_gradient_audit"]
        self.assertTrue(actual(model)["all_branches_nonzero"])
        for name,param in parameters.items():
            if name.startswith(prefix) and "gate2" not in name:
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
