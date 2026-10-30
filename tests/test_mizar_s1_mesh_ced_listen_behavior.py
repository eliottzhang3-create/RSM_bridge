"""Execute production listen math with NumPy tensors; not PyTorch/autograd evidence."""
import ast
from pathlib import Path
from types import SimpleNamespace, MethodType
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer36912_listen"


class Array(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    @property
    def device(self):
        return "cpu"

    def unsqueeze(self, dim):
        return Array(np.expand_dims(np.asarray(self), dim))

    def transpose(self, first, second):
        return Array(np.swapaxes(np.asarray(self), first, second))

    def sum(self, dim=None, keepdim=False, **kwargs):
        return Array(np.asarray(self).sum(axis=dim, keepdims=keepdim, **kwargs))

    def mean(self, dim=None, keepdim=False, **kwargs):
        return Array(np.asarray(self).mean(axis=dim, keepdims=keepdim, **kwargs))

    def to(self, device=None, dtype=None):
        if isinstance(device, (np.dtype, type)):
            dtype = device
        return Array(np.asarray(self, dtype=dtype))

    def float(self):
        return self.to(dtype=np.float32)


def softmax(x, dim):
    value = np.asarray(x)
    exp = np.exp(value - value.max(axis=dim, keepdims=True))
    return Array(exp / exp.sum(axis=dim, keepdims=True))


TORCH = SimpleNamespace(float32=np.float32, sigmoid=lambda x: Array(1 / (1 + np.exp(-x))),
    tanh=lambda x: Array(np.tanh(x)), softmax=softmax,
    stack=lambda values, dim: Array(np.stack(values, axis=dim)),
    cat=lambda values, dim: Array(np.concatenate(values, axis=dim)))


def production_functions(path, class_name, names):
    source = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
        node.returns = None
    scope = {"torch": TORCH}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return {name: scope[name] for name in names}


METHODS = production_functions(ROUTE / "models/audio_listen.py", "MultilayerAudioListen",
    ("project_audio", "validate_prompt_mask", "pool_prompt", "forward"))
MEMORY = production_functions(ROUTE / "models/recursive_mesh_listen.py", "MeshLlamaModel", ("_write", "_read"))


def norm(x):
    centered = np.asarray(x) - np.asarray(x).mean(axis=-1, keepdims=True)
    return Array(centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-5))


class Linear:
    def __init__(self, rng, n_in, n_out, zero=False):
        self.weight = Array(np.zeros((n_out, n_in), np.float32) if zero else
                            rng.normal(0, 1 / np.sqrt(n_in), (n_out, n_in)).astype(np.float32))
        self.bias = Array(np.zeros(n_out, np.float32))

    def __call__(self, x):
        return Array(np.asarray(x) @ np.asarray(self.weight).T + self.bias)


def fixture():
    rng = np.random.default_rng(987)
    module = SimpleNamespace(width=8, audio_tokens=3, prompt_tokens=4, text_start=4,
        gamma=.3, collect_diagnostics=False, last_diagnostics=[], last_rounds=[],
        projections=[Linear(rng, 1536, 8) for _ in range(4)],
        audio_input_norms=[norm] * 4, audio_output_norms=[norm] * 4,
        text_norm=norm, query_norm=norm,
        gates=[Linear(rng, 8, 4, True) for _ in range(2)],
        listen_routers=[Linear(rng, 8, 5) for _ in range(2)],
        amplitudes=[Array(np.zeros(1, np.float32)) for _ in range(2)])
    for name, fn in METHODS.items():
        setattr(module, name, MethodType(fn, module))
    hidden = Array(rng.normal(size=(2, 11, 8)).astype(np.float32))
    grids = Array(rng.normal(size=(2, 4, 3, 1536)).astype(np.float32))
    mask = Array(np.array([[1, 1, 0, 0], [1, 1, 1, 0]]))
    memory = Array(rng.normal(size=(2, 5, 11, 8)).astype(np.float32))
    reads = softmax(Array(rng.normal(size=(2, 11, 5)).astype(np.float32)), -1)
    bank = module.project_audio(grids, hidden, mask)
    return rng, module, hidden, grids, mask, memory, reads, bank


class ProductionMathTests(unittest.TestCase):
    def test_zero_amplitudes_produce_exactly_zero_writes_both_rounds(self):
        _, module, hidden, _, mask, memory, reads, bank = fixture()
        for r in (0, 1):
            np.testing.assert_array_equal(module.forward(memory, hidden, bank, mask, r, reads), memory)
        self.assertEqual(module.last_rounds, [0, 1])

    def test_only_audio_memory_positions_change_when_open(self):
        _, module, hidden, _, mask, memory, reads, bank = fixture()
        for r in (0, 1):
            module.amplitudes[r][0] = .7
            updated = module.forward(memory, hidden, bank, mask, r, reads)
            np.testing.assert_array_equal(updated[:, :, 3:], memory[:, :, 3:])
            self.assertGreater(np.abs(updated[:, :, :3] - memory[:, :, :3]).max(), 0)
            self.assertEqual(updated.shape, memory.shape)

    def test_stress_aligned_depths_and_concentrated_routes_obey_gamma_bound(self):
        rng, module, hidden, _, mask, memory, reads, _ = fixture()
        audio = norm(Array(rng.normal(size=(2, 3, 8)).astype(np.float32)))
        bank = Array(np.stack([audio] * 4, axis=1))
        reads[:] = 0; reads[:, :, 0] = 1
        for r in (0, 1):
            module.gates[r].bias[:] = 30
            module.listen_routers[r].weight[:] = 0
            module.listen_routers[r].bias[:] = -30
            module.listen_routers[r].bias[0] = 30
            for sign in (-1, 1):
                module.amplitudes[r][0] = sign * 30
                updated = module.forward(memory, hidden, bank, mask, r, reads)
                delta = Array(updated - memory)
                maximum = np.sqrt(np.asarray(delta) ** 2 .__class__(1)) if False else np.sqrt(np.mean(np.asarray(delta) ** 2, axis=-1)).max()
                self.assertLessEqual(maximum, .3 + 1e-6)
                self.assertGreater(maximum, .299)
                direct = MEMORY["_read"](delta, reads)
                self.assertLessEqual(np.sqrt(np.mean(np.asarray(direct) ** 2, axis=-1)).max(), .3 + 1e-6)

    def test_prompt_pool_excludes_pad_answer_sep_and_audio(self):
        rng, module, hidden, _, mask, _, _, _ = fixture()
        expected = module.pool_prompt(hidden, mask)
        changed = hidden.copy()
        changed[:, :4] += rng.normal(size=changed[:, :4].shape) * 100
        changed[:, 8:] += rng.normal(size=changed[:, 8:].shape) * 100
        for b in range(2):
            for t in range(4):
                if not mask[b, t]:
                    changed[b, 4 + t] += rng.normal(size=8) * 100
        np.testing.assert_array_equal(module.pool_prompt(changed, mask), expected)
        changed[0, 4, 0] += 100
        self.assertFalse(np.array_equal(module.pool_prompt(changed, mask), expected))

    def test_sigmoid_selects_depths_independently_and_gates_are_round_specific(self):
        _, module, hidden, _, mask, memory, reads, bank = fixture()
        for amplitude in module.amplitudes:
            amplitude[0] = 1
        module.gates[0].bias[:] = -30
        module.gates[1].bias[:] = 30
        closed = module.forward(memory, hidden, bank, mask, 0, reads)
        opened = module.forward(memory, hidden, bank, mask, 1, reads)
        self.assertLess(np.abs(closed - memory).max(), 1e-6)
        self.assertGreater(np.abs(opened - memory).max(), 1e-3)

    def test_actual_math_has_beta_gradient_at_zero_and_upstream_unlocks_after_opening(self):
        rng, module, hidden, _, mask, memory, reads, bank = fixture()
        memory[:] = 0
        objective_weights = rng.normal(size=memory.shape)
        def objective():
            return float((module.forward(memory, hidden, bank, mask, 0, reads) * objective_weights).sum())
        def derivative(array, index, eps=1e-3):
            old = array[index]
            array[index] = old + eps; plus = objective()
            array[index] = old - eps; minus = objective()
            array[index] = old
            return (plus - minus) / (2 * eps)
        self.assertGreater(abs(derivative(module.amplitudes[0], 0)), 1e-5)
        self.assertEqual(derivative(module.gates[0].bias, 0), 0)
        self.assertEqual(derivative(module.listen_routers[0].bias, 0), 0)
        module.amplitudes[0][0] = .5
        self.assertGreater(abs(derivative(module.gates[0].bias, 0)), 1e-6)
        self.assertGreater(abs(derivative(module.listen_routers[0].bias, 0)), 1e-6)

    def test_invalid_context_and_empty_or_nonbinary_masks_fail(self):
        _, module, hidden, grids, mask, _, _, _ = fixture()
        for bad in (None, Array(np.zeros_like(mask)), Array(mask * .5), Array(mask[:, :3])):
            with self.subTest(mask=type(bad)), self.assertRaises(ValueError):
                module.project_audio(grids, hidden, bad)
        for bad in (None, grids[:, :3], grids.astype(np.float64)):
            with self.assertRaises(ValueError):
                module.project_audio(bad, hidden, mask)

    def test_original_write_read_remain_slot_weighted_operations(self):
        rng, _, hidden, _, _, memory, reads, _ = fixture()
        weights = softmax(Array(rng.normal(size=(2, 11, 5)).astype(np.float32)), -1)
        result = MEMORY["_write"](memory, hidden, weights)
        expected = np.asarray(memory) + np.einsum("btd,bts->bstd", hidden, weights)
        np.testing.assert_allclose(result, expected, rtol=0, atol=0)
        np.testing.assert_allclose(MEMORY["_read"](result, reads), np.einsum("bstd,bts->btd", result, reads), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
