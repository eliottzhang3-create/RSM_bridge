"""Execute the actual LoRA algebra/dispatch with NumPy, without mocking autograd.

These checks run on the code-only Windows host. Real torch gradients, HF
loading and logical KV caches are tested separately in the tensor suite.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'code/RSmol/mizar_text_5_10x2_5_loop_lora_r8/loop_lora.py'
MODEL = ROOT / 'code/RSmol/recursive_model_mizar_text_5_10x2_5_loop_lora_r8.py'


def linear(inputs, weight, bias=None):
    result = inputs @ weight.T
    return result if bias is None else result + bias


def executable_node(path, name, namespace, *, class_name=None):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    if class_name:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class NumpyLoRATests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(47)
        self.forward = executable_node(HELPER, 'forward', {'F': types.SimpleNamespace(linear=linear)},
                                       class_name='LoopLoRALinear')
        self.targets = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                        'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
        namespace = {'TARGETS': self.targets}
        executable_node(HELPER, 'projections', namespace)
        self.scope = executable_node(HELPER, 'call_with_loop_lora', namespace)
        self.inputs = self.rng.normal(size=(2, 3, 4))
        self.module = types.SimpleNamespace(
            weight=self.rng.normal(size=(3, 4)), bias=self.rng.normal(size=3), scaling=2.,
            lora_A=[self.rng.normal(size=(8, 4)) for _ in range(2)],
            lora_B=[np.zeros((3, 8)) for _ in range(2)], loop_index=None)

    def layer(self):
        modules = {name: types.SimpleNamespace(loop_index=None) for name in self.targets}
        return types.SimpleNamespace(get_submodule=modules.__getitem__), list(modules.values())

    def test_zero_b_preserves_base_and_nonzero_branches_differ(self):
        base = linear(self.inputs, self.module.weight, self.module.bias)
        before = self.module.weight.copy()
        for loop in (0, 1):
            self.module.loop_index = loop
            np.testing.assert_array_equal(self.forward(self.module, self.inputs), base)
        self.module.lora_B = [self.rng.normal(size=(3, 8)) for _ in range(2)]
        results = []
        for loop in (0, 1):
            self.module.loop_index = loop
            actual = self.forward(self.module, self.inputs)
            expected = base + self.inputs @ (2 * self.module.lora_B[loop] @ self.module.lora_A[loop]).T
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
            results.append(actual)
        self.assertFalse(np.allclose(*results))
        np.testing.assert_array_equal(before, self.module.weight)

    def test_finite_difference_of_actual_forward_for_both_branches(self):
        self.module.lora_B = [self.rng.normal(size=(3, 8)) for _ in range(2)]
        x = self.inputs.reshape(-1, 4)
        upstream = self.rng.normal(size=(x.shape[0], 3))
        for loop in (0, 1):
            self.module.loop_index = loop
            expected = [upstream.T @ x,
                        2 * (upstream @ self.module.lora_B[loop]).T @ x,
                        2 * upstream.T @ (x @ self.module.lora_A[loop].T),
                        np.zeros_like(self.module.lora_A[1-loop]),
                        np.zeros_like(self.module.lora_B[1-loop])]
            arrays = [self.module.weight, self.module.lora_A[loop], self.module.lora_B[loop],
                      self.module.lora_A[1-loop], self.module.lora_B[1-loop]]
            for array, derivative in zip(arrays, expected):
                numeric = np.zeros_like(array)
                for index in np.ndindex(array.shape):
                    original = array[index]
                    array[index] = original + 1e-5
                    plus = (self.forward(self.module, x) * upstream).sum()
                    array[index] = original - 1e-5
                    minus = (self.forward(self.module, x) * upstream).sum()
                    array[index] = original
                    numeric[index] = (plus-minus) / 2e-5
                np.testing.assert_allclose(numeric, derivative, rtol=1e-7, atol=1e-7)

    def test_selection_is_explicit_and_scope_resets_even_on_error(self):
        with self.assertRaises(RuntimeError):
            self.forward(self.module, self.inputs)
        layer, modules = self.layer()
        for loop in (0, 1):
            result = self.scope(layer, loop, lambda current: tuple(m.loop_index for m in modules))
            self.assertEqual(result, (loop,) * 7)
            self.assertTrue(all(m.loop_index is None for m in modules))
        def fail(current):
            self.assertEqual([m.loop_index for m in modules], [1] * 7)
            raise ValueError('decoder failed')
        with self.assertRaisesRegex(ValueError, 'decoder failed'):
            self.scope(layer, 1, fail)
        self.assertTrue(all(m.loop_index is None for m in modules))
        with self.assertRaises(RuntimeError):
            self.scope(layer, 2, lambda _: None)
        modules[0].loop_index = 0
        with self.assertRaises(RuntimeError):
            self.scope(layer, 1, lambda _: None)
        self.assertEqual(modules[0].loop_index, 0)

    def test_actual_decoder_dispatch_preserves_outer_and_selects_middle(self):
        calls = []
        layer, modules = self.layer()
        def decoder(current, hidden, **kwargs):
            calls.append((tuple(m.loop_index for m in modules), hidden, kwargs))
            return ('decoder-result',)
        dispatch = executable_node(MODEL, '_call_loop_decoder_layer',
                                   {'_call_decoder_layer': decoder, 'call_with_loop_lora': self.scope})
        for loop in (None, 0, 1):
            result = dispatch(layer, 'hidden', loop_index=loop, cache='cache', use_cache=False)
            self.assertEqual(result, ('decoder-result',))
            self.assertEqual(calls[-1], ((loop,) * 7, 'hidden', {'cache': 'cache', 'use_cache': False}))
            self.assertTrue(all(m.loop_index is None for m in modules))


if __name__ == '__main__':
    unittest.main()
