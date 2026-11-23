"""Execute actual three-loop LoRA, stack and optimizer code with NumPy fixtures."""
import ast
import copy
import math
import types
import unittest
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from test_mizar_s1_text71437_ced48_audio8_read3 import (
    A, C, ROUTE, T, function,
)
import test_mizar_s1_text71437_ced48_audio8_read3 as core_tests


class ExecutionTests(unittest.TestCase):
    def fixture(self, closed=False):
        driver = core_tests.RoutingSourceTests()
        obj, x, pair, scope = driver.fixture()
        rng = np.random.default_rng(789)
        projection_scope = dict(F=types.SimpleNamespace(
            linear=lambda value, weight, bias=None: value @ weight.T + (0 if bias is None else bias)))
        module = A.tree(ROUTE / "models/loop_lora.py")
        owner = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "LoopLoRALinear")
        A.execute([A.find_function(owner, "forward")], projection_scope)
        forward = projection_scope["forward"]
        targets = C.LOOP_LORA_CONTRACT["targets"]
        scope2 = dict(TARGETS=targets)
        for name in ("projections", "call_with_loop_lora"):
            A.execute([A.find_function(module, name)], scope2)

        class Layer:
            def __init__(self, index):
                self.index = index
                self.ps = {}
                if 7 <= index < 21:
                    for target in targets:
                        self.ps[target] = types.SimpleNamespace(
                            weight=np.eye(3) * .995, bias=np.full(3, .0001),
                            scaling=2., loop_index=None,
                            lora_A=[rng.normal(0, .04, (8, 3)) for _ in range(3)],
                            lora_B=[rng.normal(0, 0. if closed else .01, (3, 8)) for _ in range(3)])

            def get_submodule(self, name):
                return self.ps[name]

            def __call__(self, hidden):
                for projection in self.ps.values():
                    obj.projection_calls.append((self.index, projection.loop_index))
                    hidden = forward(projection, hidden)
                return T(hidden + (self.index + 1) * .001)

        obj.layers = [Layer(i) for i in range(28)]
        obj.projection_calls = []
        obj.events = []
        stack_scope = dict(call_with_loop_lora=scope2["call_with_loop_lora"],
                           _call_layer=lambda layer, hidden, **kwargs: (layer(hidden),))
        stack = function("models/recursive_audio8_read3.py", "_run_stack", stack_scope)

        def run(hidden, indices, start, **kwargs):
            obj.events.append((start, hidden.copy()))
            return stack(obj, hidden, indices, start, **kwargs)
        obj._run_stack = run
        return driver, obj, x, pair, scope, forward, scope2["call_with_loop_lora"]

    def reference(self, obj, x, pair, lora=True):
        def stack(h, indices, loop=None):
            for index in indices:
                for p in obj.layers[index].ps.values():
                    old = h
                    h = old @ p.weight.T + p.bias
                    if lora:
                        h = h + 2 * (old @ p.lora_A[loop].T @ p.lora_B[loop].T)
                h = h + (index + 1) * .001
            return h

        def softmax(x):
            x = x - x.max(axis=-1, keepdims=True)
            x = np.exp(x)
            return x / x.sum(axis=-1, keepdims=True)
        ma = np.concatenate((x[:, None, :126], np.zeros((2, 5, 126, 3)), np.stack(pair, axis=1)), axis=1)
        mt = np.concatenate((x[:, None, 126:], np.zeros((2, 5, 5, 3))), axis=1)
        h = x.copy()
        queries = []
        for index in range(4):
            value = stack(h, range(7) if index == 0 else range(7, 21), None if index == 0 else index - 1)
            query = value if index == 0 else h
            queries.append(query.copy())
            wr = obj.write_routers[index]
            w = softmax(query @ wr.weight.T + wr.bias)
            ma[:, :6] += value[:, None, :126] * w[:, :126].swapaxes(1, 2)[..., None]
            mt += value[:, None, 126:] * w[:, 126:].swapaxes(1, 2)[..., None]
            ar = obj.audio_read_routers[index] if index < 3 else obj.read_routers[index]
            r = softmax(query[:, :126] @ ar.weight.T + ar.bias)
            ah = (ma[:, :r.shape[-1]] * r.swapaxes(1, 2)[..., None]).sum(axis=1)
            tr = obj.read_routers[index]
            r = softmax(query[:, 126:] @ tr.weight.T + tr.bias)
            th = (mt * r.swapaxes(1, 2)[..., None]).sum(axis=1)
            h = np.concatenate((ah, th), axis=1)
        return stack(h, range(21, 28)), queries

    def test_actual_56_layers_294_projection_calls_and_independent_recurrence(self):
        driver, obj, x, pair, scope, _, _ = self.fixture()
        actual, memory = driver.forward(obj, x, pair, scope)
        expected, queries = self.reference(obj, x, pair)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(len(obj.last_forward_trace), 56)
        self.assertEqual(len(obj.projection_calls), 294)
        self.assertEqual([v[0] for v in obj.events], [0, 7, 21, 35, 49])
        for loop in range(3):
            self.assertEqual(sum(index == loop for _, index in obj.projection_calls), 98)
            suffix = "pre" if loop == 0 else str(loop - 1)
            np.testing.assert_allclose(obj.last_router_queries["audio_read_" + suffix], queries[loop][:, :126],
                                       rtol=1e-5, atol=1e-5)
        self.assertTrue(all(p.loop_index is None for layer in obj.layers for p in layer.ps.values()))
        np.testing.assert_array_equal(memory[0][:, 6:], np.stack(pair, axis=1))

    def test_zero_B_matches_same_audio8_recurrence_without_adapter_delta(self):
        driver, obj, x, pair, scope, _, _ = self.fixture(closed=True)
        actual, _ = driver.forward(obj, x, pair, scope)
        expected, _ = self.reference(obj, x, pair, lora=False)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_third_lora_perturbation_changes_output_and_preserves_other_parameters(self):
        driver, obj, x, pair, scope, _, _ = self.fixture()
        before, _ = driver.forward(obj, x, pair, scope)
        p = obj.layers[7].ps[C.LOOP_LORA_CONTRACT["targets"][0]]
        first, second = p.lora_B[0].copy(), p.lora_B[1].copy()
        p.lora_B[2] += .02
        after, _ = driver.forward(obj, x, pair, scope)
        self.assertGreater(np.max(abs(before - after)), 1e-8)
        np.testing.assert_array_equal(first, p.lora_B[0])
        np.testing.assert_array_equal(second, p.lora_B[1])

    def test_stack_rejects_wrong_loop_ranges_and_scope_always_resets(self):
        _, obj, x, _, _, _, call = self.fixture()
        kw = dict(attention_mask=None, position_ids=None, cache=None, use_cache=False,
                  cache_position=None, position_embeddings=None, output_attentions=False, all_attentions=[])
        obj.last_forward_trace, obj.last_loop_lora_trace = [], ()
        for loop, start in ((None, 7), (0, 35), (2, 21), (3, 49)):
            with self.assertRaises(ValueError):
                obj._run_stack(x, range(7, 21), start, loop_index=loop, **kw)
        for loop in range(3):
            def fail(*args, **kwargs):
                raise RuntimeError("synthetic decoder failure")
            with self.assertRaises(RuntimeError):
                call(obj.layers[7], loop, fail, x)
            self.assertTrue(all(p.loop_index is None for p in obj.layers[7].ps.values()))

    def test_zero_B_finite_difference_and_third_round_gradient_isolation(self):
        _, obj, x, _, _, forward, _ = self.fixture(closed=True)
        p = obj.layers[7].ps[C.LOOP_LORA_CONTRACT["targets"][0]]
        p.loop_index = 2
        def loss():
            return (forward(p, x[:, :3]) ** 2).sum()
        def diff(matrix):
            old, eps = matrix[0, 0], 1e-5
            matrix[0, 0] = old + eps
            plus = loss()
            matrix[0, 0] = old - eps
            minus = loss()
            matrix[0, 0] = old
            return (plus - minus) / (2 * eps)
        self.assertEqual(diff(p.lora_A[2]), 0.)
        self.assertNotEqual(diff(p.lora_B[2]), 0.)
        self.assertEqual(diff(p.lora_B[0]), 0.)
        p.lora_B[2][0, 0] = .1
        self.assertNotEqual(diff(p.lora_A[2]), 0.)


class OptimizerTests(unittest.TestCase):
    def test_actual_resume_branch_preserves_third_lora_read_and_rank_specific_state(self):
        saved, config, geometry = core_tests.valid_resume()
        config["resume_checkpoint"] = "synthetic_full_checkpoint"
        saved["state_dict"] = {
            "caption_decoder.lm.model.audio_read_routers.2.weight": T(np.ones((8, 3))),
            "caption_decoder.lm.model.layers.7.self_attn.q_proj.lora_B.2": T(np.ones((3, 8))),
        }
        saved["random_state_by_rank"] = [dict(python=i, numpy=i+10, torch=i+20, cuda=i+30) for i in range(8)]
        fn = A.find_function(A.tree(ROUTE / "train.py"), "train")
        branch = next(n for n in fn.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "config['resume_checkpoint']")

        class State:
            def load_state_dict(self, value):
                self.loaded = copy.deepcopy(value)
                self.last_epoch = value.get("last_epoch", 0)

        class Model:
            def state_dict(self):
                return {name: T(np.zeros(value.shape)) for name, value in saved["state_dict"].items()}
            def load_state_dict(self, state, strict):
                assert strict
                self.loaded = copy.deepcopy(state)

        def execute(payload):
            model, optimizer, scheduler, scaler, tracker = Model(), State(), State(), State(), State()
            scope = dict(config=config, geometry=geometry, groups=saved["optimizer_inventory"], rank=3,
                         model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, tracker=tracker,
                         audio_core=types.SimpleNamespace(audio_gradient_state=None), EPOCHS=3,
                         validate_resume_payload=C.validate_resume_payload,
                         torch=types.SimpleNamespace(Tensor=core_tests.Tensor, isfinite=np.isfinite,
                                                     load=lambda *a, **k: copy.deepcopy(payload)))
            A.execute([copy.deepcopy(branch)], scope)
            return model, scope

        model, scope = execute(saved)
        for name, value in saved["state_dict"].items():
            np.testing.assert_array_equal(model.loaded[name], value)
        self.assertEqual((scope["step"], scope["start_epoch"], scope["start_batch"]), (2, 0, 2))
        self.assertEqual(scope["resume_rng"], saved["random_state_by_rank"][3])
        self.assertEqual(scope["audio_core"].audio_gradient_state, saved["audio_gradient_state"])
        for component, key in (("optimizer", "optimizer"), ("scheduler", "scheduler"),
                               ("scaler", "grad_scaler"), ("tracker", "grad_norm_tracker")):
            self.assertEqual(scope[component].loaded, saved[key])
        for failure in ("inventory", "shape", "finite", "route"):
            payload = copy.deepcopy(saved)
            if failure == "inventory":
                payload["optimizer_inventory"] = list(reversed(payload["optimizer_inventory"]))
            elif failure == "shape":
                key = next(iter(payload["state_dict"]))
                payload["state_dict"][key] = T(np.zeros((7, 3)))
            elif failure == "finite":
                next(iter(payload["state_dict"].values())).fill(np.nan)
            else:
                payload["route_variant"] = "old_read2_route"
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                execute(payload)

    def test_real_group_builder_includes_588_matrices_13_roles_and_entire_schedule(self):
        class Parameter:
            requires_grad = True
            def __init__(self, ndim, count):
                self.ndim, self.count = ndim, count
            def numel(self):
                return self.count
        params = []
        for prefix in ("audio_encoder.base.mapper", "caption_decoder.lm.model.layers.0",
                       "audio_encoder.base.layer_mlps.4", "audio_encoder.base.layer_mlps.8",
                       *(f"caption_decoder.lm.model.audio_read_routers.{i}" for i in range(3))):
            for suffix, ndim, count in ((".weight", 2, 24), (".bias", 1, 8)):
                params.append((prefix + suffix, Parameter(ndim, count)))
        shapes = {"q_proj": (576, 576), "k_proj": (576, 192), "v_proj": (576, 192),
                  "o_proj": (576, 576), "gate_proj": (576, 1536), "up_proj": (576, 1536), "down_proj": (1536, 576)}
        for layer in range(7, 21):
            for target in C.LOOP_LORA_CONTRACT["targets"]:
                inputs, outputs = shapes[target.rsplit(".", 1)[1]]
                for loop in range(3):
                    for kind, count in (("A", 8 * inputs), ("B", 8 * outputs)):
                        params.append((f"caption_decoder.lm.model.layers.{layer}.{target}.lora_{kind}.{loop}", Parameter(2, count)))
        class AdamW:
            def __init__(self, groups, **kwargs):
                self.param_groups, self.kwargs = groups, kwargs
        scope = dict(torch=types.SimpleNamespace(optim=types.SimpleNamespace(
            AdamW=AdamW, lr_scheduler=types.SimpleNamespace(LambdaLR=lambda optimizer, fn: fn))), lr_factor=C.lr_factor)
        tree = A.tree(ROUTE / "optim.py")
        A.execute([n for n in tree.body if isinstance(n, ast.Assign)], scope)
        role = function("optim.py", "parameter_role", scope)
        build = function("optim.py", "build_optimizer", scope)
        optimizer, schedule, inventory = build(types.SimpleNamespace(named_parameters=lambda: params), 34626)
        self.assertEqual(len({entry["role"] for entry in inventory}), 13)
        lora = [entry for entry in inventory if "lora" in entry["role"]]
        self.assertEqual(sum(len(entry["names"]) for entry in lora), 588)
        self.assertEqual(sum(entry["numel"] for entry in lora), 3419136)
        for entry in inventory:
            self.assertEqual(entry["base_lr"], 3e-4 if entry["role"] == "default" else 6e-4)
            self.assertTrue(all(entry["weight_decay"] == (0. if name.endswith(".bias") else .01) for name in entry["names"]))
        for step in range(34627):
            expected = max(.1, (step + 1) / 1731) if step < 1731 else .1 + .45 * (1 + math.cos(math.pi * (step - 1731) / (34626 - 1731)))
            self.assertAlmostEqual(schedule(step), expected, places=14)
        self.assertEqual(optimizer.kwargs["betas"], (.9, .999))
        for name in ("caption_decoder.lm.model.layers.6.self_attn.q_proj.lora_A.2",
                     "caption_decoder.lm.model.layers.7.self_attn.q_proj.lora_A.3"):
            with self.assertRaises(ValueError):
                role(name)


if __name__ == "__main__":
    unittest.main()
