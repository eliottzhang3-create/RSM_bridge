"""Execute production loader/audits/control flow with explicit CPU fixtures.

NumPy tensors and simulated decoder layers do not claim torch autograd, CED
decoding or GPU validation. Real small-model checks live in the tensor suite.
"""
import ast
import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71437_mesh_ced_small"
sys.path.insert(0, str(ROUTE))
import contract as C
import audit_reference as A
from models import recursive_text as L
from test_mizar_s1_text71437_ced_small_route import checkpoint


class Array:
    def __init__(self, value, dtype=None): self.value = np.asarray(value, dtype=dtype)
    @property
    def shape(self): return self.value.shape
    @property
    def ndim(self): return self.value.ndim
    @property
    def dtype(self): return self.value.dtype
    device = "cpu"
    requires_grad = False
    def __getitem__(self, key): return Array(self.value[key])
    def __setitem__(self, key, value): self.value[key] = value.value if isinstance(value, Array) else value
    def __add__(self, other): return Array(self.value + (other.value if isinstance(other, Array) else other))
    def __mul__(self, other): return Array(self.value * (other.value if isinstance(other, Array) else other))
    def unsqueeze(self, dim): return Array(np.expand_dims(self.value, dim))
    def expand(self, *shape): return Array(np.broadcast_to(self.value, tuple(old if new == -1 else new for old, new in zip(self.shape, shape))))
    def transpose(self, a, b): return Array(np.swapaxes(self.value, a, b))
    def sum(self, dim=None): return Array(self.value.sum(axis=dim))
    def float(self): return Array(self.value.astype(np.float32))
    def to(self, device=None, dtype=None): return Array(self.value.astype(dtype or self.dtype))
    def all(self): return bool(self.value.all())
    def any(self): return bool(self.value.any())
    def ne(self, value): return Array(self.value != value)
    def numel(self): return self.value.size
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.value


def softmax(value):
    value = value - value.max(axis=-1, keepdims=True)
    exp = np.exp(value)
    return exp / exp.sum(axis=-1, keepdims=True)


TORCH = SimpleNamespace(float32=np.dtype("float32"),
    equal=lambda a, b: np.array_equal(a.value, b.value),
    isfinite=lambda a: Array(np.isfinite(a.value)),
    zeros=lambda shape, dtype, device: Array(np.zeros(shape, dtype=dtype)),
    arange=lambda a, b, device: Array(np.arange(a, b)))


class BehaviorTests(unittest.TestCase):
    def test_original_tracked_sources_and_optimizer_bytes_are_unchanged(self):
        snapshot = json.loads((ROUTE / "protected_sources.json").read_text(encoding="utf-8"))
        self.assertGreater(len(snapshot["sha256_normalized_lf"]), 3000)
        for name, expected in snapshot["sha256_normalized_lf"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), expected, name)
        parent = ROUTE.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small")
        self.assertEqual((parent / "optim.py").read_bytes(), (ROUTE / "optim.py").read_bytes())

    def test_source_final_step_complete_marker_and_all_schedule_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            source, config = checkpoint(Path(directory))
            for alias in ("logical_to_physical", "recursive_logical_to_physical", "logical_to_physical_schedule", "recursive_logical_to_physical_schedule"):
                C.write_json(source / "config.json", dict(config, **{alias: list(range(56))}))
                with self.assertRaises(ValueError): C.recursive_checkpoint_contract(source)
            C.write_json(source / "config.json", config)
            partial = source.with_name("checkpoint-001200")
            source.rename(partial)
            for name in ("checkpoint_complete.json", "checkpoint_manifest.json", "mesh_checkpoint_metadata.json"):
                raw = json.loads((partial / name).read_text())
                raw["optimizer_step"] = 1200
                C.write_json(partial / name, raw)
            with self.assertRaisesRegex(ValueError, "step 9244"): C.recursive_checkpoint_contract(partial)

    def fixture_state(self):
        embedding = Array(np.ones((4, 3), np.float32))
        state = {"model.embed_tokens.weight": embedding, "lm_head.weight": embedding}
        for group in range(4):
            for kind in ("write", "read"):
                state[f"model.{kind}_routers.{group}.weight"] = Array(np.full((6, 3), .01 * (group + 1), np.float32))
                state[f"model.{kind}_routers.{group}.bias"] = Array(np.full((6,), .02 * (group + 1), np.float32))
        return state

    def audit_inventory(self, state, payloads):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = {}
            for index, values in enumerate(payloads):
                path = root / f"model-{index}.safetensors"
                path.write_bytes(b"explicit non-tensor fixture")
                shards[str(path)] = values
            class Handle:
                def __init__(self, path, **kwargs): self.values = shards[path]
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def keys(self): return self.values.keys()
                def get_tensor(self, key): return self.values[key]
            with patch.dict(sys.modules, {"torch": TORCH, "safetensors": SimpleNamespace(safe_open=Handle)}):
                return L.audit_checkpoint_tensors(SimpleNamespace(state_dict=lambda: state), root)

    def test_actual_loader_inventory_accepts_all_sixteen_routers_across_shards_and_one_tied_alias(self):
        state = self.fixture_state()
        names = [key for key in state if key != "lm_head.weight"]
        audit = self.audit_inventory(state, [{k: state[k] for k in names[::2]}, {k: state[k] for k in names[1::2]}])
        self.assertEqual(audit["router_tensors"], 16)
        self.assertEqual(audit["omitted_tied_aliases"], ["lm_head.weight"])

    def test_actual_loader_inventory_rejects_missing_duplicate_extra_dtype_nonfinite_shape_and_tied_conflict(self):
        state = self.fixture_state()
        for failure in ("missing", "duplicate", "extra", "dtype", "nonfinite", "shape", "tied"):
            with self.subTest(failure=failure):
                payload = copy.deepcopy(state)
                key = "model.read_routers.3.weight"
                if failure == "missing": del payload[key]
                elif failure == "extra": payload["extra"] = Array([1.], np.float32)
                elif failure == "dtype": payload[key] = Array(payload[key].value, np.float16)
                elif failure == "nonfinite": payload[key].value[0, 0] = np.nan
                elif failure == "shape": payload[key] = Array(np.zeros((5, 3), np.float32))
                elif failure == "tied": payload["lm_head.weight"] = Array(np.zeros((4, 3), np.float32))
                shards = [payload, {key: payload[key]}] if failure == "duplicate" else [payload]
                with self.assertRaises(ValueError): self.audit_inventory(state, shards)

    def simulated_backbone(self):
        module = A.tree(ROOT / "code/RSmol/recursive_model_mizar_text_7_14x3_7_mesh.py")
        owner = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "MeshLlamaModel")
        methods = [copy.deepcopy(node) for node in owner.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"forward", "_route", "_write", "_read", "_run_stack"}]
        scope = dict(torch=TORCH, F=SimpleNamespace(softmax=lambda x, dim: Array(softmax(x.value))),
            _cfg=lambda config, key, default: getattr(config, key, default), _cache_seq_length=lambda cache: 0,
            _causal_mask=lambda *args, **kwargs: None, record_function=lambda *args: nullcontext(),
            BaseModelOutputWithPast=SimpleNamespace, PREFIX_LAYER_COUNT=7, MIDDLE_LAYER_COUNT=14,
            PHYSICAL_LAYER_COUNT=28, RECURSIVE_LOOPS=3, MEMORY_SLOT_COUNT=6,
            _call_layer=lambda layer, hidden, **kwargs: (Array(np.tanh(hidden.value @ layer[0] + layer[1])),))
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        A.execute([future, ast.ClassDef(name="Backbone", bases=[], keywords=[], body=methods, decorator_list=[])], scope)
        model = scope["Backbone"]()
        rng = np.random.default_rng(42)
        model.config = SimpleNamespace()
        model.layers = [(rng.normal(0, .2, (4, 4)).astype(np.float32), rng.normal(0, .1, (4,)).astype(np.float32)) for _ in range(28)]
        model.weights = [rng.normal(0, .2, (6, 4)).astype(np.float32) for _ in range(8)]
        model.biases = [rng.normal(0, .1, (6,)).astype(np.float32) for _ in range(8)]
        model.write_routers = [lambda x, index=i: Array(x.value @ model.weights[index].T + model.biases[index]) for i in range(4)]
        model.read_routers = [lambda x, index=i: Array(x.value @ model.weights[4+index].T + model.biases[4+index]) for i in range(4)]
        model.audit_mode, model.gradient_audit_mode, model.routing_stats_mode = True, False, False
        model.rotary_emb = lambda hidden, position_ids: None
        model.norm = lambda hidden: hidden
        return model

    def test_actual_forward_control_flow_matches_independent_six_slot_three_loop_recurrence(self):
        model = self.simulated_backbone()
        embedding = np.random.default_rng(4).normal(size=(1, 683, 4)).astype(np.float32)
        def stack(value, indices):
            for index in indices:
                weight, bias = model.layers[index]
                value = np.tanh(value @ weight + bias)
            return value
        expected_memory = np.zeros((1, 6, 683, 4), np.float32)
        expected_memory[:, 0] = embedding
        value = stack(embedding, range(7))
        queries = [value.copy()]
        for group in range(4):
            if group:
                queries.append(hidden.copy())
                value = stack(hidden, range(7, 21))
            query = queries[-1]
            write = softmax(query @ model.weights[group].T + model.biases[group])
            read = softmax(query @ model.weights[4+group].T + model.biases[4+group])
            expected_memory = expected_memory + value[:, None] * np.swapaxes(write, 1, 2)[..., None]
            hidden = (expected_memory * np.swapaxes(read, 1, 2)[..., None]).sum(axis=1)
        expected = stack(hidden, range(21, 28))
        result = model.forward(inputs_embeds=Array(embedding), use_cache=False)
        np.testing.assert_allclose(result.last_hidden_state.value, expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(model.last_memory_shape, (1, 6, 683, 4))
        self.assertEqual([item["physical_index"] for item in model.last_forward_trace], list(C.TEXT_MODEL_SCHEDULE))
        self.assertEqual(model.last_forward_trace[49], {"logical_index": 49, "physical_index": 21})
        self.assertEqual(len(model.last_memory_write_history), 4)
        for group, suffix in enumerate(("pre", "0", "1", "2")):
            np.testing.assert_array_equal(model.last_router_queries[f"write_{suffix}"].value, queries[group])
            np.testing.assert_array_equal(model.last_router_queries[f"read_{suffix}"].value, queries[group])
        np.testing.assert_allclose(model.last_memory.value, expected_memory, rtol=1e-6, atol=1e-7)
        # A second full forward must start from its own embeddings and five zero slots.
        model.forward(inputs_embeds=Array(embedding * 2), use_cache=False)
        np.testing.assert_array_equal(model.last_initial_memory.value[:, 0], embedding * 2)
        self.assertFalse(model.last_initial_memory.value[:, 1:].any())

    def test_forward_and_backward_audits_require_56_calls_three_boundaries_and_release_refs(self):
        scope = dict(torch=TORCH, TEXT_MODEL_SCHEDULE=C.TEXT_MODEL_SCHEDULE, PREFIX_TOKENS=383, ANSWER_TOKENS=300)
        A.execute([A.find_function(A.tree(ROUTE / "train.py"), name) for name in
                   ("recursive_forward_audit", "recursive_backward_audit", "router_gradient_audit")], scope)
        model = self.simulated_backbone()
        model.forward(inputs_embeds=Array(np.ones((1, 683, 4), np.float32)), use_cache=False)
        lm = SimpleNamespace(model=model, config=SimpleNamespace(hidden_size=4))
        wrapper = SimpleNamespace(caption_decoder=SimpleNamespace(lm=lm))
        def boundary():
            tensor = Array(np.ones((1, 683, 4), np.float32))
            tensor.grad = Array(np.ones((1, 683, 4), np.float32))
            return tensor
        model.last_core_input_refs, model.last_core_output_refs = [boundary() for _ in range(3)], [boundary() for _ in range(3)]
        result = scope["recursive_backward_audit"](wrapper, 1)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["boundary_gradients"]), 6)
        self.assertFalse(model.gradient_audit_mode)
        self.assertEqual(model.last_core_input_refs, [])
        model.last_core_input_refs, model.last_core_output_refs = [boundary() for _ in range(2)], [boundary() for _ in range(3)]
        with self.assertRaisesRegex(ValueError, "three"): scope["recursive_backward_audit"](wrapper, 1)
        model.last_forward_trace[-1]["physical_index"] = 0
        with self.assertRaisesRegex(ValueError, "complete"): scope["recursive_forward_audit"](wrapper, 1)

    def test_router_audit_includes_the_third_loop_weight_and_bias_gradients(self):
        scope = dict(torch=TORCH)
        A.execute([A.find_function(A.tree(ROUTE / "train.py"), "router_gradient_audit")], scope)
        parameters = self.fixture_state()
        parameters = {name: value for name, value in parameters.items() if "_routers." in name}
        for value in parameters.values(): value.grad = Array(np.ones_like(value.value))
        wrapper = SimpleNamespace(caption_decoder=SimpleNamespace(lm=SimpleNamespace(named_parameters=lambda: parameters.items())))
        self.assertTrue(scope["router_gradient_audit"](wrapper)["passed"])
        target = parameters["model.read_routers.3.bias"]
        for bad in (None, Array(np.zeros(6, np.float32)), Array(np.full(6, np.nan, np.float32))):
            target.grad = bad
            self.assertFalse(scope["router_gradient_audit"](wrapper)["passed"])

    def resume_fixture(self):
        prepared = dict(rows=1477492, source_rows=C.SOURCE_COUNTS, rows_canonical_sha256="rows", mapping_sha256="mapping")
        contract = C.semantic_contract(prepared, C.DEFAULT_TEXT_MODEL, "synthetic-ced")
        config = {"contract": contract, "contract_sha256": C.digest(contract), "resume_checkpoint": "synthetic.ckpt"}
        saved = dict(schema_version=1, route_variant=C.ROUTE, mode="formal", contract=contract,
            contract_sha256=config["contract_sha256"], state_dict={"learned_router": "retained"},
            optimizer={"moments": "retained"}, scheduler={"last_epoch": 5771}, grad_scaler={"scale": 65536},
            grad_norm_tracker={"running": "retained"}, total_step=5771, epoch=0, next_optimizer_step=5771,
            num_epochs=3, batch_geometry=dict(world_size=8, microbatch_size=4, gradient_accumulation_steps=4, global_batch_size=128),
            loss_reduction="mean_of_rank_local_microbatch_token_means_then_DDP_mean",
            random_state_by_rank=[dict(python=i, numpy=i+10, torch=i+20, cuda=i+30) for i in range(8)])
        return config, saved

    def test_resume_rejects_wrong_route_partial_state_rng_batch_scheduler_cursor_source_and_completed_run(self):
        config, saved = self.resume_fixture()
        C.validate_resume_metadata(saved, config)
        cases = [dict(route_variant="old"), dict(contract_sha256="old"), dict(contract={}),
                 dict(batch_geometry={}), dict(num_epochs=1), dict(loss_reduction="global_token_mean"),
                 dict(random_state_by_rank=saved["random_state_by_rank"][:7]),
                 dict(scheduler={"last_epoch": 5770}), dict(next_optimizer_step=5770), dict(total_step=True),
                 dict(total_step=34626, epoch=3, next_optimizer_step=0)]
        for update in cases:
            with self.subTest(update=update), self.assertRaises(ValueError): C.validate_resume_metadata(dict(saved, **update), config)
        for key in ("state_dict", "optimizer", "grad_scaler", "grad_norm_tracker"):
            bad = dict(saved); del bad[key]
            with self.assertRaises(ValueError): C.validate_resume_metadata(bad, config)

    def test_actual_resume_branch_restores_full_audio_state_and_rank_specific_rng(self):
        config, saved = self.resume_fixture()
        class Receiver:
            def load_state_dict(self, state, **kwargs): self.loaded = state; self.last_epoch = state.get("last_epoch", None)
        model, optimizer, scheduler, scaler, tracker = [Receiver() for _ in range(5)]
        scope = dict(config=config, torch=SimpleNamespace(load=lambda *args, **kwargs: copy.deepcopy(saved)),
            validate_resume_metadata=C.validate_resume_metadata, ROUTE=C.ROUTE, EPOCHS=3, WORLD_SIZE=8,
            geometry=config["contract"]["geometry"], rank=3, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, tracker=tracker, step=0, start_epoch=0, start_batch=0, resume_rng=None)
        node = next(n for n in A.find_function(A.tree(ROUTE / "train.py"), "train").body
                    if isinstance(n, ast.If) and ast.unparse(n.test) == "config['resume_checkpoint']")
        A.execute([node], scope)
        self.assertEqual((scope["step"], scope["start_epoch"], scope["start_batch"]), (5771, 0, 5771))
        self.assertEqual(scope["resume_rng"], saved["random_state_by_rank"][3])
        for target, key in ((model, "state_dict"), (optimizer, "optimizer"), (scheduler, "scheduler"),
                            (scaler, "grad_scaler"), (tracker, "grad_norm_tracker")):
            self.assertEqual(target.loaded, saved[key])
        config["resume_checkpoint"] = ""
        with patch.object(scope["torch"], "load", side_effect=AssertionError("fresh audio must not restore training state")):
            A.execute([node], scope)

    def test_actual_optimizer_keeps_discriminative_lr_wd_and_full_horizon(self):
        class Optimizer:
            def __init__(self, groups, **kwargs): self.param_groups, self.options = groups, kwargs
        class Scheduler:
            def __init__(self, optimizer, formula): self.formula = formula
        scope = dict(torch=SimpleNamespace(optim=SimpleNamespace(AdamW=Optimizer,
            lr_scheduler=SimpleNamespace(LambdaLR=Scheduler))), lr_factor=C.lr_factor)
        A.execute([A.find_function(A.tree(ROUTE / "optim.py"), name) for name in ("parameter_role", "build_optimizer")], scope)
        parameters = []
        for name, shape in (("audio_encoder.base.mapper.net.weight", (3, 3)), ("audio_encoder.base.mapper.net.bias", (3,)),
                            ("caption_decoder.lm.model.read_routers.3.weight", (6, 3)), ("caption_decoder.lm.model.norm.weight", (3,))):
            tensor = Array(np.ones(shape)); tensor.requires_grad = True; parameters.append((name, tensor))
        optimizer, scheduler, groups = scope["build_optimizer"](SimpleNamespace(named_parameters=lambda: parameters), 34626)
        self.assertEqual(optimizer.options["betas"], (.9, .999))
        self.assertEqual({(group["role"], group["base_lr"], group["weight_decay"]) for group in groups},
                         {("map", 6e-4, 0.), ("map", 6e-4, .01), ("default", 3e-4, 0.), ("default", 3e-4, .01)})
        self.assertEqual([scheduler.formula(step) for step in (0, 1730, 34626)], [.1, 1., .1])


if __name__ == "__main__": unittest.main()
