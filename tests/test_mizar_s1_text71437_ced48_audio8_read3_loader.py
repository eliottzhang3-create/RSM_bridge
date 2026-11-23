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
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71437_mesh_ced48_audio8_read3_lora_r8"
sys.path.insert(0, str(ROUTE))
import contract as C
import audit_reference as A
from models import recursive_text as L
from test_mizar_s1_text71437_ced48_audio8_read3_route import checkpoint


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
        self.assertTrue(A.audit()["complete_epoch_update_loop_equal"])

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


if __name__ == "__main__": unittest.main()
