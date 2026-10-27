"""Stdlib checks for training parity and strict MeSH loading, without real weights."""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine"
ROUTE = FAMILY / "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_usad2_small"
sys.path.insert(0, str(ROUTE))
import audit_reference
import contract


def extract_loader_functions(*names, **scope):
    tree = ast.parse((ROUTE / "models/recursive_text.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(Path=Path, Any=Any, TEXT_MODEL_CONTRACT=contract.TEXT_MODEL_CONTRACT,
                     TEXT_MODEL_ARCHITECTURE=contract.TEXT_MODEL_ARCHITECTURE,
                     TEXT_MODEL_SCHEDULE=contract.TEXT_MODEL_SCHEDULE, **scope)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "actual-mesh-loader", "exec"), namespace)
    return namespace


def model_config():
    return SimpleNamespace(hidden_size=576, num_hidden_layers=30, recursive_layer_count=20,
                           recursive_loops=2, mesh_memory_slots=5, mesh_router_count=6,
                           mesh_architecture_contract=contract.TEXT_MODEL_CONTRACT,
                           model_architecture_contract=contract.TEXT_MODEL_ARCHITECTURE,
                           logical_to_physical=list(contract.TEXT_MODEL_SCHEDULE))


class MeSHUSAD2ParityTests(unittest.TestCase):
    def test_original_mesh_semantics_and_ordinary_usad2_recipe_match(self):
        original = audit_reference.audit_parent()
        baseline = audit_reference.audit_usad2_baseline()
        self.assertEqual(len(original["byte_identical_files"]), 9)
        self.assertEqual(len(baseline["byte_identical_files"]), 16)
        self.assertEqual(baseline["optimizer_epoch_loop"], "identical including one-time gradient diagnostic")

    def test_mapper_build_order_and_first_gradient_checks_include_all_routers(self):
        base = FAMILY / "mizar_stage1_fixed20_gbs256_3epochs_usad2_small"
        self.assertEqual((ROUTE / "models/mellow.py").read_bytes(), (base / "models/mellow.py").read_bytes())
        mellow = (ROUTE / "models/mellow.py").read_text(encoding="utf-8")
        self.assertLess(mellow.index("self.audio_encoder = AudioEncoder("),
                        mellow.index("self.caption_decoder = get_decoder("))
        train = (ROUTE / "train.py").read_text(encoding="utf-8")
        for name in ("mapper_initial_sha256", "mapper_start_sha256", "mapper_final_sha256",
                     "nonzero_by_router", "mesh_forward_audit", "usad2_unchanged"):
            self.assertIn(name, train)
        self.assertIn("restore_rng(comparison_rng, device)", train)
        self.assertNotIn("torchlibrosa", (ROUTE / "write_config.py").read_text(encoding="utf-8"))

    def test_strict_local_loader_options_and_all_loading_mismatches(self):
        for bad_key in (None, "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            with self.subTest(bad_key=bad_key), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "config.json").write_text("{}")
                sentinel = object()
                loader = Mock(return_value=(sentinel, {} if bad_key is None else {bad_key: ["synthetic mismatch"]}))
                register, auditor = Mock(), Mock()
                scope = extract_loader_functions(
                    "load_recursive_text_model", torch=SimpleNamespace(float32="FP32"),
                    _AUTO_CLASS_REGISTERED=False,
                    _import_recursive_model=lambda: (SimpleNamespace(from_pretrained=loader), Mock(), register),
                    audit_loaded_recursive_model=auditor)
                call = lambda: scope["load_recursive_text_model"](path)
                if bad_key:
                    with self.assertRaisesRegex(RuntimeError, "not exact"):
                        call()
                    auditor.assert_not_called()
                else:
                    self.assertIs(call(), sentinel)
                    auditor.assert_called_once_with(sentinel)
                register.assert_called_once()
                loader.assert_called_once_with(path.resolve(), local_files_only=True, torch_dtype="FP32",
                                               output_loading_info=True, ignore_mismatched_sizes=False)

    def test_loaded_model_rejects_shared_physical_objects_or_persistent_memory(self):
        expected = dict(logical_layer_count=30, physical_layer_count=20, memory_slots=5, router_count=6,
                        schedule=list(contract.TEXT_MODEL_SCHEDULE), router_objects_independent=True,
                        router_parameter_tensor_count=12)
        scope = extract_loader_functions(
            "validate_recursive_config", "audit_loaded_recursive_model",
            _import_recursive_model=lambda: (None, lambda model: expected, None))
        core = SimpleNamespace(layers=[object() for _ in range(20)],
                               write_routers=[object() for _ in range(3)],
                               read_routers=[object() for _ in range(3)])
        model = SimpleNamespace(config=model_config(), model=core, state_dict=lambda: {})
        self.assertEqual(scope["audit_loaded_recursive_model"](model), expected)
        core.layers[1] = core.layers[0]
        with self.assertRaisesRegex(ValueError, "independent physical"):
            scope["audit_loaded_recursive_model"](model)
        core.layers[1] = object()
        model.state_dict = lambda: {"model.saved_memory": object()}
        with self.assertRaisesRegex(ValueError, "transient"):
            scope["audit_loaded_recursive_model"](model)

    def test_tokenizer_and_recursive_checkpoint_validation_still_reject_wrong_geometry(self):
        scope = extract_loader_functions("validate_recursive_config")
        scope["validate_recursive_config"](model_config())
        for key, value in (("recursive_loops", 4), ("mesh_memory_slots", 7),
                           ("mesh_router_count", 8), ("hidden_size", 384)):
            config = model_config()
            setattr(config, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                scope["validate_recursive_config"](config)

    def test_audit_report_matches_current_route_and_shared_recursive_source(self):
        report = json.loads((ROUTE / "reference_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["route_code_sha256"], contract.code_digest())
        self.assertEqual(report["shared_recursive_model_sha256"], contract.recursive_model_source_sha256())
        self.assertEqual(report["scheduler_positions"], 17314)
        self.assertEqual(report["scheduler_max_abs_error"], 0.)


if __name__ == "__main__":
    unittest.main()
