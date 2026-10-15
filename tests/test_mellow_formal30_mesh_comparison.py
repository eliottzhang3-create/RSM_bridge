"""Formal30 equivalence and loader contracts without local ML packages."""

import ast
import contextlib
import copy
import gc
import importlib.util
import json
import random
import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch256_formal_30epochs_5_10x2_5_mesh"
sys.path.insert(0, str(ROUTE))
import comparison
from route_contract import MESH_TEXT, BASELINE_TEXT, ROUTE_CONTRACT, TEXT_CONTRACT, validate_mesh_config


def valid_config():
    config = {"data": {"datafiles": ["metadata.json"], "datapath": "stage"},
              "model": {"encoder": {"pretrained_audioencoder_path": "htsat"}},
              "save_dir": "checkpoints"}
    config = comparison.reference_runtime(config)
    config["model"]["decoder"]["text_decoder"] = str(MESH_TEXT.resolve())
    config.update(route_contract=ROUTE_CONTRACT, text_model_contract=TEXT_CONTRACT,
                  comparison_identity={"source": "test fixture"})
    return config


def mesh_config():
    return {"hidden_size": 576, "num_hidden_layers": 30, "recursive_layer_count": 20,
            "recursive_loops": 2, "mesh_memory_slots": 5, "mesh_router_count": 6,
            "mesh_architecture_contract": TEXT_CONTRACT,
            "logical_to_physical": list(range(15)) + list(range(5, 15)) + list(range(15, 20))}


class EquivalenceTests(unittest.TestCase):
    def test_actual_writer_output_matches_baseline(self):
        path = ROUTE / "scripts/rsmol/write_reasonaqa_runtime_config.py"
        emitted = {}
        yaml = types.ModuleType("yaml")
        yaml.safe_dump = lambda value, handle, **kwargs: emitted.update(copy.deepcopy(value))
        with patch.dict(sys.modules, yaml=yaml):
            spec = importlib.util.spec_from_file_location("_formal30_mesh_writer", path)
            writer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(writer)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / "stage"
            (stage / ".mellow_stage").mkdir(parents=True)
            (stage / ".mellow_stage/READY.json").touch()
            metadata = stage / "metadata.json"
            metadata.touch()
            argv = ["writer", "--stage-root", str(stage), "--data-json", str(metadata),
                    "--output-config", str(root / "runtime.yaml"), "--save-dir", str(root / "checkpoints")]
            # Only filesystem/remote provenance are mocked. The real writer
            # validates arguments and builds the exact dictionary under test.
            with patch.object(sys, "argv", argv), patch.object(writer, "validate_sources"), \
                 patch.object(writer, "make_identity", return_value={"fixture": True}), \
                 patch.object(Path, "is_relative_to", return_value=True), \
                 patch.object(Path, "exists", return_value=True):
                self.assertEqual(writer.main(), 0)
            self.assertEqual(comparison.audit_runtime_config(emitted)["status"], "PASS")
            self.assertEqual(emitted["data"]["tokenizer_type"], str(BASELINE_TEXT.resolve()))
            self.assertEqual(emitted["model"]["decoder"]["text_decoder"], str(MESH_TEXT.resolve()))
            for option, value in (("--max-lr", "2e-4"), ("--max-optimizer-steps", "8"),
                                  ("--num-epochs", "5"), ("--batch-size", "4")):
                with self.subTest(option=option), patch.object(sys, "argv", argv + [option, value]), \
                     patch.object(writer, "validate_sources"), self.assertRaises(ValueError):
                    writer.main()

    def test_actual_sources_match_baseline(self):
        self.assertEqual(comparison.audit_source_equivalence()["status"], "PASS")

    def test_generated_config_has_only_text_difference(self):
        self.assertEqual(comparison.audit_runtime_config(valid_config())["status"], "PASS")

    def test_training_drift_rejected(self):
        mutations = [("train", "batch_size", 4), ("train", "gradient_accumulation_steps", 8),
                     ("train", "num_epochs", 5), ("train", "num_workers", 0),
                     ("train", "max_grad_norm", 1.0), ("data", "tokenizer_type", str(MESH_TEXT)),
                     ("data", "ip_text_len", 128)]
        for section, key, value in mutations:
            with self.subTest(field=(section, key)):
                config = valid_config()
                config[section][key] = value
                with self.assertRaises(ValueError):
                    comparison.audit_runtime_config(config)
        for key, value in (("max_lr", 2e-4), ("min_lr", 1e-5), ("weight_decay", 0.1)):
            with self.subTest(field=key):
                config = valid_config()
                config["train"]["optimizer"][key] = value
                with self.assertRaises(ValueError):
                    comparison.audit_runtime_config(config)

    def test_wrong_model_and_cross_route_resume_rejected(self):
        config = valid_config()
        config["model"]["decoder"]["text_decoder"] = str(BASELINE_TEXT)
        with self.assertRaises(ValueError):
            comparison.audit_runtime_config(config)
        config = valid_config()
        checkpoint = {field: copy.deepcopy(config[field]) for field in
                      ("route_contract", "text_model_contract", "comparison_identity")}
        comparison.validate_resume_identity(checkpoint, config)
        checkpoint["comparison_identity"]["source"] = "another checkpoint"
        with self.assertRaises(ValueError):
            comparison.validate_resume_identity(checkpoint, config)

    def test_recursive_geometry_rejected(self):
        validate_mesh_config(mesh_config())
        for field, value in (("mesh_memory_slots", 7), ("recursive_loops", 4),
                             ("recursive_layer_count", 30), ("mesh_router_count", 10),
                             ("model_architecture_contract", "another_model"),
                             ("logical_to_physical", list(range(30)))):
            config = mesh_config()
            config[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_mesh_config(config)

    def test_semantic_trainer_mutation_detected(self):
        trainer = comparison.ROOT / "training/trainer.py"
        original = Path.read_text
        def altered(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            return value.replace("max_norm=max_grad_norm", "max_norm=1.0") if path == trainer else value
        with patch.object(Path, "read_text", altered), self.assertRaisesRegex(ValueError, "trainer differs"):
            comparison.audit_source_equivalence()

    def test_overlay_import_resolution(self):
        # models/data are namespace packages; training is explicitly copied.
        search = [str(ROUTE), str(comparison.BASE)]
        from importlib.machinery import PathFinder
        models = PathFinder.find_spec("models", search)
        self.assertEqual(Path(PathFinder.find_spec("decoder", models.submodule_search_locations).origin),
                         ROUTE / "models/decoder.py")
        self.assertEqual(Path(PathFinder.find_spec("htsat", models.submodule_search_locations).origin),
                         comparison.BASE / "models/htsat.py")
        data = PathFinder.find_spec("data", search)
        self.assertEqual(Path(PathFinder.find_spec("audiotext_dataset", data.submodule_search_locations).origin),
                         ROUTE / "data/audiotext_dataset.py")


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.cpu = 10
        self.cuda = 20
        self.numpy_state = 30
        self.calls = []
        self.loading_info = {}
        self.dtype = "float32"
        self.config = types.SimpleNamespace(**mesh_config(), _attn_implementation="sdpa",
                    use_cache=False, return_dict=True, output_attentions=False, output_hidden_states=False)
        self.config.to_dict = lambda: vars(self.config)
        self.reference = copy.copy(self.config)
        self.reference.use_cache = True
        self.parameter = types.SimpleNamespace(dtype="float32", is_floating_point=lambda: True)
        self.model = types.SimpleNamespace(config=self.config, parameters=lambda: [self.parameter],
                    get_input_embeddings=lambda: types.SimpleNamespace(weight=types.SimpleNamespace(shape=(49152, 576))),
                    get_output_embeddings=lambda: types.SimpleNamespace(weight=types.SimpleNamespace(shape=(49152, 576))))
        self.torch = types.SimpleNamespace(float32="float32", get_rng_state=lambda: self.cpu,
                    set_rng_state=lambda value: setattr(self, "cpu", value),
                    cuda=types.SimpleNamespace(is_initialized=lambda: True, get_rng_state_all=lambda: [self.cuda],
                    set_rng_state_all=lambda value: setattr(self, "cuda", value[0])))
        self.numpy = types.SimpleNamespace(random=types.SimpleNamespace(
                    get_state=lambda: self.numpy_state, set_state=lambda value: setattr(self, "numpy_state", value)))
        def baseline_load(path):
            self.calls.append(("baseline", path))
            self.cpu += 3
            return types.SimpleNamespace(config=self.reference, parameters=lambda: [self.parameter])
        def mesh_load(path, **kwargs):
            self.calls.append(("mesh", path, kwargs))
            self.cpu += 100
            self.cuda += 100
            self.numpy_state += 100
            random.random()
            self.parameter.dtype = self.dtype
            return self.model, self.loading_info
        self.recursive = types.ModuleType("recursive_model_5_10x2_5_mesh")
        self.recursive.RecursiveLlamaForCausalLM = types.SimpleNamespace(from_pretrained=mesh_load)
        self.recursive.parameter_audit = lambda model: dict(logical_layer_count=30, physical_layer_count=20,
                                                           memory_slots=5, router_count=6)
        source = ROUTE / "models/recursive_text.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        self.ns = dict(contextlib=contextlib, gc=gc, random=random, sys=sys, Path=Path, np=self.numpy,
                       torch=self.torch, AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=baseline_load),
                       BASELINE_TEXT=BASELINE_TEXT, validate_sources=lambda *args: None,
                       validate_mesh_config=validate_mesh_config, __file__=str(source))
        exec(compile(tree, str(source), "exec"), self.ns)

    def load(self):
        with patch.dict(sys.modules, recursive_model_5_10x2_5_mesh=self.recursive):
            return self.ns["load_recursive_text_model"](str(MESH_TEXT))

    def test_load_keeps_baseline_rng_consumption_only(self):
        python_state = random.getstate()
        model = self.load()
        self.assertEqual((self.cpu, self.cuda, self.numpy_state), (13, 20, 30))
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(model.config.use_cache)
        self.assertEqual(self.calls[1][2]["torch_dtype"], "float32")
        self.assertEqual(self.calls[1][2]["attn_implementation"], "sdpa")
        self.assertEqual(self.calls[0][0], "baseline")

    def test_incomplete_weights_rejected_and_rng_restored(self):
        self.loading_info = {"missing_keys": ["model.write_routers.0.weight"]}
        with self.assertRaisesRegex(ValueError, "completely"):
            self.load()
        self.assertEqual((self.cpu, self.cuda, self.numpy_state), (13, 20, 30))

    def test_unexpected_and_mismatched_weights_rejected(self):
        for field in ("unexpected_keys", "mismatched_keys", "error_msgs"):
            with self.subTest(field=field):
                self.loading_info = {field: ["bad state"]}
                with self.assertRaises(ValueError):
                    self.load()

    def test_non_fp32_loaded_model_rejected(self):
        self.dtype = "bfloat16"
        with self.assertRaisesRegex(ValueError, "FP32"):
            self.load()

    def test_attention_implementation_mismatch_rejected(self):
        self.config._attn_implementation = "eager"
        with self.assertRaisesRegex(ValueError, "attention implementation differs"):
            self.load()

    def test_missing_baseline_attention_rejected(self):
        del self.reference._attn_implementation
        with self.assertRaisesRegex(ValueError, "establish baseline attention"):
            self.load()


if __name__ == "__main__":
    unittest.main()
