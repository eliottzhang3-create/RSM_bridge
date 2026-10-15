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
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch256_formal_30epochs_5_10x4_5_mesh"


def load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Private modules allow both routes' tests to run without import collisions.
contract = load_local_module("_formal30_x4_contract", ROUTE / "route_contract.py")
with patch.dict(sys.modules, route_contract=contract):
    comparison = load_local_module("_formal30_x4_comparison", ROUTE / "comparison.py")
MESH_TEXT, BASELINE_TEXT = contract.MESH_TEXT, contract.BASELINE_TEXT
ROUTE_CONTRACT, TEXT_CONTRACT = contract.ROUTE_CONTRACT, contract.TEXT_CONTRACT
LOGICAL_TO_PHYSICAL = contract.LOGICAL_TO_PHYSICAL
validate_mesh_config = contract.validate_mesh_config


def valid_config():
    config = {"data": {"datafiles": ["metadata.json"], "datapath": "stage"},
              "model": {"encoder": {"pretrained_audioencoder_path": "htsat"}},
              "save_dir": "checkpoints"}
    config = comparison.reference_runtime(config)
    config["model"]["decoder"]["text_decoder"] = str(MESH_TEXT.resolve())
    config.update(route_contract=ROUTE_CONTRACT, text_model_contract=TEXT_CONTRACT,
                  comparison_identity=valid_identity())
    return config


def valid_identity():
    return dict(identity_schema_version=comparison.IDENTITY_VERSION,
                route_contract=ROUTE_CONTRACT, text_checkpoint=str(MESH_TEXT),
                baseline_text=str(BASELINE_TEXT), training_code_ast_sha256={"trainer.py": "ast"},
                code_sha256_lf={"wrapper.sh": "original"}, text_artifacts={"model": "weights"},
                baseline_text_artifacts={"model": "baseline weights"},
                tokenizer={"vocab": "canonical"}, environment={"torch": "same version"},
                htsat_sha256="htsat weights",
                initialization="text_weights_only_fresh_audio_bridge_optimizer_scheduler")


def mesh_config():
    return {"hidden_size": 576, "num_hidden_layers": 50, "recursive_layer_count": 20,
            "recursive_loops": 4, "mesh_memory_slots": 7, "mesh_router_count": 10,
            "mesh_architecture_contract": TEXT_CONTRACT,
            "logical_to_physical": list(LOGICAL_TO_PHYSICAL)}


def final_checkpoint_fixture():
    config = valid_config()
    checkpoint = {field: copy.deepcopy(config[field]) for field in
                  ("route_contract", "text_model_contract", "comparison_identity")}
    tensor = types.SimpleNamespace(is_floating_point=lambda: True, dtype="float32")
    checkpoint.update(epoch_completed=30, num_epochs=30, total_step=300,
        initialization_audit={"reference_audio_and_cpu_rng": "exact_match", "lm": {"parameter_audit": dict(
            logical_layer_count=50, physical_layer_count=20, recursive_loops=4,
            logical_cache_slot_count=50, memory_slots=7, router_count=10,
            schedule=list(LOGICAL_TO_PHYSICAL))}},
        state_dict={f"caption_decoder.lm.model.layers.{i}.weight": tensor for i in range(20)})
    for kind in ("write", "read"):
        for i in range(5):
            for field, shape in (("weight", (7, 576)), ("bias", (7,))):
                checkpoint["state_dict"][f"caption_decoder.lm.model.{kind}_routers.{i}.{field}"] = types.SimpleNamespace(
                    shape=shape, is_floating_point=lambda: True, dtype="float32")
    return config, checkpoint


class EquivalenceTests(unittest.TestCase):
    def test_actual_writer_output_matches_baseline(self):
        path = ROUTE / "scripts/rsmol/write_reasonaqa_runtime_config.py"
        emitted = {}
        yaml = types.ModuleType("yaml")
        yaml.safe_dump = lambda value, handle, **kwargs: emitted.update(copy.deepcopy(value))
        with patch.dict(sys.modules, yaml=yaml, comparison=comparison, route_contract=contract):
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
        checkpoint["comparison_identity"]["text_artifacts"]["model"] = "another checkpoint"
        with self.assertRaises(ValueError):
            comparison.validate_resume_identity(checkpoint, config)

    def test_recursive_geometry_rejected(self):
        validate_mesh_config(mesh_config())
        for field, value in (("mesh_memory_slots", 8), ("mesh_memory_slots", 5),
                             ("num_hidden_layers", 30), ("recursive_loops", 2),
                             ("recursive_layer_count", 30), ("mesh_router_count", 6),
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

    def test_wrapper_drift_is_provenance_only(self):
        wrapper = comparison.RSMOL / "run_mellow_official_reasonaqa_gbs256_formal30_5090.sh"
        original = comparison.source_hash
        with patch.object(comparison, "source_hash",
                          side_effect=lambda p: "remote-wrapper-hash" if p == wrapper else original(p)):
            report = comparison.audit_source_equivalence()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["provenance_only_changes"][wrapper.name]["actual_sha256_lf"],
                         "remote-wrapper-hash")
        self.assertNotIn(wrapper.name, report["protected_python_files"])

    def test_python_formatting_allowed_but_prompt_and_training_changes_rejected(self):
        path = comparison.BASE / "data/template.py"
        original = Path.read_text
        def commented(p, *args, **kwargs):
            value = original(p, *args, **kwargs)
            return value + "\n# harmless checkout formatting\n" if p == path else value
        with patch.object(Path, "read_text", commented):
            report = comparison.audit_source_equivalence()
        self.assertEqual(report["status"], "PASS")
        self.assertIn(path.relative_to(comparison.RSMOL).as_posix(),
                      report["provenance_only_changes"])
        def changed(p, *args, **kwargs):
            value = original(p, *args, **kwargs)
            return value + "\nDETAIL = 'different prompt'\n" if p == path else value
        with patch.object(Path, "read_text", changed), \
             self.assertRaisesRegex(ValueError, "protected training Python changed"):
            comparison.audit_source_equivalence()

    def test_ast_hash_portable_across_bom_newlines_comments_and_type_params(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / "left.py", Path(temp) / "right.py"
            left.write_bytes(b"def f(x):\n    return x + 1\n")
            right.write_bytes(b"\xef\xbb\xbf# remote checkout\r\ndef f(x):\r\n    return (x+1) # same code\r\n")
            expected = comparison.python_source_hash(left)
            self.assertEqual(expected, comparison.python_source_hash(right))
            parse = ast.parse
            def newer_ast(*args, **kwargs):
                tree = parse(*args, **kwargs)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and "type_params" not in node._fields:
                        node._fields = (*node._fields, "type_params")
                        node.type_params = []
                return tree
            with patch.object(ast, "parse", newer_ast):
                self.assertEqual(expected, comparison.python_source_hash(right))
            right.write_text("def f(x):\n    return x + 2\n", encoding="utf-8")
            self.assertNotEqual(expected, comparison.python_source_hash(right))

    def test_new_node_and_rebuilt_stage_paths_do_not_bind_resume(self):
        saved, current = valid_config(), valid_config()
        checkpoint = {field: copy.deepcopy(saved[field]) for field in
                      ("route_contract", "text_model_contract", "comparison_identity")}
        with tempfile.TemporaryDirectory() as temp:
            old = Path(temp) / "old_stage"
            old.mkdir()
            saved["data"]["datapath"] = str(old)
            saved["data"]["datafiles"] = [str(old / "metadata.json")]
            old.rmdir()
            current["data"]["datapath"] = str(Path(temp) / "new_stage")
            current["data"]["datafiles"] = [str(Path(temp) / "new_stage/metadata.json")]
            current["save_dir"] = str(Path(temp) / "new_output")
            current["comparison_identity"]["code_sha256_lf"] = {"wrapper.sh": "remote edits"}
            checkpoint["comparison_identity"]["hostname"] = "node-old"
            checkpoint["comparison_identity"]["staging_report"] = {"stage_root": str(old), "pid": 12}
            current["comparison_identity"]["hostname"] = "node-new"
            current["comparison_identity"]["staging_report"] = {"pid": 34, "started_unix": 987}
            self.assertEqual(comparison.audit_runtime_config(current)["status"], "PASS")
            comparison.validate_resume_identity(checkpoint, current)

    def test_stable_identity_drift_still_rejected(self):
        for field in ("training_code_ast_sha256", "text_artifacts", "baseline_text_artifacts",
                      "tokenizer", "environment", "htsat_sha256", "text_checkpoint"):
            saved, current = valid_identity(), valid_identity()
            current[field] = "changed"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                comparison.validate_comparison_identity(saved, current)
        identity = valid_identity()
        del identity["tokenizer"]
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            comparison.validate_comparison_identity(identity, valid_identity())

    def test_final_checkpoint_audit_needs_no_surviving_stage(self):
        path = ROUTE / "scripts/rsmol/audit_comparison.py"
        spec = importlib.util.spec_from_file_location("_mesh_final_audit", path)
        audit = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, comparison=comparison, route_contract=contract):
            spec.loader.exec_module(audit)
        config, checkpoint = final_checkpoint_fixture()
        torch = types.ModuleType("torch")
        torch.float32 = "float32"
        torch.load = lambda *args, **kwargs: checkpoint
        yaml = types.ModuleType("yaml")
        yaml.safe_load = lambda *args: config
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / "deleted_stage"
            stage.mkdir()
            (stage / "metadata.json").write_text("[]", encoding="utf-8")
            config["data"]["datapath"] = str(stage)
            config["data"]["datafiles"] = [str(stage / "metadata.json")]
            (stage / "metadata.json").unlink()
            stage.rmdir()
            runtime, report = root / "runtime.yaml", root / "report.json"
            runtime.write_text("fixture", encoding="utf-8")
            argv = ["audit", "--runtime-config", str(runtime), "--output-report", str(report),
                    "--checkpoint", str(root / "model.ckpt")]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, torch=torch, yaml=yaml), \
                 patch.object(audit, "make_identity", side_effect=AssertionError("must not recheck stage")):
                audit.main()
            self.assertEqual(json.loads(report.read_text())["status"], "PASS")

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
        self.model = types.SimpleNamespace(config=self.config, model=types.SimpleNamespace(ablate_after_two_loops=False), parameters=lambda: [self.parameter],
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
        self.recursive = types.ModuleType("recursive_model_5_10x4_5_mesh")
        self.recursive.RecursiveLlamaForCausalLM = types.SimpleNamespace(from_pretrained=mesh_load)
        self.recursive.parameter_audit = lambda model: dict(logical_layer_count=50, physical_layer_count=20,
            recursive_loops=4, logical_cache_slot_count=50, memory_slots=7, router_count=10,
            schedule=list(LOGICAL_TO_PHYSICAL), router_objects_independent=True, memory_persistent=False)
        source = ROUTE / "models/recursive_text.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        self.ns = dict(contextlib=contextlib, gc=gc, random=random, sys=sys, Path=Path, np=self.numpy,
                       torch=self.torch, AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=baseline_load),
                       BASELINE_TEXT=BASELINE_TEXT, LOGICAL_TO_PHYSICAL=LOGICAL_TO_PHYSICAL, validate_sources=lambda *args: None,
                       validate_mesh_config=validate_mesh_config, __file__=str(source))
        exec(compile(tree, str(source), "exec"), self.ns)

    def load(self):
        with patch.dict(sys.modules, recursive_model_5_10x4_5_mesh=self.recursive):
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
