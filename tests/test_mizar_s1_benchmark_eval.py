"""CPU-only contracts for Mizar Stage 1 benchmark evaluation."""
from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code" / "RSmol" / "scripts"
SOURCE = SCRIPTS / "evaluate_mizar_s1_benchmark.py"
RUNTIME_SOURCE = SCRIPTS / "mizar_s1_eval_runtime.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MizarS1BenchmarkEvalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("evaluate_mizar_s1_benchmark", SOURCE)
        cls.runtime = load_module("mizar_s1_eval_runtime", RUNTIME_SOURCE)

    def test_fixed_protocol_is_training_aligned_and_raw(self):
        protocol = self.runtime.PROTOCOL
        self.assertEqual(protocol["prefix_layout"], [129, 129, 1, 256])
        self.assertEqual(protocol["prefix_tokens"], 515)
        self.assertEqual(protocol["answer_start_position"], 515)
        self.assertEqual(protocol["first_answer_logit_position"], 514)
        self.assertEqual(protocol["max_new_tokens"], 300)
        self.assertFalse(protocol["do_sample"])
        self.assertIsNone(protocol["attention_mask"])
        self.assertEqual(protocol["prediction_preprocessing"],
                         "none_including_eos_no_strip_no_label_extraction")

    def test_runtime_tuple_contract_has_loader_audit_and_route_modules(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "model, tokenizer, device, text_encoder, _, _, _ = runtime",
            source,
        )
        runtime_source = RUNTIME_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "return model, tokenizer, device, text_encoder, modules[\"data.fixed20\"], audit, modules",
            runtime_source,
        )

    def test_benchmark_defaults_use_separate_full_evaluation_directories(self):
        mmau = self.module.parse_args(["--benchmark", "mmau"])
        mmar = self.module.parse_args(["--benchmark", "mmar"])
        self.assertEqual(mmau.parquet.name, "test_mini.parquet")
        self.assertEqual(mmau.metadata_json.name, "mmau-test-mini.json")
        self.assertEqual(mmau.audio_root.name, "test-mini-audios")
        self.assertEqual(mmar.metadata_json.name, "MMAR-meta.json")
        self.assertEqual(mmar.audio_root.name, "mmar-audio")
        self.assertIn("mmau_test_mini_step122937", str(mmau.output_dir))
        self.assertIn("mmar_step122937", str(mmar.output_dir))

    def test_gbs256_checkpoint_contract_is_selectable(self):
        args = self.module.parse_args([
            "--benchmark", "mmau", "--expected-step", "17313",
            "--expected-geometry", "8,8,4",
        ])
        self.assertEqual(args.expected_step, 17313)
        self.assertEqual(args.expected_geometry, (8, 8, 4))
        self.assertIn("mmau_test_mini_step17313", str(args.output_dir))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact"
            source.touch()
            for key in ("checkpoint", "runtime_config", "metadata_json", "parquet", "evaluation_script"):
                setattr(args, key, source)
            identity = self.module._identity(args, "model_output")
        self.assertEqual(identity["expected_step"], 17313)
        self.assertEqual(identity["expected_geometry"], [8, 8, 4])

    def checkpoint_fixture(self, variant):
        folder = {
            "gbs32": "mizar_stage1_fixed20_3epochs",
            "gbs256": "mizar_stage1_fixed20_gbs256_3epochs",
            "recursive_mesh": "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh",
            "recursive_mesh_gbs128": "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_5_10x4_5",
            "recursive_5_10_5": "mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5",
        }[variant]
        route = load_module("test_contract_" + variant, ROOT / "code/RSmol" /
                            "mellow_official_training_c8204d8_adamw_cosine" / folder / "contract.py")
        contract = route.semantic_contract({
            "rows": (1311334 if variant == "gbs32" else 1477492), "source_rows": route.SOURCE_COUNTS,
            "rows_canonical_sha256": "manifest", "mapping_sha256": "mapping",
        }, "text-model", "htsat")
        identity = route.digest(contract)
        saved = {
            "schema_version": 1, "route_variant": route.ROUTE, "mode": "formal",
            "state_dict": {"test_parameter": [1]}, "contract": contract,
            "contract_sha256": identity, "total_step": contract["geometry"]["total_steps"],
            "num_epochs": 3,
            "loss_reduction": (
                "mean_of_rank_local_microbatch_token_means_then_DDP_mean"
                if variant != "gbs32" else "rank_local_token_mean_then_DDP_mean"),
        }
        runtime = {"contract": contract, "contract_sha256": identity,
                   "route": route.ROUTE, "mode": "formal"}
        return saved, runtime, route

    def test_checkpoint_schemas_validate_and_cross_route_use_fails(self):
        for variant, geometry, step in (("gbs32", (8, 4, 1), 122937),
                                        ("gbs256", (8, 8, 4), 17313),
                                        ("recursive_mesh", (8, 8, 4), 17313),
                                        ("recursive_mesh_gbs128", (8, 4, 4), 34626),
                                        ("recursive_5_10_5", (8, 8, 4), 17313)):
            with self.subTest(variant=variant):
                saved, runtime, route = self.checkpoint_fixture(variant)
                contract = self.runtime.validate_checkpoint(saved, runtime, route, step, geometry)
                self.assertEqual(contract["geometry"]["total_steps"], step)
                wrong_geometry = (8, 8, 4) if variant == "gbs32" else (8, 4, 1)
                with self.assertRaisesRegex(ValueError, "expected training geometry"):
                    self.runtime.validate_checkpoint(saved, runtime, route, step, wrong_geometry)
                with self.assertRaisesRegex(ValueError, "step / epoch horizon"):
                    self.runtime.validate_checkpoint(saved, runtime, route, step + 1, geometry)
                other = self.checkpoint_fixture("gbs32" if variant != "gbs32" else "gbs256")[2]
                with self.assertRaisesRegex(ValueError, "complete formal"):
                    self.runtime.validate_checkpoint(saved, runtime, other, step, geometry)

    def test_recursive_contracts_require_text_architecture_schedule_and_source(self):
        for variant in ("recursive_mesh", "recursive_mesh_gbs128", "recursive_5_10_5"):
            with self.subTest(variant=variant):
                saved, runtime, route = self.checkpoint_fixture(variant)
                step = 34626 if variant == "recursive_mesh_gbs128" else 17313
                geometry = (8, 4, 4) if variant == "recursive_mesh_gbs128" else (8, 8, 4)
                contract = self.runtime.validate_checkpoint(saved, runtime, route, step, geometry)
                self.assertEqual(contract["text_model_contract"], route.TEXT_MODEL_CONTRACT)
                self.assertEqual(contract["text_model_schedule"], list(route.TEXT_MODEL_SCHEDULE))
                with patch.object(route, "recursive_model_source_sha256", return_value="changed"):
                    with self.assertRaisesRegex(ValueError, "recursive model source differs"):
                        self.runtime.validate_checkpoint(saved, runtime, route, step, geometry)
                for field, value in (("text_model_architecture", "wrong-model"),
                                     ("text_model_schedule", list(reversed(route.TEXT_MODEL_SCHEDULE)))):
                    previous = saved["contract"][field]
                    saved["contract"][field] = value
                    saved["contract_sha256"] = route.digest(saved["contract"])
                    runtime["contract_sha256"] = saved["contract_sha256"]
                    with self.assertRaisesRegex(ValueError, "recursive text training contract"):
                        self.runtime.validate_checkpoint(saved, runtime, route, step, geometry)
                    saved["contract"][field] = previous

    def test_recursive_mesh_and_non_mesh_checkpoints_are_not_interchangeable(self):
        for variant, other_variant in (("recursive_mesh", "recursive_5_10_5"),
                                       ("recursive_mesh_gbs128", "recursive_mesh"),
                                       ("recursive_mesh", "recursive_mesh_gbs128"),
                                       ("recursive_5_10_5", "recursive_mesh")):
            saved, runtime, _ = self.checkpoint_fixture(variant)
            other = self.checkpoint_fixture(other_variant)[2]
            step = 34626 if variant == "recursive_mesh_gbs128" else 17313
            geometry = (8, 4, 4) if variant == "recursive_mesh_gbs128" else (8, 8, 4)
            with self.assertRaisesRegex(ValueError, "complete formal"):
                self.runtime.validate_checkpoint(saved, runtime, other, step, geometry)

    def test_gbs256_keeps_source_runtime_and_loss_checks(self):
        saved, runtime, route = self.checkpoint_fixture("gbs256")
        with patch.object(route, "code_digest", return_value="different-source"):
            with self.assertRaisesRegex(ValueError, "training source differs"):
                self.runtime.validate_checkpoint(saved, runtime, route, 17313, (8, 8, 4))
        with self.assertRaisesRegex(ValueError, "runtime.json does not belong"):
            self.runtime.validate_checkpoint(saved, {**runtime, "contract_sha256": "other"},
                                             route, 17313, (8, 8, 4))
        saved["loss_reduction"] = "rank_local_token_mean_then_DDP_mean"
        with self.assertRaisesRegex(ValueError, "training loss contract mismatch"):
            self.runtime.validate_checkpoint(saved, runtime, route, 17313, (8, 8, 4))

    def test_scorer_key_is_taken_from_official_script(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.py"
            path.write_text("output_key = 'answer_prediction'\n", encoding="utf-8")
            self.assertEqual(self.module.scorer_prediction_key(path), "answer_prediction")
            path.write_text("output_key = 'model_output'\n", encoding="utf-8")
            self.assertEqual(self.module.scorer_prediction_key(path), "model_output")
            path.write_text("output_key = 'selected_option'\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.module.scorer_prediction_key(path)

    def test_materialization_preserves_raw_text_and_official_order(self):
        records = [
            {"id": "a", "answer": "one", "choices": ["one", "two"]},
            {"id": "b", "answer": "two", "choices": ["one", "two"]},
        ]
        raw = "  b) two <|endoftext|>  "
        state = [{"id": "b", "row_index": 1, "status": "generated", "generated_text": raw}]
        result = self.module._materialize(state, records, "answer_prediction", "mmar")
        self.assertEqual([row["id"] for row in result], ["a", "b"])
        self.assertEqual(result[1]["answer_prediction"], raw)
        self.assertEqual(result[0]["answer_prediction"], "")

    def test_output_identity_refuses_cross_checkpoint_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = type("Args", (), {"output_dir": output})()
            self.module.claim_output(args, {"checkpoint": "one"})
            self.module.claim_output(args, {"checkpoint": "one"})
            with self.assertRaises(RuntimeError):
                self.module.claim_output(args, {"checkpoint": "two"})

    def test_mmau_audio_directory_is_optional_but_mmar_audio_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.ckpt"
            runtime = root / "runtime.json"
            route = root / "route"
            metadata = root / "metadata.json"
            scorer = root / "evaluation.py"
            parquet = root / "test_mini.parquet"
            audio_root = root / "missing-audios"
            checkpoint.touch()
            runtime.touch()
            route.mkdir()
            metadata.touch()
            scorer.touch()
            parquet.touch()
            args = type("Args", (), {
                "benchmark": "mmau", "checkpoint": checkpoint,
                "runtime_config": runtime, "route_root": route,
                "metadata_json": metadata, "evaluation_script": scorer,
                "parquet": parquet, "audio_root": audio_root,
            })()
            self.module._validate_input_paths(args)

            args.benchmark = "mmar"
            with self.assertRaisesRegex(FileNotFoundError, "audio root not found"):
                self.module._validate_input_paths(args)

    def test_submission_wrappers_request_one_gpu_and_official_full_runner(self):
        mmau_submit = (ROOT / "code/RSmol/run_mizar_s1_mmau_eval_5090.sh").read_text(encoding="utf-8")
        mmar_submit = (ROOT / "code/RSmol/run_mizar_s1_mmar_eval_5090.sh").read_text(encoding="utf-8")
        inner = (SCRIPTS / "evaluate_mizar_s1_benchmark.sh").read_text(encoding="utf-8")
        for content, benchmark in ((mmau_submit, "mmau"), (mmar_submit, "mmar")):
            self.assertIn("-p pdgpu-5090", content)
            self.assertIn("-c 8 -m 32G -g 1 -n 1", content)
            self.assertIn("-g 1", content)
            self.assertIn(f"--benchmark {benchmark}", content)
            self.assertIn("model--step-122937.ckpt", content)
            self.assertIn("--expected-step", content)
            self.assertIn("--expected-geometry", content)
            self.assertIn("vc submit", content)
        self.assertIn("RSMOL_MIZAR_S1_EVAL_VARIANT", mmau_submit)
        self.assertIn("model--step-17313.ckpt", mmau_submit)
        self.assertIn("recursive_mesh_gbs128", mmau_submit)
        self.assertIn("RSMOL_MIZAR_S1_EVAL_VARIANT", mmar_submit)
        self.assertIn("model--step-17313.ckpt", mmar_submit)
        self.assertIn("recursive_mesh_gbs128", mmar_submit)
        self.assertIn("conda activate mellow_c8204d8", inner)

    def test_recursive_loader_uses_text_module_validator(self):
        source = RUNTIME_SOURCE.read_text(encoding="utf-8")
        self.assertIn('importlib.import_module("models.recursive_text")', source)
        self.assertNotIn("route_contract.validate_recursive_config", source)

    def recursive_audit_fixture(self, variant):
        """Exercise real route validators with CPU stand-ins for Torch state."""
        route = self.checkpoint_fixture(variant)[2]
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        config = {
            "model_type": "llama", "architectures": ["RecursiveLlamaForCausalLM"],
            "hidden_size": 576, "num_hidden_layers": 30, "recursive_layer_count": 20,
            "recursive_loops": 2, "logical_to_physical": list(route.TEXT_MODEL_SCHEDULE),
            "vocab_size": 49152, "eos_token_id": 0,
        }
        parameters = {
            "logical_layer_count": 30, "physical_layer_count": 20,
            "logical_cache_slot_count": 30, "recursive_loops": 2,
            "schedule": list(route.TEXT_MODEL_SCHEDULE),
        }
        if variant == "recursive_mesh":
            implementation = "recursive_model_5_10x2_5_mesh"
            config.update(mesh_memory_slots=5, mesh_router_count=6,
                          mesh_architecture_contract=route.TEXT_MODEL_CONTRACT)
            parameters.update(memory_slots=5, router_count=6)
        elif variant == "recursive_mesh_gbs128":
            implementation = "recursive_model_5_10x4_5_mesh"
            config.update(num_hidden_layers=50, recursive_loops=4,
                          mesh_memory_slots=7, mesh_router_count=10,
                          mesh_architecture_contract=route.TEXT_MODEL_CONTRACT)
            parameters.update(logical_layer_count=50, logical_cache_slot_count=50,
                              recursive_loops=4, memory_slots=7, router_count=10,
                              router_objects_independent=True, memory_persistent=False)
        else:
            implementation = "recursive_model_5_10_5"
            config.update(
                recursive_prefix_layer_count=5, recursive_middle_layer_count=10,
                recursive_suffix_layer_count=5, recursive_loops_scope="middle_only",
                recursive_source_layer_indices_0based=list(route.TEXT_MODEL_SOURCE_MAPPING))
            parameters.update(recursive_layer_count=20, middle_recurrent_count=10,
                              model_physical_module_count=20)
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"CPU fixture, not model weights")
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        contract = {"text_model_dir": str(root),
                    "recursive_text_checkpoint": route.recursive_checkpoint_contract(root)}
        with patch.dict(sys.modules, {"contract": route, "torch": SimpleNamespace(float32="float32")}):
            text = load_module("eval_text_" + variant,
                               Path(route.__file__).parent / "models/recursive_text.py")
        loaded_config = SimpleNamespace(**config)
        loaded_config.to_dict = lambda: vars(loaded_config).copy()
        lm = SimpleNamespace(
            config=loaded_config,
            model=SimpleNamespace(layers=[object() for _ in range(20)],
                                  logical_to_physical=list(route.TEXT_MODEL_SCHEDULE)),
            named_parameters=Mock(return_value=[("model.layers.0.weight", object())]),
            named_buffers=Mock(return_value=[]))
        model = SimpleNamespace(caption_decoder=SimpleNamespace(lm=lm))
        backend = SimpleNamespace(
            __file__=str(ROOT / "code/RSmol" / f"{implementation}.py"),
            parameter_audit=Mock(return_value=parameters))
        modules = {"models.recursive_text": text, implementation: backend}
        return route, contract, model, modules

    def test_non_mesh_audit_executes_training_validator_without_mesh_counters(self):
        route, contract, model, modules = self.recursive_audit_fixture("recursive_5_10_5")
        text = modules["models.recursive_text"]
        backend = modules["recursive_model_5_10_5"]
        with patch.object(self.runtime.importlib, "import_module", side_effect=modules.__getitem__) as importer:
            with patch.object(text, "validate_loaded_model", wraps=text.validate_loaded_model) as validator:
                result = self.runtime.audit_recursive_route(route, contract, model)
                validator.assert_called_once_with(model.caption_decoder.lm, backend.parameter_audit.return_value)
        self.assertEqual([call.args[0] for call in importer.call_args_list],
                         ["models.recursive_text", "recursive_model_5_10_5"])
        self.assertEqual(result["implementation"], "recursive_model_5_10_5")
        self.assertNotIn("memory_slots", result["parameters"])
        self.assertNotIn("router_count", result["parameters"])
        self.assertEqual(result["checkpoint"]["config"]["memory_slots"], 0)
        self.assertEqual(result["checkpoint"]["config"]["routers"], 0)
        model.caption_decoder.lm.named_parameters.assert_called_once_with()
        model.caption_decoder.lm.named_buffers.assert_called_once_with()

    def test_mesh_audit_still_executes_its_real_config_validator_and_router_checks(self):
        route, contract, model, modules = self.recursive_audit_fixture("recursive_mesh")
        text = modules["models.recursive_text"]
        with patch.object(self.runtime.importlib, "import_module", side_effect=modules.__getitem__):
            with patch.object(text, "validate_recursive_config", wraps=text.validate_recursive_config) as validator:
                result = self.runtime.audit_recursive_route(route, contract, model)
                validator.assert_called_once_with(model.caption_decoder.lm.config)
            self.assertEqual(result["parameters"]["memory_slots"], 5)
            self.assertEqual(result["parameters"]["router_count"], 6)
            modules["recursive_model_5_10x2_5_mesh"].parameter_audit.return_value["router_count"] = 0
            with self.assertRaisesRegex(ValueError, "parameter audit mismatch"):
                self.runtime.audit_recursive_route(route, contract, model)

    def test_x4_mesh_audit_requires_four_loops_seven_slots_and_ten_routers(self):
        route, contract, model, modules = self.recursive_audit_fixture("recursive_mesh_gbs128")
        text = modules["models.recursive_text"]
        backend = modules["recursive_model_5_10x4_5_mesh"]
        with patch.object(self.runtime.importlib, "import_module", side_effect=modules.__getitem__):
            with patch.object(text, "validate_recursive_config", wraps=text.validate_recursive_config) as validator:
                result = self.runtime.audit_recursive_route(route, contract, model)
                validator.assert_called_once_with(model.caption_decoder.lm.config)
            self.assertEqual(result["implementation"], "recursive_model_5_10x4_5_mesh")
            self.assertEqual(result["parameters"]["logical_layer_count"], 50)
            self.assertEqual(result["parameters"]["physical_layer_count"], 20)
            self.assertEqual(result["parameters"]["recursive_loops"], 4)
            self.assertEqual(result["parameters"]["memory_slots"], 7)
            self.assertEqual(result["parameters"]["router_count"], 10)
            self.assertTrue(result["parameters"]["router_objects_independent"])
            self.assertFalse(result["parameters"]["memory_persistent"])
            for field, wrong in (("logical_layer_count", 30), ("recursive_loops", 2),
                                 ("memory_slots", 5), ("router_count", 6)):
                original = backend.parameter_audit.return_value[field]
                backend.parameter_audit.return_value[field] = wrong
                with self.assertRaisesRegex(ValueError, "parameter audit mismatch"):
                    self.runtime.audit_recursive_route(route, contract, model)
                backend.parameter_audit.return_value[field] = original
            model.caption_decoder.lm.config.mesh_memory_slots = 5
            with self.assertRaisesRegex(ValueError, "geometry mismatch"):
                self.runtime.audit_recursive_route(route, contract, model)

    def test_non_mesh_audit_rejects_router_memory_parameters_and_buffers(self):
        for method, name in (("named_parameters", "model.routers.0.weight"),
                              ("named_parameters", "model.memory_slots"),
                              ("named_buffers", "model.mesh_memory")):
            with self.subTest(method=method, name=name):
                route, contract, model, modules = self.recursive_audit_fixture("recursive_5_10_5")
                getattr(model.caption_decoder.lm, method).return_value = [(name, object())]
                with patch.object(self.runtime.importlib, "import_module", side_effect=modules.__getitem__):
                    with self.assertRaisesRegex(ValueError, "contains MeSH parameter/buffer"):
                        self.runtime.audit_recursive_route(route, contract, model)

    def test_non_mesh_audit_rejects_wrong_physical_layers_schedule_and_config(self):
        for problem in ("shared_layers", "execution_schedule", "loop_scope", "source_mapping"):
            with self.subTest(problem=problem):
                route, contract, model, modules = self.recursive_audit_fixture("recursive_5_10_5")
                lm = model.caption_decoder.lm
                if problem == "shared_layers":
                    lm.model.layers = [object()] * 20
                    message = "20 independent physical layers"
                elif problem == "execution_schedule":
                    lm.model.logical_to_physical.reverse()
                    message = "execution schedule mismatch"
                elif problem == "loop_scope":
                    lm.config.recursive_loops_scope = "entire_stack"
                    message = "recursive_loops_scope must be middle_only"
                else:
                    lm.config.recursive_source_layer_indices_0based = list(range(20))
                    message = "source-layer mapping mismatch"
                with patch.object(self.runtime.importlib, "import_module", side_effect=modules.__getitem__):
                    with self.assertRaisesRegex(ValueError, message):
                        self.runtime.audit_recursive_route(route, contract, model)

    def test_recursive_audit_rejects_changed_text_artifact_and_stale_imports(self):
        for problem in ("text_artifact", "loader", "implementation"):
            with self.subTest(problem=problem):
                route, contract, model, modules = self.recursive_audit_fixture("recursive_5_10_5")
                if problem == "text_artifact":
                    contract["recursive_text_checkpoint"]["checkpoint"] = "another-initialization"
                    error, message = ValueError, "recursive text checkpoint changed"
                elif problem == "loader":
                    modules["models.recursive_text"].__file__ = __file__
                    error, message = RuntimeError, "loader escaped training route"
                else:
                    modules["recursive_model_5_10_5"].__file__ = __file__
                    error, message = RuntimeError, "implementation escaped source root"
                with patch.object(self.runtime.importlib, "import_module", side_effect=modules.__getitem__):
                    with self.assertRaisesRegex(error, message):
                        self.runtime.audit_recursive_route(route, contract, model)

    def test_submission_selects_checkpoint_runtime_route_and_geometry_together(self):
        bash = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
        if not Path(bash).is_file():
            self.skipTest("Bash is not installed")

        def bash_path(path):
            value = path.resolve().as_posix()
            return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submit = root / "run_mizar_s1_mmau_eval_5090.sh"
            submit.write_bytes((ROOT / "code/RSmol" / submit.name).read_bytes().replace(b"\r\n", b"\n"))
            binary = root / "bin"
            binary.mkdir()
            stub = binary / "vc"
            stub.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$RSM_TEST_CAPTURE"\n',
                            encoding="utf-8", newline="\n")
            stub.chmod(0o755)
            clean_env = {key: value for key, value in os.environ.items()
                         if not key.startswith("RSMOL_")}
            for variant, run, step, geometry in (
                ("gbs32", "mizar_s1_htsat_5090/formal_async_v3", "122937", "8,4,1"),
                ("gbs256", "mizar_s1_htsat_gbs256_5090/formal_hybrid_gbs256_v1", "17313", "8,8,4"),
                ("recursive_mesh", "mizar_s1_htsat_gbs256_recursive_mesh_5090/formal_recursive_mesh_padfix_20261025_003124507033014", "17313", "8,8,4"),
                ("recursive_mesh_gbs128", "mizar_s1_htsat_gbs128_recursive_mesh_5_10x4_5_3090/formal_20261026_003638413364331", "34626", "8,4,4"),
                ("recursive_5_10_5", "mizar_s1_htsat_gbs256_recursive_5_10_5_5090/formal_20261025_160348996763174", "17313", "8,8,4"),
            ):
                with self.subTest(variant=variant):
                    capture = root / f"{variant}.args"
                    env = dict(clean_env, RSMOL_MIZAR_S1_EVAL_VARIANT=variant,
                               RSM_TEST_CAPTURE=bash_path(capture),
                               PATH=f"{bash_path(binary)}:/usr/bin:/bin")
                    subprocess.run([bash, bash_path(submit)], env=env,
                                   capture_output=True, text=True, check=True)
                    submission = capture.read_text(encoding="utf-8").splitlines()
                    for flag, value in (("-c", "8"), ("-m", "32G"), ("-g", "1"), ("-n", "1")):
                        self.assertEqual(submission[submission.index(flag) + 1], value)
                    command = shlex.split(submission[submission.index("--cmd") + 1])
                    self.assertEqual(command[:2], ["bash", "scripts/evaluate_mizar_s1_benchmark.sh"])
                    options = dict(zip(command[2::2], command[3::2]))
                    self.assertTrue(options["--checkpoint"].endswith(
                        f"{run}/formal/checkpoints/model--step-{step}.ckpt"))
                    self.assertTrue(options["--runtime-config"].endswith(f"{run}/formal/runtime.json"))
                    self.assertTrue(options["--output-dir"].endswith(f"{run}/eval/mmau_test_mini_step{step}"))
                    route = {
                        "gbs32": "mizar_stage1_fixed20_3epochs",
                        "gbs256": "mizar_stage1_fixed20_gbs256_3epochs",
                        "recursive_mesh": "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh",
                        "recursive_mesh_gbs128": "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_5_10x4_5",
                        "recursive_5_10_5": "mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5",
                    }[variant]
                    self.assertTrue(options["--route-root"].endswith(route))
                    self.assertEqual(options["--expected-step"], step)
                    self.assertEqual(options["--expected-geometry"], geometry)

    def test_mmar_submission_selects_checkpoint_runtime_route_and_geometry_together(self):
        bash = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
        if not Path(bash).is_file():
            self.skipTest("Bash is not installed")

        def bash_path(path):
            value = path.resolve().as_posix()
            return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submit = root / "run_mizar_s1_mmar_eval_5090.sh"
            submit.write_bytes((ROOT / "code/RSmol" / submit.name).read_bytes().replace(b"\r\n", b"\n"))
            binary = root / "bin"
            binary.mkdir()
            stub = binary / "vc"
            stub.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$RSM_TEST_CAPTURE"\n',
                            encoding="utf-8", newline="\n")
            stub.chmod(0o755)
            clean_env = {key: value for key, value in os.environ.items()
                         if not key.startswith("RSMOL_")}
            for variant, run, step, geometry in (
                ("gbs32", "mizar_s1_htsat_5090/formal_async_v3", "122937", "8,4,1"),
                ("gbs256", "mizar_s1_htsat_gbs256_5090/formal_hybrid_gbs256_v1", "17313", "8,8,4"),
                ("recursive_mesh", "mizar_s1_htsat_gbs256_recursive_mesh_5090/formal_recursive_mesh_padfix_20261025_003124507033014", "17313", "8,8,4"),
                ("recursive_mesh_gbs128", "mizar_s1_htsat_gbs128_recursive_mesh_5_10x4_5_3090/formal_20261026_003638413364331", "34626", "8,4,4"),
                ("recursive_5_10_5", "mizar_s1_htsat_gbs256_recursive_5_10_5_5090/formal_20261025_160348996763174", "17313", "8,8,4"),
            ):
                with self.subTest(variant=variant):
                    capture = root / f"mmar_{variant}.args"
                    env = dict(clean_env, RSMOL_MIZAR_S1_EVAL_VARIANT=variant,
                               RSM_TEST_CAPTURE=bash_path(capture),
                               PATH=f"{bash_path(binary)}:/usr/bin:/bin")
                    subprocess.run([bash, bash_path(submit)], env=env,
                                   capture_output=True, text=True, check=True)
                    submission = capture.read_text(encoding="utf-8").splitlines()
                    for flag, value in (("-c", "8"), ("-m", "32G"), ("-g", "1"), ("-n", "1")):
                        self.assertEqual(submission[submission.index(flag) + 1], value)
                    command = shlex.split(submission[submission.index("--cmd") + 1])
                    self.assertEqual(command[:2], ["bash", "scripts/evaluate_mizar_s1_benchmark.sh"])
                    options = dict(zip(command[2::2], command[3::2]))
                    self.assertTrue(options["--checkpoint"].endswith(
                        f"{run}/formal/checkpoints/model--step-{step}.ckpt"))
                    self.assertTrue(options["--runtime-config"].endswith(f"{run}/formal/runtime.json"))
                    self.assertTrue(options["--output-dir"].endswith(f"{run}/eval/mmar_step{step}"))
                    route = {
                        "gbs32": "mizar_stage1_fixed20_3epochs",
                        "gbs256": "mizar_stage1_fixed20_gbs256_3epochs",
                        "recursive_mesh": "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh",
                        "recursive_mesh_gbs128": "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_5_10x4_5",
                        "recursive_5_10_5": "mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5",
                    }[variant]
                    self.assertTrue(options["--route-root"].endswith(route))
                    self.assertEqual(options["--expected-step"], step)
                    self.assertEqual(options["--expected-geometry"], geometry)

    def test_inference_path_has_no_legacy_audio_or_prefix_contract(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("prefer_official_audio_file=True", source)
        self.assertIn("center_crop_before_split_or_right_pad_to20s", source)
        self.assertIn("_prepare_mmar", source)
        self.assertIn("_run_official_evaluation", source) if "_run_official_evaluation" in source else self.assertIn("run_official_scorer", source)
        self.assertNotIn("prepare_model_output_for_official_scorer", source)
        self.assertNotIn("max_new_tokens 32", source)
        runtime_source = RUNTIME_SOURCE.read_text(encoding="utf-8")
        self.assertIn("attention_mask=None, labels=None, use_cache=False", runtime_source)
        self.assertIn("skip_special_tokens=False", runtime_source)
        self.assertNotIn("split(tokenizer.eos_token", runtime_source)


if __name__ == "__main__":
    unittest.main()
