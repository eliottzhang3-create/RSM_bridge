"""USAD2/MeSH evaluation parity and failure-path checks without model weights."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code/RSmol/scripts"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module("ced_eval_test_helpers", ROOT / "tests/test_mizar_s1_usad2_benchmark_eval.py")


class MeshUSAD2EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_module("mesh_ced_runner", SCRIPTS / "evaluate_mizar_s1_usad2_recursive_mesh_benchmark.py")
        cls.runtime = cls.runner.runtime_adapter
        cls.route = load_module("mesh_ced_contract", cls.runtime.ROUTE_ROOT / "contract.py")
        cls.baseline = load_module("baseline_ced_runner", SCRIPTS / "evaluate_mizar_s1_usad2_benchmark.py")

    def fixture(self):
        contract, saved, runtime = BASE.USAD2BenchmarkEvaluationTest.fixture(self)
        contract["recursive_text_checkpoint"] = {
            "checkpoint": "text-model", "config": {"hidden_size": 576, "logical_layers": 30,
            "physical_layers": 20, "recursive_loops": 2, "memory_slots": 5, "routers": 6}}
        contract["assets"]={"usad2_small":{"root":"usad2-small","config":{"encoder_dim":384}}}
        contract["comparison_text_model_dir"]="baseline-text"
        contract["comparison_tokenizer_assets"]={"root":"baseline-text","vocab_size":32,"files":[]}
        self.resign(saved, runtime)
        return contract, saved, runtime

    resign = BASE.USAD2BenchmarkEvaluationTest.resign
    decode_fixture = BASE.USAD2BenchmarkEvaluationTest.decode_fixture
    test_defaults_point_to_completed_mesh_run = BASE.USAD2BenchmarkEvaluationTest.test_defaults_point_to_completed_usad2_training_run
    test_checkpoint_route_step_geometry = BASE.USAD2BenchmarkEvaluationTest.test_checkpoint_validates_and_rejects_wrong_route_step_and_geometry
    test_source_runtime_loss_identity = BASE.USAD2BenchmarkEvaluationTest.test_source_runtime_and_loss_identity_remain_strict
    test_training_audio_mapper_contract = BASE.USAD2BenchmarkEvaluationTest.test_training_audio_and_mapper_contracts_are_enforced
    test_all_seven_usad2_asset_identities = BASE.USAD2BenchmarkEvaluationTest.test_asset_audit_checks_all_seven_usad2_files_and_empty_init
    test_generation_audio_prompt_and_raw_text = BASE.USAD2BenchmarkEvaluationTest.test_generation_preserves_whole_waveform_and_uses_training_tokenizer
    test_first_eos_and_full_recompute = BASE.USAD2BenchmarkEvaluationTest.test_greedy_stops_on_first_eos_and_recomputes_without_cache
    test_300_tokens_prefix_and_finite_logits = BASE.USAD2BenchmarkEvaluationTest.test_greedy_enforces_300_horizon_prefix_and_finite_logits
    test_raw_eos_and_prediction_order = BASE.USAD2BenchmarkEvaluationTest.test_materialized_predictions_keep_raw_eos_and_official_order

    def test_entire_runner_greedy_and_protocol_are_identical_to_usad2_base(self):
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False)
                    for n in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

        self.assertEqual(functions(Path(self.runner.__file__)), functions(Path(self.baseline.__file__)))
        original = functions(Path(self.baseline.runtime_adapter.__file__))
        mesh = functions(Path(self.runtime.__file__))
        for name in ("greedy_decode", "audit_assets", "parse_training_geometry"):
            self.assertEqual(original[name], mesh[name], name)
        self.assertEqual(self.runtime.PROTOCOL, self.baseline.runtime_adapter.PROTOCOL)
        self.assertEqual(self.runner.EXPECTED_ROWS, 1000)

    def test_shared_source_and_recursive_contract_mismatches_are_rejected(self):
        _, saved, runtime = self.fixture()
        with patch.object(self.route, "recursive_model_source_sha256", return_value="different"):
            with self.assertRaisesRegex(ValueError, "recursive model source differs"):
                self.runtime.validate_checkpoint(saved, runtime, self.route, 17313)
        for key, value in (("text_model_contract", "other"), ("text_model_architecture", "other"),
                           ("text_model_schedule", list(range(30))),
                           ("recursive_text_checkpoint", None), ("recursive_text_checkpoint", {})):
            with self.subTest(key=key, value=value):
                contract, saved, runtime = self.fixture()
                contract[key] = value
                self.resign(saved, runtime)
                with self.assertRaisesRegex(ValueError, "recursive text"):
                    self.runtime.validate_checkpoint(saved, runtime, self.route, 17313)

    def test_runtime_contract_and_embedded_hash_cannot_be_substituted(self):
        _, saved, runtime = self.fixture()
        changed = copy.deepcopy(runtime)
        changed["contract"]["seed"] = 1
        with self.assertRaisesRegex(ValueError, "runtime.json does not belong"):
            self.runtime.validate_checkpoint(saved, changed, self.route, 17313)
        saved["contract_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "embedded USAD2 contract SHA256 mismatch"):
            self.runtime.validate_checkpoint(saved, runtime, self.route, 17313)

    def test_output_claim_preserves_resume_and_rejects_other_model_or_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            args = types.SimpleNamespace(output_dir=Path(directory) / "eval")
            identity = {"checkpoint": "mesh.ckpt", "generation": dict(self.runtime.PROTOCOL),
                        "evaluation_code": {"runtime_adapter_sha256": "mesh-adapter"}}
            self.runner.claim_output(args, identity)
            self.runner.claim_output(args, identity)
            for changed in ({**identity, "checkpoint": "base.ckpt"},
                            {**identity, "evaluation_code": {"runtime_adapter_sha256": "base-adapter"}},
                            {**identity, "generation": {**self.runtime.PROTOCOL, "use_cache": True}}):
                with self.assertRaisesRegex(RuntimeError, "another checkpoint/dataset/protocol"):
                    self.runner.claim_output(args, changed)
            self.assertEqual(json.loads((args.output_dir / "run_config.json").read_text(encoding="utf-8")), identity)

    def structure_fixture(self):
        class Router:
            def __init__(self):
                self.weight = types.SimpleNamespace(shape=(5, 576))
                self.bias = types.SimpleNamespace(shape=(5,))

            def parameters(self):
                return (self.weight, self.bias)

        class LM:
            def __init__(self):
                self.config = types.SimpleNamespace(hidden_size=576)
                self.model = types.SimpleNamespace(
                    layers=[object() for _ in range(20)],
                    write_routers=[Router() for _ in range(3)],
                    read_routers=[Router() for _ in range(3)],
                    logical_layer_count=30, recursive_layer_count=20, recursive_loops=2,
                    memory_slots=5, audit_mode=False, gradient_audit_mode=False, routing_stats_mode=False)
                self.state = {"model.layers.0.weight": object()}

            def state_dict(self):
                return self.state

        lm = LM()
        model = types.SimpleNamespace(caption_decoder=types.SimpleNamespace(
            lm=lm, is_recursive_mesh=True, text_model_contract=self.runtime.TEXT_MODEL_CONTRACT))
        recursive = types.ModuleType("models.recursive_text")
        recursive.__file__ = str(self.runtime.ROUTE_ROOT / "models/recursive_text.py")
        recursive.validate_recursive_config = Mock()
        implementation = types.ModuleType("recursive_model_5_10x2_5_mesh")
        implementation.__file__ = str(self.runtime.ROOT / "recursive_model_5_10x2_5_mesh.py")
        implementation.RecursiveLlamaForCausalLM = LM
        implementation.parameter_audit = Mock(return_value={
            "logical_layer_count": 30, "physical_layer_count": 20,
            "logical_cache_slot_count": 30, "recursive_loops": 2, "memory_slots": 5,
            "router_count": 6, "router_objects_independent": True, "memory_persistent": False,
            "schedule": list(self.runtime.TEXT_MODEL_SCHEDULE)})
        contract, _, _ = self.fixture()
        return model, contract, recursive, implementation

    def check_structure(self, fixture):
        model, contract, recursive, implementation = fixture
        with (patch.dict(sys.modules, {recursive.__name__: recursive, implementation.__name__: implementation}),
              patch.object(self.route, "recursive_checkpoint_contract", return_value=contract["recursive_text_checkpoint"])):
            return self.runtime.audit_recursive_route(self.route, contract, model)

    def test_actual_layer_router_geometry_and_transient_memory(self):
        fixture = self.structure_fixture()
        audit = self.check_structure(fixture)
        self.assertEqual(audit["loaded_physical_layers"], 20)
        self.assertEqual(audit["loaded_routers"], 6)
        self.assertEqual(audit["loaded_router_parameter_tensors"], 12)
        self.assertEqual(audit["persistent_memory_tensors"], [])

        def layer_alias(model):
            model.caption_decoder.lm.model.layers[1] = model.caption_decoder.lm.model.layers[0]

        def router_alias(model):
            core = model.caption_decoder.lm.model
            core.read_routers[0] = core.write_routers[0]

        def parameter_alias(model):
            core = model.caption_decoder.lm.model
            core.read_routers[0].weight = core.write_routers[0].weight

        for mutate in (layer_alias, router_alias, parameter_alias,
                       lambda m: setattr(m.caption_decoder.lm.model.write_routers[0].weight, "shape", (6, 576)),
                       lambda m: setattr(m.caption_decoder.lm.model, "memory_slots", 6),
                       lambda m: setattr(m.caption_decoder.lm.model, "audit_mode", True),
                       lambda m: m.caption_decoder.lm.state.update({"model.memory": object()}),
                       lambda m: setattr(m.caption_decoder, "is_recursive_mesh", False)):
            with self.subTest(mutation=mutate):
                fixture = self.structure_fixture()
                mutate(fixture[0])
                with self.assertRaises(ValueError):
                    self.check_structure(fixture)

    def test_recursive_asset_and_import_origin_changes_are_rejected(self):
        fixture = self.structure_fixture()
        with (patch.dict(sys.modules, {fixture[2].__name__: fixture[2], fixture[3].__name__: fixture[3]}),
              patch.object(self.route, "recursive_checkpoint_contract", return_value={"changed": True})):
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                self.runtime.audit_recursive_route(self.route, fixture[1], fixture[0])
        for index in (2, 3):
            fixture = self.structure_fixture()
            fixture[index].__file__ = str(ROOT / "wrong_loader.py")
            with self.assertRaisesRegex(RuntimeError, "escaped"):
                self.check_structure(fixture)

    def test_forward_hook_observes_every_schedule_without_changing_output(self):
        lm = Mock()
        lm.model.last_forward_trace = [{"logical_index": i, "physical_index": p}
                                      for i, p in enumerate(self.runtime.TEXT_MODEL_SCHEDULE)]
        audit = self.runtime.install_mesh_trace_audit(lm)
        hook = lm.register_forward_hook.call_args.args[0]
        for length in (508, 509, 807):
            output = types.SimpleNamespace(past_key_values=None, logits=types.SimpleNamespace(shape=(1, length, 32)))
            self.assertIsNone(hook(lm, (), output))
            self.assertEqual(audit["last_sequence_tokens"], length)
        self.assertEqual(audit["checked_forwards"], 3)
        self.assertEqual(audit["status"], "PASS")
        self.assertFalse(audit["use_cache"])

    def test_forward_hook_rejects_wrong_loops_logical_indices_and_cache(self):
        for error in ("physical", "logical", "cache"):
            with self.subTest(error=error):
                lm = Mock()
                lm.model.last_forward_trace = [{"logical_index": i, "physical_index": p}
                                              for i, p in enumerate(self.runtime.TEXT_MODEL_SCHEDULE)]
                output = types.SimpleNamespace(past_key_values=None, logits=types.SimpleNamespace(shape=(1, 508, 32)))
                audit = self.runtime.install_mesh_trace_audit(lm)
                if error == "physical":
                    lm.model.last_forward_trace[15]["physical_index"] = 15
                elif error == "logical":
                    lm.model.last_forward_trace[15]["logical_index"] = 14
                else:
                    output.past_key_values = object()
                hook = lm.register_forward_hook.call_args.args[0]
                with self.assertRaises(RuntimeError):
                    hook(lm, (), output)
                self.assertEqual(audit["checked_forwards"], 0)

    def loader_fixture(self, directory, state=None):
        contract, saved, runtime = self.fixture()
        if state is not None:
            saved["state_dict"] = state
        runtime_path = Path(directory) / "runtime.json"
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
        args = types.SimpleNamespace(checkpoint=Path(directory) / "model.ckpt", runtime_config=runtime_path,
                                     route_root=self.runtime.ROUTE_ROOT, expected_step=17313,
                                     expected_geometry=(8, 8, 4))
        parameter = types.SimpleNamespace(dtype="float32", requires_grad=False, numel=lambda: 10)
        model = Mock()
        for name in ("float", "to", "eval"):
            getattr(model, name).return_value = model
        model.modules.return_value = [types.SimpleNamespace(training=False)]
        model.parameters.return_value = [parameter]
        model.audio_encoder.base.usad2.modules.return_value = [types.SimpleNamespace(training=False)]
        model.audio_encoder.base.usad2.parameters.return_value = [parameter]
        model.caption_decoder.lm.config.vocab_size = 32
        model.caption_decoder.lm.get_input_embeddings.return_value.num_embeddings = 32

        class Tokenizer:
            eos_token_id, pad_token_id, pad_token = 0, 0, "<|endoftext|>"

            def __len__(self):
                return 32

            def encode(self, text, **kwargs):
                return [17] if text == "!" else [0]

            def add_special_tokens(self, tokens):
                self.pad_token, self.pad_token_id = tokens["pad_token"], 17
                return 1  # Existing token registration, with zero vocabulary growth.

        tokenizer = Tokenizer()
        modules = {"contract": self.route, "models.mellow": types.SimpleNamespace(Mellow=Mock(return_value=model)),
                   "data.audiotext_dataset": types.SimpleNamespace(AudioTextDataset=type("Dataset", (), {})),
                   "data.fixed20": object()}
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True, manual_seed_all=Mock(),
                                       set_device=Mock(), get_device_name=lambda _: "fake GPU"),
            manual_seed=Mock(), set_num_threads=Mock(), device=lambda _: "cuda:0", float32="float32",
            backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()),
                                          cudnn=types.SimpleNamespace()),
            load=Mock(return_value=saved), __version__="fake", version=types.SimpleNamespace(cuda="fake"))
        packages = {"torch": torch, "numpy": types.SimpleNamespace(random=types.SimpleNamespace(seed=Mock())),
                    "transformers": types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
                        from_pretrained=Mock(return_value=tokenizer)))}
        return args, saved, model, modules, packages

    def run_loader(self, fixture):
        args, saved, model, modules, packages = fixture

        def audit_after_load(*_):
            model.load_state_dict.assert_called_once()
            return {"status": "PASS"}

        with (patch.dict(sys.modules, packages),
              patch.object(self.runtime, "import_training_route", return_value=modules),
              patch.object(self.runtime, "audit_assets", return_value={"status": "PASS"}),
              patch.object(self.route, "validate_usad2_assets", return_value=saved["contract"]["assets"]["usad2_small"]),
              patch.object(self.runtime, "audit_comparison_assets", return_value={"status": "PASS"}),
              patch.dict(sys.modules, {"comparison": types.SimpleNamespace(compare_tokenizers=Mock(return_value={"status":"PASS"}))}),
              patch.object(self.runtime, "audit_recursive_route", side_effect=audit_after_load),
              patch.object(self.runtime, "install_mesh_trace_audit", return_value={"status": "PENDING"})):
            return self.runtime.load_runtime(args)

    def test_complete_checkpoint_is_loaded_strictly_then_audited(self):
        for ddp in (False, True):
            with self.subTest(ddp=ddp), tempfile.TemporaryDirectory() as directory:
                state = {name: [i] for i, name in enumerate(("audio_encoder.base.usad2.weight",
                         "audio_encoder.projection.weight", "caption_decoder.lm.model.layers.5.weight",
                         "caption_decoder.lm.model.write_routers.0.weight"))}
                fixture = self.loader_fixture(directory, {"module." + k: v for k, v in state.items()} if ddp else state)
                result = self.run_loader(fixture)
                model, modules, packages = fixture[2:]
                model.load_state_dict.assert_called_once_with(state, strict=True)
                model.float.assert_called_once_with()
                calls = [call[0] for call in model.method_calls]
                self.assertLess(calls.index("float"), calls.index("load_state_dict"))
                model.to.assert_called_once_with("cuda:0")
                model.eval.assert_called_once_with()
                model.requires_grad_.assert_called_once_with(False)
                model.caption_decoder.lm.resize_token_embeddings.assert_not_called()
                modules["models.mellow"].Mellow.assert_called_once_with(
                    audioenc_name="USAD2-Small", d_in=384, d_out=576, text_decoder="text-model", prefix_length=508,
                    freeze_text_decoder_weights=False, use_pretrained_audioencoder=True,
                    freeze_audio_encoder_weights=True, pretrained_audioencoder_path="usad2-small")
                self.assertEqual(packages["transformers"].AutoTokenizer.from_pretrained.call_args_list[0].args, ("text-model",))
                self.assertEqual(packages["transformers"].AutoTokenizer.from_pretrained.call_args_list[1].args, ("baseline-text",))
                self.assertEqual(result[5]["tokenizer_comparison"]["status"], "PASS")
                self.assertEqual(len(result), 7)
                self.assertEqual(result[5]["tokenizer"]["checkpoint_source"]["vocabulary_growth"], 0)
                self.assertEqual(result[5]["tokenizer"]["pad_id"], 17)
                self.assertEqual(result[5]["ddp_state_dict_prefix_normalized"], ddp)

    def test_mixed_ddp_namespaces_and_strict_load_errors_are_not_suppressed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.loader_fixture(directory, {"module.a": [1], "b": [2]})
            with self.assertRaisesRegex(ValueError, "mixed DDP"):
                self.run_loader(fixture)
            fixture[2].load_state_dict.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.loader_fixture(directory)
            fixture[2].load_state_dict.side_effect = RuntimeError("Missing key: trained router")
            with self.assertRaisesRegex(RuntimeError, "Missing key: trained router"):
                self.run_loader(fixture)
            fixture[2].eval.assert_not_called()

    def test_non_eval_or_non_float32_models_are_rejected(self):
        for field in ("mode", "dtype"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                fixture = self.loader_fixture(directory)
                if field == "mode":
                    fixture[2].modules.return_value = [types.SimpleNamespace(training=True)]
                else:
                    fixture[2].parameters.return_value = [types.SimpleNamespace(dtype="float16")]
                with self.assertRaisesRegex(RuntimeError, "eval mode|float32"):
                    self.run_loader(fixture)


    def test_comparison_tokenizer_asset_changes_and_wrong_import_are_rejected(self):
        contract,_,_=self.fixture()
        module=types.ModuleType("comparison")
        module.__file__=str(self.runtime.ROUTE_ROOT/"comparison.py")
        module.tokenizer_asset_identity=Mock(return_value=contract["comparison_tokenizer_assets"])
        with patch.dict(sys.modules,{"comparison":module}):
            result=self.runtime.audit_comparison_assets(self.route,contract)
            self.assertEqual(result["status"],"PASS")
            module.tokenizer_asset_identity.assert_called_once_with("baseline-text")
            module.tokenizer_asset_identity.return_value={"changed":True}
            with self.assertRaisesRegex(ValueError,"baseline tokenizer assets changed"):
                self.runtime.audit_comparison_assets(self.route,contract)
            module.__file__=str(ROOT/"wrong_comparison.py")
            with self.assertRaisesRegex(RuntimeError,"comparison escaped"):
                self.runtime.audit_comparison_assets(self.route,contract)


if __name__ == "__main__":
    unittest.main()
