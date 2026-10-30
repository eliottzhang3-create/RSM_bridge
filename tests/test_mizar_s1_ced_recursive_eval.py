"""Contracts and executable CPU stand-ins for the four non-MeSH evaluators."""
from __future__ import annotations

import ast
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
sys.path.insert(0, str(SCRIPTS))
import evaluate_mizar_s1_ced_recursive_benchmark as runner

adapter = runner.runtime_adapter
VARIANTS = tuple(adapter.ROUTE_SPECS)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module("ced_recursive_reference_helpers", ROOT / "tests/test_mizar_s1_ced_benchmark_eval.py")


class RecursiveCedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.runner, self.runtime = runner, adapter
        self.select(VARIANTS[0])

    def select(self, variant):
        self.variant = variant
        self.spec = adapter.variant_spec(variant)
        self.route = load_module("eval_contract_" + variant, self.spec["route_root"] / "contract.py")

    def fixture(self):
        contract, saved, runtime = BASE.CedBenchmarkEvaluationTest.fixture(self)
        contract["recursive_text_checkpoint"] = {"checkpoint": "text-model", "fixture_only": True}
        self.resign(saved, runtime)
        return contract, saved, runtime

    resign = BASE.CedBenchmarkEvaluationTest.resign
    decode_fixture = BASE.CedBenchmarkEvaluationTest.decode_fixture
    test_raw_predictions_preserve_eos_and_order = BASE.CedBenchmarkEvaluationTest.test_materialized_predictions_keep_raw_eos_and_official_order

    def test_asset_hashes_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for name in adapter.reference.CED_ASSET_FILES:
                path = root / name
                path.write_bytes(name.encode())
                entries.append({"name": name, "bytes": path.stat().st_size,
                                "sha256": adapter.sha256_file(path)})
            contract = {"text_model_dir": str(root), "ced_model_dir": str(root),
                        "assets": {"text": [entries[0]], "ced": entries}}
            self.assertEqual(adapter.audit_assets(contract)["verified_files"], 7)
            (root / "model.safetensors").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                adapter.audit_assets(contract)

    def validate(self, saved, runtime, **kwargs):
        return adapter.validate_checkpoint(saved, runtime, self.route, 17313,
                                           eval_variant=self.variant, **kwargs)

    def test_all_four_contracts_and_eight_default_paths(self):
        stamps = ("20261028_160823559722647", "20261030_000807991327585",
                  "20261029_210116592758429", "20261029_110929631468450")
        for variant, stamp in zip(VARIANTS, stamps):
            with self.subTest(variant=variant):
                self.select(variant)
                contract, saved, runtime = self.fixture()
                self.assertIs(self.validate(saved, runtime), contract)
                self.assertEqual(self.spec["route"], self.route.ROUTE)
                self.assertEqual(self.spec["run_root"].name, "formal_" + stamp)
                for benchmark in ("mmau", "mmar"):
                    args = runner.parse_args(["--benchmark", benchmark, "--eval-variant", variant])
                    self.assertEqual(args.checkpoint, (self.spec["run_root"] / "formal/checkpoints/model--step-17313.ckpt").resolve())
                    self.assertEqual(args.runtime_config, (self.spec["run_root"] / "formal/runtime.json").resolve())
                    self.assertEqual(args.route_root, self.spec["route_root"].resolve())
                    self.assertEqual(args.expected_geometry, (8, 8, 4))
                    self.assertEqual(args.output_dir.name, "mmau_test_mini_step17313" if benchmark == "mmau" else "mmar_step17313")

    def test_all_cross_route_pairings_are_rejected(self):
        for variant in VARIANTS:
            self.select(variant)
            _, saved, runtime = self.fixture()
            for other in set(VARIANTS) - {variant}:
                with self.subTest(checkpoint=variant, selected=other), self.assertRaises(ValueError):
                    adapter.validate_checkpoint(saved, runtime, self.route, 17313, eval_variant=other)

    def test_checkpoint_runtime_source_and_geometry_checks(self):
        for variant in VARIANTS:
            self.select(variant)
            for key, value in (("schema_version", 2), ("mode", "smoke"), ("state_dict", {}),
                               ("total_step", 20), ("num_epochs", 2), ("loss_reduction", "wrong"),
                               ("contract_sha256", "wrong")):
                with self.subTest(variant=variant, key=key):
                    _, saved, runtime = self.fixture()
                    saved[key] = value
                    with self.assertRaises(ValueError):
                        self.validate(saved, runtime)
            _, saved, runtime = self.fixture()
            with self.assertRaises(ValueError):
                self.validate(saved, runtime, expected_geometry=(8, 4, 1))
            with self.assertRaises(ValueError):
                self.validate(saved, {**runtime, "contract_sha256": "other"})
            for method in ("code_digest", "recursive_model_source_sha256"):
                with patch.object(self.route, method, return_value="changed"), self.assertRaises(ValueError):
                    self.validate(saved, runtime)

    def test_architecture_audio_mapper_and_injection_contracts_are_enforced(self):
        for variant in VARIANTS:
            self.select(variant)
            mutations = [(None, "text_model_schedule", list(range(30))),
                         (None, "text_model_architecture", "mesh"),
                         (None, "recursive_text_checkpoint", {}), (None, "prefix", [130, 256]),
                         (None, "attention_mask", "pad_mask"), (None, "forward_dtype", "bfloat16"),
                         ("audio_encoder", "hidden_cache", True), ("mapper", "dropout", 0.)]
            if self.spec["layer_order"]:
                mutations += [("layer_injection", "ced_layers_1based", list(reversed(self.spec["layer_order"]))),
                              ("layer_injection", "logical_indices_0based", [6, 16])]
            if self.spec["reinjection"]:
                mutations += [("audio_reinjection", "shared_gate", False),
                              ("audio_reinjection", "source", "current_hidden")]
            for section, key, value in mutations:
                with self.subTest(variant=variant, section=section, key=key):
                    contract, saved, runtime = self.fixture()
                    (contract if section is None else contract[section])[key] = value
                    self.resign(saved, runtime)
                    with self.assertRaises(ValueError):
                        self.validate(saved, runtime)

    def test_protocol_and_fifteen_common_runner_functions_match_base(self):
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False)
                    for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
        original = functions(SCRIPTS / "evaluate_mizar_s1_ced_benchmark.py")
        actual = functions(Path(runner.__file__))
        self.assertEqual(set(original), set(actual))
        self.assertEqual({name for name in original if original[name] != actual[name]},
                         {"parse_args", "generate", "_identity"})
        self.assertEqual(adapter.PROTOCOL, adapter.reference.PROTOCOL)
        self.assertEqual(runner.EXPECTED_ROWS, 1000)
        for relative in ("data/fixed20.py", "data/audiotext_dataset.py", "models/ced_offline.py"):
            expected = (adapter.reference.ROUTE_ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
            for spec in adapter.ROUTE_SPECS.values():
                self.assertEqual((spec["route_root"] / relative).read_bytes().replace(b"\r\n", b"\n"), expected)

    def decode(self, ids, *, finite=True, information=True):
        model, tokenizer, prefix, torch = self.decode_fixture(ids, finite)
        features = None
        if self.spec["layer_order"] and information:
            class AudioTensor:
                shape, device, dtype = (1, 126, 576), "cuda:0", "float32"
            torch.Tensor = AudioTensor
            prefix.dtype = "float32"
            features = (AudioTensor(), AudioTensor())
        return model, tokenizer, prefix, torch, features

    def test_greedy_eos_recompute_raw_text_and_feature_order_for_all_variants(self):
        for variant in VARIANTS:
            self.select(variant)
            model, tokenizer, prefix, torch, features = self.decode([42, 0, 99])
            with patch.dict(sys.modules, {"torch": torch}):
                result = adapter.greedy_decode(model, tokenizer, prefix, eval_variant=variant, audio_information=features)
            self.assertEqual(result["generated_token_ids"], [42, 0])
            self.assertEqual(result["generated_text"], "  raw <|endoftext|>  ")
            self.assertEqual(result["stop_reason"], "eos_token")
            calls = model.caption_decoder.lm.call_args_list
            self.assertEqual([c.kwargs["inputs_embeds"].shape[1] for c in calls], [383, 384])
            for call in calls:
                self.assertIsNone(call.kwargs["attention_mask"])
                self.assertIsNone(call.kwargs["labels"])
                self.assertIs(call.kwargs["use_cache"], False)
                if features:
                    self.assertIs(call.kwargs["audio_information"], features)
                else:
                    self.assertNotIn("audio_information", call.kwargs)

    def test_300_token_limit_prefix_and_finite_logits(self):
        for variant in VARIANTS:
            self.select(variant)
            model, tokenizer, prefix, torch, features = self.decode([42] * 300)
            with patch.dict(sys.modules, {"torch": torch}):
                result = adapter.greedy_decode(model, tokenizer, prefix, eval_variant=variant, audio_information=features)
                self.assertEqual(result["generated_token_count"], 300)
                self.assertEqual(result["stop_reason"], "max_new_tokens")
                with self.assertRaisesRegex(ValueError, "exactly max_new_tokens"):
                    adapter.greedy_decode(model, tokenizer, prefix, 32, eval_variant=variant)
                prefix.shape = (1, 515, 576)
                with self.assertRaisesRegex(ValueError, "prefix shape"):
                    adapter.greedy_decode(model, tokenizer, prefix, eval_variant=variant)
            model, tokenizer, prefix, torch, features = self.decode([42], finite=False)
            with patch.dict(sys.modules, {"torch": torch}), self.assertRaises(ValueError):
                adapter.greedy_decode(model, tokenizer, prefix, eval_variant=variant, audio_information=features)

    def test_missing_wrong_or_unexpected_sample_features_are_rejected(self):
        for variant in VARIANTS:
            self.select(variant)
            model, tokenizer, prefix, torch, features = self.decode([0])
            bad_values = [None, (), (object(), object())] if features else [(object(), object())]
            if features:
                wrong = type(features[0])()
                wrong.shape = (1, 127, 576)
                bad_values += [(wrong, features[1])]
                wrong = type(features[0])()
                wrong.dtype = "float16"
                bad_values += [(wrong, features[1])]
                wrong = type(features[0])()
                wrong.device = "cpu"
                bad_values += [(wrong, features[1])]
            for value in bad_values:
                with patch.dict(sys.modules, {"torch": torch}), self.assertRaises(ValueError):
                    adapter.greedy_decode(model, tokenizer, prefix, eval_variant=variant, audio_information=value)
            model.caption_decoder.lm.assert_not_called()

    def test_generate_encodes_once_and_passes_each_samples_features(self):
        from contextlib import nullcontext
        class Tensor:
            shape = (1, 383, 576)
            def numel(self): return 320000
            def unsqueeze(self, _): return self
            def to(self, _): return self
            def eq(self, _): return self
            def sum(self): return 11
        for variant in VARIANTS:
            self.select(variant)
            model, tokenizer, encoder = Mock(), Mock(pad_token_id=17), Mock()
            encoder.tokenize.return_value = {"input_ids": Tensor(), "attention_mask": Tensor()}
            tokenizer.encode.return_value = [1, 2]
            samples = [((object(), object()) if self.spec["layer_order"] else None) for _ in range(2)]
            prefix = Tensor()
            model.generate_prefix_inference.side_effect = [(prefix, {}, f) for f in samples]
            with (patch.dict(sys.modules, {"torch": types.SimpleNamespace(inference_mode=nullcontext)}),
                  patch.object(runner, "_fixed20_pair", return_value=(Tensor(), Tensor())),
                  patch.object(adapter, "greedy_decode", return_value={"generated_text": " raw EOS ", "generated_token_count": 1}) as decode):
                for _ in samples:
                    result = runner.generate((model, tokenizer, "cuda:0", encoder, None,
                                              {"eval_variant": variant}, None), {"prompt": "A Question?"})
                    self.assertEqual(result["generated_text"], " raw EOS ")
                    self.assertEqual(result["first_answer_logit_position"], 382)
            self.assertEqual(model.generate_prefix_inference.call_count, 2)
            for call, features in zip(decode.call_args_list, samples):
                self.assertIs(call.kwargs["audio_information"], features)
                self.assertEqual(call.kwargs["eval_variant"], variant)
            encoder.tokenize.assert_called_with("a question?", 256)

    def structure_fixture(self):
        config = dict(model_type="llama", architectures=["RecursiveLlamaForCausalLM"],
                      hidden_size=576, num_hidden_layers=30, recursive_layer_count=20,
                      recursive_loops=2, recursive_prefix_layer_count=5, recursive_middle_layer_count=10,
                      recursive_suffix_layer_count=5, recursive_loops_scope="middle_only",
                      logical_to_physical=list(adapter.TEXT_MODEL_SCHEDULE), vocab_size=32, eos_token_id=0,
                      recursive_source_layer_indices_0based=list(self.route.TEXT_MODEL_SOURCE_MAPPING))
        parameters = dict(logical_layer_count=30, physical_layer_count=20, recursive_layer_count=20,
                          recursive_loops=2, middle_recurrent_count=10, model_physical_module_count=20,
                          schedule=list(adapter.TEXT_MODEL_SCHEDULE))
        class Module:
            def __init__(self, count=1): self.count = count
            def parameters(self): return [types.SimpleNamespace(numel=lambda: self.count)]
        class LM:
            def __init__(self):
                self.config = types.SimpleNamespace(**config, to_dict=lambda: config)
                self.model = types.SimpleNamespace(layers=[object() for _ in range(20)],
                    logical_to_physical=list(adapter.TEXT_MODEL_SCHEDULE))
                self.params, self.buffers = [("model.layers.0.weight", object())], []
            def named_parameters(self): return self.params
            def named_buffers(self): return self.buffers
        lm = LM()
        model = types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=lm, is_recursive_5_10_5=True,
            text_model_contract=adapter.TEXT_MODEL_CONTRACT), audio_encoder=types.SimpleNamespace(base=types.SimpleNamespace()))
        if self.spec["layer_order"]:
            gates = {str(k): types.SimpleNamespace(weight=types.SimpleNamespace(shape=(1, 576)),
                     bias=types.SimpleNamespace(shape=(1,))) for k in (4, 8)}
            class Gates(dict):
                def parameters(self): return [types.SimpleNamespace(numel=lambda: 1154)]
            class MLPs(dict):
                def parameters(self): return [types.SimpleNamespace(numel=lambda: 2443776)]
            lm.model.audio_gates = Gates(gates)
            lm.model.layer_inject_tokens, lm.model.layer_inject_indices = 126, (5, 15)
            lm.model.collect_injection_stats = False
            model.audio_encoder.base.layer_mlps = MLPs({"4": Module(), "8": Module()})
        if self.spec["reinjection"]:
            lm.model.audio_reinject_tokens, lm.model.audio_reinject_indices = 126, (5, 15)
            lm.model.collect_reinject_stats = False
            lm.model.audio_linear = types.SimpleNamespace(weight=types.SimpleNamespace(shape=(576, 576)), bias=types.SimpleNamespace(shape=(576,)))
            lm.model.audio_gate = types.SimpleNamespace(weight=types.SimpleNamespace(shape=(1, 576)), bias=types.SimpleNamespace(shape=(1,)))
            lm.params += [("model." + key, object()) for key in
                          ("audio_linear.weight", "audio_linear.bias", "audio_gate.weight", "audio_gate.bias")]
        with patch.dict(sys.modules, {"contract": self.route}):
            text = load_module("actual_text_" + self.variant, self.spec["route_root"] / "models/recursive_text.py")
        implementation = types.ModuleType(self.spec["implementation"])
        implementation.__file__ = str(adapter.ROOT / "recursive_model_5_10_5.py" if self.variant == VARIANTS[0]
            else self.spec["route_root"] / (self.spec["implementation"].replace(".", "/") + ".py"))
        implementation.RecursiveLlamaForCausalLM = LM
        implementation.parameter_audit = Mock(return_value=parameters)
        contract, _, _ = self.fixture()
        return model, contract, {"models.recursive_text": text, self.spec["implementation"]: implementation}

    def audit_structure(self, fixture):
        model, contract, modules = fixture
        with (patch.dict(sys.modules, modules), patch.object(self.route, "recursive_checkpoint_contract",
              return_value=contract["recursive_text_checkpoint"])):
            return adapter.audit_recursive_route(self.route, contract, model, eval_variant=self.variant)

    def test_real_training_validators_accept_each_structure_and_trained_gates(self):
        for variant, count in zip(VARIANTS, (0, 2444930, 2444930, 332929)):
            self.select(variant)
            fixture = self.structure_fixture()
            audit = self.audit_structure(fixture)
            self.assertEqual(audit["new_parameter_count"], count)
            self.assertEqual(audit["routers"], 0)
            self.assertEqual(audit["layer_order"], list(self.spec["layer_order"]))

    def test_real_training_validators_reject_aliases_mesh_state_and_bad_geometry(self):
        for variant in VARIANTS:
            self.select(variant)
            for kind in ("layer_alias", "schedule", "router_parameter", "memory_buffer", "config"):
                fixture = self.structure_fixture()
                lm = fixture[0].caption_decoder.lm
                if kind == "layer_alias": lm.model.layers[1] = lm.model.layers[0]
                elif kind == "schedule": lm.model.logical_to_physical = list(range(30))
                elif kind == "router_parameter": lm.params.append(("model.router.weight", object()))
                elif kind == "memory_buffer": lm.buffers.append(("model.memory_slots.state", object()))
                else: lm.config.to_dict = lambda: {"hidden_size": 32}
                with self.subTest(variant=variant, kind=kind), self.assertRaises(ValueError):
                    self.audit_structure(fixture)
            if self.spec["layer_order"]:
                fixture = self.structure_fixture()
                mlps = fixture[0].audio_encoder.base.layer_mlps
                mlps["8"] = mlps["4"]
                with self.assertRaises(ValueError): self.audit_structure(fixture)

    def test_changed_recursive_assets_and_stale_imports_are_rejected(self):
        for variant in VARIANTS:
            self.select(variant)
            fixture = self.structure_fixture()
            with patch.object(self.route, "recursive_checkpoint_contract", return_value={}), self.assertRaises(ValueError):
                adapter.audit_recursive_route(self.route, fixture[1], fixture[0], eval_variant=variant)
            for module in fixture[2].values():
                old = module.__file__
                module.__file__ = str(ROOT / "wrong_module.py")
                with self.assertRaises(RuntimeError): self.audit_structure(fixture)
                module.__file__ = old

    def test_hooks_observe_physical_calls_injection_order_and_cache(self):
        for variant in VARIANTS:
            self.select(variant)
            lm = Mock()
            lm.model.layers = [Mock() for _ in range(20)]
            expected = [[i, k] for i, k in zip((5, 15), self.spec["layer_order"])] if self.spec["layer_order"] else [5, 15] if self.spec["reinjection"] else []
            lm.model.last_injection_trace, lm.model.last_reinject_trace = expected, expected
            audit = adapter.install_recursive_trace_audit(lm, eval_variant=variant)
            begin = lm.register_forward_pre_hook.call_args.args[0]
            finish = lm.register_forward_hook.call_args.args[0]
            output = types.SimpleNamespace(past_key_values=None, logits=types.SimpleNamespace(shape=(1, 383, 32)))
            for _ in range(2):
                self.assertIsNone(begin(lm, (), {"use_cache": False}))
                for index in adapter.TEXT_MODEL_SCHEDULE:
                    self.assertIsNone(lm.model.layers[index].register_forward_hook.call_args.args[0](None, (), None))
                self.assertIsNone(finish(lm, (), output))
            self.assertEqual(audit["checked_forwards"], 2)
            self.assertEqual(audit["last_injection_trace"], expected)
            with self.assertRaises(RuntimeError): begin(lm, (), {"use_cache": True})
            with self.assertRaises(RuntimeError): begin(lm, (), {"use_cache": False, "past_key_values": object()})
            begin(lm, (), {"use_cache": False})
            with self.assertRaisesRegex(RuntimeError, "physical schedule"): finish(lm, (), output)
            for index in adapter.TEXT_MODEL_SCHEDULE:
                lm.model.layers[index].register_forward_hook.call_args.args[0](None, (), None)
            output.past_key_values = object()
            with self.assertRaisesRegex(RuntimeError, "KV cache"): finish(lm, (), output)
            output.past_key_values = None
            if expected:
                lm.model.last_injection_trace = lm.model.last_reinject_trace = list(reversed(expected))
                with self.assertRaisesRegex(RuntimeError, "injection trace"): finish(lm, (), output)

    def loader_fixture(self, directory, state):
        _, saved, runtime = self.fixture()
        saved["state_dict"] = state
        runtime_path = Path(directory) / "runtime.json"
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
        args = types.SimpleNamespace(checkpoint=Path(directory)/"model.ckpt", runtime_config=runtime_path,
            route_root=self.spec["route_root"], expected_step=17313, expected_geometry=(8, 8, 4), eval_variant=self.variant)
        parameter = types.SimpleNamespace(dtype="float32", requires_grad=False, numel=lambda: 10)
        model = Mock()
        for name in ("float", "to", "eval"): getattr(model, name).return_value = model
        model.modules.return_value = [types.SimpleNamespace(training=False)]
        model.parameters.return_value = [parameter]
        model.audio_encoder.base.ced.modules.return_value = [types.SimpleNamespace(training=False)]
        model.audio_encoder.base.ced.parameters.return_value = [parameter]
        model.caption_decoder.lm.config.vocab_size = 32
        model.caption_decoder.lm.get_input_embeddings.return_value.num_embeddings = 32
        class Tokenizer:
            eos_token_id, pad_token_id, pad_token = 0, 0, "<|endoftext|>"
            def __len__(self): return 32
            def encode(self, text, **kwargs): return [17] if text == "!" else [0]
            def add_special_tokens(self, tokens):
                self.pad_token, self.pad_token_id = tokens["pad_token"], 17
                return 1
        modules = {"contract": self.route, "models.mellow": types.SimpleNamespace(Mellow=Mock(return_value=model)),
            "data.fixed20": object(), "data.audiotext_dataset": types.SimpleNamespace(AudioTextDataset=type("Dataset", (), {}))}
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, manual_seed_all=Mock(),
            set_device=Mock(), get_device_name=lambda _: "fake GPU"), manual_seed=Mock(), set_num_threads=Mock(),
            device=lambda _: "cuda:0", float32="float32", load=Mock(return_value=saved), __version__="mock",
            version=types.SimpleNamespace(cuda="mock"), backends=types.SimpleNamespace(
                cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()), cudnn=types.SimpleNamespace()))
        packages = {"torch": torch, "numpy": types.SimpleNamespace(random=types.SimpleNamespace(seed=Mock())),
            "transformers": types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=Mock(return_value=Tokenizer())))}
        return args, model, modules, packages

    def run_loader(self, fixture):
        args, model, modules, packages = fixture
        def after_load(*_, **__):
            model.load_state_dict.assert_called_once()
            return {"status": "PASS"}
        with (patch.dict(sys.modules, packages), patch.object(adapter, "import_training_route", return_value=modules),
              patch.object(adapter, "audit_assets", return_value={"status": "PASS"}),
              patch.object(adapter, "audit_recursive_route", side_effect=after_load),
              patch.object(adapter, "install_recursive_trace_audit", return_value={"status": "PENDING"})):
            return adapter.load_runtime(args)

    def test_full_audio_state_loads_strictly_in_fp32_and_tokenizer_has_no_growth(self):
        for variant in VARIANTS:
            self.select(variant)
            for ddp in (False, True):
                with tempfile.TemporaryDirectory() as directory:
                    names = ("audio_encoder.base.ced.weight", "audio_encoder.base.mapper.weight",
                             "caption_decoder.lm.model.layers.5.weight", "trained_branch.weight")
                    state = {name: [i] for i, name in enumerate(names)}
                    fixture = self.loader_fixture(directory, {"module."+k: v for k, v in state.items()} if ddp else state)
                    result = self.run_loader(fixture)
                    model = fixture[1]
                    model.load_state_dict.assert_called_once_with(state, strict=True)
                    calls = [c[0] for c in model.method_calls]
                    self.assertLess(calls.index("float"), calls.index("load_state_dict"))
                    model.requires_grad_.assert_called_once_with(False)
                    model.caption_decoder.lm.resize_token_embeddings.assert_not_called()
                    self.assertEqual(len(result), 7)
                    self.assertEqual(result[5]["tokenizer"]["checkpoint_source"]["vocabulary_growth"], 0)
                    self.assertEqual(result[5]["tokenizer"]["pad_id"], 17)
                    self.assertEqual(result[5]["ddp_state_dict_prefix_normalized"], ddp)

    def test_loader_does_not_suppress_missing_keys_mixed_ddp_or_bad_model_state(self):
        for kind in ("ddp", "strict", "dtype", "mode"):
            with tempfile.TemporaryDirectory() as directory:
                fixture = self.loader_fixture(directory, {"module.a": [1], "b": [2]} if kind == "ddp" else {"a": [1]})
                model = fixture[1]
                if kind == "strict": model.load_state_dict.side_effect = RuntimeError("Missing trained gate")
                elif kind == "dtype": model.parameters.return_value = [types.SimpleNamespace(dtype="float16")]
                elif kind == "mode": model.modules.return_value = [types.SimpleNamespace(training=True)]
                with self.subTest(kind=kind), self.assertRaises((ValueError, RuntimeError)):
                    self.run_loader(fixture)


if __name__ == "__main__":
    unittest.main()
