"""Executable contract/loading/decoding checks without production assets or CUDA."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code/RSmol/scripts"
sys.path.insert(0, str(SCRIPTS))
import evaluate_mizar_s1_ced_multilayer_benchmark as runner

adapter = runner.runtime_adapter
VARIANTS = tuple(adapter.ROUTE_SPECS)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module("ced_multilayer_test_helpers", ROOT / "tests/test_mizar_s1_ced_benchmark_eval.py")
PREVIOUS = load_module("ced_multilayer_loader_helpers", ROOT / "tests/test_mizar_s1_ced_recursive_eval.py")


class MultilayerEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.runner, self.runtime = runner, adapter
        self.select(VARIANTS[0])

    def select(self, variant):
        self.variant, self.spec = variant, adapter.variant_spec(variant)
        self.route = load_module("multilayer_contract_" + variant, self.spec["route_root"] / "contract.py")

    def fixture(self):
        contract, saved, runtime = BASE.CedBenchmarkEvaluationTest.fixture(self)
        saved["total_step"] = self.spec["default_step"]
        contract["recursive_text_checkpoint"] = {"fixture_only": True}
        self.resign(saved, runtime)
        return contract, saved, runtime

    resign = BASE.CedBenchmarkEvaluationTest.resign
    def loader_fixture(self, directory, state):
        fixture = PREVIOUS.RecursiveCedEvaluationTests.loader_fixture(self, directory, state)
        fixture[0].expected_step = self.spec["default_step"]
        fixture[0].expected_geometry = self.spec["default_geometry"]
        return fixture
    test_raw_predictions_preserve_eos_and_order = BASE.CedBenchmarkEvaluationTest.test_materialized_predictions_keep_raw_eos_and_official_order

    def validate(self, saved, runtime, **kwargs):
        return adapter.validate_checkpoint(saved, runtime, self.route, self.spec["default_step"],
                                           eval_variant=self.variant, **kwargs)

    def test_twelve_actual_contracts_accept_completed_checkpoints_and_bind_eleven_runs(self):
        stamps = ("20261030_140459065784533", "20261030_162028139130855", None,
                  "20261030_172450546025072", "20261101_113133327944502",
                  "20261101_170211458358358", "20261102_002211827036990", "20261102_112040402999384",
                  "20261102_132232677352730", "20261102_112121702891801", "20261103_165410438279511", "20261103_093224843011222")
        counts = (4892741, 6241733, 2444930, 2450700, 3552788, 2468010, 2469164, 2444930, 2802093, 2777859, 4406019, 0)
        self.assertEqual(len(VARIANTS), 12)
        for variant, stamp, count in zip(VARIANTS, stamps, counts):
            self.select(variant)
            contract, saved, runtime = self.fixture()
            self.assertIs(self.validate(saved, runtime), contract)
            self.assertEqual(self.spec["route"], self.route.ROUTE)
            self.assertEqual(self.spec["new_parameters"], count)
            for benchmark in ("mmau", "mmar"):
                argv = ["--benchmark", benchmark, "--eval-variant", variant]
                if stamp:
                    args = runner.parse_args(argv)
                    self.assertEqual(args.run_root.name, "formal_" + stamp)
                else:
                    with patch("sys.stderr"), self.assertRaises(SystemExit): runner.parse_args(argv)
                    args = runner.parse_args(argv + ["--run-root", "/actual completed run"])
                step = 34626 if variant == "recursive_mesh_5_10x4_5" else 17313
                self.assertEqual(args.expected_step, step)
                self.assertEqual(args.checkpoint, args.run_root / f"formal/checkpoints/model--step-{step}.ckpt")
                self.assertEqual(args.runtime_config, args.run_root / "formal/runtime.json")
                self.assertEqual(args.expected_geometry, (8, 4, 4) if step == 34626 else (8, 8, 4))
                self.assertEqual(args.output_dir.parent, args.run_root / "eval")

    def test_explicit_checkpoint_infers_its_own_run_and_noncanonical_paths_need_full_triple(self):
        for variant in VARIANTS:
            argv = ["--benchmark", "mmar", "--eval-variant", variant]
            args = runner.parse_args(argv + ["--checkpoint", "/new run/formal/checkpoints/model.ckpt"])
            self.assertEqual(args.runtime_config, Path("/new run/formal/runtime.json").resolve())
        argv = ["--benchmark", "mmar", "--eval-variant", VARIANTS[2]]
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            runner.parse_args(argv + ["--checkpoint", "/weights.ckpt"])
        args = runner.parse_args(argv + ["--checkpoint", "/weights.ckpt", "--runtime-config", "/runtime.json",
                                         "--output-dir", "/independent output"])
        self.assertIsNone(args.run_root)
        self.assertEqual(args.output_dir, Path("/independent output").resolve())

    def test_all_cross_route_pairings_are_rejected(self):
        for variant in VARIANTS:
            self.select(variant)
            _, saved, runtime = self.fixture()
            for other in set(VARIANTS) - {variant}:
                with self.subTest(checkpoint=variant, selected=other), self.assertRaises(ValueError):
                    adapter.validate_checkpoint(saved, runtime, self.route, 17313, eval_variant=other)

    def test_runtime_step_loss_source_and_geometry_are_strict(self):
        for variant in VARIANTS:
            self.select(variant)
            for key, value in (("schema_version", 2), ("mode", "smoke"), ("state_dict", {}),
                               ("total_step", 20), ("num_epochs", 2), ("loss_reduction", "wrong"),
                               ("contract_sha256", "wrong")):
                _, saved, runtime = self.fixture()
                saved[key] = value
                with self.subTest(variant=variant, key=key), self.assertRaises(ValueError):
                    self.validate(saved, runtime)
            _, saved, runtime = self.fixture()
            with self.assertRaises(ValueError): self.validate(saved, {**runtime, "route": "other"})
            with self.assertRaises(ValueError): self.validate(saved, runtime, expected_geometry=(8, 1, 4))
            for method in ("code_digest", "recursive_model_source_sha256"):
                with patch.object(self.route, method, return_value="changed"), self.assertRaises(ValueError):
                    self.validate(saved, runtime)

    def test_branch_and_input_contract_mutations_are_rejected(self):
        for variant in VARIANTS:
            self.select(variant)
            mutations = [(None, "prefix", [126, 1, 129]), (None, "attention_mask", "pad"),
                    (None, "forward_dtype", "bfloat16"), (None, "text_model_schedule", list(range(30))),
                    (None, "text_model_architecture", "other"), (None, "recursive_text_checkpoint", {}),
                    ("audio_encoder", "frozen", False), ("mapper", "name", "other")]
            mutations += [(key, "wrong_branch_contract", True)
                          for key in adapter.branch_contracts(self.route, self.spec)]
            for section, name, value in mutations:
                contract, saved, runtime = self.fixture()
                (contract if section is None else contract[section])[name] = value
                self.resign(saved, runtime)
                with self.subTest(variant=variant, field=name), self.assertRaises(ValueError):
                    self.validate(saved, runtime)
            contract, saved, runtime = self.fixture()
            contract["audio_cross_attention" if self.spec["kind"] == "serial84" else "audio_reinjection"] = {}
            self.resign(saved, runtime)
            with self.assertRaises(ValueError): self.validate(saved, runtime)

    def test_loop_lora_contract_is_required_and_cannot_be_added_to_parent(self):
        parent = "recursive_5_10_5_layer8_4_inject_audio_reinject"
        self.select(parent + "_loop_lora_r8")
        for key, value in (("rank", 4), ("alpha", 8), ("targets", []),
                           ("physical_layers_0based", list(range(20))), ("loops", 1)):
            contract, saved, runtime = self.fixture()
            contract["loop_lora"][key] = value
            self.resign(saved, runtime)
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, "loop_lora"):
                self.validate(saved, runtime)
        contract, saved, runtime = self.fixture()
        lora_contract = contract.pop("loop_lora")
        self.resign(saved, runtime)
        with self.assertRaisesRegex(ValueError, "loop_lora"): self.validate(saved, runtime)
        self.select(parent)
        contract, saved, runtime = self.fixture()
        contract["loop_lora"] = lora_contract
        self.resign(saved, runtime)
        with self.assertRaisesRegex(ValueError, "unexpected CED branch"): self.validate(saved, runtime)

    def test_original_protocol_and_fifteen_runner_functions_are_unchanged(self):
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False)
                    for n in ast.parse(path.read_text(encoding="utf-8-sig")).body if isinstance(n, ast.FunctionDef)}
        original = functions(SCRIPTS / "evaluate_mizar_s1_ced_benchmark.py")
        actual = functions(Path(runner.__file__))
        self.assertEqual(set(original), set(actual))
        self.assertEqual({name for name in original if original[name] != actual[name]},
                         {"parse_args", "generate", "_identity"})
        self.assertEqual(adapter.PROTOCOL, adapter.reference.PROTOCOL)
        self.assertEqual(runner.EXPECTED_ROWS, 1000)
        for relative in ("data/fixed20.py", "data/audiotext_dataset.py", "models/ced_offline.py"):
            expected = ast.dump(ast.parse((adapter.reference.ROUTE_ROOT / relative).read_text(encoding="utf-8-sig")))
            for spec in adapter.ROUTE_SPECS.values():
                self.assertEqual(ast.dump(ast.parse((spec["route_root"] / relative).read_text(encoding="utf-8-sig"))), expected)

    def run_loader(self, fixture):
        args, model, modules, packages = fixture
        def after_load(*_, **__):
            model.load_state_dict.assert_called_once()
            model.requires_grad_.assert_not_called()
            return {"status": "PASS"}
        with (patch.dict(sys.modules, packages), patch.object(adapter, "import_training_route", return_value=modules),
              patch.object(adapter, "audit_assets", return_value={"status": "PASS"}),
              patch.object(adapter, "audit_recursive_route", side_effect=after_load),
              patch.object(adapter, "install_recursive_trace_audit", return_value={"status": "PENDING"})):
            return adapter.load_runtime(args)

    def test_full_audio_state_loads_strictly_before_audit_and_freeze_without_vocab_growth(self):
        for variant in VARIANTS:
            self.select(variant)
            for ddp in (False, True):
                with tempfile.TemporaryDirectory() as directory:
                    state = {name: [i] for i, name in enumerate(("audio_encoder.base.ced.weight",
                        "audio_encoder.base.mapper.weight", "caption_decoder.lm.model.layers.5.weight",
                        "trained_branch.weight"))}
                    fixture = self.loader_fixture(directory, {"module." + k: v for k, v in state.items()} if ddp else state)
                    result, model = self.run_loader(fixture), fixture[1]
                    model.load_state_dict.assert_called_once_with(state, strict=True)
                    calls = [c[0] for c in model.method_calls]
                    self.assertLess(calls.index("float"), calls.index("load_state_dict"))
                    model.requires_grad_.assert_called_once_with(False)
                    model.caption_decoder.lm.resize_token_embeddings.assert_not_called()
                    self.assertEqual(result[5]["tokenizer"]["checkpoint_source"]["vocabulary_growth"], 0)
                    self.assertEqual(result[5]["tokenizer"]["pad_id"], 17)
                    self.assertEqual(result[5]["ddp_state_dict_prefix_normalized"], ddp)

    def test_loader_rejects_missing_keys_mixed_ddp_and_wrong_precision_or_mode(self):
        for variant in VARIANTS:
            self.select(variant)
            for kind in ("ddp", "strict", "dtype", "mode"):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = self.loader_fixture(directory, {"module.a": [1], "b": [2]} if kind == "ddp" else {"a": [1]})
                    model = fixture[1]
                    if kind == "strict": model.load_state_dict.side_effect = RuntimeError("Missing trained gate")
                    elif kind == "dtype": model.parameters.return_value = [types.SimpleNamespace(dtype="float16")]
                    elif kind == "mode": model.modules.return_value = [types.SimpleNamespace(training=True)]
                    with self.subTest(variant=variant, kind=kind), self.assertRaises((ValueError, RuntimeError)):
                        self.run_loader(fixture)

    def test_generate_encodes_once_per_sample_and_uses_exact_training_context(self):
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
            mask = Tensor()
            encoder.tokenize.return_value = {"input_ids": Tensor(), "attention_mask": mask}
            tokenizer.encode.return_value = [1, 2]
            contexts = []
            for _ in range(2):
                if self.spec["context_kind"] == "pair":
                    context, output = (object(), object()), {}
                elif self.spec["context_kind"] == "seeds":
                    context, output = (object(), object(), object()), {}
                elif self.spec["context_kind"] == "none":
                    context, output = None, {}
                else:
                    grids = object()
                    output = {"layer_grids": grids}
                    context = None if self.spec["kind"] == "listen36912" else {"audio_grids": grids, "prompt_mask": mask}
                contexts.append((output, context))
            model.generate_prefix_inference.side_effect = [(Tensor(), out, ctx) for out, ctx in contexts]
            with (patch.dict(sys.modules, {"torch": types.SimpleNamespace(inference_mode=nullcontext)}),
                  patch.object(runner, "_fixed20_pair", return_value=(Tensor(), Tensor())),
                  patch.object(adapter, "greedy_decode", return_value={"generated_text": " raw EOS ", "generated_token_count": 1}) as decode):
                for _ in contexts:
                    result = runner.generate((model, tokenizer, "cuda:0", encoder, None,
                                              {"eval_variant": variant}, None), {"prompt": "Question?"})
                    self.assertEqual(result["first_answer_logit_position"], 382)
            self.assertEqual(model.generate_prefix_inference.call_count, 2)
            for call, (out, ctx) in zip(decode.call_args_list, contexts):
                got = call.kwargs["context"]
                if self.spec["context_kind"] == "pair":
                    self.assertIs(got["audio_information"], ctx)
                elif self.spec["context_kind"] == "seeds":
                    self.assertIs(got["audio_memory_seeds"], ctx)
                elif self.spec["context_kind"] == "none":
                    self.assertEqual(got, {})
                else:
                    self.assertIs(got["audio_grids"], out["layer_grids"])
                    self.assertIs(got["prompt_mask"], mask)
            encoder.tokenize.assert_called_with("question?", 256)

    def test_sample_context_rejects_missing_or_stale_four_depth_context(self):
        for variant in VARIANTS[:2]:
            for context in (None, {}, {"audio_grids": object()}):
                with self.assertRaises(ValueError): adapter.sample_context(variant, {}, context, {})
        with self.assertRaises(ValueError): adapter.sample_context(VARIANTS[4], {}, {}, {})
        with self.assertRaises(ValueError): adapter.sample_context("recursive_mesh_5_10x4_5", {}, (object(), object()), {})

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires NumPy parameter stand-ins")
    def test_actual_seven_slot_validators_preserve_learned_rows_and_gates(self):
        import numpy as np
        class Parameter(np.ndarray):
            def data_ptr(self): return self.__array_interface__["data"][0]
            def detach(self): return self
            def ne(self, value): return self != value
        class Linear:
            def __init__(self, rows):
                self.weight = np.arange(rows * 576, dtype=np.float32).reshape(rows, 576).view(Parameter)
                self.bias = np.arange(rows, dtype=np.float32).view(Parameter) + .25
            def parameters(self): return (self.weight, self.bias)
        for suffix in ("layer4_8_audio7slot_split3", "layer4_8_audio7slot_split3_inject", "layer4_8_audio7slot_split3_prompt_mix"):
            self.select("recursive_mesh_" + suffix)
            core = types.SimpleNamespace(listen_audio_tokens=126, audio_memory_slots=7, text_memory_slots=5,
                audio_router_initialization="split_slot0_three_ways", config=types.SimpleNamespace(
                    audio_injection="post_read_tanh_linear_zero"),
                write_routers=[Linear(5) for _ in range(3)], read_routers=[Linear(5) for _ in range(3)],
                audio_listen_write_routers=[Linear(7) for _ in range(3)],
                audio_listen_read_routers=[Linear(7) for _ in range(3)])
            if suffix.endswith("_inject"): core.audio_gates = {str(k): Linear(1) for k in (4, 8)}
            model = types.SimpleNamespace(model=core, config=types.SimpleNamespace(hidden_size=576))
            routers = core.audio_listen_write_routers + core.audio_listen_read_routers
            parameters = [p for r in routers for p in r.parameters()]
            if hasattr(core, "audio_gates"):
                parameters += [p for r in core.audio_gates.values() for p in r.parameters()]
            before = [p.copy() for p in parameters]
            with patch.dict(sys.modules, {"contract": self.route, "torch": types.SimpleNamespace(equal=np.array_equal)}):
                loader = load_module("seven_slot_validator", self.spec["route_root"] / "models/recursive_text.py")
                branch = loader.validate_audio_listen(model, fresh=False)
                self.assertEqual((branch["audio_memory_slots"], branch["text_memory_slots"], branch["total_routers"]), (7, 5, 12))
                with self.assertRaises(ValueError): loader.validate_audio_listen(model, fresh=True)
                for actual, original in zip(parameters, before): np.testing.assert_array_equal(actual, original)
                core.audio_listen_read_routers[0] = core.audio_listen_write_routers[0]
                with self.assertRaises(ValueError): loader.validate_audio_listen(model, fresh=False)

    def test_trace_hooks_check_physical_calls_branches_routers_and_query32(self):
        for variant in VARIANTS:
            self.select(variant)
            lm = Mock()
            core, kind = lm.model, self.spec["kind"]
            core.ablate_after_two_loops = False
            core.layers = [Mock() for _ in range(20)]
            groups = self.spec["loops"] + 1
            core.write_routers, core.read_routers = [Mock() for _ in range(groups)], [Mock() for _ in range(groups)]
            core.audio_listen_write_routers, core.audio_listen_read_routers = [Mock() for _ in range(3)], [Mock() for _ in range(3)]
            core.audio_gates = {str(k): Mock() for k in (4, 8)}
            core.listen_routers = {str(k): Mock() for k in (4, 8)}
            core.audio_listen.listen_routers = [Mock() for _ in range(2)]
            core.audio_listen.last_rounds = [0, 1]
            core.last_injection_trace = core.last_listen_trace = tuple(zip((5, 15), self.spec["interaction_layers"]))
            core.last_crossattn_trace = (5, 15)
            core.last_reinject_trace = (5, 15)
            core.last_serial_injection_trace = ((5, "input"), (5, "layer8"), (15, "input"), (15, "layer4"))
            lora_projections = {}
            if self.spec.get("loop_lora"):
                core.loop_lora_enabled, core.gradient_checkpointing = True, False
                core.last_loop_lora_trace = tuple((i, self.spec["schedule"][i], 0 if i < 15 else 1) for i in range(5, 25))
                for physical in range(5, 15):
                    lora_projections[physical] = {target: Mock(loop_index=None,
                        in_features=1536 if target == "mlp.down_proj" else 576) for target in adapter.LOOP_LORA_TARGETS}
                    core.layers[physical].get_submodule.side_effect = lora_projections[physical].__getitem__
            audit = adapter.install_recursive_trace_audit(lm, eval_variant=variant)
            begin = lm.register_forward_pre_hook.call_args.args[0]
            finish = lm.register_forward_hook.call_args.args[0]
            output = types.SimpleNamespace(past_key_values=None, logits=types.SimpleNamespace(shape=(1, 383, 576)))
            def hook(module, inputs=()):
                self.assertIsNone(module.register_forward_hook.call_args.args[0](module, inputs, None))
            def execute(*, bad_order=False, bad_compression=False, reverse_injection=False, omit_audio_router=False,
                        reverse_serial=False, duplicate_projection=False, wrong_loop=False,
                        omit_lora=False, duplicate_lora=False):
                begin(lm, (), {"use_cache": False, "attention_mask": None})
                if kind == "serial84":
                    for _ in range(2 if duplicate_projection else 1):
                        hook(core.audio_linear, (types.SimpleNamespace(shape=(1, 126, 576)),))
                if kind == "query32":
                    q = types.SimpleNamespace(shape=(4, 31 if bad_compression else 32, 576))
                    kv = types.SimpleNamespace(shape=(4, 126, 576))
                    hook(core.audio_crossattn.compression_attention, (q, kv, kv))
                for logical, physical in enumerate(self.spec["schedule"]):
                    if logical >= 5 and (logical - 5) % 10 == 0 and self.spec["mesh"]:
                        index = (logical - 5) // 10
                        if self.spec["split_memory"]:
                            hook(core.audio_listen_write_routers[index])
                            if not omit_audio_router: hook(core.audio_listen_read_routers[index])
                        if bad_order: hook(core.read_routers[index])
                        hook(core.write_routers[index])
                        if not bad_order: hook(core.read_routers[index])
                        if index < 2:
                            if kind == "listen36912":
                                hook(core.audio_listen.listen_routers[index])
                                hook(core.audio_listen, (None, None, None, None, index, None))
                            elif self.spec["interaction_layers"]:
                                layer = str(self.spec["interaction_layers"][1 - index if reverse_injection else index])
                                hook(core.audio_gates[layer])
                                if kind == "listen48": hook(core.listen_routers[layer])
                    elif logical in (5, 15):
                        if kind == "serial84":
                            if not reverse_serial: hook(core.audio_gate)
                            order = (4, 8) if reverse_injection else (8, 4)
                            hook(core.audio_gates[str(order[(logical - 5) // 10])])
                            if reverse_serial: hook(core.audio_gate)
                        else:
                            hook(core.audio_crossattn, (None, None, None, 0 if logical == 5 else 1))
                    if self.spec.get("loop_lora") and 5 <= logical < 25:
                        try:
                            for target, projection in lora_projections[physical].items():
                                projection.loop_index = (1 if logical < 15 else 0) if wrong_loop else (0 if logical < 15 else 1)
                                if omit_lora and target == "self_attn.q_proj": continue
                                for _ in range(2 if duplicate_lora else 1):
                                    projection.register_forward_pre_hook.call_args.args[0](projection,
                                        (types.SimpleNamespace(shape=(1, 383, projection.in_features)),))
                            hook(core.layers[physical])
                        finally:
                            for projection in lora_projections[physical].values(): projection.loop_index = None
                    else:
                        hook(core.layers[physical])
                finish(lm, (), output)
            execute()
            execute()
            self.assertEqual(audit["checked_forwards"], 2)
            self.assertEqual(audit["last_physical_schedule"], list(self.spec["schedule"]))
            for kwargs in ({"use_cache": True}, {"use_cache": False, "past_key_values": object()},
                           {"use_cache": False, "attention_mask": object()}):
                with self.assertRaises(RuntimeError): begin(lm, (), kwargs)
            begin(lm, (), {"use_cache": False})
            with self.assertRaisesRegex(RuntimeError, "physical schedule"): finish(lm, (), output)
            if self.spec["mesh"]:
                with self.assertRaisesRegex(RuntimeError, "router invocation order"): execute(bad_order=True)
            if self.spec["split_memory"]:
                with self.assertRaisesRegex(RuntimeError, "router invocation order"): execute(omit_audio_router=True)
            if self.spec["interaction_layers"]:
                with self.assertRaisesRegex(RuntimeError, "interaction order"): execute(reverse_injection=True)
            if kind == "query32":
                with self.assertRaisesRegex(RuntimeError, "compress"): execute(bad_compression=True)
            if kind == "serial84":
                with self.assertRaisesRegex(RuntimeError, "interaction order"): execute(reverse_serial=True)
                with self.assertRaisesRegex(RuntimeError, "projected once"): execute(duplicate_projection=True)
            if self.spec.get("loop_lora"):
                self.assertEqual(audit["loop_lora_projection_calls"], 140)
                self.assertEqual(audit["loop_lora_calls_per_loop"], [70, 70])
                with self.assertRaisesRegex(RuntimeError, "wrong physical layer or loop"): execute(wrong_loop=True)
                for kwargs in ({"omit_lora": True}, {"duplicate_lora": True}):
                    with self.assertRaisesRegex(RuntimeError, "140 full-sequence"): execute(**kwargs)
                projection = lora_projections[5]["self_attn.q_proj"]
                projection.loop_index = 0
                with self.assertRaisesRegex(RuntimeError, "reset between forwards"): execute()
                projection.loop_index = None
                core.gradient_checkpointing = True
                with self.assertRaisesRegex(RuntimeError, "gradient checkpointing"): execute()
                core.gradient_checkpointing = False
                execute()
            if kind == "mesh_x4":
                core.ablate_after_two_loops = True
                with self.assertRaisesRegex(RuntimeError, "four trained"): execute()


@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy for executable decode stand-ins")
class NumpyDecodeTests(unittest.TestCase):
    def fixture(self, variant, ids):
        import numpy as np
        class Tensor(np.ndarray):
            device = "cpu"
            def __new__(cls, value, dtype=np.float32): return np.asarray(value, dtype=dtype).view(cls)
            def sum(self, dim=None, **kwargs): return super().sum(axis=dim, **kwargs)
        torch = types.SimpleNamespace(Tensor=Tensor, float32=np.dtype("float32"), long=np.int64,
            isfinite=np.isfinite, argmax=lambda value, dim: np.argmax(value, axis=dim),
            cat=lambda values, dim: Tensor(np.concatenate(values, axis=dim)),
            tensor=lambda value, device, dtype: Tensor(value, dtype))
        prefix = Tensor(np.zeros((1, 383, 576)))
        kind = adapter.variant_spec(variant)["context_kind"]
        if kind in ("pair", "seeds"):
            key, count = ("audio_information", 2) if kind == "pair" else ("audio_memory_seeds", 3)
            context = {key: tuple(Tensor(np.full((1, 126, 576), i)) for i in range(count))}
        elif kind == "none":
            context = {}
        else:
            context = {"audio_grids": Tensor(np.zeros((1, 4, 126, 1536))),
                       "prompt_mask": Tensor([[1] * 180 + [0] * 76], np.int64)}
        def forward(**kwargs):
            token = next(tokens)
            logits = Tensor(np.zeros((1, kwargs["inputs_embeds"].shape[1], 8)))
            logits[0, -1, token] = 1
            return types.SimpleNamespace(logits=logits)
        tokens = iter(ids)
        lm = Mock(side_effect=forward)
        lm.get_input_embeddings.return_value = lambda ids: Tensor(np.zeros((1, 1, 576)))
        model = types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=lm))
        tokenizer = types.SimpleNamespace(eos_token_id=0, decode=Mock(return_value="  A) raw <|endoftext|>  "))
        return model, tokenizer, prefix, context, torch

    def test_eos_full_recomputation_explicit_context_and_raw_text(self):
        for variant in VARIANTS:
            model, tokenizer, prefix, context, torch = self.fixture(variant, [3, 0, 7])
            with patch.dict(sys.modules, {"torch": torch}):
                result = adapter.greedy_decode(model, tokenizer, prefix, eval_variant=variant, context=context)
            self.assertEqual(result["generated_token_ids"], [3, 0])
            self.assertEqual(result["generated_text"], "  A) raw <|endoftext|>  ")
            self.assertEqual(result["stop_reason"], "eos_token")
            self.assertEqual([c.kwargs["inputs_embeds"].shape[1] for c in model.caption_decoder.lm.call_args_list], [383, 384])
            for call in model.caption_decoder.lm.call_args_list:
                self.assertIsNone(call.kwargs["attention_mask"])
                self.assertIs(call.kwargs["use_cache"], False)
                for key, value in context.items(): self.assertIs(call.kwargs[key], value)
            tokenizer.decode.assert_called_once_with([3, 0], skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def test_both_benchmarks_keep_300_token_limit_and_finite_context(self):
        for variant in VARIANTS:
            model, tokenizer, prefix, context, torch = self.fixture(variant, [3] * 300)
            with patch.dict(sys.modules, {"torch": torch}):
                result = adapter.greedy_decode(model, tokenizer, prefix, eval_variant=variant, context=context)
                self.assertEqual(result["generated_token_count"], 300)
                self.assertEqual(result["stop_reason"], "max_new_tokens")
                with self.assertRaises(ValueError):
                    adapter.greedy_decode(model, tokenizer, prefix, 32, eval_variant=variant, context=context)
                invalid = [None, {"audio_grids": object(), "prompt_mask": object()}]
                if context: invalid.append({})
                for bad in invalid:
                    with self.assertRaises(ValueError): adapter.validate_context(variant, bad, prefix)
                if not context: continue
                value = next(iter(context.values()))
                target = value[0] if isinstance(value, tuple) else value
                target.flat[0] = float("nan")
                with self.assertRaises(ValueError): adapter.validate_context(variant, context, prefix)


if __name__ == "__main__":
    unittest.main()
