"""Seven additional routes: actual contracts, loader ordering and execution audits."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import shlex
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/RSmol/scripts"))
sys.path.insert(0, str(ROOT / "code/RSmol"))
import mizar_s1_ced_multilayer_eval_runtime as adapter
import mizar_s1_ced_official_recursive_eval_runtime as official
import evaluate_mizar_s1_ced_multilayer_benchmark as runner


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load("extension_eval_helpers", ROOT / "tests/test_mizar_s1_ced_multilayer_eval.py")
shell = load("extension_shell_helpers", ROOT / "tests/test_mizar_s1_ced_multilayer_eval_shell.py")
NEW = {name: spec for name, spec in adapter.ROUTE_SPECS.items() if spec.get("extension")}


class ExtensionTests(unittest.TestCase):
    def select(self, variant):
        self.variant, self.spec = variant, NEW[variant]
        self.route = load("extension_contract_" + variant, self.spec["route_root"] / "contract.py")

    def fixture(self):
        prepared = {"rows": 1477492, "source_rows": self.route.SOURCE_COUNTS,
                    "rows_canonical_sha256": "rows", "mapping_sha256": "mapping",
                    "read_policy": "shared_until_complete"}
        if self.spec.get("official"):
            artifact = {"manifest_sha256": "manifest", "files": {"model.ckpt": "sha"}}
            contract = self.route.semantic_contract(prepared, str(ROOT / "fake-model"), artifact)
        else:
            contract = self.route.semantic_contract(prepared, "text-model", "ced-small")
            contract["recursive_text_checkpoint"] = {"fixture_only": True}
        saved = {"schema_version": 1, "route_variant": self.route.ROUTE, "mode": "formal",
                 "state_dict": {"trained_tensor": [1]}, "contract": contract,
                 "total_step": self.spec["default_step"], "num_epochs": 3,
                 "loss_reduction": adapter.TRAINING_LOSS_REDUCTION}
        runtime = {"contract": contract, "route": self.route.ROUTE, "mode": "formal"}
        self.resign(saved, runtime)
        return contract, saved, runtime

    def resign(self, saved, runtime):
        saved["contract_sha256"] = runtime["contract_sha256"] = self.route.digest(saved["contract"])

    def validate(self, saved, runtime, **kwargs):
        return adapter.validate_checkpoint(saved, runtime, self.route, self.spec["default_step"],
                                           eval_variant=self.variant, **kwargs)

    def test_seven_actual_contracts_exact_runs_steps_and_completed_acceptance(self):
        stamps = ("20261107_203129899582015", "20261106_092419698981503", "20261107_215615398982617",
                  "20261107_215916042121396", "20261107_215838489409042", "20261107_215949376143529",
                  "20261107_003434623913704")
        self.assertEqual(len(NEW), 7)
        for (variant, spec), stamp in zip(NEW.items(), stamps):
            self.select(variant)
            contract, saved, runtime = self.fixture()
            self.assertIs(self.validate(saved, runtime), contract)
            self.assertEqual(spec["run_root"].name, "formal_" + stamp)
            for benchmark in ("mmau", "mmar"):
                args = runner.parse_args(["--benchmark", benchmark, "--eval-variant", variant])
                self.assertEqual(args.checkpoint.name, f'model--step-{spec["default_step"]}.ckpt')
                self.assertEqual(args.runtime_config, args.run_root / "formal/runtime.json")
                self.assertEqual(args.expected_geometry, spec["default_geometry"])
                self.assertEqual(args.route_root, spec["route_root"].resolve())

    def test_source_runtime_step_geometry_precision_and_branch_mutations_rejected(self):
        for variant in NEW:
            self.select(variant)
            _, saved, runtime = self.fixture()
            for key, value in (("schema_version", 2), ("route_variant", "wrong"), ("mode", "smoke"),
                               ("num_epochs", 2), ("total_step", 100), ("state_dict", {}),
                               ("contract_sha256", "wrong"), ("loss_reduction", "wrong")):
                with self.subTest(variant=variant, key=key), self.assertRaises(ValueError):
                    self.validate({**saved, key: value}, runtime)
            for key, value in (("route", "wrong"), ("contract_sha256", "wrong"), ("mode", "smoke")):
                with self.assertRaises(ValueError): self.validate(saved, {**runtime, key: value})
            for method in ("code_digest", "recursive_model_source_sha256"):
                with patch.object(self.route, method, return_value="changed"), self.assertRaises(ValueError):
                    self.validate(saved, runtime)
            with self.assertRaises(ValueError): self.validate(saved, runtime, expected_geometry=(8, 1, 4))
            mutations = [("prefix", [126, 1, 128]), ("forward_dtype", "bfloat16"),
                         ("attention_mask", "padding"), ("text_model_contract", "wrong")]
            mutations += [(key, {"wrong": True}) for key in adapter.branch_contracts(self.route, self.spec)]
            if self.spec.get("official"): mutations += [("architecture", {}), ("initialization_artifact", {})]
            else: mutations += [("text_model_architecture", "wrong"), ("recursive_text_checkpoint", {})]
            for key, value in mutations:
                _, changed, current = self.fixture()
                changed["contract"][key] = value
                self.resign(changed, current)
                with self.subTest(variant=variant, contract=key), self.assertRaises(ValueError):
                    self.validate(changed, current)

    def test_cross_route_rejection_includes_all_existing_variants(self):
        for variant in NEW:
            self.select(variant)
            _, saved, runtime = self.fixture()
            for other in adapter.ROUTE_SPECS.keys() - {variant}:
                with self.subTest(saved=variant, selected=other), self.assertRaises(ValueError):
                    adapter.validate_checkpoint(saved, runtime, self.route, self.spec["default_step"], eval_variant=other)

    def test_input_processing_ast_and_protocol_are_identical(self):
        def parsed(path): return ast.parse(path.read_text(encoding="utf-8-sig"))
        def dump(value): return ast.dump(value, include_attributes=False)
        def method(path, name):
            return next(n for c in parsed(path).body if isinstance(c, ast.ClassDef) and c.name == "CEDSmallWrapper"
                        for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
        reference = adapter.ROUTE_SPECS["recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8"]["route_root"]
        self.assertEqual(adapter.PROTOCOL, adapter.reference.PROTOCOL)
        for spec in NEW.values():
            for relative in ("data/fixed20.py", "data/audiotext_dataset.py"):
                self.assertEqual(dump(parsed(spec["route_root"] / relative)), dump(parsed(reference / relative)))
            ced = (adapter.ROOT / "mizar_official_recursive_5_20x2_5/ced.py" if spec.get("official")
                   else spec["route_root"] / "models/ced.py")
            for name in ("_waveform_to_mel", "_prepare_ced_input", "_hidden_to_temporal_grid"):
                self.assertEqual(dump(method(ced, name)), dump(method(reference / "models/ced.py", name)))

    def test_lora_families_have_distinct_counts_scales_and_short_directory_mappings(self):
        values = [s["loop_lora_config"] for s in NEW.values() if s.get("loop_lora")]
        self.assertEqual([s["matrix_count"] for s in values], [280, 400, 520, 560])
        self.assertEqual([s["parameter_count"] for s in values], [814080, 2641920, 3655680, 3256320])
        self.assertEqual(values[0]["rank"], 4)
        self.assertEqual(values[2]["scale_by_family"],
                         {"attention_shared": 2., "mlp_shared": 1., "mlp_audio": 1., "mlp_text": 1.})
        for name, spec in NEW.items():
            if "attn_shared" in name: self.assertIn("in84", spec["route_root"].name)

    def test_official_assets_must_match_the_entire_original_manifest(self):
        self.select("official_final_s1_5_20x2_5")
        contract, _, _ = self.fixture()
        with patch.object(self.route, "artifact_identity", return_value=contract["initialization_artifact"]):
            self.assertEqual(official.audit_assets(self.route, contract)["evaluated_weights"], "training_checkpoint_state_dict")
        with patch.object(self.route, "artifact_identity", return_value={"changed": True}), self.assertRaises(ValueError):
            official.audit_assets(self.route, contract)

    def test_full_checkpoint_is_strictly_loaded_before_audit_and_freeze_for_all_seven(self):
        for variant, spec in NEW.items():
            self.select(variant)
            for ddp in (False, True):
                with self.subTest(variant=variant, ddp=ddp), tempfile.TemporaryDirectory() as directory:
                    state = {"trained_gate": [1], "trained_mapper": [2], "trained_lora": [3]}
                    fixture = legacy.PREVIOUS.RecursiveCedEvaluationTests.loader_fixture(
                        self, directory, {"module." + k: v for k, v in state.items()} if ddp else state)
                    args, model, modules, packages = fixture
                    args.expected_step, args.expected_geometry = spec["default_step"], spec["default_geometry"]
                    sequence = []
                    model.load_state_dict.side_effect = lambda *a, **k: sequence.append("trained_state")
                    model.requires_grad_.side_effect = lambda *a: sequence.append("freeze")
                    def audit(*a, **k):
                        sequence.append("audit")
                        model.load_state_dict.assert_called_once_with(state, strict=True)
                        return {"status": "PASS"}
                    if spec.get("official"):
                        modules["runtime"] = types.SimpleNamespace(load_model=Mock(return_value=model), validate_state=Mock())
                    importer = official if spec.get("official") else adapter
                    with (patch.dict(sys.modules, packages), patch.object(importer, "import_training_route", return_value=modules),
                          patch.object(importer, "audit_assets", return_value={"status": "PASS"}),
                          patch.object(adapter, "audit_recursive_route", side_effect=audit),
                          patch.object(adapter, "install_recursive_trace_audit", return_value={"status": "PENDING"})):
                        result = adapter.load_runtime(args)
                    self.assertEqual(sequence, ["trained_state", "audit", "freeze"])
                    self.assertEqual(result[5]["ddp_state_dict_prefix_normalized"], ddp)
                    if spec.get("official"):
                        modules["runtime"].validate_state.assert_called_once_with(state, model)
                        packages["transformers"].AutoTokenizer.from_pretrained.assert_called_once_with(
                            ROOT / "fake-model/tokenizer", local_files_only=True)

    def test_new_decode_routes_preserve_eos_300_horizon_context_and_recomputation(self):
        helper = legacy.NumpyDecodeTests()
        for variant in NEW:
            for ids in ([3, 0, 7], [3] * 300):
                model, tokenizer, prefix, context, torch = helper.fixture(variant, ids)
                with patch.dict(sys.modules, {"torch": torch}):
                    result = adapter.greedy_decode(model, tokenizer, prefix, eval_variant=variant, context=context)
                count = 2 if len(ids) == 3 else 300
                self.assertEqual(result["generated_token_count"], count)
                self.assertEqual(result["generated_text"], "  A) raw <|endoftext|>  ")
                self.assertEqual([c.kwargs["inputs_embeds"].shape[1] for c in model.caption_decoder.lm.call_args_list], list(range(383, 383 + count)))
                for call in model.caption_decoder.lm.call_args_list:
                    self.assertIsNone(call.kwargs["attention_mask"])
                    self.assertFalse(call.kwargs["use_cache"])
                    for key, value in context.items(): self.assertIs(call.kwargs[key], value)

    def test_execution_hooks_cover_all_slots_families_and_fifty_call_boundaries(self):
        for variant, spec in NEW.items():
            self.select(variant)
            lm = Mock()
            core, length = lm.model, 384
            core.layers = [Mock() for _ in range(spec.get("physical_layers", 20))]
            core.ablate_after_two_loops = False
            core.write_routers, core.read_routers = [Mock() for _ in range(3)], [Mock() for _ in range(3)]
            core.loop_lora_enabled, core.gradient_checkpointing = True, False
            core.audio_gates = {str(k): Mock() for k in (4, 8)}
            boundaries, end = spec.get("boundaries", (5, 15)), spec.get("middle_end", 15)
            core.last_serial_injection_trace = tuple(event for b, k in zip(boundaries, (8, 4))
                                                       for event in ((b, "input"), (b, f"layer{k}")))
            core.last_injection_trace, core.last_reinject_trace = tuple(zip(boundaries, (8, 4))), boundaries
            core.last_layer_trace, core.last_input_projection_calls = spec["schedule"], 1
            core.last_sequence_length, core.last_batch_size = length, 1
            core.last_loop_lora_trace = tuple((logical, physical, int(logical >= end))
                for logical, physical in enumerate(spec["schedule"]) if 5 <= logical < 2 * end - 5)
            projections = {}
            if spec.get("loop_lora"):
                for physical in range(5, end):
                    projections[physical] = {target: Mock(loop_index=None, in_features=1536 if target == "mlp.down_proj" else 576)
                                             for target in adapter.LOOP_LORA_TARGETS}
                    core.layers[physical].get_submodule.side_effect = projections[physical].__getitem__
            namespace = {"contract": self.route}
            if spec.get("family_lora"):
                path = spec["route_root"] / "models/loop_lora.py"
                tree = ast.parse(path.read_text(encoding="utf-8"))
                scope = {"MIDDLE_LAYERS": tuple(range(5, 15)), "TARGETS": adapter.LOOP_LORA_TARGETS,
                         "ATTENTION_TARGETS": adapter.LOOP_LORA_TARGETS[:4], "AUDIO_TOKENS": 126,
                         "LOOPS": 2, "RANK": 8, "ATTENTION_ALPHA": 16, "MLP_ALPHA": 8}
                nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name in ("projections", "validate_loop_lora_execution")]
                exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
                namespace["models.loop_lora"] = types.SimpleNamespace(__file__=str(path), **scope)
            with patch.dict(sys.modules, namespace):
                audit = adapter.install_recursive_trace_audit(lm, eval_variant=variant)
            begin = lm.register_forward_pre_hook.call_args.args[0]
            finish = lm.register_forward_hook.call_args.args[0]
            def hook(module):
                module.register_forward_hook.call_args.args[0](module, (types.SimpleNamespace(shape=(1, length, 576)),), None)
            def execute(missing=False, reversed_gate=False, wrong_shape=False, wrong_family=False):
                begin(lm, (), {"use_cache": False, "attention_mask": None})
                if spec.get("loop_lora"):
                    for targets in projections.values():
                        for target, projection in targets.items():
                            projection.forward_counts = [0, 0]
                            projection.last_token_shapes = [None, None]
                            projection.rank, projection.alpha, projection.scaling = 8, 16, 2.
                            projection.mode = "shared"
                            branches = ("shared",)
                            if spec.get("family_lora") and target.startswith("mlp."):
                                both = spec["loop_lora_config"]["mlp"] == "shared_plus_modality"
                                projection.mode = "shared_modality" if both else "modality"
                                if both: projection.alpha, projection.scaling = 8, 1.
                                branches = ("shared", "audio", "text") if both else ("audio", "text")
                            projection.branch_counts = [{b: 0 for b in branches} for _ in range(2)]
                if spec["interaction_layers"]:
                    core.audio_linear.register_forward_hook.call_args.args[0](core.audio_linear,
                        (types.SimpleNamespace(shape=(1, 126, 576)),), None)
                for logical, physical in enumerate(spec["schedule"]):
                    if spec["mesh"] and logical in (5, 15, 25):
                        index = (logical - 5) // 10
                        for router in (core.write_routers[index], core.read_routers[index]):
                            router.register_forward_hook.call_args.args[0](router,
                                (types.SimpleNamespace(shape=(1, length - int(wrong_shape), 576)),), None)
                    if spec["interaction_layers"] and logical in boundaries:
                        layer = (8, 4)[boundaries.index(logical)]
                        if reversed_gate: hook(core.audio_gates[str(layer)])
                        hook(core.audio_gate)
                        if not reversed_gate: hook(core.audio_gates[str(layer)])
                    for target, projection in projections.get(physical, {}).items():
                        if missing and target == "self_attn.q_proj": continue
                        loop = int(logical >= end)
                        projection.loop_index = loop
                        projection.register_forward_pre_hook.call_args.args[0](projection,
                            (types.SimpleNamespace(shape=(1, length, projection.in_features)),))
                        projection.forward_counts[loop] += 1
                        projection.last_token_shapes[loop] = (1, 126, length - 126)
                        for branch in projection.branch_counts[loop]: projection.branch_counts[loop][branch] += 1
                        if wrong_family and target.startswith("mlp."): projection.mode = "wrong"
                    hook(core.layers[physical])
                    for projection in projections.get(physical, {}).values(): projection.loop_index = None
                output = types.SimpleNamespace(past_key_values=None, logits=types.SimpleNamespace(shape=(1, length, 32)))
                finish(lm, (), output)
            execute()
            execute()
            self.assertEqual(audit["checked_forwards"], 2)
            self.assertEqual(audit["last_physical_schedule"], list(spec["schedule"]))
            if spec.get("loop_lora"):
                self.assertEqual(audit["loop_lora_projection_calls"], 280 if spec.get("official") else 140)
                with self.assertRaisesRegex(RuntimeError, "full-sequence projection"): execute(missing=True)
                with self.assertRaisesRegex(RuntimeError, "interaction order"): execute(reversed_gate=True)
                if spec.get("family_lora"):
                    self.assertEqual(audit["family_lora_execution"]["adapter_calls"], spec["loop_lora_config"]["matrix_count"] // 2)
                    with self.assertRaises(ValueError): execute(wrong_family=True)
            else:
                self.assertEqual(audit["full_memory_shape_from_router_geometry"], [1, spec["memory_slots"], length, 576])
                with self.assertRaisesRegex(RuntimeError, "complete multimodal"): execute(wrong_shape=True)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires NumPy source execution")
    def test_actual_lora_forward_nonzero_branches_scales_and_base_projection_once(self):
        import numpy as np
        for spec in NEW.values():
            if not spec.get("loop_lora"): continue
            path = (adapter.ROOT / "mizar_official_recursive_5_20x2_5/loop_lora.py" if spec.get("official")
                    else spec["route_root"] / "models/loop_lora.py")
            tree = ast.parse(path.read_text(encoding="utf-8"))
            forward = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "LoopLoRALinear"
                           for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
            for mlp in (False, True):
                rng = np.random.default_rng(1234)
                rank = spec["loop_lora_config"]["rank"]
                both = spec.get("family_lora") and spec["loop_lora_config"]["mlp"] == "shared_plus_modality"
                mode = "shared_modality" if both and mlp else "modality" if spec.get("family_lora") and mlp else "shared"
                branches = ("shared", "audio", "text") if mode == "shared_modality" else ("audio", "text") if mode == "modality" else ("shared",)
                a = {b: [rng.normal(size=(rank, 3)) * .1 for _ in range(2)] for b in branches}
                b = {k: [rng.normal(size=(3, rank)) * .1 for _ in range(2)] for k in branches}
                module = types.SimpleNamespace(weight=np.eye(3), bias=np.ones(3) * .1, scaling=1. if both and mlp else 2.,
                    lora_A=a if spec.get("family_lora") else a["shared"],
                    lora_B=b if spec.get("family_lora") else b["shared"], loop_index=0, mode=mode, audio_tokens=126,
                    forward_counts=[0, 0], last_token_shapes=[None, None], branch_counts=[{k: 0 for k in branches} for _ in range(2)])
                base_calls = []
                def linear(x, weight, bias=None):
                    if weight is module.weight: base_calls.append(1)
                    return x @ weight.T + (0 if bias is None else bias)
                scope = {"F": types.SimpleNamespace(linear=linear),
                         "torch": types.SimpleNamespace(cat=lambda xs, dim: np.concatenate(xs, axis=dim))}
                exec(compile(ast.Module(body=[forward], type_ignores=[]), str(path), "exec"), scope)
                value = rng.normal(size=(1, 130, 3))
                outputs = []
                for loop in (0, 1):
                    module.loop_index = loop
                    expected = value @ module.weight.T + module.bias
                    if "shared" in branches: expected += module.scaling * (value @ a["shared"][loop].T @ b["shared"][loop].T)
                    for branch, region in (("audio", slice(None, 126)), ("text", slice(126, None))):
                        if branch in branches: expected[:, region] += module.scaling * (value[:, region] @ a[branch][loop].T @ b[branch][loop].T)
                    actual = scope["forward"](module, value)
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                    outputs.append(actual)
                self.assertEqual(len(base_calls), 2)
                self.assertFalse(np.allclose(*outputs))
                module.loop_index = None
                with self.assertRaises(RuntimeError): scope["forward"](module, value)


class ExtensionShellTests(unittest.TestCase):
    # Reuse the real Bash/fake-vc harness; old launcher tests remain in their
    # original suite, while this suite exercises every new route and override.
    setUp = shell.MultilayerCedShellTests.setUp
    run_launcher = shell.MultilayerCedShellTests.run_launcher
    assert_resources = shell.MultilayerCedShellTests.assert_resources

    def test_seven_default_runs_and_explicit_checkpoint_overrides(self):
        for variant, spec in NEW.items():
            for benchmark in ("mmau", "mmar"):
                for override in (False, True):
                    with self.subTest(variant=variant, benchmark=benchmark, override=override), tempfile.TemporaryDirectory() as directory:
                        extra = ("--checkpoint", f'/replacement run/formal/checkpoints/model--step-{spec["default_step"]}.ckpt') if override else ()
                        result, submission = self.run_launcher(directory, benchmark, variant, extra=extra)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assert_resources(submission)
                        command = shlex.split(submission[submission.index("--cmd") + 1])
                        options = dict(zip(command[2::2], command[3::2]))
                        run = "/replacement run" if override else spec["run_root"].as_posix()
                        self.assertEqual(options["--checkpoint"], run + f'/formal/checkpoints/model--step-{spec["default_step"]}.ckpt')
                        self.assertEqual(options["--runtime-config"], run + "/formal/runtime.json")
                        self.assertEqual(options["--expected-geometry"], ",".join(map(str, spec["default_geometry"])))
                        self.assertTrue(options["--route-root"].endswith("/" + spec["route_root"].name))
                        self.assertEqual(options["--eval-variant"], variant)


if __name__ == "__main__":
    unittest.main()
