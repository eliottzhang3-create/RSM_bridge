"""Contracts, strict loader ordering, prefix APIs, observers and real Bash submission."""
from __future__ import annotations

import ast
import copy
import importlib.util
import shlex
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code/RSmol/scripts"), str(ROOT / "code/RSmol")]
import mizar_s1_ced_multilayer_eval_runtime as adapter
import mizar_s1_ced_official_recursive_eval_runtime as official
import mizar_s1_ced_input_ca_eval_runtime as observer
import evaluate_mizar_s1_ced_multilayer_benchmark as runner


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = load("input_ca_previous", ROOT / "tests/test_mizar_s1_ced_eval_extensions.py")
NEW = {name: spec for name, spec in adapter.ROUTE_SPECS.items() if spec.get("input_ca_extension")}
CA, INPUT = tuple(NEW)


class InputCaEvaluationTests(unittest.TestCase):
    fixture = previous.ExtensionTests.fixture
    resign = previous.ExtensionTests.resign
    validate = previous.ExtensionTests.validate

    def select(self, variant):
        self.variant, self.spec = variant, NEW[variant]
        self.route = load("input_ca_contract_" + variant, self.spec["route_root"] / "contract.py")

    def test_actual_completed_contracts_runs_steps_and_exact_paths(self):
        self.assertEqual(len(adapter.ROUTE_SPECS), 26)
        for variant, stamp, step, count in ((CA, "20261109_005039335295104", 17313, 4334405),
                                           (INPUT, "20261107_220147075105401", 34626, 332929)):
            self.select(variant)
            contract, saved, runtime = self.fixture()
            self.assertIs(self.validate(saved, runtime), contract)
            self.assertEqual(self.spec["new_parameters"], count)
            self.assertFalse(self.spec.get("loop_lora"))
            for benchmark in ("mmau", "mmar"):
                args = runner.parse_args(["--benchmark", benchmark, "--eval-variant", variant])
                self.assertEqual(args.run_root.name, "formal_" + stamp)
                self.assertEqual(args.checkpoint, args.run_root / f"formal/checkpoints/model--step-{step}.ckpt")
                self.assertEqual(args.runtime_config, args.run_root / "formal/runtime.json")
                self.assertEqual(args.expected_geometry, (8, 8, 4) if variant == CA else (8, 4, 4))

    def test_source_runtime_step_precision_geometry_and_cross_route_rejection(self):
        for variant in NEW:
            self.select(variant)
            _, saved, runtime = self.fixture()
            for key, value in (("total_step", 1), ("mode", "smoke"), ("contract_sha256", "bad"),
                               ("state_dict", {}), ("num_epochs", 2), ("route_variant", "wrong")):
                with self.subTest(variant=variant, field=key), self.assertRaises(ValueError):
                    self.validate({**saved, key: value}, runtime)
            with self.assertRaises(ValueError): self.validate(saved, {**runtime, "route": "wrong"})
            for key, value in (("code_sha256", "wrong"), ("recursive_model_source_sha256", "wrong"),
                               ("forward_dtype", "bfloat16"), ("attention_mask", "padding")):
                changed, current = copy.deepcopy(saved), copy.deepcopy(runtime)
                changed["contract"][key] = value
                current["contract"] = changed["contract"]
                self.resign(changed, current)
                with self.subTest(variant=variant, contract=key), self.assertRaises(ValueError):
                    self.validate(changed, current)
            with self.assertRaises(ValueError): self.validate(saved, runtime, expected_geometry=(8, 1, 4))
            for other in adapter.ROUTE_SPECS.keys() - {variant}:
                with self.subTest(selected=other), self.assertRaises(ValueError):
                    adapter.validate_checkpoint(saved, runtime, self.route, self.spec["default_step"], eval_variant=other)

    def test_ca_contract_embeds_reinjection_and_input_only_rejects_removed_branches(self):
        for variant in NEW:
            self.select(variant)
            contract, saved, runtime = self.fixture()
            if variant == CA:
                self.assertEqual(adapter.branch_contracts(self.route, self.spec),
                                 {"audio_cross_attention": self.route.cross_attention_contract()})
                self.assertNotIn("audio_reinjection", contract)
                fields = (("audio_cross_attention", {}), ("audio_reinjection", {"old": True}))
            else:
                self.assertIs(contract["loop_lora"], False)
                self.assertIs(contract["encoder_intermediate_injection"], False)
                fields = (("loop_lora", {"rank": 8}), ("encoder_intermediate_injection", {}),
                          ("audio_reinjection", {}))
            for key, value in fields:
                changed, current = copy.deepcopy(saved), copy.deepcopy(runtime)
                changed["contract"][key] = value
                current["contract"] = changed["contract"]
                self.resign(changed, current)
                with self.assertRaises(ValueError): self.validate(changed, current)

    def test_trained_state_overwrites_initial_bundle_before_model_audit_and_freeze(self):
        for variant, spec in NEW.items():
            self.select(variant)
            for ddp in (False, True):
                with tempfile.TemporaryDirectory() as directory:
                    state = {"trained_mapper": [1], "trained_gate": [2]}
                    fixture = previous.legacy.PREVIOUS.RecursiveCedEvaluationTests.loader_fixture(
                        self, directory, {"module." + k: v for k, v in state.items()} if ddp else state)
                    args, model, modules, packages = fixture
                    args.expected_step, args.expected_geometry = spec["default_step"], spec["default_geometry"]
                    sequence = []
                    model.load_state_dict.side_effect = lambda *a, **k: sequence.append("trained_state")
                    model.requires_grad_.side_effect = lambda *a: sequence.append("freeze")
                    def audit(*a, **k):
                        sequence.append("audit")
                        model.load_state_dict.assert_called_once_with(state, strict=True)
                        return {}
                    if spec.get("official"):
                        modules["runtime"] = types.SimpleNamespace(load_model=Mock(return_value=model), validate_state=Mock())
                    importer = official if spec.get("official") else adapter
                    with (patch.dict(sys.modules, packages), patch.object(importer, "import_training_route", return_value=modules),
                          patch.object(importer, "audit_assets", return_value={}),
                          patch.object(adapter, "audit_recursive_route", side_effect=audit),
                          patch.object(adapter, "install_recursive_trace_audit", return_value={})):
                        adapter.load_runtime(args)
                    self.assertEqual(sequence, ["trained_state", "audit", "freeze"])
                    if spec.get("official"): modules["runtime"].validate_state.assert_called_once_with(state, model)

    def test_preprocessing_and_protocol_ast_match_original(self):
        self.assertEqual(adapter.PROTOCOL, adapter.reference.PROTOCOL)
        ref = adapter.reference.ROUTE_ROOT
        def tree(path): return ast.parse(path.read_text(encoding="utf-8-sig"))
        def dump(node): return ast.dump(node, include_attributes=False)
        for name, spec in NEW.items():
            for rel in ("data/fixed20.py", "data/audiotext_dataset.py"):
                self.assertEqual(dump(tree(spec["route_root"] / rel)), dump(tree(ref / rel)))
            ced = (adapter.ROOT / "mizar_official_recursive_5_20x2_5_input_only/ced.py" if name == INPUT
                   else spec["route_root"] / "models/ced.py")
            for method in ("_waveform_to_mel", "_prepare_ced_input", "_hidden_to_temporal_grid"):
                def get(path): return next(n for n in ast.walk(tree(path)) if isinstance(n, ast.FunctionDef) and n.name == method)
                self.assertEqual(dump(get(ced)), dump(get(ref / "models/ced.py")))

    def test_official_model_class_state_count_and_complete_manifest(self):
        self.select(INPUT)
        contract, _, _ = self.fixture()
        root = self.spec["route_root"]
        class Model:
            def state_dict(self): return dict.fromkeys(range(458))
        model = Model()
        model.caption_decoder = types.SimpleNamespace(lm=types.SimpleNamespace(model=types.SimpleNamespace(logical_to_physical=self.spec["schedule"])))
        runtime = types.SimpleNamespace(__file__=str(root / "runtime.py"), model_audit=Mock(return_value={"parameter_count": 159632769}))
        module = types.SimpleNamespace(__file__=str(adapter.implementation_path(self.spec)), MizarInputOnlyRecursive=Model)
        with patch.dict(sys.modules, {"runtime": runtime, self.spec["implementation"]: module}):
            result = official.audit_model(self.route, contract, model, self.spec)
            self.assertEqual(result["interaction_layers"], [])
            runtime.model_audit.return_value = {"parameter_count": 165334019}
            with self.assertRaises(ValueError): official.audit_model(self.route, contract, model, self.spec)
        with patch.object(self.route, "artifact_identity", return_value=contract["initialization_artifact"]):
            self.assertEqual(official.audit_assets(self.route, contract)["evaluated_weights"], "training_checkpoint_state_dict")
        with patch.object(self.route, "artifact_identity", return_value={}), self.assertRaises(ValueError):
            official.audit_assets(self.route, contract)

    def prefix_fixture(self, variant):
        model = Mock()
        encoder = model.audio_encoder.base
        encoder.ced.blocks = [Mock() for _ in range(12)]
        encoder.last_audio_layout = dict(embedding=[1,126,576], hidden=[1,504,384], grid=[1,126,4,384],
            waveform_samples=640000, encoder_eval=True, tap_layers=[4,8,12], tap_norm="pre_final_norm",
            layer_grids=[1,3,126,1536], intermediate_capture=False, encoder_feature_calls=1)
        mask, prefix, grids = object(), object(), object()
        inputs = {"input": {"attention_mask": mask}}
        output = {"layer_grids": grids} if variant == CA else {}
        result = (prefix, output, {"audio_grids": grids, "prompt_mask": mask}) if variant == CA else (prefix, output)
        def encode(_):
            for module in encoder.ced.blocks:
                module.register_forward_hook.call_args.args[0](module, (), types.SimpleNamespace(shape=(2,252,384)))
            encoder.register_forward_hook.call_args.args[0](encoder, (), output)
            return result
        model.generate_prefix_inference.side_effect = encode
        return model, inputs, result

    def test_actual_prefix_api_normalization_context_identity_and_hook_cleanup(self):
        for variant, spec in NEW.items():
            model, inputs, result = self.prefix_fixture(variant)
            audit = {}
            for sample in range(2):
                prefix, output, context = adapter.generate_prefix_inference(model, inputs, eval_variant=variant, audit=audit)
                self.assertIs(prefix, result[0])
                kwargs = adapter.sample_context(variant, output, context, inputs["input"])
                self.assertEqual(set(kwargs), {"audio_grids", "prompt_mask"} if variant == CA else set())
                self.assertEqual(audit["prefix_execution_audit"]["checked_samples"], sample + 1)
            for block in model.audio_encoder.base.ced.blocks:
                self.assertEqual(block.register_forward_hook.return_value.remove.call_count, 2)
            model.generate_prefix_inference.side_effect = RuntimeError("encoder failure")
            with self.assertRaises(RuntimeError): adapter.generate_prefix_inference(model, inputs, eval_variant=variant, audit=audit)
            for block in model.audio_encoder.base.ced.blocks:
                self.assertEqual(block.register_forward_hook.return_value.remove.call_count, 3)

    def test_prefix_observer_rejects_wrong_depth_capture_and_duplicate_encoding(self):
        for variant, spec in NEW.items():
            for failure in ("layout", "duplicate", "api"):
                model, inputs, result = self.prefix_fixture(variant)
                original = model.generate_prefix_inference.side_effect
                def encode(value):
                    encoded = original(value)
                    if failure == "layout": model.audio_encoder.base.last_audio_layout["embedding"] = [1,127,576]
                    if failure == "duplicate":
                        encoder = model.audio_encoder.base
                        encoder.register_forward_hook.call_args.args[0](encoder, (), {})
                    return encoded[:1] if failure == "api" else encoded
                model.generate_prefix_inference.side_effect = encode
                with self.subTest(variant=variant, failure=failure), self.assertRaises((ValueError, RuntimeError)):
                    observer.generate_prefix(model, inputs, spec, {})

    def trace_fixture(self, variant):
        spec, lm = NEW[variant], Mock()
        core = lm.model
        core.layers = [Mock() for _ in range(30 if variant == INPUT else 20)]
        core.last_layer_trace, core.last_input_injection_trace = spec["schedule"], (5,25)
        core.last_input_projection_calls, core.last_crossattn_trace = 1, (5,15)
        branch = core.audio_crossattn
        branch.last_injection_trace = ((0,"input","cross_attention"),(1,"input","cross_attention"))
        branch.audio_input_norms = [Mock() for _ in range(3)]
        branch.projections = [Mock() for _ in range(3)]
        branch.audio_output_norms = [Mock() for _ in range(3)]
        audit = adapter.install_recursive_trace_audit(lm, eval_variant=variant)
        return lm, audit

    def execute_trace(self, variant, lm, *, failure=None, length=383):
        core, spec = lm.model, NEW[variant]
        def value(width, tokens=126): return types.SimpleNamespace(shape=(1,tokens,width))
        def call(module, args, output): module.register_forward_hook.call_args.args[0](module,args,output)
        lm.register_forward_pre_hook.call_args.args[0](lm,(),dict(inputs_embeds=value(576,length),use_cache=False))
        if variant == CA:
            branch = core.audio_crossattn
            for index in range(3):
                call(branch.audio_input_norms[index], (value(1536),), value(1536))
                call(branch.projections[index], (value(1536),), value(576))
                call(branch.audio_output_norms[index], (value(576),), value(576))
            call(branch.input_projection, (value(576),), value(576))
            if failure == "projection": call(branch.input_projection,(value(576),),value(576))
        else:
            call(core.audio_linear,(value(576),),value(576))
            if failure == "projection": call(core.audio_linear,(value(576),),value(576))
        for logical, physical in enumerate(spec["schedule"]):
            if logical in ((5,15) if variant == CA else (5,25)):
                if variant == CA:
                    if not (failure == "missing" and logical == 15):
                        for module, width in ((branch.input_gate,1),(branch.gate1,3),(branch.query_norm,576)):
                            call(module,(value(576),),value(width))
                        call(branch.cross_attention,(value(576),)*3,(value(576),None))
                        call(branch.gate2,(value(576),),value(1))
                        call(branch,(value(576,length),types.SimpleNamespace(shape=(1,3,126,576)),value(576),0 if logical==5 else 1),value(576,length))
                elif not (failure == "missing" and logical == 25): call(core.audio_gate,(value(576),),value(1))
            call(core.layers[physical],(value(576,length),),None)
        output = types.SimpleNamespace(logits=types.SimpleNamespace(shape=(1,length,32)),past_key_values=None)
        if failure == "cache": output.past_key_values = object()
        lm.register_forward_hook.call_args.args[0](lm,(),output)

    def test_observed_full_forward_counts_both_boundaries_and_bank_preparation(self):
        for variant in NEW:
            lm, audit = self.trace_fixture(variant)
            for length in (383,384): self.execute_trace(variant,lm,length=length)
            self.assertEqual(audit["checked_forwards"],2)
            self.assertEqual(audit["last_physical_schedule"],list(NEW[variant]["schedule"]))
            self.assertEqual(audit["input_projection_calls"],1)
            for failure in ("projection","missing","cache"):
                with self.subTest(variant=variant,failure=failure), self.assertRaises(RuntimeError):
                    self.execute_trace(variant,lm,failure=failure)

    def test_decode_preserves_raw_eos_horizon_and_recomputation_for_both_contexts(self):
        for variant in NEW:
            for tokens in ([3,0,7],[3]*300):
                fixture = previous.legacy.NumpyDecodeTests().fixture(variant,tokens)
                model, tokenizer, prefix, context, torch = fixture
                if variant == CA: context["audio_grids"] = context["audio_grids"][:,:3]
                with patch.dict(sys.modules,{"torch":torch}):
                    result = adapter.greedy_decode(model,tokenizer,prefix,eval_variant=variant,context=context)
                    wrong = {**context,"audio_grids":torch.tensor([0],device="cpu",dtype=torch.float32)}
                    with self.assertRaises(ValueError): adapter.validate_context(variant,wrong,prefix)
                count = 2 if len(tokens)==3 else 300
                self.assertEqual(result["generated_token_count"],count)
                self.assertEqual(result["generated_text"],"  A) raw <|endoftext|>  ")
                self.assertEqual([c.kwargs["inputs_embeds"].shape[1] for c in model.caption_decoder.lm.call_args_list],list(range(383,383+count)))


class InputCaShellTests(unittest.TestCase):
    setUp = previous.ExtensionShellTests.setUp
    run_launcher = previous.ExtensionShellTests.run_launcher
    assert_resources = previous.ExtensionShellTests.assert_resources

    def test_four_real_bash_submissions_bind_exact_steps_runs_and_overrides(self):
        for variant,spec in NEW.items():
            for benchmark in ("mmau","mmar"):
                for override in (False,True):
                    with tempfile.TemporaryDirectory() as directory:
                        extra = ("--checkpoint",f'/replacement run/formal/checkpoints/model--step-{spec["default_step"]}.ckpt') if override else ()
                        result,submission = self.run_launcher(directory,benchmark,variant,extra=extra)
                        self.assertEqual(result.returncode,0,result.stderr)
                        self.assert_resources(submission)
                        command = shlex.split(submission[submission.index("--cmd")+1])
                        options = dict(zip(command[2::2],command[3::2]))
                        run = "/replacement run" if override else spec["run_root"].as_posix()
                        self.assertEqual(options["--checkpoint"],run+f'/formal/checkpoints/model--step-{spec["default_step"]}.ckpt')
                        self.assertEqual(options["--runtime-config"],run+"/formal/runtime.json")
                        self.assertEqual(options["--expected-geometry"],",".join(map(str,spec["default_geometry"])))
                        self.assertTrue(options["--route-root"].endswith("/"+spec["route_root"].name))
                        self.assertEqual(options["--eval-variant"],variant)


if __name__ == "__main__": unittest.main()
