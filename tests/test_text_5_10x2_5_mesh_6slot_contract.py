"""Exercise actual conversion/architecture/checkpoint guards without ML packages."""
from __future__ import annotations

import ast
import copy
import json
import math
import random
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code/RSmol"
MODEL = CODE / "recursive_model_5_10x2_5_mesh_6slot.py"
CONVERTER = CODE / "scripts/convert_stepwise_5_10x2_5_mesh_6slot.py"
TRAINER = CODE / "scripts/train_stage4_5_10x2_5_mesh_6slot_ddp.py"


def production_functions(path, names, namespace=None):
    scope = dict(namespace or {})
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), scope)
    return types.SimpleNamespace(**scope)


def model_constants():
    scope = {}
    tree = ast.parse(MODEL.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets):
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(MODEL), "exec"), scope)
    return {key: value for key, value in scope.items() if key.isupper()}


class SixSlotConversionContractTest(unittest.TestCase):
    def setUp(self):
        self.constants = model_constants()
        self.converter = production_functions(CONVERTER, {"detect_source", "_training_markers", "_target_config", "_restore_tied_weight_aliases", "_resolved", "reject_forbidden_output"}, {
            **self.constants, "copy": copy, "json": json, "Path": Path,
            "TRAINING_MARKERS": {"training_state.pt", "optimizer.pt", "scheduler.pt"},
            "LOCAL_CHECKOUT": ROOT, "FORBIDDEN_CHECKOUT": Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM"),
        })

    def source(self, root, config, metadata=None):
        path = root / "source"
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        if metadata is not None:
            (path / "recursive_conversion_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def test_original_thirty_layer_mapping_and_six_slot_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.source(Path(folder), {"num_hidden_layers": 30})
            info = self.converter.detect_source(path)
            self.assertEqual(info["mapping"], (0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29))
            target = self.converter._target_config(types.SimpleNamespace(hidden_size=576), info, path)
            self.assertEqual((target.num_hidden_layers, target.recursive_layer_count, target.recursive_loops), (30,20,2))
            self.assertEqual((target.mesh_memory_slots, target.mesh_router_count), (6,6))
            self.assertEqual(target.mesh_architecture_contract, self.constants["MODEL_ARCHITECTURE_CONTRACT"])
            self.assertIn("6d", target.mesh_router_init)
            self.assertEqual(target.architectures, ["RecursiveLlama5_10x2_5Mesh6SlotForCausalLM"])

    def test_clean_twenty_layer_conversion_is_copied_without_remapping_again(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.source(Path(folder), {"num_hidden_layers":30, "recursive_layer_count":20, "recursive_loops":2}, {"conversion":"SmolLM2-5-10-5", "status":"ok"})
            self.assertEqual(self.converter.detect_source(path)["mapping"], tuple(range(20)))

    def test_trained_or_existing_mesh_sources_are_rejected(self):
        for config, marker in (({"num_hidden_layers":30}, "training_state.pt"),
                               ({"num_hidden_layers":30,"mesh_memory_slots":7}, None),
                               ({"num_hidden_layers":50,"recursive_layer_count":20,"recursive_loops":4}, None)):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as folder:
                path = self.source(Path(folder), config)
                if marker:
                    (path / marker).write_bytes(b"")
                with self.assertRaises(ValueError):
                    self.converter.detect_source(path)

    def test_tied_embedding_alias_is_restored(self):
        tensor = types.SimpleNamespace(shape=(64,32))
        remapped = {"model.embed_tokens.weight":tensor}
        target = {"model.embed_tokens.weight":tensor, "lm_head.weight":tensor}
        self.assertEqual(self.converter._restore_tied_weight_aliases(remapped, target, types.SimpleNamespace(tie_word_embeddings=True)), ["lm_head.weight"])
        self.assertIs(remapped["lm_head.weight"], tensor)

    def test_converter_refuses_model_artifacts_inside_checkout(self):
        with self.assertRaises(ValueError):
            self.converter.reject_forbidden_output(ROOT / "models/accidental")

    def test_model_routing_and_loop_implementation_matches_seven_slot_parent(self):
        parent = ast.parse((CODE / "recursive_model_5_10x2_5_mesh_7slot.py").read_text(encoding="utf-8"))
        current = ast.parse(MODEL.read_text(encoding="utf-8"))
        for name in ("_init_router",):
            left = next(node for node in parent.body if isinstance(node, ast.FunctionDef) and node.name == name)
            right = next(node for node in current.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(left), ast.dump(right))
        classes = [next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MeshLlamaModel") for tree in (parent,current)]
        for name in ("forward", "_route", "_read", "_write", "_run_stack"):
            nodes = [next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name) for cls in classes]
            self.assertEqual(ast.dump(nodes[0]), ast.dump(nodes[1]), name)


class SixSlotCheckpointContractTest(unittest.TestCase):
    def setUp(self):
        self.constants = model_constants()
        self.scope = {**self.constants, "TRAINING_CONTRACT":"text_mesh_x2_6slot_fp32_full_epoch_v1", "PRECISION_CONTRACT":"fp32", "json":json, "Path":Path, "math":math, "random":random}

    def test_loaded_model_requires_six_slots_and_independent_routers(self):
        helpers = production_functions(TRAINER, {"_validate_model_audit", "_validate_loading_info"}, self.scope)
        audit = {"logical_layer_count":30,"physical_layer_count":20,"recursive_loops":2,"memory_slots":6,"router_count":6,"router_parameter_tensor_count":12,"router_objects_independent":True,"memory_persistent":False,"schedule":list(self.constants["LOGICAL_TO_PHYSICAL"])}
        helpers._validate_model_audit(audit)
        for key, bad in (("memory_slots",7),("router_count",10),("router_objects_independent",False),("schedule",list(range(30)))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                helpers._validate_model_audit({**audit,key:bad})
        helpers._validate_loading_info({})
        for key in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                helpers._validate_loading_info({key:["invalid"]})

    def test_rng_roundtrip_includes_python_cpu_and_cuda(self):
        original = random.getstate()
        observed = {}
        fake_torch = types.SimpleNamespace(get_rng_state=lambda:"cpu-state", set_rng_state=lambda value:observed.update(cpu=value), cuda=types.SimpleNamespace(get_rng_state=lambda device:"cuda-state",set_rng_state=lambda value,device:observed.update(cuda=value)))
        helpers = production_functions(TRAINER, {"_capture_rng_state","_restore_rng_state"}, {**self.scope,"torch":fake_torch})
        try:
            device = types.SimpleNamespace(type="cuda")
            state = helpers._capture_rng_state(device)
            expected = random.random()
            helpers._restore_rng_state(state, device)
            self.assertEqual(random.random(), expected)
            self.assertEqual(observed, {"cpu":"cpu-state","cuda":"cuda-state"})
            with self.assertRaises(ValueError):
                helpers._restore_rng_state({**state,"torch_cuda":None}, device)
        finally:
            random.setstate(original)

    def test_checkpoint_rejects_cross_route_step_mismatch_or_missing_rank(self):
        state = {"optimizer":{},"scheduler":{},"optimizer_step":500,"configuration":{"world_size":8},"data_cursors_by_rank":{str(i):{} for i in range(8)},"rng_state":None,"rng_states_by_rank":{str(i):{} for i in range(8)},"manifest":[],"checkpoint_contract":"model_config_tokenizer_optimizer_scheduler_step_data_cursors_rng_manifest"}
        fake_torch = types.SimpleNamespace(load=lambda *args,**kwargs:state)
        helpers = production_functions(TRAINER, {"_validate_checkpoint_complete"}, {**self.scope,"torch":fake_torch})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint-000500"
            path.mkdir()
            base = {"status":"complete","complete_marker":True,"architecture_contract":self.constants["MODEL_ARCHITECTURE_CONTRACT"],"optimizer_step":500,"files":["model.safetensors"]}
            metadata = {**base,"training_contract":self.scope["TRAINING_CONTRACT"],"precision":"fp32","memory_slots":6,"router_groups":3,"router_module_count":6,"router_parameters_in_optimizer":True,"logical_to_physical":list(self.constants["LOGICAL_TO_PHYSICAL"])}
            def write():
                for name, data in (("checkpoint_complete.json",base),("checkpoint_manifest.json",base),("mesh_checkpoint_metadata.json",metadata)):
                    (path / name).write_text(json.dumps(data), encoding="utf-8")
            (path / "model.safetensors").write_bytes(b"fixture")
            write()
            helpers._validate_checkpoint_complete(path)
            for key, invalid in (("memory_slots",7),("training_contract","other-route"),("optimizer_step",499)):
                old = metadata[key]
                metadata[key] = invalid
                write()
                with self.subTest(key=key), self.assertRaises(ValueError):
                    helpers._validate_checkpoint_complete(path)
                metadata[key] = old
            write()
            state["rng_states_by_rank"].pop("7")
            with self.assertRaises(ValueError):
                helpers._validate_checkpoint_complete(path)

    def test_data_and_optimizer_math_match_five_slot_parent(self):
        parent = ast.parse((CODE / "scripts/train_stage4_5_10x2_5_mesh_ddp.py").read_text(encoding="utf-8"))
        current = ast.parse(TRAINER.read_text(encoding="utf-8"))
        for name in ("DistributedParquetStream","token_weighted_gradient_scale","_cosine_lr","_configure_fp32","_assert_fp32_model"):
            nodes = [next(node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==name) for tree in (parent,current)]
            self.assertEqual(ast.dump(nodes[0]),ast.dump(nodes[1]),name)


if __name__ == "__main__":
    unittest.main()
