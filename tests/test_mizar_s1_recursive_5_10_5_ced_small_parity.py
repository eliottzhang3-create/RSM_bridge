"""Three-way training parity and real non-MeSH loader rejection checks, without torch."""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine"
ROUTE = FAMILY / "mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small"
ORIGINAL = FAMILY / "mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5"
sys.path.insert(0, str(ROUTE))
import audit_reference
import contract
from models import recursive_text


def text_config():
    return dict(model_type="llama", architectures=["RecursiveLlamaForCausalLM"],
                hidden_size=576, num_hidden_layers=30, recursive_layer_count=20,
                recursive_loops=2, recursive_prefix_layer_count=5,
                recursive_middle_layer_count=10, recursive_suffix_layer_count=5,
                recursive_loops_scope="middle_only", vocab_size=49152, eos_token_id=0,
                logical_to_physical=list(contract.TEXT_MODEL_SCHEDULE),
                recursive_source_layer_indices_0based=list(contract.TEXT_MODEL_SOURCE_MAPPING))


def fake_model():
    cfg = text_config()
    model = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: cfg))
    model.model = SimpleNamespace(layers=[object() for _ in range(20)],
                                 logical_to_physical=contract.TEXT_MODEL_SCHEDULE)
    model.named_parameters = lambda: iter([("model.layers.5.weight", object())])
    model.named_buffers = lambda: iter([])
    return model


def parameter_audit():
    return dict(logical_layer_count=30, physical_layer_count=20, recursive_layer_count=20,
                recursive_loops=2, middle_recurrent_count=10, model_physical_module_count=20,
                schedule=list(contract.TEXT_MODEL_SCHEDULE))


class NonMeshCEDParityTests(unittest.TestCase):
    def test_three_way_audio_data_loss_and_update_loop(self):
        parent = audit_reference.audit_parent()
        self.assertEqual(len(parent["byte_identical_files"]), 11)
        self.assertEqual(len(parent["nonmesh_text_contract_function_asts"]), 5)
        self.assertEqual(len(audit_reference.audit_ced_baseline()["identical_files_normalized_lf"]), 16)
        self.assertEqual(len(audit_reference.audit_mesh_ced()["identical_files_normalized_lf"]), 17)

    def test_text_checkpoint_config_and_rejections(self):
        cfg = text_config()
        self.assertEqual(contract.validate_recursive_config_values(cfg)["memory_slots"], 0)
        bad_values = {"mesh_memory_slots": 5, "mesh_router_count": 6, "mesh_enabled": True,
                      "mesh_architecture_contract": "mesh", "hidden_size": 384,
                      "recursive_loops": 4, "recursive_layer_count": 19,
                      "recursive_loops_scope": "all_layers",
                      "recursive_source_layer_indices_0based": list(range(20)),
                      "logical_to_physical": list(range(30)),
                      "architectures": ["LlamaForCausalLM"]}
        for key, value in bad_values.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                contract.validate_recursive_config_values(dict(cfg, **{key: value}))

    def test_checkpoint_contract_requires_model_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            contract.write_json(path / "config.json", text_config())
            with self.assertRaisesRegex(ValueError, "weight files"):
                contract.recursive_checkpoint_contract(path)
            (path / "model.safetensors").write_bytes(b"synthetic fixture, not real weights")
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                contract.recursive_checkpoint_contract(path)
            (path / "tokenizer.json").write_text("{}")
            report = contract.recursive_checkpoint_contract(path)
            self.assertEqual((report["config"]["physical_layers"], report["config"]["memory_slots"],
                              report["config"]["routers"]), (20, 0, 0))
            self.assertEqual(report["config"]["source_layer_indices_0based"], list(contract.TEXT_MODEL_SOURCE_MAPPING))

    def test_loaded_model_checks_real_layers_schedule_parameters_and_buffers(self):
        model, audit = fake_model(), parameter_audit()
        recursive_text.validate_loaded_model(model, audit)
        model.model.layers = [object()] * 20
        with self.assertRaisesRegex(ValueError, "independent physical"):
            recursive_text.validate_loaded_model(model, audit)
        for kind in ("parameters", "buffers"):
            model = fake_model()
            setattr(model, "named_" + kind, lambda: iter([("model.memory_slots.state", object())]))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "MeSH parameter/buffer"):
                recursive_text.validate_loaded_model(model, audit)
        model = fake_model()
        model.model.logical_to_physical = tuple(range(30))
        with self.assertRaisesRegex(ValueError, "execution schedule"):
            recursive_text.validate_loaded_model(model, audit)
        with self.assertRaisesRegex(ValueError, "parameter audit"):
            recursive_text.validate_loaded_model(fake_model(), dict(audit, physical_layer_count=19))

    def test_loader_uses_explicit_class_offline_and_rejects_every_loading_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            contract.write_json(path / "config.json", text_config())
            cls, model = Mock(), fake_model()
            for failure in (None, "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
                cls.from_pretrained.return_value = (model, {} if failure is None else {failure: ["fixture mismatch"]})
                with self.subTest(failure=failure), patch.object(
                        recursive_text, "_import_recursive_model", return_value=(cls, lambda _: parameter_audit())):
                    if failure:
                        with self.assertRaisesRegex(ValueError, "weights do not match"):
                            recursive_text.load_recursive_text_model(path)
                    else:
                        self.assertIs(recursive_text.load_recursive_text_model(path), model)
                    self.assertEqual(cls.from_pretrained.call_args.kwargs,
                                     dict(local_files_only=True, output_loading_info=True))

    def test_mapper_initialization_order_and_nonmesh_decoder_identity(self):
        source = (ROUTE / "models/mellow.py").read_text(encoding="utf-8")
        self.assertLess(source.index("self.audio_encoder = AudioEncoder("),
                        source.index("self.caption_decoder = get_decoder("))
        source = (ROUTE / "train.py").read_text(encoding="utf-8")
        self.assertLess(source.index("torch.manual_seed(local_seed)"), source.index('model = Mellow(audioenc_name="CEDSmall"'))
        self.assertLess(source.index("ddp = DistributedDataParallel"), source.index("mapper_before = tensor_digest"))
        self.assertIn('"mapper_initial_sha256"', source)
        self.assertIn('"is_recursive_5_10_5"', source)
        source = (ROUTE / "models/decoder.py").read_text(encoding="utf-8")
        self.assertIn("self.is_recursive_5_10_5 = True", source)
        self.assertNotIn("self.is_recursive_mesh", source)
        self.assertNotIn("AutoModelForCausalLM", source)

    def test_submission_defaults_and_staging_namespace_are_isolated(self):
        launcher = (ROOT / "code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_5090.sh").read_text(encoding="utf-8")
        self.assertIn("TEXT_MODEL_DIR=" + contract.DEFAULT_TEXT_MODEL_DIR, launcher)
        self.assertIn("mizar_s1_ced_small_gbs256_recursive_5_10_5_5090/index_cache", launcher)
        self.assertNotIn("/stage4_5_10x2_5_mesh/", launcher)
        runner = (ROUTE / "run.sh").read_text(encoding="utf-8")
        self.assertIn('/dev/shm/mizar_s1_gbs256_recursive_5_10_5_ced_small_${STAGE_TOKEN}', runner)
        self.assertIn('conda activate mellow_c8204d8', runner)
        self.assertNotIn('smoke', launcher)

    def test_shared_source_identity_and_all_four_routes_are_distinct(self):
        contracts = []
        for name in (ROUTE.name, ORIGINAL.name, "mizar_stage1_fixed20_gbs256_3epochs_ced_small",
                     "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small"):
            scope = {"__file__": str(FAMILY / name / "contract.py")}
            exec(compile((FAMILY / name / "contract.py").read_text(encoding="utf-8"), name, "exec"), scope)
            contracts.append(scope["ROUTE"])
        self.assertEqual(len(set(contracts)), 4)
        self.assertEqual(contract.recursive_model_source_sha256(),
                         __import__("hashlib").sha256((ROOT / "code/RSmol/recursive_model_5_10_5.py")
                                                      .read_bytes().replace(b"\r\n", b"\n")).hexdigest())

    def test_tokenizer_registering_existing_pad_does_not_resize_vocabulary(self):
        class Tokenizer:
            pad_token, pad_token_id, eos_token_id = "<|endoftext|>", 0, 0
            def __len__(self):
                return 49152
            def encode(self, value, **kwargs):
                return [17] if value == "!" else [0]
            def add_special_tokens(self, values):
                self.pad_token, self.pad_token_id = values["pad_token"], 17
                return 1
        report = contract.validate_checkpoint_tokenizer(Tokenizer(), 49152)
        self.assertEqual((report["added_tokens"], report["vocabulary_growth"], report["mizar_pad_id"]), (1, 0, 17))

    def test_fresh_reference_report_matches_current_code(self):
        report = json.loads((ROUTE / "reference_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["route_code_sha256"], contract.code_digest())
        self.assertIn("mesh_ced_audit", report)
        self.assertEqual(report["official_mapper_class_asts_equal"], ["CEDFrequencyMerger", "CEDTemporalMapper"])


if __name__ == "__main__":
    unittest.main()
