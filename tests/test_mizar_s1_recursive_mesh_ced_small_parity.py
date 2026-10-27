"""Stdlib parity checks for the isolated CED-Small / MeSH training route."""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine"
ROUTE = FAMILY / "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small"
MESH = FAMILY / "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh"
sys.path.insert(0, str(ROUTE))
import audit_reference
import contract


class MeSHCEDParityTests(unittest.TestCase):
    def test_original_mesh_semantics_and_ced_baseline_match(self):
        self.assertEqual(len(audit_reference.audit_parent()["byte_identical_files"]), 11)
        result = audit_reference.audit_ced_baseline()
        self.assertEqual(len(result["identical_files_normalized_lf"]), 16)
        self.assertEqual(result["optimizer_epoch_loop"], "identical including gradient diagnostic")

    def test_mesh_loader_and_checkpoint_contract_preserved(self):
        self.assertEqual((ROUTE / "models/recursive_text.py").read_bytes(),
                         (MESH / "models/recursive_text.py").read_bytes())
        actual = ast.parse((ROUTE / "contract.py").read_text(encoding="utf-8"))
        original = ast.parse((MESH / "contract.py").read_text(encoding="utf-8"))
        for name in ("recursive_checkpoint_contract", "recursive_model_source_sha256",
                     "validate_checkpoint_tokenizer", "validate_tokenizer_contract"):
            self.assertEqual(ast.dump(audit_reference.find_function(actual, name)),
                             ast.dump(audit_reference.find_function(original, name)), name)
        source = (ROUTE / "models/decoder.py").read_text(encoding="utf-8")
        self.assertIn("self.lm = load_recursive_text_model(text_decoder)", source)
        self.assertIn("self.is_recursive_mesh = True", source)
        self.assertNotIn("AutoModelForCausalLM", source)
        self.assertEqual(contract.TEXT_MODEL_ARCHITECTURE, "logical_30_physical_20_5_10x2_5_mesh")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            cfg = dict(model_type="llama", hidden_size=576, num_hidden_layers=30,
                       recursive_layer_count=20, recursive_loops=2, mesh_memory_slots=5,
                       mesh_router_count=6, logical_to_physical=list(contract.TEXT_MODEL_SCHEDULE),
                       mesh_architecture_contract=contract.TEXT_MODEL_CONTRACT,
                       model_architecture_contract=contract.TEXT_MODEL_ARCHITECTURE,
                       vocab_size=49152, eos_token_id=0)
            (checkpoint / "model.safetensors").write_bytes(b"isolated synthetic test weights")
            (checkpoint / "tokenizer.json").write_text("{}")
            contract.write_json(checkpoint / "config.json", cfg)
            report = contract.recursive_checkpoint_contract(checkpoint)
            self.assertEqual((report["config"]["physical_layers"], report["config"]["memory_slots"],
                              report["config"]["routers"]), (20, 5, 6))
            for name, wrong in (("mesh_memory_slots", 7), ("mesh_router_count", 8),
                                ("recursive_loops", 4), ("hidden_size", 768)):
                with self.subTest(name=name):
                    contract.write_json(checkpoint / "config.json", dict(cfg, **{name: wrong}))
                    with self.assertRaisesRegex(ValueError, "invalid recursive text checkpoint"):
                        contract.recursive_checkpoint_contract(checkpoint)

    def test_model_build_order_preserves_mapper_initialization(self):
        baseline = FAMILY / "mizar_stage1_fixed20_gbs256_3epochs_ced_small"
        self.assertEqual((ROUTE / "models/mellow.py").read_bytes(),
                         (baseline / "models/mellow.py").read_bytes())
        source = (ROUTE / "models/mellow.py").read_text(encoding="utf-8")
        self.assertLess(source.index("self.audio_encoder = AudioEncoder("),
                        source.index("self.caption_decoder = get_decoder("))
        source = (ROUTE / "train.py").read_text(encoding="utf-8")
        self.assertLess(source.index("torch.manual_seed(local_seed)"),
                        source.index('model = Mellow(audioenc_name="CEDSmall"'))
        self.assertIn("mapper_initial_sha256", source)
        self.assertIn("ced_state_unchanged", source)

    def test_official_mapper_and_report_identity(self):
        report = json.loads((ROUTE / "reference_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["route_code_sha256"], contract.code_digest())
        self.assertEqual(report["official_mapper_class_asts_equal"],
                         ["CEDFrequencyMerger", "CEDTemporalMapper"])


if __name__ == "__main__":
    unittest.main()
