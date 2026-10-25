"""CPU-only checks for the non-MeSH 5+10x2+5 Mizar ablation route."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5"
OUTER = ROOT / "code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_5090.sh"
spec = importlib.util.spec_from_file_location("recursive_5_10_5_contract_test", ROUTE / "contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


class Recursive5105Tests(unittest.TestCase):
    def test_geometry_and_identity(self):
        self.assertEqual(contract.ROUTE, "htsat_mizar_s1_fixed20_gbs256_recursive_5_10_5_v1")
        self.assertEqual(contract.geometry(1477492)["total_steps"], 17313)
        self.assertEqual(contract.geometry(1477492)["warmup_steps"], 866)
        self.assertEqual(contract.TEXT_MODEL_SCHEDULE, tuple(list(range(15)) + list(range(5, 15)) + list(range(15, 20))))

    def test_non_mesh_config_contract(self):
        config = {
            "model_type": "llama", "architectures": ["RecursiveLlamaForCausalLM"],
            "hidden_size": 576, "num_hidden_layers": 30, "recursive_layer_count": 20,
            "recursive_loops": 2, "recursive_prefix_layer_count": 5,
            "recursive_middle_layer_count": 10, "recursive_suffix_layer_count": 5,
            "recursive_loops_scope": "middle_only", "logical_to_physical": list(contract.TEXT_MODEL_SCHEDULE),
            "recursive_source_layer_indices_0based": list(contract.TEXT_MODEL_SOURCE_MAPPING),
            "vocab_size": 49152, "eos_token_id": 0,
        }
        result = contract.validate_recursive_config_values(config)
        self.assertEqual(result["memory_slots"], 0)
        for bad in ({**config, "mesh_memory_slots": 5},
                    {**config, "mesh_enabled": True},
                    {**config, "recursive_layer_count": 19}):
            with self.assertRaises(ValueError):
                contract.validate_recursive_config_values(bad)

    def test_checkpoint_contract_accepts_non_mesh_shape(self):
        config = {
            "model_type": "llama", "architectures": ["RecursiveLlamaForCausalLM"],
            "hidden_size": 576, "num_hidden_layers": 30, "recursive_layer_count": 20,
            "recursive_loops": 2, "recursive_prefix_layer_count": 5,
            "recursive_middle_layer_count": 10, "recursive_suffix_layer_count": 5,
            "recursive_loops_scope": "middle_only", "logical_to_physical": list(contract.TEXT_MODEL_SCHEDULE),
            "recursive_source_layer_indices_0based": list(contract.TEXT_MODEL_SOURCE_MAPPING),
            "vocab_size": 49152, "eos_token_id": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"fixture")
            (root / "tokenizer.json").write_bytes(b"fixture")
            report = contract.recursive_checkpoint_contract(root)
            self.assertEqual(report["config"]["memory_slots"], 0)
            self.assertEqual(report["config"]["routers"], 0)

    def test_submission_isolated_and_checkpoint_is_correct(self):
        source = OUTER.read_text(encoding="utf-8")
        self.assertIn("stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244", source)
        self.assertIn("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5/run.sh", source)
        self.assertIn("-c 32 -m 256G -g 8 -n 1", source)
        self.assertIn("recursive 5-10-5 non-MeSH", source)
        self.assertNotIn("stage4_5_10x2_5_mesh", source)

    def test_alignment_audit_passes(self):
        audit = ROUTE / "audit_alignment.py"
        spec = importlib.util.spec_from_file_location("recursive_5_10_5_alignment", audit)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.audit_alignment()
        self.assertEqual(result["status"], "PASS", result["problems"])


if __name__ == "__main__":
    unittest.main()
