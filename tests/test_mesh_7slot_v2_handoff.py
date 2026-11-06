"""Exercise the real text writer -> audio preflight boundary without ML assets."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code/RSmol"
ROUTE = CODE / "mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_7slot_v2_ced_small"
sys.path.insert(0, str(ROUTE))
import contract
from text_checkpoint import final_text_checkpoint
from test_text_5_10x2_5_mesh_7slot_v2_training import trainer_namespace
from test_text_5_10x2_5_mesh_7slot_v2_contract import production_functions, model_constants


class HandoffTests(unittest.TestCase):
    def checkpoint(self, output):
        """Execute the production text checkpoint writer, with model IO stand-ins."""
        constants = model_constants()
        trainer = trainer_namespace()
        config = trainer.Stage4Config(gate="FORMAL", output_dir=output, max_optimizer_steps=9244)
        def save_model(path, **kwargs):
            config_json = dict(model_type="llama", hidden_size=576, num_hidden_layers=30,
                               recursive_layer_count=20, recursive_loops=2, mesh_memory_slots=7,
                               mesh_router_count=6, mesh_architecture_contract=constants["MODEL_ARCHITECTURE_CONTRACT"],
                               logical_to_physical=list(constants["LOGICAL_TO_PHYSICAL"]), eos_token_id=0)
            contract.write_json(path / "config.json", config_json)
            (path / "model.safetensors").write_bytes(b"synthetic-model-writer")
        tokenizer = SimpleNamespace(save_pretrained=lambda path: (path / "tokenizer.json").write_text("{}"))
        from dataclasses import asdict
        import shutil
        scope = dict(constants, TRAINING_CONTRACT=trainer.TRAINING_CONTRACT,
                     PRECISION_CONTRACT=trainer.PRECISION_CONTRACT, PRECISION_POLICY=trainer.PRECISION_POLICY,
                     asdict=asdict, json=json, shutil=shutil,
                     torch=SimpleNamespace(save=lambda value, path: path.write_bytes(b"synthetic-optimizer"),
                                           get_rng_state=lambda: "cpu-state"))
        writer = production_functions(CODE / "scripts/train_stage4_5_10x2_5_mesh_7slot_v2_ddp.py", {"_checkpoint"}, scope)
        return writer._checkpoint(SimpleNamespace(save_pretrained=save_model), tokenizer,
                                  SimpleNamespace(state_dict=lambda: {}), 9244, config, 0, {})

    def test_actual_text_writer_is_accepted_by_audio_config_and_cpu_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(Path(directory))
            result = contract.recursive_checkpoint_contract(checkpoint)
            self.assertEqual(result["config"]["memory_slots"], 7)
            self.assertEqual(result["text_pretraining"]["optimizer_step"], 9244)
            self.assertFalse(result["text_pretraining"]["text_optimizer_scheduler_rng_cursor_restored"])
            completed = subprocess.run([sys.executable, "-B", str(ROUTE / "check_text_checkpoint.py"),
                                        "--text-model-dir", str(checkpoint)], capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["status"], "PASS_METADATA_ONLY")
            # Adding a BF16 training source does not change the FP32 audio policy.
            prepared = dict(mode="formal", rows=1477492, source_rows=contract.SOURCE_COUNTS,
                            rows_canonical_sha256="rows", mapping_sha256="mapping")
            contract.write_json(Path(directory) / "prepared.json", prepared)
            import write_config
            args = ["write_config.py", "--prepared", str(Path(directory) / "prepared.json"),
                    "--output", str(Path(directory) / "runtime.json"), "--run-dir", str(Path(directory)),
                    "--mode", "formal", "--text-model-dir", str(checkpoint), "--ced-model-dir", str(Path(directory))]
            with patch.object(sys, "argv", args), patch.object(write_config, "StagingGate", return_value=SimpleNamespace(info=None)), \
                 patch.object(write_config, "asset_identity", return_value={"synthetic_ced": True}), \
                 patch.object(write_config.importlib.metadata, "version", return_value="fixture"):
                write_config.main()
            runtime = json.loads((Path(directory) / "runtime.json").read_text())
            self.assertEqual(runtime["contract"]["recursive_text_checkpoint"], result)
            self.assertEqual((runtime["contract"]["forward_dtype"], runtime["contract"]["autocast"]), ("float32", False))
            self.assertEqual(runtime["contract_sha256"], contract.digest(runtime["contract"]))

    def test_final_provenance_rejects_wrong_route_step_precision_or_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(Path(directory))
            metadata_path = checkpoint / "mesh_checkpoint_metadata.json"
            original = json.loads(metadata_path.read_text())
            for key, value in (("optimizer_step", 3081), ("memory_slots", 4), ("memory_slots", 6),
                               ("architecture_contract", "logical_30_physical_20_5_10x2_5_mesh_7slot_3router"),
                               ("training_contract", "old-seven-slot"), ("precision", "fp32"),
                               ("epochs", 2), ("precision_policy", {}), ("router_parameters_in_optimizer", False)):
                with self.subTest(key=key, value=value):
                    contract.write_json(metadata_path, dict(original, **{key: value}))
                    with self.assertRaises(ValueError):
                        contract.recursive_checkpoint_contract(checkpoint)
            contract.write_json(metadata_path, original)
            for name in ("checkpoint_complete.json", "training_state.pt", "model.safetensors", "tokenizer.json"):
                path = checkpoint / name
                saved = path.read_bytes()
                path.unlink()
                with self.subTest(name=name), self.assertRaises(ValueError):
                    final_text_checkpoint(checkpoint)
                path.write_bytes(saved)

    def test_final_manifest_rejects_empty_files_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(Path(directory))
            (checkpoint / "training_state.pt").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty"):
                final_text_checkpoint(checkpoint)
            (checkpoint / "training_state.pt").write_bytes(b"state")
            for name in ("checkpoint_complete.json", "checkpoint_manifest.json"):
                value = json.loads((checkpoint / name).read_text())
                value["files"].append("../outside")
                contract.write_json(checkpoint / name, value)
            with self.assertRaises(ValueError):
                final_text_checkpoint(checkpoint)

    def test_complete_text_model_converter_and_trainer_preserve_four_slot_parent(self):
        pairs = [(CODE / "recursive_model_5_10x2_5_mesh_7slot_v2.py", CODE / "recursive_model_5_10x2_5_mesh_4slot.py"),
                 (CODE / "scripts/convert_stepwise_5_10x2_5_mesh_7slot_v2.py", CODE / "scripts/convert_stepwise_5_10x2_5_mesh_4slot.py"),
                 (CODE / "scripts/train_stage4_5_10x2_5_mesh_7slot_v2_ddp.py", CODE / "scripts/train_stage4_5_10x2_5_mesh_4slot_ddp.py")]
        for current, parent in pairs:
            normalized = current.read_text(encoding="utf-8")
            for new, old in (("mesh-7slot-v2", "mesh-4slot"), ("7slot_v2", "4slot"),
                             ("7SLOT_V2", "4SLOT"), ("7SlotV2", "4Slot"),
                             ("seven-slot-v2", "four-slot"), ("seven memory slots", "four memory slots"),
                             ("seven slots", "four slots"), ("MEMORY_SLOT_COUNT = 7", "MEMORY_SLOT_COUNT = 4"),
                             ("sqrt_2_over_7d", "sqrt_2_over_4d")):
                normalized = normalized.replace(new, old)
            self.assertEqual(ast.dump(ast.parse(normalized)), ast.dump(ast.parse(parent.read_text(encoding="utf-8"))), current.name)


if __name__ == "__main__":
    unittest.main()
