"""CPU-only contract checks for the isolated recursive MeSH Mizar route."""

import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs"
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh"
OUTER = ROOT / "code/RSmol/run_mizar_s1_gbs256_recursive_mesh_5090.sh"

spec = importlib.util.spec_from_file_location("recursive_mesh_contract", ROUTE / "contract.py")
contract = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(contract)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExistingBangTokenizer:
    """Match the remote checkpoint's existing token and registration result."""

    eos_token_id = 0
    pad_token = "<|endoftext|>"
    pad_token_id = 0
    bang_id = 17

    def __len__(self):
        return 49152

    def add_special_tokens(self, mapping):
        self.pad_token = mapping["pad_token"]
        self.pad_token_id = self.bang_id
        return 1

    def encode(self, text, add_special_tokens=False):
        return [self.bang_id] if text == "!" else [0]


class RecursiveMizarRouteTests(unittest.TestCase):
    def test_route_identity_and_geometry(self):
        self.assertEqual(contract.ROUTE, "htsat_mizar_s1_fixed20_gbs256_recursive_mesh_v1")
        self.assertEqual(contract.TEXT_MODEL_CONTRACT, "logical_30_physical_20_5_10x2_5")
        self.assertEqual(tuple(contract.TEXT_MODEL_SCHEDULE), tuple(
            list(range(15)) + list(range(5, 15)) + list(range(15, 20))))
        geometry = contract.geometry(1477492)
        self.assertEqual(
            (geometry["steps_per_epoch"], geometry["total_steps"],
             geometry["warmup_steps"], geometry["dropped_rows_per_epoch"]),
            (5771, 17313, 866, 116),
        )

    def test_optimizer_and_sequence_contract_are_baseline_values(self):
        self.assertEqual(contract.PREFIX_TOKENS, 515)
        self.assertEqual(contract.PREFIX_TOKENS + contract.ANSWER_TOKENS, 815)
        self.assertEqual(contract.semantic_contract(
            {"rows_canonical_sha256": "rows", "mapping_sha256": "mapping",
             "source_rows": contract.SOURCE_COUNTS, "rows": 1477492},
            "/remote/mesh", "/remote/htsat")["adamw"], {
                "decoder_lr": 3e-4, "map_lr": 6e-4, "betas": [0.9, 0.999],
                "eps": 1e-8, "weight_decay": 0.01, "exclude_bias_ndim_le1": True,
            })
        self.assertEqual(contract.semantic_contract(
            {"rows_canonical_sha256": "rows", "mapping_sha256": "mapping",
             "source_rows": contract.SOURCE_COUNTS, "rows": 1477492},
            "/remote/mesh", "/remote/htsat")["prefix"], [129, 129, 1, 256])

    def test_only_model_loader_decoder_contract_changes(self):
        for relative in (
            "data/audiotext_dataset.py", "data/fixed20.py", "data/audio_paths.py",
            "data/sampler.py", "data/staging.py", "stage_data.py", "prepare_index.py",
            "optim.py", "models/audio.py", "models/htsat.py", "models/mellow.py",
        ):
            self.assertEqual(sha256(BASELINE / relative), sha256(ROUTE / relative), relative)
        decoder = (ROUTE / "models/decoder.py").read_text(encoding="utf-8")
        loader = (ROUTE / "models/recursive_text.py").read_text(encoding="utf-8")
        self.assertIn("load_recursive_text_model", decoder)
        self.assertIn("inputs_embeds=embedding_cat", decoder)
        self.assertIn('"separators": 1', decoder)
        self.assertIn("prefix.shape[1] != 515", decoder)
        self.assertIn("local_files_only=True", loader)
        self.assertIn("register_auto_class", loader)
        self.assertIn("TEXT_MODEL_SCHEDULE", loader)
        self.assertNotIn("AutoModelForCausalLM.from_pretrained", decoder)

    def test_outer_launcher_isolated_and_uses_remote_mesh_checkpoint(self):
        source = OUTER.read_text(encoding="utf-8")
        self.assertIn("stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244", source)
        self.assertIn("mizar_s1_htsat_gbs256_recursive_mesh_5090", source)
        self.assertIn("mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh/run.sh", source)
        self.assertIn("-c 32 -m 256G -g 8 -n 1", source)
        self.assertNotIn("SmolLM2", source)

    def test_recursive_checkpoint_contract_rejects_missing_assets(self):
        with self.assertRaises(FileNotFoundError):
            contract.recursive_checkpoint_contract(ROOT / "does-not-exist")

    def test_recursive_checkpoint_contract_accepts_expected_remote_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text(json.dumps({
                "model_type": "llama",
                "hidden_size": 576,
                "num_hidden_layers": 30,
                "recursive_layer_count": 20,
                "recursive_loops": 2,
                "mesh_memory_slots": 5,
                "mesh_router_count": 6,
                "logical_to_physical": list(contract.TEXT_MODEL_SCHEDULE),
                "mesh_architecture_contract": contract.TEXT_MODEL_CONTRACT,
                "model_architecture_contract": contract.TEXT_MODEL_ARCHITECTURE,
                "vocab_size": 49152,
                "eos_token_id": 0,
            }), encoding="utf-8")
            for name in ("model-00001-of-00001.safetensors", "tokenizer.json", "tokenizer_config.json"):
                (checkpoint / name).write_bytes(b"fixture")
            report = contract.recursive_checkpoint_contract(checkpoint)
            self.assertEqual(report["config"]["hidden_size"], 576)
            self.assertEqual(report["config"]["routers"], 6)
            self.assertIn("tokenizer.json", report["tokenizer_files"])

    def test_source_eos_pad_is_overridden_by_mizar_bang_pad(self):
        class SourceTokenizer:
            eos_token_id = 0
            pad_token = "<|endoftext|>"
            pad_token_id = 0

            def __len__(self):
                return 49152

            def add_special_tokens(self, mapping):
                self.pad_token = mapping["pad_token"]
                self.pad_token_id = 17
                return 0

            def encode(self, text, add_special_tokens=False):
                return [17] if text == "!" else [0]

        audit = contract.validate_checkpoint_tokenizer(SourceTokenizer(), 49152)
        self.assertEqual(audit["source_pad_token"], "<|endoftext|>")
        self.assertEqual(audit["source_pad_id"], 0)
        self.assertEqual(audit["mizar_pad_ids"], [17])
        self.assertEqual(audit["mizar_pad_id"], 17)
        self.assertTrue(audit["existing_pad_contract"])

    def test_source_tokenizer_rejects_bang_aliasing_eos(self):
        class InvalidTokenizer:
            eos_token_id = 0
            pad_token = "<|endoftext|>"
            pad_token_id = 0

            def __len__(self):
                return 49152

            def add_special_tokens(self, mapping):
                self.pad_token = mapping["pad_token"]
                self.pad_token_id = 0
                return 0

            def encode(self, text, add_special_tokens=False):
                return [0]

        with self.assertRaisesRegex(ValueError, "aliases EOS/SEP"):
            contract.validate_checkpoint_tokenizer(InvalidTokenizer(), 49152)

    def test_existing_bang_special_registration_can_return_one(self):
        audit = contract.validate_checkpoint_tokenizer(ExistingBangTokenizer(), 49152)
        self.assertEqual(audit["added_tokens"], 1)
        self.assertEqual(audit["vocabulary_growth"], 0)
        self.assertEqual(audit["source_vocab_rows"], audit["tokenizer_vocab_rows"])
        self.assertEqual(audit["source_mizar_pad_ids"], audit["mizar_pad_ids"])
        self.assertEqual(audit["mizar_pad_id"], 17)
        self.assertEqual(audit["eos_id"], 0)
        self.assertTrue(audit["existing_pad_contract"])

    def test_source_tokenizer_still_rejects_actual_vocabulary_growth(self):
        class GrowingTokenizer:
            eos_token_id = 0
            pad_token = "<|endoftext|>"
            pad_token_id = 0
            rows = 49152

            def __len__(self):
                return self.rows

            def add_special_tokens(self, mapping):
                self.pad_token = mapping["pad_token"]
                self.pad_token_id = self.rows
                self.rows += 1
                return 1

            def encode(self, text, add_special_tokens=False):
                return [self.pad_token_id] if text == "!" else [0]

        with self.assertRaisesRegex(ValueError, "changed vocabulary rows from 49152 to 49153"):
            contract.validate_checkpoint_tokenizer(GrowingTokenizer(), 49152)

    def test_source_tokenizer_rejects_changed_bang_id_without_vocabulary_growth(self):
        class ChangingTokenizer(ExistingBangTokenizer):
            def add_special_tokens(self, mapping):
                self.bang_id = 18
                return super().add_special_tokens(mapping)

        with self.assertRaisesRegex(ValueError, "changed '!' token IDs from \\[17\\] to \\[18\\]"):
            contract.validate_checkpoint_tokenizer(ChangingTokenizer(), 49152)

    def test_startup_tokenizer_audit_uses_resolved_model_vocab_size(self):
        # Execute the real startup helper/call without importing the GPU stack.
        tree = ast.parse((ROUTE / "train.py").read_text(encoding="utf-8"))
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "audit_checkpoint_tokenizer")
        train = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "train")
        assignment = next(node for node in train.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "source_tokenizer_audit"
                                  for target in node.targets))
        startup = compile(ast.Module(body=[helper, assignment], type_ignores=[]), str(ROUTE / "train.py"), "exec")
        for vocab_size in (49152, 49151):
            with self.subTest(model_vocab_size=vocab_size):
                loader = Mock(return_value=ExistingBangTokenizer())
                scope = {
                    "AutoTokenizer": SimpleNamespace(from_pretrained=loader),
                    "validate_checkpoint_tokenizer": contract.validate_checkpoint_tokenizer,
                    "contract": {"text_model_dir": "/remote/mesh",
                                 "recursive_text_checkpoint": {"config": {"vocab_size": vocab_size}}},
                }
                if vocab_size == 49152:
                    exec(startup, scope)
                    self.assertEqual(scope["source_tokenizer_audit"]["model_vocab_size"], vocab_size)
                else:
                    with self.assertRaisesRegex(ValueError, "model vocab_size=49151"):
                        exec(startup, scope)
                loader.assert_called_once_with("/remote/mesh", local_files_only=True)

    def test_startup_logs_serialize_both_tokenizer_audit_levels(self):
        tokenizer = ExistingBangTokenizer()
        source_audit = contract.validate_checkpoint_tokenizer(tokenizer, 49152)
        runtime_audit = contract.validate_tokenizer_contract(tokenizer, 17, 49152)
        runtime_audit.update(ignore_index=17, checkpoint_source=source_audit)
        tree = ast.parse((ROUTE / "train.py").read_text(encoding="utf-8"))
        train = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "train")
        events = {}
        for node in ast.walk(train):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "print" and node.args):
                continue
            expression = node.args[0]
            if not (isinstance(expression, ast.Call) and expression.args
                    and isinstance(expression.args[0], ast.Dict)):
                continue
            event_values = [value.value for key, value in zip(expression.args[0].keys, expression.args[0].values)
                            if isinstance(key, ast.Constant) and key.value == "event"
                            and isinstance(value, ast.Constant)]
            if not event_values or event_values[0] not in {"CHECKPOINT_TOKENIZER_PASS", "PREFLIGHT_PASS"}:
                continue
            payload = eval(compile(ast.Expression(expression), str(ROUTE / "train.py"), "eval"), {
                "json": json, "geometry": {}, "tokenizer_audit": runtime_audit,
                "source_tokenizer_audit": source_audit, "audit": {"initial_optimizer_lrs": []},
            })
            events[event_values[0]] = json.loads(payload)
        self.assertEqual(set(events), {"CHECKPOINT_TOKENIZER_PASS", "PREFLIGHT_PASS"})
        self.assertEqual(events["CHECKPOINT_TOKENIZER_PASS"]["tokenizer"], source_audit)
        self.assertEqual(events["PREFLIGHT_PASS"]["checkpoint_tokenizer"], source_audit)
        self.assertEqual(events["PREFLIGHT_PASS"]["tokenizer"]["embedding_rows"], 49152)


if __name__ == "__main__":
    unittest.main()
