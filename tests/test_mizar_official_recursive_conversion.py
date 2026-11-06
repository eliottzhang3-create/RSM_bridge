"""Real stdlib file/path/contract checks; synthetic fixtures are not weight audits."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/RSmol"))
from mizar_official_recursive_5_20x2_5 import audit, contract
from mizar_official_recursive_5_20x2_5.assets import CED_EXPECTED_CONFIG, validate_ced_config
from mizar_official_recursive_5_20x2_5.conversion import atomic_publish

REF = ROOT / ("code/RSmol/mellow_official_training_c8204d8_adamw_cosine/"
              "mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8")


def source_config():
    return dict(model_type="llama", num_hidden_layers=30, hidden_size=576, intermediate_size=1536,
                num_attention_heads=9, num_key_value_heads=3, vocab_size=49152,
                tie_word_embeddings=True, rms_norm_eps=1e-5, custom_preserved={"sentinel": 42})


def definition(path, name):
    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    return ast.dump(node, include_attributes=False)


class ContractTests(unittest.TestCase):
    def test_full_thirty_layer_schedule_and_serial_boundaries(self):
        schedule = contract.SCHEDULE
        self.assertEqual(len(schedule), 50)
        self.assertEqual(schedule[:5], tuple(range(5)))
        self.assertEqual(schedule[5:25], tuple(range(5, 25)))
        self.assertEqual(schedule[25:45], schedule[5:25])
        self.assertEqual(schedule[45:], tuple(range(25, 30)))
        self.assertEqual(contract.INJECTION_INDICES, (5, 25))
        self.assertEqual(contract.SERIAL_TRACE, ((5, "input"), (5, "layer8"), (25, "input"), (25, "layer4")))

    def test_rank8_counts_from_projection_geometries(self):
        hidden, kv, mlp, rank = 576, 192, 1536, 8
        # q/o, k/v, gate/up/down, two loops, twenty physical middle layers.
        lora = 20 * 2 * rank * (2 * (hidden + hidden) + 2 * (hidden + kv) + 3 * (hidden + mlp))
        merger = 2 * 1536 + 1536 * hidden + hidden + hidden * hidden + hidden + 2 * hidden
        audio = hidden * hidden + hidden + 3 * (hidden + 1) + 2 * merger
        self.assertEqual(lora, contract.LORA_PARAMETERS)
        self.assertEqual(audio, contract.AUDIO_PARAMETERS)
        self.assertEqual(159299840 + lora + audio, 165334019)
        self.assertEqual(20 * 7 * 2 * 2 + 24, contract.NEW_TENSORS)

    def test_learning_rates_and_layer_norm_boundaries(self):
        for i in (0, 4, 5, 24, 25, 29):
            lr = 6e-4 if 5 <= i <= 24 else 3e-4
            for suffix in ("self_attn.q_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight"):
                self.assertEqual(contract.parameter_role(f"caption_decoder.lm.model.layers.{i}.{suffix}")[1], lr)
        for name in ("audio_encoder.base.mapper.pos_in", "caption_decoder.lm.model.embed_tokens.weight",
                     "caption_decoder.lm.lm_head.weight", "caption_decoder.lm.model.norm.weight"):
            self.assertEqual(contract.parameter_role(name)[1], 3e-4)
        for name in ("audio_encoder.base.layer_mlps.4.net.1.weight", "caption_decoder.lm.model.audio_gate.bias",
                     "caption_decoder.lm.model.layers.24.self_attn.q_proj.lora_B.1"):
            self.assertEqual(contract.parameter_role(name)[1], 6e-4)
        self.assertIsNone(contract.parameter_role("audio_encoder.base.ced.blocks.0.weight")[1])

    def test_invalid_parameter_roles_fail_closed(self):
        for name in ("router.weight", "caption_decoder.lm.model.layers.30.input_layernorm.weight",
                     "caption_decoder.lm.model.layers.4.self_attn.q_proj.lora_A.0",
                     "caption_decoder.lm.model.layers.5.self_attn.q_proj.lora_B.2"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                contract.parameter_role(name)

    def test_all_source_config_fields_preserved_without_mutation(self):
        original = source_config()
        before = copy.deepcopy(original)
        converted = contract.build_text_config(original)
        self.assertEqual(original, before)
        self.assertEqual(converted["custom_preserved"], original["custom_preserved"])
        self.assertEqual(converted["recursive_source_layer_indices_0based"], list(range(30)))
        contract.validate_text_config(converted)

    def test_wrong_geometry_cache_or_schedule_rejected(self):
        for name, value in (("hidden_size", 768), ("use_cache", True), ("num_hidden_layers", 30),
                            ("recursive_layer_count", 20), ("logical_to_physical", list(range(30)))):
            converted = contract.build_text_config(source_config())
            converted[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                contract.validate_text_config(converted)

    def test_gbs128_training_intent_is_not_resume(self):
        recipe = contract.training_intent()
        self.assertFalse(recipe["implemented"])
        self.assertEqual(recipe["world_size"] * recipe["microbatch"] * recipe["accumulation"], 128)
        self.assertEqual(recipe["steps_per_epoch"], recipe["rows"] // 128)
        self.assertEqual(recipe["total_steps"], 3 * recipe["steps_per_epoch"])
        self.assertEqual(recipe["warmup_steps"], round(.05 * recipe["total_steps"]))
        self.assertEqual(recipe["rows"] % 128, 116)

    def test_original_mapper_ast_matches_official_release(self):
        for name, expected in audit.release_manifest()["official_mapper_ast_sha256"].items():
            actual = hashlib.sha256(definition(contract.HERE / "ced.py", name).encode()).hexdigest()
            self.assertEqual(actual, expected, name)

    def test_ced_feature_mapper_and_compatibility_helpers_preserved(self):
        for name in ("CEDFrequencyMerger", "CEDTemporalMapper"):
            self.assertEqual(definition(contract.HERE / "ced.py", name), definition(REF / "models/ced.py", name))
        for name in ("_assert_supported_llama_api", "_causal_mask", "_call_decoder_layer"):
            self.assertEqual(definition(contract.HERE / "llama_compat.py", name),
                             definition(REF / "models/recursive_layer_inject.py", name))

    def test_pinned_official_release_inventory(self):
        release = audit.release_manifest()
        self.assertEqual(len(release["files"]) + 1, 18)
        self.assertEqual(release["tensor_count"], 454)
        self.assertEqual(release["checkpoint"]["sha256"],
                         "0c542f63e11ed55d360feb0d66e7be0e8c6a878f44cffd302a92388f915388a2")
        assets = audit.artifact_asset_records()
        self.assertEqual(len(assets), 14)
        self.assertNotIn("tokenizer/model.safetensors", {x["path"] for x in assets})

    def test_help_has_no_tensor_dependency(self):
        result = subprocess.run([sys.executable, "-B", "-m", "mizar_official_recursive_5_20x2_5", "--help"],
                                cwd=ROOT / "code/RSmol", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("convert", result.stdout)


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mizar-recursive-audit-")
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.assertTrue(self.root.name.startswith("mizar-recursive-audit-"))
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.temp.cleanup()

    def test_real_sha256_missing_and_tampered_files(self):
        path = self.root / "asset.bin"
        path.write_bytes(b"original")
        entry = audit.file_record(self.root, path)
        self.assertEqual(audit.verify_files(self.root, [entry]), ["asset.bin"])
        path.write_bytes(b"modified")
        with self.assertRaises(ValueError): audit.verify_files(self.root, [entry])
        path.unlink()
        with self.assertRaises(FileNotFoundError): audit.verify_files(self.root, [entry])

    def test_traversal_absolute_and_nonportable_paths_rejected(self):
        for name in ("../escape", "/absolute", "C:/absolute", "a\\b", "a/../b", "a/./b", "a//b", ""):
            with self.subTest(name=name), self.assertRaises(ValueError): audit.checked_path(self.root, name)

    def test_duplicate_manifest_entry_rejected(self):
        path = self.root / "asset"
        path.write_bytes(b"x")
        entry = audit.file_record(self.root, path)
        with self.assertRaises(ValueError): audit.verify_files(self.root, [entry, entry])

    def test_existing_output_and_source_overlap_rejected(self):
        source = self.root / "source"
        source.mkdir()
        output = self.root / "output"
        self.assertEqual(audit.validate_output_location(source, output), output)
        with self.assertRaises(ValueError): audit.validate_output_location(source, source / "new")
        with self.assertRaises(FileExistsError): audit.validate_output_location(source, source)
        output.mkdir()
        with self.assertRaises(FileExistsError): audit.validate_output_location(source, output)

    def test_output_inside_git_checkout_rejected(self):
        checkout = self.root / "repo"
        checkout.mkdir()
        (checkout / ".git").write_text("gitdir: /irrelevant")
        with self.assertRaises(ValueError): audit.validate_output_location(self.root / "source", checkout / "models")

    def test_atomic_publish_preserves_content_and_refuses_any_existing_output(self):
        staging = self.root / "stage"
        staging.mkdir()
        (staging / "marker").write_text("complete")
        output = self.root / "published"
        atomic_publish(staging, output)
        self.assertEqual((output / "marker").read_text(), "complete")
        self.assertFalse(staging.exists())
        for name in ("existing-empty", "existing-file", "published"):
            target = self.root / name
            if name == "existing-empty": target.mkdir()
            if name == "existing-file": target.write_text("preserve")
            stage = self.root / (name + "-stage")
            stage.mkdir()
            with self.assertRaises(OSError): atomic_publish(stage, target)
            self.assertTrue(stage.exists())

    def test_ced_label_indices_and_geometry_reject_incompatible_assets(self):
        config = {**CED_EXPECTED_CONFIG, "model_type": "ced", "auto_map": {
            "AutoConfig": "configuration_ced.CedConfig",
            "AutoModelForAudioClassification": "modeling_ced.CedForAudioClassification"},
            "id2label": {str(i): f"label{i}" for i in range(527)}}
        self.assertEqual(validate_ced_config(config)["temporal_tokens"], 126)
        config["embed_dim"] = 768
        with self.assertRaises(ValueError): validate_ced_config(config)

    def test_artifact_identity_tamper_rejected_before_tensor_load(self):
        manifest = {"schema": contract.SCHEMA, "route": "wrong-route"}
        contract.write_json(self.root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "identity mismatch"): audit.audit_artifact(self.root)

    def test_files_only_source_audit_never_claims_tensor_pass(self):
        checkpoint = self.root / "weights.ckpt"
        checkpoint.write_bytes(b"synthetic fixture, not real weights")
        config = self.root / "assets/smollm2-135m/config.json"
        config.parent.mkdir(parents=True)
        contract.write_json(config, source_config())
        release = {"checkpoint": audit.file_record(self.root, checkpoint),
                   "files": [audit.file_record(self.root, config)], "release": {"name": "test-fixture"}}
        with patch.object(audit, "release_manifest", return_value=release), \
                patch.object(audit, "validate_ced_assets", return_value={"fixture": True}):
            result = audit.audit_source(self.root)
        self.assertEqual(result["scope"], "source_files_only")
        self.assertEqual(result["tensor_audit"], "NOT_RUN")
        self.assertEqual(result["verified_file_count"], 2)

    def fixture_artifact(self):
        """Synthetic file fixture only; no torch.load or production artifact claim."""
        cfg = self.root / "tokenizer/config.json"
        cfg.parent.mkdir()
        contract.write_json(cfg, source_config())
        for name in ("model.ckpt", "LICENSE", "NOTICE"):
            (self.root / name).write_text("synthetic unit-test fixture")
        (self.root / "licenses").mkdir()
        (self.root / "licenses/Mizar-ClearBSD.txt").write_text("synthetic unit-test fixture")
        contract.write_json(self.root / "config.json", contract.build_text_config(source_config()))
        release = audit.release_manifest()
        contract.write_json(self.root / "source_audit.json", {
            "status": "PASS", "scope": "source_files_only",
            "verified_files": [release["checkpoint"], *release["files"]],
            "release_manifest_sha256": contract.sha256_file(contract.RELEASE_FILE)})
        contract.write_json(self.root / "conversion_report.json", {
            "status": "PASS", "scope": "full_conversion", "reload": {"status": "PASS"}})
        assets = [audit.file_record(self.root, cfg)]
        manifest = {"schema": contract.SCHEMA, "route": contract.ROUTE,
                    "architecture": contract.architecture(), "training_intent": contract.training_intent(),
                    "implementation": contract.implementation_identity(),
                    "source_release_manifest_sha256": contract.sha256_file(contract.RELEASE_FILE)}
        self.refresh_fixture_manifest(manifest)
        return manifest, assets

    def refresh_fixture_manifest(self, manifest):
        manifest["files"] = [audit.file_record(self.root, p) for p in sorted(self.root.rglob("*"))
                             if p.is_file() and p.name != "manifest.json"]
        contract.write_json(self.root / "manifest.json", manifest)

    def test_artifact_inventory_and_config_byte_tamper_are_detected(self):
        manifest, assets = self.fixture_artifact()
        with patch.object(audit, "artifact_asset_records", return_value=assets):
            self.assertEqual(audit.audit_artifact(self.root)["tensor_audit"], "NOT_RUN")
            (self.root / "optimizer.pt").write_text("unexpected training state")
            with self.assertRaisesRegex(ValueError, "unlisted"): audit.audit_artifact(self.root)
            (self.root / "optimizer.pt").unlink()
            (self.root / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA256"): audit.audit_artifact(self.root)

    def test_config_must_preserve_all_source_fields_even_after_rehash(self):
        manifest, assets = self.fixture_artifact()
        cfg = audit.read_json(self.root / "config.json")
        cfg["custom_preserved"]["sentinel"] = -1
        contract.write_json(self.root / "config.json", cfg)
        self.refresh_fixture_manifest(manifest)
        with patch.object(audit, "artifact_asset_records", return_value=assets), \
                self.assertRaisesRegex(ValueError, "every original source config field"):
            audit.audit_artifact(self.root)

    def test_incomplete_reload_report_is_not_a_published_artifact(self):
        manifest, assets = self.fixture_artifact()
        report = audit.read_json(self.root / "conversion_report.json")
        report["reload"] = "PENDING"
        contract.write_json(self.root / "conversion_report.json", report)
        self.refresh_fixture_manifest(manifest)
        with patch.object(audit, "artifact_asset_records", return_value=assets):
            with self.assertRaisesRegex(ValueError, "mandatory tensor/logits reload"):
                audit.audit_artifact(self.root)
            self.assertEqual(audit.audit_artifact(self.root, _pending_reload=True)["scope"], "artifact_files_only")

    def test_any_implementation_change_is_refused_before_deserialization(self):
        manifest, assets = self.fixture_artifact()
        manifest["implementation"]["model.py"] = "0" * 64
        contract.write_json(self.root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "identity mismatch"): audit.audit_artifact(self.root)


if __name__ == "__main__":
    unittest.main()
