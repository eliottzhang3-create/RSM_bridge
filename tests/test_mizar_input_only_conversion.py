"""Real stdlib file/path/contract checks; synthetic fixtures are not weight audits."""
from __future__ import annotations

import ast
import copy
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/RSmol"))
from mizar_official_recursive_5_20x2_5_input_only import audit, contract
from mizar_official_recursive_5_20x2_5_input_only import conversion
from mizar_official_recursive_5_20x2_5_input_only.assets import CED_EXPECTED_CONFIG, validate_ced_config
from mizar_official_recursive_5_20x2_5_input_only.conversion import atomic_publish
from mizar_official_recursive_5_20x2_5_input_only.tensor_runtime import config_from_dict

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


def mutating_config_library():
    """Configuration-only test double; does not simulate tensors or model PASS."""
    module = types.ModuleType("transformers")

    class MutatingLlamaConfig:
        @classmethod
        def from_dict(cls, value):
            # Reproduce the library's top-level mutation and exercise nested
            # ownership too, so a shallow dict(value) fix cannot pass.
            value["attn_implementation"] = None
            value["custom_preserved"]["sentinel"] = -1
            return types.SimpleNamespace(**value)

    module.LlamaConfig = MutatingLlamaConfig
    return patch.dict(sys.modules, {"transformers": module})


class ContractTests(unittest.TestCase):
    def test_full_thirty_layer_schedule_and_serial_boundaries(self):
        schedule = contract.SCHEDULE
        self.assertEqual(len(schedule), 50)
        self.assertEqual(schedule[:5], tuple(range(5)))
        self.assertEqual(schedule[5:25], tuple(range(5, 25)))
        self.assertEqual(schedule[25:45], schedule[5:25])
        self.assertEqual(schedule[45:], tuple(range(25, 30)))
        self.assertEqual(contract.INJECTION_INDICES, (5, 25))
        self.assertEqual(contract.INPUT_TRACE, (5, 25))

    def test_input_only_counts_from_projection_geometries(self):
        hidden = 576
        audio = hidden * hidden + hidden + hidden + 1
        self.assertEqual(audio, contract.AUDIO_PARAMETERS)
        self.assertEqual(159299840 + audio, 159632769)
        self.assertEqual(contract.TOTAL_PARAMETERS, 159632769)
        self.assertEqual(contract.NEW_TENSORS, 4)
        self.assertFalse(contract.architecture()["lora"])
        self.assertFalse(contract.architecture()["encoder_intermediate_injection"])

    def test_learning_rates_and_layer_norm_boundaries(self):
        for i in (0, 4, 5, 24, 25, 29):
            lr = 6e-4 if 5 <= i <= 24 else 3e-4
            for suffix in ("self_attn.q_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight"):
                self.assertEqual(contract.parameter_role(f"caption_decoder.lm.model.layers.{i}.{suffix}")[1], lr)
        for name in ("audio_encoder.base.mapper.pos_in", "caption_decoder.lm.model.embed_tokens.weight",
                     "caption_decoder.lm.lm_head.weight", "caption_decoder.lm.model.norm.weight"):
            self.assertEqual(contract.parameter_role(name)[1], 3e-4)
        for name in ("caption_decoder.lm.model.audio_linear.weight", "caption_decoder.lm.model.audio_gate.bias"):
            self.assertEqual(contract.parameter_role(name)[1], 6e-4)
        self.assertIsNone(contract.parameter_role("audio_encoder.base.ced.blocks.0.weight")[1])

    def test_invalid_parameter_roles_fail_closed(self):
        for name in ("router.weight", "audio_encoder.base.layer_mlps.4.net.1.weight",
                     "caption_decoder.lm.model.audio_gates.4.bias",
                     "caption_decoder.lm.model.layers.24.self_attn.q_proj.lora_B.1",
                     "caption_decoder.lm.model.layers.30.input_layernorm.weight",
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

    def test_config_constructor_mutation_cannot_reach_persisted_or_source_dict(self):
        for value in (source_config(), contract.build_text_config(source_config())):
            before = copy.deepcopy(value)
            with mutating_config_library():
                runtime = config_from_dict(value)
            self.assertEqual(value, before)
            self.assertEqual(runtime._attn_implementation, "eager")
            self.assertEqual(runtime.custom_preserved["sentinel"], -1)
            self.assertNotIn("attn_implementation", value)

    def test_preservation_error_reports_runtime_fields_without_allowing_them(self):
        value = contract.build_text_config(source_config())
        value["attn_implementation"] = None
        with self.assertRaisesRegex(ValueError, r"extra=\['attn_implementation'\]"):
            contract.validate_config_preservation(value, source_config())

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
        result = subprocess.run([sys.executable, "-B", "-m", "mizar_official_recursive_5_20x2_5_input_only", "--help"],
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

    @staticmethod
    def emulate_posix_empty_directory_replace(source, destination):
        # Windows cannot replace an empty directory. Emulate just that POSIX
        # primitive to exercise the Linux reservation path on this local host.
        destination = Path(destination)
        destination.rmdir()
        FileTests.native_rename(source, destination)

    native_rename = staticmethod(os.rename)

    def test_linux_unsupported_rename_flags_use_reservation_with_real_file_content(self):
        for number in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            staging, output = self.root / f"stage-{number}", self.root / f"out-{number}"
            staging.mkdir()
            (staging / "model").write_bytes(b"verified-model-fixture")
            with patch.object(conversion.sys, "platform", "linux"), \
                    patch.object(conversion, "_linux_noreplace", side_effect=OSError(number, "unsupported")), \
                    patch.object(conversion.os, "rename", side_effect=self.emulate_posix_empty_directory_replace):
                method = atomic_publish(staging, output)
            self.assertEqual(method, "posix_rename_into_exclusive_empty_reservation")
            self.assertEqual((output / "model").read_bytes(), b"verified-model-fixture")
            self.assertFalse(staging.exists())

    def test_linux_permission_failure_does_not_trigger_compatibility_fallback(self):
        staging, output = self.root / "stage", self.root / "out"
        staging.mkdir()
        with patch.object(conversion.sys, "platform", "linux"), \
                patch.object(conversion, "_linux_noreplace", side_effect=OSError(errno.EACCES, "denied")), \
                patch.object(conversion, "_reserved_posix_rename") as fallback:
            with self.assertRaises(PermissionError): atomic_publish(staging, output)
            fallback.assert_not_called()
        self.assertTrue(staging.exists())
        self.assertFalse(output.exists())

    def test_failed_posix_rename_removes_only_its_empty_reservation_and_keeps_stage(self):
        staging, output = self.root / "stage", self.root / "out"
        staging.mkdir()
        (staging / "model").write_bytes(b"preserve")
        with patch.object(conversion.sys, "platform", "linux"), \
                patch.object(conversion, "_linux_noreplace", side_effect=OSError(errno.EINVAL, "flags")), \
                patch.object(conversion.os, "rename", side_effect=OSError(errno.EIO, "I/O")):
            with self.assertRaises(OSError): atomic_publish(staging, output)
        self.assertFalse(output.exists())
        self.assertEqual((staging / "model").read_bytes(), b"preserve")

    def test_foreign_content_in_reservation_survives_failed_publication(self):
        staging, output = self.root / "stage", self.root / "out"
        staging.mkdir()
        def fail_after_foreign_write(*args):
            (output / "foreign").write_text("preserve")
            raise OSError(errno.ENOTEMPTY, "directory is no longer empty")
        with patch.object(conversion.sys, "platform", "linux"), \
                patch.object(conversion, "_linux_noreplace", side_effect=OSError(errno.EINVAL, "flags")), \
                patch.object(conversion.os, "rename", side_effect=fail_after_foreign_write):
            with self.assertRaises(OSError): atomic_publish(staging, output)
        self.assertEqual((output / "foreign").read_text(), "preserve")
        self.assertTrue(staging.exists())

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
            "status": "PASS", "scope": "full_conversion", "reload": {"status": "PASS",
                "all_saved_tensors_equal": 458, "full_683_token_logits_equal": True, "strict_state_dict": True}})
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

    def test_constructor_then_json_save_still_passes_strict_artifact_config_check(self):
        manifest, assets = self.fixture_artifact()
        value = audit.read_json(self.root / "config.json")
        with mutating_config_library():
            config_from_dict(value)
        # Follow the actual conversion order: construct first, serialize the
        # original canonical dictionary later, then run the strict file audit.
        contract.write_json(self.root / "config.json", value)
        self.refresh_fixture_manifest(manifest)
        with patch.object(audit, "artifact_asset_records", return_value=assets):
            self.assertEqual(audit.audit_artifact(self.root)["tensor_audit"], "NOT_RUN")

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

    def completed_stage_fixture(self):
        outer = self.root
        staging = outer / "model.staging-verified"
        staging.mkdir()
        try:
            self.root = staging
            manifest, assets = self.fixture_artifact()
            # Simulate the previous publisher implementation, keeping every
            # architecture/model/config/helper identity exactly current.
            manifest["implementation"]["conversion.py"] = "0" * 64
            report = audit.read_json(staging / "conversion_report.json")
            report.update(source_alignment={"status": "PASS"}, recursive_forward={"status": "PASS"})
            report["reload"].update(all_saved_tensors_equal=458, full_683_token_logits_equal=True,
                                    strict_state_dict=True)
            contract.write_json(staging / "conversion_report.json", report)
            self.refresh_fixture_manifest(manifest)
            contract.write_json(staging / "failure.json", {"status": "FAIL", "error": "[Errno 22] Invalid argument"})
        finally:
            self.root = outer
        return staging, outer / "model", assets

    def test_publication_only_reuses_completed_stage_and_preserves_diagnostics(self):
        staging, output, assets = self.completed_stage_fixture()
        weights = (staging / "model.ckpt").read_bytes()
        with patch.object(audit, "artifact_asset_records", return_value=assets):
            result = conversion.publish_staged(staging, output, progress=lambda x: None)
            self.assertEqual(audit.audit_artifact(output)["status"], "PASS")
        self.assertEqual(result["scope"], "publication_only")
        self.assertFalse(result["tensor_reexecution"])
        self.assertFalse(staging.exists())
        self.assertEqual((output / "model.ckpt").read_bytes(), weights)
        self.assertFalse((output / "failure.json").exists())
        backup = Path(result["diagnostics_backup"])
        self.assertTrue((backup / "failure.json").is_file())
        self.assertEqual(audit.read_json(backup / "manifest.json")["implementation"]["conversion.py"], "0" * 64)
        self.assertFalse(output.with_name("model.conversion.lock").exists())

    def test_publication_only_rejects_model_identity_change(self):
        staging, output, assets = self.completed_stage_fixture()
        manifest = audit.read_json(staging / "manifest.json")
        manifest["implementation"]["model.py"] = "0" * 64
        contract.write_json(staging / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "model/contract implementation changed"):
            conversion.publish_staged(staging, output, progress=lambda x: None)
        self.assertFalse(output.exists())
        self.assertTrue((staging / "failure.json").exists())

    def test_publication_only_rejects_prior_incomplete_reload(self):
        staging, output, assets = self.completed_stage_fixture()
        report = audit.read_json(staging / "conversion_report.json")
        report["reload"] = "PENDING"
        contract.write_json(staging / "conversion_report.json", report)
        manifest = audit.read_json(staging / "manifest.json")
        manifest["files"] = [audit.file_record(staging, staging / i["path"]) for i in manifest["files"]]
        contract.write_json(staging / "manifest.json", manifest)
        with patch.object(audit, "artifact_asset_records", return_value=assets), \
                self.assertRaisesRegex(ValueError, "mandatory tensor/logits reload"):
            conversion.publish_staged(staging, output, progress=lambda x: None)
        self.assertFalse(output.exists())

    def test_publication_only_requires_actual_completed_reload_fields_beyond_pass_label(self):
        staging, output, assets = self.completed_stage_fixture()
        report = audit.read_json(staging / "conversion_report.json")
        report["reload"] = {"status": "PASS"}
        contract.write_json(staging / "conversion_report.json", report)
        manifest = audit.read_json(staging / "manifest.json")
        manifest["files"] = [audit.file_record(staging, staging / i["path"]) for i in manifest["files"]]
        contract.write_json(staging / "manifest.json", manifest)
        with patch.object(audit, "artifact_asset_records", return_value=assets), \
                self.assertRaisesRegex(ValueError, "mandatory tensor/logits reload"):
            conversion.publish_staged(staging, output, progress=lambda x: None)
        self.assertFalse(output.exists())

    def test_cli_publish_runs_without_importing_tensor_dependencies(self):
        from mizar_official_recursive_5_20x2_5_input_only.__main__ import main
        staging, output, assets = self.completed_stage_fixture()
        captured, errors = io.StringIO(), io.StringIO()
        with patch.object(audit, "artifact_asset_records", return_value=assets), \
                patch.dict(sys.modules, {"torch": None, "transformers": None, "torchaudio": None}), \
                redirect_stdout(captured), redirect_stderr(errors):
            result = main(["publish", "--staging", str(staging), "--output", str(output)])
        self.assertEqual(result, 0, errors.getvalue())
        self.assertIn('"tensor_reexecution": false', captured.getvalue())
        self.assertTrue((output / "model.ckpt").is_file())

    def test_publication_only_detects_saved_weight_tamper(self):
        staging, output, assets = self.completed_stage_fixture()
        (staging / "model.ckpt").write_text("tampered")
        with patch.object(audit, "artifact_asset_records", return_value=assets), \
                self.assertRaisesRegex(ValueError, "SHA256/size mismatch"):
            conversion.publish_staged(staging, output, progress=lambda x: None)
        self.assertFalse(output.exists())

    def test_publication_failure_restores_metadata_and_preserves_new_diagnostics(self):
        staging, output, assets = self.completed_stage_fixture()
        manifest_before = (staging / "manifest.json").read_bytes()
        report_before = (staging / "conversion_report.json").read_bytes()
        with patch.object(audit, "artifact_asset_records", return_value=assets), \
                patch.object(conversion, "atomic_publish", side_effect=OSError(errno.EIO, "failed publish")), \
                self.assertRaises(OSError):
            conversion.publish_staged(staging, output, progress=lambda x: None)
        self.assertEqual((staging / "manifest.json").read_bytes(), manifest_before)
        self.assertEqual((staging / "conversion_report.json").read_bytes(), report_before)
        self.assertIn("failed publish", audit.read_json(staging / "failure.json")["error"])
        self.assertFalse(output.exists())

    def test_publication_only_rejects_existing_empty_final_directory(self):
        staging, output, assets = self.completed_stage_fixture()
        output.mkdir()
        with self.assertRaises(FileExistsError):
            conversion.publish_staged(staging, output, progress=lambda x: None)
        self.assertTrue(staging.exists())
        self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
