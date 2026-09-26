"""Dependency-light contracts for matched native Mellow-v0 MMAU/MMAR evaluation."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"
SCRIPTS = RSMOL / "scripts"
PREFLIGHT = SCRIPTS / "audit_mellow_v0_artifact.py"
EVALUATOR = SCRIPTS / "evaluate_mmau_test_mini_mellow_v0.py"
PREFLIGHT_SH = RSMOL / "run_mellow_v0_artifact_preflight.sh"
SMOKE_SH = RSMOL / "run_mmau_test_mini_mellow_v0_smoke_4090.sh"
FULL_SH = RSMOL / "run_mmau_test_mini_mellow_v0_full_4090.sh"
MMAR_EVALUATOR = SCRIPTS / "evaluate_mmar_mellow_v0.py"
MMAR_RUNTIME = SCRIPTS / "evaluate_mmar_mellow_v0.sh"
MMAR_SUBMIT = RSMOL / "run_mmar_mellow_v0_4090.sh"


def load_evaluator():
    scripts_text = str(SCRIPTS)
    if scripts_text not in sys.path:
        sys.path.insert(0, scripts_text)
    spec = importlib.util.spec_from_file_location("native_mellow_v0_mmau", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_mmar_evaluator():
    scripts_text = str(SCRIPTS)
    if scripts_text not in sys.path:
        sys.path.insert(0, scripts_text)
    spec = importlib.util.spec_from_file_location("native_mellow_v0_mmar", MMAR_EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeMellowV0MMAUStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.preflight = PREFLIGHT.read_text(encoding="utf-8")
        cls.evaluator_text = EVALUATOR.read_text(encoding="utf-8")
        cls.evaluator = load_evaluator()
        cls.smoke = SMOKE_SH.read_text(encoding="utf-8")
        cls.full = FULL_SH.read_text(encoding="utf-8")
        cls.mmar_text = MMAR_EVALUATOR.read_text(encoding="utf-8")
        cls.mmar = load_mmar_evaluator()
        cls.mmar_runtime = MMAR_RUNTIME.read_text(encoding="utf-8")
        cls.mmar_submit = MMAR_SUBMIT.read_text(encoding="utf-8")

    def test_preflight_strictly_loads_complete_native_checkpoint_on_cpu(self) -> None:
        for marker in (
            '"htsat": "audio_encoder.base.htsat."',
            '"c2l": "audio_encoder.base.c2l."',
            '"projection": "audio_encoder.projection."',
            '"text_decoder": "caption_decoder.lm."',
            'map_location="cpu"',
            'load_state_dict(state, strict=True)',
            '"runtime_state_shapes_match": True',
            '"runtime_state_dtypes_match": True',
            '"example_audio_check": "NOT_REQUESTED"',
            "decoder.py independently hardcodes id 0",
            '"separator_token_id": 0',
        ):
            self.assertIn(marker, self.preflight)
        self.assertNotIn("resource/1.wav", self.preflight)
        self.assertNotIn("resource/2.wav", self.preflight)
        self.assertNotIn("torch.cuda", self.preflight)

    def test_evaluator_uses_native_two_slot_prefix_with_separate_encoding(self) -> None:
        self.assertEqual(self.evaluator.MELLOW_PREFIX_TOKENS, 389)
        self.assertEqual(self.evaluator.MELLOW_AUDIO_TOKENS_PER_SLOT, 129)
        for marker in (
            '"audio1": audio',
            '"audio2": audio',
            "model.generate_prefix_inference",
            '"audio2_reused": False',
            '"audio2_encoded_separately": True',
            '"native_audio_encoder_invocations": 2',
            '"compact_single_audio_prefix_used": False',
        ):
            self.assertIn(marker, self.evaluator_text)

    def test_author_reply_protocol_covers_prompt_audio_generation_and_dual_scoring(self) -> None:
        for marker in (
            "build_mellow_author_reply_prompt",
            "decode_mellow_author_reply_audio",
            "mellow_author_reply_audio_segment",
            "DEFAULT_MAX_NEW_TOKENS = 300",
            "write_mellow_author_reply_evaluation",
            '"mellow_author_reply_evaluation"',
            '"mmau_v051525_evaluation"',
            "MELLOW_AUTHOR_REPLY_CONTEXT",
        ):
            self.assertIn(marker, self.evaluator_text)

    def test_generation_matches_author_reply_top_p_argmax_and_is_verbatim(self) -> None:
        prepare = self.evaluator.prepare_model_output_for_official_scorer
        for value in ("a) answer", "  D) untouched  ", "free text"):
            self.assertEqual(prepare(value), value)
        common = Path(self.evaluator.official.__file__).read_text(encoding="utf-8")
        for marker in (
            "torch.argmax",
            "language_model_default_exactly_as_wrapper",
            "model.caption_decoder.lm(inputs_embeds=generated)",
            '"do_sample": False',
            '"top_p": 0.8',
            '"comparable": bool(',
            '"temperature": 1.0',
            "cumulative_probs",
            "sorted_indices_to_remove",
            "generated_text = str(generation.get",
        ):
            self.assertIn(marker, self.evaluator_text + "\n" + common)

    def test_preflight_report_is_a_required_fresh_artifact_gate(self) -> None:
        for marker in (
            "_load_and_validate_preflight",
            'report.get("status") != "PASS"',
            'report.get("strict_state_dict_load") is not True',
            "mellow_checkpoint_sha256",
            "snapshot_config_sha256",
            "current_source_hashes",
            "current_smollm2_inventory",
            "preflight report is stale",
        ):
            self.assertIn(marker, self.evaluator_text)

    def test_cluster_storage_mount_aliases_have_the_same_identity(self) -> None:
        hpc = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main")
        cloud = "/mnt/cloudstorfs/sjtu_home/jinwei.zhang/models/mellow-main"
        self.assertEqual(
            self.evaluator._shared_storage_identity(hpc),
            self.evaluator._shared_storage_identity(cloud),
        )
        self.assertTrue(self.evaluator._same_artifact_path(hpc, cloud))
        self.assertFalse(
            self.evaluator._same_artifact_path(
                hpc,
                "/mnt/cloudstorfs/sjtu_home/jinwei.zhang/models/other-model",
            )
        )

    def test_persistent_model_contract_uses_mount_independent_identities(self) -> None:
        args = SimpleNamespace(
            preflight_report=Path("/hpc_stor03/user/preflight.json"),
            mellow_source_root=Path("/hpc_stor03/user/mellow"),
            mellow_snapshot=Path("/hpc_stor03/user/snapshot"),
            mellow_checkpoint=Path("/hpc_stor03/user/snapshot/v0.ckpt"),
            base_smollm2=Path("/hpc_stor03/user/SmolLM2"),
        )
        with mock.patch.object(self.evaluator.preflight, "_sha256", return_value="report-hash"):
            contract = self.evaluator._model_contract(
                args,
                {"mellow_checkpoint_sha256": "checkpoint-hash"},
            )
        self.assertEqual(contract["path_identity_policy"], "known_shared_storage_alias_v1")
        self.assertEqual(contract["mellow_source_root"], "shared-storage:/user/mellow")
        self.assertEqual(contract["mellow_snapshot"], "shared-storage:/user/snapshot")
        self.assertEqual(
            contract["mellow_checkpoint"],
            "shared-storage:/user/snapshot/v0.ckpt",
        )
        self.assertEqual(contract["base_smollm2"], "shared-storage:/user/SmolLM2")

    def test_preflight_accepts_mount_alias_but_still_checks_all_content_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            snapshot = root / "snapshot"
            base = root / "base"
            source.mkdir()
            snapshot.mkdir()
            base.mkdir()
            checkpoint = snapshot / "v0.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            report_path = root / "preflight.json"
            source_inventory = {"mellow/model/model.py": "source-hash"}
            base_inventory = {"required_file_sha256": {"config.json": "base-hash"}}
            report = {
                "status": "PASS",
                "artifact_contract": self.evaluator.preflight.ARTIFACT_CONTRACT,
                "strict_state_dict_load": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "mellow_source_root": str(source),
                "mellow_snapshot": str(snapshot),
                "mellow_checkpoint": str(checkpoint),
                "base_smollm2": str(base),
                "mellow_checkpoint_sha256": "checkpoint-hash",
                "snapshot_config_sha256": "snapshot-hash",
                "mellow_source_sha256": source_inventory,
                "base_smollm2_inventory": base_inventory,
            }
            report_path.write_text(__import__("json").dumps(report), encoding="utf-8")
            args = SimpleNamespace(
                preflight_report=report_path,
                mellow_source_root=source,
                mellow_snapshot=snapshot,
                mellow_checkpoint=checkpoint,
                base_smollm2=base,
            )
            real_same_path = self.evaluator._same_artifact_path

            def same_path_with_alias(expected, reported):
                return real_same_path(expected, reported)

            with (
                mock.patch.object(self.evaluator, "_same_artifact_path", side_effect=same_path_with_alias),
                mock.patch.object(self.evaluator.preflight, "_sha256", side_effect=["checkpoint-hash", "snapshot-hash"]),
                mock.patch.object(self.evaluator.preflight, "_source_inventory", return_value=source_inventory),
                mock.patch.object(self.evaluator.preflight, "_smollm2_inventory", return_value=base_inventory),
            ):
                loaded = self.evaluator._load_and_validate_preflight(args)
            self.assertEqual(loaded["runtime_path_validation"]["status"], "PASS")
            self.assertIn("content_hashes", loaded["runtime_path_validation"]["policy"])

    def test_full_can_start_directly_without_smoke_gate(self) -> None:
        for marker in (
            "SMOKE_GATE_FILENAME",
            "_require_completed_smoke",
            "_smoke_gate_payload",
            "requires the completed first-five smoke gate",
        ):
            self.assertNotIn(marker, self.evaluator_text)
        self.assertIn("--mode full", self.full)
        self.assertIn("--run-official-evaluation", self.full)

    def test_smoke_and_full_share_output_and_respect_scheduler_limits(self) -> None:
        default_output = (
            "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0/"
            "mmau_test_mini_mellow_author_reply_matched_smollm2_113430_v2"
        )
        for wrapper in (self.smoke, self.full):
            self.assertIn(default_output, wrapper)
            self.assertIn("-p pdgpu-4090", wrapper)
            self.assertIn("-c 8 -m 32G -g 1", wrapper)
            self.assertIn("--preflight-report", wrapper)
            self.assertIn("--max-prompt-tokens 129", wrapper)
            self.assertIn("--max-new-tokens 300", wrapper)
            self.assertIn("--dtype fp32", wrapper)
        self.assertIn("--mode smoke", self.smoke)
        self.assertNotIn("--run-official-evaluation", self.smoke)
        self.assertIn("--mode full", self.full)
        self.assertIn("--run-official-evaluation", self.full)
        self.assertIn("scripts/audit_mellow_v0_artifact.sh", PREFLIGHT_SH.read_text(encoding="utf-8"))

    def test_existing_complete_denominator_and_smoke_resume_backend_is_reused(self) -> None:
        combined = self.evaluator_text + "\n" + Path(
            self.evaluator.official.__file__
        ).read_text(encoding="utf-8")
        for marker in (
            "official.run(",
            "SMOKE_ROWS",
            "EXPECTED_FULL_ROWS",
            "prompt_exceeds_max_tokens",
            "skipped_rows_scored_as_incorrect",
            "prepare_prediction=prepare_model_output_for_official_scorer",
        ):
            self.assertIn(marker, combined)

    def test_mmau_records_exact_comparison_reference(self) -> None:
        target = (
            "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
            "audio_smollm2_135m_mellow_shared_store_configurable_epochs/"
            "formal_30epochs_20260923_v1/"
            "mmau_test_mini_checkpoint_113430_mellow_author_reply_protocol_v1"
        )
        self.assertEqual(
            self.evaluator.MMAU_COMPARISON_REFERENCE["evaluation_output"], target
        )
        self.assertIn("MMAU-v05.15.25", self.evaluator_text)
        self.assertIn('report["comparison_reference"]', self.evaluator_text)

    def test_native_mmar_files_and_fixed_runtime_contract(self) -> None:
        for path in (MMAR_EVALUATOR, MMAR_RUNTIME, MMAR_SUBMIT):
            self.assertTrue(path.is_file(), path)
        args = self.mmar.parse_args([
            "--dataset-dir", "/tmp/mmar",
            "--output-dir", "/tmp/mellow-mmar",
        ])
        self.assertEqual(args.mode, "full")
        self.assertEqual(args.max_prompt_tokens, 129)
        self.assertEqual(args.max_new_tokens, 32)
        self.assertEqual(args.dtype, "fp32")
        with self.assertRaises(SystemExit):
            self.mmar.parse_args([
                "--output-dir", "/tmp/mellow-mmar",
                "--max-prompt-tokens", "476",
            ])
        with self.assertRaises(SystemExit):
            self.mmar.parse_args([
                "--output-dir", "/tmp/mellow-mmar",
                "--max-new-tokens", "300",
            ])

    def test_native_mmar_matches_target_data_and_dual_scoring(self) -> None:
        target = (
            "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
            "audio_smollm2_135m_mellow_shared_store_configurable_epochs/"
            "formal_30epochs_20260923_v1/mmar_checkpoint_113430_dual_scoring_v1"
        )
        self.assertEqual(
            self.mmar.MMAR_COMPARISON_REFERENCE["evaluation_output"], target
        )
        for marker in (
            "evaluate_mmar_5_10x2_5_mesh_mellow as official",
            "evaluate_mmau_test_mini_mellow_v0 as mellow",
            "load_runtime_model=_load_runtime_model",
            'runtime["runtime_model_contract"] = _model_contract',
            "run_model_generation=mellow._run_model_generation",
            "prepare_prediction=mellow.prepare_model_output_for_official_scorer",
            "write_choice_label_prefix_evaluation",
            'report["choice_label_prefix_evaluation"]',
            'report["dual_scoring"]',
            '"prediction_text_shared_without_preparse": True',
            '"official_metadata_and_scorer_shared_with_comparison": True',
            '"top_p_argmax_is_greedy_equivalent": True',
            '"model_inherent_difference"',
            '"language_model_default_exactly_as_released_wrapper"',
        ):
            self.assertIn(marker, self.mmar_text)

    def test_native_mmar_submission_is_isolated_and_official(self) -> None:
        for marker in (
            "-p pdgpu-4090",
            "-c 8 -m 32G -g 1 -n 1",
            "MMAR-meta.json",
            "mmar-audio",
            "code/evaluation.py",
            "--max-prompt-tokens 129",
            "--max-new-tokens 32",
            "--dtype fp32",
            "--run-official-evaluation",
            "mmar_mellow_v0_dual_scoring_matched_smollm2_113430_v1",
        ):
            self.assertIn(marker, self.mmar_submit)
        self.assertIn("evaluate_mmar_mellow_v0.py", self.mmar_runtime)


if __name__ == "__main__":
    unittest.main()
