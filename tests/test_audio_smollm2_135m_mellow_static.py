"""Dependency-light static contracts for the original SmolLM2 audio baseline."""
from __future__ import annotations

import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "code" / "RSmol" / "audio_smollm2_135m_mellow"
TRAIN = ROOT / "code" / "RSmol" / "scripts" / "train_audio_smollm2_135m_mellow_ddp.py"
AUDIT = ROOT / "code" / "RSmol" / "scripts" / "audit_audio_smollm2_135m_mellow_checkpoint.py"
INNER = tuple(ROOT / "code" / "RSmol" / "scripts" / name for name in (
    "train_audio_smollm2_135m_mellow_smoke20_ddp.sh",
    "train_audio_smollm2_135m_mellow_resume2_ddp.sh",
    "train_audio_smollm2_135m_mellow_formal_ddp.sh",
    "audit_audio_smollm2_135m_mellow_checkpoint.sh",
))
WRAPPERS = tuple(ROOT / "code" / "RSmol" / name for name in (
    "run_audio_smollm2_135m_mellow_smoke20_5090.sh",
    "run_audio_smollm2_135m_mellow_resume2_5090.sh",
    "run_audio_smollm2_135m_mellow_formal_5090.sh",
    "run_audio_smollm2_135m_mellow_checkpoint_audit_5090.sh",
))


class SmolLM2AudioBaselineStaticTest(unittest.TestCase):
    def test_files_exist(self) -> None:
        for path in (PKG / "__init__.py", PKG / "data.py", PKG / "model.py", PKG / "README.md", TRAIN, AUDIT, *INNER, *WRAPPERS):
            self.assertTrue(path.is_file(), path)

    def test_standard_text_contract_is_explicit(self) -> None:
        text = (PKG / "model.py").read_text(encoding="utf-8")
        for marker in (
            "LlamaForCausalLM",
            "model_type",
            "SMOLLM2_LAYER_COUNT = 30",
            "SMOLLM2_HIDDEN_SIZE = 576",
            "independent_decoder_layers",
            "embedding_lm_head_tied",
            "AudioSmolLM2Model",
            "use_cache=False",
            "inputs_embeds=inputs_embeds",
            "AUDIO_TOKENS_PER_CLIP = 129",
            "AUDIO_PREFIX_TOKENS = 260",
            "all_text_trainable",
            'convert_tokens_to_ids("!")',
        ):
            self.assertIn(marker, text)
        self.assertNotIn("RecursiveLlamaForCausalLM", text)
        self.assertNotIn("write_routers", text)
        self.assertNotIn("read_routers", text)

    def test_shared_audio_data_and_labels_are_explicit(self) -> None:
        data = (PKG / "data.py").read_text(encoding="utf-8")
        model = (PKG / "model.py").read_text(encoding="utf-8")
        self.assertIn("ReasonAQADataset", data)
        self.assertIn("collate_reasonaqa", data)
        for marker in ("AudioBridge", "MAPPER_CONTRACT", "build_labels", "audio2_reused_mask", "prefix_length"):
            self.assertIn(marker, model + data)

    def test_checkpoint_and_resume_contract(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        for marker in (
            "AutoModelForCausalLM.from_pretrained",
            "text_model/",
            "audio_smollm2_config.json",
            "checkpoint_complete.json",
            "rng_states_by_rank",
            "_validate_optimizer_coverage",
            "batch_in_epoch",
            "tempfile.mkdtemp",
            "refusing to overwrite existing checkpoint",
            "refusing to prune checkpoint outside output directory",
            "broadcast_buffers=False",
            "find_unused_parameters=False",
            "DistributedSampler(dataset",
            "sampler.set_epoch(epoch)",
            "math.ceil(total_steps * 0.05)",
            "schedule_total_steps",
            "run_target_step",
            "execution_max_steps",
            '"gate": str(args.gate)',
            '"run_kind": run_kind',
            "expected_run_kind",
            "STAGE7 checkpoints cannot resume FORMAL",
            '"scheduler": "cosine_lambda"',
            '"scheduler_name": "cosine_lambda"',
            "training_state gate/run_kind disagrees with config",
            '"sample_rate": 32000',
            '"audio_seconds": 10',
            '"save_every": int(args.save_every)',
            '"checkpoint_retention": int(args.checkpoint_retention)',
            '"effective_global_batch_size": int(args.micro_batch_size * args.world_size * args.gradient_accumulation_steps)',
            '"frozen_audio_encoder": True',
            '"periodic_validation": False',
            '"trainable_parameter_names": [',
            'marker.required is invalid',
            "FORMAL checkpoint canonical contract mismatch",
            "FORMAL checkpoint warmup/schedule/run-target contract mismatch",
            "--tokenizer-path cannot be combined with --resume-from",
            "resume_from",
            "resume_start_step",
            "parameter_change_audit",
            "_select_resume_representatives",
            "_compute_resume_parameter_change_audit",
            "_validate_parameter_change_audit",
            'item.get("finite") is not True',
            "text_model_config_sha256",
            "saved_source",
            "current_mellow_sha",
            "lr_before_optimizer_step",
            "resume_lr_before_optimizer_step",
            "next_optimizer_step == 21",
            "FORMAL schedule-total-steps",
            "FORMAL resume checkpoint is not a full canonical three-epoch run",
            "resume output-dir must be separate",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("FORMAL requires a fresh run from --model-path; resume-from is not allowed", text)
        smoke = INNER[0].read_text(encoding="utf-8")
        resume = INNER[1].read_text(encoding="utf-8")
        formal = INNER[2].read_text(encoding="utf-8")
        self.assertIn("--max-steps 20", smoke)
        self.assertIn("--schedule-total-steps 22", smoke)
        self.assertIn("/smoke20", smoke)
        self.assertIn("--max-steps 22", resume)
        self.assertIn("--schedule-total-steps 22", resume)
        self.assertIn("--expected-resume-step 20", resume)
        self.assertIn("/resume2_from20", resume)
        self.assertEqual(smoke.count("--schedule-total-steps 22"), resume.count("--schedule-total-steps 22"))
        self.assertNotIn("--schedule-total-steps", formal)
        self.assertIn("/formal", formal)
        self.assertIn("--save-every 500", formal)

    def test_short_scheduler_index_contract(self) -> None:
        # LambdaLR evaluates the factor at the current zero-based scheduler
        # index before each optimizer update.  With total=22 and warmup=2,
        # step 21 therefore uses index 20 and must still be above min_lr.
        total_steps = 22
        warmup_steps = 2

        def scale(step: int) -> float:
            if step < warmup_steps:
                return min(1.0, float(step + 1) / max(1, warmup_steps))
            progress = min(1.0, max(0.0, float(step + 1 - warmup_steps) / max(1, total_steps - warmup_steps)))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # optimizer step N observes scheduler index N-1.
        self.assertGreater(scale(20), 0.0)  # optimizer step 21
        self.assertGreater(scale(19), 0.0)  # optimizer step 20
        self.assertEqual(scale(21), 0.0)  # optimizer step 22 reaches min_lr

    def test_shell_environment_and_vc_submit_contract(self) -> None:
        for path in INNER:
            text = path.read_text(encoding="utf-8")
            self.assertIn("source \"$USER_CONDA_BASE/etc/profile.d/conda.sh\"", text)
            self.assertIn("conda activate rsmol", text)
        for path in WRAPPERS:
            text = path.read_text(encoding="utf-8")
            self.assertIn("vc submit", text)
            self.assertIn("docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1", text)
            self.assertIn("-p pdgpu-5090", text)
        self.assertIn("-g 1", WRAPPERS[-1].read_text(encoding="utf-8"))
        self.assertIn("-g 8", WRAPPERS[0].read_text(encoding="utf-8"))

    def test_audit_checks_real_reload_and_gradients(self) -> None:
        text = AUDIT.read_text(encoding="utf-8")
        for marker in (
            "_audit_saved_checkpoint",
            "_load_training_state",
            "forward_backward",
            "gradient_audit",
            "external_audio_provenance_verified",
            "expected-step",
            "expected-gate",
            "expected-parent-step",
            "parent-checkpoint",
            "parent_artifact",
            "resume_start_step",
            "parent_checkpoint_global_step",
            "parameter_change_audit",
            "_validate_parameter_change_audit",
            "_assert_parent_contract",
            "_canonical_contract_value",
            "_VOLATILE_TEXT_CONFIG_KEYS",
            "parent/child immutable contract mismatch",
            "parent/child gate/run_kind mismatch",
            "parent_contract_verified",
            "saved text model source differs from --model-path",
            "saved validation manifest path differs from --val-manifest",
            "checkpoint gate/run_kind mismatch",
            "fresh FORMAL checkpoint has invalid resume lineage",
            "torch.autocast",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("write_routers", text)

    def test_separate_smoke_resume_output_contract_is_documented(self) -> None:
        text = (PKG / "README.md").read_text(encoding="utf-8")
        for marker in (
            "SMOKE_OUT=",
            "RESUME_OUT=",
            "FORMAL_OUT=",
            "FORMAL_RESUME_OUT=",
            "--resume-from",
            "--output-dir",
            "--parent-checkpoint",
            "--model-path",
            "--manifest",
            "--val-manifest",
            "resume2_from20",
            "checkpoint-000500",
            "FORMAL checkpoint",
            '--resume-from "$FORMAL_OUT/checkpoint-000500"',
            "--expected-gate FORMAL",
            "--expected-step 500",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("same output directory", text.lower())


if __name__ == "__main__":
    unittest.main()
