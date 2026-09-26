"""Static contracts for the isolated Mellow-faithful SmolLM2 route."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code" / "RSmol"
SCRIPTS = RSMOL / "scripts"
LEGACY = RSMOL / "audio_smollm2_135m_mellow_shared_store"
PACKAGE = RSMOL / "audio_smollm2_135m_mellow_shared_store_configurable_epochs"
TRAIN = SCRIPTS / "train_audio_smollm2_shared_store_mellow_faithful_135m_ddp.py"
ENTRY = SCRIPTS / "train_audio_smollm2_shared_store_configurable_epochs_135m_mellow_ddp.py"
STORE = SCRIPTS / "prepare_mellow_faithful_unique_audio_waveform_store.py"
STAGE = SCRIPTS / "stage_audio_smollm2_shared_store_configurable_epochs_135m_mellow.sh"
SMOKE = SCRIPTS / "train_audio_smollm2_shared_store_smoke20_configurable_epochs_135m_mellow_ddp.sh"
REFERENCE = SCRIPTS / "train_audio_smollm2_shared_store_reference22_configurable_epochs_135m_mellow_ddp.sh"
RESUME = SCRIPTS / "train_audio_smollm2_shared_store_resume2_configurable_epochs_135m_mellow_ddp.sh"
FORMAL = SCRIPTS / "train_audio_smollm2_shared_store_formal_configurable_epochs_135m_mellow_ddp.sh"
SUBMITS = (
    RSMOL / "run_audio_smollm2_shared_store_smoke20_configurable_epochs_135m_mellow_3090.sh",
    RSMOL / "run_audio_smollm2_shared_store_reference22_configurable_epochs_135m_mellow_3090.sh",
    RSMOL / "run_audio_smollm2_shared_store_resume2_configurable_epochs_135m_mellow_3090.sh",
    RSMOL / "run_audio_smollm2_shared_store_formal_configurable_epochs_135m_mellow_3090.sh",
)


class MellowFaithfulSmolLM2StaticTest(unittest.TestCase):
    def test_files_exist_and_python_parses(self) -> None:
        python_files = (
            PACKAGE / "__init__.py",
            PACKAGE / "data.py",
            PACKAGE / "model.py",
            PACKAGE / "sampler.py",
            PACKAGE / "mellow_templates.py",
            TRAIN,
            ENTRY,
            STORE,
        )
        for path in (
            *python_files,
            PACKAGE / "README.md",
            STAGE,
            SMOKE,
            REFERENCE,
            RESUME,
            FORMAL,
            *SUBMITS,
        ):
            self.assertTrue(path.is_file(), path)
        for path in python_files:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_legacy_shared_store_surface_is_unchanged(self) -> None:
        init = (LEGACY / "__init__.py").read_text(encoding="utf-8")
        data = (LEGACY / "data.py").read_text(encoding="utf-8")
        model = (LEGACY / "model.py").read_text(encoding="utf-8")
        self.assertIn("smollm2_node_shared_unique_store_fullshuffle_fixed260_audio_reuse_answer_eos_v2", init)
        self.assertIn("audio_5_10x2_5_mesh_mellow_shared_store.data", data)
        self.assertIn("AudioSmolLM2Model", model)
        self.assertNotIn("mellow_faithful", init + data + model)
        self.assertNotIn("shared_store_configurable_epochs", init + data + model)

    def test_isolated_contract_and_official_reference(self) -> None:
        package = "\n".join(
            (PACKAGE / name).read_text(encoding="utf-8")
            for name in ("__init__.py", "data.py", "model.py", "sampler.py")
        )
        self.assertIn("c8204d8eb99b4384fd7a76ad57995731e0c0c2bf", package)
        self.assertIn("4d3722d904734c7b2ae1b55309002c82bf1d11bc", package)
        self.assertIn("mellow_faithful_variable_waveform_store_v2", package)
        self.assertIn("gbs32_exact_resume_v3", package)
        self.assertNotIn("audio_5_10x2_5_mesh", package)

    def test_stateful_mellow_data_call_order(self) -> None:
        text = (PACKAGE / "data.py").read_text(encoding="utf-8")
        for marker in (
            "pool_paths = sorted({",
            "random.choice(self.random_audio_pool_ids)",
            "random.randint(0, samples - SEGMENT_SAMPLES)",
            "audio1, offset1 = self._crop_or_pad(full1)",
            "audio2, offset2 = self._crop_or_pad(full2)",
            "answer, prompt, template = self._answer_prompt(row)",
            "PROMPT_TOKENS = 129",
            "ANSWER_TOKENS = 250",
            "value + \" <|endoftext|>\"",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("sha256(f\"{self.seed}", text)
        self.assertNotIn("random_audio_pool.json", text)

    def test_variable_store_is_strict_torchaudio_and_full_length(self) -> None:
        text = STORE.read_text(encoding="utf-8")
        for marker in (
            "torchaudio.load(str(path), channels_first=True)",
            "torchaudio.functional.resample",
            'getattr(torchaudio, "info", None)',
            "from torchcodec.decoders import AudioDecoder",
            "decoder.metadata",
            "duration_seconds",
            "soundfile.info(str(path))",
            "soundfile.read(",
            "decode_audio_file(path)",
            "TorchAudio cannot decode because TorchCodec is missing",
            "byte_offset",
            "byte_length",
            "num_samples",
            "estimated_total_waveform_bytes",
            "BUILDING",
            "--resume",
            "--dry-run",
            "waveform_verification",
            "resume waveform file is shorter than durable progress",
            "if not partial.exists() and final.is_file():",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("torchaudio.info(", text)
        for forbidden in ("wave.open", "F.interpolate", "SEGMENT_SAMPLES"):
            self.assertNotIn(forbidden, text)

    def test_sampler_matches_confirmed_official_process(self) -> None:
        text = (PACKAGE / "sampler.py").read_text(encoding="utf-8")
        for marker in (
            "generator.manual_seed(self.epoch)",
            "torch.randperm(self.dataset_size, generator=generator)",
            "rank_start = self.rank * self.samples_per_rank",
            "rank_indices[self.start_sample:]",
            "start_optimizer_step",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("self.rank::self.num_replicas", text)

    def test_model_uses_fixed_639_layout_and_independent_audio_passes(self) -> None:
        model = (PACKAGE / "model.py").read_text(encoding="utf-8")
        for marker in (
            "MELLOW_SEQUENCE_TOKENS",
            "prefix1, prefix2 = self.encode_audio(audio1, audio2, audio2_reused_mask=None)",
            "output.logits[:, MELLOW_TEXT_PREFIX_TOKENS - 1:-1]",
            "ignore_index=pad_id",
            "inputs_embeds=inputs_embeds",
        ):
            self.assertIn(marker, model)
        self.assertNotIn("attention_mask=", model)

    def test_optimizer_batch_scheduler_and_checkpoint_contract(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        for marker in (
            "CANONICAL_MICRO_BATCH = 4",
            "CANONICAL_GRAD_ACCUM = 1",
            "CANONICAL_GLOBAL_BATCH = 32",
            "CANONICAL_CHECKPOINT_RETENTION = 3",
            "torch.optim.Adam(",
            "CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)",
            "output.loss / args.gradient_accumulation_steps",
            "ddp.no_sync()",
            "clip_grad_norm_(ddp.parameters(), 0.5",
            "prune_formal_checkpoints",
            "resume_comparison_trace",
            "training_state_fingerprint",
            "compare_reference22",
            "route_code_identity",
            "if rank != 0:",
            "mellow_faithful_full_waveforms_32k_f32_v2",
            "scheduler.step()",
            "shape=shape",
            "inventory=inventory",
        ):
            self.assertIn(marker, text)
        self.assertIn('"global_batch_size": global_batch', text)
        self.assertIn('"warmup_steps": 0', text)
        self.assertNotIn("AdamW", text)
        self.assertNotIn("autocast(", text)

    def test_shell_and_submission_contracts(self) -> None:
        stage = STAGE.read_text(encoding="utf-8")
        for marker in (
            "mellow_faithful_variable_waveform_store_v2",
            "/dev/shm/rsmol_smollm2_mellow_faithful_",
            "SHM_MARGIN_KIB=$((10 * 1024 * 1024))",
            "--nproc_per_node=8",
        ):
            self.assertIn(marker, stage)
        for path in (SMOKE, REFERENCE, RESUME, FORMAL):
            text = path.read_text(encoding="utf-8")
            for marker in (
                "--epochs 30",
                "--micro-batch-size 4",
                "--gradient-accumulation-steps 1",
                "--checkpoint-retention 3",
            ):
                self.assertIn(marker, text)
        self.assertIn("--reference22-report", RESUME.read_text(encoding="utf-8"))
        formal = FORMAL.read_text(encoding="utf-8")
        for marker in ("--smoke20-report", "--reference22-report", "--smoke-resume-report"):
            self.assertIn(marker, formal)
        for path in SUBMITS:
            text = path.read_text(encoding="utf-8")
            for marker in ("vc submit", "-p pdgpu-3090", "-c 32", "-m 256G", "-g 8", "-n 1"):
                self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()
