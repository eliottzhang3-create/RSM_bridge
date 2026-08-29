"""Dependency-light Stage 4 contract checks.

These checks intentionally avoid importing torch/Transformers so they can run
on the Windows development checkout as well as on the remote environment.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import types
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "code" / "RSmol" / "scripts" / "train_stage4_ddp.py"
AUDIT = ROOT / "code" / "RSmol" / "scripts" / "audit_stage4_dataset.py"
RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "train_stage4_ddp.sh"
SUBMIT = ROOT / "code" / "RSmol" / "run_stage4_5090.sh"


def load_stage4():
    spec = importlib.util.spec_from_file_location("stage4_static", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_gate_b():
    spec = importlib.util.spec_from_file_location("gate_b_static", AUDIT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {AUDIT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage4StaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stage4 = load_stage4()
        cls.gate_b = load_gate_b()
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.audit_source = AUDIT.read_text(encoding="utf-8")
        cls.runtime = RUNTIME.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_gate_b_has_independent_cpu_entrypoint(self) -> None:
        self.assertTrue(AUDIT.is_file())
        for marker in (
            "--model-path",
            "--tokenizer-path",
            "--data-dir",
            "--output-dir",
            "--report-path",
            "--context-length",
            "--world-size",
            "--seed",
            "--sample-shards",
            "--progress-every-rows",
            "--cache-root",
            "discover_parquet_files",
            "audit_parquet_shards",
            "assign_shards",
            "select_sample_shards",
            "rank_valid_trainable_rows",
            "formal_global_samples",
            "formal_remaining_raw_rows",
        ):
            self.assertIn(marker, self.audit_source)
        for forbidden in ("torchrun", "torch.distributed", "_prepare_distributed", "AutoModelForCausalLM"):
            self.assertNotIn(forbidden, self.audit_source)

    def test_fixed_defaults_and_shards(self) -> None:
        self.assertEqual(len(self.stage4.expected_parquet_names()), 85)
        self.assertEqual(self.stage4.DEFAULT_TARGET_SAMPLES_PER_RANK, 1_280)
        self.assertEqual(self.stage4.DEFAULT_MAX_OPTIMIZER_STEPS, 10)
        self.assertEqual(self.stage4.DEFAULT_FORMAL_OPTIMIZER_STEPS, 9_244)
        self.assertEqual(self.stage4.DEFAULT_FORMAL_SAMPLES_PER_RANK, 1_183_232)
        self.assertEqual(self.stage4.DEFAULT_FORMAL_LOCAL_MICROBATCHES, 147_904)
        self.assertEqual(self.stage4.DEFAULT_WORLD_SIZE, 8)
        files = [Path(f"train-{index:05d}-of-00085.parquet") for index in range(85)]
        self.assertEqual(
            [path.name for path in self.stage4.select_sample_shards(files, sample_shards=3, seed=0)],
            [path.name for path in self.stage4.select_sample_shards(files, sample_shards=3, seed=0)],
        )
        self.assertEqual(len(self.stage4.select_sample_shards(files, sample_shards=3, seed=0)), 3)
        assignment = self.stage4.assign_shards(
            [Path(f"train-{index:05d}-of-00085.parquet") for index in range(85)],
            world_size=8,
            seed=0,
        )
        self.assertEqual(assignment["rank_shard_counts"], {str(i): (11 if i < 5 else 10) for i in range(8)})

    def test_gate_d_default_and_scheduler_contract(self) -> None:
        config = self.stage4._parse_args(["--gate", "D", "--dry-run"])
        self.assertEqual(config.max_optimizer_steps, 10)
        self.assertEqual(config.scheduler_total_steps, 10)
        self.assertEqual(config.warmup_steps, 1)
        self.assertEqual(config.max_lr, 2e-4)
        self.assertEqual(config.min_lr, 2e-5)
        self.assertEqual(config.log_interval_steps, 10)
        default_config = json.loads(
            (ROOT / "code" / "RSmol" / "stage4_default_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(default_config["max_optimizer_steps"], 10)
        self.assertEqual(default_config["formal_optimizer_steps"], 9244)
        self.assertEqual(default_config["scheduler_type"], "linear_warmup_cosine")
        self.assertEqual(default_config["log_interval_steps"], 10)
        with self.assertRaisesRegex(ValueError, "fixed ten-optimizer-step"):
            self.stage4._parse_args(
                ["--gate", "D", "--max-optimizer-steps", "9244", "--dry-run"]
            )
        with self.assertRaisesRegex(ValueError, "fixed ten-optimizer-step"):
            self.stage4._parse_args(
                ["--gate", "D", "--max-optimizer-steps", "9", "--dry-run"]
            )
        self.assertEqual(
            self.stage4._parse_args(["--gate", "C", "--dry-run"]).max_optimizer_steps,
            2,
        )
        self.assertEqual(
            self.stage4._parse_args(
                ["--gate", "C", "--max-optimizer-steps", "10", "--dry-run"]
            ).max_optimizer_steps,
            10,
        )

    def test_formal_parser_defaults_and_conflicts(self) -> None:
        config = self.stage4._parse_args(["--gate", "formal", "--dry-run"])
        self.assertEqual(config.gate, "FORMAL")
        self.assertEqual(config.max_optimizer_steps, 9244)
        self.assertEqual(config.formal_optimizer_steps, 9244)
        self.assertEqual(config.scheduler_total_steps, 9244)
        self.assertEqual(config.warmup_steps, 463)
        self.assertEqual(config.save_every, 500)
        self.assertEqual(config.checkpoint_retention, 3)
        default_config = json.loads(
            (ROOT / "code" / "RSmol" / "stage4_default_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(default_config["formal_gate"], "FORMAL")
        self.assertEqual(default_config["formal_scheduler_total_steps"], 9244)
        self.assertEqual(default_config["formal_warmup_steps"], 463)
        self.assertEqual(default_config["formal_save_every"], 500)
        self.assertEqual(default_config["checkpoint_retention"], 3)
        self.assertEqual(default_config["formal"]["optimizer_steps"], 9244)
        self.assertEqual(default_config["formal"]["scheduler_total_steps"], 9244)
        self.assertEqual(default_config["formal"]["warmup_steps"], 463)
        self.assertEqual(default_config["formal"]["save_every"], 500)
        self.assertEqual(default_config["formal"]["checkpoint_retention"], 3)
        self.assertEqual(default_config["optimizer"]["type"], "AdamW")
        self.assertEqual(default_config["optimizer"]["betas"], [0.9, 0.95])
        self.assertEqual(default_config["optimizer"]["eps"], 1e-8)
        self.assertEqual(default_config["optimizer"]["weight_decay"], 0.1)
        self.assertFalse(default_config["optimizer"]["amsgrad"])
        self.assertTrue(default_config["formal"]["resume_supported"])
        self.assertFalse(default_config["formal"]["bounded_shuffle_bitwise_exact_resume"])
        self.assertEqual(self.stage4.formal_save_steps(), [*range(500, 9001, 500), 9244])
        self.assertAlmostEqual(self.stage4.cosine_warmup_factor(0, 9244, 463), 0.1, places=8)
        self.assertAlmostEqual(self.stage4.cosine_warmup_factor(463, 9244, 463), 1.0, places=8)
        self.assertAlmostEqual(self.stage4.cosine_warmup_factor(9244, 9244, 463), 0.1, places=8)
        self.assertAlmostEqual(
            self.stage4.token_weighted_gradient_scale(
                world_size=8, global_window_tokens=16 * 1024
            ),
            8.0 / (16 * 1024),
            places=12,
        )
        self.assertNotAlmostEqual(
            self.stage4.token_weighted_gradient_scale(
                world_size=8, global_window_tokens=16 * 1024
            ),
            8.0 / (16 * 16 * 1024),
            places=12,
        )
        self.assertNotIn(
            "config.gradient_accumulation_steps * global_window_tokens",
            self.source,
        )
        self.assertNotIn(
            "config.gradient_accumulation_steps * window_global_tokens",
            self.source,
        )
        self.assertGreaterEqual(self.source.count("token_weighted_gradient_scale("), 2)
        with self.assertRaisesRegex(ValueError, "FORMAL requires max_optimizer_steps=9244"):
            self.stage4._parse_args(
                ["--gate", "FORMAL", "--max-optimizer-steps", "10", "--dry-run"]
            )
        with self.assertRaisesRegex(ValueError, "FORMAL requires max_optimizer_steps=9244"):
            self.stage4._parse_args(
                ["--gate=FORMAL", "--max-steps=10", "--dry-run"]
            )
        with self.assertRaisesRegex(ValueError, "FORMAL requires scheduler_total_steps=9244"):
            self.stage4._parse_args(
                ["--gate", "FORMAL", "--scheduler-total-steps", "10", "--dry-run"]
            )
        with self.assertRaisesRegex(ValueError, "FORMAL requires backend=nccl"):
            self.stage4._parse_args(
                ["--gate", "FORMAL", "--backend", "gloo", "--dry-run"]
            )

    def test_formal_retention_only_removes_verified_step_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in (500, 1000, 1500, 2000):
                checkpoint = root / f"checkpoint-step-{step:06d}"
                checkpoint.mkdir()
                (checkpoint / "checkpoint_complete.json").write_text(
                    json.dumps({"complete": True, "optimizer_step": step}) + "\n",
                    encoding="utf-8",
                )
            (root / "checkpoint-step-999999.tmp").mkdir()
            (root / "checkpoint-step-000001").mkdir()
            (root / "checkpoint-step-000002").mkdir()
            (root / "latest_complete_checkpoint.json").write_text("{}\n", encoding="utf-8")
            retained = self.stage4.retain_formal_checkpoints(root, keep=3)
            self.assertEqual([path.name for path in retained], [
                "checkpoint-step-001000", "checkpoint-step-001500", "checkpoint-step-002000"
            ])
            self.assertFalse((root / "checkpoint-step-000500").exists())
            self.assertTrue((root / "checkpoint-step-999999.tmp").exists())
            self.assertTrue((root / "checkpoint-step-000001").exists())
            pointer = json.loads((root / "latest_complete_checkpoint.json").read_text(encoding="utf-8"))
            self.assertTrue(pointer["complete"])
            self.assertEqual(pointer["optimizer_step"], 2000)

    def test_formal_resume_parser_and_metadata_contract(self) -> None:
        config = self.stage4._parse_args(
            ["--gate", "FORMAL", "--resume-from", "/tmp/formal-checkpoint", "--dry-run"]
        )
        self.assertEqual(config.gate, "FORMAL")
        self.assertEqual(config.resume_from, Path("/tmp/formal-checkpoint"))
        self.assertEqual(config.max_optimizer_steps, 9244)
        state = {
            "configuration": {"gate": "FORMAL"},
            "scheduler_config": {"total_steps_for_schedule": 9244, "warmup_steps": 463},
            "optimizer_step": 9000,
        }
        self.assertEqual(
            self.stage4.validate_formal_resume_state(state),
            {"optimizer_step": 9000, "scheduler_total_steps": 9244, "warmup_steps": 463},
        )
        with self.assertRaisesRegex(ValueError, "checkpoint gate=D"):
            invalid_mode = dict(state)
            invalid_mode["configuration"] = {"gate": "D"}
            self.stage4.validate_formal_resume_state(invalid_mode)
        complete_state = dict(state)
        complete_state["optimizer_step"] = 9244
        self.assertEqual(
            self.stage4.validate_formal_resume_state(complete_state)["optimizer_step"],
            9244,
        )
        with self.assertRaisesRegex(ValueError, "scheduler domain must be 9244"):
            invalid = dict(state)
            invalid["scheduler_config"] = {"total_steps_for_schedule": 10, "warmup_steps": 1}
            self.stage4.validate_formal_resume_state(invalid)
        with self.assertRaisesRegex(ValueError, "created by FORMAL mode"):
            invalid = dict(state)
            invalid["configuration"] = {"gate": "D"}
            self.stage4.validate_formal_resume_state(invalid)

    def test_warmup_ceil_and_cosine_boundaries(self) -> None:
        self.assertEqual(self.stage4.compute_warmup_steps(10), 1)
        self.assertEqual(self.stage4.compute_warmup_steps(21), 2)
        self.assertEqual(self.stage4.compute_warmup_steps(1), 1)
        values = [
            self.stage4.cosine_warmup_factor(step, 10, 1)
            for step in range(11)
        ]
        self.assertAlmostEqual(values[0], 0.1, places=8)
        self.assertAlmostEqual(values[1], 1.0, places=8)
        self.assertAlmostEqual(values[-1], 0.1, places=8)
        self.assertTrue(all(values[index] >= values[index + 1] for index in range(1, 10)))

    def test_progress_report_and_resume_scheduler_continuity_contract(self) -> None:
        for marker in (
            "stage4-progress",
            "stage4_progress.jsonl",
            "progress_logging",
            "log_interval_steps",
            "total_steps_for_schedule",
            "last_samples_per_second",
            "scheduler_config",
            "scheduler_type",
            "linear_warmup_cosine",
        ):
            self.assertIn(marker, self.source)
        old_schedule = [
            self.stage4.cosine_warmup_factor(step, 10, 1)
            for step in range(13)
        ]
        # Gate E extends the smoke target by two steps but restores the
        # checkpoint's original ten-step schedule, so LR remains continuous
        # at the old final boundary rather than restarting warmup.
        self.assertAlmostEqual(old_schedule[10], old_schedule[11], places=8)
        self.assertAlmostEqual(old_schedule[11], old_schedule[12], places=8)

    def test_external_output_guard_and_markers(self) -> None:
        with self.assertRaises(ValueError):
            self.stage4.ensure_external_output(ROOT / "outputs" / "stage4")
        with tempfile.TemporaryDirectory() as temporary:
            self.assertTrue(self.stage4.ensure_external_output(Path(temporary) / "safe").is_absolute())
            incomplete = Path(temporary) / "checkpoint"
            incomplete.mkdir()
            with self.assertRaises(FileNotFoundError):
                self.stage4.ensure_external_resume(incomplete)
            (incomplete / "config.json").write_text("{}\n", encoding="utf-8")
            (incomplete / "training_state.pt").write_bytes(b"placeholder")
            (incomplete / "model.safetensors").write_bytes(b"placeholder")
            (incomplete / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            (incomplete / "checkpoint_complete.json").write_text(
                '{"complete": true}\n', encoding="utf-8"
            )
            self.assertEqual(self.stage4.ensure_external_resume(incomplete), incomplete.resolve())
        for marker in (
            "ParquetFile.iter_batches",
            "content_audit",
            "content_paths",
            "content_is_full_corpus",
            "sampled_shards",
            "progress_callback",
            "global_loss_sum",
            "global_valid_token_count",
            "world_size",
            "gradient_accumulation_steps",
            "all_reduce_min_flag",
            "partial_accumulation_window_discarded",
            "save_complete_checkpoint",
            "checkpoint_complete.json",
            "bounded_shuffle_bitwise_exact",
            "coarse_cursor_skip_applied",
            "data_cursor_restored",
            "coarse_epoch_shard_complete_microbatch_skip",
            "stop_reason",
            "steps_at_stop",
            "final_checkpoint",
            "final_checkpoint_saved_after_coordinated_stop",
            "Gate E resume did not apply a coarse data cursor skip",
            "all_gather_object",
            "data_cursors_by_rank",
            "checkpoint_contract",
            "logical_30_physical_15_loops_2",
            "checkpoint_complete.json world_size mismatch",
            "checkpoint step mismatch between checkpoint_complete.json and training_state.pt",
            "FORMAL resume is not implemented",
            "retain_formal_checkpoints",
            "formal_save_steps",
            "token_weighted_gradient_scale",
            "betas=DEFAULT_ADAMW_BETAS",
            "eps=DEFAULT_ADAMW_EPS",
            "amsgrad=DEFAULT_ADAMW_AMSGRAD",
            "Stage 4 requires AdamW weight_decay",
            "Stage 4 requires AdamW, got",
            "validate_formal_resume_state",
            "formal_resume",
            "torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16)",
            "labels = input_ids.clone()",
            "shift_logits = logits[..., :-1, :]",
            "shift_labels = labels[..., 1:].contiguous()",
        ):
            self.assertIn(marker, self.source)
        for marker in (
            "sample_shards",
            "sample_statistics_are_not_full_corpus_estimates",
            "progress_every_rows",
            "STAGE4_GATE_B_CACHE_ROOT",
            "cache_root",
            "actual_arrow_files",
            "CPU-only",
            "full_corpus_tokenizer_scan",
        ):
            self.assertIn(marker, self.audit_source)

    def test_remote_wrappers_use_eight_gpu_contract(self) -> None:
        self.assertIn("torchrun --standalone", self.runtime)
        self.assertIn("Gate B", self.runtime)
        self.assertIn("vc submit", self.submit)
        self.assertIn('if [[ "$GATE" == "B" ]]', self.submit)
        self.assertIn('Gate B is CPU-only', self.submit)
        self.assertNotIn('GPU_COUNT=1', self.submit)
        self.assertIn('-c 32 -m 256G -g 8 -n 1', self.submit)
        self.assertIn("pdgpu-5090", self.submit)
        self.assertIn('RESUME_PATH="${RSMOL_STAGE4_RESUME_FROM:-}"', self.runtime)
        self.assertIn('if [[ "$GATE" == "E"', self.runtime)
        self.assertIn("checkpoint_complete.json", self.runtime)
        self.assertIn("Gate E requires RSMOL_STAGE4_RESUME_FROM", self.runtime)
        self.assertIn("training_state.pt", self.runtime)
        self.assertIn("Gate B is CPU-only", self.runtime)
        self.assertIn('if [[ "$GATE" == "FORMAL" ]]', self.runtime)
        self.assertIn('("$GATE" == "E" || "$GATE" == "FORMAL")', self.runtime)
        self.assertIn('&& -n "$RESUME_PATH"', self.runtime)
        self.assertIn('elif [[ "$GATE" != "A"', self.runtime)
        self.assertIn("RSMOL_STAGE4_CHECKPOINT_RETENTION", self.runtime)
        self.assertIn("RSMOL_STAGE4_CHECKPOINT_RETENTION", self.submit)

    def test_resume_parser_and_cursor_contract(self) -> None:
        config = self.stage4._parse_args(
            ["--gate", "E", "--resume-from", "/tmp/external-checkpoint", "--dry-run"]
        )
        self.assertEqual(config.gate, "E")
        self.assertEqual(config.resume_from, Path("/tmp/external-checkpoint"))
        self.assertIn("restore_cursor", self.source)
        self.assertIn("shard_microbatches_seen", self.source)
        self.assertIn("partial_accumulation_window_discarded", self.source)
        self.assertIn("Legacy single data_cursor checkpoints are supported only for world_size=1", self.source)
        self.assertIn("Gate E resume rejected: checkpoint contains legacy single data_cursor", self.source)
        save_parameters = inspect.signature(self.stage4.save_complete_checkpoint).parameters
        self.assertIn("data_cursors_by_rank", save_parameters)
        self.assertNotIn("data_cursor", save_parameters)

    def test_gate_b_is_explicitly_single_process(self) -> None:
        self.assertIn('if config.gate == "B" and world_size != 1:', self.source)
        self.assertIn('if config.gate != "B" and world_size != DEFAULT_WORLD_SIZE', self.source)
        self.assertIn('world_size = 1', self.source)

    def test_gate_b_default_sample_and_bounded_stats(self) -> None:
        config = self.gate_b._parse_args(
            ["--tokenizer-path", "/tmp/tokenizer", "--output-dir", "/tmp/gate-b-test"]
        )
        self.assertEqual(config.sample_shards, 3)
        self.assertEqual(config.progress_every_rows, 10000)
        stats = self.stage4._LengthStats(reservoir_size=2, seed=0)
        for value in (1, 2, 100):
            stats.add(value)
        summary = stats.as_dict()
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["min"], 1)
        self.assertEqual(summary["max"], 100)
        self.assertIn("mean", summary)
        self.assertIn("p99", summary["quantiles"])
        self.assertEqual(summary["reservoir_size"], 2)

    def test_gate_b_streams_only_selected_shards_and_reports_progress(self) -> None:
        class Field:
            def __init__(self, name, field_type="string"):
                self.name = name
                self.type = field_type
                self.nullable = True

        class Column:
            def __init__(self, values):
                self.values = values

            def to_pylist(self):
                return self.values

        class Batch:
            def __init__(self, texts, sources):
                self.columns = {"text": Column(texts), "source": Column(sources)}

            def column(self, name):
                return self.columns[name]

        class FakeParquetFile:
            opened = []

            def __init__(self, path):
                self.path = Path(path)
                self.schema_arrow = [Field("text"), Field("source")]
                self.metadata = types.SimpleNamespace(num_rows=3)
                self.num_row_groups = 1
                self.opened.append(self.path.name)

            def iter_batches(self, **kwargs):
                if self.path.name == "train-00000-of-00085.parquet":
                    yield Batch(["short", "", "x" * 20], ["wiki", None, "book"])
                elif self.path.name == "train-00001-of-00085.parquet":
                    yield Batch([None, "ok", "long"], ["code", "", "code"])
                elif self.path.name == "train-00002-of-00085.parquet":
                    yield Batch(["tiny", "tiny", "tiny"], ["news", "news", "news"])

        parquet_module = types.ModuleType("pyarrow.parquet")
        parquet_module.ParquetFile = FakeParquetFile
        pyarrow_module = types.ModuleType("pyarrow")
        pyarrow_module.__path__ = []
        pyarrow_module.parquet = parquet_module
        old_pyarrow = sys.modules.get("pyarrow")
        old_parquet = sys.modules.get("pyarrow.parquet")
        sys.modules["pyarrow"] = pyarrow_module
        sys.modules["pyarrow.parquet"] = parquet_module
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                files = []
                for index in range(85):
                    path = root / f"train-{index:05d}-of-00085.parquet"
                    path.write_bytes(b"fixture")
                    files.append(path)

                class FakeTokenizer:
                    def __call__(self, text, **kwargs):
                        # Each character is a token, making x*20 over context=10.
                        return {"input_ids": list(range(len(text)))}

                selected = files[:3]
                progress = []
                report = self.stage4.audit_parquet_shards(
                    files,
                    tokenizer=FakeTokenizer(),
                    context_length=10,
                    content=True,
                    content_paths=selected,
                    progress_callback=progress.append,
                    progress_every_rows=2,
                )
                self.assertEqual(report["file_count"], 85)
                self.assertEqual(report["content_scope"], "sampled_shards")
                self.assertFalse(report["content_is_full_corpus"])
                self.assertEqual(report["sampled_shard_count"], 3)
                self.assertEqual(len(FakeParquetFile.opened), 85)
                self.assertTrue(any("footer 85/85" in item for item in progress))
                self.assertTrue(any("read_rows=2/3" in item for item in progress))
                self.assertTrue(any("tokenized_rows=" in item for item in progress))
                self.assertEqual(report["content"]["none_text"], 1)
                self.assertEqual(report["content"]["empty_text"], 1)
                self.assertEqual(report["content"]["over_context_length"], 1)
                self.assertEqual(report["content"]["over_context_ratio_denominator"], "tokenized_rows")
                self.assertEqual(report["content"]["token_length"]["count"], 7)
                self.assertFalse(report["shards"][3]["content"]["audited"])
        finally:
            if old_pyarrow is None:
                sys.modules.pop("pyarrow", None)
            else:
                sys.modules["pyarrow"] = old_pyarrow
            if old_parquet is None:
                sys.modules.pop("pyarrow.parquet", None)
            else:
                sys.modules["pyarrow.parquet"] = old_parquet

    def test_gate_b_cache_root_is_local_and_does_not_create_arrow_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            config = self.gate_b.AuditConfig(cache_root=root)
            audit = self.gate_b._configure_local_cache(config)
            self.assertTrue(audit["safe"])
            self.assertEqual(audit["actual_arrow_files"], [])
            self.assertEqual(Path(audit["environment"]["HF_DATASETS_CACHE"]).parent, root)
            with self.assertRaises(ValueError):
                self.gate_b._configure_local_cache(
                    self.gate_b.AuditConfig(cache_root=Path("/hpc_stor03/shared-cache"))
                )

    def test_local_pretrained_paths_are_validated_before_hf_loading(self) -> None:
        self.assertIn("_require_local_artifact_dir", self.source)
        self.assertIn("is not an existing local directory", self.source)
        self.assertIn("HFValidationError", self.source)

    def test_final_cursor_gather_is_outside_rank0_branch(self) -> None:
        tree = ast.parse(self.source)
        run_training = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_training"
        )
        for node in ast.walk(run_training):
            if not isinstance(node, ast.If):
                continue
            test = ast.unparse(node.test)
            if "rank == 0" not in test:
                continue
            nested_calls = [
                child for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "_gather_rank_cursors"
            ]
            self.assertFalse(
                nested_calls,
                "_gather_rank_cursors is collective and must not be nested under if rank == 0",
            )
        final_anchor = self.source.index("if coordinated_stop and optimizer_step > start_step:")
        final_block = self.source[final_anchor : self.source.index("finally:", final_anchor)]
        self.assertLess(final_block.index("_gather_rank_cursors"), final_block.index("if rank == 0:"))


if __name__ == "__main__":
    unittest.main()
