"""Dependency-light Stage 4 contract checks.

These checks intentionally avoid importing torch/Transformers so they can run
on the Windows development checkout as well as on the remote environment.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "code" / "RSmol" / "scripts" / "train_stage4_ddp.py"
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


class Stage4StaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stage4 = load_stage4()
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.runtime = RUNTIME.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_fixed_defaults_and_shards(self) -> None:
        self.assertEqual(len(self.stage4.expected_parquet_names()), 85)
        self.assertEqual(self.stage4.DEFAULT_TARGET_SAMPLES_PER_RANK, 1_183_232)
        self.assertEqual(self.stage4.DEFAULT_MAX_OPTIMIZER_STEPS, 9_244)
        self.assertEqual(self.stage4.DEFAULT_WORLD_SIZE, 8)
        assignment = self.stage4.assign_shards(
            [Path(f"train-{index:05d}-of-00085.parquet") for index in range(85)],
            world_size=8,
            seed=0,
        )
        self.assertEqual(assignment["rank_shard_counts"], {str(i): (11 if i < 5 else 10) for i in range(8)})

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
            "torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16)",
            "labels = input_ids.clone()",
            "shift_logits = logits[..., :-1, :]",
            "shift_labels = labels[..., 1:].contiguous()",
        ):
            self.assertIn(marker, self.source)

    def test_remote_wrappers_use_eight_gpu_contract(self) -> None:
        self.assertIn("torchrun --standalone", self.runtime)
        self.assertIn("Gate B", self.runtime)
        self.assertIn("vc submit", self.submit)
        self.assertIn("-c 32 -m 256G -g 8 -n 1", self.submit)
        self.assertIn("pdgpu-5090", self.submit)
        self.assertIn('RESUME_PATH="${RSMOL_STAGE4_RESUME_FROM:-}"', self.runtime)
        self.assertIn('if [[ "$GATE" == "E"', self.runtime)
        self.assertIn("checkpoint_complete.json", self.runtime)
        self.assertIn("Gate E requires RSMOL_STAGE4_RESUME_FROM", self.runtime)
        self.assertIn("training_state.pt", self.runtime)

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
