"""CPU/stdlib checks for the isolated Mizar S1 route; no local datasets/weights."""
import ast
import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_3epochs"
sys.path.insert(0, str(ROUTE))
import contract
import stage_data
from data.indexed_jsonl import IndexedRows
from write_config import validate_smoke


def passed_smoke():
    gradients = {role: {"finite": True, "nonzero_parameters": 2} for role in ("c2l", "projection", "lm")}
    return {"status": "PASS", "mode": "smoke", "optimizer_steps": 8, "contract_sha256": "recipe-a",
            "rank_checks": [{"rank": rank, "seed": 1234 + rank, "optimizer_steps": 8,
                             "first8_rows": list(range(rank * 32, (rank + 1) * 32)),
                             "frozen_htsat_equal": True, "parameters_finite": True, "htsat_all_eval": True,
                             "gradients": {"1": copy.deepcopy(gradients), "8": copy.deepcopy(gradients)},
                             "trainable_weight_changes": {role: {"changed": True} for role in gradients}}
                            for rank in range(8)],
            "checkpoint": {"status": "PASS", "optimizer_steps": 8, "model_tensors_equal": True, "rng_ranks": 8},
            "layout": {"prefix": 515, "total": 815}}


class MizarS1RouteTests(unittest.TestCase):
    def test_rank_rng_and_data_order_are_independent(self):
        self.assertEqual([contract.rank_seed(rank) for rank in range(8)], list(range(1234, 1242)))
        before = random.getstate()
        expected = list(range(64))
        random.Random(1234).shuffle(expected)
        self.assertEqual(contract.data_order(64), expected)
        self.assertEqual(random.getstate(), before)

    def test_filtered_geometry_and_official_schedule_boundaries(self):
        geometry = contract.geometry(sum(contract.SOURCE_COUNTS.values()))
        self.assertEqual(geometry["rows"], 1311334)
        self.assertEqual((geometry["world_size"], geometry["microbatch"], geometry["gradient_accumulation_steps"]), (8, 4, 1))
        self.assertEqual(geometry["global_batch_size"], 32)
        self.assertEqual(geometry["steps_per_epoch"], 40979)
        self.assertEqual(geometry["total_steps"], 122937)
        self.assertEqual(geometry["warmup_steps"], 6147)
        self.assertEqual(geometry["dropped_rows_per_epoch"], 6)
        total, warmup = 122937, 6147
        self.assertEqual(contract.lr_factor(0, total), .1)
        self.assertEqual(contract.lr_factor(8, total), .1)  # smoke keeps formal horizon
        self.assertAlmostEqual(contract.lr_factor(warmup - 1, total), 1.)
        self.assertAlmostEqual(contract.lr_factor(warmup, total), 1.)
        self.assertAlmostEqual(contract.lr_factor(total, total), .1)
        self.assertGreater(contract.lr_factor(total - 1, total), .1)

    def test_original_recording_not_historical_chunk2(self):
        row = {"source_dataset": "reasonaqa_oneaudio", "filepath1_original": "raw.wav",
               "filepath1": "processed_chunk1.wav", "filepath2": "processed_chunk2.wav"}
        self.assertEqual(contract.raw_references(row), ("reasonaqa_oneaudio", [("filepath1_original", "raw.wav")]))
        row = {"source_dataset": "Clotho", "audio_path": "raw.wav", "filepath2": "chunk2.wav"}
        self.assertEqual(contract.raw_references(row)[1], [("audio_path", "raw.wav")])
        row["audio_path2"] = "actual_second_recording.wav"
        self.assertEqual(len(contract.raw_references(row)[1]), 2)

    def test_unfiltered_or_missing_raw_row_fails(self):
        for row in ({"source_dataset": "MusicCaps", "audio_path": "mc.wav"},
                    {"source_dataset": "SpeechCraft", "audio_path": "/x/train-other-500/a.wav"},
                    {"source_dataset": "Clotho", "filepath1": "processed.wav"}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                contract.raw_references(row)

    def test_resolution_tsv_ignores_failed_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolved.tsv"
            header = "field\traw_path\tstatus\tresolved_path\n"
            path.write_text(header + "audio_path\ta.wav\tmissing\t/x/a.wav\n"
                            + "audio_path\tb.wav\texists\t/x/b.wav\n", encoding="utf-8")
            self.assertEqual(stage_data.load_resolutions(path), {("audio_path", "b.wav"): "/x/b.wav"})
            with path.open("a", encoding="utf-8") as handle:
                handle.write("audio_path\tb.wav\texists\t/other/b.wav\n")
            with self.assertRaisesRegex(ValueError, "conflicting"):
                stage_data.load_resolutions(path)

    def test_streamed_staging_preserves_fields_and_deduplicates_actual_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "recording.wav"
            original.write_bytes(b"raw-audio-placeholder")
            rows = [{"source_dataset": "reasonaqa_oneaudio", "filepath1_original": "first-id.wav",
                     "filepath1": "old/chunk1.wav", "filepath2": "old/chunk2.wav", "input": "ABC", "answer": "DEF"},
                    {"source_dataset": "AudioCaps", "audio_path": "second-alias.wav", "input": "XYZ", "answer": "Q"}]
            manifest, metadata = root / "s1.json", root / "rows.jsonl"
            manifest.write_text(json.dumps(rows), encoding="utf-8")
            resolutions = {("filepath1_original", "first-id.wav"): str(original),
                           ("audio_path", "second-alias.wav"): str(original)}
            report, files = stage_data.prepare(manifest, resolutions, metadata, strict_counts=False)
            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["unique_raw_files"], 1)
            self.assertEqual(len(files), 1)
            indexed = IndexedRows(metadata)
            try:
                self.assertEqual(indexed[0]["s1_raw_paths"], indexed[1]["s1_raw_paths"])
                for i, original_row in enumerate(rows):
                    self.assertEqual({k: indexed[i][k] for k in original_row}, original_row)
                self.assertEqual(len(indexed[0]["s1_raw_paths"]), 1)
                with self.assertRaises(IndexError):
                    indexed[2]
            finally:
                indexed.close()
            report2, _ = stage_data.prepare(manifest, resolutions, root / "repeat.jsonl", strict_counts=False)
            self.assertEqual(report["rows_canonical_sha256"], report2["rows_canonical_sha256"])
            self.assertEqual(report["mapping_sha256"], report2["mapping_sha256"])

    def test_retained_mapping_miss_aborts_instead_of_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "s1.json"
            manifest.write_text(json.dumps([{"source_dataset": "Clotho", "audio_path": "missing.wav"}]))
            with self.assertRaisesRegex(ValueError, "no successful audited path"):
                stage_data.prepare(manifest, {}, root / "metadata.jsonl", strict_counts=False)

    def test_changed_mixture_aborts_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "recording.wav"
            original.write_bytes(b"audio")
            manifest = root / "s1.json"
            manifest.write_text(json.dumps([{"source_dataset": "Clotho", "audio_path": "one.wav"}]))
            with self.assertRaisesRegex(ValueError, "source counts changed"):
                stage_data.prepare(manifest, {("audio_path", "one.wav"): str(original)}, root / "rows.jsonl")

    def test_smoke_gate_checks_all_ranks_not_just_a_pass_label(self):
        report = passed_smoke()
        validate_smoke(report, "recipe-a")
        with self.assertRaises(ValueError):
            validate_smoke(report, "recipe-b")
        for key, value in (("optimizer_steps", 7), ("seed", 1234), ("frozen_htsat_equal", False),
                           ("parameters_finite", False), ("first8_rows", [])):
            broken = copy.deepcopy(report)
            broken["rank_checks"][7][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_smoke(broken, "recipe-a")
        broken = copy.deepcopy(report)
        broken["checkpoint"]["model_tensors_equal"] = False
        with self.assertRaises(ValueError):
            validate_smoke(broken, "recipe-a")

    def test_gradnorm_scalar_policy_matches_reference_behavior(self):
        tree = ast.parse((ROUTE / "optim.py").read_text(encoding="utf-8"))
        tracker_class = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "GradNormTracker")
        scope = {}
        exec(compile(ast.Module(body=[tracker_class], type_ignores=[]), "gradnorm-policy", "exec"), scope)
        tracker = scope["GradNormTracker"]()
        running, scale = tracker._clip_grad(100., 1., 1.)
        self.assertAlmostEqual(scale, .025)
        self.assertAlmostEqual(running, .995 + .005 * 6.25)
        running, scale = tracker._clip_grad(.5, 1., 1.)
        self.assertAlmostEqual(scale, 1.)
        self.assertAlmostEqual(running, .9975)

    def test_layout_dimensions_and_geometry_in_runner(self):
        self.assertEqual(contract.PREFIX_TOKENS, 515)
        self.assertEqual(contract.PREFIX_TOKENS + contract.ANSWER_TOKENS, 815)
        outer = (ROOT / "code/RSmol/run_mizar_s1_5090.sh").read_text(encoding="utf-8")
        inner = (ROUTE / "run.sh").read_text(encoding="utf-8")
        self.assertIn("-c 32 -m 256G -g 8 -n 1", outer)
        self.assertIn("--nproc_per_node=8", inner)
        self.assertIn("s1_stage1_filtered.json", inner)
        self.assertIn("trap cleanup EXIT", inner)


if __name__ == "__main__":
    unittest.main()
