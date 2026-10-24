"""Dependency-light checks for the isolated GBS256 training route."""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs"
sys.path.insert(0, str(ROUTE))

import contract
import stage_data


class MizarS1Gbs256Tests(unittest.TestCase):
    def test_geometry_matches_effective_batch_and_epoch_tail(self):
        result = contract.geometry(1477492)
        self.assertEqual(contract.ROUTE, "htsat_mizar_s1_fixed20_gbs256_v1")
        self.assertEqual((contract.WORLD_SIZE, contract.BATCH_SIZE, contract.ACCUMULATION_STEPS), (8, 8, 4))
        self.assertEqual(contract.GLOBAL_BATCH_SIZE, 256)
        self.assertEqual(contract.RANK_UPDATE_SAMPLES, 32)
        self.assertEqual(result["steps_per_epoch"], 5771)
        self.assertEqual(result["total_steps"], 17313)
        self.assertEqual(result["warmup_steps"], 866)
        self.assertEqual(result["dropped_rows_per_epoch"], 116)

    def test_staging_priorities_cover_each_complete_accumulation_group(self):
        rows = 2073
        per_rank = rows // contract.WORLD_SIZE
        permutation = list(range(rows))
        order = contract.data_order(rows)
        priorities, initial = stage_data.epoch_priorities(rows, permutation=permutation)
        for rank in range(contract.WORLD_SIZE):
            offset = rank * per_rank
            expected = [order[i] for i in permutation[offset:offset + 8 * contract.RANK_UPDATE_SAMPLES]]
            self.assertEqual(initial[str(rank)], expected)
            for cursor, index in enumerate(permutation[offset:offset + len(expected)]):
                self.assertEqual(priorities[order[index]], cursor // contract.RANK_UPDATE_SAMPLES)
            for cursor in range(8 * contract.RANK_UPDATE_SAMPLES, per_rank):
                self.assertEqual(priorities[order[permutation[offset + cursor]]], 8)

        resumed, _ = stage_data.epoch_priorities(rows, epoch=1, start_batch=7, permutation=permutation)
        for rank in range(contract.WORLD_SIZE):
            offset = rank * per_rank
            for cursor in range(per_rank):
                expected_priority = 8 if cursor < 7 * contract.RANK_UPDATE_SAMPLES or cursor >= 8 * contract.RANK_UPDATE_SAMPLES else 7
                self.assertEqual(resumed[order[permutation[offset + cursor]]], expected_priority)

    def test_staging_window_preserves_sample_coverage_scale(self):
        files = {
            "audio/probe.wav": {"bytes": 1, "first_batch": -1},
            "audio/first.wav": {"bytes": 2, "first_batch": 0},
            "audio/prefill-last.wav": {"bytes": 3, "first_batch": 63},
            "audio/next.wav": {"bytes": 4, "first_batch": 64},
        }
        ordered, initial, budgets = stage_data.staging_plan(files, 5771, 0, 64)
        self.assertEqual(initial, 64)
        self.assertEqual([name for name, _ in ordered], [
            "audio/probe.wav", "audio/first.wav", "audio/prefill-last.wav", "audio/next.wav"])
        self.assertEqual(next(item for item in budgets if item["steps"] == 64),
                         {"steps": 64, "files": 3, "bytes": 6})
        self.assertEqual(64 * contract.GLOBAL_BATCH_SIZE, 512 * 32)
        self.assertEqual(16 * contract.GLOBAL_BATCH_SIZE, 128 * 32)

    def test_trainer_accumulates_four_microbatches_and_syncs_once(self):
        source = (ROUTE / "train.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "no_sync" for node in ast.walk(tree)))
        self.assertIn("scaler.scale(loss / ACCUMULATION_STEPS).backward()", source)
        self.assertIn("scaler.unscale_(optimizer)", source)
        self.assertIn("tracker.track_and_clip_(model.named_parameters())", source)
        self.assertIn('"next_optimizer_step": next_optimizer_step', source)
        self.assertIn('"gradient_accumulation_steps": ACCUMULATION_STEPS', source)

    def test_submission_is_formal_only_and_uses_isolated_route(self):
        launcher = (ROOT / "code/RSmol/run_mizar_s1_gbs256_5090.sh").read_text(encoding="utf-8")
        runner = (ROUTE / "run.sh").read_text(encoding="utf-8")
        config = (ROUTE / "write_config.py").read_text(encoding="utf-8")
        self.assertIn("mizar_stage1_fixed20_gbs256_3epochs/run.sh", launcher)
        self.assertIn("8 GPUs x microbatch8 x accumulation4", launcher)
        self.assertIn("INDEX_CACHE_ROOT=", launcher)
        self.assertIn("PREPARED_INDEX=$7", runner)
        self.assertIn("shared-path training starts immediately", launcher)
        self.assertIn('choices=("formal",)', config)
        self.assertNotIn("smoke-report", launcher + runner + config)
        self.assertNotIn("smoke-then-formal", launcher + runner)

    def test_hybrid_manifest_and_mapping_are_the_active_inputs(self):
        launcher = (ROOT / "code/RSmol/run_mizar_s1_gbs256_5090.sh").read_text(encoding="utf-8")
        runner = (ROUTE / "run.sh").read_text(encoding="utf-8")
        expected_manifest = "/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json"
        expected_audit = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv"
        for source in (launcher, runner):
            self.assertIn(expected_manifest, source)
            self.assertNotIn("s1_stage1_filtered.json", source)
        self.assertIn(expected_audit, launcher)
        self.assertNotIn("mizar_s1_audio_audit_audiocaps_csv_retry1", launcher)

    def test_hybrid_sources_are_allowed_by_raw_reference_contract(self):
        self.assertEqual(contract.raw_references(
            {"source_dataset": "MusicCaps", "audio_path": "musiccaps_audios/a.wav"}),
            ("MusicCaps", [("audio_path", "musiccaps_audios/a.wav")]))
        self.assertEqual(contract.raw_references(
            {"source_dataset": "SpeechCraft", "audio_path": "libritts_r_audios/train-other-500/a.wav"}),
            ("SpeechCraft", [("audio_path", "libritts_r_audios/train-other-500/a.wav")]))


if __name__ == "__main__":
    unittest.main()
