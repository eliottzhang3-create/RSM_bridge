"""Tests for filtering Mizar S1 rows against the CPU audio audit."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code/RSmol/scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "filter_mizar_s1_manifest.py"
SPEC = importlib.util.spec_from_file_location("filter_mizar_s1_manifest", SCRIPT)
filtering = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = filtering
SPEC.loader.exec_module(filtering)


class FilterMizarS1ManifestTest(unittest.TestCase):
    def create_fixture(self, root: Path) -> tuple[Path, Path, list[dict]]:
        rows = [
            {
                "source": "reasonaqa_oneaudio",
                "filepath1_original": "AudioCapsLarger/train/keep.wav",
                "filepath1": "/historical/processed/one.wav",
                "filepath2": "/historical/processed/two.wav",
                "answer": "keep this row exactly",
            },
            {
                "source": "MusicCaps",
                "audio_path": "musiccaps_audios/clip.wav",
                "id": "mc-1",
            },
            {
                "source": "SpeechCraft",
                "audio_path": "libritts_r_audios/train-other-500/1/2/sample.wav",
                "id": "sp-other",
            },
            {
                "source": "AudioCaps",
                "audio_path": "audiocaps_audios/missing.wav",
                "id": "ac-missing",
            },
            {
                "source": "Clotho",
                "audio_path": "clotho_audio/ambiguous.wav",
                "id": "cl-ambiguous",
            },
            {
                "source": "LP-MusicCaps-MTT",
                "audio_path": "lp_musiccaps_mtt/keep.mp3",
                "id": "lp-keep",
                "metadata": {"unchanged": True},
            },
            {
                "source": "SpeechCraft",
                "audio_path": "libritts_r_audios/train-clean-100/1/2/clean.wav",
                "id": "sp-clean",
            },
            {
                "source": "MusicCaps",
                "audio_path": "musiccaps_audios/available.wav",
                "id": "mc-existing",
            },
        ]
        manifest = root / "s1.json"
        manifest.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

        audit_dir = root / "audit"
        audit_dir.mkdir()
        (audit_dir / "summary.json").write_text(json.dumps({
            "manifest": str(manifest),
            "counts": {
                "rows": len(rows),
                "existing_rows": len(rows) - 3,
                "decode_failures": 0,
            },
            "unique_resolved_audio_bytes": 123,
            "unique_resolved_audio_gib": 123 / (1024 ** 3),
            "size_stat_failures": 0,
            "source_rows": {
                source: sum(row["source"] == source for row in rows)
                for source in {row["source"] for row in rows}
            },
        }), encoding="utf-8")

        issue_records = {
            "allowed_missing.jsonl": [{
                "row_index": 1,
                "source": "MusicCaps",
                "field": "audio_path",
                "raw_path": "musiccaps_audios/clip.wav",
                "status": "unmapped",
            }],
            "retained_missing.jsonl": [{
                "row_index": 3,
                "source": "AudioCaps",
                "field": "audio_path",
                "raw_path": "audiocaps_audios/missing.wav",
                "status": "missing",
            }],
            "ambiguous.jsonl": [{
                "row_index": 4,
                "source": "Clotho",
                "field": "audio_path",
                "raw_path": "clotho_audio/ambiguous.wav",
                "status": "ambiguous",
            }],
            "unmapped.jsonl": [],
            "decode_failures.jsonl": [],
        }
        for filename, records in issue_records.items():
            (audit_dir / filename).write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
        return manifest, audit_dir, rows

    def run_filter(
        self,
        manifest: Path,
        audit_dir: Path,
        output: Path,
        *,
        keep_musiccaps: bool = False,
        keep_train_other_500: bool = False,
        max_unique_audio_gib: float = 0.0,
    ) -> dict:
        return filtering.filter_manifest(filtering.argparse.Namespace(
            manifest=manifest,
            audit_dir=audit_dir,
            output_manifest=output,
            excluded_jsonl=None,
            report=None,
            keep_musiccaps=keep_musiccaps,
            keep_train_other_500=keep_train_other_500,
            max_unique_audio_gib=max_unique_audio_gib,
            progress_every=0,
        ))

    def test_filters_intentional_sources_and_audit_issues_preserving_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, audit_dir, original_rows = self.create_fixture(root)
            output = root / "filtered.json"

            report = self.run_filter(manifest, audit_dir, output)
            filtered_rows = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(filtered_rows, [original_rows[0], original_rows[5], original_rows[6]])
            self.assertEqual(report["counts"]["input_rows"], 8)
            self.assertEqual(report["counts"]["kept_rows"], 3)
            self.assertEqual(report["counts"]["excluded_rows"], 5)
            self.assertEqual(report["excluded_by_reason"]["dataset_musiccaps"], 2)
            self.assertEqual(report["excluded_by_reason"]["speechcraft_train_other_500"], 1)
            self.assertEqual(report["excluded_by_reason"]["audit_allowed_missing"], 1)
            self.assertEqual(report["excluded_by_reason"]["audit_retained_missing"], 1)
            self.assertEqual(report["excluded_by_reason"]["audit_ambiguous"], 1)

            excluded = [
                json.loads(line)
                for line in (root / "filtered.excluded_rows.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            mc_reasons = next(item["exclude_reasons"] for item in excluded if item["row_index"] == 1)
            self.assertEqual(mc_reasons, ["audit_allowed_missing", "dataset_musiccaps"])
            self.assertTrue((root / "filtered.filter_report.json").is_file())

    def test_expanded_policy_keeps_resolved_musiccaps_and_train_other_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, audit_dir, original_rows = self.create_fixture(root)
            output = root / "expanded.json"

            report = self.run_filter(
                manifest,
                audit_dir,
                output,
                keep_musiccaps=True,
                keep_train_other_500=True,
                max_unique_audio_gib=1.0,
            )

            kept = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(kept, [original_rows[index] for index in (0, 2, 5, 6, 7)])
            self.assertEqual(report["counts"]["kept_rows"], 5)
            self.assertTrue(report["retention_policy"]["keep_musiccaps"])
            self.assertTrue(report["retention_policy"]["keep_speechcraft_train_other_500"])

    def test_capacity_guard_refuses_manifest_over_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, audit_dir, _ = self.create_fixture(root)

            with self.assertRaisesRegex(ValueError, "exceeds the requested staging limit"):
                self.run_filter(
                    manifest,
                    audit_dir,
                    root / "over_limit.json",
                    keep_musiccaps=True,
                    keep_train_other_500=True,
                    max_unique_audio_gib=0.000000001,
                )
            self.assertFalse((root / "over_limit.json").exists())

    def test_rejects_issue_index_that_disagrees_with_manifest_row(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, audit_dir, _ = self.create_fixture(root)
            issue = {
                "row_index": 3,
                "source": "Clotho",
                "field": "audio_path",
                "raw_path": "audiocaps_audios/missing.wav",
                "status": "missing",
            }
            (audit_dir / "retained_missing.jsonl").write_text(
                json.dumps(issue) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                self.run_filter(manifest, audit_dir, root / "should_not_exist.json")
            self.assertFalse((root / "should_not_exist.json").exists())

    def test_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, audit_dir, _ = self.create_fixture(root)
            output = root / "existing.json"
            output.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                self.run_filter(manifest, audit_dir, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_rejects_manifest_with_matching_row_count_but_changed_source_totals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, audit_dir, rows = self.create_fixture(root)
            rows[0]["source"] = "AudioCaps"
            manifest.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source counts changed since audit"):
                self.run_filter(manifest, audit_dir, root / "should_not_exist.json")
            self.assertFalse((root / "should_not_exist.json").exists())


if __name__ == "__main__":
    unittest.main()
