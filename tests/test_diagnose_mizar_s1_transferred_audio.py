"""Targeted tests for the read-only transferred-audio diagnosis."""
from __future__ import annotations

import csv
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/RSmol/scripts/diagnose_mizar_s1_transferred_audio.py"
SPEC = importlib.util.spec_from_file_location("diagnose_mizar_audio", SCRIPT)
diagnosis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnosis)


class DiagnoseTransferredAudioTest(unittest.TestCase):
    def test_advisor_mapping_recovers_renamed_file_and_preserves_row_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            speechcraft = root / "data" / "SpeechCraft"
            speechcraft.mkdir(parents=True)
            mapped_audio = speechcraft / "103_1241_000000_000001.flac"
            mapped_audio.write_bytes(b"fixture")
            raw = "libritts_r_audios/train-clean-100/103/1241/original_name.wav"
            with (audit_dir / "resolved_paths.tsv").open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(["field", "raw_path", "references", "status", "rule"])
                writer.writerow(["audio_path", raw, 4, "missing", "speechcraft_libritts_r"])
            mapping_path = root / "mapping.jsonl"
            mapping_path.write_text(json.dumps({
                "source_dataset": "SpeechCraft", "audio_path": raw,
                "status": "file", "local_path": "/s2/materialized/" + mapped_audio.name,
                "archive_path": None, "archive_member": None,
            }) + "\n", encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                report = diagnosis.diagnose(audit_dir, mapping_path, root / "data")

            source = report["datasets"]["SpeechCraft"]
            self.assertEqual(source["failed_unique_references_by_diagnosis"], {
                "advisor_filename_found_unique": 1,
            })
            self.assertEqual(source["failed_manifest_rows_by_diagnosis"], {
                "advisor_filename_found_unique": 4,
            })
            self.assertEqual(source["examples"]["advisor_filename_found_unique"][0]["candidate"], str(mapped_audio))
            self.assertEqual(mapped_audio.read_bytes(), b"fixture")

    def test_extension_change_is_a_hint_and_never_accepted_as_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "Yvideo_30.flac"
            audio.write_bytes(b"fixture")
            names, stems, _ = diagnosis.inventory(root)

            kind, details = diagnosis.classify({
                "status": "file", "local_path": "/s2/Yvideo_30.wav",
            }, names, stems)

            self.assertEqual(kind, "advisor_filename_absent_same_stem_only")
            self.assertNotIn("candidate", details)
            self.assertEqual(details["hints_not_accepted"], [str(audio)])

    def test_conflicting_mapping_records_are_retained_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mapping.jsonl"
            first = {
                "source_dataset": "MusicCaps", "audio_path": "musiccaps_audios/a.wav",
                "status": "file", "local_path": "/s2/a.wav",
            }
            second = {**first, "local_path": "/s2/b.wav"}
            path.write_text("".join(
                json.dumps(row) + "\n" for row in (first, first, second)
            ), encoding="utf-8")
            key = ("MusicCaps", "musiccaps_audios/a.wav")

            records, counts = diagnosis.read_s2_mapping(path, {key: {}})

            self.assertEqual(len(records[key]), 2)
            self.assertEqual(counts["MusicCaps"]["file"], 3)

    def test_advisor_missing_and_duplicate_destination_names_are_distinct(self):
        kind, _ = diagnosis.classify({"status": "missing", "local_path": ""}, {}, {})
        self.assertEqual(kind, "advisor_also_missing")
        kind, details = diagnosis.classify({
            "status": "file", "local_path": "/s2/a.wav",
        }, {"a.wav": ["/d6/first/a.wav", "/d6/second/a.wav"]}, {})
        self.assertEqual(kind, "advisor_filename_has_multiple_candidates")
        self.assertEqual(len(details["candidates"]), 2)

    def test_summary_shows_archive_member_and_directory_scan_status(self):
        report = {
            "current_audit": {}, "s2_mapping_records_by_source_status": {},
            "datasets": {"SpeechCraft": {
                "directory_inventory": {
                    "root_exists": True, "audio_file_entries": 0, "file_entries": 0,
                    "extensions": {}, "example_audio_relative_paths": [],
                    "walk_error_count": 1, "walk_errors": ["permission denied"],
                    "skipped_symlink_directory_count": 1,
                    "skipped_symlink_directories_first_10": ["/audio/linked"],
                },
                "failed_unique_references_by_diagnosis": {"advisor_filename_absent": 1},
                "failed_manifest_rows_by_diagnosis": {"advisor_filename_absent": 2},
                "examples": {"advisor_filename_absent": [{
                    "manifest_rows": 2, "raw_path": "libritts_r_audios/a.wav",
                    "s2_record": {"status": "archive", "local_path": "",
                                  "archive_path": "/s2/audio.tar",
                                  "archive_member": "LibriTTS_R/a.wav"},
                }]},
            }},
        }
        output = io.StringIO()
        with redirect_stdout(output):
            diagnosis.print_summary(report)
        summary = output.getvalue()
        self.assertIn("s2_archive_member=LibriTTS_R/a.wav", summary)
        self.assertIn("s2_archive_path=/s2/audio.tar", summary)
        self.assertIn("directory walk errors: 1", summary)
        self.assertIn("skipped symlink directories: 1", summary)


if __name__ == "__main__":
    unittest.main()
