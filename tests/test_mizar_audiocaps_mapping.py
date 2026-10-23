"""Tests for AudioCaps v2 ID/start-time path resolution in the Mizar audit."""
from __future__ import annotations

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/RSmol/scripts/audit_mizar_s1_raw_audio.py"
SPEC = importlib.util.spec_from_file_location("mizar_audio_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class AudioCapsV2MappingTest(unittest.TestCase):
    def make_rules(
        self,
        root: Path,
        train_rows: list[tuple[str, int]],
        supplemental_root: Path | None = None,
    ) -> list[dict]:
        for split, rows in {
            "train": train_rows,
            "val": [],
            "test": [],
        }.items():
            contents = "youtube_id,start_time,caption\n" + "".join(
                f"{youtube_id},{start_time},caption\n" for youtube_id, start_time in rows
            )
            (root / f"{split}.csv").write_text(contents, encoding="utf-8")
        config = root / "mapping.json"
        rule = {
            "name": "audiomcq_audiocaps",
            "field": "audio_path",
            "prefix": "audiocaps_audios/",
            "match": "audiocaps_v2",
            "roots": [str(root)],
            "metadata_root": str(root),
        }
        if supplemental_root is not None:
            rule["flat_roots"] = [str(supplemental_root)]
        config.write_text(json.dumps({"rules": [rule]}), encoding="utf-8")
        rules = audit.load_rules(config)
        audit.build_recursive_indexes(rules)
        return rules

    def test_legacy_youtube_filename_maps_after_csv_key_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "train" / "YpgLXVFvo5GI.wav"
            audio.parent.mkdir()
            audio.touch()
            rules = self.make_rules(root, [("pgLXVFvo5GI", 27), ("pgLXVFvo5GI", 27)])
            result = audit.resolve("audiocaps_audios/pgLXVFvo5GI_27.wav", "audio_path", rules)
            self.assertEqual(result["status"], "exists")
            self.assertEqual(Path(result["path"]), audio.resolve())
            self.assertEqual(result["method"], "audiocaps_v2_csv_legacy_youtube_id")

    def test_legacy_filename_is_ambiguous_when_start_time_was_discarded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "train" / "YpgLXVFvo5GI.wav"
            audio.parent.mkdir()
            audio.touch()
            rules = self.make_rules(root, [("pgLXVFvo5GI", 27), ("pgLXVFvo5GI", 55)])
            result = audit.resolve("audiocaps_audios/pgLXVFvo5GI_27.wav", "audio_path", rules)
            self.assertEqual(result["status"], "ambiguous")
            self.assertEqual(result["method"], "audiocaps_v2_legacy_name_loses_start_time")

    def test_canonical_filename_preserves_start_time_even_if_id_repeats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "train" / "pgLXVFvo5GI_27.wav"
            audio.parent.mkdir()
            audio.touch()
            rules = self.make_rules(root, [("pgLXVFvo5GI", 27), ("pgLXVFvo5GI", 55)])
            result = audit.resolve("audiocaps_audios/pgLXVFvo5GI_27.wav", "audio_path", rules)
            self.assertEqual(result["status"], "exists")
            self.assertEqual(Path(result["path"]), audio.resolve())
            self.assertEqual(result["method"], "audiocaps_v2_csv_canonical")

    def test_supplemental_tree_is_searched_recursively_only_after_preferred_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "existing"
            supplemental = Path(temp) / "supplemental"
            root.mkdir()
            new_audio = supplemental / "nested" / "YpgLXVFvo5GI.wav"
            new_audio.parent.mkdir(parents=True)
            new_audio.touch()
            rules = self.make_rules(
                root, [("pgLXVFvo5GI", 27)], supplemental_root=supplemental
            )

            result = audit.resolve(
                "audiocaps_audios/pgLXVFvo5GI_27.wav", "audio_path", rules
            )

            self.assertEqual(result["status"], "exists")
            self.assertEqual(Path(result["path"]), new_audio.resolve())
            self.assertEqual(result["method"], "audiocaps_v2_csv_legacy_youtube_id")

    def test_preferred_legacy_path_wins_over_supplemental_canonical_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "existing"
            supplemental = Path(temp) / "supplemental"
            old_audio = root / "train" / "YpgLXVFvo5GI.wav"
            new_audio = supplemental / "pgLXVFvo5GI_27.wav"
            old_audio.parent.mkdir(parents=True)
            new_audio.parent.mkdir(parents=True)
            old_audio.touch()
            new_audio.touch()
            rules = self.make_rules(
                root, [("pgLXVFvo5GI", 27)], supplemental_root=supplemental
            )

            result = audit.resolve(
                "audiocaps_audios/pgLXVFvo5GI_27.wav", "audio_path", rules
            )

            self.assertEqual(Path(result["path"]), old_audio.resolve())
            self.assertEqual(result["method"], "audiocaps_v2_csv_legacy_youtube_id")

    def test_supplemental_copy_does_not_hide_ambiguous_preferred_legacy_name(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "existing"
            supplemental = Path(temp) / "supplemental"
            old_audio = root / "train" / "YpgLXVFvo5GI.wav"
            new_audio = supplemental / "pgLXVFvo5GI_27.wav"
            old_audio.parent.mkdir(parents=True)
            new_audio.parent.mkdir(parents=True)
            old_audio.touch()
            new_audio.touch()
            rules = self.make_rules(
                root,
                [("pgLXVFvo5GI", 27), ("pgLXVFvo5GI", 55)],
                supplemental_root=supplemental,
            )

            result = audit.resolve(
                "audiocaps_audios/pgLXVFvo5GI_27.wav", "audio_path", rules
            )

            self.assertEqual(result["status"], "ambiguous")
            self.assertEqual(
                result["method"], "audiocaps_v2_legacy_name_loses_start_time"
            )

    def test_csv_key_and_audio_file_failures_are_distinguished(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rules = self.make_rules(root, [("pgLXVFvo5GI", 27)])
            missing_file = audit.resolve(
                "audiocaps_audios/pgLXVFvo5GI_27.wav", "audio_path", rules
            )
            unknown_key = audit.resolve(
                "audiocaps_audios/pgLXVFvo5GI_28.wav", "audio_path", rules
            )
            self.assertEqual(missing_file["status"], "missing")
            self.assertEqual(missing_file["method"], "audiocaps_v2_csv_key_file_missing")
            self.assertEqual(unknown_key["status"], "unmapped")
            self.assertEqual(unknown_key["method"], "audiocaps_metadata_key_not_found")

    def test_full_audit_reports_legacy_resolution_method(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "train" / "YpgLXVFvo5GI.wav"
            audio.parent.mkdir()
            audio.write_bytes(b"audio")
            self.make_rules(root, [("pgLXVFvo5GI", 27)])
            manifest = root / "s1.json"
            manifest.write_text(json.dumps([{
                "source_dataset": "AudioMCQ",
                "audio_path": "audiocaps_audios/pgLXVFvo5GI_27.wav",
            }]), encoding="utf-8")
            output = root / "audit"
            with redirect_stdout(io.StringIO()):
                code = audit.main([
                    "--manifest", str(manifest),
                    "--mapping-json", str(root / "mapping.json"),
                    "--output-dir", str(output),
                    "--progress-every", "0",
                ])
            report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(report["counts"]["exists"], 1)
            self.assertEqual(report["method_rows"]["audiocaps_v2_csv_legacy_youtube_id"], 1)
            self.assertEqual(report["unique_resolved_files"], 1)
            self.assertEqual(report["unique_resolved_audio_bytes"], 5)
            self.assertEqual((output / "unique_files.tsv").read_text().splitlines()[1].split("\t")[1], "5")

    def test_unique_audio_limit_returns_capacity_exit_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "train" / "YpgLXVFvo5GI.wav"
            audio.parent.mkdir()
            audio.write_bytes(b"audio")
            self.make_rules(root, [("pgLXVFvo5GI", 27)])
            manifest = root / "s1.json"
            manifest.write_text(json.dumps([{
                "source_dataset": "AudioMCQ",
                "audio_path": "audiocaps_audios/pgLXVFvo5GI_27.wav",
            }]), encoding="utf-8")
            output = root / "audit"
            with redirect_stdout(io.StringIO()):
                code = audit.main([
                    "--manifest", str(manifest),
                    "--mapping-json", str(root / "mapping.json"),
                    "--output-dir", str(output),
                    "--max-unique-audio-gib", "0.000000001",
                    "--progress-every", "0",
                ])
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 3)
            self.assertTrue(summary["capacity_check"]["exceeded"])

    def test_full_audit_writes_ambiguous_rows_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "train" / "YpgLXVFvo5GI.wav"
            audio.parent.mkdir()
            audio.touch()
            self.make_rules(root, [("pgLXVFvo5GI", 27), ("pgLXVFvo5GI", 55)])
            manifest = root / "s1.json"
            manifest.write_text(json.dumps([{
                "source_dataset": "AudioMCQ",
                "audio_path": "audiocaps_audios/pgLXVFvo5GI_27.wav",
            }]), encoding="utf-8")
            output = root / "audit"
            with redirect_stdout(io.StringIO()):
                code = audit.main([
                    "--manifest", str(manifest),
                    "--mapping-json", str(root / "mapping.json"),
                    "--output-dir", str(output),
                    "--progress-every", "0",
                ])
            issues = [json.loads(line) for line in (output / "ambiguous.jsonl").read_text().splitlines()]
            self.assertEqual(code, 2)
            self.assertEqual(len(issues), 1)
            self.assertEqual(issues[0]["method"], "audiocaps_v2_legacy_name_loses_start_time")
            self.assertEqual((output / "retained_missing.jsonl").read_text(), "")

    def test_empty_csv_start_time_is_skipped_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "train" / "YpgLXVFvo5GI.wav"
            audio.parent.mkdir()
            audio.touch()
            self.make_rules(root, [("pgLXVFvo5GI", 27)])
            with (root / "train.csv").open("a", encoding="utf-8") as handle:
                handle.write("anotherVideoId,,bad metadata row\n")
            manifest = root / "s1.json"
            manifest.write_text(json.dumps([{
                "source_dataset": "AudioMCQ",
                "audio_path": "audiocaps_audios/pgLXVFvo5GI_27.wav",
            }]), encoding="utf-8")
            output = root / "audit"
            with redirect_stdout(io.StringIO()):
                code = audit.main([
                    "--manifest", str(manifest),
                    "--mapping-json", str(root / "mapping.json"),
                    "--output-dir", str(output),
                    "--progress-every", "0",
                ])
            report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            train_audit = report["metadata_audit"]["audiomcq_audiocaps"]["train"]
            self.assertEqual(code, 0)
            self.assertEqual(train_audit["skipped_records"], 1)
            self.assertEqual(train_audit["skipped_by_reason"], {"empty_start_time": 1})
            self.assertEqual(train_audit["skipped_examples_first_20"][0]["record"], 3)


if __name__ == "__main__":
    unittest.main()
