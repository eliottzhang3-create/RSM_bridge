"""Check exact advisor keys and filenames without guessed audio matches."""
from __future__ import annotations

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mizar_s2_audit", ROOT / "code/RSmol/scripts/audit_mizar_s1_raw_audio.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class S2MappingTest(unittest.TestCase):
    def make_rules(self, root, records, source="MusicCaps"):
        metadata = root / "s2.jsonl"
        metadata.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        config = root / "mapping.json"
        prefix = "musiccaps_audios/" if source == "MusicCaps" else "libritts_r_audios/"
        config.write_text(json.dumps({"rules": [{
            "name": "advisor_audio", "field": "audio_path", "prefix": prefix,
            "roots": [str(root / "audio")], "match": "s2_mapping",
            "s2_mapping_jsonl": str(metadata), "source_dataset": source,
        }]}), encoding="utf-8")
        rules = audit.load_rules(config)
        audit.build_recursive_indexes(rules)
        return rules

    def test_y_prefixed_advisor_file_is_used_and_other_dataset_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "audio" / "nested" / "Y-0SdAVK79lg.wav"
            candidate.parent.mkdir(parents=True)
            candidate.write_bytes(b"audio")
            raw = "musiccaps_audios/-0SdAVK79lg.wav"
            record = {"source_dataset": "MusicCaps", "audio_path": raw,
                      "status": "file", "local_path": "/s2/AudioSet/" + candidate.name}
            rules = self.make_rules(root, [record, record, {
                **record, "source_dataset": "AudioCaps", "local_path": "/s2/another.wav",
            }])
            result = audit.resolve(raw, "audio_path", rules)
            self.assertEqual(result["status"], "exists")
            self.assertEqual(result["path"], str(candidate.resolve()))
            self.assertEqual(result["method"], "s2_mapping_exact_filename")

    def test_archive_member_not_empty_local_path_supplies_speech_filename(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "audio" / "103_1241_000014_000003.wav"
            candidate.parent.mkdir()
            candidate.touch()
            raw = "libritts_r_audios/train-clean-100/103/1241/" + candidate.name
            rules = self.make_rules(root, [{
                "source_dataset": "SpeechCraft", "audio_path": raw,
                "status": "archive", "local_path": None,
                "archive_path": "/s2/train_clean_100.tar.gz",
                "archive_member": "LibriTTS_R/train-clean-100/103/1241/" + candidate.name,
            }], source="SpeechCraft")
            self.assertEqual(audit.resolve(raw, "audio_path", rules)["path"], str(candidate.resolve()))

    def test_missing_mapping_cannot_be_replaced_by_a_guessed_local_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "audio").mkdir()
            (root / "audio" / "a.wav").touch()
            raw = "musiccaps_audios/a.wav"
            rules = self.make_rules(root, [])
            self.assertEqual(audit.resolve(raw, "audio_path", rules)["method"], "s2_mapping_key_not_found")
            rules = self.make_rules(root, [{
                "source_dataset": "MusicCaps", "audio_path": raw,
                "status": "missing", "local_path": None,
            }])
            result = audit.resolve(raw, "audio_path", rules)
            self.assertEqual(result["status"], "missing")
            self.assertEqual(result["method"], "s2_mapping_advisor_missing_or_unusable")

    def test_conflicting_advisor_paths_and_duplicate_local_names_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = "musiccaps_audios/a.wav"
            record = {"source_dataset": "MusicCaps", "audio_path": raw,
                      "status": "file", "local_path": "/s2/first/a.wav"}
            for folder in ("first", "second"):
                candidate = root / "audio" / folder / "a.wav"
                candidate.parent.mkdir(parents=True)
                candidate.touch()
            rules = self.make_rules(root, [record])
            self.assertEqual(audit.resolve(raw, "audio_path", rules)["method"], "s2_mapping_multiple_local_files")
            rules = self.make_rules(root, [record, {**record, "local_path": "/s2/second/a.wav"}])
            result = audit.resolve(raw, "audio_path", rules)
            self.assertEqual(result["status"], "ambiguous")
            self.assertEqual(result["method"], "s2_mapping_conflicting_records")

    def test_case_and_extension_changes_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "audio").mkdir()
            (root / "audio" / "Yvideo.flac").touch()
            (root / "audio" / "yvideo.wav").touch()
            raw = "musiccaps_audios/video.wav"
            rules = self.make_rules(root, [{
                "source_dataset": "MusicCaps", "audio_path": raw,
                "status": "file", "local_path": "/s2/Yvideo.wav",
            }])
            self.assertEqual(audit.resolve(raw, "audio_path", rules)["status"], "missing")

    def test_full_audit_deduplicates_multiple_rows_using_the_same_resolved_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "audio").mkdir()
            (root / "audio" / "Ya.wav").write_bytes(b"12345")
            raw = "musiccaps_audios/a.wav"
            self.make_rules(root, [{
                "source_dataset": "MusicCaps", "audio_path": raw,
                "status": "file", "local_path": "/s2/Ya.wav",
            }])
            manifest = root / "s1.json"
            manifest.write_text(json.dumps([
                {"source_dataset": "MusicCaps", "audio_path": raw},
                {"source_dataset": "MusicCaps", "audio_path": raw},
            ]), encoding="utf-8")
            output = root / "audit"
            with redirect_stdout(io.StringIO()):
                code = audit.main([
                    "--manifest", str(manifest), "--mapping-json", str(root / "mapping.json"),
                    "--output-dir", str(output), "--canonicalize-resolved-paths",
                    "--progress-every", "0",
                ])
            report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(report["counts"]["exists"], 2)
            self.assertEqual(report["unique_resolved_files"], 1)
            self.assertEqual(report["unique_resolved_audio_bytes"], 5)


if __name__ == "__main__":
    unittest.main()
