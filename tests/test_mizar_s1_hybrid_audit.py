"""CPU-only checks for the shared-LibriTTS / private-LibriTTS-R audit."""
from __future__ import annotations

import importlib.util
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code/RSmol/scripts"
SPEC = importlib.util.spec_from_file_location(
    "mizar_hybrid_audit", SCRIPTS / "audit_mizar_s1_raw_audio.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class HybridAuditTest(unittest.TestCase):
    def fixture(self, root):
        config = json.loads(
            (SCRIPTS / "mizar_s1_audio_mapping_s2_hybrid.json").read_text(encoding="utf-8")
        )
        data = root / "data"
        shared = root / "shared_libritts"
        for rule in config["rules"]:
            rule["roots"] = [
                str(root / "data" / Path(path).relative_to(
                    "/hpc_stor03/sjtu_home/jinwei.zhang/data"
                )) if path.startswith("/hpc_stor03/sjtu_home/")
                else str(shared / path.rsplit("/", 1)[-1])
                for path in rule["roots"]
            ]
            if "flat_roots" in rule:
                rule["flat_roots"] = [str(data / "AudioCaps")]
            if "metadata_root" in rule:
                rule["metadata_root"] = str(data / "audiocaps_v2")
            if "s2_mapping_jsonl" in rule:
                rule["s2_mapping_jsonl"] = str(root / "s2.jsonl")

        def audio(path, content):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

        audio(data / "audiocaps_v2/train/YpgLXVFvo5GI.wav", b"original audio")
        # A second physical copy must not become a second staging input.
        audio(data / "AudioCaps/pgLXVFvo5GI_27.wav", b"duplicate audio")
        for split in ("train", "val", "test"):
            (data / f"audiocaps_v2/{split}.csv").write_text(
                "youtube_id,start_time\n" + ("pgLXVFvo5GI,27\n" if split == "train" else ""),
                encoding="utf-8",
            )
        utterance = "103/1241/103_1241_000014_000003.wav"
        audio(shared / "train-clean-100" / utterance, b"clean100")
        audio(shared / "train-clean-360" / utterance, b"clean360")
        audio(data / "SpeechCraft/libritts_r_audios/train-other-500" / utterance, b"restored500")
        # This decoy in shared storage must never substitute for LibriTTS-R other500.
        audio(shared / "train-other-500" / utterance, b"wrong version")
        audio(data / "MusicCaps/Ymusic.wav", b"music")
        (root / "s2.jsonl").write_text(json.dumps({
            "source_dataset": "MusicCaps", "audio_path": "musiccaps_audios/music.wav",
            "status": "file", "local_path": "/s2/AudioSet/Ymusic.wav",
        }) + "\n", encoding="utf-8")
        rows = [
            {"source": "reasonaqa_oneaudio", "filepath1_original": "AudioCapsLarger/train/YpgLXVFvo5GI.wav"},
            {"source": "AudioCaps", "audio_path": "audiocaps_audios/pgLXVFvo5GI_27.wav"},
            *({"source": "SpeechCraft", "audio_path": f"libritts_r_audios/{split}/{utterance}"}
              for split in ("train-clean-100", "train-clean-360", "train-other-500")),
            {"source": "MusicCaps", "audio_path": "musiccaps_audios/music.wav"},
        ]
        manifest = root / "s1.json"
        manifest.write_text(json.dumps(rows), encoding="utf-8")
        mapping = root / "mapping.json"
        mapping.write_text(json.dumps(config), encoding="utf-8")
        return manifest, mapping, data, shared, utterance

    def run_audit(self, manifest, mapping, output, *extra):
        with redirect_stdout(io.StringIO()):
            code = audit.main([
                "--manifest", str(manifest), "--mapping-json", str(mapping),
                "--output-dir", str(output), "--canonicalize-resolved-paths",
                "--progress-every", "0", *extra,
            ])
        return code, json.loads((output / "summary.json").read_text(encoding="utf-8"))

    def test_all_splits_and_musiccaps_resolve_without_duplicate_audiocaps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, mapping, data, shared, utterance = self.fixture(root)
            original_manifest = manifest.read_bytes()
            original_mapping = mapping.read_bytes()
            code, report = self.run_audit(manifest, mapping, root / "audit", "--require-all-sources")
            self.assertEqual(code, 0)
            self.assertEqual(report["counts"]["exists"], 6)
            self.assertEqual(report["unique_resolved_files"], 5)
            self.assertEqual(report["unique_resolved_audio_bytes"], 46)
            for split in ("train-clean-100", "train-clean-360", "train-other-500"):
                self.assertEqual(report["speechcraft_by_split"][split], {
                    "exists": {"unique_references": 1, "manifest_rows": 1},
                })
            self.assertEqual(manifest.read_bytes(), original_manifest)
            self.assertEqual(mapping.read_bytes(), original_mapping)
            with (root / "audit/unique_files.tsv").open(encoding="utf-8") as handle:
                paths = handle.read()
            self.assertNotIn(str((data / "AudioCaps").resolve()), paths)
            self.assertNotIn(str((shared / "train-other-500").resolve()), paths)

    def test_missing_other500_is_a_failure_and_shared_decoy_is_not_used(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, mapping, data, _, utterance = self.fixture(root)
            (data / "SpeechCraft/libritts_r_audios/train-other-500" / utterance).unlink()
            (data / "MusicCaps/Ymusic.wav").unlink()
            code, report = self.run_audit(manifest, mapping, root / "audit", "--require-all-sources")
            self.assertEqual(code, 2)
            self.assertEqual(report["counts"]["retained_missing_rows"], 2)
            self.assertEqual(report["counts"].get("allowed_missing_rows", 0), 0)
            self.assertEqual((root / "audit/allowed_missing.jsonl").read_text(), "")
            self.assertEqual(report["speechcraft_by_split"]["train-other-500"], {
                "missing": {"unique_references": 1, "manifest_rows": 1},
            })
            # Old callers retain their allowed-missing policy.
            legacy_code, legacy_report = self.run_audit(manifest, mapping, root / "legacy")
            self.assertEqual(legacy_code, 0)
            self.assertEqual(legacy_report["counts"]["allowed_missing_rows"], 2)

    def test_existing_audit_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, mapping, *_ = self.fixture(root)
            output = root / "audit"
            self.run_audit(manifest, mapping, output, "--require-all-sources")
            original = {p.name: p.read_bytes() for p in output.iterdir()}
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self.run_audit(manifest, mapping, output, "--require-all-sources")
            self.assertEqual(error.exception.code, 2)
            self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, original)


if __name__ == "__main__":
    unittest.main()
