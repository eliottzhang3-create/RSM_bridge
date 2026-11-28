"""Real WAV/JSON/TSV/CLI checks for conservative S3 preparation (CPU only)."""
from __future__ import annotations

from collections import Counter
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "RSmol"))
from mizar_s3_data_prepare.audio import AudioAuditor
from mizar_s3_data_prepare.io_utils import canonical, digest, iter_rows, publish, sha256_file, write_json
from mizar_s3_data_prepare.mapping import (AUDIO_CONTRACT, AVQAMapper, AudioMCQMapper,
                                         Unmatched, candidate_mcq_key, validate_candidate)
from mizar_s3_data_prepare.prepare import main, prepare, verify_dataset


def wav(path, *, rate=16000, channels=2, frames=800):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        samples = [((i % 100) - 50) * 100 for i in range(frames * channels)]
        handle.writeframes(struct.pack("<" + "h" * len(samples), *samples))


def candidate(name, *, source="AVQA", source_id="avqa_train_00000184", raw="/old/hash.wav"):
    return {"id": name, "sample_key": name, "training_original_sample_key": name,
            "source": "avqa" if source == "AVQA" else "audiomcq", "source_dataset": source,
            "source_sample_key": source_id, "filepath1": raw, "filepath2": "",
            "audio_input_contract": AUDIO_CONTRACT, "question": "based only on the audio, what is heard?",
            "choices": ["car", "train", "bird", "rain"], "gold_option_index": 0,
            "gold_letter": "a", "gold_choice": "car", "answer": "a) car",
            "input": "based only on the audio, what is heard? a) car b) train c) bird d) rain",
            "schema_version": "table14_raw_direct_single4_v1", "opaque_extra": {"preserve": [1, True]}}


def annotation(identifier=184, *, video="-HIPq7T3eFI_000011", name="-HIPq7T3eFI_11.wav"):
    return {"id": identifier, "video_name": video, "video_id": 342,
            "question_text": "What is the main source of sound in the video?",
            "multi_choice": ["Car", "train", "bird", "rain"], "answer": 0,
            "question_relation": "Both", "question_type": "Come From", "dataset_name": "AVQA",
            "audio_path": "./Joysw909/AVQA/VGG10000/" + name}


class Fixture:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.root = self.folder / "AVQA"
        self.root.mkdir()
        self.avwav = self.root / "VGG10000" / "-HIPq7T3eFI_11.wav"
        wav(self.avwav)
        self.mcqwav = self.folder / "historical audio" / "clip.wav"
        wav(self.mcqwav, rate=32000, channels=1)
        self.ann = self.folder / "avqa.jsonl"
        self.history = self.folder / "history.json"
        self.tsv = self.folder / "resolved.tsv"
        self.source = self.folder / "s3.json"
        self.output = self.folder / "prepared"
        self.annotations = [annotation()]
        self.rows = [candidate("ac", source="AudioCaps", source_id="42", raw="/old/ac/clip.wav"),
                     candidate("av"), candidate("missing", source_id="avqa_train_00000999", raw="/old/unmatched.wav")]
        self.histories = [{"source": "audiomcq", "source_dataset": "AudioCaps", "id": "42",
                           "stageE_variant": "audiomcq_mcq", "audio_contribution": "strong",
                           "choices": ["Car", "train", "bird", "rain"], "gold_letter": "a",
                           "audio_path": "audiocaps_audios/clip.wav", "input": "historical prompt",
                           "answer": "a) Car"}]
        self.save()

    def save(self):
        write_json(self.source, self.rows)
        write_json(self.history, self.histories)
        self.ann.write_text("".join(canonical(r) + "\n" for r in self.annotations), encoding="utf-8")
        with self.tsv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["field", "raw_path", "status", "resolved_path"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"field": "audio_path", "raw_path": "audiocaps_audios/clip.wav",
                             "status": "exists", "resolved_path": str(self.mcqwav)})

    def run(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare(self.source, self.ann, self.root, self.history, self.tsv, self.output,
                           expected_sha=sha256_file(self.source),
                           expected_counts=dict(Counter(r["source_dataset"] for r in self.rows)))


class S3PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="s3-audit-", dir=os.environ.get("RSMOL_TEST_TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(self.temp.name)

    def test_full_preparation_preserves_metadata_order_and_denominator(self):
        before = {p: sha256_file(p) for p in (self.f.source, self.f.ann, self.f.history, self.f.tsv,
                                             self.f.avwav, self.f.mcqwav)}
        manifest = self.f.run()
        self.assertEqual(manifest["rows"], 2)
        self.assertEqual(list(iter_rows(self.f.output / "train.json")), self.f.rows[:2])
        rejected = list(iter_rows(self.f.output / "excluded.jsonl"))
        self.assertEqual(rejected[0]["metadata"], self.f.rows[2])
        self.assertEqual(rejected[0]["reason"], "avqa_annotation_id_missing")
        report = json.loads((self.f.output / "dataset_report.json").read_text(encoding="utf-8"))
        self.assertEqual((report["input_rows"], report["kept_rows"], report["excluded_rows"]), (3, 2, 1))
        self.assertEqual(report["unique_retained_audio"], 2)
        self.assertEqual(before, {p: sha256_file(p) for p in before})
        self.assertEqual(verify_dataset(self.f.output)["seal_sha256"], manifest["seal_sha256"])
        original = {p.name: sha256_file(p) for p in self.f.output.iterdir()}
        self.f.run()
        self.assertEqual(original, {p.name: sha256_file(p) for p in self.f.output.iterdir()})

    def test_shared_recording_rescues_only_missing_annotation(self):
        self.f.rows[2]["filepath1"] = self.f.rows[1]["filepath1"]
        self.f.save()
        self.assertEqual(self.f.run()["rows"], 3)
        self.assertEqual((self.f.output / "excluded.jsonl").stat().st_size, 0)
        selected = list(iter_rows(self.f.output / "selection.jsonl"))
        self.assertEqual(selected[2]["evidence"]["validated_anchors"][0]["annotation_id"], 184)
        self.assertEqual(len(list(iter_rows(self.f.output / "audio_audit.jsonl"))), 2)

    def test_label_mismatch_not_rescued_by_shared_recording(self):
        self.f.annotations.append(annotation(999))
        self.f.annotations[1]["answer"] = 1
        self.f.rows[2]["filepath1"] = self.f.rows[1]["filepath1"]
        self.f.save()
        self.assertEqual(self.f.run()["rows"], 2)
        self.assertEqual(next(iter_rows(self.f.output / "excluded.jsonl"))["reason"], "avqa_options_or_gold_mismatch")

    def test_recording_start_case_ambiguity_and_traversal_are_rejected(self):
        mapper = AVQAMapper(self.f.ann, self.f.root)
        row = self.f.rows[1]
        mapper.observe(row)
        self.assertEqual(Path(mapper.resolve(row)[0]), self.f.avwav.resolve())
        wav(self.f.root / "VGG20000" / "-HIPq7T3eFI_000011.wav")
        mapper = AVQAMapper(self.f.ann, self.f.root)
        mapper.observe(row)
        with self.assertRaisesRegex(Unmatched, "multiple_wavs"):
            mapper.resolve(row)
        self.f.annotations[0]["audio_path"] = "../VGG10000/-HIPq7T3eFI_11.wav"
        self.f.save()
        mapper = AVQAMapper(self.f.ann, self.f.root)
        mapper.observe(row)
        with self.assertRaisesRegex(Unmatched, "unsafe"):
            mapper.resolve(row)

    def test_wrong_recording_name_and_visual_only_excluded(self):
        for field, value, reason in (("video_name", "-HIPq7T3eFI_000012", "video_and_audio"),
                                     ("question_relation", "View", "relation_not")):
            with self.subTest(field=field):
                self.f.annotations[0] = annotation()
                self.f.annotations[0][field] = value
                self.f.save()
                mapper = AVQAMapper(self.f.ann, self.f.root)
                mapper.observe(self.f.rows[1])
                with self.assertRaisesRegex(Unmatched, reason):
                    mapper.resolve(self.f.rows[1])

    def test_conflicting_avqa_recording_excludes_entire_old_path(self):
        second = annotation(999, video="ABCDEFGHIJK_000012", name="ABCDEFGHIJK_12.wav")
        self.f.annotations.append(second)
        wav(self.f.root / "VGG10000" / "ABCDEFGHIJK_12.wav")
        self.f.rows[2]["filepath1"] = self.f.rows[1]["filepath1"]
        self.f.save()
        self.assertEqual(self.f.run()["rows"], 1)
        rejected = list(iter_rows(self.f.output / "excluded.jsonl"))
        self.assertTrue(all(r["reason"] == "avqa_shared_s3_path_has_conflicting_recordings" for r in rejected))

    def test_history_identity_not_basename_or_new_absolute_path_fallback(self):
        row = self.f.rows[0]
        needed = {candidate_mcq_key(row)}
        mapper = AudioMCQMapper(self.f.history, self.f.tsv, needed)
        self.assertEqual(mapper.resolve(row)[0], str(self.f.mcqwav))
        wrong = copy.deepcopy(row)
        wrong["source_sample_key"] = "wrong"
        wrong["filepath1"] = str(self.f.mcqwav)
        with self.assertRaisesRegex(Unmatched, "no_historical"):
            mapper.resolve(wrong)
        self.f.histories[0]["audio_contribution"] = "weak"
        self.f.save()
        with self.assertRaisesRegex(ValueError, "no strongAC"):
            AudioMCQMapper(self.f.history, self.f.tsv, needed)

    def test_conflicting_historical_audit_is_excluded(self):
        with self.f.tsv.open("a", encoding="utf-8") as handle:
            handle.write("audio_path\taudiocaps_audios/clip.wav\texists\t/another/clip.wav\n")
        self.assertEqual(self.f.run()["rows"], 1)
        rejected = list(iter_rows(self.f.output / "excluded.jsonl"))
        self.assertEqual(rejected[0]["reason"], "audiomcq_historical_audit_conflict")

    def test_s3_gold_corrections_preserved_while_old_history_only_maps_audio(self):
        self.f.histories[0]["gold_letter"] = "d"
        self.f.histories[0]["answer"] = "d) rain"
        self.f.save()
        self.assertEqual(self.f.run()["rows"], 2)
        self.assertEqual(next(iter_rows(self.f.output / "train.json"))["answer"], "a) car")
        evidence = next(iter_rows(self.f.output / "selection.jsonl"))["evidence"]
        self.assertFalse(evidence["historical_gold_matches_s3"])
        self.assertEqual(evidence["historical_gold_letters"], ["d"])
        report = json.loads((self.f.output / "dataset_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["retained_audiomcq_historical_gold_differences"], 1)

    def test_missing_and_corrupt_audio_are_recorded_and_filtered(self):
        self.f.mcqwav.write_bytes(b"not audio")
        result = self.f.run()
        self.assertEqual(result["rows"], 1)
        rejected = list(iter_rows(self.f.output / "excluded.jsonl"))
        self.assertEqual(rejected[0]["reason"], "audio_decode_or_file_check_failed")
        self.assertIn("resolved_path", rejected[0]["details"])

    def test_full_decode_finite_stereo_and_zero_policy(self):
        import numpy as np
        import soundfile as sf
        decoder = AudioAuditor()
        record = decoder.check(self.f.avwav)
        self.assertEqual((record["channels"], record["samplerate"], record["decoded_frames"]), (2, 16000, 800))
        self.assertEqual(decoder.check(self.f.avwav), record)
        self.assertTrue(record["first20_will_pad"])
        nonfinite = self.f.folder / "bad float.wav"
        samples = np.zeros(330000, dtype=np.float32)
        samples[-1] = np.nan
        sf.write(nonfinite, samples, 16000, subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "NaN"):
            decoder.check(nonfinite)
        zero = self.f.folder / "zero.wav"
        sf.write(zero, np.zeros(320001, dtype=np.float32), 16000)
        record = decoder.check(zero)
        self.assertTrue(record["all_zero"])
        self.assertTrue(record["first20_will_crop"])

    def test_duplicate_candidates_and_invalid_gold_fail_without_publication(self):
        self.f.rows.append(copy.deepcopy(self.f.rows[0]))
        self.f.save()
        with self.assertRaisesRegex(ValueError, "duplicate S3"):
            self.f.run()
        self.assertFalse(self.f.output.exists())
        bad = candidate("bad")
        bad["gold_option_index"] = True
        with self.assertRaises(ValueError):
            validate_candidate(bad)

    def test_source_output_and_audio_tampering_rejected(self):
        self.f.run()
        original = (self.f.output / "train.json").read_bytes()
        (self.f.output / "train.json").write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "prepared file changed"):
            verify_dataset(self.f.output)
        (self.f.output / "train.json").write_bytes(original)
        wav(self.f.avwav, frames=801)
        with self.assertRaisesRegex(ValueError, "audio identity changed"):
            verify_dataset(self.f.output)

    def test_failure_keeps_private_diagnostics_and_releases_lock(self):
        with patch("mizar_s3_data_prepare.prepare.publish", side_effect=OSError("simulated mount failure")):
            with self.assertRaisesRegex(OSError, "mount failure"):
                self.f.run()
        self.assertFalse(self.f.output.exists())
        self.assertFalse(self.f.output.with_name("prepared.prepare.lock").exists())
        stages = list(self.f.folder.glob("prepared.building-*"))
        self.assertEqual(len(stages), 1)
        self.assertEqual(json.loads((stages[0] / "failure.json").read_text())["status"], "FAILED")

    def test_strict_streaming_parser_and_no_replace(self):
        path = self.f.folder / "parser.json"
        for payload in ('[{"x":1},]', '[{"x":1}] trailing', '[{"x":1}', '{"x":1,"x":2}\n', '{"x":NaN}\n'):
            path.write_text(payload, encoding="utf-8")
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                list(iter_rows(path))
        staging, existing = self.f.folder / "stage", self.f.folder / "exists"
        staging.mkdir()
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            publish(staging, existing)
        self.assertTrue(staging.is_dir())

    def test_preview_cli_does_not_read_assets(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "code" / "RSmol") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        result = subprocess.run([sys.executable, "-B", "-m", "mizar_s3_data_prepare", "--check-args",
                                 "--source", "missing source.json", "--output-dir", "with spaces"],
                                env=env, text=True, capture_output=True, check=True)
        value = json.loads(result.stdout)
        self.assertEqual(value["paths"]["output_dir"], "with spaces")
        self.assertEqual(sum(value["expected_sources"].values()), 289952)
        self.assertFalse(value["gpu"])

    def test_busy_lock_and_wrong_input_identity_are_not_overwritten(self):
        lock = self.f.output.with_name("prepared.prepare.lock")
        lock.write_text("another owner", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.f.run()
        self.assertEqual(lock.read_text(encoding="utf-8"), "another owner")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "SHA disagrees"):
            prepare(self.f.source, self.f.ann, self.f.root, self.f.history, self.f.tsv,
                    self.f.output, expected_sha="0" * 64)
        self.assertFalse(self.f.output.exists())

    def test_case_sensitive_recording_join_does_not_use_similar_name(self):
        self.f.annotations[0]["video_name"] = "-hipq7t3efi_000011"
        self.f.annotations[0]["audio_path"] = "./VGG10000/-hipq7t3efi_11.wav"
        self.f.save()
        mapper = AVQAMapper(self.f.ann, self.f.root)
        mapper.observe(self.f.rows[1])
        with self.assertRaisesRegex(Unmatched, "wav_missing"):
            mapper.resolve(self.f.rows[1])

    def test_actual_bash_wrapper_quotes_arguments_and_disables_cuda(self):
        import shutil
        bash = shutil.which("bash")
        if os.name == "nt":
            bash = "C:/Program Files/Git/bin/bash.exe"
        self.assertTrue(bash and Path(bash).is_file(), "Bash is required for this wrapper check")
        base = self.f.folder / "conda base with spaces"
        profile = base / "etc" / "profile.d" / "conda.sh"
        profile.parent.mkdir(parents=True)
        profile.write_text('conda() { [ "$1" = activate ] && [ "$2" = mellow_c8204d8 ]; }\n', encoding="utf-8")
        bin_dir = self.f.folder / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "python"
        fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n'
                        'printf "CUDA=%s\\n" "$CUDA_VISIBLE_DEVICES" >> "$CAPTURE"\n', encoding="utf-8")
        fake.chmod(0o755)
        capture = self.f.folder / "arguments.txt"
        env = dict(os.environ, RSMOL_CONDA_BASE=base.as_posix(), CAPTURE=capture.as_posix())
        # Set the fake executable search path inside Bash, avoiding Windows PATH
        # conversion differences. The wrapper itself is executed unmodified.
        script = ROOT / "code" / "RSmol" / "prepare_mizar_s3_data.sh"
        result = subprocess.run([bash, "-c", 'bin="$1"; if command -v cygpath >/dev/null; then bin=$(cygpath -u "$bin"); fi; export PATH="$bin:$PATH"; shift; bash "$@"', "wrapper-test",
                        bin_dir.as_posix(), script.as_posix(), "--check-args", "--output-dir", "with spaces"],
                       env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(capture.read_text(encoding="utf-8").splitlines(),
                         ["-B", "-u", "-m", "mizar_s3_data_prepare", "--check-args",
                          "--output-dir", "with spaces", "CUDA="])


if __name__ == "__main__":
    unittest.main(verbosity=2)
