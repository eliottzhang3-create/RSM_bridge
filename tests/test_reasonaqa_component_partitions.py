"""Executable CPU-only fixtures: no torch, audio decoder or remote assets."""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "code/RSmol/scripts/plan_reasonaqa_component_partitions.py"
SPEC = importlib.util.spec_from_file_location("component_planner_test", SCRIPT)
planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def fixture(root: Path, rows=None, num_audio=12):
    manifest = root / "train.jsonl"
    if rows is None:
        rows = []
        for aid in range(num_audio):
            for repeat in range(3):
                rows.append({"audio1_path": f"/audio/{aid}.wav", "audio2_path": "",
                             "prompt": "Question?", "answer": "Answer", "original_id": f"{aid}_{repeat}"})
        # Chain is transitive; reverse and repeated edges must not split it.
        for a, b in [(0, 1), (1, 2), (2, 1), (0, 1)]:
            rows.append({"audio1_path": f"/audio/{a}.wav", "audio2_path": f"/audio/{b}.wav",
                         "prompt": "Compare", "answer": "Same", "original_id": f"{a}_{b}"})
    manifest.write_text("\n" + "\n\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    store = root / "store"
    store.mkdir()
    entries = [{"audio_id": a, "source_path": f"/audio/{a}.wav", "manifest_aliases": [f"/alias/{a}.wav"],
                "source_group": "audiocaps" if a < 8 else "clotho", "byte_offset": a * 1280000,
                "byte_length": 1280000, "shape": [1, 320000], "dtype": "float32"} for a in range(num_audio)]
    (store / "index.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    # Sparse payload: planner may stat it but must never read/decode it.
    with (store / "waveforms.f32").open("wb") as handle:
        handle.truncate(num_audio * 1280000)
    meta = {"format": planner.STORE_FORMAT, "status": "PASS", "manifest_sha256": planner.sha256_file(manifest),
            "sample_rate": 32000, "seconds": 10, "samples_per_audio": 320000, "bytes_per_audio": 1280000,
            "dtype": "float32", "byte_order": "little", "data_file": "waveforms.f32",
            "waveform_verification": {"passed": True}, "num_unique_audio_files": num_audio,
            "total_waveform_bytes": num_audio * 1280000, "index_sha256": planner.sha256_file(store / "index.jsonl"),
            "waveform_sha256": "fixture-provenance-not-reverified"}
    planner.write_json(store / "metadata.json", meta)
    return manifest, store, rows


def run_fixture(manifest, store, output):
    with redirect_stdout(io.StringIO()):
        planner.main(["--manifest", str(manifest), "--waveform-store-dir", str(store), "--output-dir", str(output),
                      "--skip-tokenization", "--trials", "4", "--refine-candidates", "2",
                      "--local-rounds", "2", "--swap-attempts", "100"])
    return json.loads((output / "partition_audit.json").read_text(encoding="utf-8"))


class ComponentPartitionsTest(unittest.TestCase):
    def test_end_to_end_coverage_and_double_audio_locality(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, store, original = fixture(root)
            output = root / "plan"
            audit = run_fixture(manifest, store, output)
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["qa_rows"], len(original))
            self.assertEqual(audit["connected_components"], 10)
            self.assertEqual(audit["duplicated_audio"], 0)
            self.assertEqual(audit["cross_partition_qa"], 0)
            self.assertFalse((output / "BUILDING").exists())
            observed = []
            owner = {}
            for p in range(6):
                entries = [r for _, _, r, _ in planner.iter_rows(output / f"partition_{p}_audio.jsonl")]
                local = {r["source_path"] for r in entries}
                for e in entries:
                    self.assertNotIn(e["audio_id"], owner)
                    owner[e["audio_id"]] = p
                for _, _, row, _ in planner.iter_rows(output / f"partition_{p}_rows.jsonl"):
                    observed.append(row)
                    self.assertIn(row["audio1_path"], local)
                    self.assertIn(row["audio2_path"] or row["audio1_path"], local)
            self.assertEqual(owner[0], owner[1])
            self.assertEqual(owner[1], owner[2])
            self.assertEqual(Counter(json.dumps(r, sort_keys=True) for r in observed),
                             Counter(json.dumps(r, sort_keys=True) for r in original))
            sidecar = [r for _, _, r, _ in planner.iter_rows(output / "row_assignments.jsonl")]
            self.assertEqual([r["row_index"] for r in sidecar], list(range(len(original))))
            self.assertEqual(sidecar[0]["manifest_line_number"], 2)

    def test_deterministic_outputs_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, store, _ = fixture(root)
            run_fixture(manifest, store, root / "a")
            run_fixture(manifest, store, root / "b")
            for path in (root / "a").glob("*.json*"):
                self.assertEqual(path.read_bytes(), (root / "b" / path.name).read_bytes())
            with self.assertRaises(FileExistsError):
                run_fixture(manifest, store, root / "a")

    def test_balances_equal_components(self):
        components = [planner.Component(i, [i], rows=10, waveform_bytes=1280000,
                                        compute_proxy=1000, source_rows=Counter({"a": 10})) for i in range(18)]
        assignment, report = planner.pack_components(components, trials=4, local_rounds=2, swap_attempts=50)
        self.assertEqual(Counter(assignment), Counter({p: 3 for p in range(6)}))
        self.assertTrue(report["row_balance_within_tolerance"])

    def test_qa_balance_has_priority_over_secondary_extreme_load(self):
        components = [planner.Component(i, [i], rows=10, waveform_bytes=10**12 if i == 0 else 1,
                                        compute_proxy=10**12 if i == 0 else 1,
                                        source_rows=Counter({"a": 10})) for i in range(18)]
        assignment, report = planner.pack_components(components, trials=4, local_rounds=2, swap_attempts=50)
        self.assertTrue(report["row_balance_within_tolerance"])
        self.assertEqual(Counter(assignment), Counter({p: 3 for p in range(6)}))

    def test_impossible_balance_is_not_silently_claimed(self):
        components = [planner.Component(i, [i], rows=100 if i == 0 else 1, waveform_bytes=1280000,
                                        compute_proxy=1000, source_rows=Counter({"a": 1})) for i in range(6)]
        assignment, report = planner.pack_components(components, trials=2, local_rounds=1, swap_attempts=10)
        self.assertEqual(len(set(assignment)), 6)
        self.assertFalse(report["row_balance_within_tolerance"])

    def test_reject_too_few_components(self):
        components = [planner.Component(0, [0], rows=1)]
        with self.assertRaises(ValueError):
            planner.pack_components(components)

    def test_tokenizer_matches_training_limits_and_fixed_prefix(self):
        calls = []
        def tokenizer(texts, **kwargs):
            calls.append(kwargs)
            return {"input_ids": [[1] * (3 if kwargs["add_special_tokens"] else 2) for _ in texts]}
        with tempfile.TemporaryDirectory() as temp:
            manifest, store, _ = fixture(Path(temp))
            _, audio, aliases, _ = planner.load_store(store, planner.sha256_file(manifest))
            components, ac, first, second, lengths = planner.collect_components(manifest, audio, aliases, tokenizer, 7)
            self.assertEqual(set(lengths), {265})
            self.assertEqual(sum(c.compute_proxy for c in components), len(first) * 265**2)
            self.assertEqual(calls[0], {"max_length": 129, "truncation": True, "padding": False, "add_special_tokens": True})
            self.assertEqual(calls[1]["max_length"], 250)
            self.assertFalse(calls[1]["add_special_tokens"])

    def test_store_metadata_index_and_building_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, store, _ = fixture(Path(temp))
            sha = planner.sha256_file(manifest)
            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                planner.load_store(store, "wrong")
            (store / "BUILDING").touch()
            with self.assertRaisesRegex(ValueError, "BUILDING"):
                planner.load_store(store, sha)
            (store / "BUILDING").unlink()
            with (store / "index.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                planner.load_store(store, sha)

    def test_aliases_missing_second_and_missing_store_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{"audio1_path": "/alias/0.wav", "answer": "yes"}]
            manifest, store, _ = fixture(root, rows, 1)
            _, audio, aliases, _ = planner.load_store(store, planner.sha256_file(manifest))
            components, _, first, second, _ = planner.collect_components(manifest, audio, aliases)
            self.assertEqual(first, second)
            rows[0]["audio1_path"] = "/not-in-store.wav"
            manifest.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "absent"):
                planner.collect_components(manifest, audio, aliases)

    def test_invalid_json_and_missing_answer_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, store, _ = fixture(Path(temp))
            _, audio, aliases, _ = planner.load_store(store, planner.sha256_file(manifest))
            manifest.write_text("{bad\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                planner.collect_components(manifest, audio, aliases)
            manifest.write_text(json.dumps({"audio1_path": "/audio/0.wav"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing answer"):
                planner.collect_components(manifest, audio, aliases)

    def test_cli_rejects_invalid_numeric_options(self):
        for option, value in [("--trials", "0"), ("--num-partitions", "1"), ("--waveform-weight", "nan"),
                              ("--row-tolerance", "inf"), ("--swap-attempts", "-1")]:
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                planner.parse_args(["--output-dir", "unused", option, value])

    def test_independent_audit_catches_corrupted_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, store, _ = fixture(root)
            output = root / "plan"
            audit = run_fixture(manifest, store, output)
            _, audio, aliases, _ = planner.load_store(store, planner.sha256_file(manifest))
            _, _, first, second, _ = planner.collect_components(manifest, audio, aliases)
            owner = [0] * len(audio)
            for p in range(6):
                for _, _, row, _ in planner.iter_rows(output / f"partition_{p}_audio.jsonl"):
                    owner[row["audio_id"]] = p
            path = output / "row_assignments.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join([lines[0], *lines]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "duplicated"):
                planner.audit_written_assignments(output, owner, first, second, audit["partitions"])

    def test_planner_never_opens_waveform_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, store, _ = fixture(root)
            original_open = Path.open
            def guarded_open(path, *args, **kwargs):
                if path.name == "waveforms.f32":
                    raise AssertionError("planner must not read waveform payload")
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", guarded_open):
                run_fixture(manifest, store, root / "plan")


if __name__ == "__main__":
    unittest.main()
