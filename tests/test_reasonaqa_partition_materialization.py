"""CPU-only materialization tests with a small fixed-stride payload."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "code/RSmol/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("materializer_test", SCRIPT_DIR / "materialize_reasonaqa_component_partitions.py")
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)
import plan_reasonaqa_component_partitions as planner


def make_fixture(root: Path):
    manifest = root / "train.jsonl"
    rows = [{"audio1_path": f"/audio/{aid}.wav", "audio2_path": "",
             "prompt": f"question {aid}", "answer": f"answer {aid}"} for aid in range(6)]
    manifest.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    store = root / "source_store"
    store.mkdir()
    entries = []
    for aid in range(6):
        entries.append({"audio_id": aid, "source_path": f"/audio/{aid}.wav",
                        "source_group": "audiocaps", "source_size_bytes": 123 + aid,
                        "source_mtime_ns": 456 + aid, "manifest_aliases": [f"/alias/{aid}.wav"],
                        "slot_reference_count": 2, "qa_incidence_count": 1,
                        "byte_offset": aid * materializer.BYTES_PER_AUDIO,
                        "byte_length": materializer.BYTES_PER_AUDIO, "shape": [1, 320000], "dtype": "float32"})
    index = store / "index.jsonl"
    index.write_text("".join(json.dumps(x) + "\n" for x in entries), encoding="utf-8")
    data = store / materializer.DATA_FILE
    digest = hashlib.sha256()
    with data.open("wb") as handle:
        for aid in range(6):
            payload = bytes([aid + 1]) * materializer.BYTES_PER_AUDIO
            handle.write(payload)
            digest.update(payload)
    metadata = {"format": planner.STORE_FORMAT, "status": "PASS",
                "manifest": str(manifest), "manifest_sha256": planner.sha256_file(manifest),
                "source_inventory_sha256": "fixture", "num_unique_audio_files": 6,
                "sample_rate": 32000, "seconds": 10, "samples_per_audio": 320000,
                "bytes_per_audio": materializer.BYTES_PER_AUDIO, "dtype": "float32",
                "byte_order": "little", "data_file": materializer.DATA_FILE,
                "total_waveform_bytes": 6 * materializer.BYTES_PER_AUDIO,
                "index_sha256": planner.sha256_file(index), "waveform_sha256": digest.hexdigest(),
                "waveform_verification": {"passed": True},
                "preprocessing_contract": "fixture fixed waveform"}
    planner.write_json(store / "metadata.json", metadata)
    plan = root / "plan"
    with redirect_stdout(io.StringIO()):
        planner.main(["--manifest", str(manifest), "--waveform-store-dir", str(store),
                      "--output-dir", str(plan), "--skip-tokenization", "--trials", "2",
                      "--refine-candidates", "1", "--local-rounds", "1", "--swap-attempts", "4"])
    return manifest, store, plan, entries


class PartitionMaterializationTest(unittest.TestCase):
    def test_dry_run_creates_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, store, plan, _ = make_fixture(root)
            output = root / "output"
            with redirect_stdout(io.StringIO()):
                rc = materializer.main(["--plan-dir", str(plan), "--source-store-dir", str(store),
                                        "--output-dir", str(output), "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertFalse(output.exists())

    def test_materializes_six_reader_compatible_zero_copy_stores(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, store, plan, source_entries = make_fixture(root)
            output = root / "output"
            with redirect_stdout(io.StringIO()):
                materializer.main(["--plan-dir", str(plan), "--source-store-dir", str(store),
                                   "--output-dir", str(output), "--checkpoint-every", "2",
                                   "--verify-samples-per-partition", "1", "--free-space-margin-gib", "0"])
            report = json.loads((output / "materialization_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(report["source_payload_sha256_reverified"])
            self.assertEqual(report["duplicated_audio"], 0)
            self.assertEqual(report["total_materialized_waveform_bytes"], 6 * materializer.BYTES_PER_AUDIO)
            observed_global = []
            for p in range(6):
                directory = output / f"partition_{p}"
                self.assertFalse((directory / "BUILDING").exists())
                metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["format"], planner.STORE_FORMAT)
                self.assertEqual(metadata["status"], "PASS")
                self.assertEqual(metadata["manifest_sha256"], planner.sha256_file(directory / "rows.jsonl"))
                self.assertEqual(metadata["source_global_inventory_sha256"], "fixture")
                index = [x for _, _, x, _ in planner.iter_rows(directory / "index.jsonl")]
                expected_inventory = materializer._partition_source_inventory_sha256(
                    [source_entries[x["global_audio_id"]] for x in index])
                self.assertEqual(metadata["source_inventory_sha256"], expected_inventory)
                self.assertEqual([x["audio_id"] for x in index], list(range(len(index))))
                self.assertEqual((directory / materializer.DATA_FILE).stat().st_size,
                                 len(index) * materializer.BYTES_PER_AUDIO)
                for local, entry in enumerate(index):
                    gid = entry["global_audio_id"]
                    observed_global.append(gid)
                    self.assertEqual(entry["global_byte_offset"], source_entries[gid]["byte_offset"])
                    self.assertEqual(entry["byte_offset"], local * materializer.BYTES_PER_AUDIO)
                    with (directory / materializer.DATA_FILE).open("rb") as handle:
                        handle.seek(local * materializer.BYTES_PER_AUDIO)
                        self.assertEqual(handle.read(1), bytes([gid + 1]))
            self.assertEqual(sorted(observed_global), list(range(6)))
            self.assertFalse((output / "BUILDING").exists())

    def test_resume_truncates_nondurable_tail_and_completes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, store, plan, _ = make_fixture(root)
            contract = materializer.load_contract(plan, store)
            output = root / "output"
            config = materializer._initialize_output(output, contract)
            completed = 3
            counts = materializer._prefix_counts(contract["owner"], completed, 6)
            with (store / materializer.DATA_FILE).open("rb") as source:
                for gid in range(completed):
                    payload = source.read(materializer.BYTES_PER_AUDIO)
                    p = contract["owner"][gid]
                    with (output / f"partition_{p}" / f".{materializer.DATA_FILE}.partial").open("ab") as target:
                        target.write(payload)
            # Simulate bytes written after the last durable progress update.
            tail_partition = contract["owner"][completed]
            with (output / f"partition_{tail_partition}" / f".{materializer.DATA_FILE}.partial").open("ab") as target:
                target.write(b"nondurable")
            materializer.write_json_atomic(output / "progress.json", {
                "status": "BUILDING", "completed_global_audio": completed,
                "per_partition_completed_audio": counts, "updated_unix": 1,
            })
            with redirect_stdout(io.StringIO()):
                report = materializer.materialize(contract, output, resume=True, checkpoint_every=2,
                                                   verify_samples_per_partition=1, free_space_margin_gib=0)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(sum(x["total_waveform_bytes"] for x in config["partitions"]),
                             report["total_materialized_waveform_bytes"])

    def test_rejects_changed_plan_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, store, plan, _ = make_fixture(root)
            with (plan / "partition_0_rows.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "hash/size"):
                materializer.load_contract(plan, store)

    def test_refuses_overwrite_and_resume_of_absent_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, store, plan, _ = make_fixture(root)
            contract = materializer.load_contract(plan, store)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                materializer.materialize(contract, output, resume=False, checkpoint_every=1,
                                         verify_samples_per_partition=1, free_space_margin_gib=0)
            missing = root / "missing"
            with self.assertRaises(FileNotFoundError):
                materializer.materialize(contract, missing, resume=True, checkpoint_every=1,
                                         verify_samples_per_partition=1, free_space_margin_gib=0)


if __name__ == "__main__":
    unittest.main()
