"""Concurrency, byte-copy and eight-rank ordering checks without torch/CUDA."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_3epochs"
sys.path.insert(0, str(ROUTE))
import stage_data
from contract import data_order, write_json
from data.staging import StagingGate, process_alive, wait_for_start


def fixture(root, *, steps=4):
    root.mkdir(exist_ok=True)
    (root / "audio").mkdir()
    info = {"token": "test-job", "producer_pid": os.getpid(), "epoch": 0, "start_batch": 0,
            "status_path": str(root / "STAGING.json")}
    write_json(root / ".owner.json", {"root": str(root), "token": info["token"]})
    state = {**info, "phase": "COPYING", "heartbeat": time.time(), "ready_through_batch": 1,
             "copied_files": 0, "copied_bytes": 0}
    prepared = {"status": "TRAINING_READY", "stage_root": str(root), "async_staging": info,
                "planned_files": 5, "planned_bytes": 5, "initial_steps": 1,
                "staging_epoch_steps": steps, "phase_seconds": {}, "precopy_seconds": 0.}
    write_json(root / "STAGING.json", state)
    return prepared, state


class BackgroundStagingTests(unittest.TestCase):
    def test_priority_covers_the_next_batches_of_all_eight_contiguous_rank_shards(self):
        count, per_rank, steps = 1033, 129, 32
        permutation = list(reversed(range(count)))
        order = data_order(count)
        priorities, first8 = stage_data.epoch_priorities(count, permutation=permutation)
        resumed, _ = stage_data.epoch_priorities(count, start_batch=7, permutation=permutation)
        for rank in range(8):
            expected = [order[i] for i in permutation[rank * per_rank:(rank + 1) * per_rank]]
            self.assertEqual(first8[str(rank)], expected[:32])
            for cursor, original_index in enumerate(expected[:128]):
                self.assertEqual(priorities[original_index], cursor // 4)
                self.assertEqual(resumed[original_index], steps if cursor < 28 else cursor // 4)
            self.assertEqual(priorities[expected[128]], steps)
        self.assertEqual(sum(value == steps for value in priorities), 9)

    def test_prepare_deduplicates_by_earliest_use_and_always_prioritizes_source_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("probe.wav", "shared.wav"):
                (root / name).write_bytes(b"original raw bytes")
            rows = [{"source_dataset": "Clotho", "audio_path": name} for name in
                    ("probe.wav", "shared.wav", "shared.wav")]
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(rows), encoding="utf-8")
            resolutions = {("audio_path", name): str(root / name) for name in ("probe.wav", "shared.wav")}
            _, files = stage_data.prepare(manifest, resolutions, root / "rows.jsonl", strict_counts=False,
                                          row_priorities=[7, 5, 1])
            priorities = {Path(info["source"]).name: info["first_batch"] for info in files.values()}
            self.assertEqual(priorities, {"probe.wav": -1, "shared.wav": 1})

    def test_partial_copy_never_becomes_a_readable_final_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audio").mkdir()
            source = root / "original.wav"
            raw = b"this is deliberately not decodable audio"
            source.write_bytes(raw)
            started, release = threading.Event(), threading.Event()
            errors = []

            def slow_copy(original, temporary):
                Path(temporary).write_bytes(raw[:4])
                started.set()
                if not release.wait(3):
                    raise TimeoutError("test did not release copy")
                Path(temporary).write_bytes(raw)

            def producer():
                try:
                    stage_data.copy_one(root, ("audio/test.wav", {"source": str(source), "bytes": len(raw)}))
                except BaseException as exc:
                    errors.append(exc)

            with patch.object(stage_data.shutil, "copyfile", slow_copy):
                thread = threading.Thread(target=producer)
                thread.start()
                try:
                    self.assertTrue(started.wait(3))
                    self.assertFalse((root / "audio/test.wav").exists())
                    self.assertTrue((root / "audio/test.wav.partial").exists())
                finally:
                    release.set()
                    thread.join(3)
            self.assertFalse(errors)
            self.assertEqual((root / "audio/test.wav").read_bytes(), raw)
            self.assertFalse((root / "audio/test.wav.partial").exists())

    def test_training_crosses_prefill_boundary_and_later_epoch_waits_for_remaining_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, state = fixture(root)
            ordered = []
            for i, priority in enumerate((-1, 0, 1, 3, 4)):
                source = root / f"source{i}.wav"
                source.write_bytes(bytes([i]))
                ordered.append((f"audio/{i}.wav", {"source": str(source), "bytes": 1, "first_batch": priority}))
            report = root / "report.json"
            later_started, later_release = threading.Event(), threading.Event()
            tail_started, tail_release = threading.Event(), threading.Event()
            errors = []
            original_copy = stage_data.copy_one

            def controlled_copy(destination, entry):
                priority = entry[1]["first_batch"]
                if priority in (1, 4):
                    started, release = ((later_started, later_release) if priority == 1 else (tail_started, tail_release))
                    started.set()
                    if not release.wait(5):
                        raise TimeoutError("test did not release background copy")
                return original_copy(destination, entry)

            def producer():
                try:
                    stage_data.copy_background(root, prepared, ordered, report, state, workers=1, window_steps=1)
                except BaseException as exc:
                    errors.append(exc)

            def consume(gate, epoch, batch, done):
                try:
                    gate.wait_for_batch(epoch, batch)
                    done.set()
                except BaseException as exc:
                    errors.append(exc)

            with patch.object(stage_data, "copy_one", controlled_copy):
                worker = threading.Thread(target=producer)
                worker.start()
                consumers = []
                try:
                    self.assertTrue(later_started.wait(3))
                    snapshot = json.loads(report.read_text(encoding="utf-8"))
                    self.assertEqual((snapshot["status"], snapshot["staged_files"]), ("TRAINING_READY", 2))
                    gate = StagingGate(snapshot, poll=.005, timeout=3)
                    gate.wait_for_batch(0, 0)
                    wait_for_start(report, root, os.getpid(), timeout=.2, poll=.005)
                    done = threading.Event()
                    consumers.append(threading.Thread(target=consume, args=(gate, 0, 1, done)))
                    consumers[-1].start()
                    self.assertFalse(done.wait(.03))
                    later_release.set()
                    self.assertTrue(done.wait(3), errors)
                    self.assertTrue(tail_started.wait(3), errors)
                    gate.wait_for_batch(0, 3)
                    tail_done = threading.Event()
                    consumers.append(threading.Thread(target=consume, args=(gate, 1, 0, tail_done)))
                    consumers[-1].start()
                    self.assertFalse(tail_done.wait(.03))
                    tail_release.set()
                    self.assertTrue(tail_done.wait(3), errors)
                finally:
                    later_release.set()
                    tail_release.set()
                    worker.join(5)
                    for consumer in consumers:
                        consumer.join(5)
            self.assertFalse(errors)
            self.assertFalse(worker.is_alive())
            final = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual((final["status"], final["staged_files"], final["staged_bytes"]), ("PASS", 5, 5))
            self.assertTrue(gate.complete)
            self.assertTrue((root / "READY.json").exists())
            for i in range(5):
                self.assertEqual((root / f"audio/{i}.wav").read_bytes(), bytes([i]))

    def test_copy_failure_is_published_and_never_passes_the_training_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, state = fixture(root)
            source = root / "source.wav"
            source.write_bytes(b"wrong length")
            entries = [("audio/x.wav", {"source": str(source), "bytes": 1, "first_batch": 0})]
            with self.assertRaisesRegex(IOError, "short staging copy"):
                stage_data.copy_background(root, prepared, entries, root / "report.json", state, workers=1)
            self.assertEqual(json.loads((root / "STAGING.json").read_text())["phase"], "FAILED")
            self.assertFalse((root / "audio/x.wav").exists())
            self.assertFalse((root / "audio/x.wav.partial").exists())
            with self.assertRaisesRegex(RuntimeError, "background staging failed"):
                StagingGate(prepared).wait_for_batch(0, 0)

    def test_gate_rejects_dead_stale_incomplete_and_wrong_owner_states(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, state = fixture(root)
            self.assertTrue(process_alive(os.getpid()))
            with patch("data.staging.process_alive", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "exited before completion"):
                    StagingGate(prepared).wait_for_batch(0, 0)
            write_json(root / "STAGING.json", dict(state, heartbeat=time.time() - 121))
            with self.assertRaisesRegex(RuntimeError, "heartbeat is stale"):
                StagingGate(prepared).wait_for_batch(0, 0)
            write_json(root / "STAGING.json", dict(state, phase="COMPLETE", copied_files=4, copied_bytes=4))
            with self.assertRaisesRegex(RuntimeError, "counts do not match"):
                StagingGate(prepared).wait_complete()
            write_json(root / ".owner.json", {"root": str(root), "token": "other-job"})
            with self.assertRaisesRegex(ValueError, "ownership differs"):
                StagingGate(prepared)

    def test_gate_timeout_and_unfinished_start_are_explicit_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, _ = fixture(root)
            with self.assertRaises(TimeoutError):
                StagingGate(prepared, timeout=.01, poll=.005).wait_for_batch(0, 1)
            with self.assertRaises(TimeoutError):
                wait_for_start(root / "no-report.json", root, os.getpid(), timeout=.01, poll=.005)
            stale = dict(prepared, stage_root=str(root / "previous-allocation"))
            write_json(root / "old-report.json", stale)
            with self.assertRaises(TimeoutError):
                wait_for_start(root / "old-report.json", root, os.getpid(), timeout=.01, poll=.005)

    def test_initial_budget_counts_unique_files_and_smoke_uses_two_steps(self):
        files = {f"audio/{i}.wav": {"bytes": i + 1, "first_batch": priority} for i, priority in
                 enumerate((-1, 0, 1, 3, 4))}
        _, initial, budgets = stage_data.staging_plan(files, 4, 0, 2)
        self.assertEqual(initial, 2)
        self.assertEqual(next(x for x in budgets if x["steps"] == 2), {"steps": 2, "files": 3, "bytes": 6})
        self.assertEqual(stage_data.staging_plan(files, 4, 0, 512, smoke=True)[1], 2)

    def test_runner_overlaps_real_work_and_stops_producer_before_cleanup_on_failure(self):
        bash = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
        if not Path(bash).is_file():
            self.skipTest("Bash is unavailable")

        def bash_path(path):
            value = path.as_posix()
            return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value

        python_stub = r'''#!/usr/bin/env bash
set -eu
case "$1" in
  *stage_data.py)
    if [[ "$2" == --cleanup ]]; then
      if [[ -f "$RSM_TEST_DIR/producer_pid" ]] && kill -0 "$(cat "$RSM_TEST_DIR/producer_pid")" 2>/dev/null; then
        echo 'cleanup encountered a live producer' >&2
        exit 9
      fi
      touch "$RSM_TEST_DIR/cleaned"
      exit 0
    fi
    echo $$ > "$RSM_TEST_DIR/producer_pid"
    touch "$RSM_TEST_DIR/prefill"
    for ((i=0; i<300; i++)); do
      if [[ -f "$RSM_TEST_DIR/train_started" ]]; then
        touch "$RSM_TEST_DIR/complete"
        exit 0
      fi
      sleep .01
    done
    exit 9
    ;;
  -m)
    for ((i=0; i<300; i++)); do
      [[ ! -f "$RSM_TEST_DIR/prefill" ]] || exit 0
      sleep .01
    done
    exit 9
    ;;
  write_config.py)
    [[ " $* " == *" --mode formal "* ]]
    [[ " $* " != *" --smoke-report "* ]]
    [[ "$RSM_TEST_FAIL" != 1 ]] || exit 7
    ;;
  -)
    [[ "$3" == formal && -f "$RSM_TEST_DIR/training_done" ]]
    cat > /dev/null
    ;;
  *) exit 9 ;;
esac
'''
        torchrun_stub = r'''#!/usr/bin/env bash
set -eu
[[ -f "$RSM_TEST_DIR/prefill" && ! -f "$RSM_TEST_DIR/complete" ]]
touch "$RSM_TEST_DIR/train_started"
for ((i=0; i<300; i++)); do
  if [[ -f "$RSM_TEST_DIR/complete" ]]; then
    touch "$RSM_TEST_DIR/training_done"
    exit 0
  fi
  sleep .01
done
exit 9
'''
        for fail in (False, True):
            with self.subTest(config_fails=fail), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                root = Path(directory)
                (root / "bin").mkdir()
                (root / "etc/profile.d").mkdir(parents=True)
                (root / "etc/profile.d/conda.sh").write_text("conda() { :; }\n", encoding="utf-8")
                script = root / "run.sh"
                script.write_text((ROUTE / "run.sh").read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
                for name, text in (("python", python_stub), ("torchrun", torchrun_stub)):
                    path = root / "bin" / name
                    path.write_text(text, encoding="utf-8", newline="\n")
                    path.chmod(0o755)
                resolution = root / "resolved.tsv"
                resolution.write_text("fixture\n", encoding="utf-8")
                env = dict(os.environ, MELLOW_CONDA_BASE=bash_path(root), RSM_TEST_DIR=bash_path(root),
                           RSM_TEST_FAIL=str(int(fail)), PATH=f"{bash_path(root / 'bin')}:/usr/bin:/bin")
                result = subprocess.run([bash, bash_path(script), "formal", bash_path(resolution),
                                         bash_path(root / "output"), "", bash_path(root), bash_path(root), "512", "2"],
                                        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
                self.assertEqual(result.returncode, 7 if fail else 0, result.stdout + result.stderr)
                self.assertTrue((root / "cleaned").exists(), result.stdout + result.stderr)
                self.assertEqual((root / "training_done").exists(), not fail)


if __name__ == "__main__":
    unittest.main()
