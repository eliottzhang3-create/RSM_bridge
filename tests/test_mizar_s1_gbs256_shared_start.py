"""CPU tests for shared-file training during copying, followed by local reads."""
import contextlib
import io
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
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs"
sys.path.insert(0, str(ROUTE))

import prepare_index
import stage_data
import write_config
from contract import SOURCE_COUNTS, code_digest, write_json
from data.audio_paths import AudioPaths
from data.prepared_index import load_index
from data.staging import StagingGate, wait_for_start


def fixture(root):
    (root / "audio").mkdir()
    info = {"token": "shared-test", "producer_pid": os.getpid(), "epoch": 0, "start_batch": 0,
            "status_path": str(root / "STAGING.json")}
    write_json(root / ".owner.json", {"root": str(root), "token": info["token"]})
    state = {**info, "phase": "COPYING", "heartbeat": time.time(), "ready_through_batch": 0,
             "copied_files": 0, "copied_bytes": 0}
    prepared = {"status": "TRAINING_READY", "read_policy": "shared_until_complete",
                "stage_root": str(root), "async_staging": info, "planned_files": 1, "planned_bytes": 3,
                "initial_steps": 0, "staging_epoch_steps": 5122, "phase_seconds": {}, "precopy_seconds": 0.}
    write_json(root / "STAGING.json", state)
    return prepared, state


class SharedStartTests(unittest.TestCase):
    def test_training_starts_before_first_copy_and_crosses_epoch_without_waiting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, state = fixture(root)
            source = root / "source.wav"
            source.write_bytes(b"raw")  # deliberately not audio: copying never decodes
            report = root / "report.json"
            started, release = threading.Event(), threading.Event()
            errors = []
            original_copy = stage_data.copy_one

            def blocked_copy(destination, entry):
                started.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release copying")
                return original_copy(destination, entry)

            def producer():
                try:
                    stage_data.copy_background(root, prepared,
                                               [("audio/x.wav", {"source": str(source), "bytes": 3})],
                                               report, state, workers=1)
                except BaseException as exc:
                    errors.append(exc)

            with patch.object(stage_data, "copy_one", blocked_copy):
                thread = threading.Thread(target=producer)
                thread.start()
                try:
                    self.assertTrue(started.wait(3))
                    snapshot = json.loads(report.read_text(encoding="utf-8"))
                    self.assertEqual((snapshot["status"], snapshot["staged_files"]), ("TRAINING_READY", 0))
                    gate = StagingGate(snapshot, timeout=.02, poll=.001)
                    gate.wait_for_batch(0, 0)
                    gate.wait_for_batch(0, 5000)
                    gate.wait_for_batch(1, 0)
                    self.assertFalse(gate.complete)
                    wait_for_start(report, root, os.getpid(), timeout=.02, poll=.001)
                    self.assertFalse((root / "audio/x.wav").exists())
                    release.set()
                    thread.join(3)
                    gate.wait_for_batch(1, 20)
                    self.assertTrue(gate.complete)
                    self.assertEqual((root / "audio/x.wav").read_bytes(), b"raw")
                    self.assertEqual(json.loads(report.read_text())["status"], "PASS")
                finally:
                    release.set()
                    thread.join(5)
            self.assertFalse(errors)

    def test_shared_path_remains_selected_until_explicit_boundary_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, local = root / "original.wav", root / "audio/x.wav"
            source.write_bytes(b"same bytes")
            row = {"s1_raw_paths": ["audio/x.wav"], "s1_original_paths": [str(source)]}
            paths = AudioPaths(root, "shared_until_complete")
            self.assertEqual(paths.resolve(row), [source])
            local.parent.mkdir()
            local.with_name("x.wav.partial").write_bytes(b"same")
            self.assertEqual(paths.resolve(row), [source])
            stage_data.copy_one(root, ("audio/x.wav", {"source": str(source), "bytes": 10}))
            self.assertEqual(paths.resolve(row), [source])
            paths.use_local()
            source.unlink()
            self.assertEqual(paths.resolve(row), [local])
            self.assertEqual(local.read_bytes(), b"same bytes")
            self.assertEqual(paths.read_files, {"shared": 3, "local": 1})
            with self.assertRaisesRegex(ValueError, "escaped"):
                paths.resolve({"s1_raw_paths": ["../outside.wav"]})

    def test_copy_failure_dead_process_and_stale_heartbeat_still_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, state = fixture(root)
            gate = StagingGate(prepared)
            with patch("data.staging.process_alive", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "exited before completion"):
                    gate.wait_for_batch(0, 0)
            write_json(root / "STAGING.json", dict(state, heartbeat=time.time() - 121))
            with self.assertRaisesRegex(RuntimeError, "heartbeat is stale"):
                gate.wait_for_batch(1, 0)
            write_json(root / "STAGING.json", dict(state, phase="FAILED", error="I/O failed"))
            with self.assertRaisesRegex(RuntimeError, "I/O failed"):
                gate.wait_for_batch(0, 0)
            write_json(root / "STAGING.json", dict(state, phase="COMPLETE", copied_files=0, copied_bytes=0))
            with self.assertRaisesRegex(RuntimeError, "counts do not match"):
                gate.wait_for_batch(0, 0)

    def test_final_report_requires_full_copy_even_in_shared_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, state = fixture(root)
            gate = StagingGate(prepared, timeout=.01, poll=.001)
            with self.assertRaises(TimeoutError):
                gate.wait_complete()
            write_json(root / "STAGING.json", dict(state, phase="COMPLETE", copied_files=1, copied_bytes=3))
            gate.wait_complete()
            self.assertTrue(gate.complete)

    def test_source_identity_change_is_rejected_and_partial_copy_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audio").mkdir()
            source = root / "source.wav"
            source.write_bytes(b"raw")
            info = {"source": str(source), "bytes": 3, "mtime_ns": source.stat().st_mtime_ns}
            os.utime(source, ns=(info["mtime_ns"] + 10000000, info["mtime_ns"] + 10000000))
            with self.assertRaisesRegex(IOError, "source changed"):
                stage_data.copy_one(root, ("audio/x.wav", info))
            self.assertFalse((root / "audio/x.wav").exists())
            self.assertFalse((root / "audio/x.wav.partial").exists())

    def test_cpu_index_reuses_mapping_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            source.write_bytes(b"raw")
            manifest, resolutions = root / "s1.json", root / "resolved.tsv"
            manifest.write_text(json.dumps([{"source_dataset": "Clotho", "audio_path": "one.wav"}]))
            resolutions.write_text("field\traw_path\tstatus\tresolved_path\n"
                                   f"audio_path\tone.wav\texists\t{source}\n")
            original_prepare = stage_data.prepare

            def small_prepare(*args):
                report, files = original_prepare(*args, strict_counts=False)
                # Keep production row/source validation enabled in load_index.
                report.update(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS)
                return report, files

            with patch.object(prepare_index, "prepare", small_prepare), contextlib.redirect_stdout(io.StringIO()):
                index = prepare_index.ensure_index(manifest, resolutions, root / "cache")
            data = load_index(index, manifest, resolutions)
            augmented = json.loads(Path(data["metadata"]).read_text())
            self.assertEqual(augmented["s1_original_paths"], [str(source.resolve())])
            files = json.loads(Path(data["files_path"]).read_text())
            self.assertEqual(next(iter(files.values()))["mtime_ns"], source.stat().st_mtime_ns)
            with patch.object(prepare_index, "prepare", side_effect=AssertionError("must not rescan audio")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prepare_index.ensure_index(manifest, resolutions, root / "cache"), index)
            with patch.object(prepare_index, "code_digest", return_value="different-code"), \
                    patch("data.prepared_index.code_digest", return_value="different-code"), \
                    patch.object(prepare_index, "prepare", small_prepare), contextlib.redirect_stdout(io.StringIO()):
                # A code change selects another cache entry rather than overwriting.
                self.assertNotEqual(prepare_index.ensure_index(manifest, resolutions, root / "cache"), index)
            manifest.write_text(manifest.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "changed since CPU"):
                load_index(index, manifest, resolutions)

    def test_formal_config_is_created_with_zero_staged_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, _ = fixture(root)
            prepared.update(mode="formal", rows=1311334, source_rows=SOURCE_COUNTS,
                            rows_canonical_sha256="rows", mapping_sha256="mapping")
            text, htsat = root / "text", root / "htsat"
            text.mkdir()
            htsat.mkdir()
            (text / "model.safetensors").write_bytes(b"weights")
            (htsat / "HTSAT_AudioSet_Saved_1.ckpt").write_bytes(b"weights")
            report, output = root / "report.json", root / "runtime.json"
            write_json(report, prepared)
            argv = ["write_config.py", "--mode", "formal", "--prepared", str(report), "--output", str(output),
                    "--run-dir", str(root), "--text-model-dir", str(text), "--htsat-root", str(htsat)]
            with patch.object(sys, "argv", argv), patch.object(write_config.importlib.metadata, "version", return_value="fixture"), \
                    contextlib.redirect_stdout(io.StringIO()):
                write_config.main()
            config = json.loads(output.read_text())
            self.assertEqual(config["max_optimizer_steps"], 15366)
            self.assertEqual(config["staging"]["initial_steps"], 0)
            self.assertEqual(config["staging"]["read_policy"], "shared_until_complete")

    def test_switch_is_collective_before_loading_all_four_microbatches(self):
        source = (ROUTE / "train.py").read_text(encoding="utf-8")
        switch = source.index("dataset.audio_paths.use_local()")
        self.assertLess(source.index("dist.all_reduce(ready, op=dist.ReduceOp.MIN)"), switch)
        self.assertLess(switch, source.index("for micro_index in range(ACCUMULATION_STEPS)"))
        self.assertLess(switch, source.index("batch = next(iterator)"))
        runner = (ROUTE / "run.sh").read_text(encoding="utf-8")
        launcher = (ROOT / "code/RSmol/run_mizar_s1_gbs256_5090.sh").read_text(encoding="utf-8")
        self.assertNotIn("--start-steps", runner)
        self.assertLess(launcher.index("/prepare_index.py"), launcher.index("vc submit"))
        self.assertIn("STAGE_WORKERS=4", launcher)

    def test_runner_starts_real_training_during_copy_and_cleans_on_failure(self):
        bash = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
        if not Path(bash).is_file():
            self.skipTest("Bash unavailable")

        def bash_path(path):
            value = path.as_posix()
            return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value

        python_stub = r'''#!/usr/bin/env bash
set -eu
case "$1" in
  *stage_data.py)
    if [[ "$2" == --cleanup ]]; then
      if [[ -f "$RSM_TEST_DIR/producer_pid" ]] && kill -0 "$(cat "$RSM_TEST_DIR/producer_pid")" 2>/dev/null; then
        exit 9
      fi
      touch "$RSM_TEST_DIR/cleaned"
      exit 0
    fi
    [[ " $* " == *" --prepared-index "* && " $* " != *" --start-steps "* ]]
    echo $$ > "$RSM_TEST_DIR/producer_pid"
    touch "$RSM_TEST_DIR/index_loaded"
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
      [[ ! -f "$RSM_TEST_DIR/index_loaded" ]] || exit 0
      sleep .01
    done
    exit 9
    ;;
  write_config.py) [[ "$RSM_TEST_FAIL" != 1 ]] || exit 7 ;;
  -) [[ -f "$RSM_TEST_DIR/training_done" ]]; cat > /dev/null ;;
  *) exit 9 ;;
esac
'''
        torchrun_stub = r'''#!/usr/bin/env bash
set -eu
[[ -f "$RSM_TEST_DIR/index_loaded" && ! -f "$RSM_TEST_DIR/complete" ]]
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
                (root / "resolved.tsv").write_text("fixture\n")
                (root / "index").mkdir()
                env = dict(os.environ, MELLOW_CONDA_BASE=bash_path(root), RSM_TEST_DIR=bash_path(root),
                           RSM_TEST_FAIL=str(int(fail)), PATH=f"{bash_path(root / 'bin')}:/usr/bin:/bin")
                result = subprocess.run([bash, bash_path(script), "formal", bash_path(root / "resolved.tsv"),
                                         bash_path(root / "output"), "", bash_path(root), bash_path(root),
                                         bash_path(root / "index"), "4"], env=env, capture_output=True,
                                        text=True, encoding="utf-8", errors="replace", timeout=20)
                self.assertEqual(result.returncode, 7 if fail else 0, result.stdout + result.stderr)
                self.assertTrue((root / "cleaned").exists(), result.stdout + result.stderr)
                self.assertEqual((root / "training_done").exists(), not fail)


if __name__ == "__main__":
    unittest.main()
