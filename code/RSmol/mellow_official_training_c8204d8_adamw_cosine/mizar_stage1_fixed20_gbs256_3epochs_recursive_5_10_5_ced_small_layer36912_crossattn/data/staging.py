"""CPU-only readiness protocol between the raw-byte copier and all DDP ranks."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def process_alive(pid):
    if os.name == "nt":
        # Windows os.kill(pid, 0) can terminate the process; use a query handle.
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5  # access denied, but it exists
        try:
            code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A terminated child can remain a zombie until the shell reaps it.
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        if state == "Z":
            return False
    except (OSError, IndexError):
        pass
    return True


class StagingGate:
    def __init__(self, prepared, *, log=False, timeout=600., poll=.2):
        self.info = prepared.get("async_staging")
        self.complete = self.info is None
        self.read_policy = prepared.get("read_policy", "staged_only")
        if self.read_policy not in {"staged_only", "shared_until_complete"}:
            raise ValueError("unknown staging read policy")
        self.log, self.timeout, self.poll = log, timeout, poll
        self.wait_seconds = 0.
        self.last_ready_batch = 0
        if prepared.get("status") not in {"PASS", "TRAINING_READY"}:
            raise ValueError("staging has not published a training-ready manifest")
        if self.info is None and prepared.get("status") != "PASS":
            raise ValueError("partial staging requires the asynchronous readiness protocol")
        if self.info:
            self.root = Path(prepared["stage_root"])
            self.status_path = self.root / "STAGING.json"
            if Path(self.info["status_path"]) != self.status_path:
                raise ValueError("staging status path does not match this job's root")
            owner = json.loads((self.root / ".owner.json").read_text(encoding="utf-8"))
            if owner != {"root": str(self.root), "token": self.info["token"]}:
                raise ValueError("staging ownership differs from the training-ready report")
            self.total_files = prepared["planned_files"]
            self.total_bytes = prepared["planned_bytes"]

    def _read(self):
        state = json.loads(self.status_path.read_text(encoding="utf-8"))
        for key in ("token", "producer_pid", "epoch", "start_batch"):
            if state.get(key) != self.info[key]:
                raise RuntimeError(f"staging identity changed: {key}")
        if state.get("phase") == "FAILED":
            raise RuntimeError(f"background staging failed: {state.get('error', 'unknown error')}")
        if state.get("phase") == "COMPLETE":
            if state.get("copied_files") != self.total_files or state.get("copied_bytes") != self.total_bytes:
                raise RuntimeError("background staging COMPLETE counts do not match the plan")
            self.complete = True
        elif not process_alive(self.info["producer_pid"]):
            raise RuntimeError("background staging process exited before completion")
        elif time.time() - state["heartbeat"] > 120:
            raise RuntimeError("background staging heartbeat is stale for more than 120 seconds")
        self.last_ready_batch = state["ready_through_batch"]
        return state

    def wait_for_batch(self, epoch, batch_index):
        """Wait BEFORE loading data or entering this batch's DDP collectives.

        ready_through_batch is an exclusive, rank-global batch watermark.
        Later epochs wait for the entire retained raw-file set, including files
        dropped from the first epoch's batches by drop_last.
        """
        if self.complete:
            return
        if self.read_policy == "shared_until_complete":
            # Check producer health, but training reads original shared files
            # until the trainer collectively selects the complete local copy.
            self._read()
            return
        start, next_log = time.monotonic(), 0.
        try:
            while True:
                state = self._read()
                if self.complete:
                    return
                if epoch < self.info["epoch"] or (epoch == self.info["epoch"] and batch_index < self.info["start_batch"]):
                    raise RuntimeError("training cursor precedes the staging plan")
                if epoch == self.info["epoch"] and batch_index < self.last_ready_batch:
                    return
                elapsed = time.monotonic() - start
                if elapsed > self.timeout:
                    raise TimeoutError(f"waited {elapsed:.1f}s for staging: epoch={epoch + 1}, batch={batch_index + 1}, state={state}")
                if self.log and elapsed >= next_log:
                    print(f"[staging-wait] epoch={epoch + 1} batch={batch_index + 1} "
                          f"ready_through_batch={self.last_ready_batch} copied={state['copied_files']:,}/{self.total_files:,} "
                          f"wait_seconds={elapsed:.1f}", flush=True)
                    next_log = elapsed + 15.
                time.sleep(self.poll)
        finally:
            self.wait_seconds += time.monotonic() - start

    def wait_complete(self):
        if self.info and self.read_policy == "shared_until_complete":
            start = time.monotonic()
            while not self.complete:
                self._read()
                if self.complete:
                    break
                if time.monotonic() - start > self.timeout:
                    raise TimeoutError("background audio copy did not complete before final report")
                if not self.complete:
                    time.sleep(self.poll)
            self.wait_seconds += time.monotonic() - start
        elif self.info:
            self.wait_for_batch(self.info["epoch"] + 1, 0)


def wait_for_start(report, root, producer_pid, *, timeout=3600., poll=.5):
    start, next_log = time.monotonic(), 0.
    while True:
        state_path = Path(root) / "STAGING.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        if state.get("phase") == "FAILED":
            raise RuntimeError(f"staging failed before training: {state.get('error')}")
        if Path(report).exists():
            prepared = json.loads(Path(report).read_text(encoding="utf-8"))
            # A resume can reuse its output directory, whose old report still
            # names the previous allocation. Wait for this producer to replace
            # it; never mistake the prior job's PASS for current readiness.
            if Path(prepared.get("stage_root", "")) == Path(root):
                if prepared.get("async_staging", {}).get("producer_pid") != producer_pid:
                    raise RuntimeError("training-ready report belongs to another staging process")
                gate = StagingGate(prepared)
                gate.wait_for_batch(gate.info["epoch"], gate.info["start_batch"])
                print(f"[staging] training can start: read_policy={gate.read_policy} prefill_steps={prepared['initial_steps']} "
                      f"ready_through_batch={gate.last_ready_batch} report={report}", flush=True)
                return
        if not process_alive(producer_pid):
            raise RuntimeError("staging process exited without publishing training-ready data")
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            raise TimeoutError(f"staging did not become training-ready within {timeout}s")
        if elapsed >= next_log:
            print(f"[staging-start] phase={state.get('phase', 'STARTING')} elapsed={elapsed:.1f}s", flush=True)
            next_log = elapsed + 30.
        time.sleep(poll)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--producer-pid", type=int, required=True)
    args = parser.parse_args()
    wait_for_start(args.report, args.stage_root, args.producer_pid)
