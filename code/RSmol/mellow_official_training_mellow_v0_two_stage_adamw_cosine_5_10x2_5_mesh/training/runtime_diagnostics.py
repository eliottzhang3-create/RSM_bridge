"""Opt-in phase timings for the existing training loop; no training-state changes."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from collections import defaultdict

import torch


class StepRuntimeDiagnostics:
    """Measure a short window after ten startup steps on every rank.

    CUDA events use the training stream. A single stream synchronization at the
    end of a measured step makes those events readable; there is no additional
    synchronization between phases. Timings include stream waits, rather than
    claiming to measure kernel execution alone. All diagnostics are off unless
    MELLOW_RUNTIME_DIAGNOSTICS_STEPS is a positive integer.
    """

    def __init__(self, device, distributed, logger, *, dataset, data_json, output_path):
        raw_steps = os.environ.get("MELLOW_RUNTIME_DIAGNOSTICS_STEPS", "0")
        if not raw_steps.isdecimal() or not 0 <= int(raw_steps) <= 1000:
            raise ValueError("MELLOW_RUNTIME_DIAGNOSTICS_STEPS must be an integer in [0, 1000]")
        self.steps = int(raw_steps)
        self.device = device
        self.distributed = distributed
        self.logger = logger
        self.output_path = Path(output_path)
        self.data_source = self._data_source(dataset, data_json) if self.steps else {}
        self.cuda = device.type == "cuda"
        self.seen = 0
        self.active = False
        self.sample_count = 0
        self.step_ms_total = 0.0
        self.phase_wall_total = defaultdict(float)
        self.phase_stream_total = defaultdict(float)
        self.phase_names = set()
        self.initial_memory = self._memory() if self.steps else {}
        self.window_start_memory = {}
        if self.steps and distributed.rank() == 0:
            logger.info("Runtime diagnostics: skip 10 steps, measure the next %d; report once", self.steps)
            logger.info("Runtime diagnostics data source: %s", json.dumps(self.data_source, sort_keys=True))

    @staticmethod
    def _mount(path):
        target = os.path.realpath(path)
        selected = None
        with open("/proc/self/mountinfo", encoding="utf-8") as handle:
            for line in handle:
                before, separator, after = line.partition(" - ")
                if not separator:
                    continue
                fields = before.split()
                mount_point = fields[4].replace("\\040", " ")
                if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
                    if selected is None or len(mount_point) > len(selected["mount_point"]):
                        mounted = after.split()
                        selected = {"mount_point": mount_point, "filesystem": mounted[0], "source": mounted[1]}
        return selected

    @classmethod
    def _data_source(cls, dataset, data_json):
        root = os.path.realpath(dataset.data_path)
        entries = dataset.all_data_json
        samples = []
        for i in range(min(64, len(entries))):
            entry = entries[i * len(entries) // min(64, len(entries))]
            for field in ("filepath1", "filepath2"):
                name = entry.get(field, "")
                if not name:
                    continue
                path = os.path.realpath(os.path.join(root, name))
                samples.append({
                    "field": field, "raw_path": name, "resolved_path": path,
                    "under_stage_root": os.path.commonpath((root, path)) == root,
                    "exists": os.path.isfile(path),
                    "mount": cls._mount(path),
                })
                if len(samples) >= 16:
                    break
            if len(samples) >= 16:
                break
        return {
            "data_root": root, "data_root_mount": cls._mount(root),
            "data_json": os.path.realpath(data_json),
            "data_json_mount": cls._mount(data_json),
            "row_count": len(entries), "sampled_audio_paths": samples,
        }

    @staticmethod
    def _gpu_snapshot():
        command = ["nvidia-smi", "--query-gpu=index,uuid,name,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
            return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"error": repr(exc)}

    def _memory(self):
        if not self.cuda:
            return {}
        stats = torch.cuda.memory_stats(self.device)
        return {
            "allocated_mib": torch.cuda.memory_allocated(self.device) / 2**20,
            "reserved_mib": torch.cuda.memory_reserved(self.device) / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(self.device) / 2**20,
            "allocation_retries": stats.get("num_alloc_retries", 0),
            "oom_count": stats.get("num_ooms", 0),
        }

    def begin_step(self):
        if not self.steps:
            return
        self.seen += 1
        self.active = 10 < self.seen <= 10 + self.steps
        if self.active:
            if self.seen == 11:
                self.window_start_memory = self._memory()
            self.started = time.perf_counter()
            self.wall = defaultdict(float)
            self.events = defaultdict(list)

    def phase(self, name):
        if not self.active:
            return contextlib.nullcontext()
        return self._measure(name)

    @contextlib.contextmanager
    def _measure(self, name):
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(self.device))
        started = time.perf_counter()
        try:
            yield
        finally:
            self.wall[name] += time.perf_counter() - started
            if self.cuda:
                end.record(torch.cuda.current_stream(self.device))
                self.events[name].append((start, end))

    def end_step(self):
        if not self.active:
            return
        if self.cuda:
            torch.cuda.current_stream(self.device).synchronize()
        step_ms = (time.perf_counter() - self.started) * 1000
        stream_ms = {
            name: sum(start.elapsed_time(end) for start, end in pairs)
            for name, pairs in self.events.items()
        }
        self.sample_count += 1
        self.step_ms_total += step_ms
        for name, duration in self.wall.items():
            self.phase_names.add(name)
            self.phase_wall_total[name] += duration * 1000
            self.phase_stream_total[name] += stream_ms.get(name, 0.0)
        self.events.clear()
        self.active = False
        if self.sample_count != self.steps:
            return

        phase_names = sorted(self.phase_names)
        phases = {
            name: {
                "wall_ms": self.phase_wall_total[name] / self.steps,
                "stream_ms": self.phase_stream_total[name] / self.steps,
            }
            for name in phase_names
        }
        memory = self._memory()
        for key in ("allocation_retries", "oom_count"):
            if memory:
                memory[key + "_during_window"] = memory[key] - self.window_start_memory[key]
        payload = {
            "rank": self.distributed.rank(),
            "hostname": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(self.device) if self.cuda else "cpu",
            "gpu_total_mib": torch.cuda.get_device_properties(self.device).total_memory / 2**20 if self.cuda else None,
            "torch_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mean_step_ms": self.step_ms_total / self.steps,
            "phases": phases,
            "memory_after_initialization": self.initial_memory,
            "memory_after_window": memory,
            "data_source": self.data_source,
        }
        ranks = self.distributed.all_gather_object(payload)
        if self.distributed.rank() == 0:
            summary = {
                "contract": "mellow_two_stage_runtime_diagnostics_v1",
                "measured_steps_this_run": [11, 10 + self.steps],
                "slowest_rank_mean_step_ms": max(rank["mean_step_ms"] for rank in ranks),
                "phase_max_rank_mean_ms": {
                    name: {
                        metric: max(rank["phases"][name][metric] for rank in ranks)
                        for metric in ("wall_ms", "stream_ms")
                    }
                    for name in phase_names
                },
                "ranks": ranks,
            }
            self.logger.info("Runtime diagnostics GPU snapshot: %s", json.dumps(self._gpu_snapshot(), sort_keys=True))
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.logger.info("Runtime diagnostics report: %s", self.output_path)
            self.logger.info("Runtime diagnostics: %s", json.dumps(summary, sort_keys=True))
