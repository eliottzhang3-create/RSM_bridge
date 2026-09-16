#!/usr/bin/env python3
"""Read-only storage and memory discovery inside a submitted training container."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


NETWORK_FS = {
    "9p", "afs", "ceph", "cifs", "davfs", "fuse.ceph", "fuse.glusterfs",
    "fuse.sshfs", "gcsfuse", "glusterfs", "lustre", "nfs", "nfs4", "smb3",
}
MEMORY_FS = {"devtmpfs", "hugetlbfs", "ramfs", "tmpfs"}
LOCAL_FS = {"btrfs", "ext2", "ext3", "ext4", "f2fs", "xfs", "zfs"}
DEFAULT_CANDIDATES = (
    "/",
    "/tmp",
    "/var/tmp",
    "/dev/shm",
    "/scratch",
    "/local",
    "/local_nvme",
    "/mnt/local",
    "/mnt/nvme",
    "/workspace",
    "/hpc_stor03",
    "/hpc_stor03/sjtu_home/jinwei.zhang",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-path", action="append", default=[])
    parser.add_argument("--minimum-local-free-gib", type=float, default=150.0)
    parser.add_argument("--expected-cache-gib", type=float, default=115.30)
    return parser.parse_args()


def _run(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        return {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": repr(exc)}


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _scalar(value: str | None) -> int | str | None:
    if value is None:
        return None
    if value == "max":
        return value
    try:
        return int(value)
    except ValueError:
        return value


def _mount_for(path: Path) -> dict[str, Any]:
    result = _run(["findmnt", "--json", "--target", str(path), "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"])
    if result["returncode"] != 0:
        return {"target": None, "source": None, "fstype": None, "options": None, "error": result["stderr"].strip()}
    try:
        filesystems = json.loads(result["stdout"]).get("filesystems", [])
        item = filesystems[0] if filesystems else {}
        return {
            "target": item.get("target"),
            "source": item.get("source"),
            "fstype": item.get("fstype"),
            "options": item.get("options"),
            "error": None,
        }
    except (IndexError, json.JSONDecodeError) as exc:
        return {"target": None, "source": None, "fstype": None, "options": None, "error": repr(exc)}


def _storage_kind(mount: dict[str, Any]) -> str:
    fstype = str(mount.get("fstype") or "").lower()
    source = str(mount.get("source") or "")
    if fstype in MEMORY_FS:
        return "memory_tmpfs"
    if fstype == "overlay":
        return "container_overlay"
    if fstype in NETWORK_FS or fstype.startswith("fuse."):
        return "network_filesystem"
    if fstype in LOCAL_FS and (source.startswith("/dev/") or source.startswith("UUID=")):
        return "local_block_device"
    if fstype in LOCAL_FS:
        return "local_filesystem_unconfirmed_source"
    return "unknown"


def _candidate(path_text: str, minimum_free_bytes: int) -> dict[str, Any]:
    path = Path(path_text)
    payload: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_directory": path.is_dir(),
        "writable_by_access_check": os.access(path, os.W_OK) if path.exists() else False,
    }
    if not path.exists() or not path.is_dir():
        payload.update({"mount": None, "storage_kind": "missing", "eligible_for_full_cache": False})
        return payload

    usage = shutil.disk_usage(path)
    mount = _mount_for(path)
    kind = _storage_kind(mount)
    payload.update({
        "mount": mount,
        "storage_kind": kind,
        "capacity_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "capacity_gib": float(usage.total / 1024**3),
        "used_gib": float(usage.used / 1024**3),
        "free_gib": float(usage.free / 1024**3),
        "eligible_for_full_cache": bool(
            kind == "local_block_device"
            and payload["writable_by_access_check"]
            and int(usage.free) >= int(minimum_free_bytes)
        ),
        "exclusion_reason": None,
    })
    if kind == "memory_tmpfs":
        payload["exclusion_reason"] = "RAM-backed tmpfs; cache bytes count against job memory"
    elif kind == "container_overlay":
        payload["exclusion_reason"] = "container writable layer; capacity/lifetime is not a guaranteed scratch contract"
    elif kind == "network_filesystem":
        payload["exclusion_reason"] = "network/shared filesystem; not a node-local staging target"
    elif not payload["writable_by_access_check"]:
        payload["exclusion_reason"] = "not writable by the submitted process"
    elif int(usage.free) < int(minimum_free_bytes):
        payload["exclusion_reason"] = "less free space than the configured safety threshold"
    elif kind == "local_filesystem_unconfirmed_source":
        payload["exclusion_reason"] = "local filesystem type, but findmnt did not confirm a block-device source"
    elif kind == "unknown":
        payload["exclusion_reason"] = "filesystem type/source is not recognized as a local disk"
    return payload


def _unique_candidates(extra: list[str]) -> list[str]:
    values = list(DEFAULT_CANDIDATES)
    for name in ("TMPDIR", "TMP", "TEMP", "SCRATCH", "LOCAL_SCRATCH", "JOB_TMPDIR"):
        value = os.environ.get(name)
        if value:
            values.append(value)
    values.extend(extra)
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = os.path.abspath(os.path.expanduser(value))
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _raw_diagnostics() -> list[dict[str, Any]]:
    commands = [
        ["df", "-hT"],
        ["df", "-ih"],
        ["findmnt", "--real", "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"],
        ["lsblk", "--bytes", "--output", "NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS"],
        ["bash", "-lc", "ulimit -a"],
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
    ]
    return [_run(command) for command in commands]


def main() -> int:
    args = parse_args()
    if args.minimum_local_free_gib <= 0 or args.expected_cache_gib <= 0:
        raise ValueError("cache and free-space sizes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    minimum_free_bytes = int(args.minimum_local_free_gib * 1024**3)
    candidates = [_candidate(path, minimum_free_bytes) for path in _unique_candidates(args.candidate_path)]
    raw = _raw_diagnostics()

    cgroup_paths = (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    )
    proc_paths = (
        "/proc/meminfo",
        "/proc/mounts",
        "/proc/sys/vm/max_map_count",
        "/proc/sys/vm/overcommit_memory",
        "/proc/sys/vm/overcommit_ratio",
    )
    eligible = [item["path"] for item in candidates if item.get("eligible_for_full_cache")]
    report = {
        "stage": "audio_storage_probe_5090",
        "status": "PASS",
        "timestamp_unix": time.time(),
        "read_only_probe": True,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "uid": os.getuid() if hasattr(os, "getuid") else None,
        "expected_cache_gib": float(args.expected_cache_gib),
        "minimum_local_free_gib": float(args.minimum_local_free_gib),
        "candidate_paths": candidates,
        "eligible_local_stage_paths": eligible,
        "decision": (
            "LOCAL_STAGE_CANDIDATE_FOUND"
            if eligible else
            "NO_CONFIRMED_LOCAL_PATH_WITH_REQUIRED_FREE_SPACE"
        ),
        "environment": {
            key: value for key, value in sorted(os.environ.items())
            if any(token in key.upper() for token in ("TMP", "SCRATCH", "LOCAL", "JOB", "CUDA_VISIBLE", "NVIDIA_VISIBLE"))
        },
        "cgroup_memory": {path: _scalar(_read_text(path)) for path in cgroup_paths},
        "vm": {path: _scalar(_read_text(path)) for path in proc_paths[2:]},
        "raw_diagnostics_file": str(args.output_dir / "raw_diagnostics.txt"),
        "notes": [
            "The vc -m 256G request is job memory, not local disk capacity.",
            "A file-backed mmap does not reserve the mapped file size as resident RAM.",
            "tmpfs/dev/shm is RAM-backed and is never selected as the full-cache staging target.",
            "overlay is reported but never selected because its capacity and lifecycle are not a stable scratch contract.",
            "Free local capacity is dynamic; the eventual training launcher must repeat the capacity check before staging.",
        ],
    }

    raw_lines = []
    for item in raw:
        raw_lines.extend([
            f"===== {' '.join(item['command'])} =====",
            f"returncode: {item['returncode']}",
            item["stdout"].rstrip(),
            item["stderr"].rstrip(),
            "",
        ])
    for path in proc_paths[:2]:
        raw_lines.extend([f"===== {path} =====", _read_text(path) or "<unavailable>", ""])
    (args.output_dir / "raw_diagnostics.txt").write_text("\n".join(raw_lines), encoding="utf-8")
    report_path = args.output_dir / "storage_probe_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("======== AUDIO STORAGE PROBE ========")
    print(f"report: {report_path}")
    print(f"hostname: {report['hostname']}")
    print(f"decision: {report['decision']}")
    print(f"expected_cache_gib: {report['expected_cache_gib']}")
    print(f"minimum_local_free_gib: {report['minimum_local_free_gib']}")
    for item in candidates:
        if not item["exists"]:
            continue
        mount = item.get("mount") or {}
        print(
            f"path={item['path']} kind={item['storage_kind']} "
            f"fstype={mount.get('fstype')} source={mount.get('source')} "
            f"free_gib={item.get('free_gib')} writable={item['writable_by_access_check']} "
            f"eligible={item.get('eligible_for_full_cache')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
