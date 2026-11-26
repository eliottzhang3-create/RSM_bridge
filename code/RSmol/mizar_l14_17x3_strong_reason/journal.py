"""Recover the metrics cursor without losing superseded or partial bytes."""
import json
import os
from pathlib import Path
import time


def recover(path, step):
    path = Path(path)
    if not path.exists():
        if step:
            raise ValueError("resume checkpoint has no metrics journal")
        return
    data = path.read_bytes()
    previous, keep = 0, 0
    offset = 0
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break  # A killed write is preserved in the backup below.
        record = json.loads(line)
        cursor = record.get("step")
        if type(cursor) is not int or cursor != previous + 1:
            raise ValueError("metrics journal has duplicate, missing, or reordered complete steps")
        previous = cursor
        offset += len(line)
        if cursor <= step:
            keep = offset
    if previous < step:
        raise ValueError("metrics journal ends before the checkpoint cursor")
    if not step and data:
        raise ValueError("fresh training requires an empty journal")
    if keep != len(data):
        backup = path.with_name(path.name + f".before-resume-{time.time_ns()}")
        with backup.open("xb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        temporary = path.with_name(path.name + ".recovering")
        with temporary.open("wb") as handle:
            handle.write(data[:keep]); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
