"""Strict streaming metadata I/O and no-replace dataset publication."""
from __future__ import annotations

import ctypes
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


DECODER = json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_constant)


def loads(value):
    return json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    result = {"path": str(path), "bytes": before.st_size, "sha256": sha256_file(path)}
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"input changed while hashing: {path}")
    return result


def iter_rows(path):
    """JSON arrays (including pretty JSON) or JSONL; reject incomplete/trailing data."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        first = handle.read(1)
        while first and first.isspace():
            first = handle.read(1)
        if not first:
            raise ValueError(f"empty metadata: {path}")
        if first != "[":
            for number, line in enumerate(iter([first + handle.readline()]), 1):
                row = loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"non-object row {number}: {path}")
                yield row
            for number, line in enumerate(handle, 2):
                if not line.strip():
                    continue
                row = loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"non-object row {number}: {path}")
                yield row
            return
        buffer, pos, eof, initial = "", 0, False, True

        def more():
            nonlocal buffer, eof
            block = handle.read(1024 * 1024)
            buffer += block
            eof = not block

        def whitespace():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) or eof:
                    return
                more()

        while True:
            whitespace()
            if initial and pos < len(buffer) and buffer[pos] == "]":
                pos += 1
                break
            initial = False
            while True:
                try:
                    row, end = DECODER.raw_decode(buffer, pos)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"invalid/truncated JSON array: {path}") from None
                    more()
            if not isinstance(row, dict):
                raise ValueError(f"non-object array row: {path}")
            pos = end
            yield row
            whitespace()
            if pos >= len(buffer):
                raise ValueError(f"unterminated JSON array: {path}")
            char = buffer[pos]
            pos += 1
            if char == "]":
                break
            if char != ",":
                raise ValueError(f"expected comma or closing bracket: {path}")
            if pos > 1024 * 1024:
                buffer, pos = buffer[pos:], 0
        if (buffer[pos:] + handle.read()).strip():
            raise ValueError(f"trailing content after JSON array: {path}")


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def sync(handle):
    handle.flush()
    os.fsync(handle.fileno())


def publish(staging, output):
    """Publish beside staging without replacing an existing output, including CloudStor."""
    staging, output = Path(staging).resolve(strict=True), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output: {output}")
    output = output.parent.resolve(strict=True) / output.name
    if staging.parent != output.parent or not staging.is_dir():
        raise ValueError("publication requires sibling directories")
    if sys.platform == "win32":
        os.rename(staging, output)
        return
    if not sys.platform.startswith("linux"):
        raise RuntimeError("publication supports Linux and Windows")
    libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1) == 0:
            return
        error = ctypes.get_errno()
        if error not in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP}:
            raise OSError(error, os.strerror(error), str(output))
    # Claim a new empty target. Ordinary POSIX rename supports CloudStor mounts
    # which do not implement RENAME_NOREPLACE. Never remove another owner's data.
    output.mkdir(mode=0o700)
    owner = output.lstat()
    try:
        current = output.lstat()
        if output.is_symlink() or (current.st_dev, current.st_ino) != (owner.st_dev, owner.st_ino):
            raise FileExistsError(f"publication reservation replaced: {output}")
        os.rename(staging, output)
    except BaseException:
        current = output.lstat()
        if not output.is_symlink() and (current.st_dev, current.st_ino) == (owner.st_dev, owner.st_ino):
            try:
                output.rmdir()
            except OSError:
                pass
        raise
