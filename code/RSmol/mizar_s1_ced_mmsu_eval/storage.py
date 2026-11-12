"""Durable raw generations, exact resume identity and a process lifetime lock."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def output_lock(directory):
    """OS lock automatically releases on crash. Never delete the lock inode."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".mmsu.lock"
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"MMSU output directory is in use: {directory}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def claim_output(directory, identity):
    path = directory / "run_config.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError("MMSU output belongs to another checkpoint/data/scorer/code/protocol; use a new directory")
    elif any(p.name != ".mmsu.lock" for p in directory.iterdir()):
        raise FileExistsError(f"refusing nonempty unowned MMSU output: {directory}")
    else:
        write_json(path, identity)


def recover_jsonl(path):
    """Only a non-newline-terminated final write can be discarded after a crash."""
    if not path.exists():
        return []
    records, offset = [], 0
    with path.open("r+b") as stream:
        while line := stream.readline():
            if not line.endswith(b"\n"):
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
                break
            try:
                records.append(json.loads(line))
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"corrupt complete JSONL record in {path} at byte {offset}") from exc
            offset += len(line)
    return records


class GenerationStore:
    def __init__(self, directory, records):
        self.path = directory / "raw_generations.jsonl"
        self.progress = directory / "progress.jsonl"
        self.state = recover_jsonl(self.path)
        # Raw is authoritative if a process died between the two appends.
        recover_jsonl(self.progress)
        if len(self.state) > len(records):
            raise ValueError("too many saved MMSU generations")
        for index, saved in enumerate(self.state):
            if (saved.get("row_index") != index or saved.get("id") != records[index]["id"]
                    or saved.get("official_record") != records[index]
                    or saved.get("record_sha256") != object_sha(records[index])
                    or saved.get("generation_sha256") != object_sha(saved.get("generation"))):
                raise ValueError(f"saved MMSU row identity/content mismatch at index {index}")
            if not isinstance(saved["generation"].get("generated_text"), str):
                raise ValueError(f"saved MMSU response is not text at index {index}")

    @staticmethod
    def append_line(path, value):
        with path.open("ab") as stream:
            stream.write((json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())

    def append(self, index, record, generation):
        if index != len(self.state) or not isinstance(generation.get("generated_text"), str):
            raise ValueError("MMSU generations must be contiguous rows with a raw text response")
        saved = {"row_index": index, "id": record["id"], "official_record": record,
                 "record_sha256": object_sha(record), "generation": generation,
                 "generation_sha256": object_sha(generation)}
        self.append_line(self.path, saved)
        self.state.append(saved)
        self.append_line(self.progress, {"row_index": index, "id": record["id"], "status": "generated"})
