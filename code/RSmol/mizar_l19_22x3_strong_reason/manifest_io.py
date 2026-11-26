import gzip,json
from pathlib import Path
from typing import Any,Iterator
CHUNK_SIZE=1024*1024

def _open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")

def _iter_json_array(handle, first_chunk: str) -> Iterator[Any]:
    """Stream a JSON array without materialising the 1.5M-row manifest."""

    decoder = json.JSONDecoder()
    buffer = first_chunk
    position = 0
    ended = False

    def refill() -> bool:
        nonlocal buffer, position
        extra = handle.read(CHUNK_SIZE)
        if extra:
            buffer += extra
            return True
        return False

    while True:
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            if not refill():
                raise ValueError("unexpected end of manifest JSON array")
        if not ended:
            if buffer[position] != "[":
                raise ValueError("manifest is not a JSON array")
            position += 1
            ended = True
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if not refill():
                    raise ValueError("unexpected end of manifest JSON array")
            if buffer[position] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if not refill():
                    raise ValueError("invalid or truncated JSON array")
                continue
            position = end
            yield value
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if not refill():
                    raise ValueError("unexpected end after JSON row")
            if buffer[position] == ",":
                position += 1
            elif buffer[position] == "]":
                return
            else:
                raise ValueError("expected ',' or ']' after JSON row")
            # Compact consumed data so a long manifest does not grow in RAM.
            if position > CHUNK_SIZE:
                buffer = buffer[position:]
                position = 0

def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Read a JSON array, JSONL file, or their gzip-compressed equivalent."""

    with _open_text(path) as handle:
        first = handle.read(1)
        while first and first.isspace():
            first = handle.read(1)
        if not first:
            return
        if first == "[":
            for row in _iter_json_array(handle, first):
                if not isinstance(row, dict):
                    raise ValueError("manifest contains a non-object row")
                yield row
            return
        first_line = first + handle.readline()
        for line_number, line in enumerate(first_line.splitlines(), start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                yield row
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            yield row
