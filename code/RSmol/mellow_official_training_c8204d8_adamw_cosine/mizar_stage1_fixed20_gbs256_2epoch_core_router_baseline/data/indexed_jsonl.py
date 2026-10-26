"""Read a shared JSONL manifest without duplicating 1.3M Python dicts per rank."""
import json
import mmap
import struct
from pathlib import Path


class IndexedRows:
    def __init__(self, path):
        self.path = Path(path)
        self._data_file = self.path.open("rb")
        self._index_file = Path(str(self.path) + ".idx").open("rb")
        size = Path(str(self.path) + ".idx").stat().st_size
        if size == 0 or size % 8:
            raise ValueError("invalid JSONL offset index")
        self._data = mmap.mmap(self._data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self._index = mmap.mmap(self._index_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.count = size // 8

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        start = struct.unpack_from("<Q", self._index, index * 8)[0]
        end = struct.unpack_from("<Q", self._index, (index + 1) * 8)[0] if index + 1 < self.count else len(self._data)
        return json.loads(self._data[start:end])

    def close(self):
        self._data.close()
        self._index.close()
        self._data_file.close()
        self._index_file.close()
