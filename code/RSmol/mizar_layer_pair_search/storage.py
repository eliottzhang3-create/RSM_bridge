"""Atomic identities, journal recovery and process locks, adapted in isolation."""
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def text_sha(path):
    return hashlib.sha256(Path(path).read_text(encoding='utf-8').encode('utf-8')).hexdigest()


def file_identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = file_sha(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'file changed while hashing: {path}')
    return {'path': str(path), 'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns, 'sha256': digest}


def verify_file(identity, *, full=True):
    path = Path(identity['path'])
    stat = path.stat()
    if (stat.st_size != identity['bytes'] or stat.st_mtime_ns != identity['mtime_ns']
            or (full and file_sha(path) != identity['sha256'])):
        raise ValueError(f'file identity changed: {path}')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def seal(value, key='sha256'):
    return {**value, key: object_sha(value)}


def verify_seal(value, key='sha256'):
    if value.get(key) != object_sha({k: v for k, v in value.items() if k != key}):
        raise ValueError(f'content hash mismatch: {key}')
    return value


@contextmanager
def output_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.pair-search.lock').open('a+b') as stream:
        if not stream.tell():
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'output is already in use: {directory}') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def claim(directory, identity):
    directory = Path(directory)
    path = directory / 'run_config.json'
    if path.exists():
        if read_json(path) != identity:
            raise ValueError('output belongs to a different data/model/code/protocol identity')
    elif any(p.name != '.pair-search.lock' for p in directory.iterdir()):
        raise FileExistsError(f'refusing unowned nonempty output: {directory}')
    else:
        write_json(path, identity)


def recover_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    result, offset = [], 0
    with path.open('r+b') as stream:
        while line := stream.readline():
            if not line.endswith(b'\n'):
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
                break
            try:
                item = json.loads(line)
                verify_seal(item)
            except (ValueError, UnicodeError, AttributeError) as exc:
                raise ValueError(f'corrupt complete journal record: {path} offset={offset}') from exc
            result.append(item)
            offset += len(line)
    return result


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(seal(value), ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
