"""Atomic metadata, durable sequential raw records, crash-safe process lock."""
from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def text_sha(path):
    return hashlib.sha256(Path(path).read_text(encoding='utf-8').encode('utf-8')).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_jsonl(path, values):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def output_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.reasonaqa.lock').open('a+b') as stream:
        if stream.tell() == 0:
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
            raise RuntimeError(f'ReasonAQA output in use: {directory}') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def claim_output(directory, identity):
    path = directory / 'run_config.json'
    if path.is_file():
        if json.loads(path.read_text(encoding='utf-8')) != identity:
            raise ValueError('ReasonAQA checkpoint/data/audio/code/protocol identity changed; use a new output directory')
    elif any(p.name != '.reasonaqa.lock' for p in directory.iterdir()):
        raise FileExistsError(f'refusing nonempty unowned output: {directory}')
    else:
        write_json(path, identity)


def recover_jsonl(path):
    if not path.exists():
        return []
    records, offset = [], 0
    with path.open('r+b') as stream:
        while line := stream.readline():
            if not line.endswith(b'\n'):
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
                break
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError('record is not an object')
                records.append(value)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f'corrupt complete raw record at byte {offset}: {path}') from exc
            offset += len(line)
    return records


def validate_generation(generation, record):
    if not isinstance(generation, dict):
        raise ValueError('generation must be an object')
    text, ids = generation.get('generated_text'), generation.get('generated_token_ids')
    tokens, audio = generation.get('prompt_token_ids'), generation.get('audio_half_sha256')
    if (not isinstance(text, str) or generation.get('generated_text_raw') != text
            or not isinstance(ids, list) or not 1 <= len(ids) <= 300
            or any(type(token) is not int or token < 0 for token in ids)
            or generation.get('generated_token_count') != len(ids)
            or generation.get('max_new_tokens') != 300
            or generation.get('stop_reason') not in ('eos_token', 'max_new_tokens')
            or (generation['stop_reason'] == 'max_new_tokens' and len(ids) != 300)
            or generation.get('prefix_shape') != [1, 383, 576]
            or generation.get('prompt_used') != record['original_record']['input'].lower()
            or not isinstance(tokens, list) or len(tokens) != 256
            or any(type(token) is not int or token < 0 for token in tokens)
            or not isinstance(audio, list) or len(audio) != 2
            or any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value) for value in audio)
            or generation.get('input_fingerprint_sha256') != object_sha(
                {'prompt_token_ids': tokens, 'audio_half_sha256': audio})):
        raise ValueError('invalid raw generation or model-input fingerprint')


class GenerationStore:
    def __init__(self, directory, records):
        self.path = directory / 'raw_generations.jsonl'
        self.progress = directory / 'progress.jsonl'
        self.state = recover_jsonl(self.path)
        if len(self.state) > len(records):
            raise ValueError('too many saved ReasonAQA records')
        for index, saved in enumerate(self.state):
            generation = saved.get('generation')
            if (saved.get('row_index') != index or saved.get('record') != records[index]
                    or saved.get('record_sha256') != object_sha(records[index])
                    or saved.get('generation_sha256') != object_sha(generation)
                    or not isinstance(generation, dict)
                    or not isinstance(generation.get('generated_text'), str)):
                raise ValueError(f'saved ReasonAQA row identity/content mismatch: {index}')
            validate_generation(generation, records[index])
        # Raw is authoritative, including a crash between raw and progress writes.
        write_jsonl(self.progress, ({'row_index': row['row_index'], 'id': row['record']['id'],
                                    'status': 'generated'} for row in self.state))

    @staticmethod
    def append_line(path, value):
        with path.open('ab') as stream:
            stream.write((json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())

    def append(self, index, record, generation):
        if index != len(self.state):
            raise ValueError('generations require contiguous rows and raw text')
        validate_generation(generation, record)
        saved = {'row_index': index, 'record': record, 'record_sha256': object_sha(record),
                 'generation': generation, 'generation_sha256': object_sha(generation)}
        self.append_line(self.path, saved)
        self.state.append(saved)
        self.append_line(self.progress, {'row_index': index, 'id': record['id'], 'status': 'generated'})
