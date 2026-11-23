"""Copied durable primitives; MyEval validates every expanded task and input hash."""
from pathlib import Path
import re

from mizar_s1_ced_reasonaqa5000_eval.storage import (
    object_sha, file_sha, write_json, write_jsonl, output_lock, claim_output, recover_jsonl)


def validate_generation(generation, record):
    if not isinstance(generation, dict):
        raise ValueError('missing generation object')
    ids = generation.get('generated_token_ids')
    tokens, mask = generation.get('prompt_token_ids'), generation.get('prompt_attention_mask')
    audio = generation.get('audio_half_sha256')
    if (not isinstance(generation.get('generated_text'), str)
            or generation.get('generated_text_raw') != generation['generated_text']
            or not isinstance(ids, list) or not 1 <= len(ids) <= 300
            or any(type(x) is not int or x < 0 for x in ids)
            or generation.get('generated_token_count') != len(ids)
            or generation.get('max_new_tokens') != 300
            or generation.get('stop_reason') not in ('eos_token', 'max_new_tokens')
            or (generation['stop_reason'] == 'max_new_tokens' and len(ids) != 300)
            or generation.get('prefix_shape') != [1, 383, 576]
            or generation.get('first_answer_logit_position') != 382
            or generation.get('prompt_used') != record['prompt']
            or not isinstance(tokens, list) or len(tokens) != 256
            or any(type(x) is not int or x < 0 for x in tokens)
            or not isinstance(mask, list) or len(mask) != 256
            or any(type(x) is not int or x not in (0, 1) for x in mask) or not any(mask)
            or not isinstance(audio, list) or len(audio) != 2
            or any(not isinstance(x, str) or not re.fullmatch('[0-9a-f]{64}', x) for x in audio)
            or generation.get('input_fingerprint_sha256') != object_sha({
                'prompt_token_ids': tokens, 'prompt_attention_mask': mask, 'audio_half_sha256': audio})):
        raise ValueError('invalid generation or actual model-input fingerprint')


class GenerationStore:
    def __init__(self, directory, records, *, read_only=False, expected_intervention=None):
        self.path = Path(directory) / 'raw_generations.jsonl'
        self.progress = Path(directory) / 'progress.jsonl'
        self.expected_intervention = expected_intervention
        if read_only:
            import json
            raw = self.path.read_bytes()
            if raw and not raw.endswith(b'\n'):
                raise ValueError('read-only analysis refuses an incomplete raw tail')
            self.state = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
        else:
            self.state = recover_jsonl(self.path)
        if len(self.state) > len(records):
            raise ValueError('too many saved MyEval rows')
        for index, saved in enumerate(self.state):
            if (saved.get('row_index') != index or saved.get('record') != records[index]
                    or saved.get('record_sha256') != object_sha(records[index])
                    or saved.get('generation_sha256') != object_sha(saved.get('generation'))):
                raise ValueError(f'saved task/generation identity mismatch: {index}')
            validate_generation(saved['generation'], records[index])
            if expected_intervention is not None and saved['generation'].get('intervention') != expected_intervention:
                raise ValueError('saved generation intervention differs from this run')
        if not read_only:
            write_jsonl(self.progress, ({'row_index': row['row_index'], 'id': row['record']['id']}
                                       for row in self.state))

    @staticmethod
    def append_line(path, value):
        import json
        import os
        with path.open('ab') as stream:
            stream.write((json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
            stream.flush(); os.fsync(stream.fileno())

    def append(self, index, record, generation):
        if index != len(self.state):
            raise ValueError('raw generations must be contiguous')
        validate_generation(generation, record)
        if self.expected_intervention is not None and generation.get('intervention') != self.expected_intervention:
            raise ValueError('generation intervention differs from this run')
        saved = {'row_index': index, 'record': record, 'record_sha256': object_sha(record),
                 'generation': generation, 'generation_sha256': object_sha(generation)}
        self.append_line(self.path, saved)
        self.state.append(saved)
        self.append_line(self.progress, {'row_index': index, 'id': record['id']})
