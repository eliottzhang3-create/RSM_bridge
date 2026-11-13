"""Keep the historical ReasonAQA snapshot; authorize only four later MMSU edits."""
import hashlib
import json


def effective_benchmark_baseline(root):
    original = json.loads((root / 'code/RSmol/mizar_s1_ced_reasonaqa_eval/reference_audit.json').read_text(encoding='utf-8'))['files']
    extensions = json.loads((root / 'code/RSmol/mizar_s1_ced_mmsu_eval/normalization_reference.json').read_text(encoding='utf-8'))['authorized_mmsu_extensions']
    result = dict(original)
    for path, record in extensions.items():
        if original[path] != record['before_raw_sha256']:
            raise ValueError(f'MMSU extension baseline mismatch: {path}')
        result[path] = record['after_raw_sha256']
    return result
