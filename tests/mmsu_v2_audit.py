"""Keep historical snapshots and apply explicit authorized MMSU transitions."""
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
    # The user authorized a new input template on 2026-11-22. Do not relabel
    # these intentional edits as unchanged originals or replace old snapshots.
    compact = json.loads((root / 'code/RSmol/mizar_s1_ced_mmsu_eval/compact_prompt_reference.json').read_text(encoding='utf-8'))
    for path, record in compact['files'].items():
        if path not in result:
            continue
        if result[path] != record['before_raw_sha256']:
            raise ValueError(f'compact MMSU transition baseline mismatch: {path}')
        result[path] = record['after_raw_sha256']
    # Existing selected4/init0904 registrations happened after this snapshot.
    # Their own audit checks the full registration transition and old ASTs.
    from mizar_selected4_eval.audit import effective_reference_files
    for path, hashes in effective_reference_files(root).items():
        if path in result:
            result[path] = hashes['raw_sha256']
    return result
