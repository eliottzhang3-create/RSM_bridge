"""Reuse original frozen inputs across one audited, unused prompt-helper edit."""
import json
from pathlib import Path
from mizar_layer_pair_search import data, manifest as reference, storage

MMSU_SOURCE = 'mizar_s1_ced_mmsu_eval/dataset.py'
HISTORICAL_MMSU = '7fd1abda9070166767d2b46b8145517c47d4ec6770cf0d7b659c30f7649aba1b'
CURRENT_MMSU = 'a512cb89bda8e41f47ed1b1cccfd676c4f7bc6aaad9cb0dbd3948f0eac6a2ed5'
COMPATIBILITY = {
    'source': MMSU_SOURCE, 'manifest_source_lf_sha256': HISTORICAL_MMSU,
    'current_source_lf_sha256': CURRENT_MMSU,
    'historical_git': '4e97331f050c0f7bad01e4ec02f3928dbb530639',
    'prompt_update_git': '444ff7f6ee6c400f491535e23dcded619571a461',
    'reason': 'Only ordinary MMSU prompt helpers/constants changed; frozen manifest uses its own saved prompt/token IDs',
    'metadata_reader_unchanged': ['row_id', 'validate_record', 'shard_paths', 'iter_rows', 'preflight', 'audit_records'],
}


def source_audit(frozen):
    previous, current = frozen.get('generator_sources', {}), reference.generator_sources()
    differences = []
    for name in sorted(set(previous) | set(current)):
        before, after = previous.get(name), current.get(name)
        if before == after:
            continue
        allowed = name == MMSU_SOURCE and before == HISTORICAL_MMSU and after == CURRENT_MMSU
        differences.append({'source': name, 'manifest_lf_sha256': before,
                            'current_lf_sha256': after, 'compatible': allowed})
    audit = {'event': 'FROZEN_MANIFEST_SOURCE_AUDIT', 'manifest_sha256': frozen['manifest_sha256'],
             'status': 'EXACT' if not differences else 'COMPATIBLE' if all(d['compatible'] for d in differences) else 'REJECTED',
             'differences': differences}
    if audit['status'] == 'COMPATIBLE':
        audit['compatibility'] = COMPATIBILITY
    if audit['status'] == 'REJECTED':
        raise ValueError('unapproved frozen manifest source differences: '+json.dumps(audit, ensure_ascii=False))
    return audit


def load(path, *, full=True, log=False):
    # Do not modify/reseal the manifest or reparse samples using today's prompts.
    frozen = data.validate_manifest(storage.read_json(path))
    audit = source_audit(frozen)
    for item in frozen['files']:
        storage.verify_file(item, full=full)
    if log:
        print(json.dumps(audit, ensure_ascii=False), flush=True)
    return frozen


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    frozen = load(args.manifest, log=True)
    print(json.dumps({'status':'PASS_FROZEN_MANIFEST', 'manifest_sha256':frozen['manifest_sha256'],
                      'rows':len(frozen['records'])}), flush=True)


if __name__ == '__main__':
    main()
