"""Generate one immutable uniform sample from the original ReasonAQA test pool."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import config, dataset, storage

SAMPLING = {'version': 'reasonaqa_close_ended_sample5000_seed1234_v1',
            'algorithm': 'python_random_Random_sample_range_without_replacement_v1',
            'population_order': 'original_json_order_after_close_ended_filter',
            'size': config.SAMPLE_SIZE, 'seed': config.SAMPLE_SEED,
            'output_order': 'original_source_row_index_ascending', 'stratified': False}


def select_sample(pool):
    if len(pool) < config.SAMPLE_SIZE:
        raise ValueError(f'need at least {config.SAMPLE_SIZE} eligible rows, got {len(pool)}')
    indices = sorted(random.Random(config.SAMPLE_SEED).sample(range(len(pool)), config.SAMPLE_SIZE))
    return [pool[index] for index in indices]


def build_manifest(test_json):
    test_json = test_json.resolve()
    source_sha = storage.file_sha(test_json)
    pool, audit = dataset.select_records(test_json)
    if storage.file_sha(test_json) != source_sha:
        raise ValueError('source test.json changed while selecting records')
    if audit['selected_category_counts'] != config.POOL_COUNTS or len(pool) != 55379:
        raise ValueError(f'eligible pool differs from user-confirmed 55379 rows: {audit["selected_category_counts"]}')
    records = select_sample(pool)
    manifest = {'schema': SAMPLING['version'], 'sampling': SAMPLING,
                'source_test_json': {'path': str(test_json), 'bytes': test_json.stat().st_size, 'sha256': source_sha},
                'pool_audit': audit,
                'sample_category_counts': {name: sum(r['category'] == name for r in records) for name in dataset.CATEGORIES},
                'sample_ids': [r['id'] for r in records],
                'sample_records_sha256': storage.object_sha(records),
                'sampled_test_sha256': storage.object_sha([r['original_record'] for r in records]),
                'records': records}
    manifest['manifest_sha256'] = storage.object_sha(manifest)
    return manifest


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError('manifest must be an object')
    body = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
    records = manifest.get('records')
    if (manifest.get('schema') != SAMPLING['version'] or manifest.get('sampling') != SAMPLING
            or manifest.get('manifest_sha256') != storage.object_sha(body)
            or not isinstance(records, list) or len(records) != config.SAMPLE_SIZE
            or manifest.get('sample_records_sha256') != storage.object_sha(records)
            or manifest.get('sample_ids') != [r['id'] for r in records]
            or len(set(manifest['sample_ids'])) != config.SAMPLE_SIZE
            or manifest.get('pool_audit', {}).get('selected_category_counts') != config.POOL_COUNTS):
        raise ValueError('invalid/tampered fixed-5000 sampling manifest')
    return manifest


def load_manifest(path, test_json=None):
    manifest = validate_manifest(json.loads(path.read_text(encoding='utf-8')))
    source = test_json.resolve() if test_json else Path(manifest['source_test_json']['path'])
    if (source.stat().st_size != manifest['source_test_json']['bytes']
            or storage.file_sha(source) != manifest['source_test_json']['sha256']):
        raise ValueError('source test.json differs from sampling manifest')
    # Regenerate independently to verify seed, uniform-selection algorithm and original records.
    recomputed = build_manifest(source)
    for field in ('records', 'pool_audit', 'sample_ids', 'sample_category_counts',
                  'sample_records_sha256', 'sampled_test_sha256', 'sampling'):
        if manifest[field] != recomputed[field]:
            raise ValueError(f'sampling manifest disagrees with original test.json: {field}')
    return manifest


def generate_bundle(test_json, directory):
    manifest = build_manifest(test_json)
    ownership = {'benchmark': 'reasonaqa5000_dataset', 'manifest_sha256': manifest['manifest_sha256']}
    with storage.output_lock(directory):
        storage.claim_output(directory, ownership)
        target = directory / 'manifest.json'
        if target.exists() and json.loads(target.read_text(encoding='utf-8')) != manifest:
            raise ValueError('existing sampling manifest differs; use another output directory')
        storage.write_json(directory / 'sampled_test.json', [r['original_record'] for r in manifest['records']])
        storage.write_jsonl(directory / 'selected_records.jsonl', manifest['records'])
        storage.write_json(directory / 'dataset_audit.json', {key: value for key, value in manifest.items() if key != 'records'})
        # Publish manifest last; it alone is authoritative for evaluation.
        storage.write_json(target, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-json', type=Path, default=config.DATA / 'reasonaqa/test.json')
    parser.add_argument('--output-dir', type=Path, default=config.DEFAULT_BUNDLE)
    args = parser.parse_args(argv)
    try:
        manifest = generate_bundle(args.test_json, args.output_dir.resolve())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'ReasonAQA5000 sampling failed: {exc}\n')
    print(json.dumps({'status': 'PASS', 'eligible_rows': 55379, 'sample_rows': len(manifest['records']),
                      'sample_category_counts': manifest['sample_category_counts'],
                      'manifest_sha256': manifest['manifest_sha256'],
                      'manifest': str(args.output_dir / 'manifest.json')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
