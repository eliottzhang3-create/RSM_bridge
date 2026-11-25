"""CPU-only deterministic 4x250 manifest; no model weights or CUDA required."""
import json
from mizar_layer_pair_search import audio, storage
from mizar_layer_pair_search.manifest import load_tokenizer
from . import arguments, config, data


def log_event(event, **fields):
    print(json.dumps({'event': event, **fields}, ensure_ascii=False), flush=True)


def source_arguments(args):
    names = ('model_dir', 'mmau_metadata', 'mmau_parquet', 'mmau_audio_root', 'mmar_metadata',
             'mmar_audio_root', 'mmsu_dir', 'reasonaqa_test', 'audiocaps_root', 'clotho_root',
             'compare_manifest')
    return {name: str(getattr(args, name)) if getattr(args, name) is not None else None for name in names}


def generator_sources():
    paths = [config.PACKAGE / name for name in ('config.py', 'arguments.py', 'data.py', 'manifest.py')]
    paths.extend(config.ROOT / name for name in (
        'mizar_layer_pair_search/config.py', 'mizar_layer_pair_search/storage.py',
        'mizar_layer_pair_search/audio.py', 'mizar_layer_pair_search/data.py',
        'mizar_layer_pair_search/manifest.py', 'mizar_official_eval/release_manifest.json',
        'mizar_s1_ced_mmsu_eval/dataset.py', 'mizar_s1_ced_reasonaqa_eval/dataset.py',
        'mizar_s1_ced_reasonaqa_eval/storage.py',
        'scripts/evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.py'))
    return {p.relative_to(config.ROOT).as_posix(): storage.text_sha(p) for p in paths}


def overlap_audit(records, path):
    if path is None:
        return {'status': 'NOT_REQUESTED', 'sampling_affected': False}
    old = storage.verify_seal(storage.read_json(path), 'manifest_sha256')
    keys = {r['key'] for r in old['records']}
    common = [r for r in records if r['key'] in keys]
    return {'status': 'AUDITED', 'path': str(path), 'manifest_sha256': old['manifest_sha256'],
            'sampling_affected': False, 'rows_in_common': len(common),
            'per_dataset': {name: sum(r['dataset'] == name for r in common) for name in config.DATASETS},
            'keys_in_common': [r['key'] for r in common]}


def build(args):
    sources = generator_sources()
    log_event('MANIFEST_BUILD_START', rows=config.ROWS, rows_per_dataset=config.PER_DATASET, seed=config.SEED)
    tokenizer, tokenizer_files = load_tokenizer(args.model_dir)
    from mizar_s1_ced_mmsu_eval import dataset as mmsu
    source_paths = [args.mmau_metadata, args.mmar_metadata, args.reasonaqa_test, *mmsu.shard_paths(args.mmsu_dir)]
    if args.compare_manifest is not None:
        source_paths.append(args.compare_manifest)
    source_files = [storage.file_identity(path) for path in source_paths]
    pools, _, mmsu_audit = data.read_pools(args)
    log_event('MANIFEST_POOLS_READY', source_rows={name: len(rows) for name, rows in pools.items()})
    records, selection = data.select_records(pools, tokenizer)
    log_event('MANIFEST_SAMPLED', selection_audit=selection)
    records = data.attach_audio(records, pools, args)
    inventory = {item['path']: item for item in source_files}
    try:
        if any(r['audio_ref']['kind'] == 'parquet' for r in records):
            audio.prepare_selected_rows(records)
        for record in records:
            path = record['audio_ref']['path']
            if path not in inventory:
                inventory[path] = storage.file_identity(path)
            _, audit = audio.load_fixed20(record['audio_ref'])
            record['audio_audit'] = audit
            log_event('MANIFEST_AUDIO_CHECKED', index=record['global_index'], key=record['key'],
                      prompt_truncated=record['prompt_truncated'])
    finally:
        audio.clear_selected_rows()
    overlap = overlap_audit(records, args.compare_manifest)
    files = [*inventory.values(), *tokenizer_files]
    for item in files:
        storage.verify_file(item)
    if generator_sources() != sources:
        raise ValueError('manifest generator changed during build')
    frozen = storage.seal({'schema': config.MANIFEST_SCHEMA, 'seed': config.SEED,
        'sampling': 'independent_Random1234_sample_without_replacement_sorted_eligible_indices_per_dataset',
        'protocol': config.MANIFEST_PROTOCOL, 'generator_sources': sources, 'source_arguments': source_arguments(args),
        'selection_audit': selection, 'mmsu_source_audit': mmsu_audit, 'previous_manifest_overlap': overlap,
        'files': files, 'records': records}, 'manifest_sha256')
    return data.validate_manifest(frozen)


def source_audit(frozen):
    previous, current = frozen.get('generator_sources', {}), generator_sources()
    differences = [{'source': name, 'manifest_lf_sha256': previous.get(name), 'current_lf_sha256': current.get(name)}
                   for name in sorted(set(previous) | set(current)) if previous.get(name) != current.get(name)]
    if differences:
        raise ValueError('1000-row manifest parser/tokenizer/audio source mismatch: '+json.dumps(differences))
    return {'event': 'FROZEN_MANIFEST_SOURCE_AUDIT', 'status': 'EXACT',
            'manifest_sha256': frozen['manifest_sha256'], 'differences': []}


def load(path, *, full=True, log=False):
    frozen = data.validate_manifest(storage.read_json(path))
    audit = source_audit(frozen)
    for item in frozen['files']:
        storage.verify_file(item, full=full)
    if log:
        print(json.dumps({**audit, 'rows': len(frozen['records'])}, ensure_ascii=False), flush=True)
    return frozen


def ensure(args):
    with storage.output_lock(args.manifest.parent):
        if args.manifest.exists():
            result = load(args.manifest)
            if result['source_arguments'] != source_arguments(args):
                raise ValueError('frozen manifest source arguments differ; reuse original arguments or a new path')
            log_event('REUSED_FROZEN_MANIFEST', manifest_sha256=result['manifest_sha256'])
        else:
            result = build(args)
            storage.write_json(args.manifest, result)
    return result


def main(argv=None):
    args = arguments.parse_args(argv)
    if args.check_args:
        print(json.dumps({**vars(args), 'manifest_protocol': config.MANIFEST_PROTOCOL}, default=str, indent=2))
        return 0
    result = ensure(args)
    print(json.dumps({'status': 'PASS_MANIFEST', 'manifest': str(args.manifest),
                      'manifest_sha256': result['manifest_sha256'], 'rows': len(result['records']),
                      'selection_audit': result['selection_audit'],
                      'previous_manifest_overlap': result['previous_manifest_overlap']}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
