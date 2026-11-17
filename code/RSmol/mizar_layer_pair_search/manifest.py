"""CPU-only deterministic selection/tokenization/audio audit; no model loading."""
import json
from pathlib import Path

from . import arguments, audio, config, data, storage


def source_arguments(args):
    names = ('model_dir', 'mmau_metadata', 'mmau_parquet', 'mmau_audio_root', 'mmar_metadata',
             'mmar_audio_root', 'mmsu_dir', 'reasonaqa_test', 'audiocaps_root', 'clotho_root')
    return {name: str(getattr(args, name)) for name in names}


def generator_sources():
    paths = [config.PACKAGE / name for name in
             ('config.py', 'storage.py', 'arguments.py', 'data.py', 'audio.py', 'manifest.py')]
    paths.extend([config.ROOT / 'mizar_s1_ced_mmsu_eval/dataset.py',
                  config.ROOT / 'mizar_s1_ced_reasonaqa_eval/dataset.py',
                  config.ROOT / 'mizar_s1_ced_reasonaqa_eval/storage.py',
                  config.ROOT / 'scripts/evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.py'])
    return {p.relative_to(config.ROOT).as_posix(): storage.text_sha(p) for p in paths}


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    root = Path(model_dir) / 'assets/smollm2-135m'
    release = storage.read_json(config.ROOT / 'mizar_official_eval/release_manifest.json')
    files = []
    for item in release['files']:
        if item['path'].startswith('assets/smollm2-135m/') and not item['path'].endswith('model.safetensors'):
            actual = storage.file_identity(Path(model_dir) / item['path'])
            if actual['bytes'] != item['bytes'] or actual['sha256'] != item['sha256']:
                raise ValueError(f'tokenizer/config differs from official release: {item["path"]}')
            files.append(actual)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    size = len(tokenizer)
    tokenizer.add_special_tokens({'pad_token': '!'})
    tokenizer.padding_side = tokenizer.truncation_side = 'right'
    if len(tokenizer) != size or size != 49152 or tokenizer.pad_token_id != 17 or tokenizer.eos_token_id != 0:
        raise ValueError('official tokenizer vocabulary/PAD/EOS mismatch')
    return tokenizer, files


def build(args):
    tokenizer, tokenizer_files = load_tokenizer(args.model_dir)
    # Hash before reading; repeated verification below catches concurrent source writes.
    from mizar_s1_ced_mmsu_eval import dataset as mmsu
    source_paths = [args.mmau_metadata, args.mmar_metadata, args.reasonaqa_test, *mmsu.shard_paths(args.mmsu_dir)]
    source_files = [storage.file_identity(path) for path in source_paths]
    pools, _, mmsu_audit = data.read_pools(args)
    records, selection = data.select_records(pools, tokenizer)
    records = data.attach_audio(records, pools, args)
    if any(r['audio_ref']['kind'] == 'parquet' for r in records):
        audio.prepare_selected_rows(records)
    inventory = {item['path']: item for item in source_files}
    for record in records:
        path = record['audio_ref']['path']
        if path not in inventory:
            inventory[path] = storage.file_identity(path)
        _, audit = audio.load_fixed20(record['audio_ref'])
        record['audio_audit'] = audit
        print(json.dumps({'event': 'MANIFEST_AUDIO_CHECKED', 'index': record['global_index'],
                          'key': record['key'], 'prompt_truncated': record['prompt_truncated']}, ensure_ascii=False), flush=True)
    audio.clear_selected_rows()
    files = [*inventory.values(), *tokenizer_files]
    for identity in files:
        storage.verify_file(identity)
    manifest = storage.seal({'schema': config.SCHEMA, 'seed': config.SEED,
        'sampling': 'independent_Random1234_sample_without_replacement_sorted_eligible_indices_per_dataset',
        'protocol': config.PROTOCOL, 'generator_sources': generator_sources(), 'source_arguments': source_arguments(args),
        'selection_audit': selection, 'mmsu_source_audit': mmsu_audit,
        'files': files, 'records': records}, 'manifest_sha256')
    return data.validate_manifest(manifest)


def load(path, *, full=True, verify_sources=True):
    manifest = data.validate_manifest(storage.read_json(path))
    if verify_sources and manifest['generator_sources'] != generator_sources():
        raise ValueError('manifest was generated with a different parser/tokenizer/audio implementation')
    for identity in manifest['files']:
        storage.verify_file(identity, full=full)
    return manifest


def ensure(args):
    directory = args.manifest.parent
    with storage.output_lock(directory):
        if args.manifest.exists():
            result = load(args.manifest)
            if result['source_arguments'] != source_arguments(args):
                raise ValueError('frozen manifest source arguments differ; reuse its original arguments or choose a new manifest path')
            print(json.dumps({'event': 'REUSED_FROZEN_MANIFEST', 'manifest_sha256': result['manifest_sha256']}), flush=True)
        else:
            result = build(args)
            storage.write_json(args.manifest, result)
    return result


def main(argv=None):
    args = arguments.parse_args(argv)
    if args.check_args:
        print(json.dumps(vars(args), default=str, ensure_ascii=False, indent=2))
        return 0
    result = ensure(args)
    print(json.dumps({'status': 'PASS_MANIFEST', 'manifest': str(args.manifest),
                      'manifest_sha256': result['manifest_sha256'], 'rows': len(result['records']),
                      'selection_audit': result['selection_audit']}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
