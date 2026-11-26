"""CPU first500 import, source cache references only; never runs screening."""
import json
from mizar_layer_pair_search import storage
from . import arguments, config, identity, source


def publish(args, frozen):
    path = args.output_dir/'selected/manifest.json'
    if path.exists() and storage.read_json(path) != frozen:
        raise ValueError('existing first500 manifest differs')
    storage.write_json(path, frozen)
    storage.write_jsonl(args.output_dir/'selected/selected_records.jsonl', frozen['records'])
    storage.write_json(args.output_dir/'selected/sampled_test.json', [r['original_record'] for r in frozen['records']])
    storage.write_json(args.output_dir/'selection_report.json', storage.seal({
        'status': 'SELECTED', 'required': config.TARGET, 'accepted': len(frozen['records']),
        'source_scanned': frozen['source_scanned'], 'source_qualifying': frozen['source_total_qualifying'],
        'selection': frozen['selection'], 'new_screening_forwards': 0,
        'stop_shuffle_position': frozen['stop_shuffle_position'], 'manifest_sha256': frozen['manifest_sha256']}))
    print(json.dumps({'event': 'REUSED_FIRST500_READY', 'source_qualifying': frozen['source_total_qualifying'],
        'selected': len(frozen['records']), 'category_counts': frozen['category_counts'], 'new_screening_forwards': 0,
        'selected_manifest': str(path), 'source_output_dir': str(args.source_output_dir)}, ensure_ascii=False), flush=True)


def main(argv=None):
    args = arguments.parse_args(argv)
    if args.check_args:
        print(json.dumps(vars(args), default=str, indent=2)); return 0
    context = source.load(args.source_output_dir, full=True, verify_caches=True)
    frozen = source.manifest(context, args.source_output_dir)
    expected = identity.build(args, frozen, context)
    with storage.output_lock(args.output_dir):
        storage.claim(args.output_dir, expected)
        publish(args, frozen)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
