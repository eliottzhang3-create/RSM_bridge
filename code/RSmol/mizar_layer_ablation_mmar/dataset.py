"""Freeze all 1000 MMAR rows; retain blank slots and audit unmappable GT."""
from collections import Counter
from pathlib import Path
import json
from mizar_layer_pair_search import audio, data, manifest as reference_manifest, storage
from . import arguments, config, scoring


def source_args(args):
    return {key: str(getattr(args, key)) for key in
            ('model_dir', 'mmar_metadata', 'mmar_audio_root', 'evaluation_script')}


def generator_sources():
    paths = [config.PACKAGE/name for name in ('config.py', 'arguments.py', 'dataset.py', 'scoring.py')]
    paths.extend(Path(module.__file__) for module in (audio, data, reference_manifest, storage))
    paths.extend([config.ROOT/'mizar_layer_pair_search/config.py',
                  config.ROOT/'mizar_official_eval/release_manifest.json'])
    return {p.relative_to(config.ROOT).as_posix(): storage.text_sha(p) for p in paths}


def normalize(index, row):
    choices, question, answer, identifier = (row.get(key) for key in ('choices', 'question', 'answer', 'id'))
    if (not isinstance(choices, list) or not 2 <= len(choices) <= 26 or any(not isinstance(c, str) for c in choices)
            or not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip()
            or not isinstance(identifier, str) or not identifier.strip() or row.get('category') not in config.CATEGORIES):
        raise ValueError(f'invalid full-MMAR metadata at {index}')
    bodies = [data.body(c).lower() for c in choices]
    valid = [bool(c) for c in bodies]
    if sum(valid) < 2 or '<|endoftext|>' in question.lower() or any('<|endoftext|>' in c for c in bodies):
        raise ValueError(f'insufficient valid choices or special tokens at MMAR row {index}')
    matches = [i for i, body in enumerate(bodies) if body and body == data.body(answer).lower()]
    gold = matches[0] if len(matches) == 1 else None
    prompt = (question.strip()+' '+' '.join(f'{chr(97+i)}) {c}' for i, c in enumerate(bodies))).lower()
    return {
        'dataset': 'mmar', 'global_index': index, 'source_row_index': index,
        'id': identifier, 'key': 'mmar:'+identifier, 'question_original': question,
        'choices_original': choices, 'choice_bodies': bodies, 'choice_valid_mask': valid,
        'valid_original_slots': [i for i, enabled in enumerate(valid) if enabled],
        'empty_original_slots': [i for i, enabled in enumerate(valid) if not enabled],
        'prompt': prompt, 'gold_index': gold, 'gold_original': answer,
        'gold_body_matching_indices': matches,
        'gold_mapping_status': 'UNIQUE_BODY_MATCH' if gold is not None else 'UNMAPPED_COUNT_INCORRECT',
        'duplicate_choice_bodies': len(set(c for c in bodies if c)) < sum(valid),
        'official_correct_by_slot': [enabled and scoring.string_match(answer, body, choices)
                                     for enabled, body in zip(valid, bodies)],
        'category': row['category'], 'sub_category': str(row.get('sub-category', '')),
        'modality': str(row.get('modality', '')), 'language': str(row.get('language', '')),
        'audio_path_original': row.get('audio_path'), 'original_record_sha256': storage.object_sha(row),
    }


def tokenize(record, tokenizer):
    prompt = tokenizer(record['prompt'], padding='max_length', truncation=True, max_length=256,
                       add_special_tokens=True)
    original = tokenizer.encode(record['prompt'], add_special_tokens=True)
    choices = [tokenizer.encode(body, add_special_tokens=False) if valid else []
               for body, valid in zip(record['choice_bodies'], record['choice_valid_mask'])]
    if len(prompt['input_ids']) != 256 or len(prompt['attention_mask']) != 256:
        raise ValueError('expected original prompt256')
    if any(valid and (not ids or len(ids)+383 > 1024) for ids, valid in zip(choices, record['choice_valid_mask'])):
        raise ValueError('empty tokenized nonempty choice or full option exceeds context; never filter/truncate')
    return {**record, 'prompt_token_ids': list(prompt['input_ids']),
            'prompt_attention_mask_audit_only': list(prompt['attention_mask']),
            'prompt_original_token_count': len(original), 'prompt_truncated': len(original) > 256,
            'choice_token_ids': [list(ids) for ids in choices]}


def attach_audio(records, args):
    result = []
    for record in records:
        value = record['audio_path_original']
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'invalid MMAR audio_path: {record["id"]}')
        path = data.contained(args.mmar_audio_root, value)
        if not path.is_file():
            raise FileNotFoundError(f'missing MMAR audio for {record["id"]}: {path}')
        result.append({**record, 'audio_ref': {'kind': 'file', 'path': str(path)}})
    return result


def metadata_audit(records):
    return {
        'rows': len(records), 'category_counts': dict(Counter(r['category'] for r in records)),
        'modality_counts': dict(Counter(r['modality'] for r in records)),
        'sub_category_counts': dict(Counter(r['sub_category'] for r in records)),
        'empty_option_rows': [{'index': r['global_index'], 'id': r['id'], 'slots': r['empty_original_slots']}
                              for r in records if r['empty_original_slots']],
        'unmapped_gold_rows': [{'index': r['global_index'], 'id': r['id'], 'category': r['category'],
                               'choices': r['choices_original'], 'answer': r['gold_original'],
                               'policy': 'retained_in_full_denominator_count_incorrect'}
                              for r in records if r['gold_index'] is None],
        'official_accepted_slot_count_distribution': dict(Counter(str(sum(r['official_correct_by_slot'])) for r in records)),
        'prompt_truncated_rows': sum(r.get('prompt_truncated', False) for r in records),
    }


def validate(frozen):
    storage.verify_seal(frozen, 'manifest_sha256')
    records = frozen['records']
    if (frozen['schema'] != config.MANIFEST_SCHEMA or frozen['input_protocol'] != config.INPUT_PROTOCOL
            or len(records) != config.ROWS or [r['global_index'] for r in records] != list(range(config.ROWS))
            or len({r['id'] for r in records}) != config.ROWS
            or Counter(r['category'] for r in records) != Counter(config.CATEGORY_COUNTS)):
        raise ValueError('not exact 1000-row/four-category MMAR coverage')
    for record in records:
        reconstructed = normalize(record['global_index'], {
            'id': record['id'], 'question': record['question_original'], 'choices': record['choices_original'],
            'answer': record['gold_original'], 'category': record['category'], 'sub-category': record['sub_category'],
            'modality': record['modality'], 'language': record['language'], 'audio_path': record['audio_path_original']})
        for key, value in reconstructed.items():
            if key != 'original_record_sha256' and record.get(key) != value:
                raise ValueError(f'inconsistent frozen MMAR field {record["id"]}: {key}')
        ids = record['choice_token_ids']
        if (len(ids) != len(record['choice_bodies']) or len(record['prompt_token_ids']) != 256
                or len(record['prompt_attention_mask_audit_only']) != 256 or not record.get('audio_audit')
                or record['audio_ref']['kind'] != 'file'
                or any(type(t) is not int or not 0 <= t < 49152 for t in record['prompt_token_ids'])
                or any((bool(tokens) != valid or len(tokens)+383 > 1024
                        or any(type(t) is not int or not 0 <= t < 49152 for t in tokens))
                       for tokens, valid in zip(ids, record['choice_valid_mask']))):
            raise ValueError(f'invalid frozen MMAR tokens/audio: {record["id"]}')
    if metadata_audit(records) != frozen['dataset_audit']:
        raise ValueError('frozen MMAR metadata audit changed')
    return frozen


def load(path, full=True):
    frozen = validate(storage.read_json(path))
    if frozen['generator_sources'] != generator_sources():
        raise ValueError('MMAR manifest parser/tokenizer/audio/scoring implementation changed')
    for item in frozen['files']:
        storage.verify_file(item, full=full)
    return frozen


def build(args):
    sources = generator_sources()
    tokenizer, tokenizer_files = reference_manifest.load_tokenizer(args.model_dir)
    files = {str(path): storage.file_identity(path) for path in (args.mmar_metadata, args.evaluation_script)}
    scoring.verify_reference(args.evaluation_script)
    rows = storage.read_json(args.mmar_metadata)
    if (not isinstance(rows, list) or len(rows) != config.ROWS
            or len({r['id'] for r in rows}) != config.ROWS
            or Counter(r['category'] for r in rows) != Counter(config.CATEGORY_COUNTS)):
        raise ValueError('expected all 1000 original MMAR rows and Signal43/Perception404/Semantic412/Cultural141')
    records = attach_audio([tokenize(normalize(i, row), tokenizer) for i, row in enumerate(rows)], args)
    for record in records:
        path = record['audio_ref']['path']
        if path not in files:
            files[path] = storage.file_identity(path)
    for record in records:
        _, record['audio_audit'] = audio.load_fixed20(record['audio_ref'])
        print(json.dumps({'event': 'MMAR_INPUT_FROZEN', 'index': record['global_index'], 'id': record['id']},
                         ensure_ascii=False), flush=True)
    identities = [*files.values(), *tokenizer_files]
    for item in identities:
        storage.verify_file(item)
    if generator_sources() != sources:
        raise ValueError('MMAR input implementation changed during build')
    return validate(storage.seal({
        'schema': config.MANIFEST_SCHEMA, 'input_protocol': config.INPUT_PROTOCOL, 'records': records,
        'files': identities, 'source_arguments': source_args(args), 'generator_sources': sources,
        'metadata_canonical_sha256': storage.object_sha(rows), 'dataset_audit': metadata_audit(records),
    }, 'manifest_sha256'))


def ensure(args):
    with storage.output_lock(args.manifest.parent):
        if args.manifest.exists():
            frozen = load(args.manifest)
            if frozen['source_arguments'] != source_args(args):
                raise ValueError('existing MMAR manifest uses different source paths')
        else:
            frozen = build(args)
            storage.write_json(args.manifest, frozen)
    print(json.dumps({'event': 'MMAR_MANIFEST_READY', 'manifest': str(args.manifest),
                      'manifest_sha256': frozen['manifest_sha256'], 'audit': frozen['dataset_audit']},
                     ensure_ascii=False), flush=True)
    return frozen


def main(argv=None):
    args = arguments.parse_args(argv)
    if args.check_args:
        print(json.dumps(vars(args), default=str, indent=2))
        return 0
    frozen = ensure(args)
    print(json.dumps({'status': 'PASS_MANIFEST', 'manifest': str(args.manifest), 'rows': len(frozen['records'])}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
