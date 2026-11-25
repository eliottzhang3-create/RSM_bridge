"""CPU metadata/tokenizer preparation of the exact pre-existing 5000 rows."""
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path
from mizar_layer_pair_search import storage
from mizar_layer_pair_search.data import reasonaqa_mcq
from mizar_layer_pair_search.manifest import load_tokenizer
from mizar_s1_ced_reasonaqa5000_eval import dataset, sample
from . import arguments, config


def source_arguments(args):
    return {k: str(getattr(args, k)) for k in ('model_dir', 'source_manifest', 'test_json',
            'audiocaps_root', 'clotho_root', 'clotho_aqa_root')}


def generator_sources():
    from .identity import reused_sources
    return {**reused_sources(), **{f'mizar_reasonaqa_tracing/{name}': storage.text_sha(config.PACKAGE/name)
            for name in ('config.py', 'prepare.py')}}


def single_ids(tokenizer, labels):
    encoded = [tokenizer.encode(s, add_special_tokens=False) for s in labels]
    special = set(tokenizer.all_special_ids) | {0, 17}
    if (any(len(ids) != 1 or type(ids[0]) is not int or ids[0] in special
            or not 0 <= ids[0] < 49152 for ids in encoded)
            or len({ids[0] for ids in encoded}) != len(labels)):
        raise ValueError(f'each target must be one distinct ordinary token: {dict(zip(labels, encoded))}')
    return [ids[0] for ids in encoded]


def transform(record, index, tokenizer, resolver):
    row = record['original_record']
    if row['filepath2'].strip():
        raise ValueError('existing 5000 unexpectedly contains a second recording; do not silently filter')
    if record['category'] in config.CATEGORIES[:2]:
        question, choices = reasonaqa_mcq(row)
        labels = list('abcd')
        match = re.match(r'^\s*([a-dA-D])[).]\s*', row['answer'])
        if match is None:
            raise ValueError('expected explicit A-D gold label in the original 5000')
        gold = ord(match[1].lower())-97
        # Preserve slots even when bodies duplicate; never infer GT from a body.
        base_prompt = (question.strip()+' '+' '.join(f'{c}) {s.strip()}' for c, s in zip(labels, choices))).lower()
    elif record['category'] == config.CATEGORIES[2]:
        labels, choices = ['yes', 'no'], ['yes', 'no']
        answer = row['answer'].strip().casefold()
        if answer not in labels:
            raise ValueError('not an original close-ended yes/no row')
        gold = labels.index(answer)
        base_prompt = row['input'].strip().lower()
    else:
        raise ValueError('unknown original 5000 category')
    if '<|endoftext|>' in base_prompt:
        raise ValueError('source input already contains EOT')
    prompt = base_prompt+config.EOT
    tokens = tokenizer(prompt, padding='max_length', truncation=True, max_length=256, add_special_tokens=True)
    full = tokenizer.encode(prompt, add_special_tokens=True)
    ref = resolver.resolve(row['filepath1'], record['category'])
    return {**copy.deepcopy(record), 'source_manifest_index': index, 'key': record['id'],
        'source_record_sha256': storage.object_sha(record), 'prompt': prompt,
        'prompt_token_ids': list(tokens['input_ids']), 'prompt_attention_mask_audit_only': list(tokens['attention_mask']),
        'prompt_original_token_count': len(full), 'prompt_truncated': len(full)>256,
        'prompt_eot_retained': 0 in tokens['input_ids'], 'candidate_labels': labels,
        'candidate_token_ids': single_ids(tokenizer, labels), 'choice_bodies': [s.lower() for s in choices],
        'gold_index': gold, 'gold_label': labels[gold], 'gold_original': row['answer'],
        'audio_ref': {'kind': 'file', 'path': ref['path']}, 'audio_file_stat': ref}


def validate(value):
    storage.verify_seal(value, 'manifest_sha256')
    rows, order = value['records'], value['shuffle_order']
    if (value['schema'] != config.SCHEMA or value['protocol'] != config.PROTOCOL
            or len(rows) != config.POOL_SIZE or sorted(order) != list(range(config.POOL_SIZE))
            or order != permutation() or value['shuffle_sha256'] != storage.object_sha(order)
            or len({r['id'] for r in rows}) != config.POOL_SIZE
            or [r['source_manifest_index'] for r in rows] != list(range(config.POOL_SIZE))
            or [r['source_row_index'] for r in rows] != sorted(r['source_row_index'] for r in rows)):
        raise ValueError('pool identity/order/size differs from fixed 5000')
    for r in rows:
        labels = ['yes', 'no'] if r['category'] == config.CATEGORIES[2] else list('abcd')
        if (r['candidate_labels'] != labels or len(r['candidate_token_ids']) != len(labels)
                or len(set(r['candidate_token_ids'])) != len(labels)
                or any(type(x) is not int or not 0 <= x < 49152 or x in (0, 17) for x in r['candidate_token_ids'])
                or len(r['prompt_token_ids']) != 256 or len(r['prompt_attention_mask_audit_only']) != 256
                or any(type(x) is not int or not 0 <= x < 49152 for x in r['prompt_token_ids'])
                or not 0 <= r['gold_index'] < len(labels) or r['gold_label'] != labels[r['gold_index']]
                or not r['prompt'].endswith(config.EOT)
                or r['prompt_eot_retained'] != (0 in r['prompt_token_ids'])
                or r['source_record_sha256'] != storage.object_sha({k: r[k] for k in
                    ('id', 'source_row_index', 'category', 'original_record')})):
            raise ValueError(f'invalid pool record: {r["id"]}')
    return value


def permutation():
    order = list(range(config.POOL_SIZE))
    random.Random(config.SEED).shuffle(order)
    return order


def load(path, full=False):
    value = validate(storage.read_json(path))
    if value['generator_sources'] != generator_sources():
        raise ValueError('pool parser/tokenizer implementation changed; use a new pool/output version')
    for item in value['files']:
        storage.verify_file(item, full=full)
    # Audio hashing/decoding is lazy, before each candidate's first forward.
    for r in value['records']:
        stat = Path(r['audio_ref']['path']).stat()
        if stat.st_size != r['audio_file_stat']['bytes'] or stat.st_mtime_ns != r['audio_file_stat']['mtime_ns']:
            raise ValueError(f'audio stat changed: {r["id"]}')
    return value


def ensure(args):
    with storage.output_lock(args.pool_manifest.parent):
        if args.pool_manifest.is_file():
            value = load(args.pool_manifest, full=True)
            if value['source_arguments'] != source_arguments(args):
                raise ValueError('pool belongs to different input paths')
        else:
            source_id = storage.file_identity(args.source_manifest)
            original = sample.load_manifest(args.source_manifest, args.test_json)
            tokenizer, files = load_tokenizer(args.model_dir)
            resolver = dataset.AudioResolver(args)
            rows = [transform(r, i, tokenizer, resolver) for i, r in enumerate(original['records'])]
            order = permutation()
            files = [source_id, storage.file_identity(args.test_json), *files]
            for f in files:
                storage.verify_file(f)
            value = validate(storage.seal({'schema': config.SCHEMA, 'protocol': config.PROTOCOL,
                'source_arguments': source_arguments(args), 'source_manifest_sha256': original['manifest_sha256'],
                'files': files, 'generator_sources': generator_sources(), 'records': rows,
                'category_counts': dict(Counter(r['category'] for r in rows)),
                'shuffle_order': order, 'shuffle_sha256': storage.object_sha(order)}, 'manifest_sha256'))
            storage.write_json(args.pool_manifest, value)
    print(json.dumps({'event': 'TRACING_POOL_READY', 'pool_manifest': str(args.pool_manifest),
        'rows': len(value['records']), 'category_counts': value['category_counts'],
        'manifest_sha256': value['manifest_sha256'], 'shuffle_sha256': value['shuffle_sha256']}, ensure_ascii=False), flush=True)
    return value


def main(argv=None):
    args = arguments.parse_args(argv)
    if args.check_args:
        print(json.dumps(vars(args), default=str, indent=2)); return 0
    ensure(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
