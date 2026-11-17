"""Metadata conversion without model inputs derived from reference answers."""
import random
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from . import config, storage

LABEL = re.compile(r'^\s*(?:\([A-Za-z]\)|[A-Za-z][).])\s*')
OPTION = re.compile(r'(?<!\S)([a-dA-D])\)\s+')


class Ineligible(ValueError):
    pass


def body(text):
    if not isinstance(text, str):
        raise Ineligible('non_text_option')
    return LABEL.sub('', text, count=1).strip()


def reasonaqa_mcq(row):
    text = row.get('input')
    if not isinstance(text, str):
        raise Ineligible('missing_question')
    matches = list(OPTION.finditer(text))
    if [m.group(1).lower() for m in matches] != list('abcd'):
        raise Ineligible('ambiguous_or_missing_ABCD_option_boundaries')
    question = text[:matches[0].start()].strip()
    choices = [text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)].strip()
               for i, m in enumerate(matches)]
    return question, choices


def normalize_record(dataset, source_index, row):
    if dataset == 'reasonaqa':
        if row.get('subtype') not in ('AudioCaps-MCQ.json', 'Clotho-MCQ.json'):
            raise Ineligible('not_two_MCQ_subtypes')
        if row.get('filepath2', '').strip() not in ('', row.get('filepath1', '').strip()):
            raise Ineligible('two_distinct_recordings')
        question, original = reasonaqa_mcq(row)
        answer, identifier = row['answer'], f'reasonaqa_test:{source_index}'
        category = row['subtype']
    elif dataset == 'mmsu':
        question = row['question']
        original = [row[f'choice_{c}'] for c in 'abcd']
        answer, identifier = row['answer_gt'], str(row['id'])
        category = row.get('task_name', '')
    else:
        question, original = row['question'], row['choices']
        answer, identifier = row['answer'], str(row['id'])
        category = row.get('task', row.get('category', ''))
    if not isinstance(question, str) or not question.strip() or not identifier.strip():
        raise Ineligible('empty_question_or_id')
    if not isinstance(original, list) or not 2 <= len(original) <= 26:
        raise Ineligible('invalid_options')
    slots = [i for i, value in enumerate(original) if value is not None and str(value).strip()]
    choices = [body(original[i]) for i in slots]
    normalized = [c.lower() for c in choices]
    if (len(choices) < 2 or any(not c for c in choices) or len(set(normalized)) != len(choices)):
        raise Ineligible('empty_or_duplicate_normalized_options')
    correct = [i for i, c in enumerate(normalized) if c == body(answer).lower()]
    if len(correct) != 1:
        raise Ineligible('GT_does_not_uniquely_match_option_body')
    prompt = (question.strip() + ' ' + ' '.join(f'{chr(97+i)}) {c}' for i, c in enumerate(choices))).lower()
    if '<|endoftext|>' in prompt or any('<|endoftext|>' in c for c in normalized):
        raise Ineligible('question_or_option_contains_EOT')
    return {'dataset': dataset, 'source_row_index': source_index, 'id': identifier,
            'key': f'{dataset}:{identifier}', 'category': str(category),
            'question_original': question, 'choices_original': original, 'valid_original_slots': slots,
            'choice_bodies': normalized, 'prompt': prompt, 'gold_index': correct[0],
            'gold_original': answer, 'original_record_sha256': storage.object_sha(row)}


def tokenize_record(record, tokenizer):
    tokens = tokenizer(record['prompt'], padding='max_length', truncation=True, max_length=256,
                       add_special_tokens=True)
    untruncated = tokenizer.encode(record['prompt'], add_special_tokens=True)
    choices = [tokenizer.encode(c, add_special_tokens=False) for c in record['choice_bodies']]
    if any(not ids for ids in choices):
        raise Ineligible('empty_tokenized_option')
    if any(len(ids) + 383 > 1024 for ids in choices):
        raise Ineligible('full_option_exceeds_1024_context')
    if len(tokens['input_ids']) != 256:
        raise ValueError('tokenizer did not return prompt256')
    return {**record, 'prompt_token_ids': list(tokens['input_ids']),
            'prompt_attention_mask_audit_only': list(tokens['attention_mask']),
            'prompt_original_token_count': len(untruncated), 'prompt_truncated': len(untruncated) > 256,
            'choice_token_ids': choices}


def select_records(pools, tokenizer):
    selected, audits = [], {}
    for dataset in config.DATASETS:
        eligible, excluded, seen = [], Counter(), set()
        for source_index, row in pools[dataset]:
            try:
                record = tokenize_record(normalize_record(dataset, source_index, row), tokenizer)
            except Ineligible as exc:
                excluded[str(exc)] += 1
                continue
            if record['key'] in seen:
                raise ValueError(f'duplicate eligible ID: {record["key"]}')
            seen.add(record['key'])
            eligible.append(record)
        if len(eligible) < config.PER_DATASET:
            raise ValueError(f'{dataset}: need 25 eligible rows, got {len(eligible)}; exclusions={excluded}')
        indices = sorted(random.Random(config.SEED).sample(range(len(eligible)), config.PER_DATASET))
        sample = [eligible[i] for i in indices]
        audits[dataset] = {'eligible_rows': len(eligible), 'excluded': dict(excluded),
                           'eligible_sha256': storage.object_sha(eligible), 'selected_eligible_indices': indices,
                           'selected_categories': dict(Counter(r['category'] for r in sample))}
        selected.extend(sample)
    return [{**r, 'global_index': i} for i, r in enumerate(selected)], audits


def legacy_mmau():
    scripts = str(config.ROOT / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as mmau
    return mmau


def read_pools(args):
    from mizar_s1_ced_mmsu_eval import dataset as mmsu
    mmau_rows, mmar_rows = storage.read_json(args.mmau_metadata), storage.read_json(args.mmar_metadata)
    if len(mmau_rows) != 1000 or len(mmar_rows) != 1000:
        raise ValueError('MMAU test-mini and MMAR must each have exactly 1000 source rows')
    paths = mmsu.shard_paths(args.mmsu_dir)
    mmsu_rows, mmsu_audit = mmsu.preflight(paths)
    reasons = storage.read_json(args.reasonaqa_test)
    for name, rows in [('mmau', mmau_rows), ('mmar', mmar_rows)]:
        if len({str(r['id']) for r in rows}) != len(rows):
            raise ValueError(f'duplicate source ID: {name}')
    files = [args.mmau_metadata, args.mmar_metadata, args.reasonaqa_test, *paths]
    return {'mmau': list(enumerate(mmau_rows)), 'mmsu': list(enumerate(mmsu_rows)),
            'mmar': list(enumerate(mmar_rows)), 'reasonaqa': list(enumerate(reasons))}, files, mmsu_audit


def contained(root, relative):
    root = Path(root).resolve()
    raw = str(relative).replace('\\', '/')
    if Path(raw).is_absolute() or '..' in Path(raw).parts or re.match(r'^[A-Za-z]:', raw):
        raise ValueError(f'unsafe relative audio path: {raw}')
    path = (root / raw).resolve()
    if not path.is_relative_to(root):
        raise ValueError('audio escapes root')
    return path


def attach_audio(records, pools, args):
    """Resolve only the selected rows. Missing audio is fatal, never resampled."""
    from mizar_s1_ced_reasonaqa_eval.dataset import AudioResolver
    from mizar_s1_ced_mmsu_eval import dataset as mmsu
    resolver = AudioResolver(SimpleNamespace(audiocaps_root=args.audiocaps_root,
                             clotho_root=args.clotho_root, clotho_aqa_root=args.clotho_root))
    raw = {name: dict(rows) for name, rows in pools.items()}
    missing_mmau = {r['id'] for r in records if r['dataset'] == 'mmau'
                    and not contained(args.mmau_audio_root, r['id'] + '.wav').is_file()}
    mmau_refs = {}
    if missing_mmau:
        import pyarrow.parquet as pq
        mmau = legacy_mmau()
        if not args.mmau_parquet.is_file():
            raise FileNotFoundError(f'MMAU fallback parquet missing: {args.mmau_parquet}')
        with pq.ParquetFile(args.mmau_parquet) as parquet:
            if parquet.metadata.num_rows != 1000:
                raise ValueError('MMAU fallback parquet must contain test-mini 1000 rows')
            columns = [name for name in ('id', 'other_attributes') if name in parquet.schema_arrow.names]
            if not columns:
                raise ValueError('MMAU fallback parquet lacks ID metadata columns')
            parquet_ids = [row for batch in parquet.iter_batches(batch_size=64, columns=columns, use_threads=False)
                           for row in batch.to_pylist()]
        for index, row in enumerate(parquet_ids):
            identifier = mmau.extract_row_id(row)[0]
            if identifier in missing_mmau:
                if identifier in mmau_refs:
                    raise ValueError('duplicate MMAU fallback ID')
                mmau_refs[identifier] = {'kind': 'parquet', 'path': str(args.mmau_parquet.resolve()),
                    'row_index': index, 'expected_id': identifier, 'format': 'mmau',
                    'base_dir': str(args.mmau_parquet.parent.resolve())}
        if set(mmau_refs) != missing_mmau:
            raise ValueError(f'MMAU audio IDs absent from parquet: {missing_mmau - mmau_refs.keys()}')
    shard_paths = mmsu.shard_paths(args.mmsu_dir)
    result = []
    for record in records:
        name, index = record['dataset'], record['source_row_index']
        row = raw[name][index]
        if name == 'mmau':
            if record['id'] in mmau_refs:
                ref = mmau_refs[record['id']]
            else:
                ref = {'kind': 'file', 'path': str(contained(args.mmau_audio_root, record['id'] + '.wav'))}
        elif name == 'mmar':
            ref = {'kind': 'file', 'path': str(contained(args.mmar_audio_root, row['audio_path']))}
        elif name == 'reasonaqa':
            category = 'audiocaps_mcq' if row['subtype'] == 'AudioCaps-MCQ.json' else 'clotho_mcq'
            ref = {'kind': 'file', 'path': resolver.resolve(row['filepath1'], category)['path']}
        else:
            shard = 0 if index < 1667 else 1 if index < 3334 else 2
            ref = {'kind': 'parquet', 'path': str(shard_paths[shard].resolve()),
                   'row_index': index - (0, 1667, 3334)[shard], 'expected_id': record['id'],
                   'format': 'mmsu', 'base_dir': str(args.mmsu_dir.resolve())}
        if not Path(ref['path']).is_file():
            raise FileNotFoundError(f'missing selected audio source: {ref}')
        result.append({**record, 'audio_ref': ref})
    return result


def validate_manifest(manifest):
    storage.verify_seal(manifest, 'manifest_sha256')
    records = manifest.get('records', [])
    if (manifest.get('schema') != config.SCHEMA or manifest.get('protocol') != config.PROTOCOL
            or manifest.get('seed') != config.SEED or len(records) != config.ROWS
            or len({r['key'] for r in records}) != config.ROWS
            or [r['global_index'] for r in records] != list(range(config.ROWS))
            or Counter(r['dataset'] for r in records) != Counter({d: 25 for d in config.DATASETS})):
        raise ValueError('manifest schema/protocol/counts/unique keys mismatch')
    for r in records:
        n = len(r['choice_bodies'])
        if (n < 2 or len(r['choice_token_ids']) != n or not 0 <= r['gold_index'] < n
                or len(r['prompt_token_ids']) != 256 or not r.get('audio_audit')
                or any(not ids or len(ids) + 383 > 1024 for ids in r['choice_token_ids'])):
            raise ValueError(f'invalid manifest record: {r["key"]}')
    return manifest
