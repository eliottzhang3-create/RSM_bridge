"""Retokenize the exact old questions with prompt EOT and pure letter targets."""
from collections import Counter
import copy
import json
from pathlib import Path
from mizar_layer_pair_search import storage
from mizar_layer_pair_search.manifest import load_tokenizer
from . import arguments, config


def mapping():
    value = storage.read_json(config.PACKAGE/'mmau_mapping.json')
    groups = value['groups']
    names = [name for group in ('perception', 'semantic', 'reasoning') for name in groups[group]['subcategories']]
    if len(names) != 27 or len(set(names)) != 27:
        raise ValueError('MMAU mapping must cover the original 27 distinct subcategories')
    return {name: group for group in groups for name in groups[group]['subcategories']}


def source_module(task):
    if task == 'cutoff':
        from mizar_audio_cutoff1000 import manifest as module
    elif task == 'mmau':
        from mizar_layer_ablation import dataset as module
    else:
        from mizar_layer_ablation_mmar import dataset as module
    return module


def load_source(args, full=True):
    module = source_module(args.task)
    if not args.source_manifest.is_file():
        original = copy.copy(args)
        original.manifest = args.source_manifest
        original.compare_manifest = None
        # Existing proven generators retain their original eligibility/selection.
        module.ensure(original)
    source = module.load(args.source_manifest, full=full)
    source_model = source['source_arguments'].get('model_dir')
    if source_model != str(args.model_dir):
        raise ValueError('old manifest tokenizer model-dir differs; use its original model-dir')
    if len(source['records']) != config.ROWS:
        raise ValueError('source must be the original complete 1000-row manifest')
    return source


def source_args(args):
    return {'task': args.task, 'source_manifest': str(args.source_manifest), 'model_dir': str(args.model_dir)}


def generator_sources(task):
    paths = [config.PACKAGE/name for name in ('config.py', 'manifest.py', 'mmau_mapping.json')]
    paths.extend(config.ROOT/name for name in ('mizar_layer_pair_search/manifest.py',
                 'mizar_layer_pair_search/config.py', 'mizar_layer_pair_search/storage.py'))
    module = source_module(task)
    paths.extend(config.ROOT/name for name in module.generator_sources())
    return {p.relative_to(config.ROOT).as_posix(): storage.text_sha(p) for p in sorted(set(paths))}


def transform_record(record, tokenizer, task, subcategory_map=None):
    value = copy.deepcopy(record)
    n = len(record['choice_bodies'])
    valid = record.get('choice_valid_mask', [True]*n)
    if not 2 <= n <= 26 or len(valid) != n or sum(valid) < 2:
        raise ValueError('invalid original option slots')
    letters = [chr(97+i) for i in range(n)]
    encoded = [tokenizer.encode(letter, add_special_tokens=False) for letter in letters]
    if any(len(ids) != 1 or type(ids[0]) is not int or ids[0] in (0,17) or not 0 <= ids[0] < 49152 for ids in encoded):
        raise ValueError('each bare lowercase letter must be one ordinary vocabulary token')
    if len({ids[0] for ids in encoded}) != n:
        raise ValueError('option letters must have distinct token IDs')
    prompt = record['prompt']+config.EOT_SUFFIX
    tokens = tokenizer(prompt, padding='max_length', truncation=True, max_length=256, add_special_tokens=True)
    full_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if len(tokens['input_ids']) != 256 or len(tokens['attention_mask']) != 256:
        raise ValueError('expected original right-padded prompt256 including EOT suffix')
    if task == 'cutoff':
        group = record['dataset']
    elif task == 'mmau':
        if subcategory_map is None:
            subcategory_map = mapping()
        if record['sub_category'] not in subcategory_map:
            raise ValueError(f'unknown MMAU user grouping: {record["sub_category"]}')
        group = subcategory_map[record['sub_category']]
    else:
        group = record['category']
    if group not in config.GROUP_COUNTS[task]:
        raise ValueError('unexpected analysis group')
    value.update(
        source_record_sha256=storage.object_sha(record), source_prompt=record['prompt'],
        source_body_token_ids_audit_only=record['choice_token_ids'],
        prompt=prompt, prompt_token_ids=list(tokens['input_ids']),
        prompt_attention_mask_audit_only=list(tokens['attention_mask']),
        prompt_original_token_count=len(full_ids), prompt_truncated=len(full_ids)>256,
        prompt_eot_retained=0 in tokens['input_ids'],
        option_letters=letters, letter_token_ids=[ids[0] for ids in encoded],
        choice_token_ids=[ids if enabled else [] for ids, enabled in zip(encoded, valid)],
        choice_valid_mask=list(valid), analysis_group=group,
    )
    return value


def validate(value):
    storage.verify_seal(value, 'manifest_sha256')
    task = value['task']
    records = value['records']
    if (value['schema'] != config.SCHEMA or value['protocol'] != config.protocol(task)
            or len(records) != config.ROWS or [r['global_index'] for r in records] != list(range(config.ROWS))
            or len({r['key'] for r in records}) != config.ROWS
            or Counter(r['analysis_group'] for r in records) != Counter(config.GROUP_COUNTS[task])):
        raise ValueError('letter manifest protocol/counts/groups/coverage mismatch')
    inventory = []
    for r in records:
        n = len(r['choice_bodies'])
        valid = r['choice_valid_mask']
        if (r['prompt'] != r['source_prompt']+config.EOT_SUFFIX or len(r['prompt_token_ids']) != 256
                or len(r['prompt_attention_mask_audit_only']) != 256
                or r['prompt_eot_retained'] is not (0 in r['prompt_token_ids'])
                or r['option_letters'] != [chr(97+i) for i in range(n)] or len(valid) != n or sum(valid)<2
                or any(type(b) is not bool for b in valid) or len(r['letter_token_ids']) != n
                or len(set(r['letter_token_ids'])) != n or len(r['choice_token_ids']) != n
                or r['choice_token_ids'] != [[tid] if enabled else [] for tid, enabled in zip(r['letter_token_ids'],valid)]
                or any(type(t) is not int or not 0 <= t < 49152 for t in r['prompt_token_ids']+r['letter_token_ids'])
                or any(t in (0,17) for t in r['letter_token_ids']) or not r.get('audio_audit')
                or r['gold_index'] is not None and (type(r['gold_index']) is not int or not 0<=r['gold_index']<n or not valid[r['gold_index']])):
            raise ValueError(f'invalid frozen EOT/letter record: {r["key"]}')
        inventory.append({'key':r['key'],'source_record_sha256':r['source_record_sha256']})
    if inventory != value['source_record_inventory']:
        raise ValueError('original source row identity/order changed')
    if value['group_counts'] != dict(Counter(r['analysis_group'] for r in records)):
        raise ValueError('changed group denominators')
    return value


def build(args, source):
    sources = generator_sources(args.task)
    source_identity = storage.file_identity(args.source_manifest)
    tokenizer, tokenizer_files = load_tokenizer(args.model_dir)
    subcategories = mapping() if args.task == 'mmau' else None
    records = [transform_record(r, tokenizer, args.task, subcategories) for r in source['records']]
    files = {item['path']: item for item in [*source['files'], *tokenizer_files, source_identity]}
    for item in files.values():
        storage.verify_file(item)
    if generator_sources(args.task) != sources:
        raise ValueError('letter generator changed during conversion')
    return validate(storage.seal({
        'schema':config.SCHEMA, 'task':args.task, 'protocol':config.protocol(args.task),
        'source_arguments':source_args(args), 'source_manifest_sha256':source['manifest_sha256'],
        'source_record_inventory':[{'key':r['key'],'source_record_sha256':r['source_record_sha256']} for r in records],
        'generator_sources':sources, 'records':records, 'files':list(files.values()),
        'group_counts':dict(Counter(r['analysis_group'] for r in records)),
        'input_audit':{'same_original_questions':True,'new_sampling':False,
            'prompt_eot_retained_rows':sum(r['prompt_eot_retained'] for r in records),
            'prompt_truncated_rows':sum(r['prompt_truncated'] for r in records),
            'option_count_distribution':dict(Counter(str(len(r['choice_bodies'])) for r in records)),
            'empty_option_rows':sum(not all(r['choice_valid_mask']) for r in records),
            'unmapped_gold_rows':sum(r['gold_index'] is None for r in records)},
    }, 'manifest_sha256'))


def load(path, full=False):
    value = validate(storage.read_json(path))
    if value['generator_sources'] != generator_sources(value['task']):
        raise ValueError('letter manifest tokenizer/parser/source implementation changed')
    for item in value['files']:
        storage.verify_file(item, full=full)
    return value


def ensure(args):
    with storage.output_lock(args.manifest.parent):
        if args.manifest.exists():
            value = load(args.manifest, full=True)
            if value['source_arguments'] != source_args(args):
                raise ValueError('letter manifest belongs to different original input paths/task')
        else:
            source = load_source(args)
            value = build(args, source)
            storage.write_json(args.manifest, value)
    print(json.dumps({'event':'LETTER_EOT_MANIFEST_READY','task':args.task,'manifest':str(args.manifest),
        'manifest_sha256':value['manifest_sha256'],'source_manifest_sha256':value['source_manifest_sha256'],
        'group_counts':value['group_counts'],'input_audit':value['input_audit']}, ensure_ascii=False),flush=True)
    return value


def main(argv=None):
    args=arguments.parse_args(argv)
    if args.check_args:
        print(json.dumps(vars(args),default=str,indent=2));return 0
    ensure(args)
    return 0


if __name__=='__main__':raise SystemExit(main())
