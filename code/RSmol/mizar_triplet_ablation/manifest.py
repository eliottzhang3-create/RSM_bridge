"""Freeze exact full1000 letter/EOT inputs, original source paths read-only."""
import copy
import json
from collections import Counter
from mizar_layer_pair_search import storage
from mizar_layer_pair_search.manifest import load_tokenizer
from mizar_letter_analysis import manifest as letters
from . import arguments,config


def source_args(a):
    keys=('task','model_dir','source_manifest','body_manifest',*(('mmau_metadata','mmau_parquet','mmau_audio_root')
        if a.task=='mmau' else ('mmar_metadata','mmar_audio_root','evaluation_script')))
    return {k:str(getattr(a,k)) for k in keys}


def generator_sources():
    return {p.relative_to(config.ROOT).as_posix():storage.text_sha(p) for p in
        [config.PACKAGE/'config.py',config.PACKAGE/'manifest.py',config.PACKAGE/'mmau_mapping.json',config.PACKAGE/'reference_guard.json']}


def validate(value):
    storage.verify_seal(value,'manifest_sha256');task=value['task'];records=value['records']
    if (value['schema']!=config.SCHEMA or value['protocol']!=config.protocol(task) or len(records)!=1000
            or [r['global_index'] for r in records]!=list(range(1000)) or len({r['key'] for r in records})!=1000
            or Counter(r['analysis_group'] for r in records)!=Counter(config.GROUP_COUNTS[task])):raise ValueError('wrong full1000 coverage/groups')
    for r in records:
        n=len(r['choice_bodies']);valid=r['choice_valid_mask'];gold=r['gold_index']
        if (r['dataset']!=task or r['option_letters']!=[chr(97+i) for i in range(n)] or len(valid)!=n
                or any(type(x) is not bool for x in valid) or sum(valid)<2 or len(r['letter_token_ids'])!=n
                or len(set(r['letter_token_ids']))!=n
                or any(type(x) is not int or not 0<=x<49152 or x in (0,17) for x in r['letter_token_ids'])
                or r['choice_token_ids']!=[[tid] if enabled else [] for tid,enabled in zip(r['letter_token_ids'],valid)]
                or len(r['prompt_token_ids'])!=256 or len(r['prompt_attention_mask_audit_only'])!=256
                or r['prompt']!=r['source_prompt']+' <|endoftext|>' or not r.get('audio_audit')
                or gold is not None and (type(gold) is not int or not 0<=gold<n or not valid[gold])):
            raise ValueError(f'invalid letter input {r["key"]}')
    if value['record_inventory']!=[storage.object_sha(r) for r in records]:raise ValueError('input row changed')
    return value


def build(a):
    tokenizer,tokens=load_tokenizer(a.model_dir);files=[]
    if a.source_manifest.is_file():
        src=letters.load(a.source_manifest,full=True)
        if src['task']!=a.task or src['source_arguments']['model_dir']!=str(a.model_dir):raise ValueError('source task/model differs')
        records=copy.deepcopy(src['records']);files=[*src['files'],storage.file_identity(a.source_manifest)]
        origin={'kind':'existing_letter_manifest','path':str(a.source_manifest),'sha256':src['manifest_sha256']}
    else:
        module=letters.source_module(a.task)
        if a.body_manifest.is_file():
            src=module.load(a.body_manifest,full=True);files=[storage.file_identity(a.body_manifest)]
            if src['source_arguments']['model_dir']!=str(a.model_dir):raise ValueError('source model path differs')
            origin={'kind':'existing_body_manifest','path':str(a.body_manifest),'sha256':src['manifest_sha256']}
        else:
            src=module.build(a);origin={'kind':'isolated_full_metadata_build','sha256':src['manifest_sha256']}
        records=[letters.transform_record(r,tokenizer,a.task) for r in src['records']];files.extend(src['files'])
    special=set(tokenizer.all_special_ids)|{0,17}
    for r in records:
        ids=[tokenizer.encode(c,add_special_tokens=False) for c in r['option_letters']]
        if any(len(x)!=1 or x[0] in special for x in ids) or [x[0] for x in ids]!=r['letter_token_ids']:
            raise ValueError('actual release tokenizer differs from saved letter targets')
    inventory={f['path']:f for f in [*files,*tokens]}
    for f in inventory.values():storage.verify_file(f)
    return validate(storage.seal({'schema':config.SCHEMA,'task':a.task,'protocol':config.protocol(a.task),
        'source_arguments':source_args(a),'origin':origin,'generator_sources':generator_sources(),'files':list(inventory.values()),
        'record_inventory':[storage.object_sha(r) for r in records],'records':records,
        'input_audit':{'same_full1000':True,'sampling':False,'unmapped_gold_rows':sum(r['gold_index'] is None for r in records),
            'empty_option_rows':sum(not all(r['choice_valid_mask']) for r in records),
            'group_counts':config.GROUP_COUNTS[a.task]}},'manifest_sha256'))


def load(path,full=False):
    value=validate(storage.read_json(path))
    if value['generator_sources']!=generator_sources():raise ValueError('triplet input generator changed')
    for f in value['files']:storage.verify_file(f,full=full)
    return value


def ensure(a):
    from .identity import sources
    sources()
    with storage.output_lock(a.manifest.parent):
        if a.manifest.is_file():
            frozen=load(a.manifest,full=True)
            if frozen['source_arguments']!=source_args(a):raise ValueError('input paths differ')
        else:frozen=build(a);storage.write_json(a.manifest,frozen)
    print(json.dumps({'event':'TRIPLET_INPUT_READY','task':a.task,'rows':1000,'manifest':str(a.manifest),
        'groups':config.GROUP_COUNTS[a.task]}),flush=True);return frozen


def main(argv=None):
    a=arguments.parse_args(argv)
    if a.check_args:print(json.dumps(vars(a),default=str,indent=2));return 0
    ensure(a);return 0


if __name__=='__main__':raise SystemExit(main())
