"""CPU frozen1000 inputs; existing manifests are read-only, fallbacks isolated."""
import copy
import json
import random
from collections import Counter
from mizar_layer_pair_search import storage
from mizar_layer_pair_search.manifest import load_tokenizer
from mizar_letter_analysis import manifest as letters
from . import arguments, config


def mapping():
    data=storage.read_json(config.PACKAGE/'mmau_mapping.json')['groups']
    result={name:group for group,body in data.items() for name in body['subcategories']}
    if len(result)!=27:raise ValueError('MMAU mapping needs all27 distinct subcategories')
    return result


def source_args(a):
    keys=('task','model_dir','source_manifest','body_manifest',*(('mmau_metadata','mmau_parquet','mmau_audio_root')
        if a.task=='mmau' else ('mmar_metadata','mmar_audio_root','evaluation_script')))
    return {k:str(getattr(a,k)) for k in keys}


def generator_sources():
    return {p.relative_to(config.ROOT).as_posix():storage.text_sha(p) for p in
        [config.PACKAGE/'config.py',config.PACKAGE/'prepare.py',config.PACKAGE/'mmau_mapping.json',
         config.PACKAGE/'reference_guard.json']}


def transform(r,index,tokenizer,task):
    value=copy.deepcopy(r)
    labels=r['option_letters'];ids=[tokenizer.encode(x,add_special_tokens=False) for x in labels]
    if (any(len(x)!=1 or x[0] in set(tokenizer.all_special_ids)|{0,17} for x in ids)
            or [x[0] for x in ids]!=r['letter_token_ids']):
        raise ValueError('actual release tokenizer does not match frozen single letter targets')
    expected_group=mapping()[r['sub_category']] if task=='mmau' else r['category']
    if r['analysis_group']!=expected_group:raise ValueError('source analysis group differs')
    value.update(source_manifest_index=index,candidate_labels=list(labels),candidate_token_ids=[x[0] for x in ids],
        gold_label=None if r['gold_index'] is None else labels[r['gold_index']],
        tracing_source_record_sha256=storage.object_sha(r))
    return value


def validate(value):
    storage.verify_seal(value,'manifest_sha256')
    task=value['task'];records=value['records']
    if (value['schema']!=config.SCHEMA or value['protocol']!=config.protocol(task)
            or len(records)!=1000 or [r['source_manifest_index'] for r in records]!=list(range(1000))
            or len({r['key'] for r in records})!=1000
            or Counter(r['analysis_group'] for r in records)!=Counter(config.GROUPS[task])
            or sorted(value['shuffle_order'])!=list(range(1000))):raise ValueError('input coverage/groups/order differ')
    order=list(range(1000));random.Random(config.SEED).shuffle(order)
    if order!=value['shuffle_order']:raise ValueError('input shuffle differs from seed1234')
    for r in records:
        n=len(r['choice_bodies']);valid=r['choice_valid_mask'];gold=r['gold_index']
        base={k:v for k,v in r.items() if k not in ('source_manifest_index','candidate_labels','candidate_token_ids','gold_label','tracing_source_record_sha256')}
        if (r['candidate_labels']!=[chr(97+i) for i in range(n)] or len(r['candidate_token_ids'])!=n
                or r['candidate_token_ids']!=r['letter_token_ids'] or len(set(r['candidate_token_ids']))!=n
                or len(valid)!=n or any(type(x) is not bool for x in valid) or sum(valid)<2
                or any(type(x) is not int or not 0<=x<49152 or x in (0,17) for x in r['candidate_token_ids'])
                or len(r['prompt_token_ids'])!=256 or len(r['prompt_attention_mask_audit_only'])!=256
                or not r['prompt'].endswith(' <|endoftext|>') or not r.get('audio_audit')
                or r['dataset']!=task or r['tracing_source_record_sha256']!=storage.object_sha(base)
                or gold is not None and (type(gold) is not int or not 0<=gold<n or not valid[gold])
                or r['gold_label']!=(None if gold is None else r['candidate_labels'][gold])):
            raise ValueError(f'invalid tracing input {r["key"]}')
    return value


def build(a):
    tokenizer,tokens=load_tokenizer(a.model_dir);files=[]
    if a.source_manifest.is_file():
        src=letters.load(a.source_manifest,full=True)
        if src['task']!=a.task or src['source_arguments']['model_dir']!=str(a.model_dir):raise ValueError('wrong source task/model')
        rows=src['records'];files=[*src['files'],storage.file_identity(a.source_manifest)]
        origin={'kind':'existing_letter_manifest','sha256':src['manifest_sha256'],'path':str(a.source_manifest)}
    else:
        module=letters.source_module(a.task)
        if a.body_manifest.is_file():
            src=module.load(a.body_manifest,full=True);files=[storage.file_identity(a.body_manifest)]
            if src['source_arguments']['model_dir']!=str(a.model_dir):raise ValueError('wrong source tokenizer path')
            origin={'kind':'existing_body_manifest','sha256':src['manifest_sha256'],'path':str(a.body_manifest)}
        else:
            # Existing builders return data. They do not publish to old paths.
            src=module.build(a);origin={'kind':'isolated_metadata_build','sha256':src['manifest_sha256']}
        rows=[letters.transform_record(r,tokenizer,a.task,mapping() if a.task=='mmau' else None) for r in src['records']]
        files.extend(src['files'])
    records=[transform(r,i,tokenizer,a.task) for i,r in enumerate(rows)]
    inventory={f['path']:f for f in [*files,*tokens]}
    for f in inventory.values():storage.verify_file(f)
    order=list(range(1000));random.Random(config.SEED).shuffle(order)
    return validate(storage.seal({'schema':config.SCHEMA,'task':a.task,'protocol':config.protocol(a.task),
        'source_arguments':source_args(a),'origin':origin,'files':list(inventory.values()),'generator_sources':generator_sources(),
        'records':records,'shuffle_order':order,'group_counts':config.GROUPS[a.task]},'manifest_sha256'))


def load(path,full=False):
    value=validate(storage.read_json(path))
    if value['generator_sources']!=generator_sources():raise ValueError('tracing input generator changed')
    for f in value['files']:storage.verify_file(f,full=full)
    return value


def ensure(a):
    from .identity import sources
    sources()
    with storage.output_lock(a.manifest.parent):
        if a.manifest.is_file():
            frozen=load(a.manifest,full=True)
            if frozen['source_arguments']!=source_args(a):raise ValueError('input source paths/task differ')
        else:
            frozen=build(a);storage.write_json(a.manifest,frozen)
    print(json.dumps({'event':'TRACING_INPUT_READY','task':a.task,'manifest':str(a.manifest),'rows':1000,
        'group_counts':frozen['group_counts'],'unmapped_gold':sum(r['gold_index'] is None for r in frozen['records'])}),flush=True)
    return frozen


def main(argv=None):
    a=arguments.parse_args(argv)
    if a.check_args:print(json.dumps(vars(a),default=str,indent=2));return 0
    ensure(a);return 0


if __name__=='__main__':raise SystemExit(main())
