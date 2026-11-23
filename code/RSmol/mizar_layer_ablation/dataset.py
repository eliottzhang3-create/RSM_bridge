"""Freeze all MMAU rows without sampling, filtering duplicates or dropping slots."""
from collections import Counter
from pathlib import Path
from mizar_layer_pair_search import audio,data,manifest,storage
from mizar_layer_analysis_common import config

SCHEMA='mizar_mmau1000_likelihood_inputs_v1'


def source_args(args):
    return {name:str(getattr(args,name)) for name in ('model_dir','mmau_metadata','mmau_audio_root','mmau_parquet')}


def generator_sources():
    paths=[Path(__file__),Path(data.__file__),Path(audio.__file__),Path(manifest.__file__),Path(storage.__file__)]
    return {p.relative_to(config.ROOT).as_posix():storage.text_sha(p) for p in paths}


def normalize(index,row):
    choices=row.get('choices'); question=row.get('question'); answer=row.get('answer'); identifier=row.get('id')
    if (not isinstance(choices,list) or not 2<=len(choices)<=26 or not isinstance(question,str) or not question.strip()
            or not isinstance(answer,str) or not isinstance(identifier,str) or not identifier.strip()):
        raise ValueError(f'invalid full-MMAU metadata at {index}')
    bodies=[data.body(c).lower() for c in choices]
    if any(not c or '<|endoftext|>' in c for c in bodies) or '<|endoftext|>' in question:
        raise ValueError('empty/special-token MMAU question or choice')
    matches=[i for i,c in enumerate(bodies) if c==data.body(answer).lower()]
    exact=[i for i,c in enumerate(choices) if c==answer]
    if not matches: raise ValueError('GT body does not match original option')
    gold=exact[0] if len(exact)==1 else matches[0]
    prompt=(question.strip()+' '+' '.join(f'{chr(97+i)}) {c}' for i,c in enumerate(bodies))).lower()
    return {'dataset':'mmau','global_index':index,'source_row_index':index,'id':identifier,'key':'mmau:'+identifier,
        'question_original':question,'choices_original':choices,'valid_original_slots':list(range(len(choices))),
        'choice_bodies':bodies,'prompt':prompt,'gold_index':gold,'gold_original':answer,
        'gold_body_matching_indices':matches,'duplicate_choice_bodies':len(set(bodies))<len(bodies),
        'task':str(row.get('task','')),'difficulty':str(row.get('difficulty','')),'category':str(row.get('category','')),
        'sub_category':str(row.get('sub-category','')),'original_record_sha256':storage.object_sha(row)}


def attach_audio(records,args):
    missing={r['id'] for r in records if not data.contained(args.mmau_audio_root,r['id']+'.wav').is_file()}
    lookup={}
    if missing:
        import pyarrow.parquet as pq
        helper=data.legacy_mmau()
        with pq.ParquetFile(args.mmau_parquet) as parquet:
            if parquet.metadata.num_rows!=1000: raise ValueError('MMAU fallback must have exactly 1000 rows')
            columns=[c for c in ('id','other_attributes') if c in parquet.schema_arrow.names]
            if not columns: raise ValueError('MMAU parquet lacks ID metadata')
            rows=[r for b in parquet.iter_batches(batch_size=64,columns=columns,use_threads=False) for r in b.to_pylist()]
        ids=[helper.extract_row_id(r)[0] for r in rows]
        if len(set(ids))!=1000 or set(ids)!={r['id'] for r in records}: raise ValueError('MMAU metadata/parquet ID sets differ')
        lookup={identifier:index for index,identifier in enumerate(ids)}
    result=[]
    for r in records:
        if r['id'] in missing:
            ref={'kind':'parquet','path':str(args.mmau_parquet),'row_index':lookup[r['id']],
                 'expected_id':r['id'],'format':'mmau','base_dir':str(args.mmau_parquet.parent)}
        else: ref={'kind':'file','path':str(data.contained(args.mmau_audio_root,r['id']+'.wav'))}
        if not Path(ref['path']).is_file(): raise FileNotFoundError(ref['path'])
        result.append({**r,'audio_ref':ref})
    return result


def validate(frozen):
    storage.verify_seal(frozen,'manifest_sha256')
    records=frozen['records']
    if (frozen['schema']!=SCHEMA or len(records)!=1000 or [r['global_index'] for r in records]!=list(range(1000))
            or len({r['key'] for r in records})!=1000): raise ValueError('not exact 1000-row MMAU coverage')
    for r in records:
        n=len(r['choice_bodies'])
        if (n<2 or len(r['choice_token_ids'])!=n or not 0<=r['gold_index']<n or len(r['prompt_token_ids'])!=256
                or any(not ids or len(ids)+383>1024 for ids in r['choice_token_ids']) or not r.get('audio_audit')):
            raise ValueError('invalid complete-MMAU frozen record')
    return frozen


def load(path,full=True):
    frozen=validate(storage.read_json(path))
    if frozen['generator_sources']!=generator_sources(): raise ValueError('full-MMAU input implementation changed')
    for item in frozen['files']: storage.verify_file(item,full=full)
    return frozen


def build(args):
    tokenizer,token_files=manifest.load_tokenizer(args.model_dir)
    meta=storage.file_identity(args.mmau_metadata); rows=storage.read_json(args.mmau_metadata)
    if not isinstance(rows,list) or len(rows)!=1000 or len({r['id'] for r in rows})!=1000:
        raise ValueError('expected all 1000 distinct original MMAU test-mini rows')
    if Counter(str(r.get('task','')) for r in rows)!=Counter(music=334,sound=333,speech=333):
        raise ValueError('MMAU test-mini task distribution changed')
    records=[data.tokenize_record(normalize(i,row),tokenizer) for i,row in enumerate(rows)]
    records=attach_audio(records,args)
    files={meta['path']:meta}
    # Pin file bytes before decoding; no selected row may be silently skipped.
    for r in records:
        path=r['audio_ref']['path']
        if path not in files: files[path]=storage.file_identity(path)
    try:
        audio.prepare_selected_rows(records)
        for r in records:
            _,r['audio_audit']=audio.load_fixed20(r['audio_ref'])
            print(__import__('json').dumps({'event':'MMAU_INPUT_FROZEN','index':r['global_index'],'id':r['id']}),flush=True)
    finally: audio.clear_selected_rows()
    identities=[*files.values(),*token_files]
    for item in identities: storage.verify_file(item)
    return validate(storage.seal({'schema':SCHEMA,'records':records,'files':identities,'source_arguments':source_args(args),
        'generator_sources':generator_sources(),'metadata_canonical_sha256':storage.object_sha(rows),
        'duplicate_choice_rows':sum(r['duplicate_choice_bodies'] for r in records),
        'gold_slot_policy':'unique_exact_original_choice_else_first_matching_body',
        'task_counts':dict(Counter(r['task'] for r in records))},'manifest_sha256'))


def ensure(args):
    with storage.output_lock(args.manifest.parent):
        if args.manifest.exists():
            frozen=load(args.manifest)
            if frozen['source_arguments']!=source_args(args): raise ValueError('existing MMAU manifest uses different source paths')
        else:
            frozen=build(args); storage.write_json(args.manifest,frozen)
    return frozen
