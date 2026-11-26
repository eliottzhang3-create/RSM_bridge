"""Full1000 coverage, complete baselines and all qualifying selected rows."""
import copy
import csv
import json
from collections import Counter
from mizar_layer_pair_search import storage
from . import config,scoring


def rank_path(directory,rank):return directory/'ranks'/f'rank{rank:02d}'/'screening.jsonl'


def make_row(record,expected,position,clean,corrupted,audio_audit,cache_sha):
    reason=scoring.reason(record,clean,corrupted)
    return {'identity_sha256':expected['identity_sha256'],'task':record['dataset'],
        'source_manifest_index':record['source_manifest_index'],'shuffle_position':position,'key':record['key'],
        'analysis_group':record['analysis_group'],'record_sha256':storage.object_sha(record),
        'clean':clean,'corrupted':corrupted,'accepted':reason=='accepted','reason':reason,
        'probability_gap':None if record['gold_index'] is None else clean['target_probability']-corrupted['target_probability'],
        'audio_audit':audio_audit,'cache_meta_sha256':cache_sha}


def validate(row,record,expected,position):
    storage.verify_seal(row);scoring.validate(record,row['clean']);scoring.validate(record,row['corrupted'])
    recomputed=storage.seal(make_row(record,expected,position,row['clean'],row['corrupted'],row['audio_audit'],row['cache_meta_sha256']))
    if row!=recomputed or bool(row['cache_meta_sha256'])!=row['accepted'] or row['audio_audit']!=record['audio_audit']:
        raise ValueError('screening identity/qualification/audio/cache mismatch')
    return row


def read_rank(directory,rank,inputs,expected):
    rows=storage.recover_jsonl(rank_path(directory,rank));positions=config.indices(rank,config.ROWS)
    if len(rows)>len(positions):raise ValueError('too many screening rows')
    for row,pos in zip(rows,positions):validate(row,inputs['records'][inputs['shuffle_order'][pos]],expected,pos)
    return rows


def merge(parts,inputs,expected):
    if len(parts)!=8 or any(len(p)!=125 for p in parts):raise ValueError('screening needs exactly eight125-row shards')
    ordered=sorted([r for part in parts for r in part],key=lambda r:r['shuffle_position'])
    if [r['shuffle_position'] for r in ordered]!=list(range(1000)):raise ValueError('incomplete/duplicate screening positions')
    for row,pos in zip(ordered,range(1000)):validate(row,inputs['records'][inputs['shuffle_order'][pos]],expected,pos)
    return ordered


def counts(rows):
    n=len(rows);clean=sum(r['clean']['correct'] for r in rows);corrupt=sum(r['corrupted']['correct'] for r in rows)
    selected=sum(r['accepted'] for r in rows)
    return {'total':n,'clean_correct':clean,'clean_accuracy':clean/n if n else None,
        'corrupted_correct':corrupt,'corrupted_accuracy':corrupt/n if n else None,
        'selected':selected,'selection_rate':selected/n if n else None,'reasons':dict(Counter(r['reason'] for r in rows))}


def selected_manifest(ordered,inputs,expected):
    by_source=sorted([r for r in ordered if r['accepted']],key=lambda r:r['source_manifest_index'])
    rows=[]
    for row in by_source:
        r=copy.deepcopy(inputs['records'][row['source_manifest_index']])
        r.update(selected_index=len(rows),shuffle_position=row['shuffle_position'],screening={k:row[k] for k in
            ('clean','corrupted','probability_gap','audio_audit','cache_meta_sha256')})
        rows.append(r)
    return storage.seal({'schema':config.SCHEMA,'task':inputs['task'],'identity_sha256':expected['identity_sha256'],
        'input_manifest_sha256':inputs['manifest_sha256'],'screening_sha256':storage.object_sha(ordered),
        'selection':'all_qualifying_no_quota','records':rows,'selected':len(rows),
        'group_counts':{g:sum(r['analysis_group']==g for r in rows) for g in config.GROUPS[inputs['task']]}},'manifest_sha256')


def publish(directory,ordered,inputs,expected):
    frozen=selected_manifest(ordered,inputs,expected)
    report=storage.seal({'schema':config.SCHEMA,'task':inputs['task'],'identity_sha256':expected['identity_sha256'],
        'status':'SELECTED' if frozen['selected'] else 'NO_QUALIFYING','selection_quota':None,
        **counts(ordered),'groups':{g:counts([r for r in ordered if r['analysis_group']==g]) for g in config.GROUPS[inputs['task']]},
        'screening_sha256':storage.object_sha(ordered),'selected_manifest_sha256':frozen['manifest_sha256']})
    path=directory/'selected/manifest.json'
    if path.exists() and storage.read_json(path)!=frozen:raise ValueError('selected manifest differs from complete screening')
    storage.write_jsonl(directory/'selected/selected_records.jsonl',frozen['records'])
    storage.write_jsonl(directory/'screening/shuffled_journal.jsonl',ordered)
    storage.write_jsonl(directory/'screening/per_question.jsonl',sorted(ordered,key=lambda r:r['source_manifest_index']))
    storage.write_json(directory/'selection_report.json',report)
    fields=('group','total','clean_correct','clean_accuracy','corrupted_correct','corrupted_accuracy','selected','selection_rate')
    with (directory/'screening_summary.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for group,body in [('overall',report),*report['groups'].items()]:writer.writerow({'group':group,**{k:body[k] for k in fields if k!='group'}})
    # Selected manifest is the completion marker, published last.
    storage.write_json(path,frozen)
    print(json.dumps({'event':'SCREENING_DONE','task':inputs['task'],**counts(ordered),'groups':report['groups']},ensure_ascii=False),flush=True)
    return frozen,report


def reconstruct(directory,inputs,expected):
    parts=[read_rank(directory,r,inputs,expected) for r in range(8)]
    if any(len(part)!=125 for part in parts):return None
    ordered=merge(parts,inputs,expected);frozen=selected_manifest(ordered,inputs,expected)
    if storage.read_json(directory/'selected/manifest.json')!=frozen:raise ValueError('selected file differs from rank journals')
    report=storage.verify_seal(storage.read_json(directory/'selection_report.json'))
    if (report['schema']!=config.SCHEMA or report['task']!=inputs['task'] or report['identity_sha256']!=expected['identity_sha256']
            or report['status']!=('SELECTED' if frozen['selected'] else 'NO_QUALIFYING') or report['selection_quota'] is not None
            or report['selected_manifest_sha256']!=frozen['manifest_sha256'] or report['screening_sha256']!=storage.object_sha(ordered)):
        raise ValueError('screening report changed')
    for g in config.GROUPS[inputs['task']]:
        if report['groups'][g]!=counts([r for r in ordered if r['analysis_group']==g]):raise ValueError('screening group stats changed')
    if any(report[k]!=v for k,v in counts(ordered).items()):raise ValueError('screening full1000 stats changed')
    return frozen,report
