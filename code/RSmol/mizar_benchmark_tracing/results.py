"""Dynamic N, question-weighted RR and all original dataset groups."""
import csv
from mizar_layer_pair_search import storage
from mizar_reasonaqa_tracing500.results import recovery,stats as original_stats
from . import config,scoring


def stats(rows):
    if rows:return original_stats(rows)
    return {'total':0,'mean_RR':None,'median_RR':None,'mean_RR_CI95':None,'RR_quantiles':None,
        'negative_RR':0,'above_one_RR':0,'small_gap_rows':0,'gap_quantiles':None,'sensitivity_mean_RR_gap_at_least':{},
        'patched_correct':0,'patched_accuracy':None,'mean_Pclean':None,'mean_Pcorrupted':None,'mean_Ppatched':None}


def make_row(r,expected,frozen,layer,patched):
    s=r['screening'];pc=s['clean']['target_probability'];ps=s['corrupted']['target_probability'];pp=patched['target_probability']
    return {'identity_sha256':expected['identity_sha256'],'manifest_sha256':frozen['manifest_sha256'],'task':frozen['task'],
        'selected_index':r['selected_index'],'source_manifest_index':r['source_manifest_index'],'key':r['key'],
        'record_sha256':storage.object_sha(r),'analysis_group':r['analysis_group'],'layer_1based':layer,'gold_label':r['gold_label'],
        'patched':patched,'Pclean':pc,'Pcorrupted':ps,'Ppatched':pp,'probability_gap':s['probability_gap'],
        'RR':recovery(pp,pc,ps),'small_gap':s['probability_gap']<1e-6,'executed_layers_1based':list(range(1,31)),
        'patched_text_slice_0based':[127,383]}


def validate(row,r,expected,frozen,layer):
    storage.verify_seal(row);scoring.validate(r,row['patched'])
    if scoring.reason(r,r['screening']['clean'],r['screening']['corrupted'])!='accepted':raise ValueError('nonqualifying patched record')
    if not 1<=layer<=30 or row!=storage.seal(make_row(r,expected,frozen,layer,row['patched'])):raise ValueError('patch result changed')
    if layer==30:scoring.same(r,row['patched'],r['screening']['clean'])
    return row


def rank_path(directory,rank,layer):return directory/'ranks'/f'rank{rank:02d}'/f'patch_{layer:02d}.jsonl'


def read_rank(directory,rank,frozen,expected,layer):
    rows=storage.recover_jsonl(rank_path(directory,rank,layer));indices=config.indices(rank,frozen['selected'])
    if len(rows)>len(indices):raise ValueError('too many patched rows')
    for row,index in zip(rows,indices):validate(row,frozen['records'][index],expected,frozen,layer)
    return rows


def merge(parts,frozen,expected,layer):
    if len(parts)!=8 or any(len(p)!=len(config.indices(rank,frozen['selected'])) for rank,p in enumerate(parts)):
        raise ValueError('incomplete dynamic eight-rank patch coverage')
    rows=sorted([r for p in parts for r in p],key=lambda r:r['selected_index'])
    if [r['selected_index'] for r in rows]!=list(range(frozen['selected'])):raise ValueError('patch coverage differs from selected manifest')
    for row,r in zip(rows,frozen['records']):validate(row,r,expected,frozen,layer)
    return rows


def save_layer(directory,rows,frozen,expected,layer,elapsed):
    summary=storage.seal({'identity_sha256':expected['identity_sha256'],'manifest_sha256':frozen['manifest_sha256'],
        'task':frozen['task'],'layer_1based':layer,**stats(rows),'groups':{g:stats([r for r in rows if r['analysis_group']==g])
        for g in config.GROUPS[frozen['task']]},'raw_rows_sha256':storage.object_sha(rows),'elapsed_seconds':elapsed})
    folder=directory/'layers'/f'patch_{layer:02d}'
    storage.write_jsonl(folder/'per_question.jsonl',rows);storage.write_json(folder/'summary.json',summary)
    return summary


def tables(directory,summaries):
    storage.write_json(directory/'layer_scores.json',summaries)
    fields=('layer_1based','group','total','mean_RR','median_RR','CI95_low','CI95_high','patched_accuracy',
        'mean_Pclean','mean_Pcorrupted','mean_Ppatched','small_gap_rows','negative_RR','above_one_RR')
    with (directory/'layer_scores.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for s in summaries:
            for g,body in [('overall',s),*s['groups'].items()]:
                ci=body['mean_RR_CI95'] or [None,None]
                writer.writerow({'layer_1based':s['layer_1based'],'group':g,'CI95_low':ci[0],'CI95_high':ci[1],
                    **{k:body[k] for k in fields if k not in ('layer_1based','group','CI95_low','CI95_high')}})
