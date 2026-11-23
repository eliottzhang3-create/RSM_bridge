"""Exact 1000-row sum-likelihood scores and paired baseline changes."""
from mizar_layer_pair_search import storage
from mizar_layer_pair_search.results import choose
from mizar_layer_analysis_common import config

CONDITIONS=(None,*range(2,30))


def name(layer):
    if layer is not None and (type(layer) is not int or not 2<=layer<=29): raise ValueError('invalid ablation layer')
    return 'baseline' if layer is None else f'ablate_{layer:02d}'


def validate(row,record,layer,expected):
    storage.verify_seal(row)
    scores=row.get('option_loglikelihood',[]); counts=[len(ids) for ids in record['choice_token_ids']]
    if (row.get('identity_sha256')!=expected['identity_sha256'] or row.get('condition')!=name(layer)
            or row.get('global_index')!=record['global_index'] or row.get('key')!=record['key']
            or row.get('record_sha256')!=storage.object_sha(record) or len(scores)!=len(counts) or row.get('generation') is not None):
        raise ValueError('wrong ablation/input/identity')
    pred=choose(scores)
    if (row.get('prediction_index')!=pred or row.get('correct') is not (pred==record['gold_index'])
            or row.get('option_token_counts')!=counts or row.get('option_mean_loglikelihood')!=[s/n for s,n in zip(scores,counts)]
            or row.get('option_score_tie') is not (scores.count(scores[pred])>1)):
        raise ValueError('decision does not follow raw sum scores')
    trace=row.get('execution',{})
    if (trace.get('drop_layer_1based')!=layer or trace.get('executed_original_layers_1based')!=[i for i in range(1,31) if i!=layer]
            or trace.get('effective_layers')!=(30 if layer is None else 29) or trace.get('checked_forwards',0)<1):
        raise ValueError('wrong actual original-layer route')


def read_rank(directory,rank,records,layer,expected):
    rows=storage.recover_jsonl(directory/'ranks'/f'rank{rank:02d}'/(name(layer)+'.jsonl'))
    indices=config.indices(rank,1000)
    if len(rows)>125: raise ValueError('extra ablation rows')
    for row,index in zip(rows,indices): validate(row,records[index],layer,expected)
    return rows


def merge(parts,records,layer,expected):
    if len(parts)!=8: raise ValueError('need all eight ranks')
    rows=[]
    for rank,part in enumerate(parts):
        if [r['global_index'] for r in part]!=config.indices(rank,1000): raise ValueError('not 125 distinct assigned rows per GPU')
        for r in part: validate(r,records[r['global_index']],layer,expected)
        rows.extend(part)
    rows.sort(key=lambda r:r['global_index'])
    if [r['global_index'] for r in rows]!=list(range(1000)): raise ValueError('not exact 1000-row denominator')
    return rows


def summarize(rows,base,records,layer):
    if len(rows)!=1000 or len(base)!=1000: raise ValueError('incomplete condition or baseline')
    correct=sum(r['correct'] for r in rows); old=sum(r['correct'] for r in base)
    groups={}
    for field in ('task','difficulty','category','sub_category'):
        field_groups={}
        for value in sorted({r[field] for r in records}):
            indices=[i for i,r in enumerate(records) if r[field]==value]
            n=sum(rows[i]['correct'] for i in indices); b=sum(base[i]['correct'] for i in indices)
            field_groups[value]={'correct':n,'total':len(indices),'accuracy':n/len(indices),'delta_vs_baseline':(n-b)/len(indices)}
        groups[field]=field_groups
    better=[i for i in range(1000) if rows[i]['correct'] and not base[i]['correct']]
    worse=[i for i in range(1000) if base[i]['correct'] and not rows[i]['correct']]
    return {'condition':name(layer),'drop_layer_1based':layer,'drop_layer_0based':None if layer is None else layer-1,
        'correct':correct,'total':1000,'accuracy':correct/1000,'baseline_accuracy':old/1000,'delta_vs_baseline':(correct-old)/1000,
        'newly_correct_indices':better,'newly_wrong_indices':worse,'newly_correct':len(better),'newly_wrong':len(worse),
        'option_score_ties':sum(r['option_score_tie'] for r in rows),'groups':groups,
        'raw_rows_sha256':storage.object_sha(rows),'baseline_rows_sha256':storage.object_sha(base)}


def save_condition(directory,rows,base,records,layer,expected,elapsed):
    summary={**summarize(rows,base,records,layer),'identity_sha256':expected['identity_sha256'],'elapsed_seconds':elapsed}
    folder=directory/'conditions'/name(layer)
    storage.write_jsonl(folder/'likelihood.jsonl',rows); storage.write_json(folder/'summary.json',storage.seal(summary))
    changes=[]
    for index in summary['newly_correct_indices']+summary['newly_wrong_indices']:
        r=records[index]
        changes.append({'index':index,'key':r['key'],'prompt':r['prompt'],'choice_bodies':r['choice_bodies'],'gold_index':r['gold_index'],
            'baseline_prediction':base[index]['prediction_index'],'prediction':rows[index]['prediction_index'],
            'baseline_scores':base[index]['option_loglikelihood'],'scores':rows[index]['option_loglikelihood'],
            'change':'newly_correct' if rows[index]['correct'] else 'newly_wrong'})
    storage.write_jsonl(folder/'paired_changes.jsonl',changes)
    return summary
