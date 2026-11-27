"""Reproduce every letter decision and count-weighted group baseline change."""
import csv
import os
from collections import Counter
from mizar_layer_pair_search import storage
from . import config, runtime


def validate(row, record, task, layer, expected):
    storage.verify_seal(row)
    if (row.get('identity_sha256')!=expected['identity_sha256'] or row.get('task')!=task
            or row.get('condition')!=config.condition_name(task,layer) or row.get('condition_layer_1based')!=layer
            or row.get('global_index')!=record['global_index'] or row.get('key')!=record['key']
            or row.get('record_sha256')!=storage.object_sha(record) or row.get('generation') is not None):
        raise ValueError('letter row has wrong experiment/input/condition')
    for key,value in runtime.decision(record,row.get('option_loglikelihood',[])).items():
        if row.get(key)!=value or isinstance(value,bool) and type(row.get(key)) is not bool:
            raise ValueError(f'letter decision inconsistent with raw scores: {key}')
    trace=row.get('execution',{})
    dropped=config.drop_layers(layer)
    valid=(trace.get('executed_original_layers_1based')==[i for i in range(1,31) if i not in dropped]
        and trace.get('window_start_1based')==layer and trace.get('window_end_1based')==(None if layer is None else layer+2)
        and trace.get('drop_layers_1based')==dropped and trace.get('effective_layers')==30-len(dropped))
    if not valid or trace.get('checked_forwards',0)<1:
        raise ValueError('actual layer/mask route differs from the requested condition')


def read_rank(directory, rank, records, task, layer, expected):
    rows=storage.recover_jsonl(directory/'ranks'/f'rank{rank:02d}'/(config.condition_name(task,layer)+'.jsonl'))
    assigned=config.indices(rank)
    if len(rows)>len(assigned):raise ValueError('extra rank journal rows')
    for row,index in zip(rows,assigned):validate(row,records[index],task,layer,expected)
    return rows


def merge(parts, records, task, layer, expected):
    if len(parts)!=config.WORLD:raise ValueError('need exactly eight rank results')
    combined=[]
    for rank,part in enumerate(parts):
        if [r['global_index'] for r in part]!=config.indices(rank):raise ValueError('rank must cover its 125 exact indices')
        for row in part:validate(row,records[row['global_index']],task,layer,expected)
        combined.extend(part)
    combined.sort(key=lambda r:r['global_index'])
    if [r['global_index'] for r in combined]!=list(range(config.ROWS)):raise ValueError('not exact 1000-question coverage')
    return combined


def group_score(indices, rows, baseline, key='correct'):
    n=len(indices);correct=sum(rows[i][key] for i in indices);old=sum(baseline[i][key] for i in indices)
    return {'correct':correct,'total':n,'accuracy':correct/n,'baseline_correct':old,'baseline_accuracy':old/n,
            'delta_vs_baseline':(correct-old)/n,'delta_percentage_points':100*(correct-old)/n,
            'newly_correct':sum(rows[i][key] and not baseline[i][key] for i in indices),
            'newly_wrong':sum(baseline[i][key] and not rows[i][key] for i in indices),
            'prediction_changed':sum(rows[i]['prediction_index']!=baseline[i]['prediction_index'] for i in indices)}


def summarize(rows, baseline, records, task, layer):
    if (len(rows)!=config.ROWS or len(baseline)!=config.ROWS
            or [r['global_index'] for r in rows]!=list(range(config.ROWS))
            or [r['global_index'] for r in baseline]!=list(range(config.ROWS))
            or Counter(r['analysis_group'] for r in records)!=Counter(config.GROUP_COUNTS[task])):
        raise ValueError('wrong full denominator or analysis group counts')
    grouped={g:group_score([i for i,r in enumerate(records) if r['analysis_group']==g],rows,baseline)
             for g in config.GROUP_COUNTS[task]}
    subgroups={}
    for field in ('dataset','task','category','sub_category','modality','difficulty'):
        values=sorted({str(r[field]) for r in records if field in r})
        if values:
            subgroups[field]={v:group_score([i for i,r in enumerate(records) if str(r.get(field))==v],rows,baseline) for v in values}
    improved=[i for i in range(config.ROWS) if rows[i]['correct'] and not baseline[i]['correct']]
    harmed=[i for i in range(config.ROWS) if baseline[i]['correct'] and not rows[i]['correct']]
    value={'task':task,'condition':config.condition_name(task,layer),'condition_layer_1based':layer,
           'window_start_1based':layer,'window_end_1based':None if layer is None else layer+2,
           'drop_layers_1based':config.drop_layers(layer),'effective_layers':30-len(config.drop_layers(layer)),
           **group_score(list(range(config.ROWS)),rows,baseline),'groups':grouped,'subgroups':subgroups,
           'newly_correct_indices':improved,'newly_wrong_indices':harmed,
           'unmapped_gold_rows':sum(r['gold_index'] is None for r in records),
           'option_score_ties':sum(r['option_score_tie'] for r in rows),
           'raw_rows_sha256':storage.object_sha(rows),'baseline_rows_sha256':storage.object_sha(baseline)}
    if task=='mmar':
        value['official_diagnostic']={**group_score(list(range(config.ROWS)),rows,baseline,'official_correct'),
            'groups':{g:group_score([i for i,r in enumerate(records) if r['analysis_group']==g],rows,baseline,'official_correct')
                      for g in config.GROUP_COUNTS[task]},'primary_metric':False}
    return value


def save_condition(directory, rows, baseline, records, task, layer, expected, elapsed):
    summary={**summarize(rows,baseline,records,task,layer),'identity_sha256':expected['identity_sha256'],'elapsed_seconds':elapsed}
    folder=directory/'conditions'/config.condition_name(task,layer)
    storage.write_jsonl(folder/'likelihood.jsonl',rows)
    storage.write_json(folder/'summary.json',storage.seal(summary))
    changes=[]
    for i in summary['newly_correct_indices']+summary['newly_wrong_indices']:
        r=records[i]
        changes.append({'global_index':i,'key':r['key'],'analysis_group':r['analysis_group'],'prompt':r['prompt'],
            'choices':r['choice_bodies'],'letters':r['option_letters'],'gold_index':r['gold_index'],
            'gold_original':r['gold_original'],'baseline_prediction':baseline[i]['prediction_letter'],
            'prediction':rows[i]['prediction_letter'],'baseline_letter_loglikelihood':baseline[i]['option_loglikelihood'],
            'letter_loglikelihood':rows[i]['option_loglikelihood'],
            'change':'newly_correct' if rows[i]['correct'] else 'newly_wrong'})
    storage.write_jsonl(folder/'paired_changes.jsonl',changes)
    return summary


def write_csv(path, fields, rows):
    temporary=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with temporary.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def ranking(summaries, group=None):
    key=lambda row:row['correct'] if group is None else row['groups'][group]['correct']
    ordered=sorted(summaries[1:],key=lambda row:(-key(row),row['condition_layer_1based']))
    ranked=[];last=None;rank=None
    for position,row in enumerate(ordered,1):
        if key(row)!=last:rank,last=position,key(row)
        ranked.append({'rank':rank,**row})
    return ranked


def save_analysis(directory, summaries, frozen):
    task=frozen['task']
    if [r['condition_layer_1based'] for r in summaries]!=list(config.conditions(task)):
        raise ValueError('need all conditions baseline-first')
    storage.write_json(directory/'condition_scores.json',summaries)
    storage.write_json(directory/'baseline.json',summaries[0])
    storage.write_json(directory/'group_baseline.json',summaries[0]['groups'])
    storage.write_json(directory/'input_audit.json',frozen['input_audit'])
    ranked=ranking(summaries);storage.write_json(directory/'ranking.json',ranked)
    storage.write_json(directory/'group_ranking.json',{g:ranking(summaries,g) for g in config.GROUP_COUNTS[task]})
    fields=['condition','window_start_1based','window_end_1based','effective_layers','correct','total','accuracy','baseline_accuracy','delta_vs_baseline',
            'delta_percentage_points','newly_correct','newly_wrong','prediction_changed']
    table=[{k:r[k] for k in fields} for r in summaries]
    write_csv(directory/'accuracy_by_window.csv',fields,table)
    group_fields=['condition','window_start_1based','window_end_1based','group',*fields[4:]]
    grouped=[]
    for row in summaries:
        for group,value in row['groups'].items():
            grouped.append({'condition':row['condition'],'window_start_1based':row['window_start_1based'],
                'window_end_1based':row['window_end_1based'],'group':group,**{k:value[k] for k in fields[4:]}})
    write_csv(directory/'accuracy_by_group.csv',group_fields,grouped)
    if task=='mmau':
        storage.write_json(directory/'group_mapping.json',storage.read_json(config.PACKAGE/'mmau_mapping.json'))
        write_csv(directory/'psr_scores.csv',group_fields,grouped)
    if task=='mmar':
        diagnostic=[]
        for row in summaries:
            for group in ('overall',*config.GROUP_COUNTS[task]):
                value=row['official_diagnostic'] if group=='overall' else row['official_diagnostic']['groups'][group]
                diagnostic.append({'condition':row['condition'],'window_start_1based':row['window_start_1based'],
                    'window_end_1based':row['window_end_1based'],'group':group,**{k:value[k] for k in fields[4:]}})
        write_csv(directory/'official_diagnostic.csv',group_fields,diagnostic)
    return ranked
