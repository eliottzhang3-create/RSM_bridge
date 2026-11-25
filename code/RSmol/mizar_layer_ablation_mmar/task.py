import csv
import json
import time
from mizar_layer_pair_search import storage
from . import audit,config,runtime
from . import dataset,results


def protocol(args):
    return {**config.COMMON_PROTOCOL,'task':config.SCHEMA,'rows':1000,'conditions':list(results.CONDITIONS),
        'intervention':'h_out_equals_h_in_bypass_single_original_block','protected_layers_1based':[1,30],
        'scoring':'FP32_log_softmax_sum_full_body_no_BOS_EOS_label_or_added_space','tie':'first_original_option',
        'gold_slot_policy':config.INPUT_PROTOCOL['gold'],'original_duplicate_slots_retained':True,
        'empty_option_policy':config.INPUT_PROTOCOL['empty_choices'], 'categories':config.CATEGORY_COUNTS,
        'official_matcher_diagnostic_only':True, 'input_protocol':config.INPUT_PROTOCOL,
        'source_arguments':dataset.source_args(args)}


prepare=dataset.ensure
runtime_check=runtime.check_skip_equivalence


def evaluate_condition(args,loaded,prefixes,records,expected,layer,rank,dist):
    started=time.monotonic(); control=[(layer,expected['identity_sha256']) if rank==0 else None]
    dist.broadcast_object_list(control,src=0)
    if tuple(control[0])!=(layer,expected['identity_sha256']): raise ValueError('ranks disagree on active condition')
    loaded['controller'].set_drop(layer)
    rows=results.read_rank(args.output_dir,rank,records,layer,expected)
    path=args.output_dir/'ranks'/f'rank{rank:02d}'/(results.name(layer)+'.jsonl')
    for index in config.indices(rank,1000)[len(rows):]:
        record=records[index]
        scored=runtime.score(loaded,prefixes[index],record)
        value={'identity_sha256':expected['identity_sha256'],'condition':results.name(layer),'global_index':index,
               'key':record['key'],'record_sha256':storage.object_sha(record),'execution':loaded['controller'].evidence(),**scored}
        sealed=storage.seal(value); results.validate(sealed,record,layer,expected)
        storage.append_jsonl(path,value); rows.append(sealed)
    parts=[None]*8 if rank==0 else None; dist.gather_object(rows,parts,dst=0)
    if rank==0:
        combined=results.merge(parts,records,layer,expected)
        base=combined if layer is None else results.merge([results.read_rank(args.output_dir,r,records,None,expected) for r in range(8)],records,None,expected)
        summary=results.save_condition(args.output_dir,combined,base,records,layer,expected,time.monotonic()-started)
        print(json.dumps({'event':'BASELINE_DONE' if layer is None else 'ABLATION_DONE',
            'completed':results.CONDITIONS.index(layer)+1,'total_conditions':29,
            **{k:summary[k] for k in ('condition','drop_layer_1based','correct','total','accuracy',
                                      'delta_vs_baseline','newly_correct','newly_wrong','elapsed_seconds')},
            'categories':summary['groups']['category'],
            'official_diagnostic_accuracy':summary['official_diagnostic']['accuracy']},ensure_ascii=False),flush=True)
    dist.barrier()


def work(args,loaded,prefixes,records,expected,rank,dist):
    for layer in results.CONDITIONS: evaluate_condition(args,loaded,prefixes,records,expected,layer,rank,dist)


def finalize(args,frozen,expected):
    directory=args.output_dir; records=frozen['records']; base=None; summaries=[]
    for layer in results.CONDITIONS:
        parts=[results.read_rank(directory,r,records,layer,expected) for r in range(8)]
        if any(len(rows)!=125 for rows in parts): return None
        rows=results.merge(parts,records,layer,expected)
        if layer is None: base=rows
        elapsed=0; path=directory/'conditions'/results.name(layer)/'summary.json'
        if path.exists():
            previous=storage.verify_seal(storage.read_json(path))
            if previous['identity_sha256']!=expected['identity_sha256']: raise ValueError('summary from different run')
            elapsed=previous['elapsed_seconds']
        summaries.append(results.save_condition(directory,rows,base,records,layer,expected,elapsed))
    if not audit.ranks_complete(directory): return None
    checks=audit.verify(directory,expected,records)
    for rank_check in checks['rank_checks']:
        if ([c['layer_1based'] for c in rank_check['bypass_vs_identity_output']]!=[2,29]
                or any(c['status']!='PASS' or c['max_abs_error']!=0 for c in rank_check['bypass_vs_identity_output'])):
            raise ValueError('missing actual bypass/identity equivalence check')
    storage.write_json(directory/'condition_scores.json',summaries); storage.write_json(directory/'baseline.json',summaries[0])
    storage.write_json(directory/'dataset_audit.json',frozen['dataset_audit'])
    storage.write_json(directory/'category_baseline.json',summaries[0]['groups']['category'])
    last,rank=None,None; ranking=[]
    for position,row in enumerate(sorted(summaries[1:],key=lambda r:(-r['correct'],r['drop_layer_1based'])),1):
        if row['correct']!=last: rank,last=position,row['correct']
        ranking.append({**row,'rank':rank})
    storage.write_json(directory/'ranking.json',ranking)
    fields=['condition','drop_layer_1based','correct','total','accuracy','delta_vs_baseline','newly_correct','newly_wrong','option_score_ties']
    write_csv(directory/'accuracy_by_layer.csv',fields,[{k:row[k] for k in fields} for row in summaries])
    category_fields=['condition','drop_layer_1based','category','correct','total','accuracy','baseline_accuracy',
                     'delta_vs_baseline','delta_percentage_points','newly_correct','newly_wrong','prediction_changed']
    category_rows=[]; official_rows=[]
    for row in summaries:
        for category in config.CATEGORIES:
            value=row['groups']['category'][category]
            category_rows.append({'condition':row['condition'],'drop_layer_1based':row['drop_layer_1based'],
                'category':category,**{k:value[k] for k in category_fields if k in value},
                'delta_percentage_points':100*value['delta_vs_baseline']})
        for category in ('overall',*config.CATEGORIES):
            value=row['official_diagnostic'] if category=='overall' else row['official_diagnostic']['groups']['category'][category]
            official_rows.append({'condition':row['condition'],'drop_layer_1based':row['drop_layer_1based'],
                'category':category,**{k:value[k] for k in category_fields if k in value},
                'delta_percentage_points':100*value['delta_vs_baseline']})
    write_csv(directory/'accuracy_by_category.csv',category_fields,category_rows)
    write_csv(directory/'official_diagnostic.csv',category_fields,official_rows)
    category_ranking={}
    for category in config.CATEGORIES:
        ordered=sorted(summaries[1:],key=lambda row:(-row['groups']['category'][category]['correct'],row['drop_layer_1based']))
        values=[]; last=None; rank=None
        for position,row in enumerate(ordered,1):
            value=row['groups']['category'][category]
            if value['correct']!=last: rank,last=position,value['correct']
            values.append({'rank':rank,'drop_layer_1based':row['drop_layer_1based'],**value})
        category_ranking[category]=values
    storage.write_json(directory/'category_ranking.json',category_ranking)
    from .plot import make_plot
    plots=make_plot(directory,summaries)
    return {'status':'PASS','conditions_completed':29,'decisions':29000,'rows_per_condition':1000,'generation_count':0,
        'baseline_accuracy':summaries[0]['accuracy'],'top_accuracy':ranking[0]['accuracy'],
        'category_counts':config.CATEGORY_COUNTS, 'primary_metric':'strict_option_match_on_likelihood_argmax',
        'unmapped_gold_rows':summaries[0]['unmapped_gold_rows'], 'empty_choice_rows':summaries[0]['empty_choice_rows'],
        'top_tied_layers_1based':[r['drop_layer_1based'] for r in ranking if r['rank']==1],
        'runtime_checks_sha256':checks['sha256'],'plots':plots}


def write_csv(path, fields, rows):
    import os
    temporary=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with temporary.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)
