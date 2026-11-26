"""Original slots, nullable gold, full-vocabulary probability and portable checks."""
import math
from mizar_reasonaqa_tracing500.validation import probability_error


def decision(record,logprobs):
    valid=record['choice_valid_mask'];labels=record['candidate_labels'];gold=record['gold_index']
    if len(logprobs)!=len(valid) or len(labels)!=len(valid) or sum(valid)<2:
        raise ValueError('candidate slot layout differs')
    for enabled,p in zip(valid,logprobs):
        if enabled:
            if type(p) not in (float,int) or not math.isfinite(p) or p>0:raise ValueError('invalid candidate logprob')
        elif p is not None:raise ValueError('blank option must have null logprob')
    if gold is not None and (type(gold) is not int or not 0<=gold<len(valid) or not valid[gold]):
        raise ValueError('invalid gold slot')
    best=max((i for i,v in enumerate(valid) if v),key=lambda i:logprobs[i])
    probs=[math.exp(p) if enabled else None for enabled,p in zip(valid,logprobs)]
    return {'candidate_logprobs':[float(p) if p is not None else None for p in logprobs],
        'candidate_probabilities':probs,'prediction_index':best,'prediction_label':labels[best],
        'correct':gold is not None and best==gold,'target_logprob':None if gold is None else float(logprobs[gold]),
        'target_probability':None if gold is None else probs[gold],
        'tie':sum(logprobs[i]==logprobs[best] for i,v in enumerate(valid) if v)>1}


def from_logits(record,logits):
    import torch
    if logits.ndim!=1 or not bool(torch.isfinite(logits).all()):raise ValueError('nonfinite readout logits')
    lp=logits.float().log_softmax(-1)
    ids=torch.tensor(record['candidate_token_ids'],dtype=torch.long,device=lp.device)
    values=lp.index_select(0,ids).tolist()
    return decision(record,[float(p) if v else None for p,v in zip(values,record['choice_valid_mask'])])


def compare(record,actual,expected):
    if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('decision fields differ')
    for key in expected.keys()-{'candidate_probabilities','target_probability'}:
        if actual[key]!=expected[key]:raise ValueError(f'decision {key} differs')
    if (type(actual['prediction_index']) is not int or type(actual['correct']) is not bool or type(actual['tie']) is not bool
            or not isinstance(actual['candidate_logprobs'],list)
            or record['gold_index'] is not None and type(actual['target_logprob']) is not float
            or any(p is not None and type(p) is not float for p in actual['candidate_logprobs'])):
        raise ValueError('invalid decision scalar types')
    ps=actual['candidate_probabilities'];qs=expected['candidate_probabilities']
    if not isinstance(ps,list) or len(ps)!=len(qs):raise ValueError('wrong probability count')
    errors=[]
    for i,(p,q) in enumerate(zip(ps,qs)):
        if q is None:
            if p is not None:raise ValueError('blank slot probability must be null')
        else:errors.append(probability_error(p,q,f'candidate_probability[{i}]'))
    if record['gold_index'] is None:
        if actual['target_probability'] is not None:raise ValueError('unmapped gold needs null target probability')
    else:
        errors.append(probability_error(actual['target_probability'],expected['target_probability'],'target_probability'))
        if actual['target_probability']!=ps[record['gold_index']]:raise ValueError('gold probability differs from its own slot')
    return max(errors,default=0.0)


def validate(record,value):
    try:return compare(record,value,decision(record,value['candidate_logprobs']))
    except (KeyError,TypeError,ValueError) as exc:raise ValueError(f'invalid decision for {record["key"]}: {exc}') from exc


def same(record,actual,saved):
    validate(record,actual);validate(record,saved)
    return compare(record,actual,saved)


def reason(record,clean,corrupted):
    if record['gold_index'] is None:return 'unmapped_gold'
    if not clean['correct']:return 'clean_wrong'
    if corrupted['correct']:return 'silence_correct'
    if clean['target_probability']<=corrupted['target_probability']:return 'nonpositive_probability_gap'
    return 'accepted'
