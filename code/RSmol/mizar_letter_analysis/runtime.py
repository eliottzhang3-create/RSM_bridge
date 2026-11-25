"""Same original Mizar routes; gather bare lowercase letter likelihoods once."""
import math
from mizar_layer_pair_search import runtime as reference
from mizar_audio_cutoff1000 import runtime as cutoff_reference
from mizar_layer_analysis_common import runtime as ablation_reference
from mizar_layer_ablation_mmar.runtime import check_skip_equivalence
from . import config

autocast = reference.autocast
prefix = reference.prefix


def decision(record, scores):
    valid = record['choice_valid_mask']
    if len(scores)!=len(valid):
        raise ValueError('wrong letter score count')
    candidates=[]
    for i,(enabled,score) in enumerate(zip(valid,scores)):
        if enabled:
            if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score):
                raise ValueError('expected finite letter loglikelihood')
            candidates.append(i)
        elif score is not None:
            raise ValueError('invalid original option must have null letter score')
    if len(candidates)<2:
        raise ValueError('need at least two valid original choices')
    predicted=max(candidates,key=lambda i:scores[i])
    value={'option_loglikelihood':scores,'option_letters':record['option_letters'],
        'letter_token_ids':record['letter_token_ids'],'choice_valid_mask':valid,
        'prediction_index':predicted,'prediction_letter':record['option_letters'][predicted],
        'prediction_text':record['choice_bodies'][predicted], 'gold_index':record['gold_index'],
        'gold_letter':None if record['gold_index'] is None else record['option_letters'][record['gold_index']],
        'correct':record['gold_index'] is not None and predicted==record['gold_index'],
        'option_score_tie':sum(scores[i]==scores[predicted] for i in candidates)>1}
    if 'official_correct_by_slot' in record:
        value['official_correct']=record['official_correct_by_slot'][predicted]
    return value


def load(args, bundle, binding):
    loaded=(cutoff_reference if args.task=='cutoff' else ablation_reference).load(args,bundle,binding)
    loaded['task']=args.task
    loaded['audit'].update(experiment=config.RUN_NAMES[args.task],letter_scoring=True,prompt_eot=True)
    return loaded


def set_condition(loaded, layer):
    config.condition_name(loaded['task'],layer)
    if loaded['task']=='cutoff':loaded['cutoff_controller'].set_cutoff(layer)
    else:loaded['controller'].set_drop(layer)


def evidence(loaded):
    return loaded['cutoff_controller' if loaded['task']=='cutoff' else 'controller'].evidence()


def score(loaded, cached_prefix, record):
    import torch
    lm=loaded['lm']
    with torch.inference_mode(),autocast(loaded['device']):
        # No candidate tokens are appended: first answer logits at position 382.
        output=lm(inputs_embeds=cached_prefix,attention_mask=None,labels=None,use_cache=False)
        logits=output.logits[0,-1].float()
    if not bool(torch.isfinite(logits).all()):raise ValueError('nonfinite first-answer logits')
    logprobs=logits.log_softmax(-1)
    ids=torch.tensor(record['letter_token_ids'],device=logprobs.device,dtype=torch.long)
    selected=logprobs.index_select(0,ids).tolist()
    scores=[float(s) if valid else None for s,valid in zip(selected,record['choice_valid_mask'])]
    return {**decision(record,scores),'execution':evidence(loaded)}


def checks(loaded, cached_prefix, record):
    # Both reference checks consume one letter token, hence the same prefix-only
    # input as the actual scoring forward (no option-body teacher forcing).
    if loaded['task']=='cutoff':return cutoff_reference.runtime_checks(loaded,cached_prefix,record)
    return check_skip_equivalence(loaded,cached_prefix,record)
