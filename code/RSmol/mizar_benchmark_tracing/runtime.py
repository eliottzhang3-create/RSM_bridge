"""Original all30 native forward/capture/patch; scoring supports blank slots."""
from mizar_reasonaqa_tracing import runtime as original
from . import scoring,config

forward,prefix,silent_audio,corrupt_prefix = original.forward,original.prefix,original.silent_audio,original.corrupt_prefix
from_logits = scoring.from_logits


def load(a,bundle,binding):
    loaded=original.load(a,bundle,binding)
    loaded['audit'].update(experiment=config.SCHEMA,task=a.task,all_30_blocks_executed=True)
    return loaded


def controls(loaded,cp,sp,states,record,clean_saved,corrupt_saved):
    import torch
    clean,captured=forward(loaded,cp,capture=True);native,_=forward(loaded,cp)
    corrupt,_=forward(loaded,sp)
    zero,_=forward(loaded,sp,patch_input=cp[:,loaded.get('text_start',127):])
    last,_=forward(loaded,sp,layer=30,clean_states=states)
    for actual,expected in ((clean,native),(zero,corrupt),(last,clean)):
        if not torch.equal(actual,expected):raise ValueError('capture/layer0/layer30 full vocabulary logits differ')
    if not torch.equal(states,captured):raise ValueError('cache differs from repeated raw clean states')
    clean_score=from_logits(record,clean);corrupt_score=from_logits(record,corrupt)
    errors=[scoring.same(record,clean_score,clean_saved),scoring.same(record,corrupt_score,corrupt_saved)]
    gap=clean_saved['target_probability']-corrupt_saved['target_probability']
    return {'status':'PASS','capture_native_bitwise':True,'layer0_corrupted_bitwise':True,'layer30_clean_bitwise':True,
        'cache_recapture_bitwise':True,'state_shape':list(states.shape),'state_dtype':str(states.dtype),
        'saved_logprobs_exact':True,'exp_max_ULPs':8,'saved_probability_max_abs_error':max(errors),
        'layer30_RR':(clean_score['target_probability']-corrupt_saved['target_probability'])/gap}
