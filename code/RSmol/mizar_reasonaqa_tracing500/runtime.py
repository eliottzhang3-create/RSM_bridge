"""Original inference; portable validation of already saved FP64 exp values."""
from mizar_reasonaqa_tracing.runtime import (load, forward, from_logits,
    decision, silent_audio, corrupt_prefix, autocast, prefix)
from .validation import validate_decision, assert_same_decision, EXP_MAX_ULPS


def controls(loaded,clean_prefix,corrupted_prefix,clean_states,record,clean_result,corrupt_result):
    import torch
    clean_logits,captured=forward(loaded,clean_prefix,capture=True)
    native,_=forward(loaded,clean_prefix)
    corrupt_logits,_=forward(loaded,corrupted_prefix)
    layer0,_=forward(loaded,corrupted_prefix,patch_input=clean_prefix[:,loaded.get('text_start',127):])
    last,_=forward(loaded,corrupted_prefix,layer=30,clean_states=clean_states)
    for actual,expected in ((native,clean_logits),(layer0,corrupt_logits),(last,clean_logits)):
        if not torch.equal(actual,expected):
            raise ValueError('no-patch/layer0/layer30 control failed (full vocabulary logits)')
    if not torch.equal(captured,clean_states):
        raise ValueError('cached clean states differ from repeated clean capture')
    clean=from_logits(record,clean_logits);corrupt=from_logits(record,corrupt_logits)
    errors=[assert_same_decision(record,clean,clean_result),assert_same_decision(record,corrupt,corrupt_result)]
    gap=clean_result['target_probability']-corrupt_result['target_probability']
    last_probability=from_logits(record,last)['target_probability']
    return {'status':'PASS','capture_vs_native_logits_bitwise_equal':True,
        'layer0_equals_corrupted_logits_bitwise':True,'layer30_equals_clean_logits_bitwise':True,
        'cache_equals_repeated_clean_states_bitwise':True,'state_shape':list(clean_states.shape),
        'state_dtype':str(clean_states.dtype),'text_start':loaded.get('text_start',127),
        'saved_logprobs_exact':True,'saved_exp_max_ulps':EXP_MAX_ULPS,
        'saved_probability_max_abs_error':max(errors),
        'layer30_RR':None if gap<=0 else (last_probability-corrupt_result['target_probability'])/gap}
