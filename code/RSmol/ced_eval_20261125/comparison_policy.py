"""Validate native execution separately from the common benchmark protocol."""
import copy
from .registry import ROUTES

def known_variants():
    from ced_eval_20261123.registry import ROUTES as previous
    from ced_eval_20261122.registry import ROUTES as earlier
    from text71427_ced_eval.registry import ROUTES as original
    from mizar_s1_ced_reasonaqa5000_eval.config import VARIANT_TARGETS
    return {*ROUTES,*previous,*earlier,*original,*VARIANT_TARGETS}

def common_protocol(identity,benchmark):
    from . import runtime,reasonaqa_runtime,mmsu_runtime
    base=(reasonaqa_runtime.PROTOCOL if benchmark=='reasonaqa5000' else
          mmsu_runtime.PROTOCOL if benchmark=='mmsu' else runtime.PROTOCOL)
    variant=identity['eval_variant']
    if variant not in known_variants():raise ValueError('unregistered model execution policy')
    actual=identity['generation' if benchmark in ('mmau','mmar') else 'protocol']
    expected=runtime.protocol_for(variant,base) if variant in ROUTES else base
    if actual!=expected:raise ValueError('benchmark/native execution protocol differs from registered policy')
    result=copy.deepcopy(base)
    for field in ('attention_mask','prompt_padding'):result.pop(field,None)
    return result

def shared_evaluation_code(identity,benchmark):
    if benchmark=='mmsu':
        names=['mizar_s1_ced_mmsu_eval/'+n+'.py' for n in ('dataset','parsing','scoring','storage')]
    elif benchmark=='reasonaqa5000':
        names=['mizar_s1_ced_reasonaqa5000_eval/'+n+'.py' for n in ('config','dataset','sample','scoring','storage')]
    else:names=['scripts/'+n for n in ('evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.py','evaluate_mmar_5_10x2_5_mesh_mellow.py')]
    return {name:identity['evaluation_code'][name] for name in names}

def execution_columns(identity):
    protocol=identity.get('protocol',identity.get('generation'))
    return {'attention_policy':protocol['attention_mask'] or 'causal_no_padding_key_mask',
            'prompt_padding':protocol['prompt_padding']}
