"""Compare the four registered native models under the common reference protocol."""
import copy
from .registry import ROUTES


def known_variants():
    return set(ROUTES)


def common_protocol(identity,benchmark):
    from . import runtime,reasonaqa_runtime,mmsu_runtime
    base = (reasonaqa_runtime.PROTOCOL if benchmark == 'reasonaqa5000' else
            mmsu_runtime.PROTOCOL if benchmark == 'mmsu' else runtime.PROTOCOL)
    variant = identity['eval_variant']
    if variant not in ROUTES:
        raise ValueError('unregistered model')
    actual = identity['generation' if benchmark in ('mmau','mmar') else 'protocol']
    if actual != runtime.protocol_for(variant,base):
        raise ValueError('benchmark/native execution policy differs')
    return copy.deepcopy(base)


def shared_evaluation_code(identity,benchmark):
    if benchmark == 'mmsu':
        names = ['mizar_s1_ced_mmsu_eval/'+n+'.py' for n in ('dataset','parsing','scoring','storage')]
    elif benchmark == 'reasonaqa5000':
        names = ['mizar_s1_ced_reasonaqa5000_eval/'+n+'.py' for n in ('config','dataset','sample','scoring','storage')]
    else:
        names = ['scripts/'+n for n in ('evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.py','evaluate_mmar_5_10x2_5_mesh_mellow.py')]
    return {name:identity['evaluation_code'][name] for name in names}


def execution_columns(identity):
    protocol = identity.get('protocol',identity.get('generation'))
    execution = protocol['model_execution']
    return dict(attention_policy='causal_no_padding_key_mask',prompt_padding=protocol['prompt_padding'],
        physical_layers=30,block_executions=len(execution['schedule']))
