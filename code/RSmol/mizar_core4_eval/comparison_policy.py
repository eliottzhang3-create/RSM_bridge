"""Validate common protocol and native execution before comparing complete results."""
import copy
from .registry import ROUTES


def known_variants():
    return {*ROUTES, 'official_mizar'}


def common_protocol(identity, benchmark):
    from . import runtime, mmsu_runtime
    base = mmsu_runtime.PROTOCOL if benchmark == 'mmsu' else runtime.PROTOCOL
    variant = identity['eval_variant']
    if variant not in known_variants():
        raise ValueError('unregistered model')
    if variant == 'official_mizar':
        from mizar_core3_eval import runtime as official_runtime
        expected = official_runtime.protocol_for(variant, base)
    else:
        expected = runtime.protocol_for(variant, base)
    actual = identity['generation' if benchmark in ('mmau', 'mmar') else 'protocol']
    if actual != expected:
        raise ValueError('benchmark/native execution policy differs')
    return copy.deepcopy(base)


def shared_evaluation_code(identity, benchmark):
    if benchmark == 'mmsu':
        names = ['mizar_s1_ced_mmsu_eval/'+n+'.py' for n in ('dataset', 'parsing', 'scoring', 'storage')]
    else:
        names = ['scripts/'+n for n in ('evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.py',
                                        'evaluate_mmar_5_10x2_5_mesh_mellow.py')]
    return {name:identity['evaluation_code'][name] for name in names}


def execution_columns(identity):
    protocol = identity.get('protocol', identity.get('generation'))
    execution = protocol['model_execution']
    return dict(attention_policy='causal_no_padding_key_mask', prompt_padding=protocol['prompt_padding'],
                physical_layers=30, block_executions=len(execution['schedule']))
