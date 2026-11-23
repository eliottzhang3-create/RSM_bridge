"""Nineteen conditions; layer numbers exposed to users are one-based."""
from pathlib import Path
from mizar_layer_pair_search import config as reference

ROOT = reference.ROOT
PACKAGE = Path(__file__).resolve().parent
SCHEMA = 'mizar_audio_sep_cutoff03_20_bf16_v1'
CONDITIONS = (None, *range(3, 21))
DATASETS = reference.DATASETS
TOKENS = 127  # 126 mapped audio tokens plus the separator.
PROTOCOL = {
    'schema': SCHEMA, 'conditions': list(CONDITIONS), 'layers': 30,
    'cutoff_semantics': 'before_one_based_layer_L_through_layer30',
    'blocked_keys_0based': [0, 126], 'blocked_queries_0based': [127, 'sequence_end'],
    'causal_attention': True, 'keep_original_positions': True, 'visible_prompt_padding': True,
    'method': 'per_layer_additive_negative_infinity_attention_mask',
    'no_token_deletion': True, 'no_layer_pruning': True, 'use_cache': False,
    'weights': 'strict_final_FP32', 'forward': 'cuda_BF16_autocast',
    'audio_frontend': 'FP32', 'log_softmax_and_sum': 'FP32', 'tf32': False,
    'scoring': reference.PROTOCOL['likelihood'], 'choice_tie': 'first_original_valid_option',
    'generation': False, 'rows': 100, 'rows_per_dataset': 25, 'baseline_first': True,
    'manifest': 'reuse_frozen_layer_pair_search_manifest_no_regeneration',
}


def condition_name(cutoff):
    if cutoff not in CONDITIONS or isinstance(cutoff, bool):
        raise ValueError('expected baseline or cutoff layer 3..20')
    return 'baseline' if cutoff is None else f'cutoff_{cutoff:02d}'


def masked_layers(cutoff):
    condition_name(cutoff)
    return [] if cutoff is None else list(range(cutoff, 31))


def indices(rank, world):
    if type(world) is not int or world not in (1, 8) or type(rank) is not int or not 0 <= rank < world:
        raise ValueError('expected one or eight workers and a valid rank')
    return list(range(rank, 100, world))
