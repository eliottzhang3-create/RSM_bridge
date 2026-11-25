"""Independent 4x250 manifest and 27 full-model conditions; one-based layers."""
from pathlib import Path
from mizar_layer_pair_search import config as reference

ROOT = reference.ROOT
PACKAGE = Path(__file__).resolve().parent
REMOTE, DATA = reference.REMOTE, reference.DATA
MODEL_DIR, CHECKPOINT = reference.MODEL_DIR, reference.CHECKPOINT
SEED, PER_DATASET, ROWS = 1234, 250, 1000
DATASETS = reference.DATASETS
SCHEMA = 'mizar_audio_sep_cutoff03_28_1000_bf16_v1'
MANIFEST_SCHEMA = 'mizar_audio_cutoff_balanced1000_seed1234_v1'
DEFAULT_MANIFEST = DATA / MANIFEST_SCHEMA / 'manifest.json'
CONDITIONS = (None, *range(3, 29))
ISOLATION_CHECKS = (3, 28)
TOKENS = 127
MANIFEST_PROTOCOL = {
    'version': MANIFEST_SCHEMA, 'seed': SEED, 'rows': ROWS, 'rows_per_dataset': PER_DATASET,
    'dataset_order': list(DATASETS), 'sampling': 'independent_seed_per_dataset_without_replacement',
    'eligibility': 'original_pair_search_unique_normalized_options_unique_GT_tokenizable',
    'reasonaqa': reference.PROTOCOL['reasonaqa'],
    **{key: reference.PROTOCOL[key] for key in (
        'audio', 'prompt', 'prompt_tokens', 'prefix_tokens', 'audio_tokens', 'hidden_size',
        'pad_token_id', 'eos_token_id', 'separator_id', 'max_context', 'likelihood')},
    'decoder': 'soundfile_not_torchaudio_load', 'missing_audio': 'fail_without_resampling',
}
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
    'generation': False, 'rows': ROWS, 'rows_per_dataset': PER_DATASET, 'baseline_first': True,
    'manifest_schema': MANIFEST_SCHEMA, 'manifest_protocol': MANIFEST_PROTOCOL,
    'sharding': 'global_index_mod_world_size_no_padding', 'synchronize_after_every_condition': True,
    'isolation_check_layers': list(ISOLATION_CHECKS),
}


def condition_name(cutoff):
    if cutoff is not None and (type(cutoff) is not int or cutoff not in CONDITIONS):
        raise ValueError('expected baseline or cutoff layer 3..28')
    return 'baseline' if cutoff is None else f'cutoff_{cutoff:02d}'


def masked_layers(cutoff):
    condition_name(cutoff)
    return [] if cutoff is None else list(range(cutoff, 31))


def indices(rank, world):
    if type(world) is not int or world not in (1, 8) or type(rank) is not int or not 0 <= rank < world:
        raise ValueError('expected one or eight workers and a valid rank')
    return list(range(rank, ROWS, world))
