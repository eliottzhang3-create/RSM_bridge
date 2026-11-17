"""Frozen search and original-Mizar input semantics; no heavy imports."""
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
REMOTE = Path('/hpc_stor03/sjtu_home/jinwei.zhang')
DATA = REMOTE / 'data'
MODEL_DIR = REMOTE / 'models/Mizar159M'
CHECKPOINT = Path('checkpoints/final/stage3_q4_seed20260905.ckpt')
DEFAULT_OUTPUT = REMOTE / 'outputs/RSmol/mizar_30to28_pair_search_bf16_seed1234_v1'
DATASETS = ('mmau', 'mmsu', 'mmar', 'reasonaqa')
SEED, PER_DATASET, ROWS, WORLD = 1234, 25, 100, 8
PAIRS = tuple(combinations(range(1, 29), 2))
SCHEMA = 'mizar_30to28_likelihood_sum_bf16_v1'
ROUTE = ROOT / 'mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_ced_small'
PROTOCOL = {
    'version': SCHEMA, 'seed': SEED, 'protected_layers_0based': [0, 29],
    'mask_semantics': '1_drop_0_execute', 'pair_order': 'lexicographic_original_0based',
    'candidate_masks': len(PAIRS), 'world_size': WORLD,
    'sharding': 'global_index_mod8_no_padding', 'synchronize_after_every_mask': True,
    'audio': 'resample32k_then_mono_first20s_right_zero_pad',
    'prompt': 'question_contiguous_lowercase_letter_options_no_question_EOT',
    'prompt_tokens': 256, 'prefix_tokens': 383, 'audio_tokens': 126, 'hidden_size': 576,
    'pad_token_id': 17, 'eos_token_id': 0, 'separator_id': 0,
    'attention_mask': None, 'use_cache': False, 'tf32': False,
    'model_weights': 'strict_FP32_source', 'forward': 'cuda_BF16_autocast',
    'audio_frontend': 'FP32', 'prefix_dtype': 'bfloat16',
    'attention_backend': 'sdpa_math_or_eager_no_flash_no_mem_efficient',
    'likelihood': 'FP32_log_softmax_sum_body_only_no_BOS_EOS_label_leading_space',
    'choice_score_tie': 'first_original_valid_option', 'ranking_tie': 'shared_rank',
    'generation': {'first_mask_only': True, 'temperature': 0, 'top_p': 1,
                   'max_new_tokens': 300, 'stop_token_ids': [0], 'constrained': False},
    'max_context': 1024, 'baseline': '30_layers_likelihood_only_after_378_masks',
    'reasonaqa': 'test_AudioCaps-MCQ_and_Clotho-MCQ_single_recording_only',
}


def rank_indices(rank):
    if type(rank) is not int or not 0 <= rank < WORLD:
        raise ValueError('rank must be in 0..7')
    return list(range(rank, ROWS, WORLD))


def drop_mask(pair):
    pair = tuple(pair)
    if pair and pair not in PAIRS:
        raise ValueError(f'invalid original layer pair: {pair}')
    return [int(i in pair) for i in range(30)]


def mask_name(index):
    if type(index) is not int or not 0 <= index <= len(PAIRS):
        raise ValueError('invalid mask index')
    return f'mask_{index:03d}' if index < len(PAIRS) else 'baseline'
