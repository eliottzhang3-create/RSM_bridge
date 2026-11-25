"""Fixed MMAR coverage, original Mizar inference and one-based ablations."""
from pathlib import Path
from mizar_layer_pair_search import config as reference

ROOT = reference.ROOT
PACKAGE = Path(__file__).resolve().parent
REMOTE, DATA = reference.REMOTE, reference.DATA
MODEL_DIR, CHECKPOINT, ROUTE = reference.MODEL_DIR, reference.CHECKPOINT, reference.ROUTE
SCHEMA = 'mizar_mmar_ablation_bf16_8gpu5090_v1'
MANIFEST_SCHEMA = 'mizar_mmar1000_likelihood_inputs_v1'
ROWS, WORLD = 1000, 8
CONDITIONS = (None, *range(2, 30))
CATEGORIES = ('Signal Layer', 'Perception Layer', 'Semantic Layer', 'Cultural Layer')
CATEGORY_COUNTS = dict(zip(CATEGORIES, (43, 404, 412, 141)))
COMMON_PROTOCOL = {
    'schema': SCHEMA, 'weights': 'strict_official_final_FP32', 'forward': 'CUDA_BF16_autocast',
    'prefix': 'audio126_SEP0_prompt256_hidden576_BF16',
    'audio': 'SoundFile_FP32_resample32k_then_mono_first20_right_pad',
    'prompt': 'compact_lowercase_no_question_EOT_PAD17_visible_original_option_slots',
    'use_cache': False, 'attention_mask': None, 'TF32': False, 'seed': 1234,
    'world_size': WORLD, 'node_count': 1, 'generation': False,
}
INPUT_PROTOCOL = {
    'schema': MANIFEST_SCHEMA, 'rows': ROWS, 'categories': CATEGORY_COUNTS,
    'sampling': False, 'original_order_and_slots': True,
    'empty_choices': 'retain_prompt_slot_exclude_from_argmax_save_null_scores',
    'gold': 'unique_normalized_body_match_else_null_count_incorrect_in_full_denominator',
    'scoring': 'FP32_log_softmax_sum_full_body_no_BOS_EOS_label_or_added_space',
    'official_diagnostic': 'unmodified_MMAR_string_match_on_likelihood_selected_body',
    'audio': COMMON_PROTOCOL['audio'], 'prompt': COMMON_PROTOCOL['prompt'],
}


def indices(rank, rows=ROWS):
    if type(rank) is not int or not 0 <= rank < WORLD or rows != ROWS:
        raise ValueError('expected eight workers and all 1000 MMAR rows')
    return list(range(rank, rows, WORLD))
