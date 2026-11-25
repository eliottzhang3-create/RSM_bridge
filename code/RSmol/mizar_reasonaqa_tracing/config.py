"""Frozen experiment; no model/data imports in argument parsing."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
REMOTE = Path('/hpc_stor03/sjtu_home/jinwei.zhang')
DATA = REMOTE / 'data'
MODEL_DIR = REMOTE / 'models/Mizar159M'
CHECKPOINT = Path('checkpoints/final/stage3_q4_seed20260905.ckpt')
SOURCE = DATA / 'reasonaqa/test_close_ended_sample5000_seed1234_v1/manifest.json'
POOL = DATA / 'reasonaqa/mizar_tracing_pool5000_letters_eot_v1/manifest.json'
OUTPUT = REMOTE / 'outputs/RSmol/mizar_reasonaqa_tracing1000_letters_eot_bf16_8gpu5090_v1'
SCHEMA = 'mizar_reasonaqa_silence_text_patch1000_v1'
POOL_SIZE, TARGET, SEED, WORLD, LAYERS = 5000, 1000, 1234, 8, 30
CATEGORIES = ('audiocaps_mcq', 'clotho_mcq', 'clotho_aqa_yesno')
EOT = ' <|endoftext|>'
PROTOCOL = {
    'schema': SCHEMA, 'candidate_pool': 'existing_fixed_5000_only', 'shuffle_seed': SEED,
    'selection': 'clean_correct_and_corrupted_wrong_and_Pclean_strictly_greater',
    'target_rows': TARGET, 'stop': 'ordered_windows_min8_remaining_target_remaining_pool_no_overrun',
    'selected_order': 'original_source_row_index_ascending',
    'model': 'original_full_Mizar_final_encoder_trained_projector_distinct30_LM',
    'source_weights': 'FP32', 'forward': 'BF16_autocast', 'audio_frontend': 'FP32', 'tf32': False,
    'audio': 'SoundFile_resample32k_then_mono_first20_right_zero_pad',
    'corruption': 'zero_waveform_640000_through_unchanged_encoder_and_projector',
    'prompt': 'compact_lowercase_original_options_then_space_EOT_right_truncate_pad256',
    'prefix': [126, 1, 256, 576], 'separator_id': 0, 'pad_id': 17,
    'attention_mask': None, 'use_cache': False, 'readout_0based': 382,
    'classification': 'bare_lowercase_abcd_or_yes_no_single_ordinary_tokens_argmax',
    'probability': 'target_token_full_vocabulary_FP32_log_softmax_exp_FP64',
    'patch': 'single_layer_raw_post_block_before_final_norm_text_slice127_383_all256_slots',
    'patch_layers_1based': list(range(1, 31)), 'patch_cache': 'exact_unpooled_actual_dtype',
    'RR': '(Ppatched-Pcorrupted)/(Pclean-Pcorrupted)_FP64_no_epsilon_no_clip',
    'aggregate': 'mean_of_per_question_RR', 'small_gap_threshold': 1e-6,
    'world': WORLD, 'patch_sharding': 'selected_index_mod8_no_padding_same_layer_barrier',
}


def indices(rank):
    if type(rank) is not int or not 0 <= rank < WORLD:
        raise ValueError('rank must be 0..7')
    return list(range(rank, TARGET, WORLD))
