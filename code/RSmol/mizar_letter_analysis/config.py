"""Only letter scoring and prompt EOT change; retain original inputs and routes."""
from pathlib import Path
from mizar_layer_pair_search import config as reference

ROOT = reference.ROOT
PACKAGE = Path(__file__).resolve().parent
REMOTE, DATA, MODEL_DIR, CHECKPOINT = reference.REMOTE, reference.DATA, reference.MODEL_DIR, reference.CHECKPOINT
SCHEMA = 'mizar_lowercase_letter_eot_bf16_v1'
TASKS = ('cutoff', 'mmau', 'mmar')
ROWS, WORLD = 1000, 8
EOT_SUFFIX = ' <|endoftext|>'
ISOLATION_CHECKS = (3, 28)
SOURCE_MANIFESTS = {
    'cutoff': DATA/'mizar_audio_cutoff_balanced1000_seed1234_v1/manifest.json',
    'mmau': DATA/'mizar_mmau1000_likelihood_v1/manifest.json',
    'mmar': DATA/'mizar_mmar1000_likelihood_v1/manifest.json',
}
RUN_NAMES = {
    'cutoff': 'mizar_audio_sep_cutoff03_28_1000_letters_eot_bf16_8gpu5090_v1',
    'mmau': 'mizar_mmau_ablation_letters_eot_bf16_8gpu5090_v1',
    'mmar': 'mizar_mmar_ablation_letters_eot_bf16_8gpu5090_v1',
}
GROUP_COUNTS = {
    'cutoff': dict(zip(reference.DATASETS, (250, 250, 250, 250))),
    'mmau': {'perception': 331, 'semantic': 230, 'reasoning': 439},
    'mmar': {'Signal Layer': 43, 'Perception Layer': 404, 'Semantic Layer': 412, 'Cultural Layer': 141},
}


def conditions(task):
    check_task(task)
    return (None, *range(3, 29)) if task == 'cutoff' else (None, *range(2, 30))


def check_task(task):
    if task not in TASKS:
        raise ValueError('expected cutoff, mmau or mmar')


def condition_name(task, layer):
    if layer is not None and (type(layer) is not int or layer not in conditions(task)):
        raise ValueError('invalid original layer for this task')
    check_task(task)
    return 'baseline' if layer is None else f'{"cutoff" if task == "cutoff" else "ablate"}_{layer:02d}'


def indices(rank, rows=ROWS):
    if type(rank) is not int or not 0 <= rank < WORLD or rows != ROWS:
        raise ValueError('requires eight ranks and all 1000 original questions')
    return list(range(rank, rows, WORLD))


def protocol(task):
    return {
        'schema': SCHEMA, 'task': task, 'rows': ROWS, 'world_size': WORLD,
        'conditions': list(conditions(task)), 'baseline_first': True,
        'weights': 'strict_official_final_FP32', 'forward': 'CUDA_BF16_autocast',
        'audio': 'SoundFile_resample32k_then_mono_first20_right_zero_pad_FP32',
        'prompt': 'original_saved_compact_lowercase_prompt_plus_space_EOT_then_tokenize256',
        'prompt_suffix': EOT_SUFFIX, 'prompt_padding': 'right_PAD17_visible',
        'prompt_truncation': 'right_max256_including_EOT_suffix_same_as_formal_tokenize',
        'prefix': 'audio126_SEP0_prompt256_BF16_383x576',
        'scoring': 'FP32_full_vocab_log_softmax_at_position382_gather_single_lowercase_letter',
        'letter_format': 'a_b_c_d_or_actual_original_slots_no_parenthesis_no_space_no_body_no_answer_EOS',
        'single_LM_forward_per_question': True, 'choice_tie': 'first_valid_original_slot',
        'group_counts': GROUP_COUNTS[task], 'input_selection': 'exact_old_manifest_records_no_resampling_or_reordering',
        'TF32': False, 'use_cache': False, 'generation': False,
        'intervention': 'audio_plus_SEP_cutoff_from_layer_L_through30' if task == 'cutoff' else 'single_block_h_out_equals_h_in',
    }
