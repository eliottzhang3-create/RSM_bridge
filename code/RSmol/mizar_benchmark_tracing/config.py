from pathlib import Path
from mizar_layer_pair_search import config as original

ROOT,REMOTE,DATA = original.ROOT,original.REMOTE,original.DATA
MODEL_DIR,CHECKPOINT = original.MODEL_DIR,original.CHECKPOINT
PACKAGE = Path(__file__).resolve().parent
SCHEMA = 'mizar_mmau_mmar_all_eligible_silence_tracing_v1'
TASKS = ('mmau','mmar')
ROWS,WORLD,SEED = 1000,8,1234
GROUPS = {'mmau':{'perception':331,'semantic':230,'reasoning':439},
    'mmar':{'Signal Layer':43,'Perception Layer':404,'Semantic Layer':412,'Cultural Layer':141}}


def protocol(task):
    if task not in TASKS:raise ValueError('expected mmau/mmar')
    return {'schema':SCHEMA,'task':task,'source_rows':ROWS,'selection':'all_clean_correct_silence_wrong_positive_target_probability_gap',
        'selection_quota':None,'shuffle_seed':SEED,'group_counts':GROUPS[task],'screen_both_inputs_for_all1000':True,
        'model':'strict_original_final_full30_Mizar_encoder_trained_projector','weights':'FP32','forward':'BF16_autocast',
        'frontend':'FP32','audio':'SoundFile_resample32k_then_mono_first20_right_pad',
        'prompt':'compact_lowercase_plus_space_EOT_then_right_truncate_pad256_visible_PAD17',
        'prefix':[126,1,256,576],'readout_0based':382,'separator_id':0,'attention_mask':None,'use_cache':False,'tf32':False,
        'classification':'single_lowercase_letter_valid_original_slots_argmax_first_tie',
        'empty_slot':'null_score_excluded','unmapped_gold':'count_wrong_in1000_exclude_RR',
        'log_softmax':'full_vocabulary_FP32_CPU_after_forward_like_ReasonAQA_tracing','probability':'FP64_exp_no_candidate_renormalization',
        'corruption':'zero640000_waveform_through_native_encoder_projector_once_per_rank',
        'patch':'single_raw_post_block_text127_383_all256_before_final_norm_layers1_to30',
        'cache':'exact_full_unpooled_states_actual_dtype','RR':'per_question_FP64_probability_difference_ratio_no_epsilon_no_clip',
        'aggregate':'mean_per_question_RR','small_gap_threshold':1e-6,'exp_validation_max_FP64_ULPs':8,
        'bootstrap':{'replicates':2000,'seed':SEED,'unit':'question','CI':'pointwise95_percent'},
        'world':WORLD,'patch_shard':'selected_index_mod8_dynamic_N','barrier_after_layer':True}


def indices(rank,rows):
    if type(rank) is not int or not 0<=rank<WORLD or type(rows) is not int or not 0<=rows<=ROWS:
        raise ValueError('invalid rank or question count')
    return list(range(rank,rows,WORLD))
