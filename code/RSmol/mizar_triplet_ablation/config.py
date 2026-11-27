from pathlib import Path
from mizar_layer_pair_search import config as reference

ROOT,REMOTE,DATA = reference.ROOT,reference.REMOTE,reference.DATA
MODEL_DIR,CHECKPOINT,ROUTE = reference.MODEL_DIR,reference.CHECKPOINT,reference.ROUTE
PACKAGE=Path(__file__).resolve().parent
SCHEMA='mizar_triplet_start06_25_letters_eot_bf16_8gpu3090_v1'
TASKS=('mmau','mmar');ROWS,WORLD=1000,8
CONDITIONS=(None,*range(6,26))
GROUP_COUNTS={'mmau':{'perception':331,'semantic':230,'reasoning':439},
    'mmar':{'Signal Layer':43,'Perception Layer':404,'Semantic Layer':412,'Cultural Layer':141}}


def conditions(task):
    if task not in TASKS:raise ValueError('expected mmau/mmar')
    return CONDITIONS


def drop_layers(start):
    if start is None:return []
    if type(start) is not int or not 6<=start<=25:raise ValueError('triplet start must be6..25')
    return list(range(start,start+3))


def condition_name(task,start):
    conditions(task);drop_layers(start)
    return 'baseline' if start is None else f'drop_{start:02d}_{start+2:02d}'


def indices(rank,rows=ROWS):
    if type(rank) is not int or not 0<=rank<8 or rows!=1000:raise ValueError('requires8 ranks and all1000 rows')
    return list(range(rank,1000,8))


def protocol(task):
    conditions(task)
    return {'schema':SCHEMA,'task':task,'rows':1000,'world':8,'partition':'pdgpu-3090','conditions':list(CONDITIONS),
        'triplet_size':3,'start_layers_1based':list(range(6,26)),'stride':1,'baseline_first':True,
        'intervention':'each_selected_full_decoder_block_h_out_equals_h_in_all_tokens_nonaccumulating',
        'retained_layers':27,'protected_layers_1based':[1,2,3,4,5,28,29,30],
        'weights':'strict_original_final_FP32','forward':'BF16_autocast','frontend':'FP32','tf32':False,
        'audio':'SoundFile_resample32k_then_mono_first20_right_pad',
        'prompt':'compact_lowercase_plus_space_EOT_then_tokenize_right_truncate_pad256_PAD17_visible',
        'prefix':[126,1,256,576],'separator_id':0,'readout_0based':382,'attention_mask':None,'use_cache':False,
        'scoring':'FP32_full_vocab_log_softmax_GPU_gather_bare_lowercase_single_letter_valid_slots',
        'tie':'first_valid_original_slot','generation':False,'sampling':False,'tracing_subset':False,
        'empty_option':'null_exclude_argmax_preserve_slots','unmapped_gold':'retain1000_denominator_count_wrong',
        'groups':GROUP_COUNTS[task],'cache':'one_prefix_per_question_per_task_reused_across_all21_conditions',
        'shard':'global_index_mod8_125_each','barrier_after_condition':True}
