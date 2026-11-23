from pathlib import Path
from mizar_layer_pair_search import config as reference

ROOT = reference.ROOT
PACKAGE = Path(__file__).resolve().parent
REMOTE, DATA, MODEL_DIR, CHECKPOINT, ROUTE = reference.REMOTE, reference.DATA, reference.MODEL_DIR, reference.CHECKPOINT, reference.ROUTE
FROZEN100 = DATA/'mizar_prune_calibration100_seed1234_v1/manifest.json'
COMMON_PROTOCOL = {
    'weights':'strict_official_final_FP32', 'forward':'CUDA_BF16_autocast',
    'prefix':'audio126_SEP0_prompt256_hidden576_BF16', 'audio':'SoundFile_FP32_resample32k_then_mono_first20_right_pad',
    'prompt':'compact_lowercase_no_question_EOT_PAD17_visible', 'use_cache':False, 'attention_mask':None,
    'TF32':False, 'seed':1234, 'world_size':8, 'node_count':1, 'generation':False,
}


def indices(rank, rows):
    if type(rank) is not int or not 0 <= rank < 8 or rows not in (100,1000):
        raise ValueError('expected eight workers and 100 or 1000 rows')
    return list(range(rank,rows,8))
