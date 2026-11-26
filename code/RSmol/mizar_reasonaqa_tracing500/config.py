import copy
from pathlib import Path
from mizar_reasonaqa_tracing import config as original

ROOT, REMOTE = original.ROOT, original.REMOTE
PACKAGE = Path(__file__).resolve().parent
SOURCE_OUTPUT = original.OUTPUT
OUTPUT = REMOTE/'outputs/RSmol/mizar_reasonaqa_tracing500_letters_eot_bf16_8gpu5090_v1'
SCHEMA = 'mizar_reasonaqa_silence_text_patch500_reuse_v1'
TARGET, WORLD, SEED, CATEGORIES = 500, 8, original.SEED, original.CATEGORIES
PROTOCOL = copy.deepcopy(original.PROTOCOL)
PROTOCOL.update(schema=SCHEMA, target_rows=TARGET,
    stop='reuse_first500_accepted_in_existing_shuffle_journal_no_screening_forward',
    selected_order='original_source_row_index_ascending',
    cache_source='read_only_original_screening_full_clean_states_and_prefixes')


def indices(rank):
    if type(rank) is not int or not 0 <= rank < WORLD:
        raise ValueError('rank must be 0..7')
    return list(range(rank, TARGET, WORLD))
