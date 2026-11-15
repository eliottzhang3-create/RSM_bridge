"""Reuse the frozen thirty targets and the existing immutable 5000-row sample."""
from pathlib import Path

from mizar_s1_ced_reasonaqa5000_eval.config import (
    ROOT, REMOTE, TARGETS, EXCLUDED, DEFAULT_MANIFEST, SAMPLE_SIZE, target_name,
)

PACKAGE = Path(__file__).resolve().parent
WORLD_SIZE = 8
ROWS_PER_RANK = SAMPLE_SIZE // WORLD_SIZE
OUTPUT_NAME = 'reasonaqa_close_ended_sample5000_seed1234_step{step}_8gpu_v1'
PARALLEL = {
    'version': 'reasonaqa_sample5000_8gpu_v1',
    'world_size': WORLD_SIZE,
    'sharding': 'manifest_index_modulo_world_size_no_padding_no_shuffle',
    'rows_per_rank': ROWS_PER_RANK,
    'batch_size_per_gpu': 1,
    'device_binding': 'mask_one_inherited_visible_gpu_before_torch_import',
    'launcher': 'single_node_torchrun_no_restarts',
    'model_replica_per_gpu': True,
    'ddp_gradient_wrapper': False,
}


def rank_indices(rank):
    if type(rank) is not int or not 0 <= rank < WORLD_SIZE:
        raise ValueError('rank must be an integer from 0 to 7')
    return list(range(rank, SAMPLE_SIZE, WORLD_SIZE))


def rank_dir(output, rank):
    rank_indices(rank)
    return output / 'ranks' / f'rank{rank:02d}'
