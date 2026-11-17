"""Execution topology is independent of the frozen 100-question manifest."""
import sys

from . import arguments, config

SINGLE_OUTPUT = config.REMOTE / 'outputs/RSmol/mizar_30to28_pair_search_bf16_seed1234_1gpu4090_v1'


def validate_world(world_size):
    if type(world_size) is not int or world_size not in (1, 8):
        raise ValueError('only single-GPU or eight-GPU execution is supported')
    return world_size


def rank_indices(rank, world_size=8):
    validate_world(world_size)
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError('rank is outside the requested execution world')
    return list(range(rank, config.ROWS, world_size))


def protocol(world_size=8):
    validate_world(world_size)
    if world_size == 8:
        return config.PROTOCOL
    return {**config.PROTOCOL, 'world_size': 1, 'sharding': 'global_index_mod1_no_padding'}


def parse_args(argv=None, world_size=8):
    validate_world(world_size)
    argv = list(sys.argv[1:] if argv is None else argv)
    if world_size == 1 and not any(a == '--output-dir' or a.startswith('--output-dir=') for a in argv):
        argv += ['--output-dir', str(SINGLE_OUTPUT)]
    return arguments.parse_args(argv)
