"""Select one inherited visible GPU before any Torch/model import."""
import os
import sys

from . import execution


def device_binding(environ, world_size=8):
    execution.validate_world(world_size)
    try:
        rank, local = int(environ['RANK']), int(environ['LOCAL_RANK'])
        world, local_world = int(environ['WORLD_SIZE']), int(environ['LOCAL_WORLD_SIZE'])
    except (KeyError, ValueError) as exc:
        raise ValueError('single-node torchrun is required') from exc
    if world != world_size or local_world != world_size or rank != local or not 0 <= rank < world_size:
        raise ValueError('torchrun topology differs from the requested GPU count')
    inherited = environ.get('CUDA_VISIBLE_DEVICES')
    devices = [str(i) for i in range(world_size)] if inherited is None else [s.strip() for s in inherited.split(',')]
    if len(devices) != world_size or len(set(devices)) != world_size or any(not s or s == '-1' for s in devices):
        raise ValueError('inherited CUDA_VISIBLE_DEVICES must match the requested GPU count')
    return {'rank': rank, 'local_rank': local, 'world_size': world_size,
            'inherited_cuda_visible_devices': inherited,
            'worker_cuda_visible_devices': devices[rank], 'logical_device': 'cuda:0'}


def bind_worker(world_size=8):
    if 'torch' in sys.modules:
        raise RuntimeError('GPU visibility must be bound before importing torch')
    result = device_binding(os.environ, world_size)
    os.environ['CUDA_VISIBLE_DEVICES'] = result['worker_cuda_visible_devices']
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    return result
