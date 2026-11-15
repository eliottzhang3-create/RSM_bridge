"""Bind inherited physical GPU visibility before importing any Torch modules."""
import os
import sys


def device_binding(environ):
    try:
        rank = int(environ['RANK'])
        local_rank = int(environ['LOCAL_RANK'])
        world = int(environ['WORLD_SIZE'])
        local_world = int(environ['LOCAL_WORLD_SIZE'])
    except (KeyError, ValueError) as exc:
        raise ValueError('worker requires single-node torchrun with eight ranks') from exc
    if world != 8 or local_world != 8 or rank != local_rank or not 0 <= rank < 8:
        raise ValueError('only single-node RANK=LOCAL_RANK, WORLD_SIZE=LOCAL_WORLD_SIZE=8 is supported')
    visible = environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        devices = [str(index) for index in range(8)]
    else:
        devices = [item.strip() for item in visible.split(',')]
        if len(devices) != 8 or len(set(devices)) != 8 or any(not item or item == '-1' for item in devices):
            raise ValueError('inherited CUDA_VISIBLE_DEVICES must expose exactly eight distinct GPUs')
    return {'rank': rank, 'local_rank': local_rank, 'world_size': world,
            'inherited_cuda_visible_devices': visible,
            'worker_cuda_visible_devices': devices[rank], 'logical_device': 'cuda:0'}


def bind_worker():
    if 'torch' in sys.modules:
        raise RuntimeError('worker GPU visibility must be bound before importing Torch')
    binding = device_binding(os.environ)
    os.environ['CUDA_VISIBLE_DEVICES'] = binding['worker_cuda_visible_devices']
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    return binding
