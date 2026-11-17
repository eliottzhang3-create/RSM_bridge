"""Select one inherited visible GPU before any Torch/model import."""
import os
import sys


def device_binding(environ):
    try:
        rank, local = int(environ['RANK']), int(environ['LOCAL_RANK'])
        world, local_world = int(environ['WORLD_SIZE']), int(environ['LOCAL_WORLD_SIZE'])
    except (KeyError, ValueError) as exc:
        raise ValueError('single-node eight-rank torchrun is required') from exc
    if world != 8 or local_world != 8 or rank != local or not 0 <= rank < 8:
        raise ValueError('only one node with eight distinct GPU workers is supported')
    inherited = environ.get('CUDA_VISIBLE_DEVICES')
    devices = [str(i) for i in range(8)] if inherited is None else [s.strip() for s in inherited.split(',')]
    if len(devices) != 8 or len(set(devices)) != 8 or any(not s or s == '-1' for s in devices):
        raise ValueError('inherited CUDA_VISIBLE_DEVICES must expose eight distinct devices')
    return {'rank': rank, 'local_rank': local, 'world_size': 8,
            'inherited_cuda_visible_devices': inherited,
            'worker_cuda_visible_devices': devices[rank], 'logical_device': 'cuda:0'}


def bind_worker():
    if 'torch' in sys.modules:
        raise RuntimeError('GPU visibility must be bound before importing torch')
    result = device_binding(os.environ)
    os.environ['CUDA_VISIBLE_DEVICES'] = result['worker_cuda_visible_devices']
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    return result
