"""Atomic, lossless full-state caches, bound to the screened question and run."""
import os
from pathlib import Path
from mizar_layer_pair_search import storage, audio


def folder(directory, index):
    return Path(directory)/'cache'/f'source_{index:04d}'


def save(directory, record, expected, clean_prefix, corrupted_prefix, states, audio_audit, precision):
    import torch
    target = folder(directory, record['source_manifest_index'])
    target.mkdir(parents=True, exist_ok=True)
    data = {k: v.detach().clone().cpu().contiguous() for k, v in
            (('clean_prefix', clean_prefix), ('corrupted_prefix', corrupted_prefix), ('clean_states', states))}
    validate_tensors(data)
    path = target/'tensors.pt'
    temporary = path.with_name(f'tensors.pt.tmp.{os.getpid()}')
    with temporary.open('wb') as stream:
        torch.save(data, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    meta = storage.seal({'identity_sha256': expected['identity_sha256'],
        'record_sha256': storage.object_sha(record), 'source_manifest_index': record['source_manifest_index'],
        'audio_audit': audio_audit, 'precision': precision, 'file_sha256': storage.file_sha(path),
        'tensor_sha256': {k: audio.tensor_sha(v) for k, v in data.items()},
        'tensor_shapes': {k: list(v.shape) for k, v in data.items()},
        'tensor_dtypes': {k: str(v.dtype) for k, v in data.items()}})
    storage.write_json(target/'entry.json', meta)
    return meta


def validate_tensors(data):
    import torch
    if set(data) != {'clean_prefix', 'corrupted_prefix', 'clean_states'}:
        raise ValueError('incorrect full-state cache keys')
    for key in ('clean_prefix', 'corrupted_prefix'):
        if data[key].shape != (1, 383, 576) or data[key].dtype != torch.bfloat16:
            raise ValueError('invalid cached original BF16 prefix383')
    if data['clean_states'].shape != (30, 256, 576) or data['clean_states'].dtype not in (torch.bfloat16, torch.float32):
        raise ValueError('invalid full raw clean activation cache')
    if (any(not bool(torch.isfinite(x).all()) for x in data.values())
            or not torch.equal(data['clean_prefix'][:, 126:], data['corrupted_prefix'][:, 126:])):
        raise ValueError('cache nonfinite or corruption changed text/SEP')


def load(directory, record, expected, *, digest=None, precision=None, tensors=True):
    import torch
    from mizar_layer_pair_search import storage
    target = folder(directory, record['source_manifest_index'])
    meta = storage.verify_seal(storage.read_json(target/'entry.json'))
    if (meta['identity_sha256'] != expected['identity_sha256'] or meta['record_sha256'] != storage.object_sha(record)
            or meta['source_manifest_index'] != record['source_manifest_index']
            or digest is not None and meta['sha256'] != digest
            or precision is not None and meta['precision'] != precision
            or storage.file_sha(target/'tensors.pt') != meta['file_sha256']):
        raise ValueError('activation cache identity/precision/content changed')
    if not tensors:
        return None, meta
    data = torch.load(target/'tensors.pt', map_location='cpu', weights_only=True)
    validate_tensors(data)
    if ({k: audio.tensor_sha(v) for k, v in data.items()} != meta['tensor_sha256']
            or {k: str(v.dtype) for k, v in data.items()} != meta['tensor_dtypes']
            or {k: list(v.shape) for k, v in data.items()} != meta['tensor_shapes']):
        raise ValueError('activation tensor fingerprint changed')
    return data, meta


def precision(audit):
    return {k: audit[k] for k in ('torch', 'transformers', 'cuda', 'gpu', 'weights_dtype', 'forward_dtype',
            'audio_frontend_dtype', 'prefix_dtype', 'attention_implementation', 'sdpa_backends')}
