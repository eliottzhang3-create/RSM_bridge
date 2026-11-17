"""Rank-zero atomic per-record BF16 prefixes; completed entries survive restarts."""
import os

from . import audio, config, runtime, storage


def precision_identity(audit):
    return {k: audit[k] for k in ('torch', 'cuda', 'gpu', 'weights_dtype', 'forward_dtype',
                                  'audio_frontend_dtype', 'prefix_dtype', 'attention_implementation', 'sdpa_backends')}


def cache_identity(identity, audit):
    return {'schema': config.SCHEMA, 'run_identity_sha256': identity['identity_sha256'],
            'precision': precision_identity(audit), 'shape': [1, 383, 576], 'dtype': 'bfloat16'}


def load_entry(directory, record, expected):
    import torch
    index = record['global_index']
    meta = storage.verify_seal(storage.read_json(directory / f'prefix_{index:03d}.json'))
    path = directory / f'prefix_{index:03d}.pt'
    if (meta.get('cache_identity') != expected or meta.get('record_sha256') != storage.object_sha(record)
            or meta.get('file_sha256') != storage.file_sha(path)):
        raise ValueError('prefix cache identity/file mismatch')
    x = torch.load(path, map_location='cpu', weights_only=True)
    if (not torch.is_tensor(x) or x.dtype != torch.bfloat16 or tuple(x.shape) != (1, 383, 576)
            or not bool(torch.isfinite(x).all()) or audio.tensor_sha(x) != meta['tensor_sha256']):
        raise ValueError('invalid cached BF16 prefix')
    return x, meta


def prepare(loaded, records, directory, identity):
    import torch
    directory.mkdir(parents=True, exist_ok=True)
    expected = cache_identity(identity, loaded['audit'])
    storage.claim(directory, expected)
    missing = [r for r in records if not (directory / f'prefix_{r["global_index"]:03d}.json').exists()]
    if any(r['audio_ref']['kind'] == 'parquet' for r in missing):
        audio.prepare_selected_rows(missing)
    entries, check = [], None
    for record in records:
        index = record['global_index']
        meta_path = directory / f'prefix_{index:03d}.json'
        if meta_path.exists():
            _, meta = load_entry(directory, record, expected)
        else:
            wave, audit = audio.load_fixed20(record['audio_ref'])
            if audit != record['audio_audit']:
                raise ValueError(f'waveform/source differs from manifest: {record["key"]}')
            prefix = runtime.prefix(loaded, record, wave).cpu()
            path = directory / f'prefix_{index:03d}.pt'
            temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
            with temporary.open('wb') as stream:
                torch.save(prefix, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            meta = storage.seal({'cache_identity': expected, 'record_sha256': storage.object_sha(record),
                                 'file_sha256': storage.file_sha(path), 'tensor_sha256': audio.tensor_sha(prefix),
                                 'audio_audit': audit})
            storage.write_json(meta_path, meta)
            restored, _ = load_entry(directory, record, expected)
            if not torch.equal(prefix, restored):
                raise ValueError('online prefix/cache serialization is not exact')
            check = {'status': 'PASS', 'record': record['key'], 'bitwise_BF16_prefix_equal': True}
        entries.append(meta['sha256'])
        print(__import__('json').dumps({'event': 'BF16_PREFIX_READY', 'index': index, 'key': record['key']}), flush=True)
    prior_path = directory / 'cache_manifest.json'
    prior = storage.read_json(prior_path) if prior_path.exists() else None
    if check is None and prior is not None:
        check = prior['online_serialized_check']
    if check is None:
        # Crash after the last entry but before publishing cache_manifest: compare
        # online assembly from the persisted mapped tokens, without encoding again.
        first, _ = load_entry(directory, records[0], expected)
        device, embed = loaded['device'], loaded['lm'].get_input_embeddings()
        with torch.inference_mode():
            ids = torch.tensor([records[0]['prompt_token_ids']], device=device)
            sep = torch.tensor([[0]], device=device)
            assembled = torch.cat([first[:, :126].to(device), embed(sep).to(torch.bfloat16),
                                   embed(ids).to(torch.bfloat16)], dim=1).cpu()
        if not torch.equal(first, assembled):
            raise ValueError('cached prefix does not match online separator/prompt embedding assembly')
        check = {'status': 'PASS', 'record': records[0]['key'], 'bitwise_BF16_prefix_equal': True,
                 'recovered_after_entry_commit': True}
    result = storage.seal({'cache_identity': expected, 'entry_meta_sha256': entries,
                           'rows': 100, 'online_serialized_check': check})
    storage.write_json(prior_path, result)
    audio.clear_selected_rows()
    return result


def load_all(loaded, records, directory, identity):
    import torch
    expected = cache_identity(identity, loaded['audit'])
    meta = storage.verify_seal(storage.read_json(directory / 'cache_manifest.json'))
    if meta['cache_identity'] != expected or meta['rows'] != 100 or meta['online_serialized_check']['status'] != 'PASS':
        raise ValueError('cache ready identity/check mismatch')
    result, hashes = {}, []
    for record in records:
        tensor, entry = load_entry(directory, record, expected)
        result[record['global_index']] = tensor.to(loaded['device'], dtype=torch.bfloat16)
        hashes.append(entry['sha256'])
    if hashes != meta['entry_meta_sha256']:
        raise ValueError('cache inventory changed after publication')
    return result, meta
