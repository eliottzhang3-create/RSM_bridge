"""Native-Mizar first20 audio; deliberately separate from center-crop evaluators."""
import hashlib
import io
from pathlib import Path

from . import data, storage

_SELECTED_ROWS = {}


def prepare_selected_rows(records):
    """Stream each parquet once; keep only the selected audio payloads in RAM."""
    import pyarrow.parquet as pq
    wanted = {}
    for record in records:
        ref = record['audio_ref']
        if ref['kind'] == 'parquet':
            wanted.setdefault(ref['path'], set()).add(ref['row_index'])
    for path, indices in wanted.items():
        with pq.ParquetFile(path) as parquet:
            offset = 0
            for batch in parquet.iter_batches(batch_size=8, use_threads=False):
                if any(offset <= i < offset+batch.num_rows for i in indices):
                    for local, row in enumerate(batch.to_pylist()):
                        if offset+local in indices:
                            _SELECTED_ROWS[(path, offset+local)] = row
                offset += batch.num_rows
                if offset > max(indices):
                    break
        if any((path, i) not in _SELECTED_ROWS for i in indices):
            raise ValueError('selected audio rows missing from parquet')


def clear_selected_rows():
    _SELECTED_ROWS.clear()


def parquet_row(path, index):
    if (str(path), index) in _SELECTED_ROWS:
        return _SELECTED_ROWS[(str(path), index)]
    import pyarrow.parquet as pq
    with pq.ParquetFile(path) as parquet:
        start = 0
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            if start <= index < start + count:
                return parquet.read_row_group(group, use_threads=False).slice(index-start, 1).to_pylist()[0]
            start += count
    raise IndexError(f'parquet row missing: {path}:{index}')


def payload(ref):
    if ref['kind'] == 'file':
        return Path(ref['path']), {'audio_source_sha256': storage.file_sha(ref['path']), 'decoder': 'torchaudio_file'}
    row = parquet_row(ref['path'], ref['row_index'])
    if ref['format'] == 'mmau':
        helper = data.legacy_mmau()
        identifier = helper.extract_row_id(row)[0]
        value = helper.extract_audio_payload(row)[0]
    else:
        identifier, value = str(row['id']), row['audio']
    if identifier != ref['expected_id']:
        raise ValueError('parquet row ID differs from frozen metadata')
    if isinstance(value, dict) and value.get('bytes') is not None:
        raw = bytes(value['bytes'])
        if not raw:
            raise ValueError('empty embedded audio')
        return io.BytesIO(raw), {'audio_source_sha256': hashlib.sha256(raw).hexdigest(),
                                 'decoder': 'soundfile_embedded_bytes'}
    if isinstance(value, dict) and value.get('array') is not None:
        rate = value.get('sampling_rate') or value.get('sample_rate')
        if not rate:
            raise ValueError('array audio lacks actual sampling rate')
        return (value['array'], int(rate)), {'decoder': 'embedded_array'}
    raw_path = value.get('path') if isinstance(value, dict) else value
    if not isinstance(raw_path, (str, Path)) or not str(raw_path):
        raise ValueError('audio payload has neither bytes, array nor path')
    path = Path(raw_path)
    if not path.is_absolute():
        path = data.contained(ref['base_dir'], str(path))
    return path, {'audio_source_sha256': storage.file_sha(path), 'decoder': 'torchaudio_payload_path'}


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def load_fixed20(ref):
    import numpy as np
    import torch
    import torchaudio
    source, audit = payload(ref)
    if isinstance(source, io.BytesIO):
        import soundfile as sf
        array, rate = sf.read(source, dtype='float32', always_2d=True)
        wave = torch.from_numpy(np.ascontiguousarray(array.T))
    elif isinstance(source, tuple):
        array, rate = source
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError('embedded array must be mono; do not guess channel axis')
        wave = torch.from_numpy(array.copy()).unsqueeze(0)
    else:
        wave, rate = torchaudio.load(str(source))
    if wave.ndim != 2 or not wave.numel() or rate <= 0 or not bool(torch.isfinite(wave).all()):
        raise ValueError('empty/nonfinite/invalid decoded audio')
    audit.update(original_sample_rate=int(rate), original_channels=wave.shape[0], original_samples=wave.shape[1])
    wave = wave.float()
    # Match official prepare_inference.py: resample channels first, then mono.
    if rate != 32000:
        wave = torchaudio.transforms.Resample(int(rate), 32000)(wave)
    wave = wave.mean(dim=0)
    length = wave.numel()
    wave = torch.nn.functional.pad(wave[:640000], (0, max(0, 640000-length))).contiguous()
    if wave.shape != (640000,) or not bool(torch.isfinite(wave).all()):
        raise ValueError('invalid native fixed20 waveform')
    audit.update(resampled_samples=length, cropped_samples=max(0, length-640000),
                 right_pad_samples=max(0, 640000-length), crop_start=0,
                 waveform_sha256=tensor_sha(wave), policy='resample_then_mono_first20_right_pad_FP32')
    return wave, audit
