"""Bounded-memory full-file CPU decoding; no encoder, resampling or audio rewrite."""
from __future__ import annotations

import hashlib
from pathlib import Path


class AudioAuditor:
    def __init__(self):
        # Fail BEFORE scanning/filtering if dependencies are unavailable. A
        # missing decoder must never masquerade as thousands of bad recordings.
        import numpy as np
        import soundfile as sf
        self.np, self.sf = np, sf
        self.decoder = {"name": "soundfile", "version": sf.__version__,
                        "libsndfile": sf.__libsndfile_version__, "numpy": np.__version__}
        self.cache = {}

    @staticmethod
    def stat(path):
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        if not path.is_file() or stat.st_size <= 0:
            raise ValueError("audio missing, not a regular file or empty")
        return path, {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                      "device": stat.st_dev, "inode": stat.st_ino}

    def check(self, path):
        path, stamp = self.stat(path)
        key = str(path)
        if key in self.cache:
            saved = self.cache[key]
            if saved["stat"] != stamp:
                raise ValueError(f"audio changed during audit: {path}")
            return saved
        count, peak, energy = 0, 0.0, 0.0
        decoded_hash = hashlib.sha256()
        with self.sf.SoundFile(str(path)) as handle:
            rate, channels, header_frames = handle.samplerate, handle.channels, handle.frames
            if rate <= 0 or channels <= 0 or header_frames <= 0:
                raise ValueError("audio has invalid samplerate/channels/frames")
            for block in handle.blocks(blocksize=65536, dtype="float32", always_2d=True):
                if not self.np.isfinite(block).all():
                    raise ValueError("audio contains NaN or infinity")
                decoded_hash.update(block.astype("<f4", copy=False).tobytes())
                count += len(block)
                peak = max(peak, float(self.np.abs(block).max(initial=0)))
                energy += float(self.np.square(block, dtype=self.np.float64).sum())
            subtype, format_name = handle.subtype, handle.format
        if not count:
            raise ValueError("audio decoder produced zero frames")
        if self.stat(path)[1] != stamp:
            raise ValueError(f"audio changed while decoding: {path}")
        result = {"resolved_path": key, "stat": stamp, "samplerate": rate,
                  "channels": channels, "decoded_frames": count,
                  "header_frames": header_frames, "header_frame_disagreement": count != header_frames,
                  "duration_seconds": count / rate, "peak": peak,
                  "rms": (energy / (count * channels)) ** 0.5, "all_zero": peak == 0,
                  "first20_will_pad": count < rate * 20,
                  "first20_will_crop": count > rate * 20,
                  "decoded_native_f32_sha256": decoded_hash.hexdigest(),
                  "subtype": subtype, "format": format_name}
        self.cache[key] = result
        return result

    def recheck(self):
        for path, result in self.cache.items():
            if self.stat(path)[1] != result["stat"]:
                raise ValueError(f"audio changed before publication: {path}")
