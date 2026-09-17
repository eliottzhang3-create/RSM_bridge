"""Static contracts for the manifest-scoped shared waveform store."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "RSmol" / "scripts" / "prepare_unique_audio_waveform_store.py"
DATA = ROOT / "code" / "RSmol" / "audio_5_10x2_5_mesh_mellow" / "data.py"


class UniqueAudioWaveformStoreStaticTest(unittest.TestCase):
    def test_builder_is_manifest_scoped_single_file_and_resumable(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        ast.parse(text)
        for marker in (
            "UNIQUE_WAVEFORM_STORE_FORMAT",
            'DATA_FILE = "waveforms.f32"',
            'parser.add_argument("--manifest"',
            'parser.add_argument("--resume"',
            'parser.add_argument("--dry-run"',
            'parser.add_argument("--checkpoint-every"',
            'parser.add_argument("--verify-samples"',
            '"BUILDING"',
            '"progress.json"',
            '"index.jsonl"',
            '"waveform_sha256"',
            '"waveform_verification"',
            "refusing to overwrite existing waveform store",
            "resume build configuration/manifest/source inventory mismatch",
            "source changed while decoding",
            "verification byte mismatch",
            "os.fsync",
            "os.replace(partial_path, final_path)",
        ):
            self.assertIn(marker, text)

    def test_builder_reuses_exact_training_waveform_contract(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for marker in (
            "load_waveform(path, sample_rate=SAMPLE_RATE, seconds=SECONDS)",
            "SAMPLE_RATE = 32_000",
            "SECONDS = 10",
            "SAMPLES_PER_AUDIO = SAMPLE_RATE * SECONDS",
            "BYTES_PER_AUDIO = SAMPLES_PER_AUDIO * 4",
            '"dtype": "float32"',
            'waveform.to(dtype=torch.float32).contiguous()',
            "torch.isfinite(waveform).all()",
        ):
            self.assertIn(marker, text)

    def test_reader_maps_one_immutable_inode_with_fixed_stride(self) -> None:
        text = DATA.read_text(encoding="utf-8")
        ast.parse(text)
        for marker in (
            "class UniqueWaveformStore",
            'UNIQUE_WAVEFORM_STORE_FORMAT = "manifest_unique_fixed_waveform_store_v1"',
            '"data_file": "waveforms.f32"',
            'dtype="<f4"',
            'mode="c"',
            "shape=(self.num_audio, self.samples_per_audio)",
            "audio path is absent from unique waveform store",
            "audio_id * self.bytes_per_audio",
        ):
            self.assertIn(marker, text)

    def test_old_multishard_experiment_remains_isolated(self) -> None:
        builder = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("--num-shards", builder)
        self.assertNotIn("ShardAwareDistributedBatchSampler", builder)
        self.assertNotIn('f"shard-', builder)


if __name__ == "__main__":
    unittest.main()
