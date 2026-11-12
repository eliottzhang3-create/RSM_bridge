"""Real Parquet/audio/tensor checks; explicitly skip unavailable dependencies."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code/RSmol"), str(ROOT / "tests")]
from mizar_s1_ced_mmsu_eval import dataset, runtime
from test_mizar_s1_ced_mmsu_eval import record

HAS_ARROW = importlib.util.find_spec("pyarrow") is not None
HAS_SOUNDFILE = importlib.util.find_spec("soundfile") is not None
HAS_AUDIO = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchaudio", "soundfile"))


@unittest.skipUnless(HAS_SOUNDFILE, "requires soundfile")
class RealAudioReadTests(unittest.TestCase):
    def test_bytes_precede_nonexistent_path_and_preserve_header_rate_channels(self):
        import numpy as np
        import soundfile as sf
        samples = np.column_stack([np.ones(8000, dtype=np.float32) * .5, np.zeros(8000, dtype=np.float32)])
        stream = io.BytesIO()
        sf.write(stream, samples, 8000, format="WAV", subtype="FLOAT")
        actual, rate, source = runtime.read_audio({"bytes": stream.getvalue(), "path": "/missing/audio.wav"}, Path("."))
        self.assertTrue(np.array_equal(actual, samples))
        self.assertEqual(rate, 8000)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(source, "parquet_embedded_bytes")

    def test_path_fallback_and_corrupt_bytes_are_not_silently_replaced(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            sf.write(path / "audio.wav", np.zeros(320, dtype=np.float32), 32000)
            _, rate, source = runtime.read_audio({"bytes": None, "path": "audio.wav"}, path)
            self.assertEqual(rate, 32000)
            self.assertTrue(source.startswith("path_fallback:"))
            with self.assertRaises(Exception):
                runtime.read_audio({"bytes": b"corrupt", "path": "audio.wav"}, path)
            with self.assertRaises(ValueError):
                runtime.read_audio({"bytes": b"", "path": "audio.wav"}, path)


@unittest.skipUnless(HAS_ARROW, "requires pyarrow; run in mellow_c8204d8")
class RealParquetTests(unittest.TestCase):
    def shards(self, root):
        import pyarrow as pa
        import pyarrow.parquet as pq
        directory = root / "data"
        directory.mkdir()
        offset = 0
        for name, count in zip(dataset.SHARD_NAMES, dataset.SHARD_ROWS):
            rows = [{**record(offset + i), "audio": {"bytes": b"fixture", "path": "stale/path.wav"}}
                    for i in range(count)]
            table = pa.Table.from_pylist(rows)
            table = table.set_column(table.schema.get_field_index("choice_b"), "choice_b",
                                     pa.array([None] * count, type=pa.string()))
            pq.write_table(table, directory / name)
            offset += count
        return dataset.shard_paths(root)

    def test_real_three_shard_global_order_and_nullable_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.shards(Path(directory))
            rows, audit = dataset.preflight(paths, 13)
            self.assertEqual(len(rows), 5000)
            self.assertEqual(audit["null_choices"], {"choice_b": 5000})
            streamed = list(dataset.iter_rows(paths, audio=True, batch_size=7))
            self.assertEqual([i for i, _ in streamed], list(range(5000)))
            self.assertEqual([r["id"] for _, r in streamed], [f"sample-{i}" for i in range(5000)])
            self.assertEqual(streamed[1667][1]["audio"]["bytes"], b"fixture")

    def test_real_metadata_and_schema_errors(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            paths = self.shards(Path(directory))
            table = pq.read_table(paths[1])
            changed = table.to_pylist()
            changed[0]["id"] = "sample-0"
            pq.write_table(pa.Table.from_pylist(changed, schema=table.schema), paths[1])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                dataset.preflight(paths)
            pq.write_table(table.drop(["question"]), paths[1])
            with self.assertRaisesRegex(ValueError, "schema"):
                dataset.preflight(paths)

    def test_local_uploaded_dataset_metadata_if_available(self):
        root = Path("C:/Xlance/assets/MMSU/MMSU data")
        if not root.is_dir():
            self.skipTest("optional local MMSU asset copy unavailable")
        rows, audit = dataset.preflight(dataset.shard_paths(root))
        self.assertEqual(len(rows), 5000)
        self.assertEqual(audit["task_count"], 47)
        self.assertEqual(audit["null_choices"], dict(choice_a=214, choice_b=225, choice_c=237, choice_d=230))


@unittest.skipUnless(HAS_AUDIO, "requires torch, torchaudio, soundfile; run in mellow_c8204d8")
class RealAudioTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        spec = importlib.util.spec_from_file_location("mmsu_training_fixed20", runtime.base.ROUTE_ROOT / "data/fixed20.py")
        # fixed_len itself is the exact training implementation; utils.audio_io is
        # loaded via the selected training route for the genuine module import.
        sys.path.insert(0, str(runtime.base.ROUTE_ROOT))
        cls.fixed20 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fixed20)

    def wav(self, samples, rate):
        import soundfile as sf
        stream = io.BytesIO()
        sf.write(stream, samples, rate, format="WAV", subtype="FLOAT")
        return stream.getvalue()

    def test_embedded_bytes_precede_stale_path_stereo_downmix_and_real_resampling(self):
        import numpy as np
        import torchaudio
        torch = self.torch
        samples = np.column_stack([np.linspace(-.5, .5, 16000, dtype=np.float32), np.ones(16000, dtype=np.float32) * .25])
        waveform, audit = runtime.decode_audio({"bytes": self.wav(samples, 16000), "path": "/stale/missing.wav"}, Path("."))
        expected = torchaudio.functional.resample(torch.from_numpy(samples.T.copy()).mean(0, keepdim=True), 16000, 32000)[0]
        self.assertTrue(torch.equal(waveform, expected))
        self.assertEqual(audit["audio_original_sample_rate"], 16000)
        self.assertEqual(audit["audio_original_channels"], 2)
        self.assertEqual(audit["audio_right_pad_samples"], 608000)
        self.assertEqual(audit["audio_source"], "parquet_embedded_bytes")

    def test_fixed20_actual_crop_before_split_and_right_padding(self):
        torch = self.torch
        long = torch.arange(864000, dtype=torch.float32)
        result = self.fixed20.fixed_len(long, 640000)
        self.assertTrue(torch.equal(result[:320000], long[112000:432000]))
        self.assertTrue(torch.equal(result[320000:], long[432000:752000]))
        short = self.fixed20.fixed_len(torch.ones(16000), 640000)
        self.assertEqual(short.numel(), 640000)
        self.assertEqual(int(short[16000:].count_nonzero()), 0)
        self.assertEqual(int(short[320000:].count_nonzero()), 0)

    def test_generation_preserves_uppercase_and_actual_greedy_decoder_eos(self):
        torch = self.torch
        calls = []
        class LM(torch.nn.Module):
            def forward(self, **kwargs):
                calls.append(kwargs)
                length = kwargs["inputs_embeds"].shape[1]
                logits = torch.zeros((1, length, 8))
                logits[:, -1, 3 if length == 383 else 0] = 10
                return SimpleNamespace(logits=logits)
            def get_input_embeddings(self):
                return lambda ids: torch.zeros((*ids.shape, 576))
        model = SimpleNamespace(caption_decoder=SimpleNamespace(lm=LM()), generate_prefix_inference=Mock(
            return_value=(torch.zeros((1, 383, 576)), None, None)))
        tokenizer = SimpleNamespace(encode=lambda *a, **k: list(range(280)), decode=Mock(return_value="A<|endoftext|>"), eos_token_id=0)
        text_encoder = SimpleNamespace(tokenize=Mock(return_value={"input_ids": torch.ones((1, 256), dtype=torch.long)}))
        loaded = (model, tokenizer, torch.device("cpu"), text_encoder, self.fixed20, {}, {})
        text = dataset.prompt(record())
        result = runtime.generate(loaded, {"prompt": text, "waveform": torch.ones(16000), "audio_audit": {}}, variant="base")
        text_encoder.tokenize.assert_called_once_with(text, 256)
        self.assertEqual(result["generated_token_ids"], [3, 0])
        self.assertEqual(result["generated_text"], "A<|endoftext|>")
        self.assertTrue(result["prompt_truncated"])
        self.assertEqual([c["inputs_embeds"].shape[1] for c in calls], [383, 384])
        self.assertTrue(all(c["attention_mask"] is None and c["use_cache"] is False for c in calls))
        tokenizer.decode.assert_called_once_with([3, 0], skip_special_tokens=False, clean_up_tokenization_spaces=False)


if __name__ == "__main__":
    unittest.main()
