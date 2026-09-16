"""Dependency-light contracts for deterministic CPU waveform shard generation."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "RSmol" / "scripts" / "prepare_audio_waveform_shards.py"


class AudioWaveformShardStaticTest(unittest.TestCase):
    def test_script_parses_and_exposes_expected_contract(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        ast.parse(text)
        for marker in (
            "SAMPLES_PER_AUDIO = SAMPLE_RATE * SECONDS",
            "BYTES_PER_AUDIO = SAMPLES_PER_AUDIO * 4",
            "--num-shards",
            "--seed",
            "--workers",
            "random.Random(int(seed)).shuffle(shuffled)",
            "raw_fixed_waveform_shards_v1",
            '"index.jsonl"',
            '"dtype": "float32"',
            '"epoch_read_contract"',
            '"--resume"',
            '"BUILDING"',
            "os.replace(partial_path, path)",
            "refusing to overwrite existing shard output",
            '"build_error.json"',
            "resume build configuration/source inventory mismatch",
            "free-space-margin-gib",
        ):
            self.assertIn(marker, text)

    def test_source_roots_and_audio_contract_are_explicit(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for marker in (
            "DEFAULT_AUDIOCAPS_ROOT",
            "DEFAULT_CLOTHO_AQA_ROOT",
            "DEFAULT_CLOTHO_ROOT",
            "32000",
            "10",
            "float32",
            "row",
            "byte_offset",
            "sha256",
        ):
            self.assertIn(marker, text)

    def test_generation_does_not_create_one_file_per_audio(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('f"shard-{shard_id:05d}.bin"', text)
        self.assertNotIn("source_path + \".npy\"", text)
        self.assertNotIn("torch.save(waveform", text)

    def test_balanced_ranges_cover_every_audio_once(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_shard_bounds"
        )
        namespace = {"tuple": tuple}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SCRIPT), "exec"), namespace)
        bounds = [namespace["_shard_bounds"](96_724, 64, shard_id) for shard_id in range(64)]
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], 96_724)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(bounds, bounds[1:])))
        counts = [end - start for start, end in bounds]
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(sum(counts), 96_724)


if __name__ == "__main__":
    unittest.main()
