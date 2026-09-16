"""Dependency-light contracts for the submitted storage discovery job."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "code" / "RSmol" / "scripts" / "probe_audio_storage.py"
INNER = ROOT / "code" / "RSmol" / "scripts" / "probe_audio_storage_5090.sh"
SUBMIT = ROOT / "code" / "RSmol" / "run_audio_storage_probe_5090.sh"


class AudioStorageProbeStaticTest(unittest.TestCase):
    def test_entrypoints_and_formal_resource_match(self) -> None:
        for path in (PROBE, INNER, SUBMIT):
            self.assertTrue(path.is_file(), path)
        submit = SUBMIT.read_text(encoding="utf-8")
        for marker in (
            "vc submit",
            "pdgpu-5090",
            "docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1",
            "-c 32 -m 256G -g 8 -n 1",
            "probe_audio_storage_5090.sh",
        ):
            self.assertIn(marker, submit)

    def test_probe_is_read_only_and_classifies_unsafe_targets(self) -> None:
        source = PROBE.read_text(encoding="utf-8")
        ast.parse(source)
        for marker in (
            '"read_only_probe": True',
            '"memory_tmpfs"',
            '"container_overlay"',
            '"network_filesystem"',
            '"local_block_device"',
            '"eligible_local_stage_paths"',
            '"minimum_local_free_gib"',
            '"expected_cache_gib"',
            '"/proc/sys/vm/max_map_count"',
            '"/sys/fs/cgroup/memory.max"',
            '"raw_diagnostics.txt"',
            '"storage_probe_report.json"',
        ):
            self.assertIn(marker, source)
        for forbidden in ("fio", "dd if=", "fallocate", "truncate(", "os.remove", "unlink("):
            self.assertNotIn(forbidden, source)

    def test_persistent_unique_output_and_threshold(self) -> None:
        inner = INNER.read_text(encoding="utf-8")
        self.assertIn("RSMOL_STORAGE_PROBE_RUN_ID", inner)
        self.assertIn("/outputs/RSmol/audio_storage_probe/", inner)
        self.assertIn("--expected-cache-gib 115.30", inner)
        self.assertIn("--minimum-local-free-gib 150", inner)
        self.assertIn('"$@"', inner)


if __name__ == "__main__":
    unittest.main()
