"""Dependency-light contract checks for the isolated fixed-260 MMAU route."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code" / "RSmol" / "scripts"
EVALUATOR = SCRIPTS / "evaluate_mmau_test_mini_audio_5_10x2_5_mesh_mellow_silence_slot.py"
RUNTIME = SCRIPTS / "evaluate_mmau_test_mini_audio_5_10x2_5_mesh_mellow_silence_slot.sh"
SUBMIT = ROOT / "code" / "RSmol" / "run_mmau_test_mini_audio_5_10x2_5_mesh_mellow_silence_slot_5090.sh"


def load_module():
    spec = importlib.util.spec_from_file_location("silence_slot_mmau_static", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SilenceSlotMMAUStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()
        cls.source = EVALUATOR.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_route_is_fixed260_and_uses_exact_checkpoint(self):
        self.assertEqual(self.module.CONFIG_FILENAME, "audio_mesh_fixed260_silence_slot_config.json")
        self.assertEqual(self.module.CONTRACT, "component_partitions6_rank_ram_fixed260_runtime_silence_second_slot_answer_eos_v2")
        self.assertEqual(self.module.EXPECTED_PREFIX_TOKENS, {"single": 260, "dual": 260})
        self.assertIn("audio_5_10x2_5_mesh_mellow_silence_slot/partition_formal_answer_eos_v2_10epochs_20260921", str(self.module.DEFAULT_CHECKPOINT).replace("\\", "/"))

    def test_adapter_reuses_official_pipeline_but_not_compact_loader(self):
        for marker in ("import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as official", "official.run(", "load_runtime_model=_load_runtime_model", "run_model_generation=_run_model_generation"):
            self.assertIn(marker, self.source)
        for marker in ("def _audit_checkpoint", "def _load_runtime_model", "def _build_fixed260_silence_prefix", "runtime_zero_wave", "AudioMeshSilenceSlotModel"):
            self.assertIn(marker, self.source)
        self.assertNotIn("_validate_checkpoint_contract(args)", self.source)
        self.assertNotIn("compact_single_audio_prefix=True", self.source)

    def test_generation_contract_and_submission_isolation(self):
        for marker in ("max_new_tokens", "--max-new-tokens 32", "--run-official-evaluation", "pdgpu-5090"):
            self.assertIn(marker, self.source + self.submit)
        self.assertIn("audio_5_10x2_5_mesh_mellow_silence_slot/mmau_test_mini_checkpoint_037810_fixed260_runtime_silence_v1", self.submit)
        self.assertIn("evaluate_mmau_test_mini_audio_5_10x2_5_mesh_mellow_silence_slot.sh", self.submit)
        self.assertTrue(RUNTIME.is_file())


if __name__ == "__main__":
    unittest.main()
