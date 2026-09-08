"""Dependency-light contracts for the isolated MeSH audio route."""
from __future__ import annotations

import unittest
import json
import tempfile
import wave
import struct
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "code" / "RSmol" / "audio_5_10x2_5_mesh_mellow"
STAGE3 = ROOT / "code" / "RSmol" / "scripts" / "audit_audio_stage3_5_10x2_5_mesh_mellow.py"
STAGE4 = ROOT / "code" / "RSmol" / "scripts" / "audit_audio_stage4_5_10x2_5_mesh_mellow.py"
TRAIN = ROOT / "code" / "RSmol" / "scripts" / "train_audio_5_10x2_5_mesh_mellow_ddp.py"
SUBMIT_WRAPPERS = tuple(ROOT / "code" / "RSmol" / name for name in (
    "run_audio_stage4_5_10x2_5_mesh_mellow_5090.sh",
    "run_audio_stage5_5_10x2_5_mesh_mellow_5090.sh",
    "run_audio_stage7_5_10x2_5_mesh_mellow_5090.sh",
    "run_audio_formal_5_10x2_5_mesh_mellow_5090.sh",
))


class AudioMeshStaticContractTest(unittest.TestCase):
    def test_route_is_isolated(self) -> None:
        for path in (PKG / "model.py", PKG / "data.py", PKG / "stage3.py", STAGE3, STAGE4, TRAIN):
            self.assertTrue(path.is_file(), path)
        combined = "\n".join(path.read_text(encoding="utf-8") for path in (PKG / "model.py", PKG / "data.py", PKG / "stage3.py", STAGE3, STAGE4, TRAIN))
        self.assertIn("5_10x2_5", combined)
        self.assertIn("HTSAT", combined)
        self.assertNotIn("recursive_model_5_10_5", combined)

    def test_answer_only_loss_contract_is_explicit(self) -> None:
        text = (PKG / "model.py").read_text(encoding="utf-8")
        self.assertIn("torch.full", text)
        self.assertIn("-100", text)
        self.assertIn("non-answer prefix participates in loss", text)
        self.assertIn("answer supervision count mismatch", text)
        self.assertIn("separator = _find_embedding", text)
        self.assertNotIn("self._find_embedding", text)
        self.assertIn("prefix_mask = torch.ones", text)
        self.assertIn("attention mask/embedding shape mismatch", text)

    def test_audio_contract_and_reuse(self) -> None:
        data = (PKG / "data.py").read_text(encoding="utf-8")
        model = (PKG / "model.py").read_text(encoding="utf-8")
        self.assertIn("sample_rate: int = 32000", data)
        self.assertIn("target_samples = sample_rate * seconds", data)
        self.assertIn("waveform[..., :target_samples]", data)
        self.assertIn("audio2 = _path(row, False)", data)
        self.assertIn("audio2 = audio2 or audio1", data)
        self.assertIn("audio2_reused_mask", data + model)

    def test_formal_schedule(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        self.assertIn('default=3', text)
        self.assertIn('math.ceil(total_steps * 0.05)', text)
        self.assertIn('default=4', text)
        self.assertIn('default=1', text)
        self.assertIn('default=1e-3', text)
        self.assertIn('broadcast_buffers=False', text)
        self.assertIn('"STAGE7"', text)
        self.assertIn("_actual_resume_audit", text)
        self.assertIn("rng_states_by_rank", text)
        self.assertIn("batch_in_epoch", text)
        self.assertIn("args.save_every", text)
        self.assertIn("learning_rate", text)
        self.assertIn("answer_only_labels", text)
        self.assertIn("progress_percent", text)
        self.assertIn("step={optimizer_step}/{max_steps}", text)
        self.assertIn('"--num-workers"', text)

    def test_gpu_stages_have_submission_wrappers(self) -> None:
        for wrapper in SUBMIT_WRAPPERS:
            self.assertTrue(wrapper.is_file(), wrapper)
            text = wrapper.read_text(encoding="utf-8")
            self.assertIn("vc submit", text)
            self.assertIn("-g 1" if "stage4" in wrapper.name else "-g 8", text)
            self.assertIn("docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1", text)

    def test_stage3_fast_contracts(self) -> None:
        text = (PKG / "stage3.py").read_text(encoding="utf-8")
        self.assertIn("cross_split_overlap", text)
        self.assertIn("MAX_CONTEXT_LENGTH = 768", text)
        self.assertIn("dynamic_longest_in_batch", text)
        self.assertNotIn("_inspect_audio", text)
        self.assertNotIn("AutoTokenizer", text)

    def test_stage3_runtime_reports_paths_and_overlap(self) -> None:
        source = PKG / "stage3.py"
        spec = importlib.util.spec_from_file_location("audio_mesh_stage3_test", source)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.wav"
            with wave.open(str(valid), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(struct.pack("<h", 0) * 16000)
            manifests = {}
            for split, path in (("train", valid), ("val", valid), ("test", valid)):
                manifest = root / f"{split}.jsonl"
                manifest.write_text(json.dumps({"audio1_path": str(path), "audio2_path": "", "question": "q", "answer": "a"}) + "\n", encoding="utf-8")
                manifests[split] = manifest
            args = type("Args", (), {"train_manifest": manifests["train"], "val_manifest": manifests["val"], "test_manifest": manifests["test"], "tokenizer_path": None, "max_prompt_tokens": 129, "max_answer_tokens": 250})()
            report = module.audit(args)
            self.assertGreater(report["cross_split_overlap"]["train__val"]["count"], 0)
            self.assertEqual(report["status"], "PASS_WITH_WARNINGS")


if __name__ == "__main__":
    unittest.main()
