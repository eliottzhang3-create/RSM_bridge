from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MCQ = ROOT / "code" / "RSmol" / "mellow_official_training_c8204d8_adamw_cosine" / "scripts" / "rsmol" / "prepare_reasonaqa_mcq_manifest.py"
CKPT = ROOT / "code" / "RSmol" / "mellow_official_training_c8204d8_adamw_cosine" / "scripts" / "rsmol" / "audit_reasonaqa_mellow_init_checkpoint.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MellowOfficialMcqAuditStaticTests(unittest.TestCase):
    def test_mcq_manifest_filters_exact_subtypes_and_writes_audit(self):
        module = load_module(MCQ, "mcq_manifest_test")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "train.json"
            output = root / "mcq" / "reasonaqa_mcq_train.json"
            rows = [
                {"subtype": "AudioCaps-MCQ.json", "filepath1": "a.wav", "filepath2": "", "input": "q", "answer": "a", "keep": 1},
                {"subtype": "Clotho-MCQ.json", "filepath1": "b.wav", "filepath2": "", "input": "q", "answer": "b", "keep": 2},
                {"subtype": "AudioCaps.json", "filepath1": "c.wav", "filepath2": "", "input": "q", "answer": "c", "keep": 3},
            ]
            source.write_text(json.dumps(rows), encoding="utf-8")
            report = module.build_manifest(source, output)
            selected = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([row["keep"] for row in selected], [1, 2])
            self.assertEqual(report["selected_rows"], 2)
            self.assertEqual(report["selected_subtype_distribution"], {"AudioCaps-MCQ.json": 1, "Clotho-MCQ.json": 1})
            self.assertTrue((output.with_suffix(output.suffix + ".audit.json")).is_file())
            self.assertFalse(report["gpu_required"])

    def test_checkpoint_audit_requires_state_dict_and_reports_full_contract(self):
        text = CKPT.read_text(encoding="utf-8")
        for marker in (
            "state_dict",
            "schema_version",
            "missing_full_checkpoint_fields",
            "initialization_semantics",
            "load state_dict only; create fresh optimizer/scheduler/RNG",
            "gpu_required",
        ):
            self.assertIn(marker, text)

    def test_tools_are_cpu_only_and_do_not_submit_gpu_jobs(self):
        for path in (MCQ, CKPT):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("vc submit", text)
            self.assertNotIn("torchrun", text)


if __name__ == "__main__":
    unittest.main()
