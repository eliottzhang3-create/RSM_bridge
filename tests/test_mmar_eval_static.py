"""Dependency-light contracts for the official MMAR evaluator."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code" / "RSmol" / "scripts"
SOURCE = SCRIPTS / "evaluate_mmar_5_10x2_5_mesh_mellow.py"
INNER_SH = SCRIPTS / "evaluate_mmar_5_10x2_5_mesh_mellow.sh"
SUBMIT_SH = ROOT / "code" / "RSmol" / "run_mmar_5_10x2_5_mesh_mellow_5090.sh"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("mmar_audio_mesh_evaluator", SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


class MMAREvaluatorStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_loads_official_json_and_jsonl_shapes(self) -> None:
        records = [{"id": "0"}, {"id": "1"}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "meta.json"
            jsonl_path = root / "meta.jsonl"
            json_path.write_text(json.dumps(records), encoding="utf-8")
            jsonl_path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
            self.assertEqual(self.module.load_mmar_records(json_path), records)
            self.assertEqual(self.module.load_mmar_records(jsonl_path), records)

    def test_audio_path_is_confined_to_extracted_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = (root / "audio" / "clip.wav").resolve()
            self.assertEqual(self.module.resolve_mmar_audio_path(root, "./audio/clip.wav"), expected)
            with self.assertRaises(ValueError):
                self.module.resolve_mmar_audio_path(root, "../clip.wav")
            with self.assertRaises(ValueError):
                self.module.resolve_mmar_audio_path(root, expected)

    def test_prediction_file_keeps_official_order_and_denominator(self) -> None:
        official = [
            {"id": "0", "answer": "a", "choices": ["a", "b"]},
            {"id": "1", "answer": "b", "choices": ["a", "b"]},
        ]
        state = [
            {"id": "1", "row_index": 1, "status": "parsed", "answer_prediction": "b"},
        ]
        predictions = self.module.materialize_predictions(state, official, full=True)
        self.assertEqual([item["id"] for item in predictions], ["0", "1"])
        self.assertEqual([item["answer_prediction"] for item in predictions], ["", "b"])

    def test_eos_v2_choice_output_uses_strict_common_parser(self) -> None:
        result = self.module.common.parse_model_output(
            "c) It is plausible",
            ["It is impossible", "It is unlikely", "It is plausible", "It is certain"],
        )
        self.assertEqual(result["selected_option"], "It is plausible")
        self.assertEqual(result["parse_method"], "leading_label")
        self.assertEqual(
            self.module.common.parse_model_output(
                "It is plausible because the event can occur",
                ["It is impossible", "It is unlikely", "It is plausible", "It is certain"],
            )["selected_option"],
            "",
        )

    def test_smoke_directory_can_resume_as_full(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = dict(
                output_dir=root,
                checkpoint=Path("/checkpoint"),
                metadata_json=Path("/data/MMAR-meta.json"),
                audio_root=Path("/data/mmar-audio"),
                evaluation_script=Path("/data/code/evaluation.py"),
                htsat_checkpoint=Path("/models/htsat.ckpt"),
                mellow_root=Path("/code/mellow"),
                max_prompt_tokens=self.module.MMAR_MAX_PROMPT_TOKENS,
                max_new_tokens=self.module.common.DEFAULT_MAX_NEW_TOKENS,
            )
            self.module._ensure_output_dir(SimpleNamespace(mode="smoke", **common))
            self.module._ensure_output_dir(SimpleNamespace(mode="full", **common))
            config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["mode"], "full")
            self.assertEqual(config["mode_history"], ["smoke", "full"])

    def test_dataset_override_derives_all_official_artifact_paths(self) -> None:
        args = self.module.parse_args([
            "--dataset-dir", "/custom/MMAR",
            "--output-dir", "/tmp/mmar-output",
        ])
        self.assertEqual(args.metadata_json, Path("/custom/MMAR/MMAR-meta.json"))
        self.assertEqual(args.audio_root, Path("/custom/MMAR/mmar-audio"))
        self.assertEqual(args.evaluation_script, Path("/custom/MMAR/code/evaluation.py"))

    def test_source_and_wrappers_lock_official_contract(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        for marker in (
            "answer_prediction",
            "MMAR_CORE_CANONICAL_SHA256",
            "MMAR_EVALUATION_SHA256",
            "first 10 seconds",
            "compact single-audio prefix",
            "official MMAR order",
        ):
            self.assertIn(marker, source)
        self.assertIn("checkpoint-037810", self.module.DEFAULT_CHECKPOINT)
        common_source = Path(self.module.common.__file__).read_text(encoding="utf-8")
        self.assertIn("skipped.jsonl", common_source)
        self.assertIn("progress.jsonl", common_source)
        self.assertIn("conda activate rsmol", INNER_SH.read_text(encoding="utf-8"))
        submit = SUBMIT_SH.read_text(encoding="utf-8")
        self.assertIn("vc submit", submit)
        self.assertIn("-p pdgpu-5090", submit)
        self.assertIn("-g 1", submit)
        self.assertIn("MMAR-meta.json", submit)
        self.assertIn("code/evaluation.py", submit)
        self.assertIn("--max-prompt-tokens 622", submit)


if __name__ == "__main__":
    unittest.main()
