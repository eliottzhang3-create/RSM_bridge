"""CPU-only regression coverage for rescoring existing raw generations."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "code/RSmol/scripts/rescore_mmau_mellow_author_reply.py"
spec = importlib.util.spec_from_file_location("rescore_mmau_mellow_author_reply", SOURCE)
assert spec is not None and spec.loader is not None
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def raw_row(index: int, text: str = "b) second") -> dict:
    official = {
        "id": f"sample-{index}", "question": "Which sound?",
        "choices": ["first", "Second"], "answer": "SECOND",
        "task": ("sound", "music", "speech")[index % 3],
        "difficulty": ("easy", "medium", "hard")[index % 3],
    }
    return {
        "status": "generated", "row_index": index, "id": official["id"],
        "question": official["question"], "choices": official["choices"],
        "official_record": official, "generated_text": text,
        "model_output": "a) deliberately different processed field",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class RescoreRawGenerationsTest(unittest.TestCase):
    def test_full_cli_runs_without_site_packages_and_preserves_raw_outputs(self):
        outputs = [
            "B) unrelated body<|endoftext|>", " b) second", "\nb) second",
            "answer is b) second", "b", "b<|endoftext|>", "(b) second", "b) one)two",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "raw_generations.jsonl"
            rows = [raw_row(index, outputs[index % 8]) for index in range(1000)]
            write_jsonl(path, rows[::-1])
            original = path.read_bytes()
            official_result = root / "official_evaluation.txt"
            official_result.write_text("existing official score", encoding="utf-8")
            process = subprocess.run(
                [sys.executable, "-S", str(SOURCE), str(path), "--expected-rows", "1000"],
                capture_output=True, text=True, check=True,
            )
            report = json.loads(process.stdout)
            self.assertEqual(report["score"], {
                "correct": 375, "total": 1000, "accuracy_percent": 37.5,
            })
            self.assertEqual(report["source_predictions_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(report["input_audit"]["unique_ids"], 1000)
            self.assertFalse(report["official_mmau_scorer"])
            details = json.loads((root / "choice_label_rescore/mellow_author_reply_evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual([row["id"] for row in details["rows"]], [row["id"] for row in rows])
            self.assertEqual([row["prediction"] for row in details["rows"]], [row["generated_text"] for row in rows])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(official_result.read_text(encoding="utf-8"), "existing official score")
            for section in ("task", "difficulty"):
                self.assertEqual(sum(item["total"] for item in report[section].values()), 1000)

    def test_jsonl_defaults_to_full_denominator_and_allows_explicit_five(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw_generations.jsonl"
            write_jsonl(path, [raw_row(index) for index in range(5)])
            args = scorer.parse_args([str(path)])
            with self.assertRaisesRegex(ValueError, "expected=1000 actual=5"):
                scorer.run(args)
            self.assertFalse(args.output_dir.exists())
            report = scorer.run(scorer.parse_args([str(path), "--expected-rows", "5"]))
            self.assertEqual(report["score"]["correct"], 5)

    def test_duplicate_indices_and_ids_are_rejected(self):
        for field, message in (("row_index", "duplicate row_index"), ("id", "duplicate id")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "raw_generations.jsonl"
                rows = [raw_row(index) for index in range(5)]
                rows[1][field] = rows[0][field]
                write_jsonl(path, rows)
                with self.assertRaisesRegex(ValueError, message):
                    scorer.load_raw_predictions(path, 5)

    def test_truncated_jsonl_fails_with_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw_generations.jsonl"
            write_jsonl(path, [raw_row(0)])
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"row_index": 1')
            with self.assertRaisesRegex(ValueError, "line 2: invalid JSON"):
                scorer.load_raw_predictions(path, 5)

    def test_incorrect_metadata_or_prediction_fields_are_rejected(self):
        for case in ("metadata_id", "choices", "generated_text", "status"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                rows = [raw_row(index) for index in range(5)]
                if case == "metadata_id":
                    rows[0]["official_record"]["id"] = "different-id"
                elif case == "choices":
                    rows[0]["choices"] = ["swapped", "order"]
                else:
                    rows[0][case] = None
                path = Path(directory) / "raw_generations.jsonl"
                write_jsonl(path, rows)
                with self.assertRaises(ValueError):
                    scorer.load_raw_predictions(path, 5)

    def test_row_index_gaps_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw_generations.jsonl"
            write_jsonl(path, [raw_row(index) for index in range(1, 6)])
            with self.assertRaisesRegex(ValueError, "row_index coverage differs"):
                scorer.load_raw_predictions(path, 5)

    def test_unmatched_gold_and_empty_generation_keep_full_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw_generations.jsonl"
            rows = [raw_row(index) for index in range(5)]
            rows[0]["official_record"]["answer"] = "not an exact choice"
            rows[1]["generated_text"] = ""
            write_jsonl(path, rows)
            report = scorer.run(scorer.parse_args([str(path), "--expected-rows", "5"]))
            self.assertEqual(report["score"]["total"], 5)
            self.assertEqual(report["score"]["correct"], 3)
            self.assertEqual(report["record_errors"]["total"], 1)

    def test_existing_prediction_json_interface_still_works(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "predictions.json"
            predictions = [{**raw_row(index)["official_record"], "model_output": "b) second"}
                           for index in range(5)]
            path.write_text(json.dumps(predictions), encoding="utf-8")
            args = scorer.parse_args([str(path), "--expected-rows", "5"])
            self.assertEqual(args.output_dir, root)
            report = scorer.run(args)
            self.assertEqual(report["score"]["correct"], 5)
            self.assertEqual(report["input_format"], "predictions_json")


if __name__ == "__main__":
    unittest.main()
