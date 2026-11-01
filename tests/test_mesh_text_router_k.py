"""Dependency-light math, input, recorded-data and offline workflow checks."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/RSmol/scripts/calibrate_mesh_text_router_k.py"
spec = importlib.util.spec_from_file_location("mesh_k", SCRIPT)
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


def config():
    # These values match the existing converter and CED text loader; the mesh
    # checkpoint contract deliberately differs from model_architecture_contract.
    return {"model_type": "llama", "hidden_size": 576, "num_hidden_layers": 30,
            "recursive_layer_count": 20, "recursive_loops": 2, "mesh_memory_slots": 5,
            "mesh_router_count": 6, "mesh_architecture_contract": "logical_30_physical_20_5_10x2_5",
            "model_architecture_contract": "logical_30_physical_20_5_10x2_5_mesh",
            "logical_to_physical": list(cal.SCHEDULE)}


def records():
    result = []
    for sample in range(10):
        for position in range(4):
            for index, name in enumerate(cal.ROUTERS):
                # Nonuniform probabilities, different router scales; one rare
                # low-logsumexp input must control the common upper bound.
                shift = -30 if (sample, position, index) == (9, 3, 5) else index - sample / 10
                logits = [shift + value for value in (-2, -1, 0, 1, 2)]
                lse = cal.logsumexp(logits)
                row = {"sample_id": f"p{sample:02d}", "generation_step": max(0, position - 2),
                       "forward_kind": "generation", "sequence_position": position,
                       "token_region": "prompt" if position < 2 else "generated", "token_id": position,
                       "token_text": "a token, with \"quotes\"\nand newline", "router": name, "logsumexp": lse}
                row.update({f"logit_{i}": value for i, value in enumerate(logits)})
                row.update({f"prob_{i}": math.exp(value - lse) for i, value in enumerate(logits)})
                result.append(row)
    return result


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cal.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class RouterCalibrationTests(unittest.TestCase):
    def test_formula_matches_explicit_seven_slot_softmax_and_preserves_old_ratios(self):
        logits, k = [-4., -2., .2, 2., 3.], 6.
        seven = [math.exp(z) for z in (*logits, -k, -k)]
        seven = [p / sum(seven) for p in seven]
        old = [math.exp(z - cal.logsumexp(logits)) for z in logits]
        q = cal.new_mass(k, cal.logsumexp(logits))
        self.assertAlmostEqual(q, sum(seven[5:]), places=15)
        self.assertEqual(seven[5], seven[6])
        for p, actual in zip(old, seven[:5]):
            self.assertAlmostEqual((1 - q) * p, actual, places=15)

    def test_probability_alone_cannot_determine_k(self):
        a, b = [0.] * 5, [10.] * 5
        for left, right in zip(a, b):
            self.assertAlmostEqual(math.exp(left - cal.logsumexp(a)), math.exp(right - cal.logsumexp(b)), places=14)
        self.assertNotEqual(cal.new_mass(5, cal.logsumexp(a)), cal.new_mass(5, cal.logsumexp(b)))

    def test_boundary_equality_and_margin_strictly_bound_both_new_slots(self):
        for lse in (-1000., -10., 0., 1000.):
            for cap in (.01, .005):
                boundary = cal.k_boundary(lse, cap)
                self.assertAlmostEqual(cal.new_mass(boundary, lse), cap, places=14)
                self.assertLess(cal.new_mass(boundary + 1, lse), cap)

    def test_stable_extreme_logits_and_sigmoid(self):
        self.assertTrue(math.isfinite(cal.logsumexp([10000.] * 5)))
        self.assertAlmostEqual(cal.logsumexp([-10000.] * 5), -10000 + math.log(5))
        self.assertEqual(cal.new_mass(10000, 10000), 0)
        self.assertEqual(cal.new_mass(-10000, -10000), 1)
        for bad in ([math.nan] * 5, [math.inf] * 5, [0.] * 4):
            with self.assertRaises(ValueError): cal.logsumexp(bad)

    def test_common_k_uses_worst_token_instead_of_means_and_reports_all_six(self):
        rows = records()
        report = cal.calibrate(rows)
        k = report["recommended_common_k"]
        self.assertGreaterEqual(k, report["common_k_boundary_at_cap"] + 1 - 1e-12)
        self.assertLess(k, report["common_k_boundary_at_cap"] + 1.1 + 1e-12)
        self.assertEqual(set(report["routers"]), set(cal.ROUTERS))
        for row in rows:
            self.assertLess(cal.new_mass(k, row["logsumexp"]), .005)
        self.assertEqual(report["routers"]["read_1"]["worst_position"]["sample_id"], "p09")
        self.assertLess(report["worst_combined_probability_at_common_k"], .005 * .38)
        self.assertNotEqual(report["routers"]["write_pre"]["combined_new_slot_probability_at_common_k"]["max"],
                            report["routers"]["read_1"]["combined_new_slot_probability_at_common_k"]["max"])

    def test_prompt_and_generated_are_separate_and_token_counts_are_not_repeated(self):
        report = cal.calibrate(records())
        self.assertEqual(report["record_count"], 10 * 4 * 6)
        for item in report["routers"].values():
            for region in ("prompt", "generated"):
                self.assertEqual(item["by_region"][region]["count"], 20)

    def test_records_reject_duplicate_missing_router_bad_prob_or_bad_lse(self):
        cases = [records() + records()[:1], records()[1:]]
        for field, value in (("logsumexp", 999.), ("prob_0", .99), ("token_region", "audio")):
            rows = records()
            rows[0][field] = value
            cases.append(rows)
        for rows in cases:
            with self.assertRaises(ValueError): cal.calibrate(rows)

    def test_records_reject_incomplete_samples_missing_positions_and_identity_conflicts(self):
        cases = [[row for row in records() if row["sample_id"] != "p09"],
                 [row for row in records() if row["sequence_position"] != 1]]
        changed = records()
        changed[0]["token_id"] = 99
        cases.append(changed)
        for rows in cases:
            with self.assertRaises(ValueError): cal.validate_records(rows)

    def test_csv_roundtrip_with_quotes_and_multiline_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.csv"
            write_csv(path, records())
            self.assertEqual(cal.load_records(path), records())

    def test_template_exactly_ten_utf8_objects_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sub/prompts.jsonl"
            cal.write_template(path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 10)
            with self.assertRaises(FileExistsError): cal.write_template(path)
            with self.assertRaisesRegex(ValueError, "fill the template"): cal.load_prompts(path)
            for row in rows: row["prompt"] = "  保留空白\nwith \"quotes\"  "
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8-sig")
            self.assertEqual(cal.load_prompts(path), rows)

    def test_prompts_reject_duplicate_id_json_key_count_and_typo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            good = [{"id": str(i), "prompt": "text"} for i in range(10)]
            duplicate = copy.deepcopy(good)
            duplicate[9]["id"] = "0"
            typo = copy.deepcopy(good)
            typo[0]["promtp"] = typo[0].pop("prompt")
            for rows in (good[:9], duplicate, typo):
                path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
                with self.assertRaises(ValueError): cal.load_prompts(path)
            path.write_text('{"id":"a","prompt":"one","prompt":"two"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"): cal.load_prompts(path)

    def test_config_matches_existing_loader_and_rejects_wrong_route(self):
        cal.validate_config(config())
        for key, value in (("mesh_memory_slots", 7), ("mesh_router_count", 8), ("hidden_size", 384),
                           ("mesh_architecture_contract", cal.TEXT_ARCHITECTURE), ("recursive_loops", 4),
                           ("logical_to_physical", list(range(30)))):
            bad = config(); bad[key] = value
            with self.assertRaises(ValueError): cal.validate_config(bad)

    def test_check_inputs_does_not_import_torch_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "weights"; model.mkdir()
            (model / "config.json").write_text(json.dumps(config()), encoding="utf-8")
            for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json"):
                (model / name).write_text("{}", encoding="utf-8")
            prompts = root / "prompts.jsonl"
            prompts.write_text("\n".join(json.dumps({"id": str(i), "prompt": "text"}) for i in range(10)), encoding="utf-8")
            out = root / "output"
            with patch.dict(sys.modules, {"torch": None, "transformers": None}):
                result = cal.main(["--prompts-jsonl", str(prompts), "--text-model-dir", str(model),
                                   "--output-dir", str(out), "--check-inputs"])
            self.assertEqual(result, 0)
            self.assertFalse(out.exists())

    def test_offline_workflow_recomputes_without_models_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "router.csv"; out = root / "result"
            write_csv(source, records())
            args = ["--records-csv", str(source), "--output-dir", str(out), "--new-slot-mass-cap", "0.001"]
            with patch.dict(sys.modules, {"torch": None, "transformers": None}):
                self.assertEqual(cal.main(args), 0)
            report = json.loads((out / "calibration_report.json").read_text(encoding="utf-8"))
            self.assertLess(report["worst_combined_probability_at_common_k"], .001)
            self.assertEqual(report["sample_count"], 10)
            before = (out / "generation_report.json").read_bytes()
            self.assertEqual(cal.main(args), 1)
            self.assertEqual((out / "generation_report.json").read_bytes(), before)

    def test_incomplete_input_csv_records_failure_without_recommending_k(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "partial.csv"; out = root / "failed"
            write_csv(source, records()[1:])
            with patch("sys.stderr"):
                self.assertEqual(cal.main(["--records-csv", str(source), "--output-dir", str(out)]), 1)
            report = json.loads((out / "generation_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(json.loads((out / "run_config.json").read_text())["status"], "FAIL")
            self.assertFalse((out / "calibration_report.json").exists())

    def test_fixed_generation_horizon_and_invalid_mass_settings_are_rejected(self):
        for args in (("--max-new-tokens", "99"), ("--new-slot-mass-cap", "nan"),
                     ("--new-slot-mass-cap", "0"), ("--k-margin", "0"), ("--k-margin", "inf")):
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                cal.parse_args(["--prompts-jsonl", "test.jsonl", *args])

    def test_script_can_create_template_with_no_tensor_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "prompts.jsonl"
            result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--write-prompts-template", str(template)],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(template.read_text().splitlines()), 10)


if __name__ == "__main__":
    unittest.main()
