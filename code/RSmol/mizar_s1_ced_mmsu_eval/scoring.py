"""Execute the untouched official scorer, retaining its actual denominator."""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import runpy
import subprocess
import sys

from .dataset import EXPECTED_ROWS, METADATA
from .storage import file_sha, text_sha

SCORER_LF_SHA256 = "b6d5d35d2755244019a78e601495ea74e722681bf6f7e1fd0179bab7cabca3eb"


def audit_scorer(path):
    digest = text_sha(path)
    if digest != SCORER_LF_SHA256:
        raise ValueError(f"MMSU official scorer source differs from inspected release: {path}: {digest}")
    return {"path": str(path), "sha256": file_sha(path), "lf_sha256": digest,
            "source": "https://github.com/dingdongwang/MMSU/blob/main/mmsu_evaluation.py"}


def write_predictions(path, saved_rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for saved in saved_rows:
            row = {key: saved["official_record"][key] for key in METADATA}
            row["response"] = saved["generation"]["generated_text"]
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def run_scorer(script, predictions, output_dir):
    audit = audit_scorer(script)
    with predictions.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    if (len(records) != EXPECTED_ROWS or len({str(row["id"]) for row in records}) != EXPECTED_ROWS
            or any(not isinstance(row.get("response"), str) for row in records)):
        raise ValueError("refusing official scoring without 5000 unique text predictions")
    command = [sys.executable, str(script), str(predictions)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", check=False,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    (output_dir / "official_evaluation.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8")
    result = {"execution_status": "PASS" if completed.returncode == 0 else "FAILED",
              "returncode": completed.returncode, "command": command, "scorer": audit,
              "input_rows": len(records), "status": "FAILED"}
    if completed.returncode != 0:
        return result
    # Call the very same pinned official function for unrounded values and counts.
    # No private reimplementation or response rewriting is used.
    with contextlib.redirect_stdout(io.StringIO()):
        functions = runpy.run_path(str(script), run_name="mmsu_official_scorer")
        groups, categories, accuracy, count = functions["calculate_accuracy_per_task_and_category"](records)
    matches = re.findall(r"^Total count: (\d+)\s*$", completed.stdout, re.MULTILINE)
    scores = re.findall(r"^Overall Accuracy: ([0-9.]+)\s*$", completed.stdout, re.MULTILINE)
    agrees = (matches == [str(count)] and scores == [f"{accuracy:.4f}"])
    result.update(actual_denominator=count, excluded_rows=EXPECTED_ROWS - count,
                  overall_accuracy=accuracy, category_accuracy=dict(categories),
                  category_subcategory_accuracy={key: dict(value) for key, value in groups.items()},
                  cli_function_agreement=agrees,
                  status="PASS" if agrees and count == EXPECTED_ROWS else "FAILED")
    return result
