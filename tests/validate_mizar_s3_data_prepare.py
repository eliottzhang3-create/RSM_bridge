"""Run isolated S3 CPU checks and optionally inspect local reference metadata."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "RSmol"))
from mizar_s3_data_prepare.io_utils import iter_rows, sha256_file, write_json
from mizar_s3_data_prepare.mapping import candidate_mcq_key, mcq_key, norm, validate_candidate
from mizar_s3_data_prepare.prepare import OFFICIAL_SHA256, OFFICIAL_COUNTS


def references(source, avqa, s1):
    annotations = {r["id"]: r for r in iter_rows(avqa)}
    needed = {}
    counts, avqa_counts = Counter(), Counter()
    for row in iter_rows(source):
        validate_candidate(row)
        counts[row["source_dataset"]] += 1
        if row["source"] == "audiomcq":
            needed.setdefault(candidate_mcq_key(row), []).append(row)
            continue
        raw = annotations.get(int(row["source_sample_key"].rsplit("_", 1)[1]))
        if raw is None:
            avqa_counts["original_id_missing"] += 1
        elif ([norm(x) for x in raw["multi_choice"]] == [norm(x) for x in row["choices"]]
              and raw["answer"] == row["gold_option_index"]):
            avqa_counts["id_options_gold_agree"] += 1
        else:
            avqa_counts["options_or_gold_disagree"] += 1
    found = defaultdict(dict)
    for index, row in enumerate(iter_rows(s1)):
        if (row.get("source") != "audiomcq" or row.get("audio_contribution") != "strong"
                or row.get("stageE_variant") != "audiomcq_mcq"):
            continue
        key = mcq_key(row["source_dataset"], row["id"], row["choices"])
        if key in needed:
            found[key].setdefault(row["audio_path"], set()).add(row["gold_letter"])
        if (index + 1) % 200000 == 0:
            print(f"[reference metadata] S1 rows scanned={index+1:,}", flush=True)
    joins, gold_changes = Counter(), 0
    for key, rows in needed.items():
        entries = found.get(key, {})
        for row in rows:
            if len(entries) != 1:
                joins["missing_or_ambiguous_historical_identity"] += 1
                continue
            raw = next(iter(entries))
            if PurePosixPath(raw.replace("\\", "/")).name != PurePosixPath(row["filepath1"]).name:
                joins["recording_name_disagrees"] += 1
                continue
            joins["identity_options_recording_match"] += 1
            gold_changes += entries[raw] != {row["gold_letter"]}
    source_sha = sha256_file(source)
    if source_sha != OFFICIAL_SHA256 or counts != Counter(OFFICIAL_COUNTS):
        raise ValueError("local S3 reference does not match official release")
    return {"scope": "local reference metadata only; no production audio/TSV or filtered production count",
            "source_sha256": source_sha, "source_counts": dict(counts),
            "avqa_annotation_rows": len(annotations), "avqa_id_join": dict(avqa_counts),
            "audiomcq_reference_s1_join": dict(joins),
            "audiomcq_s3_gold_differs_from_reference_s1": gold_changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "audit_reports/mizar_s3_data_prepare_validation.json")
    parser.add_argument("--reference-s3", type=Path)
    parser.add_argument("--reference-avqa", type=Path)
    parser.add_argument("--reference-s1", type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    spec = importlib.util.spec_from_file_location("s3_checks", ROOT / "tests/test_mizar_s3_data_prepare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    print(stream.getvalue())
    folder = ROOT / "code/RSmol/mizar_s3_data_prepare"
    python_files = sorted(folder.glob("*.py")) + [Path(__file__), ROOT / "tests/test_mizar_s3_data_prepare.py"]
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 10))
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else "bash"
    subprocess.run([bash, "-n", str(ROOT / "code/RSmol/prepare_mizar_s3_data.sh")], check=True)
    report = {"status": "PASS_LOCAL_CPU" if result.wasSuccessful() and not result.skipped else "FAILED",
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "python": platform.python_version(),
              "python310_syntax": "PASS", "bash_syntax": "PASS", "test_output": stream.getvalue(),
              "production_audio_checked": False, "gpu_submitted": False,
              "source_files": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                               python_files + [ROOT / "code/RSmol/prepare_mizar_s3_data.sh"]}}
    if any((args.reference_s3, args.reference_avqa, args.reference_s1)):
        if not all((args.reference_s3, args.reference_avqa, args.reference_s1)):
            parser.error("all three reference metadata options are required together")
        report["reference_metadata"] = references(args.reference_s3, args.reference_avqa, args.reference_s1)
    report["elapsed_seconds"] = time.monotonic() - start
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.is_file():
        previous = json.loads(args.report.read_text(encoding="utf-8"))
        report["previous_runs"] = previous.get("previous_runs", []) + [
            {key: value for key, value in previous.items() if key != "previous_runs"}]
    write_json(args.report, report)
    print(f"{report['status']}: {result.testsRun} checks; report={args.report}")
    return 0 if report["status"] == "PASS_LOCAL_CPU" else 1


if __name__ == "__main__":
    raise SystemExit(main())
