"""Run isolated four-slot checks and the unchanged six-slot regression suite.

This reports local evidence only. Production conversion and DDP training must
be checked on the remote Linux host with the actual assets.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "audit_reports/text_mesh_4slot_bf16_validation.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true", help="fail on any skipped tensor check, including CUDA")
    args = parser.parse_args()
    os.chdir(ROOT)
    git_bin = Path("C:/Program Files/Git/bin")
    if os.name == "nt" and (git_bin / "bash.exe").is_file():
        os.environ["PATH"] = str(git_bin) + os.pathsep + os.environ.get("PATH", "")
    previous = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.is_file() else {}
    protected = previous.get("protected_original_files", {})
    changed = [name for name, digest in protected.items() if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    sources = [p for base in (ROOT / "code/RSmol", ROOT / "code/RSmol/scripts", ROOT / "tests") for p in base.glob("*5_10x2_5_mesh_4slot*") if p.is_file()]
    sources += [Path(__file__).resolve(), ROOT / "code/RSmol/text_5_10x2_5_mesh_4slot/README.md"]
    python_sources = [p for p in sources if p.suffix == ".py"]
    for path in python_sources:
        compile(ast.parse(path.read_text(encoding="utf-8")), str(path), "exec")
    suites = {}
    passed = not changed
    for route in ("4slot", "6slot"):
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=f"test_text_5_10x2_5_mesh_{route}_*.py")
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        suites[route] = {
            "run": result.testsRun, "passed": result.testsRun - len(result.skipped) - len(result.failures) - len(result.errors),
            "skipped": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
            "failures": [{"test": test.id(), "traceback": error} for test, error in result.failures],
            "errors": [{"test": test.id(), "traceback": error} for test, error in result.errors],
        }
        passed = passed and result.wasSuccessful() and not (args.require_tensors and result.skipped)
    report = {
        "status": "PASS_LOCAL_WITH_SKIPS" if passed and any(suite["skipped"] for suite in suites.values()) else ("PASS_LOCAL" if passed else "FAIL"),
        "validation_time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "local syntax, production contract helpers, Parquet cursor simulation and real Bash mock submissions; optional tensor tests",
        "production_weights_read": False, "remote_conversion_verified": False, "remote_gpu_training_verified": False,
        "protected_original_files": protected, "protected_files_changed": changed,
        "python_syntax_files": [p.relative_to(ROOT).as_posix() for p in python_sources],
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "suites": suites,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{report['status']}: passed={sum(suite['passed'] for suite in suites.values())}, skipped={sum(len(suite['skipped']) for suite in suites.values())}, protected={len(protected)}, syntax={len(python_sources)}; report={REPORT}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
