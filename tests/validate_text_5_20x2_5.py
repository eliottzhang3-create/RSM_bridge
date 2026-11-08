"""Run local route validation; --require-tensors rejects dependency skips."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "tests"))
    if os.name == "nt":
        os.environ["PATH"] = "C:/Program Files/Git/bin" + os.pathsep + os.environ["PATH"]
    patterns = ("test_text_5_20x2_5_*.py", "test_5_10_5_static.py", "test_stage4_static.py")
    files = [p for directory in (ROOT / "code/RSmol", ROOT / "code/RSmol/scripts", ROOT / "tests")
             for p in directory.glob("*5_20x2_5*.py")]
    for path in files:
        compile(ast.parse(path.read_text(encoding="utf-8")), str(path), "exec")
    source_files = files + [p for directory in (ROOT / "code/RSmol", ROOT / "code/RSmol/scripts")
                            for p in directory.glob("*5_20x2_5*.sh")]
    source_files += [ROOT / "code/RSmol/text_5_20x2_5" / name for name in ("README.md", "source_guard.json")]
    suites, success = {}, True
    for pattern in patterns:
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=pattern)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        suites[pattern] = {
            "run": result.testsRun,
            "passed": result.testsRun - len(result.skipped) - len(result.failures) - len(result.errors),
            "skipped": [{"test": t.id(), "reason": reason} for t, reason in result.skipped],
            "failures": [{"test": t.id(), "traceback": message} for t, message in result.failures],
            "errors": [{"test": t.id(), "traceback": message} for t, message in result.errors],
        }
        success = success and result.wasSuccessful() and not (args.require_tensors and result.skipped)
    skipped = sum(len(s["skipped"]) for s in suites.values())
    report = {"status": ("PASS_LOCAL_WITH_SKIPS" if skipped else "PASS_LOCAL") if success else "FAIL",
              "time_utc": datetime.now(timezone.utc).isoformat(), "suites": suites,
              "production_assets_read": False, "remote_conversion_verified": False,
              "gpu_training_verified": False, "eight_rank_resume_verified": False,
              "syntax_checked": [p.relative_to(ROOT).as_posix() for p in files],
              "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in source_files if p.is_file()}}
    target = ROOT / "audit_reports/text_5_20x2_5_validation.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{report['status']}: passed={sum(s['passed'] for s in suites.values())}, skipped={skipped}; report={target}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
