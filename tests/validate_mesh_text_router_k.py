#!/usr/bin/env python3
"""Local/remote CPU validation only; never downloads assets or submits jobs."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "audit_reports/mesh_text_router_k_validation.json")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_mesh_text_router_k*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    syntax = []
    for path in [ROOT / "code/RSmol/scripts/calibrate_mesh_text_router_k.py", *ROOT.glob("tests/*mesh_text_router_k*.py")]:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        syntax.append({"path": str(path.relative_to(ROOT)), "status": "PASS"})
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
    for path in (ROOT / "code/RSmol/run_mesh_text_router_k_5090.sh", ROOT / "code/RSmol/scripts/calibrate_mesh_text_router_k.sh"):
        value = path.resolve().as_posix()
        if os.name == "nt": value = f"/{value[0].lower()}{value[2:]}"
        check = subprocess.run([bash, "-n", value], text=True, capture_output=True) if bash else None
        syntax.append({"path": str(path.relative_to(ROOT)), "status": "PASS" if check and check.returncode == 0 else "FAIL",
                       "stderr": check.stderr if check else "Bash unavailable"})
    ok = result.wasSuccessful() and all(item["status"] == "PASS" for item in syntax) and not (args.require_tensors and result.skipped)
    report = {"status": "PASS" if ok else "FAIL", "scope": "local CPU/code checks only; no remote checkpoint, CUDA or submission",
              "tests_run": result.testsRun, "passed": result.testsRun - len(result.skipped) - len(result.errors) - len(result.failures),
              "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
              "failures": [{"test": str(test), "traceback": trace} for test, trace in result.failures + result.errors],
              "syntax": syntax, "require_tensors": args.require_tensors, "remote_gpu_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Validation {report['status']}: {report['passed']} passed; {len(result.skipped)} skipped; report={args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
