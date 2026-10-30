"""Validate the isolated four-route CED evaluator and base/MeSH regressions.

No downloads, production weights, CUDA or job submission. --require-tensors
turns any missing tensor dependency or skipped tensor check into failure.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code/RSmol"
TEST_FILES = (
    "test_mizar_s1_ced_recursive_eval.py",
    "test_mizar_s1_ced_recursive_eval_workflow.py",
    "test_mizar_s1_ced_recursive_eval_shell.py",
    "test_mizar_s1_ced_recursive_eval_tensor.py",
    "test_mizar_s1_ced_benchmark_eval.py",
    "test_mizar_s1_ced_recursive_mesh_eval.py",
    "test_mizar_s1_ced_recursive_mesh_eval_shell.py",
    "test_mizar_s1_ced_recursive_mesh_eval_tensor.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "audit_reports/mizar_s1_ced_recursive_eval_validation.json")
    args = parser.parse_args()
    results = []
    sources = [RSMOL / "scripts" / name for name in (
        "evaluate_mizar_s1_ced_recursive_benchmark.py", "mizar_s1_ced_recursive_eval_runtime.py")]
    sources += [ROOT / "tests" / name for name in TEST_FILES] + [Path(__file__)]
    for path in sources:
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
    results.append({"check": "python_syntax", "status": "PASS", "files": len(sources)})
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    if bash:
        environment["PATH"] = str(Path(bash).parent) + os.pathsep + environment.get("PATH", "")
    for path in (RSMOL / "run_mizar_s1_ced_small_mmau_eval_5090.sh",
                 RSMOL / "run_mizar_s1_ced_small_mmar_eval_5090.sh",
                 RSMOL / "scripts/evaluate_mizar_s1_ced_recursive_benchmark.sh"):
        if bash:
            completed = subprocess.run([bash, "-n", path.as_posix()], cwd=ROOT, env=environment,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
            results.append({"check": path.name, "status": "PASS" if completed.returncode == 0 else "FAIL",
                            "output": completed.stdout + completed.stderr})
        else:
            results.append({"check": path.name, "status": "FAIL", "output": "Bash is required"})
    for name in TEST_FILES:
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "tests" / name), "-v"],
            cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        output = completed.stdout + completed.stderr
        count = re.search(r"Ran (\d+) tests?", output)
        skipped = re.search(r"skipped=(\d+)", output)
        result = {"check": name, "status": "PASS" if completed.returncode == 0 and count else "FAIL",
                  "tests": int(count[1]) if count else 0, "skipped": int(skipped[1]) if skipped else 0,
                  "exit_code": completed.returncode, "output": output}
        if result["status"] == "PASS" and result["tests"] == result["skipped"]:
            result["status"] = "SKIPPED"
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "output"}), flush=True)
        if result["status"] == "FAIL": print(output, flush=True)
    missing = [name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None]
    tensor_skips = sum(r.get("skipped", 0) for r in results if "tensor" in r["check"])
    if args.require_tensors:
        results.append({"check": "required_tensor_coverage", "status": "FAIL" if missing or tensor_skips else "PASS",
                        "missing_dependencies": missing, "skipped_tensor_tests": tensor_skips})
    failed = any(r["status"] == "FAIL" for r in results)
    report = {"status": "FAIL" if failed else "PASS", "scope": "local code validation only; no remote assets or GPU evaluation",
        "require_tensors": args.require_tensors, "tensor_dependencies_available": not missing,
        "missing_dependencies": missing, "executed_tests": sum(r.get("tests", 0) - r.get("skipped", 0) for r in results),
        "skipped_tests": sum(r.get("skipped", 0) for r in results), "checks": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}), flush=True)
    print(f"report={args.output}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
