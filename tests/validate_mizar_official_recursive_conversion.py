"""Local code validation only. No downloads, production conversion or training."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "code/RSmol/mizar_official_recursive_5_20x2_5"
SHELL = ROOT / "code/RSmol/scripts/mizar_official_recursive_5_20x2_5.sh"
TESTS = ("test_mizar_official_recursive_conversion.py", "test_mizar_official_recursive_tensor.py")
REF = ROOT / ("code/RSmol/mellow_official_training_c8204d8_adamw_cosine/"
              "mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8")


def protected_hashes():
    paths = list(REF.rglob("*.py")) + list(REF.rglob("*.sh"))
    paths += list((ROOT / "code/RSmol/mizar_official_eval").glob("*.py"))
    paths += [ROOT / "code/RSmol/mizar_official_eval/release_manifest.json"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "audit_reports/mizar_official_recursive_conversion_validation.json")
    args = parser.parse_args()
    protected = protected_hashes()
    results = []
    sources = list(PACKAGE.glob("*.py")) + [Path(__file__)] + [ROOT / "tests" / n for n in TESTS]
    try:
        for path in sources: compile(path.read_text(encoding="utf-8"), str(path), "exec")
        results.append({"check": "python_syntax", "status": "PASS", "files": len(sources)})
    except Exception as error:
        results.append({"check": "python_syntax", "status": "FAIL", "error": str(error)})
    bash_path = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(bash_path) if os.name == "nt" and bash_path.is_file() else shutil.which("bash")
    if bash:
        check = subprocess.run([bash, "-n", SHELL.as_posix()], capture_output=True, text=True, timeout=30)
        results.append({"check": "bash_syntax_and_LF", "status": "PASS" if check.returncode == 0 and b"\r" not in SHELL.read_bytes() else "FAIL",
                        "output": check.stdout + check.stderr})
    else:
        results.append({"check": "bash_syntax_and_LF", "status": "FAIL", "error": "Bash is required"})
    # unittest reports real skip counts directly; no dependency substitution.
    import unittest
    sys.path.insert(0, str(ROOT / "tests"))
    for name in TESTS:
        suite = unittest.defaultTestLoader.loadTestsFromName(Path(name).stem)
        import io
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        item = {"check": name, "status": "PASS" if result.wasSuccessful() else "FAIL",
                "tests": result.testsRun, "skipped": len(result.skipped), "output": stream.getvalue()}
        if result.wasSuccessful() and result.testsRun == len(result.skipped): item["status"] = "SKIPPED"
        results.append(item)
        print(json.dumps({k: v for k, v in item.items() if k != "output"}), flush=True)
        if not result.wasSuccessful(): print(stream.getvalue(), flush=True)
    missing = [n for n in ("torch", "transformers") if importlib.util.find_spec(n) is None]
    if args.require_tensors:
        results.append({"check": "required_real_cpu_tensors", "status": "FAIL" if missing or any(r.get("skipped") for r in results) else "PASS",
                        "missing": missing})
    results.append({"check": "reference_sources_unchanged", "status": "PASS" if protected_hashes() == protected else "FAIL",
                    "files": len(protected)})
    failed = any(r["status"] == "FAIL" for r in results)
    report = {"status": "FAIL" if failed else "PASS", "scope": "local_code_only_no_production_weights",
              "executed_tests": sum(r.get("tests", 0) - r.get("skipped", 0) for r in results),
              "skipped_tests": sum(r.get("skipped", 0) for r in results), "missing_tensor_dependencies": missing,
              "remote_weights_audited": False, "production_conversion_executed": False,
              "gpu_training_executed": False, "checks": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}), flush=True)
    print(f"report={args.output}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
