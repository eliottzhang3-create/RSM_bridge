"""Validate the new training route locally; never submit or use production data.

--require-tensors rejects missing real CPU tensor coverage. Small-dimension CPU
tests do not replace remote full-model/eight-CUDA-rank validation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_official_final_s1_gbs128_5_20x2_5_input_only"
LAUNCHER = ROOT / "code/RSmol/run_mizar_official_final_s1_gbs128_5_20x2_5_input_only_5090.sh"
PACKAGE = ROOT / "code/RSmol/mizar_official_recursive_5_20x2_5_input_only"
CONVERTER = ROOT / "code/RSmol/scripts/mizar_official_recursive_5_20x2_5_input_only.sh"
SUITES = ["test_mizar_input_only_conversion.py", "test_mizar_input_only_training.py", "test_mizar_input_only_shell.py", "test_mizar_input_only_tensor.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT/"audit_reports/mizar_input_only_validation.json")
    args = parser.parse_args()
    results = []
    python_files = [*PACKAGE.rglob("*.py"), *ROUTE.rglob("*.py"), *(ROOT/"tests"/n for n in SUITES), Path(__file__)]
    for path in python_files:
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
    results.append({"check": "python_compile", "status": "PASS", "files": len(python_files)})
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing/bash")
    environment = dict(os.environ, PYTHONUTF8="1")
    if not bash.is_file():
        results.append({"check": "bash_available", "status": "FAIL"})
    else:
        for path in (CONVERTER, LAUNCHER, ROUTE/"run.sh"):
            if b"\r" in path.read_bytes():
                raise RuntimeError(f"Bash entry must use LF: {path}")
            result = subprocess.run([str(bash), "-n", path.as_posix()], capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", env=environment, timeout=30)
            results.append({"check": "bash_syntax/"+path.name, "status": "PASS" if result.returncode == 0 else "FAIL",
                            "output": result.stdout+result.stderr})
    def run(name):
        result = subprocess.run([sys.executable, "-B", str(ROOT/"tests"/name), "-v"],
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                env=environment, cwd=ROOT, timeout=180)
        output = result.stdout+result.stderr
        count = re.search(r"Ran (\d+) tests?", output)
        skipped = re.search(r"skipped=(\d+)", output)
        return {"check": name, "status": "PASS" if result.returncode == 0 and count else "FAIL",
                "tests": int(count[1]) if count else 0, "skipped": int(skipped[1]) if skipped else 0,
                "exit_code": result.returncode, "output": output}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(run, SUITES):
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "output"}), flush=True)
            if result["status"] == "FAIL":
                print(result["output"], flush=True)
    missing = [n for n in ("torch", "transformers", "torchaudio", "numpy", "soundfile", "safetensors")
               if importlib.util.find_spec(n) is None]
    skipped = sum(r.get("skipped", 0) for r in results)
    if args.require_tensors:
        results.append({"check": "required_tensor_coverage", "status": "FAIL" if missing or skipped else "PASS",
                        "missing_dependencies": missing, "skipped_tests": skipped})
    report = {"status": "FAIL" if any(r["status"] == "FAIL" for r in results) else "PASS",
              "scope": "local file/protocol/Bash fixtures and optional small CPU tensors; no remote weights or GPU/DDP PASS",
              "executed_tests": sum(r.get("tests", 0)-r.get("skipped", 0) for r in results),
              "skipped_tests": skipped, "missing_dependencies": missing, "require_tensors": args.require_tensors,
              "checks": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({k:v for k,v in report.items() if k != "checks"}), flush=True)
    raise SystemExit(report["status"] != "PASS")


if __name__ == "__main__":
    main()
