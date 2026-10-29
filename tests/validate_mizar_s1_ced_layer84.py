"""Run isolated new-route checks and the unchanged parent regressions.

Use --require-tensors in remote mellow_c8204d8 to require real CPU tensor checks.
This script never downloads assets, submits jobs or runs CUDA training.
"""
import argparse
import importlib.util
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject"


def test_result(name, completed):
    output = completed.stdout + completed.stderr
    count = re.search(r"Ran (\d+) tests?", output)
    skipped = re.search(r"skipped=(\d+)", output)
    return dict(check=name,
                status="PASS" if completed.returncode == 0 and count and int(count[1]) else "FAIL",
                tests=int(count[1]) if count else 0, skipped=int(skipped[1]) if skipped else 0,
                exit_code=completed.returncode, output=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true")
    parser.add_argument("--output", type=Path, default=ROUTE / "validation_report.json")
    args = parser.parse_args()
    files = ["test_mizar_s1_ced_layer84_behavior.py", "test_mizar_s1_ced_layer84_route.py",
             "test_mizar_s1_ced_layer84_tensor.py",
             "test_mizar_s1_ced_layer48_behavior.py", "test_mizar_s1_ced_layer48_route.py",
             "test_mizar_s1_ced_layer48_tensor.py",
             "test_mizar_s1_recursive_5_10_5_ced_small_route.py",
             "test_mizar_s1_recursive_5_10_5_ced_small_parity.py",
             "test_mizar_s1_recursive_5_10_5_ced_small_offline_config.py",
             "test_mizar_s1_recursive_5_10_5_ced_small_shared_start.py",
             "test_mizar_s1_recursive_5_10_5_ced_small_tensor.py"]
    environment = dict(os.environ, PYTHONUTF8="1")
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else None
    if bash and bash.is_file():
        environment["PATH"] = str(bash.parent) + os.pathsep + environment.get("PATH", "")
    else:
        bash_name = shutil.which("bash")
        bash = Path(bash_name) if bash_name else None
    results = []
    compiled = list(ROUTE.rglob("*.py")) + [ROOT/"tests"/name for name in files] + [Path(__file__)]
    for path in compiled:
        py_compile.compile(str(path), doraise=True)
    results.append(dict(check="python_compile", status="PASS", files=len(compiled)))
    if not bash or not bash.is_file():
        raise RuntimeError("Bash is required to validate the formal entry")
    launcher = ROOT/"code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_5090.sh"
    for path in (launcher, ROUTE/"run.sh"):
        # Git Bash accepts the native forward-slash Windows absolute path.
        result = subprocess.run([str(bash), "-n", path.as_posix()], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", env=environment, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        results.append(dict(check=path.name, status="PASS", scope="bash syntax"))

    def run(name):
        path = ROOT/"tests"/name
        result = subprocess.run([sys.executable, str(path), "-v"], cwd=ROOT, env=environment,
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        return test_result(name, result)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(run, files):
            results.append(result)
            print(json.dumps({key:value for key,value in result.items() if key != "output"}), flush=True)
            if result["status"] != "PASS":
                print(result["output"], flush=True)
    missing_dependencies = [name for name in ("torch", "torchaudio", "transformers", "soundfile")
                            if importlib.util.find_spec(name) is None]
    tensors = not missing_dependencies
    if args.require_tensors:
        skipped_checks = [result["check"] for result in results
                          if result["status"] == "SKIPPED" or result.get("skipped", 0)]
        results.append(dict(check="required_tensor_coverage",
                            status="FAIL" if missing_dependencies or skipped_checks else "PASS",
                            missing_dependencies=missing_dependencies, skipped_checks=skipped_checks))
    failed = any(result["status"] == "FAIL" for result in results)
    report = dict(status="FAIL" if failed else "PASS",scope="local code validation only; no remote assets or GPU training",
                  tensor_dependencies_available=tensors,require_tensors=args.require_tensors,
                  missing_dependencies=missing_dependencies,
                  executed_tests=sum(r.get("tests",0)-r.get("skipped",0) for r in results),
                  skipped_tests=sum(r.get("skipped",0) for r in results),checks=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:value for key,value in report.items() if key != "checks"}),flush=True)
    print(f"report={args.output}",flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
