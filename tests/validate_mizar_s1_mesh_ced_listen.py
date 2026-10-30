"""Optional CPU-only validation; no assets, downloads, CUDA training or vc jobs.

Use --require-tensors in remote mellow_c8204d8 to reject any skipped tensor
coverage. This command is never a prerequisite of the formal launcher.
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer36912_listen"
FILES = ["test_mizar_s1_mesh_ced_listen_behavior.py", "test_mizar_s1_mesh_ced_listen_route.py",
         "test_mizar_s1_mesh_ced_listen_shared_start.py", "test_mizar_s1_mesh_ced_listen_tensor.py",
         "test_mizar_s1_recursive_mesh_ced_small_route.py", "test_mizar_s1_recursive_mesh_ced_small_parity.py",
         "test_mizar_s1_recursive_mesh_ced_small_shared_start.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "audit_reports/mizar_s1_mesh_ced_listen_validation.json")
    args = parser.parse_args()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else None
    if not bash or not bash.is_file():
        found = shutil.which("bash")
        bash = Path(found) if found else None
    if bash:
        env["PATH"] = str(bash.parent) + os.pathsep + env.get("PATH", "")
    results = []
    sources = list(ROUTE.rglob("*.py")) + [ROOT / "tests" / name for name in FILES] + [Path(__file__)]
    for path in sources:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    results.append(dict(check="python_compile_in_memory", status="PASS", files=len(sources)))
    launcher = ROOT / "code/RSmol/run_mizar_s1_gbs256_recursive_mesh_ced_small_layer36912_listen_5090.sh"
    for path in (launcher, ROUTE / "run.sh"):
        if bash is None:
            results.append(dict(check=path.name, status="FAIL", error="Bash unavailable"))
        else:
            done = subprocess.run([str(bash), "-n", path.as_posix()], env=env, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=30)
            results.append(dict(check=path.name, status="PASS" if done.returncode == 0 else "FAIL",
                                scope="bash syntax", output=done.stdout+done.stderr))

    def run(name):
        done = subprocess.run([sys.executable, "-B", str(ROOT / "tests" / name), "-v"], cwd=ROOT,
            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        output = done.stdout+done.stderr
        count = re.search(r"Ran (\d+) tests?", output)
        skips = re.search(r"skipped=(\d+)", output)
        tests, skipped = int(count[1]) if count else 0, int(skips[1]) if skips else 0
        return dict(check=name, status="PASS" if done.returncode == 0 and tests else "FAIL",
                    tests=tests, skipped=skipped, exit_code=done.returncode, output=output)

    # Isolated subprocesses prevent same-named route packages contaminating imports.
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(run, FILES):
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "output"}), flush=True)
            if result["status"] == "FAIL":
                print(result["output"], flush=True)
    missing = [n for n in ("torch", "torchaudio", "transformers", "soundfile", "numpy")
               if importlib.util.find_spec(n) is None]
    tensors = [r for r in results if r["check"].endswith("_tensor.py")]
    tensor_pass = bool(tensors) and not missing and all(r["status"] == "PASS" and r["tests"] > 0
                                                      and r["skipped"] == 0 for r in tensors)
    if args.require_tensors:
        results.append(dict(check="required_tensor_coverage", status="PASS" if tensor_pass else "FAIL",
                            missing_dependencies=missing))
    failed = any(r["status"] == "FAIL" for r in results)
    skips = sum(r.get("skipped", 0) for r in results)
    report = dict(status="FAIL" if failed else "PASS_WITH_SKIPS" if skips else "PASS",
        scope="local code/CPU validation only; no real assets, remote data, CUDA or eight-rank integration",
        route=ROUTE.name, require_tensors=args.require_tensors, missing_dependencies=missing,
        real_cpu_tensor_checks_passed=tensor_pass,
        executed_tests=sum(r.get("tests", 0)-r.get("skipped", 0) for r in results),
        skipped_tests=skips, checks=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}), flush=True)
    print("report="+str(args.output), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
