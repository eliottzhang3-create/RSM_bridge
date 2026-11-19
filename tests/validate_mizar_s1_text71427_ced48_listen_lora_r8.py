"""Validate new GBS128 audio route and unchanged parent/text routes, offline.

No real assets, network, CUDA or vc submission. Optional --require-tensors turns
missing ML dependencies/skips into a failure; it is not a formal-job prerequisite.
"""
import argparse
import ast
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
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced48_listen_lora_r8"
TESTS = ["test_mizar_s1_text71427_ced48_listen_lora_r8_route.py", "test_mizar_s1_text71427_ced48_listen_lora_r8_shared_start.py",
         "test_mizar_s1_text71427_ced48_listen_lora_r8_tensors.py", "test_mizar_s1_text71427_ced48_listen_lora_r8_behavior.py", "test_mizar_s1_text71427_ced_small_route.py", "test_mizar_s1_recursive_mesh_ced_small_route.py",
         "test_mizar_s1_recursive_mesh_ced_small_parity.py", "test_mizar_s1_recursive_mesh_ced_small_shared_start.py",
         "test_mizar_text_7_14x2_7_mesh.py", "test_mizar_text_7_14x2_7_mesh_shell.py",
         "test_mizar_text_7_14x2_7_mesh_tensors.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "audit_reports/mizar_s1_text71427_ced48_listen_lora_r8_validation.json")
    args = parser.parse_args()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "")
    if not bash.is_file(): raise RuntimeError("Bash is required for actual shell checks")
    env["PATH"] = str(bash.parent) + os.pathsep + env.get("PATH", "")
    checks = []
    files = list(ROUTE.rglob("*.py")) + [ROOT / "tests" / n for n in TESTS] + [Path(__file__)]
    for path in files: ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path), feature_version=(3, 10))
    checks.append(dict(check="python_3_10_syntax", status="PASS", files=len(files)))
    for path in (ROOT / "code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced48_listen_lora_r8_3090.sh", ROUTE / "run.sh"):
        result = subprocess.run([str(bash), "-n", path.as_posix()], env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=30)
        checks.append(dict(check=path.name, status="FAIL" if result.returncode else "PASS", output=result.stdout + result.stderr))
    def run(path):
        result = subprocess.run([sys.executable, "-B", str(path), "-v"], cwd=ROOT, env=env, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=300)
        output = result.stdout + result.stderr
        count, skip = re.search(r"Ran (\d+) tests?", output), re.search(r"skipped=(\d+)", output)
        skipped = int(skip[1]) if skip else 0
        return dict(check=path.name, status="FAIL" if result.returncode or not count else "PASS_WITH_SKIPS" if skipped else "PASS",
                    tests=int(count[1]) if count else 0, skipped=skipped, exit_code=result.returncode, output=output)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(run, [ROOT / "tests" / name for name in TESTS]):
            checks.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "output"}), flush=True)
            if result["status"] == "FAIL": print(result["output"][-7000:], flush=True)
    missing = [n for n in ("torch", "torchaudio", "transformers", "soundfile", "safetensors", "numpy")
               if importlib.util.find_spec(n) is None]
    if missing:
        runtime_tests = sum(isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
                            for node in ast.walk(ast.parse((ROUTE / "runtime_checks.py").read_text(encoding="utf-8"))))
        checks.append(dict(check="runtime_checks.py", status="PASS_WITH_SKIPS", tests=runtime_tests, skipped=runtime_tests,
                           reason="ML dependencies missing; no real tensor/audio/CUDA execution"))
    else:
        checks.append(run(ROUTE / "runtime_checks.py"))
    reference = json.loads((ROUTE / "reference_audit.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROUTE))
    import contract
    checks.append(dict(check="official_reference_audit_identity", status="PASS" if reference["status"] == "PASS"
                       and reference["route_code_sha256"] == contract.code_digest() else "FAIL"))
    skipped = sum(item.get("skipped", 0) for item in checks)
    if args.require_tensors: checks.append(dict(check="require_real_tensors", status="FAIL" if skipped or missing else "PASS"))
    failed = any(item["status"] == "FAIL" for item in checks)
    report = dict(status="FAIL" if failed else "PASS_LOCAL_WITH_SKIPS" if skipped else "PASS_LOCAL",
                  scope="local source, synthetic fixtures and simulated commands; no production assets or GPU training",
                  executed_tests=sum(item.get("tests", 0) - item.get("skipped", 0) for item in checks),
                  skipped_tests=skipped, missing_dependencies=missing, checks=checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}), flush=True)
    return int(failed)


if __name__ == "__main__": raise SystemExit(main())
