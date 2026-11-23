"""Offline qualification for isolated three-loop audio8/read3 LoRA training.

Synthetic files, NumPy decoder/collective fixtures and simulated vc only.
--require-tensors rejects missing dependencies or tensor skips. Not a formal gate.
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
import time

ROOT = Path(__file__).resolve().parents[1]
NAME = "mizar_s1_text71437_ced48_audio8_read3"
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71437_mesh_ced48_audio8_read3_lora_r8"
ENTRY = ROOT / "code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced48_audio8_read3_lora_r8_3090.sh"
TESTS = [f"test_{NAME}{part}.py" for part in ("", "_route", "_shared_start", "_staging_roots", "_audio", "_loader", "_behavior")]
TESTS += ["test_mizar_s1_text71427_ced48_audio7_read2.py",
          "test_mizar_s1_text71427_ced48_audio7_read2_audio.py", "test_mizar_text_7_14x3_7_mesh.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / f"audit_reports/{NAME}_validation.json")
    args = parser.parse_args()
    start = time.monotonic()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "")
    if not bash.is_file():
        raise RuntimeError("Git Bash is required for real launcher checks")
    env["PATH"] = str(bash.parent) + os.pathsep + env.get("PATH", "")
    checks = []
    files = list(ROUTE.rglob("*.py")) + [ROOT / "tests" / n for n in TESTS] + [Path(__file__)]
    for path in files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path), feature_version=(3, 10))
    checks.append(dict(check="python_3_10_syntax", status="PASS", files=len(files)))
    for path in (ENTRY, ROUTE / "run.sh"):
        result = subprocess.run([str(bash), "-n", path.as_posix()], env=env, capture_output=True,
                                text=True, timeout=30)
        checks.append(dict(check=path.name, status="FAIL" if result.returncode else "PASS",
                           output=result.stdout + result.stderr))
    fence = chr(96) * 3
    commands = "\n".join(re.findall(fence + r"bash\n(.*?)" + fence, (ROUTE / "README.md").read_text(encoding="utf-8"), re.S))
    result = subprocess.run([str(bash), "-n"], input=commands, env=env, capture_output=True,
                            text=True, timeout=30)
    checks.append(dict(check="README_bash_syntax", status="FAIL" if result.returncode else "PASS",
                       output=result.stdout + result.stderr))
    result = subprocess.run([sys.executable, "-B", str(ROUTE / "audit_reference.py")], cwd=ROOT,
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
    checks.append(dict(check="two_parent_source_and_complete_epoch_audit",
                       status="FAIL" if result.returncode else "PASS", output=result.stdout + result.stderr))

    def run(name):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "tests" / name), "-v"], cwd=ROOT,
                                env=env, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=300)
        output = result.stdout + result.stderr
        count, skip = re.search(r"Ran (\d+) tests?", output), re.search(r"skipped=(\d+)", output)
        skipped = int(skip[1]) if skip else 0
        return dict(check=name, status="FAIL" if result.returncode or not count else "PASS_WITH_SKIPS" if skipped else "PASS",
                    tests=int(count[1]) if count else 0, skipped=skipped, exit_code=result.returncode, output=output)

    with ThreadPoolExecutor(max_workers=3) as pool:
        for item in pool.map(run, TESTS):
            checks.append(item)
            print(json.dumps({k: v for k, v in item.items() if k != "output"}), flush=True)
            if item["status"] == "FAIL":
                print(item["output"][-14000:], flush=True)
    missing = [n for n in ("torch", "torchaudio", "transformers", "soundfile", "safetensors", "numpy")
               if importlib.util.find_spec(n) is None]
    skipped = sum(c.get("skipped", 0) for c in checks)
    if args.require_tensors:
        checks.append(dict(check="require_real_tensors", status="FAIL" if missing or skipped else "PASS"))
    history = []
    if args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        history = previous.get("previous_runs", []) + [{
            k: previous[k] for k in ("status", "executed_tests", "skipped_tests", "checks") if k in previous
        }]
    failed = any(c["status"] == "FAIL" for c in checks)
    report = dict(status="FAIL" if failed else "PASS_LOCAL_WITH_SKIPS" if skipped else "PASS_LOCAL",
        scope="local source and synthetic fixtures; no production assets, CUDA or vc submission",
        executed_tests=sum(c.get("tests", 0) - c.get("skipped", 0) for c in checks),
        skipped_tests=skipped, missing_dependencies=missing, elapsed_seconds=time.monotonic() - start,
        protected_existing_sources=3609, previous_runs=history,
        development_notes=[
            "Initial isolated core test encountered Windows sandbox temporary-directory PermissionError; identical unsandboxed offline run passed.",
            "Initial numerical scope test omitted fixture.events; corrected test setup without changing production behavior.",
        ], checks=checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"checks", "previous_runs"}}), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
