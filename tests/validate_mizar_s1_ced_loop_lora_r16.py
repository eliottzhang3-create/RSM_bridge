"""Validate isolated CED8->4 rank16/alpha32 training and unchanged rank8 routes.

No downloads, remote assets, CUDA or job submission. --require-tensors rejects skips.
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
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16"
TASKS = [(f"test_mizar_s1_ced_loop_lora_r16_{kind}.py", "")
         for kind in ("route", "behavior", "shared_start", "tensor")]
TASKS += [(f"test_mizar_s1_ced_loop_lora_{kind}.py", order)
          for order in ("84", "48") for kind in ("route", "behavior", "tensor")]
TASKS += [(f"test_mizar_s1_ced_layer84_audio_reinject_{kind}.py", "")
          for kind in ("behavior", "route", "tensor")]
TASKS += [("test_mizar_s1_recursive_5_10_5_ced_small_shared_start.py", "")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT/"audit_reports/mizar_s1_ced_loop_lora_r16_validation.json")
    args = parser.parse_args()
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "")
    if not bash.is_file():
        raise RuntimeError("Bash is required for formal launcher validation")
    environment["PATH"] = str(bash.parent) + os.pathsep + environment.get("PATH", "")
    results = []
    result = subprocess.run([sys.executable, "-B", str(ROUTE/"audit_reference.py")], cwd=ROOT,
                            env=environment, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=60)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    results.append(dict(check="complete_rank8_rank16_source_parity", status="PASS"))
    files = list(ROUTE.rglob("*.py")) + [ROOT/"tests"/name for name in sorted({n for n, _ in TASKS})] + [Path(__file__)]
    for path in files:
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
    results.append(dict(check="python_syntax", status="PASS", files=len(files)))
    launcher = ROOT/"code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16_5090.sh"
    for path in (launcher, ROUTE/"run.sh"):
        result = subprocess.run([str(bash), "-n", path.as_posix()], env=environment,
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        results.append(dict(check=path.name, status="FAIL" if result.returncode else "PASS",
                            output=result.stdout+result.stderr))

    def run(task):
        name, order = task
        env = dict(environment, RSM_LORA_ORDER=order) if order else environment
        result = subprocess.run([sys.executable, "-B", str(ROOT/"tests"/name), "-v"], cwd=ROOT,
                                env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=300)
        output = result.stdout+result.stderr
        count = re.search(r"Ran (\d+) tests?", output)
        skip = re.search(r"skipped=(\d+)", output)
        skipped = int(skip[1]) if skip else 0
        status = "FAIL" if result.returncode or not count else "PASS_WITH_SKIPS" if skipped else "PASS"
        return dict(check=name+("/"+order if order else ""), status=status,
                    tests=int(count[1]) if count else 0, skipped=skipped,
                    exit_code=result.returncode, output=output)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(run, TASKS):
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k != "output"}), flush=True)
            if result["status"] == "FAIL":
                print(result["output"][-6000:], flush=True)
    missing = [n for n in ("torch", "torchaudio", "transformers", "soundfile")
               if importlib.util.find_spec(n) is None]
    skipped = sum(r.get("skipped", 0) for r in results)
    if args.require_tensors:
        results.append(dict(check="required_tensor_coverage", status="FAIL" if missing or skipped else "PASS"))
    failed = any(r["status"] == "FAIL" for r in results)
    report = dict(status="FAIL" if failed else "PASS", scope="local code checks; no real assets or GPU training",
                  require_tensors=args.require_tensors, tensor_status="SKIPPED" if missing else "EXECUTED",
                  missing_dependencies=missing, executed_tests=sum(r.get("tests",0)-r.get("skipped",0) for r in results),
                  skipped_tests=skipped, checks=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "checks"}), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
