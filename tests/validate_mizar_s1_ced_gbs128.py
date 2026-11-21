"""Local GBS128/3090 validation; no remote assets, network, CUDA or submission."""
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
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs128_3epochs_ced_small"
LAUNCHER = ROOT / "code/RSmol/run_mizar_s1_ced_small_gbs128_3090.sh"
TESTS = ["test_mizar_s1_ced_gbs128_route.py", "test_mizar_s1_ced_gbs128_shared_start.py",
         "test_mizar_s1_ced_gbs128_offline_config.py", "test_mizar_s1_ced_gbs128_staging_roots.py",
         "test_mizar_s1_ced_gbs128_behavior.py", "test_mizar_s1_ced_gbs128_tensor.py",
         "test_mizar_s1_ced_small_route.py", "test_mizar_s1_ced_small_shared_start.py",
         "test_mizar_s1_ced_offline_config.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true")
    parser.add_argument("--reference-root", type=Path,
                        default=Path("C:/Xlance/assets/Mizar/Mizar_159M-main/Mizar_159M-main"))
    parser.add_argument("--output", type=Path, default=ROOT / "audit_reports/mizar_s1_ced_gbs128_validation.json")
    args = parser.parse_args()
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "")
    if not bash.is_file():
        raise RuntimeError("Git Bash/POSIX Bash is required to verify the actual entry points")
    environment["PATH"] = str(bash.parent) + os.pathsep + environment.get("PATH", "")
    results = []
    files = list(ROUTE.rglob("*.py")) + [ROOT / "tests" / name for name in TESTS] + [Path(__file__)]
    for path in files:
        source = path.read_text(encoding="utf-8-sig")
        ast.parse(source, filename=str(path), feature_version=(3, 10))
        compile(source, str(path), "exec")
    results.append(dict(check="python_3_10_syntax", status="PASS", files=len(files)))
    for path in (LAUNCHER, ROUTE / "run.sh"):
        completed = subprocess.run([str(bash), "-n", path.as_posix()], env=environment, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=30)
        results.append(dict(check=path.name, status="FAIL" if completed.returncode else "PASS",
                            output=completed.stdout + completed.stderr))
    for number, command in enumerate(re.findall(r"```bash\n(.*?)```", (ROUTE / "README.md").read_text(encoding="utf-8"), re.S)):
        completed = subprocess.run([str(bash), "-n"], input=command, env=environment,
                                   capture_output=True, text=True, encoding="utf-8", timeout=30)
        results.append(dict(check=f"readme_bash_block_{number + 1}", status="FAIL" if completed.returncode else "PASS",
                            output=completed.stdout + completed.stderr))

    def run(name):
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "tests" / name), "-v"], cwd=ROOT,
                                   env=environment, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=300)
        output = completed.stdout + completed.stderr
        count, skip = re.search(r"Ran (\d+) tests?", output), re.search(r"skipped=(\d+)", output)
        skipped = int(skip[1]) if skip else 0
        return dict(check=name, status="FAIL" if completed.returncode or not count else "PASS_WITH_SKIPS" if skipped else "PASS",
                    tests=int(count[1]) if count else 0, skipped=skipped, exit_code=completed.returncode, output=output)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(run, TESTS):
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "output"}), flush=True)
            if result["status"] == "FAIL":
                print(result["output"][-10000:], flush=True)
    for path in (LAUNCHER, *ROUTE.rglob("*.py"), ROUTE / "run.sh",
                 ROUTE / "reference_audit.json", ROUTE / "parent_reference_audit.json",
                 ROUTE / "original_files_sha256.json"):
        completed = subprocess.run(["git", "check-ignore", "--no-index", "--", str(path)], cwd=ROOT,
                                   env=environment, capture_output=True, text=True, encoding="utf-8", timeout=30)
        if completed.returncode != 1:
            results.append(dict(check="git_sync_" + path.name, status="FAIL",
                                output=completed.stdout + completed.stderr))
    results.append(dict(check="git_sync_manifest", status="PASS" if not any(
        r["check"].startswith("git_sync_") and r["status"] == "FAIL" for r in results) else "FAIL"))
    reference_present = (args.reference_root / "runtime/stage1/training/trainer.py").is_file()
    if reference_present:
        completed = subprocess.run([sys.executable, "-B", str(ROUTE / "audit_reference.py"),
                                    "--reference-root", str(args.reference_root),
                                    "--output", str(ROUTE / "reference_audit.json")], cwd=ROOT,
                                   env=environment, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=120)
        result = dict(check="official_reference_and_parent_source_audit", status="FAIL" if completed.returncode else "PASS",
                      output=completed.stdout + completed.stderr)
        results.append(result)
        if completed.returncode:
            print(result["output"][-10000:], flush=True)
    else:
        results.append(dict(check="official_reference_audit", status="NOT_RUN", reason="local official reference tree unavailable"))
    missing = [name for name in ("torch", "torchaudio", "transformers", "soundfile", "safetensors", "numpy")
               if importlib.util.find_spec(name) is None]
    skipped = sum(r.get("skipped", 0) for r in results)
    if args.require_tensors:
        results.append(dict(check="required_tensor_coverage", status="FAIL" if missing or skipped else "PASS"))
    failed = any(r["status"] == "FAIL" for r in results)
    report = dict(status="FAIL" if failed else "PASS_LOCAL_WITH_SKIPS" if skipped else "PASS_LOCAL",
                  scope="local offline tests; synthetic tensor-free fixtures explicitly separated from real CPU tensor coverage",
                  remote_assets_read=False, gpu_training_run=False, vc_jobs_submitted=False,
                  require_tensors=args.require_tensors, missing_dependencies=missing,
                  geometry=dict(world_size=8, microbatch=4, accumulation=4, global_batch=128,
                                epochs=3, steps_per_epoch=11542, total_steps=34626, warmup=1731),
                  resources=dict(queue="pdgpu-3090", cpu=64, memory="256G", gpu=8, nodes=1),
                  executed_tests=sum(r.get("tests", 0) - r.get("skipped", 0) for r in results),
                  skipped_tests=skipped, checks=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
