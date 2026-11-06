"""Local validation of seven-slot v2 conversion, text training and audio handoff."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code/RSmol"
ROUTE = CODE / "mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_7slot_v2_ced_small"
TESTS = [f"test_text_5_10x2_5_mesh_{slot}_{kind}.py"
         for slot in ("7slot_v2", "4slot") for kind in ("contract", "training", "shell", "tensor")]
TESTS += [f"test_mizar_s1_mesh_{slot}_ced_small_{kind}.py"
          for slot in ("7slot_v2", "4slot") for kind in ("route", "shared_start", "behavior", "tensor")]
TESTS += ["test_mesh_7slot_v2_handoff.py", "test_text_5_10x2_5_mesh_7slot_static.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true", help="fail on all tensor skips, including CUDA")
    parser.add_argument("--output", type=Path, default=ROOT / "audit_reports/mesh_7slot_v2_validation.json")
    args = parser.parse_args()
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "")
    if not bash.is_file():
        raise RuntimeError("Bash is required for actual launcher checks")
    environment["PATH"] = str(bash.parent) + os.pathsep + environment.get("PATH", "")
    results = []
    sources = list(ROUTE.rglob("*.py"))
    for base in (CODE, CODE / "scripts", ROOT / "tests"):
        sources += [path for path in base.glob("*7slot_v2*.py") if path.is_file()]
    sources = sorted(set(sources))
    for path in sources:
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
    results.append(dict(check="python_syntax", status="PASS", files=len(sources)))
    shells = list(CODE.glob("*7slot_v2*.sh")) + list((CODE / "scripts").glob("*7slot_v2*.sh")) + [ROUTE / "run.sh"]
    for path in shells:
        completed = subprocess.run([str(bash), "-n", path.as_posix()], env=environment,
                                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        results.append(dict(check=path.name, status="FAIL" if completed.returncode or b"\r" in path.read_bytes() else "PASS",
                            output=completed.stdout + completed.stderr))
    document_blocks = 0
    for path in (CODE / "text_5_10x2_5_mesh_7slot_v2/README.md", ROUTE / "README.md"):
        for block in re.findall(r"```bash\n(.*?)```", path.read_text(encoding="utf-8"), flags=re.S):
            completed = subprocess.run([str(bash), "-n"], input=block, env=environment,
                                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
            document_blocks += 1
            if completed.returncode:
                raise RuntimeError(f"documented command syntax failed: {path}: {completed.stderr}")
    results.append(dict(check="documented_bash_commands", status="PASS", blocks=document_blocks))
    audit = subprocess.run([sys.executable, "-B", str(ROUTE / "audit_reference.py")], cwd=ROOT,
                           env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    results.append(dict(check="original_sources_and_parent_parity", status="FAIL" if audit.returncode else "PASS", output=audit.stdout + audit.stderr))
    if audit.returncode:
        print(audit.stdout + audit.stderr, flush=True)

    def run(name):
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "tests" / name), "-v"], cwd=ROOT,
                                   env=environment, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=300)
        output = completed.stdout + completed.stderr
        count = re.search(r"Ran (\d+) tests?", output)
        skip = re.search(r"skipped=(\d+)", output)
        skipped = int(skip[1]) if skip else 0
        return dict(check=name, status="FAIL" if completed.returncode or not count else "PASS_WITH_SKIPS" if skipped else "PASS",
                    tests=int(count[1]) if count else 0, skipped=skipped, exit_code=completed.returncode, output=output)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(run, TESTS):
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "output"}), flush=True)
            if result["status"] == "FAIL":
                print(result["output"][-10000:], flush=True)
    missing = [name for name in ("torch", "torchaudio", "transformers", "soundfile", "safetensors")
               if importlib.util.find_spec(name) is None]
    skipped = sum(result.get("skipped", 0) for result in results)
    if args.require_tensors:
        results.append(dict(check="required_tensor_coverage", status="FAIL" if missing or skipped else "PASS"))
    failed = any(result["status"] == "FAIL" for result in results)
    source_audit = json.loads(audit.stdout) if audit.returncode == 0 else None
    report = dict(status="FAIL" if failed else "PASS_WITH_SKIPS" if skipped else "PASS",
                  scope="local checks; synthetic files and mock submissions; no production assets or GPU training",
                  remote_conversion_verified=False, remote_assets_loaded=False, remote_gpu_training_verified=False,
                  source_audit=source_audit, require_tensors=args.require_tensors,
                  source_sha256={path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                 for path in sources + shells},
                  missing_dependencies=missing, executed_tests=sum(r.get("tests", 0) - r.get("skipped", 0) for r in results),
                  skipped_tests=skipped, checks=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"checks", "source_audit", "source_sha256"}}), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
