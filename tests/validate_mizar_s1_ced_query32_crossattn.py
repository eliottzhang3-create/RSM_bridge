"""Validate this isolated route and its parent, without assets, downloads or GPU jobs.

Use --require-tensors in remote mellow_c8204d8 to reject missing CPU tensor coverage.
The formal launcher does not execute this optional validation command.
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
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_query32_crossattn"
FILES = ["test_mizar_s1_ced_multilayer_crossattn_behavior.py",
         "test_mizar_s1_ced_multilayer_crossattn_route.py",
         "test_mizar_s1_ced_multilayer_crossattn_tensor.py",
"test_mizar_s1_ced_query32_crossattn_behavior.py",
         "test_mizar_s1_ced_query32_crossattn_route.py",
         "test_mizar_s1_ced_query32_crossattn_shared_start.py",
         "test_mizar_s1_ced_query32_crossattn_tensor.py",
         "test_mizar_s1_recursive_5_10_5_ced_small_route.py",
         "test_mizar_s1_recursive_5_10_5_ced_small_parity.py",
         "test_mizar_s1_recursive_5_10_5_ced_small_offline_config.py",
         "test_mizar_s1_recursive_5_10_5_ced_small_shared_start.py",
         "test_mizar_s1_recursive_5_10_5_ced_small_tensor.py"]


def test_result(name, completed):
    output = completed.stdout + completed.stderr
    count = re.search(r"Ran (\d+) tests?", output)
    skipped = re.search(r"skipped=(\d+)", output)
    tests, skips = int(count[1]) if count else 0, int(skipped[1]) if skipped else 0
    return dict(check=name, status="PASS" if completed.returncode == 0 and tests else "FAIL",
                tests=tests, skipped=skips, exit_code=completed.returncode, output=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "audit_reports/mizar_s1_ced_query32_crossattn_validation.json")
    args = parser.parse_args()
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else None
    if not bash or not bash.is_file():
        found = shutil.which("bash")
        bash = Path(found) if found else None
    if bash:
        environment["PATH"] = str(bash.parent) + os.pathsep + environment.get("PATH", "")
    results = []
    sources = list(ROUTE.rglob("*.py")) + [ROOT / "tests" / name for name in FILES] + [Path(__file__)]
    # Compile in memory so validation does not populate source directories with bytecode.
    for path in sources:
        compile(path.read_text(encoding="utf-8"),str(path),"exec")
    results.append(dict(check="python_compile", status="PASS", files=len(sources)))
    launcher = ROOT / "code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer36912_query32_crossattn_5090.sh"
    for path in (launcher,ROUTE / "run.sh"):
        if bash is None:
            results.append(dict(check=path.name,status="FAIL",error="Bash unavailable"))
            continue
        result = subprocess.run([str(bash),"-n",path.as_posix()],capture_output=True,text=True,
                                encoding="utf-8",errors="replace",env=environment,timeout=30)
        results.append(dict(check=path.name,status="PASS" if result.returncode==0 else "FAIL",
                            scope="bash syntax",output=result.stdout+result.stderr))

    def run(name):
        completed = subprocess.run([sys.executable,"-B",str(ROOT / "tests" / name),"-v"],
            cwd=ROOT,env=environment,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=300)
        return test_result(name,completed)

    # Separate processes isolate routes with the same Python module names.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(run,FILES):
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!="output"}),flush=True)
            if result["status"] == "FAIL":
                print(result["output"],flush=True)
    missing = [name for name in ("torch","torchaudio","transformers","soundfile","numpy")
               if importlib.util.find_spec(name) is None]
    tensor_results = [r for r in results if r["check"].endswith("_tensor.py")]
    tensor_pass = not missing and all(r["status"]=="PASS" and r["tests"]>0 and not r["skipped"]
                                      for r in tensor_results)
    if args.require_tensors:
        results.append(dict(check="required_tensor_coverage",status="PASS" if tensor_pass else "FAIL",
                            missing_dependencies=missing))
    failed = any(r["status"]=="FAIL" for r in results)
    skipped = sum(r.get("skipped",0) for r in results)
    report = dict(status="FAIL" if failed else "PASS_WITH_SKIPS" if skipped else "PASS",
        scope="local code validation; no real weights, remote data, CUDA or eight-rank integration",
        route=ROUTE.name,require_tensors=args.require_tensors,missing_dependencies=missing,
        real_cpu_tensor_checks_passed=tensor_pass,
        executed_tests=sum(r.get("tests",0)-r.get("skipped",0) for r in results),
        skipped_tests=skipped,checks=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="checks"}),flush=True)
    print(f"report={args.output}",flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
