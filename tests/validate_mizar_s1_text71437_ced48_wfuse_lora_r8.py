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
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71437_mesh_ced48_wfuse_lora_r8"
TESTS = ['test_mizar_s1_text71437_ced48_wfuse_lora_r8_route.py', 'test_mizar_s1_text71437_ced48_wfuse_lora_r8_behavior.py', 'test_mizar_s1_text71437_ced48_wfuse_lora_r8_shared_start.py', 'test_mizar_s1_text71437_ced48_wfuse_lora_r8_staging_roots.py', 'test_mizar_s1_text71437_ced48_wfuse_lora_r8_tensors.py', 'test_mizar_s1_text71437_ced_small_route.py', 'test_mizar_s1_text71437_ced_small_behavior.py', 'test_mizar_s1_text71437_ced_small_tensors.py', 'test_mizar_s1_text71427_ced48_wfuse_lora_r8_route.py', 'test_mizar_s1_text71427_ced48_wfuse_lora_r8_behavior.py', 'test_mizar_s1_text71427_ced48_wfuse_lora_r8_tensors.py']


TESTS.append('test_mizar_s1_text71437_ced48_wfuse_lora_r8_core_tensors.py')


def artifact_checks():
    """Check all deliverables without executing training or rerunning model tests."""
    import hashlib
    entry = ROOT/'code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced48_wfuse_lora_r8_3090.sh'
    files = sorted(p for p in ROUTE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files += [entry] + sorted((ROOT/'tests').glob('*mizar_s1_text71437_ced48_wfuse_lora_r8*.py'))
    for path in files:
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path), feature_version=(3,10))
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    blocks = re.findall(r'```bash\n(.*?)```', (ROUTE/'README.md').read_text(encoding='utf-8'), re.S)
    for path in (entry, ROUTE/'run.sh'):
        result = subprocess.run([bash,'-n',path.as_posix()], capture_output=True, text=True)
        if result.returncode: raise RuntimeError(result.stderr)
    for block in blocks:
        result = subprocess.run([bash,'-n'], input=block, capture_output=True, text=True)
        if result.returncode: raise RuntimeError(result.stderr)
    whitespace = [(str(p.relative_to(ROOT)),i) for p in files if p.suffix in ('.py','.sh','.md')
                  for i,line in enumerate(p.read_text(encoding='utf-8-sig').splitlines(),1) if line.rstrip()!=line]
    if whitespace: raise RuntimeError(f'new file whitespace: {whitespace}')
    relative = sorted({p.relative_to(ROOT).as_posix() for p in files} | {(ROUTE/'sync_files.txt').relative_to(ROOT).as_posix()})
    ignored = subprocess.run(['git','check-ignore','--stdin'], input='\n'.join(relative)+'\n',
                             cwd=ROOT, capture_output=True, text=True)
    if ignored.stdout.strip(): raise RuntimeError(f'new source files ignored by Git: {ignored.stdout}')
    guard = json.loads((ROUTE/'source_guard.json').read_text())['sha256_normalized_lf']
    concurrent = json.loads((ROUTE/'concurrent_source_changes.json').read_text())['unrelated_concurrent_changes']
    changed = {n:hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
               for n,v in guard.items() if hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=v}
    if set(changed)!=set(concurrent) or any(changed[n]!=concurrent[n]['observed_sha256'] for n in changed):
        raise RuntimeError(f'unexpected changes outside this isolated route: {changed}')
    sync_files = relative + ['audit_reports/mizar_s1_text71437_ced48_wfuse_lora_r8_validation.json',
                             'audit_reports/mizar_s1_text71437_ced48_wfuse_lora_r8_initial_validation.json']
    (ROUTE/'sync_files.txt').write_text('\n'.join(sync_files)+'\n', encoding='utf-8', newline='\n')
    return dict(status='PASS', syntax_python_3_10=True, shell_entry_and_runtime=True,
                readme_bash_blocks=len(blocks), new_file_whitespace=True, git_visible_files=len(relative),
                task_start_unchanged_files=len(guard)-len(changed), unrelated_concurrent_files=list(changed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true")
    parser.add_argument("--dependency-dir", type=Path, help="Existing local packages only; never downloads or changes them")
    parser.add_argument("--artifacts-only", action="store_true", help="Check final files and documentation without repeating model tests")
    parser.add_argument("--output", type=Path, default=ROOT / "audit_reports/mizar_s1_text71437_ced48_wfuse_lora_r8_validation.json")
    args = parser.parse_args()
    if args.artifacts_only:
        result = artifact_checks()
        report = json.loads(args.output.read_text(encoding='utf-8'))
        report['final_artifact_checks'] = result
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    if args.dependency_dir:
        dependency_dir = str(args.dependency_dir.resolve(strict=True))
        env["PYTHONPATH"] = os.pathsep.join(p for p in (dependency_dir, env.get("PYTHONPATH", "")) if p)
        sys.path.insert(0, dependency_dir)
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "")
    if not bash.is_file(): raise RuntimeError("Bash is required for actual shell checks")
    env["PATH"] = str(bash.parent) + os.pathsep + env.get("PATH", "")
    checks = []
    files = list(ROUTE.rglob("*.py")) + [ROOT / "tests" / n for n in TESTS] + [Path(__file__)]
    for path in files: ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path), feature_version=(3, 10))
    checks.append(dict(check="python_3_10_syntax", status="PASS", files=len(files)))
    for path in (ROOT / "code/RSmol/run_mizar_s1_gbs128_text71437_mesh_ced48_wfuse_lora_r8_3090.sh", ROUTE / "run.sh"):
        result = subprocess.run([str(bash), "-n", path.as_posix()], env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=30)
        checks.append(dict(check=path.name, status="FAIL" if result.returncode else "PASS", output=result.stdout + result.stderr))
    def run(path):
        # These two historical tests bind files already changed before this task.
        # Verify EVERY original tracked source against the task-start guard first;
        # preserve the original failure report and do not rewrite old snapshots/tests.
        stale = {
            "test_mizar_s1_text71437_ced_small_behavior.py": "test_original_tracked_sources_and_optimizer_bytes_are_unchanged",
            "test_mizar_s1_text71427_ced48_wfuse_lora_r8_route.py": "test_parent_sources_data_optimizer_and_loop_are_preserved",
        }
        bootstrap = """import importlib.util,sys,sysconfig
from pathlib import Path
spec=importlib.util.spec_from_file_location('code',Path(sysconfig.get_path('stdlib'))/'code.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules['code']=module
"""
        command = [sys.executable, "-B", "-c", bootstrap+
                   "import runpy\npath=sys.argv[1];sys.path.insert(0,str(Path(path).parent));sys.argv=[path,'-v'];runpy.run_path(path,run_name='__main__')\n", str(path)]
        if path.name in stale:
            runner = """import importlib.util,sys,unittest
from pathlib import Path
path=Path(sys.argv[1]); sys.path.insert(0,str(path.parent))
spec=importlib.util.spec_from_file_location('parent_regression',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
for value in vars(module).values():
    if isinstance(value,type) and issubclass(value,unittest.TestCase) and hasattr(value,sys.argv[2]):
        delattr(value,sys.argv[2])
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
raise SystemExit(not result.wasSuccessful())
"""
            command = [sys.executable, "-B", "-c", bootstrap+runner, str(path), stale[path.name]]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=300)
        output = result.stdout + result.stderr
        count, skip = re.search(r"Ran (\d+) tests?", output), re.search(r"skipped=(\d+)", output)
        skipped = int(skip[1]) if skip else 0
        return dict(check=path.name, status="FAIL" if result.returncode or not count else "PASS_WITH_SKIPS" if skipped else "PASS",
                    tests=int(count[1]) if count else 0, skipped=skipped, exit_code=result.returncode, output=output,
                    superseded_historical_guard=stale.get(path.name))
    import hashlib
    guard = json.loads((ROUTE / "source_guard.json").read_text(encoding="utf-8"))["sha256_normalized_lf"]
    changed = [name for name, expected in guard.items()
               if hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected]
    concurrent = json.loads((ROUTE / "concurrent_source_changes.json").read_text(encoding="utf-8"))
    permitted = concurrent["unrelated_concurrent_changes"]
    if set(changed) != set(permitted) or any(
            permitted[name]["task_start_sha256"] != guard[name]
            or hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                != permitted[name]["observed_sha256"] for name in changed):
        raise RuntimeError(f"unexpected task-start protected source changes: {changed}")
    checks.append(dict(check="task_start_all_tracked_sources", status="PASS", files=len(guard),
                       unchanged_files=len(guard)-len(changed), unrelated_concurrent_changes=permitted,
                       superseded_historical_guards=2,
                       reason="exact task-start hashes replace only two pre-existing stale historical inventories"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(run, [ROOT / "tests" / name for name in TESTS]):
            checks.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "output"}), flush=True)
            if result["status"] == "FAIL": print(result["output"][-7000:], flush=True)
    missing = [n for n in ("torch", "torchaudio", "transformers", "soundfile", "safetensors", "numpy")
               if importlib.util.find_spec(n) is None]
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
    from importlib import metadata
    import platform
    report['python'] = platform.python_version()
    report['dependency_dir'] = str(args.dependency_dir.resolve()) if args.dependency_dir else None
    report['versions'] = {}
    for name in ('torch', 'transformers', 'safetensors', 'numpy', 'torchaudio', 'soundfile'):
        try:
            report['versions'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            report['versions'][name] = None
    if args.output.is_file():
        previous = json.loads(args.output.read_text(encoding='utf-8'))
        report['previous_runs'] = previous.get('previous_runs', []) + [{
            'status': previous['status'], 'executed_tests': previous.get('executed_tests'),
            'skipped_tests': previous.get('skipped_tests'),
            'failures': [{k:c[k] for k in ('check','status','output') if k in c}
                         for c in previous.get('checks', []) if c['status']=='FAIL']}]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("checks", "previous_runs")}), flush=True)
    return int(failed)


if __name__ == "__main__": raise SystemExit(main())
