"""Validate the isolated init0904 route without production assets or GPU jobs."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine"
PARENT_NAME = "mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small"
NAME = PARENT_NAME + "_init0904"
PARENT, ROUTE = FAMILY / PARENT_NAME, FAMILY / NAME
LAUNCHER_NAME = "run_mizar_s1_gbs256_recursive_5_10_5_ced_small_init0904_5090.sh"
LAUNCHER = ROOT / "code/RSmol" / LAUNCHER_NAME
OLD_LAUNCHER = "run_mizar_s1_gbs256_recursive_5_10_5_ced_small_5090.sh"
OLD_CHECKPOINT = "formal-epoch2-continue-20260902_184936/checkpoint-step-009244"
CHECKPOINT = "formal-maxlr8e-4-resume4500-20260904_154711/checkpoint-step-009244"
OLD_ID = "ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_v1"
ID = "ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_init0904_v1"
OLD_OUTPUT = "mizar_s1_ced_small_gbs256_recursive_5_10_5_5090"
OUTPUT = "mizar_s1_ced_small_gbs256_recursive_5_10_5_init0904_5090"
OLD_STAGE = "mizar_s1_gbs256_recursive_5_10_5_ced_small_${STAGE_TOKEN}"
STAGE = "mizar_s1_gbs256_recursive_5_10_5_ced_small_init0904_${STAGE_TOKEN}"


def read(path):
    return Path(path).read_text(encoding="utf-8-sig")


def contract_scope(path):
    scope = {"__file__": str(path)}
    exec(compile(read(path), str(path), "exec"), scope)
    return scope


NEW, OLD = contract_scope(ROUTE / "contract.py"), contract_scope(PARENT / "contract.py")


def bash_executable():
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    return str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")


class IsolationTests(unittest.TestCase):
    def test_parent_sources_unchanged(self):
        guard = json.loads(read(ROUTE / "source_guard.json"))
        self.assertEqual(len(guard["files"]), 38)
        for relative, expected in guard["files"].items():
            with self.subTest(file=relative):
                actual = hashlib.sha256((ROOT / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                self.assertEqual(actual, expected)

    def test_complete_copy_diff_allowlist(self):
        names = {p.relative_to(PARENT).as_posix() for p in PARENT.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts}
        for name in names:
            if name in {"README.md", "reference_audit.json"}:
                continue
            expected = read(PARENT / name)
            if name == "contract.py":
                expected = expected.replace(OLD_CHECKPOINT, CHECKPOINT).replace(OLD_ID, ID)
            elif name == "run.sh":
                expected = expected.replace(OLD_LAUNCHER, LAUNCHER_NAME).replace(OLD_STAGE, STAGE)
            elif name == "NOTICE":
                expected = expected.replace("recorded in reference_audit.json.",
                    "recorded in parent_reference_audit.json (historical parent audit).")
            with self.subTest(file=name):
                self.assertEqual(read(ROUTE / name), expected)
        self.assertEqual((PARENT / "reference_audit.json").read_bytes(),
                         (ROUTE / "parent_reference_audit.json").read_bytes())
        self.assertFalse((ROUTE / "reference_audit.json").exists())
        actual = {p.relative_to(ROUTE).as_posix() for p in ROUTE.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts}
        self.assertEqual(actual, names - {"reference_audit.json"} |
                         {"parent_reference_audit.json", "source_guard.json"})

    def test_entire_launcher_changes_only_identity_and_source(self):
        expected = read(ROOT / "code/RSmol" / OLD_LAUNCHER)
        expected = expected.replace(PARENT_NAME, NAME).replace(OLD_CHECKPOINT, CHECKPOINT)
        expected = expected.replace(OLD_OUTPUT, OUTPUT)
        expected = expected.replace("mizar_s1_gbs256_recursive_5_10_5_ced_small.",
                                    "mizar_s1_gbs256_recursive_5_10_5_ced_small_init0904.")
        expected = expected.replace("mizar-s1-ced-rec515-gbs256-${RUN_TAG}", "s1-ced-r515-i0904-${RUN_TAG}")
        expected = expected.replace("recursive 5-10-5 non-MeSH:", "recursive 5-10-5 non-MeSH init0904:")
        self.assertEqual(read(LAUNCHER), expected)
        self.assertIn("TEXT_MODEL_DIR=" + NEW["DEFAULT_TEXT_MODEL_DIR"], expected)
        self.assertIn(OUTPUT + "/index_cache", expected)

    def test_whole_trainer_loader_data_and_optimizer_identical(self):
        names = ["train.py", "optim.py", "write_config.py", "stage_data.py", "prepare_index.py",
                 "models/recursive_text.py", "models/decoder.py", "models/mellow.py", "models/ced.py"]
        names += [p.relative_to(PARENT).as_posix() for p in (PARENT / "data").glob("*.py")]
        for name in names:
            with self.subTest(file=name):
                self.assertEqual(ast.dump(ast.parse(read(PARENT / name))),
                                 ast.dump(ast.parse(read(ROUTE / name))))

    def test_entire_semantic_contract_and_all_lr_positions(self):
        prepared = dict(rows=1477492, source_rows=NEW["SOURCE_COUNTS"],
                        rows_canonical_sha256="synthetic_rows", mapping_sha256="synthetic_mapping")
        new = NEW["semantic_contract"](prepared, NEW["DEFAULT_TEXT_MODEL_DIR"], "ced")
        old = OLD["semantic_contract"](prepared, OLD["DEFAULT_TEXT_MODEL_DIR"], "ced")
        changes = {key for key in new if new[key] != old[key]}
        self.assertEqual(changes, {"route", "code_sha256", "text_model_dir"})
        self.assertEqual(new["route"], ID)
        self.assertEqual(new["geometry"], dict(rows=1477492, world_size=8, microbatch=8,
            gradient_accumulation_steps=4, global_batch_size=256, rank_samples_per_optimizer_step=32,
            samples_per_rank=184686, steps_per_epoch=5771, total_steps=17313,
            warmup_steps=866, dropped_rows_per_epoch=116))
        self.assertEqual(new["text_model_dir"], "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/" + CHECKPOINT)
        for step in range(17314):
            self.assertEqual(NEW["lr_factor"](step, 17313), OLD["lr_factor"](step, 17313))

    @staticmethod
    def initialization_scope(saved=None):
        """Execute the real optimizer/resume block, replacing only tensor dependencies."""
        function = next(n for n in ast.parse(read(ROUTE / "train.py")).body
                        if isinstance(n, ast.FunctionDef) and n.name == "train")
        start = next(i for i, n in enumerate(function.body)
                     if ast.unparse(n).startswith("optimizer, scheduler, groups ="))
        stop = next(i for i, n in enumerate(function.body)
                    if ast.unparse(n).startswith("checkpoint_dir ="))
        module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(function.body[start:stop]), type_ignores=[]))
        optimizer, scheduler, scaler, tracker, model = (Mock() for _ in range(5))
        scheduler.last_epoch = 5771
        torch = SimpleNamespace(load=Mock(return_value=saved) if saved else
            Mock(side_effect=AssertionError("fresh initialization must not deserialize training state")),
            cuda=SimpleNamespace(amp=SimpleNamespace(GradScaler=Mock(return_value=scaler))))
        scope = dict(torch=torch, model=model, geometry=dict(total_steps=17313, steps_per_epoch=5771),
            config=dict(resume_checkpoint="synthetic_audio.ckpt" if saved else "", contract_sha256="new-contract"),
            build_optimizer=Mock(return_value=(optimizer, scheduler, [])),
            GradNormTracker=Mock(return_value=tracker), ROUTE=ID, EPOCHS=3, rank=3,
            staging=SimpleNamespace(info=None))
        return module, scope

    def test_fresh_actual_control_flow_starts_at_zero_without_old_training_state(self):
        module, scope = self.initialization_scope()
        exec(compile(module, "actual-init0904-initialization", "exec"), scope)
        self.assertEqual((scope["step"], scope["start_epoch"], scope["start_batch"]), (0, 0, 0))
        self.assertIsNone(scope["resume_rng"])
        scope["torch"].load.assert_not_called()
        scope["build_optimizer"].assert_called_once_with(scope["model"], 17313)
        scope["optimizer"].load_state_dict.assert_not_called()

    @staticmethod
    def saved_state():
        return dict(schema_version=1, route_variant=ID, mode="formal", contract_sha256="new-contract",
                    state_dict={"lm": "synthetic trained weight"}, optimizer={"moments": "synthetic moments"},
                    scheduler={"last_epoch": 5771}, grad_scaler={"scale": 1}, grad_norm_tracker={"running": 2},
                    total_step=5771, epoch=1, next_optimizer_step=0,
                    random_state_by_rank=[{"rank": r} for r in range(8)])

    def test_actual_resume_rejects_parent_route_and_different_identity(self):
        for change in (dict(route_variant=OLD_ID), dict(contract_sha256="parent-contract"),
                       dict(schema_version=2), dict(mode="smoke")):
            with self.subTest(change=change):
                saved = {**self.saved_state(), **change}
                module, scope = self.initialization_scope(saved)
                with self.assertRaisesRegex(ValueError, "identical route/code/data/recipe"):
                    exec(compile(module, "actual-init0904-resume-rejection", "exec"), scope)
                scope["model"].load_state_dict.assert_not_called()

    def test_actual_full_resume_restores_all_states_and_selected_rank_rng(self):
        saved = self.saved_state()
        module, scope = self.initialization_scope(saved)
        exec(compile(module, "actual-init0904-resume", "exec"), scope)
        self.assertEqual((scope["step"], scope["start_epoch"], scope["start_batch"]), (5771, 1, 0))
        self.assertEqual(scope["resume_rng"], {"rank": 3})
        scope["model"].load_state_dict.assert_called_once_with(saved["state_dict"], strict=True)
        for name, key in (("optimizer", "optimizer"), ("scheduler", "scheduler"),
                          ("scaler", "grad_scaler"), ("tracker", "grad_norm_tracker")):
            scope[name].load_state_dict.assert_called_once_with(saved[key])

    def test_python_and_shell_syntax(self):
        for path in sorted(ROUTE.rglob("*.py")):
            compile(read(path), str(path), "exec")
        for path in sorted((ROOT / "tests").glob("*ced_small_init0904_*.py")):
            compile(read(path), str(path), "exec")
        bash = bash_executable()
        self.assertTrue(bash, "Bash is required to validate submission scripts")
        for path in (LAUNCHER, ROUTE / "run.sh"):
            result = subprocess.run([bash, "-n", path.as_posix()], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn(b"\r\n", path.read_bytes())
        for block in re.findall(r"```bash\n(.*?)```", read(ROUTE / "README.md"), re.S):
            result = subprocess.run([bash, "-n"], input=block, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensors", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "audit_reports/mizar_s1_ced_init0904_validation.json")
    args = parser.parse_args()
    start = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IsolationTests))
    checks = [dict(suite="isolation", run=result.testsRun, skipped=len(result.skipped),
                   failures=len(result.failures), errors=len(result.errors), passed=result.wasSuccessful())]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    bash = bash_executable()
    if bash:
        env["PATH"] = str(Path(bash).parent) + os.pathsep + env.get("PATH", "")
    for kind in ("route", "parity", "offline_config", "shared_start", "tensor"):
        path = ROOT / f"tests/test_mizar_s1_recursive_5_10_5_ced_small_init0904_{kind}.py"
        child = subprocess.run([sys.executable, "-B", str(path), "-v"], cwd=ROOT, env=env,
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        output = child.stdout + child.stderr
        print(output, flush=True)
        run = re.search(r"Ran (\d+) tests?", output)
        skipped = re.search(r"skipped=(\d+)", output)
        checks.append(dict(suite=kind, run=int(run[1]) if run else 0,
                           skipped=int(skipped[1]) if skipped else 0,
                           returncode=child.returncode, passed=child.returncode == 0 and bool(run)))
    tensor = next(item for item in checks if item["suite"] == "tensor")
    passed = all(item["passed"] for item in checks) and (
        not args.require_tensors or tensor["skipped"] == 0)
    report = dict(schema_version=1, status="PASS" if passed else "FAIL", route=ID,
                  text_initialization=NEW["DEFAULT_TEXT_MODEL_DIR"], require_tensors=args.require_tensors,
                  checks=checks, passed_tests=sum(item["run"] - item["skipped"] for item in checks if item["passed"]),
                  skipped_tests=sum(item["skipped"] for item in checks),
                  guarded_parent_files=38, scheduler_positions=17314,
                  code_sha256=NEW["code_digest"](), source_guard_sha256=hashlib.sha256((ROUTE / "source_guard.json").read_bytes()).hexdigest(),
                  scope="Local source, stdlib control-flow, synthetic concurrency and Bash mock tests; no production assets or GPU jobs.",
                  production_weights_loaded=False, gpu_training_verified=False, eight_rank_resume_verified=False,
                  seconds=round(time.monotonic() - start, 3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
