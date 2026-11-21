"""Isolated stdlib checks; actual tensors and GPU are covered separately."""
import ast
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced_small"
ENTRY = ROOT / "code/RSmol/run_mizar_s1_gbs128_text71427_mesh_ced_small_3090.sh"
sys.path.insert(0, str(ROUTE))
import contract as C
import audit_reference as A
from models.recursive_text import validate_loading_info
from mizar_text_7_14x2_7_mesh.contract import architecture_fields, ARCHITECTURE, ROUTER_KEYS


def checkpoint(root):
    path = root / "checkpoint-009244"
    path.mkdir()
    config = dict(architecture_fields(), model_type="llama", hidden_size=576,
                  vocab_size=49152, tie_word_embeddings=True, eos_token_id=0, pad_token_id=0)
    C.write_json(path / "config.json", config)
    common = dict(architecture_contract=ARCHITECTURE, optimizer_step=9244)
    C.write_json(path / "checkpoint_complete.json", dict(common, status="complete", complete_marker=True))
    C.write_json(path / "checkpoint_manifest.json", dict(common, status="complete"))
    C.write_json(path / "mesh_checkpoint_metadata.json", dict(
        common, memory_slots=5, router_groups=3, router_module_count=6,
        logical_to_physical=list(C.TEXT_MODEL_SCHEDULE)))
    (path / "model.safetensors").write_bytes(b"synthetic; tensor loading deliberately not exercised here")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "training_state.pt").write_bytes(b"invalid pickle: must never deserialize")
    return path, config


def bash_path(path):
    value = Path(path).as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


class RouteTests(unittest.TestCase):
    def test_parent_sources_data_optimizer_and_loop_are_preserved(self):
        audit = A.audit_parent()
        self.assertEqual(len(audit["identical_files_normalized_lf"]), 15)
        self.assertEqual(audit["protected_source_files"], 35)

    def test_gbs128_horizon_lr_and_save_points(self):
        g = C.geometry(sum(C.SOURCE_COUNTS.values()))
        self.assertEqual((C.WORLD_SIZE, C.BATCH_SIZE, C.ACCUMULATION_STEPS, C.GLOBAL_BATCH_SIZE), (8, 4, 4, 128))
        self.assertEqual((g["steps_per_epoch"], g["total_steps"], g["warmup_steps"], g["dropped_rows_per_epoch"]),
                         (11542, 34626, 1731, 116))
        self.assertEqual([round(g["total_steps"] * i / 6) for i in range(1, 7)],
                         [5771, 11542, 17313, 23084, 28855, 34626])
        self.assertEqual(C.lr_factor(0, 34626), .1)
        self.assertEqual(C.lr_factor(1730, 34626), 1.)
        self.assertEqual(C.lr_factor(34626, 34626), .1)
        self.assertEqual(list(C.TEXT_MODEL_SCHEDULE), list(range(7)) + list(range(7, 21)) * 2 + list(range(21, 28)))

    def test_weights_only_checkpoint_does_not_read_old_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = checkpoint(Path(directory))
            before = (path / "training_state.pt").read_bytes()
            result = C.recursive_checkpoint_contract(path)
            self.assertEqual(result["initialization"]["source_checkpoint_step"], 9244)
            self.assertEqual(result["initialization"]["start_optimizer_step"], 0)
            self.assertFalse(result["initialization"]["restore_optimizer_scheduler_rng_cursor"])
            self.assertEqual(result["config"]["routers"], 6)
            self.assertEqual((path / "training_state.pt").read_bytes(), before)
            (path / "training_state.pt").unlink()
            self.assertEqual(C.recursive_checkpoint_contract(path), result)

    def test_wrong_architecture_schedule_and_incomplete_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, config = checkpoint(Path(directory))
            for key, wrong in (("recursive_layer_count", 20), ("num_hidden_layers", 30),
                               ("router_groups", 4), ("memory_slots", 7), ("hidden_size", 768),
                               ("logical_to_physical_schedule", list(range(42)))):
                with self.subTest(key=key):
                    C.write_json(path / "config.json", dict(config, **{key: wrong}))
                    with self.assertRaises(ValueError): C.recursive_checkpoint_contract(path)
            C.write_json(path / "config.json", config)
            C.write_json(path / "checkpoint_complete.json", dict(status="complete", complete_marker=True,
                         architecture_contract=ARCHITECTURE, optimizer_step=1000))
            with self.assertRaisesRegex(ValueError, "steps disagree"): C.recursive_checkpoint_contract(path)
            (path / "checkpoint_complete.json").unlink()
            with self.assertRaisesRegex(ValueError, "complete route checkpoint"): C.recursive_checkpoint_contract(path)

    def test_loading_info_rejects_missing_routers_extra_and_mismatched_tensors(self):
        validate_loading_info({})
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "strict"):
                validate_loading_info({key: ["model.write_routers.0.weight"]})

    def test_prompt_and_answer_keep_eos_and_original_tokenizer_arguments(self):
        scope = {}
        A.execute([A.find_function(A.tree(ROUTE / "data/audiotext_dataset.py"), "tokenize")], scope)
        class RecordingTokenizer:
            def encode_plus(self, **kwargs): return kwargs
        data = SimpleNamespace(tokenizer=RecordingTokenizer())
        for length in (256, 300):
            for text in ("hello", "hello " * 2000):
                result = scope["tokenize"](data, text, length)
                self.assertEqual(result, dict(text=text + " <|endoftext|>", add_special_tokens=True,
                                             max_length=length, padding="max_length", truncation=True,
                                             return_tensors="pt"))

    def test_center_crop_before_split_and_independent_real_pair(self):
        class Wave(list):
            def numel(self): return len(self)
            def contiguous(self): return self
            def __getitem__(self, key):
                result = super().__getitem__(key)
                return Wave(result) if isinstance(key, slice) else result
        rows = {"long": Wave(range(27)), "short": Wave(range(7)), "other": Wave(range(40, 55))}
        scope = dict(HALF_SAMPLES=10, load_mono=lambda p: rows[p], torch=SimpleNamespace(
            cat=lambda xs: Wave(sum((list(x) for x in xs), [])),
            nn=SimpleNamespace(functional=SimpleNamespace(pad=lambda x, pad: Wave(list(x) + [0] * pad[1])))))
        source = A.tree(ROUTE / "data/fixed20.py")
        A.execute([A.find_function(source, "fixed_len"), A.find_function(source, "load_fixed20")], scope)
        a, b = scope["load_fixed20"]("long")
        self.assertEqual(a + b, list(range(3, 23)))
        a, b = scope["load_fixed20"]("short")
        self.assertEqual(a + b, list(range(7)) + [0] * 13)
        a, b = scope["load_fixed20"]("long", "other")
        self.assertEqual(a, list(range(8, 18)))
        self.assertEqual(b, list(range(42, 52)))

    def test_resume_identity_excludes_job_staging_and_separates_architecture_batch(self):
        prepared = dict(rows=1477492, rows_canonical_sha256="rows", mapping_sha256="mapping",
                        source_rows=C.SOURCE_COUNTS, stage_root="job-a")
        first = C.semantic_contract(prepared, "text", "ced")
        prepared["stage_root"] = "job-b"
        self.assertEqual(first, C.semantic_contract(prepared, "text", "ced"))
        for key, wrong in (("route", "old_route"), ("text_model_architecture", "old_model"),
                           ("geometry", dict(first["geometry"], microbatch=8, global_batch_size=256))):
            self.assertNotEqual(C.digest(first), C.digest(dict(first, **{key: wrong})))
        self.assertEqual(first["prefix"], [126, 1, 256])

    def test_launcher_defaults_and_tmpfs_guard(self):
        source = ENTRY.read_text(encoding="utf-8")
        self.assertIn("TEXT_MODEL_DIR=" + C.DEFAULT_TEXT_MODEL, source)
        self.assertIn("INIT_CHECKPOINT=" + C.DEFAULT_INIT_CHECKPOINT, source)
        self.assertNotIn("stage4_5_10x2_5_mesh", source)
        runtime = (ROUTE / "run.sh").read_text(encoding="utf-8")
        self.assertIn("expected = 34626", runtime)
        self.assertIn('/dev/shm/mizar_s1_gbs256_text71427_mesh_ced_small_', runtime)
        scope = dict(Path=Path)
        A.execute([A.find_function(A.tree(ROUTE / "stage_data.py"), "validate_stage_root")], scope)
        if os.name != "nt":
            scope["validate_stage_root"](Path("/dev/shm/mizar_s1_gbs256_text71427_mesh_ced_small_unique"))

    def test_real_bash_launcher_argument_flow_and_early_rejections(self):
        bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
        if not bash or not Path(bash).is_file(): self.skipTest("Bash unavailable")
        python_stub = '''#!/usr/bin/env bash
set -eu
case "$1" in
  *check_text_assets.py) touch "$RSM_TEST_DIR/text_checked"; exit "$RSM_TEXT_FAIL" ;;
  *check_ced_assets.py) [[ -f "$RSM_TEST_DIR/text_checked" ]]; touch "$RSM_TEST_DIR/ced_checked"; exit "$RSM_ASSET_FAIL" ;;
  *audio_initialization.py) [[ -f "$RSM_TEST_DIR/ced_checked" ]]; touch "$RSM_TEST_DIR/init_checked"; exit "$RSM_INIT_FAIL" ;;
  *prepare_index.py)
    [[ -f "$RSM_TEST_DIR/ced_checked" ]]
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == --path-output ]]; then printf '%s\\n' "$RSM_TEST_DIR/index" > "$2"; break; fi
      shift
    done ;;
  *) exit 98 ;;
esac
'''
        for case in ("fresh", "resume", "bad-init", "bad-text", "bad-ced", "bad-workers", "nonempty"):
            with self.subTest(case=case), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                root = Path(directory)
                for name in ("bin", "text", "CED small", "index"): (root / name).mkdir()
                for name in ("text/config.json", "manifest", "resolved.tsv", "resume.ckpt", "init.ckpt"):
                    (root / name).write_text("fixture", encoding="utf-8")
                script = root / "submit.sh"
                script.write_text(ENTRY.read_text(encoding="utf-8").replace(
                    "MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json",
                    f'MANIFEST="{bash_path(root / "manifest")}"'), encoding="utf-8", newline="\n")
                for name, value in (("python", python_stub), ("vc", '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$RSM_TEST_DIR/vc_args"\n')):
                    p = root / "bin" / name
                    p.write_text(value, encoding="utf-8", newline="\n"); p.chmod(0o755)
                output = root / "output with spaces"
                if case == "nonempty":
                    output.mkdir(); (output / "prior.txt").write_text("preserve")
                args = ["--output-dir", bash_path(output), "--resolved-paths", bash_path(root / "resolved.tsv"),
                        "--text-model-dir", bash_path(root / "text"), "--ced-model-dir", bash_path(root / "CED small"),
                        "--index-cache-root", bash_path(root / "index"), "--stage-workers", "0" if case == "bad-workers" else "4",
                        "--init-checkpoint", bash_path(root / "init.ckpt")]
                if case == "resume": args += ["--resume-checkpoint", bash_path(root / "resume.ckpt")]
                env = dict(os.environ, RSM_TEST_DIR=bash_path(root), RSM_TEXT_FAIL="9" if case == "bad-text" else "0",
                           RSM_INIT_FAIL="8" if case == "bad-init" else "0",
                           RSM_ASSET_FAIL="7" if case == "bad-ced" else "0", PATH=f"{bash_path(root / 'bin')}:/usr/bin:/bin")
                result = subprocess.run([bash, bash_path(script), *args], env=env, capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", timeout=30)
                expected = 0 if case in ("fresh", "resume") else 9 if case == "bad-text" else 8 if case == "bad-init" else 7 if case == "bad-ced" else 2
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                capture = root / "vc_args"
                self.assertEqual(capture.exists(), expected == 0)
                if expected: continue
                submitted = capture.read_text().splitlines()
                for flag, value in (("-p", "pdgpu-3090"), ("-c", "64"), ("-m", "256G"), ("-g", "8"), ("-n", "1")):
                    self.assertEqual(submitted[submitted.index(flag) + 1], value)
                self.assertLess(len(submitted[submitted.index("-j") + 1]), 60)
                command = shlex.split(submitted[submitted.index("--cmd") + 1])
                self.assertEqual(command[:3], ["bash", f"mellow_official_training_c8204d8_adamw_cosine/{ROUTE.name}/run.sh", "formal"])
                self.assertEqual(command[4], bash_path(output))
                self.assertEqual(command[5], bash_path(root / "resume.ckpt") if case == "resume" else "")
                self.assertEqual(command[6:], [bash_path(root / "text"), bash_path(root / "CED small"), bash_path(root / "index"), "4", bash_path(root / "init.ckpt")])


if __name__ == "__main__": unittest.main()
