"""Stdlib-only regression checks for the isolated USAD2-Small training route."""
import ast
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_usad2_small"
PARENT = ROUTE.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh")
sys.path.insert(0, str(ROUTE))

import contract
import stage_data
import write_config


def make_assets(root):
    root.mkdir()
    for name in contract.USAD2_ASSET_FILES:
        (root / name).write_bytes(b"fixture")
    config = {**contract.USAD2_EXPECTED_CONFIG, "input_dim": 128,
              "use_framewise_subsample": True, "conv_subsample_rate": 2}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")


def bash_path(path):
    value = path.as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


class MeSHUSAD2RouteTests(unittest.TestCase):
    def test_gbs256_schedule_and_new_audio_sequence_shape(self):
        actual = contract.geometry(1477492)
        self.assertEqual(contract.ROUTE, "usad2_small_mizar_s1_fixed20_gbs256_recursive_mesh_v1")
        self.assertEqual((contract.WORLD_SIZE, contract.BATCH_SIZE, contract.ACCUMULATION_STEPS,
                          contract.GLOBAL_BATCH_SIZE), (8, 8, 4, 256))
        self.assertEqual((actual["steps_per_epoch"], actual["total_steps"], actual["warmup_steps"],
                          actual["dropped_rows_per_epoch"]), (5771, 17313, 866, 116))
        self.assertEqual((contract.USAD2_FRAMES, contract.USAD2_DIM, contract.AUDIO_GROUP_SIZE,
                          contract.AUDIO_TOKENS, contract.PREFIX_TOKENS,
                          contract.PREFIX_TOKENS + contract.ANSWER_TOKENS),
                         (999, 384, 4, 251, 508, 808))

    def test_usad2_asset_contract_is_offline_and_strict(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "USAD2"
            make_assets(root)
            result = contract.validate_usad2_assets(root)
            self.assertEqual(result["config"]["encoder_dim"], 384)
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["sample_rate"] = 32000
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompatible USAD2-Small"):
                contract.validate_usad2_assets(root)
            (root / "config.json").write_text(json.dumps({**contract.USAD2_EXPECTED_CONFIG,
                "input_dim": 128, "use_framewise_subsample": True, "conv_subsample_rate": 2}),
                encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"")
            with self.assertRaises(FileNotFoundError):
                contract.validate_usad2_assets(root)

    def test_recipe_keeps_mother_data_and_optimizer_contract(self):
        prepared = {"rows": 1477492, "rows_canonical_sha256": "rows", "mapping_sha256": "mapping",
                    "source_rows": contract.SOURCE_COUNTS, "stage_root": "one"}
        first = contract.semantic_contract(prepared, "text", "usad2")
        prepared["stage_root"] = "two"
        self.assertEqual(first, contract.semantic_contract(prepared, "text", "usad2"))
        self.assertEqual(first["geometry"], contract.geometry(1477492))
        self.assertEqual(first["usad2"]["frames_20s"], 999)
        self.assertEqual(first["usad2"]["partial_group_policy"], "mean_only_valid_frames")
        self.assertEqual(first["gradient_accumulation"]["steps"], 4)
        self.assertEqual(first["adamw"]["map_lr"], 6e-4)
        self.assertEqual(first["adamw"]["decoder_lr"], 3e-4)
        specification = importlib.util.spec_from_file_location("usad2_mother_contract", PARENT / "contract.py")
        mother = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(mother)
        previous = mother.semantic_contract(prepared, "text", "htsat")
        encoder_fields = {"route", "code_sha256", "audio_policy", "prefix",
                          "htsat_root", "usad2_root", "usad2"}
        self.assertEqual({key: value for key, value in first.items() if key not in encoder_fields},
                         {key: value for key, value in previous.items() if key not in encoder_fields})
        self.assertNotEqual(contract.digest(first), contract.digest(
            dict(first, route="htsat_mizar_s1_fixed20_gbs256_v1")))

    def test_raw_decode_staging_and_sampling_files_remain_byte_identical(self):
        unchanged = ["prepare_index.py", "stage_data.py", "utils/audio_io.py", "data/audio_paths.py",
                     "data/indexed_jsonl.py", "data/prepared_index.py", "data/sampler.py",
                     "data/staging.py", "data/template.py"]
        for name in unchanged:
            with self.subTest(file=name):
                self.assertEqual((ROUTE / name).read_bytes(), (PARENT / name).read_bytes())

    def test_audio_path_is_one_fixed20_tensor_and_explicit_local_loader(self):
        dataset = (ROUTE / "data/audiotext_dataset.py").read_text(encoding="utf-8")
        fixed20 = (ROUTE / "data/fixed20.py").read_text(encoding="utf-8")
        loader = (ROUTE / "models/usad2.py").read_text(encoding="utf-8")
        mellow = (ROUTE / "models/mellow.py").read_text(encoding="utf-8")
        self.assertIn('"audio": waveform', dataset)
        self.assertIn('"audio": torch.stack', dataset)
        self.assertIn("return whole.contiguous()", fixed20)
        self.assertIn("importlib.import_module(\"USAD2.modeling_usad2\")", loader)
        self.assertIn("use_safetensors=True", loader)
        self.assertIn("ignore_mismatched_sizes=False", loader)
        self.assertNotIn("AutoModel.from_pretrained", loader)
        self.assertIn("torchaudio.functional.resample", loader)
        self.assertIn("temporal_average_groups(projected[:, 1:], lengths)", mellow)
        self.assertIn("nn.Dropout(p)", mellow)

    def test_parameter_roles_include_only_projection_and_full_lm(self):
        tree = ast.parse((ROUTE / "optim.py").read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "parameter_role")
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "role", "exec"), namespace)
        role = namespace["parameter_role"]
        self.assertEqual(role("audio_encoder.projection.linear1.weight"), "map")
        self.assertEqual(role("caption_decoder.lm.model.layers.0.weight"), "default")
        for name in ("audio_encoder.base.usad2.encoder.weight", "audio_encoder.base.c2l.weight"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                role(name)

    def test_formal_submitter_uses_separate_route_assets_and_parent_resources(self):
        launcher = (ROOT / "code/RSmol/run_mizar_s1_gbs256_recursive_mesh_usad2_small_5090.sh").read_text(encoding="utf-8")
        runner = (ROUTE / "run.sh").read_text(encoding="utf-8")
        self.assertIn("--usad2-model-dir", launcher)
        self.assertIn("check_usad2_assets.py", launcher)
        self.assertIn("mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_usad2_small/run.sh", launcher)
        self.assertIn("pdgpu-5090", launcher)
        self.assertIn("-c 32 -m 256G -g 8 -n 1", launcher)
        self.assertIn("conda activate mellow_c8204d8", runner)
        self.assertIn("--usad2-model-dir", runner)
        self.assertIn("--nproc_per_node=8", runner)
        self.assertNotIn("HTSAT_ROOT", launcher + runner)
        self.assertNotIn("--mode smoke", launcher + runner)

    def test_staging_covers_whole_accumulation_groups(self):
        count = 2073
        priorities, initial = stage_data.epoch_priorities(count, permutation=list(range(count)))
        self.assertEqual(len(initial), contract.WORLD_SIZE)
        self.assertEqual(len(initial["0"]), 8 * contract.RANK_UPDATE_SAMPLES)
        self.assertTrue(all(isinstance(value, int) and value >= 0 for value in priorities))
        ordered, initial_steps, _ = stage_data.staging_plan(
            {"probe.wav": {"bytes": 1, "first_batch": -1},
             "first.wav": {"bytes": 2, "first_batch": 0}}, 5771, 0, 64)
        self.assertEqual(initial_steps, 64)
        self.assertEqual([name for name, _ in ordered], ["probe.wav", "first.wav"])


    def test_launcher_resources_arguments_and_rejections_before_submit(self):
        bash = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
        if not Path(bash).is_file():
            self.skipTest("Bash unavailable")
        launcher = (ROOT / "code/RSmol/run_mizar_s1_gbs256_recursive_mesh_usad2_small_5090.sh").read_text(encoding="utf-8")
        python_stub = r'''#!/usr/bin/env bash
set -eu
case "$1" in
  *check_usad2_assets.py)
    [[ "$2" == --usad2-model-dir && "$3" == "$RSM_TEST_DIR/USAD2" ]]
    touch "$RSM_TEST_DIR/assets_checked"
    exit "$RSM_ASSET_FAIL" ;;
  *check_text_assets.py)
    [[ "$2" == --text-model-dir && "$3" == "$RSM_TEST_DIR/text" ]]
    [[ "$4" == --comparison-text-model-dir && "$5" == "$RSM_TEST_DIR/comparison" ]]
    touch "$RSM_TEST_DIR/text_checked"
    exit "$RSM_TEXT_FAIL" ;;
  *prepare_index.py)
    [[ -f "$RSM_TEST_DIR/assets_checked" && -f "$RSM_TEST_DIR/text_checked" ]]
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == --path-output ]]; then printf '%s\n' "$RSM_TEST_DIR/index" > "$2"; break; fi
      shift
    done ;;
  *) exit 98 ;;
esac
'''
        vc_stub = '''#!/usr/bin/env bash
set -eu
printf '%s\\n' "$@" > "$RSM_TEST_DIR/vc_args"
'''
        for case in ("fresh", "resume", "bad-assets", "bad-text", "bad-workers", "nonempty"):
            with self.subTest(case=case), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                root = Path(directory)
                for name in ("bin", "text", "USAD2", "index", "comparison"):
                    (root / name).mkdir()
                for name in ("manifest", "resolved.tsv", "resume.ckpt"):
                    (root / name).write_text("fixture")
                script = root / "submit.sh"
                text = launcher.replace(
                    "MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json",
                    f'MANIFEST="{bash_path(root / "manifest")}"')
                script.write_text(text, encoding="utf-8", newline="\n")
                for name, value in (("python", python_stub), ("vc", vc_stub)):
                    path = root / "bin" / name
                    path.write_text(value, encoding="utf-8", newline="\n")
                    path.chmod(0o755)
                output = root / "output with spaces"
                if case == "nonempty":
                    output.mkdir()
                    (output / "prior.txt").write_text("preserve")
                args = ["--output-dir", bash_path(output), "--resolved-paths", bash_path(root / "resolved.tsv"),
                        "--text-model-dir", bash_path(root / "text"), "--usad2-model-dir", bash_path(root / "USAD2"),
                        "--comparison-text-model-dir", bash_path(root / "comparison"),
                        "--index-cache-root", bash_path(root / "index"), "--stage-workers", "0" if case == "bad-workers" else "4"]
                if case == "resume":
                    args.extend(("--resume-checkpoint", bash_path(root / "resume.ckpt")))
                env = dict(os.environ, RSM_TEST_DIR=bash_path(root), RSM_ASSET_FAIL="7" if case == "bad-assets" else "0",
                           RSM_TEXT_FAIL="8" if case == "bad-text" else "0",
                           PATH=f"{bash_path(root / 'bin')}:/usr/bin:/bin")
                result = subprocess.run([bash, bash_path(script), *args], env=env, capture_output=True,
                                        text=True, encoding="utf-8", errors="replace", timeout=20)
                expected = 0 if case in ("fresh", "resume") else 7 if case == "bad-assets" else 8 if case == "bad-text" else 2
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                captured = root / "vc_args"
                self.assertEqual(captured.exists(), expected == 0)
                if expected:
                    continue
                submitted = captured.read_text().splitlines()
                self.assertEqual(submitted[0], "submit")
                for flag, value in (("-p", "pdgpu-5090"), ("-c", "32"), ("-m", "256G"), ("-g", "8"), ("-n", "1")):
                    self.assertEqual(submitted[submitted.index(flag) + 1], value)
                self.assertLessEqual(len(submitted[submitted.index("-j") + 1]), 60)
                command = shlex.split(submitted[submitted.index("--cmd") + 1])
                self.assertEqual(command[:3], ["bash", "mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_usad2_small/run.sh", "formal"])
                self.assertEqual(command[4], bash_path(output))
                self.assertEqual(command[5], bash_path(root / "resume.ckpt") if case == "resume" else "")
                self.assertEqual(command[7:], [bash_path(root / "USAD2"), bash_path(root / "index"), "4", bash_path(root / "comparison")])


if __name__ == "__main__":
    unittest.main()
