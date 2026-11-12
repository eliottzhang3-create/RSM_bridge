"""Exercise selected checkpoints through both real shells and argument parser."""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import test_mizar_s1_ced_mmsu_shell as harness

RSMOL = harness.RSMOL
sys.path.insert(0, str(RSMOL))
from mizar_s1_ced_mmsu_eval import runtime
OUTPUT_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol"
MODEL_DIR = "/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M"
LAUNCHER = "run_mizar_mmsu_selected5_5090.sh"
TRAINED = {
    "base": ("base", "mizar_s1_ced_small_gbs256_5090/formal_20261026_193830617147305",
             "mizar_stage1_fixed20_gbs256_3epochs_ced_small", "base"),
    "recursive_5_10_5": ("recursive_5_10_5",
             "mizar_s1_ced_small_gbs256_recursive_5_10_5_5090/formal_20261028_160823559722647",
             "mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small", "recursive"),
    "serial84_lora_r8": ("recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8",
             "mizar_s1_ced_small_gbs256_recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8_5090/formal_20261103_165410438279511",
             "mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8", "multilayer"),
    "recursive_mesh": ("recursive_mesh",
             "mizar_s1_ced_small_gbs256_recursive_mesh_5090/formal_20261027_081123150934256",
             "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small", "mesh"),
}
TARGETS = ("official_mizar", *TRAINED)


class SelectedTargetTests(unittest.TestCase):
    setUp = harness.ShellTests.setUp

    def fixture(self, directory):
        result = harness.ShellTests.fixture(self, directory)
        shutil.copy2(RSMOL / LAUNCHER, result[1] / LAUNCHER)
        return result

    def launch(self, rs, env, args):
        return subprocess.run([self.bash, str(rs / LAUNCHER), *args], cwd=rs, env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)

    def test_all_five_exact_bindings_resources_and_real_parser(self):
        outputs = set()
        for target in TARGETS:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                path, rs, env = self.fixture(directory)
                result = self.launch(rs, env, [target])
                self.assertEqual(result.returncode, 0, result.stderr)
                # Generic submitter prints the actual CPU-parsed configuration.
                config = json.loads(result.stdout.splitlines()[0])
                self.assertTrue(config["check_args"])  # submit-side CPU parse only
                submission = (path / "submission.bin").read_bytes().decode().split("\0")[:-1]
                for key, expected in (("-p", "pdgpu-5090"), ("-c", "8"), ("-m", "32G"),
                        ("-g", "1"), ("-n", "1"), ("-i", "docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1")):
                    self.assertEqual(submission[submission.index(key) + 1], expected)
                self.assertEqual(submission.count("submit"), 1)
                command = shlex.split(submission[submission.index("--cmd") + 1])
                self.assertEqual(command[:2], ["bash", "mizar_s1_ced_mmsu_eval/run.sh"])
                options = dict(zip(command[2::2], command[3::2]))
                outputs.add(options["--output-dir"])
                self.assertEqual(config["eval_variant"], options["--eval-variant"])
                if target == "official_mizar":
                    self.assertEqual(options["--model-dir"], MODEL_DIR)
                    self.assertEqual(options["--checkpoint"], MODEL_DIR + "/checkpoints/final/stage3_q4_seed20260905.ckpt")
                    self.assertEqual(options["--output-dir"], OUTPUT_ROOT + "/mizar_official_stage3_q4_seed20260905_matched_ced_v1/mmsu")
                    for forbidden in ("--runtime-config", "--route-root", "--expected-step", "--expected-geometry"):
                        self.assertNotIn(forbidden, options)
                    self.assertIsNone(config["runtime_config"])
                else:
                    variant, run, route, family = TRAINED[target]
                    root = OUTPUT_ROOT + "/" + run
                    self.assertEqual(options["--eval-variant"], variant)
                    self.assertEqual(options["--checkpoint"], root + "/formal/checkpoints/model--step-17313.ckpt")
                    self.assertEqual(options["--runtime-config"], root + "/formal/runtime.json")
                    self.assertTrue(options["--route-root"].endswith("/" + route))
                    self.assertEqual(options["--output-dir"], root + "/eval/mmsu_step17313")
                    self.assertEqual(options["--expected-step"], "17313")
                    self.assertEqual(options["--expected-geometry"], "8,8,4")
                    self.assertEqual(config["expected_geometry"], [8, 8, 4])
                    _, actual_family, actual_route = runtime.adapter(variant)
                    self.assertEqual(actual_family, family)
                    self.assertEqual(actual_route.name, route)
        self.assertEqual(len(outputs), 5)

    def test_check_args_for_each_target_never_calls_vc(self):
        for target in TARGETS:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                path, rs, env = self.fixture(directory)
                result = self.launch(rs, env, [target, "--check-args"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(json.loads(result.stdout)["check_args"])
                self.assertFalse((path / "submission.bin").exists())

    def test_io_overrides_keep_model_identity_and_quote_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            root = "/shared data's path/MMSU"
            output = "/new output's path/mmsu"
            result = self.launch(rs, env, ["serial84_lora_r8", "--mmsu-root", root,
                                          "--output-dir=" + output, "--parquet-batch-size=3"])
            self.assertEqual(result.returncode, 0, result.stderr)
            submission = (path / "submission.bin").read_bytes().decode().split("\0")[:-1]
            args = shlex.split(submission[submission.index("--cmd") + 1])
            self.assertIn(root, args)
            self.assertIn("--output-dir=" + output, args)
            config = json.loads(result.stdout.splitlines()[0])
            self.assertEqual(config["parquet_batch_size"], 3)
            self.assertTrue(config["output_dir"].replace("\\", "/").endswith(output))
            self.assertTrue(config["dataset_dir"].replace("\\", "/").endswith(root + "/MMSU data"))
            self.assertTrue(config["evaluation_script"].replace("\\", "/").endswith(root + "/MMSU code/mmsu_evaluation.py"))
            self.assertEqual(config["expected_step"], 17313)
            self.assertEqual(config["eval_variant"], TRAINED["serial84_lora_r8"][0])

    def test_unknown_target_missing_value_and_model_overrides_fail_before_vc(self):
        bad = [[], ["unknown"], ["base", "--mmsu-root"], ["base", "--output-dir="],
               ["base", "--checkpoint", "/wrong.ckpt"], ["base", "--eval-variant=recursive_mesh"],
               ["base", "--expected-step=1"], ["base", "--runtime-config", "/wrong.json"]]
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            for args in bad:
                with self.subTest(args=args):
                    result = self.launch(rs, env, args)
                    self.assertEqual(result.returncode, 2)
                    self.assertFalse((path / "submission.bin").exists())

    def test_official_pin_matches_existing_mmau_mmar_release_manifest(self):
        release = json.loads((RSMOL / "mizar_official_eval/release_manifest.json").read_text())
        self.assertEqual(release["checkpoint"]["path"], "checkpoints/final/stage3_q4_seed20260905.ckpt")
        self.assertEqual(release["release"]["stage"], 3)
        self.assertEqual(release["release"]["recipe"], "Q4")
        self.assertEqual(release["release"]["seed"], 20260905)


if __name__ == "__main__":
    unittest.main()
