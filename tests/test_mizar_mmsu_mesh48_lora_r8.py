"""Exercise the fixed MeSH48 MMSU target through real submit and parse shells."""
from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

import test_mizar_s1_ced_mmsu_shell as harness
from mmsu_v2_audit import effective_benchmark_baseline

ROOT, RSMOL = harness.ROOT, harness.RSMOL
sys.path.insert(0, str(RSMOL))
from mizar_s1_ced_mmsu_eval import parsing, runtime

LAUNCHER = "run_mizar_mmsu_mesh48_lora_r8_5090.sh"
VARIANT = "recursive_mesh_layer4_8_inject_loop_lora_r8"
RUN_ROOT = ("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
            "mizar_s1_ced_small_gbs256_recursive_mesh_layer4_8_inject_loop_lora_r8_5090/"
            "formal_20261104_161727489909082")
ROUTE = "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject_loop_lora_r8"


class Mesh48MMSUTests(unittest.TestCase):
    setUp = harness.ShellTests.setUp

    def fixture(self, directory):
        path, rs, env = harness.ShellTests.fixture(self, directory)
        shutil.copy2(RSMOL / LAUNCHER, rs / LAUNCHER)
        # This target is bound by explicit arguments even with stale settings.
        env.update(RSMOL_MIZAR_CED_EVAL_VARIANT="base", RSMOL_MIZAR_CED_RUN_ROOT="/wrong",
                   RSMOL_MIZAR_CED_CHECKPOINT="/wrong.ckpt", RSMOL_MIZAR_CED_EXPECTED_STEP="1")
        return path, rs, env

    def launch(self, rs, env, args):
        return subprocess.run([self.bash, str(rs / LAUNCHER), *args], cwd=rs, env=env,
                              capture_output=True, text=True, encoding="utf-8", timeout=30)

    def assert_target(self, config):
        self.assertEqual(config["eval_variant"], VARIANT)
        for key, suffix in (("checkpoint", RUN_ROOT + "/formal/checkpoints/model--step-17313.ckpt"),
                            ("runtime_config", RUN_ROOT + "/formal/runtime.json"),
                            ("route_root", "/" + ROUTE)):
            self.assertTrue(config[key].replace("\\", "/").endswith(suffix), config[key])
        self.assertEqual(config["expected_step"], 17313)
        self.assertEqual(config["expected_geometry"], [8, 8, 4])

    def test_existing_mmau_mmar_adapter_is_used_with_original_mesh_and_lora_contract(self):
        adapter, family, route = runtime.adapter(VARIANT)
        self.assertIs(adapter, runtime.multilayer)
        self.assertEqual((family, route.name), ("multilayer", ROUTE))
        spec = adapter.variant_spec(VARIANT)
        self.assertEqual(spec["run_root"].as_posix(), RUN_ROOT)
        self.assertEqual((spec["default_step"], spec["default_geometry"]), (17313, (8, 8, 4)))
        self.assertEqual((spec["mesh"], spec["interaction_layers"], spec["context_kind"]),
                         (True, (4, 8), "pair"))
        self.assertEqual((spec["memory_slots"], spec["loops"]), (5, 2))
        self.assertEqual(spec["loop_lora_config"]["rank"], 8)
        self.assertEqual(spec["loop_lora_config"]["parameter_count"], 1628160)
        self.assertEqual(spec["new_parameters"], 4073090)
        self.assertEqual(runtime.PROTOCOL["scoring"],
                         parsing.VERSION + "_then_pinned_external_official_scorer")

    def test_full_submission_binds_checkpoint_and_preserves_resources_and_job_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            result = self.launch(rs, env, [])
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(result.stdout.splitlines()[0])
            self.assert_target(config)
            self.assertTrue(config["output_dir"].replace("\\", "/").endswith(RUN_ROOT + "/eval/mmsu_step17313"))
            self.assertTrue(config["dataset_dir"].replace("\\", "/").endswith("/data/MMSU/MMSU data"))
            self.assertTrue(config["evaluation_script"].replace("\\", "/").endswith("/data/MMSU/MMSU code/mmsu_evaluation.py"))
            submission = (path / "submission.bin").read_bytes().decode().split("\0")[:-1]
            self.assertEqual(submission.count("submit"), 1)
            for option, value in (("-p", "pdgpu-5090"), ("-c", "8"), ("-m", "32G"),
                                  ("-g", "1"), ("-n", "1"),
                                  ("-i", "docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1")):
                self.assertEqual(submission[submission.index(option) + 1], value)
            command = shlex.split(submission[submission.index("--cmd") + 1])
            self.assertEqual(command[:2], ["bash", "mizar_s1_ced_mmsu_eval/run.sh"])
            self.assertNotIn("--check-args", command)
            compute = subprocess.run([self.bash, str(rs / command[1]), "--check-args", *command[2:]],
                                     cwd=rs, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(compute.returncode, 0, compute.stderr)
            self.assert_target(json.loads(compute.stdout))
            self.assertEqual((path / "environment.bin").read_bytes().split(b"\0")[:-1],
                             [b"1", b"1", b"false", b"utf-8"])

    def test_check_args_and_help_do_not_submit(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            checked = self.launch(rs, env, ["--check-args"])
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assert_target(json.loads(checked.stdout))
            for option in ("--help", "-h"):
                result = self.launch(rs, env, [option])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("step17313", result.stdout)
            self.assertFalse((path / "submission.bin").exists())

    def test_io_overrides_with_spaces_and_quotes_keep_fixed_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            mmsu = "/shared data's path/MMSU"
            output = "/new output's path/mmsu"
            result = self.launch(rs, env, ["--mmsu-root", mmsu, "--output-dir=" + output,
                                          "--parquet-batch-size=3"])
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(result.stdout.splitlines()[0])
            self.assert_target(config)
            self.assertEqual(config["parquet_batch_size"], 3)
            self.assertTrue(config["output_dir"].replace("\\", "/").endswith(output))
            self.assertTrue(config["dataset_dir"].replace("\\", "/").endswith(mmsu + "/MMSU data"))
            submission = (path / "submission.bin").read_bytes().decode().split("\0")[:-1]
            command = shlex.split(submission[submission.index("--cmd") + 1])
            self.assertIn(mmsu, command)
            self.assertIn("--output-dir=" + output, command)

    def test_bad_options_and_model_overrides_fail_before_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            for args in (["--checkpoint", "/wrong"], ["--eval-variant=base"],
                         ["--runtime-config", "/wrong"], ["--expected-step=1"],
                         ["--route-root=/wrong"], ["--expected-geometry=1,1,1"],
                         ["--mmsu-root"], ["--output-dir="], ["--parquet-batch-size=0"]):
                with self.subTest(args=args):
                    result = self.launch(rs, env, args)
                    self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse((path / "submission.bin").exists())

    def test_existing_benchmark_and_training_sources_unchanged(self):
        for path, digest in effective_benchmark_baseline(ROOT).items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)
        audit = json.loads((RSMOL / "mizar_selected4_eval/reference_audit.json").read_text(encoding="utf-8"))
        for path, digest in audit["preserved_training_lf_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_text(encoding="utf-8").encode()).hexdigest(), digest, path)

    def test_new_shell_syntax(self):
        result = subprocess.run([self.bash, "-n", str(RSMOL / LAUNCHER)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
