"""CPU-only contracts for Mizar Stage 1 benchmark evaluation."""
from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code" / "RSmol" / "scripts"
SOURCE = SCRIPTS / "evaluate_mizar_s1_benchmark.py"
RUNTIME_SOURCE = SCRIPTS / "mizar_s1_eval_runtime.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MizarS1BenchmarkEvalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("evaluate_mizar_s1_benchmark", SOURCE)
        cls.runtime = load_module("mizar_s1_eval_runtime", RUNTIME_SOURCE)

    def test_fixed_protocol_is_training_aligned_and_raw(self):
        protocol = self.runtime.PROTOCOL
        self.assertEqual(protocol["prefix_layout"], [129, 129, 1, 256])
        self.assertEqual(protocol["prefix_tokens"], 515)
        self.assertEqual(protocol["answer_start_position"], 515)
        self.assertEqual(protocol["first_answer_logit_position"], 514)
        self.assertEqual(protocol["max_new_tokens"], 300)
        self.assertFalse(protocol["do_sample"])
        self.assertIsNone(protocol["attention_mask"])
        self.assertEqual(protocol["prediction_preprocessing"],
                         "none_including_eos_no_strip_no_label_extraction")

    def test_runtime_tuple_contract_has_loader_audit_and_route_modules(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "model, tokenizer, device, text_encoder, _, _, _ = runtime",
            source,
        )
        runtime_source = RUNTIME_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "return model, tokenizer, device, text_encoder, modules[\"data.fixed20\"], audit, modules",
            runtime_source,
        )

    def test_benchmark_defaults_use_separate_full_evaluation_directories(self):
        mmau = self.module.parse_args(["--benchmark", "mmau"])
        mmar = self.module.parse_args(["--benchmark", "mmar"])
        self.assertEqual(mmau.parquet.name, "test_mini.parquet")
        self.assertEqual(mmau.metadata_json.name, "mmau-test-mini.json")
        self.assertEqual(mmau.audio_root.name, "test-mini-audios")
        self.assertEqual(mmar.metadata_json.name, "MMAR-meta.json")
        self.assertEqual(mmar.audio_root.name, "mmar-audio")
        self.assertIn("mmau_test_mini_step122937", str(mmau.output_dir))
        self.assertIn("mmar_step122937", str(mmar.output_dir))

    def test_gbs256_checkpoint_contract_is_selectable(self):
        args = self.module.parse_args([
            "--benchmark", "mmau", "--expected-step", "17313",
            "--expected-geometry", "8,8,4",
        ])
        self.assertEqual(args.expected_step, 17313)
        self.assertEqual(args.expected_geometry, (8, 8, 4))
        self.assertIn("mmau_test_mini_step17313", str(args.output_dir))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact"
            source.touch()
            for key in ("checkpoint", "runtime_config", "metadata_json", "parquet", "evaluation_script"):
                setattr(args, key, source)
            identity = self.module._identity(args, "model_output")
        self.assertEqual(identity["expected_step"], 17313)
        self.assertEqual(identity["expected_geometry"], [8, 8, 4])

    def checkpoint_fixture(self, variant):
        folder = ("mizar_stage1_fixed20_gbs256_3epochs" if variant == "gbs256"
                  else "mizar_stage1_fixed20_3epochs")
        route = load_module("test_contract_" + variant, ROOT / "code/RSmol" /
                            "mellow_official_training_c8204d8_adamw_cosine" / folder / "contract.py")
        contract = route.semantic_contract({
            "rows": (1477492 if variant == "gbs256" else 1311334), "source_rows": route.SOURCE_COUNTS,
            "rows_canonical_sha256": "manifest", "mapping_sha256": "mapping",
        }, "text-model", "htsat")
        identity = route.digest(contract)
        saved = {
            "schema_version": 1, "route_variant": route.ROUTE, "mode": "formal",
            "state_dict": {"test_parameter": [1]}, "contract": contract,
            "contract_sha256": identity, "total_step": contract["geometry"]["total_steps"],
            "num_epochs": 3,
            "loss_reduction": (
                "mean_of_rank_local_microbatch_token_means_then_DDP_mean"
                if variant == "gbs256" else "rank_local_token_mean_then_DDP_mean"),
        }
        runtime = {"contract": contract, "contract_sha256": identity,
                   "route": route.ROUTE, "mode": "formal"}
        return saved, runtime, route

    def test_both_checkpoint_schemas_validate_and_cross_route_use_fails(self):
        for variant, geometry, step in (("gbs32", (8, 4, 1), 122937),
                                        ("gbs256", (8, 8, 4), 17313)):
            with self.subTest(variant=variant):
                saved, runtime, route = self.checkpoint_fixture(variant)
                contract = self.runtime.validate_checkpoint(saved, runtime, route, step, geometry)
                self.assertEqual(contract["geometry"]["total_steps"], step)
                wrong_geometry = (8, 8, 4) if variant == "gbs32" else (8, 4, 1)
                with self.assertRaisesRegex(ValueError, "expected training geometry"):
                    self.runtime.validate_checkpoint(saved, runtime, route, step, wrong_geometry)
                with self.assertRaisesRegex(ValueError, "step / epoch horizon"):
                    self.runtime.validate_checkpoint(saved, runtime, route, step + 1, geometry)
                other = self.checkpoint_fixture("gbs32" if variant == "gbs256" else "gbs256")[2]
                with self.assertRaisesRegex(ValueError, "complete formal"):
                    self.runtime.validate_checkpoint(saved, runtime, other, step, geometry)

    def test_gbs256_keeps_source_runtime_and_loss_checks(self):
        saved, runtime, route = self.checkpoint_fixture("gbs256")
        with patch.object(route, "code_digest", return_value="different-source"):
            with self.assertRaisesRegex(ValueError, "training source differs"):
                self.runtime.validate_checkpoint(saved, runtime, route, 17313, (8, 8, 4))
        with self.assertRaisesRegex(ValueError, "runtime.json does not belong"):
            self.runtime.validate_checkpoint(saved, {**runtime, "contract_sha256": "other"},
                                             route, 17313, (8, 8, 4))
        saved["loss_reduction"] = "rank_local_token_mean_then_DDP_mean"
        with self.assertRaisesRegex(ValueError, "training loss contract mismatch"):
            self.runtime.validate_checkpoint(saved, runtime, route, 17313, (8, 8, 4))

    def test_scorer_key_is_taken_from_official_script(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.py"
            path.write_text("output_key = 'answer_prediction'\n", encoding="utf-8")
            self.assertEqual(self.module.scorer_prediction_key(path), "answer_prediction")
            path.write_text("output_key = 'model_output'\n", encoding="utf-8")
            self.assertEqual(self.module.scorer_prediction_key(path), "model_output")
            path.write_text("output_key = 'selected_option'\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.module.scorer_prediction_key(path)

    def test_materialization_preserves_raw_text_and_official_order(self):
        records = [
            {"id": "a", "answer": "one", "choices": ["one", "two"]},
            {"id": "b", "answer": "two", "choices": ["one", "two"]},
        ]
        raw = "  b) two <|endoftext|>  "
        state = [{"id": "b", "row_index": 1, "status": "generated", "generated_text": raw}]
        result = self.module._materialize(state, records, "answer_prediction", "mmar")
        self.assertEqual([row["id"] for row in result], ["a", "b"])
        self.assertEqual(result[1]["answer_prediction"], raw)
        self.assertEqual(result[0]["answer_prediction"], "")

    def test_output_identity_refuses_cross_checkpoint_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = type("Args", (), {"output_dir": output})()
            self.module.claim_output(args, {"checkpoint": "one"})
            self.module.claim_output(args, {"checkpoint": "one"})
            with self.assertRaises(RuntimeError):
                self.module.claim_output(args, {"checkpoint": "two"})

    def test_mmau_audio_directory_is_optional_but_mmar_audio_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.ckpt"
            runtime = root / "runtime.json"
            route = root / "route"
            metadata = root / "metadata.json"
            scorer = root / "evaluation.py"
            parquet = root / "test_mini.parquet"
            audio_root = root / "missing-audios"
            checkpoint.touch()
            runtime.touch()
            route.mkdir()
            metadata.touch()
            scorer.touch()
            parquet.touch()
            args = type("Args", (), {
                "benchmark": "mmau", "checkpoint": checkpoint,
                "runtime_config": runtime, "route_root": route,
                "metadata_json": metadata, "evaluation_script": scorer,
                "parquet": parquet, "audio_root": audio_root,
            })()
            self.module._validate_input_paths(args)

            args.benchmark = "mmar"
            with self.assertRaisesRegex(FileNotFoundError, "audio root not found"):
                self.module._validate_input_paths(args)

    def test_submission_wrappers_request_one_gpu_and_official_full_runner(self):
        mmau_submit = (ROOT / "code/RSmol/run_mizar_s1_mmau_eval_5090.sh").read_text(encoding="utf-8")
        mmar_submit = (ROOT / "code/RSmol/run_mizar_s1_mmar_eval_5090.sh").read_text(encoding="utf-8")
        inner = (SCRIPTS / "evaluate_mizar_s1_benchmark.sh").read_text(encoding="utf-8")
        for content, benchmark in ((mmau_submit, "mmau"), (mmar_submit, "mmar")):
            self.assertIn("-p pdgpu-5090", content)
            self.assertIn("-g 1", content)
            self.assertIn(f"--benchmark {benchmark}", content)
            self.assertIn("model--step-122937.ckpt", content)
            self.assertIn("--expected-step", content)
            self.assertIn("--expected-geometry", content)
            self.assertIn("vc submit", content)
        self.assertIn("RSMOL_MIZAR_S1_EVAL_VARIANT", mmau_submit)
        self.assertIn("model--step-17313.ckpt", mmau_submit)
        self.assertIn("RSMOL_MIZAR_S1_EVAL_VARIANT", mmar_submit)
        self.assertIn("model--step-17313.ckpt", mmar_submit)
        self.assertIn("conda activate mellow_c8204d8", inner)

    def test_submission_selects_checkpoint_runtime_route_and_geometry_together(self):
        bash = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
        if not Path(bash).is_file():
            self.skipTest("Bash is not installed")

        def bash_path(path):
            value = path.resolve().as_posix()
            return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submit = root / "run_mizar_s1_mmau_eval_5090.sh"
            submit.write_bytes((ROOT / "code/RSmol" / submit.name).read_bytes().replace(b"\r\n", b"\n"))
            binary = root / "bin"
            binary.mkdir()
            stub = binary / "vc"
            stub.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$RSM_TEST_CAPTURE"\n',
                            encoding="utf-8", newline="\n")
            stub.chmod(0o755)
            clean_env = {key: value for key, value in os.environ.items()
                         if not key.startswith("RSMOL_")}
            for variant, run, step, geometry in (
                ("gbs32", "mizar_s1_htsat_5090/formal_async_v3", "122937", "8,4,1"),
                ("gbs256", "mizar_s1_htsat_gbs256_5090/formal_hybrid_gbs256_v1", "17313", "8,8,4"),
            ):
                with self.subTest(variant=variant):
                    capture = root / f"{variant}.args"
                    env = dict(clean_env, RSMOL_MIZAR_S1_EVAL_VARIANT=variant,
                               RSM_TEST_CAPTURE=bash_path(capture),
                               PATH=f"{bash_path(binary)}:/usr/bin:/bin")
                    subprocess.run([bash, bash_path(submit)], env=env,
                                   capture_output=True, text=True, check=True)
                    submission = capture.read_text(encoding="utf-8").splitlines()
                    self.assertEqual(submission[submission.index("-g") + 1], "1")
                    command = shlex.split(submission[submission.index("--cmd") + 1])
                    self.assertEqual(command[:2], ["bash", "scripts/evaluate_mizar_s1_benchmark.sh"])
                    options = dict(zip(command[2::2], command[3::2]))
                    self.assertTrue(options["--checkpoint"].endswith(
                        f"{run}/formal/checkpoints/model--step-{step}.ckpt"))
                    self.assertTrue(options["--runtime-config"].endswith(f"{run}/formal/runtime.json"))
                    self.assertTrue(options["--output-dir"].endswith(f"{run}/eval/mmau_test_mini_step{step}"))
                    route = "mizar_stage1_fixed20_gbs256_3epochs" if variant == "gbs256" else "mizar_stage1_fixed20_3epochs"
                    self.assertTrue(options["--route-root"].endswith(route))
                    self.assertEqual(options["--expected-step"], step)
                    self.assertEqual(options["--expected-geometry"], geometry)

    def test_mmar_submission_selects_checkpoint_runtime_route_and_geometry_together(self):
        bash = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
        if not Path(bash).is_file():
            self.skipTest("Bash is not installed")

        def bash_path(path):
            value = path.resolve().as_posix()
            return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submit = root / "run_mizar_s1_mmar_eval_5090.sh"
            submit.write_bytes((ROOT / "code/RSmol" / submit.name).read_bytes().replace(b"\r\n", b"\n"))
            binary = root / "bin"
            binary.mkdir()
            stub = binary / "vc"
            stub.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$RSM_TEST_CAPTURE"\n',
                            encoding="utf-8", newline="\n")
            stub.chmod(0o755)
            clean_env = {key: value for key, value in os.environ.items()
                         if not key.startswith("RSMOL_")}
            for variant, run, step, geometry in (
                ("gbs32", "mizar_s1_htsat_5090/formal_async_v3", "122937", "8,4,1"),
                ("gbs256", "mizar_s1_htsat_gbs256_5090/formal_hybrid_gbs256_v1", "17313", "8,8,4"),
            ):
                with self.subTest(variant=variant):
                    capture = root / f"mmar_{variant}.args"
                    env = dict(clean_env, RSMOL_MIZAR_S1_EVAL_VARIANT=variant,
                               RSM_TEST_CAPTURE=bash_path(capture),
                               PATH=f"{bash_path(binary)}:/usr/bin:/bin")
                    subprocess.run([bash, bash_path(submit)], env=env,
                                   capture_output=True, text=True, check=True)
                    submission = capture.read_text(encoding="utf-8").splitlines()
                    self.assertEqual(submission[submission.index("-g") + 1], "1")
                    command = shlex.split(submission[submission.index("--cmd") + 1])
                    self.assertEqual(command[:2], ["bash", "scripts/evaluate_mizar_s1_benchmark.sh"])
                    options = dict(zip(command[2::2], command[3::2]))
                    self.assertTrue(options["--checkpoint"].endswith(
                        f"{run}/formal/checkpoints/model--step-{step}.ckpt"))
                    self.assertTrue(options["--runtime-config"].endswith(f"{run}/formal/runtime.json"))
                    self.assertTrue(options["--output-dir"].endswith(f"{run}/eval/mmar_step{step}"))
                    route = "mizar_stage1_fixed20_gbs256_3epochs" if variant == "gbs256" else "mizar_stage1_fixed20_3epochs"
                    self.assertTrue(options["--route-root"].endswith(route))
                    self.assertEqual(options["--expected-step"], step)
                    self.assertEqual(options["--expected-geometry"], geometry)

    def test_inference_path_has_no_legacy_audio_or_prefix_contract(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("prefer_official_audio_file=True", source)
        self.assertIn("center_crop_before_split_or_right_pad_to20s", source)
        self.assertIn("_prepare_mmar", source)
        self.assertIn("_run_official_evaluation", source) if "_run_official_evaluation" in source else self.assertIn("run_official_scorer", source)
        self.assertNotIn("prepare_model_output_for_official_scorer", source)
        self.assertNotIn("max_new_tokens 32", source)
        runtime_source = RUNTIME_SOURCE.read_text(encoding="utf-8")
        self.assertIn("attention_mask=None, labels=None, use_cache=False", runtime_source)
        self.assertIn("skip_special_tokens=False", runtime_source)
        self.assertNotIn("split(tokenizer.eos_token", runtime_source)


if __name__ == "__main__":
    unittest.main()
