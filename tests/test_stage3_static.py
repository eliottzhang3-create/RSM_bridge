"""Dependency-light Stage 3 offline benchmark contract checks."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3.py"
RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3.sh"
SUBMIT = ROOT / "code" / "RSmol" / "run_stage3_eval_5090.sh"


def load_stage3():
    spec = importlib.util.spec_from_file_location("stage3_evaluation_static", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage3StaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stage3 = load_stage3()
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.runtime = RUNTIME.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_task_protocol_is_explicit(self) -> None:
        self.assertEqual(
            self.stage3.STAGE3_TASKS,
            ("hellaswag", "mmlu", "gsm8k", "arc_easy", "arc_challenge"),
        )
        self.assertEqual(len(self.stage3.EXPECTED_MMLU_SUBJECTS), 57)
        self.assertIn("econometrics", self.stage3.EXPECTED_MMLU_SUBJECTS)
        self.assertIn("us_foreign_policy", self.stage3.EXPECTED_MMLU_SUBJECTS)
        self.assertNotIn("construction", self.stage3.EXPECTED_MMLU_SUBJECTS)
        self.assertNotIn("criminology", self.stage3.EXPECTED_MMLU_SUBJECTS)
        for marker in (
            "hellaswag",
            "mmlu",
            "gsm8k",
            "arc_easy",
            "arc_challenge",
            "EXPECTED_MMLU_SUBJECTS",
            "num_fewshot is intentionally omitted",
            "num_fewshot=5 if task == \"mmlu\" else None",
            "fewshot_split",
            "metric_names",
            "activity_label",
            "ctx_a",
            "ctx_b",
            '"choices", "answerKey"',
            "acc_norm",
            "exact_match",
            "overlay_modified_fields",
            "TaskManager(include_path",
            "_expand_official_group",
            "_official_task_tags",
            "expanded_leaf_tasks",
            "mmlu_stem_tasks",
            "validation",
            "main",
            "auxiliary_train_used",
            "gsm8k_socratic_used",
        ):
            self.assertIn(marker, self.source)

    def test_local_mapping_uses_observed_snapshot_and_correct_splits(self) -> None:
        root = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/eval_datasets")
        hellaswag = self.stage3.local_data_files("hellaswag", root)
        self.assertIn("validation", hellaswag)
        self.assertIn(
            "Rowan_hellaswag/data/validation-00000-of-00001.parquet",
            hellaswag["validation"].replace("\\", "/"),
        )
        mmlu = self.stage3.local_data_files("mmlu_abstract_algebra", root)
        self.assertIn("/cais_mmlu/abstract_algebra/test-00000-of-00001.parquet", mmlu["test"].replace("\\", "/"))
        self.assertIn("dev", mmlu)
        self.assertNotIn("auxiliary_train", " ".join(mmlu.values()))
        gsm = self.stage3.local_data_files("gsm8k", root)
        self.assertIn("/openai_gsm8k/main/train-00000-of-00001.parquet", gsm["train"].replace("\\", "/"))
        self.assertNotIn("socratic", " ".join(gsm.values()))
        self.assertIn("ARC-Easy", self.stage3.local_data_files("arc_easy", root)["test"])
        self.assertIn("ARC-Challenge", self.stage3.local_data_files("arc_challenge", root)["test"])

    def test_official_mmlu_group_expansion_is_leaf_complete(self) -> None:
        leaves = {f"mmlu_{subject}": {} for subject in self.stage3.EXPECTED_MMLU_SUBJECTS}
        groups = {
            "mmlu": {"task": [{"group": "mmlu_stem"}]},
            "mmlu_stem": {"task": list(leaves)},
        }
        self.assertEqual(
            self.stage3._expand_official_group("mmlu", groups, leaves),
            set(leaves),
        )
        self.assertEqual(
            self.stage3._expand_official_group(
                "mmlu",
                {"mmlu": {"task": ["mmlu_stem"]}, "mmlu_stem": {"task": ["stem_tasks"]}},
                leaves,
                tags={"stem_tasks": set(leaves)},
            ),
            set(leaves),
        )
        tag_index = self.stage3._official_task_tags(
            {
                "mmlu_abstract_algebra": (
                    Path("mmlu_abstract_algebra.yaml"),
                    {"tag": "mmlu_stem_tasks"},
                ),
                "mmlu_anatomy": (
                    Path("mmlu_anatomy.yaml"),
                    {"tag": ["mmlu_stem_tasks", "mmlu_all_tasks"]},
                ),
            }
        )
        self.assertEqual(
            tag_index["mmlu_stem_tasks"],
            {"mmlu_abstract_algebra", "mmlu_anatomy"},
        )
        groups["mmlu_stem"]["task"] = list(leaves)[:-1]
        self.assertNotEqual(
            self.stage3._expand_official_group("mmlu", groups, leaves),
            set(leaves),
        )

    def test_offline_and_external_output_guards(self) -> None:
        self.assertIn('"HF_HUB_OFFLINE": "1"', self.source)
        self.assertIn('"TRANSFORMERS_OFFLINE": "1"', self.source)
        self.assertIn('"HF_DATASETS_OFFLINE": "1"', self.source)
        self.assertIn("local_files_only=True", self.source)
        self.assertIn("pyarrow is required", self.source)
        self.assertIn("dataset_path_after_overlay", self.source)
        with self.assertRaises(ValueError):
            self.stage3.ensure_external_output(ROOT / "outputs" / "stage3")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-output"
            self.assertEqual(self.stage3.ensure_external_output(output), output.resolve())
            output.mkdir()
            (output / "existing.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.stage3.ensure_external_output(output)

    def test_yaml_function_reference_detection_contract(self) -> None:
        # lm_eval 0.4.12 load_yaml(resolve_func=False) materializes
        # ``!function utils.process_docs`` as an absolute path string.  Keep
        # the unresolved relative spelling covered too, since it is valid in
        # task YAMLs and must not be emitted as a plain scalar by an overlay.
        self.assertTrue(self.stage3._is_function_ref_string("utils.process_docs"))
        self.assertTrue(
            self.stage3._is_function_ref_string("/site-packages/lm_eval/tasks/hellaswag/utils.process_docs")
        )
        self.assertTrue(
            self.stage3._is_function_ref_string(r"C:\lm_eval\tasks\hellaswag\utils.process_docs")
        )
        self.assertTrue(
            self.stage3._is_function_ref_string(
                r"C:\Program Files\lm_eval\tasks\hellaswag\utils.process_docs"
            )
        )
        self.assertFalse(self.stage3._is_function_ref_string("{{question.strip()}}"))
        self.assertFalse(self.stage3._is_function_ref_string("A normal sentence"))
        self.assertIn("aggregation", self.stage3._FUNCTION_REFERENCE_KEYS)
        self.assertIn('represent_scalar("!function"', self.source)
        self.assertIn("_normalise_function_reference(value", self.source)
        self.assertNotIn("show_config=True", self.source)

    def test_yaml_function_reference_roundtrip_when_pyyaml_is_available(self) -> None:
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML is an evaluation-environment dependency")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "overlay.yaml"
            self.stage3._dump_yaml(
                output,
                {
                    "process_docs": "utils.process_docs",
                    "process_results": "/opt/lm_eval/tasks/example/utils.process_results",
                    "metric_list": [{"aggregation": "utils.aggregate"}],
                },
            )
            text = output.read_text(encoding="utf-8")
            self.assertIn("!function utils.process_docs", text)
            self.assertIn("!function /opt/lm_eval/tasks/example/utils.process_results", text)
            self.assertIn("!function utils.aggregate", text)

    def test_simple_evaluate_0412_argument_and_fewshot_contract(self) -> None:
        # 0.4.12 exposes task_manager and the four seed kwargs, but not the
        # newer CLI-only show_config kwarg.  Passing num_fewshot=5 is scoped
        # to the mmlu invocation; GSM8K keeps its YAML-native value of 5 and
        # zero-shot tasks receive None (the harness then uses 0).
        for marker in (
            "task_manager=TaskManager(include_path=str(overlay_dir))",
            "random_seed=config.seed",
            "numpy_random_seed=config.seed",
            "torch_random_seed=config.seed",
            "fewshot_random_seed=config.seed",
            'num_fewshot=5 if task == "mmlu" else None',
        ):
            self.assertIn(marker, self.source)
        self.assertNotIn("show_config=True", self.source)

    def test_relative_function_reference_becomes_loadable_absolute_ref(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = Path(temporary) / "site-packages" / "lm_eval" / "tasks" / "hellaswag"
            task_dir.mkdir(parents=True)
            (task_dir / "utils.py").write_text(
                "def process_docs(dataset): return dataset\n", encoding="utf-8"
            )
            source = task_dir / "hellaswag.yaml"
            resolved = self.stage3._normalise_function_reference(
                "utils.process_docs", source
            )
            self.assertEqual(resolved, "lm_eval.tasks.hellaswag.utils.process_docs")

    def test_recursive_registration_and_audit_contract(self) -> None:
        for marker in (
            "register_auto_class()",
            "AutoModelForCausalLM.from_pretrained",
            "logical_layer_count",
            "physical_layer_count",
            "recursive_loops",
            "forward_trace",
            "expected_forward_trace",
            "no_duplicate_parameter_storage",
            "unique_physical_layer_object_count",
            "torch_dtype=getattr(torch, dtype)",
        ):
            self.assertIn(marker, self.source)
        self.assertNotIn("device_map=", self.source)

    def test_result_audit_and_machine_readable_outputs(self) -> None:
        for marker in (
            "lm_eval_results.json",
            "log_samples.json",
            "summary.json",
            "summary.csv",
            "audit_report.json",
            "run_config.json",
            "all_parquet_manifest",
            "failed_count",
            "skipped_count",
            "stderr.log",
            "git_commit",
            "gpu",
            "sample_counts",
        ):
            self.assertIn(marker, self.source)

    def test_remote_wrapper_is_one_gpu_and_runtime_only(self) -> None:
        self.assertIn("vc submit", self.submit)
        self.assertIn("-c 8 -m 32G -g 1 -n 1", self.submit)
        self.assertIn("pdgpu-5090", self.submit)
        self.assertIn("SUBMIT_LOG_ROOT", self.submit)
        self.assertIn("absolute external path", self.submit)
        self.assertNotIn('mkdir -p log', self.submit)
        self.assertIn("code/RSLAM/code/RSmol/log", self.submit)
        self.assertNotIn("outputs/RSmol/stage3-submit-logs", self.submit)
        self.assertIn("bash scripts/evaluate_stage3.sh", self.submit)
        self.assertIn("conda activate", self.runtime)
        self.assertIn("python -u code/RSmol/scripts/evaluate_stage3.py", self.runtime)
        self.assertIn("HF_HUB_OFFLINE=1", self.runtime)
        self.assertIn("RSMOL_STAGE3_MODEL", self.runtime)
        self.assertIn("code/RSmol/log", self.runtime)
        self.assertIn("--log-root", self.runtime)
        self.assertIn("Refusing to overwrite non-empty Stage 3 output root", self.runtime)
        self.assertIn("RSMOL_STAGE3_OUTPUT_DIR must be an absolute external path", self.runtime)
        self.assertIn("RSMOL_STAGE3_TASKS must contain at least one task", self.runtime)
        self.assertIn("original", self.runtime)
        self.assertIn("recursive", self.runtime)
        self.assertIn("both", self.runtime)
        self.assertNotIn("torchrun", self.submit)


if __name__ == "__main__":
    unittest.main()
