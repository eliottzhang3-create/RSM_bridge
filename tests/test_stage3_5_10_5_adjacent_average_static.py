"""Dependency-light contracts for adjacent-average Stage 3 evaluation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3_5_10_5_adjacent_average.py"
RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3_5_10_5_adjacent_average.sh"
SUBMIT = ROOT / "code" / "RSmol" / "run_stage3_eval_5_10_5_adjacent_average_3090.sh"
LEGACY = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3_5_10_5.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("stage3_5_10_5_adjavg_static", EVALUATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_legacy_evaluator():
    spec = importlib.util.spec_from_file_location("stage3_5_10_5_legacy_for_adjavg_test", LEGACY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage3AdjacentAverageStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_evaluator()
        cls.source = EVALUATOR.read_text(encoding="utf-8")
        cls.runtime = RUNTIME.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")
        cls.legacy = LEGACY.read_text(encoding="utf-8")
        cls.legacy_module = load_legacy_evaluator()

    def _artifact(self, root: Path, *, adjacent: bool) -> None:
        config = {
            "model_type": "llama",
            "vocab_size": 3,
            "architectures": ["RecursiveLlamaForCausalLM"],
            "num_hidden_layers": 30,
            "recursive_layer_count": 20,
            "recursive_loops": 2,
            "recursive_loops_scope": "middle_only",
            "recursive_prefix_layer_count": 5,
            "recursive_middle_layer_count": 10,
            "recursive_suffix_layer_count": 5,
            "logical_to_physical": list(self.module.LOGICAL_TO_PHYSICAL),
            "recursive_logical_to_physical": list(self.module.LOGICAL_TO_PHYSICAL),
            "logical_to_physical_schedule": list(self.module.LOGICAL_TO_PHYSICAL),
            "recursive_logical_to_physical_schedule": list(self.module.LOGICAL_TO_PHYSICAL),
        }
        if adjacent:
            config.update({
                "recursive_initialization_policy": self.module.INITIALIZATION_POLICY,
                "recursive_initialization_contract": self.module.INITIALIZATION_CONTRACT,
                "recursive_average_accumulator_dtype": "float32",
                "recursive_prefix_source_layers_0based": list(self.module.PREFIX_SOURCE_LAYERS_0BASED),
                "recursive_middle_source_layer_pairs_0based": [list(pair) for pair in self.module.MIDDLE_SOURCE_LAYER_PAIRS_0BASED],
                "recursive_suffix_source_layers_0based": list(self.module.SUFFIX_SOURCE_LAYERS_0BASED),
                "recursive_source_layer_coverage_0based": list(range(30)),
                "recursive_source_layer_indices_0based": None,
                "recursive_source_mapping_0based": None,
            })
        else:
            config["recursive_source_layer_indices_0based"] = [
                0, 1, 2, 3, 4, 5, 7, 9, 11, 13,
                15, 17, 19, 21, 23, 25, 26, 27, 28, 29,
            ]
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (root / "tokenizer.json").write_text(
            '{"model": {"vocab": {"a": 0, "b": 1, "c": 2}}}', encoding="utf-8"
        )
        (root / "model.safetensors").write_bytes(b"placeholder")
        if adjacent:
            metadata = {
                "status": "ok",
                "initialization_policy": self.module.INITIALIZATION_POLICY,
                "initialization_contract": self.module.INITIALIZATION_CONTRACT,
                "middle_source_layer_pairs_0based": [list(pair) for pair in self.module.MIDDLE_SOURCE_LAYER_PAIRS_0BASED],
                "source_layer_coverage_0based": list(range(30)),
                "source_layer_coverage_exactly_once": True,
                "pair_average_audits": [
                    {"target": 5 + index, "source_pair": list(pair)}
                    for index, pair in enumerate(self.module.MIDDLE_SOURCE_LAYER_PAIRS_0BASED)
                ],
                "saved_artifact_verification": {"status": "PASS"},
            }
            (root / "adjacent_average_conversion_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

    def test_artifact_accepts_only_adjacent_average_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            self._artifact(model, adjacent=True)
            info = self.module.inspect_model_artifacts_5_10_5(model)
            self.assertEqual(info["label"], "recursive_5_10_5_adjacent_average")
            self.assertTrue(all(info["recursive_audit"]["contract_checks"].values()))
            with self.assertRaisesRegex(ValueError, "checkpoint contract"):
                self.legacy_module.inspect_model_artifacts_5_10_5(model)
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            self._artifact(model, adjacent=False)
            with self.assertRaises((FileNotFoundError, ValueError)):
                self.module.inspect_model_artifacts_5_10_5(model)

    def test_official_tasks_and_isolated_registration(self):
        self.assertEqual(
            self.module.STAGE3_TASKS,
            ("hellaswag", "mmlu", "gsm8k", "arc_easy", "arc_challenge"),
        )
        self.assertIn('"code.RSmol.recursive_model_5_10_5_adjacent_average"', self.source)
        self.assertIn("num_fewshot=5 if task == \"mmlu\" else None", self.source)
        self.assertIn("TaskManager(include_path=str(overlay_dir))", self.source)
        self.assertIn("adjacent_average_conversion_metadata.json", self.source)

    def test_wrappers_use_3090_and_safe_single_gpu_resources(self):
        self.assertIn("RSMOL_STAGE3_5_10_5_ADJAVG_MODEL", self.runtime + self.submit)
        self.assertIn("hellaswag,mmlu,gsm8k,arc_easy,arc_challenge", self.runtime)
        self.assertIn("-p pdgpu-3090", self.submit)
        self.assertIn("-c 8 -m 32G -g 1 -n 1", self.submit)
        self.assertNotIn("ADJAVG", self.legacy)


if __name__ == "__main__":
    unittest.main()
