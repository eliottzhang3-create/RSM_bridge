from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "code/RSmol/recursive_model_5_10x4_5_mesh.py"
GENERATOR = ROOT / "code/RSmol/scripts/generate_reasonaqa_x4_two_loop_ablation.py"
SUBMIT = ROOT / "code/RSmol/run_reasonaqa_x4_two_loop_ablation_generation_3090.sh"
LAUNCH = ROOT / "code/RSmol/scripts/generate_reasonaqa_x4_two_loop_ablation.sh"


class TwoLoopAblationStaticTest(unittest.TestCase):
    def test_sources_parse_and_shell_contract(self):
        for path in (MODEL, GENERATOR):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in (SUBMIT, LAUNCH):
            content = path.read_bytes()
            self.assertTrue(content.startswith(b"#!"))
            self.assertNotIn(b"\r", content)
            self.assertIn(b"set -euo pipefail", content)
        submit = SUBMIT.read_text(encoding="utf-8")
        self.assertIn("--num-samples 5", submit)
        self.assertIn("/data/reasonaqa/test.json", submit)
        self.assertIn("formal_3epochs_20261002_configfix_v3/checkpoint-011343", submit)
        self.assertIn("pdgpu-3090", submit)

    def test_exact_short_trace(self):
        tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "expected_trace")
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(GENERATOR), "exec"), namespace)
        trace = namespace["expected_trace"]()
        self.assertEqual(len(trace), 30)
        self.assertEqual([item["physical_index"] for item in trace], list(range(5)) + list(range(5, 15)) * 2 + list(range(15, 20)))
        self.assertEqual([item["logical_index"] for item in trace], list(range(30)))

    def test_router_and_model_isolation(self):
        model = MODEL.read_text(encoding="utf-8")
        generator = GENERATOR.read_text(encoding="utf-8")
        self.assertIn("self.ablate_after_two_loops = False", model)
        self.assertIn("range(2 if self.ablate_after_two_loops else RECURSIVE_LOOPS)", model)
        self.assertIn("read_index = 4 if self.ablate_after_two_loops and loop == 1 else loop + 1", model)
        self.assertIn("write_1 / write_routers[2]", generator)
        self.assertIn("read_3 / read_routers[4]", generator)
        self.assertIn("_load_runtime_model(args, mesh_eval.ROUTES[\"x4\"])", generator)
        self.assertIn("router_calls != expected_calls", generator)


if __name__ == "__main__":
    unittest.main()
