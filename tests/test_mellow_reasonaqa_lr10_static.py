import ast
import copy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch256_lr_ablation_10epochs"
FORMAL = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch256_formal_30epochs"
BASELINE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/reasonaqa_global_batch256_lr_ablation_5epochs"
WRAPPER = ROOT / "code/RSmol/run_mellow_official_reasonaqa_gbs256_lr_ablation_10epochs_5090.sh"


def read(path):
    return Path(path).read_text(encoding="utf-8")


def function_node(path, name):
    tree = ast.parse(read(path))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


class MellowReasonAqaLr10StaticTests(unittest.TestCase):
    def test_route_isolated_from_original_routes(self):
        self.assertTrue((ROUTE / "train.py").is_file())
        self.assertTrue((ROUTE / "data/audiotext_dataset.py").is_file())
        self.assertTrue((ROUTE / "scripts/rsmol/run_reasonaqa_8gpu_formal.sh").is_file())
        self.assertTrue(WRAPPER.is_file())
        self.assertNotEqual(ROUTE, FORMAL)
        self.assertNotEqual(ROUTE, BASELINE)

    def test_dataset_keeps_formal_compact_text_contract(self):
        route_dataset = ROUTE / "data/audiotext_dataset.py"
        formal_dataset = FORMAL / "data/audiotext_dataset.py"
        route_tree = ast.parse(read(route_dataset))
        formal_tree = ast.parse(read(formal_dataset))
        route_reader = next(
            node for node in ast.walk(route_tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_read_audio"
        )
        formal_reader = next(
            node for node in ast.walk(formal_tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_read_audio"
        )
        route_reader.body = copy.deepcopy(formal_reader.body)
        self.assertEqual(ast.dump(route_tree, include_attributes=False), ast.dump(formal_tree, include_attributes=False))
        for name in ("_tokenize_text", "_read_text", "collate_fn"):
            self.assertEqual(
                ast.dump(function_node(route_dataset, name), include_attributes=False),
                ast.dump(function_node(formal_dataset, name), include_attributes=False),
                name,
            )

    def test_trainer_matches_formal30_except_route_identity(self):
        route_tree = ast.parse(read(ROUTE / "training/trainer.py"))
        formal_tree = ast.parse(read(FORMAL / "training/trainer.py"))
        substitutions = {
            "formal_10epochs_lr_ablation_random_audio2": "formal_30epochs_compact",
            "random_fill_if_missing_audio2": "zero_if_missing_audio2",
            "unexpected route_variant for the 10-epoch LR-ablation training": "unexpected route_variant for compact 30-epoch training",
            "the 10-epoch LR-ablation route requires global_l2_hard_clip": "compact 30-epoch route requires global_l2_hard_clip",
            "the 10-epoch LR-ablation route requires max_grad_norm=0.5": "compact 30-epoch route requires max_grad_norm=0.5",
            "the LR-ablation route requires a 260-position audio prefix": "compact training requires a 260-position audio prefix",
        }
        for node in ast.walk(route_tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = substitutions.get(node.value, node.value)
        self.assertEqual(
            ast.dump(route_tree, include_attributes=False),
            ast.dump(formal_tree, include_attributes=False),
        )

    def test_dataset_randomly_fills_missing_second_audio(self):
        source = read(ROUTE / "data/audiotext_dataset.py")
        start = source.index("    def _read_audio")
        end = source.index("    def _create_answer_input", start)
        audio_reader = source[start:end]
        self.assertIn("if file_path2 == '':", audio_reader)
        self.assertIn("file_path2 = random.choice(self.exists_filepaths1)", audio_reader)
        self.assertNotIn("zero_audio2", audio_reader)
        self.assertNotIn("torch.zeros_like(audio_data1)", audio_reader)

    def test_runtime_writer_contract(self):
        source = read(ROUTE / "scripts/rsmol/write_reasonaqa_runtime_config.py")
        self.assertIn('ROUTE_VARIANT = "formal_10epochs_lr_ablation_random_audio2"', source)
        self.assertIn('AUDIO_SLOT_POLICY = "random_fill_if_missing_audio2"', source)
        self.assertIn('parser.add_argument("--max-lr", type=float, required=True)', source)
        self.assertIn("if args.num_epochs != 10:", source)
        self.assertIn("min_lr = 0.1 * args.max_lr", source)
        self.assertNotIn('parser.add_argument("--min-lr"', source)
        self.assertNotIn('args.max_lr, 4e-5', source)

    def test_trainer_and_final_audit_use_new_route_identity(self):
        trainer = read(ROUTE / "training/trainer.py")
        runner = read(ROUTE / "scripts/rsmol/run_reasonaqa_8gpu_formal.sh")
        for source in (trainer, runner):
            self.assertIn("formal_10epochs_lr_ablation_random_audio2", source)
            self.assertIn("random_fill_if_missing_audio2", source)
            self.assertNotIn("zero_if_missing_audio2", source)
        self.assertIn("NUM_EPOCHS=10", runner)
        self.assertIn('RUNTIME_CONFIG="$OUTPUT_DIR/runtime_10epochs.yaml"', runner)
        self.assertIn("min_lr = 0.1 * max_lr", runner)
        self.assertLess(runner.index("MIN_LR=$("), runner.index("staging raw audio"))

    def test_wrapper_exposes_lr_and_preserves_resources(self):
        source = read(WRAPPER)
        self.assertIn("--max-lr", source)
        self.assertIn("-p pdgpu-5090", source)
        self.assertIn("-c 32 -m 256G -g 8 -n 1", source)
        self.assertIn("reasonaqa_global_batch256_lr_ablation_10epochs/scripts/rsmol/run_reasonaqa_8gpu_formal.sh", source)
        self.assertIn("formal_10epochs_maxlr_", source)
        self.assertNotIn("formal30_5090", source)


if __name__ == "__main__":
    unittest.main()
