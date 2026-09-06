"""Dependency-light static contract tests for the isolated MeSH pipeline."""

from __future__ import annotations

import ast
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "code" / "RSmol" / "recursive_model_5_10x2_5_mesh.py"
CONVERTER = ROOT / "code" / "RSmol" / "scripts" / "convert_stepwise_5_10x2_5_mesh.py"
AUDIT = ROOT / "code" / "RSmol" / "scripts" / "audit_stage1_5_10x2_5_mesh.py"
TRAIN = ROOT / "code" / "RSmol" / "scripts" / "train_stage4_5_10x2_5_mesh_ddp.py"
FILES = [
    ROOT / "code" / "RSmol" / "scripts" / "convert_stepwise_5_10x2_5_mesh.sh",
    ROOT / "code" / "RSmol" / "scripts" / "audit_stage1_5_10x2_5_mesh.sh",
    ROOT / "code" / "RSmol" / "scripts" / "train_stage4_5_10x2_5_mesh_ddp.sh",
    ROOT / "code" / "RSmol" / "run_convert_stepwise_5_10x2_5_mesh_3090.sh",
    ROOT / "code" / "RSmol" / "run_audit_stage1_5_10x2_5_mesh_3090.sh",
    ROOT / "code" / "RSmol" / "run_stage4_5_10x2_5_mesh_3090.sh",
]


def _literal(name: str, text: str):
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"constant {name} not found")


class MeshStaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL.read_text(encoding="utf-8")
        cls.converter = CONVERTER.read_text(encoding="utf-8")
        cls.audit = AUDIT.read_text(encoding="utf-8")
        cls.train = TRAIN.read_text(encoding="utf-8")
        cls.shell = "\n".join(path.read_text(encoding="utf-8") for path in FILES)

    def test_all_files_are_present_and_isolated(self):
        self.assertTrue(MODEL.is_file())
        self.assertTrue(CONVERTER.is_file())
        self.assertTrue(AUDIT.is_file())
        self.assertTrue(TRAIN.is_file())
        self.assertTrue(all(path.is_file() for path in FILES))
        self.assertNotIn("recursive_model_5_10_5.py", self.model)
        self.assertNotIn("train_stage4_5_10_5_ddp.py", self.train)

    def test_exact_schedule_and_source_mapping(self):
        self.assertEqual(_literal("LOGICAL_TO_PHYSICAL", self.model), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19))
        self.assertEqual(_literal("SOURCE_LAYER_INDICES_0BASED", self.model), (0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29))
        for marker in ("LOGICAL_LAYER_COUNT = 30", "PHYSICAL_LAYER_COUNT = 20", "RECURSIVE_LOOPS = 2", "MEMORY_SLOT_COUNT = 5", "ROUTER_COUNT = 3", "LogicalSlotCacheView"):
            self.assertIn(marker, self.model)

    def test_mesh_recurrence_contract(self):
        for marker in ("memory = torch.zeros((batch_size, MEMORY_SLOT_COUNT", "memory[:, 0] = hidden", "memory[:, 1:]", "prefix_output", "write_pre", "read_pre", "self._write(memory, prefix_output, write_pre)", "for loop in range(2)", "self._write(memory, core, write)", "self._read(memory, read)", "self.norm(hidden)"):
            self.assertIn(marker, self.model)
        self.assertIn("self.write_routers = nn.ModuleList", self.model)
        self.assertIn("self.read_routers = nn.ModuleList", self.model)
        self.assertIn("for _ in range(ROUTER_COUNT)", self.model)
        self.assertIn("transition_query", self.converter)
        self.assertIn("prefix_output", self.converter)
        self.assertNotIn("sqrt(hidden_size)", self.model)
        self.assertNotIn("A_bar", self.model)
        self.assertNotIn("B_bar", self.model)
        self.assertNotIn("Poisson", self.model + self.converter + self.train)
        self.assertNotIn("dynamic depth", self.model.lower() + self.train.lower())

    def test_router_initialization_is_meta_device_safe(self):
        function_start = self.model.index("def _init_router")
        function_end = self.model.index("\n\nclass MeshLlamaModel", function_start)
        function = self.model[function_start:function_end]
        self.assertIn("router.weight.is_meta", function)
        self.assertIn("router.bias.is_meta", function)
        self.assertLess(function.index("router.weight.is_meta"), function.index("torch.no_grad()"))
        self.assertLess(function.index("router.weight.is_meta"), function.index("torch.isfinite(values).all()"))

    def test_converter_clean_source_and_atomic_contract(self):
        for marker in ("TRAINING_MARKERS", "mesh_checkpoint_metadata.json", "checkpoint_complete.json", "detect_source", "conversion_only_5_10_5", "original_smolLM2_30_layer", "reject_forbidden_output", "tempfile.mkdtemp", "staging.replace(output)", "allow-overwrite", "mesh_conversion_metadata.json", "source_kind"):
            self.assertIn(marker, self.converter)
        self.assertIn("DEFAULT_OUTPUT_DIR = Path(\"/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10x2-5-mesh\")", self.converter)
        self.assertIn("safe_serialization=True", self.converter)

    def test_converter_restores_only_declared_tied_weight_aliases(self):
        for marker in ("_restore_tied_weight_aliases", "tie_word_embeddings", "model.embed_tokens.weight", "lm_head.weight", "restored_tied_weight_aliases"):
            self.assertIn(marker, self.converter)
        self.assertIn('if not bool(getattr(source_config, "tie_word_embeddings", False)):', self.converter)
        self.assertIn("if tuple(source_tensor.shape) != tuple(target_state[missing_key].shape):", self.converter)

        tree = ast.parse(self.converter)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_restore_tied_weight_aliases")
        namespace = {"Any": object}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(CONVERTER), "exec"), namespace)
        restore = namespace["_restore_tied_weight_aliases"]

        class Tensor:
            def __init__(self, shape):
                self.shape = shape

        class Config:
            tie_word_embeddings = True

        embedding = Tensor((16, 8))
        target = {"model.embed_tokens.weight": Tensor((16, 8)), "lm_head.weight": Tensor((16, 8))}
        remapped = {"model.embed_tokens.weight": embedding}
        self.assertEqual(restore(remapped, target, Config()), ["lm_head.weight"])
        self.assertIs(remapped["lm_head.weight"], embedding)

        Config.tie_word_embeddings = False
        untied = {"model.embed_tokens.weight": embedding}
        self.assertEqual(restore(untied, target, Config()), [])
        self.assertNotIn("lm_head.weight", untied)

        Config.tie_word_embeddings = True
        mismatched_target = {"model.embed_tokens.weight": Tensor((16, 8)), "lm_head.weight": Tensor((17, 8))}
        with self.assertRaisesRegex(ValueError, "cannot restore tied weight"):
            restore({"model.embed_tokens.weight": embedding}, mismatched_target, Config())

    def test_stage1_contract(self):
        self.assertIn("torch.cuda.is_available", self.audit)
        for marker in ("memory_shape", "six_router_outputs", "slot_sum_one", "physical_trace_5_10_10_5", "logical_cache_slots_0_29", "prefill_incremental_logits_close", "six_router_gradients", "save_reload_logits_close", "memory_not_serialized", "report_path"):
            self.assertIn(marker, self.audit)

    def test_stage4_formal_contract_and_checkpoint_cursor(self):
        for marker in ("MODEL_ARCHITECTURE_CONTRACT = \"logical_30_physical_20_5_10x2_5_mesh\"", "DEFAULT_WORLD_SIZE = 8", "DEFAULT_MICRO_BATCH_SIZE = 8", "DEFAULT_GRADIENT_ACCUMULATION_STEPS = 16", "DEFAULT_FORMAL_OPTIMIZER_STEPS = 9244", "DEFAULT_FORMAL_WARMUP_STEPS = 463", "DEFAULT_MAX_LR = 8e-4", "DEFAULT_MIN_LR = 8e-5", "DEFAULT_SAVE_EVERY = 500", "DEFAULT_LOG_INTERVAL_STEPS = 10", "ParquetFile", "iter_batches", "columns=[\"text\"]", "data_cursors_by_rank", "token_weighted_gradient_scale", "checkpoint_contract", "checkpoint_complete.json", "checkpoint_manifest.json", "rng_states_by_rank", "routing_audit_due", "slot_probabilities", "router collapse", "use_cache=False", "router_parameters_in_optimizer", "routing_stats", "tokens_per_second", "step_time_seconds", "gpu_memory_allocated_gib", "gpu_memory_reserved_gib", "print("):
            self.assertIn(marker, self.train)
        self.assertIn("pdgpu-3090", self.shell)
        self.assertIn("docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1", self.shell)
        self.assertIn("-c 32 -m 256G -g 8", self.shell)
        self.assertIn("stage4_5_10x2_5_mesh", self.shell)

    def test_stage4_parquet_directory_contract(self):
        expected = "/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset/data"
        self.assertIn(f'DATA_ROOT_DEFAULT = Path("{expected}")', self.train)
        self.assertIn(expected, self.shell)
        self.assertIn('nested_data = data_dir / "data"', self.train)
        self.assertIn('candidate.glob("*.parquet")', self.train)
        self.assertIn("no parquet shards found; checked:", self.train)

        tree = ast.parse(self.train)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_manifest")
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(TRAIN), "exec"), namespace)
        manifest = namespace["_manifest"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "data"
            nested.mkdir()
            shard = nested / "000.parquet"
            shard.touch()
            self.assertEqual(manifest(root), [shard])
            self.assertEqual(manifest(nested), [shard])


if __name__ == "__main__":
    unittest.main()
