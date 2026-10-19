"""Dependency-light replication checks for the LR5-based LR30 route."""
import ast
import copy
import importlib.util
import re
import unittest
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine"
SOURCE_ROUTE = PARENT / "reasonaqa_global_batch256_lr_ablation_10epochs"
ROUTE = PARENT / "reasonaqa_global_batch256_lr_ablation_30epochs"
GLOBAL_CLIP = PARENT / "reasonaqa_global_batch256_global_clip_5epochs"
HTSAT_FROZEN = PARENT / "reasonaqa_global_batch256_htsat_frozen_5epochs"
WRAPPER = ROOT / "code/RSmol/run_mellow_official_reasonaqa_gbs256_lr_ablation_30epochs_5090.sh"
VARIANT = "lr_ablation_30epochs_htsat_frozen_global_clip"


def read(path):
    return Path(path).read_text(encoding="utf-8-sig")


def tree(path):
    return ast.parse(read(path))


def function_node(path, name, class_name=None):
    parent = tree(path)
    if class_name:
        parent = next(node for node in parent.body
                      if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in ast.walk(parent)
                if isinstance(node, ast.FunctionDef) and node.name == name)


class MellowReasonAqaLr30StaticTests(unittest.TestCase):
    def test_runner_stage_name_passes_actual_staging_resolver(self):
        stage_script = ROUTE / "scripts/rsmol/stage_reasonaqa_raw_audio.py"
        spec = importlib.util.spec_from_file_location("lr30_staging_test", stage_script)
        staging = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(staging)
        runner = read(ROUTE / "scripts/rsmol/run_reasonaqa_8gpu_formal.sh")
        stage_template = re.search(r'^STAGE_ROOT="([^"]+)"$', runner, re.MULTILINE).group(1)

        # Exercise the actual Linux path checks on Windows without creating
        # /dev/shm or requiring remote audio. Only the parent directory exists.
        class VirtualPosixPath(PurePosixPath):
            def resolve(self, strict=False):
                if strict and self != PurePosixPath("/dev/shm"):
                    raise FileNotFoundError(str(self))
                return self

            def expanduser(self):
                return self

            def is_dir(self):
                return self == PurePosixPath("/dev/shm")

            def exists(self):
                return self == PurePosixPath("/dev/shm")

            def is_symlink(self):
                return False

        with patch.object(staging, "Path", VirtualPosixPath):
            for run_id in ("20_20261018_100246029693787", "91_20261019_120000000000001"):
                candidate = VirtualPosixPath(stage_template.replace("$RUN_ID", run_id))
                with self.subTest(run_id=run_id):
                    shm, resolved = staging.resolve_stage_root(candidate)
                    self.assertEqual(shm, PurePosixPath("/dev/shm"))
                    self.assertEqual(resolved, candidate)
            broken = VirtualPosixPath(
                "/dev/shm/mellow_adamw_cosine_reasonaqa_gbs256_lr10_frozen_clip_20")
            with self.assertRaisesRegex(ValueError, "basename must start"):
                staging.resolve_stage_root(broken)

    def test_copied_lr10_files_are_identical_except_epoch_contract(self):
        exceptions = {"README.md", "training/trainer.py",
                      "scripts/rsmol/run_reasonaqa_8gpu_formal.sh",
                      "scripts/rsmol/write_reasonaqa_runtime_config.py"}
        source_files = {file.relative_to(SOURCE_ROUTE).as_posix() for file in SOURCE_ROUTE.rglob("*")
                        if file.is_file() and "__pycache__" not in file.parts}
        target_files = {file.relative_to(ROUTE).as_posix() for file in ROUTE.rglob("*")
                        if file.is_file() and "__pycache__" not in file.parts}
        self.assertEqual(target_files, source_files)
        for source in SOURCE_ROUTE.rglob("*"):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            relative = source.relative_to(SOURCE_ROUTE)
            if relative.as_posix() not in exceptions:
                with self.subTest(file=relative.as_posix()):
                    self.assertEqual(source.read_bytes(), (ROUTE / relative).read_bytes())

    def test_htsat_model_is_exactly_the_frozen_reference(self):
        self.assertEqual(read(ROUTE / "models/mellow.py"),
                         read(HTSAT_FROZEN / "models/mellow.py"))

    def test_trainer_matches_global_clip_reference_except_contract_guards(self):
        route_tree = tree(ROUTE / "training/trainer.py")
        reference_tree = tree(GLOBAL_CLIP / "training/trainer.py")
        for current in (route_tree, reference_tree):
            train = next(node for node in ast.walk(current)
                         if isinstance(node, ast.FunctionDef) and node.name == "train")
            first_log = next(index for index, node in enumerate(train.body)
                             if isinstance(node, ast.Expr)
                             and ast.unparse(node).startswith("self.logger.info("))
            train.body = train.body[first_log:]
        train = next(node for node in ast.walk(route_tree)
                     if isinstance(node, ast.FunctionDef) and node.name == "train")
        audio_index = next(index for index, node in enumerate(train.body)
                           if isinstance(node, ast.Assign)
                           and ast.unparse(node.targets[0]) == "audio_encoder")
        del train.body[audio_index:audio_index + 4]
        self.assertEqual(ast.dump(route_tree), ast.dump(reference_tree))

    def test_decoder_keeps_no_extra_padding_mask(self):
        class Tensor:
            def __init__(self, shape):
                self.shape, self.device = shape, "cpu"

            def to(self, _device):
                return self

            def contiguous(self):
                return self

            def unsqueeze(self, dim):
                return Tensor(self.shape[:dim] + (1,) + self.shape[dim:])

            def repeat(self, *counts):
                return Tensor(tuple(size * count for size, count in zip(self.shape, counts)))

        class LM:
            def __init__(self):
                self.model = SimpleNamespace(embed_tokens=lambda value: Tensor(value.shape + (576,)))

            def __call__(self, **kwargs):
                return kwargs

        def cat(values, dim):
            shape = list(values[0].shape)
            shape[dim] = sum(value.shape[dim] for value in values)
            return Tensor(tuple(shape))

        torch = SimpleNamespace(Tensor=Tensor, tensor=lambda value: Tensor((len(value),)), cat=cat)
        namespace = {"torch": torch, "Optional": Optional, "downsample": lambda value: value}
        node = copy.deepcopy(function_node(ROUTE / "models/decoder.py", "forward", "DecoderModel"))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     "<decoder forward>", "exec"), namespace)
        mask = Tensor((2, 639))
        result = namespace["forward"](
            SimpleNamespace(text_decoder="smollm2", lora=False, lm=LM()),
            Tensor((2, 129, 576)), Tensor((2, 129, 576)),
            {"input_ids": Tensor((2, 129))},
            {"input_ids": Tensor((2, 250)), "attention_mask": mask})
        # Even though the trainer constructs a padding mask, LR5's Mellow
        # forward does not supply it to the decoder. Preserve that behavior.
        self.assertIsNone(result["attention_mask"])
        self.assertEqual(result["inputs_embeds"].shape, (2, 639, 576))

    def test_runtime_writer_contract(self):
        source = read(ROUTE / "scripts/rsmol/write_reasonaqa_runtime_config.py")
        self.assertIn(f'ROUTE_VARIANT = "{VARIANT}"', source)
        self.assertIn('ATTENTION_MASK_POLICY = "prefix_only"', source)
        self.assertIn('HTSAT_FREEZE_POLICY = "fully_frozen_eval_no_grad"', source)
        self.assertIn('AUDIO_SLOT_POLICY = "upstream_random_fallback"', source)
        self.assertIn('parser.add_argument("--max-lr", type=float, required=True)', source)
        self.assertIn("if args.num_epochs != 30:", source)
        self.assertIn('"total_prefix_length": 389', source)
        self.assertIn("min_lr = 0.1 * args.max_lr", source)
        self.assertIn('"max_grad_norm": 0.5', source)
        # Evaluate both config expressions with the same resolved paths and
        # arguments; only epochs and the route identifier may differ.
        def config_for(root, epochs, variant):
            writer = root / "scripts/rsmol/write_reasonaqa_runtime_config.py"
            assign = next(node for node in ast.walk(tree(writer))
                          if isinstance(node, ast.AnnAssign)
                          and isinstance(node.target, ast.Name) and node.target.id == "config")
            namespace = {
                "ROUTE_VARIANT": variant, "GRADIENT_CLIP_POLICY": "global_l2_hard_clip",
                "ATTENTION_MASK_POLICY": "prefix_only",
                "HTSAT_FREEZE_POLICY": "fully_frozen_eval_no_grad",
                "AUDIO_SLOT_POLICY": "upstream_random_fallback",
                "resolved_stage_root": ROOT, "resume_checkpoint": "",
                "init_model_checkpoint": "", "min_lr": 4e-6,
                "args": SimpleNamespace(
                    data_json=ROOT / "metadata.json", text_model_dir=ROOT / "SmolLM2",
                    htsat_root=ROOT / "HTSAT", save_dir=ROOT / "checkpoints",
                    max_lr=4e-5, warmup_ratio=0.05, batch_size=8,
                    gradient_accumulation_steps=4, num_epochs=epochs, num_workers=4,
                    max_epochs_this_run=0, max_optimizer_steps=0),
            }
            return eval(compile(ast.Expression(assign.value), str(writer), "eval"), namespace)
        reference = config_for(SOURCE_ROUTE, 10, "lr_ablation_10epochs_htsat_frozen_global_clip")
        actual = config_for(ROUTE, 30, VARIANT)
        self.assertEqual(actual["train"]["num_epochs"], 30)
        self.assertEqual(actual["train"]["optimizer"]["max_lr"], 4e-5)
        self.assertEqual(actual["train"]["optimizer"]["min_lr"], 4e-6)
        actual["train"]["num_epochs"] = 10
        actual["route_variant"] = "lr_ablation_10epochs_htsat_frozen_global_clip"
        self.assertEqual(actual, reference)

    def test_checkpoint_and_resume_use_new_contract(self):
        trainer = read(ROUTE / "training/trainer.py")
        runner = read(ROUTE / "scripts/rsmol/run_reasonaqa_8gpu_formal.sh")
        for source in (trainer, runner):
            self.assertIn(VARIANT, source)
            self.assertIn("fully_frozen_eval_no_grad", source)
            self.assertIn("global_l2_hard_clip", source)
        self.assertIn("NUM_EPOCHS=30", runner)
        self.assertIn('RUNTIME_CONFIG="$OUTPUT_DIR/runtime_30epochs.yaml"', runner)
        self.assertIn('MELLOW_JOB_ID="mellow_adamw_cosine_reasonaqa_gbs256_lr30_frozen_clip_', runner)
        self.assertIn('checkpoint.get("grad_norm_tracker") is not None', runner)
        self.assertNotIn("audit_source_equivalence", trainer)

    def test_wrapper_exposes_lr_and_preserves_resources(self):
        source = read(WRAPPER)
        self.assertIn("--max-lr", source)
        self.assertIn("-p pdgpu-5090", source)
        self.assertIn("-c 32 -m 256G -g 8 -n 1", source)
        self.assertIn("reasonaqa_global_batch256_lr_ablation_30epochs/scripts/rsmol/run_reasonaqa_8gpu_formal.sh", source)
        self.assertIn("formal_30epochs_lr5_frozen_clip_maxlr_", source)
        self.assertIn("mellow_official_reasonaqa_gbs256_lr_ablation_30epochs_5090", source)
        self.assertIn("MELLOW_REASONAQA_LR30_OUTPUT_DIR", source)
        self.assertIn(
            "run_mellow_official_reasonaqa_gbs256_lr_ablation_30epochs_5090.sh --max-lr 4e-5",
            read(ROUTE / "README.md"),
        )


if __name__ == "__main__":
    unittest.main()
