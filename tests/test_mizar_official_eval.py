"""Release integrity, strict official loading, and unchanged evaluation semantics."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/RSmol"))
from mizar_official_eval import evaluate as runner
from mizar_official_eval import runtime


class OfficialEvalTests(unittest.TestCase):
    def test_protocol_and_generation_functions_are_shared_objects(self):
        self.assertIs(runtime.PROTOCOL, runner.base.runtime_adapter.PROTOCOL)
        self.assertIs(runtime.greedy_decode, runner.base.runtime_adapter.greedy_decode)
        for name in ("generate", "make_mmau_audio_decoder", "_metadata_records", "_prepare_mmar",
                     "_materialize", "_write_predictions", "run_official_scorer", "claim_output"):
            self.assertIs(getattr(runner, name), getattr(runner.base, name), name)
        self.assertEqual(runner.EXPECTED_ROWS, 1000)

    def test_orchestration_only_changes_official_release_check(self):
        def run_tree(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        original = run_tree(Path(runner.base.__file__))
        official = run_tree(Path(runner.__file__))
        # Remove only the one loaded-model provenance check; traversal, resume,
        # fatal-row handling, coverage, raw text and scoring remain AST-identical.
        for tree in (original, official):
            body = next(node for node in tree.body if isinstance(node, ast.Try)).body
            checks = [node for node in body if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Compare)
                      and ast.unparse(node.test).startswith("runtime[5]")]
            self.assertEqual(len(checks), 1)
            body.remove(checks[0])
        self.assertEqual(ast.dump(original), ast.dump(official))

    def test_defaults_and_model_directory_overrides(self):
        for benchmark, folder in (("mmau", "mmau_test_mini"), ("mmar", "mmar")):
            args = runner.parse_args(["--benchmark", benchmark])
            self.assertEqual(args.model_dir, runtime.DEFAULT_MODEL_DIR.resolve())
            self.assertEqual(args.checkpoint.name, "stage3_q4_seed20260905.ckpt")
            self.assertEqual(args.output_dir, (runner.OUTPUT_ROOT / folder).resolve())
            self.assertFalse(hasattr(args, "runtime_config"))
            self.assertFalse(hasattr(args, "expected_step"))
            args = runner.parse_args(["--benchmark", benchmark, "--model-dir", "other model"])
            self.assertEqual(args.checkpoint, args.model_dir / "checkpoints/final/stage3_q4_seed20260905.ckpt")
        args = runner.parse_args(["--benchmark", "mmar", "--model-dir", "other", "--checkpoint", "relocated.ckpt"])
        self.assertEqual(args.checkpoint, Path("relocated.ckpt").resolve())

    def test_manifest_pins_the_exact_official_release_and_all_assets(self):
        manifest = runtime.release_manifest()
        self.assertEqual(manifest["release"]["optimizer_step"], 200)
        self.assertEqual(manifest["release"]["parameters"], 159299840)
        self.assertEqual(manifest["checkpoint"]["sha256"],
                         "0c542f63e11ed55d360feb0d66e7be0e8c6a878f44cffd302a92388f915388a2")
        paths = [entry["path"] for entry in manifest["files"]]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(sum(p.startswith("assets/") for p in paths), 15)
        self.assertIn("model_info.json", paths)
        self.assertIn("provenance/local_bundle.json", paths)

    def test_mapper_classes_match_the_official_stage3_source(self):
        path = runtime.ROUTE_ROOT / "models/ced.py"
        classes = {node.name: node for node in ast.parse(path.read_text(encoding="utf-8")).body
                   if isinstance(node, ast.ClassDef)}
        for name, expected in runtime.release_manifest()["official_mapper_ast_sha256"].items():
            # Python 3.12 added empty type_params fields to ordinary classes/functions.
            canonical = ast.dump(classes[name], include_attributes=False).replace(", type_params=[]", "")
            actual = hashlib.sha256(canonical.encode()).hexdigest()
            self.assertEqual(actual, expected, name)

    def bundle_fixture(self, root):
        manifest = copy.deepcopy(runtime.release_manifest())
        for item in [manifest["checkpoint"], *manifest["files"]]:
            data = item["path"].encode()
            path = root / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            item.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        return manifest, root / manifest["checkpoint"]["path"]

    def test_bundle_audit_and_stability_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, checkpoint = self.bundle_fixture(root)
            with patch.object(runtime, "release_manifest", return_value=manifest):
                audit = runtime.audit_bundle(root, checkpoint)
                self.assertEqual(len(audit["verified_files"]), 18)
                runtime.verify_bundle_unchanged(audit, root, checkpoint)
                with self.assertRaises(ValueError):
                    runtime.verify_bundle_unchanged(audit, root / "other", checkpoint)
                checkpoint.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "changed after"):
                    runtime.verify_bundle_unchanged(audit, root, checkpoint)

    def test_each_asset_and_checkpoint_is_required_and_hash_checked(self):
        for missing in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, checkpoint = self.bundle_fixture(root)
                for item in [manifest["checkpoint"], *manifest["files"]]:
                    path = root / item["path"]
                    original = path.read_bytes()
                    if missing:
                        path.unlink()
                    else:
                        path.write_bytes(b"x" * len(original))  # same-size corruption
                    with self.subTest(file=item["path"], missing=missing):
                        with patch.object(runtime, "release_manifest", return_value=manifest):
                            with self.assertRaises((ValueError, FileNotFoundError)):
                                runtime.audit_bundle(root, checkpoint)
                    path.write_bytes(original)

    @staticmethod
    def state_fixture():
        class Tensor:
            dtype = "float32"
            def __init__(self, n=0): self.n = n
            def numel(self): return self.n
            def is_floating_point(self): return True
        state = {f"tensor.{i}": Tensor() for i in range(454)}
        state["tensor.0"].n = 187628509
        torch = NS(is_tensor=lambda v: isinstance(v, Tensor), float32="float32")
        return state, torch

    def test_plain_full_fp32_state_only(self):
        state, torch = self.state_fixture()
        self.assertEqual(runtime.validate_state_dict(state, torch)["tensor_count"], 454)
        for bad in ({"state_dict": state}, {**state, "extra": object()}, {},
                    {"module." + k: v for k, v in state.items()}):
            with self.assertRaises(ValueError): runtime.validate_state_dict(bad, torch)
        state["tensor.0"].n -= 1
        with self.assertRaisesRegex(ValueError, "element count"):
            runtime.validate_state_dict(state, torch)
        state["tensor.0"].n += 1
        state["tensor.0"].dtype = "bfloat16"
        with self.assertRaisesRegex(ValueError, "float32"):
            runtime.validate_state_dict(state, torch)

    @staticmethod
    def structure_fixture():
        weight = NS(shape=(49152, 576))
        embed = NS(weight=weight, num_embeddings=49152)
        lm = NS(model=NS(layers=[Mock() for _ in range(30)]),
                config=NS(hidden_size=576, num_hidden_layers=30),
                state_dict=lambda: {}, get_input_embeddings=lambda: embed,
                get_output_embeddings=lambda: embed)
        encoder = NS(mapper_type="freq_merge_126", mapper=NS(resample=False, output_tokens=126))
        return NS(caption_decoder=NS(lm=lm), audio_encoder=NS(base=encoder),
                  parameters=lambda: [NS(numel=lambda: 159299840)])

    def test_model_geometry_and_tied_embeddings(self):
        model = self.structure_fixture()
        self.assertEqual(runtime.audit_model(model)["physical_layers"], 30)
        mutations = [
            lambda m: m.caption_decoder.lm.model.layers.pop(),
            lambda m: m.caption_decoder.lm.model.layers.__setitem__(1, m.caption_decoder.lm.model.layers[0]),
            lambda m: setattr(m.caption_decoder.lm, "state_dict", lambda: {"model.read_routers.0.weight": 1}),
            lambda m: setattr(m.caption_decoder.lm, "get_output_embeddings", lambda: NS(weight=object())),
            lambda m: setattr(m, "parameters", lambda: [NS(numel=lambda: 159299839)]),
            lambda m: setattr(m.audio_encoder.base.mapper, "resample", True),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                model = self.structure_fixture()
                mutate(model)
                with self.assertRaises(ValueError): runtime.audit_model(model)

    def test_forward_hooks_only_observe_and_require_standard_sequence(self):
        lm = Mock()
        lm.model.layers = [Mock() for _ in range(30)]
        audit = runtime.install_forward_audit(lm)
        begin = lm.register_forward_pre_hook.call_args.args[0]
        finish = lm.register_forward_hook.call_args.args[0]
        output = NS(past_key_values=None)
        for length in (383, 384, 682):
            self.assertIsNone(begin(lm, (), {"inputs_embeds": NS(shape=(1, length, 576)),
                                            "use_cache": False, "attention_mask": None}))
            for layer in lm.model.layers:
                self.assertIsNone(layer.register_forward_pre_hook.call_args.args[0](layer, ()))
            self.assertIsNone(finish(lm, (), output))
        self.assertEqual(audit["checked_forwards"], 3)
        self.assertEqual(audit["last_layers"], list(range(30)))
        for kwargs in ({"use_cache": True}, {"attention_mask": object()}, {"past_key_values": object()},
                       {"inputs_embeds": NS(shape=(1, 382, 576))}):
            with self.assertRaises(RuntimeError):
                begin(lm, (), {"inputs_embeds": NS(shape=(1, 383, 576)), "use_cache": False, **kwargs})
        begin(lm, (), {"inputs_embeds": NS(shape=(1, 383, 576)), "use_cache": False})
        with self.assertRaises(RuntimeError): finish(lm, (), output)

    def loader_fixture(self):
        model = self.structure_fixture()
        lifecycle = Mock()
        for name in ("float", "load_state_dict", "requires_grad_"):
            setattr(model, name, getattr(lifecycle, name))
        model.to = lifecycle.to
        model.to.return_value = model
        model.eval = lifecycle.eval
        model.eval.return_value = model
        model.modules = lambda: [NS(training=False)]
        model.parameters = lambda: [NS(numel=lambda: 159299840, dtype="float32")]
        state, torch = self.state_fixture()
        torch.load = Mock(return_value=state)
        torch.cuda = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_name.return_value = "fake GPU"
        torch.manual_seed = Mock()
        torch.set_num_threads = Mock()
        torch.device = lambda name: name
        torch.backends = NS(cuda=NS(matmul=NS()), cudnn=NS())
        torch.__version__ = "mock"
        torch.version = NS(cuda="mock")
        class Tokenizer:
            pad_token_id, eos_token_id = 17, 0
            def __len__(self): return 49152
            def add_special_tokens(self, _): return 1
        tokenizer = Tokenizer()
        auto = Mock()
        auto.from_pretrained.return_value = tokenizer
        modules = {"models.mellow": NS(Mellow=Mock(return_value=model)),
                   "contract": NS(validate_tokenizer_contract=Mock(return_value={}), code_digest=lambda: "source"),
                   "data.fixed20": object(), "data.audiotext_dataset": NS(AudioTextDataset=type("Text", (), {}))}
        return model, lifecycle, torch, tokenizer, auto, modules

    def load_fixture(self, fixture):
        model, lifecycle, torch, tokenizer, auto, modules = fixture
        args = runner.parse_args(["--benchmark", "mmar"])
        with (patch.dict(sys.modules, {"torch": torch, "transformers": NS(AutoTokenizer=auto),
                                     "numpy": NS(random=NS(seed=Mock()))}),
              patch.object(runtime.base, "import_training_route", return_value=modules),
              patch.object(runtime, "verify_bundle_unchanged"),
              patch.object(runtime, "install_forward_audit", return_value={"status": "PENDING"})):
            result = runtime.load_runtime(args, {"status": "PASS"})
        return args, result

    def test_loader_applies_final_weights_strictly_after_fp32_construction(self):
        fixture = self.loader_fixture()
        args, result = self.load_fixture(fixture)
        model, lifecycle, torch, tokenizer, auto, modules = fixture
        torch.load.assert_called_once_with(args.checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual([call[0] for call in lifecycle.mock_calls],
                         ["float", "load_state_dict", "to", "eval", "requires_grad_"])
        self.assertEqual(lifecycle.load_state_dict.call_args.kwargs, {"strict": True})
        lifecycle.requires_grad_.assert_called_once_with(False)
        self.assertIs(result[0], model)
        self.assertIs(result[3].tokenizer, tokenizer)
        self.assertEqual(result[5]["release"]["optimizer_step"], 200)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertTrue(torch.backends.cudnn.deterministic)

    def test_loader_never_falls_back_on_strict_load_failure(self):
        fixture = self.loader_fixture()
        fixture[1].load_state_dict.side_effect = RuntimeError("missing official key")
        with self.assertRaisesRegex(RuntimeError, "missing official key"):
            self.load_fixture(fixture)
        self.assertEqual(fixture[1].load_state_dict.call_count, 1)
        fixture[1].to.assert_not_called()

    def test_loader_rejects_incorrect_tokenizer_or_precision(self):
        for kind in ("eos", "pad", "dtype", "train"):
            fixture = self.loader_fixture()
            if kind == "eos": fixture[3].eos_token_id = 17
            elif kind == "pad": fixture[3].pad_token_id = 0
            elif kind == "dtype": fixture[0].parameters = lambda: [NS(numel=lambda: 159299840, dtype="float16")]
            else: fixture[0].modules = lambda: [NS(training=True)]
            with self.subTest(kind=kind), self.assertRaises((ValueError, RuntimeError)):
                self.load_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
