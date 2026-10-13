"""Output-ownership contracts for the gbs256 MMAU evaluator."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code" / "RSmol" / "scripts"
SOURCE = SCRIPTS / "evaluate_mmau_test_mini_mellow_gbs256.py"


def load_module():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("mmau_test_mini_mellow_gbs256", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        import yaml  # noqa: F401
    except ModuleNotFoundError:
        # Output-ownership tests do not parse a runtime YAML file.  Keep the
        # evaluator importable in the CPU-only local checkout, where PyYAML is
        # intentionally absent.
        yaml_stub = types.ModuleType("yaml")
        yaml_stub.safe_load = lambda _value: {}
        with mock.patch.dict(sys.modules, {"yaml": yaml_stub}):
            spec.loader.exec_module(module)
    else:
        spec.loader.exec_module(module)
    return module


class GBS256OutputClaimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    @staticmethod
    def _args(root: Path) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            checkpoint_file=root / "model--epo-5.ckpt",
            runtime_config=root / "runtime_5epochs.yaml",
            route_root=root / "route_root",
            dataset_dir=root / "dataset",
            output_dir=root / "eval" / "full",
        )

    @staticmethod
    def _hash(path: Path) -> str:
        return f"sha256:{path.name}"

    def _claim(self, args: types.SimpleNamespace) -> None:
        with mock.patch.object(self.module.adapter, "sha256_file", side_effect=self._hash):
            self.module._claim_output(args)

    def _identity(self, args: types.SimpleNamespace) -> dict[str, str]:
        with mock.patch.object(self.module.adapter, "sha256_file", side_effect=self._hash):
            return self.module._evaluation_identity(args)

    def _official_args(self, args: types.SimpleNamespace) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            mode="full",
            output_dir=args.output_dir,
            checkpoint=args.checkpoint_file.parent,
            dataset_dir=args.dataset_dir,
            parquet=args.dataset_dir / "test_mini.parquet",
            metadata_json=args.dataset_dir / "mmau-test-mini.json",
            evaluation_script=args.dataset_dir / "evaluation.py",
            htsat_checkpoint=args.dataset_dir / "HTSAT",
            mellow_root=args.route_root,
            max_prompt_tokens=129,
            max_new_tokens=300,
            audio_root=args.dataset_dir / "test-mini-audios",
        )

    def _initialize_official_store(self, args: types.SimpleNamespace) -> None:
        official_training = self.module.official_training
        official_training.official._ensure_output_dir(
            self._official_args(args),
            prediction_format=official_training.PREDICTION_FORMAT,
            prompt_format=official_training.official.MELLOW_AUTHOR_REPLY_PROMPT_FORMAT,
            protocol=official_training.PROTOCOL_CONTRACT,
        )

    def test_new_claim_keeps_official_output_directory_empty_for_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(Path(temporary))
            self._claim(args)

            sidecar = self.module._identity_sidecar(args.output_dir)
            self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8")), self._identity(args))
            self.assertFalse(args.output_dir.exists())

            self._initialize_official_store(args)
            self.assertTrue((args.output_dir / "run_config.json").is_file())
            self._claim(args)

    def test_legacy_in_directory_marker_is_migrated_before_official_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(Path(temporary))
            args.output_dir.mkdir(parents=True)
            legacy = args.output_dir / self.module.LEGACY_IDENTITY_FILENAME
            legacy.write_text(json.dumps(self._identity(args)), encoding="utf-8")

            self._claim(args)

            self.assertFalse(legacy.exists())
            self.assertEqual(
                json.loads(self.module._identity_sidecar(args.output_dir).read_text(encoding="utf-8")),
                self._identity(args),
            )
            self.assertEqual(list(args.output_dir.iterdir()), [])
            self._initialize_official_store(args)
            self.assertTrue((args.output_dir / "run_config.json").is_file())

    def test_mismatched_sidecar_refuses_to_reuse_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(Path(temporary))
            sidecar = self.module._identity_sidecar(args.output_dir)
            sidecar.parent.mkdir(parents=True)
            identity = self._identity(args)
            identity["checkpoint_sha256"] = "different-checkpoint"
            sidecar.write_text(json.dumps(identity), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "different checkpoint or config"):
                self._claim(args)


class GBS256NewRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_grouped_learning_rate_checkpoint_and_runtime(self) -> None:
        for bridge_max_lr in (4e-4, 8e-4):
            with self.subTest(bridge_max_lr=bridge_max_lr), tempfile.TemporaryDirectory() as temporary:
                checkpoint = {
                    "schema_version": 2,
                    "state_dict": {"weight": object()},
                    "epoch_completed": 5,
                    "num_epochs": 5,
                    "batch_geometry": {
                        "per_rank_batch_size": 8,
                        "world_size": 8,
                        "gradient_accumulation_steps": 4,
                    },
                    "loss_reduction": "global_token_mean",
                    "optimizer_contract": {
                        "type": "AdamW",
                        "betas": [0.9, 0.95],
                        "lr_groups": [{"lr_group": "bridge"}, {"lr_group": "lm"}],
                    },
                    "scheduler": {
                        "scheduler_type": "step_cosine_warmup",
                        "max_lr_by_group": {"bridge": bridge_max_lr, "lm": 1e-4},
                        "min_lr_by_group": {"bridge": bridge_max_lr * 0.1, "lm": 1e-5},
                        "total_steps": 100,
                        "warmup_steps": 5,
                        "last_step": 100,
                    },
                    "total_step": 100,
                }
                torch_stub = types.ModuleType("torch")
                torch_stub.load = lambda *_args, **_kwargs: checkpoint
                with mock.patch.dict(sys.modules, {"torch": torch_stub}):
                    contract = self.module._load_checkpoint(Path(temporary) / "model--epo-5.ckpt")
                runtime_config = {
                    "data": {"ip_text_len": 129},
                    "model": {
                        "model_type": "Mellow",
                        "encoder": {
                            "audioenc_name": "HTSAT",
                            "d_proj": 576,
                            "pretrained_audioencoder_path": "/models/HTSAT",
                        },
                        "decoder": {
                            "prefix_length": 40,
                            "total_prefix_length": 389,
                            "text_decoder": "/models/SmolLM2",
                        },
                    },
                    "train": {
                        "batch_size": 8,
                        "gradient_accumulation_steps": 4,
                        "num_epochs": 5,
                        "optimizer": {
                            "optimizer_type": "AdamW",
                            "betas": [0.9, 0.95],
                            "learning_rate": bridge_max_lr,
                            "lr_groups": {
                                "bridge": {"max_lr": bridge_max_lr, "min_lr": bridge_max_lr * 0.1},
                                "lm": {"max_lr": 1e-4, "min_lr": 1e-5},
                            },
                            "warmup_ratio": 0.05,
                            "weight_decay": 1e-4,
                        },
                    },
                }
                runtime_path = Path(temporary) / "runtime_5epochs.yaml"
                runtime_path.write_text(json.dumps(runtime_config), encoding="utf-8")
                with mock.patch.object(self.module.yaml, "safe_load", side_effect=json.loads):
                    self.assertEqual(self.module._runtime_route_variant(runtime_path), "bridge_lm_lr")
                    runtime_contract = self.module._validate_runtime(runtime_path, contract)
                self.assertTrue(contract["lr_grouped"])
                self.assertTrue(runtime_contract["lr_grouped"])
                self.assertEqual(
                    self.module._resolve_route(
                        contract["loss_reduction"], contract["route_variant"], runtime_contract["route_variant"]
                    )[0],
                    "reasonaqa_global_batch256_bridge_lm_lr_5epochs",
                )

    def test_attention_mask_flag_is_route_specific(self) -> None:
        for route_variant in ("attention_mask", "global_clip", "bridge_lm_lr", None):
            with self.subTest(route_variant=route_variant):
                args = types.SimpleNamespace(
                    route_variant=route_variant,
                    mode="full",
                    checkpoint_file=Path("model--epo-5.ckpt"),
                    runtime_config=Path("runtime_5epochs.yaml"),
                    route_root=Path("route"),
                    dataset_dir=Path("dataset"),
                    output_dir=Path("output"),
                    parquet_batch_size=8,
                    max_prompt_tokens=129,
                    max_new_tokens=300,
                    dtype="fp32",
                    parquet=None,
                    metadata_json=None,
                    evaluation_script=None,
                    audio_root=None,
                )
                delegated = self.module._delegate_args(args)
                self.assertEqual(delegated.attention_mask_generation, route_variant == "attention_mask")
                with mock.patch.object(self.module.adapter, "sha256_file", return_value="digest"):
                    identity = self.module._evaluation_identity(args)
                self.assertEqual(
                    "generation_attention_mask" in identity,
                    route_variant == "attention_mask",
                )

    def test_attention_mask_prefix_preserves_right_padding(self) -> None:
        class Mask:
            def __init__(self, values):
                self.values = values
                self.shape = (1, len(values))
                self.dtype = "long"
                self.device = "cpu"

        torch_stub = types.ModuleType("torch")
        torch_stub.inference_mode = nullcontext
        torch_stub.ones = lambda shape, **_kwargs: Mask([1] * shape[1])
        torch_stub.cat = lambda tensors, dim: Mask([value for tensor in tensors for value in tensor.values])
        prompt_mask = Mask([1] * 40 + [0] * 89)
        prefix = types.SimpleNamespace(shape=(1, 389, 576), device="cpu")
        adapter = self.module.adapter
        with (
            mock.patch.dict(sys.modules, {"torch": torch_stub}),
            mock.patch.object(adapter, "tokenize_training_prompt", return_value=({"attention_mask": prompt_mask}, {})),
            mock.patch.object(adapter, "build_native_prefix", return_value=(prefix, {})),
            mock.patch.object(adapter, "greedy_decode_native", return_value={}) as decode,
        ):
            adapter.run_generation(
                None, None, "cpu", {"prompt": "question", "waveform": None},
                max_prompt_tokens=129, max_new_tokens=300, common=None,
                use_padding_attention_mask=True,
            )
            mask = decode.call_args.kwargs["attention_mask"]
            self.assertEqual(mask.values, [1] * 300 + [0] * 89)
            adapter.run_generation(
                None, None, "cpu", {"prompt": "question", "waveform": None},
                max_prompt_tokens=129, max_new_tokens=300, common=None,
            )
            self.assertIsNone(decode.call_args.kwargs["attention_mask"])

    def test_generated_tokens_extend_mask_without_changing_legacy_calls(self) -> None:
        class Tensor:
            def __init__(self, values, width=None):
                self.values = values
                self.shape = (1, len(values)) if width is None else (1, len(values), width)
                self.dtype = "long"
                self.device = "cpu"

            def to(self, **_kwargs):
                return self

        torch_stub = types.ModuleType("torch")
        torch_stub.long = "long"
        torch_stub.tensor = lambda *_args, **_kwargs: object()
        torch_stub.ones = lambda shape, **_kwargs: Tensor([1] * shape[1])
        torch_stub.cat = lambda tensors, dim: Tensor(
            [value for tensor in tensors for value in tensor.values],
            tensors[0].shape[2] if len(tensors[0].shape) == 3 else None,
        )
        torch_stub.sort = lambda *_args, **_kwargs: (mock.MagicMock(), mock.MagicMock())
        torch_stub.softmax = lambda *_args, **_kwargs: mock.MagicMock()
        removable = mock.MagicMock()
        removable.shape = (0,)
        cumulative = mock.MagicMock()
        cumulative.__gt__.return_value = removable
        torch_stub.cumsum = lambda *_args, **_kwargs: cumulative
        torch_stub.argmax = mock.Mock(side_effect=[types.SimpleNamespace(item=lambda: 7)] * 3)

        logits = mock.MagicMock()
        logits.__getitem__.return_value = mock.MagicMock()
        lm = mock.Mock(return_value=types.SimpleNamespace(logits=logits))
        lm.model = types.SimpleNamespace(embed_tokens=lambda _token: Tensor([0], 576))
        model = types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=lm))
        tokenizer = types.SimpleNamespace(
            eos_token_id=99,
            eos_token="<|endoftext|>",
            decode=lambda *_args, **_kwargs: "answer",
        )
        prefix = Tensor([0] * 389, 576)
        initial_mask = Tensor([1] * 300 + [0] * 89)
        with mock.patch.dict(sys.modules, {"torch": torch_stub}):
            self.module.adapter.greedy_decode_native(
                model, tokenizer, prefix, max_new_tokens=2, attention_mask=initial_mask
            )
            self.assertEqual(lm.call_args_list[0].kwargs["attention_mask"].values, initial_mask.values)
            self.assertEqual(lm.call_args_list[1].kwargs["attention_mask"].values, initial_mask.values + [1])
            lm.reset_mock()
            self.module.adapter.greedy_decode_native(model, tokenizer, prefix, max_new_tokens=1)
            self.assertNotIn("attention_mask", lm.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
