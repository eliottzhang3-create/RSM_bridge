from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contracts as C
import runtime


class _FiniteResult:
    @staticmethod
    def all():
        return True


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...], dtype: str = "float32"):
        self.shape = shape
        self.dtype = dtype

    def to(self, *, device, dtype):
        self.dtype = dtype
        return self

    def contiguous(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return b"same-waveform"


class _FakeTorch:
    float32 = "float32"

    @staticmethod
    def isfinite(_value):
        return _FiniteResult()


def contract_fixture(*, route_variant: str, epochs: int, epoch: int,
                     max_lr: float, min_lr: float | None = None) -> tuple[dict, dict]:
    if min_lr is None:
        min_lr = 0.1 * max_lr if epochs == 10 else 4e-6
    policies = C.LR_ABLATION_POLICIES if epochs == 10 else C.POLICIES
    total_steps = epochs * 100
    current_step = epoch * 100
    checkpoint = {
        "schema_version": 2,
        "num_epochs": epochs,
        "epoch_completed": epoch,
        "total_step": current_step,
        "loss_reduction": "global_token_mean",
        **policies,
        "batch_geometry": {
            "per_rank_batch_size": 8,
            "world_size": 8,
            "gradient_accumulation_steps": 4,
        },
        "route_contract": None,
        "text_model_contract": None,
        "state_dict": {
            f"caption_decoder.lm.model.layers.{index}.self_attn.q_proj.weight": 0
            for index in range(30)
        },
        "optimizer_contract": {"type": "AdamW", "betas": [0.9, 0.95], "weight_decay": 1e-4},
        "scheduler": {
            "scheduler_type": "step_cosine_warmup",
            "max_lr": max_lr,
            "min_lr": min_lr,
            "total_steps": total_steps,
            "last_step": current_step,
            "warmup_steps": math.ceil(total_steps * 0.05),
        },
    }
    config = {
        **policies,
        "route_contract": None,
        "text_model_contract": None,
        "train": {
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "num_epochs": epochs,
            "gradient_clip_policy": "global_l2_hard_clip",
            "max_grad_norm": 0.5,
            "mixed_precision": {"use_mixed_precision": False},
            "optimizer": {
                "optimizer_type": "AdamW",
                "betas": [0.9, 0.95],
                "learning_rate": max_lr,
                "max_lr": max_lr,
                "min_lr": min_lr,
                "weight_decay": 1e-4,
                "warmup_ratio": 0.05,
                "scheduler": "step_cosine_warmup",
            },
        },
        "data": {
            "sampling_rate": 32000,
            "segment_seconds": 10,
            "ip_text_len": 129,
            "op_text_len": 250,
            "tokenizer_type": "/models/SmolLM2",
        },
        "model": {
            "model_type": "Mellow",
            "encoder": {
                "audioenc_name": "HTSAT",
                "out_emb": 768,
                "d_proj": 576,
                "freeze_audio_encoder_weights": True,
                "use_pretrained_audioencoder": True,
                "pretrained_audioencoder_path": "/models/HTSAT",
            },
            "decoder": {
                "prefix_length": 40,
                "total_prefix_length": 260,
                "freeze_gpt_weights": False,
                "text_decoder": "/models/SmolLM2",
            },
        },
    }
    return checkpoint, config


class MellowCheckpointContractTests(unittest.TestCase):
    def test_accepts_any_saved_epoch_and_positive_lr_for_10epoch_route(self):
        for epoch, max_lr in ((1, 1e-7), (5, 1e-3), (10, 0.25)):
            with self.subTest(epoch=epoch, max_lr=max_lr):
                checkpoint, config = contract_fixture(
                    route_variant=C.LR_ABLATION_ROUTE_VARIANT,
                    epochs=10, epoch=epoch, max_lr=max_lr,
                )
                architecture, model_contract = C.validate_contract(checkpoint, config)
                self.assertEqual(architecture, "smollm2")
                self.assertEqual(model_contract["training_epochs"], 10)
                self.assertEqual(model_contract["max_lr"], max_lr)
                self.assertEqual(model_contract["min_lr"], 0.1 * max_lr)

    def test_rejects_lr_mismatch_and_non_smollm2_route(self):
        checkpoint, config = contract_fixture(
            route_variant=C.LR_ABLATION_ROUTE_VARIANT, epochs=10, epoch=3, max_lr=1e-3,
        )
        checkpoint["scheduler"]["min_lr"] = 2e-4
        with self.assertRaisesRegex(ValueError, "min_lr"):
            C.validate_contract(checkpoint, config)

        checkpoint, config = contract_fixture(
            route_variant=C.LR_ABLATION_ROUTE_VARIANT, epochs=10, epoch=3, max_lr=1e-3,
        )
        checkpoint["route_contract"] = C.ARCHITECTURES["mesh_x2"]["route_contract"]
        checkpoint["text_model_contract"] = C.ARCHITECTURES["mesh_x2"]["text_model_contract"]
        config["route_contract"] = checkpoint["route_contract"]
        config["text_model_contract"] = checkpoint["text_model_contract"]
        with self.assertRaisesRegex(ValueError, "ordinary SmolLM2"):
            C.validate_contract(checkpoint, config)

    def test_keeps_formal30_lr_and_route_contract_strict(self):
        checkpoint, config = contract_fixture(
            route_variant=C.POLICIES["route_variant"], epochs=30, epoch=30, max_lr=4e-5,
        )
        architecture, _ = C.validate_contract(checkpoint, config)
        self.assertEqual(architecture, "smollm2")
        checkpoint["scheduler"]["max_lr"] = 5e-5
        with self.assertRaisesRegex(ValueError, "scheduler.max_lr"):
            C.validate_contract(checkpoint, config)

    def test_lr_route_uses_its_isolated_model_sources(self):
        self.assertEqual(
            C.route_for_architecture("smollm2", C.LR_ABLATION_ROUTE_VARIANT),
            C.LR_ABLATION_ROUTE,
        )
        with self.assertRaisesRegex(ValueError, "only produces ordinary SmolLM2"):
            C.route_for_architecture("mesh_x2", C.LR_ABLATION_ROUTE_VARIANT)

    def test_protocol_records_audio1_reuse_in_second_slot(self):
        protocol = C.protocol("mmau_test_mini")
        self.assertEqual(protocol["audio2"], "reuse_audio1_waveform_encoded_independently")
        self.assertIn("reused_audio2", protocol["version"])

    def test_prefix_builder_passes_identical_waveform_object_to_both_slots(self):
        audio = _FakeTensor((1, 320000))
        prefix = _FakeTensor((1, 267, 576))

        class Model:
            inputs = None

            def generate_prefix_inference(self, inputs):
                self.inputs = inputs
                return prefix, None, None

        model = Model()
        text_input = {"input_ids": _FakeTensor((1, 7))}
        with mock.patch.dict(sys.modules, {"torch": _FakeTorch()}):
            result, audit = runtime.make_prefix(model, audio, text_input, "cuda:0")
        self.assertIs(result, prefix)
        self.assertIs(model.inputs["audio1"], audio)
        self.assertIs(model.inputs["audio2"], audio)
        self.assertTrue(audit["audio2_equals_audio1"])
        self.assertEqual(audit["audio1_waveform_sha256"], audit["audio2_waveform_sha256"])


if __name__ == "__main__":
    unittest.main()
