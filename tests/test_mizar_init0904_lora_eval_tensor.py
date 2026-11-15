"""Real selected training LM with nonzero trained gates and both LoRA branches.

No downloads; unavailable tensor dependencies are reported as explicit skips.
Only production assets, audio construction and CUDA use synthetic fixtures.
"""
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import test_mizar_init0904_lora_eval as helpers
import test_mizar_s1_ced_multilayer_eval_tensor as tensors


@unittest.skipUnless(tensors.HAS_TENSORS, "requires real torch and transformers; no downloaded dependencies")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = helpers.adapter.variant_spec(helpers.VARIANT)
        with patch.object(tensors, "LEGACY_SPECS", {helpers.VARIANT: spec}):
            tensors.MultilayerTensorTests.setUpClass.__func__(cls)

    model = tensors.MultilayerTensorTests.model
    inputs = tensors.MultilayerTensorTests.inputs
    call = staticmethod(tensors.MultilayerTensorTests.call)

    def single_route_test(self, name):
        with patch.object(tensors, "LEGACY_SPECS", {helpers.VARIANT: helpers.adapter.variant_spec(helpers.VARIANT)}):
            getattr(tensors.MultilayerTensorTests, name)(self)

    def test_every_decode_step_matches_direct_training_model_logits_and_tokens(self):
        self.single_route_test("test_adapter_matches_direct_model_logits_and_tokens_at_every_decode_step")

    def test_hooks_preserve_actual_model_state_logits_and_execution(self):
        self.single_route_test("test_hooks_preserve_real_logits_state_and_transient_memory_for_all_routes")

    def test_first_answer_logits_match_training_position382(self):
        self.single_route_test("test_training_decoder_and_prefix_evaluation_first_answer_logits_align")

    def test_sample_pair_order_changes_output_without_cross_sample_leaks(self):
        self.single_route_test("test_samples_and_prompt_masks_never_leak_across_full_forwards")

    def test_final_checkpoint_strict_load_keeps_trained_lora_and_gates_without_reset(self):
        t = self.torch
        def model():
            result = t.nn.Module()
            result.audio_encoder = t.nn.Module()
            result.audio_encoder.base = t.nn.Module()
            result.audio_encoder.base.ced = t.nn.Linear(1, 1)
            result.audio_encoder.base.mapper = t.nn.Linear(1, 1)
            result.caption_decoder = t.nn.Module()
            result.caption_decoder.lm = self.model(helpers.VARIANT)
            return result.float().eval()
        source, restored = model(), model()
        with t.no_grad():
            for name, value in source.named_parameters():
                if ".lora_" in name or "audio_gate" in name:
                    value.normal_(.03, .005)
        helper = helpers.RuntimeTests()
        helper.setUp()
        with tempfile.TemporaryDirectory() as directory:
            state = {"module." + key: value.clone() for key, value in source.state_dict().items()}
            args, _, modules, packages = helpers.helpers.plain.RecursiveCedEvaluationTests.loader_fixture(helper, directory, state)
            modules["models.mellow"].Mellow.return_value = restored
            t.save({**helper.fixture()[1], "state_dict": state}, args.checkpoint)
            cpu = t.device("cpu")
            proxy = types.ModuleType("torch")
            proxy.__getattr__ = lambda name: getattr(t, name)
            proxy.device = lambda _: cpu
            proxy.cuda = types.SimpleNamespace(is_available=lambda: True, manual_seed_all=Mock(),
                set_device=Mock(), get_device_name=lambda _: "real CPU tensor fixture")
            packages["torch"] = proxy
            def audit(*_a, **_k):
                for key, value in source.state_dict().items():
                    t.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
                return {"tiny_fixture": True}
            with (patch.dict(sys.modules, packages),
                  patch.object(helpers.adapter, "import_training_route", return_value=modules),
                  patch.object(helpers.adapter, "audit_assets", return_value={"fixture_only": True}),
                  patch.object(helpers.adapter, "audit_recursive_route", side_effect=audit),
                  patch.object(helpers.adapter, "install_recursive_trace_audit", return_value={})):
                loaded = helpers.adapter.load_runtime(args)[0]
        self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
        self.assertTrue(all(not module.training for module in loaded.modules()))
        prefix, context = self.inputs(helpers.VARIANT)
        with t.inference_mode():
            expected = self.call(source.caption_decoder.lm, prefix, context).logits
            actual = self.call(loaded.caption_decoder.lm, prefix, context).logits
        t.testing.assert_close(actual, expected, rtol=0, atol=0)
        missing = dict(source.state_dict())
        missing.pop(next(key for key in missing if ".lora_B." in key))
        with self.assertRaises(RuntimeError):
            loaded.load_state_dict(missing, strict=True)


if __name__ == "__main__":
    unittest.main()
