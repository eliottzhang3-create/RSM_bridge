"""Real tiny CPU LoRA checkpoint through the production evaluation loader.

Only audio construction, production assets/geometry, CUDA and tokenization are
fixtures. The final state load, model tensors and recursive LM forward are real.
"""
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import test_mizar_s1_ced48_text_epoch1_loop_lora_r8_tensor as inherited
import test_mizar_selected4_eval as helpers


@unittest.skipUnless(inherited.HAS_TENSORS, "requires torch, transformers and safetensors; no downloads")
class FinalAudioTensorTests(unittest.TestCase):
    setUpClass = classmethod(inherited.TensorTests.setUpClass.__func__)
    lm = inherited.TensorTests.lm
    fixture = inherited.TensorTests.fixture

    def test_trained_final_audio_lora_overwrites_text_values_and_matches_logits(self):
        torch = self.torch
        variant = helpers.TARGETS["text_lora_epoch1_serial48_r8"][0]
        selected = helpers.runtime.adapter(variant)[0]
        helper = helpers.RuntimeTests()
        helper.select(variant)
        source, restored = self.fixture(), self.fixture()
        for model in (source, restored):
            model.audio_encoder.base.ced = torch.nn.Linear(1, 1)
        with torch.no_grad():
            for name, value in source.named_parameters():
                if ".lora_" in name or "audio_gate" in name:
                    value.normal_(mean=.03, std=.005)
        source.float().eval()
        with tempfile.TemporaryDirectory() as directory:
            state = {"module." + name: value.clone() for name, value in source.state_dict().items()}
            args, _, modules, packages = helpers.plain.RecursiveCedEvaluationTests.loader_fixture(helper, directory, state)
            modules["models.mellow"].Mellow.return_value = restored
            checkpoint = {**helper.fixture()[1], "state_dict": state}
            torch.save(checkpoint, args.checkpoint)
            cpu = torch.device("cpu")
            proxy = types.ModuleType("torch")
            proxy.__getattr__ = lambda name: getattr(torch, name)
            proxy.device = lambda _: cpu
            proxy.cuda = types.SimpleNamespace(is_available=lambda: True, manual_seed_all=Mock(),
                set_device=Mock(), get_device_name=lambda _: "CPU tensor fixture")
            packages["torch"] = proxy
            def audit(*a, **k):
                for name, value in source.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
                return {"tiny_geometry_fixture": True}
            with (patch.dict(sys.modules, packages),
                  patch.object(selected, "import_training_route", return_value=modules),
                  patch.object(selected, "audit_assets", return_value={"fixture_only": True}),
                  patch.object(selected, "audit_recursive_route", side_effect=audit),
                  patch.object(selected, "install_recursive_trace_audit", return_value={})):
                loaded = selected.load_runtime(args)[0]
        self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
        self.assertTrue(all(not module.training for module in loaded.modules()))
        embeds = torch.randn(1, 130, 32)
        information = tuple(torch.randn(1, 126, 32) for _ in range(2))
        with torch.inference_mode():
            expected = source.caption_decoder.lm(inputs_embeds=embeds, audio_information=information, use_cache=False).logits
            actual = loaded.caption_decoder.lm(inputs_embeds=embeds, audio_information=information, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
