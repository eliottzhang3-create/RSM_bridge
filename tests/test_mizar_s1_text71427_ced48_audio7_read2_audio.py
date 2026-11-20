"""Real CPU tensors in mellow_c8204d8; synthetic CED, no assets/downloads/CUDA."""
import copy
import importlib.util
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / 'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_lora_r8'
HAS_TENSORS = all((importlib.util.find_spec(n) is not None for n in ('torch', 'torchaudio', 'transformers', 'soundfile')))

@unittest.skipUnless(HAS_TENSORS, 'requires torch/torchaudio/transformers/soundfile in mellow_c8204d8')
class TensorTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(ROUTE), str(ROOT / 'code/RSmol')]
        import torch
        import contract
        from models.ced import CEDSmallWrapper, CEDTemporalMapper, CEDFrequencyMerger
        cls.torch, cls.contract = (torch, contract)
        cls.wrapper_class, cls.mapper_class, cls.mlp_class = (CEDSmallWrapper, CEDTemporalMapper, CEDFrequencyMerger)
        torch.set_num_threads(1)

    def toy_wrapper(self):
        torch, nn = (self.torch, self.torch.nn)

        class Block(nn.Module):

            def __init__(self, index):
                super().__init__()
                self.offset = nn.Parameter(torch.tensor(float(index)), requires_grad=False)

            def forward(self, x):
                return x + self.offset

        class CED(nn.Module):
            maximal_allowed_length = 1012

            def __init__(self):
                super().__init__()
                self.init_bn = nn.BatchNorm2d(64)
                self.blocks = nn.Sequential(*(Block(i) for i in range(1, 13)))
                self.norm = nn.LayerNorm(384)

            def forward_features(self, x):
                self.calls = getattr(self, 'calls', 0) + 1
                self.base = torch.arange(x.shape[0] * 252 * 384, dtype=x.dtype).reshape(x.shape[0], 252, 384) / 1000
                return self.norm(self.blocks(self.base))
        wrapper = self.wrapper_class.__new__(self.wrapper_class)
        nn.Module.__init__(wrapper)
        wrapper.ced = CED().eval()
        for p in wrapper.ced.parameters():
            p.requires_grad = False
        wrapper.mapper = self.mapper_class(384, 4, 576, output_tokens=126, resample=False)
        wrapper.layer_mlps = nn.ModuleDict({str(k): self.mlp_class(384, 4, 576, dropout=0.0) for k in (4, 8)})
        wrapper.mapper_type = 'freq_merge_126'
        wrapper._waveform_to_mel = lambda wave: torch.zeros(wave.shape[0], 64, 2001)
        return wrapper

    def test_single_ced_forward_capture_block4_8_before_final_norm_and_hook_cleanup(self):
        torch = self.torch
        wrapper = self.toy_wrapper()
        final, intermediate = wrapper._forward_hidden(torch.zeros(2, 64, 2001))
        base = wrapper.ced.base
        self.assertEqual(wrapper.ced.calls, 1)
        torch.testing.assert_close(intermediate[4], (base + 10).reshape(2, 504, 384))
        torch.testing.assert_close(intermediate[8], (base + 36).reshape(2, 504, 384))
        torch.testing.assert_close(final, wrapper.ced.norm(base + 78).reshape(2, 504, 384))
        self.assertTrue(all((not b._forward_hooks for b in wrapper.ced.blocks)))
        with patch.object(wrapper.ced.blocks[5], 'forward', side_effect=RuntimeError('fixture failure')):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                wrapper._forward_hidden(torch.zeros(2, 64, 2001))
        self.assertTrue(all((not b._forward_hooks for b in wrapper.ced.blocks)))

    def test_wrapper_mlps_train_outside_no_grad_and_encoder_stays_frozen_eval(self):
        torch = self.torch
        wrapper = self.toy_wrapper().train()
        out = wrapper(torch.zeros(2, 640000))
        self.assertEqual(wrapper.ced.calls, 1)
        for name in ('embedding', 'layer4_audio', 'layer8_audio'):
            self.assertEqual(tuple(out[name].shape), (2, 126, 576))
            self.assertTrue(out[name].requires_grad)
        sum((v.square().mean() for k, v in out.items() if k != 'clipwise_output')).backward()
        self.assertTrue(all((p.grad is None for p in wrapper.ced.parameters())))
        self.assertTrue(all((p.grad is not None for p in wrapper.layer_mlps.parameters())))
        self.assertFalse(wrapper.ced.training)
        self.assertTrue(wrapper.mapper.training and wrapper.layer_mlps.training)

    def test_new_parameter_count_and_constructor_rng_mapper_parity(self):
        torch, nn = (self.torch, self.torch.nn)
        cfg = dict(self.contract.CED_EXPECTED_CONFIG, model_type='ced', id2label={str(i): str(i) for i in range(527)}, auto_map={'AutoConfig': 'configuration_ced.CedConfig', 'AutoModelForAudioClassification': 'modeling_ced.CedForAudioClassification'})
        encoder = self.toy_wrapper().ced
        encoder.config = SimpleNamespace(**cfg, to_dict=lambda: cfg)
        loaded = nn.Module()
        loaded.encoder = encoder
        spec = importlib.util.spec_from_file_location('mother_ced', ROUTE.with_name('mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small') / 'models/ced.py')
        parent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parent)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in self.contract.CED_ASSET_FILES:
                (root / name).write_bytes(b'synthetic')
            self.contract.write_json(root / 'config.json', cfg)
            with patch('models.ced.local_ced_config', return_value=nullcontext(encoder.config)), patch.object(parent, 'local_ced_config', return_value=nullcontext(encoder.config)), patch('models.ced.AutoModelForAudioClassification.from_pretrained', return_value=(loaded, {})):
                torch.manual_seed(12)
                old = parent.CEDSmallWrapper({'pretrained_audioencoder_path': directory})
                old_rng = torch.get_rng_state()
                torch.manual_seed(12)
                new = self.wrapper_class({'pretrained_audioencoder_path': directory})
                new_rng = torch.get_rng_state()
            self.assertTrue(torch.equal(old_rng, new_rng))
            for name, value in old.mapper.state_dict().items():
                torch.testing.assert_close(value, new.mapper.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(sum((p.numel() for p in new.layer_mlps.parameters())), 2443776)
            self.assertEqual(2443776 + 2 * (576 + 1), 2444930)
if __name__ == '__main__':
    unittest.main()
