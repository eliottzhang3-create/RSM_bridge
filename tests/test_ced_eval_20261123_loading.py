"""Run actual new checkpoint loader control flow with explicit CPU/CUDA API fixtures."""
import copy
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/RSmol'),str(ROOT/'tests')]
from ced_eval_20261123 import registry, runtime, arguments, sources, observers
from test_ced_eval_20261123 import checkpoint_fixture, cache


class Tensor(np.ndarray):
    def is_floating_point(self):return self.dtype.kind=='f'
    def numel(self):return self.size


def tensor(value,dtype=np.float32):return np.asarray(value,dtype=dtype).view(Tensor)


E='caption_decoder.lm.model.embed_tokens.weight'
H='caption_decoder.lm.lm_head.weight'


class LoadingTests(unittest.TestCase):
    def test_actual_normalizer_rejects_missing_extra_wrong_dtype_shape_nonfinite_and_tied_conflict(self):
        target={E:tensor([[1,2],[3,4]]),H:tensor([[1,2],[3,4]])}
        model=types.SimpleNamespace(state_dict=lambda:target)
        torch=types.SimpleNamespace(Tensor=Tensor,isfinite=np.isfinite,equal=np.array_equal)
        with patch.dict(sys.modules,{'torch':torch}):
            actual,ddp=runtime.normalize_state({'state_dict':copy.deepcopy(target)},model)
            self.assertFalse(ddp);self.assertEqual(set(actual),set(target))
            _,ddp=runtime.normalize_state({'state_dict':{'module.'+k:v for k,v in target.items()}},model)
            self.assertTrue(ddp)
            bads=[{E:target[E]},{**target,'extra':target[E]},
                  {**target,E:tensor([[1,2],[3,4]],np.float64)},
                  {**target,E:tensor([[1,2]])},{**target,E:tensor([[np.nan,2],[3,4]])},
                  {**target,H:tensor([[8,2],[3,4]])},{E:target[E],'module.'+H:target[H]}]
            for state in bads:
                with self.assertRaises(ValueError):runtime.normalize_state({'state_dict':state},model)

    def test_actual_loader_strict_overwrite_audit_then_freeze_and_pad_override_both_models(self):
        for alias in registry.ROUTES:
            with self.subTest(alias=alias),tempfile.TemporaryDirectory() as directory:
                path=pathlib.Path(directory)/'run/formal/checkpoints/model--step-5771.ckpt'
                path.parent.mkdir(parents=True);path.write_bytes(b'explicit model fixture')
                c,saved,config=checkpoint_fixture(alias,5771)
                (path.parent.parent/'runtime.json').write_text(__import__('json').dumps(config))
                saved['state_dict']={E:tensor([[5,6],[7,8]]),H:tensor([[5,6],[7,8]])}
                events=[]
                class Model:
                    def __init__(self):
                        self.training=True;self.frozen=False
                        self.state={E:tensor([[0,0],[0,0]]),H:tensor([[0,0],[0,0]])}
                        emb=types.SimpleNamespace(num_embeddings=49152)
                        self.caption_decoder=types.SimpleNamespace(lm=types.SimpleNamespace(
                            config=types.SimpleNamespace(vocab_size=49152),get_input_embeddings=lambda:emb))
                        self.audio_encoder=types.SimpleNamespace(base=types.SimpleNamespace(ced=types.SimpleNamespace(
                            modules=lambda:[types.SimpleNamespace(training=False)],parameters=lambda:[])))
                    def float(self):events.append('float');return self
                    def state_dict(self):return self.state
                    def load_state_dict(self,state,strict):
                        self_outer.assertTrue(strict);events.append('strict_load');self.state=copy.deepcopy(state)
                    def to(self,device):events.append('to_device');return self
                    def eval(self):events.append('eval');self.training=False;return self
                    def requires_grad_(self,value):events.append('freeze');self.frozen=not value;return self
                    def modules(self):return [self]
                    def parameters(self):return self.state.values()
                class Tokenizer:
                    pad_token='<|endoftext|>'
                    def add_special_tokens(self,value):
                        events.append('pad_override');self.pad_token=value['pad_token'];return 1
                    def encode(self,text):return [17]
                self_outer=self
                model,tokenizer=Model(),Tokenizer()
                def audit_model(*a):
                    events.append('audit_trained_state');self.assertFalse(model.frozen)
                    self.assertTrue(np.array_equal(model.state[E],saved['state_dict'][E]));return {'status':'PASS'}
                def source_tokens(tok,rows):
                    self.assertEqual(tok.pad_token,'<|endoftext|>');events.append('source_token_audit');return {}
                def tokens(tok,pad,rows):
                    self.assertEqual((tok.pad_token,pad,rows),('!',17,49152));events.append('mizar_token_audit');return {}
                cuda=types.SimpleNamespace(is_available=lambda:True,manual_seed_all=lambda *a:None,
                    set_device=lambda *a:None,get_device_name=lambda *a:'MOCK_CUDA_NO_GPU')
                torch=types.SimpleNamespace(Tensor=Tensor,float32=np.float32,isfinite=np.isfinite,equal=np.array_equal,
                    load=Mock(return_value=saved),manual_seed=lambda *a:None,set_num_threads=lambda *a:None,
                    cuda=cuda,device=lambda *a:'cuda:0',version=types.SimpleNamespace(cuda='FIXTURE'),__version__='FIXTURE',
                    backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()),cudnn=types.SimpleNamespace()))
                modules={'contract':c,'models.mellow':types.SimpleNamespace(Mellow=Mock(return_value=model)),
                    'data.audiotext_dataset':types.SimpleNamespace(AudioTextDataset=type('Dataset',(),{})),
                    'data.fixed20':object()}
                args=arguments.parse_args('mmsu',['--eval-variant',alias,'--checkpoint',str(path),'--sources-cache',str(cache())])
                with patch.dict(sys.modules,{'torch':torch,'transformers':types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)))}), \
                     patch.object(sources,'import_training',return_value=modules), \
                     patch.object(runtime.reference,'audit_assets',return_value={'status':'FIXTURE'}), \
                     patch.object(runtime,'audit_model',side_effect=audit_model), \
                     patch.object(c,'validate_checkpoint_tokenizer',side_effect=source_tokens,create=True), \
                     patch.object(c,'validate_tokenizer_contract',side_effect=tokens), \
                     patch.object(observers,'install_trace_audit',return_value={'status':'PENDING'}):
                    loaded=runtime.load_runtime(args)
                self.assertIs(loaded[0],model);self.assertTrue(model.frozen)
                self.assertLess(events.index('strict_load'),events.index('audit_trained_state'))
                self.assertLess(events.index('audit_trained_state'),events.index('freeze'))
                if alias!='smol_gbs128':self.assertLess(events.index('source_token_audit'),events.index('pad_override'))
                self.assertLess(events.index('pad_override'),events.index('mizar_token_audit'))
                self.assertTrue(loaded[5]['strict_state_dict']);torch.load.assert_called_once()
                self.assertFalse(hasattr(args,'_checkpoint_payload'))

    def test_cached_checkpoint_detects_replacement_between_identity_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'model--epo-3.ckpt';path.write_bytes(b'first')
            args=types.SimpleNamespace(checkpoint=path,expected_step=None,eval_variant='mesh_71437',sources_cache=cache())
            with patch.dict(sys.modules,{'torch':types.SimpleNamespace(load=lambda *a,**k:{'total_step':34626})}):
                runtime.read_checkpoint(args)
                path.write_bytes(b'different replacement')
                with self.assertRaisesRegex(ValueError,'changed after identity'):
                    runtime.read_checkpoint(args)


if __name__=='__main__':unittest.main()
