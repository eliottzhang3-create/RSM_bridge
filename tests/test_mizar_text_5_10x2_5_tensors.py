"""Real small CPU tensor checks; never substitute these for production validation."""
from __future__ import annotations
import copy
from pathlib import Path
import sys
import tempfile
import unittest

from test_mizar_text_5_10x2_5 import C, T, V

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    import safetensors
    from code.RSmol.recursive_model_5_10_5 import RecursiveLlamaForCausalLM
    AVAILABLE=True
except ImportError as error:
    AVAILABLE=False
    MISSING=str(error)


@unittest.skipUnless(AVAILABLE, 'CPU tensor dependencies unavailable: ' + (globals().get('MISSING','')))
class TensorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)
        self.config=LlamaConfig(num_hidden_layers=30,hidden_size=16,intermediate_size=32,
                                num_attention_heads=4,num_key_value_heads=2,vocab_size=43,
                                tie_word_embeddings=True,max_position_embeddings=128,
                                pad_token_id=0,bos_token_id=1,eos_token_id=2,
                                _attn_implementation='eager')
        self.source=LlamaForCausalLM(self.config).float().eval()
        self.target=RecursiveLlamaForCausalLM(C.build_target_config(self.source.config)).float().eval()

    def test_real_final_namespace_extraction_and_strict_load(self):
        state={C.LM_PREFIX+k:v for k,v in self.source.state_dict().items()}
        state['audio_encoder.fake']=torch.ones(3)
        text=V.extract_lm_state(state)
        self.assertEqual(set(text),set(self.source.state_dict()))
        V.strict_load(self.source,text)
        with self.assertRaises(ValueError):
            V.extract_lm_state({'state_dict':state})
        with self.assertRaises(ValueError):
            V.strict_load(self.source,{**text,'extra':torch.ones(1)})

    def test_real_tied_head_mismatch_fails_before_load(self):
        state={C.LM_PREFIX+k:v.clone() for k,v in self.source.state_dict().items()}
        state[C.LM_PREFIX+'lm_head.weight'].add_(1)
        with self.assertRaisesRegex(ValueError,'tied'):
            V.extract_lm_state(state)

    def test_selected_inheritance_trace_and_save_reload(self):
        count=V.inherited_state(self.source.state_dict(),self.target)
        self.assertEqual(count,len(self.target.state_dict()))
        audit,logits=V.audit_forward(self.target)
        self.assertEqual(audit['physical_execution_trace'],list(C.SCHEDULE))
        with tempfile.TemporaryDirectory() as tmp:
            self.target.save_pretrained(tmp,safe_serialization=True)
            reloaded=RecursiveLlamaForCausalLM.from_pretrained(tmp,local_files_only=True,
                       torch_dtype=torch.float32,attn_implementation='eager')
            V.assert_inherited(self.source.state_dict(),reloaded)
            _,other=V.audit_forward(reloaded)
            torch.testing.assert_close(logits,other,rtol=0,atol=0)

    def test_global_token_weighted_gradient_matches_concatenated_tokens(self):
        # Simulate the DDP average over two unequal token populations.
        w=torch.tensor(0.2,requires_grad=True)
        logits1=torch.stack([w,-w,w*2]).repeat(2,2,1)
        logits2=torch.stack([w*3,-w,w]).repeat(1,3,1)
        labels1=torch.tensor([[0,1],[1,-100]])
        labels2=torch.tensor([[0,2,1]])
        sum1,n1=T._causal_loss_sum(logits1,labels1)
        sum2,n2=T._causal_loss_sum(logits2,labels2)
        scale=T.token_weighted_gradient_scale(world_size=2,global_window_tokens=int(n1+n2))
        actual=torch.autograd.grad((sum1*scale+sum2*scale)/2,w,retain_graph=True)[0]
        expected=torch.autograd.grad((sum1+sum2)/(n1+n2),w)[0]
        torch.testing.assert_close(actual,expected)

    def test_epoch2_optimizer_and_scheduler_start_fresh(self):
        param=torch.nn.Parameter(torch.ones(3))
        first=torch.optim.AdamW([param],lr=8e-4,betas=(.9,.95),eps=1e-8,weight_decay=.1)
        sched1=T._load_scheduler(first,463,9244,max_lr=8e-4,min_lr=8e-5)
        param.sum().backward(); first.step(); sched1.step()
        state=copy.deepcopy(first.state_dict())
        second=torch.optim.AdamW([param],lr=2e-4,betas=(.9,.95),eps=1e-8,weight_decay=.1)
        sched2=T._load_scheduler(second,463,9244,max_lr=2e-4,min_lr=2e-5)
        self.assertTrue(state['state'])
        self.assertEqual(second.state_dict()['state'],{})
        self.assertEqual(sched2.last_epoch,0)
        self.assertAlmostEqual(second.param_groups[0]['lr'],2e-5)


if __name__=='__main__':
    unittest.main()
