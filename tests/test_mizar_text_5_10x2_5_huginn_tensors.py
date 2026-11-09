"""Real small CPU tensor checks; never substitute these for production validation."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_mizar_text_5_10x2_5_huginn import C, T, V

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    import safetensors
    from code.RSmol.recursive_model_mizar_text_5_10x2_5_huginn import MizarHuginnForCausalLM
    from code.RSmol import recursive_model_mizar_text_5_10x2_5_huginn as M
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
        self.target=MizarHuginnForCausalLM(C.build_target_config(self.source.config)).float().eval()

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
        self.assertEqual(count,len(self.target.state_dict())-1)
        audit,logits=V.audit_forward(self.target)
        self.assertEqual(audit['physical_execution_trace'],list(C.SCHEDULE))
        with tempfile.TemporaryDirectory() as tmp:
            self.target.save_pretrained(tmp,safe_serialization=True)
            reloaded=MizarHuginnForCausalLM.from_pretrained(tmp,local_files_only=True,
                       torch_dtype=torch.float32,attn_implementation='eager')
            V.assert_inherited(self.source.state_dict(),reloaded)
            _,other=V.audit_forward(reloaded)
            torch.testing.assert_close(logits,other,rtol=0,atol=0)

    def test_actual_two_adapter_calls_prelude_reuse_and_full_gradients(self):
        V.inherited_state(self.source.state_dict(),self.target)
        captured=[]; core=[]; prelude=[]
        def value(out): return out if torch.is_tensor(out) else out[0]
        handles=[self.target.model.recurrent_adapter.register_forward_pre_hook(
                    lambda _m,args:captured.append(args[0])),
                 self.target.model.layers[4].register_forward_hook(
                    lambda _m,_a,out:prelude.append(value(out))),
                 self.target.model.layers[14].register_forward_hook(
                    lambda _m,_a,out:core.append(value(out)))]
        try:
            with patch.object(M,'sample_initial_state',wraps=M.sample_initial_state) as sample:
                logits=self.target(input_ids=torch.tensor([[3,4,5,6]]),use_cache=False).logits
                self.assertEqual(sample.call_count,1)
            self.assertEqual(len(captured),2)
            self.assertEqual(len(core),2)
            for v in captured: torch.testing.assert_close(v[...,16:],prelude[0],rtol=0,atol=0)
            torch.testing.assert_close(captured[1][...,:16],core[0],rtol=0,atol=0)
            for v in (prelude[0],core[0]): v.retain_grad()
            logits.square().mean().backward()
            for v in (prelude[0],core[0]):
                self.assertIsNotNone(v.grad)
                self.assertGreater(v.grad.abs().sum().item(),0)
            for half in self.target.model.recurrent_adapter.weight.grad.split(16,dim=1):
                self.assertGreater(half.abs().sum().item(),0)
            for layer in self.target.model.layers:
                self.assertGreater(sum(p.grad.abs().sum().item() for p in layer.parameters() if p.grad is not None),0)
        finally:
            for h in handles: h.remove()

    def test_noise_distribution_and_adapter_deterministic_initialization(self):
        state=M.sample_initial_state(torch.empty(1000,100))
        self.assertTrue((state.abs()<=3*C.STATE_STD).all())
        self.assertLess(abs(state.mean().item()),.01)
        # A truncated normal has slightly smaller observed std than its
        # underlying normal; this guards against ordinary/unscaled Gaussian.
        self.assertAlmostEqual(state.std().item(),.624,delta=.01)
        before=torch.get_rng_state().clone()
        a=M.make_huginn_adapter(576)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        b=M.make_huginn_adapter(576)
        self.assertTrue(torch.equal(a.weight,b.weight))
        std=(2/(5*576))**.5
        self.assertTrue((a.weight.abs()<=3*std).all())
        self.assertAlmostEqual(a.weight.std().item(),std*.9866,delta=.0002)
        self.assertEqual(a.weight.numel(),663552)
        self.assertIsNone(a.bias)

    def test_controlled_forward_manual_reference_and_bf16_residual_dtype(self):
        model=self.target.model
        ids=torch.tensor([[3,4,5]])
        state=M.sample_initial_state(torch.empty(1,3,16))
        with torch.no_grad():
            actual=self.target(input_ids=ids,initial_state=state,use_cache=False).logits
            hidden=model.embed_tokens(ids)
            positions=torch.arange(3).unsqueeze(0)
            mask=M._causal_mask(attention_mask=None,batch_size=1,query_length=3,
                    past_length=0,dtype=hidden.dtype,device=hidden.device)
            rotary=model.rotary_emb(hidden,position_ids=positions)
            for logical,physical in enumerate(C.SCHEDULE):
                if logical==5:
                    prelude=hidden
                    hidden=model.recurrent_adapter(torch.cat([state,prelude],-1))
                if logical==15:
                    hidden=model.recurrent_adapter(torch.cat([hidden,prelude],-1))
                hidden=M._call_decoder_layer(model.layers[physical],hidden,attention_mask=mask,
                    position_ids=positions,cache=None,use_cache=False,cache_position=positions[0],
                    position_embeddings=rotary,output_attentions=False)[0]
            expected=self.target.lm_head(model.norm(hidden))
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            seen=[]
            handle=model.layers[5].register_forward_pre_hook(lambda _m,args:seen.append(args[0].dtype))
            try:
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    self.target(input_ids=ids,initial_state=state,use_cache=False)
            finally: handle.remove()
            self.assertEqual(seen,[torch.float32,torch.float32])

    def test_trained_adapter_survives_epoch2_weights_only_reload(self):
        with torch.no_grad(): self.target.model.recurrent_adapter.weight.add_(.25)
        expected=self.target.model.recurrent_adapter.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            self.target.save_pretrained(tmp,safe_serialization=True)
            other,loading=MizarHuginnForCausalLM.from_pretrained(tmp,local_files_only=True,
                    torch_dtype=torch.float32,output_loading_info=True)
            self.assertFalse(any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')))
            self.assertTrue(torch.equal(expected,other.model.recurrent_adapter.weight))
            self.assertEqual(set(other.state_dict())-set(self.source.state_dict()),{C.ADAPTER_KEY})

    def test_missing_adapter_rejected_by_runtime_strict_loader(self):
        import safetensors.torch
        with tempfile.TemporaryDirectory() as tmp:
            self.target.save_pretrained(tmp,safe_serialization=True)
            # Satisfy only the pre-load tokenizer artifact check. The runtime
            # must reject missing A before attempting tokenizer parsing.
            (Path(tmp)/'tokenizer.json').write_text('{}')
            path=Path(tmp)/'model.safetensors'
            state=safetensors.torch.load_file(path)
            del state[C.ADAPTER_KEY]
            safetensors.torch.save_file(state,path,metadata={'format':'pt'})
            with self.assertRaisesRegex(ValueError,'Strict recursive model load failed'):
                T._load_runtime_model(Path(tmp),torch.device('cpu'))

    def test_cache_matches_full_forward_with_same_per_token_initial_state(self):
        ids=torch.tensor([[3,4,5,6]])
        state=M.sample_initial_state(torch.empty(1,4,16))
        with torch.no_grad():
            full=self.target(input_ids=ids,initial_state=state,use_cache=False).logits
            cache=None; pieces=[]
            for i in range(4):
                result=self.target(input_ids=ids[:,i:i+1],initial_state=state[:,i:i+1],
                    attention_mask=torch.ones(1,i+1,dtype=torch.long),
                    past_key_values=cache,use_cache=True)
                cache=result.past_key_values
                pieces.append(result.logits)
            torch.testing.assert_close(torch.cat(pieces,dim=1),full,rtol=1e-5,atol=1e-5)
            self.assertEqual(len(cache),30)

    def test_rank_rng_independence_restore_and_trace_audit_preservation(self):
        T._set_seed(0,rank=0); zero=T._capture_rank_rng(rank=0)
        first=M.sample_initial_state(torch.empty(1,4,16))
        T._set_seed(0,rank=1); one=T._capture_rank_rng(rank=1)
        second=M.sample_initial_state(torch.empty(1,4,16))
        self.assertFalse(torch.equal(first,second))
        for rank,saved,expected in ((0,zero,first),(1,one,second)):
            T._restore_rank_rng(saved,rank=rank)
            audit=T.recursive_forward_trace_audit(self.target,device=torch.device('cpu'))
            self.assertTrue(audit['training_rng_preserved'])
            self.assertTrue(torch.equal(saved['torch_cpu'],torch.get_rng_state()))
            self.assertTrue(torch.equal(expected,M.sample_initial_state(torch.empty(1,4,16))))
        with self.assertRaises(ValueError): T._restore_rank_rng(zero,rank=1)

    def test_complete_checkpoint_roundtrip_restores_each_rank_rng(self):
        # Exercise the actual save/restore path, including serialization of
        # independent CPU RNG states, optimizer moments, scheduler and cursors.
        config=T.Stage4Config(gate='A',world_size=2)
        optimizer=torch.optim.AdamW(self.target.parameters(),lr=8e-4,betas=(.9,.95),
                                     eps=1e-8,weight_decay=.1)
        scheduler=T._load_scheduler(optimizer,463,9244,max_lr=8e-4,min_lr=8e-5)
        self.target(torch.tensor([[3,4,5]]),use_cache=False).logits.square().mean().backward()
        optimizer.step(); scheduler.step()
        rng={}; expected={}
        for rank in range(2):
            T._set_seed(0,rank=rank)
            rng[str(rank)]=T._capture_rank_rng(rank=rank)
            expected[rank]=M.sample_initial_state(torch.empty(1,4,16))
        def gather(values,local): values[:]=[rng['0'],rng['1']]
        with patch('torch.distributed.is_initialized',return_value=True), \
             patch('torch.distributed.get_world_size',return_value=2), \
             patch('torch.distributed.all_gather_object',side_effect=gather):
            gathered=T._gather_rank_rng(rank=0,world_size=2)
        for rank in range(2):
            self.assertTrue(torch.equal(gathered[str(rank)]['torch_cpu'],rng[str(rank)]['torch_cpu']))
        cursors={str(i):{'epoch':0,'shard_index':0,'shard_microbatches_seen':16,
                         'samples_seen':128,'microbatches_seen':16} for i in range(2)}
        class Tokenizer:
            def save_pretrained(self,path): (path/'tokenizer.json').write_text('{}')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint-step-000500'
            T.save_complete_checkpoint(path,model=self.target,tokenizer=Tokenizer(),
                optimizer=optimizer,scheduler=scheduler,optimizer_step=500,
                cumulative_samples=1024,cumulative_tokens=2048,data_cursors_by_rank=cursors,
                rng_states_by_rank=rng,manifest={},config=config,report={},rank=0)
            marker=json.loads((path/'checkpoint_complete.json').read_text())
            self.assertEqual(marker['rng_states_by_rank'],['0','1'])
            for rank in range(2):
                T._set_seed(999,rank=rank)
                restored=T._restore_training_state(path,optimizer=optimizer,scheduler=scheduler,
                                                   config=config,rank=rank)
                self.assertEqual(restored[:3],(500,1024,2048))
                self.assertEqual(restored[3],cursors[str(rank)])
                self.assertTrue(torch.equal(expected[rank],M.sample_initial_state(torch.empty(1,4,16))))
            state=torch.load(path/'training_state.pt',map_location='cpu',weights_only=False)
            state['rng_states_by_rank']={'0':rng['0']}
            torch.save(state,path/'training_state.pt')
            with self.assertRaisesRegex(ValueError,'per-rank RNG'):
                T._restore_training_state(path,optimizer=optimizer,scheduler=scheduler,config=config,rank=1)

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
