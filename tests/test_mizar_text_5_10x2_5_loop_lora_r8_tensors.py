"""Real small CPU tensor checks; never substitute these for production validation."""
from __future__ import annotations
import copy
from pathlib import Path
import random
import tempfile
import unittest

from test_mizar_text_5_10x2_5_loop_lora_r8 import C, T, V

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    import safetensors
    from code.RSmol.recursive_model_mizar_text_5_10x2_5_loop_lora_r8 import MizarTextLoopLoRAR8ForCausalLM as RecursiveLlamaForCausalLM
    from code.RSmol.recursive_model_5_10_5 import RecursiveLlamaForCausalLM as BaseRecursiveLM
    from code.RSmol.mizar_text_5_10x2_5_loop_lora_r8.loop_lora import (
        LoopLoRALinear, validate_loop_lora, call_with_loop_lora, install_loop_lora,
    )
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

    def optimizer(self, model, max_lr=8e-4):
        parameters=list(model.parameters())
        optimizer=torch.optim.AdamW(parameters,lr=max_lr,betas=(.9,.95),eps=1e-8,weight_decay=.1)
        self.assertTrue(T.optimizer_parameter_audit(optimizer,parameters)['optimizer_matches_model_exactly_once'])
        return optimizer

    def task_loss(self, model):
        ids=torch.tensor([[3,4,5,6,7,8],[9,10,11,12,13,14]])
        output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
        summed,count=T._causal_loss_sum(output.logits,ids)
        return summed/count

    def test_scope_independence_inherited_parameters_and_rng(self):
        config=C.build_target_config(self.source.config)
        torch.manual_seed(29)
        baseline=BaseRecursiveLM(copy.deepcopy(config)).float()
        expected_rng=torch.get_rng_state().clone()
        torch.manual_seed(29)
        target=RecursiveLlamaForCausalLM(copy.deepcopy(config)).float()
        torch.testing.assert_close(torch.get_rng_state(),expected_rng,rtol=0,atol=0)
        for name,value in baseline.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name],value,rtol=0,atol=0)
        original_parameters=dict(baseline.named_parameters())
        install_loop_lora(baseline.model)
        for name,param in original_parameters.items():
            self.assertIs(dict(baseline.named_parameters())[name],param)
        audit=validate_loop_lora(target.model)
        self.assertEqual(audit['matrix_count'],280)
        for index,layer in enumerate(target.model.layers):
            for name in C.LORA_CONTRACT['targets']:
                module=layer.get_submodule(name)
                if 5 <= index < 15:
                    self.assertIsInstance(module,LoopLoRALinear)
                    self.assertFalse(torch.equal(*module.lora_A))
                    self.assertTrue(all(torch.count_nonzero(b)==0 for b in module.lora_B))
                else:
                    self.assertIs(type(module),torch.nn.Linear)
        self.assertTrue(all(p.requires_grad for p in target.parameters()))
        with self.assertRaises(ValueError): install_loop_lora(target.model)

    def test_real_task_gradients_reach_both_loops_and_shared_base(self):
        self.target.train()
        optimizer=self.optimizer(self.target)
        self.task_loss(self.target).backward()
        for name,param in self.target.named_parameters():
            self.assertIsNotNone(param.grad,name)
            if '.lora_A.' in name:
                self.assertEqual(int(torch.count_nonzero(param.grad)),0,name)
            elif '.lora_B.' in name:
                self.assertGreater(int(torch.count_nonzero(param.grad)),0,name)
        before=self.target.model.layers[5].self_attn.q_proj.weight.detach().clone()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        self.assertFalse(torch.equal(before,self.target.model.layers[5].self_attn.q_proj.weight))
        self.task_loss(self.target).backward()
        for name,param in self.target.named_parameters():
            if '.lora_A.' in name or '.lora_B.' in name:
                self.assertGreater(int(torch.count_nonzero(param.grad)),0,name)
        self.assertTrue(all(module.loop_index is None for module in self.target.modules()
                            if isinstance(module,LoopLoRALinear)))

    def test_trained_nonzero_lora_full_checkpoint_restores_model_moments_and_cursor(self):
        config=T._parse_args([])
        optimizer=self.optimizer(self.target)
        scheduler=T._load_scheduler(optimizer,463,9244,max_lr=8e-4,min_lr=8e-5)
        self.task_loss(self.target).backward(); optimizer.step(); scheduler.step()
        self.target.eval()
        _,expected_logits=V.audit_forward(self.target)
        cursor={str(rank):{'epoch':0,'shard_index':rank,'shard_microbatches_seen':16,
                          'samples_seen':128,'microbatches_seen':16} for rank in range(8)}
        class TokenizerStub:
            def save_pretrained(self,path):
                (Path(path)/'tokenizer.json').write_text('{}',encoding='utf-8')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint-step-000001'
            expected_rng=torch.get_rng_state().clone()
            expected_python=random.getstate()
            T.save_complete_checkpoint(path,model=self.target,tokenizer=TokenizerStub(),
                optimizer=optimizer,scheduler=scheduler,optimizer_step=1,
                cumulative_samples=1024,cumulative_tokens=6000,data_cursors_by_rank=cursor,
                manifest={'test':'synthetic'},config=config,report={},rank=0)
            restored=RecursiveLlamaForCausalLM.from_pretrained(path,local_files_only=True,
                         torch_dtype=torch.float32,attn_implementation='eager')
            V.assert_reload_equal(self.target,restored)
            _,other=V.audit_forward(restored)
            torch.testing.assert_close(other,expected_logits,rtol=0,atol=0)
            opt2=self.optimizer(restored)
            sched2=T._load_scheduler(opt2,463,9244,max_lr=8e-4,min_lr=8e-5)
            torch.manual_seed(99); random.seed(99)
            result=T._restore_training_state(path,optimizer=opt2,scheduler=sched2,config=config,rank=3)
            self.assertEqual(result,(1,1024,6000,cursor['3'],{'test':'synthetic'},'data_cursors_by_rank'))
            torch.testing.assert_close(torch.get_rng_state(),expected_rng,rtol=0,atol=0)
            self.assertEqual(random.getstate(),expected_python)
            self.assertEqual(scheduler.state_dict(),sched2.state_dict())
            original,reloaded=optimizer.state_dict(),opt2.state_dict()
            self.assertEqual(original['param_groups'],reloaded['param_groups'])
            self.assertEqual(set(original['state']),set(reloaded['state']))
            for index,state in original['state'].items():
                for name,value in state.items():
                    torch.testing.assert_close(value,reloaded['state'][index][name],rtol=0,atol=0)
            # A new epoch retains this exact model while starting new moments/schedule.
            epoch2=self.optimizer(restored,max_lr=2e-4)
            sched_epoch2=T._load_scheduler(epoch2,463,9244,max_lr=2e-4,min_lr=2e-5)
            self.assertEqual(epoch2.state_dict()['state'],{})
            self.assertEqual(sched_epoch2.last_epoch,0)
            V.assert_reload_equal(self.target,restored)

    def test_missing_trained_lora_is_rejected_by_actual_runtime_loader(self):
        from safetensors.torch import load_file,save_file
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            self.target.save_pretrained(path,safe_serialization=True)
            (path/'tokenizer.json').write_text('{}',encoding='utf-8')
            state=load_file(str(path/'model.safetensors'))
            del state['model.layers.5.self_attn.q_proj.lora_B.1']
            save_file(state,str(path/'model.safetensors'),metadata={'format':'pt'})
            with self.assertRaisesRegex(ValueError,'Strict recursive model load failed'):
                T._load_runtime_model(path,torch.device('cpu'))

    def test_nonzero_branches_prefill_incremental_cache_match_full_forward(self):
        with torch.no_grad():
            for name,param in self.target.named_parameters():
                if '.lora_B.' in name: param.normal_(std=.005)
            ids=torch.tensor([[3,4,5,6],[7,8,9,10]])
            full=self.target(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
            prefix=self.target(input_ids=ids[:,:3],attention_mask=torch.ones_like(ids[:,:3]),use_cache=True)
            incremental=self.target(input_ids=ids[:,3:],attention_mask=torch.ones_like(ids),
                                    past_key_values=prefix.past_key_values,use_cache=True)
        torch.testing.assert_close(incremental.logits,full[:,3:],rtol=2e-5,atol=1e-6)
        cache=incremental.past_key_values
        self.assertEqual(len(cache),30)
        self.assertEqual([cache.get_seq_length(i) for i in range(30)],[4]*30)

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
        self.assertEqual(count,len(self.target.state_dict())-280)
        audit,logits=V.audit_forward(self.target)
        self.assertEqual(audit['physical_execution_trace'],list(C.SCHEDULE))
        self.assertTrue(V.assert_zero_lora_parity(self.source.state_dict(),self.target,logits))
        with tempfile.TemporaryDirectory() as tmp:
            self.target.save_pretrained(tmp,safe_serialization=True)
            reloaded=RecursiveLlamaForCausalLM.from_pretrained(tmp,local_files_only=True,
                       torch_dtype=torch.float32,attn_implementation='eager')
            V.assert_inherited(self.source.state_dict(),reloaded)
            V.assert_reload_equal(self.target,reloaded)
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
