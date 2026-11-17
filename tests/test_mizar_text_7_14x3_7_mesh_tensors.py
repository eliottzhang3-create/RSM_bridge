"""Real CPU tensor tests; skips explicitly if this code-only machine lacks ML deps."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/RSmol'))
from mizar_text_7_14x3_7_mesh import contract as C, conversion as V
from mizar_text_7_14x3_7_mesh import data as D

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    import recursive_model_mizar_text_7_14x3_7_mesh as M
    spec = importlib.util.spec_from_file_location('isolated_71437_trainer', ROOT/'code/RSmol/scripts/train_stage4_mizar_text_7_14x3_7_mesh_ddp.py')
    T = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = T
    spec.loader.exec_module(T)
    AVAILABLE, MISSING = True, ''
except ImportError as error:
    AVAILABLE, MISSING = False, str(error)


@unittest.skipUnless(AVAILABLE, 'Real tensor dependencies unavailable: '+MISSING)
class TensorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)
        config = LlamaConfig(num_hidden_layers=30, hidden_size=16, intermediate_size=32,
                             num_attention_heads=4, num_key_value_heads=2, vocab_size=43,
                             tie_word_embeddings=True, max_position_embeddings=128,
                             pad_token_id=0, bos_token_id=1, eos_token_id=2,
                             _attn_implementation='eager')
        self.source = LlamaForCausalLM(config).float().eval()
        self.model = M.RecursiveLlamaForCausalLM(C.build_target_config(config)).float().eval()
        V.inherited_state(self.source.state_dict(), self.model)
        self.ids = torch.tensor([[3,4,5,6,7,8]])

    def test_official_truncated_normal_exact_seed_and_meta_safe(self):
        router = torch.nn.Linear(576,6)
        torch.manual_seed(101); M._init_router(router)
        std = math.sqrt(2/(6*576))
        expected = torch.empty_like(router.weight)
        torch.manual_seed(101)
        torch.nn.init.trunc_normal_(expected, mean=0, std=std, a=-3*std, b=3*std)
        self.assertTrue(torch.equal(router.weight, expected))
        self.assertEqual(int(torch.count_nonzero(router.bias)), 0)
        self.assertLessEqual(float(router.weight.abs().max()), 3*std)
        M._init_router(torch.nn.Linear(576,6,device='meta'))

    def test_source_namespace_tied_values_and_strict_errors(self):
        state = {C.LM_PREFIX+k:v.clone() for k,v in self.source.state_dict().items()}
        state['audio_encoder.fake'] = torch.ones(2)
        text = V.extract_lm_state(state)
        V.strict_load(self.source, text)
        state[C.LM_PREFIX+'lm_head.weight'].add_(1)
        with self.assertRaisesRegex(ValueError, 'tied'):
            V.extract_lm_state(state)
        with self.assertRaises(ValueError):
            V.extract_lm_state({'state_dict':state})
        with self.assertRaises(ValueError):
            V.strict_load(self.source, {**text, 'extra':torch.ones(1)})
        bad = dict(text); del bad['model.layers.8.self_attn.q_proj.weight']
        with self.assertRaisesRegex(ValueError, 'Missing inherited'):
            V.inherited_state(bad, self.model)

    def test_all_retained_weights_and_independent_physical_layers(self):
        V.assert_inherited(self.source.state_dict(), self.model)
        actual = self.model.state_dict()
        expected_count = len(self.source.state_dict()) - 2*len(self.source.model.layers[0].state_dict()) + 16
        self.assertEqual(len(actual), expected_count)
        qweights = [layer.self_attn.q_proj.weight for layer in self.model.model.layers]
        self.assertEqual(len({p.data_ptr() for p in qweights}),28)
        self.assertIs(self.model.model.embed_tokens.weight,self.model.lm_head.weight)
        self.assertEqual({k for k in actual if '_routers.' in k}, C.ROUTER_KEYS)

    def test_memory_queries_trace_and_all_three_loop_gradients(self):
        self.model.model.audit_mode = True
        self.model.model.gradient_audit_mode = True
        out = self.model(self.ids, use_cache=False)
        owner = self.model.model
        self.assertEqual([x['physical_index'] for x in owner.last_forward_trace], list(C.SCHEDULE))
        self.assertEqual([x['logical_index'] for x in owner.last_forward_trace], list(range(56)))
        self.assertEqual(owner.last_forward_trace[49], {'logical_index':49, 'physical_index':21})
        self.assertEqual(len(owner.last_core_inputs),3)
        self.assertEqual(len(owner.last_core_outputs),3)
        self.assertEqual(len(owner.last_core_input_refs),3)
        self.assertEqual(len(owner.last_core_output_refs),3)
        self.assertEqual(len(owner.last_memory_write_history),4)
        self.assertEqual(len(owner.last_router_weights),8)
        self.assertEqual(set(owner.last_router_queries),
                         {'write_pre','read_pre'} | {f'{kind}_{i}' for kind in ('write','read') for i in range(3)})
        self.assertEqual(tuple(owner.last_initial_memory.shape),(1,6,6,16))
        torch.testing.assert_close(owner.last_initial_memory[:,0], owner.embed_tokens(self.ids))
        self.assertEqual(int(torch.count_nonzero(owner.last_initial_memory[:,1:])),0)
        torch.testing.assert_close(owner.last_router_queries['write_pre'], owner.last_prefix_output)
        torch.testing.assert_close(owner.last_router_queries['read_pre'], owner.last_prefix_output)
        for loop in range(3):
            torch.testing.assert_close(owner.last_router_queries[f'write_{loop}'], owner.last_core_inputs[loop])
            torch.testing.assert_close(owner.last_router_queries[f'read_{loop}'], owner.last_core_inputs[loop])
        out.logits.square().mean().backward()
        for value in owner.last_core_output_refs + owner.last_core_input_refs:
            self.assertIsNotNone(value.grad)
            self.assertGreater(float(value.grad.abs().sum()),0)
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad,name)
            self.assertTrue(torch.isfinite(parameter.grad).all(),name)
            if name in C.ROUTER_KEYS:
                self.assertGreater(float(parameter.grad.abs().sum()),0,name)
        self.assertFalse(any('memory' in key for key in self.model.state_dict()))

    def test_strict_save_reload_preserves_every_tensor_and_logits(self):
        before = {k:v.clone() for k,v in self.model.state_dict().items()}
        _, logits = V.audit_forward(self.model)
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save_pretrained(tmp,safe_serialization=True)
            reloaded, info = M.RecursiveLlamaForCausalLM.from_pretrained(tmp, local_files_only=True,
                    torch_dtype=torch.float32, attn_implementation='eager', output_loading_info=True)
            T._validate_loading_info(info)
            C.validate_model_config(reloaded.config.to_dict(),production=False)
            for key,value in reloaded.state_dict().items():
                self.assertTrue(torch.equal(value,before[key]),key)
            V.assert_inherited(self.source.state_dict(),reloaded)
            _, after = V.audit_forward(reloaded)
            torch.testing.assert_close(logits,after,rtol=0,atol=0)

    def test_cache_56_slots_matches_full_context_and_generation(self):
        with torch.no_grad():
            full = self.model(self.ids,use_cache=False).logits
            first = self.model(self.ids[:,:3],use_cache=True)
            cache = first.past_key_values
            for pos in range(3,6):
                result = self.model(self.ids[:,pos:pos+1],past_key_values=cache,
                                    attention_mask=torch.ones(1,pos+1,dtype=torch.long),use_cache=True)
                torch.testing.assert_close(result.logits[:,-1],full[:,pos],rtol=1e-5,atol=1e-6)
            self.assertEqual(len(cache),56)
            self.assertTrue(all(cache.get_seq_length(i)==6 for i in range(56)))
            a = self.model.generate(self.ids,max_new_tokens=3,do_sample=False,use_cache=True)
            b = self.model.generate(self.ids,max_new_tokens=3,do_sample=False,use_cache=False)
            self.assertTrue(torch.equal(a,b))

    def test_ddp_and_accumulation_gradient_equals_global_token_mean(self):
        weight = torch.tensor(0.2,requires_grad=True)
        sums, counts = [], []
        for size in (2,3,1,4):
            logits = torch.stack([weight, -weight, weight*2]).repeat(size,1)
            labels = torch.arange(size)%3
            sums.append(torch.nn.functional.cross_entropy(logits,labels,reduction='sum'))
            counts.append(size)
        scale = T.token_weighted_gradient_scale(world_size=2,global_window_tokens=sum(counts),gradient_accumulation_steps=2)
        actual = torch.autograd.grad(sum(sums)*scale/2,weight,retain_graph=True)[0]
        expected = torch.autograd.grad(sum(sums)/sum(counts),weight)[0]
        torch.testing.assert_close(actual,expected)

    def test_padding_masks_only_padding_and_keeps_real_pad_id_supervised(self):
        batch = D.collate([[3,0,5],[7]],pad_token_id=0)
        self.assertEqual(batch['attention_mask'].tolist(),[[1,1,1],[1,0,0]])
        self.assertEqual(batch['labels'].tolist(),[[3,0,5],[7,-100,-100]])
        self.assertEqual(int((batch['labels'][:,1:]!=-100).sum()),2)

    def test_optimizer_groups_rng_and_complete_checkpoint_roundtrip(self):
        config = T.Stage4Config(gate='A',world_size=1,micro_batch_size=2,gradient_accumulation_steps=2,
                              max_optimizer_steps=4,scheduler_total_steps=4,warmup_steps=1)
        optimizer,audit = T._optimizer(self.model,config)
        self.assertTrue(audit['router_parameters_in_optimizer'])
        optimized = [p for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(optimized),len({id(p) for p in optimized}))
        for group in optimizer.param_groups:
            for router in list(self.model.model.write_routers)+list(self.model.model.read_routers):
                if any(p is router.bias for p in group['params']): self.assertEqual(group['weight_decay'],0)
                if any(p is router.weight for p in group['params']): self.assertEqual(group['weight_decay'],0.1)
        self.model(self.ids,use_cache=False).logits.sum().backward(); optimizer.step()
        rng = T._capture_rng_state(torch.device('cpu'))
        expected = torch.rand(3)
        T._restore_rng_state(rng,torch.device('cpu'))
        self.assertTrue(torch.equal(torch.rand(3),expected))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({f't{i}':i for i in range(43)},unk_token='t0')),pad_token='t0',bos_token='t1',eos_token='t2')
        state = {'manifest':[], 'data_cursors_by_rank':{'0':{'synthetic':True}}, 'rng_states_by_rank':{'0':rng}}
        with tempfile.TemporaryDirectory() as tmp:
            config.output_dir = Path(tmp)/'run'
            for step in range(1,5):
                saved = T._checkpoint(self.model,tokenizer,optimizer,step,config,0,state)
            self.assertEqual(len(list(config.output_dir.glob('checkpoint-*'))),3)
            # The fixture reduces hidden size only; keep all identity fields.
            with patch.object(T,'validate_model_config',side_effect=lambda raw:C.validate_model_config(raw,production=False)):
                loaded = T._load_checkpoint_state(saved)
            T._validate_resume_contract(loaded,config)
            restored,_ = T._optimizer(self.model,config); restored.load_state_dict(loaded['optimizer'])
            self.assertEqual(len(restored.state),len(optimizer.state))
            with self.assertRaises(FileExistsError):
                T._checkpoint(self.model,tokenizer,optimizer,4,config,0,state)
            (saved/'checkpoint_complete.json').unlink()
            with self.assertRaises(ValueError): T._validate_checkpoint_complete(saved)


if __name__ == '__main__':
    unittest.main()
