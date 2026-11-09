"""Optional real CPU conversion, gradient, cache and full-state resume checks."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from text_x3_test_support import BASE, ROUTES, SCHEDULE, SOURCE_MAPPING, load_file

HAS_TENSORS = all(importlib.util.find_spec(name) is not None for name in ('torch','transformers','safetensors','tokenizers'))


@unittest.skipUnless(HAS_TENSORS, 'requires torch, transformers, safetensors and tokenizers; run remotely with --require-tensors')
class RealTensorChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        cls.torch, cls.Config, cls.Source = torch, LlamaConfig, LlamaForCausalLM
        torch.set_num_threads(2)
        sys.path.insert(0,str(BASE))
        cls.mods = {route: (load_file('x3_converter_'+route, BASE/f'scripts/convert_stepwise_{route}.py'),
                            load_file('x3_trainer_'+route, BASE/f'scripts/train_stage4_{route}_ddp.py')) for route in ROUTES}
        engine=Tokenizer(WordLevel({'[PAD]':0,'[BOS]':1,'[EOS]':2,'[UNK]':3,**{f't{i}':i for i in range(4,64)}},unk_token='[UNK]'))
        engine.pre_tokenizer=Whitespace()
        cls.tokenizer=PreTrainedTokenizerFast(tokenizer_object=engine,pad_token='[PAD]',bos_token='[BOS]',eos_token='[EOS]',unk_token='[UNK]')

    def source(self):
        self.torch.manual_seed(123)
        config=self.Config(num_hidden_layers=30,hidden_size=16,intermediate_size=24,num_attention_heads=4,num_key_value_heads=2,vocab_size=64,max_position_embeddings=128,tie_word_embeddings=True,pad_token_id=0,attention_dropout=0.0)
        config._attn_implementation='eager'
        return self.Source(config)

    def model(self, route):
        converter,_=self.mods[route]
        source=self.source()
        target=converter.RecursiveLlamaForCausalLM(converter._target_config(source.config,{'kind':'original_smolLM2_30_layer'},Path('/synthetic')))
        mapping={source:target for target,source in enumerate(SOURCE_MAPPING)}
        inherited={converter._remap_key(k,'original_smolLM2_30_layer',mapping):v for k,v in source.state_dict().items()}
        inherited={k:v for k,v in inherited.items() if k is not None}
        target.load_state_dict(inherited,strict=False)
        target.tie_weights()
        return target

    def test_full_cpu_converters_both_routes_save_reload_and_reject_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'
            original=self.source(); original.save_pretrained(source,safe_serialization=True)
            self.tokenizer.save_pretrained(source)
            targets=[]
            for route in ROUTES:
                converter,trainer=self.mods[route]
                args=types.SimpleNamespace(source_checkpoint=source,output_dir=root/route,seed=0,threads=2)
                metadata=converter.convert(args)
                for key in ('inherited_weights_exact','save_reload_tensors_exact','save_reload_logits_exact','tokenizer_roundtrip_exact'):
                    self.assertTrue(metadata[key],key)
                self.assertEqual(metadata['forward_trace'],SCHEDULE)
                trainer._validate_fresh_model(args.output_dir)
                restored,info=converter.RecursiveLlamaForCausalLM.from_pretrained(args.output_dir,local_files_only=True,output_loading_info=True)
                trainer._validate_loading_info(info)
                trainer._validate_model_audit(converter.parameter_audit(restored))
                self.assertEqual(restored.lm_head.weight.data_ptr(),restored.model.embed_tokens.weight.data_ptr())
                for i,j in enumerate(SOURCE_MAPPING):
                    for name,value in restored.model.layers[i].state_dict().items():
                        self.assertTrue(self.torch.equal(value,original.model.layers[j].state_dict()[name]))
                targets.append(restored)
                with self.assertRaises(FileExistsError): converter.convert(args)
            for name,value in targets[0].state_dict().items():
                self.assertTrue(self.torch.equal(value,targets[1].state_dict()[name]),name)

    def test_clean_twenty_layer_source_is_not_subsampled_twice(self):
        old=load_file('x3_old_pure_source',BASE/'recursive_model_5_10_5.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); src=root/'clean20'; original=self.source()
            config=copy.deepcopy(original.config)
            config.recursive_layer_count=20; config.recursive_loops=2
            config.logical_to_physical=[*range(15),*range(5,15),*range(15,20)]
            oldmodel=old.RecursiveLlamaForCausalLM(config)
            converter,_=self.mods[ROUTES[0]]
            mapping={src:dst for dst,src in enumerate(SOURCE_MAPPING)}
            state={converter._remap_key(k,'original_smolLM2_30_layer',mapping):v for k,v in original.state_dict().items()}
            oldmodel.load_state_dict({k:v for k,v in state.items() if k is not None},strict=True)
            oldmodel.save_pretrained(src,safe_serialization=True); self.tokenizer.save_pretrained(src)
            (src/'recursive_conversion_metadata.json').write_text(json.dumps({'conversion':'SmolLM2-5-10-5','status':'ok'}))
            for route in ROUTES:
                c,_=self.mods[route]
                report=c.convert(types.SimpleNamespace(source_checkpoint=src,output_dir=root/route,seed=0,threads=2))
                self.assertEqual(report['source_kind'],'conversion_only_5_10_5')

    def test_real_cpu_training_control_flow_both_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'
            self.source().save_pretrained(source,safe_serialization=True)
            self.tokenizer.save_pretrained(source)
            for route in ROUTES:
                converter,trainer=self.mods[route]
                model_dir=root/(route+'_model')
                converter.convert(types.SimpleNamespace(source_checkpoint=source,output_dir=model_dir,seed=0,threads=2))
                config=trainer.Stage4Config(gate='A',model_path=model_dir,output_dir=root/(route+'_run'),world_size=1,micro_batch_size=2,gradient_accumulation_steps=2,context_length=8,max_optimizer_steps=2,scheduler_total_steps=10,warmup_steps=1)
                report=trainer.run_training(config)
                self.assertEqual(report['status'],'PASS',report.get('hard_failures'))
                self.assertEqual(report['optimizer_steps'],2)
                self.assertEqual(report['precision_policy']['parameter_dtype'],'float32')
                self.assertTrue((Path(report['final_checkpoint'])/'checkpoint_complete.json').is_file())
                self.assertEqual(report['router_module_count'],8 if 'mesh' in route else 0)

    def test_pure_middle_gradients_sum_three_uses(self):
        torch=self.torch
        shared=self.model(ROUTES[0]).train()
        unrolled=copy.deepcopy(shared)
        unrolled.model.layers=torch.nn.ModuleList([copy.deepcopy(shared.model.layers[i]) for i in SCHEDULE])
        unrolled.model.logical_to_physical=tuple(range(40))
        ids=torch.tensor([[3,5,7,11]])
        for model in (shared,unrolled): model(input_ids=ids,labels=ids,use_cache=False).loss.backward()
        for physical,layer in enumerate(shared.model.layers):
            logical=[i for i,index in enumerate(SCHEDULE) if index==physical]
            self.assertEqual(len(logical),3 if 5<=physical<15 else 1)
            for name,parameter in layer.named_parameters():
                expected=sum(dict(unrolled.model.layers[i].named_parameters())[name].grad for i in logical)
                torch.testing.assert_close(parameter.grad,expected,rtol=5e-4,atol=1e-6)
        self.assertFalse(any('router' in name for name,_ in shared.named_parameters()))

    def test_mesh_six_slots_four_queries_writes_and_all_gradients(self):
        torch=self.torch; route=ROUTES[1]
        model=self.model(route).train(); core=model.model
        core.audit_mode=True; core.gradient_audit_mode=True
        ids=torch.tensor([[3,4,5,6],[7,8,9,10]])
        model(input_ids=ids,labels=ids,use_cache=False).loss.backward()
        self.assertEqual(core.last_memory_shape,(2,6,4,16))
        self.assertEqual([x['physical_index'] for x in core.last_forward_trace],SCHEDULE)
        torch.testing.assert_close(core.last_initial_memory[:,0],core.embed_tokens(ids))
        self.assertEqual(torch.count_nonzero(core.last_initial_memory[:,1:]).item(),0)
        for kind in ('write','read'):
            torch.testing.assert_close(core.last_router_queries[kind+'_pre'],core.last_prefix_output)
            for i in range(3): torch.testing.assert_close(core.last_router_queries[f'{kind}_{i}'],core.last_core_inputs[i])
        memory=core.last_initial_memory
        for name,value,written in zip(('pre','0','1','2'),[core.last_prefix_output,*core.last_core_outputs],core.last_memory_write_history):
            expected=core._write(memory,value,core.last_router_weights['write_'+name])
            torch.testing.assert_close(written,expected)
            memory=written
        for ref in core.last_core_input_refs+core.last_core_output_refs:
            self.assertIsNotNone(ref.grad); self.assertTrue(torch.isfinite(ref.grad).all())
        routers=[*core.write_routers,*core.read_routers]
        self.assertEqual(len({id(r) for r in routers}),8)
        for router in routers:
            for p in router.parameters():
                self.assertIsNotNone(p.grad); self.assertTrue(torch.isfinite(p.grad).all()); self.assertGreater(p.grad.abs().sum().item(),0)
        self.assertEqual(sum(p.numel() for r in routers for p in r.parameters()),8*(16*6+6))

    def test_forty_cache_slots_and_cached_recompute(self):
        torch=self.torch; ids=torch.tensor([[3,4,5,6]])
        for route in ROUTES:
            model=self.model(route).eval()
            with torch.no_grad():
                prefix=model(input_ids=ids[:,:-1],use_cache=True)
                cache=prefix.past_key_values
                self.assertEqual(len(cache),40)
                self.assertEqual([cache.get_seq_length(i) for i in range(40)],[3]*40)
                cached=model(input_ids=ids[:,-1:],attention_mask=torch.ones_like(ids),past_key_values=cache,use_cache=True)
                full=model(input_ids=ids,use_cache=False)
                torch.testing.assert_close(cached.logits[:,-1],full.logits[:,-1],rtol=3e-4,atol=1e-6)

    def test_full_checkpoint_next_update_matches_uninterrupted(self):
        torch=self.torch
        for route in ROUTES:
            _,trainer=self.mods[route]
            with tempfile.TemporaryDirectory() as tmp:
                model=self.model(route).train()
                config=trainer.Stage4Config(gate='D',output_dir=Path(tmp),world_size=1,max_optimizer_steps=100,scheduler_total_steps=100,warmup_steps=1)
                optimizer,_=trainer._optimizer(model,config)
                ids=torch.tensor([[3,4,5,6]])
                def update(m,opt):
                    opt.zero_grad(set_to_none=True); m(input_ids=ids,labels=ids,use_cache=False).loss.backward(); opt.step()
                update(model,optimizer)
                rng=trainer._capture_rng_state(torch.device('cpu'))
                checkpoint=trainer._checkpoint(model,self.tokenizer,optimizer,10,config,0,{'manifest':[],'data_cursors_by_rank':{'0':{'row_offset':40}},'rng_states_by_rank':{'0':rng}})
                state=trainer._load_checkpoint_state(checkpoint); trainer._validate_resume_contract(state,config)
                restored=trainer.RecursiveLlamaForCausalLM.from_pretrained(checkpoint,local_files_only=True,torch_dtype=torch.float32,attn_implementation=model.config._attn_implementation).train()
                resumed_optimizer,_=trainer._optimizer(restored,config); resumed_optimizer.load_state_dict(state['optimizer'])
                trainer._restore_rng_state(state['rng_states_by_rank']['0'],torch.device('cpu'))
                self.assertTrue(torch.equal(torch.get_rng_state(),rng['torch_cpu']))
                update(model,optimizer); update(restored,resumed_optimizer)
                for key,value in model.state_dict().items(): torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
                self.assertEqual(state['data_cursors_by_rank']['0']['row_offset'],40)
                with self.assertRaises(FileExistsError): trainer._checkpoint(model,self.tokenizer,optimizer,10,config,0,{})
                wrong=self.mods[ROUTES[1] if route==ROUTES[0] else ROUTES[0]][1]
                with self.assertRaises(ValueError): wrong._validate_checkpoint_complete(checkpoint)

    def test_global_effective_token_gradient_matches_reference(self):
        torch=self.torch; torch.manual_seed(8)
        shared=torch.nn.Linear(3,7,bias=False)
        reference=copy.deepcopy(shared)
        micros=[(torch.randn(3,3),torch.tensor([1,2,3])),(torch.randn(5,3),torch.tensor([2,3,4,5,6])),(torch.randn(1,3),torch.tensor([0])),(torch.randn(4,3),torch.tensor([0,1,2,3]))]
        for x,y in micros: (torch.nn.functional.cross_entropy(shared(x).float(),y,reduction='sum')/2/2).backward()
        scale=self.mods[ROUTES[0]][1].token_weighted_gradient_scale(world_size=2,gradient_accumulation_steps=2,global_window_tokens=13)
        shared.weight.grad.mul_(scale)
        torch.nn.functional.cross_entropy(reference(torch.cat([x for x,_ in micros])),torch.cat([y for _,y in micros]),reduction='mean').backward()
        torch.testing.assert_close(shared.weight.grad,reference.weight.grad,rtol=1e-5,atol=1e-7)

    def test_wrong_config_routes_rejected_and_meta_router_init_safe(self):
        torch=self.torch
        pure=self.model(ROUTES[0]); mesh=self.model(ROUTES[1])
        for route,wrong in ((ROUTES[0],mesh.config),(ROUTES[1],pure.config)):
            with self.assertRaises(ValueError): self.mods[route][0].RecursiveLlamaForCausalLM(wrong)
        module=sys.modules['recursive_model_5_10x3_5_mesh_6slot']
        module._init_router(torch.nn.Linear(16,6,device='meta'))


if __name__ == '__main__': unittest.main()
