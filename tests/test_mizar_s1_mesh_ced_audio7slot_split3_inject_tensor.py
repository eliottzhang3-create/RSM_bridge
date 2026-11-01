"""Real CPU tensors in mellow_c8204d8; synthetic CED, no assets/downloads/CUDA."""
import copy
import math
import importlib.util
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT/"code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_audio7slot_split3_inject"
HAS_TENSORS = all(importlib.util.find_spec(n) is not None for n in ("torch","torchaudio","transformers","soundfile"))


@unittest.skipUnless(HAS_TENSORS, "requires torch/torchaudio/transformers/soundfile in mellow_c8204d8")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(ROUTE), str(ROOT/"code/RSmol")]
        import torch
        import contract
        import train
        from transformers import LlamaConfig
        from models.ced import CEDSmallWrapper, CEDTemporalMapper, CEDFrequencyMerger
        from models.decoder import DecoderModel
        from models.mellow import Mellow, AudioEncoder
        from models.recursive_mesh_listen import RecursiveLlamaForCausalLM
        from recursive_model_5_10x2_5_mesh import RecursiveLlamaForCausalLM as OriginalLM
        from optim import build_optimizer, GradNormTracker
        cls.torch, cls.contract, cls.train = torch, contract, train
        cls.config_class, cls.lm_class, cls.original_class = LlamaConfig, RecursiveLlamaForCausalLM, OriginalLM
        cls.wrapper_class, cls.mapper_class, cls.mlp_class = CEDSmallWrapper, CEDTemporalMapper, CEDFrequencyMerger
        cls.decoder_class, cls.mellow_class, cls.encoder_class = DecoderModel, Mellow, AudioEncoder
        cls.build_optimizer, cls.tracker_class = staticmethod(build_optimizer), GradNormTracker
        torch.set_num_threads(1)

    def lm(self):
        config = self.config_class(vocab_size=32, hidden_size=32, intermediate_size=48,
            num_hidden_layers=30, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=1024, pad_token_id=17, eos_token_id=0,
            use_cache=False, attention_dropout=0.)
        config.recursive_layer_count, config.recursive_loops = 20, 2
        return self.lm_class(config).float()

    def toy_wrapper(self):
        torch, nn = self.torch, self.torch.nn
        class Block(nn.Module):
            def __init__(self,index):
                super().__init__()
                self.offset = nn.Parameter(torch.tensor(float(index)),requires_grad=False)
            def forward(self,x): return x+self.offset
        class CED(nn.Module):
            maximal_allowed_length = 1012
            def __init__(self):
                super().__init__()
                self.init_bn=nn.BatchNorm2d(64)
                self.blocks=nn.Sequential(*(Block(i) for i in range(1,13)))
                self.norm=nn.LayerNorm(384)
            def forward_features(self,x):
                self.calls=getattr(self,"calls",0)+1
                self.base=torch.arange(x.shape[0]*252*384,dtype=x.dtype).reshape(x.shape[0],252,384)/1000
                return self.norm(self.blocks(self.base))
        wrapper=self.wrapper_class.__new__(self.wrapper_class)
        nn.Module.__init__(wrapper)
        wrapper.ced=CED().eval()
        for p in wrapper.ced.parameters(): p.requires_grad=False
        wrapper.mapper=self.mapper_class(384,4,576,output_tokens=126,resample=False)
        wrapper.layer_mlps=nn.ModuleDict({str(k):self.mlp_class(384,4,576,dropout=0.) for k in (4,8)})
        wrapper.mapper_type="freq_merge_126"
        wrapper._waveform_to_mel=lambda wave: torch.zeros(wave.shape[0],64,2001)
        return wrapper

    def fixture(self):
        torch,nn=self.torch,self.torch.nn
        torch.manual_seed(1234)
        decoder=self.decoder_class.__new__(self.decoder_class)
        nn.Module.__init__(decoder)
        decoder.lm,decoder.lm_embedding_size=self.lm(),32
        decoder.lm.model.enable_audio_listen()
        model=nn.Module()
        model.audio_encoder,model.caption_decoder=nn.Module(),decoder
        model.audio_encoder.base=nn.Module()
        model.audio_encoder.base.mapper=self.mapper_class(384,4,32,output_tokens=126,resample=False)
        model.audio_encoder.base.layer_mlps=nn.ModuleDict({str(k):self.mlp_class(384,4,32,dropout=0.) for k in (4,8)})
        targets=torch.randint(1,16,(1,300))
        targets[:,199],targets[:,200:]=0,17
        inputs=dict(grids={k:torch.randn(1,126,4,384) for k in (4,8,12)},
                    prompt={"input_ids":torch.ones(1,256,dtype=torch.long)},
                    answer={"input_ids":targets,"attention_mask":targets.ne(17).long()})
        return model.train(),inputs

    def loss(self,model,inputs):
        base=model.audio_encoder.base
        audio=base.mapper(inputs["grids"][12])
        information=tuple(base.layer_mlps[str(k)](inputs["grids"][k]) for k in (4,8))
        out=model.caption_decoder(audio,None,inputs["prompt"],inputs["answer"],audio_information=information)
        self.assertEqual(tuple(out.logits.shape),(1,683,32))
        self.assertIsNone(out.past_key_values)
        return self.train.answer_loss(out,{"answer":inputs["answer"]},17)

    def test_single_ced_forward_capture_block4_8_before_final_norm_and_hook_cleanup(self):
        torch=self.torch
        wrapper=self.toy_wrapper()
        final,intermediate=wrapper._forward_hidden(torch.zeros(2,64,2001))
        base=wrapper.ced.base
        self.assertEqual(wrapper.ced.calls,1)
        torch.testing.assert_close(intermediate[4],(base+10).reshape(2,504,384))
        torch.testing.assert_close(intermediate[8],(base+36).reshape(2,504,384))
        torch.testing.assert_close(final,wrapper.ced.norm(base+78).reshape(2,504,384))
        self.assertTrue(all(not b._forward_hooks for b in wrapper.ced.blocks))
        with patch.object(wrapper.ced.blocks[5],"forward",side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError,"fixture failure"):
                wrapper._forward_hidden(torch.zeros(2,64,2001))
        self.assertTrue(all(not b._forward_hooks for b in wrapper.ced.blocks))

    def test_wrapper_mlps_train_outside_no_grad_and_encoder_stays_frozen_eval(self):
        torch=self.torch
        wrapper=self.toy_wrapper().train()
        out=wrapper(torch.zeros(2,640000))
        self.assertEqual(wrapper.ced.calls,1)
        for name in ("embedding","layer4_audio","layer8_audio"):
            self.assertEqual(tuple(out[name].shape),(2,126,576))
            self.assertTrue(out[name].requires_grad)
        sum(v.square().mean() for k,v in out.items() if k!="clipwise_output").backward()
        self.assertTrue(all(p.grad is None for p in wrapper.ced.parameters()))
        self.assertTrue(all(p.grad is not None for p in wrapper.layer_mlps.parameters()))
        self.assertFalse(wrapper.ced.training)
        self.assertTrue(wrapper.mapper.training and wrapper.layer_mlps.training)

    def test_inherited_tensors_rng_and_audio_router_split3_rows_independent(self):
        torch=self.torch
        from models.recursive_text import validate_audio_listen
        torch.manual_seed(3)
        model=self.lm()
        inherited={k:v.clone() for k,v in model.state_dict().items()}
        state=torch.get_rng_state().clone()
        model.model.enable_audio_listen()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for name,value in inherited.items():torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
        audit=validate_audio_listen(model,fresh=True)
        self.assertEqual(audit['total_routers'],12)
        with self.assertRaises(RuntimeError):model.model.enable_audio_listen()
        for kind in ('write','read'):
            for source,target in zip(getattr(model.model,f'{kind}_routers'),getattr(model.model,f'audio_listen_{kind}_routers')):
                for i in (0,5,6):
                    torch.testing.assert_close(target.weight[i],source.weight[0],rtol=0,atol=0)
                    torch.testing.assert_close(target.bias[i],source.bias[0]-math.log(3.),rtol=0,atol=0)
                row0=target.weight[0].detach().clone()
                with torch.no_grad():target.weight[6,0].add_(1.)
                torch.testing.assert_close(target.weight[0],row0,rtol=0,atol=0)
                torch.testing.assert_close(source.weight[0],row0,rtol=0,atol=0)
                self.assertNotEqual(source.weight[0,0].item(),target.weight[6,0].item())

    def test_all_six_actual_router_softmaxes_split_slot0_and_preserve_remaining_slots(self):
        torch=self.torch
        model=self.lm().eval();core=model.model;core.enable_audio_listen()
        query=torch.randn(2,126,32)
        for kind in ('write','read'):
            for source,target in zip(getattr(core,f'{kind}_routers'),getattr(core,f'audio_listen_{kind}_routers')):
                with torch.no_grad():
                    old=torch.softmax(source(query),dim=-1);new=torch.softmax(target(query),dim=-1)
                torch.testing.assert_close(new[...,1:5],old[...,1:5],rtol=2e-6,atol=1e-7)
                for i in (0,5,6):torch.testing.assert_close(new[...,i],old[...,0]/3,rtol=2e-6,atol=1e-7)
                torch.testing.assert_close(new[...,[0,5,6]].sum(-1),old[...,0],rtol=2e-6,atol=1e-7)

    def test_first_backward_all_new_rows_and_mlps_have_finite_nonzero_paths(self):
        model,inputs=self.fixture()
        optimizer,_,groups=self.build_optimizer(model,17313)
        self.loss(model,inputs).backward()
        audit=self.train.first_gradient_audit(model)
        self.assertTrue(audit['passed'],audit)
        self.assertTrue(all(audit['nonzero_by_role'].values()),audit)
        for role,value in audit['audio_branches'].items():
            self.assertTrue(value['present'] and value['finite'] and value['nonzero'],(role,value))
            if role.startswith('audio_'):
                for label in ('weight','bias'):
                    self.assertTrue(value[f'new_{label}_rows']['nonzero'],(role,value))
        for group in groups:
            self.assertEqual(group['base_lr'],3e-4 if group['role']=='default' else 6e-4)
            for name in group['names']:
                parameter=dict(model.named_parameters())[name]
                expected=0. if parameter.ndim<=1 or name.endswith('.bias') else .01
                self.assertEqual(group['weight_decay'],expected)
        new_weight=model.caption_decoder.lm.model.audio_listen_write_routers[0].weight
        before=new_weight[5:].detach().clone()
        optimizer.step()
        self.assertFalse(self.torch.equal(before,new_weight[5:]))

    def test_rejects_missing_wrong_information_and_kv_cache(self):
        torch=self.torch
        model=self.lm()
        model.model.enable_audio_listen()
        x=torch.randn(1,130,32)
        pair=(torch.randn(1,126,32),torch.randn(1,126,32))
        for kwargs in ({},{"audio_information":pair[:1]},
                       {"audio_information":(pair[0],pair[1][:,:125])},
                       {"audio_information":(pair[0].double(),pair[1])}):
            with self.subTest(kwargs=list(kwargs)),self.assertRaises(ValueError):
                model(inputs_embeds=x,use_cache=False,**kwargs)
        with self.assertRaises(ValueError): model(inputs_embeds=x,use_cache=True,audio_information=pair)

    def test_diagnostics_leave_logits_and_rng_unchanged(self):
        torch=self.torch
        model=self.lm().eval();core=model.model;core.enable_audio_listen()
        x=torch.randn(1,131,32);pair=tuple(torch.randn(1,126,32) for _ in range(2))
        with torch.no_grad():baseline=model(inputs_embeds=x,use_cache=False,audio_information=pair).logits
        before=torch.get_rng_state().clone();core.collect_listen_stats=True
        with torch.no_grad():actual=model(inputs_embeds=x,use_cache=False,audio_information=pair).logits
        torch.testing.assert_close(actual,baseline,rtol=0,atol=0)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        records=core.listen_diagnostics()
        routers=[item for item in records if item['name'].startswith('audio_')]
        self.assertEqual(len(routers),6)
        for item in routers:
            self.assertAlmostEqual(sum(item[f'slot{i}_mean'] for i in range(7)),1.,places=6)
            self.assertAlmostEqual(item['slot5_mean']+item['slot6_mean'],item['new_combined_mean'],places=7)
            self.assertAlmostEqual(item['slot0_mean']+item['new_combined_mean'],item['slot0_group_mean'],places=6)
            self.assertEqual(item['split_group_spread_max'],0.)
            self.assertNotIn('old5_logsumexp_mean',item)
        self.assertEqual(core._diagnostic_audio_seeds,())

    def test_causality_and_original_recovery_after_test_only_unsplitting(self):
        torch=self.torch
        model=self.lm().eval();original=self.original_class(model.config).eval()
        original.load_state_dict(model.state_dict(),strict=True)
        core=model.model;core.enable_audio_listen()
        x=torch.randn(1,132,32);pair=tuple(torch.randn(1,126,32) for _ in range(2))
        with torch.no_grad():
            baseline=model(inputs_embeds=x,use_cache=False,audio_information=pair).logits
            later=x.clone();later[:,130:]+=torch.randn_like(later[:,130:])
            changed=model(inputs_embeds=later,use_cache=False,audio_information=pair).logits
            torch.testing.assert_close(changed[:,:130],baseline[:,:130],rtol=1e-6,atol=1e-7)
            # Test-only unsplitting: restore slot0 bias as well as suppressing
            # rows5/6. Neither operation occurs in formal split3 training.
            for kind in ('write','read'):
                for source,router in zip(getattr(core,f'{kind}_routers'),getattr(core,f'audio_listen_{kind}_routers')):
                    router.bias[0].copy_(source.bias[0])
                    router.bias[5:].fill_(-1000.)
            recovered=model(inputs_embeds=x,use_cache=False,audio_information=pair).logits
            expected=original(inputs_embeds=x,use_cache=False).logits
        torch.testing.assert_close(recovered,expected,rtol=2e-5,atol=2e-6)

    def test_zero_real_sample_gradients_are_reported_without_unlock_requirement(self):
        torch=self.torch
        model,inputs=self.fixture();self.loss(model,inputs).backward()
        for p in model.parameters():
            if p.grad is not None:p.grad.zero_()
        audit=self.train.first_gradient_audit(model)
        self.assertTrue(audit['passed'],audit)
        self.assertFalse(any(audit['nonzero_by_role'].values()))
        with tempfile.TemporaryDirectory() as directory:
            state={'checks':0,'last_checked_update':0}
            with patch.object(self.train,'collective_require',side_effect=lambda ok,msg,device:self.assertTrue(ok,msg)):
                branches=self.train.audit_listen_update(model,state,1,Path(directory),0,'cpu')
            self.assertEqual(state,{'checks':1,'last_checked_update':1})
            self.assertTrue(all(item['present'] and item['finite'] for item in branches.values()))

    def test_actual_initial_memories_queries_writes_reads_and_transience(self):
        torch=self.torch
        model=self.lm().eval();core=model.model;core.enable_audio_listen();core.audit_mode=True
        x=torch.randn(1,131,32)
        pair=tuple(torch.randn(1,126,32) for _ in range(2))
        with torch.no_grad():first=model(inputs_embeds=x,use_cache=False,audio_information=pair).logits
        audio,text=core.last_initial_memory
        torch.testing.assert_close(audio[:,0],x[:,:126],rtol=0,atol=0)
        torch.testing.assert_close(audio[:,5],pair[0],rtol=0,atol=0)
        torch.testing.assert_close(audio[:,6],pair[1],rtol=0,atol=0)
        self.assertFalse(bool(audio[:,1:5].ne(0).any()))
        self.assertFalse(bool(text[:,1:].ne(0).any()))
        for stage,suffix in enumerate(('pre','0','1')):
            query=core.last_prefix_output if stage==0 else core.last_core_inputs[stage-1]
            value=core.last_prefix_output if stage==0 else core.last_core_outputs[stage-1]
            for branch,stem,region in ((0,'audio_',slice(None,126)),(1,'',slice(126,None))):
                torch.testing.assert_close(core.last_router_queries[f'{stem}write_{suffix}'],query[:,region],rtol=0,atol=0)
                torch.testing.assert_close(core.last_router_queries[f'{stem}read_{suffix}'],query[:,region],rtol=0,atol=0)
                weights=core.last_router_weights[f'{stem}write_{suffix}']
                old=core.last_initial_memory[branch] if stage==0 else core.last_memory_write_history[stage-1][branch]
                expected=old+value[:,region].unsqueeze(1)*weights.transpose(1,2).unsqueeze(-1)
                torch.testing.assert_close(core.last_memory_write_history[stage][branch],expected,rtol=0,atol=0)
        for stage in range(2):
            suffix='pre' if stage==0 else '0'
            values=[]
            for branch,stem in ((0,'audio_'),(1,'')):
                weights=core.last_router_weights[f'{stem}read_{suffix}']
                values.append(core._read(core.last_memory_write_history[stage][branch],weights))
            torch.testing.assert_close(core.last_core_inputs[stage],torch.cat(values,dim=1),rtol=0,atol=0)
        self.assertEqual(core.last_memory_shape,((1,7,126,32),(1,5,5,32)))
        self.assertEqual([e['physical_index'] for e in core.last_forward_trace],list(self.contract.TEXT_MODEL_SCHEDULE))
        with torch.no_grad():second=model(inputs_embeds=x,use_cache=False,audio_information=pair).logits
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertFalse(any('memory' in key for key in model.state_dict()))

    def test_strict_mesh_loader_inherits_every_tensor_before_audio_copy(self):
        from models import recursive_text
        config=self.config_class(vocab_size=64,hidden_size=576,intermediate_size=32,
                                num_hidden_layers=30,num_attention_heads=9,num_key_value_heads=3,
                                max_position_embeddings=1024,pad_token_id=17,eos_token_id=0,use_cache=False)
        config.recursive_layer_count,config.recursive_loops=20,2
        config.mesh_memory_slots,config.mesh_router_count=5,6
        config.mesh_architecture_contract=self.contract.TEXT_MODEL_CONTRACT
        config.model_architecture_contract=self.contract.TEXT_MODEL_ARCHITECTURE
        config.logical_to_physical=list(self.contract.TEXT_MODEL_SCHEDULE)
        original=self.original_class(config)
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            loaded=recursive_text.load_recursive_text_model(directory)
        for name,value in original.state_dict().items():
            self.torch.testing.assert_close(loaded.state_dict()[name],value,rtol=0,atol=0)
        from models.recursive_text import validate_audio_listen
        self.assertEqual(validate_audio_listen(loaded,fresh=True)['new_router_parameter_count'],24234)

    def test_full_checkpoint_resume_matches_uninterrupted_next_update(self):
        torch=self.torch
        model,inputs=self.fixture()
        optimizer,scheduler,_=self.build_optimizer(model,17313)
        tracker=self.tracker_class()
        def update(m,o,s,t):
            o.zero_grad(set_to_none=True)
            loss=self.loss(m,inputs)
            loss.backward()
            t.track_and_clip_(m.named_parameters())
            o.step();s.step()
        update(model,optimizer,scheduler,tracker)
        model.caption_decoder.lm.model.audio_gradient_state = {'checks':1,'last_checked_update':1}
        # Exercise the real full-checkpoint writer/reloader with eight synthetic rank RNG records.
        rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"model--step-1.ckpt"
            recipe={"route":self.contract.ROUTE,"geometry":self.contract.geometry(1477492),"audio_listen":self.contract.LISTEN_CONTRACT}
            cfg={"mode":"formal","contract":recipe,"contract_sha256":self.contract.digest(recipe)}
            class Scaler:
                def state_dict(self): return {"synthetic_cpu":True}
            def gather(output,value): output[:]=[copy.deepcopy(value) for _ in range(8)]
            with patch.object(self.train,"rng_state",return_value={"torch":rng,"python":None,"numpy":None,"cuda":None}), \
                 patch.object(self.train.dist,"all_gather_object",side_effect=gather), \
                 patch.object(self.train.dist,"get_rank",return_value=0), \
                 patch.object(self.train.dist,"barrier"):
                self.train.save_checkpoint(path,model,optimizer,scheduler,Scaler(),tracker,cfg,1,0,1,"cpu")
            self.assertEqual(self.train.reload_audit(path,model,cfg,1)["status"],"PASS")
            saved=torch.load(path,weights_only=False)
            # Construct the fresh architecture first, then strict-load learned
            # rows/biases; initialization must never run again after this load.
            restored,_=self.fixture()
            restored.load_state_dict(saved["state_dict"],strict=True)
            restored.caption_decoder.lm.model.audio_gradient_state=saved['audio_gradient_state']
            with self.assertRaises(RuntimeError):restored.caption_decoder.lm.model.enable_audio_listen()
            opt2,sched2,_=self.build_optimizer(restored,17313)
            opt2.load_state_dict(saved["optimizer"]);sched2.load_state_dict(saved["scheduler"])
            track2=self.tracker_class();track2.load_state_dict(saved["grad_norm_tracker"])
            torch.set_rng_state(rng);update(model,optimizer,scheduler,tracker)
            torch.set_rng_state(saved["random_state_by_rank"][0]["torch"]);update(restored,opt2,sched2,track2)
            for name,p in model.state_dict().items():
                torch.testing.assert_close(p,restored.state_dict()[name],rtol=0,atol=0)
            self.assertEqual(tracker.state_dict(),track2.state_dict())
            self.assertEqual(scheduler.state_dict(),sched2.state_dict())

    def test_zero_gate_full_logits_match_split3_and_registration_preserves_rng(self):
        torch=self.torch
        spec=importlib.util.spec_from_file_location("split3_parent_gate_test",ROUTE.with_name(
            ROUTE.name.removesuffix("_inject"))/"models/recursive_mesh_listen.py")
        parent_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent_module)
        model=self.lm().eval()
        parent=parent_module.RecursiveLlamaForCausalLM(copy.deepcopy(model.config)).float().eval()
        parent.load_state_dict(model.state_dict(),strict=True)
        inherited={k:v.clone() for k,v in model.state_dict().items()}
        before=torch.get_rng_state().clone()
        model.model.enable_audio_listen()
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        parent.model.enable_audio_listen()
        for name,value in inherited.items():
            torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters())-
                         sum(p.numel() for p in parent.parameters()),2*(32+1))
        for gate in model.model.audio_gates.values():
            for parameter in gate.parameters():self.assertFalse(bool(parameter.ne(0).any()))
        x=torch.randn(1,131,32);pair=tuple(torch.randn(1,126,32) for _ in range(2))
        with torch.no_grad():
            actual=model(inputs_embeds=x,audio_information=pair,use_cache=False).logits
            expected=parent(inputs_embeds=x,audio_information=pair,use_cache=False).logits
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_nonzero_gate_hooks_confirm_post_read_input_and_audio_only_residual(self):
        torch=self.torch
        model=self.lm().eval();core=model.model;core.enable_audio_listen();core.audit_mode=True
        seen={};handles=[]
        for layer,gate in core.audio_gates.items():
            with torch.no_grad():
                gate.weight.copy_(torch.linspace(-.03,.06,32).reshape(1,32))
                gate.bias.fill_(.13 if layer=='4' else -.15)
            def capture(module,args,key=layer):seen[key]=args[0].detach().clone()
            handles.append(gate.register_forward_pre_hook(capture))
        x=torch.randn(1,132,32);pair=tuple(torch.randn(1,126,32) for _ in range(2))
        try:
            with torch.no_grad():model(inputs_embeds=x,audio_information=pair,use_cache=False)
        finally:
            for handle in handles:handle.remove()
        for stage,layer in enumerate(('4','8')):
            record=core.last_injection_audit[stage]
            h=record['read_hidden']
            torch.testing.assert_close(seen[layer],h[:,:126],rtol=0,atol=0)
            expected_gate=torch.tanh(core.audio_gates[layer](h[:,:126]))
            torch.testing.assert_close(record['gate'],expected_gate,rtol=0,atol=0)
            torch.testing.assert_close(core.last_core_inputs[stage][:,:126],
                h[:,:126]+expected_gate*pair[stage],rtol=0,atol=0)
            torch.testing.assert_close(core.last_core_inputs[stage][:,126:],h[:,126:],rtol=0,atol=0)
            suffix='pre' if stage==0 else '0'
            self.assertFalse(torch.equal(seen[layer],core.last_router_queries[f'audio_read_{suffix}']))
            torch.testing.assert_close(core.last_router_queries[f'audio_read_{stage}'],
                core.last_core_inputs[stage][:,:126],rtol=0,atol=0)
        self.assertEqual(core.last_injection_trace,((5,4),(15,8)))
        with torch.no_grad():
            baseline=model(inputs_embeds=x,audio_information=pair,use_cache=False).logits
            changed=x.clone();changed[:,130:]+=torch.randn_like(changed[:,130:])
            later=model(inputs_embeds=changed,audio_information=pair,use_cache=False).logits
        torch.testing.assert_close(later[:,:130],baseline[:,:130],rtol=1e-6,atol=1e-7)

    def test_zero_gate_blocks_only_direct_seed_gradient_not_existing_memory_path(self):
        torch=self.torch
        model=self.lm();core=model.model;core.enable_audio_listen()
        for layer,index in ((4,5),(8,15)):
            h=torch.randn(1,131,32,requires_grad=True)
            information=torch.randn(1,126,32,requires_grad=True)
            out=core.inject_audio(h,information,layer,index)
            (out*torch.randn_like(out)).sum().backward()
            self.assertTrue(torch.equal(information.grad,torch.zeros_like(information)))
            gate=core.audio_gates[str(layer)]
            for parameter in gate.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertTrue(bool(parameter.grad.ne(0).any()))
        model.zero_grad(set_to_none=True)
        x=torch.randn(1,131,32)
        pair=tuple(torch.randn(1,126,32,requires_grad=True) for _ in range(2))
        model(inputs_embeds=x,audio_information=pair,use_cache=False).logits.square().mean().backward()
        for information in pair:
            self.assertIsNotNone(information.grad)
            self.assertTrue(bool(torch.isfinite(information.grad).all()))
            self.assertTrue(bool(information.grad.ne(0).any()))

    def test_new_parameter_count_and_constructor_rng_mapper_parity(self):
        torch,nn=self.torch,self.torch.nn
        cfg=dict(self.contract.CED_EXPECTED_CONFIG,model_type="ced",
                 id2label={str(i):str(i) for i in range(527)},auto_map={
                 "AutoConfig":"configuration_ced.CedConfig",
                 "AutoModelForAudioClassification":"modeling_ced.CedForAudioClassification"})
        encoder=self.toy_wrapper().ced
        encoder.config=SimpleNamespace(**cfg,to_dict=lambda:cfg)
        loaded=nn.Module();loaded.encoder=encoder
        spec=importlib.util.spec_from_file_location("mother_ced",ROUTE.with_name(
            "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small")/"models/ced.py")
        parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in self.contract.CED_ASSET_FILES: (root/name).write_bytes(b"synthetic")
            self.contract.write_json(root/"config.json",cfg)
            with patch("models.ced.local_ced_config",return_value=nullcontext(encoder.config)), \
                 patch.object(parent,"local_ced_config",return_value=nullcontext(encoder.config)), \
                 patch("models.ced.AutoModelForAudioClassification.from_pretrained",return_value=(loaded,{})):
                torch.manual_seed(12);old=parent.CEDSmallWrapper({"pretrained_audioencoder_path":directory});old_rng=torch.get_rng_state()
                torch.manual_seed(12);new=self.wrapper_class({"pretrained_audioencoder_path":directory});new_rng=torch.get_rng_state()
            self.assertTrue(torch.equal(old_rng,new_rng))
            for name,value in old.mapper.state_dict().items():
                torch.testing.assert_close(value,new.mapper.state_dict()[name],rtol=0,atol=0)
            self.assertEqual(sum(p.numel() for p in new.layer_mlps.parameters()),2443776)
            self.assertEqual(2443776+6*7*(576+1)+2*(576+1),2469164)


if __name__ == "__main__":
    unittest.main()
