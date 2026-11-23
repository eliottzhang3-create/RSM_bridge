"""Real CPU autograd, strict loading and restart tests; no production assets/CUDA."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / 'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71437_mesh_ced48_wfuse_lora_r8'
MISSING = [n for n in ('torch','transformers','safetensors','torchaudio','soundfile','numpy') if importlib.util.find_spec(n) is None]


@unittest.skipIf(MISSING, 'Real ML dependencies missing: ' + ', '.join(MISSING))
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(ROUTE), str(ROOT/'code/RSmol')]
        import torch
        import contract as C
        import train as T
        from models import recursive_text as L
        from models import recursive_mesh_wfuse_lora as M
        from models.loop_lora import install_loop_lora, validate_loop_lora, LoopLoRALinear
        from optim import build_optimizer, GradNormTracker, NEW_ROLES
        from transformers import LlamaConfig
        from mizar_text_7_14x3_7_mesh.contract import build_target_config, validate_model_config
        from recursive_model_mizar_text_7_14x3_7_mesh import MizarText7_14x3_7MeshForCausalLM as OriginalLM
        cls.torch, cls.C, cls.T, cls.L, cls.M = torch,C,T,L,M
        cls.config_class, cls.original_class = LlamaConfig, OriginalLM
        cls.build_config, cls.validate_config = staticmethod(build_target_config), staticmethod(validate_model_config)
        cls.install, cls.validate_lora = staticmethod(install_loop_lora), staticmethod(validate_loop_lora)
        cls.build_optimizer, cls.tracker_class = staticmethod(build_optimizer), GradNormTracker
        cls.NEW_ROLES, cls.lora_class = NEW_ROLES, LoopLoRALinear
        from models.ced import CEDSmallWrapper, CEDTemporalMapper, CEDFrequencyMerger
        cls.wrapper_class, cls.mapper_class, cls.mlp_class = CEDSmallWrapper, CEDTemporalMapper, CEDFrequencyMerger
        torch.set_num_threads(1)

    def setUp(self):
        self.torch.manual_seed(12)

    def lm(self, enable=True):
        config=self.config_class(num_hidden_layers=30,hidden_size=16,intermediate_size=32,
            num_attention_heads=4,num_key_value_heads=2,vocab_size=43,tie_word_embeddings=True,
            pad_token_id=17,eos_token_id=0,max_position_embeddings=1024,use_cache=False,
            attention_dropout=0.,_attn_implementation='eager')
        model=self.M.MizarText7_14x3_7MeshForCausalLM(self.build_config(config)).float()
        if enable:
            model.model.enable_audio_fusion()
            self.install(model.model)
        return model

    def inputs(self, length=131):
        torch=self.torch
        return torch.randn(1,length,16),tuple(torch.randn(1,126,16,requires_grad=True) for _ in range(2))

    def fixture(self):
        nn=self.torch.nn
        model=nn.Module()
        model.caption_decoder=nn.Module();model.caption_decoder.lm=self.lm()
        model.audio_encoder=nn.Module();model.audio_encoder.base=nn.Module()
        model.audio_encoder.base.mapper=nn.Linear(16,16)
        model.audio_encoder.base.layer_mlps=nn.ModuleDict({str(i):nn.Linear(16,16) for i in (4,8)})
        core=model.caption_decoder.lm.model
        core.fusion_gradient_state={'complete':False,'seen':{n:False for n in self.NEW_ROLES},'deadline':32}
        return model

    def loss(self, model):
        torch=self.torch
        base=model.audio_encoder.base
        x=base.mapper(torch.randn(1,131,16))
        pair=tuple(base.layer_mlps[str(i)](torch.randn(1,126,16)) for i in (4,8))
        return model.caption_decoder.lm(inputs_embeds=x,audio_information=pair,use_cache=False).logits.square().mean()

    def test_strict_loader_inherits_all_tensors_and_parameter_objects_before_registration(self):
        torch=self.torch
        source=self.lm(enable=False)
        with torch.no_grad():
            for i,r in enumerate(list(source.model.write_routers)+list(source.model.read_routers)):
                r.weight.fill_(.01*(i+1));r.bias.fill_(.001*(i+1))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);source.save_pretrained(path,safe_serialization=True)
            (path/'training_state.pt').write_bytes(b'never deserialize text training state')
            with patch.object(self.L,'recursive_checkpoint_contract',return_value={'source_checkpoint_step':9244}), \
                 patch.object(self.L,'validate_recursive_config',side_effect=lambda c:self.validate_config(c.to_dict(),production=False)), \
                 patch.object(self.L,'validate_parameter_audit',side_effect=lambda a:self.assertEqual(
                     (a['logical_layer_count'],a['physical_layer_count'],a['router_parameter_tensor_count'],a['router_parameter_count']),
                     (56,28,16,816))), \
                 patch.object(torch,'load',side_effect=AssertionError('text pickle read')):
                restored=self.L.load_recursive_text_model(path)
            for k,v in source.state_dict().items():
                self.assertTrue(torch.equal(v,restored.state_dict()[k]),k)
            self.assertIs(restored.model.embed_tokens.weight,restored.lm_head.weight)
            self.assertTrue(restored.audio_initialization_audit['inherited_parameter_objects_preserved'])
            self.assertEqual(restored.audio_initialization_audit['tensor_inheritance']['router_tensors'],16)
            self.assertEqual(restored.audio_initialization_audit['audio_optimizer_start_step'],0)
            # A missing persisted original router can never be hidden by new tensors.
            from safetensors.torch import save_file
            state={k:v.detach().clone() for k,v in source.state_dict().items()}
            del state['model.read_routers.1.bias']
            save_file(state,str(path/'model.safetensors'))
            with self.assertRaises(ValueError): self.L.audit_checkpoint_tensors(source,path)

    def test_zero_gate_and_zero_B_equal_original_logits_and_inherited_gradients(self):
        torch=self.torch
        model=self.lm(enable=False)
        original=self.original_class(model.config).float()
        original.load_state_dict(model.state_dict(),strict=True)
        state=torch.get_rng_state().clone()
        inherited=dict(model.named_parameters())
        model.model.enable_audio_fusion();self.install(model.model)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(all(dict(model.named_parameters())[k] is v for k,v in inherited.items()))
        model.eval();original.eval()
        x,pair=self.inputs()
        y=model(inputs_embeds=x,audio_information=pair,use_cache=False).logits
        z=original(inputs_embeds=x,use_cache=False).logits
        torch.testing.assert_close(y,z,rtol=0,atol=0)
        y.square().sum().backward()
        grads={n:p.grad.clone() for n,p in model.named_parameters() if n in inherited}
        z.square().sum().backward()
        for n,p in original.named_parameters(): torch.testing.assert_close(grads[n],p.grad,rtol=0,atol=0)
        self.assertTrue(all(v.grad is not None and not v.grad.ne(0).any() for v in pair))

    def test_all_588_matrices_independent_and_first_then_opened_gradient_audits(self):
        torch=self.torch
        model=self.fixture();core=model.caption_decoder.lm.model
        lora=self.validate_lora(core)
        self.assertEqual(lora['matrix_count'],588)
        matrices=[p for n,p in core.named_parameters() if '.lora_A.' in n or '.lora_B.' in n]
        self.assertEqual(len({p.untyped_storage().data_ptr() for p in matrices}),588)
        optimizer,_,_=self.build_optimizer(model,34626)
        self.assertEqual(len(optimizer.state),0)
        self.loss(model).backward()
        first=self.T.first_gradient_audit(model)
        self.assertTrue(first['passed'],first)
        self.assertTrue(all(first['closed_encoder_weights'].values()) and all(first['closed_lora_B'].values()))
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        self.loss(model).backward()
        second=self.T.first_gradient_audit(model)
        self.assertTrue(second['passed'],second)
        self.assertTrue(all(second['nonzero_by_role'].values()),second)

    def test_actual_56_calls_683_memory_both_loop_gradients_and_answer_shift(self):
        torch=self.torch
        model=self.lm();core=model.model
        core.audit_mode=core.gradient_audit_mode=True
        x,pair=self.inputs(683)
        out=model(inputs_embeds=x,audio_information=pair,use_cache=False)
        wrapper=NS(caption_decoder=NS(lm=model))
        audit=self.T.recursive_forward_audit(wrapper,1,require_memory=True)
        self.assertEqual(audit['memory_shape'],[1,6,683,16])
        self.assertEqual(core.last_fusion_trace,((7,4),(21,8)))
        self.assertEqual(core.last_loop_lora_trace,tuple((l,p,(l-7)//14) for l,p in enumerate(self.C.TEXT_MODEL_SCHEDULE) if 7<=p<21))
        targets=torch.randint(1,16,(1,300));targets[:,20:]=17;targets[:,19]=0
        inputs={'answer':{'input_ids':targets,'attention_mask':targets.ne(17).long()}}
        loss=self.T.answer_loss(out,inputs,17)
        expected=torch.nn.functional.cross_entropy(out.logits[:,382:402].reshape(-1,43),targets[:,:20].reshape(-1))
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertTrue(self.T.recursive_backward_audit(wrapper,1)['passed'])
        self.assertTrue(self.T.router_gradient_audit(wrapper)['passed'])
        self.assertEqual(core.last_core_input_refs,[])

    def test_nonzero_fusion_changes_write_values_but_preserves_original_queries(self):
        torch=self.torch
        model=self.lm().eval();core=model.model
        core.audit_mode=True
        with torch.no_grad():
            for layer in (4,8): core.audio_fusions[str(layer)].weight[:,16:].copy_(torch.eye(16)*(.2 if layer==4 else -.3))
        x,pair=self.inputs()
        values=[]
        write=core._write
        def recording_write(memory,value,weights):
            values.append(value.detach().clone());return write(memory,value,weights)
        with patch.object(core,'_write',side_effect=recording_write),torch.no_grad():
            model(inputs_embeds=x,audio_information=pair,use_cache=False)
        boundaries=[core.last_prefix_output,*core.last_core_outputs]
        for i,layer in enumerate((4,8)):
            expected=core.audio_fusions[str(layer)](torch.cat((boundaries[i][:,:126],pair[i].detach()),dim=-1))
            torch.testing.assert_close(values[i][:,:126],expected)
            torch.testing.assert_close(values[i][:,126:],boundaries[i][:,126:],rtol=0,atol=0)
        torch.testing.assert_close(values[2],boundaries[2],rtol=0,atol=0)
        torch.testing.assert_close(values[3],boundaries[3],rtol=0,atol=0)
        torch.testing.assert_close(core.last_router_queries['write_pre'],boundaries[0],rtol=0,atol=0)
        for i in (0,1,2):
            torch.testing.assert_close(core.last_router_queries[f'write_{i}'],core.last_core_inputs[i],rtol=0,atol=0)
            torch.testing.assert_close(core.last_router_queries[f'read_{i}'],core.last_core_inputs[i],rtol=0,atol=0)
        self.assertFalse(hasattr(core,'audio_gates') or hasattr(core,'listen_routers'))

    def test_identity_initialization_independence_and_encoder_half_gradients(self):
        torch=self.torch;model=self.lm();core=model.model
        for layer in (4,8):
            f=core.audio_fusions[str(layer)]
            self.assertIsNone(f.bias)
            torch.testing.assert_close(f.weight[:,:16],torch.eye(16),rtol=0,atol=0)
            self.assertFalse(f.weight[:,16:].ne(0).any())
        self.assertIsNot(core.audio_fusions['4'].weight,core.audio_fusions['8'].weight)
        x,pair=self.inputs()
        model(inputs_embeds=x,audio_information=pair,use_cache=False).logits.square().sum().backward()
        for layer,info in zip((4,8),pair):
            self.assertTrue(core.audio_fusions[str(layer)].weight.grad[:,16:].ne(0).any())
            self.assertIsNotNone(info.grad)
            self.assertFalse(info.grad.ne(0).any())
        with self.assertRaises(RuntimeError): core.enable_audio_fusion()

    def test_rejects_cache_missing_audio_and_gradient_checkpoint_replay(self):
        torch=self.torch
        model=self.lm();x,pair=self.inputs()
        for bad in (None,pair[:1],(pair[0],pair[1][:,:125]),(pair[0].double(),pair[1])):
            with self.assertRaises(ValueError): model(inputs_embeds=x,audio_information=bad,use_cache=False)
        with self.assertRaises(ValueError): model(inputs_embeds=x,audio_information=pair,use_cache=True)
        model.model.gradient_checkpointing=True
        with self.assertRaises(RuntimeError): model(inputs_embeds=x,audio_information=pair,use_cache=False)

    def test_full_state_save_resume_matches_uninterrupted_next_update(self):
        torch=self.torch
        model=self.fixture()
        opt,sched,_=self.build_optimizer(model,34626);tracker=self.tracker_class()
        scaler=torch.cuda.amp.GradScaler(enabled=False)
        def update(candidate,optimizer,scheduler,clip):
            optimizer.zero_grad(set_to_none=True)
            self.loss(candidate).backward();clip.track_and_clip_(candidate.named_parameters())
            optimizer.step();scheduler.step()
        update(model,opt,sched,tracker)
        # Save the first-update partial unlock state, including both LoRA B roles.
        model.caption_decoder.lm.model.fusion_gradient_state['seen'].update({n:n.endswith('_fusion') or n.endswith('_B') for n in self.NEW_ROLES})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'step1.ckpt'
            recipe={'geometry':self.C.geometry(1477492)}
            cfg={'mode':'formal','contract':recipe,'contract_sha256':self.C.digest(recipe)}
            rng={'python':__import__('random').getstate(),'numpy':__import__('numpy').random.get_state(),
                 'torch':torch.get_rng_state(),'cuda':torch.get_rng_state()}
            def gather(states,state): states[:]=[copy.deepcopy(state) for _ in range(8)]
            with patch.object(self.T,'rng_state',return_value=rng),patch.object(self.T.dist,'all_gather_object',side_effect=gather), \
                 patch.object(self.T.dist,'get_rank',return_value=0),patch.object(self.T.dist,'barrier'):
                self.T.save_checkpoint(path,model,opt,sched,scaler,tracker,cfg,1,0,1,'cpu')
            self.T.reload_audit(path,model,cfg,1)
            saved=torch.load(path,map_location='cpu',weights_only=False)
            restored=self.fixture()
            other_opt,other_sched,_=self.build_optimizer(restored,34626)
            self.assertFalse(restored.caption_decoder.lm.model.audio_fusions['4'].weight[:,16:].ne(0).any())
            self.T.validate_resume_model_state(saved,restored,other_opt)
            restored.load_state_dict(saved['state_dict'],strict=True)
            self.assertTrue(restored.caption_decoder.lm.model.audio_fusions['4'].weight[:,16:].ne(0).any())
            other_clip=self.tracker_class();other_clip.load_state_dict(saved['grad_norm_tracker'])
            other_opt.load_state_dict(saved['optimizer']);other_sched.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['grad_scaler'])
            torch.set_rng_state(saved['random_state_by_rank'][0]['torch']);update(model,opt,sched,tracker)
            torch.set_rng_state(saved['random_state_by_rank'][0]['torch']);update(restored,other_opt,other_sched,other_clip)
            for name,value in model.state_dict().items(): torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)
            self.assertEqual(tracker.state_dict(),other_clip.state_dict())
            self.assertEqual(sched.state_dict(),other_sched.state_dict())


    def test_resume_rejects_inventory_reordering_tensor_dtype_and_nonfinite_moments(self):
        torch=self.torch;model=self.fixture();opt,sched,_=self.build_optimizer(model,34626)
        self.loss(model).backward();opt.step();sched.step()
        saved=dict(total_step=1,state_dict=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
            optimizer_parameter_inventory=self.T.optimizer_parameter_inventory(model,opt))
        self.T.validate_resume_model_state(saved,model,opt)
        changed=copy.deepcopy(saved);changed['optimizer_parameter_inventory'][0][0]['name']='wrong'
        with self.assertRaises(ValueError): self.T.validate_resume_model_state(changed,model,opt)
        key=next(k for k,v in saved['state_dict'].items() if v.is_floating_point())
        changed=copy.deepcopy(saved);changed['state_dict'][key]=changed['state_dict'][key].double()
        with self.assertRaises(ValueError): self.T.validate_resume_model_state(changed,model,opt)
        changed=copy.deepcopy(saved);next(iter(changed['optimizer']['state'].values()))['exp_avg'].fill_(float('nan'))
        with self.assertRaises(ValueError): self.T.validate_resume_model_state(changed,model,opt)

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


if __name__=='__main__': unittest.main()
