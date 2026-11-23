"""Real CPU LM/autograd/restart checks without audio dependencies or production assets.

The checkpoint tests execute production save/restore functions with CPU RNG and
explicit collective fixtures; they do not claim eight-rank or CUDA validation.
"""
import ast
import copy
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT/'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71437_mesh_ced48_wfuse_lora_r8'
MISSING = [n for n in ('torch', 'transformers', 'safetensors', 'numpy') if importlib.util.find_spec(n) is None]


@unittest.skipIf(MISSING, 'Real LM dependencies missing: '+', '.join(MISSING))
class CoreTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The repository's top-level code package can shadow Python's code.py
        # when pdb is imported by torch.distributed. Load the stdlib module first.
        import sysconfig
        code_spec = importlib.util.spec_from_file_location('code', Path(sysconfig.get_path('stdlib'))/'code.py')
        stdlib_code = importlib.util.module_from_spec(code_spec)
        code_spec.loader.exec_module(stdlib_code)
        sys.modules['code'] = stdlib_code
        sys.path[:0] = [str(ROUTE), str(ROOT/'code/RSmol')]
        import torch
        import numpy as np
        import contract as C
        import audit_reference as A
        from models import recursive_mesh_wfuse_lora as M, recursive_text as L
        from models.loop_lora import install_loop_lora, validate_loop_lora, LoopLoRALinear
        from optim import build_optimizer, GradNormTracker, parameter_role, TRAINABLE_ROLES, NEW_ROLES
        from transformers import LlamaConfig
        from mizar_text_7_14x3_7_mesh.contract import build_target_config, validate_model_config
        from recursive_model_mizar_text_7_14x3_7_mesh import MizarText7_14x3_7MeshForCausalLM
        cls.torch, cls.np, cls.C, cls.M, cls.L, cls.A = torch, np, C, M, L, A
        cls.config_class, cls.original_class = LlamaConfig, MizarText7_14x3_7MeshForCausalLM
        cls.build_config, cls.validate_config = staticmethod(build_target_config), staticmethod(validate_model_config)
        cls.install, cls.validate_lora, cls.lora_class = staticmethod(install_loop_lora), staticmethod(validate_loop_lora), LoopLoRALinear
        cls.optimizer, cls.tracker_class = staticmethod(build_optimizer), GradNormTracker
        cls.new_roles = NEW_ROLES
        scope = dict(torch=torch, F=torch.nn.functional, np=np, Path=Path, os=__import__('os'),
                     WORLD_SIZE=8, BATCH_SIZE=4, ACCUMULATION_STEPS=4, RANK_UPDATE_SAMPLES=16, EPOCHS=3,
                     PREFIX_TOKENS=383, ANSWER_TOKENS=300, ROUTE=C.ROUTE, TEXT_MODEL_SCHEDULE=C.TEXT_MODEL_SCHEDULE,
                     NEW_ROLES=NEW_ROLES, TRAINABLE_ROLES=TRAINABLE_ROLES, parameter_role=parameter_role,
                     validate_resume_payload=C.validate_resume_payload)
        functions = ('answer_loss', 'first_gradient_audit', 'recursive_forward_audit', 'recursive_backward_audit',
                     'router_gradient_audit', 'optimizer_parameter_inventory', 'validate_resume_model_state', 'save_checkpoint')
        A.execute([copy.deepcopy(A.find_function(A.tree(ROUTE/'train.py'), n)) for n in functions], scope)
        cls.scope = scope
        torch.set_num_threads(1)

    def setUp(self):
        self.torch.manual_seed(123)

    def lm(self, branches=True):
        cfg = self.config_class(num_hidden_layers=30, hidden_size=16, intermediate_size=32,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=43, tie_word_embeddings=True,
            pad_token_id=17, eos_token_id=0, max_position_embeddings=1024, use_cache=False,
            attention_dropout=0., _attn_implementation='eager')
        result = self.M.MizarText7_14x3_7MeshForCausalLM(self.build_config(cfg)).float()
        if branches:
            result.model.enable_audio_fusion()
            self.install(result.model)
        return result

    def inputs(self, length=131):
        return self.torch.randn(1,length,16), tuple(self.torch.randn(1,126,16,requires_grad=True) for _ in (4,8))

    def fixture(self):
        nn = self.torch.nn
        model = nn.Module()
        model.caption_decoder = nn.Module(); model.caption_decoder.lm = self.lm()
        model.audio_encoder = nn.Module(); model.audio_encoder.base = nn.Module()
        model.audio_encoder.base.mapper = nn.Linear(16,16)
        model.audio_encoder.base.layer_mlps = nn.ModuleDict({str(i):nn.Linear(16,16) for i in (4,8)})
        model.caption_decoder.lm.model.fusion_gradient_state = dict(complete=False, seen={n:False for n in self.new_roles}, deadline=32)
        return model

    def loss(self, model):
        base = model.audio_encoder.base
        x = base.mapper(self.torch.randn(1,131,16))
        pair = tuple(base.layer_mlps[str(i)](self.torch.randn(1,126,16)) for i in (4,8))
        return model.caption_decoder.lm(inputs_embeds=x,audio_information=pair,use_cache=False).logits.square().mean()

    def test_strict_trained_router_inheritance_registration_rng_and_missing_tensor_rejection(self):
        torch = self.torch
        source = self.lm(False)
        with torch.no_grad():
            for i,r in enumerate(list(source.model.write_routers)+list(source.model.read_routers)):
                r.weight.fill_(.005*(i+1)); r.bias.fill_(.001*(i+1))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); source.save_pretrained(path,safe_serialization=True)
            (path/'training_state.pt').write_bytes(b'never deserialize text state')
            with patch.object(self.L,'recursive_checkpoint_contract',return_value={'source_checkpoint_step':9244}), \
                 patch.object(self.L,'validate_recursive_config',side_effect=lambda c:self.validate_config(c.to_dict(),production=False)), \
                 patch.object(self.L,'validate_parameter_audit',side_effect=lambda a:self.assertEqual(
                     (a['logical_layer_count'],a['router_parameter_tensor_count'],a['router_parameter_count']),(56,16,816))), \
                 patch.object(torch,'load',side_effect=AssertionError('text training pickle read')):
                restored = self.L.load_recursive_text_model(path)
            for n,t in source.state_dict().items(): self.assertTrue(torch.equal(t,restored.state_dict()[n]),n)
            self.assertTrue(restored.audio_initialization_audit['inherited_parameter_objects_preserved'])
            self.assertEqual(restored.audio_initialization_audit['tensor_inheritance']['router_tensors'],16)
            self.assertIs(restored.model.embed_tokens.weight,restored.lm_head.weight)
            from safetensors.torch import save_file
            state={n:t.detach().clone() for n,t in source.state_dict().items()}
            del state['model.read_routers.3.bias']; save_file(state,str(path/'model.safetensors'))
            with self.assertRaises(ValueError): self.L.audit_checkpoint_tensors(source,path)
        model=self.lm(False); rng=torch.get_rng_state().clone(); inherited=dict(model.named_parameters())
        model.model.enable_audio_fusion(); self.install(model.model)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(dict(model.named_parameters())[n] is p for n,p in inherited.items()))

    def test_identity_fusion_and_zero_lora_match_original_three_loop_logits_and_gradients(self):
        torch=self.torch; model=self.lm(False)
        original=self.original_class(model.config).float(); original.load_state_dict(model.state_dict(),strict=True)
        inherited=set(dict(model.named_parameters()))
        model.model.enable_audio_fusion(); self.install(model.model)
        model.eval(); original.eval(); x,pair=self.inputs()
        y=model(inputs_embeds=x,audio_information=pair,use_cache=False).logits
        z=original(inputs_embeds=x,use_cache=False).logits
        torch.testing.assert_close(y,z,rtol=1e-6,atol=1e-7)
        y.square().mean().backward(); z.square().mean().backward()
        current=dict(model.named_parameters())
        for n,p in original.named_parameters():
            self.assertIn(n,inherited); torch.testing.assert_close(current[n].grad,p.grad,rtol=1e-5,atol=1e-7)
        self.assertTrue(all(p.grad is not None and not p.grad.ne(0).any() for p in pair))

    def test_56_calls_four_writes_683_positions_three_loop_gradients_and_all_routers(self):
        torch=self.torch; model=self.lm(); core=model.model; core.audit_mode=core.gradient_audit_mode=True
        x,pair=self.inputs(683); out=model(inputs_embeds=x,audio_information=pair,use_cache=False)
        wrapper=NS(caption_decoder=NS(lm=model))
        self.assertEqual(self.scope['recursive_forward_audit'](wrapper,1,True)['memory_shape'],[1,6,683,16])
        self.assertEqual(core.last_fusion_trace,((7,4),(21,8)))
        self.assertEqual(len(core.last_memory_write_history),4)
        self.assertEqual(core.last_loop_lora_trace,tuple((i,p,(i-7)//14) for i,p in enumerate(self.C.TEXT_MODEL_SCHEDULE) if 7<=p<21))
        target=torch.randint(1,16,(1,300)); target[:,19]=0; target[:,20:]=17
        loss=self.scope['answer_loss'](out,{'answer':{'input_ids':target,'attention_mask':target.ne(17)}},17)
        expected=torch.nn.functional.cross_entropy(out.logits[:,382:402].reshape(-1,43),target[:,:20].reshape(-1))
        torch.testing.assert_close(loss,expected); loss.backward()
        self.assertTrue(self.scope['recursive_backward_audit'](wrapper,1)['passed'])
        self.assertTrue(self.scope['router_gradient_audit'](wrapper)['passed'])
        self.assertEqual(core.last_core_input_refs,[])

    def test_588_independent_matrices_third_round_selection_and_gradient_unlock(self):
        torch=self.torch; model=self.fixture(); core=model.caption_decoder.lm.model
        self.assertEqual(self.validate_lora(core)['matrix_count'],588)
        matrices=[p for n,p in core.named_parameters() if '.lora_A.' in n or '.lora_B.' in n]
        self.assertEqual(len({p.untyped_storage().data_ptr() for p in matrices}),588)
        opt,_,groups=self.optimizer(model,34626)
        self.assertEqual(len({g['role'] for g in groups}),12)
        self.loss(model).backward(); first=self.scope['first_gradient_audit'](model)
        self.assertTrue(first['passed'],first)
        opt.step(); opt.zero_grad(set_to_none=True); self.loss(model).backward()
        opened=self.scope['first_gradient_audit'](model); self.assertTrue(opened['passed'],opened)
        self.assertTrue(all(opened['nonzero_by_role'].values()),opened)
        module=core.layers[7].self_attn.q_proj
        from models.loop_lora import call_with_loop_lora
        x=torch.randn(1,4,16); outputs=[]
        for loop in (0,1,2):
            outputs.append(call_with_loop_lora(core.layers[7],loop,lambda layer,value:layer.self_attn.q_proj(value),x))
        self.assertFalse(torch.equal(outputs[0],outputs[2]))
        self.assertFalse(torch.equal(outputs[1],outputs[2]))
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            call_with_loop_lora(core.layers[7],2,lambda *a:(_ for _ in ()).throw(RuntimeError('synthetic')))
        self.assertIsNone(module.loop_index)

    def test_nonzero_fusion_only_two_values_and_no_third_loop_injection(self):
        torch=self.torch; model=self.lm().eval(); core=model.model; core.audit_mode=True
        with torch.no_grad():
            for layer in (4,8): core.audio_fusions[str(layer)].weight[:,16:].copy_(torch.eye(16)*(.1*layer))
        values=[]; original_write=core._write
        def write(memory,value,weights):
            values.append(value.detach().clone()); return original_write(memory,value,weights)
        x,pair=self.inputs()
        with patch.object(core,'_write',side_effect=write),torch.no_grad(): model(inputs_embeds=x,audio_information=pair,use_cache=False)
        boundaries=[core.last_prefix_output,*core.last_core_outputs]
        self.assertEqual(len(values),4)
        for i,layer in enumerate((4,8)):
            expected=core.audio_fusions[str(layer)](torch.cat((boundaries[i][:,:126],pair[i]),dim=-1))
            torch.testing.assert_close(values[i][:,:126],expected)
            torch.testing.assert_close(values[i][:,126:],boundaries[i][:,126:],rtol=0,atol=0)
        for i in (2,3): torch.testing.assert_close(values[i],boundaries[i],rtol=0,atol=0)
        for i in (0,1,2):
            torch.testing.assert_close(core.last_router_queries[f'write_{i}'],core.last_core_inputs[i],rtol=0,atol=0)
        with self.assertRaises(ValueError): model(inputs_embeds=x,audio_information=pair,use_cache=True)
        with self.assertRaises(ValueError): model(inputs_embeds=x,use_cache=False)
        core.gradient_checkpointing=True
        with self.assertRaises(RuntimeError): model(inputs_embeds=x,audio_information=pair,use_cache=False)

    def test_full_checkpoint_real_optimizer_moments_rng_and_next_update_match(self):
        torch=self.torch; model=self.fixture(); opt,sched,_=self.optimizer(model,34626); tracker=self.tracker_class()
        def update(m,o,s,t):
            o.zero_grad(set_to_none=True); self.loss(m).backward(); t.track_and_clip_(m.named_parameters()); o.step(); s.step()
        update(model,opt,sched,tracker)
        core=model.caption_decoder.lm.model
        core.fusion_gradient_state['seen'].update({n:n.endswith('_fusion') or n.endswith('_B') for n in self.new_roles})
        rng=dict(python=random.getstate(),numpy=self.np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.get_rng_state())
        def gather(states,value): states[:]=[copy.deepcopy(value) for _ in range(8)]
        scope=self.scope
        scope.update(dist=NS(all_gather_object=gather,get_rank=lambda:0,barrier=lambda:None),rng_state=lambda device:rng)
        recipe={'geometry':self.C.geometry(1477492)}
        config=dict(mode='formal',contract=recipe,contract_sha256=self.C.digest(recipe))
        scaler=NS(state_dict=lambda:{'scale':65536})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'step1.ckpt'
            scope['save_checkpoint'](path,model,opt,sched,scaler,tracker,config,1,0,1,'cpu')
            saved=torch.load(path,map_location='cpu',weights_only=False)
            self.C.validate_resume_payload(saved,config,recipe['geometry'])
            restored=self.fixture(); other_opt,other_sched,_=self.optimizer(restored,34626); other_tracker=self.tracker_class()
            scope['validate_resume_model_state'](saved,restored,other_opt)
            restored.load_state_dict(saved['state_dict'],strict=True); other_opt.load_state_dict(saved['optimizer'])
            other_sched.load_state_dict(saved['scheduler']); other_tracker.load_state_dict(saved['grad_norm_tracker'])
            restored.caption_decoder.lm.model.fusion_gradient_state=copy.deepcopy(saved['fusion_gradient_state'])
            for m,o,s,t in ((model,opt,sched,tracker),(restored,other_opt,other_sched,other_tracker)):
                torch.set_rng_state(saved['random_state_by_rank'][3]['torch'])
                random.setstate(saved['random_state_by_rank'][3]['python']); self.np.random.set_state(saved['random_state_by_rank'][3]['numpy'])
                update(m,o,s,t)
            for name,value in model.state_dict().items(): self.assertTrue(torch.equal(value,restored.state_dict()[name]),name)
            for key,state in opt.state_dict()['state'].items():
                for kind,value in state.items(): torch.testing.assert_close(value,other_opt.state_dict()['state'][key][kind],rtol=0,atol=0)
            self.assertEqual(tracker.state_dict(),other_tracker.state_dict())
            invalid=copy.deepcopy(saved); invalid['fusion_gradient_state']['seen'].pop('loop3_lora_A')
            with self.assertRaises(ValueError): self.C.validate_resume_payload(invalid,config,recipe['geometry'])
            invalid=copy.deepcopy(saved); name=next(n for n in invalid['state_dict'] if '.lora_B.2' in n)
            invalid['state_dict'][name].fill_(float('nan'))
            with self.assertRaises(ValueError): scope['validate_resume_model_state'](invalid,restored,other_opt)


if __name__ == '__main__': unittest.main()
