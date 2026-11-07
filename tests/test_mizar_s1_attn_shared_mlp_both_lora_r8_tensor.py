"""Optional real CPU PyTorch checks. Synthetic inputs; no CED assets or CUDA.

Checkpoint collectives/scaler are explicit CPU fixtures, not an eight-GPU claim.
Run via validate_mizar_s1_attn_shared_mlp_both_lora_r8.py.
"""
import ast
import copy
import importlib.util
import os
import random
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORDER = "84"
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_recursive_5_10_5_ced_small_in84_loop_attn_shared_mlp_both_lora_r8"
HAS_TENSORS = all(importlib.util.find_spec(n) for n in ("torch", "transformers"))


@unittest.skipUnless(HAS_TENSORS, "requires torch and transformers; no downloads")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(ROUTE), str(ROOT / "code/RSmol")]
        import torch
        import contract
        import audit_reference
        from transformers import LlamaConfig
        from models.recursive_layer_inject import RecursiveLlamaForCausalLM
        from models.loop_lora import LoopLoRALinear, validate_loop_lora, call_with_loop_lora, validate_loop_lora_execution
        from optim import build_optimizer, GradNormTracker, parameter_role
        cls.torch, cls.contract, cls.audit = torch, contract, audit_reference
        cls.config_class, cls.lm_class, cls.linear = LlamaConfig, RecursiveLlamaForCausalLM, LoopLoRALinear
        cls.validate = staticmethod(validate_loop_lora)
        cls.call_scope = staticmethod(call_with_loop_lora)
        cls.build_optimizer, cls.tracker = staticmethod(build_optimizer), GradNormTracker
        cls.role = staticmethod(parameter_role)
        # Execute the actual audit/checkpoint functions without audio dependencies.
        nodes = audit_reference.tree(ROUTE / "train.py")
        scope = dict(torch=torch, WORLD_SIZE=8, EPOCHS=3, RANK_UPDATE_SAMPLES=32, BATCH_SIZE=8,
                     ACCUMULATION_STEPS=4, ROUTE=contract.ROUTE, TEXT_MODEL_SCHEDULE=contract.TEXT_MODEL_SCHEDULE,
                     SERIAL_INJECTION_CONTRACT=contract.SERIAL_INJECTION_CONTRACT, parameter_role=parameter_role,
                     validate_loop_lora_execution=validate_loop_lora_execution,
                     os=os, hashlib=__import__("hashlib"))
        for node in nodes.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"TRAINABLE_ROLES", "NEW_ROLES"} for t in node.targets):
                audit_reference.execute([node], scope)
        for name in ("first_gradient_audit", "role_snapshots", "weights_changed", "save_checkpoint", "reload_audit"):
            audit_reference.execute([audit_reference.find_function(nodes, name)], scope)
        cls.train_scope = scope
        torch.set_num_threads(1)

    def lm(self, initialize=True, width=32, intermediate=48):
        cfg = self.config_class(vocab_size=32, hidden_size=width, intermediate_size=intermediate,
            num_hidden_layers=30, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=1024, pad_token_id=17, eos_token_id=0,
            use_cache=False, attention_dropout=0., tie_word_embeddings=False)
        cfg.recursive_layer_count, cfg.recursive_loops = 20, 2
        model = self.lm_class(cfg).float()
        if initialize:
            model.model.enable_layer_injection()
            model.model.enable_audio_reinjection()
            model.model.enable_loop_lora()
        return model

    def test_exact_scope_storage_independence_base_identity_and_rng(self):
        torch = self.torch
        model = self.lm(initialize=False)
        before = {n:p for n,p in model.named_parameters()}
        inherited = {n:p.detach().clone() for n,p in model.named_parameters()}
        state = torch.get_rng_state().clone()
        model.model.enable_layer_injection(); model.model.enable_audio_reinjection(); model.model.enable_loop_lora()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for name, old in before.items():
            self.assertIs(dict(model.named_parameters())[name], old)
            torch.testing.assert_close(old, inherited[name], rtol=0, atol=0)
        audit = self.validate(model.model)
        self.assertEqual(audit["matrix_count"], 520)
        expected = 10*2*8*(2*(32+32) + 2*(32+16) + 3*3*(32+48))
        self.assertEqual(audit["parameter_count"], expected)
        for index, layer in enumerate(model.model.layers):
            for target in self.contract.LOOP_LORA_CONTRACT["targets"]:
                module = layer.get_submodule(target)
                self.assertEqual(isinstance(module,self.linear), 5<=index<15)
                if 5<=index<15:
                    expected_keys = ('shared',) if target.startswith('self_attn.') else ('shared','audio','text')
                    self.assertEqual(tuple(module.lora_A),expected_keys)
                    self.assertEqual((module.rank,module.alpha,module.scaling),
                                     (8,16,2.) if target.startswith('self_attn.') else (8,8,1.))
                    for branch in expected_keys:
                        self.assertFalse(torch.equal(module.lora_A[branch][0],module.lora_A[branch][1]))
                        self.assertTrue(all(not b.ne(0).any() for b in module.lora_B[branch]))
                    if module.mode == 'shared_modality':
                        self.assertFalse(torch.equal(module.lora_A['audio'][0],module.lora_A['text'][0]))
        with self.assertRaises(ValueError): model.model.enable_loop_lora()
        module = model.model.layers[5].self_attn.q_proj
        module.lora_A["shared"][1] = module.lora_A["shared"][0]
        with self.assertRaises(ValueError): self.validate(model.model)

    def test_zero_lora_matches_matching_serial_reference_logits_and_base_gradients(self):
        torch = self.torch
        parent = ROUTE.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject")
        source = (parent/"models/recursive_layer_inject.py").read_text(encoding="utf-8")
        if ORDER=="48": source=source.replace("branch = 1 - self.layer_inject_indices.index(logical_index)","branch = self.layer_inject_indices.index(logical_index)")
        module = types.ModuleType("matching_serial_reference")
        exec(compile(source,str(parent/"models/recursive_layer_inject.py"),"exec"),module.__dict__)
        model = self.lm().eval()
        reference = module.RecursiveLlamaForCausalLM(copy.deepcopy(model.config)).float().eval()
        reference.model.enable_layer_injection(); reference.model.enable_audio_reinjection()
        with torch.no_grad():
            for gate in (model.model.audio_gate,*model.model.audio_gates.values()):
                gate.weight.normal_(0,.02); gate.bias.fill_(.1)
        reference.load_state_dict({n:p for n,p in model.state_dict().items() if ".lora_" not in n},strict=True)
        x = torch.randn(1,130,32)
        pair = tuple(torch.randn(1,126,32) for _ in range(2))
        a=model(inputs_embeds=x,audio_information=pair,use_cache=False).logits
        b=reference(inputs_embeds=x,audio_information=pair,use_cache=False).logits
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.square().sum().backward(); b.square().sum().backward()
        params = dict(model.named_parameters())
        for name,p in reference.named_parameters():
            if p.grad is None: self.assertIsNone(params[name].grad)
            else: torch.testing.assert_close(params[name].grad,p.grad,rtol=0,atol=0)

    def test_nonzero_lora_each_projection_uses_own_loop_and_backward_survives_reset(self):
        torch = self.torch
        model = self.lm()
        calls, handles = [], []
        def check(module, args, output):
            loop=module.loop_index
            calls.append(loop)
            x=args[0]
            if module.mode == 'shared':
                delta=torch.nn.functional.linear(torch.nn.functional.linear(x,module.lora_A['shared'][loop]),module.lora_B['shared'][loop])
            else:
                delta=torch.cat([torch.nn.functional.linear(torch.nn.functional.linear(tokens,module.lora_A[branch][loop]),module.lora_B[branch][loop])
                                 for branch,tokens in (('audio',x[:,:126]),('text',x[:,126:]))],dim=1)
            if module.mode != 'shared':
                delta=torch.nn.functional.linear(torch.nn.functional.linear(x,module.lora_A['shared'][loop]),module.lora_B['shared'][loop])+delta
            expected=torch.nn.functional.linear(x,module.weight,module.bias)+(2. if module.mode=='shared' else 1.)*delta
            torch.testing.assert_close(output,expected,rtol=0,atol=0)
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m,self.linear):
                    for branches in m.lora_B.values():
                        for b in branches: b.normal_(0,.02)
                    handles.append(m.register_forward_hook(check))
        try:
            out=model(inputs_embeds=torch.randn(1,130,32),audio_information=tuple(torch.randn(1,126,32) for _ in range(2)),use_cache=False)
            self.assertEqual(calls,[0]*70+[1]*70)
            self.assertTrue(all(m.loop_index is None for m in model.modules() if isinstance(m,self.linear)))
            out.logits.square().mean().backward()
            for name,p in model.named_parameters():
                if ".lora_" in name:
                    self.assertIsNotNone(p.grad,name)
                    self.assertTrue(p.grad.ne(0).any(),name)
        finally:
            for h in handles: h.remove()

    def fixture(self):
        torch,nn=self.torch,self.torch.nn
        model=nn.Module()
        model.caption_decoder=nn.Module(); model.caption_decoder.lm=self.lm()
        model.audio_encoder=nn.Module(); model.audio_encoder.base=nn.Module()
        base=model.audio_encoder.base
        base.mapper=nn.Linear(32,32)
        base.layer_mlps=nn.ModuleDict({str(k):nn.Linear(32,32) for k in (4,8)})
        return model

    def loss(self,model):
        torch=self.torch
        grid=torch.randn(1,126,32)
        base=model.audio_encoder.base
        embeds=torch.cat((base.mapper(grid),model.caption_decoder.lm.get_input_embeddings()(torch.tensor([[1,2,3,4]]))),dim=1)
        info=tuple(base.layer_mlps[str(k)](grid) for k in (4,8))
        return model.caption_decoder.lm(inputs_embeds=embeds,audio_information=info,use_cache=False).logits.square().mean()

    def test_first_then_opened_gradients_all_roles_optimizer_and_snapshots(self):
        torch=self.torch
        model=self.fixture()
        optimizer,scheduler,groups=self.build_optimizer(model,17313)
        before=self.train_scope["role_snapshots"](model)
        self.assertEqual(len(before),24)
        self.loss(model).backward()
        audit=self.train_scope["first_gradient_audit"](model)
        self.assertTrue(audit["passed"],audit)
        self.assertTrue(all(audit["closed_lora_B"].values()))
        for n,p in model.named_parameters():
            if ".lora_A." in n: self.assertFalse(p.grad.ne(0).any(),n)
            if ".lora_B." in n: self.assertTrue(p.grad.ne(0).any(),n)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        self.loss(model).backward()
        audit=self.train_scope["first_gradient_audit"](model)
        self.assertTrue(audit["passed"],audit)
        self.assertFalse(any(audit["closed_lora_B"].values()))
        for n,p in model.named_parameters():
            if ".lora_A." in n: self.assertTrue(p.grad.ne(0).any(),n)
        # Missing gradients in either adapter must fail the actual trainer audit.
        p=next(p for n,p in model.named_parameters() if ".lora_B.text.1" in n)
        old=p.grad; p.grad=None
        self.assertFalse(self.train_scope["first_gradient_audit"](model)["passed"])
        p.grad=old
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        changed=self.train_scope["weights_changed"](model,before)
        self.assertTrue(all(v["changed"] for v in changed.values()))
        ids=[id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        for group in groups:
            if "lora" in group["role"]:
                self.assertEqual(group["base_lr"],6e-4); self.assertEqual(group["weight_decay"],.01)

    def test_complete_cpu_checkpoint_resume_matches_next_update_including_rng_and_tracker(self):
        torch=self.torch
        model=self.fixture()
        optimizer,scheduler,_=self.build_optimizer(model,17313)
        tracker=self.tracker()
        def step(m,opt,sched,track):
            loss=self.loss(m)*(1+random.random()+np.random.random())
            loss.backward(); track.track_and_clip_(m.named_parameters())
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step(model,optimizer,scheduler,tracker)
        def rng(_): return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
        collective=NS(all_gather_object=lambda items,state:items.__setitem__(slice(None),[copy.deepcopy(state) for _ in range(8)]),
                      get_rank=lambda:0,barrier=lambda:None)
        scaler=NS(state_dict=lambda:{"cpu_fixture": True,"scale":1.})
        config=dict(mode="formal",contract={"loop_lora":self.contract.LOOP_LORA_CONTRACT},contract_sha256="cpu_fixture")
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"checkpoint.ckpt"
            with patch.dict(self.train_scope,dist=collective,rng_state=rng):
                self.train_scope["save_checkpoint"](path,model,optimizer,scheduler,scaler,tracker,config,1,0,1,None)
            self.assertEqual(self.train_scope["reload_audit"](path,model,config,1)["rng_ranks"],8)
            step(model,optimizer,scheduler,tracker)
            expected={n:p.clone() for n,p in model.state_dict().items()}
            tracker_expected=tracker.state_dict()
            restored=self.fixture(); opt2,sched2,_=self.build_optimizer(restored,17313); track2=self.tracker()
            payload=torch.load(path,map_location="cpu",weights_only=False)
            restored.load_state_dict(payload["state_dict"],strict=True)
            opt2.load_state_dict(payload["optimizer"]); sched2.load_state_dict(payload["scheduler"])
            track2.load_state_dict(payload["grad_norm_tracker"])
            state=payload["random_state_by_rank"][0]
            random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
            step(restored,opt2,sched2,track2)
            for name,p in restored.state_dict().items(): torch.testing.assert_close(p,expected[name],rtol=0,atol=0)
            self.assertEqual(track2.state_dict(),tracker_expected)
            self.assertEqual(sched2.state_dict(),scheduler.state_dict())
            for key,state in optimizer.state_dict()["state"].items():
                for part,value in state.items(): torch.testing.assert_close(opt2.state_dict()["state"][key][part],value,rtol=0,atol=0)

    def test_strict_text_loader_inherits_all_tensors_then_adds_lora_and_audio(self):
        from models.recursive_text import load_recursive_text_model
        torch=self.torch
        source=self.lm(initialize=False,width=576,intermediate=8)
        cfg=source.config
        cfg.recursive_prefix_layer_count=cfg.recursive_suffix_layer_count=5
        cfg.recursive_middle_layer_count=10; cfg.recursive_loops_scope="middle_only"
        cfg.logical_to_physical=list(self.contract.TEXT_MODEL_SCHEDULE)
        cfg.recursive_source_layer_indices_0based=list(self.contract.TEXT_MODEL_SOURCE_MAPPING)
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            loaded=load_recursive_text_model(directory)
        for name,p in source.state_dict().items(): torch.testing.assert_close(loaded.state_dict()[name],p,rtol=0,atol=0)
        extra=set(loaded.state_dict())-set(source.state_dict())
        self.assertEqual(len([n for n in extra if ".lora_" in n]),520)
        self.assertEqual(len(extra),528)

    def test_repeated_forward_metadata_modes_and_scope_recovery(self):
        from models.loop_lora import validate_loop_lora_execution
        torch=self.torch
        model=self.lm().eval();core=model.model
        x=torch.randn(1,130,32);pair=tuple(torch.randn(1,126,32) for _ in range(2))
        a=model(inputs_embeds=x,audio_information=pair,use_cache=False).logits
        self.assertEqual(validate_loop_lora_execution(core)['adapter_calls'],260)
        model(inputs_embeds=torch.randn(2,137,32),audio_information=tuple(torch.randn(2,126,32) for _ in range(2)),use_cache=False)
        self.assertEqual(validate_loop_lora_execution(core)['text_tokens'],11)
        b=model(inputs_embeds=x,audio_information=pair,use_cache=False).logits
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        projection=core.layers[5].self_attn.q_proj
        projection.mode='shared_modality'
        with self.assertRaises(ValueError):self.validate(core)
        with self.assertRaises(ValueError):validate_loop_lora_execution(core)
        projection.mode='shared'
        projection.forward_counts[0]=2
        with self.assertRaises(ValueError):validate_loop_lora_execution(core)
        layer=core.layers[5]
        def fail(*args,**kwargs):raise RuntimeError('synthetic decoder failure')
        with self.assertRaises(RuntimeError):self.call_scope(layer,0,fail)
        self.assertTrue(all(layer.get_submodule(t).loop_index is None for t in self.contract.LOOP_LORA_CONTRACT['targets']))
        model(inputs_embeds=x,audio_information=pair,use_cache=False)
        self.assertEqual(validate_loop_lora_execution(core)['projection_calls'],140)

    def test_attention_shared_autograd_is_sum_of_both_token_regions(self):
        torch=self.torch
        projection=self.linear(torch.nn.Linear(3,4),mode='shared').float()
        with torch.no_grad():
            for b in projection.lora_B['shared']:b.normal_(0,.1)
        x=torch.randn(2,131,3)
        for loop in (0,1):
            projection.loop_index=loop
            y=projection(x)
            matrices=(projection.lora_A['shared'][loop],projection.lora_B['shared'][loop])
            ga=torch.autograd.grad(y[:,:126].square().sum(),matrices,retain_graph=True)
            gt=torch.autograd.grad(y[:,126:].square().sum(),matrices,retain_graph=True)
            gf=torch.autograd.grad(y.square().sum(),matrices)
            for audio,text,full in zip(ga,gt,gf):
                self.assertTrue(audio.ne(0).any());self.assertTrue(text.ne(0).any())
                torch.testing.assert_close(full,audio+text,rtol=1e-5,atol=1e-5)
        projection.loop_index=None

    def test_mlp_projection_direct_boundary_isolation_and_autograd_by_round(self):
        torch=self.torch
        projection=self.linear(torch.nn.Linear(3,4),mode='shared_modality').float()
        with torch.no_grad():
            for branches in projection.lora_B.values():
                for b in branches:b.normal_(0,.1)
        x=torch.randn(2,131,3)
        for loop in (0,1):
            for branch,tokens,other in (('audio',slice(None,126),slice(126,None)),('text',slice(126,None),slice(None,126))):
                projection.loop_index=loop
                before=projection(x).detach()
                matrix=projection.lora_B[branch][loop]
                with torch.no_grad():saved=matrix.clone();matrix.add_(.2)
                after=projection(x).detach()
                torch.testing.assert_close(after[:,other],before[:,other],rtol=0,atol=0)
                self.assertFalse(torch.equal(after[:,tokens],before[:,tokens]))
                for position,owner in ((125,'audio'),(126,'text')):
                    self.assertEqual(torch.equal(after[:,position],before[:,position]),owner!=branch)
                with torch.no_grad():matrix.copy_(saved)
                projection.zero_grad(set_to_none=True)
                loss=projection(x)[:,tokens].square().sum()
                projection.loop_index=None
                loss.backward()
                for modality in ('shared','audio','text'):
                    for index in (0,1):
                        for collection in (projection.lora_A,projection.lora_B):
                            gradient=collection[modality][index].grad
                            if index==loop and modality in ('shared',branch):
                                self.assertIsNotNone(gradient);self.assertTrue(gradient.ne(0).any())
                            else:self.assertTrue(gradient is None or not gradient.ne(0).any())

    def test_mlp_parallel_sum_shared_autograd_and_scaling_validation(self):
        torch=self.torch
        projection=self.linear(torch.nn.Linear(3,4),mode='shared_modality').double()
        with torch.no_grad():
            for branches in projection.lora_B.values():
                for b in branches:b.normal_(0,.1)
        x=torch.randn(2,131,3,dtype=torch.float64)
        for loop in (0,1):
            projection.loop_index=loop
            y=projection(x)
            shared=torch.nn.functional.linear(torch.nn.functional.linear(x,projection.lora_A['shared'][loop]),projection.lora_B['shared'][loop])
            private=torch.cat([torch.nn.functional.linear(torch.nn.functional.linear(tokens,projection.lora_A[branch][loop]),projection.lora_B[branch][loop])
                               for branch,tokens in (('audio',x[:,:126]),('text',x[:,126:]))],dim=1)
            expected=torch.nn.functional.linear(x,projection.weight,projection.bias)+(shared+private)
            torch.testing.assert_close(y,expected,rtol=0,atol=0)
            matrices=(projection.lora_A['shared'][loop],projection.lora_B['shared'][loop])
            ga=torch.autograd.grad(y[:,:126].square().sum(),matrices,retain_graph=True)
            gt=torch.autograd.grad(y[:,126:].square().sum(),matrices,retain_graph=True)
            gf=torch.autograd.grad(y.square().sum(),matrices)
            for audio,text,full in zip(ga,gt,gf):
                self.assertTrue(audio.ne(0).any());self.assertTrue(text.ne(0).any())
                torch.testing.assert_close(full,audio+text,rtol=1e-10,atol=1e-10)
        model=self.lm()
        module=model.model.layers[5].mlp.gate_proj
        module.scaling=2.
        with self.assertRaises(ValueError):self.validate(model.model)
        module.scaling=1.
        module.alpha=16
        with self.assertRaises(ValueError):self.validate(model.model)
        module.alpha=8
        module.lora_A['shared'][0]=module.lora_A['audio'][0]
        with self.assertRaises(ValueError):self.validate(model.model)


if __name__=="__main__":
    unittest.main()
