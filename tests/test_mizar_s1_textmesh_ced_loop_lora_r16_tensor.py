"""Optional real CPU PyTorch checks. Synthetic inputs; no CED assets or CUDA.

Checkpoint collectives/scaler are explicit CPU fixtures, not an eight-GPU claim.
Run via validate_mizar_s1_textmesh_ced_loop_lora_r16.py; routers only process text.
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
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_textmesh_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16"
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
        from models.loop_lora import LoopLoRALinear, validate_loop_lora, call_with_loop_lora
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
                     parameter_role=parameter_role, SERIAL_INJECTION_CONTRACT=contract.SERIAL_INJECTION_CONTRACT,
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
        self.assertEqual(audit["matrix_count"], 280)
        expected = 10*2*16*(2*(32+32) + 2*(32+16) + 3*(32+48))
        self.assertEqual(audit["parameter_count"], expected)
        for index, layer in enumerate(model.model.layers):
            for target in self.contract.LOOP_LORA_CONTRACT["targets"]:
                module = layer.get_submodule(target)
                self.assertEqual(isinstance(module,self.linear), 5<=index<15)
                if 5<=index<15:
                    self.assertFalse(torch.equal(module.lora_A[0],module.lora_A[1]))
                    self.assertTrue(all(not b.ne(0).any() for b in module.lora_B))
        with self.assertRaises(ValueError): model.model.enable_loop_lora()
        module = model.model.layers[5].self_attn.q_proj
        module.lora_A[1] = module.lora_A[0]
        with self.assertRaises(ValueError): self.validate(model.model)

    def test_zero_lora_matches_same_hybrid_with_direct_base_projections(self):
        torch=self.torch
        model=self.lm().eval()
        reference=copy.deepcopy(model)
        x=torch.randn(1,130,32);pair=tuple(torch.randn(1,126,32) for _ in range(2))
        a=model(inputs_embeds=x,audio_information=pair,use_cache=False).logits
        def base_only(module,inputs): return torch.nn.functional.linear(inputs,module.weight,module.bias)
        with patch.object(self.linear,"forward",base_only):
            b=reference(inputs_embeds=x,audio_information=pair,use_cache=False).logits
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.square().sum().backward();b.square().sum().backward()
        params=dict(model.named_parameters())
        for name,p in reference.named_parameters():
            if ".lora_" in name: continue
            if p.grad is None:self.assertIsNone(params[name].grad)
            else:torch.testing.assert_close(params[name].grad,p.grad,rtol=0,atol=0)

    def test_nonzero_lora_each_projection_uses_own_loop_and_backward_survives_reset(self):
        torch = self.torch
        model = self.lm()
        calls, handles = [], []
        def check(module, args, output):
            loop=module.loop_index
            calls.append(loop)
            x=args[0]
            expected=torch.nn.functional.linear(x,module.weight,module.bias)+2*torch.nn.functional.linear(
                torch.nn.functional.linear(x,module.lora_A[loop]),module.lora_B[loop])
            torch.testing.assert_close(output,expected,rtol=0,atol=0)
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m,self.linear):
                    for b in m.lora_B: b.normal_(0,.02)
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
        self.assertEqual(len(before),12)
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
        p=next(p for n,p in model.named_parameters() if ".lora_B.1" in n)
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
        from models.recursive_layer_inject import SOURCE_LAYER_INDICES_0BASED
        torch=self.torch
        source=self.lm(initialize=False,width=576,intermediate=8)
        cfg=source.config
        cfg.recursive_prefix_layer_count=cfg.recursive_suffix_layer_count=5
        cfg.recursive_middle_layer_count=10; cfg.recursive_loops_scope="middle_only"
        cfg.mesh_memory_slots=5; cfg.mesh_router_count=6
        cfg.mesh_architecture_contract=self.contract.TEXT_MODEL_CONTRACT
        cfg.model_architecture_contract=self.contract.TEXT_MODEL_ARCHITECTURE
        cfg.logical_to_physical=list(self.contract.TEXT_MODEL_SCHEDULE)
        cfg.recursive_source_layer_indices_0based=list(SOURCE_LAYER_INDICES_0BASED)
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            loaded=load_recursive_text_model(directory)
        for name,p in source.state_dict().items(): torch.testing.assert_close(loaded.state_dict()[name],p,rtol=0,atol=0)
        extra=set(loaded.state_dict())-set(source.state_dict())
        self.assertEqual(len([n for n in extra if ".lora_" in n]),280)
        self.assertEqual(len(extra),288)

    def test_text_memory_queries_writes_serial_audio_and_final_bypass(self):
        torch=self.torch
        model=self.lm().eval();core=model.model
        core.audit_mode=True;core.gradient_audit_mode=True
        with torch.no_grad():
            for gate in list(core.audio_gates.values())+[core.audio_gate]:gate.weight.normal_(0,.02);gate.bias.fill_(.1)
            for module in model.modules():
                if isinstance(module,self.linear):
                    for matrix in module.lora_B:matrix.normal_(0,.02)
        x=torch.randn(1,130,32);pair=tuple(torch.randn(1,126,32) for _ in range(2))
        reads=[];stacks={};gate_inputs={};projections=[]
        original_read=core._read;original_stack=core._run_stack
        def read(memory,weights):
            self.assertEqual(tuple(memory.shape),(1,5,4,32))
            out=original_read(memory,weights);reads.append(out);return out
        def stack(hidden,indices,start,**kwargs):
            stacks[start]=hidden.detach().clone();return original_stack(hidden,indices,start,**kwargs)
        handles=[core.audio_linear.register_forward_hook(lambda m,a,o:projections.append(o.detach().clone()))]
        def hook(key):
            def capture(m,args):gate_inputs[key]=args[0].detach().clone()
            return capture
        for key,gate in core.audio_gates.items():handles.append(gate.register_forward_pre_hook(hook(key)))
        try:
            with patch.object(core,"_read",side_effect=read),patch.object(core,"_run_stack",side_effect=stack):
                result=model(inputs_embeds=x,audio_information=pair,use_cache=False)
        finally:
            for h in handles:h.remove()
        self.assertEqual(len(projections),1)
        torch.testing.assert_close(core.last_initial_memory[:,0],x[:,126:],rtol=0,atol=0)
        self.assertFalse(core.last_initial_memory[:,1:].ne(0).any())
        self.assertEqual(core.last_memory_shape,(1,5,4,32))
        self.assertEqual(core.last_serial_injection_trace,((5,"input"),(5,"layer8"),(15,"input"),(15,"layer4")))
        self.assertEqual(len(reads),3)
        sources=[core.last_prefix_output[:,:126],core.last_core_outputs[0][:,:126]]
        for loop,layer in enumerate((8,4)):
            before=sources[loop];updated=before+torch.tanh(core.audio_gate(before))*projections[0]
            torch.testing.assert_close(gate_inputs[str(layer)],updated,rtol=0,atol=0)
            expected=updated+torch.tanh(core.audio_gates[str(layer)](updated))*pair[1-loop]
            torch.testing.assert_close(core.last_core_inputs[loop][:,:126],expected,rtol=0,atol=0)
            torch.testing.assert_close(core.last_core_inputs[loop][:,126:],reads[loop],rtol=0,atol=0)
            for kind in ("write","read"):
                torch.testing.assert_close(core.last_router_queries[f"{kind}_{loop}"],core.last_core_inputs[loop][:,126:],rtol=0,atol=0)
            weights=core.last_router_weights[f"write_{loop}"]
            memory=core.last_memory_write_history[loop]+core.last_core_outputs[loop][:,126:].unsqueeze(1)*weights.transpose(1,2).unsqueeze(-1)
            torch.testing.assert_close(core.last_memory_write_history[loop+1],memory,rtol=0,atol=0)
        torch.testing.assert_close(stacks[25][:,:126],core.last_core_outputs[1][:,:126],rtol=0,atol=0)
        torch.testing.assert_close(stacks[25][:,126:],reads[2],rtol=0,atol=0)
        result.logits.square().mean().backward()
        for ref in core.last_core_input_refs+core.last_core_output_refs:
            self.assertIsNotNone(ref.grad);self.assertTrue(torch.isfinite(ref.grad).all())

    def test_repeated_forward_isolation_cache_rejection_and_memory_lengths(self):
        torch=self.torch;model=self.lm().eval();core=model.model
        pair=tuple(torch.randn(1,126,32) for _ in range(2));x=torch.randn(1,130,32)
        a=model(inputs_embeds=x,audio_information=pair,use_cache=False).logits
        model(inputs_embeds=torch.randn(1,137,32),audio_information=pair,use_cache=False)
        self.assertEqual(core.last_memory_shape,(1,5,11,32))
        b=model(inputs_embeds=x,audio_information=pair,use_cache=False).logits
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(len(core.last_serial_injection_trace),4)
        for kw in ({"use_cache":True},{"past_key_values":object()},{"audio_information":None}):
            args=dict(inputs_embeds=x,audio_information=pair,use_cache=False);args.update(kw)
            with self.assertRaises(ValueError):model(**args)






if __name__=="__main__":
    unittest.main()
