"""Optional real CPU PyTorch checks. Synthetic inputs; no CED assets or CUDA.

Checkpoint collectives/scaler are explicit CPU fixtures, not an eight-GPU claim.
Run via validate_mizar_s1_ced_text_x3_loop_lora.py --require-tensors.
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
ORDER = "48"
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8"
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
                     SERIAL_INJECTION_CONTRACT=contract.SERIAL_INJECTION_CONTRACT, parameter_role=parameter_role,
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
            num_hidden_layers=40, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=1024, pad_token_id=17, eos_token_id=0,
            use_cache=False, attention_dropout=0., tie_word_embeddings=False)
        cfg.recursive_layer_count, cfg.recursive_loops = 20, 3
        cfg.recursive_architecture_contract = self.contract.TEXT_MODEL_ARCHITECTURE
        model = self.lm_class(cfg).float()
        if initialize:
            model.model.enable_layer_injection()
            model.model.enable_audio_reinjection()
            model.model.enable_loop_lora()
        return model

    def pure_text_class(self):
        spec = importlib.util.spec_from_file_location('x3_pure_text_reference', ROOT/'code/RSmol/recursive_model_5_10x3_5.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.RecursiveLlamaForCausalLM

    def test_zero_adapters_preserve_real_pure_x3_text_forward(self):
        torch = self.torch
        source = self.lm(initialize=False).eval()
        pure = self.pure_text_class()(copy.deepcopy(source.config)).float().eval()
        pure.load_state_dict(source.state_dict(), strict=True)
        source.model.enable_layer_injection(); source.model.enable_audio_reinjection(); source.model.enable_loop_lora()
        inputs = torch.randn(1, 130, 32)
        information = tuple(torch.randn(1, 126, 32) for _ in range(2))
        with torch.no_grad():
            expected = pure(inputs_embeds=inputs, use_cache=False).logits
            actual = source(inputs_embeds=inputs, audio_information=information, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(source.model.last_loop_lora_trace), 30)
        self.assertEqual(source.model.last_reinject_trace, (5, 15, 25))

    def test_real_forward_projects_original_audio_once_and_ced_reads_post_input_only_first_two(self):
        torch = self.torch
        model = self.lm().eval()
        core = model.model
        with torch.no_grad():
            core.audio_gate.weight.normal_(0, .03); core.audio_gate.bias.fill_(.12)
            for gate in core.audio_gates.values():
                gate.weight.normal_(0, .03); gate.bias.fill_(.07)
            for module in core.modules():
                if isinstance(module, self.linear):
                    for b in module.lora_B: b.normal_(0, .02)
        inputs = torch.randn(1, 130, 32, requires_grad=True)
        information = tuple(torch.randn(1, 126, 32, requires_grad=True) for _ in range(2))
        projected, input_gate_hidden, ced_hidden, entrance_hidden, schedule = [], [], {4: [], 8: []}, [], []
        handles = [core.audio_linear.register_forward_hook(lambda m, a, o: projected.append((a[0].detach().clone(), o.detach().clone()))),
                   core.audio_gate.register_forward_pre_hook(lambda m, a: input_gate_hidden.append(a[0].detach().clone())),
                   core.layers[5].register_forward_pre_hook(lambda m, a: entrance_hidden.append(a[0].detach().clone()))]
        for layer, gate in core.audio_gates.items():
            handles.append(gate.register_forward_pre_hook(lambda m, a, k=int(layer): ced_hidden[k].append(a[0].detach().clone())))
        for index, layer in enumerate(core.layers):
            handles.append(layer.register_forward_pre_hook(lambda m, a, k=index: schedule.append(k)))
        try:
            output = model(inputs_embeds=inputs, audio_information=information, use_cache=False)
        finally:
            for handle in handles: handle.remove()
        self.assertEqual(len(projected), 1)
        self.assertEqual(schedule, list(self.contract.TEXT_MODEL_SCHEDULE))
        torch.testing.assert_close(projected[0][0], inputs[:, :126], rtol=0, atol=0)
        self.assertEqual((len(input_gate_hidden), len(ced_hidden[4]), len(ced_hidden[8])), (3, 1, 1))
        with torch.no_grad():
            for loop in range(3):
                hidden = input_gate_hidden[loop]
                post_input = hidden + torch.tanh(torch.nn.functional.linear(hidden, core.audio_gate.weight, core.audio_gate.bias)) * projected[0][1]
                if loop < 2:
                    ced_layer = (4, 8)[loop]
                    torch.testing.assert_close(ced_hidden[ced_layer][0], post_input, rtol=0, atol=0)
                    gate = core.audio_gates[str(ced_layer)]
                    expected = post_input + torch.tanh(torch.nn.functional.linear(post_input, gate.weight, gate.bias)) * information[loop]
                else:
                    expected = post_input
                torch.testing.assert_close(entrance_hidden[loop][:, :126], expected, rtol=0, atol=0)
        self.assertEqual([list(x) for x in core.last_serial_injection_trace], self.contract.SERIAL_INJECTION_CONTRACT['trace'])
        output.logits.square().mean().backward()
        self.assertTrue(inputs.grad[:, :126].ne(0).any())
        self.assertTrue(core.audio_linear.weight.grad.ne(0).any())
        for info in information: self.assertTrue(info.grad.ne(0).any())
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
        self.assertEqual(audit["matrix_count"], 420)
        expected = 10*3*8*(2*(32+32) + 2*(32+16) + 3*(32+48))
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
            self.assertEqual(calls,[0]*70+[1]*70+[2]*70)
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
        self.assertEqual(len(before),14)
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
        # Missing third-loop gradients must fail the actual trainer audit.
        p=next(p for n,p in model.named_parameters() if ".lora_B.2" in n)
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
        source=self.pure_text_class()(self.lm(initialize=False,width=576,intermediate=8).config).float()
        cfg=source.config
        cfg.recursive_prefix_layer_count=cfg.recursive_suffix_layer_count=5
        cfg.recursive_middle_layer_count=10; cfg.recursive_loops_scope="middle_only"
        cfg.logical_to_physical=list(self.contract.TEXT_MODEL_SCHEDULE)
        cfg.recursive_source_layer_indices_0based=list(self.contract.TEXT_MODEL_SOURCE_MAPPING)
        cfg.recursive_logical_to_physical_schedule=list(self.contract.LEGACY_X2_SCHEDULE)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/"checkpoint-009244"
            source.save_pretrained(checkpoint)
            (checkpoint/"tokenizer.json").write_text("{}")
            from mizar_text_x3_support import final_text_metadata
            final_text_metadata(checkpoint)
            config_bytes=(checkpoint/'config.json').read_bytes()
            loaded=load_recursive_text_model(checkpoint)
            self.assertEqual((checkpoint/'config.json').read_bytes(),config_bytes)
        self.assertEqual(loaded.config.recursive_logical_to_physical_schedule,list(self.contract.TEXT_MODEL_SCHEDULE))
        self.assertIn(self.contract.LEGACY_SCHEDULE_ALIAS,loaded.recursive_config_compatibility['overrides'])
        for name,p in source.state_dict().items(): torch.testing.assert_close(loaded.state_dict()[name],p,rtol=0,atol=0)
        extra=set(loaded.state_dict())-set(source.state_dict())
        self.assertEqual(len([n for n in extra if ".lora_" in n]),420)
        self.assertEqual(len(extra),428)



if __name__=="__main__":
    unittest.main()
