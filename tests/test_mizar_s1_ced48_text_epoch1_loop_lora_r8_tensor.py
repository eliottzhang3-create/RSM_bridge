"""Optional real CPU PyTorch checks. Synthetic inputs; no CED assets or CUDA.

Checkpoint collectives/scaler are explicit CPU fixtures, not an eight-GPU claim.
Run via validate_mizar_s1_ced48_text_epoch1_loop_lora_r8.py.
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
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORDER = "48"
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced48_text_epoch1_loop_lora_r8"
HAS_TENSORS = all(importlib.util.find_spec(n) for n in ("torch", "transformers", "safetensors"))


@unittest.skipUnless(HAS_TENSORS, "requires torch, transformers and safetensors; no downloads")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(ROUTE), str(ROOT), str(ROOT / "code/RSmol")]
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

    def text_checkpoint(self, directory, sharded=False):
        # torch/pdb may already have imported the stdlib module named `code`.
        # Add only package lookup support, preserving its existing public API.
        if 'code' in sys.modules and not hasattr(sys.modules['code'], '__path__'):
            sys.modules['code'].__path__ = [str(ROOT / 'code')]
        from code.RSmol.recursive_model_mizar_text_5_10x2_5_loop_lora_r8 import MizarTextLoopLoRAR8ForCausalLM
        cfg = self.config_class(vocab_size=32, hidden_size=32, intermediate_size=48,
            num_hidden_layers=30, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=1024, pad_token_id=17, eos_token_id=0,
            use_cache=False, attention_dropout=0., tie_word_embeddings=True)
        cfg.recursive_layer_count, cfg.recursive_loops = 20, 2
        source = MizarTextLoopLoRAR8ForCausalLM(cfg).float().eval()
        # Both loops carry trained, distinct, nonzero A AND B, including every target.
        with self.torch.no_grad():
            for name, value in source.named_parameters():
                if '.lora_' in name:
                    value.normal_(mean=.01 if name.endswith('.0') else -.01, std=.02)
        options = dict(max_shard_size='20KB') if sharded else {}
        source.save_pretrained(directory, safe_serialization=True, **options)
        (Path(directory) / 'training_state.pt').write_bytes(b'POISON: never read text optimizer/RNG state')
        return source

    @contextmanager
    def tiny_loader(self):
        from models import recursive_text as loader
        # Production shape/JSON checks require hidden576/vocab49152/final-step metadata.
        # Replace only those boundaries for this tiny CPU fixture. HF loading, the
        # twenty physical layers, schedule, tensor equality and all A/B checks stay real.
        with patch.object(loader, 'recursive_checkpoint_contract', return_value={'initialization': {'fixture': True}}), \
             patch.object(loader, 'validate_recursive_config'):
            yield loader

    def test_real_hf_inheritance_shards_tied_weights_and_zero_audio_gate_logits(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            source = self.text_checkpoint(directory, sharded=True)
            with self.tiny_loader() as loader:
                loaded = loader.load_recursive_text_model(directory).eval()
            original = source.state_dict()
            for name, value in original.items():
                torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
            self.assertEqual(sum('.lora_' in n for n in original), 280)
            self.assertIs(loaded.get_input_embeddings().weight, loaded.get_output_embeddings().weight)
            self.assertEqual(set(loaded.state_dict()) - set(original), {
                'model.audio_gates.4.weight', 'model.audio_gates.4.bias',
                'model.audio_gates.8.weight', 'model.audio_gates.8.bias',
                'model.audio_linear.weight', 'model.audio_linear.bias',
                'model.audio_gate.weight', 'model.audio_gate.bias'})
            audit = loaded.text_initialization_audit
            self.assertTrue(audit['all_source_tensors_equal'])
            self.assertEqual(audit['lora_tensor_count'], 280)
            self.assertFalse(audit['lora_reset_after_load'])
            self.assertFalse(audit['read_text_training_state'])
            for gate in (*loaded.model.audio_gates.values(), loaded.model.audio_gate):
                self.assertFalse(gate.weight.ne(0).any())
                self.assertFalse(gate.bias.ne(0).any())
            embeds = torch.randn(1, 130, 32)
            info = tuple(torch.randn(1, 126, 32) for _ in range(2))
            # The same attention backend makes this an exact arithmetic comparison.
            source.config._attn_implementation = loaded.config._attn_implementation
            with torch.no_grad():
                expected = source(inputs_embeds=embeds, use_cache=False).logits
                actual = loaded(inputs_embeds=embeds, audio_information=info, use_cache=False).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_real_hf_loader_rejects_missing_trained_a_or_b(self):
        from safetensors.torch import load_file, save_file
        with tempfile.TemporaryDirectory() as directory:
            self.text_checkpoint(directory)
            path = Path(directory) / 'model.safetensors'
            original = load_file(str(path))
            with self.tiny_loader() as loader:
                for part in ('lora_A.0', 'lora_B.1'):
                    name = next(n for n in original if part in n)
                    save_file({n: v for n, v in original.items() if n != name}, str(path), metadata={'format': 'pt'})
                    with self.subTest(missing=name), self.assertRaisesRegex(ValueError, 'weights do not match'):
                        loader.load_recursive_text_model(directory)

    def test_fresh_audio_registration_preserves_rng_and_pretrained_tensors_and_refuses_reset(self):
        torch = self.torch
        model = self.lm(initialize=False)
        with torch.no_grad():
            for name, value in model.named_parameters():
                if '.lora_B.' in name:
                    value.normal_(std=.02)
        before = {n: p.clone() for n, p in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        model.model.enable_layer_injection()
        model.model.enable_audio_reinjection()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, value in before.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        for method in (model.model.enable_loop_lora, model.model.enable_layer_injection,
                       model.model.enable_audio_reinjection):
            with self.assertRaises((RuntimeError, ValueError)):
                method()

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
        return model



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
        # Represent trained text LoRA before the first audio optimizer step.
        with torch.no_grad():
            for name,param in model.named_parameters():
                if '.lora_B.' in name: param.normal_(std=.005)
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
        self.assertFalse(any(audit["closed_lora_B"].values()))
        for n,p in model.named_parameters():
            if ".lora_A." in n: self.assertTrue(p.grad.ne(0).any(),n)
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




if __name__=="__main__":
    unittest.main()
