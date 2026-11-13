"""Optional CPU tensor checks in mellow_c8204d8; no remote weights or CUDA required."""
import ast
import copy
import importlib.util
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine"
ROUTE = FAMILY / "mizar_s1_ced_small_huginn_text_epoch1"
HAS_TENSORS = all(importlib.util.find_spec(name) is not None
                  for name in ("torch", "torchaudio", "transformers", "soundfile", "safetensors"))


@unittest.skipUnless(HAS_TENSORS, "requires torch/torchaudio/transformers/soundfile in mellow_c8204d8")
class NonMeshCEDTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(ROUTE), str(ROOT / "code/RSmol")]
        import torch
        from transformers import LlamaConfig
        from models.ced import CEDTemporalMapper
        from models.decoder import DecoderModel
        from optim import build_optimizer, GradNormTracker
        from models.huginn import MizarHuginnForCausalLM
        from models import huginn, recursive_text
        from mizar_text_5_10x2_5_huginn.contract import build_target_config
        import train
        cls.torch, cls.config_class, cls.model_class = torch, LlamaConfig, MizarHuginnForCausalLM
        cls.huginn, cls.recursive_text = huginn, recursive_text
        cls.build_config = staticmethod(build_target_config)
        cls.mapper_class, cls.decoder_class = CEDTemporalMapper, DecoderModel
        cls.build_optimizer = staticmethod(build_optimizer)
        cls.tracker_class, cls.train = GradNormTracker, train
        torch.set_num_threads(1)

    def make_model_and_inputs(self, *, normalized=True):
        torch, nn = self.torch, self.torch.nn
        torch.manual_seed(1234)
        config = self.config_class(vocab_size=32, hidden_size=32, intermediate_size=48,
            num_hidden_layers=30, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=1024, pad_token_id=17, eos_token_id=0,
            use_cache=False, attention_dropout=0.)
        config.recursive_layer_count, config.recursive_loops = 20, 2
        config=self.build_config(config)
        decoder = self.decoder_class.__new__(self.decoder_class)
        nn.Module.__init__(decoder)
        decoder.lm, decoder.lm_embedding_size = self.model_class(config).float(), 32
        if normalized:
            decoder.lm.model.enable_recurrent_boundary_norms()
        model = nn.Module()
        model.audio_encoder, model.caption_decoder = nn.Module(), decoder
        model.audio_encoder.base = nn.Module()
        model.audio_encoder.base.mapper = self.mapper_class(384, 4, 32, output_tokens=126, resample=False)
        targets = torch.randint(1, 16, (1, 300))
        targets[:, 199], targets[:, 200:] = 0, 17
        inputs = dict(grid=torch.randn(1, 126, 4, 384),
                      prompt={"input_ids": torch.ones(1, 256, dtype=torch.long)},
                      answer={"input_ids": targets, "attention_mask": targets.ne(17).long()})
        return model.train(), inputs

    def loss(self, model, inputs):
        audio = model.audio_encoder.base.mapper(inputs["grid"])
        output = model.caption_decoder(audio, None, inputs["prompt"], inputs["answer"])
        self.assertIsNone(output.past_key_values)
        self.assertEqual(tuple(output.logits.shape), (1, 683, 32))
        return self.train.answer_loss(output, {"answer": inputs["answer"]}, 17)

    def test_real_nonmesh_schedule_full_prefix_and_mapper_lm_backward(self):
        torch = self.torch
        model, inputs = self.make_model_and_inputs()
        calls = []
        handles = [layer.register_forward_pre_hook(lambda _, args, index=i: calls.append(index))
                   for i, layer in enumerate(model.caption_decoder.lm.model.layers)]
        try:
            self.loss(model, inputs).backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, list(range(15)) + list(range(5, 15)) + list(range(15, 20)))
        self.assertEqual(len({id(layer) for layer in model.caption_decoder.lm.model.layers}), 20)
        for name, parameter in model.named_parameters():
            self.assertFalse(any(part.startswith(("router", "mesh_", "memory_slot")) for part in name.split(".")))
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertTrue(any(p.grad.ne(0).any() for p in model.audio_encoder.base.mapper.parameters()))
        adapter=model.caption_decoder.lm.model.recurrent_adapter
        for half in adapter.weight.grad.split(32,dim=1):
            self.assertTrue(half.ne(0).any())

    def test_full_checkpoint_cpu_roundtrip_preserves_next_stochastic_update(self):
        torch = self.torch
        model, inputs = self.make_model_and_inputs()
        optimizer, scheduler, _ = self.build_optimizer(model, 200)
        tracker = self.tracker_class()

        def update(candidate, opt, sched, grad_tracker):
            self.loss(candidate, inputs).backward()
            grad_tracker.track_and_clip_(candidate.named_parameters())
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()

        update(model, optimizer, scheduler, tracker)
        state = {"cpu": torch.get_rng_state()}
        scaler = SimpleNamespace(state_dict=lambda: {"synthetic_cpu_scaler": True})
        config = dict(mode="formal", contract={"synthetic_cpu_test": True}, contract_sha256="fixture")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model--step-1.ckpt"
            def gather(output, local):
                output[:] = [copy.deepcopy(local) for _ in range(8)]
            with patch.object(self.train, "rng_state", return_value=state), \
                    patch.object(self.train.dist, "all_gather_object", side_effect=gather), \
                    patch.object(self.train.dist, "get_rank", return_value=0), \
                    patch.object(self.train.dist, "barrier"):
                self.train.save_checkpoint(path, model, optimizer, scheduler, scaler, tracker,
                                           config, 1, 0, 1, torch.device("cpu"))
            saved = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(saved["route_variant"], "ced_small_mizar_s1_fixed20_gbs256_huginn_text_epoch1_v1")
        self.assertEqual(len(saved["random_state_by_rank"]), 8)
        self.assertEqual((saved["total_step"], saved["epoch"], saved["next_optimizer_step"]), (1, 0, 1))
        self.assertIn("grad_scaler", saved)
        update(model, optimizer, scheduler, tracker)
        restored, _ = self.make_model_and_inputs()
        opt, sched, _ = self.build_optimizer(restored, 200)
        restored.load_state_dict(saved["state_dict"], strict=True)
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        other_tracker = self.tracker_class()
        other_tracker.load_state_dict(saved["grad_norm_tracker"])
        torch.set_rng_state(saved["random_state_by_rank"][0]["cpu"])
        update(restored, opt, sched, other_tracker)
        for (name, a), (_, b) in zip(model.named_parameters(), restored.named_parameters()):
            self.assertTrue(torch.equal(a, b), name)
        self.assertEqual(scheduler.state_dict(), sched.state_dict())
        self.assertEqual(tracker.state_dict(), other_tracker.state_dict())

    def test_three_real_mappers_share_initial_weights_before_different_text_initializers(self):
        torch, nn = self.torch, self.torch.nn
        class ToyCED(nn.Module):
            def __init__(self, encoder_config, d_out):
                super().__init__()
                self.encoder = nn.Linear(3, 3)
                self.mapper = NonMeshCEDTensorTests.mapper_class(384, 4, d_out, output_tokens=126, resample=False)
        states = []
        for index, route in enumerate(("mizar_stage1_fixed20_gbs256_3epochs_ced_small",
                "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small", ROUTE.name)):
            tree = ast.parse((FAMILY / route / "models/mellow.py").read_text(encoding="utf-8"))
            scope = dict(torch=torch, nn=nn, get_audio_encoder=lambda _: (ToyCED, 384),
                         get_decoder=lambda _: lambda *args: nn.Linear(5 + index, 5 + index))
            nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "actual-mellow-construction", "exec"), scope)
            torch.manual_seed(1234)
            model = scope["Mellow"]("CEDSmall", 384, "synthetic-text", 383, False, 576, True, True, "synthetic-ced")
            states.append({name: value.detach().clone() for name, value in model.audio_encoder.base.mapper.state_dict().items()})
        for name in states[0]:
            self.assertTrue(torch.equal(states[0][name], states[1][name]), name)
            self.assertTrue(torch.equal(states[0][name], states[2][name]), name)

    def test_trained_adapter_hf_weights_only_load_and_missing_key_rejected(self):
        import safetensors.torch
        torch=self.torch
        model,_=self.make_model_and_inputs(normalized=False)
        lm=model.caption_decoder.lm
        with torch.no_grad(): lm.model.recurrent_adapter.weight.add_(.125)
        expected=lm.model.recurrent_adapter.weight.detach().clone()
        inherited={name:value.detach().clone() for name,value in lm.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            lm.save_pretrained(path,safe_serialization=True)
            (path/'training_state.pt').write_bytes(b'poison audio initialization must not load')
            # Production geometry is checked by dependency-light tests. This
            # fixture uses tiny tensors while exercising the real strict load.
            with patch.object(self.recursive_text,'validate_text_initialization'), \
                 patch.object(self.recursive_text,'validate_loaded_model'):
                loaded=self.recursive_text.load_recursive_text_model(path)
                self.assertTrue(torch.equal(loaded.model.recurrent_adapter.weight,expected))
                self.assertIs(loaded.get_input_embeddings().weight,loaded.lm_head.weight)
                for name, value in inherited.items():
                    self.assertTrue(torch.equal(loaded.state_dict()[name],value), name)
                for norm in (loaded.model.recurrent_input_norm,loaded.model.recurrent_output_norm):
                    self.assertTrue(torch.equal(norm.weight,torch.ones(32)))
                weights=safetensors.torch.load_file(path/'model.safetensors')
                del weights['model.recurrent_adapter.weight']
                safetensors.torch.save_file(weights,path/'model.safetensors',metadata={'format':'pt'})
                with self.assertRaisesRegex(ValueError,'weights do not match'):
                    self.recursive_text.load_recursive_text_model(path)

    def test_multimodal_huginn_one_noise_sample_and_two_shared_adapter_calls(self):
        model,inputs=self.make_model_and_inputs()
        core=model.caption_decoder.lm.model
        seen=[]; prelude=[]; normalized_states=[]; norm_inputs=[]; core_inputs=[]
        def tensor(out): return out if self.torch.is_tensor(out) else out[0]
        handles=[core.recurrent_adapter.register_forward_pre_hook(lambda _m,args:seen.append(args[0])),
                 core.layers[4].register_forward_hook(lambda _m,_a,out:prelude.append(tensor(out))),
                 core.recurrent_output_norm.register_forward_hook(lambda _m,_a,out:normalized_states.append(out)),
                 core.recurrent_input_norm.register_forward_hook(lambda _m,_a,out:norm_inputs.append(out)),
                 core.layers[5].register_forward_pre_hook(lambda _m,args:core_inputs.append(args[0]))]
        try:
            with patch.object(self.huginn,'sample_initial_state',wraps=self.huginn.sample_initial_state) as sample:
                self.loss(model,inputs)
                self.assertEqual(sample.call_count,1)
            self.assertEqual(len(seen),2)
            self.assertEqual(tuple(seen[0].shape),(1,683,64))
            for value in seen: self.torch.testing.assert_close(value[...,32:],prelude[0],rtol=0,atol=0)
            self.assertEqual((len(norm_inputs),len(normalized_states),len(core_inputs)),(2,2,2))
            self.torch.testing.assert_close(seen[1][...,:32],normalized_states[0],rtol=0,atol=0)
            for actual, expected in zip(core_inputs,norm_inputs):
                self.torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        finally:
            for h in handles: h.remove()

    def test_adapter_uses_lm_lr_and_weight_decay_and_receives_optimizer_update(self):
        torch=self.torch
        model,inputs=self.make_model_and_inputs()
        optimizer,scheduler,audit=self.build_optimizer(model,17313)
        key='caption_decoder.lm.model.recurrent_adapter.weight'
        group=next(g for g in audit if key in g['names'])
        self.assertEqual((group['role'],group['base_lr'],group['weight_decay']),('default',3e-4,.01))
        before=model.caption_decoder.lm.model.recurrent_adapter.weight.detach().clone()
        self.loss(model,inputs).backward()
        self.tracker_class().track_and_clip_(model.named_parameters())
        optimizer.step(); scheduler.step()
        self.assertFalse(torch.equal(before,model.caption_decoder.lm.model.recurrent_adapter.weight))
        for name in ('recurrent_input_norm','recurrent_output_norm'):
            norm=getattr(model.caption_decoder.lm.model,name)
            group=next(g for g in audit if 'caption_decoder.lm.model.'+name+'.weight' in g['names'])
            self.assertEqual((group['role'],group['base_lr'],group['weight_decay']),('default',3e-4,0.))
            self.assertIsNotNone(norm.weight.grad)
            self.assertTrue(norm.weight.grad.ne(0).any())
            self.assertFalse(torch.equal(norm.weight,torch.ones_like(norm.weight)))

    def test_norm_installation_preserves_loaded_values_rng_and_normalized_hf_reload(self):
        torch=self.torch
        model,_=self.make_model_and_inputs(normalized=False)
        lm=model.caption_decoder.lm
        before={name:t.detach().clone() for name,t in lm.state_dict().items()}
        rng=torch.get_rng_state().clone()
        lm.model.enable_recurrent_boundary_norms()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,t in before.items(): self.assertTrue(torch.equal(t,lm.state_dict()[name]),name)
        slots=(lm.model.recurrent_input_norm,lm.model.recurrent_output_norm)
        self.assertIsNot(*slots)
        lm.model.enable_recurrent_boundary_norms()
        self.assertIs(lm.model.recurrent_input_norm,slots[0])
        with torch.no_grad():
            slots[0].weight.add_(.125)
            slots[1].weight.sub_(.25)
        with tempfile.TemporaryDirectory() as directory:
            lm.save_pretrained(directory,safe_serialization=True)
            restored,info=self.model_class.from_pretrained(directory,local_files_only=True,output_loading_info=True)
            self.assertFalse(any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')))
            for name,t in lm.state_dict().items(): self.assertTrue(torch.equal(t,restored.state_dict()[name]),name)
            broken=dict(restored.state_dict())
            del broken['model.recurrent_input_norm.weight']
            with self.assertRaises(RuntimeError): restored.load_state_dict(broken,strict=True)

    def test_gradient_diagnostics_inspect_values_before_clipping_and_keep_normal_policy(self):
        torch=self.torch
        for value,reason in ((1e6,'finite_gradient_norm_reached_emergency_threshold'),
                             (1e20,'fp32_norm_overflow_with_finite_gradients'),
                             (float('inf'),'nonfinite_gradient_values')):
            parameter=torch.nn.Parameter(torch.ones(2))
            parameter.grad=torch.full_like(parameter,value)
            tracker=self.tracker_class()
            tracker.track_and_clip_([('synthetic.weight',parameter)])
            report=tracker.last_anomaly
            self.assertIsNotNone(report)
            self.assertEqual(report['reason'],reason)
            if value==1e20:
                self.assertTrue(self.torch.isfinite(torch.tensor(report['gradient_l2_fp64_before_clip'])))
            if value==1e6:
                self.assertEqual(report['largest_gradients_fp64'][0]['max_abs'],1e6)
                self.assertTrue(parameter.grad.abs().max()<1e6)

    def test_audio_rank_seed_noise_and_source_probe_preserve_training_stream(self):
        import numpy as np
        from contextlib import nullcontext
        torch=self.torch
        streams=[]
        for rank in (0,1):
            seed=1234+rank
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            streams.append(self.huginn.sample_initial_state(torch.empty(1,683,32)))
        self.assertFalse(torch.equal(*streams))
        # Execute probe_sources on CPU, substituting its device RNG helper.
        class ProbeModel:
            def eval(self): pass
            def train(self): pass
            def __call__(self,inputs):
                self_noise=self_outer.huginn.sample_initial_state(torch.empty(1,683,32))
                raise RuntimeError('intentional probe failure after sampling')
        self_outer=self
        saved=torch.get_rng_state().clone()
        with patch.object(self.train,'rng_state',return_value=saved), \
             patch.object(self.train,'restore_rng',side_effect=lambda state,device:torch.set_rng_state(state)), \
             patch.object(self.train,'collate_fn',side_effect=lambda items:items), \
             patch.object(self.train,'move_batch',side_effect=lambda batch,device:batch):
            with self.assertRaisesRegex(RuntimeError,'after sampling'):
                self.train.probe_sources(ProbeModel(),SimpleNamespace(original_item=lambda i:{}),
                    {'probe_rows':{'AudioCaps':[0]}},torch.device('cpu'))
        self.assertTrue(torch.equal(saved,torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
