"""Real CPU/autograd checks; synthetic assets, no downloads, CUDA or job submission.

Eight-rank collectives and CUDA RNG storage are mocked in the checkpoint test.
That test is CPU continuation evidence, not an eight-GPU resume qualification.
"""
import ast
import copy
import importlib.util
import random
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_crossattn"
REQUIRED = ("torch", "torchaudio", "transformers", "soundfile", "numpy")
HAS_TENSORS = all(importlib.util.find_spec(name) is not None for name in REQUIRED)


@unittest.skipUnless(HAS_TENSORS, "requires torch/torchaudio/transformers/soundfile/numpy")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(ROUTE), str(ROOT / "code/RSmol")]
        import numpy as np
        import torch
        import contract
        import train
        from transformers import LlamaConfig
        from models.ced import CEDSmallWrapper, CEDTemporalMapper
        from models.decoder import DecoderModel
        from models.mellow import Mellow
        from models.multilayer_audio import MultilayerAudioCrossAttention
        from models.recursive_crossattn import RecursiveLlamaForCausalLM
        from recursive_model_5_10_5 import RecursiveLlamaForCausalLM as OriginalLM
        from optim import build_optimizer, GradNormTracker
        cls.torch, cls.np, cls.contract, cls.train = torch, np, contract, train
        cls.config_class, cls.lm_class, cls.original_class = LlamaConfig, RecursiveLlamaForCausalLM, OriginalLM
        cls.wrapper_class, cls.mapper_class = CEDSmallWrapper, CEDTemporalMapper
        cls.decoder_class, cls.mellow_class, cls.adapter_class = DecoderModel, Mellow, MultilayerAudioCrossAttention
        cls.build_optimizer, cls.tracker_class = staticmethod(build_optimizer), GradNormTracker
        torch.set_num_threads(1)

    def lm(self, enable=True):
        cfg = self.config_class(vocab_size=32, hidden_size=36, intermediate_size=48,
            num_hidden_layers=30, num_attention_heads=6, num_key_value_heads=2,
            max_position_embeddings=1024, pad_token_id=17, eos_token_id=0,
            use_cache=False, attention_dropout=0.)
        cfg.recursive_layer_count, cfg.recursive_loops = 20, 2
        model = self.lm_class(cfg).float()
        if enable:
            model.model.enable_audio_cross_attention()
        return model

    def context(self, length=389):
        torch = self.torch
        return (torch.randn(1, length, 36), torch.randn(1, 4, 126, 1536),
                torch.cat((torch.ones(1, 180), torch.zeros(1, 76)), dim=1).long())

    def toy_wrapper(self):
        torch, nn = self.torch, self.torch.nn
        class Block(nn.Module):
            def __init__(self, index):
                super().__init__()
                self.offset = nn.Parameter(torch.tensor(float(index)), requires_grad=False)
            def forward(self, x):
                return x + self.offset
        class CED(nn.Module):
            maximal_allowed_length = 1012
            def __init__(self):
                super().__init__()
                self.init_bn = nn.BatchNorm2d(64)
                self.blocks = nn.Sequential(*(Block(i) for i in range(1, 13)))
                self.norm = nn.LayerNorm(384)
            def forward_features(self, x):
                self.calls = getattr(self, "calls", 0) + 1
                self.base = torch.arange(x.shape[0]*252*384, dtype=x.dtype).reshape(x.shape[0],252,384)/1024
                return self.norm(self.blocks(self.base))
        wrapper = self.wrapper_class.__new__(self.wrapper_class)
        nn.Module.__init__(wrapper)
        wrapper.ced = CED().eval()
        for p in wrapper.ced.parameters():
            p.requires_grad = False
        wrapper.mapper = self.mapper_class(384,4,576,output_tokens=126,resample=False)
        wrapper.mapper_type = "freq_merge_126"
        wrapper._waveform_to_mel = lambda wave: torch.zeros(wave.shape[0],64,2001)
        return wrapper

    def fixture(self):
        torch, nn = self.torch, self.torch.nn
        torch.manual_seed(1234)
        decoder = self.decoder_class.__new__(self.decoder_class)
        nn.Module.__init__(decoder)
        decoder.lm, decoder.lm_embedding_size = self.lm(), 36
        model = nn.Module()
        model.audio_encoder, model.caption_decoder = nn.Module(), decoder
        model.audio_encoder.base = nn.Module()
        model.audio_encoder.base.mapper = self.mapper_class(384,4,36,output_tokens=126,resample=False)
        targets = torch.randint(1,16,(1,300))
        targets[:,199], targets[:,200:] = 0,17
        prompt = {"input_ids":torch.ones(1,256,dtype=torch.long),
                  "attention_mask":torch.cat((torch.ones(1,180),torch.zeros(1,76)),dim=1).long()}
        return model.train(), dict(grid=torch.randn(1,126,4,384), grids=torch.randn(1,4,126,1536),
            prompt=prompt, answer={"input_ids":targets,"attention_mask":targets.ne(17).long()})

    def loss(self, model, inputs):
        audio = model.audio_encoder.base.mapper(inputs["grid"])
        output = model.caption_decoder(audio,None,inputs["prompt"],inputs["answer"],audio_grids=inputs["grids"])
        self.assertEqual(tuple(output.logits.shape),(1,683,32))
        self.assertIsNone(output.past_key_values)
        return self.train.answer_loss(output,{"answer":inputs["answer"]},17)

    def test_production_geometry_initialization_parameter_count_and_rng(self):
        torch = self.torch
        state = torch.get_rng_state().clone()
        adapter = self.adapter_class()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in adapter.parameters()),4892741)
        self.assertEqual(adapter.cross_attention.num_heads,9)
        self.assertEqual(adapter.cross_attention.dropout,0.)
        self.assertEqual(len({id(p.weight) for p in adapter.projections}),4)
        norms = list(adapter.audio_input_norms) + list(adapter.audio_output_norms) + [adapter.text_norm]
        self.assertEqual(len({id(n) for n in norms + [adapter.output_norm]}),10)
        for norm in norms:
            self.assertEqual(norm.eps,1e-5)
            self.assertTrue(norm.elementwise_affine)
            torch.testing.assert_close(norm.weight,torch.ones_like(norm.weight),rtol=0,atol=0)
            self.assertFalse(norm.bias.ne(0).any())
        self.assertFalse(adapter.output_norm.elementwise_affine)
        self.assertEqual(list(adapter.output_norm.parameters()),[])
        self.assertEqual(adapter.residual_scale,.1)
        self.assertNotIn("residual_scale",dict(adapter.named_parameters()))
        self.assertTrue(adapter.cross_attention.out_proj.weight.ne(0).any())
        for layer in (adapter.gate1,adapter.gate2):
            self.assertFalse(layer.weight.ne(0).any())
            self.assertFalse(layer.bias.ne(0).any())
        for bias in adapter.round_bias:
            self.assertFalse(bias.ne(0).any())
        torch.testing.assert_close(torch.softmax(adapter.gate1(torch.randn(2,576)),dim=-1),torch.full((2,4),.25))

    def test_one_ced_forward_raw_layers_temporal_order_and_hook_cleanup(self):
        torch = self.torch
        wrapper = self.toy_wrapper()
        final, selected = wrapper._forward_hidden(torch.zeros(2,64,2001))
        self.assertEqual(wrapper.ced.calls,1)
        for value,offset in zip(selected,(6,21,45,78)):
            torch.testing.assert_close(value,(wrapper.ced.base+offset).reshape(2,504,384),rtol=0,atol=0)
        torch.testing.assert_close(final,wrapper.ced.norm(wrapper.ced.base+78).reshape(2,504,384),rtol=0,atol=0)
        grid = wrapper._hidden_to_temporal_grid(selected[1])
        raw = selected[1].reshape(2,2,4,63,384)
        torch.testing.assert_close(grid[:,63+19,2],raw[:,1,2,19],rtol=0,atol=0)
        self.assertTrue(all(not b._forward_hooks for b in wrapper.ced.blocks))
        with patch.object(wrapper.ced.blocks[7],"forward",side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError,"synthetic failure"):
                wrapper._forward_hidden(torch.zeros(1,64,2001))
        self.assertTrue(all(not b._forward_hooks for b in wrapper.ced.blocks))

    def test_encoder_frozen_mapper_and_four_projections_outside_no_grad(self):
        torch = self.torch
        wrapper = self.toy_wrapper().train()
        output = wrapper(torch.zeros(1,640000))
        self.assertEqual(tuple(output["layer_grids"].shape),(1,4,126,1536))
        self.assertFalse(output["layer_grids"].requires_grad)
        self.assertTrue(output["embedding"].requires_grad)
        adapter = self.adapter_class()
        bank = adapter.project_audio(output["layer_grids"],output["embedding"])
        # Output LN makes squared energy nearly constant. Use a nonconstant
        # weighted objective to exercise every upstream normalization/projection.
        ((bank*torch.randn_like(bank)).mean()+output["embedding"].square().mean()).backward()
        self.assertTrue(all(p.grad is not None and p.grad.ne(0).any() for p in adapter.projections.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.ne(0).any()
                            for norms in (adapter.audio_input_norms,adapter.audio_output_norms)
                            for p in norms.parameters()))
        self.assertTrue(all(p.grad is None for p in wrapper.ced.parameters()))
        self.assertFalse(wrapper.ced.training)
        self.assertTrue(wrapper.mapper.training)

    def test_zero_gate_logits_and_original_parameter_gradients_match_parent(self):
        torch = self.torch
        torch.manual_seed(7)
        model = self.lm(enable=False)
        original = self.original_class(model.config)
        original.load_state_dict(model.state_dict(),strict=True)
        rng = torch.get_rng_state().clone()
        model.model.enable_audio_cross_attention()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        model.eval(); original.eval()
        x, grids, mask = self.context()
        y = model(inputs_embeds=x,use_cache=False,audio_grids=grids,prompt_mask=mask).logits
        z = original(inputs_embeds=x,use_cache=False).logits
        torch.testing.assert_close(y,z,rtol=0,atol=0)
        y.square().mean().backward(); z.square().mean().backward()
        for name,p in original.named_parameters():
            torch.testing.assert_close(dict(model.named_parameters())[name].grad,p.grad,rtol=0,atol=0)

    def test_schedule_shared_calls_q_only_bias_and_single_projection(self):
        torch = self.torch
        model = self.lm().eval()
        adapter = model.model.audio_crossattn
        with torch.no_grad():
            adapter.gate2.bias.fill_(.25)
            adapter.round_bias[0].fill_(.2)
            adapter.round_bias[1].fill_(-.3)
        x, grids, mask = self.context()
        calls, projections, observations, norm_calls = [],[],[],[]
        handles = [layer.register_forward_pre_hook(lambda _,args,index=i:calls.append(index))
                   for i,layer in enumerate(model.model.layers)]
        handles += [layer.register_forward_hook(lambda *args,index=i:projections.append(index))
                    for i,layer in enumerate(adapter.projections)]
        handles += [layer.register_forward_hook(lambda *args,name=name:norm_calls.append(name))
                    for name,layer in adapter.named_modules() if isinstance(layer,torch.nn.LayerNorm)]
        original_forward = adapter.forward
        def observe(hidden,bank,prompt_mask,round_index):
            with patch.object(adapter.cross_attention,"forward",wraps=adapter.cross_attention.forward) as attention, \
                 patch.object(adapter.gate2,"forward",wraps=adapter.gate2.forward) as gate:
                output = original_forward(hidden,bank,prompt_mask,round_index)
            q,k,v = attention.call_args.args
            normalized = torch.nn.functional.layer_norm(hidden[:,127:], (36,),
                adapter.text_norm.weight,adapter.text_norm.bias,eps=1e-5)
            torch.testing.assert_close(q,normalized+adapter.round_bias[round_index])
            self.assertIs(k,v)
            torch.testing.assert_close(gate.call_args.args[0],normalized,rtol=0,atol=0)
            torch.testing.assert_close(output[:,:127],hidden[:,:127],rtol=0,atol=0)
            observations.append((round_index,id(bank)))
            return output
        try:
            with patch.object(adapter,"forward",side_effect=observe),torch.no_grad():
                model(inputs_embeds=x,audio_grids=grids,prompt_mask=mask,use_cache=False)
        finally:
            for handle in handles:handle.remove()
        self.assertEqual(calls,list(self.contract.TEXT_MODEL_SCHEDULE))
        self.assertEqual(projections,[0,1,2,3])
        self.assertEqual([r for r,_ in observations],[0,1])
        self.assertEqual(observations[0][1],observations[1][1])
        for i in range(4):
            self.assertEqual(norm_calls.count(f"audio_input_norms.{i}"),1)
            self.assertEqual(norm_calls.count(f"audio_output_norms.{i}"),1)
        self.assertEqual(norm_calls.count("text_norm"),2)
        self.assertEqual(norm_calls.count("output_norm"),2)

    def test_large_ca_outputs_obey_per_token_residual_bound_in_both_rounds(self):
        torch = self.torch
        model = self.lm().eval()
        adapter = model.model.audio_crossattn
        x,grids,mask = self.context()
        with torch.no_grad():
            adapter.gate2.bias.fill_(100.)
            adapter.cross_attention.out_proj.weight.mul_(1e6)
            adapter.cross_attention.out_proj.bias.fill_(1e6)
            adapter.collect_diagnostics = True
            bank = adapter.project_audio(grids*1e5,x)
            for r in (0,1):
                output = adapter(x,bank,mask,r)
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(output[:,:127],x[:,:127],rtol=0,atol=0)
                token_rms = (output[:,127:]-x[:,127:]).square().mean(dim=-1).sqrt()
                self.assertLessEqual(token_rms.max().item(),.1+1e-6)
                self.assertGreater(token_rms.max().item(),.099)
        for entry in adapter.diagnostics():
            self.assertGreater(entry["cross_attention_rms"],1000.)
            self.assertLessEqual(entry["normalized_cross_attention_rms"],1.+1e-6)
            self.assertLessEqual(entry["residual_token_rms_max"],.1+1e-6)
            self.assertEqual(entry["residual_scale"],.1)

    def test_gate2_opens_before_each_upstream_branch_receives_gradients(self):
        torch = self.torch
        model,inputs = self.fixture()
        optimizer,scheduler,groups = self.build_optimizer(model,17313)
        for group in optimizer.param_groups:
            self.assertAlmostEqual(group["lr"], .1 * group["initial_lr"])
        for update in range(3):
            optimizer.zero_grad(set_to_none=True)
            self.loss(model,inputs).backward()
            audit = self.train.first_gradient_audit(model)
            self.assertTrue(audit["passed"],audit)
            # The initial LR is nonzero: update 0 opens Gate2, so update 1's
            # backward must reach all four projections and both round biases.
            self.assertEqual(audit["all_branches_nonzero"],update >= 1,audit)
            optimizer.step();scheduler.step()
        for group in groups:
            self.assertEqual(group["base_lr"],3e-4 if group["role"]=="default" else 6e-4)
        self.assertTrue(all(audit["nonzero_by_branch"].values()))

    def test_open_gate_future_answers_cannot_change_earlier_logits(self):
        torch = self.torch
        model = self.lm().eval()
        with torch.no_grad():
            model.model.audio_crossattn.gate2.bias.fill_(.4)
            model.model.audio_crossattn.gate1.weight.normal_(std=.03)
        x,grids,mask = self.context(length=391)
        changed = x.clone();changed[:,387:] += 100
        with torch.no_grad():
            a = model(inputs_embeds=x,audio_grids=grids,prompt_mask=mask,use_cache=False).logits
            b = model(inputs_embeds=changed,audio_grids=grids,prompt_mask=mask,use_cache=False).logits
        torch.testing.assert_close(a[:,:387],b[:,:387],rtol=0,atol=0)
        self.assertFalse(torch.equal(a[:,387:],b[:,387:]))

    def test_full_sequence_equals_growing_full_prefix_recomputation(self):
        torch = self.torch
        model = self.lm().eval()
        with torch.no_grad():
            model.model.audio_crossattn.gate2.bias.fill_(.3)
        x,grids,mask = self.context(length=389)
        with torch.no_grad():
            full = model(inputs_embeds=x,audio_grids=grids,prompt_mask=mask,use_cache=False).logits
            for end in (383,386,389):
                prefix = model(inputs_embeds=x[:,:end],audio_grids=grids,prompt_mask=mask,use_cache=False).logits
                torch.testing.assert_close(prefix,full[:,:end],rtol=1e-5,atol=1e-6)

    def test_invalid_context_masks_and_cache_are_rejected(self):
        torch = self.torch
        model = self.lm().eval()
        x,grids,mask = self.context()
        for context in ({}, {"audio_grids":grids[:,:3],"prompt_mask":mask},
                        {"audio_grids":grids.double(),"prompt_mask":mask},
                        {"audio_grids":grids,"prompt_mask":mask*0},
                        {"audio_grids":grids,"prompt_mask":mask.float()*.5}):
            with self.subTest(keys=list(context)),self.assertRaises(ValueError):
                model(inputs_embeds=x,use_cache=False,**context)
        with self.assertRaisesRegex(ValueError,"use_cache=False"):
            model(inputs_embeds=x,audio_grids=grids,prompt_mask=mask,use_cache=True)

    def assert_state_equal(self, left, right):
        torch = self.torch
        if isinstance(left,torch.Tensor):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
        elif isinstance(left,dict):
            self.assertEqual(left.keys(),right.keys())
            for key in left:self.assert_state_equal(left[key],right[key])
        elif isinstance(left,(list,tuple)):
            self.assertEqual(len(left),len(right))
            for a,b in zip(left,right):self.assert_state_equal(a,b)
        else:
            self.assertEqual(left,right)

    def test_actual_full_checkpoint_restore_block_matches_next_stochastic_update(self):
        torch = self.torch
        model,inputs = self.fixture()
        optimizer,scheduler,_ = self.build_optimizer(model,17313)
        tracker = self.tracker_class()
        class Scaler:
            def state_dict(self):return {"cpu_fixture_scale":65536.}
            def load_state_dict(self,state):self.restored=state
        def update(m,o,s,t):
            o.zero_grad(set_to_none=True)
            loss = self.loss(m,inputs)*(1+.01*random.random()+.01*self.np.random.random())
            loss.backward();t.track_and_clip_(m.named_parameters());o.step();s.step()
        random.seed(8);self.np.random.seed(9)
        for _ in range(3):update(model,optimizer,scheduler,tracker)
        cfg={"mode":"formal","contract":{"synthetic":"CPU continuation"}}
        cfg["contract_sha256"]=self.contract.digest(cfg["contract"])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"model--step-3.ckpt"
            def gather(output,value):output[:]=[copy.deepcopy(value) for _ in range(8)]
            with patch.object(torch.cuda,"get_rng_state",return_value=torch.get_rng_state().clone()), \
                 patch.object(self.train.dist,"all_gather_object",side_effect=gather), \
                 patch.object(self.train.dist,"get_rank",return_value=0),patch.object(self.train.dist,"barrier"):
                self.train.save_checkpoint(path,model,optimizer,scheduler,Scaler(),tracker,cfg,3,0,3,"cpu")
            self.assertEqual(self.train.reload_audit(path,model,cfg,3)["status"],"PASS")
            saved=torch.load(path,weights_only=False)
            names=set(saved["state_dict"])
            self.assertTrue(any("round_bias.1" in name for name in names))
            # Rebuild every module, rather than copying the trained model.
            restored,_=self.fixture()
            opt2,sched2,_=self.build_optimizer(restored,17313)
            track2,scaler2=self.tracker_class(),Scaler()
            function=next(n for n in ast.parse((ROUTE/"train.py").read_text(encoding="utf-8")).body
                          if isinstance(n,ast.FunctionDef) and n.name=="train")
            block=next(n for n in function.body if isinstance(n,ast.If)
                       and ast.unparse(n.test)=="config['resume_checkpoint']")
            scope=dict(vars(self.train),config=dict(cfg,resume_checkpoint=str(path)),
                       geometry=self.contract.geometry(1477492),model=restored,optimizer=opt2,
                       scheduler=sched2,scaler=scaler2,tracker=track2,rank=0)
            exec(compile(ast.Module(body=[block],type_ignores=[]),"<production-resume-block>","exec"),scope)
            self.assertEqual((scope["step"],scope["start_epoch"],scope["start_batch"]),(3,0,3))
            self.assertEqual(scaler2.restored,Scaler().state_dict())
            for candidate,opt,sched,track in ((model,optimizer,scheduler,tracker),(restored,opt2,sched2,track2)):
                with patch.object(torch.cuda,"set_rng_state"):
                    self.train.restore_rng(saved["random_state_by_rank"][0],"cpu")
                update(candidate,opt,sched,track)
            self.assert_state_equal(model.state_dict(),restored.state_dict())
            self.assert_state_equal(optimizer.state_dict(),opt2.state_dict())
            self.assert_state_equal(scheduler.state_dict(),sched2.state_dict())
            self.assert_state_equal(tracker.state_dict(),track2.state_dict())

    def test_original_mapper_constructor_initialization_is_unchanged(self):
        torch,nn=self.torch,self.torch.nn
        cfg=dict(self.contract.CED_EXPECTED_CONFIG,model_type="ced",
            id2label={str(i):str(i) for i in range(527)},auto_map={
                "AutoConfig":"configuration_ced.CedConfig",
                "AutoModelForAudioClassification":"modeling_ced.CedForAudioClassification"})
        encoder=self.toy_wrapper().ced
        encoder.config=SimpleNamespace(**cfg,to_dict=lambda:cfg)
        loaded=nn.Module();loaded.encoder=encoder
        parent_path=ROUTE.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small")/"models/ced.py"
        spec=importlib.util.spec_from_file_location("crossattn_parent_ced",parent_path)
        parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in self.contract.CED_ASSET_FILES:(root/name).write_bytes(b"synthetic")
            self.contract.write_json(root/"config.json",cfg)
            with patch("models.ced.local_ced_config",return_value=nullcontext(encoder.config)), \
                 patch.object(parent,"local_ced_config",return_value=nullcontext(encoder.config)), \
                 patch("models.ced.AutoModelForAudioClassification.from_pretrained",return_value=(loaded,{})):
                torch.manual_seed(12);old=parent.CEDSmallWrapper({"pretrained_audioencoder_path":directory});old_rng=torch.get_rng_state()
                torch.manual_seed(12);new=self.wrapper_class({"pretrained_audioencoder_path":directory});new_rng=torch.get_rng_state()
            self.assertTrue(torch.equal(old_rng,new_rng))
            self.assert_state_equal(old.mapper.state_dict(),new.mapper.state_dict())


if __name__ == "__main__":
    unittest.main()
