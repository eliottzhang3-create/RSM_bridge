"""Real CPU tensors in mellow_c8204d8; synthetic CED, no assets/downloads/CUDA."""
import copy
import importlib.util
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT/"code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject"
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
        from models.recursive_layer_inject import RecursiveLlamaForCausalLM
        from recursive_model_5_10_5 import RecursiveLlamaForCausalLM as OriginalLM
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
        decoder.lm.model.enable_layer_injection()
        decoder.lm.model.enable_audio_reinjection()
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

    def test_zero_gate_logits_and_inherited_parameter_gradients_equal_original(self):
        torch=self.torch
        torch.manual_seed(3)
        model=self.lm()
        original=self.original_class(model.config)
        original.load_state_dict(model.state_dict(),strict=True)
        state=torch.get_rng_state().clone()
        model.model.enable_layer_injection()
        model.model.enable_audio_reinjection()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        model.eval(); original.eval()
        x=torch.randn(1,130,32,requires_grad=True)
        a=tuple(torch.randn(1,126,32,requires_grad=True) for _ in range(2))
        y=model(inputs_embeds=x,use_cache=False,audio_information=a).logits
        z=original(inputs_embeds=x,use_cache=False).logits
        torch.testing.assert_close(y,z,rtol=0,atol=0)
        y.square().sum().backward()
        grads={n:p.grad.clone() for n,p in model.named_parameters() if not n.startswith(("model.audio_gates.", "model.audio_linear.", "model.audio_gate."))}
        z.square().sum().backward()
        for name,p in original.named_parameters():
            torch.testing.assert_close(grads[name],p.grad,rtol=0,atol=0)
        self.assertTrue(all(value.grad is not None and not value.grad.ne(0).any() for value in a))

    def test_exact_schedule_and_both_gate_then_mlp_gradient_paths(self):
        model,inputs=self.fixture()
        optimizer,_,groups=self.build_optimizer(model,17313)
        calls=[]
        handles=[layer.register_forward_pre_hook(lambda _,args,index=i:calls.append(index))
                 for i,layer in enumerate(model.caption_decoder.lm.model.layers)]
        try: self.loss(model,inputs).backward()
        finally:
            for handle in handles: handle.remove()
        self.assertEqual(calls,list(self.contract.TEXT_MODEL_SCHEDULE))
        audit=self.train.first_gradient_audit(model)
        self.assertTrue(audit["passed"],audit)
        self.assertEqual(audit["closed_gates"],{"4":True,"8":True,"input":True})
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        self.loss(model,inputs).backward()
        audit=self.train.first_gradient_audit(model)
        self.assertTrue(audit["passed"],audit)
        self.assertTrue(all(audit["nonzero_by_role"].values()))
        self.assertFalse(any(audit["closed_gates"].values()))
        for group in groups:
            self.assertEqual(group["base_lr"],3e-4 if group["role"]=="default" else 6e-4)

    def test_rejects_missing_wrong_information_and_kv_cache(self):
        torch=self.torch
        model=self.lm()
        model.model.enable_layer_injection()
        model.model.enable_audio_reinjection()
        x=torch.randn(1,130,32)
        pair=(torch.randn(1,126,32),torch.randn(1,126,32))
        for kwargs in ({},{"audio_information":pair[:1]},
                       {"audio_information":(pair[0],pair[1][:,:125])},
                       {"audio_information":(pair[0].double(),pair[1])}):
            with self.subTest(kwargs=list(kwargs)),self.assertRaises(ValueError):
                model(inputs_embeds=x,use_cache=False,**kwargs)
        with self.assertRaises(ValueError): model(inputs_embeds=x,use_cache=True,audio_information=pair)

    def test_open_gates_use_matching_layer8_then_layer4_features(self):
        torch = self.torch
        model = self.lm().eval()
        core = model.model
        core.enable_layer_injection()
        core.enable_audio_reinjection()
        with torch.no_grad():
            core.audio_gates["8"].bias.fill_(.25)
            core.audio_gates["4"].bias.fill_(-.35)
        x = torch.randn(1, 130, 32)
        a4, a8 = torch.ones(1, 126, 32), torch.full((1, 126, 32), 2.)
        inject, calls = core.inject_audio, []
        def capture(hidden, information, layer, index):
            out = inject(hidden, information, layer, index)
            calls.append((index, layer, information))
            expected = torch.tanh(core.audio_gates[str(layer)].bias) * information
            torch.testing.assert_close(out[:, :126] - hidden[:, :126], expected)
            torch.testing.assert_close(out[:, 126:], hidden[:, 126:], rtol=0, atol=0)
            return out
        with patch.object(core, "inject_audio", side_effect=capture):
            model(inputs_embeds=x, audio_information=(a4, a8), use_cache=False)
        self.assertEqual([(index, layer) for index, layer, _ in calls], [(5, 8), (15, 4)])
        self.assertIs(calls[0][2], a8)
        self.assertIs(calls[1][2], a4)
        self.assertEqual(core.last_injection_trace, ((5, 8), (15, 4)))

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
        # Exercise the real full-checkpoint writer/reloader with eight synthetic rank RNG records.
        rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"model--step-1.ckpt"
            cfg={"mode":"formal","contract":{},"contract_sha256":"synthetic"}
            class Scaler:
                def state_dict(self): return {"synthetic_cpu":True}
            def gather(output,value): output[:]=[copy.deepcopy(value) for _ in range(8)]
            with patch.object(self.train,"rng_state",return_value={"torch":rng}), \
                 patch.object(self.train.dist,"all_gather_object",side_effect=gather), \
                 patch.object(self.train.dist,"get_rank",return_value=0), \
                 patch.object(self.train.dist,"barrier"):
                self.train.save_checkpoint(path,model,optimizer,scheduler,Scaler(),tracker,cfg,1,0,1,"cpu")
            self.assertEqual(self.train.reload_audit(path,model,cfg,1)["status"],"PASS")
            saved=torch.load(path,weights_only=False)
            restored=copy.deepcopy(model)
            restored.load_state_dict(saved["state_dict"],strict=True)
            opt2,sched2,_=self.build_optimizer(restored,17313)
            opt2.load_state_dict(saved["optimizer"]);sched2.load_state_dict(saved["scheduler"])
            track2=self.tracker_class();track2.load_state_dict(saved["grad_norm_tracker"])
            torch.set_rng_state(rng);update(model,optimizer,scheduler,tracker)
            torch.set_rng_state(saved["random_state_by_rank"][0]["torch"]);update(restored,opt2,sched2,track2)
            for name,p in model.state_dict().items():
                torch.testing.assert_close(p,restored.state_dict()[name],rtol=0,atol=0)
            self.assertEqual(tracker.state_dict(),track2.state_dict())
            self.assertEqual(scheduler.state_dict(),sched2.state_dict())

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
            "mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small")/"models/ced.py")
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
            self.assertEqual(2443776+2*(576+1)+576*576+2*576+1,2777859)


    def test_serial_nonzero_gates_read_updated_hidden_and_share_input_information(self):
        torch=self.torch
        model=self.lm().eval()
        core=model.model
        core.enable_layer_injection();core.enable_audio_reinjection()
        with torch.no_grad():
            core.audio_gate.weight.fill_(.012);core.audio_gate.bias.fill_(.2)
            core.audio_gates["8"].weight.fill_(.018);core.audio_gates["8"].bias.fill_(-.1)
            core.audio_gates["4"].weight.fill_(-.014);core.audio_gates["4"].bias.fill_(.15)
        x=torch.randn(1,130,32)
        pair=tuple(torch.randn(1,126,32) for _ in range(2))
        posts,infos,project_inputs,gate_inputs,events=[],[],[],[],[]
        inject_input,inject_layer=core.inject_input_audio,core.inject_audio
        def input_call(h,a,index):
            out=inject_input(h,a,index)
            expected=h[:,:126]+torch.tanh(core.audio_gate(h[:,:126]))*a
            torch.testing.assert_close(out[:,:126],expected)
            torch.testing.assert_close(out[:,126:],h[:,126:],rtol=0,atol=0)
            posts.append(out);infos.append(a);events.append((index,"input"))
            return out
        def layer_call(h,a,layer,index):
            self.assertIs(h,posts[-1])
            self.assertIs(a,pair[1 if layer==8 else 0])
            out=inject_layer(h,a,layer,index)
            expected=h[:,:126]+torch.tanh(core.audio_gates[str(layer)](h[:,:126]))*a
            torch.testing.assert_close(out[:,:126],expected)
            torch.testing.assert_close(out[:,126:],h[:,126:],rtol=0,atol=0)
            events.append((index,f"layer{layer}"))
            return out
        handle=core.audio_linear.register_forward_pre_hook(lambda _,args:project_inputs.append(args[0].detach().clone()))
        try:
            with patch.object(core,"inject_input_audio",side_effect=input_call),patch.object(core,"inject_audio",side_effect=layer_call):
                model(inputs_embeds=x,audio_information=pair,use_cache=False)
        finally: handle.remove()
        self.assertEqual(events,[(5,"input"),(5,"layer8"),(15,"input"),(15,"layer4")])
        self.assertEqual(tuple(events),core.last_serial_injection_trace)
        self.assertEqual(len(project_inputs),1)
        torch.testing.assert_close(project_inputs[0],x[:,:126],rtol=0,atol=0)
        self.assertIs(infos[0],infos[1])
        self.assertFalse(torch.equal(posts[0],posts[1]))

    def test_disabling_each_branch_matches_the_corresponding_parent(self):
        torch=self.torch
        family=ROUTE.parent
        def load_class(folder,filename,label):
            spec=importlib.util.spec_from_file_location(label,family/folder/"models"/filename)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            return module.RecursiveLlamaForCausalLM
        layer_cls=load_class(ROUTE.name.removesuffix("_audio_reinject"),"recursive_layer_inject.py","parent_layer84")
        input_cls=load_class("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_audio_reinject","recursive_reinject.py","parent_input")
        for disable in ("input","intermediate"):
            candidate=self.lm().eval();core=candidate.model
            core.enable_layer_injection();core.enable_audio_reinjection()
            with torch.no_grad():
                core.audio_gate.weight.fill_(.01);core.audio_gate.bias.fill_(.1)
                for gate in core.audio_gates.values(): gate.weight.fill_(.02);gate.bias.fill_(.15)
                if disable=="input":
                    for p in core.audio_gate.parameters(): p.zero_()
                    reference=layer_cls(copy.deepcopy(candidate.config)).float().eval()
                    reference.model.enable_layer_injection()
                    prefixes=("model.audio_gate.","model.audio_linear.")
                else:
                    for p in core.audio_gates.parameters(): p.zero_()
                    reference=input_cls(copy.deepcopy(candidate.config)).float().eval()
                    reference.model.enable_audio_reinjection()
                    prefixes=("model.audio_gates.",)
            reference.load_state_dict({k:v for k,v in candidate.state_dict().items() if not k.startswith(prefixes)},strict=True)
            x=torch.randn(1,130,32)
            pair=tuple(torch.randn(1,126,32) for _ in range(2))
            actual=candidate(inputs_embeds=x,audio_information=pair,use_cache=False).logits
            kwargs={"audio_information":pair} if disable=="input" else {}
            expected=reference(inputs_embeds=x,use_cache=False,**kwargs).logits
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_pretrained_loader_strictly_inherits_backbone_and_registers_eight_tensors(self):
        from models.recursive_text import load_recursive_text_model,validate_serial_injection
        torch=self.torch
        config=self.config_class(vocab_size=32,hidden_size=576,intermediate_size=8,
            num_hidden_layers=30,num_attention_heads=4,num_key_value_heads=1,
            max_position_embeddings=1024,pad_token_id=17,eos_token_id=0,use_cache=False,attention_dropout=0.)
        config.recursive_layer_count,config.recursive_loops=20,2
        config.recursive_prefix_layer_count=config.recursive_suffix_layer_count=5
        config.recursive_middle_layer_count,config.recursive_loops_scope=10,"middle_only"
        config.logical_to_physical=list(self.contract.TEXT_MODEL_SCHEDULE)
        config.recursive_source_layer_indices_0based=list(self.contract.TEXT_MODEL_SOURCE_MAPPING)
        original=self.original_class(config).float()
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            candidate=load_recursive_text_model(directory)
        extra=set(candidate.state_dict())-set(original.state_dict())
        self.assertEqual(extra,{"model.audio_linear.weight","model.audio_linear.bias", "model.audio_gate.weight","model.audio_gate.bias", "model.audio_gates.4.weight","model.audio_gates.4.bias","model.audio_gates.8.weight","model.audio_gates.8.bias"})
        for name,value in original.state_dict().items():
            torch.testing.assert_close(candidate.state_dict()[name],value,rtol=0,atol=0)
        self.assertEqual(sum(candidate.state_dict()[name].numel() for name in extra)+2443776,2777859)
        audit=validate_serial_injection(candidate)
        self.assertEqual(audit["trace"],[[5,"input"],[5,"layer8"],[15,"input"],[15,"layer4"]])
        self.assertTrue(all(p.eq(0).all() for p in candidate.model.audio_gates.parameters()))
        self.assertTrue(all(p.eq(0).all() for p in candidate.model.audio_gate.parameters()))
        self.assertTrue(candidate.model.audio_linear.bias.eq(0).all())

    def test_input_initial_values_rng_and_duplicate_registration_match_original(self):
        torch=self.torch
        folder=ROUTE.parent/"mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_audio_reinject"
        spec=importlib.util.spec_from_file_location("input_rng_reference",folder/"models/recursive_reinject.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        torch.manual_seed(71);combined=self.lm()
        state=torch.get_rng_state().clone()
        combined.model.enable_layer_injection();combined.model.enable_audio_reinjection()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        torch.manual_seed(71);original=module.RecursiveLlamaForCausalLM(copy.deepcopy(combined.config)).float()
        original.model.enable_audio_reinjection()
        for name,value in original.model.audio_linear.state_dict().items():
            torch.testing.assert_close(value,combined.model.audio_linear.state_dict()[name],rtol=0,atol=0)
        with self.assertRaises(RuntimeError): combined.model.enable_audio_reinjection()
        with self.assertRaises(RuntimeError): combined.model.enable_layer_injection()


if __name__ == "__main__":
    unittest.main()
