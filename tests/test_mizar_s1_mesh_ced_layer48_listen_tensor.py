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
ROUTE = ROOT/"code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_listen"
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

    def test_zero_gate_logits_and_inherited_parameter_gradients_equal_original(self):
        torch=self.torch
        torch.manual_seed(3)
        model=self.lm()
        original=self.original_class(model.config)
        original.load_state_dict(model.state_dict(),strict=True)
        state=torch.get_rng_state().clone()
        model.model.enable_audio_listen()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        model.eval(); original.eval()
        x=torch.randn(1,130,32,requires_grad=True)
        a=tuple(torch.randn(1,126,32,requires_grad=True) for _ in range(2))
        y=model(inputs_embeds=x,use_cache=False,audio_information=a).logits
        z=original(inputs_embeds=x,use_cache=False).logits
        torch.testing.assert_close(y,z,rtol=0,atol=0)
        y.square().sum().backward()
        grads={n:p.grad.clone() for n,p in model.named_parameters() if not n.startswith(("model.audio_gates.", "model.listen_routers."))}
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
        self.assertEqual(audit["closed_gates"],{"4":True,"8":True})
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
        model.model.enable_audio_listen()
        x=torch.randn(1,130,32)
        pair=(torch.randn(1,126,32),torch.randn(1,126,32))
        for kwargs in ({},{"audio_information":pair[:1]},
                       {"audio_information":(pair[0],pair[1][:,:125])},
                       {"audio_information":(pair[0].double(),pair[1])}):
            with self.subTest(kwargs=list(kwargs)),self.assertRaises(ValueError):
                model(inputs_embeds=x,use_cache=False,**kwargs)
        with self.assertRaises(ValueError): model(inputs_embeds=x,use_cache=True,audio_information=pair)

    def test_actual_torch_reads_then_residuals_enter_both_loops(self):
        torch=self.torch
        model=self.lm().eval()
        core=model.model
        core.enable_audio_listen()
        with torch.no_grad():
            core.audio_gates["4"].bias.fill_(.2)
            core.audio_gates["8"].bias.fill_(-.3)
        x=torch.randn(1,131,32)
        pair=(torch.randn(1,126,32),torch.randn(1,126,32))
        reads,entrances=[],[]
        original_read=core._read
        def read(memory,weights):
            result=original_read(memory,weights)
            reads.append(result.detach().clone())
            return result
        handle=core.layers[5].register_forward_pre_hook(lambda _,args:entrances.append(args[0].detach().clone()))
        try:
            with patch.object(core,"_read",side_effect=read),torch.no_grad():
                model(inputs_embeds=x,use_cache=False,audio_information=pair)
        finally:
            handle.remove()
        self.assertEqual(len(reads),3)
        self.assertEqual(len(entrances),2)
        for branch,bias in enumerate((.2,-.3)):
            expected=reads[branch].clone()
            scale=torch.tanh(torch.full_like(reads[branch][:,:126,:1],bias))
            expected[:,:126]+=scale*pair[branch]
            torch.testing.assert_close(entrances[branch],expected,rtol=0,atol=0)
        self.assertEqual(core.last_listen_trace,((5,4),(15,8)))

    def test_strict_mesh_loader_inherits_every_router_before_zero_gates(self):
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
        self.assertEqual(sum(p.numel() for p in loaded.model.audio_gates.parameters()),1154)
        self.assertTrue(all(not p.ne(0).any() for p in loaded.model.audio_gates.parameters()))

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
        model.caption_decoder.lm.model.listen_gradient_state = {
            "complete": False, "seen": {f"layer{l}_{k}": k=="gate" for l in (4,8)
                                        for k in ("mlp","gate","listen")}, "deadline":32}
        # Exercise the real full-checkpoint writer/reloader with eight synthetic rank RNG records.
        rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"model--step-1.ckpt"
            recipe={"geometry":self.contract.geometry(1477492)}
            cfg={"mode":"formal","contract":recipe,"contract_sha256":self.contract.digest(recipe)}
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
            self.assertEqual(2443776+2*(576+1)+2*(576*5+5),2450700)


if __name__ == "__main__":
    unittest.main()
