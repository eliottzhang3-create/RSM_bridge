"""Isolated route tests: real Llama/AdamW/Gloo; explicit tiny audio fixtures.

Run from tests/, never the repository root (the existing code/ package shadows
Python's stdlib code module). The optional local audio adapter disables only
torchaudio's unused binary extension in this test process; Python transforms,
SoundFile and all production sources remain unchanged.
"""
from __future__ import annotations
import ast
import collections
import copy
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "code/RSmol/mizar_s3_l21_25x3_residual_gate"
sys.path.insert(0, str(PACKAGE))
import torch
torch.set_num_threads(1)
if os.environ.get("MIZAR_TEST_PYTHON_AUDIO") == "1":
    _find_spec = importlib.util.find_spec
    def _python_audio(name, *args, **kwargs):
        return None if name == "torchaudio.lib._torchaudio" else _find_spec(name, *args, **kwargs)
    with patch("importlib.util.find_spec", _python_audio):
        import torchaudio
from torch import nn
import torch.distributed as dist
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from model import MizarCoreTrain, validate_trainability
from contract import (SCHEDULE, TOTAL_ROWS, SOURCE_COUNTS, accumulation_for_update,
                      code_digest, digest, geometry, lr_factor, parameter_role, sha256_file)
from data.sampler import CustomDistributedSampler
from data.audio_paths import AudioPaths
from data.fixed20 import load_fixed20
from data.audiotext_dataset import AudioTextDataset, text_pair
from training_helpers import answer_loss, rng_state, restore_rng
from optim import build_optimizer, GradNormTracker
from train import optimizer_update, source_probe, gradient_audit, compare_probe_logits
import checkpoint
import prepare_manifest as pm
import stage_data as sd
from data.staging import StagingGate
from journal import recover


class TinyAudio(nn.Module):
    """Explicit fixture, not a substitute used by the production loader."""
    def __init__(self, hidden):
        super().__init__()
        self.map = nn.Linear(1, hidden)
        self.register_buffer("fixed", torch.ones(1))
    def forward(self, waveform):
        values = waveform[:, :126, None]
        return {"embedding": self.map(values)}


def config(implementation="sdpa", production=False):
    c = LlamaConfig(hidden_size=576 if production else 16,
        intermediate_size=1536 if production else 32, num_hidden_layers=30,
        num_attention_heads=9 if production else 4,
        num_key_value_heads=3 if production else 2,
        vocab_size=49152 if production else 64, tie_word_embeddings=True,
        pad_token_id=17, eos_token_id=0, attention_dropout=0., rope_theta=100000., use_cache=False)
    c._attn_implementation = implementation
    return c


def tiny(implementation="sdpa"):
    torch.manual_seed(29)
    model = MizarCoreTrain(config(implementation), audio_encoder=TinyAudio(16))
    model.enable_training()
    return model


def batch(rank=0, micro=0):
    generator = torch.Generator().manual_seed(700 + rank * 10 + micro)
    answer = torch.full((1,96),17,dtype=torch.long)
    count = 3 + rank + micro
    answer[:,:count] = torch.randint(1,16,(1,count),generator=generator)
    answer[:,count-1] = 0
    return {"audio1":torch.randn(1,126,generator=generator), "audio2":torch.zeros(1,2),
            "input":{"input_ids":torch.randint(1,64,(1,256),generator=generator)},
            "answer":{"input_ids":answer,"attention_mask":answer.ne(17).long()}}


def scalar_update(model, batches, optimizer, tracker):
    optimizer.zero_grad(set_to_none=True)
    losses = [answer_loss(model(b), b, 17) for b in batches]
    (sum(losses)/len(losses)).backward()
    tracker.track_and_clip_(model.named_parameters())
    optimizer.step()


def ddp_worker(rank, port, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method=f"tcp://127.0.0.1:{port}",rank=rank,world_size=2)
    try:
        model=tiny()
        ddp=torch.nn.parallel.DistributedDataParallel(model,broadcast_buffers=False)
        optimizer,scheduler,_=build_optimizer(model,5)
        scaler=torch.amp.GradScaler("cuda",enabled=False)
        tracker=GradNormTracker()
        for count in (4,1):
            optimizer_update(ddp,[batch(rank,m) for m in range(count)],torch.device("cpu"),
                             optimizer,scaler,tracker,17)
            scheduler.step()
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict()},Path(directory)/f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def empty_tail_worker(rank,port,directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2)
    try:
        model=tiny();ddp=torch.nn.parallel.DistributedDataParallel(model,broadcast_buffers=False)
        op,scheduler,_=build_optimizer(model,5);scaler=torch.amp.GradScaler('cuda',enabled=False);tracker=GradNormTracker()
        windows=([batch(rank,0),batch(rank,1)] if rank==0 else [batch(rank,0),None],
                 [batch(rank,2)] if rank==0 else [None])
        for batches in windows:
            optimizer_update(ddp,batches,torch.device('cpu'),op,scaler,tracker,17,dummy_batch=batch(rank,4));scheduler.step()
        torch.save(model.state_dict(),Path(directory)/f'empty{rank}.pt')
    finally:dist.destroy_process_group()


import contract as recipe
from contract import checkpoint_steps, training_cursor, GROUP_LRS, EPOCHS, group_lr, TRAINABLE_ROLES



TOKENIZER=Path(os.environ.get('MIZAR_TEST_MODEL','C:/Xlance/assets/Mizar/Mizar159M'))/'assets/smollm2-135m'

class ModelTests(unittest.TestCase):
    def test_native_30_parity_and_independent_initial_gates(self):
        for impl in ('eager','sdpa'):
            model=tiny(impl).eval();core=model.caption_decoder.lm.model
            self.assertEqual(tuple(core.loop_bias.shape),(2,16));self.assertEqual(core.loop_bias.count_nonzero(),0)
            self.assertEqual(len(core.repeat_gates),2)
            self.assertIsNot(core.repeat_gates[0].linear.weight,core.repeat_gates[1].linear.weight)
            for gate in core.repeat_gates:
                self.assertEqual(set(gate.state_dict()),{'linear.weight','linear.bias'})
                self.assertEqual(gate.linear.weight.count_nonzero(),0)
                torch.testing.assert_close(gate.linear.bias,torch.full((1,),-4.595),rtol=0,atol=0)
                out=gate(torch.randn(2,135,16))
                self.assertEqual(tuple(out.shape),(2,135,1))
                torch.testing.assert_close(out,torch.full_like(out,1/(1+math.exp(4.595))))
            native=LlamaForCausalLM(config(impl)).eval()
            native.load_state_dict({k:v for k,v in model.caption_decoder.lm.state_dict().items()
                                    if 'repeat_gates' not in k and 'loop_bias' not in k},strict=True)
            x=torch.randn(1,135,16)
            with torch.no_grad():
                a=core._execute(x,single_pass=True).last_hidden_state
                b=native.model(inputs_embeds=x,use_cache=False).last_hidden_state
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                # Force gates to exactly zero: extra passes have zero effect.
                for gate in core.repeat_gates:gate.linear.bias.fill_(-1000.)
                actual=core(inputs_embeds=x).last_hidden_state
                torch.testing.assert_close(actual,b,rtol=0,atol=0)
            with self.assertRaises(ValueError):core.enable_adapters()
            with self.assertRaises(ValueError):core(inputs_embeds=x,use_cache=True)
        cold=MizarCoreTrain(config(),audio_encoder=TinyAudio(16));state=torch.get_rng_state().clone()
        cold.enable_training();self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_independent_formula_all_tokens_bias_order_and_l26_once(self):
        model=tiny().eval();core=model.caption_decoder.lm.model
        with torch.no_grad():
            core.loop_bias.copy_(torch.arange(32).view(2,16)*.001)
            for i,gate in enumerate(core.repeat_gates):
                gate.linear.weight.copy_(torch.linspace(-.2,.3,16)[None]*(i+1))
                gate.linear.bias.fill_(-.7+i*.4)
        traces={i:[] for i in range(30)};gate_inputs=[[],[]];handles=[];norm_calls=[]
        def hook(i):
            def capture(module,args,result):
                value=result[0] if isinstance(result,tuple) else result
                traces[i].append((args[0].clone(),value.clone()))
            return capture
        for i,layer in enumerate(core.layers):handles.append(layer.register_forward_hook(hook(i)))
        for i,gate in enumerate(core.repeat_gates):
            handles.append(gate.register_forward_pre_hook(lambda m,args,i=i:gate_inputs[i].append(args[0].clone())))
        handles.append(core.norm.register_forward_hook(lambda *args:norm_calls.append(1)))
        x=torch.randn(1,479,16)
        with torch.no_grad():core(inputs_embeds=x)
        for handle in handles:handle.remove()
        self.assertEqual([len(traces[i]) for i in range(30)],[3 if 20<=i<=24 else 1 for i in range(30)])
        self.assertEqual(core.last_layer_trace,tuple(list(range(20))+list(range(20,25))*3+list(range(25,30))))
        self.assertEqual(len(core.last_layer_trace),40);self.assertEqual(core.last_loop_trace,(0,1,2))
        self.assertEqual(core.last_gated_trace,(1,2));self.assertEqual(norm_calls,[1])
        # First pass has neither gate nor bias and preserves the original path.
        torch.testing.assert_close(traces[20][0][0],traces[19][0][1],rtol=0,atol=0)
        previous=traces[24][0][1]
        for i in range(2):
            value=previous+core.loop_bias[i]
            torch.testing.assert_close(traces[20][i+1][0],value,rtol=1e-5,atol=1e-7)
            torch.testing.assert_close(gate_inputs[i][0],value,rtol=1e-5,atol=1e-7)
            # Independent scalar dot/RMS calculation over every token.
            norm=value/(value.square().sum(-1,keepdim=True)/16+core.config.rms_norm_eps).sqrt()
            gate=((norm*core.repeat_gates[i].linear.weight[0]).sum(-1,keepdim=True)
                  +core.repeat_gates[i].linear.bias[0]).sigmoid()
            previous=value+gate*(traces[24][i+1][1]-value)
            observed=traces[20][2][0]-core.loop_bias[1] if i==0 else traces[25][0][0]
            torch.testing.assert_close(previous,observed,rtol=1e-5,atol=1e-7)
        with torch.no_grad():
            a=core(inputs_embeds=x).last_hidden_state.clone()
            core(inputs_embeds=x+1)
            b=core(inputs_embeds=x).last_hidden_state
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_full_bptt_both_biases_gates_core_l26_and_frozen_suffix(self):
        model=tiny();validate_trainability(model,production=False);core=model.caption_decoder.lm.model
        frozen=checkpoint.sha_frozen(model);outputs=[];suffix=[]
        def keep(target):
            def hook(module,args,result):
                value=result[0] if isinstance(result,tuple) else result
                value.retain_grad();target.append(value)
            return hook
        handles=[core.layers[i].register_forward_hook(keep(outputs)) for i in range(20,25)]
        handles.append(core.layers[26].register_forward_hook(keep(suffix)))
        inputs=batch();answer_loss(model(inputs),inputs,17).backward()
        for handle in handles:handle.remove()
        self.assertEqual(len(outputs),15)
        self.assertTrue(all(v.grad is not None and v.grad.abs().sum()>0 for v in outputs+suffix))
        self.assertTrue(all(gradient_audit(model)['nonzero_task_gradients'].values()))
        self.assertTrue(all(row.abs().sum()>0 for row in core.loop_bias.grad))
        for gate in core.repeat_gates:
            self.assertGreater(gate.linear.weight.grad.abs().sum(),0)
            self.assertGreater(gate.linear.bias.grad.abs().sum(),0)
        self.assertGreater(core.layers[25].self_attn.q_proj.weight.grad.abs().sum(),0)
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if parameter_role(n)=='frozen'))
        op,_,_=build_optimizer(model,300);op.step();self.assertEqual(checkpoint.sha_frozen(model),frozen)
        model.eval();x=torch.randn(1,135,16);y=x.clone();y[:,132:]+=2
        with torch.no_grad():
            u=core(inputs_embeds=x).last_hidden_state;v=core(inputs_embeds=y).last_hidden_state
        torch.testing.assert_close(u[:,:132],v[:,:132],rtol=0,atol=0)

    def test_production_meta_six_layers_and_2306_added_parameters(self):
        with torch.device('meta'):
            model=MizarCoreTrain(config(production=True),audio_encoder=TinyAudio(576));model.enable_training()
        roles=collections.Counter()
        for n,p in model.named_parameters():roles[parameter_role(n)]+=p.numel()
        self.assertEqual([roles[f'layer{i}'] for i in range(21,27)],[3540096]*6)
        self.assertEqual((roles['loop_bias'],roles['gate2'],roles['gate3']),(1152,577,577))
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),21242882)
        self.assertIs(model.caption_decoder.lm.lm_head.weight,model.caption_decoder.lm.model.embed_tokens.weight)

    def test_exact_479_layout_independent_answer_ce_and_source_probe(self):
        model=tiny();inputs=batch();out=model(inputs);self.assertEqual(out.logits.shape,(1,479,64))
        target=inputs['answer']['input_ids'][0];valid=target.ne(17)
        expected=-torch.log_softmax(out.logits[0,382:478],-1)[torch.arange(96)[valid],target[valid]].mean()
        torch.testing.assert_close(answer_loss(out,inputs,17),expected)
        self.assertEqual(model.caption_decoder.last_layout['total'],479)
        model.eval()
        with torch.no_grad():audit=source_probe(model,inputs,17)
        self.assertEqual(audit['future_answer_causality'],'PASS')


class ProbeRegressionTests(unittest.TestCase):
    def test_reported_49152_vocab_near_zero_roundoff_passes(self):
        # Reproduce the reported 12/49152 near-zero failures without pretending
        # that a CPU fixture reproduces the remote CUDA kernel itself.
        reference=torch.zeros(1,49152);reference[0,0]=10.
        actual=reference.clone()
        actual[0,1:13]=1.4748889952898026e-5
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(actual,reference,rtol=1e-4,atol=1e-5)
        report=compare_probe_logits(actual,reference,'reported_roundoff')
        self.assertEqual(report['mismatched_elements'],0)
        self.assertTrue(report['argmax_equal']);self.assertEqual(report['total_elements'],49152)
        self.assertGreater(report['max_abs_error'],1e-5)
        self.assertLess(report['max_tolerance_ratio'],1.)
        actual[0,1]=1e-3
        with self.assertRaisesRegex(ValueError,'reported_roundoff'):
            compare_probe_logits(actual,reference,'reported_roundoff')

    def test_small_numerical_error_still_rejects_changed_greedy_token(self):
        a=torch.tensor([[1.00001,1.]],dtype=torch.float32)
        b=torch.tensor([[1.,1.00001]],dtype=torch.float32)
        torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-4)
        with self.assertRaisesRegex(ValueError,'"argmax_equal": false'):
            compare_probe_logits(a,b,'greedy_flip')

    def test_nonfinite_wrong_shape_and_non_fp32_rejected(self):
        valid=torch.zeros(1,4)
        for value in (float('nan'),float('inf'),float('-inf')):
            bad=valid.clone();bad[0,1]=value
            for a,b in ((bad,valid),(valid,bad)):
                with self.assertRaisesRegex(ValueError,'NaN/Inf'):compare_probe_logits(a,b,'nonfinite')
        for bad in (torch.zeros(2,4),torch.zeros(4),torch.zeros(1,0),valid.double()):
            with self.assertRaises(ValueError):compare_probe_logits(bad,valid,'shape_dtype')

    def test_actual_source_probe_length_drift_and_causal_failure(self):
        model=tiny().eval();lm=model.caption_decoder.lm;inputs=batch()
        original=lm.forward
        def perturb(*args,**kwargs):
            out=original(*args,**kwargs)
            if kwargs['inputs_embeds'].shape[1] in (383,384):
                logits=out.logits.clone()
                # Deterministic perturbation of every non-winning vocabulary
                # entry simulates cross-length FP32 roundoff only.
                top=logits[:,-1].argmax(-1)
                logits[:,-1] += 3e-5
                for row,index in enumerate(top):logits[row,-1,index] -= 3e-5
                out.logits=logits
            return out
        with torch.no_grad(),patch.object(lm,'forward',side_effect=perturb):
            report=source_probe(model,inputs,17)
        self.assertEqual(len(report['logit_comparisons']),2)
        self.assertTrue(all(r['argmax_equal'] for r in report['logit_comparisons']))
        self.assertEqual(report['future_answer_causality'],'PASS')
        full_calls=0
        def leak(*args,**kwargs):
            nonlocal full_calls
            out=original(*args,**kwargs)
            if kwargs['inputs_embeds'].shape[1]==479:
                full_calls+=1
                if full_calls==2:
                    out.logits=out.logits.clone();out.logits[:,0,0]+=1e-3
            return out
        with torch.no_grad(),patch.object(lm,'forward',side_effect=leak):
            with self.assertRaises(AssertionError):source_probe(model,inputs,17)

class DataTests(unittest.TestCase):
    def test_complete_two_epoch_shards_tail_and_half_epoch_saves(self):
        n=272796;g=geometry(n)
        self.assertEqual((g['epochs'],g['steps_per_epoch'],g['total_steps'],g['warmup_steps'],g['last_update_rows']),(2,2132,4264,100,28))
        self.assertEqual(checkpoint_steps(n),[1066,2132,3198,4264])
        epoch_orders=[]
        for epoch in range(2):
            shards=[]
            for rank in range(8):
                sampler=CustomDistributedSampler(range(n),8,rank,seed=1234);sampler.set_epoch(epoch)
                shard=list(sampler);shards+=shard;sampler.start_index=2131*16
                self.assertEqual(list(sampler),shard[2131*16:])
                self.assertEqual(len(sampler),4 if rank<4 else 3)
                self.assertEqual(recipe.rank_microbatch_size(n,rank,2131,0),4 if rank<4 else 3)
            expected=torch.randperm(n,generator=torch.Generator().manual_seed(1234+epoch)).tolist()
            self.assertEqual(shards,expected);self.assertEqual(len(set(shards)),n);epoch_orders.append(shards)
        self.assertNotEqual(*epoch_orders)
        self.assertEqual(recipe.accumulation_for_update(2131,n),1)
        self.assertEqual(recipe.training_cursor(2132,g),(1,0));self.assertEqual(recipe.training_cursor(4264,g),(2,0))
        self.assertEqual(recipe.GROUP_LRS,{'adapters':(6e-5,6e-6),'core':(1.5e-5,1.5e-6),'layer26':(5e-6,1e-6)})
        with self.assertRaises(ValueError):recipe.validate_data_counts(136398,recipe.SOURCE_COUNTS)

    def test_real_audio_resample_then_mono_center20_pad_and_text_eos(self):
        import numpy as np,soundfile as sf,torchaudio
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'stereo.wav';wave=np.stack((np.arange(336000)/672000,np.zeros(336000)),axis=1).astype('float32')
            sf.write(p,wave,16000,subtype='FLOAT');first,second=load_fixed20(p)
            whole=torchaudio.transforms.Resample(16000,32000)(torch.from_numpy(wave.T)).mean(0)
            start=(whole.numel()-640000)//2
            expected=whole[start:start+640000]
            self.assertFalse(torch.equal(expected,whole[:640000]))
            torch.testing.assert_close(torch.cat((first,second)),expected,rtol=0,atol=0)
            sf.write(p,wave[:100],16000,subtype='FLOAT');a,b=load_fixed20(p)
            self.assertEqual(a.shape,(320000,));self.assertEqual(b.count_nonzero(),0);self.assertEqual(a[200:].count_nonzero(),0)
            with self.assertRaises(ValueError):load_fixed20(p,p)
        dataset=AudioTextDataset.__new__(AudioTextDataset)
        dataset.tokenizer=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True)
        recipe.validate_checkpoint_tokenizer(dataset.tokenizer,49152)
        prompt=dataset.tokenize('what is heard? a) car b) train c) bird d) rain',256)
        answer=dataset.tokenize('a) car',96,answer=True)
        self.assertNotIn(0,prompt['input_ids'][0].tolist())
        count=int(answer['attention_mask'].sum());self.assertEqual(int(answer['input_ids'][0,count-1]),0)
        self.assertTrue(answer['input_ids'][0,count:].eq(17).all())
        with self.assertRaises(ValueError):dataset.tokenize(' sound'*300,256)

    def test_real_sealed_s3_adapter_full_index_and_mapping(self):
        from test_mizar_s3_data_prepare import Fixture
        import prepare_manifest as pm,prepare_index as pi
        from data.prepared_index import load_index
        from data.indexed_jsonl import IndexedRows
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp)
            for row in f.rows:row['prompt_contract']='alignkd_direct_mcq_v1'
            f.save();f.run()
            before={p.name:recipe.sha256_file(p) for p in f.output.iterdir()}
            with patch.object(pm,'validate_data_counts',lambda *a:2),patch('data.prepared_index.validate_data_counts',lambda *a:2):
                index=pi.ensure_index(f.output/'train.json',f.output/'resolved_paths.tsv',Path(tmp)/'cache',TOKENIZER)
                self.assertEqual(index,pi.ensure_index(f.output/'train.json',f.output/'resolved_paths.tsv',Path(tmp)/'cache',TOKENIZER))
                report=load_index(index);self.assertEqual(report['text_audit']['rows'],2)
                rows=IndexedRows(report['metadata'])
                for i,row in enumerate(rows):
                    self.assertEqual({k:v for k,v in row.items() if not k.startswith('s1_')},f.rows[i])
                    self.assertEqual(row['s1_original_paths'],[str((f.mcqwav if i==0 else f.avwav).resolve())])
                self.assertEqual(before,{p.name:recipe.sha256_file(p) for p in f.output.iterdir()})
                rows.close()
                (index/'rows.jsonl').write_text('corrupt')
                with self.assertRaises(ValueError):load_index(index)

class OptimizerTests(unittest.TestCase):
    def test_three_group_max_min_fixed_100_warmup_and_actual_adamw_lrs(self):
        model=tiny();op,sc,audit=build_optimizer(model,300)
        seen={id(p) for g in op.param_groups for p in g['params']}
        self.assertEqual(seen,{id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual(sum(len(g['params']) for g in op.param_groups),len(seen))
        self.assertEqual({g['lr_group'] for g in op.param_groups},set(GROUP_LRS))
        for g in op.param_groups:
            self.assertEqual((g['initial_lr'],g['min_lr']),GROUP_LRS[g['lr_group']])
            if g['role']=='loop_bias':self.assertEqual(g['weight_decay'],0)
        lr_used=[]
        for update in range(300):
            lr_used.append([g['lr'] for g in op.param_groups])
            op.step();sc.step()
        for i,g in enumerate(op.param_groups):
            maximum,minimum=GROUP_LRS[g['lr_group']]
            self.assertAlmostEqual(lr_used[99][i],maximum)
            self.assertAlmostEqual(lr_used[-1][i],minimum)
            self.assertAlmostEqual(lr_used[0][i],maximum*.1)
            self.assertTrue(all(lr_used[s][i]>=lr_used[s+1][i] for s in range(100,299)))
        self.assertEqual(geometry(325210)['warmup_steps'],100)
        self.assertAlmostEqual(recipe.group_lr('layer26',299,300)/GROUP_LRS['layer26'][0],.2)

class RecoveryTests(unittest.TestCase):
    def test_real_gloo_empty_rank_has_zero_task_weight(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as tmp:
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            mp.spawn(empty_tail_worker,args=(port,tmp),nprocs=2,join=True)
            actual=torch.load(Path(tmp)/'empty0.pt',weights_only=True);other=torch.load(Path(tmp)/'empty1.pt',weights_only=True)
            ref=tiny();op,scheduler,_=build_optimizer(ref,5);tracker=GradNormTracker()
            op.zero_grad(set_to_none=True)
            losses=[answer_loss(ref(batch(r,0)),batch(r,0),17) for r in range(2)]
            loss=(sum(losses)/2+answer_loss(ref(batch(0,1)),batch(0,1),17))/2
            loss.backward();tracker.track_and_clip_(ref.named_parameters());op.step();scheduler.step()
            scalar_update(ref,[batch(0,2)],op,tracker);scheduler.step()
            for n,v in actual.items():
                torch.testing.assert_close(v,other[n],rtol=0,atol=0)
                torch.testing.assert_close(v,ref.state_dict()[n],rtol=2e-5,atol=2e-7)

    def test_actual_two_epoch_trainer_mid_epoch_resume_matches(self):
        import train as trainer
        from contextlib import ExitStack
        class MiniDataset:
            def __init__(self,*args,**kwargs):
                self.order=recipe.data_order(81);self.pad_id=17;self.audio_paths=AudioPaths(Path.cwd(),'shared_until_complete')
            def __len__(self):return 81
            def original_item(self,index):
                b=batch(micro=index%3)
                return {**b,'audio1':b['audio1'][0],'audio2':b['audio2'][0],'row_index':index,'source':'AudioCaps'}
            def __getitem__(self,index):return self.original_item(self.order[index])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dist.init_process_group('gloo',init_method=(root/'init').as_uri(),rank=0,world_size=1)
            try:
                prepared=root/'prepared.json';prepared.write_text(json.dumps({'status':'PASS','metadata':'fixture','stage_root':tmp,
                    'read_policy':'shared_until_complete','probe_rows':{'AudioCaps':[0]}}))
                g=dict(geometry(81),world_size=1,global_batch_size=16,rank_samples=[81],samples_per_rank=81,steps_per_epoch=6,total_steps=12,warmup_steps=11)
                c={'route':recipe.ROUTE,'geometry':g,'code_sha256':code_digest(),'model_dir':tmp,'official_bundle':{}}
                cfg={'contract':c,'contract_sha256':digest(c),'run_dir':str(root/'full'),'prepared_report':str(prepared),
                     'checkpoint_steps':list(range(1,13)),'max_optimizer_steps':12,'resume_checkpoint':'','attempt_id':'fixture'}
                with ExitStack() as stack:
                    for name,value in (('WORLD_SIZE',1),('geometry',lambda n:g),('checkpoint_steps',lambda n:list(range(1,13))),('AudioTextDataset',MiniDataset),('load_official',lambda p:tiny()),
                        ('verify_bundle',lambda b:None),('inspect_tokenizer',lambda *a:{}),
                        ('validate_trainability',lambda m:validate_trainability(m,production=False)),
                        ('accumulation_for_update',lambda u,n:4 if u<5 else 1),
                        ('rank_microbatch_size',lambda n,r,u,m:max(0,min(4,81-u*16-m*4)))):
                        stack.enter_context(patch.object(trainer,name,value))
                    actual_save=trainer.save;resume_copy=root/'model--step-2.ckpt'
                    def keep(path,*args):
                        actual_save(path,*args)
                        if Path(path).name=='model--step-2.ckpt':
                            import shutil;shutil.copyfile(path,resume_copy)
                    stack.enter_context(patch.object(trainer,'save',keep))
                    Path(cfg['run_dir']).mkdir();trainer.train(cfg,0,torch.device('cpu'))
                    final=torch.load(Path(cfg['run_dir'])/'checkpoints/model--step-12.ckpt',weights_only=False)
                    self.assertEqual(json.loads((Path(cfg['run_dir'])/'training_report.json').read_text())['status'],'PASS')
                    self.assertEqual(len(list((Path(cfg['run_dir'])/'checkpoints').glob('*.ckpt'))),4)
                    resumed=root/'resumed';resumed.mkdir()
                    lines=(Path(cfg['run_dir'])/'metrics.jsonl').read_text().splitlines();(resumed/'metrics.jsonl').write_text('\n'.join(lines[:2])+'\n')
                    trainer.train(dict(cfg,run_dir=str(resumed),resume_checkpoint=str(resume_copy)),0,torch.device('cpu'))
                    result=torch.load(resumed/'checkpoints/model--step-12.ckpt',weights_only=False)
                    for n,v in final['state_dict'].items():torch.testing.assert_close(v,result['state_dict'][n],rtol=0,atol=0)
                    self.assertEqual(json.loads((resumed/'epoch1_rank0.json').read_text())['rows_consumed'],49)
                    self.assertEqual(json.loads((resumed/'epoch2_rank0.json').read_text())['rows_consumed'],81)
            finally:dist.destroy_process_group()

    def test_checkpoint_group_lrs_inventory_moments_and_four_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist.init_process_group('gloo',init_method=(Path(tmp)/'init').as_uri(),rank=0,world_size=1)
            try:
                model=tiny();g=geometry(321);op,sc,_=build_optimizer(model,6);tr=GradNormTracker();scaler=torch.amp.GradScaler('cuda',enabled=False)
                frozen=checkpoint.sha_frozen(model);c={'geometry':dict(g,world_size=1),'code_sha256':code_digest()};cfg={'contract':c,'contract_sha256':digest(c),'max_optimizer_steps':6}
                for step in range(1,7):
                    scalar_update(model,[batch(micro=step%3)],op,tr);sc.step()
                    p=Path(tmp)/f'model--step-{step}.ckpt';checkpoint.save(p,model,op,sc,scaler,tr,cfg,step,frozen,torch.device('cpu'))
                    saved=torch.load(p,weights_only=False);clean=tiny();cleanop=build_optimizer(clean,6)[0]
                    checkpoint.validate_saved(saved,clean,cleanop,cfg,allow_complete=step==6)
                    if step==2:resume=copy.deepcopy(saved)
                self.assertEqual(sorted(p.name for p in Path(tmp).glob('*.ckpt')),[f'model--step-{s}.ckpt' for s in (3,4,5,6)])
                for mutation in (lambda s:s.update(route_variant='mizar_s3_l11_14x3_residual_sigmoid_gate2_bias2_l15_center20_2epochs_v1'),
                    lambda s:s['optimizer']['param_groups'][0].update(min_lr=1e-7),
                    lambda s:s['scheduler']['base_lrs'].__setitem__(0,1e-3),
                    lambda s:s['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),
                    lambda s:s['state_dict']['audio_encoder.fixed'].add_(1),lambda s:s.update(next_optimizer_step=1)):
                    bad=copy.deepcopy(resume);mutation(bad);clean=tiny();cleanop=build_optimizer(clean,6)[0]
                    with self.assertRaises(ValueError):checkpoint.validate_saved(bad,clean,cleanop,cfg)
            finally:dist.destroy_process_group()

    def test_real_two_rank_gloo_ga4_and_tail_ga1(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as tmp:
            with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
            mp.spawn(ddp_worker,args=(port,tmp),nprocs=2,join=True)
            actual=torch.load(Path(tmp)/'rank0.pt',weights_only=False)['model'];other=torch.load(Path(tmp)/'rank1.pt',weights_only=False)['model']
            ref=tiny();op,sc,_=build_optimizer(ref,5);tr=GradNormTracker()
            for count in (4,1):scalar_update(ref,[batch(r,m) for r in range(2) for m in range(count)],op,tr);sc.step()
            for n,v in actual.items():
                torch.testing.assert_close(v,other[n],rtol=0,atol=0);torch.testing.assert_close(v,ref.state_dict()[n],rtol=2e-5,atol=2e-7)

    def test_journal_backup_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'metrics.jsonl';old=b'{"step":1}\n{"step":2}\n{"step":3}\n{"step":'
            p.write_bytes(old);recover(p,2);self.assertEqual(p.read_bytes(),b'{"step":1}\n{"step":2}\n')
            self.assertEqual(next(Path(tmp).glob('*.before-resume-*')).read_bytes(),old)
            p.write_bytes(b'{"step":1}\nBAD\n')
            with self.assertRaises(ValueError):recover(p,1)

class StagingTests(unittest.TestCase):
    def test_async_shared_start_complete_files_and_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"audio").mkdir();source=root/"source.wav";source.write_bytes(b"audio"*40)
            token="fixture";info={"token":token,"producer_pid":os.getpid(),"epoch":0,"start_batch":0}
            (root/".owner.json").write_text(json.dumps({"root":str(root),"token":token}))
            prepared={"stage_root":str(root),"read_policy":"shared_until_complete","planned_files":1,"planned_bytes":200,
                "staging_epoch_steps":2,"initial_steps":0,"phase_seconds":{},"precopy_seconds":0.,
                "async_staging":dict(info,status_path=str(root/"STAGING.json"))}
            out=root/"report.json";started=threading.Event();release=threading.Event();real_copy=sd.copy_one
            def delayed(*args):started.set();release.wait(10);return real_copy(*args)
            state=dict(info,heartbeat=time.time())
            errors=[]
            def produce():
                try:sd.copy_background(root,prepared,[("audio/a.wav",{"source":str(source),"bytes":200})],out,state,workers=1)
                except Exception as e:errors.append(e)
            with patch.object(sd,"copy_one",delayed):
                thread=threading.Thread(target=produce);thread.start();self.assertTrue(started.wait(10))
                current=json.loads(out.read_text());self.assertEqual(current["status"],"TRAINING_READY")
                gate=StagingGate(current);gate.wait_for_batch(0,0);self.assertFalse(gate.complete)
                paths=AudioPaths(root,"shared_until_complete")
                row={"s1_raw_paths":["audio/a.wav"],"s1_original_paths":[str(source)]}
                self.assertEqual(paths.resolve(row),[source])
                release.set();thread.join(15)
            self.assertFalse(thread.is_alive());self.assertFalse(errors)
            gate.wait_for_batch(0,1);self.assertTrue(gate.complete)
            paths.use_local();self.assertEqual(paths.resolve(row),[root/"audio/a.wav"])
            self.assertEqual((root/"audio/a.wav").read_bytes(),source.read_bytes())
            state.update(phase="FAILED",error="copy failure");(root/"STAGING.json").write_text(json.dumps(state))
            gate=StagingGate(current)
            with self.assertRaises(RuntimeError):gate.wait_for_batch(0,0)
            with self.assertRaises(ValueError):sd.cleanup(root,token)
            for value in ("/dev/shm", "/tmp/mizar_s1_gbs256_x", "/dev/shm/../x"):
                with self.assertRaises(ValueError):sd.validate_stage_root(value)

    def test_copy_changed_source_fails_without_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"source";source.write_bytes(b"abc")
            with self.assertRaises(IOError):sd.copy_one(root,("target",{"source":str(source),"bytes":4}))
            self.assertFalse((root/"target").exists());self.assertFalse((root/"target.partial").exists())

class ShellTests(unittest.TestCase):
    def test_default_preview_and_resume_output_root(self):
        bash=os.environ.get('MIZAR_TEST_BASH','C:/Program Files/Git/bin/bash.exe')
        wrapper=ROOT/'code/RSmol/run_mizar_s3_l21_25x3_residual_gate_5090.sh'
        r=subprocess.run([bash,wrapper.as_posix(),'--check-args'],capture_output=True,text=True,check=True)
        self.assertIn('queue=pdgpu-5090 cpus=32 memory=256G gpus=8 nodes=1',r.stdout)
        self.assertIn('s3_full_filtered_v1',r.stdout)
        self.assertIn('/mizar_s3_l21_25x3_residual_gate_2epochs_gbs128_5090/formal_',r.stdout)
        self.assertIn('GBS128 epochs2 warmup100',r.stdout)
        self.assertIn('rows_per_epoch=272796 steps_per_epoch=2132 total_steps=4264 tail_rows=28',r.stdout)
        r=subprocess.run([bash,wrapper.as_posix(),'--resume-checkpoint','/old/run/formal/checkpoints/model--step-533.ckpt',
                          '--check-args'],capture_output=True,text=True,check=True)
        self.assertIn('output_root=/old/run',r.stdout)

    def test_real_bash_cpu_admission_once_submission_quoted_args_and_failures(self):
        bash=os.environ.get('MIZAR_TEST_BASH','C:/Program Files/Git/bin/bash.exe')
        with tempfile.TemporaryDirectory(prefix='s3 shell ') as tmp:
            root=Path(tmp);scripts=root/'scripts with spaces';scripts.mkdir()
            wrapper=scripts/'submit.sh';wrapper.write_bytes((ROOT/'code/RSmol/run_mizar_s3_l21_25x3_residual_gate_5090.sh').read_bytes())
            conda=root/'env with spaces';profile=conda/'etc/profile.d';profile.mkdir(parents=True)
            (profile/'conda.sh').write_text('''conda() { printf 'conda:%s\\n' "$*" >> "$MOCK_TRACE"; }
python() {
  printf 'python:%s\\n' "$*" >> "$MOCK_TRACE"
  if [[ "$1" == -u ]]; then shift; fi
  if [[ "$1" == */prepare_index.py ]]; then
    [[ "${MOCK_INDEX_FAIL:-0}" == 0 ]] || return 18
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == --path-output ]]; then printf '%s\\n' "$MOCK_INDEX" > "$2"; break; fi
      shift
    done
  elif [[ "$1" == */admission.py && "${MOCK_ADMISSION_FAIL:-0}" == 1 ]]; then return 19; fi
}
vc() { printf '%s\\0' "$@" > "$MOCK_SUBMIT"; }
''',encoding='utf-8')
            data=root/'data with spaces';output=root/'outer run';index=root/'index cache';model=root/'model with spaces'
            trace=root/'trace';submit=root/'submit'
            env=dict(os.environ,RSMOL_CONDA_BASE=conda.as_posix(),MOCK_TRACE=trace.as_posix(),
                MOCK_INDEX=(root/'indexed metadata').as_posix(),MOCK_SUBMIT=submit.as_posix())
            args=[bash,wrapper.as_posix(),'--data-dir',data.as_posix(),'--model-dir',model.as_posix(),
                  '--output-dir',output.as_posix(),'--index-cache-root',index.as_posix()]
            r=subprocess.run(args+['--check-args'],env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr);self.assertFalse(trace.exists());self.assertFalse(submit.exists())
            r=subprocess.run(args,env=env,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
            submitted=submit.read_bytes().decode().split('\0')[:-1];self.assertEqual(submitted.count('submit'),1)
            for flag,value in (('-p','pdgpu-5090'),('-c','32'),('-m','256G'),('-g','8'),('-n','1')):
                self.assertEqual(submitted[submitted.index(flag)+1],value)
            command=submitted[submitted.index('--cmd')+1]
            parsed=subprocess.run([bash,'-c',"eval 'set -- '\"$1\"; printf '%s\\0' \"$@\"",'_',command],capture_output=True,env=env)
            self.assertEqual(parsed.returncode,0,parsed.stderr)
            self.assertEqual(parsed.stdout.decode().split('\0')[:-1],
                ['bash','mizar_s3_l21_25x3_residual_gate/run.sh',data.as_posix(),output.as_posix(),model.as_posix(),env['MOCK_INDEX'],'4',''])
            order=trace.read_text();self.assertNotIn('prepare_manifest.py',order)
            self.assertLess(order.index('prepare_index.py'),order.index('admission.py'))
            submit.unlink();r=subprocess.run(args+['--preflight-only'],env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr);self.assertFalse(submit.exists())
            for key,code in (('MOCK_INDEX_FAIL',18),('MOCK_ADMISSION_FAIL',19)):
                failed=subprocess.run(args,env=dict(env,**{key:'1'}),capture_output=True,text=True)
                self.assertEqual(failed.returncode,code,failed.stderr);self.assertFalse(submit.exists())

class IsolationTests(unittest.TestCase):
    def test_reference_route_and_old_forwarder_remain_unchanged(self):
        parents=json.loads((PACKAGE/'parent_sources.json').read_text(encoding='utf-8'))['files']
        import hashlib
        for name,expected in parents.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),expected,name)

class LifecycleTests(unittest.TestCase):
    def test_inner_lifecycle_single_producer_stop_reap_before_cleanup(self):
        bash=os.environ.get("MIZAR_TEST_BASH", "C:/Program Files/Git/bin/bash.exe")
        with tempfile.TemporaryDirectory(prefix="mizar lifecycle ") as tmp:
            root=Path(tmp);script=root/"run.sh";script.write_bytes((PACKAGE/"run.sh").read_bytes())
            conda=root/"conda";profile=conda/"etc/profile.d";profile.mkdir(parents=True)
            (profile/"conda.sh").write_text('''conda() { :; }
flock() { :; } # Git Bash lacks Linux flock; lifecycle fixture only.
python() {
  if [[ "$1" == -u && "$2" == stage_data.py ]]; then
    trap - EXIT
    trap 'printf "producer_exit\\n" >> "$MOCK_TRACE"; exit 0' TERM
    printf 'producer_start\\n' >> "$MOCK_TRACE"
    sleep .4
    printf 'producer_exit\\n' >> "$MOCK_TRACE"
  elif [[ "$1" == */stage_data.py && "$2" == --cleanup ]]; then printf 'cleanup\\n' >> "$MOCK_TRACE"
  elif [[ "$1" == write_config.py ]]; then printf 'config\\n' >> "$MOCK_TRACE"; return "${MOCK_CONFIG_EXIT:-0}"
  elif [[ "$1" == */write_failure.py ]]; then printf 'failure:%s\\n' "$3" >> "$MOCK_TRACE"
  fi
}
torchrun() { printf 'torchrun\\n' >> "$MOCK_TRACE"; return "${MOCK_TRAIN_EXIT:-0}"; }
''',encoding="utf-8")
            for name in ("data","model","index"):(root/name).mkdir()
            for mode,exit_code in (("success",0),("train_fail",7),("config_fail",13)):
                trace=root/f"{mode}.trace"
                env=dict(os.environ,RSMOL_CONDA_BASE=conda.as_posix(),MOCK_TRACE=trace.as_posix(),
                         MOCK_TRAIN_EXIT="7" if mode=="train_fail" else "0",
                         MOCK_CONFIG_EXIT="13" if mode=="config_fail" else "0")
                result=subprocess.run([bash,script.as_posix(),(root/"data").as_posix(),(root/mode).as_posix(),
                    (root/"model").as_posix(),(root/"index").as_posix(),"4",""],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,exit_code,result.stderr)
                lines=trace.read_text().splitlines()
                self.assertEqual(lines.count("producer_start"),1)
                self.assertEqual(lines.count("cleanup"),1)
                self.assertLess(lines.index("producer_exit"),lines.index("cleanup"))
                self.assertEqual(lines.count("torchrun"),0 if mode=="config_fail" else 1)
                if exit_code:self.assertIn(f"failure:{exit_code}",lines)


if __name__=='__main__':unittest.main(verbosity=2)
