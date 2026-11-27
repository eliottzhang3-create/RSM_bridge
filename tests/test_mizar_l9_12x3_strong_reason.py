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
PACKAGE = ROOT / "code/RSmol/mizar_l9_12x3_strong_reason"
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
from train import optimizer_update
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
    answer = torch.full((1,300),17,dtype=torch.long)
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



class ModelTests(unittest.TestCase):
    def test_native_30_parity_and_zero_initialized_adapters(self):
        for impl in ('eager','sdpa'):
            model=tiny(impl).eval();core=model.caption_decoder.lm.model
            self.assertEqual(core.loop_bias.shape,(3,16));self.assertEqual(core.loop_bias.count_nonzero(),0)
            self.assertEqual(core.audio_gate.weight.count_nonzero(),0);self.assertEqual(core.audio_gate.bias.count_nonzero(),0)
            native=LlamaForCausalLM(config(impl)).eval()
            native.load_state_dict({k:v for k,v in model.caption_decoder.lm.state_dict().items() if not ('audio_gate' in k or 'loop_bias' in k)},strict=True)
            x=torch.randn(1,132,16)
            with torch.no_grad():
                a=core._execute(x,single_pass=True).last_hidden_state
                b=native.model(inputs_embeds=x,use_cache=False).last_hidden_state
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            with self.assertRaises(ValueError):core.enable_adapters()
            with self.assertRaises(ValueError):core(inputs_embeds=x,use_cache=True)
        cold=MizarCoreTrain(config(),audio_encoder=TinyAudio(16));state=torch.get_rng_state().clone()
        cold.enable_training();self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def traces(self,model,x):
        core=model.caption_decoder.lm.model;traces={i:[] for i in range(30)};handles=[];conditions=[]
        def hook(i):
            def capture(module,args,result):
                value=result[0] if isinstance(result,tuple) else result
                traces[i].append((args[0].clone(),value.clone()))
            return capture
        for i,layer in enumerate(core.layers):handles.append(layer.register_forward_hook(hook(i)))
        handles.append(core.audio_gate.register_forward_pre_hook(lambda module,args:conditions.append(args[0].clone())))
        with torch.no_grad():core(inputs_embeds=x)
        for h in handles:h.remove()
        return traces,conditions

    def test_zero_gate_audio_restart_text_carry_and_all_token_bias(self):
        model=tiny().eval();core=model.caption_decoder.lm.model
        with torch.no_grad():
            core.loop_bias.copy_(torch.stack([torch.full((16,),v) for v in (.1,.2,-.15)]))
        traces,conditions=self.traces(model,torch.randn(1,135,16))
        anchor=traces[8][0][1]
        torch.testing.assert_close(traces[9][0][0],anchor+core.loop_bias[0],rtol=0,atol=0)
        for loop in (1,2):
            before=traces[9][loop][0];previous=traces[13][loop-1][1]
            torch.testing.assert_close(before[:,:126],anchor[:,:126]+core.loop_bias[loop],rtol=0,atol=0)
            torch.testing.assert_close(before[:,126:],previous[:,126:]+core.loop_bias[loop],rtol=0,atol=0)
            torch.testing.assert_close(conditions[loop-1],previous[:,:126],rtol=0,atol=0)
        self.assertEqual(core.last_loop_trace,(0,1,2));self.assertEqual(core.last_layer_trace,SCHEDULE)
        self.assertEqual([len(traces[i]) for i in range(30)],[3 if 9<=i<=13 else 1 for i in range(30)])
        torch.testing.assert_close(traces[14][0][0],traces[13][2][1],rtol=0,atol=0)

    def test_nonzero_gate_independent_formula_before_bias_and_shared_two_calls(self):
        model=tiny().eval();core=model.caption_decoder.lm.model
        with torch.no_grad():
            core.audio_gate.weight.copy_(torch.linspace(-.15,.2,16)[None]);core.audio_gate.bias.fill_(.12)
            core.loop_bias.copy_(torch.arange(48).view(3,16)*.001)
        traces,conditions=self.traces(model,torch.randn(1,134,16));anchor=traces[8][0][1][:,:126]
        self.assertEqual(len(conditions),2)
        for loop in (1,2):
            prev=traces[13][loop-1][1];before=traces[9][loop][0]
            # Independent scalar dot product, not the production Linear call.
            weights=((prev[:,:126]*core.audio_gate.weight[0]).sum(-1,keepdim=True)+core.audio_gate.bias[0]).tanh()
            expected=anchor+weights*prev[:,:126]+core.loop_bias[loop]
            torch.testing.assert_close(before[:,:126],expected,rtol=1e-6,atol=1e-7)
            torch.testing.assert_close(before[:,126:],prev[:,126:]+core.loop_bias[loop],rtol=0,atol=0)

    def test_full_bptt_all_bias_rows_l15_and_frozen_suffix(self):
        from train import gradient_audit
        model=tiny();validate_trainability(model,production=False);core=model.caption_decoder.lm.model
        frozen=checkpoint.sha_frozen(model);outputs=[];suffix=[]
        def keep(target):
            def hook(module,args,result):
                value=result[0] if isinstance(result,tuple) else result
                value.retain_grad();target.append(value)
            return hook
        a=core.layers[13].register_forward_hook(keep(outputs));b=core.layers[15].register_forward_hook(keep(suffix))
        inputs=batch();loss=answer_loss(model(inputs),inputs,17);loss.backward();a.remove();b.remove()
        self.assertEqual(len(outputs),3)
        self.assertTrue(all(v.grad is not None and v.grad.abs().sum()>0 for v in outputs+suffix))
        audit=gradient_audit(model);self.assertTrue(all(audit['nonzero_task_gradients'].values()))
        self.assertTrue(all(row.abs().sum()>0 for row in core.loop_bias.grad))
        self.assertGreater(core.layers[14].self_attn.q_proj.weight.grad.abs().sum(),0)
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if parameter_role(n)=='frozen'))
        op,_,_=build_optimizer(model,300);op.step();self.assertEqual(checkpoint.sha_frozen(model),frozen)
        model.eval();x=torch.randn(1,135,16);y=x.clone();y[:,132:]+=2
        with torch.no_grad():
            u=core(inputs_embeds=x).last_hidden_state;v=core(inputs_embeds=y).last_hidden_state
        torch.testing.assert_close(u[:,:132],v[:,:132],rtol=0,atol=0)

    def test_production_meta_six_layers_and_2305_added_parameters(self):
        with torch.device('meta'):
            model=MizarCoreTrain(config(production=True),audio_encoder=TinyAudio(576));model.enable_training()
        roles=collections.Counter()
        for n,p in model.named_parameters():roles[parameter_role(n)]+=p.numel()
        self.assertEqual([roles[f'layer{i}'] for i in range(10,16)],[3540096]*6)
        self.assertEqual((roles['loop_bias'],roles['audio_gate']),(1728,577))
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),21242881)
        self.assertIs(model.caption_decoder.lm.lm_head.weight,model.caption_decoder.lm.model.embed_tokens.weight)

    def test_exact_683_layout_and_independent_answer_ce(self):
        model=tiny();b=batch();out=model(b);self.assertEqual(out.logits.shape,(1,683,64))
        target=b['answer']['input_ids'][0];valid=b['answer']['attention_mask'][0].bool()&target.ne(17)
        expected=-torch.log_softmax(out.logits[0,382:682],-1)[torch.arange(300)[valid],target[valid]].mean()
        torch.testing.assert_close(answer_loss(out,b,17),expected)


class DataTests(unittest.TestCase):
    def fixture(self,tmp):
        root=Path(tmp);rows=[{'source':'reasonaqa_oneaudio','input':'unused','answer':'unused','filepath1_original':'missing-do-not-stat'}]
        tsv=root/'old.tsv';records=[]
        for i,(strength,view) in enumerate((('strong','audiomcq_mcq'),('weak','audiomcq_mcq'),('strong','audiomcq_cot'),('weak','audiomcq_cot'),('strong','audiomcq_mcq'))):
            audio=root/f'a{i%2}.wav';audio.write_bytes(b'audio')
            rows.append({'id':str(i),'source':'audiomcq','source_dataset':'AudioCaps','audio_path':f'a{i%2}',
                'audio_contribution':strength,'stageE_variant':view,'choices':['one','two','three','four'],'gold_letter':'a',
                'input':'Original Question','answer':'Original Answer'})
            records.append(f'audio_path\ta{i%2}\texists\t{audio}\n')
        tsv.write_text('field\traw_path\tstatus\tresolved_path\n'+''.join(records))
        source=root/'hybrid.json';source.write_text(json.dumps(rows));return source,tsv,rows

    def test_only_strong_direct_cot_no_reason_weak_or_new_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,tsv,rows=self.fixture(tmp);out=Path(tmp)/'out';out.mkdir()
            (Path(tmp)/'a1.wav').unlink()  # Weak-only missing recording must not be inspected.
            detail=pm.build(source,tsv,out,expected_previous={'reasonaqa_oneaudio':1,'AudioCaps':5},expected_selected={'AudioCaps':3})
            got=[json.loads(l) for l in (out/'train.jsonl').read_text().splitlines()]
            self.assertEqual(got,[rows[1],rows[3],rows[5]])
            self.assertEqual((detail['omitted_reasonaqa'],detail['omitted_weak']),(1,2))
            self.assertEqual(detail['audio_contribution_counts'],{'strong':3})
            self.assertEqual(detail['selected_counts'],{'audiomcq_mcq':2,'audiomcq_cot':1})
            self.assertEqual(detail['unique_audio_files'],1)
            second=Path(tmp)/'bad';second.mkdir();(Path(tmp)/'a0.wav').unlink()
            with self.assertRaises(ValueError):pm.build(source,tsv,second,expected_previous={'reasonaqa_oneaudio':1,'AudioCaps':5})
            self.assertTrue((second/'unresolved_audio.json').exists())

    def test_bad_history_strength_view_gold_and_missing_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,tsv,rows=self.fixture(tmp)
            for key,value in (('audio_contribution','unknown'),('stageE_variant','unknown'),('gold_letter','z')):
                changed=copy.deepcopy(rows);changed[1][key]=value;source.write_text(json.dumps(changed))
                out=Path(tmp)/key;out.mkdir()
                with self.assertRaises(ValueError):pm.build(source,tsv,out,expected_previous={'reasonaqa_oneaudio':1,'AudioCaps':5})
            source.write_text(json.dumps(rows));out=Path(tmp)/'wrong_history';out.mkdir()
            with self.assertRaises(ValueError):pm.build(source,tsv,out)
            tsv.write_text('field\traw_path\tstatus\tresolved_path\n');out=Path(tmp)/'missing';out.mkdir()
            with self.assertRaises(ValueError):pm.build(source,tsv,out,expected_previous={'reasonaqa_oneaudio':1,'AudioCaps':5})

    def test_real_cli_atomic_publication_idempotence_index_and_tamper(self):
        import prepare_index
        with tempfile.TemporaryDirectory() as tmp:
            source,tsv,rows=self.fixture(tmp);out=Path(tmp)/'published'
            args=['--source',str(source),'--resolved-paths',str(tsv),'--output-dir',str(out)]
            with patch.object(pm,'PREVIOUS_SOURCE_COUNTS',{'reasonaqa_oneaudio':1,'AudioCaps':5}):
                pm.main(args);seal=(out/'manifest.json').read_bytes();pm.main(args)
                self.assertEqual(seal,(out/'manifest.json').read_bytes())
                index=prepare_index.ensure_index(out/'train.jsonl',out/'resolved_paths.tsv',Path(tmp)/'cache')
                self.assertEqual(prepare_index.ensure_index(out/'train.jsonl',out/'resolved_paths.tsv',Path(tmp)/'cache'),index)
                self.assertEqual(json.loads((index/'index_report.json').read_text())['rows'],3)
                (out/'train.jsonl').write_text('changed')
                with self.assertRaises(ValueError):pm.verify_dataset(out)

    def test_arbitrary_one_epoch_no_padding_shards_and_quarter_save_points(self):
        for n in (1,3,17,34,130,325210,325648):
            g=geometry(n);self.assertEqual(g['epochs'],1);self.assertEqual(g['total_steps'],math.ceil(n/128))
            all_rows=[]
            for rank in range(8):
                sampler=CustomDistributedSampler(range(n),8,rank,seed=1234);shard=list(sampler);all_rows.extend(shard)
                sampler.start_index=(g['total_steps']-1)*16
                self.assertEqual(list(sampler),shard[sampler.start_index:])
                count=sum(recipe.rank_microbatch_size(n,rank,u,m) for u in range(g['total_steps']) for m in range(recipe.accumulation_for_update(u,n)))
                self.assertEqual(count,len(shard))
            self.assertEqual(len(all_rows),n);self.assertEqual(len(set(all_rows)),n)
            self.assertEqual(checkpoint_steps(n),sorted({math.ceil(q*g['total_steps']/4) for q in range(1,5)}))
            self.assertEqual(training_cursor(g['total_steps'],g),(1,0))


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
        self.assertAlmostEqual(recipe.group_lr('layer15',299,300)/GROUP_LRS['layer15'][0],.2)


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

    def test_actual_one_epoch_trainer_mid_epoch_resume_matches(self):
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
                g=dict(geometry(81),world_size=1,global_batch_size=16,rank_samples=[81],samples_per_rank=81,steps_per_epoch=6,total_steps=6,warmup_steps=5)
                c={'route':recipe.ROUTE,'geometry':g,'code_sha256':code_digest(),'model_dir':tmp,'official_bundle':{}}
                cfg={'contract':c,'contract_sha256':digest(c),'run_dir':str(root/'full'),'prepared_report':str(prepared),
                     'checkpoint_steps':list(range(1,7)),'max_optimizer_steps':6,'resume_checkpoint':'','attempt_id':'fixture'}
                with ExitStack() as stack:
                    for name,value in (('WORLD_SIZE',1),('AudioTextDataset',MiniDataset),('load_official',lambda p:tiny()),
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
                    final=torch.load(Path(cfg['run_dir'])/'checkpoints/model--step-6.ckpt',weights_only=False)
                    self.assertEqual(json.loads((Path(cfg['run_dir'])/'training_report.json').read_text())['status'],'PASS')
                    self.assertEqual(len(list((Path(cfg['run_dir'])/'checkpoints').glob('*.ckpt'))),4)
                    resumed=root/'resumed';resumed.mkdir()
                    lines=(Path(cfg['run_dir'])/'metrics.jsonl').read_text().splitlines();(resumed/'metrics.jsonl').write_text('\n'.join(lines[:2])+'\n')
                    trainer.train(dict(cfg,run_dir=str(resumed),resume_checkpoint=str(resume_copy)),0,torch.device('cpu'))
                    result=torch.load(resumed/'checkpoints/model--step-6.ckpt',weights_only=False)
                    for n,v in final['state_dict'].items():torch.testing.assert_close(v,result['state_dict'][n],rtol=0,atol=0)
                    self.assertEqual(json.loads((resumed/'epoch1_rank0.json').read_text())['rows_consumed'],49)
            finally:dist.destroy_process_group()

    def test_checkpoint_group_lrs_inventory_moments_and_four_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist.init_process_group('gloo',init_method=(Path(tmp)/'init').as_uri(),rank=0,world_size=1)
            try:
                model=tiny();g=geometry(641);op,sc,_=build_optimizer(model,6);tr=GradNormTracker();scaler=torch.amp.GradScaler('cuda',enabled=False)
                frozen=checkpoint.sha_frozen(model);c={'geometry':dict(g,world_size=1),'code_sha256':code_digest()};cfg={'contract':c,'contract_sha256':digest(c),'max_optimizer_steps':6}
                for step in range(1,7):
                    scalar_update(model,[batch(micro=step%3)],op,tr);sc.step()
                    p=Path(tmp)/f'model--step-{step}.ckpt';checkpoint.save(p,model,op,sc,scaler,tr,cfg,step,frozen,torch.device('cpu'))
                    saved=torch.load(p,weights_only=False);clean=tiny();cleanop=build_optimizer(clean,6)[0]
                    checkpoint.validate_saved(saved,clean,cleanop,cfg,allow_complete=step==6)
                    if step==2:resume=copy.deepcopy(saved)
                self.assertEqual(sorted(p.name for p in Path(tmp).glob('*.ckpt')),[f'model--step-{s}.ckpt' for s in (3,4,5,6)])
                for mutation in (lambda s:s['optimizer']['param_groups'][0].update(min_lr=1e-7),
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

    def test_real_bash_cpu_admission_single_submission_and_argument_quoting(self):
        bash=os.environ.get("MIZAR_TEST_BASH", "C:/Program Files/Git/bin/bash.exe")
        with tempfile.TemporaryDirectory(prefix="mizar shell ") as tmp:
            root=Path(tmp);scripts=root/"scripts with spaces";scripts.mkdir()
            wrapper=scripts/"submit.sh"
            wrapper.write_bytes((ROOT/"code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh").read_bytes())
            conda=root/"env with spaces";profile=conda/"etc/profile.d";profile.mkdir(parents=True)
            (profile/"conda.sh").write_text('''conda() { printf 'conda:%s\\n' "$*" >> "$MOCK_TRACE"; }
python() {
  printf 'python:%s\\n' "$*" >> "$MOCK_TRACE"
  if [[ "$1" == -u && "$2" == */prepare_manifest.py && "${MOCK_PREPARE_FAIL:-0}" == 1 ]]; then return 18; fi
  if [[ "$1" == */prepare_index.py ]]; then
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == --path-output ]]; then printf '%s\\n' "$MOCK_INDEX" > "$2"; break; fi
      shift
    done
  elif [[ "$1" == */admission.py && "${MOCK_FAIL:-0}" == 1 ]]; then return 19; fi
}
vc() { printf '%s\\0' "$@" > "$MOCK_SUBMIT"; }
''',encoding="utf-8")
            data=root/"data with spaces";data.mkdir();(data/"manifest.json").write_text("{}")
            output=root/"outer run";index=root/"index cache";model=root/"model with spaces"
            trace=root/"trace";submit=root/"submit"
            env=dict(os.environ,RSMOL_CONDA_BASE=conda.as_posix(),MOCK_TRACE=trace.as_posix(),
                MOCK_INDEX=(root/"indexed metadata").as_posix(),MOCK_SUBMIT=submit.as_posix())
            args=[bash,wrapper.as_posix(),"--data-dir",data.as_posix(),"--model-dir",model.as_posix(),
                  "--output-dir",output.as_posix(),"--index-cache-root",index.as_posix()]
            check=subprocess.run(args+["--check-args"],env=env,capture_output=True,text=True)
            self.assertEqual(check.returncode,0,check.stderr);self.assertFalse(trace.exists());self.assertFalse(submit.exists())
            checkpoint_path=output/"formal/checkpoints/model--step-500.ckpt"
            resume=subprocess.run([bash,wrapper.as_posix(),"--resume-checkpoint",checkpoint_path.as_posix(),"--check-args"],
                                  env=env,capture_output=True,text=True)
            self.assertEqual(resume.returncode,0,resume.stderr)
            self.assertIn("output_root="+output.as_posix(),resume.stdout)
            result=subprocess.run(args,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            submitted=submit.read_bytes().decode().split("\0")[:-1]
            self.assertEqual(submitted.count("submit"),1)
            for flag,value in (("-p","pdgpu-5090"),("-c","32"),("-m","256G"),("-g","8"),("-n","1")):
                self.assertEqual(submitted[submitted.index(flag)+1],value)
            command=submitted[submitted.index("--cmd")+1]
            parsed=subprocess.run([bash,"-c","eval 'set -- '"+'"$1"'+"; printf '%s\\0' \"$@\"","_",command],capture_output=True,env=env)
            self.assertEqual(parsed.returncode,0,parsed.stderr)
            parsed=parsed.stdout.decode().split("\0")[:-1]
            self.assertEqual(parsed[2:],[data.as_posix(),output.as_posix(),model.as_posix(),env["MOCK_INDEX"],"4",""])
            order=trace.read_text();self.assertLess(order.index("prepare_manifest.py"),order.index("prepare_index.py"));self.assertLess(order.index("prepare_index.py"),order.index("admission.py"))
            submit.unlink();prep=subprocess.run(args+["--prepare-only"],env=env,capture_output=True,text=True)
            self.assertEqual(prep.returncode,0,prep.stderr);self.assertFalse(submit.exists())
            env["MOCK_PREPARE_FAIL"]="1"
            failed=subprocess.run(args,env=env,capture_output=True,text=True)
            self.assertEqual(failed.returncode,18);self.assertFalse(submit.exists())
            env["MOCK_PREPARE_FAIL"]="0";env["MOCK_FAIL"]="1"
            result=subprocess.run(args,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,19);self.assertFalse(submit.exists())

    def test_all_three_entrypoints_bash_syntax(self):
        bash=os.environ.get("MIZAR_TEST_BASH", "C:/Program Files/Git/bin/bash.exe")
        for path in (PACKAGE/"run.sh",ROOT/"code/RSmol/prepare_mizar_l9_12x3_strong_reason.sh",
                     ROOT/"code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh"):
            result=subprocess.run([bash,"-n",path.as_posix()],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=="__main__":unittest.main(verbosity=2)
