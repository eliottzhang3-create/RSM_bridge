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


import contract as recipe
from contract import checkpoint_steps, training_cursor, MAX_LR, EPOCHS


class ModelTests(unittest.TestCase):
    def test_native_parity_pure_four_passes_bptt_and_freeze(self):
        for impl in ('eager','sdpa'):
            model=tiny(impl);core=model.caption_decoder.lm.model
            self.assertFalse(any('audio_gate' in n or 'injection' in n for n in model.state_dict()))
            validate_trainability(model,production=False)
            native=LlamaForCausalLM(config(impl)).eval()
            native.load_state_dict(model.caption_decoder.lm.state_dict(),strict=True)
            model.eval();x=torch.randn(1,132,16)
            with torch.no_grad():
                actual=core._execute(x,single_pass=True).last_hidden_state
                expected=native.model(inputs_embeds=x,use_cache=False).last_hidden_state
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            model.train();traces={i:[] for i in range(30)};handles=[]
            def hook(i):
                def capture(module,args,result):
                    h=result[0] if isinstance(result,tuple) else result
                    if h.requires_grad:h.retain_grad()
                    traces[i].append((args[0],h))
                return capture
            for i,l in enumerate(core.layers):handles.append(l.register_forward_hook(hook(i)))
            frozen=checkpoint.sha_frozen(model);b=batch();out=model(b)
            loss=answer_loss(out,b,17);loss.backward()
            for h in handles:h.remove()
            self.assertEqual(core.last_layer_trace,tuple(list(range(8))+list(range(8,12))*4+list(range(12,30))))
            self.assertEqual([len(traces[i]) for i in range(30)],[4 if 8<=i<=11 else 1 for i in range(30)])
            for loop in (1,2,3):
                self.assertIs(traces[8][loop][0],traces[11][loop-1][1])
            self.assertTrue(all(h.grad is not None and h.grad.abs().sum()>0 for _,h in traces[11]))
            self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if parameter_role(n)=='frozen'))
            optimizer,_,_=build_optimizer(model,16890);optimizer.step()
            self.assertEqual(checkpoint.sha_frozen(model),frozen)
            model.eval();changed=x.clone();changed[:,130:]+=1
            with torch.no_grad():
                a=core(inputs_embeds=x).last_hidden_state;b=core(inputs_embeds=changed).last_hidden_state
            torch.testing.assert_close(a[:,:130],b[:,:130],rtol=0,atol=0)
            with self.assertRaises(ValueError):core(inputs_embeds=x,use_cache=True)

    def test_production_meta_four_layers_only(self):
        with torch.device('meta'):
            model=MizarCoreTrain(config(production=True),audio_encoder=TinyAudio(576));model.enable_training()
        roles=collections.Counter()
        for n,p in model.named_parameters():roles[parameter_role(n)]+=p.numel()
        self.assertEqual([roles[f'layer{i}'] for i in (9,10,11,12)],[3540096]*4)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),14160384)
        self.assertIs(model.caption_decoder.lm.lm_head.weight,model.caption_decoder.lm.model.embed_tokens.weight)

    def test_exact_683_layout_and_independent_answer_ce(self):
        model=tiny();b=batch();out=model(b)
        self.assertEqual(out.logits.shape,(1,683,64))
        target=b['answer']['input_ids'][0];valid=b['answer']['attention_mask'][0].bool()&target.ne(17)
        reference=-torch.log_softmax(out.logits[0,382:682],-1)[torch.arange(300)[valid],target[valid]].mean()
        torch.testing.assert_close(answer_loss(out,b,17),reference)


class DataTests(unittest.TestCase):
    def fixture(self,tmp):
        root=Path(tmp);rows=[{'source':'reasonaqa_oneaudio','input':'unused','answer':'unused','filepath1_original':'missing-do-not-stat'}]
        tsv=root/'old.tsv';records=[]
        for i,(strength,view) in enumerate((('strong','audiomcq_mcq'),('weak','audiomcq_mcq'),('strong','audiomcq_cot'),('weak','audiomcq_cot'),('strong','audiomcq_mcq'))):
            audio=root/f'a{i%2}.wav';audio.write_bytes(b'audio')
            rows.append({'id':str(i),'source':'audiomcq','source_dataset':'AudioCaps','audio_path':f'a{i%2}',
                'audio_contribution':strength,'stageE_variant':view,'input':'Original Question','answer':'Original Answer'})
            records.append(f'audio_path\ta{i%2}\texists\t{audio}\n')
        tsv.write_text('field\traw_path\tstatus\tresolved_path\n'+''.join(records))
        source=root/'hybrid.json';source.write_text(json.dumps(rows))
        return source,tsv,rows

    def test_all_strengths_views_no_reason_and_exact_old_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,tsv,rows=self.fixture(tmp);out=Path(tmp)/'out';out.mkdir()
            detail=pm.build(source,tsv,out,expected_previous={'reasonaqa_oneaudio':1,'AudioCaps':5},expected_selected={'AudioCaps':5})
            got=[json.loads(l) for l in (out/'train.jsonl').read_text().splitlines()]
            self.assertEqual(got,rows[1:]);self.assertEqual(detail['omitted_reasonaqa'],1)
            self.assertEqual(detail['audio_contribution_counts'],{'strong':3,'weak':2})
            self.assertEqual(detail['selected_counts'],{'audiomcq_mcq':3,'audiomcq_cot':2})
            self.assertEqual(detail['unique_audio_files'],2)
            second=Path(tmp)/'bad';second.mkdir();(Path(tmp)/'a0.wav').unlink()
            with self.assertRaises(ValueError):pm.build(source,tsv,second,expected_previous={'reasonaqa_oneaudio':1,'AudioCaps':5},expected_selected={'AudioCaps':5})
            self.assertTrue((second/'unresolved_audio.json').exists())

    def test_real_cli_atomic_publication_idempotence_index_and_tamper(self):
        import prepare_index
        with tempfile.TemporaryDirectory() as tmp:
            source,tsv,rows=self.fixture(tmp);out=Path(tmp)/'published'
            args=['--source',str(source),'--resolved-paths',str(tsv),'--output-dir',str(out)]
            with patch.object(pm,'PREVIOUS_SOURCE_COUNTS',{'reasonaqa_oneaudio':1,'AudioCaps':5}),patch.object(pm,'SOURCE_COUNTS',{'AudioCaps':5}),patch.object(recipe,'SOURCE_COUNTS',{'AudioCaps':5}),patch.object(recipe,'TOTAL_ROWS',5):
                pm.main(args);seal=(out/'manifest.json').read_bytes();pm.main(args)
                self.assertEqual(seal,(out/'manifest.json').read_bytes())
                index=prepare_index.ensure_index(out/'train.jsonl',out/'resolved_paths.tsv',Path(tmp)/'cache')
                self.assertEqual(prepare_index.ensure_index(out/'train.jsonl',out/'resolved_paths.tsv',Path(tmp)/'cache'),index)
                self.assertEqual(json.loads((index/'index_report.json').read_text())['rows'],5)
                (out/'train.jsonl').write_text('changed')
                with self.assertRaises(ValueError):pm.verify_dataset(out)

    def test_full_production_three_epoch_shards_quarters_and_resume_cursors(self):
        g=geometry(720524)
        self.assertEqual((g['steps_per_epoch'],g['total_steps'],g['warmup_steps'],g['last_update_rows']),(5630,16890,507,12))
        saves=checkpoint_steps(720524)
        expected=[e*5630+s for e in range(3) for s in (1408,2815,4223,5630)]
        self.assertEqual(saves,expected);self.assertEqual(len(saves),12)
        self.assertAlmostEqual(MAX_LR*lr_factor(506,16890),2e-5)
        self.assertAlmostEqual(MAX_LR*lr_factor(16889,16890),2e-6)
        self.assertEqual(training_cursor(5630,g),(1,0));self.assertEqual(training_cursor(11260,g),(2,0));self.assertEqual(training_cursor(16890,g),(3,0))
        signatures=[]
        for epoch in range(3):
            full=[]
            for rank in range(8):
                sampler=CustomDistributedSampler(range(720524),8,rank,seed=1234);sampler.set_epoch(epoch)
                shard=list(sampler);full.extend(shard);sampler.start_index=5629*16
                self.assertEqual(list(sampler),shard[5629*16:])
                self.assertEqual(len(list(sampler)),2 if rank<4 else 1)
            self.assertEqual(len(full),720524);self.assertEqual(len(set(full)),720524);signatures.append(digest(full[:100]))
        self.assertEqual(len(set(signatures)),3)


class RecoveryTests(unittest.TestCase):
    def test_actual_three_epoch_trainer_and_resume_across_boundary(self):
        import train as trainer
        from contextlib import ExitStack
        class MiniDataset:
            def __init__(self,*args,**kwargs):
                self.order=recipe.data_order(17);self.pad_id=17
                self.audio_paths=AudioPaths(Path.cwd(),'shared_until_complete')
            def __len__(self):return 17
            def original_item(self,index):
                b=batch(micro=index%3)
                return {**b,'audio1':b['audio1'][0],'audio2':b['audio2'][0],
                        'row_index':index,'source':'AudioCaps'}
            def __getitem__(self,index):return self.original_item(self.order[index])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dist.init_process_group('gloo',init_method=(root/'init').as_uri(),rank=0,world_size=1)
            try:
                prepared=root/'prepared.json'
                prepared.write_text(json.dumps({'status':'PASS','metadata':'fixture','stage_root':tmp,
                    'read_policy':'shared_until_complete','probe_rows':{'AudioCaps':[0]}}))
                g=dict(geometry(17),world_size=1,global_batch_size=16,rank_samples=[17],samples_per_rank=17,
                       steps_per_epoch=2,total_steps=6,warmup_steps=0)
                c={'route':recipe.ROUTE,'geometry':g,'code_sha256':code_digest(),'model_dir':tmp,'official_bundle':{}}
                cfg={'contract':c,'contract_sha256':digest(c),'run_dir':str(root/'full'),'prepared_report':str(prepared),
                     'checkpoint_steps':[1,2,3,4,5,6],'max_optimizer_steps':6,'resume_checkpoint':'','attempt_id':'fixture'}
                with ExitStack() as stack:
                    stack.enter_context(patch.object(trainer,'WORLD_SIZE',1))
                    stack.enter_context(patch.object(trainer,'AudioTextDataset',MiniDataset))
                    stack.enter_context(patch.object(trainer,'load_official',lambda path:tiny()))
                    stack.enter_context(patch.object(trainer,'verify_bundle',lambda bundle:None))
                    stack.enter_context(patch.object(trainer,'inspect_tokenizer',lambda *a:{}))
                    stack.enter_context(patch.object(trainer,'validate_trainability',lambda m:validate_trainability(m,production=False)))
                    stack.enter_context(patch.object(trainer,'accumulation_for_update',lambda u,n:4 if u==0 else 1))
                    stack.enter_context(patch.object(trainer,'rank_microbatch_size',lambda n,r,u,m:max(0,min(4,17-u*16-m*4))))
                    actual_save=trainer.save;resume_copy=root/'model--step-2.ckpt'
                    def keep_boundary(path,*args):
                        actual_save(path,*args)
                        if Path(path).name=='model--step-2.ckpt':
                            import shutil;shutil.copyfile(path,resume_copy)
                    stack.enter_context(patch.object(trainer,'save',keep_boundary))
                    Path(cfg['run_dir']).mkdir();trainer.train(cfg,0,torch.device('cpu'))
                    full=torch.load(Path(cfg['run_dir'])/'checkpoints/model--step-6.ckpt',weights_only=False)
                    self.assertEqual(json.loads((Path(cfg['run_dir'])/'training_report.json').read_text())['status'],'PASS')
                    self.assertEqual(len(list((Path(cfg['run_dir'])/'checkpoints').glob('*.ckpt'))),4)
                    resume=root/'resumed';resume.mkdir()
                    lines=(Path(cfg['run_dir'])/'metrics.jsonl').read_text().splitlines()
                    (resume/'metrics.jsonl').write_text('\n'.join(lines[:2])+'\n')
                    resumed_cfg=dict(cfg,run_dir=str(resume),resume_checkpoint=str(resume_copy))
                    trainer.train(resumed_cfg,0,torch.device('cpu'))
                    restored=torch.load(resume/'checkpoints/model--step-6.ckpt',weights_only=False)
                    for n,v in full['state_dict'].items():torch.testing.assert_close(v,restored['state_dict'][n],rtol=0,atol=0)
                    for epoch in (2,3):self.assertEqual(json.loads((resume/f'epoch{epoch}_rank0.json').read_text())['rows_consumed'],17)
            finally:dist.destroy_process_group()

    def test_full_checkpoint_epoch_boundaries_resume_and_retention_four(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist.init_process_group('gloo',init_method=(Path(tmp)/'init').as_uri(),rank=0,world_size=1)
            try:
                model=tiny();g=geometry(129);op,scheduler,_=build_optimizer(model,g['total_steps']);tracker=GradNormTracker()
                scaler=torch.amp.GradScaler('cuda',enabled=False);frozen=checkpoint.sha_frozen(model)
                c={'geometry':dict(g,world_size=1),'code_sha256':code_digest()};cfg={'contract':c,'contract_sha256':digest(c),'max_optimizer_steps':6}
                saved=None
                for step in range(1,7):
                    scalar_update(model,[batch(micro=step%3)],op,tracker);scheduler.step()
                    p=Path(tmp)/f'model--step-{step}.ckpt'
                    checkpoint.save(p,model,op,scheduler,scaler,tracker,cfg,step,frozen,torch.device('cpu'))
                    obj=torch.load(p,weights_only=False)
                    self.assertEqual((obj['epoch'],obj['next_optimizer_step']),divmod(step,2))
                    clean=tiny();cleanop=build_optimizer(clean,6)[0]
                    checkpoint.validate_saved(obj,clean,cleanop,cfg,allow_complete=step==6)
                    if step==2:saved=copy.deepcopy(obj)
                self.assertEqual(sorted(p.name for p in Path(tmp).glob('*.ckpt')),[f'model--step-{s}.ckpt' for s in (3,4,5,6)])
                restored=tiny();op2,sc2,_=build_optimizer(restored,6);tr2=GradNormTracker()
                checkpoint.validate_saved(saved,restored,op2,cfg)
                restored.load_state_dict(saved['state_dict']);op2.load_state_dict(saved['optimizer']);sc2.load_state_dict(saved['scheduler']);tr2.load_state_dict(saved['grad_norm_tracker'])
                for step in range(3,7):scalar_update(restored,[batch(micro=step%3)],op2,tr2);sc2.step()
                for n,v in model.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[n],rtol=0,atol=0)
                bad=copy.deepcopy(saved);bad['epoch']=0
                clean=tiny();cleanop=build_optimizer(clean,6)[0]
                with self.assertRaises(ValueError):checkpoint.validate_saved(bad,clean,cleanop,cfg)
            finally:dist.destroy_process_group()

    def test_real_two_rank_gloo_ga4_and_tail_ga1(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as tmp:
            with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
            mp.spawn(ddp_worker,args=(port,tmp),nprocs=2,join=True)
            actual=torch.load(Path(tmp)/'rank0.pt',weights_only=False)['model']
            other=torch.load(Path(tmp)/'rank1.pt',weights_only=False)['model']
            reference=tiny();op,sc,_=build_optimizer(reference,5);tr=GradNormTracker()
            for count in (4,1):scalar_update(reference,[batch(r,m) for r in range(2) for m in range(count)],op,tr);sc.step()
            for n,v in actual.items():
                torch.testing.assert_close(v,other[n],rtol=0,atol=0)
                torch.testing.assert_close(v,reference.state_dict()[n],rtol=2e-5,atol=2e-7)

    def test_journal_backup_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'metrics.jsonl';old=b'{"step":1}\n{"step":2}\n{"step":3}\n{"step":'
            p.write_bytes(old);recover(p,2)
            self.assertEqual(p.read_bytes(),b'{"step":1}\n{"step":2}\n')
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
