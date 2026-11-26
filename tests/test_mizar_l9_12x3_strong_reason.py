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


class ModelTests(unittest.TestCase):
    def test_native_30_control_and_38_schedule_eager_sdpa(self):
        for implementation in ("eager","sdpa"):
            model=tiny(implementation).eval()
            core=model.caption_decoder.lm.model
            native=LlamaForCausalLM(config(implementation)).eval()
            native.load_state_dict({k:v for k,v in model.caption_decoder.lm.state_dict().items()
                                    if "audio_gate" not in k},strict=True)
            inputs=torch.randn(1,130,16)
            with torch.no_grad():
                actual=core._execute(inputs,single_pass=True).last_hidden_state
                expected=native.model(inputs_embeds=inputs,use_cache=False).last_hidden_state
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                core(inputs_embeds=inputs)
            self.assertEqual(core.last_layer_trace,tuple(list(range(8))+list(range(8,12))*3+list(range(12,30))))
            self.assertEqual(core.last_injection_trace,(12,16))
            with self.assertRaises(ValueError):core(inputs_embeds=inputs,use_cache=True)
            with self.assertRaises(ValueError):core.enable_gate()

    def test_injection_previous_output_and_post8_anchor_only_audio(self):
        model=tiny().eval();core=model.caption_decoder.lm.model
        with torch.no_grad():
            core.audio_gate.weight.copy_(torch.linspace(-.1,.1,16)[None])
            core.audio_gate.bias.fill_(.2)
        traces={i:[] for i in range(30)}
        handles=[]
        def hook(i):
            def save(module,args,result):
                value=result[0] if isinstance(result,tuple) else result
                traces[i].append((args[0].detach().clone(),value.detach().clone()))
            return save
        for i,layer in enumerate(core.layers):handles.append(layer.register_forward_hook(hook(i)))
        with torch.no_grad():core(inputs_embeds=torch.randn(1,132,16))
        for handle in handles:handle.remove()
        anchor=traces[7][0][1][:,:126]
        for loop in (1,2):
            previous=traces[11][loop-1][1]
            before=traces[8][loop][0]
            expected=previous[:,:126]+torch.tanh(nn.functional.linear(previous[:,:126],core.audio_gate.weight,core.audio_gate.bias))*anchor
            torch.testing.assert_close(before[:,:126],expected,rtol=0,atol=0)
            torch.testing.assert_close(before[:,126:],previous[:,126:],rtol=0,atol=0)
        self.assertEqual([len(traces[i]) for i in range(30)],[3 if 8<=i<=11 else 1 for i in range(30)])

    def test_full_bptt_zero_gate_freeze_eval_and_causality(self):
        model=tiny();core=model.caption_decoder.lm.model
        self.assertEqual(core.audio_gate.weight.count_nonzero().item(),0)
        self.assertEqual(core.audio_gate.bias.count_nonzero().item(),0)
        validate_trainability(model,production=False)
        frozen={k:v.clone() for k,v in model.state_dict().items() if parameter_role(k)=="frozen"}
        activations=[]
        def retain(module,args,result):
            h=result[0] if isinstance(result,tuple) else result
            h.retain_grad();activations.append(h)
        handle=core.layers[11].register_forward_hook(retain)
        optimizer,_,_=build_optimizer(model,5)
        b=batch();loss=answer_loss(model(b),b,17);loss.backward();handle.remove()
        self.assertEqual(len(activations),3)
        self.assertTrue(all(h.grad is not None and h.grad.abs().sum()>0 for h in activations))
        self.assertGreater(core.audio_gate.weight.grad.abs().sum(),0)
        self.assertGreater(core.layers[8].self_attn.q_proj.weight.grad.abs().sum(),0)
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if parameter_role(n)=="frozen"))
        optimizer.step()
        self.assertTrue(all(torch.equal(model.state_dict()[k],v) for k,v in frozen.items()))
        model.eval()
        x=torch.randn(1,132,16);changed=x.clone();changed[:,130:]+=10
        with torch.no_grad():
            a=core(inputs_embeds=x).last_hidden_state
            z=core(inputs_embeds=changed).last_hidden_state
        torch.testing.assert_close(a[:,:130],z[:,:130],rtol=0,atol=0)

    def test_real_683_layout_and_independent_shifted_ce(self):
        model=tiny();b=batch();out=model(b)
        self.assertEqual(out.logits.shape,(1,683,64))
        logits=out.logits[0];targets=b["answer"]["input_ids"][0]
        valid=b["answer"]["attention_mask"][0].bool() & targets.ne(17)
        reference=-torch.log_softmax(logits[382:682],-1)[torch.arange(300)[valid],targets[valid]].mean()
        torch.testing.assert_close(answer_loss(out,b,17),reference)
        b["answer"]["attention_mask"].zero_()
        with self.assertRaises(ValueError):answer_loss(out,b,17)

    def test_production_lm_meta_inventory_and_shared_parameters(self):
        with torch.device("meta"):
            model=MizarCoreTrain(config(production=True),audio_encoder=TinyAudio(576))
            model.enable_training()
        roles=collections.Counter()
        for n,p in model.named_parameters():roles[parameter_role(n)]+=p.numel()
        self.assertEqual(roles["gate"],577)
        self.assertEqual([roles[f"layer{i}"] for i in (9,10,11,12)],[3540096]*4)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),14160961)
        self.assertIs(model.caption_decoder.lm.lm_head.weight,model.caption_decoder.lm.model.embed_tokens.weight)


class DataTests(unittest.TestCase):
    def fixture(self, directory):
        directory=Path(directory);paths=[]
        for i in range(3):
            p=directory/f"recording-{i}.wav";p.write_bytes(bytes([i+1])*10);paths.append(str(p))
        rows=[]
        for i in range(8):
            rows.append({"id":str(i),"source":"reasonaqa_oneaudio","filepath1_original":f"reason{i%3}",
                         "filepath2":"historical_second_chunk_is_not_second_recording",
                         "input":"Question","answer":"Answer"})
        for i,view in enumerate(("audiomcq_mcq","audiomcq_mcq","audiomcq_cot")):
            rows.append({"id":f"s{i}","source":"audiomcq","source_dataset":"AudioCaps",
                "audio_path":"strong_alias","audio_contribution":"strong","stageE_variant":view,
                "choices":["a","b","c","d"],"gold_letter":"a","input":"MCQ","answer":"a"})
        source=directory/"s1.json";source.write_text(json.dumps(rows),encoding="utf-8")
        tsv=directory/"old.tsv"
        tsv.write_text("field\traw_path\tstatus\tresolved_path\n"+"".join(
            f"filepath1_original\treason{i}\texists\t{paths[i]}\n" for i in range(3))+
            f"audio_path\tstrong_alias\texists\t{paths[0]}\n",encoding="utf-8")
        return source,tsv,rows

    def test_manifest_roundrobin_aliases_preserved_views_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,tsv,rows=self.fixture(tmp)
            resolver=pm.Resolver(tsv,Path(tmp)/"unused.json")
            quotas={"reason_rows":5,"strong_counts":{"audiomcq_mcq":2,"audiomcq_cot":1},"release_rows":11,"reason_pool":8}
            selected,report=pm.select(source,resolver.resolve,**quotas,progress=False)
            again,_=pm.select(source,resolver.resolve,**quotas,progress=False)
            self.assertEqual(selected,again);self.assertEqual(len(selected),8)
            self.assertTrue({8,9,10}<=selected)
            counts=collections.Counter(resolver.resolve(rows[i]) for i in selected if i<8)
            self.assertEqual(len(counts),3);self.assertLessEqual(max(counts.values())-min(counts.values()),1)
            out=Path(tmp)/"out";out.mkdir()
            tokenizer=lambda texts,**kwargs:{"input_ids":[[1,0] for _ in texts]}
            total,details=pm.build(source,out,resolver,tokenizer,quotas=quotas)
            self.assertEqual(total,8);self.assertEqual(details["unique_audio_files"],3)
            written=[json.loads(line) for line in (out/"train.jsonl").read_text().splitlines()]
            self.assertEqual(written,[rows[i] for i in sorted(selected)])
            self.assertEqual(details["selected_counts"],{"reasonaqa":5,"audiomcq_mcq":2,"audiomcq_cot":1})
            seal={"format":"mizar_l9_12x3_strong_reason_dataset_v1","status":"PASS","rows":8,
                  "files":pm.file_records(out,("train.jsonl","selection.jsonl","resolved_paths.tsv","dataset_report.json"))}
            seal["seal_sha256"]=digest(seal);(out/"manifest.json").write_text(json.dumps(seal))
            pm.verify_dataset(out,total_rows=8)
            (out/"train.jsonl").write_text("tampered")
            with self.assertRaises(ValueError):pm.verify_dataset(out,total_rows=8)

    def test_manifest_rejects_bad_universe_gold_and_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,tsv,rows=self.fixture(tmp);resolver=pm.Resolver(tsv,Path(tmp)/"unused")
            with self.assertRaises(ValueError):pm.select(source,resolver.resolve,release_rows=12,progress=False)
            bad=dict(rows[-1],gold_letter="x")
            with self.assertRaises(ValueError):pm.row_kind(bad)
            bad=dict(rows[0],filepath2_original="second_original")
            self.assertIsNone(pm.row_kind(bad))
            row=dict(rows[-1],audio_path="missing")
            resolver._fallback=lambda *args:{"status":"missing","path":""}
            resolver.resolve(row);self.assertTrue(resolver.errors)
            rows[-1]=row;source.write_text(json.dumps(rows))
            out=Path(tmp)/"failed";out.mkdir()
            with self.assertRaises(ValueError):pm.build(source,out,resolver,lambda texts,**k:{"input_ids":[[1,0] for _ in texts]},
                quotas={"reason_rows":5,"strong_counts":{"audiomcq_mcq":2,"audiomcq_cot":1},"release_rows":11,"reason_pool":8})
            self.assertTrue((out/"unresolved_audio.json").exists())

    def test_json_gzip_and_no_replace_publication(self):
        from manifest_io import iter_rows
        from publication import atomic_publish
        with tempfile.TemporaryDirectory() as tmp:
            source,_,rows=self.fixture(tmp)
            gz=Path(tmp)/"s1.json.gz"
            with gzip.open(gz,"wt",encoding="utf-8") as handle:json.dump(rows,handle)
            self.assertEqual(list(iter_rows(gz)),rows)
            stage=Path(tmp)/"stage";stage.mkdir();target=Path(tmp)/"published"
            atomic_publish(stage,target)
            other=Path(tmp)/"other";other.mkdir()
            with self.assertRaises(OSError):atomic_publish(other,target)
            self.assertTrue(other.is_dir());self.assertTrue(target.is_dir())

    def test_real_manifest_cli_seal_idempotence_and_immutable_index(self):
        import prepare_index
        import data.prepared_index as index_module
        with tempfile.TemporaryDirectory() as tmp:
            source,tsv,_=self.fixture(tmp);root=Path(tmp);mapping=root/"mapping.json";mapping.write_text("{}")
            out=root/"published";counts={"reasonaqa_oneaudio":5,"AudioCaps":3}
            quotas={"reason_rows":5,"strong_counts":{"audiomcq_mcq":2,"audiomcq_cot":1},"release_rows":11,"reason_pool":8}
            real_build,real_verify=pm.build,pm.verify_dataset
            def build_fixture(source,out,resolver,tokenizer):return real_build(source,out,resolver,tokenizer,quotas=quotas)
            def verify_fixture(root):return real_verify(root,total_rows=8)
            args=["--source",str(source),"--output-dir",str(out),"--resolved-paths",str(tsv),
                  "--mapping-json",str(mapping),"--model-dir",os.environ["MIZAR_LOCAL_BUNDLE"]]
            with patch.object(pm,"SOURCE_SHA256",{False:sha256_file(source)}),patch.object(pm,"TOTAL_ROWS",8), \
                    patch.object(pm,"SOURCE_COUNTS",counts),patch.object(pm,"build",build_fixture), \
                    patch.object(pm,"verify_dataset",verify_fixture),patch.object(sd,"SOURCE_COUNTS",counts), \
                    patch.object(index_module,"SOURCE_COUNTS",counts):
                pm.main(args)
                sealed=(out/"manifest.json").read_bytes()
                with patch.object(pm,"build",side_effect=AssertionError("should reuse")):pm.main(args)
                self.assertEqual((out/"manifest.json").read_bytes(),sealed)
                self.assertFalse((root/"published.prepare.lock").exists())
                path=prepare_index.ensure_index(out/"train.jsonl",out/"resolved_paths.tsv",root/"cache")
                current=index_module.load_index(path,out/"train.jsonl",out/"resolved_paths.tsv")
                self.assertEqual((current["rows"],current["unique_raw_files"]),(8,3))
                self.assertEqual(prepare_index.ensure_index(out/"train.jsonl",out/"resolved_paths.tsv",root/"cache"),path)
                (path/"files.json").write_text("{}")
                with self.assertRaises(ValueError):index_module.load_index(path)
                mapping.write_text('{"changed":true}')
                with self.assertRaises(ValueError):pm.main(args)

    def test_full_epoch_eight_shards_tail_and_lr(self):
        shards=[]
        for rank in range(8):
            sampler=CustomDistributedSampler(range(TOTAL_ROWS),8,rank,seed=1234,drop_last=False)
            shard=list(sampler);self.assertEqual(len(shard),81412)
            sampler.start_index=5088*16
            self.assertEqual(list(sampler),shard[-4:]);shards.extend(shard)
        self.assertEqual(len(shards),651296);self.assertEqual(len(set(shards)),651296)
        self.assertEqual(sum(SOURCE_COUNTS.values()),651296)
        g=geometry(TOTAL_ROWS)
        self.assertEqual((g["total_steps"],g["full_updates"],g["last_update_rows"],g["warmup_steps"]),(5089,5088,32,254))
        self.assertEqual([accumulation_for_update(i) for i in (0,5087,5088)],[4,4,1])
        self.assertEqual(sum(accumulation_for_update(i)*32 for i in range(5089)),651296)
        self.assertAlmostEqual(lr_factor(5088,5089)*1e-4,1e-5)
        self.assertAlmostEqual(lr_factor(253,5089)*1e-4,1e-4)
        self.assertTrue(all(lr_factor(i,5089)>=lr_factor(i+1,5089) for i in range(254,5088)))
        with self.assertRaises(ValueError):geometry(651295)

    def test_actual_tokenizer_original_text_semantics_and_soundfile_fixed20(self):
        import soundfile as sf
        import numpy as np
        bundle=Path(os.environ["MIZAR_LOCAL_BUNDLE"])
        tokenizer=AutoTokenizer.from_pretrained(bundle/"assets/smollm2-135m",local_files_only=True)
        tokenizer.add_special_tokens({"pad_token":"!"})
        self.assertEqual((len(tokenizer),tokenizer.pad_token_id,tokenizer.eos_token_id),(49152,17,0))
        instance=AudioTextDataset.__new__(AudioTextDataset);instance.tokenizer=tokenizer
        prompt,answer=text_pair({"input":"Hello","answer":"A"})
        self.assertEqual((prompt,answer),("hello","a"))
        encoded=instance.tokenize(answer,300)
        expected=tokenizer.encode("a <|endoftext|>")
        self.assertEqual(encoded["input_ids"][0,:len(expected)].tolist(),expected)
        self.assertTrue(encoded["input_ids"][0,len(expected):].eq(17).all())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"long.wav"
            values=np.linspace(-.8,.8,22*32000,dtype=np.float32)
            sf.write(path,np.column_stack((values,values)),32000,subtype="FLOAT")
            a,b=load_fixed20(path)
            torch.testing.assert_close(torch.cat((a,b)),torch.from_numpy(values[32000:-32000]),rtol=0,atol=0)
            sf.write(path,np.ones(16000,dtype=np.float32)*.2,16000,subtype="FLOAT")
            a,b=load_fixed20(path)
            self.assertEqual(a.numel(),320000);self.assertEqual(b.count_nonzero(),0)
            self.assertTrue(torch.isfinite(a).all());self.assertEqual(a[32000:].count_nonzero(),0)


class RecoveryTests(unittest.TestCase):
    def test_journal_backup_partial_and_complete_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"metrics.jsonl"
            original=b'{"step":1}\n{"step":2}\n{"step":3}\n{"step":'
            p.write_bytes(original);recover(p,2)
            self.assertEqual(p.read_bytes(),b'{"step":1}\n{"step":2}\n')
            self.assertEqual(next(Path(tmp).glob("*.before-resume-*")).read_bytes(),original)
            with self.assertRaises(ValueError):recover(p,3)
            p.write_bytes(b'{"step":1}\n{"step":1}\n')
            with self.assertRaises(ValueError):recover(p,1)
            p.write_bytes(b'{"step":1}\nBROKEN\n')
            with self.assertRaises(ValueError):recover(p,1)

    def test_checkpoint_resume_next_update_identity_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            init=Path(tmp)/"init"
            dist.init_process_group("gloo",init_method=init.as_uri(),rank=0,world_size=1)
            try:
                model=tiny();optimizer,scheduler,_=build_optimizer(model,5);tracker=GradNormTracker()
                scaler=torch.amp.GradScaler("cuda",enabled=False)
                frozen=checkpoint.sha_frozen(model)
                for _ in range(2):scalar_update(model,[batch()],optimizer,tracker);scheduler.step()
                contract={"code_sha256":code_digest(),"geometry":{"total_steps":5,"world_size":1}}
                cfg={"contract":contract,"contract_sha256":digest(contract),"max_optimizer_steps":5}
                path=Path(tmp)/"model--step-2.ckpt"
                checkpoint.save(path,model,optimizer,scheduler,scaler,tracker,cfg,2,frozen,torch.device("cpu"))
                saved=torch.load(path,weights_only=False)
                restored=tiny();op2,sc2,_=build_optimizer(restored,5);tr2=GradNormTracker()
                self.assertEqual(checkpoint.validate_saved(saved,restored,op2,cfg),2)
                restored.load_state_dict(saved["state_dict"]);op2.load_state_dict(saved["optimizer"])
                sc2.load_state_dict(saved["scheduler"]);tr2.load_state_dict(saved["grad_norm_tracker"])
                for m,o,s,t in ((model,optimizer,scheduler,tracker),(restored,op2,sc2,tr2)):
                    scalar_update(m,[batch(micro=1)],o,t);s.step()
                for n,v in model.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[n],rtol=0,atol=0)
                for mutate in (lambda s:s["optimizer"]["param_groups"][0]["params"].reverse(),
                    lambda s:s["scheduler"].update(_step_count=1),
                    lambda s:s["optimizer"]["state"][0]["exp_avg"].fill_(float("nan")),
                    lambda s:s.update(next_optimizer_step=1),
                    lambda s:s["state_dict"]["audio_encoder.fixed"].add_(1)):
                    bad=copy.deepcopy(saved);mutate(bad)
                    clean=tiny();clean_optimizer=build_optimizer(clean,5)[0]
                    with self.assertRaises(ValueError):checkpoint.validate_saved(bad,clean,clean_optimizer,cfg)
                # A failed freeze audit must publish no complete checkpoint.
                model.audio_encoder.fixed.add_(1)
                with self.assertRaises(RuntimeError):checkpoint.save(Path(tmp)/"model--step-3.ckpt",model,optimizer,scheduler,scaler,tracker,cfg,3,frozen,torch.device("cpu"))
                self.assertFalse((Path(tmp)/"model--step-3.ckpt").exists())
            finally:dist.destroy_process_group()

    def test_real_two_rank_gloo_ga4_and_tail_ga1_match_reference(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as tmp:
            with socket.socket() as sock:sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
            mp.spawn(ddp_worker,args=(port,tmp),nprocs=2,join=True)
            a=torch.load(Path(tmp)/"rank0.pt",weights_only=False)
            b=torch.load(Path(tmp)/"rank1.pt",weights_only=False)
            for n,v in a["model"].items():torch.testing.assert_close(v,b["model"][n],rtol=0,atol=0)
            reference=tiny();op,sched,_=build_optimizer(reference,5);tracker=GradNormTracker()
            for count in (4,1):
                scalar_update(reference,[batch(rank,m) for rank in range(2) for m in range(count)],op,tracker)
                sched.step()
            for n,v in reference.state_dict().items():torch.testing.assert_close(v,a["model"][n],rtol=2e-5,atol=2e-7)

    def test_rng_cpu_roundtrip(self):
        random.seed(12);torch.manual_seed(13)
        state=rng_state(torch.device("cpu"));a=(random.random(),torch.randn(3))
        restore_rng(state,torch.device("cpu"));b=(random.random(),torch.randn(3))
        self.assertEqual(a[0],b[0]);self.assertTrue(torch.equal(a[1],b[1]))


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
            order=trace.read_text();self.assertLess(order.index("prepare_index.py"),order.index("admission.py"))
            submit.unlink();env["MOCK_FAIL"]="1"
            result=subprocess.run(args,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,19);self.assertFalse(submit.exists())

    def test_all_three_entrypoints_bash_syntax(self):
        bash=os.environ.get("MIZAR_TEST_BASH", "C:/Program Files/Git/bin/bash.exe")
        for path in (PACKAGE/"run.sh",ROOT/"code/RSmol/prepare_mizar_l9_12x3_strong_reason.sh",
                     ROOT/"code/RSmol/run_mizar_l9_12x3_strong_reason_5090.sh"):
            result=subprocess.run([bash,"-n",path.as_posix()],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=="__main__":unittest.main(verbosity=2)
