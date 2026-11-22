"""Real CPU tests for raw3600 initialization, BP5 gradients and the new recipe."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/RSmol'))
from text_hrm_reinit3600_bp5 import contract as C
ML = all(importlib.util.find_spec(n) for n in ('torch','transformers','safetensors','pyarrow','tokenizers','numpy'))
if ML:
    import numpy as np
    import torch
    from test_text_hrm_h2l3 import hrm_model, tokenizer
    from text_hrm_h2l3 import contract as OLD
    from text_hrm_h2l3.train import build_binding as original_binding
    from text_hrm_h2l3.checkpoint import save_checkpoint as original_save
    from text_hrm_h2l3.model import load_model
    from text_hrm_reinit3600_bp5.initialization import prepare
    from text_hrm_reinit3600_bp5.artifact import inspect_artifact
    from text_hrm_reinit3600_bp5.train import (build_binding, fresh_model_optimizer,
        make_optimizer, make_stream, data_manifest, accumulate_window, causal_loss_sum)
    from text_hrm_reinit3600_bp5.gradients import gradient_audit, prelude_parameters
    from text_hrm_reinit3600_bp5.checkpoint import (save_checkpoint,restore_checkpoint,
        capture_rng,restore_rng,validate_optimizer,optimizer_names,retain_checkpoints)
    torch.set_num_threads(1)


def binding():
    return build_binding(initialization={'source_step':3600,'weights_kind':'raw'},
        data={'seed':1},environment={'fixture':True},production=False)


def tick(model,opt,step):
    opt.zero_grad(set_to_none=True)
    opt.param_groups[0]['lr']=C.learning_rate(step)
    ids=torch.tensor([[3,4,5,6],[6,5,4,3]])
    model(input_ids=ids,labels=ids,bp_steps=C.bp_steps(step)).loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)


def ranks(step):
    return {str(rank):dict(rng=capture_rng(),cursor=dict(version=1,rank=rank,
        microbatches=64*step,epoch=1,shard=0,in_shard=64*step,
        shard_start_rng=random.Random(1+1009*rank).getstate())) for rank in range(8)}


def original_checkpoint(root,step=3600):
    _,model=hrm_model(); opt=make_optimizer(model)
    tick(model,opt,1)
    for p in model.parameters(): opt.state[p]['step'].fill_(step)
    opt.param_groups[0]['lr']=OLD.learning_rate(step)
    old_binding=original_binding(initialization={'fixture':True},data={'fixture':True},environment={'fixture':True},production=False)
    source=original_save(root,model,opt,tokenizer(),step=step,binding=old_binding,
        rank_states=ranks(step),production=False,retain=False)
    return source,model


class ContractTests(unittest.TestCase):
    def test_all_9244_lr_bp_positions_and_endpoints(self):
        self.assertEqual(round(9244*.05),462)
        values=[]
        for s in range(1,9245):
            expected=3e-4+3e-4*(s-1)/461 if s<=462 else 6e-5+2.7e-4*(1+math.cos(math.pi*(s-462)/8782))
            self.assertAlmostEqual(C.learning_rate(s),expected,places=18)
            self.assertEqual(C.bp_steps(s),5)
            values.append(C.learning_rate(s))
        self.assertEqual(values[0],3e-4); self.assertEqual(values[461],6e-4); self.assertEqual(values[-1],6e-5)
        self.assertTrue(all(a<b for a,b in zip(values[:461],values[1:462])))
        self.assertTrue(all(a>b for a,b in zip(values[461:-1],values[462:])))
        for bad in (0,9245,True,1.5):
            with self.assertRaises(ValueError): C.learning_rate(bad)
            with self.assertRaises(ValueError): C.bp_steps(bad)

    def test_isolated_recipe_and_existing_sources_unchanged(self):
        from text_hrm_h2l3.contract import RECIPE as old
        self.assertEqual(old['lr'],3e-4); self.assertEqual(old['bp_min'],3); self.assertEqual(old['seed'],0)
        self.assertEqual(C.RECIPE['data_seed'],1); self.assertEqual(C.RECIPE['global_batch_size'],1024)
        guard=json.loads((C.HERE/'protected_sources.json').read_text(encoding='utf-8'))
        for name,expected in guard.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),expected,name)

    def test_python310_and_dry_run_without_assets(self):
        for path in list(C.HERE.glob('*.py'))+[Path(__file__),ROOT/'tests/validate_text_hrm_reinit3600_bp5.py']:
            ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,10))
        r=subprocess.run([sys.executable,'-B','-m','text_hrm_reinit3600_bp5.train','--output-dir','/no/assets',
            '--dry-run'],cwd=ROOT/'code/RSmol',check=True,capture_output=True,text=True)
        obj=json.loads(r.stdout)
        self.assertFalse(obj['assets_read']); self.assertFalse(obj['gpu_initialized'])
        self.assertEqual(obj['lr_probes']['1'],3e-4); self.assertEqual(obj['lr_probes']['9244'],6e-5)
        self.assertEqual(set(obj['bp_probes'].values()),{5})


@unittest.skipUnless(ML,'requires real CPU tensor/Parquet dependencies')
class TensorTests(unittest.TestCase):
    def test_actual_formal_startup_and_first_64_micro_update_cpu_harness(self):
        """Real production orchestration/64 autograd calls; CUDA/collectives simulated."""
        from contextlib import ExitStack, nullcontext
        from types import SimpleNamespace
        import text_hrm_reinit3600_bp5.train as trainer
        from text_hrm_reinit3600_bp5 import artifact
        import pyarrow as pa
        import pyarrow.parquet as pq
        from text_hrm_h2l3.data import expected_parquet_names
        real_loader=fresh_model_optimizer; real_inspect=inspect_artifact
        cpu=torch.device('cpu'); real_device=torch.device
        class LocalDDP:
            def __init__(self,model,**kwargs): self.module=model
            def __call__(self,**kwargs): return self.module(**kwargs)
            def no_sync(self): return nullcontext()
        def all_gather(output,value):
            for i in range(len(output)): output[i]=copy.deepcopy(value)
        def all_reduce(value,op=None):
            if op is None: value.mul_(8)
        with tempfile.TemporaryDirectory(prefix='hrm3600_startup_') as tmp:
            root=Path(tmp); source,_=original_checkpoint(root)
            snapshot=root/'snapshot';prepare(source,snapshot,production=False,threads=1)
            data=root/'data';data.mkdir()
            for name in expected_parquet_names():
                pq.write_table(pa.table({'text':['w1 w2 w3 w4']*64}),data/name)
            args=SimpleNamespace(init_snapshot=snapshot,data_dir=data,output_dir=root/'run',resume_from=None)
            calls=iter([0.,1.])
            def monotonic():
                try: return next(calls)
                except StopIteration: raise RuntimeError('TEST_STOP_AFTER_FIRST_REAL_UPDATE')
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ,{'LOCAL_RANK':'0'}))
                stack.enter_context(patch.object(trainer,'time',SimpleNamespace(monotonic=monotonic)))
                stack.enter_context(patch.object(trainer,'fresh_model_optimizer',side_effect=lambda path,device:real_loader(path,device=cpu,production=False)))
                stack.enter_context(patch.object(artifact,'inspect_artifact',side_effect=lambda path,**kw:real_inspect(path,production=False,**kw)))
                stack.enter_context(patch('torch.device',side_effect=lambda kind,*a:cpu if kind=='cuda' else real_device(kind,*a)))
                stack.enter_context(patch('torch.nn.parallel.DistributedDataParallel',LocalDDP))
                for name in ('set_device','manual_seed_all','synchronize','set_rng_state'):
                    stack.enter_context(patch('torch.cuda.'+name))
                for name in ('is_available','is_bf16_supported'):
                    stack.enter_context(patch('torch.cuda.'+name,return_value=True))
                stack.enter_context(patch('torch.cuda.get_rng_state',side_effect=torch.get_rng_state))
                for name in ('memory_allocated','memory_reserved','max_memory_allocated'):
                    stack.enter_context(patch('torch.cuda.'+name,return_value=0))
                for name in ('init_process_group','broadcast_object_list','destroy_process_group'):
                    stack.enter_context(patch('torch.distributed.'+name))
                stack.enter_context(patch('torch.distributed.is_initialized',return_value=True))
                stack.enter_context(patch('torch.distributed.get_world_size',return_value=8))
                stack.enter_context(patch('torch.distributed.get_rank',return_value=0))
                stack.enter_context(patch('torch.distributed.all_gather_object',side_effect=all_gather))
                stack.enter_context(patch('torch.distributed.all_reduce',side_effect=all_reduce))
                with self.assertRaisesRegex(RuntimeError,'TEST_STOP_AFTER_FIRST_REAL_UPDATE'): trainer.run(args)
            startup=json.loads((root/'run/startup_rank0.json').read_text())
            self.assertEqual(startup['start_step'],0);self.assertEqual(startup['next_lr'],3e-4)
            self.assertTrue(json.loads((root/'run/gradient_bp5_rank0.json').read_text())['passed'])
            self.assertTrue(json.loads((root/'run/prelude_full_update_rank0.json').read_text())['passed'])
            metric=json.loads((root/'run/metrics.jsonl').read_text())
            self.assertEqual(metric['step'],1);self.assertEqual(metric['lr_used'],3e-4)
            self.assertEqual(metric['bp_steps'],5);self.assertEqual(metric['global_valid_tokens'],64*2*3*8)
            self.assertFalse((root/'run/training_report.json').exists())
            self.assertEqual(json.loads((root/'run/failure_rank0.json').read_text())['completed_steps'],1)

    def test_raw3600_snapshot_fresh_optimizer_and_trained_gate_exact(self):
        with tempfile.TemporaryDirectory(prefix='hrm3600_init_') as tmp:
            root=Path(tmp); source,model=original_checkpoint(root)
            before=OLD.file_inventory(source,include_state=True)
            output=root/'snapshot'
            with patch('torch.load',side_effect=AssertionError('old training_state must not be deserialized')):
                prepare(source,output,production=False,threads=1)
                loaded,opt,tok=fresh_model_optimizer(output,device=torch.device('cpu'),production=False)
            for name,value in model.state_dict().items(): self.assertTrue(torch.equal(value,loaded.state_dict()[name]),name)
            self.assertGreater(float(loaded.model.layers[5].self_attn.gate_proj.weight.detach().abs().sum()),0)
            for p in loaded.parameters():
                state=opt.state[p]
                self.assertEqual(float(state['step']),0)
                self.assertFalse(bool(state['exp_avg'].any())); self.assertFalse(bool(state['exp_avg_sq'].any()))
                self.assertTrue(torch.equal(p,state['param_ema']))
            self.assertFalse((output/'training_state.pt').exists()); self.assertFalse((output/'ema').exists())
            self.assertEqual(before,OLD.file_inventory(source,include_state=True))
            inspect_artifact(output,production=False)
            # The pinned snapshot stays usable after the original run's retention removes source.
            shutil.rmtree(source)
            self.assertTrue(prepare(source,output,production=False)['reused'])
            fresh_model_optimizer(output,device=torch.device('cpu'),production=False)
            with self.assertRaises(ValueError): inspect_artifact(output,resume=True,production=False)

    def test_snapshot_reject_wrong_step_corruption_or_wrong_existing_source(self):
        with tempfile.TemporaryDirectory(prefix='hrm3600_bad_') as tmp:
            root=Path(tmp); wrong,_=original_checkpoint(root,3200)
            with self.assertRaises(ValueError): prepare(wrong,root/'bad',production=False,threads=1)
            self.assertFalse((root/'bad').exists())
            source,_=original_checkpoint(root,3600)
            output=root/'snapshot'; prepare(source,output,production=False,threads=1)
            with self.assertRaises(ValueError): prepare(root/'other',output,production=False)
            with (output/'model.safetensors').open('ab') as f: f.write(b'bad')
            with self.assertRaises(ValueError): prepare(source,output,production=False)

    def test_bp5_each_prelude_layer_task_gradient_and_fault_rejection(self):
        _,model=hrm_model(); opt=make_optimizer(model)
        # Start with learned nonzero gates, then audit a fresh task backward.
        tick(model,opt,1); model.model.audit_gradients=True
        ids=torch.tensor([[3,4,5,6],[6,5,4,3]])
        model(input_ids=ids,labels=ids,bp_steps=5).loss.backward()
        report=gradient_audit(model,5)
        self.assertTrue(report['passed'])
        self.assertEqual(set(report['prelude_per_layer']['layers']),set('01234'))
        self.assertEqual({n for n,on in report['calls'] if on},{'H1','H2','L4','L5','L6'})
        parameter=next(model.model.layers[2].parameters()); parameter.grad=None
        self.assertFalse(prelude_parameters(model)['passed'])
        model.model.audit_refs.clear()

    def test_seed1_manifest_stream_reproducibility_and_exact_cursor_replay(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from text_hrm_h2l3.data import expected_parquet_names, assign_shards, ParquetStream
        with tempfile.TemporaryDirectory(prefix='hrm3600_data_') as tmp:
            root=Path(tmp)
            for name in expected_parquet_names():
                pq.write_table(pa.table({'text':[f'w{i} w{(i+1)%80} w2' for i in range(64)]}),root/name)
            manifest=data_manifest(root)
            files=[root/name for name in expected_parquet_names()]
            self.assertNotEqual(manifest['assignment'],assign_shards(files,world_size=8,seed=0))
            tok=tokenizer(); stream=make_stream(manifest,tok,rank=0); it=iter(stream)
            first=[next(it)['input_ids'] for _ in range(17)]
            cursor=copy.deepcopy(stream.cursor()); expected=[next(it)['input_ids'] for _ in range(5)]
            replay=make_stream(manifest,tok,rank=0,cursor=cursor); ri=iter(replay)
            self.assertTrue(all(torch.equal(x,next(ri)['input_ids']) for x in expected))
            again=iter(make_stream(manifest,tok,rank=0))
            self.assertTrue(all(torch.equal(x,next(again)['input_ids']) for x in first))
            old=iter(ParquetStream(manifest['assignment']['rank_shards']['0'],tok,rank=0,seed=0))
            self.assertTrue(any(not torch.equal(x,next(old)['input_ids']) for x in first))
            it.close();ri.close();again.close();old.close()

    def test_new_full_resume_ema_rng_and_next_update_at_lr_boundary(self):
        with tempfile.TemporaryDirectory(prefix='hrm3600_resume_') as tmp:
            root=Path(tmp); _,model=hrm_model(); opt=make_optimizer(model)
            # The moments are real from one update; set counters to test the LR boundary.
            tick(model,opt,461)
            for p in model.parameters(): opt.state[p]['step'].fill_(461)
            state=ranks(461); b=binding()
            ckpt=save_checkpoint(root,model,opt,tokenizer(),step=461,binding=b,rank_states=state,production=False,retain=False)
            loaded=load_model(ckpt,production=False); opt2=make_optimizer(loaded)
            step,cursor=restore_checkpoint(ckpt,loaded,opt2,binding=b,rank=3,production=False)
            self.assertEqual(step,461); self.assertEqual(cursor['rank'],3)
            self.assertTrue(torch.equal(torch.get_rng_state(),state['3']['rng']['cpu']))
            for update in (462,463):
                tick(model,opt,update);tick(loaded,opt2,update)
            for name,value in model.state_dict().items(): self.assertTrue(torch.equal(value,loaded.state_dict()[name]),name)
            for p,q in zip(model.parameters(),loaded.parameters()):
                for key in opt.state[p]: self.assertTrue(torch.equal(opt.state[p][key],opt2.state[q][key]))
            ema=load_model(ckpt/'ema',production=False)
            self.assertTrue(torch.equal(ema.model.zL_init,model.model.zL_init))
            for change in ({'recipe':C.RECIPE|{'data_seed':0}},{'code':{}},{'environment':{'wrong':True}}):
                with self.assertRaises(ValueError): restore_checkpoint(ckpt,loaded,opt2,binding=b|change,rank=0,production=False)
            with self.assertRaises(ValueError): inspect_artifact(ckpt,production=False)

    def test_retention_and_cross_route_checkpoint_rejection(self):
        with tempfile.TemporaryDirectory(prefix='hrm3600_keep_') as tmp:
            root=Path(tmp); _,model=hrm_model(); opt=make_optimizer(model)
            for step in range(1,5):
                tick(model,opt,step)
                latest=save_checkpoint(root,model,opt,tokenizer(),step=step,binding=binding(),rank_states=ranks(step),production=False)
            self.assertEqual(sorted(p.name for p in root.glob('checkpoint-step-*')),[f'checkpoint-step-{i:06d}' for i in (2,3,4)])
            from text_hrm_h2l3.conversion import inspect_artifact as old_inspect
            with self.assertRaises(ValueError): old_inspect(latest,resume=True,production=False)
            with (latest/'training_state.pt').open('ab') as f: f.write(b'bad')
            with self.assertRaises(ValueError): retain_checkpoints(root)

    def test_optimizer_resume_rejects_old_lr_and_serialized_parameter_reorder(self):
        _,model=hrm_model(); opt=make_optimizer(model);tick(model,opt,1)
        for fault in ('old_lr','reorder','negative_moment','ema_missing','wrong_step'):
            state=copy.deepcopy(opt.state_dict()); names=optimizer_names(model,opt)
            key=state['param_groups'][0]['params'][0]
            if fault=='old_lr': state['param_groups'][0]['lr']=OLD.learning_rate(1)
            if fault=='reorder': state['param_groups'][0]['params'][:2]=reversed(state['param_groups'][0]['params'][:2])
            if fault=='negative_moment': state['state'][key]['exp_avg_sq'].flatten()[0]=-1
            if fault=='ema_missing': del state['state'][key]['param_ema']
            if fault=='wrong_step': state['state'][key]['step'].fill_(3601)
            with self.assertRaises(ValueError): validate_optimizer(model,opt,state,names,1)

    def test_real_two_rank_ddp_bp5_per_layer_audit(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory(prefix='hrm3600_ddp_') as tmp:
            root=Path(tmp)
            mp.spawn(ddp_worker,args=((root/'rendezvous').as_uri(),str(root)),nprocs=2,join=True)
            for rank in range(2):
                report=json.loads((root/f'rank{rank}.json').read_text())
                self.assertTrue(report['passed']); self.assertTrue(report['prelude']['passed'])


class ShellTests(unittest.TestCase):
    def test_bash_and_real_launcher_prepare_vc_runtime_chain(self):
        bash='C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else shutil.which('bash')
        if not bash or not Path(bash).exists(): self.skipTest('Bash unavailable')
        for name in ('scripts/prepare_text_hrm_reinit3600_bp5.sh','scripts/train_text_hrm_reinit3600_bp5_ddp.sh','run_text_hrm_reinit3600_bp5_3090.sh'):
            subprocess.run([bash,'-n',str(ROOT/'code/RSmol'/name)],check=True,capture_output=True)
        with tempfile.TemporaryDirectory(prefix='hrm3600 shell ') as tmp:
            root=Path(tmp); bindir=root/'bin'; bindir.mkdir()
            conda=root/'conda base';(conda/'etc/profile.d').mkdir(parents=True)
            (conda/'etc/profile.d/conda.sh').write_text('conda() { return 0; }\n',encoding='utf-8',newline='\n')
            (bindir/'python').write_text('#!/bin/bash\nif [[ "$*" == *text_hrm_reinit3600_bp5.initialization* ]]; then\n'
                'printf "%s\\n" "$@" > "$HRM_TEST_PREPARE"\nexit "${HRM_TEST_PREPARE_FAIL:-0}"\nfi\nexec '
                +shlex.quote(Path(sys.executable).as_posix())+' "$@"\n',encoding='utf-8',newline='\n')
            (bindir/'vc').write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$HRM_TEST_VC"\n',encoding='utf-8',newline='\n')
            (bindir/'torchrun').write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$HRM_TEST_TORCHRUN"\n'
                'while [[ "$1" != --module ]]; do shift; done\nshift\nexec python -B -m "$@" --dry-run\n',encoding='utf-8',newline='\n')
            for path in bindir.iterdir(): path.chmod(0o755)
            env=os.environ.copy();env.update(PATH=str(bindir)+os.pathsep+env['PATH'],MSYS_NO_PATHCONV='1',MSYS2_ARG_CONV_EXCL='*',
                RSMOL_HRM3600_CONDA_BASE=conda.as_posix(),HRM_TEST_PREPARE=str(root/'prepare.txt'),
                HRM_TEST_VC=str(root/'vc.txt'),HRM_TEST_TORCHRUN=str(root/'torchrun.txt'),RSMOL_HRM_RESUME_FROM='/wrong/old/resume')
            launcher=ROOT/'code/RSmol/run_text_hrm_reinit3600_bp5_3090.sh'
            args=[bash,str(launcher),'--source-checkpoint','/source with spaces/checkpoint-step-003600',
                '--init-snapshot','/init with spaces','--data-dir','/data with spaces','--output-dir','/output with spaces']
            result=subprocess.run(args,env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            prepare_args=(root/'prepare.txt').read_text().splitlines()
            self.assertEqual(prepare_args[prepare_args.index('--source-checkpoint')+1],'/source with spaces/checkpoint-step-003600')
            vc_args=(root/'vc.txt').read_text().splitlines()
            for key,value in (('-p','pdgpu-3090'),('-c','64'),('-m','256G'),('-g','8'),('-n','1')):
                self.assertEqual(vc_args[vc_args.index(key)+1],value)
            self.assertLess(len(vc_args[vc_args.index('-j')+1]),60)
            command=vc_args[vc_args.index('--cmd')+1]
            runtime=subprocess.run([bash,'-c',command],env=env,text=True,capture_output=True)
            self.assertEqual(runtime.returncode,0,runtime.stderr)
            data=json.loads(runtime.stdout)
            self.assertEqual(Path(data['init_snapshot']),Path('/init with spaces'))
            self.assertIsNone(data['resume_from']);self.assertEqual(data['recipe']['data_seed'],1)
            self.assertIn('--nproc_per_node=8',(root/'torchrun.txt').read_text())
            (root/'vc.txt').unlink();(root/'prepare.txt').unlink()
            result=subprocess.run(args+['--dry-run'],env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse((root/'vc.txt').exists());self.assertFalse((root/'prepare.txt').exists())
            result=subprocess.run(args,env=env|{'HRM_TEST_PREPARE_FAIL':'27'},text=True,capture_output=True)
            self.assertEqual(result.returncode,27);self.assertFalse((root/'vc.txt').exists())
            (root/'prepare.txt').unlink()
            result=subprocess.run(args+['--resume-from','/new route/checkpoint-step-000400'],env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse((root/'prepare.txt').exists())
            vc_args=(root/'vc.txt').read_text().splitlines()
            runtime=subprocess.run([bash,'-c',vc_args[vc_args.index('--cmd')+1]],env=env,text=True,capture_output=True)
            self.assertEqual(runtime.returncode,0,runtime.stderr)
            self.assertEqual(Path(json.loads(runtime.stdout)['resume_from']),Path('/new route/checkpoint-step-000400'))


def ddp_worker(rank,init_method,output):
    import datetime
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from text_hrm_h2l3.data import collate_dynamic_padding
    dist.init_process_group('gloo',init_method=init_method,world_size=2,rank=rank,timeout=datetime.timedelta(seconds=90))
    try:
        _,model=hrm_model(); reference=copy.deepcopy(model)
        ddp=DistributedDataParallel(model,broadcast_buffers=False,find_unused_parameters=False)
        windows=[[collate_dynamic_padding([[3,4,5],[6,7]],pad_token_id=0),collate_dynamic_padding([[8,9],[10,11,12]],pad_token_id=0)],
                 [collate_dynamic_padding([[13,14,15,16],[17,18]],pad_token_id=0),collate_dynamic_padding([[19,20,21],[22,23,24,25]],pad_token_id=0)]]
        audits=[]
        loss,count=accumulate_window(ddp,model,windows[rank],device=torch.device('cpu'),world_size=2,bp=5,
            audit_callback=lambda m,k:audits.append(gradient_audit(m,k)))
        for window in windows:
            for batch in window:
                logits=reference(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],bp_steps=5).logits
                (causal_loss_sum(logits,batch['labels'])/count).backward()
        for p,q in zip(model.parameters(),reference.parameters()):
            if p.grad is None or not torch.allclose(p.grad,q.grad,atol=5e-6,rtol=5e-4): raise AssertionError('global gradient mismatch')
        if not audits[0]['passed']: raise AssertionError('first micro Prelude/BP5 audit failed')
        (Path(output)/f'rank{rank}.json').write_text(json.dumps(dict(passed=True,prelude=prelude_parameters(model))))
    finally:
        dist.destroy_process_group()


if __name__=='__main__': unittest.main(verbosity=2)
