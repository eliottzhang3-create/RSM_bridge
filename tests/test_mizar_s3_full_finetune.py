"""Real CPU Llama/autograd/audio/Gloo; explicit tiny fixtures, no production GPU claim.

Run from tests/ so the repository's code/ does not shadow Python stdlib code.
MIZAR_TEST_PYTHON_AUDIO=1 disables only the unused torchaudio binary extension
in this test process when using the existing Torch2.15/torchaudio2.9 CPU runtime.
"""
from __future__ import annotations
import ast, copy, importlib.util, json, math, os, socket, subprocess, sys, tempfile, threading, time
from pathlib import Path
from contextlib import ExitStack
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'code/RSmol/mizar_s3_full_finetune'
sys.path.insert(0,str(PACKAGE))
import torch
torch.set_num_threads(1)
if os.environ.get('MIZAR_TEST_PYTHON_AUDIO')=='1':
    _find_spec=importlib.util.find_spec
    def _python_audio(name,*args,**kwargs):
        return None if name=='torchaudio.lib._torchaudio' else _find_spec(name,*args,**kwargs)
    with patch('importlib.util.find_spec',_python_audio):import torchaudio
import torch.distributed as dist
from torch import nn
from transformers import LlamaConfig,LlamaForCausalLM,AutoTokenizer
import contract as recipe
from contract import geometry,code_digest,digest,parameter_role,checkpoint_steps,lr_factor
from model import configure_training,validate_trainability,inherit_state
from models.mellow import Mellow,AudioEncoder
from models.decoder import DecoderModel
from train import optimizer_update,source_probe,gradient_audit
from optim import build_optimizer,GradNormTracker
from training_helpers import answer_loss
from data.sampler import CustomDistributedSampler
from data.audio_paths import AudioPaths
from data.staging import StagingGate
from data.fixed20 import load_fixed20
from data.audiotext_dataset import AudioTextDataset,collate_fn,text_pair
import checkpoint
import stage_data as sd

TOKENIZER=Path(os.environ.get('MIZAR_TEST_MODEL','C:/Xlance/assets/Mizar/Mizar159M'))/'assets/smollm2-135m'

class TinyBase(nn.Module):
    def __init__(self):
        super().__init__();self.ced=nn.Linear(1,1)
        self.mapper=nn.Sequential(nn.Linear(1,16),nn.Dropout(.05))
    def train(self,mode=True):
        super().train(mode);self.ced.eval();return self
    def forward(self,waveform):
        with torch.no_grad():grid=self.ced(waveform[:,:126,None])
        return {'embedding':self.mapper(grid),'clipwise_output':None}

def tiny(implementation='sdpa'):
    torch.manual_seed(29)
    c=LlamaConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=30,
        num_attention_heads=4,num_key_value_heads=2,vocab_size=64,
        tie_word_embeddings=True,pad_token_id=17,eos_token_id=0,
        attention_dropout=0.,rope_theta=100000.,use_cache=False)
    c._attn_implementation=implementation
    model=Mellow.__new__(Mellow);nn.Module.__init__(model)
    model.audio_encoder=AudioEncoder.__new__(AudioEncoder);nn.Module.__init__(model.audio_encoder)
    model.audio_encoder.base=TinyBase();model.audio_encoder.projection=nn.Identity()
    model.caption_decoder=DecoderModel.__new__(DecoderModel);nn.Module.__init__(model.caption_decoder)
    model.caption_decoder.prefix_length=383;model.caption_decoder.lm_embedding_size=16
    model.caption_decoder.lm=LlamaForCausalLM(c);model.caption_decoder.last_layout=None
    return configure_training(model)

def batch(rank=0,micro=0,size=1):
    gen=torch.Generator().manual_seed(700+rank*10+micro)
    answer=torch.full((size,96),17,dtype=torch.long)
    for i in range(size):
        count=3+(rank+micro+i)%5
        answer[i,:count]=torch.randint(1,16,(count,),generator=gen);answer[i,count-1]=0
    return {'audio1':torch.randn(size,126,generator=gen),'audio2':torch.zeros(size,2),
        'input':{'input_ids':torch.randint(1,64,(size,256),generator=gen)},
        'answer':{'input_ids':answer,'attention_mask':answer.ne(17).long()}}

def scalar_update(model,batches,optimizer,tracker):
    optimizer.zero_grad(set_to_none=True)
    (sum(answer_loss(model(b),b,17) for b in batches)/len(batches)).backward()
    tracker.track_and_clip_(model.named_parameters());optimizer.step()

def ddp_worker(rank,port,directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2)
    try:
        model=tiny();model.audio_encoder.base.mapper[1].p=0.
        ddp=nn.parallel.DistributedDataParallel(model,broadcast_buffers=False)
        op,sc,_=build_optimizer(model,5);tr=GradNormTracker();scaler=torch.amp.GradScaler('cuda',enabled=False)
        for count in (4,3):
            optimizer_update(ddp,[batch(rank,m,size=2 if rank==0 else 1) for m in range(count)],
                torch.device('cpu'),op,scaler,tr,17);sc.step()
        torch.save({'model':model.state_dict(),'optimizer':op.state_dict()},Path(directory)/f'rank{rank}.pt')
    finally:dist.destroy_process_group()

class ModelTests(unittest.TestCase):
    def test_native30_causal_parity_full_bptt_and_roles(self):
        for impl in ('eager','sdpa'):
            model=tiny(impl);counts=validate_trainability(model,production=False)
            self.assertEqual(set(counts),{'lm','mapper','frozen'})
            state=copy.deepcopy(model.state_dict());inherit_state(model,state,production=False)
            bad=copy.deepcopy(state);bad['caption_decoder.lm.lm_head.weight']=bad['caption_decoder.lm.lm_head.weight'].clone()+1
            with self.assertRaises(ValueError):inherit_state(model,bad,production=False)
            model.eval();inputs=batch()
            with torch.no_grad():audit=source_probe(model,inputs,17)
            self.assertEqual(audit['future_answer_causality'],'PASS')
            native=LlamaForCausalLM(copy.deepcopy(model.caption_decoder.lm.config)).eval()
            native.load_state_dict(model.caption_decoder.lm.state_dict())
            with torch.no_grad():
                prefix,_,_=model.generate_prefix_inference(inputs)
                actual=model.caption_decoder.lm(inputs_embeds=prefix,attention_mask=None,use_cache=False).logits
                expected=native(inputs_embeds=prefix,attention_mask=None,use_cache=False).logits
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            with self.assertRaises(ValueError):model.caption_decoder.lm(inputs_embeds=prefix,use_cache=True)
            model.train();frozen=checkpoint.sha_frozen(model);out=model(inputs)
            self.assertEqual(out.logits.shape,(1,479,64))
            loss=answer_loss(out,inputs,17)
            target=inputs['answer']['input_ids'][0];valid=target.ne(17)
            reference=-torch.log_softmax(out.logits[0,382:478],-1)[torch.arange(96)[valid],target[valid]].mean()
            torch.testing.assert_close(loss,reference)
            loss.backward();self.assertTrue(all(gradient_audit(model)['all_30_layers_nonzero'].values()))
            op,_,groups=build_optimizer(model,1066)
            self.assertEqual(sum(len(g['params']) for g in op.param_groups),len({id(p) for p in model.parameters() if p.requires_grad}))
            self.assertTrue(all(g['base_lr']==1e-5 for g in groups))
            self.assertTrue(model.caption_decoder.lm.get_input_embeddings().weight.grad.abs().sum()>0)
            before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad};op.step()
            self.assertEqual(checkpoint.sha_frozen(model),frozen)
            for role in ('mapper','lm'):
                self.assertTrue(any(not torch.equal(p,before[n]) for n,p in model.named_parameters() if parameter_role(n)==role))

class DataTests(unittest.TestCase):
    def test_full_pool_half_budget_no_duplicate_resume_tail_and_lr(self):
        n=272796;g=geometry(n)
        self.assertEqual((g['target_rows'],g['total_steps'],g['warmup_steps'],g['last_update_rows']),(136398,1066,32,78))
        self.assertEqual(checkpoint_steps(n),[267,533,800,1066])
        shards=[]
        for rank in range(8):
            sampler=CustomDistributedSampler(range(n),8,rank,seed=1234);shard=list(sampler);shards+=shard
            sampler.start_index=1065*16
            self.assertEqual(list(sampler),shard[1065*16:])
            self.assertEqual(len(sampler),10 if rank<6 else 9)
            self.assertEqual([recipe.rank_microbatch_size(n,rank,1065,m) for m in range(3)],
                             [4,4,2 if rank<6 else 1])
        expected=torch.randperm(n,generator=torch.Generator().manual_seed(1234)).tolist()[:136398]
        self.assertEqual(shards,expected);self.assertEqual(len(set(shards)),136398)
        self.assertEqual(recipe.accumulation_for_update(1065,n),3)
        self.assertAlmostEqual(1e-5*lr_factor(31,1066),1e-5)
        self.assertAlmostEqual(1e-5*lr_factor(1065,1066),1e-6)
        self.assertEqual(recipe.training_cursor(1066,g),(0,1066))
        with self.assertRaises(ValueError):recipe.validate_data_counts(136398,recipe.SOURCE_COUNTS)

    def test_real_audio_resample_then_mono_first20_pad_and_text_eos(self):
        import numpy as np,soundfile as sf,torchaudio
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'stereo.wav';wave=np.stack((np.arange(336000)%100/200,np.zeros(336000)),axis=1).astype('float32')
            sf.write(p,wave,16000,subtype='FLOAT');first,second=load_fixed20(p)
            expected=torchaudio.transforms.Resample(16000,32000)(torch.from_numpy(wave.T)).mean(0)[:640000]
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

class RecoveryTests(unittest.TestCase):
    def test_actual_trainer_half_epoch_and_midrun_resume_bitwise(self):
        import train as trainer
        class MiniDataset:
            def __init__(self,*args,**kwargs):
                self.order=list(range(106));self.pad_id=17;self.audio_paths=AudioPaths(Path.cwd(),'shared_until_complete')
            def __len__(self):return 106
            def original_item(self,index):
                b=batch(micro=index%3)
                return dict(b,audio1=b['audio1'][0],audio2=b['audio2'][0],row_index=index,source='AudioCaps')
            def __getitem__(self,index):return self.original_item(index)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dist.init_process_group('gloo',init_method=(root/'init').as_uri(),rank=0,world_size=1)
            try:
                p=root/'prepared.json';p.write_text(json.dumps({'status':'PASS','metadata':'fixture','stage_root':tmp,
                    'read_policy':'shared_until_complete','probe_rows':{'AudioCaps':[0]}}))
                g=dict(geometry(106),world_size=1,global_batch_size=16,rank_samples=[53],steps_per_epoch=4,total_steps=4,
                       full_updates=3,last_update_rows=5,warmup_steps=0)
                c={'route':recipe.ROUTE,'geometry':g,'code_sha256':code_digest(),'model_dir':tmp,'official_bundle':{},'checkpoint_steps':[1,2,3,4]}
                cfg={'contract':c,'contract_sha256':digest(c),'run_dir':str(root/'full'),'prepared_report':str(p),
                     'checkpoint_steps':[1,2,3,4],'max_optimizer_steps':4,'resume_checkpoint':'','attempt_id':'fixture'}
                with ExitStack() as stack:
                    for name,value in {'WORLD_SIZE':1,'geometry':lambda n:g,'AudioTextDataset':MiniDataset,
                        'load_official':lambda path:tiny(),'verify_bundle':lambda b:None,'inspect_tokenizer':lambda *a:{},
                        'validate_trainability':lambda m:validate_trainability(m,production=False),
                        'accumulation_for_update':lambda u,n:min(4,math.ceil((53-u*16)/4)),
                        'rank_microbatch_size':lambda n,r,u,m:max(0,min(4,53-u*16-m*4))}.items():
                        stack.enter_context(patch.object(trainer,name,value))
                    Path(cfg['run_dir']).mkdir();trainer.train(cfg,0,torch.device('cpu'))
                    full=torch.load(root/'full/checkpoints/model--step-4.ckpt',weights_only=False)
                    lines=(root/'full/metrics.jsonl').read_text().splitlines()
                    for step in (1,3):
                        resumed=root/f'resumed{step}';resumed.mkdir()
                        (resumed/'metrics.jsonl').write_text('\n'.join(lines[:step])+ '\n')
                        trainer.train(dict(cfg,run_dir=str(resumed),resume_checkpoint=str(root/f'full/checkpoints/model--step-{step}.ckpt')),0,torch.device('cpu'))
                        actual=torch.load(resumed/'checkpoints/model--step-4.ckpt',weights_only=False)
                        for n,v in full['state_dict'].items():torch.testing.assert_close(v,actual['state_dict'][n],rtol=0,atol=0)
                        for key,values in full['optimizer']['state'].items():
                            for name,v in values.items():torch.testing.assert_close(v,actual['optimizer']['state'][key][name],rtol=0,atol=0)
                        self.assertEqual(actual['scheduler'],full['scheduler'])
                        self.assertEqual(actual['grad_norm_tracker'],full['grad_norm_tracker'])
                        torch.testing.assert_close(actual['random_state_by_rank'][0]['torch'],full['random_state_by_rank'][0]['torch'],rtol=0,atol=0)
                        self.assertEqual(json.loads((resumed/'training_report.json').read_text())['consumed_rows'],53)
            finally:dist.destroy_process_group()

    def test_checkpoint_complete_retention_and_reject_bad_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dist.init_process_group('gloo',init_method=(root/'init').as_uri(),rank=0,world_size=1)
            try:
                model=tiny();op,sc,_=build_optimizer(model,5);tr=GradNormTracker()
                scaler=torch.amp.GradScaler('cuda',enabled=False);frozen=checkpoint.sha_frozen(model)
                c={'geometry':dict(geometry(1026),world_size=1,total_steps=5),'code_sha256':code_digest()};cfg={'contract':c,'contract_sha256':digest(c)}
                for step in range(1,6):
                    scalar_update(model,[batch(micro=step)],op,tr);sc.step()
                    path=root/f'model--step-{step}.ckpt';checkpoint.save(path,model,op,sc,scaler,tr,cfg,step,frozen,torch.device('cpu'))
                    saved=torch.load(path,weights_only=False);clean=tiny();cleanop=build_optimizer(clean,5)[0]
                    checkpoint.validate_saved(saved,clean,cleanop,cfg,allow_complete=step==5)
                    if step==2:snapshot=copy.deepcopy(saved)
                self.assertEqual(len(list(root.glob('*.ckpt'))),4)
                with self.assertRaises(ValueError):checkpoint.validate_saved(saved,clean,cleanop,cfg)
                def frozen_drift(s):s['state_dict']['audio_encoder.base.ced.weight'].add_(1)
                def tied_conflict(s):s['state_dict']['caption_decoder.lm.lm_head.weight']=s['state_dict']['caption_decoder.lm.lm_head.weight'].clone()+1
                for change in (lambda s:s.update(route_variant='other'),lambda s:s.update(next_optimizer_step=3),
                    lambda s:s['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),frozen_drift,tied_conflict):
                    bad=copy.deepcopy(snapshot);change(bad)
                    with self.assertRaises(ValueError):checkpoint.validate_saved(bad,clean,cleanop,cfg)
            finally:dist.destroy_process_group()

    def test_two_real_gloo_ranks_ga4_and_ga3_uneven_sizes(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as tmp:
            with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
            mp.spawn(ddp_worker,args=(port,tmp),nprocs=2,join=True)
            a=torch.load(Path(tmp)/'rank0.pt',weights_only=False)['model'];b=torch.load(Path(tmp)/'rank1.pt',weights_only=False)['model']
            ref=tiny();ref.audio_encoder.base.mapper[1].p=0.;op,sc,_=build_optimizer(ref,5);tr=GradNormTracker()
            for count in (4,3):
                scalar_update(ref,[batch(r,m,size=2 if r==0 else 1) for m in range(count) for r in range(2)],op,tr);sc.step()
            for n,v in a.items():
                torch.testing.assert_close(v,b[n],rtol=0,atol=0)
                torch.testing.assert_close(v,ref.state_dict()[n],rtol=2e-5,atol=2e-7)

    def test_metrics_recovery_preserves_prior_log(self):
        from journal import recover
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'metrics.jsonl';old=b'{"step":1}\n{"step":2}\n{"step":3}\n{"step":'
            p.write_bytes(old);recover(p,2)
            self.assertEqual(p.read_bytes(),b'{"step":1}\n{"step":2}\n')
            self.assertEqual(next(Path(tmp).glob('*.before-resume-*')).read_bytes(),old)

class ShellTests(unittest.TestCase):
    def test_default_preview_and_resume_output_root(self):
        bash=os.environ.get('MIZAR_TEST_BASH','C:/Program Files/Git/bin/bash.exe')
        wrapper=ROOT/'code/RSmol/run_mizar_s3_full_finetune_3090.sh'
        r=subprocess.run([bash,wrapper.as_posix(),'--check-args'],capture_output=True,text=True,check=True)
        self.assertIn('queue=pdgpu-3090 cpus=64 memory=256G gpus=8 nodes=1',r.stdout)
        self.assertIn('s3_full_filtered_v1',r.stdout);self.assertIn('consumed_rows=136398 updates=1066 tail_rows=78',r.stdout)
        r=subprocess.run([bash,wrapper.as_posix(),'--resume-checkpoint','/old/run/formal/checkpoints/model--step-533.ckpt',
                          '--check-args'],capture_output=True,text=True,check=True)
        self.assertIn('output_root=/old/run',r.stdout)

    def test_real_bash_cpu_admission_once_submission_quoted_args_and_failures(self):
        bash=os.environ.get('MIZAR_TEST_BASH','C:/Program Files/Git/bin/bash.exe')
        with tempfile.TemporaryDirectory(prefix='s3 shell ') as tmp:
            root=Path(tmp);scripts=root/'scripts with spaces';scripts.mkdir()
            wrapper=scripts/'submit.sh';wrapper.write_bytes((ROOT/'code/RSmol/run_mizar_s3_full_finetune_3090.sh').read_bytes())
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
            for flag,value in (('-p','pdgpu-3090'),('-c','64'),('-m','256G'),('-g','8'),('-n','1')):
                self.assertEqual(submitted[submitted.index(flag)+1],value)
            command=submitted[submitted.index('--cmd')+1]
            parsed=subprocess.run([bash,'-c',"eval 'set -- '\"$1\"; printf '%s\\0' \"$@\"",'_',command],capture_output=True,env=env)
            self.assertEqual(parsed.returncode,0,parsed.stderr)
            self.assertEqual(parsed.stdout.decode().split('\0')[:-1],
                ['bash','mizar_s3_full_finetune/run.sh',data.as_posix(),output.as_posix(),model.as_posix(),env['MOCK_INDEX'],'4',''])
            order=trace.read_text();self.assertNotIn('prepare_manifest.py',order)
            self.assertLess(order.index('prepare_index.py'),order.index('admission.py'))
            submit.unlink();r=subprocess.run(args+['--preflight-only'],env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr);self.assertFalse(submit.exists())
            for key,code in (('MOCK_INDEX_FAIL',18),('MOCK_ADMISSION_FAIL',19)):
                failed=subprocess.run(args,env=dict(env,**{key:'1'}),capture_output=True,text=True)
                self.assertEqual(failed.returncode,code,failed.stderr);self.assertFalse(submit.exists())

class StagingTests(unittest.TestCase):
    def test_actual_shell_root_and_production_cleanup_ownership(self):
        import re
        from pathlib import PurePosixPath
        bash=os.environ.get('MIZAR_TEST_BASH','C:/Program Files/Git/bin/bash.exe')
        assignment=re.search(r'^STAGE_ROOT=.*$',(PACKAGE/'run.sh').read_text(),re.M).group()
        result=subprocess.run([bash,'-c','STAGE_TOKEN=fixture_123\n'+assignment+'\nprintf "%s" "$STAGE_ROOT"'],
                              capture_output=True,text=True,check=True)
        value=result.stdout;self.assertEqual(value,'/dev/shm/mizar_s3_full_ft_fixture_123')
        state={'exists':True,'symlink':False,'alias':False}
        class PosixFixture(PurePosixPath):
            def is_symlink(self):return state['symlink']
            def resolve(self):return self/'alias' if state['alias'] else self
            def exists(self):return state['exists']
            def read_text(self,*args,**kwargs):return json.dumps({'token':'fixture_123','root':value})
        with patch.object(sd,'Path',PosixFixture),patch.object(sd.shutil,'rmtree') as delete:
            self.assertEqual(str(sd.validate_stage_root(value)),value)
            for bad in ('/dev/shm/mizar_l14_17x3_sr_fixture_123',
                        '/dev/shm/mizar_l9_12x4_mcq_fixture_123','/tmp/mizar_s3_full_ft_fixture_123',value+'/nested'):
                with self.assertRaises(ValueError):sd.validate_stage_root(bad)
            for key in ('symlink','alias'):
                state[key]=True
                with self.assertRaises(ValueError):sd.validate_stage_root(value)
                state[key]=False
            with self.assertRaises(ValueError):sd.cleanup(value,'wrong')
            delete.assert_not_called()
            sd.cleanup(value,'fixture_123');delete.assert_called_once_with(PosixFixture(value))
            state['exists']=False;delete.reset_mock()
            sd.cleanup(value,'fixture_123');delete.assert_not_called()

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
