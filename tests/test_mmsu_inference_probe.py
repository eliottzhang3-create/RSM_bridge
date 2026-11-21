"""MMSU source review and diagnostic control-flow tests; no production weights."""
import ast
import contextlib
import importlib.util
import io
import json
import pathlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from unittest.mock import Mock, patch

import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/RSmol'),str(ROOT/'tests')]
from mizar_mmsu_inference_probe import probe
from mizar_s1_ced_mmsu_eval import dataset, runtime as old, storage
from text71427_ced_eval import mmsu_runtime as new, mmsu, benchmark, registry
from test_mizar_s1_ced_mmsu_eval import record


def function(path,name):
    return next(n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
                if isinstance(n,ast.FunctionDef) and n.name==name)


class SourceReviewTests(unittest.TestCase):
    def test_actual_new_and_previous_mmsu_preprocess_and_generation_are_identical(self):
        self.assertEqual(new.PROTOCOL,old.PROTOCOL)
        for name in ('read_audio','decode_audio','prefix_and_decode','generate'):
            self.assertEqual(ast.dump(function(pathlib.Path(new.__file__),name)),
                             ast.dump(function(pathlib.Path(old.__file__),name)),name)
        from mizar_s1_ced_mmsu_eval import evaluate,parsing,scoring
        self.assertIs(mmsu.dataset,dataset);self.assertIs(mmsu.parsing,parsing);self.assertIs(mmsu.scoring,scoring)
        for name in ('execute','run','summarize'):
            self.assertEqual(ast.dump(function(pathlib.Path(mmsu.__file__),name)),
                             ast.dump(function(pathlib.Path(evaluate.__file__),name)),name)

    def test_all_six_training_tokenization_and_qa_case_policy_and_loss_shift(self):
        for alias,spec in registry.ROUTES.items():
            path=spec['route_root']/'data/audiotext_dataset.py'
            pair=function(path,'text_pair')
            space={}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[pair],type_ignores=[])),str(path),'exec'),space)
            prompt,answer=space['text_pair']({'input':'Question: Which Country? A. India','answer':'India'})
            self.assertEqual(prompt,'question: which country? a. india',alias)
            self.assertEqual(answer,'india',alias)
            tree=ast.parse(path.read_text(encoding='utf-8-sig'))
            klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AudioTextDataset')
            tokenize=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='tokenize')
            code=ast.unparse(tokenize)
            for contract in ('<|endoftext|>', "padding='max_length'", 'truncation=True', 'add_special_tokens=True'):
                self.assertIn(contract,code)
            loss=ast.unparse(function(spec['route_root']/'train.py','answer_loss'))
            self.assertIn('logits[:, prefix_len - 1:-1]',loss)
            self.assertIn('targets = answer[\'input_ids\']',loss)

    def test_prompt_ablation_separates_case_and_format_and_never_uses_gt(self):
        row=record()
        cases=probe.prompt_cases(row,dataset,benchmark)
        changed={**row,'answer_gt':'DO_NOT_INSERT_THIS_GROUND_TRUTH'}
        self.assertEqual(cases,probe.prompt_cases(changed,dataset,benchmark))
        self.assertEqual(cases['official_original'],dataset.prompt(row))
        self.assertEqual(cases['official_lowercase'],dataset.prompt(row).lower())
        self.assertEqual(cases['compact_original'],'Which Country? a) India b) None c) China d) Japan')
        self.assertEqual(cases['compact_lowercase'],'which country? a) india b) none c) china d) japan')
        self.assertIn('only A, B, C, or D',cases['official_original'])
        self.assertNotIn('only a, b, c, or d',cases['compact_lowercase'])

    def test_preview_requires_no_assets_for_new_and_legacy_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            for name,extra in (
                ('text71427',['--eval-variant','listen','--checkpoint','/missing/formal/checkpoints/model--step-34626.ckpt']),
                ('legacy',['--eval-variant','recursive_mesh','--run-root','/missing',
                           '--expected-step','17313','--expected-geometry','8,8,4'])):
                output=pathlib.Path(directory)/name
                result=subprocess.run([sys.executable,'-B','-m','mizar_mmsu_inference_probe.probe',
                    '--backend',name,'--output-dir',str(output),'--check-args',*extra],
                    cwd=registry.ROOT,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                value=json.loads(result.stdout)
                self.assertTrue(value['diagnostic_only']);self.assertEqual(value['selected_ids'],list(probe.DEFAULT_IDS))
                self.assertFalse(output.exists())

    def test_historical_backend_root_uses_matching_git_snapshot_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory)
            result=subprocess.run(['git','archive','--format=zip',
                '8a9f1f76722e77cba4be429a578c63fdcf6ee527','code/RSmol'],
                cwd=ROOT,capture_output=True,check=True)
            with zipfile.ZipFile(io.BytesIO(result.stdout)) as archive:
                for name in archive.namelist():
                    self.assertTrue((root/name).resolve().is_relative_to(root.resolve()))
                archive.extractall(root)
            old_root=root/'code/RSmol'
            output=root/'not-created'
            result=subprocess.run([sys.executable,'-B','-m','mizar_mmsu_inference_probe.probe',
                '--implementation-root',str(old_root),'--backend','text71427','--eval-variant','base',
                '--checkpoint','/missing/formal/checkpoints/model--step-34626.ckpt',
                '--output-dir',str(output),'--check-args'],cwd=registry.ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            value=json.loads(result.stdout)
            self.assertEqual(value['implementation_root'],str(old_root.resolve()))
            self.assertTrue(pathlib.Path(value['route_root']).is_relative_to(old_root))
            self.assertFalse(output.exists())


class Tensor(np.ndarray):
    def to(self,*a,**kw):return self
    def detach(self):return self
    def cpu(self):return self
    def contiguous(self):return self
    def numpy(self):return np.asarray(self)
    def unsqueeze(self,axis):return np.expand_dims(self,axis).view(Tensor)
    def bool(self):return self.astype(bool).view(Tensor)
    def float(self):return self.astype(np.float32).view(Tensor)
    def ne(self,value):return (self!=value).view(Tensor)
    def abs(self):return np.abs(self).view(Tensor)
    def transpose(self,a,b):return np.swapaxes(self,a,b).view(Tensor)


def tensor(value,dtype=np.float32):return np.array(value,dtype=dtype).view(Tensor)


class NumericalTests(unittest.TestCase):
    def fixture(self,bad_position=False,context_required=False):
        def tokenize(text,length):
            ids=np.full((1,length),17,np.int64);mask=np.zeros((1,length),np.int64)
            ids[0,:2]=[4,0];mask[0,:2]=1
            return {'input_ids':tensor(ids,np.int64),'attention_mask':tensor(mask,np.int64)}
        encoder=types.SimpleNamespace(tokenize=tokenize)
        marker=object()
        class LM:
            config=types.SimpleNamespace(hidden_size=2)
            def get_input_embeddings(self):
                return lambda ids:tensor(np.repeat(np.asarray(ids)[...,None]+1,2,axis=-1))
            def __call__(self,inputs_embeds,**kw):
                if context_required: self_outer.assertIs(kw['audio_information'],marker)
                values=np.zeros((*inputs_embeds.shape[:2],20),np.float32)
                values[:,:,2]=inputs_embeds.sum(-1)
                return types.SimpleNamespace(logits=tensor(values))
        class Model:
            caption_decoder=types.SimpleNamespace(lm=LM())
            def generate_prefix_inference(self,inputs):
                prefix=tensor(np.full((1,383,2),3.))
                return prefix,None,marker
            def __call__(self,inputs):
                prefix=self.generate_prefix_inference(inputs)[0]
                answer=self.caption_decoder.lm.get_input_embeddings()(inputs['answer']['input_ids'])
                logits=self.caption_decoder.lm(tensor(np.concatenate((prefix,answer),axis=1)),
                    **({'audio_information':marker} if context_required else {})).logits
                if bad_position:logits[:,382,:]=tensor(np.arange(20)[None,None,:])
                return types.SimpleNamespace(logits=logits)
        self_outer=self
        model=Model()
        selected=types.SimpleNamespace(generate_prefix_inference=lambda m,i,**kw:m.generate_prefix_inference(i),
            sample_context=lambda *a:{'audio_information':marker} if context_required else {})
        mmsu_adapter=types.SimpleNamespace(adapter=lambda v:(selected,'multilayer',None))
        def cross_entropy(logits,targets,**kw):
            values=np.asarray(logits);values=values-values.max(axis=1,keepdims=True)
            lp=values-np.log(np.exp(values).sum(axis=1,keepdims=True))
            return tensor(-np.take_along_axis(lp,np.asarray(targets)[:,None,:],axis=1)[:,0,:])
        fake=types.SimpleNamespace(inference_mode=contextlib.nullcontext,
            isfinite=np.isfinite,equal=np.array_equal,allclose=np.allclose,
            cat=lambda values,dim:tensor(np.concatenate(values,axis=dim)),
            nn=types.SimpleNamespace(functional=types.SimpleNamespace(cross_entropy=cross_entropy)))
        loaded=(model,types.SimpleNamespace(pad_token_id=17,eos_token_id=0,decode=lambda ids:'fixture'),
            'cpu',encoder,types.SimpleNamespace(fixed_len=lambda w,n:w),{},
            {'data.audiotext_dataset':types.SimpleNamespace(text_pair=lambda row:(row['input'].lower(),row['answer'].lower()))})
        return loaded,mmsu_adapter,fake

    def test_actual_diagnostic_compares_training_683_with_383_and_384_and_supplies_context(self):
        for needs_context in (False,True):
            loaded,adapter,fake=self.fixture(context_required=needs_context)
            with patch.dict(sys.modules,{'torch':fake}):
                result=probe.training_layout_audit(loaded,adapter,types.SimpleNamespace(eval_variant='fixture'),
                    'Question: Which country?',tensor(np.zeros(640000)),'India',{'generated_token_ids':[2,0]})
            self.assertTrue(result['passed'])
            self.assertEqual(result['first_predicting_logit_position'],382)
            self.assertEqual(result['second_teacher_forced_logits']['max_abs_difference'],0.)
            self.assertEqual(result['gt_tokens'],2)
            self.assertFalse(result['prompt_matches_training_default_QA_transform'])

    def test_bad_position_or_different_greedy_first_token_is_exposed(self):
        for bad_position,generated in ((True,[2,0]),(False,[9,0])):
            loaded,adapter,fake=self.fixture(bad_position=bad_position)
            with patch.dict(sys.modules,{'torch':fake}):
                result=probe.training_layout_audit(loaded,adapter,types.SimpleNamespace(eval_variant='fixture'),
                    'question?',tensor(np.zeros(640000)),'India',{'generated_token_ids':generated})
            self.assertFalse(result['passed'])

    def test_all_free_generations_precede_gt_diagnostics_and_runner_mismatch_detected(self):
        row=record();row['audio']={'bytes':b'fixture'};events=[]
        generate=Mock(side_effect=lambda *a,**kw:(events.append('generate') or {'generated_token_ids':[2,0],'generated_text':'raw'}))
        paired=types.SimpleNamespace(mmau=benchmark.mmau,generate=Mock(return_value={'generated_token_ids':[9,0]}))
        adapter=types.SimpleNamespace(generate=generate)
        class Wave:
            def abs(self):return np.ones(1)
            def square(self):return types.SimpleNamespace(mean=lambda:types.SimpleNamespace(sqrt=lambda:1.))
        with patch.object(probe,'training_layout_audit',side_effect=lambda *a:(events.append('teacher') or {'passed':True})), \
             patch.object(probe,'compare_audio',return_value={'mmsu_vs_mmau_bytes_reader_equal':True}), \
             patch.object(probe,'tensor_sha',return_value='explicit-fixture'):
            result=probe.probe_sample(None,types.SimpleNamespace(eval_variant='fixture',dataset_dir='/fixture'),
                row,Wave(),{},(mmsu,adapter,paired))
        self.assertEqual(events,['generate']*4+['teacher']*4)
        self.assertFalse(result['same_input_runner_token_ids_equal'])


class WorkflowTests(unittest.TestCase):
    def run_fixture(self,path,wrong_data=False,parity=True):
        rows=[{**record(i),'id':key} for i,key in enumerate(probe.DEFAULT_IDS)]
        runner=types.SimpleNamespace(__file__=mmsu.__file__,dataset=dataset,storage=storage,identity=Mock(return_value={'fixture':True}))
        adapter=types.SimpleNamespace(load_runtime=Mock(return_value=(None,)*5+({'fixture':True},None)),
            decode_audio=Mock(return_value=(object(),{})))
        args=types.SimpleNamespace(output_dir=path,dataset_dir=path/'source',parquet_batch_size=8)
        def sample(*a):
            return {'id':a[2]['id'],'same_input_runner_token_ids_equal':parity,
                    'audio_decoder_comparison':{'mmsu_vs_mmau_bytes_reader_equal':True},
                    'cases':{'fixture':{'generation':{'generated_text':'raw'},'training_layout':{'passed':True}}}}
        with patch.object(dataset,'shard_paths',return_value=[]), \
             patch.object(dataset,'preflight',return_value=([] if wrong_data else rows,{'fixture':True})), \
             patch.object(dataset,'iter_rows',return_value=iter((i,{**r,'audio':{}}) for i,r in enumerate(rows))), \
             patch.object(probe,'probe_sample',side_effect=sample),contextlib.redirect_stdout(io.StringIO()):
            result=probe.run(args,(runner,adapter,None),list(probe.DEFAULT_IDS))
        return result,adapter

    def test_three_exact_rows_no_benchmark_score_and_source_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'new'
            result,_=self.run_fixture(path)
            self.assertEqual(result['status'],'PASS_NUMERICAL_PARITY')
            self.assertEqual(len(result['samples']),3);self.assertIsNone(result['benchmark_accuracy'])
            before=(path/'diagnostic_report.json').read_bytes()
            with self.assertRaises(FileExistsError):self.run_fixture(path)
            self.assertEqual((path/'diagnostic_report.json').read_bytes(),before)

    def test_incorrect_data_fails_before_loader_and_numerical_difference_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)
            result,adapter=self.run_fixture(path/'wrong',wrong_data=True)
            self.assertEqual(result['status'],'FAILED');adapter.load_runtime.assert_not_called()
            result,_=self.run_fixture(path/'different',parity=False)
            self.assertEqual(result['status'],'DIFFERENCE_FOUND')


class BashTests(unittest.TestCase):
    def test_shell_submission_to_4090_quotes_arguments_and_calls_real_preview_without_weights(self):
        bash=pathlib.Path('C:/Program Files/Git/bin/bash.exe') if os.name=='nt' else pathlib.Path(shutil.which('bash') or '/absent')
        if not bash.is_file():self.skipTest('Bash unavailable')
        def posix(path):
            value=path.as_posix()
            return '/'+value[0].lower()+value[2:] if os.name=='nt' else value
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);conda=root/'conda/etc/profile.d';conda.mkdir(parents=True)
            (conda/'conda.sh').write_text('conda() { :; }\n',encoding='utf-8')
            bindir=root/'bin';bindir.mkdir()
            (bindir/'python').write_text('#!/usr/bin/env bash\nexec '+shlex.quote(posix(pathlib.Path(sys.executable)))+' "$@"\n',encoding='utf-8')
            (bindir/'vc').write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$PROBE_VC_CAPTURE"\n',encoding='utf-8')
            for path in bindir.iterdir():path.chmod(0o755)
            capture=root/'vc.txt'
            env={**os.environ,'RSMOL_CONDA_BASE':posix(root/'conda'),
                'PATH':posix(bindir)+(';' if os.name=='nt' else ':')+os.environ['PATH'],
                'PROBE_VC_CAPTURE':posix(capture),'MSYS2_ARG_CONV_EXCL':'*','PYTHONIOENCODING':'utf-8'}
            wrapper=registry.ROOT/'run_mmsu_inference_probe_4090.sh'
            for path in (wrapper,pathlib.Path(probe.__file__).parent/'run.sh'):
                self.assertNotIn(b'\r',path.read_bytes())
                result=subprocess.run([str(bash),'-n',str(path)],capture_output=True,text=True,env=env)
                self.assertEqual(result.returncode,0,result.stderr)
            checkpoint=str(root/"run with space's/formal/checkpoints/model--step-34626.ckpt")
            output=str(root/"diagnostic output's")
            result=subprocess.run([str(bash),str(wrapper),'--backend','text71427','--eval-variant','listen',
                '--checkpoint',checkpoint,'--output-dir',output],capture_output=True,text=True,encoding='utf-8',env=env)
            self.assertEqual(result.returncode,0,result.stderr)
            cmd=capture.read_text().splitlines()
            self.assertEqual(cmd[cmd.index('-p')+1],'pdgpu-4090')
            self.assertEqual(cmd[cmd.index('-g')+1],'1');self.assertEqual(cmd[cmd.index('-c')+1],'8')
            self.assertLess(len(cmd[cmd.index('-j')+1]),60)
            preview=json.loads(result.stdout.splitlines()[0])
            self.assertEqual(preview['checkpoint'],str(pathlib.Path(checkpoint)))
            self.assertEqual(preview['output_dir'],str(pathlib.Path(output)))
            self.assertFalse(pathlib.Path(output).exists())


MISSING=[name for name in ('torch','transformers','safetensors','torchaudio','soundfile')
         if importlib.util.find_spec(name) is None]


@unittest.skipIf(MISSING,'Real CPU model dependencies missing: '+', '.join(MISSING))
class RealTensorTests(unittest.TestCase):
    def test_real_mesh_training_decoder_683_and_autoregressive_383_384_logits(self):
        import torch
        import importlib
        from transformers import LlamaConfig
        from mizar_text_7_14x2_7_mesh.contract import build_target_config
        torch.set_num_threads(1);torch.manual_seed(1234)
        old_path=list(sys.path)
        roots=('models','data','utils','contract')
        old_modules={k:v for k,v in sys.modules.items() if any(k==n or k.startswith(n+'.') for n in roots)}
        for key in old_modules:del sys.modules[key]
        sys.path.insert(0,str(registry.ROUTES['base']['route_root']))
        try:
            implementation=importlib.import_module('recursive_model_mizar_text_7_14x2_7_mesh')
            decoder_module=importlib.import_module('models.decoder')
            mellow_module=importlib.import_module('models.mellow')
            data_module=importlib.import_module('data.audiotext_dataset')
            cfg=LlamaConfig(num_hidden_layers=30,hidden_size=16,intermediate_size=32,
                num_attention_heads=4,num_key_value_heads=2,vocab_size=43,tie_word_embeddings=True,
                pad_token_id=17,eos_token_id=0,max_position_embeddings=1024,use_cache=False,
                attention_dropout=0.,_attn_implementation='eager')
            lm=implementation.MizarText7_14x2_7MeshForCausalLM(build_target_config(cfg)).float().eval()
            decoder=decoder_module.DecoderModel.__new__(decoder_module.DecoderModel)
            torch.nn.Module.__init__(decoder)
            decoder.prefix_length=383;decoder.lm=lm;decoder.lm_embedding_size=16
            class Audio(torch.nn.Module):
                def forward(self,w):return w[:,:126,None].expand(-1,-1,16),None,{}
            model=mellow_module.Mellow.__new__(mellow_module.Mellow)
            torch.nn.Module.__init__(model)
            model.audio_encoder=Audio();model.caption_decoder=decoder;model.eval()
            class Tokenizer:
                pad_token_id=17;eos_token_id=0
                def decode(self,ids):return str(ids)
                def encode_plus(self,*,text,max_length,**kw):
                    ids=torch.full((1,max_length),17,dtype=torch.long);mask=torch.zeros_like(ids)
                    ids[0,:3]=torch.tensor([4,5,0]);mask[0,:3]=1
                    return {'input_ids':ids,'attention_mask':mask}
            tokenizer=Tokenizer();encoder=data_module.AudioTextDataset.__new__(data_module.AudioTextDataset)
            encoder.tokenizer=tokenizer
            fixed=types.SimpleNamespace(fixed_len=lambda w,n:w)
            loaded=(model,tokenizer,'cpu',encoder,fixed,{}, {'data.audiotext_dataset':data_module})
            waveform=torch.linspace(-.1,.1,640000)
            _,inputs,_=probe.model_inputs(loaded,'question?',waveform)
            with torch.inference_mode():
                prefix=model.generate_prefix_inference(inputs)[0]
                first=int(lm(inputs_embeds=prefix,use_cache=False).logits[0,-1].argmax())
            result=probe.training_layout_audit(loaded,new,types.SimpleNamespace(eval_variant='base'),
                'question?',waveform,'India',{'generated_token_ids':[first]})
            self.assertTrue(result['passed'],result)
            self.assertTrue(result['first_answer_logits']['argmax_equal'])
        finally:
            for name in list(sys.modules):
                if any(name==n or name.startswith(n+'.') for n in roots):del sys.modules[name]
            sys.modules.update(old_modules);sys.path[:]=old_path


if __name__=='__main__':unittest.main()
