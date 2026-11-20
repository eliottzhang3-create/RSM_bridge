"""Portable contract, exact reference parity, decode and submission checks."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import pathlib
import shlex
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import Mock, patch

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/RSmol'))
from text71427_ced_eval import registry, runtime, arguments, submit, reasonaqa, mmsu, benchmark
from text71427_ced_eval import reasonaqa_runtime, mmsu_runtime, observers
from mizar_s1_ced_reasonaqa5000_eval import runtime as old_reason, scoring as reason_scoring
from mizar_s1_ced_mmsu_eval import runtime as old_mmsu, parsing


def function(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def route_module(alias):
    path = registry.ROUTES[alias]['route_root'] / 'contract.py'
    spec = importlib.util.spec_from_file_location('eval_contract_' + alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_fixture(alias, step=5771):
    c = route_module(alias)
    prepared = dict(rows=1477492, rows_canonical_sha256='rows', mapping_sha256='mapping',
                    source_rows=dict(c.SOURCE_COUNTS))
    contract = c.semantic_contract(prepared, '/source/text', '/source/ced')
    contract['recursive_text_checkpoint'] = {'fixture': True}
    saved = dict(schema_version=1, route_variant=c.ROUTE, mode='formal', contract=contract,
                 contract_sha256=c.digest(contract), state_dict={'fixture': True}, total_step=step,
                 num_epochs=3, loss_reduction=runtime.reference.TRAINING_LOSS_REDUCTION,
                 batch_geometry=dict(world_size=8, microbatch_size=4, gradient_accumulation_steps=4, global_batch_size=128))
    config = dict(mode='formal', route=c.ROUTE, contract=contract, contract_sha256=c.digest(contract))
    return c, saved, config


class ReferenceParityTests(unittest.TestCase):
    def test_all_original_tracked_sources_unchanged(self):
        protected = json.loads((registry.PACKAGE / 'protected_sources.json').read_text())['sources']
        self.assertGreater(len(protected), 1000)
        for name, expected in protected.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest(), expected, name)

    def test_inference_protocols_equal_original_dictionaries(self):
        self.assertEqual(runtime.PROTOCOL, runtime.reference.PROTOCOL)
        self.assertEqual(reasonaqa_runtime.PROTOCOL, old_reason.PROTOCOL)
        self.assertEqual(mmsu_runtime.PROTOCOL, old_mmsu.PROTOCOL)

    def test_real_generation_and_preprocessing_functions_are_ast_identical(self):
        pairs = [
            ('reasonaqa_runtime.py', 'mizar_s1_ced_reasonaqa5000_eval/runtime.py',
             ('generate','fixed20_pair','waveform_sha','prefix_and_decode')),
            ('mmsu_runtime.py', 'mizar_s1_ced_mmsu_eval/runtime.py',
             ('read_audio','decode_audio','generate','prefix_and_decode')),
            ('benchmark.py', 'scripts/evaluate_mizar_s1_ced_multilayer_benchmark.py',
             ('generate','_fixed20_pair','make_mmau_audio_decoder','_prepare_mmar','_metadata_records',
              '_materialize','_write_predictions','run_official_scorer','scorer_prediction_key')),
            ('runtime.py','scripts/mizar_s1_ced_multilayer_eval_runtime.py',('greedy_decode',)),
            ('reasonaqa.py','mizar_s1_ced_reasonaqa5000_eval/evaluate.py',('execute','run','summarize')),
            ('mmsu.py','mizar_s1_ced_mmsu_eval/evaluate.py',('execute','run','summarize')),
        ]
        for new, old, names in pairs:
            for name in names:
                with self.subTest(file=new, name=name):
                    self.assertEqual(ast.dump(function(registry.PACKAGE/new,name)),
                                     ast.dump(function(registry.ROOT/old,name)))

    def test_actual_data_and_scorer_objects_are_reused(self):
        from mizar_s1_ced_reasonaqa5000_eval import dataset, sample, storage, scoring
        from mizar_s1_ced_mmsu_eval import dataset as md, storage as ms, parsing as mp
        self.assertIs(reasonaqa.dataset,dataset); self.assertIs(reasonaqa.sample,sample)
        self.assertIs(reasonaqa.storage,storage); self.assertIs(reasonaqa.scoring,scoring)
        self.assertIs(mmsu.dataset,md); self.assertIs(mmsu.storage,ms); self.assertIs(mmsu.parsing,mp)

    def test_no_new_file_enters_old_runtime_identity_globs(self):
        old = set(registry.ROOT.glob('run_reasonaqa5000*5090.sh'))
        old.update((registry.ROOT/'scripts').glob('mizar_s1_ced_*eval_runtime.py'))
        old.update(registry.ROOT.glob('recursive_model*.py'))
        self.assertFalse(set(registry.PACKAGE.glob('*')) & old)
        self.assertNotIn(registry.ROOT/'run_text71427_ced_eval_5090.sh',old)

    def test_python310_and_lf_shell_syntax_sources(self):
        for path in registry.PACKAGE.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,10))
        for path in (registry.PACKAGE/'run.sh',registry.ROOT/'run_text71427_ced_eval_5090.sh'):
            self.assertNotIn(b'\r',path.read_bytes())


class ContractTests(unittest.TestCase):
    def test_six_actual_contracts_accept_intermediate_and_completed_artifacts(self):
        for alias in registry.ROUTES:
            for step in (5771,34626):
                with self.subTest(alias=alias,step=step):
                    c,s,r = checkpoint_fixture(alias,step)
                    self.assertIs(runtime.validate_checkpoint(s,r,c,step,(8,4,4),eval_variant=alias),s['contract'])

    def test_every_cross_route_checkpoint_is_rejected(self):
        for alias in registry.ROUTES:
            c,s,r = checkpoint_fixture(alias)
            for other in set(registry.ROUTES)-{alias}:
                with self.subTest(source=alias,target=other), self.assertRaises(ValueError):
                    runtime.validate_checkpoint(s,r,c,5771,(8,4,4),eval_variant=other)

    def test_source_runtime_precision_schedule_branch_and_step_failures(self):
        for alias in registry.ROUTES:
            c,original,config = checkpoint_fixture(alias)
            mutations=[('forward_dtype','bfloat16'),('text_model_schedule',list(range(42))),
                       ('code_sha256','changed'),('recursive_model_source_sha256','changed'),
                       ('audio_policy','first10'),('attention_mask','mask'),('loop_lora',{'wrong':True})]
            for key,value in mutations:
                saved,run=copy.deepcopy(original),copy.deepcopy(config)
                saved['contract'][key]=value;run['contract']=saved['contract']
                run['contract_sha256']=saved['contract_sha256']=c.digest(saved['contract'])
                with self.subTest(alias=alias,key=key),self.assertRaises(ValueError):
                    runtime.validate_checkpoint(saved,run,c,5771,(8,4,4),eval_variant=alias)
            for step in (True,0,34627,17313):
                with self.assertRaises(ValueError):
                    runtime.validate_checkpoint(original,config,c,step,(8,4,4),eval_variant=alias)
            with self.assertRaises(ValueError):
                runtime.validate_checkpoint(original,{**config,'route':'wrong'},c,5771,(8,4,4),eval_variant=alias)
            with self.assertRaises(ValueError):
                runtime.validate_checkpoint(original,config,c,5771,(8,8,4),eval_variant=alias)


class ArgumentsAndSubmissionTests(unittest.TestCase):
    def test_24_bindings_resource_geometry_and_independent_output_paths(self):
        outputs=[]
        for alias in registry.ROUTES:
            args=submit.parse_args([alias,'all','--checkpoint',f'/runs/{alias}/formal/checkpoints/model--step-5771.ckpt','--check-args'])
            jobs=submit.planned(args)
            self.assertEqual(len(jobs),4)
            for job in jobs:
                self.assertEqual(job['resolved'].expected_geometry,(8,4,4))
                self.assertEqual(job['resolved'].expected_step,5771)
                self.assertEqual(job['resolved'].runtime_config.name,'runtime.json')
                self.assertEqual(job['resolved'].runtime_config.parent.name,'formal')
                outputs.append(job['resolved'].output_dir)
                cmd=submit.command(job,'tag')
                self.assertEqual(cmd[cmd.index('-g')+1],'1');self.assertEqual(cmd[cmd.index('-c')+1],'8')
                self.assertEqual(cmd[cmd.index('-m')+1],'32G');self.assertEqual(cmd[cmd.index('-n')+1],'1')
                self.assertLess(len(cmd[cmd.index('-j')+1]),60)
                self.assertEqual(job['queue'],'pdgpu-4090' if job['benchmark']=='reasonaqa5000' else 'pdgpu-5090')
        self.assertEqual(len(set(outputs)),24)

    def test_explicit_checkpoint_required_and_no_default_final_step(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):arguments.parse_args('mmsu',['--eval-variant','base'])
            with self.assertRaises(SystemExit):arguments.parse_args('mmsu',['--eval-variant','base','--checkpoint','/r/formal/checkpoints/model--step-5771.ckpt','--expected-step','34626'])
            with self.assertRaises(SystemExit):arguments.parse_args('mmsu',['--eval-variant','base','--checkpoint','/r/model.tmp'])
        args=arguments.parse_args('mmsu',['--eval-variant','base','--checkpoint','/r/formal/checkpoints/model--epo-3.ckpt'])
        self.assertIsNone(args.expected_step)
        saved={'total_step':34626};fake=types.SimpleNamespace(load=Mock(return_value=saved))
        with tempfile.TemporaryDirectory() as directory:
            args.checkpoint=pathlib.Path(directory)/'model--epo-3.ckpt';args.checkpoint.write_bytes(b'fixture')
            with patch.dict(sys.modules,{'torch':fake}):
                self.assertIs(runtime.read_checkpoint(args),saved)
                self.assertIs(runtime.read_checkpoint(args),saved)
            fake.load.assert_called_once();self.assertEqual(args.expected_step,34626)

    def test_noncanonical_requires_runtime_output_and_spaces_survive_shell_quoting(self):
        path="/run with spaces/model's weights.ckpt"
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            arguments.parse_args('mmsu',['--eval-variant','base','--checkpoint',path])
        jobs=submit.planned(submit.parse_args(['wfuse','mmsu','--checkpoint',path,
            '--runtime-config',"/run with spaces/runtime's.json",'--output-dir',"/new output's",'--check-args']))
        compute=shlex.split(submit.command(jobs[0],'tag')[-1])
        self.assertEqual(compute[compute.index('--checkpoint')+1],str(pathlib.Path(path)))
        self.assertEqual(compute[compute.index('--output-dir')+1],str(pathlib.Path("/new output's")))

    def test_batch_and_failure_continue_without_real_vc(self):
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory);entries=[]
            for alias in registry.ROUTES:
                checkpoint=path/alias/'formal/checkpoints/model--step-5771.ckpt'
                checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'fixture')
                (checkpoint.parent.parent/'runtime.json').write_text('{}')
                entries.append({'model':alias,'checkpoint':str(checkpoint)})
            manifest=path/'targets.json';manifest.write_text(json.dumps(entries))
            argv=['batch','all','--targets-json',str(manifest),'--submission-report',str(path/'submit.json')]
            good=types.SimpleNamespace(returncode=0,stdout='fixture submitted',stderr='')
            bad=types.SimpleNamespace(returncode=1,stdout='',stderr='fixture failed')
            with patch.object(submit.subprocess,'run',side_effect=[bad]+[good]*23) as call, \
                 patch.object(submit,'ROOT',path),redirect_stdout(io.StringIO()):
                self.assertEqual(submit.main(argv),1)
            self.assertEqual(call.call_count,24)
            report=json.loads((path/'submit.json').read_text())
            self.assertEqual(sum(j['status']=='SUBMITTED' for j in report['jobs']),23)
            self.assertEqual(report['status'],'FAILED')
            entries.append(entries[0]);manifest.write_text(json.dumps(entries))
            with self.assertRaisesRegex(ValueError,'repeated output'):
                submit.planned(submit.parse_args(['batch','all','--targets-json',str(manifest)]))

    def test_check_args_reads_no_checkpoint_and_makes_no_submission(self):
        with patch.object(submit.subprocess,'run',side_effect=AssertionError('vc called')), \
             patch.object(runtime,'read_checkpoint',side_effect=AssertionError('weights read')),redirect_stdout(io.StringIO()) as out:
            self.assertEqual(submit.main(['base','all','--checkpoint','/absent/formal/checkpoints/model--step-5771.ckpt','--check-args']),0)
        self.assertEqual(len(out.getvalue().splitlines()),4)


class Tensor(np.ndarray):
    @property
    def device(self):return 'cpu'


class DecodeTests(unittest.TestCase):
    def test_real_greedy_loop_preserves_per_sample_context_on_each_forward(self):
        fake=types.SimpleNamespace(Tensor=Tensor,float32=np.float32,long=np.int64,
            isfinite=np.isfinite,argmax=lambda x,dim:np.argmax(x,axis=dim),
            tensor=lambda x,**kw:np.asarray(x,dtype=kw.get('dtype')).view(Tensor),
            cat=lambda x,dim:np.concatenate(x,axis=dim).view(Tensor))
        class LM:
            def __init__(self):self.calls=[]
            def __call__(self,**kw):
                self.calls.append(kw)
                value=np.zeros((1,kw['inputs_embeds'].shape[1],4),dtype=np.float32).view(Tensor)
                value[:,-1,2 if len(self.calls)==1 else 0]=1
                return types.SimpleNamespace(logits=value)
            def get_input_embeddings(self):return lambda ids:np.zeros((1,1,576),np.float32).view(Tensor)
        tokenizer=types.SimpleNamespace(eos_token_id=0,decode=lambda ids,**kw:'  B) raw<|endoftext|>  ')
        prefix=np.zeros((1,383,576),np.float32).view(Tensor)
        for alias in registry.ROUTES:
            lm=LM();model=types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=lm))
            pair=tuple(np.full((1,126,576),i,np.float32).view(Tensor) for i in (4,8))
            context=runtime.sample_context(alias,None,None if alias=='base' else pair,None)
            with patch.dict(sys.modules,{'torch':fake}):
                result=runtime.greedy_decode(model,tokenizer,prefix,eval_variant=alias,context=context)
            self.assertEqual(result['generated_token_ids'],[2,0])
            self.assertEqual(result['generated_text'],'  B) raw<|endoftext|>  ')
            self.assertEqual([x['inputs_embeds'].shape[1] for x in lm.calls],[383,384])
            for call in lm.calls:
                self.assertIs(call['use_cache'],False);self.assertIsNone(call['attention_mask'])
                if alias!='base':self.assertIs(call['audio_information'],pair)

    def test_malformed_or_other_sample_context_is_not_silently_accepted(self):
        fake=types.SimpleNamespace(Tensor=Tensor,isfinite=np.isfinite)
        prefix=np.zeros((1,383,576),np.float32).view(Tensor)
        for alias in set(registry.ROUTES)-{'base'}:
            for pair in (None,(),[prefix,prefix],tuple(np.zeros((1,126,576),np.float64).view(Tensor) for _ in range(2))):
                with patch.dict(sys.modules,{'torch':fake}),self.assertRaises(ValueError):
                    runtime.validate_context(alias,{'audio_information':pair},prefix)
        with self.assertRaises(ValueError):runtime.sample_context('base',None,'extra',None)

    def test_scoring_boundaries_remain_distinct(self):
        self.assertNotEqual(reason_scoring.normalize('B'),reason_scoring.normalize('B) India'))
        self.assertNotEqual(reason_scoring.normalize('no.'),reason_scoring.normalize('no'))
        self.assertTrue(parsing.parse_response('C) India<|endoftext|>','India')['matched'])
        self.assertFalse(parsing.parse_response('A','India')['matched'])
        self.assertFalse(parsing.parse_response('Indian','India')['matched'])


if __name__=='__main__':unittest.main()
