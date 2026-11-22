"""Offline acceptance: real Git sources/contracts; model/GPU fixtures are explicit."""
import ast
import atexit
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/RSmol'),str(ROOT/'tests')]
from ced_eval_20261122 import registry, sources, runtime, preflight, submit, plain
from ced_eval_20261122 import arguments, reasonaqa, mmsu, benchmark, reasonaqa_runtime, mmsu_runtime
from ced_eval_20261122.targets import target_records
from text71427_ced_eval import runtime as previous

_cache=None
def cache():
    global _cache
    if _cache is None:
        _cache=tempfile.TemporaryDirectory(prefix='ced22_sources_')
        atexit.register(_cache.cleanup)
        for alias in registry.ROUTES:sources.prepare(alias,_cache.name)
    return Path(_cache.name)

def checkpoint_fixture(alias,step=34626):
    c=sources.load_contract(alias,cache())
    prepared=dict(rows=1477492,rows_canonical_sha256='rows',mapping_sha256='mapping',source_rows=dict(c.SOURCE_COUNTS))
    contract=c.semantic_contract(prepared,'/source/text','/source/ced')
    if not registry.ROUTES[alias]['plain']:contract['recursive_text_checkpoint']={'fixture':True}
    if alias=='mesh_reinit':
        contract['audio_initialization']={'mode':'audio_checkpoint_weights_only',
            'checkpoint':target_records()['mesh_original']['checkpoint'],'source_checkpoint_step':34626,
            'start_optimizer_step':0,'restore_optimizer_scheduler_rng_cursor':False,
            'source_checkpoint_sha256':'fixture-source-sha','source_contract_sha256':'fixture-contract-sha'}
    saved=dict(schema_version=1,route_variant=c.ROUTE,mode='formal',contract=contract,
        contract_sha256=c.digest(contract),state_dict={'fixture':True},total_step=step,num_epochs=3,
        loss_reduction=runtime.reference.TRAINING_LOSS_REDUCTION,
        batch_geometry=dict(world_size=8,microbatch_size=4,gradient_accumulation_steps=4,global_batch_size=128))
    config=dict(mode='formal',route=c.ROUTE,contract=contract,contract_sha256=c.digest(contract))
    return c,saved,config

def function(path,name):
    return next(n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
                if isinstance(n,ast.FunctionDef) and n.name==name)

class SourceContractTests(unittest.TestCase):
    def test_real_git_snapshots_all_seven_exact_code_digests(self):
        for alias in registry.ROUTES:
            root=sources.verify(alias,cache());c=sources.load_contract(alias,cache())
            self.assertEqual(root,sources.route_root(alias,cache()).resolve())
            self.assertEqual(c.code_digest(),sources.profile(alias)[1]['training_code_sha256'][alias])
            if not registry.ROUTES[alias]['plain']:
                path=Path(sys.modules['mizar_text_7_14x2_7_mesh.contract'].__file__)
                self.assertTrue(path.is_relative_to(cache()))
    def test_content_changes_and_extra_python_rejected(self):
        root=sources.route_root('mesh_original',cache());path=root/'contract.py';before=path.read_bytes()
        try:
            path.write_bytes(before+b'\n# corruption\n')
            with self.assertRaisesRegex(ValueError,'content mismatch'):sources.prepare('mesh_original',cache())
        finally:path.write_bytes(before)
        extra=root/'extra.py'
        try:
            extra.write_text('# extra')
            with self.assertRaisesRegex(ValueError,'extra'):sources.verify('mesh_original',cache())
        finally:extra.unlink()
    def test_concurrent_cached_readers_do_not_take_the_preparation_lock(self):
        cache()
        with patch.object(sources,'output_lock',side_effect=AssertionError('cached reader took write lock')):
            for alias in registry.ROUTES:self.assertTrue(sources.prepare(alias,cache()).is_dir())
    def test_all_seven_actual_contracts_accept_model_only_headers(self):
        for alias in registry.ROUTES:
            for step in (5771,34626):
                c,s,r=checkpoint_fixture(alias,step)
                self.assertIs(runtime.validate_checkpoint(s,r,c,step,(8,4,4),eval_variant=alias),s['contract'])
                self.assertNotIn('optimizer',s)
    def test_every_cross_target_including_old_and_retrained_same_route_rejected(self):
        for alias in registry.ROUTES:
            _,s,r=checkpoint_fixture(alias)
            for other in set(registry.ROUTES)-{alias}:
                c=sources.load_contract(other,cache())
                with self.subTest(source=alias,target=other),self.assertRaises(ValueError):
                    runtime.validate_checkpoint(s,r,c,34626,(8,4,4),eval_variant=other)
    def test_wrong_precision_schedule_branch_recipe_provenance_rejected(self):
        for alias in registry.ROUTES:
            c,s,r=checkpoint_fixture(alias)
            changes=[('forward_dtype','bfloat16'),('attention_mask','mask'),('code_sha256','changed'),
                     ('audio_policy','first10'),('loop_lora',{'wrong':True}),('prefix',[126,1,255])]
            if alias=='mesh_reinit':changes += [('audio_initialization',{}),('adamw',{'decoder_lr':3e-4})]
            elif registry.ROUTES[alias]['plain']:changes += [('text_model_schedule',list(range(30)))]
            else:changes += [('text_model_schedule',list(range(42)))]
            for key,value in changes:
                bad=copy.deepcopy(s);bad['contract'][key]=value
                cfg={**r,'contract':bad['contract'],'contract_sha256':c.digest(bad['contract'])}
                bad['contract_sha256']=cfg['contract_sha256']
                with self.subTest(alias=alias,key=key),self.assertRaises(ValueError):
                    runtime.validate_checkpoint(bad,cfg,c,34626,(8,4,4),eval_variant=alias)
    def test_cpu_preflight_actual_contract_and_source_checks(self):
        for alias in registry.ROUTES:
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'formal/checkpoints/model--step-34626.ckpt';p.parent.mkdir(parents=True)
                p.write_bytes(b'explicit fixture');_,saved,config=checkpoint_fixture(alias)
                (p.parent.parent/'runtime.json').write_text(json.dumps(config))
                args=arguments.parse_args('mmsu',['--eval-variant',alias,'--checkpoint',str(p),'--sources-cache',str(cache())])
                loader=Mock(return_value=saved)
                with patch.dict(sys.modules,{'torch':types.SimpleNamespace(load=loader)}):
                    result=preflight.check(args)
                self.assertEqual(result['status'],'READY_FOR_GPU');loader.assert_called_once()
                self.assertTrue(loader.call_args.kwargs['mmap'])
                config['route']='wrong';(p.parent.parent/'runtime.json').write_text(json.dumps(config))
                with patch.dict(sys.modules,{'torch':types.SimpleNamespace(load=loader)}),self.assertRaises(ValueError):preflight.check(args)

class ProtocolTests(unittest.TestCase):
    def test_existing_2898_sources_preserved(self):
        before=json.loads((ROOT/'audit_reports/ced_eval_20261122_validation.json').read_text())['existing_sources_before']
        self.assertGreater(len(before),2800)
        for name,wanted in before.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),wanted,name)
    def test_real_generation_preprocessing_scoring_and_runner_core_ast_unchanged(self):
        pairs={'runtime.py':['normalize_state','greedy_decode','generate_prefix_inference','sample_context','validate_context'],
            'reasonaqa_runtime.py':['generate'],
            'mmsu_runtime.py':['read_audio','decode_audio','generate','prefix_and_decode'],
            'reasonaqa.py':['execute','run','summarize'], 'mmsu.py':['execute','run','summarize'],
            'benchmark.py':['generate','make_mmau_audio_decoder','_fixed20_pair','_prepare_mmar','run_official_scorer','_run']}
        for filename,names in pairs.items():
            for name in names:
                self.assertEqual(ast.dump(function(registry.PACKAGE/filename,name)),
                    ast.dump(function(registry.ROOT/'text71427_ced_eval'/filename,name)),(filename,name))
    def test_protocols_and_shared_inputs_match_current_evaluators(self):
        from mizar_s1_ced_mmsu_eval import runtime as old_mmsu,dataset
        from mizar_s1_ced_reasonaqa5000_eval import runtime as old_reason
        self.assertEqual(runtime.PROTOCOL,previous.PROTOCOL)
        self.assertEqual(mmsu_runtime.PROTOCOL,old_mmsu.PROTOCOL)
        self.assertEqual(reasonaqa_runtime.PROTOCOL,old_reason.PROTOCOL)
        self.assertIs(mmsu.dataset,dataset)
        self.assertEqual(dataset.prompt({'question':'Which Country?','choice_a':'India','choice_b':None,'choice_c':'China','choice_d':'Japan'}),
                         'which country? a) india b) none c) china d) japan')
    def test_python310_and_lf_bash(self):
        for p in registry.PACKAGE.glob('*.py'):ast.parse(p.read_text(),feature_version=(3,10))
        for p in (registry.PACKAGE/'run.sh',registry.ROOT/'run_ced_eval_20261122_4090.sh'):
            self.assertNotIn(b'\r',p.read_bytes())

class SubmitTests(unittest.TestCase):
    def test_exact19_bindings_and_4090_resources_no_duplicate_read2(self):
        jobs=submit.planned(submit.parse_args(['--check-args']))
        self.assertEqual(len(jobs),19)
        self.assertEqual([sum(j['benchmark']==b for j in jobs) for b in registry.BENCHMARKS],[4,4,4,7])
        self.assertEqual(len({str(j['resolved'].output_dir) for j in jobs}),19)
        for job in jobs:
            args=job['resolved'];cmd=submit.command(job,'fixture')
            self.assertEqual(args.eval_variant,job['target']);self.assertEqual(args.expected_step,34626)
            self.assertEqual(args.expected_geometry,(8,4,4));self.assertIn('ced_eval_20261122_v1',args.output_dir.parts)
            for flag,value in (('-p','pdgpu-4090'),('-g','1'),('-c','8'),('-m','32G'),('-n','1')):
                self.assertEqual(cmd[cmd.index(flag)+1],value)
            self.assertLess(len(cmd[cmd.index('-j')+1]),60)
            split=shlex.split(cmd[-1]);self.assertIn(target_records()[job['target']]['checkpoint'].replace('/',os.sep),
                split if os.name!='nt' else [p.replace('/',os.sep) for p in split])
        self.assertEqual(len(submit.planned(submit.parse_args(['all','mmsu']))),7)
        self.assertEqual(len(submit.planned(submit.parse_args(['audio7_read2']))),4)
        with self.assertRaises(ValueError):submit.planned(submit.parse_args(['listen','mmau']))
    def test_preflight_once_per_target_failed_target_never_submitted_other_jobs_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'submission.json';calls=[]
            def check(args):
                calls.append(args.eval_variant)
                if args.eval_variant=='listen':raise ValueError('wrong target identity')
                return {'status':'READY_FOR_GPU'}
            count=0
            def vc(*a,**kw):
                nonlocal count
                count+=1
                return types.SimpleNamespace(returncode=int(count==1),stdout='mock submit only',stderr='')
            with patch.object(preflight,'check',side_effect=check),patch.object(submit.subprocess,'run',side_effect=vc) as run,redirect_stdout(io.StringIO()):
                status=submit.main(['--submission-report',str(report)])
            data=json.loads(report.read_text());self.assertEqual(status,1)
            self.assertEqual(len(calls),7);self.assertEqual(len(set(calls)),7);self.assertEqual(run.call_count,18)
            self.assertEqual(sum(j['status']=='SUBMITTED' for j in data['jobs']),17)
            self.assertFalse(next(j for j in data['jobs'] if j['target']=='listen')['vc_attempted'])
    def test_preflight_only_and_check_args_never_submit(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(preflight,'check',return_value={'status':'READY_FOR_GPU'}) as check,patch.object(submit.subprocess,'run') as run,redirect_stdout(io.StringIO()):
                self.assertEqual(submit.main(['--check-args']),0);check.assert_not_called();run.assert_not_called()
                self.assertEqual(submit.main(['--preflight-only','--submission-report',str(Path(directory)/'report.json')]),0)
                self.assertEqual(check.call_count,7);run.assert_not_called()
    def test_real_bash_python_checkargs_19_jobs(self):
        bash=Path('C:/Program Files/Git/bin/bash.exe') if os.name=='nt' else Path(shutil.which('bash') or '/absent')
        if not bash.is_file():self.skipTest('Bash missing')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);conda=root/'conda/etc/profile.d';conda.mkdir(parents=True)
            (conda/'conda.sh').write_text('conda() { :; }\n')
            bindir=root/'bin';bindir.mkdir()
            def posix(path):
                value=Path(path).as_posix()
                return '/'+value[0].lower()+value[2:] if os.name=='nt' else value
            shim=bindir/'python';shim.write_text('#!/usr/bin/env bash\nexec '+shlex.quote(posix(sys.executable))+' "$@"\n');shim.chmod(0o755)
            env={**os.environ,'RSMOL_CONDA_BASE':posix(root/'conda'),'PYTHONIOENCODING':'utf-8',
                'PATH':posix(bindir)+(';' if os.name=='nt' else ':')+os.environ['PATH'],'MSYS2_ARG_CONV_EXCL':'*'}
            for p in (registry.PACKAGE/'run.sh',registry.ROOT/'run_ced_eval_20261122_4090.sh'):
                r=subprocess.run([str(bash),'-n',str(p)],capture_output=True,text=True,env=env);self.assertEqual(r.returncode,0,r.stderr)
            r=subprocess.run([str(bash),str(registry.ROOT/'run_ced_eval_20261122_4090.sh'),'all','--check-args'],
                             capture_output=True,text=True,encoding='utf-8',env=env)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(len([json.loads(line) for line in r.stdout.splitlines()]),19)

class PlainTraceTests(unittest.TestCase):
    def test_plain30_trace_and_fault_rejection(self):
        from test_text71427_ced_eval_tensors import HookModule
        import numpy as np
        for fault in (None,'skip','cache','mask'):
            lm=HookModule();lm.model=types.SimpleNamespace(layers=[HookModule() for _ in range(30)])
            audit=plain.install_trace_audit(lm)
            def forward():
                kwargs={'inputs_embeds':np.zeros((1,383,576)),'use_cache':False}
                if fault=='mask':kwargs['attention_mask']=True
                for cb,_ in lm.pre:cb(lm,(),kwargs)
                for index,layer in enumerate(lm.model.layers):
                    if fault=='skip' and index==29:continue
                    for cb in layer.post:cb(layer,(),None)
                out=types.SimpleNamespace(past_key_values=() if fault=='cache' else None,logits=np.zeros((1,383,4)))
                for cb in lm.post:cb(lm,(),out)
            if fault:
                with self.assertRaises(ValueError):forward()
            else:
                forward();self.assertEqual(audit['status'],'PASS');self.assertEqual(audit['last_schedule'],list(range(30)))

if __name__=='__main__':unittest.main()
