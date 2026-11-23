"""Real pinned contracts/interfaces; explicit fixtures for absent GPU dependencies."""
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
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/RSmol'),str(ROOT/'tests')]
from ced_eval_20261123 import registry,sources,runtime,preflight,submit,arguments,observers,interfaces
from ced_eval_20261123 import reasonaqa,mmsu,benchmark,reasonaqa_runtime,mmsu_runtime
from ced_eval_20261123.targets import target_records
from ced_eval_20261122 import registry as old_registry

_cache=None
def cache():
    global _cache
    if _cache is None:
        _cache=tempfile.TemporaryDirectory(prefix='ced23_sources_');atexit.register(_cache.cleanup)
        for target in registry.ROUTES:sources.prepare(target,_cache.name)
    return Path(_cache.name)

def checkpoint_fixture(target,step=34626):
    c=sources.load_contract(target,cache())
    prepared=dict(rows=1477492,rows_canonical_sha256='rows',mapping_sha256='mapping',source_rows=dict(c.SOURCE_COUNTS))
    contract=c.semantic_contract(prepared,'/source/text','/source/ced')
    contract['recursive_text_checkpoint']={'fixture':True}
    saved=dict(schema_version=1,route_variant=c.ROUTE,mode='formal',contract=contract,
        contract_sha256=c.digest(contract),state_dict={'fixture':True},total_step=step,num_epochs=3,
        loss_reduction=runtime.reference.TRAINING_LOSS_REDUCTION,
        batch_geometry=dict(world_size=8,microbatch_size=4,gradient_accumulation_steps=4,global_batch_size=128))
    config=dict(mode='formal',route=c.ROUTE,contract=contract,contract_sha256=c.digest(contract))
    return c,saved,config

def function(path,name):
    return next(n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
                if isinstance(n,ast.FunctionDef) and n.name==name)

class ContractTests(unittest.TestCase):
    def test_two_real_snapshots_and_namespace_switching(self):
        for target in ('mesh_71437','audio7_read2','mesh_71437'):
            root=sources.verify(target,cache());c=sources.load_contract(target,cache())
            spec=registry.variant_spec(target)
            self.assertEqual(c.code_digest(),sources.profile(target)[1]['training_code_sha256'][target])
            self.assertEqual(tuple(c.TEXT_MODEL_SCHEDULE),spec['schedule'])
            self.assertEqual(Path(sys.modules[spec['text_package']+'.contract'].__file__).resolve(),root.parents[1]/spec['text_package']/'contract.py')
        self.assertNotEqual(registry.SOURCE_CACHE,old_registry.SOURCE_CACHE)
    def test_actual_contracts_intermediate_final_and_cross_route_rejection(self):
        for target in registry.ROUTES:
            for step in (5771,34626):
                c,s,r=checkpoint_fixture(target,step)
                self.assertIs(runtime.validate_checkpoint(s,r,c,step,(8,4,4),eval_variant=target),s['contract'])
                other=next(k for k in registry.ROUTES if k!=target)
                with self.assertRaises(ValueError):runtime.validate_checkpoint(s,r,sources.load_contract(other,cache()),step,(8,4,4),eval_variant=other)
    def test_contract_precision_architecture_branch_source_and_initialization_rejected(self):
        for target in registry.ROUTES:
            c,s,r=checkpoint_fixture(target)
            for key,value in [('forward_dtype','bfloat16'),('text_model_schedule',list(range(56))),
                ('code_sha256','wrong'),('recursive_model_source_sha256','wrong'),('attention_mask','mask'),
                ('audio_policy','first10'),('audio_initialization',{}),('loop_lora',{'wrong':True})]:
                bad=copy.deepcopy(s);bad['contract'][key]=value
                bad['contract_sha256']=c.digest(bad['contract']);cfg={**r,'contract':bad['contract'],'contract_sha256':bad['contract_sha256']}
                with self.subTest(target=target,key=key),self.assertRaises(ValueError):
                    runtime.validate_checkpoint(bad,cfg,c,34626,(8,4,4),eval_variant=target)
    def test_cpu_preflight_checks_pinned_signatures_once_without_cuda(self):
        for target in registry.ROUTES:
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'formal/checkpoints/model--step-34626.ckpt';path.parent.mkdir(parents=True);path.write_bytes(b'fixture')
                c,s,r=checkpoint_fixture(target);(path.parent.parent/'runtime.json').write_text(json.dumps(r))
                args=arguments.parse_args('mmsu',['--eval-variant',target,'--checkpoint',str(path),'--sources-cache',str(cache())])
                loader=Mock(return_value=s)
                with patch.dict(sys.modules,{'torch':types.SimpleNamespace(load=loader)}):result=preflight.check(args)
                self.assertEqual(result['status'],'READY_FOR_GPU');self.assertEqual(result['implementation_interfaces']['status'],'PASS')
                loader.assert_called_once();self.assertTrue(loader.call_args.kwargs['mmap'])
    def test_snapshot_tampering_extra_source_and_cached_concurrency(self):
        root=sources.route_root('audio7_read2',cache());path=root/'contract.py';before=path.read_bytes()
        try:
            path.write_bytes(before+b'\n# changed\n')
            with self.assertRaises(ValueError):sources.prepare('audio7_read2',cache())
        finally:path.write_bytes(before)
        extra=root/'extra.py'
        try:
            extra.write_text('# extra')
            with self.assertRaises(ValueError):sources.verify('audio7_read2',cache())
        finally:extra.unlink()
        with patch.object(sources,'output_lock',side_effect=AssertionError('cached readers need no write lock')):
            for target in registry.ROUTES:sources.prepare(target,cache())

class ParityTests(unittest.TestCase):
    def test_existing_sources_only_two_authorized_audit_fixes(self):
        baseline=json.loads((ROOT/'audit_reports/ced_eval_20261123_validation.json').read_text())['existing_sources_before']
        expected={'code/RSmol/ced_eval_20261122/runtime.py','code/RSmol/text71427_ced_eval/runtime.py'}
        changed={name for name,wanted in baseline.items() if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=wanted}
        self.assertEqual(changed,expected)
        for name in expected:
            current=(ROOT/name).read_text();original=subprocess.check_output(['git','show','HEAD:'+name],text=True,encoding='utf-8')
            before=ast.parse(original);after=ast.parse(current)
            def without_audit(tree):return ast.dump(ast.Module(body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='audit_model'],type_ignores=[]))
            self.assertEqual(without_audit(before),without_audit(after))
    def test_protocol_core_preprocessing_generation_scoring_and_runners_unchanged(self):
        from ced_eval_20261122 import runtime as previous,reasonaqa_runtime as pr,mmsu_runtime as pm
        self.assertEqual(runtime.PROTOCOL,previous.PROTOCOL);self.assertEqual(reasonaqa_runtime.PROTOCOL,pr.PROTOCOL);self.assertEqual(mmsu_runtime.PROTOCOL,pm.PROTOCOL)
        pairs={'runtime.py':['normalize_state','greedy_decode','sample_context','validate_context','generate_prefix_inference'],
            'reasonaqa_runtime.py':['generate','waveform_sha','fixed20_pair','prefix_and_decode'],
            'mmsu_runtime.py':['read_audio','decode_audio','generate','prefix_and_decode'],
            'reasonaqa.py':['execute','run','summarize'],'mmsu.py':['execute','run','summarize'],
            'benchmark.py':['generate','make_mmau_audio_decoder','_fixed20_pair','_run','run_official_scorer']}
        for file,names in pairs.items():
            for name in names:self.assertEqual(ast.dump(function(registry.PACKAGE/file,name)),ast.dump(function(registry.ROOT/'ced_eval_20261122'/file,name)),(file,name))
    def test_latest_mmsu_prompt_actual_mmau_builder_no_gt_leak(self):
        from test_mizar_s1_ced_mmsu_eval import record
        from mizar_s1_ced_mmsu_eval import dataset
        row=record();before=copy.deepcopy(row)
        expected=benchmark.mmau.build_fixed_order_prompt(row['question'],[row[k] for k in dataset.CHOICES]).lower()
        self.assertEqual(mmsu.dataset.prompt(row),expected)
        self.assertEqual(expected,'which country? a) india b) none c) china d) japan');self.assertEqual(row,before)
        row['answer_gt']='NEVER_INCLUDE';self.assertEqual(mmsu.dataset.prompt(row),expected)
    def test_python310_and_shell_lf(self):
        for path in registry.PACKAGE.glob('*.py'):ast.parse(path.read_text(),feature_version=(3,10))
        for path in (registry.PACKAGE/'run.sh',registry.ROOT/'run_ced_eval_20261123_4090.sh'):self.assertNotIn(b'\r',path.read_bytes())

class SubmissionTests(unittest.TestCase):
    def test_exact_eight_paths_4090_resource_context_and_new_outputs(self):
        jobs=submit.planned(submit.parse_args(['--check-args']))
        self.assertEqual(len(jobs),8);self.assertEqual(len({str(j['resolved'].output_dir) for j in jobs}),8)
        self.assertEqual([sum(j['benchmark']==b for j in jobs) for b in registry.BENCHMARKS],[2]*4)
        for j in jobs:
            args=j['resolved'];cmd=submit.command(j,'fixture')
            self.assertEqual(args.checkpoint,Path(target_records()[j['target']]['checkpoint']).resolve());self.assertEqual(args.expected_step,34626)
            self.assertIn('ced_eval_20261123_v1',args.output_dir.parts)
            if j['benchmark']=='mmsu':self.assertIn('mmsu_compact_lowercase_v1',args.output_dir.parts)
            for flag,value in (('-p','pdgpu-4090'),('-g','1'),('-c','8'),('-m','32G'),('-n','1')):self.assertEqual(cmd[cmd.index(flag)+1],value)
            self.assertLess(len(cmd[cmd.index('-j')+1]),60)
        self.assertEqual(len(submit.planned(submit.parse_args(['audio7_read2']))),4)
        self.assertEqual(len(submit.planned(submit.parse_args(['mesh_71437','mmsu']))),1)
    def test_cpu_failed_target_no_gpu_other_target_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json';seen=[]
            def check(args):
                seen.append(args.eval_variant)
                if args.eval_variant=='audio7_read2':raise ValueError('fixture interface error')
                return {'status':'READY_FOR_GPU'}
            with patch.object(preflight,'check',side_effect=check),patch.object(submit.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='MOCK ONLY',stderr='')) as run,redirect_stdout(io.StringIO()):
                self.assertEqual(submit.main(['--submission-report',str(path)]),1)
            self.assertEqual(seen,list(registry.ROUTES));self.assertEqual(run.call_count,4)
            data=json.loads(path.read_text());self.assertEqual(sum(j['status']=='SUBMITTED' for j in data['jobs']),4)
            self.assertTrue(all(not j['vc_attempted'] for j in data['jobs'] if j['target']=='audio7_read2'))
    def test_vc_failure_continues_and_preflight_only_never_submits(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'submit.json'
            results=[types.SimpleNamespace(returncode=int(i==0),stdout='MOCK ONLY',stderr='') for i in range(8)]
            with patch.object(preflight,'check',return_value={'status':'READY_FOR_GPU'}) as check,patch.object(submit.subprocess,'run',side_effect=results) as run,redirect_stdout(io.StringIO()):
                self.assertEqual(submit.main(['--submission-report',str(path)]),1);self.assertEqual(run.call_count,8);self.assertEqual(check.call_count,2)
                run.reset_mock();check.reset_mock()
                self.assertEqual(submit.main(['--preflight-only','--submission-report',str(Path(directory)/'preflight.json')]),0)
                run.assert_not_called();self.assertEqual(check.call_count,2)
    def test_real_bash_python_all_and_eight_single_commands(self):
        bash=Path('C:/Program Files/Git/bin/bash.exe') if os.name=='nt' else Path(shutil.which('bash') or '/absent')
        if not bash.is_file():self.skipTest('Bash missing')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);conda=root/'conda/etc/profile.d';conda.mkdir(parents=True)
            (conda/'conda.sh').write_text('conda() { :; }\n');bindir=root/'bin';bindir.mkdir()
            def posix(path):
                v=Path(path).as_posix();return '/'+v[0].lower()+v[2:] if os.name=='nt' else v
            shim=bindir/'python';shim.write_text('#!/usr/bin/env bash\nexec '+shlex.quote(posix(sys.executable))+' "$@"\n');shim.chmod(0o755)
            env={**os.environ,'RSMOL_CONDA_BASE':posix(root/'conda'),'PYTHONIOENCODING':'utf-8',
                'PATH':posix(bindir)+(';' if os.name=='nt' else ':')+os.environ['PATH'],'MSYS2_ARG_CONV_EXCL':'*'}
            for path in (registry.PACKAGE/'run.sh',registry.ROOT/'run_ced_eval_20261123_4090.sh'):
                r=subprocess.run([str(bash),'-n',str(path)],capture_output=True,text=True,env=env);self.assertEqual(r.returncode,0,r.stderr)
            for argv in [['all'],*[[target,b] for target in registry.ROUTES for b in registry.BENCHMARKS]]:
                r=subprocess.run([str(bash),str(registry.ROOT/'run_ced_eval_20261123_4090.sh'),*argv,'--check-args'],capture_output=True,text=True,encoding='utf-8',env=env)
                self.assertEqual(r.returncode,0,r.stderr);jobs=[json.loads(line) for line in r.stdout.splitlines()]
                self.assertEqual(len(jobs),8 if argv==['all'] else 1)

if __name__=='__main__':unittest.main()
