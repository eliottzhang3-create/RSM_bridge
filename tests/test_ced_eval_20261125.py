"""Offline contracts, exact source preservation and the eight explicit bindings."""
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
from ced_eval_20261125 import registry,sources,runtime,preflight,submit,arguments,interfaces,comparison_policy
from ced_eval_20261125 import reasonaqa,mmsu,benchmark,reasonaqa_runtime,mmsu_runtime
from ced_eval_20261125.targets import target_records

_cache=None
def cache():
    global _cache
    if _cache is None:
        _cache=tempfile.TemporaryDirectory(prefix='ced25_sources_');atexit.register(_cache.cleanup)
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
    def test_two_pinned_snapshots_actual_contract_digests_and_shared_imports(self):
        for target in ('audio8_read3','hrm_prefixlm','audio8_read3'):
            root=sources.verify(target,cache());c=sources.load_contract(target,cache());spec=registry.variant_spec(target)
            self.assertEqual(c.code_digest(),sources.profile(target)[1]['training_code_sha256'][target])
            self.assertEqual(tuple(c.TEXT_MODEL_SCHEDULE),spec['schedule'])
            self.assertEqual(c.TEXT_MODEL_CONTRACT,spec['text_contract'])
            for package in spec['shared_packages']:
                self.assertEqual(Path(sys.modules[package+'.contract'].__file__).resolve(),root.parents[1]/package/'contract.py')
    def test_actual_contracts_intermediate_final_and_cross_route_rejection(self):
        for target in registry.ROUTES:
            for step in (5771,34626):
                c,s,r=checkpoint_fixture(target,step)
                self.assertIs(runtime.validate_checkpoint(s,r,c,step,(8,4,4),eval_variant=target),s['contract'])
                other=next(k for k in registry.ROUTES if k!=target)
                with self.assertRaises(ValueError):runtime.validate_checkpoint(s,r,sources.load_contract(other,cache()),step,(8,4,4),eval_variant=other)
    def test_wrong_native_mask_bp_branch_source_and_precision_are_rejected(self):
        for target in registry.ROUTES:
            c,s,r=checkpoint_fixture(target)
            changes=[('forward_dtype','bfloat16'),('text_model_schedule',list(range(56))),('code_sha256','wrong'),
                ('recursive_model_source_sha256','wrong'),('audio_policy','first10'),('audio_initialization',{}),('loop_lora',{'wrong':True})]
            changes += [('attention_mask',None),('bp_steps',4),('prefix_length',382)] if target=='hrm_prefixlm' else [('attention_mask',registry.PREFIXLM_POLICY),('audio_read3',{})]
            for key,value in changes:
                bad=copy.deepcopy(s);bad['contract'][key]=value;bad['contract_sha256']=c.digest(bad['contract'])
                cfg={**r,'contract':bad['contract'],'contract_sha256':bad['contract_sha256']}
                with self.subTest(target=target,key=key),self.assertRaises(ValueError):runtime.validate_checkpoint(bad,cfg,c,34626,(8,4,4),eval_variant=target)
    def test_preflight_actual_signatures_cpu_header_and_no_model_construction(self):
        for target in registry.ROUTES:
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'formal/checkpoints/model--step-34626.ckpt';path.parent.mkdir(parents=True);path.write_bytes(b'fixture')
                c,s,r=checkpoint_fixture(target);(path.parent.parent/'runtime.json').write_text(json.dumps(r))
                args=arguments.parse_args('mmsu',['--eval-variant',target,'--checkpoint',str(path),'--sources-cache',str(cache())])
                loader=Mock(return_value=s)
                with patch.dict(sys.modules,{'torch':types.SimpleNamespace(load=loader)}):result=preflight.check(args)
                self.assertEqual(result['status'],'READY_FOR_GPU');self.assertEqual(result['implementation_interfaces']['status'],'PASS')
                loader.assert_called_once();self.assertTrue(loader.call_args.kwargs['mmap'])
    def test_tampered_snapshot_and_cached_parallel_readers(self):
        root=sources.route_root('audio8_read3',cache());path=root/'contract.py';before=path.read_bytes()
        try:
            path.write_bytes(before+b'\n# changed\n')
            with self.assertRaises(ValueError):sources.prepare('audio8_read3',cache())
        finally:path.write_bytes(before)
        extra=root/'extra.py'
        try:
            extra.write_text('# extra')
            with self.assertRaises(ValueError):sources.verify('audio8_read3',cache())
        finally:extra.unlink()
        with patch.object(sources,'output_lock',side_effect=AssertionError('cached readers need no writer lock')):
            for target in registry.ROUTES:sources.prepare(target,cache())

class ParityTests(unittest.TestCase):
    def test_all_existing_production_sources_preserved(self):
        baseline=json.loads((ROOT/'audit_reports/ced_eval_20261125_validation.json').read_text())['existing_sources_before']
        concurrent=json.loads((ROOT/'audit_reports/ced_eval_20261125_concurrent_sources.json').read_text())['files']
        self.assertEqual(set(concurrent),{'code/RSmol/mizar_audio_cutoff1000/identity.py','code/RSmol/mizar_audio_cutoff1000/submit.py'})
        self.assertGreater(len(baseline),3000)
        for name,wanted in baseline.items():
            if name in concurrent:
                self.assertEqual(wanted,concurrent[name]['before_lf_sha256'])
                wanted=concurrent[name]['after_lf_sha256']
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),wanted,name)
    def test_generation_preprocessing_scoring_functions_and_causal_loop_unchanged(self):
        pairs={'runtime.py':['normalize_state','generate_prefix_inference'],
            'reasonaqa_runtime.py':['generate','waveform_sha','fixed20_pair','prefix_and_decode'],
            'mmsu_runtime.py':['read_audio','decode_audio','generate','prefix_and_decode'],
            'reasonaqa.py':['execute','summarize'],'mmsu.py':['execute','summarize'],
            'benchmark.py':['make_mmau_audio_decoder','_fixed20_pair','run_official_scorer','_prepare_mmar','_metadata_records']}
        for file,names in pairs.items():
            for name in names:self.assertEqual(ast.dump(function(registry.PACKAGE/file,name)),ast.dump(function(registry.ROOT/'ced_eval_20261123'/file,name)),(file,name))
        new=function(registry.PACKAGE/'runtime.py','_causal_greedy_decode');new.name='greedy_decode'
        self.assertEqual(ast.dump(new),ast.dump(function(registry.ROOT/'ced_eval_20261123/runtime.py','greedy_decode')))
    def test_latest_mmsu_compact_prompt_unchanged(self):
        from test_mizar_s1_ced_mmsu_eval import record
        row=record();expected=benchmark.mmau.build_fixed_order_prompt(row['question'],[row[k] for k in mmsu.dataset.CHOICES]).lower()
        self.assertEqual(mmsu.dataset.prompt(row),expected);self.assertEqual(expected,'which country? a) india b) none c) china d) japan')
    def test_comparison_keeps_common_protocol_but_validates_native_attention(self):
        for task in registry.BENCHMARKS:
            base=reasonaqa_runtime.PROTOCOL if task=='reasonaqa5000' else mmsu_runtime.PROTOCOL if task=='mmsu' else runtime.PROTOCOL
            key='generation' if task in ('mmau','mmar') else 'protocol'
            common=[]
            for variant in registry.ROUTES:
                identity={'eval_variant':variant,key:runtime.protocol_for(variant,base)}
                common.append(comparison_policy.common_protocol(identity,task))
                identity[key]['max_new_tokens']=32
                with self.assertRaises(ValueError):comparison_policy.common_protocol(identity,task)
            self.assertEqual(*common)
            legacy={'eval_variant':'mesh_71437',key:copy.deepcopy(base)}
            self.assertEqual(comparison_policy.common_protocol(legacy,task),common[0])
    def test_python310_and_lf_shells(self):
        for path in registry.PACKAGE.glob('*.py'):ast.parse(path.read_text(),feature_version=(3,10))
        for path in (registry.PACKAGE/'run.sh',registry.ROOT/'run_ced_eval_20261125_4090.sh'):self.assertNotIn(b'\r',path.read_bytes())

class SubmissionTests(unittest.TestCase):
    def test_exact_eight_user_paths_4090_resources_and_native_protocols(self):
        jobs=submit.planned(submit.parse_args(['--check-args']));self.assertEqual(len(jobs),8)
        self.assertEqual(len({str(j['resolved'].output_dir) for j in jobs}),8)
        for job in jobs:
            args=job['resolved'];cmd=submit.command(job,'fixture')
            self.assertEqual(args.checkpoint,Path(target_records()[job['target']]['checkpoint']).resolve());self.assertEqual(args.expected_step,34626)
            self.assertIn('ced_eval_20261125_v1',args.output_dir.parts)
            for flag,value in (('-p','pdgpu-4090'),('-g','1'),('-c','8'),('-m','32G'),('-n','1')):self.assertEqual(cmd[cmd.index(flag)+1],value)
            self.assertLess(len(cmd[cmd.index('-j')+1]),60)
    def test_cpu_failure_no_gpu_other_target_continues_and_preflight_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'
            def check(args):
                if args.eval_variant=='hrm_prefixlm':raise ValueError('fixture interface mismatch')
                return {'status':'READY_FOR_GPU'}
            with patch.object(preflight,'check',side_effect=check) as checks,patch.object(submit.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='MOCK ONLY',stderr='')) as run,redirect_stdout(io.StringIO()):
                self.assertEqual(submit.main(['--submission-report',str(path)]),1);self.assertEqual(run.call_count,4);self.assertEqual(checks.call_count,2)
                run.reset_mock()
                self.assertEqual(submit.main(['--preflight-only','--submission-report',str(Path(directory)/'pre.json')]),1);run.assert_not_called()
    def test_real_bash_python_all_and_all_eight_single_commands(self):
        bash=Path('C:/Program Files/Git/bin/bash.exe') if os.name=='nt' else Path(shutil.which('bash') or '/absent')
        if not bash.is_file():self.skipTest('Bash missing')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);conda=root/'conda/etc/profile.d';conda.mkdir(parents=True);(conda/'conda.sh').write_text('conda() { :; }\n')
            bindir=root/'bin';bindir.mkdir()
            def posix(path):
                v=Path(path).as_posix();return '/'+v[0].lower()+v[2:] if os.name=='nt' else v
            shim=bindir/'python';shim.write_text('#!/usr/bin/env bash\nexec '+shlex.quote(posix(sys.executable))+' "$@"\n');shim.chmod(0o755)
            env={**os.environ,'RSMOL_CONDA_BASE':posix(root/'conda'),'PYTHONIOENCODING':'utf-8',
                'PATH':posix(bindir)+(';' if os.name=='nt' else ':')+os.environ['PATH'],'MSYS2_ARG_CONV_EXCL':'*'}
            for argv in [['all'],*[[target,b] for target in registry.ROUTES for b in registry.BENCHMARKS]]:
                r=subprocess.run([str(bash),str(registry.ROOT/'run_ced_eval_20261125_4090.sh'),*argv,'--check-args'],capture_output=True,text=True,encoding='utf-8',env=env)
                self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(len([json.loads(line) for line in r.stdout.splitlines()]),8 if argv==['all'] else 1)

if __name__=='__main__':unittest.main()
