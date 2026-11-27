"""Exact nine-job binding, strict final-checkpoint admission and source isolation."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/RSmol'),str(ROOT/'tests')]
from mizar_core4_eval import registry,sources,submit,runtime,arguments,preflight,comparison_policy


def cache():
    return registry.SOURCE_CACHE


def synthetic(target):
    c=sources.load_contract(target)
    contract=c.semantic_contract(dict(rows=720524,rows_canonical_sha256='explicit fixture',
        mapping_sha256='explicit fixture',source_rows=c.SOURCE_COUNTS),'/fixture',{'fixture':True})
    saved=dict(schema_version=1,route_variant=c.ROUTE,mode='formal',contract=contract,
        contract_sha256=c.digest(contract),state_dict={str(i):None for i in range(454)},
        total_step=16890,num_epochs=3,epoch=3,next_optimizer_step=0)
    config=dict(route=c.ROUTE,mode='formal',contract=contract,contract_sha256=c.digest(contract),max_optimizer_steps=16890)
    return saved,config,c


class AdmissionTests(unittest.TestCase):
    def test_exact_nine_bindings_resources_and_no_reason(self):
        jobs=submit.planned(submit.parse_args(['all','--check-args']))
        self.assertEqual(len(jobs),9)
        self.assertEqual(len({str(j['resolved'].output_dir) for j in jobs}),9)
        for job in jobs:
            a=job['resolved'];cmd=submit.command(job,'fixture')
            self.assertEqual(a.checkpoint,registry.targets()[job['target']].resolve())
            self.assertEqual(a.expected_step,16890)
            self.assertEqual(a.runtime_config,a.checkpoint.parent.parent/'runtime.json')
            self.assertEqual(cmd[cmd.index('-p')+1],'pdgpu-4090')
            self.assertEqual([cmd[cmd.index(f)+1] for f in ('-c','-m','-g','-n')],['8','32G','1','1'])
            self.assertLess(len(cmd[cmd.index('-j')+1]),60)
            self.assertEqual(shlex.split(cmd[-1])[:2],['bash','mizar_core4_eval/run.sh'])
        self.assertEqual({j['benchmark'] for j in jobs},{'mmau','mmar','mmsu'})
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (['all','reasonaqa5000'],['l9_12x3'],['all','--checkpoint','x']):
                with self.assertRaises(SystemExit):submit.parse_args(argv)
            with self.assertRaises(SystemExit):
                arguments.parse_args('mmsu',['--eval-variant','l9_12x4','--checkpoint','model--step-500.ckpt'])

    def test_final_cursor_accepted_bad_contract_and_geometry_rejected(self):
        for target in registry.ROUTES:
            saved,config,c=synthetic(target)
            self.assertEqual(runtime.validate_checkpoint(saved,config,target),saved['contract'])
            self.assertEqual(c.training_cursor(16890,c.geometry(720524)),(3,0))
            for field,value in [('route_variant','old_x3'),('total_step',11260),('num_epochs',1),('epoch',2),('next_optimizer_step',5630)]:
                bad=copy.deepcopy(saved);bad[field]=value
                with self.assertRaises(ValueError):runtime.validate_checkpoint(bad,config,target)
            for field,value in [('code_sha256','bad'),('schedule',list(range(30))),('recurrence_passes',3),
                                ('added_parameters',2),('attention_mask','prefixlm'),('autocast',True),('source_new_field',1)]:
                bad=copy.deepcopy(saved);bad['contract'][field]=value
                config2=copy.deepcopy(config);config2['contract']=bad['contract']
                bad['contract_sha256']=config2['contract_sha256']=c.digest(bad['contract'])
                with self.assertRaises(ValueError):runtime.validate_checkpoint(bad,config2,target)
            for size in (453,456):
                bad=copy.deepcopy(saved);bad['state_dict']={str(i):None for i in range(size)}
                with self.assertRaises(ValueError):runtime.validate_checkpoint(bad,config,target)

    def test_real_save_read_cpu_preflight_and_changed_runtime_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for target in registry.ROUTES:
                saved,config,_=synthetic(target)
                path=root/target/'formal/checkpoints/model--step-16890.ckpt'
                path.parent.mkdir(parents=True);torch.save(saved,path)
                cfg=path.parent.parent/'runtime.json';cfg.write_text(json.dumps(config))
                a=arguments.parse_args('mmsu',['--eval-variant',target,'--checkpoint',str(path)])
                self.assertEqual(preflight.check(a)['checkpoint_step'],16890)
                self.assertFalse(hasattr(a,'_checkpoint_payload'))
                runtime.read_checkpoint(a)
                cfg.write_text(cfg.read_text()+' ')
                with self.assertRaises(ValueError):runtime.read_checkpoint(a)

    def test_cpu_preflight_once_per_model_failures_do_not_block_other_models(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)
            with patch.object(preflight,'check',side_effect=[ValueError('bad first model'),{'status':'READY_FOR_GPU'},{'status':'READY_FOR_GPU'}]) as check, \
                 patch.object(submit.subprocess,'run',side_effect=[types.SimpleNamespace(returncode=x,stdout='mock',stderr='') for x in (1,0,0,0,0,0)]) as vc:
                path=root/'submission.json'
                self.assertEqual(submit.main(['all','--submission-report',str(path)]),1)
                self.assertEqual((check.call_count,vc.call_count),(3,6))
                report=json.loads(path.read_text())
                self.assertEqual(len(report['jobs']),9)
                self.assertTrue(all(not j['vc_attempted'] for j in report['jobs'][:3]))
                self.assertEqual(sum(j['status']=='SUBMITTED' for j in report['jobs']),5)
            with patch.object(preflight,'check',return_value={'status':'READY_FOR_GPU'}) as check,patch.object(submit.subprocess,'run') as vc:
                self.assertEqual(submit.main(['all','--preflight-only','--submission-report',str(root/'preflight.json')]),0)
                self.assertEqual(check.call_count,3);vc.assert_not_called()

    def test_source_snapshot_real_pinned_git_and_tamper_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            cache_path=Path(temp)
            for target in registry.ROUTES:
                root=sources.prepare(target,cache_path)
                self.assertEqual(sources.verify(target,root),root)
                self.assertEqual(sources.profile(target)['revision'],'9575eafd0901c481fbd934cc90514f8b2554eaa3')
            with (root/'model.py').open('a') as stream:stream.write('\n# changed\n')
            with self.assertRaises(ValueError):sources.prepare(target,cache_path)

    def test_real_bash_wrapper_and_space_argument_chain(self):
        bash=Path('C:/Program Files/Git/bin/bash.exe') if sys.platform=='win32' else Path('/bin/bash')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'conda base';profile=base/'etc/profile.d/conda.sh'
            profile.parent.mkdir(parents=True)
            profile.write_text('conda() { return 0; }\npython() { "'+Path(sys.executable).as_posix()+'" "$@"; }\n')
            env={**os.environ,'RSMOL_CONDA_BASE':base.as_posix()}
            result=subprocess.run([str(bash),str(ROOT/'code/RSmol/run_mizar_core4_eval_4090.sh'),'all','--check-args'],
                env=env,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(len([json.loads(x) for x in result.stdout.splitlines()]),9)
            result=subprocess.run([str(bash),str(ROOT/'code/RSmol/run_mizar_core4_eval_4090.sh'),'l9_12x4',
                '--checkpoint','/some space/formal/checkpoints/model--step-16890.ckpt','--check-args'],
                env=env,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            records=[json.loads(x) for x in result.stdout.splitlines()]
            self.assertEqual(len(records),3)
            self.assertTrue(all('some space' in r['binding']['checkpoint'] for r in records))

    def test_baseline_common_protocol_with_native_30_execution(self):
        from mizar_core3_eval import runtime as baseline
        from mizar_core4_eval import mmsu_runtime
        for benchmark in registry.BENCHMARKS:
            base=mmsu_runtime.PROTOCOL if benchmark=='mmsu' else runtime.PROTOCOL
            key='protocol' if benchmark=='mmsu' else 'generation'
            identity={'eval_variant':'official_mizar',key:baseline.protocol_for('official_mizar',base)}
            self.assertEqual(comparison_policy.common_protocol(identity,benchmark),base)
            self.assertEqual(comparison_policy.execution_columns(identity)['block_executions'],30)
            for target in registry.ROUTES:
                new={'eval_variant':target,key:runtime.protocol_for(target,base)}
                self.assertEqual(comparison_policy.common_protocol(new,benchmark),base)
                self.assertEqual(comparison_policy.execution_columns(new)['block_executions'],42)
                new[key]['prompt_eos_suffix']='changed'
                with self.assertRaises(ValueError):comparison_policy.common_protocol(new,benchmark)

    def test_reference_protocol_functions_and_all_old_source_bytes_unchanged(self):
        baseline=json.loads((ROOT/'code/RSmol/mizar_core4_eval/source_guard.json').read_text())
        for name,sha in baseline.items():
            if name=='RM.md':continue
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),sha,name)
        original=ROOT/'code/RSmol/mizar_core3_eval';selected=ROOT/'code/RSmol/mizar_core4_eval'
        for filename,names in {'mmsu.py':['execute','summarize','run'],
                'benchmark.py':['_fixed20_pair','_materialize','run_official_scorer'],
                'mmsu_runtime.py':['decode_audio','read_audio'],
                'runtime.py':['greedy_decode']}.items():
            def definitions(path):
                return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
            a,b=definitions(original/filename),definitions(selected/filename)
            for name in names:self.assertEqual(a[name],b[name],filename+':'+name)


if __name__=='__main__':unittest.main()
