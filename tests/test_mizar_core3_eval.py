"""Real source/CLI checks and strict synthetic checkpoint admissions."""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code/RSmol'),str(ROOT/'tests')]
from mizar_core3_eval import registry,sources,submit,runtime,arguments,preflight
from mizar_s1_ced_reasonaqa5000_eval import storage


def cache():
    return registry.SOURCE_CACHE


def synthetic(target,rows=1026,step=None):
    c = sources.load_contract(target)
    contract = c.semantic_contract(dict(rows=rows,rows_canonical_sha256='fixture',
        mapping_sha256='fixture',source_rows={'fixture':rows}),'/fixture',{'fixture':True})
    step = contract['geometry']['total_steps'] if step is None else step
    saved = dict(schema_version=1,route_variant=c.ROUTE,mode='formal',contract=contract,
        contract_sha256=c.digest(contract),state_dict={str(i):None for i in range(registry.ROUTES[target]['state_count'])},
        total_step=step,num_epochs=1,epoch=int(step==contract['geometry']['total_steps']),
        next_optimizer_step=0 if step==contract['geometry']['total_steps'] else step)
    config = dict(route=c.ROUTE,mode='formal',contract=contract,contract_sha256=c.digest(contract),
        max_optimizer_steps=contract['geometry']['total_steps'])
    return saved,config,c


class AdmissionTests(unittest.TestCase):
    def test_16_plans_resources_and_explicit_paths(self):
        commands = []
        for target in registry.ROUTES:
            argv = [target,'all','--check-args']
            if target != 'official_mizar':
                argv += ['--checkpoint','/no production assets/formal/checkpoints/model--step-500.ckpt']
            args = submit.parse_args(argv)
            jobs = submit.planned(args)
            self.assertEqual(len(jobs),4)
            self.assertEqual(len({str(j['resolved'].output_dir) for j in jobs}),4)
            for job in jobs:
                cmd = submit.command(job,'test')
                self.assertEqual(cmd[cmd.index('-p')+1],'pdgpu-4090')
                self.assertEqual([cmd[cmd.index(f)+1] for f in ('-c','-m','-g','-n')],['8','32G','1','1'])
                self.assertLess(len(cmd[cmd.index('-j')+1]),60)
                parts = shlex.split(cmd[-1])
                self.assertEqual(parts[:2],['bash','mizar_core3_eval/run.sh'])
                if target != 'official_mizar':
                    self.assertEqual(parts[parts.index('--checkpoint')+1],str(args.checkpoint))
                commands.append(cmd)
        self.assertEqual(len(commands),16)
        with contextlib.redirect_stderr(io.StringIO()):
            for target in ('l9_12x3','l14_17x3','l19_22x3'):
                with self.assertRaises(SystemExit):
                    submit.parse_args([target])
            with self.assertRaises(SystemExit):
                arguments.parse_args('mmsu',['--eval-variant','l9_12x3','--checkpoint','bad.partial'])

    def test_all_dynamic_steps_and_final_checkpoint_accepted_without_resume_gate(self):
        for target in ('l9_12x3','l14_17x3','l19_22x3'):
            for rows in (2,128,130,651296):
                for step in (1,None):
                    saved,config,_ = synthetic(target,rows,step)
                    self.assertEqual(runtime.validate_checkpoint(saved,config,target),saved['contract'])

    def test_wrong_route_source_schedule_and_gate_inventory_rejected(self):
        for target in ('l9_12x3','l14_17x3','l19_22x3'):
            saved,config,c = synthetic(target)
            for field,value in [('route_variant','old'),('total_step',999),('num_epochs',3),('epoch',0)]:
                changed = copy.deepcopy(saved);changed[field]=value
                with self.assertRaises(ValueError):
                    runtime.validate_checkpoint(changed,config,target)
            for field,value in [('code_sha256','bad'),('schedule',list(range(30))),
                                ('physical_layers',28),('attention_mask','prefixlm'),('autocast',True)]:
                changed = copy.deepcopy(saved);changed['contract'][field]=value
                config2 = copy.deepcopy(config);config2['contract']=changed['contract']
                changed['contract_sha256']=config2['contract_sha256']=c.digest(changed['contract'])
                with self.assertRaises(ValueError):
                    runtime.validate_checkpoint(changed,config2,target)
            changed = copy.deepcopy(saved);changed['state_dict'].pop(next(iter(changed['state_dict'])))
            with self.assertRaises(ValueError):
                runtime.validate_checkpoint(changed,config,target)

    def test_real_checkpoint_read_preflight_including_completed_final(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for target in ('l9_12x3','l14_17x3','l19_22x3'):
                saved,config,c = synthetic(target)
                path = root/target/'formal/checkpoints/model--step-9.ckpt'
                path.parent.mkdir(parents=True)
                torch.save(saved,path)
                (path.parent.parent/'runtime.json').write_text(json.dumps(config))
                args = arguments.parse_args('mmsu',['--eval-variant',target,'--checkpoint',str(path)])
                result = preflight.check(args)
                self.assertEqual(result['status'],'READY_FOR_GPU')
                self.assertEqual(result['checkpoint_step'],9)
                self.assertFalse(hasattr(args,'_checkpoint_payload'))
                args.expected_step=8
                with self.assertRaises(ValueError):
                    preflight.check(args)

    def test_submit_preflight_once_failed_admission_no_gpu_and_submission_failure_continues(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            argv=['l14_17x3','all','--checkpoint',str(root/'formal/checkpoints/model--step-5.ckpt'),
                  '--submission-report',str(root/'submit.json')]
            with patch.object(preflight,'check',return_value={'status':'READY_FOR_GPU'}) as check, \
                 patch.object(submit.subprocess,'run',side_effect=[
                    types.SimpleNamespace(returncode=i,stdout='mock',stderr='') for i in (1,0,0,0)]) as vc, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(submit.main(argv),1)
                self.assertEqual(check.call_count,1);self.assertEqual(vc.call_count,4)
            argv[-1]=str(root/'failed.json')
            with patch.object(preflight,'check',side_effect=ValueError('wrong route')), \
                 patch.object(submit.subprocess,'run') as vc,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(submit.main(argv),1);vc.assert_not_called()
            self.assertTrue(all(not j['vc_attempted'] for j in json.loads((root/'failed.json').read_text())['jobs']))

    def test_snapshots_real_git_revision_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            cache_path=Path(temp)
            root=sources.prepare('l19_22x3',cache_path,revision='HEAD')
            self.assertEqual(sources.verify('l19_22x3',root),root)
            with (root/'model.py').open('a') as stream:
                stream.write('\n# changed\n')
            with self.assertRaises(ValueError):
                sources.prepare('l19_22x3',cache_path)

    def test_bash_wrapper_argument_chain(self):
        bash=Path('C:/Program Files/Git/bin/bash.exe') if sys.platform=='win32' else Path('/bin/bash')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'conda base';profile=base/'etc/profile.d/conda.sh'
            profile.parent.mkdir(parents=True)
            python_path=Path(sys.executable).as_posix()
            profile.write_text('conda() { return 0; }\npython() { "'+python_path+'" "$@"; }\n')
            import os
            env={**os.environ,'RSMOL_CONDA_BASE':base.as_posix()}
            command=[str(bash),str(ROOT/'code/RSmol/run_mizar_core3_eval_4090.sh'),'l9_12x3','all',
                '--checkpoint','/some space/formal/checkpoints/model--step-42.ckpt','--check-args']
            result=subprocess.run(command,env=env,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            records=[json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(records),4)
            self.assertTrue(all(r['binding']['expected_step']==42 for r in records))

    def test_old_sources_and_reference_generation_unchanged(self):
        baseline=json.loads((ROOT/'code/RSmol/mizar_core3_eval/source_guard.json').read_text())
        for name,sha in baseline.items():
            if name=='RM.md':continue
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),sha,name)
        original=ROOT/'code/RSmol/ced_eval_20261125'
        selected=ROOT/'code/RSmol/mizar_core3_eval'
        for filename,names in {
            'reasonaqa.py':['execute','summarize','run'],
            'mmsu.py':['execute','summarize','run'],
            'benchmark.py':['_fixed20_pair','_materialize','run_official_scorer'],
            'reasonaqa_runtime.py':['generate','fixed20_pair'],
            'mmsu_runtime.py':['decode_audio','read_audio']}.items():
            def definitions(path):
                return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            a,b=definitions(original/filename),definitions(selected/filename)
            for name in names:self.assertEqual(a[name],b[name],filename+':'+name)


if __name__=='__main__':unittest.main()
