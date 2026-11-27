"""Exact target/source admission and real CPU execution of the gated model."""
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
from mizar_l10_14x3_eval import registry,sources,arguments,runtime,submit,preflight


def synthetic(rows=325300):
    c=sources.load_contract(registry.TARGET)
    counts=dict(c.SOURCE_COUNTS);counts['SpeechCraft']-=c.TOTAL_ROWS-rows
    views={'audiomcq_mcq':257892,'audiomcq_cot':rows-257892}
    prepared=dict(rows=rows,source_rows=counts,rows_canonical_sha256='a'*64,mapping_sha256='b'*64,
        dataset_selection=dict(seal_sha256='c'*64,historical_inputs={'fixture':True},selected_counts=views))
    contract=c.semantic_contract(prepared,'/fixture',{'fixture':True})
    cursor=c.training_cursor(2542,contract['geometry'])
    saved=dict(schema_version=1,mode='formal',route_variant=c.ROUTE,contract=contract,contract_sha256=c.digest(contract),
        state_dict={str(i):None for i in range(457)},num_epochs=1,total_step=2542,epoch=cursor[0],next_optimizer_step=cursor[1])
    config=dict(route=c.ROUTE,mode='formal',contract=contract,contract_sha256=c.digest(contract),max_optimizer_steps=contract['geometry']['total_steps'])
    return saved,config,c


class AdmissionTests(unittest.TestCase):
    def test_two_plans_exact_checkpoint_runtime_output_and_4090_resources(self):
        jobs=submit.planned(submit.parse_args(['all','--check-args']))
        self.assertEqual([j['benchmark'] for j in jobs],['mmau','mmar'])
        self.assertEqual(len({str(j['resolved'].output_dir) for j in jobs}),2)
        for job in jobs:
            a=job['resolved'];cmd=submit.command(job,'fixture')
            self.assertEqual(a.checkpoint,registry.checkpoint_path().resolve())
            self.assertEqual(a.runtime_config,a.checkpoint.parent.parent/'runtime.json')
            self.assertEqual(a.expected_step,2542)
            self.assertEqual(cmd[cmd.index('-p')+1],'pdgpu-4090')
            self.assertEqual([cmd[cmd.index(f)+1] for f in ('-c','-m','-g','-n')],['8','32G','1','1'])
            self.assertLess(len(cmd[cmd.index('-j')+1]),60)
            self.assertEqual(shlex.split(cmd[-1])[:3],['bash','mizar_l10_14x3_eval/run.sh','benchmark'])
        with contextlib.redirect_stderr(io.StringIO()):
            for task in ('reasonaqa5000','mmsu'):
                with self.assertRaises(SystemExit):submit.parse_args([task])
            with self.assertRaises(SystemExit):
                arguments.parse_args(argv=['--benchmark','mmau','--checkpoint','model--step-16890.ckpt'])

    def test_dynamic_strong_counts_final_and_intermediate_cursors_model_only(self):
        for rows,cursor in ((325300,(1,0)),(325648,(0,2542))):
            saved,config,c=synthetic(rows)
            self.assertEqual((saved['epoch'],saved['next_optimizer_step']),cursor)
            self.assertEqual(runtime.validate_checkpoint(saved,config,registry.TARGET),saved['contract'])
            self.assertNotIn('optimizer',saved)
            for field,value in [('route_variant','old_x4'),('total_step',16890),('num_epochs',3),('epoch',8)]:
                bad=copy.deepcopy(saved);bad[field]=value
                with self.assertRaises(ValueError):runtime.validate_checkpoint(bad,config,registry.TARGET)
            for field,value in [('code_sha256','bad'),('recurrence_passes',4),('schedule',list(range(30))),
                    ('loop_bias',{'shape':[4,576]}),('added_parameters',0),('attention_mask','prefixlm')]:
                bad=copy.deepcopy(saved);bad['contract'][field]=value;cfg=copy.deepcopy(config)
                cfg['contract']=bad['contract'];cfg['contract_sha256']=bad['contract_sha256']=c.digest(bad['contract'])
                with self.assertRaises(ValueError):runtime.validate_checkpoint(bad,cfg,registry.TARGET)
            for kind in ('count','weak','selection'):
                bad=copy.deepcopy(saved)
                if kind=='count':bad['state_dict'].pop('0')
                elif kind=='weak':bad['contract']['source_rows']={'reasonaqa_oneaudio':rows}
                else:bad['contract']['dataset_selection']={}
                cfg=copy.deepcopy(config);cfg['contract']=bad['contract']
                cfg['contract_sha256']=bad['contract_sha256']=c.digest(bad['contract'])
                with self.assertRaises(ValueError):runtime.validate_checkpoint(bad,cfg,registry.TARGET)

    def test_real_checkpoint_read_cpu_preflight_and_changed_runtime_refused(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'formal/checkpoints/model--step-2542.ckpt';p.parent.mkdir(parents=True)
            saved,config,_=synthetic();torch.save(saved,p)
            cfg=p.parent.parent/'runtime.json';cfg.write_text(json.dumps(config))
            a=arguments.parse_args(argv=['--benchmark','mmau','--checkpoint',str(p)])
            result=preflight.check(a)
            self.assertEqual((result['status'],result['checkpoint_step']),('READY_FOR_GPU',2542))
            self.assertFalse(hasattr(a,'_checkpoint_payload'))
            runtime.read_checkpoint(a);cfg.write_text(cfg.read_text()+' ')
            with self.assertRaises(ValueError):runtime.read_checkpoint(a)

    def test_cpu_failure_no_gpu_vc_failure_continues_and_preflight_only(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)
            with patch.object(preflight,'check',side_effect=ValueError('bad contract')) as check,patch.object(submit.subprocess,'run') as vc:
                self.assertEqual(submit.main(['all','--submission-report',str(root/'bad.json')]),1)
                self.assertEqual(check.call_count,1);vc.assert_not_called()
            with patch.object(preflight,'check',return_value={'status':'READY_FOR_GPU'}) as check, \
                 patch.object(submit.subprocess,'run',side_effect=[types.SimpleNamespace(returncode=i,stdout='mock',stderr='') for i in (1,0)]) as vc:
                self.assertEqual(submit.main(['all','--submission-report',str(root/'submit.json')]),1)
                self.assertEqual((check.call_count,vc.call_count),(1,2))
            with patch.object(preflight,'check',return_value={'status':'READY_FOR_GPU'}),patch.object(submit.subprocess,'run') as vc:
                self.assertEqual(submit.main(['all','--preflight-only','--submission-report',str(root/'cpu.json')]),0);vc.assert_not_called()

    def test_immutable_snapshot_from_real_git_pinned_revision_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root=sources.prepare(registry.TARGET,Path(temp))
            self.assertEqual(sources.verify(registry.TARGET,root),root)
            self.assertEqual(sources.load_contract(registry.TARGET,Path(temp)).code_digest(),sources.profile(registry.TARGET)['code_sha256'])
            (root/'model.py').write_text((root/'model.py').read_text()+'\n# altered\n')
            with self.assertRaises(ValueError):sources.prepare(registry.TARGET,Path(temp))

    def test_actual_bash_entry_and_space_quoting(self):
        bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else '/bin/bash'
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)/'conda base';profile=base/'etc/profile.d/conda.sh';profile.parent.mkdir(parents=True)
            profile.write_text('conda() { return 0; }\npython() { "'+Path(sys.executable).as_posix()+'" "$@"; }\n')
            env={**os.environ,'RSMOL_CONDA_BASE':base.as_posix()}
            for task,count in (('all',2),('mmau',1),('mmar',1)):
                result=subprocess.run([bash,str(ROOT/'code/RSmol/run_mizar_l10_14x3_eval_4090.sh'),task,
                    '--checkpoint','/space path/formal/checkpoints/model--step-2542.ckpt','--check-args'],
                    cwd=ROOT,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                records=[json.loads(s) for s in result.stdout.splitlines()];self.assertEqual(len(records),count)
                self.assertTrue(all('space path' in r['binding']['checkpoint'] for r in records))

    def test_reference_numeric_protocol_and_existing_files_unchanged(self):
        baseline=json.loads((registry.PACKAGE/'source_guard.json').read_text())
        for name,sha in baseline.items():
            if name=='RM.md':continue
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),sha,name)
        for filename,names in {'runtime.py':['greedy_decode'],
                'benchmark.py':['generate','_fixed20_pair','make_mmau_audio_decoder','_materialize','run_official_scorer']}.items():
            def defs(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
            a,b=defs(registry.PACKAGE/filename),defs(ROOT/'code/RSmol/mizar_core4_eval'/filename)
            for name in names:self.assertEqual(a[name],b[name],name)


if __name__=='__main__':unittest.main()
