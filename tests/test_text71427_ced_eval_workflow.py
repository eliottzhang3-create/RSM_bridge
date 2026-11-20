"""Full-sized real runners; model/audio/parquet are explicit fixtures, not GPU evidence."""
import contextlib
import io
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/RSmol'),str(ROOT/'tests')]
from text71427_ced_eval import registry, reasonaqa, mmsu, benchmark, runtime, compare, submit
from text71427_ced_eval import reasonaqa_runtime, mmsu_runtime
from test_reasonaqa5000_eval import Fixture as OriginalReasonFixture, generation
from test_mizar_s1_ced_mmsu_eval import record as mmsu_record, generation as mmsu_generation, SCORER
import test_mizar_s1_ced_multilayer_eval_workflow as original_benchmark_fixture


def target(directory, alias, runner, extra=()):
    checkpoint=directory/alias/'formal/checkpoints/model--step-5771.ckpt'
    checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_bytes(b'explicit synthetic checkpoint')
    (checkpoint.parent.parent/'runtime.json').write_text('{"fixture":true}',encoding='utf-8')
    return runner.parse_args(['--eval-variant',alias,'--checkpoint',str(checkpoint),*extra])


class ReasonWorkflowTests(OriginalReasonFixture):
    def new_args(self,alias='base'):
        extra=['--manifest',str(self.bundle/'manifest.json')]
        for flag,value in self.roots.items():extra+=['--'+flag,str(value)]
        return target(self.directory,alias,reasonaqa,extra)

    def run_new(self,args,callback=None,forbid_loader=False):
        loaded=(None,None,None,None,None,{'status':'PASS','fixture':True},None)
        with patch.object(runtime,'read_checkpoint',return_value={}), \
             patch.object(reasonaqa_runtime,'load_runtime',return_value=loaded,
                          side_effect=AssertionError('model reload forbidden') if forbid_loader else None) as loader, \
             patch.object(reasonaqa_runtime,'generate',side_effect=callback or (lambda _,r,**kw:generation(r))), \
             patch.object(os,'fsync'),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            report=reasonaqa.run(args)
        return report,loader.call_count

    def test_six_complete_5000_runs_score_compare_and_no_model_rescore(self):
        directories=[]
        for alias in registry.ROUTES:
            args=self.new_args(alias)
            def generate(_,r,**kw):
                return generation(r,'B' if r['id']==self.manifest['sample_ids'][0] else None)
            report,calls=self.run_new(args,generate)
            self.assertEqual(report['status'],'PASS',report['fatal_error'])
            self.assertEqual(calls,1)
            self.assertEqual(report['accuracy']['overall'],{'correct':4999,'total':5000,'accuracy':4999/5000})
            before=(args.output_dir/'raw_generations.jsonl').read_bytes()
            again,calls=self.run_new(args,forbid_loader=True)
            self.assertEqual(again['status'],'PASS',again['fatal_error']);self.assertEqual(calls,0)
            self.assertEqual(before,(args.output_dir/'raw_generations.jsonl').read_bytes())
            compare.read_run(args.output_dir)
            directories.append((alias,args.output_dir))
        rows=compare.compare_runs(directories,self.directory/'compare.csv')
        self.assertEqual(len(rows),6)

    def test_interrupted_13_resumes_only_remaining_4987_and_refuses_changed_identity(self):
        args=self.new_args('audio7_read2_ar6');seen=[]
        def interrupt(_,r,**kw):
            if len(seen)==13:raise RuntimeError('fixture interruption')
            seen.append(r['id']);return generation(r)
        failed,_=self.run_new(args,interrupt)
        self.assertEqual(failed['status'],'FAILED');self.assertEqual(failed['records']['generated'],13)
        raw=args.output_dir/'raw_generations.jsonl';before=raw.read_bytes()
        remaining=[]
        def resume(_,r,**kw):remaining.append(r['id']);return generation(r)
        passed,_=self.run_new(args,resume)
        self.assertEqual(passed['status'],'PASS',passed['fatal_error'])
        self.assertEqual(remaining,self.manifest['sample_ids'][13:]);self.assertTrue(raw.read_bytes().startswith(before))
        complete=raw.read_bytes()
        args.checkpoint.write_bytes(b'changed checkpoint identity')
        failed,_=self.run_new(args)
        self.assertEqual(failed['status'],'FAILED');self.assertEqual(raw.read_bytes(),complete)

    def test_complete_bad_raw_and_score_tampering_rejected(self):
        args=self.new_args('wfuse');report,_=self.run_new(args)
        self.assertEqual(report['status'],'PASS',report['fatal_error'])
        score=args.output_dir/'accuracy.json';saved=json.loads(score.read_text());saved['overall']['correct']=0
        score.write_text(json.dumps(saved))
        with self.assertRaises(ValueError):compare.read_run(args.output_dir)
        raw=args.output_dir/'raw_generations.jsonl';before=raw.read_bytes();raw.write_bytes(before+b'not json\n')
        failed,_=self.run_new(args);self.assertEqual(failed['status'],'FAILED')
        self.assertEqual(raw.read_bytes(),before+b'not json\n')


class MmsuWorkflowTests(unittest.TestCase):
    def fixture(self,directory,alias):
        args=target(directory,alias,mmsu,['--dataset-dir',str(directory/'mmsu'),
            '--evaluation-script',str(SCORER)])
        paths=[]
        for name in mmsu.dataset.SHARD_NAMES:
            path=directory/'mmsu/data'/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'parquet/audio explicitly mocked');paths.append(path)
        records=[mmsu_record(i) for i in range(5000)]
        return args,paths,records

    def execute(self,args,paths,records,callback=None,forbid_loader=False):
        loaded=(None,None,None,None,None,{'status':'PASS','fixture':True},None)
        def generate(_,sample,**kw):
            # A wrong letter is ignored; only the complete answer text decides v2 correctness.
            return mmsu_generation('B) India<|endoftext|>')
        with patch.object(runtime,'read_checkpoint',return_value={}), \
             patch.object(mmsu.dataset,'shard_paths',return_value=paths), \
             patch.object(mmsu.dataset,'preflight',return_value=(records,{'status':'PASS','rows':5000})), \
             patch.object(mmsu.dataset,'iter_rows',side_effect=lambda *a,**kw:enumerate([{**r,'audio':{'fixture':True}} for r in records])), \
             patch.object(mmsu_runtime,'load_runtime',return_value=loaded,
                          side_effect=AssertionError('no model rescore') if forbid_loader else None) as loader, \
             patch.object(mmsu_runtime,'decode_audio',return_value=(None,{'audio_source':'explicit fixture'})), \
             patch.object(mmsu_runtime,'generate',side_effect=callback or generate) as generator, \
             patch.object(os,'fsync'),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            report=mmsu.run(args)
        return report,loader.call_count,generator.call_count

    def test_six_complete_5000_runs_execute_pinned_scorer_and_read_only_compare(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);directories=[]
            for alias in registry.ROUTES:
                args,paths,records=self.fixture(root,alias)
                report,loads,calls=self.execute(args,paths,records)
                self.assertEqual(report['status'],'PASS',report['fatal_error'])
                self.assertEqual((loads,calls),(1,5000))
                self.assertEqual(report['official_evaluation']['actual_denominator'],5000)
                self.assertEqual(report['official_evaluation']['overall_accuracy'],1.)
                before=(args.output_dir/'raw_generations.jsonl').read_bytes()
                again,loads,calls=self.execute(args,paths,records,forbid_loader=True)
                self.assertEqual(again['status'],'PASS',again['fatal_error']);self.assertEqual((loads,calls),(0,0))
                self.assertEqual(before,(args.output_dir/'raw_generations.jsonl').read_bytes())
                directories.append((alias,args.output_dir))
            self.assertEqual(len(compare.compare_benchmarks('mmsu',directories,root/'mmsu.csv')),6)

    def test_interruption_resume_keeps_full_denominator_and_rejects_corrupt_complete_line(self):
        with tempfile.TemporaryDirectory() as directory:
            args,paths,records=self.fixture(pathlib.Path(directory),'listen');seen=[]
            def stop(_,sample,**kw):
                if len(seen)==13:raise RuntimeError('fixture interruption')
                seen.append(1);return mmsu_generation('India')
            report,_,_=self.execute(args,paths,records,stop)
            self.assertEqual(report['status'],'FAILED');self.assertEqual(report['records']['generated'],13)
            raw=args.output_dir/'raw_generations.jsonl';before=raw.read_bytes()
            report,_,calls=self.execute(args,paths,records)
            self.assertEqual(report['status'],'PASS',report['fatal_error']);self.assertEqual(calls,4987)
            self.assertTrue(raw.read_bytes().startswith(before))
            raw.write_bytes(raw.read_bytes()+b'bad full line\n')
            report,_,_=self.execute(args,paths,records);self.assertEqual(report['status'],'FAILED')


class BenchmarkWorkflowTests(unittest.TestCase):
    # Execute the exact reference fixture driver against the new runner, without modifying the reference module.
    execute=types.FunctionType(original_benchmark_fixture.WorkflowTests.execute.__code__,
        {**original_benchmark_fixture.WorkflowTests.execute.__globals__,'runner':benchmark})
    generation=staticmethod(original_benchmark_fixture.WorkflowTests.generation)

    def fixture(self,directory,kind='mmar',alias='base'):
        root=pathlib.Path(directory);scorer=root/'evaluation.py'
        scorer.write_text('output_key = "model_output"\n',encoding='utf-8')
        args=target(root,alias,benchmark,['--benchmark',kind,'--evaluation-script',str(scorer)])
        records=[{'id':str(i),'question':'Question?','choices':['One','Two'],
                  'answer':'One','audio_path':'fixture.wav'} for i in range(1000)]
        loaded=(None,None,None,None,None,{'checkpoint_step':5771}, {})
        identity={'benchmark':kind,'eval_variant':alias,'checkpoint':'fixture','prediction_key':'model_output'}
        return args,records,loaded,identity

    def test_all_12_complete_1000_runs_preserve_exact_official_predictions(self):
        for alias in registry.ROUTES:
            for kind in ('mmau','mmar'):
                with self.subTest(alias=alias,kind=kind),tempfile.TemporaryDirectory() as directory:
                    args,records,loaded,identity=self.fixture(directory,kind,alias)
                    generate=Mock(side_effect=self.generation)
                    with patch.object(os,'fsync'):
                        report,scorer=self.execute(args,records,loaded,identity,generate)
                    self.assertEqual(report['status'],'PASS',report.get('fatal_error'))
                    self.assertEqual(generate.call_count,1000);scorer.assert_called_once()
                    filename='predictions_fixed_order.json' if kind=='mmau' else 'predictions_official.json'
                    predictions=json.loads((args.output_dir/filename).read_text(encoding='utf-8'))
                    self.assertEqual([r['id'] for r in predictions],[str(i) for i in range(1000)])
                    self.assertTrue(all(r['model_output']=='  A) One <|endoftext|>  ' for r in predictions))

    def test_both_benchmarks_resume_only_remaining_and_never_score_partial_coverage(self):
        for kind in ('mmau','mmar'):
            with tempfile.TemporaryDirectory() as directory:
                args,records,loaded,identity=self.fixture(directory,kind,'audio7_read2')
                with patch.object(os,'fsync'):
                    failed,scorer=self.execute(args,records,loaded,identity,Mock(side_effect=[self.generation()]*13+[RuntimeError('stop')]))
                    self.assertEqual(failed['status'],'FAILED');scorer.assert_not_called()
                    raw=args.output_dir/'raw_generations.jsonl';before=raw.read_bytes()
                    generate=Mock(side_effect=self.generation)
                    passed,scorer=self.execute(args,records,loaded,identity,generate)
                self.assertEqual(passed['status'],'PASS',passed.get('fatal_error'))
                self.assertEqual(generate.call_count,987);self.assertTrue(raw.read_bytes().startswith(before))
                self.assertEqual(generate.call_args_list[0].args[1]['id'],'13')

    def test_scorer_nonzero_or_wrong_denominator_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            args,*_=self.fixture(directory)
            args.output_dir.mkdir(parents=True);path=args.output_dir/'prediction.json';path.write_text('[]')
            for code,count in ((0,999),(1,1000),(0,1000)):
                result=types.SimpleNamespace(returncode=code,stdout=f'Total Accuracy: 50.00% over {count} samples',stderr='')
                with patch.object(benchmark.subprocess,'run',return_value=result):
                    score=benchmark.run_official_scorer(args,path,1000)
                self.assertEqual(score['status'],'PASS' if (code,count)==(0,1000) else 'FAILED')

    def test_mmau_mmar_summary_checks_raw_inputs_coverage_and_score_status(self):
        for kind in ('mmau','mmar'):
            with tempfile.TemporaryDirectory() as directory:
                root=pathlib.Path(directory)
                args,records,loaded,identity=self.fixture(directory,kind,'wfuse')
                identity.update(metadata_sha256='fixture metadata',parquet_sha256=None,
                    evaluation_sha256='fixture scorer',generation=runtime.PROTOCOL,evaluation_code={'fixture':'explicit'})
                with patch.object(os,'fsync'):
                    report,_=self.execute(args,records,loaded,identity,self.generation)
                self.assertEqual(report['status'],'PASS',report.get('fatal_error'))
                report['official_evaluation']['total_accuracy_percent']=100.
                report_path=args.output_dir/'evaluation_report.json'
                report_path.write_text(json.dumps(report),encoding='utf-8')
                rows=compare.compare_benchmarks(kind,[('fixture',args.output_dir)],root/'comparison.csv')
                self.assertEqual(rows[0]['accuracy'],1.)
                report['inference_coverage']['status']='FAILED'
                report_path.write_text(json.dumps(report),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'coverage'):
                    compare.compare_benchmarks(kind,[('fixture',args.output_dir)],root/'failed.csv')


class BashTests(unittest.TestCase):
    def test_real_shell_to_python_argument_check_all_24_bindings(self):
        bash=pathlib.Path('C:/Program Files/Git/bin/bash.exe') if os.name=='nt' else pathlib.Path(shutil.which('bash') or '/absent')
        if not bash.is_file():self.skipTest('Bash unavailable')
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);conda=root/'conda/etc/profile.d';conda.mkdir(parents=True)
            # The conda command is a fixture; the real Python module and real Bash wrappers execute.
            (conda/'conda.sh').write_text('conda() { :; }\n',encoding='utf-8')
            bindir=root/'bin';bindir.mkdir()
            executable=pathlib.Path(sys.executable).as_posix()
            if os.name=='nt':executable='/'+executable[0].lower()+executable[2:]
            shim=bindir/'python';shim.write_text('#!/usr/bin/env bash\nexec '+shlex.quote(executable)+' "$@"\n',encoding='utf-8')
            shim.chmod(0o755)
            def posix(path):
                value=path.as_posix()
                return '/'+value[0].lower()+value[2:] if os.name=='nt' else value
            env={**os.environ,'RSMOL_CONDA_BASE':posix(root/'conda'),
                 'PATH':posix(bindir)+(';' if os.name=='nt' else ':')+os.environ['PATH'],
                 'MSYS2_ARG_CONV_EXCL':'*','PYTHONIOENCODING':'utf-8'}
            for path in (registry.PACKAGE/'run.sh',registry.ROOT/'run_text71427_ced_eval_5090.sh'):
                checked=subprocess.run([str(bash),'-n',str(path)],capture_output=True,text=True,env=env)
                self.assertEqual(checked.returncode,0,checked.stderr)
            for alias in registry.ROUTES:
                result=subprocess.run([str(bash),str(registry.ROOT/'run_text71427_ced_eval_5090.sh'),alias,'all',
                    '--checkpoint',str(root/"run with space's/formal/checkpoints/model--step-5771.ckpt"),'--check-args'],
                    capture_output=True,text=True,encoding='utf-8',env=env)
                self.assertEqual(result.returncode,0,result.stderr)
                jobs=[json.loads(line) for line in result.stdout.splitlines()]
                self.assertEqual(len(jobs),4)
                self.assertTrue(all("run with space's" in j['binding']['checkpoint'] for j in jobs))


if __name__=='__main__':unittest.main()
