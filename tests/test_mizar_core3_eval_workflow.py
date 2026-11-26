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
from mizar_core3_eval import registry, reasonaqa, mmsu, benchmark, runtime, compare, submit
from mizar_core3_eval import reasonaqa_runtime, mmsu_runtime
from test_reasonaqa5000_eval import Fixture as OriginalReasonFixture, generation
from test_mizar_s1_ced_mmsu_eval import record as mmsu_record, generation as mmsu_generation, SCORER
import test_mizar_s1_ced_multilayer_eval_workflow as original_benchmark_fixture
from test_mizar_core3_eval import cache
from mizar_s1_ced_reasonaqa5000_eval import storage as input_storage

def with_inputs(generation):
    values = dict(prompt_token_ids=[3,4,0]+[17]*253,prompt_attention_mask=[1]*3+[0]*253,
                  audio_half_sha256=['a'*64,'b'*64])
    return dict(generation,**values,input_fingerprint_sha256=input_storage.object_sha(values))

_original_mmsu_generation = mmsu_generation
def mmsu_generation(*args,**kwargs):
    return with_inputs(_original_mmsu_generation(*args,**kwargs))


ALL=tuple(registry.ROUTES)
FOUR=ALL


def target(directory, alias, runner, extra=()):
    checkpoint=directory/alias/'formal/checkpoints/model--step-34626.ckpt'
    checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_bytes(b'explicit synthetic checkpoint')
    (checkpoint.parent.parent/'runtime.json').write_text('{"fixture":true}',encoding='utf-8')
    return runner.parse_args(['--eval-variant',alias,'--checkpoint',str(checkpoint),'--sources-cache',str(cache()),'--output-dir',str(directory/alias/runner.__name__.split('.')[-1]),*extra])


class ReasonWorkflowTests(OriginalReasonFixture):
    def new_args(self,alias='l9_12x3'):
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

    def test_two_complete_5000_runs_score_compare_and_no_model_rescore(self):
        directories=[]
        for alias in FOUR:
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
        self.assertEqual(len(rows),4)

    def test_interrupted_13_resumes_only_remaining_4987_and_refuses_changed_identity(self):
        args=self.new_args('l19_22x3');seen=[]
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
        args=self.new_args('l9_12x3');report,_=self.run_new(args)
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
            self.assertEqual(sample['prompt'], 'which country? a) india b) none c) china d) japan')
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

    def test_two_complete_5000_runs_execute_pinned_scorer_and_read_only_compare(self):
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
            self.assertEqual(len(compare.compare_benchmarks('mmsu',directories,root/'mmsu.csv')),4)
            raw_path=directories[-1][1]/'raw_generations.jsonl'
            saved_rows=[json.loads(line) for line in raw_path.read_text().splitlines()]
            changed=saved_rows[0]['generation']
            changed['prompt_token_ids'][0]+=1
            body={k:changed[k] for k in ('prompt_token_ids','prompt_attention_mask','audio_half_sha256')}
            changed['input_fingerprint_sha256']=input_storage.object_sha(body)
            saved_rows[0]['generation_sha256']=input_storage.object_sha(changed)
            raw_path.write_text(''.join(json.dumps(r)+'\n' for r in saved_rows))
            with self.assertRaisesRegex(ValueError,'differ'):
                compare.compare_benchmarks('mmsu',directories,root/'bad_inputs.csv')


    def test_interruption_resume_keeps_full_denominator_and_rejects_corrupt_complete_line(self):
        with tempfile.TemporaryDirectory() as directory:
            args,paths,records=self.fixture(pathlib.Path(directory),'l19_22x3');seen=[]
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
    generation=staticmethod(lambda *a:with_inputs(original_benchmark_fixture.WorkflowTests.generation(*a)))

    def fixture(self,directory,kind='mmar',alias='l9_12x3'):
        root=pathlib.Path(directory);scorer=root/'evaluation.py'
        scorer.write_text('output_key = "model_output"\n',encoding='utf-8')
        args=target(root,alias,benchmark,['--benchmark',kind,'--evaluation-script',str(scorer)])
        records=[{'id':str(i),'question':'Question?','choices':['One','Two'],
                  'answer':'One','audio_path':'fixture.wav'} for i in range(1000)]
        loaded=(None,None,None,None,None,{'checkpoint_step':args.expected_step}, {})
        identity={'benchmark':kind,'eval_variant':alias,'checkpoint':'fixture','prediction_key':'model_output'}
        return args,records,loaded,identity

    def test_all_four_complete_1000_runs_preserve_exact_official_predictions(self):
        for alias in FOUR:
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
                args,records,loaded,identity=self.fixture(directory,kind,'l19_22x3')
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
                args,records,loaded,identity=self.fixture(directory,kind,'l9_12x3')
                identity.update(metadata_sha256='fixture metadata',parquet_sha256=None,
                    evaluation_sha256='fixture scorer',generation=runtime.protocol_for(args.eval_variant),evaluation_code=runtime.evaluation_sources())
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



if __name__=='__main__':unittest.main()
