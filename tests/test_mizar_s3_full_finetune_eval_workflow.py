"""Full-size AU/AR runners; model/audio/parquet are explicit synthetic fixtures."""
import contextlib
import io
import json
import os
import pathlib
import tempfile
import types
import unittest
from unittest.mock import Mock,patch
from mizar_s3_full_finetune_eval import registry,benchmark
from mizar_s1_ced_reasonaqa5000_eval import storage as input_storage
import test_mizar_s1_ced_multilayer_eval_workflow as original_benchmark_fixture

def with_inputs(generation):
    values=dict(prompt_token_ids=[3,4,0]+[17]*253,prompt_attention_mask=[1]*3+[0]*253,audio_half_sha256=['a'*64,'b'*64])
    return dict(generation,**values,input_fingerprint_sha256=input_storage.object_sha(values))

def target(directory,alias,runner,extra=()):
    checkpoint=directory/alias/'formal/checkpoints/model--step-1066.ckpt'
    checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_bytes(b'explicit synthetic checkpoint')
    (checkpoint.parent.parent/'runtime.json').write_text('{"fixture":true}')
    return runner.parse_args(['--eval-variant',alias,'--checkpoint',str(checkpoint),
        '--output-dir',str(directory/alias/'benchmark'),*extra])

class BenchmarkWorkflowTests(unittest.TestCase):
    # Execute the exact reference fixture driver against the new runner, without modifying the reference module.
    execute=types.FunctionType(original_benchmark_fixture.WorkflowTests.execute.__code__,
        {**original_benchmark_fixture.WorkflowTests.execute.__globals__,'runner':benchmark})
    generation=staticmethod(lambda *a:with_inputs(original_benchmark_fixture.WorkflowTests.generation(*a)))

    def fixture(self,directory,kind='mmar',alias=registry.TARGET):
        root=pathlib.Path(directory);scorer=root/'evaluation.py'
        scorer.write_text('output_key = "model_output"\n',encoding='utf-8')
        args=target(root,alias,benchmark,['--benchmark',kind,'--evaluation-script',str(scorer)])
        records=[{'id':str(i),'question':'Question?','choices':['One','Two'],
                  'answer':'One','audio_path':'fixture.wav'} for i in range(1000)]
        loaded=(None,None,None,None,None,{'checkpoint_step':args.expected_step}, {})
        identity={'benchmark':kind,'eval_variant':alias,'checkpoint':'fixture','prediction_key':'model_output'}
        return args,records,loaded,identity

    def test_both_complete_1000_runs_preserve_exact_official_predictions(self):
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
                args,records,loaded,identity=self.fixture(directory,kind,registry.TARGET)
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

    def test_bad_complete_rows_progress_and_changed_identity_cannot_resume(self):
        for kind in ('mmau','mmar'):
            with tempfile.TemporaryDirectory() as directory,patch.object(os,'fsync'):
                args,records,loaded,identity=self.fixture(directory,kind)
                passed,_=self.execute(args,records,loaded,identity,self.generation)
                self.assertEqual(passed['status'],'PASS',passed.get('fatal_error'))
                raw=args.output_dir/'raw_generations.jsonl';original=raw.read_bytes()
                raw.write_bytes(original+b'bad full line\n')
                failed,scorer=self.execute(args,records,loaded,identity,self.generation)
                self.assertEqual(failed['status'],'FAILED');scorer.assert_not_called()
                self.assertEqual(raw.read_bytes(),original+b'bad full line\n')
                raw.write_bytes(original)
                progress=args.output_dir/'progress.jsonl'
                rows=[json.loads(line) for line in progress.read_text().splitlines()]
                rows[0]['record']['generated_text']='conflicting complete write'
                progress.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                failed,scorer=self.execute(args,records,loaded,identity,self.generation)
                self.assertEqual(failed['status'],'FAILED');scorer.assert_not_called()
                with self.assertRaises(ValueError):benchmark.claim_output(args,{**passed['run_identity'],'checkpoint':'different'})


if __name__=='__main__':unittest.main()
