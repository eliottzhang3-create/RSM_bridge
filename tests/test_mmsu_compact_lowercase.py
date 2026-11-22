"""Formal MMSU template migration: input parity, protocol and old-output isolation."""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/RSmol'),str(ROOT/'tests')]
from mizar_s1_ced_mmsu_eval import dataset,evaluate,runtime,parsing,storage
from text71427_ced_eval import arguments,mmsu,mmsu_runtime,submit,registry
from mizar_mmsu_inference_probe import probe
from test_mizar_s1_ced_mmsu_eval import record,SCORER
import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as mmau

OLD_PROTOCOL_VERSION='mmsu_official_prompt_ced_fixed20_raw_greedy_answer_text_v2'
OLD_PROMPT_POLICY='MMSU_official_instruction_ABCD_original_case_and_null_positions'


class CompactPromptTests(unittest.TestCase):
    def test_exact_mmau_builder_parity_case_whitespace_null_slots_and_no_mutation(self):
        for question in ('Which Country?', '  Which COUNTRY?  ', 'Which\nCountry?', 'İ Ü ẞ?', 'a) In the question?'):
            for choices in ((None,'India','NEW ZEALAND','Australia'),(' A ',None,'B',None),
                            ('same','same','',None),('A. original','b) embedded','Line\nTWO','Punctuation!')):
                row={**record(),'question':question,**dict(zip(dataset.CHOICES,choices))}
                before=copy.deepcopy(row)
                expected=mmau.build_fixed_order_prompt(question,choices).lower()
                self.assertEqual(dataset.prompt(row),expected)
                self.assertEqual(row,before)
                poisoned={**row,'answer_gt':'DO_NOT_LEAK','task_name':'DO_NOT_LEAK','category':'DO_NOT_LEAK'}
                self.assertEqual(dataset.prompt(poisoned),expected)
        row={**record(),'choice_a':None,'choice_b':'B','choice_c':None,'choice_d':'D'}
        self.assertEqual(dataset.prompt(row),'which country? a) none b) b c) none d) d')

    def test_formal_and_diagnostic_compact_case_agree_and_old_template_still_distinct(self):
        cases=probe.prompt_cases(record(),dataset,types.SimpleNamespace(mmau=mmau))
        self.assertEqual(dataset.prompt(record()),cases['compact_lowercase'])
        self.assertIn('only A, B, C, or D',cases['official_original'])
        self.assertEqual(cases['official_original'],dataset.official_prompt(record()))
        self.assertNotEqual(cases['official_original'],cases['compact_original'])
        self.assertNotEqual(cases['official_lowercase'],cases['compact_lowercase'])

    def test_protocol_changes_only_prompt_identity_not_numeric_or_score_settings(self):
        self.assertEqual(runtime.PROTOCOL,mmsu_runtime.PROTOCOL)
        self.assertEqual(runtime.PROTOCOL['version'],dataset.PROMPT_VERSION)
        self.assertEqual(runtime.PROTOCOL['prompt_policy'],dataset.PROMPT_POLICY)
        self.assertNotEqual(dataset.PROMPT_VERSION,OLD_PROTOCOL_VERSION)
        self.assertNotEqual(dataset.PROMPT_POLICY,OLD_PROMPT_POLICY)
        self.assertEqual(runtime.PROTOCOL['scoring'],'mmsu_answer_text_contains_v2_then_pinned_external_official_scorer')
        for key,value in runtime.base.PROTOCOL.items():
            if key not in ('version','prompt_policy','scoring'):
                self.assertEqual(runtime.PROTOCOL[key],value,key)
        self.assertTrue(parsing.parse_response('C) INDIA<|endoftext|>','India')['matched'])
        self.assertFalse(parsing.parse_response('A<|endoftext|>','India')['matched'])

    def test_training_decoder_audio_greedy_and_scoring_functions_unchanged_from_before(self):
        report=json.loads((ROOT/'audit_reports/mmsu_compact_lowercase_validation.json').read_text(encoding='utf-8'))
        for name,functions in report['functions_before'].items():
            actual={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
                    for n in ast.parse((ROOT/name).read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef)}
            for key,digest in functions.items():
                if key=='parse_args' and name.endswith('mizar_s1_ced_mmsu_eval/evaluate.py'):continue
                if key=='prompt' and name.endswith('/dataset.py'):continue
                self.assertEqual(actual[key],digest,f'{name}:{key}')
        for name in ('code/RSmol/mizar_s1_ced_mmsu_eval/parsing.py',
                     'code/RSmol/mizar_s1_ced_mmsu_eval/scoring.py',
                     'code/RSmol/text71427_ced_eval/runtime.py',
                     'code/RSmol/scripts/evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.py'):
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
                             report['production_sources_before'][name],name)


class OutputIsolationTests(unittest.TestCase):
    def test_all_legacy_and_six_new_aliases_use_new_defaults_and_other_tasks_unchanged(self):
        for variant in runtime.VARIANTS:
            if variant=='official_mizar':continue  # explicitly supplied output is still required
            args=evaluate.parse_args(['--eval-variant',variant,'--run-root','/run',
                '--expected-step','34626','--expected-geometry','8,4,4'])
            self.assertEqual(args.output_dir,Path('/run/eval/mmsu_compact_lowercase_v1/step34626').resolve())
        for alias in registry.ROUTES:
            argv=[alias,'all','--checkpoint',f'/run/{alias}/formal/checkpoints/model--step-34626.ckpt','--check-args']
            for extra in ([],['--output-root','/independent']):
                jobs=submit.planned(submit.parse_args(argv+extra))
                self.assertEqual(len(jobs),4)
                for job in jobs:
                    task_dir=job['resolved'].output_dir.parent.name
                    self.assertEqual(task_dir,dataset.OUTPUT_NAMESPACE if job['benchmark']=='mmsu' else job['benchmark'])

    def test_wrong_old_protocol_output_is_refused_before_model_or_raw_modification(self):
        for runner,variant in ((evaluate,'recursive_mesh'),(mmsu,'listen')):
            with self.subTest(runner=runner.__name__),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);old=root/'old-output';old.mkdir()
                checkpoint=root/'formal/checkpoints/model--step-34626.ckpt'
                argv=['--eval-variant',variant,'--checkpoint',str(checkpoint),'--output-dir',str(old)]
                if runner is evaluate:
                    argv+=['--runtime-config',str(root/'formal/runtime.json'),'--expected-step','34626','--expected-geometry','8,4,4']
                args=runner.parse_args(argv)
                old_identity={'protocol':{**runtime.PROTOCOL,'version':OLD_PROTOCOL_VERSION,'prompt_policy':OLD_PROMPT_POLICY}}
                storage.claim_output(old,old_identity)
                (old/'raw_generations.jsonl').write_bytes(b'old generations must remain byte-identical\n')
                (old/'evaluation_report.json').write_text('{"old_report":true}')
                before={p.name:p.read_bytes() for p in old.iterdir()}
                with patch.object(runner,'identity',return_value={'protocol':runtime.PROTOCOL}), \
                     patch.object(runner.dataset,'shard_paths',return_value=[]), \
                     patch.object(runner.runtime,'load_runtime',side_effect=AssertionError('model loaded')) as loader, \
                     redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                    report=runner.run(args)
                self.assertEqual(report['status'],'FAILED');self.assertFalse(report['output_claimed'])
                loader.assert_not_called()
                self.assertEqual({p.name:p.read_bytes() for p in old.iterdir() if p.name!='.mmsu.lock'},before)

    def test_remaining_dispatcher_rejects_complete_or_partial_old_prompt_as_new_result(self):
        from mizar_s1_ced_mmsu_remaining_eval import inspection
        from test_mizar_mmsu_remaining_eval import fixture
        with tempfile.TemporaryDirectory() as directory:
            args,identity,_=fixture(directory,13)
            identity['protocol']={**runtime.PROTOCOL,'version':OLD_PROTOCOL_VERSION,'prompt_policy':OLD_PROMPT_POLICY}
            identity['identity_sha256']=storage.object_sha({k:v for k,v in identity.items() if k!='identity_sha256'})
            storage.write_json(args.output_dir/'run_config.json',identity)
            before={p.name:p.read_bytes() for p in args.output_dir.iterdir()}
            with self.assertRaisesRegex(ValueError,'generation protocol differs'):
                inspection.inspect(args)
            self.assertEqual({p.name:p.read_bytes() for p in args.output_dir.iterdir()},before)


if __name__=='__main__':unittest.main()
