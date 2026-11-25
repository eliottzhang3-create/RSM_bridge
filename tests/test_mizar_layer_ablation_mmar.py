"""MMAR ablation contracts, real CPU tensors and explicit full-size fixtures."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/RSmol'))
from mizar_layer_pair_search import storage
from mizar_layer_ablation_mmar import arguments, audit, cache, config, dataset, evaluate, identity, results, runtime, scoring, submit, task


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP'))
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)


class FixtureTokenizer:
    """Explicit input fixture; production uses the release-verified tokenizer."""
    def encode(self, text, add_special_tokens=True):
        return [30 + ord(c) % 10 for c in text]

    def __call__(self, text, **kwargs):
        ids = self.encode(text)[:256]
        return {'input_ids': ids + [17]*(256-len(ids)), 'attention_mask': [1]*len(ids)+[0]*(256-len(ids))}


def raw_fixture():
    rows = []
    for category, count in config.CATEGORY_COUNTS.items():
        for _ in range(count):
            i = len(rows)
            rows.append({'id': str(i), 'question': f'Fixture question {i}?', 'choices': ['one', 'two'],
                         'answer': 'one' if i % 2 == 0 else 'two', 'category': category,
                         'sub-category': 'fixture', 'modality': 'sound', 'language': 'en', 'audio_path': './audio/fixture.wav'})
    # Include both specified failure modes while retaining original slots/order.
    rows[4]['choices'].append('')
    rows[5]['answer'] = 'one\ntwo'
    return rows


def records_fixture():
    records = [dataset.tokenize(dataset.normalize(i, r), FixtureTokenizer()) for i, r in enumerate(raw_fixture())]
    return [{**r, 'audio_ref': {'kind': 'file', 'path': 'explicit-not-production.wav'},
             'audio_audit': {'fixture': True}} for r in records]


def score_fixture(record, layer, correct=True):
    target = record['gold_index'] if record['gold_index'] is not None else 0
    prediction = target if correct else 1-target
    scores = [-4.0 if valid else None for valid in record['choice_valid_mask']]
    scores[prediction] = -1.0
    return {**scoring.decision(record, scores), 'execution': {
        'drop_layer_1based': layer, 'executed_original_layers_1based': [i for i in range(1,31) if i != layer],
        'effective_layers': 30 if layer is None else 29, 'checked_forwards': 1}}


def row_fixture(record, layer, correct=True):
    return storage.seal({'identity_sha256': 'synthetic-identity', 'condition': results.name(layer),
        'global_index': record['global_index'], 'key': record['key'], 'record_sha256': storage.object_sha(record),
        **score_fixture(record, layer, correct)})


class Contracts(TempCase):
    def test_single_5090_job_quoting_forwarded_args_and_exact_condition_grid(self):
        args = arguments.parse_args(['--output-dir', str(self.path/"output's space")])
        cmd = submit.command(args, 'fixture123')
        for flag, value in (('-p','pdgpu-5090'),('-g','8'),('-c','64'),('-m','256G'),('-n','1')):
            self.assertEqual(cmd[cmd.index(flag)+1], value)
        self.assertEqual(cmd.count('submit'), 1)
        self.assertLess(len(cmd[cmd.index('-j')+1]), 60)
        compute = shlex.split(cmd[cmd.index('--cmd')+1])
        self.assertEqual(compute[:3], ['bash','mizar_layer_ablation_mmar/run.sh','evaluate'])
        self.assertEqual(arguments.parse_args(arguments.forwarded(args)).output_dir, args.output_dir)
        self.assertEqual(config.CONDITIONS, (None,*range(2,30)))
        self.assertEqual([len(config.indices(r)) for r in range(8)], [125]*8)
        self.assertEqual(sorted(i for r in range(8) for i in config.indices(r)), list(range(1000)))
        for bad in (1,30,True,2.0):
            with self.assertRaises(ValueError): results.name(bad)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments.parse_args(['--manifest',str(self.path/'output/m.json'),'--output-dir',str(self.path/'output')])
        with patch.object(evaluate.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as launch, \
             contextlib.redirect_stdout(io.StringIO()): evaluate.launch(args)
        self.assertEqual(launch.call_count,1)
        self.assertIn('--nproc-per-node=8', launch.call_args.args[0])
        self.assertIn('mizar_layer_ablation_mmar.worker', launch.call_args.args[0])

    def test_original_experiments_unchanged_and_new_sources_guarded(self):
        frozen = storage.read_json(ROOT/'tests/fixtures/mizar_layer_ablation_mmar_original_sources.json')
        for name, digest in frozen['sources_lf_sha256'].items():
            self.assertEqual(storage.text_sha(ROOT/name), digest, name)
        self.assertIn('mizar_layer_ablation_mmar/runtime.py', identity.sources())
        with patch.object(config,'SCHEMA','other-task'):
            self.assertNotEqual(task.protocol(arguments.parse_args([]))['task'], 'mizar_mmar_ablation_bf16_8gpu5090_v1')

    def test_check_args_imports_no_torch_or_production_assets(self):
        for module in ('submit','evaluate','manifest'):
            code = f"import sys; from mizar_layer_ablation_mmar.{module} import main; assert main(['--check-args'])==0; assert 'torch' not in sys.modules"
            proc = subprocess.run([sys.executable,'-B','-c',code], cwd=config.ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)


class MetadataAndScoring(TempCase):
    def test_empty_original_slot_preserved_excluded_and_not_relabelled(self):
        raw = {**raw_fixture()[0], 'choices': ['one','','two'], 'answer':'two'}
        record = dataset.tokenize(dataset.normalize(0, raw), FixtureTokenizer())
        self.assertEqual(record['prompt'],'fixture question 0? a) one b)  c) two')
        self.assertEqual(record['choice_valid_mask'], [True,False,True])
        self.assertEqual(record['valid_original_slots'], [0,2])
        self.assertEqual(record['gold_index'], 2)
        self.assertEqual(record['choice_token_ids'][1], [])
        value = scoring.decision(record,[-3.,None,-2.])
        self.assertTrue(value['correct']); self.assertEqual(value['prediction_index'],2)
        self.assertIsNone(value['option_mean_loglikelihood'][1])
        for scores in ([-3.,0.,-2.],[-3.,float('-inf'),-2.],[-3.,None,float('nan')]):
            with self.assertRaises(ValueError): scoring.decision(record,scores)

    def test_unmapped_GT_stays_in_denominator_and_no_fuzzy_correction(self):
        record = dataset.tokenize(dataset.normalize(0,{**raw_fixture()[0],'answer':'one\ntwo'}),FixtureTokenizer())
        self.assertIsNone(record['gold_index'])
        self.assertEqual(record['gold_mapping_status'],'UNMAPPED_COUNT_INCORRECT')
        for scores in ([-1.,-3.],[-3.,-1.]):
            value=scoring.decision(record,scores)
            self.assertFalse(value['correct']); self.assertFalse(value['gold_mapped'])

    def test_sum_not_mean_ties_and_official_diagnostic_are_separate(self):
        raw={**raw_fixture()[0],'choices':['short','long long long'],'answer':'short'}
        record=dataset.tokenize(dataset.normalize(0,raw),FixtureTokenizer())
        value=scoring.decision(record,[-2.,-3.])
        self.assertEqual(value['prediction_index'],0)
        self.assertGreater(value['option_mean_loglikelihood'][1],value['option_mean_loglikelihood'][0])
        tie=scoring.decision(record,[-2.,-2.]); self.assertEqual(tie['prediction_index'],0); self.assertTrue(tie['option_score_tie'])
        # The official word-set matcher ignores word order; strict main metric does not.
        raw.update(choices=['dog bites man','man bites dog'],answer='dog bites man')
        record=dataset.tokenize(dataset.normalize(0,raw),FixtureTokenizer())
        value=scoring.decision(record,[-4.,-1.])
        self.assertFalse(value['correct']); self.assertTrue(value['official_correct'])

    def test_real_local1000_metadata_and_original_official_function_AST(self):
        path=Path(r'C:/Xlance/assets/MMAR/MMAR-meta.json')
        if not path.exists(): self.skipTest('optional local MMAR metadata backup unavailable')
        records=[dataset.normalize(i,r) for i,r in enumerate(storage.read_json(path))]
        self.assertEqual(len(records),1000)
        audit=dataset.metadata_audit(records)
        self.assertEqual(audit['category_counts'],config.CATEGORY_COUNTS)
        self.assertEqual([r['index'] for r in audit['empty_option_rows']],[28,61,876,910])
        self.assertEqual([r['index'] for r in audit['unmapped_gold_rows']],[43,517,836,894,983])
        self.assertEqual(audit['official_accepted_slot_count_distribution'],{'0':5,'1':975,'2':15,'3':2,'4':3})
        scoring.verify_reference(path.parent/'code/evaluation.py')

    def test_invalid_rows_context_and_reference_function_change_fail(self):
        raw=raw_fixture()[0]
        for edit in ({'choices':['one','']},{'question':''},{'category':'other'},{'choices':['one',None]}):
            with self.assertRaises(ValueError): dataset.normalize(0,{**raw,**edit})
        record=dataset.normalize(0,{**raw,'choices':['one','x'*642]})
        with self.assertRaisesRegex(ValueError,'exceeds context'): dataset.tokenize(record,FixtureTokenizer())
        script=self.path/'bad.py'; script.write_text('def string_match(answer, prediction, choices):\n    return True\n')
        with self.assertRaisesRegex(ValueError,'differs'): scoring.verify_reference(script)

    def builder_fixture(self):
        metadata=self.path/'meta.json'; storage.write_json(metadata,raw_fixture())
        audio=self.path/'audio/audio/fixture.wav'; audio.parent.mkdir(parents=True); audio.write_bytes(b'explicit fixture not WAV')
        script=self.path/'evaluation.py'
        source=Path(scoring.__file__).read_text(encoding='utf-8')
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='string_match')
        script.write_text('import re\n'+ast.get_source_segment(source,fn)+'\n',encoding='utf-8')
        args=arguments.parse_args(['--mmar-metadata',str(metadata),'--mmar-audio-root',str(audio.parents[1]),
            '--evaluation-script',str(script),'--manifest',str(self.path/'manifest/manifest.json'),'--output-dir',str(self.path/'output')])
        with patch.object(dataset.reference_manifest,'load_tokenizer',return_value=(FixtureTokenizer(),[])), \
             patch.object(dataset.audio,'load_fixed20',return_value=(None,{'fixture':True})) as decode, \
             contextlib.redirect_stdout(io.StringIO()):
            frozen=dataset.ensure(args)
        self.assertEqual(decode.call_count,1000)
        return args,frozen,audio

    def test_actual1000_builder_seal_reuse_source_argument_and_tamper_checks(self):
        args,frozen,audio=self.builder_fixture()
        before=args.manifest.read_bytes()
        self.assertEqual(dataset.load(args.manifest),frozen)
        with patch.object(dataset,'build',side_effect=AssertionError('must not rebuild')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(dataset.ensure(args),frozen)
        self.assertEqual(before,args.manifest.read_bytes())
        changed=copy.copy(args); changed.mmar_audio_root=self.path/'other'
        with self.assertRaisesRegex(ValueError,'different source paths'): dataset.ensure(changed)
        broken=copy.deepcopy(frozen); broken['records'][0]['prompt_token_ids'][0]+=1
        storage.write_json(args.manifest,broken)
        with self.assertRaisesRegex(ValueError,'content hash mismatch'): dataset.load(args.manifest)
        broken=copy.deepcopy(frozen); broken.pop('manifest_sha256'); broken['records'][0]['gold_index']=1
        storage.write_json(args.manifest,storage.seal(broken,'manifest_sha256'))
        with self.assertRaisesRegex(ValueError,'inconsistent frozen'): dataset.load(args.manifest)
        broken=copy.deepcopy(frozen); broken.pop('manifest_sha256'); broken['generator_sources']['mizar_layer_pair_search/audio.py']='changed'
        storage.write_json(args.manifest,storage.seal(broken,'manifest_sha256'))
        with self.assertRaisesRegex(ValueError,'implementation changed'): dataset.load(args.manifest)
        storage.write_json(args.manifest,frozen); audio.write_bytes(b'changed fixture')
        with self.assertRaisesRegex(ValueError,'file identity changed'): dataset.load(args.manifest)

    def test_missing_audio_unsafe_path_and_changed_category_denominators_fail(self):
        args=arguments.parse_args(['--mmar-audio-root',str(self.path)])
        record=dataset.normalize(0,raw_fixture()[0])
        with self.assertRaises(FileNotFoundError): dataset.attach_audio([record],args)
        for value in ('../escape.wav','/absolute.wav','C:/escape.wav'):
            record['audio_path_original']=value
            with self.assertRaises(ValueError): dataset.attach_audio([record],args)
        args,frozen,_=self.builder_fixture()
        frozen.pop('manifest_sha256'); frozen['records'][0]['category']='Semantic Layer'
        with self.assertRaisesRegex(ValueError,'four-category'): dataset.validate(storage.seal(frozen,'manifest_sha256'))


class ResultsAndControl(TempCase):
    def test_all_category_denominators_weighted_overall_and_primary_vs_diagnostic(self):
        records=records_fixture()
        base=[row_fixture(r,None,r['global_index']<290) for r in records]
        rows=[row_fixture(r,2,r['global_index']<390) for r in records]
        expected={'identity_sha256':'synthetic-identity'}
        merged=results.merge([[rows[i] for i in config.indices(rank)] for rank in range(8)],records,2,expected)
        summary=results.summarize(merged,base,records,2)
        self.assertEqual((summary['correct'],summary['total'],summary['accuracy'],summary['delta_vs_baseline']),(389,1000,.389,.1))
        self.assertEqual(summary['newly_correct'],100)
        self.assertEqual({k:v['total'] for k,v in summary['groups']['category'].items()},config.CATEGORY_COUNTS)
        self.assertEqual(sum(g['correct'] for g in summary['groups']['category'].values()),summary['correct'])
        self.assertEqual(summary['unmapped_gold_rows'],1); self.assertEqual(summary['empty_choice_rows'],1)

    def test_raw_rows_scores_trace_decision_and_journal_recovery_are_verified(self):
        records=records_fixture(); record=records[0]; expected={'identity_sha256':'synthetic-identity'}
        row=row_fixture(record,2); results.validate(row,record,2,expected)
        for edit in ({'correct':False},{'official_correct':False},{'condition':'ablate_03'},
                     {'option_loglikelihood':[-1.,float('nan')]},
                     {'execution':{**row['execution'],'effective_layers':30}},{'generation':{}}):
            broken={k:v for k,v in row.items() if k!='sha256'}; broken.update(edit)
            with self.assertRaises(ValueError):results.validate(storage.seal(broken),record,2,expected)
        path=self.path/'ranks/rank00/ablate_02.jsonl'
        storage.write_jsonl(path,[row_fixture(records[i],2) for i in (0,8)])
        with path.open('ab') as stream:stream.write(b'{partial')
        self.assertEqual(len(results.read_rank(self.path,0,records,2,expected)),2)
        with self.assertRaises(ValueError): results.read_rank(self.path,0,records,2,{'identity_sha256':'other'})
        storage.write_jsonl(path,[row_fixture(records[8],2),row_fixture(records[0],2)])
        with self.assertRaises(ValueError): results.read_rank(self.path,0,records,2,expected)

    def test_actual_synchronized_worker_baseline_boundary_layers_and_resume(self):
        records=records_fixture(); expected={'identity_sha256':'synthetic-identity'}
        args=SimpleNamespace(output_dir=self.path); gathered={}; barriers=[]
        class Controller:
            def set_drop(self,layer):self.drop=layer
            def evidence(self):return score_fixture(records[0],self.drop)['execution']
        class Dist:
            def __init__(self,rank,layer):self.rank,self.layer=rank,layer
            def broadcast_object_list(self,control,src):control[0]=(self.layer,'synthetic-identity')
            def gather_object(self,rows,destination,dst):
                gathered[self.rank]=rows
                if self.rank==0:destination[:]=[gathered[r] for r in range(8)]
            def barrier(self):barriers.append((self.rank,self.layer))
        with patch.object(runtime,'score',side_effect=lambda loaded,p,r:{k:v for k,v in score_fixture(r,loaded['controller'].drop).items() if k!='execution'}) as score, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            for layer in (None,2,29):
                for rank in [*range(1,8),0]:
                    task.evaluate_condition(args,{'controller':Controller()},dict.fromkeys(config.indices(rank),'prefix'),records,expected,layer,rank,Dist(rank,layer))
            for rank in [*range(1,8),0]:
                task.evaluate_condition(args,{'controller':Controller()},{},records,expected,2,rank,Dist(rank,2))
        self.assertEqual(score.call_count,3000)
        logs=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([r['event'] for r in logs],['BASELINE_DONE','ABLATION_DONE','ABLATION_DONE','ABLATION_DONE'])
        self.assertEqual(set(logs[0]['categories']),set(config.CATEGORIES))
        self.assertEqual(len(barriers),32)
        with patch.object(task,'evaluate_condition') as condition:
            task.work(args,None,None,records,expected,0,None)
        self.assertEqual([call.args[5] for call in condition.call_args_list],list(config.CONDITIONS))

    def test_actual_coordinator_prepares_once_and_launches_one_torchrun(self):
        args=arguments.parse_args(['--output-dir',str(self.path/'out'),'--manifest',str(self.path/'m/m.json')])
        expected=storage.seal({'fixture':True},'identity_sha256')
        with patch.object(task,'prepare',return_value={'manifest_sha256':'fixture'}) as prepare, \
             patch.object(identity,'build',return_value=expected),patch.object(identity,'verify'), \
             patch.object(task,'finalize',side_effect=[None,{'status':'PASS','decisions':29000}]), \
             patch.object(evaluate,'launch') as launch,contextlib.redirect_stdout(io.StringIO()):
            report=evaluate.run(args)
        self.assertEqual((prepare.call_count,launch.call_count,report['decisions']),(1,1,29000))


def runtime_fixture(directory,records):
    expected={'identity_sha256':'synthetic-identity'}; audits=[]; checks=[]; caches=[]
    for rank in range(8):
        model={'binding':{'rank':rank,'worker_cuda_visible_devices':str(rank)},'EXPLICIT_SYNTHETIC_CPU':True}
        check={'status':'PASS','rank':rank,'identity_sha256':'synthetic-identity','EXPLICIT_SYNTHETIC_CPU':True,
               'bypass_vs_identity_output':[{'status':'PASS','layer_1based':i,'max_abs_error':0} for i in (2,29)]}
        folder=directory/'ranks'/f'rank{rank:02d}'
        storage.write_json(folder/'model_audit.json',storage.seal(model))
        storage.write_json(folder/'runtime_check.json',storage.seal(check))
        storage.write_json(folder/'rank_report.json',storage.seal({'status':'PASS','task':config.SCHEMA,
            'identity_sha256':'synthetic-identity','indices':config.indices(rank)}))
        cache_folder=directory/'cache'/f'rank{rank:02d}'; hashes=[]
        for index in config.indices(rank):
            cache_folder.mkdir(parents=True,exist_ok=True)
            path=cache_folder/f'prefix_{index:04d}.pt'; path.write_bytes(b'EXPLICIT_SYNTHETIC_NOT_A_TENSOR')
            meta=storage.seal({'record_sha256':storage.object_sha(records[index]),'file_sha256':storage.file_sha(path)})
            storage.write_json(cache_folder/f'prefix_{index:04d}.json',meta); hashes.append(meta['sha256'])
        cache_meta=storage.seal({'cache_identity':{'run_identity_sha256':'synthetic-identity'},'indices':config.indices(rank),'entry_meta_sha256':hashes})
        storage.write_json(cache_folder/'cache_manifest.json',cache_meta)
        audits.append(model);checks.append(check);caches.append(cache_meta)
    storage.write_json(directory/'runtime_checks.json',storage.seal({'status':'PASS','identity_sha256':'synthetic-identity',
        'rank_audits':audits,'rank_checks':checks,'rank_caches':caches}))
    return expected


class FullExperiment(TempCase):
    def test_actual29x1000_finalizer_four_groups_CSV_ranking_and_six_figures(self):
        records=records_fixture(); expected=runtime_fixture(self.path,records)
        def reader(directory,rank,records,layer,expected):
            threshold=290 if layer is None else 390 if layer in (2,29) else 350
            return [row_fixture(records[i],layer,i<threshold) for i in config.indices(rank)]
        with patch.object(results,'read_rank',side_effect=reader) as read:
            report=task.finalize(SimpleNamespace(output_dir=self.path),{'records':records,'dataset_audit':dataset.metadata_audit(records)},expected)
        self.assertEqual(read.call_count,29*8); self.assertEqual(report['decisions'],29000)
        self.assertEqual(report['top_tied_layers_1based'],[2,29])
        self.assertEqual(report['baseline_accuracy'],.289)
        self.assertEqual(len(storage.read_json(self.path/'ranking.json')),28)
        self.assertEqual(len((self.path/'accuracy_by_category.csv').read_text().splitlines()),117)
        self.assertEqual(len((self.path/'official_diagnostic.csv').read_text().splitlines()),146)
        self.assertEqual(len(report['plots']['files']),6)
        for filename in report['plots']['files']:self.assertGreater((self.path/filename).stat().st_size,1000)
        # Replot into a retained QA folder for visual review, without production artifacts.
        qa=os.environ.get('RSMOL_MMAR_QA')
        if qa:
            from mizar_layer_ablation_mmar.plot import make_plot
            folder=Path(qa);folder.mkdir(parents=True,exist_ok=True)
            make_plot(folder,storage.read_json(self.path/'condition_scores.json'))
        (self.path/'cache/rank00/prefix_0000.pt').write_bytes(b'corrupted')
        with self.assertRaises(ValueError):audit.verify(self.path,expected,records)


ML=all(importlib.util.find_spec(n) for n in ('torch','transformers','safetensors'))


@unittest.skipUnless(ML,'requires real local CPU Torch and Transformers')
class RealLlama(TempCase):
    def make_model(self,backend='eager'):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.set_num_threads(1);torch.manual_seed(1234)
        cfg=LlamaConfig(vocab_size=41,hidden_size=16,intermediate_size=24,num_hidden_layers=30,num_attention_heads=2,
                        num_key_value_heads=2,max_position_embeddings=1024,attention_dropout=0.,use_cache=False,tie_word_embeddings=True)
        cfg._attn_implementation=backend
        lm=LlamaForCausalLM(cfg).float().eval().requires_grad_(False)
        return torch,lm

    def test_baseline_native_all28_identity_routes_BF16_weights_unchanged(self):
        for backend in ('eager','sdpa'):
            torch,lm=self.make_model(backend)
            prefix=lm.get_input_embeddings()(torch.tensor([[1,2,3,4,5,6]])).to(torch.bfloat16)
            weights={k:v.clone() for k,v in lm.state_dict().items()}
            with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
                native=runtime.option_logits(lm,prefix,[9,10],torch.device('cpu'))
                ctl=runtime.LayerRouter(lm,hidden_size=16,prefix_tokens=6)
                loaded={'lm':lm,'device':torch.device('cpu'),'controller':ctl}
                self.assertTrue(torch.equal(native,runtime.option_logits(lm,prefix,[9,10],loaded['device'])))
                self.assertEqual(runtime.check_skip_equivalence(loaded,prefix,{'choice_token_ids':[[],[9,10]]})['status'],'PASS')
                for layer in config.CONDITIONS:
                    ctl.set_drop(layer);value=runtime.option_logits(lm,prefix,[9,10],loaded['device'])
                    self.assertTrue(bool(torch.isfinite(value).all()))
                    self.assertEqual(ctl.trace,[i for i in range(1,31) if i!=layer])
            ctl.set_drop(None)
            for key,value in lm.state_dict().items():self.assertTrue(torch.equal(value,weights[key]))

    def test_real_prefix383_shifted_body_sum_matches_stepwise_and_skips_empty(self):
        for backend in ('eager','sdpa'):
            torch,lm=self.make_model(backend);device=torch.device('cpu')
            runtime.LayerRouter(lm,hidden_size=16,prefix_tokens=383).set_drop(17)
            prefix=lm.get_input_embeddings()(torch.randint(1,41,(1,383))).to(torch.bfloat16)
            ids=[9,10,11];expected=0.;history=prefix
            with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
                for token in ids:
                    logits=lm(inputs_embeds=history,attention_mask=None,use_cache=False).logits[0,-1].float()
                    expected+=float(logits.log_softmax(-1)[token])
                    history=torch.cat([history,lm.get_input_embeddings()(torch.tensor([[token]])).to(torch.bfloat16)],dim=1)
                actual=runtime.option_score(lm,prefix,ids,device)
                self.assertAlmostEqual(actual,expected,places=5)
                record=dataset.tokenize(dataset.normalize(0,{**raw_fixture()[0],'choices':['one','','two']}),FixtureTokenizer())
                loaded={'lm':lm,'device':device}
                with patch.object(runtime,'option_score',wraps=runtime.option_score) as score:
                    value=runtime.score(loaded,prefix,record)
                self.assertEqual(score.call_count,2);self.assertIsNone(value['option_loglikelihood'][1])

    def test_real_cached125_prefixes_encode_once_resume_and_wrong_identity_rejected(self):
        import torch
        records=records_fixture();expected={'identity_sha256':'synthetic-identity'}
        precision={k:'fixture' for k in ('torch','transformers','cuda','gpu','weights_dtype','forward_dtype',
                  'audio_frontend_dtype','prefix_dtype','attention_implementation','sdpa_backends')}
        loaded={'audit':precision,'device':torch.device('cpu')}
        with patch.object(runtime,'prefix',return_value=torch.zeros((1,383,576),dtype=torch.bfloat16)) as encode, \
             patch.object(cache.audio,'load_fixed20',return_value=(None,{'fixture':True})),contextlib.redirect_stdout(io.StringIO()):
            values,_=cache.prepare_local(loaded,records,self.path,expected,0)
            self.assertEqual(encode.call_count,125);self.assertEqual(sorted(values),config.indices(0))
            cache.prepare_local(loaded,records,self.path,expected,0);self.assertEqual(encode.call_count,125)
            with self.assertRaises(ValueError):cache.prepare_local(loaded,records,self.path,{'identity_sha256':'other'},0)

    def test_real_soundfile_stereo_resample_first20_padding_without_torchcodec(self):
        import numpy as np
        import soundfile as sf
        from mizar_layer_pair_search import audio
        path=self.path/'stereo.wav'
        values=np.stack([np.ones(160,dtype=np.float32)*.25,np.ones(160,dtype=np.float32)*.5],axis=1)
        sf.write(path,values,16000,subtype='FLOAT')
        wave,audit=audio.load_fixed20({'kind':'file','path':str(path)})
        self.assertEqual(tuple(wave.shape),(640000,));self.assertEqual(audit['original_channels'],2)
        self.assertEqual(audit['original_sample_rate'],16000);self.assertEqual(audit['resampled_samples'],320)
        self.assertEqual(audit['crop_start'],0);self.assertEqual(audit['right_pad_samples'],639680)
        self.assertNotIn('torchcodec',sys.modules)


if __name__=='__main__':unittest.main()
