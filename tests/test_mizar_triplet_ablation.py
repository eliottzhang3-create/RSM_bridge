"""Real30-layer triplet identity routing and explicit full42000 decision fixtures."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/RSmol'));sys.path.insert(0,str(ROOT/'tests'))
from mizar_layer_pair_search import storage
from mizar_triplet_ablation import arguments,audit,cache,config,evaluate,identity,manifest,results,runtime,submit,worker
from test_mizar_letter_analysis import records_fixture,FixtureTokenizer


class TokenizerFixture(FixtureTokenizer):all_special_ids=[0,17]


EXPECTED={'identity_sha256':'EXPLICIT_SYNTHETIC_TRIPLET_RUN'}


def execution(start):
    drop=config.drop_layers(start)
    return {'window_start_1based':start,'window_end_1based':None if start is None else start+2,
        'drop_layers_1based':drop,'executed_original_layers_1based':[i for i in range(1,31) if i not in drop],
        'effective_layers':30-len(drop),'checked_forwards':1}


def score_fixture(r,start,correct=True):
    valid=[i for i,v in enumerate(r['choice_valid_mask']) if v];gold=r['gold_index']
    best=gold if correct and gold is not None else next(i for i in valid if i!=gold)
    lp=[-5. if v else None for v in r['choice_valid_mask']];lp[best]=-1.
    return {**runtime.decision(r,lp),'execution':execution(start)}


def row_fixture(r,task,start,correct=True):
    return storage.seal({'identity_sha256':EXPECTED['identity_sha256'],'task':task,'condition':config.condition_name(task,start),
        'condition_layer_1based':start,'global_index':r['global_index'],'key':r['key'],'record_sha256':storage.object_sha(r),
        **score_fixture(r,start,correct)})


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP'));self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)


class ContractsAndInputs(TempCase):
    def test_exact20_windows_baseline_first31based_boundaries_and_disjoint8x3090_commands(self):
        self.assertEqual(config.CONDITIONS,(None,*range(6,26)))
        self.assertEqual([config.drop_layers(i) for i in range(6,26)],[list(range(i,i+3)) for i in range(6,26)])
        self.assertEqual(config.drop_layers(25),[25,26,27])
        for invalid in (True,False,5,26,30,-1,6.0,[6,7,8]):
            with self.assertRaises(ValueError):config.drop_layers(invalid)
        outputs=[]
        for task in config.TASKS:
            a=arguments.parse_args(['--task',task,'--output-dir',str(self.path/(task+"'s output"))]);plan=submit.command(a,'fixture')
            for flag,value in (('-p','pdgpu-3090'),('-g','8'),('-c','64'),('-m','256G'),('-n','1')):
                self.assertEqual(plan[plan.index(flag)+1],value)
            self.assertEqual(arguments.parse_args(arguments.forwarded(a)),a);self.assertIn(str(a.output_dir),shlex.split(plan[-1]))
            self.assertEqual([i for rank in range(8) for i in config.indices(rank)].__len__(),1000)
            outputs.append(a.output_dir)
            self.assertFalse(config.protocol(task)['tracing_subset']);self.assertEqual(len(config.conditions(task)),21)
        self.assertNotEqual(outputs[0],outputs[1])

    def build_fixture(self,task):
        a=arguments.parse_args(['--task',task,'--source-manifest',str(self.path/task/'old_letters/manifest.json'),
            '--body-manifest',str(self.path/task/'old_body/manifest.json'),'--manifest',str(self.path/task/'new/manifest.json')])
        a.source_manifest.parent.mkdir(parents=True);a.source_manifest.write_text('EXPLICIT_SOURCE_FIXTURE',encoding='utf-8')
        audio=self.path/task/'audio.fixture';audio.write_bytes(b'EXPLICIT_AUDIO_FIXTURE_NOT_DECODED')
        src={'task':task,'source_arguments':{'model_dir':str(a.model_dir)},'records':records_fixture(task),
            'files':[storage.file_identity(audio)],'manifest_sha256':'EXPLICIT_SOURCE_SHA'}
        with patch.object(manifest,'load_tokenizer',return_value=(TokenizerFixture(),[])),patch.object(manifest.letters,'load',return_value=src), \
             contextlib.redirect_stdout(io.StringIO()):frozen=manifest.ensure(a)
        return a,frozen,src,audio

    def test_exact_full1000_readonly_letter_manifest_publication_reuse_and_tamper(self):
        for task in config.TASKS:
            a,frozen,src,audio=self.build_fixture(task);before=a.source_manifest.read_bytes()
            self.assertEqual(frozen['records'],src['records']);self.assertEqual(len(frozen['records']),1000)
            self.assertEqual(manifest.load(a.manifest,full=True),frozen)
            with patch.object(manifest,'build',side_effect=AssertionError('existing input must be reused')),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(manifest.ensure(a),frozen)
            self.assertEqual(a.source_manifest.read_bytes(),before)
            bad=copy.deepcopy(frozen);bad.pop('manifest_sha256');bad['records'][0]['gold_index']=3
            with self.assertRaises(ValueError):manifest.validate(storage.seal(bad,'manifest_sha256'))
            audio.write_bytes(b'CHANGED')
            with self.assertRaisesRegex(ValueError,'file identity'):manifest.load(a.manifest,full=True)

    def test_source_missing_uses_body_or_isolated_builder_and_never_publishes_old_paths(self):
        a=arguments.parse_args(['--task','mmar','--source-manifest',str(self.path/'absent_letters.json'),
            '--body-manifest',str(self.path/'body/manifest.json'),'--manifest',str(self.path/'new/manifest.json')])
        letter=records_fixture('mmar');body={'source_arguments':{'model_dir':str(a.model_dir)},'records':letter,'files':[],'manifest_sha256':'body'}
        module=SimpleNamespace(load=lambda *args,**kw:body,build=lambda args:body)
        a.body_manifest.parent.mkdir();a.body_manifest.write_text('BODY_SOURCE')
        with patch.object(manifest,'load_tokenizer',return_value=(TokenizerFixture(),[])),patch.object(manifest.letters,'source_module',return_value=module), \
             patch.object(manifest.letters,'transform_record',side_effect=lambda r,*args,**kw:r):
            result=manifest.build(a);self.assertEqual(result['origin']['kind'],'existing_body_manifest')
            a.body_manifest=self.path/'absent_body.json';result=manifest.build(a)
            self.assertEqual(result['origin']['kind'],'isolated_full_metadata_build')
        self.assertFalse(a.body_manifest.exists());self.assertFalse(a.source_manifest.exists())

    def test_real2000_metadata_release_tokenizer_groups_and_special_slots(self):
        from mizar_layer_pair_search.manifest import load_tokenizer
        from mizar_layer_ablation import dataset as mmau
        from mizar_layer_ablation_mmar import dataset as mmar
        from mizar_letter_analysis.manifest import transform_record
        tokenizer,_=load_tokenizer(Path('C:/Xlance/assets/Mizar/Mizar159M'))
        paths={'mmau':Path('C:/Xlance/assets/Mizar/Mizar_159M-main/Mizar_159M-main/data/evaluation/mmau_mini_official.json'),
            'mmar':Path('C:/Xlance/assets/MMAR/MMAR-meta.json')}
        for task,path in paths.items():
            self.assertTrue(path.is_file());records=[]
            for i,row in enumerate(storage.read_json(path)):
                r=(mmau if task=='mmau' else mmar).normalize(i,row);r['choice_token_ids']=[[30] if body else [] for body in r['choice_bodies']]
                r.update(audio_ref={'kind':'file','path':'EXPLICIT_METADATA_ONLY'},audio_audit={'fixture':True})
                records.append(transform_record(r,tokenizer,task))
            self.assertEqual(dict(__import__('collections').Counter(r['analysis_group'] for r in records)),config.GROUP_COUNTS[task])
            if task=='mmau':self.assertEqual(sum(r['duplicate_choice_bodies'] for r in records),27)
            else:
                self.assertEqual(sum(r['gold_index'] is None for r in records),5)
                self.assertEqual(sum(not all(r['choice_valid_mask']) for r in records),4)
                self.assertEqual(max(len(r['option_letters']) for r in records),6)

    def test_original103_files_and_all_reused_dependency_guards_hold(self):
        snapshot=storage.read_json(ROOT/'tests/fixtures/mizar_triplet_ablation_original_sources.json')
        for name,digest in snapshot.items():self.assertEqual(storage.text_sha(ROOT/name),digest,name)
        self.assertGreaterEqual(len(identity.sources()),140)


class RealNumerics(TempCase):
    def model(self,backend='eager',hidden=16,tokens=383):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.set_num_threads(1);torch.manual_seed(20261127)
        cfg=LlamaConfig(vocab_size=41,hidden_size=hidden,intermediate_size=hidden*2,num_hidden_layers=30,num_attention_heads=2,
            num_key_value_heads=2,max_position_embeddings=1024,use_cache=False,tie_word_embeddings=True)
        cfg._attn_implementation=backend;lm=LlamaForCausalLM(cfg).float().eval().requires_grad_(False)
        prefix=lm.get_input_embeddings()(torch.randint(1,41,(1,tokens))).to(torch.bfloat16)
        return torch,lm,prefix

    def test_all20_triples_real_BF16_eager_sdpa_native_baseline_independent_identity_and_no_parameter_change(self):
        from mizar_letter_analysis.runtime import score as original_score
        for backend in ('eager','sdpa'):
            torch,lm,prefix=self.model(backend);weights={k:v.clone() for k,v in lm.state_dict().items()}
            with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
                native=lm(inputs_embeds=prefix,attention_mask=None,use_cache=False).logits[0,382].float().clone()
                ctl=runtime.TripleRouter(lm,hidden_size=16,prefix_tokens=383)
                loaded={'lm':lm,'controller':ctl,'device':torch.device('cpu'),'task':'mmar'}
                self.assertTrue(torch.equal(runtime.logits(loaded,prefix),native))
                record=records_fixture('mmar')[6]
                self.assertEqual(runtime.checks(loaded,prefix,record)['status'],'PASS')
                for start in config.CONDITIONS:
                    ctl.set_drop(start);actual=runtime.score(loaded,prefix,record)
                    self.assertEqual(actual['execution']['executed_original_layers_1based'],[i for i in range(1,31) if i not in config.drop_layers(start)])
                    self.assertEqual(actual['execution']['effective_layers'],30 if start is None else 27)
                    self.assertEqual(actual['option_loglikelihood'],original_score(loaded,prefix,record)['option_loglikelihood'])
                    self.assertEqual([module.self_attn.layer_idx for module in lm.model.layers],
                        [i-1 for i in range(1,31) if i not in config.drop_layers(start)])
                ctl.set_drop(None);self.assertTrue(torch.equal(runtime.logits(loaded,prefix),native))
                for key,v in lm.state_dict().items():self.assertTrue(torch.equal(v,weights[key]))
                for module in ctl.original:self.assertFalse(hasattr(module,'_triplet_identity_input'))

    def test_real576_dimension_short_prefix_all30_modules_distinct_and_restore(self):
        torch,lm,prefix=self.model('eager',hidden=576,tokens=6)
        with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
            native=lm(inputs_embeds=prefix,attention_mask=None,use_cache=False).logits[0,-1].float().clone()
            ctl=runtime.TripleRouter(lm,hidden_size=576,prefix_tokens=6);loaded={'lm':lm,'controller':ctl,'device':torch.device('cpu')}
            self.assertEqual(runtime.checks(loaded,prefix,records_fixture('mmau')[0])['status'],'PASS')
            self.assertTrue(torch.equal(runtime.logits(loaded,prefix),native));self.assertEqual(len({id(m) for m in ctl.original}),30)

    def test_blank_option_cannot_win_even_with_highest_logit_and_original_MMAR_gt_policy(self):
        import torch
        r=records_fixture('mmar')[4];values=torch.zeros(41);values[r['letter_token_ids'][2]]=100
        loaded={'controller':SimpleNamespace(evidence=lambda:execution(6))}
        with patch.object(runtime,'logits',return_value=values):result=runtime.score(loaded,None,r)
        self.assertIsNone(result['option_loglikelihood'][2]);self.assertNotEqual(result['prediction_index'],2)
        r=records_fixture('mmar')[5]
        with patch.object(runtime,'logits',return_value=values):result=runtime.score(loaded,None,r)
        self.assertFalse(result['correct']);self.assertIsNone(result['gold_index'])

    def test_prefix125_real_BF16_serialization_once_per_question_and_tamper_reject(self):
        import torch
        records=records_fixture('mmau');loaded={'device':torch.device('cpu'),'audit':{k:'fixture' for k in
            ('torch','transformers','cuda','gpu','weights_dtype','forward_dtype','audio_frontend_dtype','prefix_dtype','attention_implementation','sdpa_backends')}}
        with patch.object(runtime,'prefix',return_value=torch.zeros(1,383,576,dtype=torch.bfloat16)) as encode, \
             patch.object(cache.audio,'load_fixed20',return_value=(None,{'fixture':True})),contextlib.redirect_stdout(io.StringIO()):
            values,meta=cache.prepare_local(loaded,records,self.path,EXPECTED,0)
            self.assertEqual(encode.call_count,125);self.assertEqual(len(values),125)
            cache.prepare_local(loaded,records,self.path,EXPECTED,0);self.assertEqual(encode.call_count,125)
            with self.assertRaises(ValueError):cache.prepare_local(loaded,records,self.path,{'identity_sha256':'other'},0)
        path=self.path/'cache/rank00/prefix_0000.pt';path.write_bytes(b'changed')
        with self.assertRaises(ValueError):cache.entry(path.parent,records[0],meta['cache_identity'])


class ResultsAndOrchestration(TempCase):
    def test_all_group_denominators_paired_changes_ranking_ties_and_bad_trace_rejected(self):
        for task in config.TASKS:
            records=records_fixture(task)
            baseline=[row_fixture(r,task,None,r['global_index']<290) for r in records]
            rows=[row_fixture(r,task,6,r['global_index']<390) for r in records]
            summary=results.summarize(rows,baseline,records,task,6)
            self.assertEqual(summary['drop_layers_1based'],[6,7,8]);self.assertEqual(summary['effective_layers'],27)
            self.assertAlmostEqual(summary['delta_percentage_points'],10.0)
            self.assertEqual({g:v['total'] for g,v in summary['groups'].items()},config.GROUP_COUNTS[task])
            bad=copy.deepcopy(rows[0]);bad.pop('sha256');bad['execution']['drop_layers_1based']=[6,7]
            with self.assertRaises(ValueError):results.validate(storage.seal(bad),records[0],task,6,EXPECTED)
            bad=copy.deepcopy(rows[0]);bad.pop('sha256');bad['correct']=not bad['correct']
            with self.assertRaises(ValueError):results.validate(storage.seal(bad),records[0],task,6,EXPECTED)

    def test_same_window_all8rank125_sync_resume_and_complete21condition_grid(self):
        for task in config.TASKS:
            records=records_fixture(task);a=SimpleNamespace(task=task,output_dir=self.path/task);collected={};barriers=[];current={'start':None}
            class Controller:
                def set_drop(self,start):current['start']=start
            class Dist:
                def __init__(self,rank):self.rank=rank
                def broadcast_object_list(self,x,src):x[0]=(task,current['start'],EXPECTED['identity_sha256'])
                def gather_object(self,rows,out,dst):
                    collected[self.rank]=rows
                    if self.rank==0:out[:]=[collected[r] for r in range(8)]
                def barrier(self):barriers.append(current['start'])
            def scored(loaded,p,r):return score_fixture(r,current['start'])
            with patch.object(runtime,'score',side_effect=scored) as infer,contextlib.redirect_stdout(io.StringIO()):
                for start in (None,6,25):
                    current['start']=start
                    for rank in [*range(1,8),0]:worker.evaluate_condition(a,{'controller':Controller()},dict.fromkeys(config.indices(rank)),records,EXPECTED,start,rank,Dist(rank))
                self.assertEqual(infer.call_count,3000)
                for rank in [*range(1,8),0]:worker.evaluate_condition(a,{'controller':Controller()},{},records,EXPECTED,25,rank,Dist(rank))
                self.assertEqual(infer.call_count,3000)
            self.assertEqual(len(barriers),32)
            with patch.object(worker,'evaluate_condition') as cond:worker.work(a,None,None,records,EXPECTED,0,None)
            self.assertEqual([call.args[5] for call in cond.call_args_list],list(config.CONDITIONS))

    def test_partial_tail_and_wrong_identity_or_rank_order_rejected(self):
        records=records_fixture('mmau');path=self.path/'ranks/rank00/drop_06_08.jsonl'
        storage.write_jsonl(path,[row_fixture(records[i],'mmau',6) for i in (0,8)])
        with path.open('ab') as stream:stream.write(b'{partial')
        self.assertEqual(len(results.read_rank(self.path,0,records,'mmau',6,EXPECTED)),2)
        with self.assertRaises(ValueError):results.read_rank(self.path,0,records,'mmau',6,{'identity_sha256':'other'})
        storage.write_jsonl(path,[row_fixture(records[i],'mmau',6) for i in (8,0)])
        with self.assertRaises(ValueError):results.read_rank(self.path,0,records,'mmau',6,EXPECTED)

    def test_coordinator_launches_one_group_per_dataset_and_reports_rank_failure_details(self):
        for task in config.TASKS:
            a=arguments.parse_args(['--task',task,'--output-dir',str(self.path/task)])
            expected=storage.seal({'fixture':True},'identity_sha256')
            with patch.object(manifest,'ensure',return_value={'manifest_sha256':'fixture'}),patch.object(identity,'build',return_value=expected), \
                 patch.object(identity,'verify'),patch.object(evaluate,'launch') as launch,patch.object(evaluate,'finalize',side_effect=[None,{'status':'PASS'}]), \
                 contextlib.redirect_stdout(io.StringIO()):report=evaluate.run(a)
            self.assertEqual(report['status'],'PASS');self.assertEqual(launch.call_count,1)
            storage.write_json(a.output_dir/'ranks/rank02/failure.json',{'status':'FAILED','rank':2,'error':'explicit fixture'})
            with patch.object(evaluate.subprocess,'run',return_value=SimpleNamespace(returncode=1)),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'rank2: explicit fixture'):evaluate.launch(a)

    def test_actual42000_full_result_finalizers_tables_ranking_figures_and_zero_generation(self):
        total=0
        for task in config.TASKS:
            records=records_fixture(task);directory=self.path/task
            for rank in range(8):storage.write_json(directory/'ranks'/f'rank{rank:02d}'/'rank_report.json',{'EXPLICIT_SYNTHETIC':True})
            frozen={'task':task,'records':records,'input_audit':{'EXPLICIT_SYNTHETIC':True}}
            def read(directory,rank,records,task,start,expected):
                threshold=290 if start is None else 390 if start in (6,25) else 350
                return [row_fixture(records[i],task,start,i<threshold) for i in config.indices(rank)]
            with patch.object(results,'read_rank',side_effect=read) as reader,patch.object(audit,'verify',return_value={'sha256':'EXPLICIT_AUDIT'}):
                report=evaluate.finalize(SimpleNamespace(task=task,output_dir=directory),frozen,EXPECTED)
            self.assertEqual(reader.call_count,168);self.assertEqual(report['status'],'PASS');self.assertEqual(report['decisions'],21000)
            self.assertEqual(report['top_tied_windows_1based'],[[6,7,8],[25,26,27]])
            self.assertEqual(report['generation_count'],0);self.assertEqual(len(storage.read_json(directory/'ranking.json')),20)
            self.assertEqual(len((directory/'accuracy_by_window.csv').read_text().splitlines()),22)
            self.assertEqual(len((directory/'accuracy_by_group.csv').read_text().splitlines()),1+21*len(config.GROUP_COUNTS[task]))
            for name in report['plots']['files']:self.assertGreater((directory/name).stat().st_size,1000)
            if os.environ.get('RSMOL_TRIPLET_QA'):
                import shutil
                qa=Path(os.environ['RSMOL_TRIPLET_QA'])/task;qa.mkdir(parents=True,exist_ok=True)
                for name in report['plots']['files']:shutil.copy2(directory/name,qa/name)
            total+=report['decisions']
        self.assertEqual(total,42000)

    def test_final_audit_exact_triplet_probes125cache_inventory_and_wrong_condition_count(self):
        records=records_fixture('mmau');models=[];checks=[];inventories=[]
        expected={**EXPECTED,'task':'mmau'}
        for rank in range(8):
            model={k:'fixture' for k in ('torch','transformers','cuda','gpu','weights_dtype','forward_dtype','audio_frontend_dtype','prefix_dtype',
                'attention_implementation','sdpa_backends')};model['binding']={'rank':rank,'worker_cuda_visible_devices':str(rank)}
            check={'status':'PASS','rank':rank,'identity_sha256':EXPECTED['identity_sha256'],'baseline_restored_bitwise':True,
                'triplet_identity_probes':[{'status':'PASS','start':s,'drop_layers_1based':config.drop_layers(s),'max_abs_error':0.,'execution':execution(s)} for s in (6,15,25)]}
            folder=self.path/'ranks'/f'rank{rank:02d}'
            for name,value in [('model_audit.json',model),('runtime_check.json',check)]:storage.write_json(folder/name,storage.seal(value))
            storage.write_json(folder/'rank_report.json',storage.seal({'status':'PASS','task':'mmau','identity_sha256':EXPECTED['identity_sha256'],
                'indices':config.indices(rank),'conditions_completed':21}))
            directory=self.path/'cache'/f'rank{rank:02d}';directory.mkdir(parents=True);entries=[]
            key={'run_identity_sha256':EXPECTED['identity_sha256'],'precision':cache.precision(model),'rank':rank}
            for i in config.indices(rank):
                path=directory/f'prefix_{i:04d}.pt';path.write_bytes(b'EXPLICIT_SYNTHETIC_NOT_TENSOR')
                meta=storage.seal({'cache_identity':key,'record_sha256':storage.object_sha(records[i]),'audio_audit':records[i]['audio_audit'],
                    'file_sha256':storage.file_sha(path)})
                storage.write_json(directory/f'prefix_{i:04d}.json',meta);entries.append(meta['sha256'])
            inventory=storage.seal({'cache_identity':key,'indices':config.indices(rank),'entry_meta_sha256':entries,'status':'PASS'})
            storage.write_json(directory/'cache_manifest.json',inventory);models.append(model);checks.append(check);inventories.append(inventory)
        storage.write_json(self.path/'runtime_checks.json',storage.seal({'status':'PASS','identity_sha256':EXPECTED['identity_sha256'],
            'rank_audits':models,'rank_checks':checks,'cache_manifests':inventories}))
        self.assertEqual(audit.verify(self.path,expected,records)['status'],'PASS')
        report_path=self.path/'ranks/rank00/rank_report.json';saved=storage.read_json(report_path)
        broken={k:v for k,v in saved.items() if k!='sha256'};broken['conditions_completed']=20
        storage.write_json(report_path,storage.seal(broken))
        with self.assertRaisesRegex(ValueError,'125x21'):audit.verify(self.path,expected,records)
        storage.write_json(report_path,saved)
        path=self.path/'cache/rank00/prefix_0000.pt';path.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):audit.verify(self.path,expected,records)


if __name__=='__main__':unittest.main()
