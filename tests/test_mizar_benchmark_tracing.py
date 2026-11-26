"""Full1000 input contracts, real numerical tracing and synthetic eight-rank flows."""
import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import random
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/RSmol'));sys.path.insert(0,str(ROOT/'tests'))
from mizar_layer_pair_search import storage
from mizar_benchmark_tracing import arguments,audit,config,evaluate,identity,plot,prepare,results,runtime,scoring,screening,submit,worker
from test_mizar_letter_analysis import records_fixture as letter_records


class TokenizerFixture:
    all_special_ids=[0,17]
    def encode(self,text,add_special_tokens=True):
        if len(text)==1 and 'a'<=text<='z':return [3+ord(text)-97]
        return [30+ord(c)%10 for c in text]


EXPECTED={'identity_sha256':'EXPLICIT_SYNTHETIC_RUN'}


def inputs_fixture(task):
    records=[prepare.transform(r,i,TokenizerFixture(),task) for i,r in enumerate(letter_records(task))]
    order=list(range(1000));random.Random(1234).shuffle(order)
    return storage.seal({'schema':config.SCHEMA,'task':task,'protocol':config.protocol(task),'records':records,
        'shuffle_order':order,'group_counts':config.GROUPS[task],'source_arguments':{},'files':[],
        'generator_sources':prepare.generator_sources()},'manifest_sha256')


def score(r,correct=True,p=.6):
    valid=[i for i,v in enumerate(r['choice_valid_mask']) if v];gold=r['gold_index']
    best=gold if correct and gold is not None else next(i for i in valid if i!=gold)
    values=[math.log(.025) if enabled else None for enabled in r['choice_valid_mask']]
    values[best]=math.log(.7 if not correct else p)
    if gold is not None and not correct:values[gold]=math.log(p)
    return scoring.decision(r,values)


def screen_row(r,pos,accepted=True):
    clean=score(r,correct=accepted,p=.6 if accepted else .1)
    corrupted=score(r,False,p=.1)
    reason=scoring.reason(r,clean,corrupted)
    return storage.seal(screening.make_row(r,EXPECTED,pos,clean,corrupted,r['audio_audit'],
        'EXPLICIT_CACHE_DIGEST' if reason=='accepted' else None))


def full_screen(inputs,accepted_indices):
    return [screen_row(inputs['records'][index],pos,index in accepted_indices) for pos,index in enumerate(inputs['shuffle_order'])]


def shard_screen(ordered):return [[r for r in ordered if r['shuffle_position']%8==rank] for rank in range(8)]


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP'));self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)


class InputAndScoring(TempCase):
    def test_both1000_inputs_same_slots_groups_no_new_sampling_and_frozen_hash(self):
        for task in config.TASKS:
            inputs=inputs_fixture(task);self.assertEqual(prepare.validate(inputs),inputs)
            self.assertEqual(len(inputs['records']),1000)
            self.assertEqual(dict(__import__('collections').Counter(r['analysis_group'] for r in inputs['records'])),config.GROUPS[task])
            r=inputs['records'][0];bad=copy.deepcopy(inputs);bad.pop('manifest_sha256');bad['records'][0]['prompt']='changed <|endoftext|>'
            with self.assertRaises(ValueError):prepare.validate(storage.seal(bad,'manifest_sha256'))
            self.assertEqual([r['source_manifest_index'] for r in inputs['records']],list(range(1000)))

    def test_real_local_metadata_and_release_tokenizer_all2000_questions(self):
        from mizar_layer_pair_search.manifest import load_tokenizer
        from mizar_layer_ablation import dataset as mmau
        from mizar_layer_ablation_mmar import dataset as mmar
        from mizar_letter_analysis import manifest as letters
        tokenizer,_=load_tokenizer(Path('C:/Xlance/assets/Mizar/Mizar159M'))
        paths={'mmau':Path('C:/Xlance/assets/Mizar/Mizar_159M-main/Mizar_159M-main/data/evaluation/mmau_mini_official.json'),
               'mmar':Path('C:/Xlance/assets/MMAR/MMAR-meta.json')}
        for task,path in paths.items():
            self.assertTrue(path.is_file());raw=storage.read_json(path);rows=[]
            for i,r in enumerate(raw):
                base=(mmau if task=='mmau' else mmar).normalize(i,r)
                base['choice_token_ids']=[[30] if body else [] for body in base['choice_bodies']]
                base.update(audio_ref={'kind':'file','path':'EXPLICIT_METADATA_ONLY'},audio_audit={'fixture':True})
                letter=letters.transform_record(base,tokenizer,task,prepare.mapping() if task=='mmau' else None)
                rows.append(prepare.transform(letter,i,tokenizer,task))
            self.assertEqual(len(rows),1000)
            self.assertEqual(dict(__import__('collections').Counter(r['analysis_group'] for r in rows)),config.GROUPS[task])
            if task=='mmau':self.assertEqual(sum(r['duplicate_choice_bodies'] for r in rows),27)
            else:
                self.assertEqual(sum(r['gold_index'] is None for r in rows),5)
                self.assertEqual(sum(not all(r['choice_valid_mask']) for r in rows),4)
                self.assertEqual(max(len(r['candidate_labels']) for r in rows),6)

    def test_CPU_source_priority_and_fallbacks_publish_only_new_manifest(self):
        task='mmau';a=arguments.parse_args(['--task',task,'--source-manifest',str(self.path/'letters/manifest.json'),
            '--body-manifest',str(self.path/'body/manifest.json'),'--manifest',str(self.path/'new/manifest.json')])
        letter_rows=letter_records(task)
        letter_source={'task':task,'source_arguments':{'model_dir':str(a.model_dir)},'records':letter_rows,'files':[],'manifest_sha256':'letters-source'}
        body_source={'source_arguments':{'model_dir':str(a.model_dir)},'records':letter_rows,'files':[],'manifest_sha256':'body-source'}
        module=SimpleNamespace(load=lambda *a,**k:body_source,build=lambda a:body_source)
        a.source_manifest.parent.mkdir();a.source_manifest.write_text('EXPLICIT_SOURCE',encoding='utf-8')
        before=a.source_manifest.read_bytes()
        with patch.object(prepare,'load_tokenizer',return_value=(TokenizerFixture(),[])), \
             patch.object(prepare.letters,'load',return_value=letter_source):
            built=prepare.build(a)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prepare.ensure(a),built)
                self.assertEqual(prepare.ensure(a),built)
        self.assertEqual(built['origin']['kind'],'existing_letter_manifest');self.assertEqual(a.source_manifest.read_bytes(),before)
        self.assertEqual(prepare.load(a.manifest,full=True),built)
        a.source_manifest=self.path/'absent_letters.json';a.body_manifest.parent.mkdir();a.body_manifest.write_text('BODY_FIXTURE')
        with patch.object(prepare,'load_tokenizer',return_value=(TokenizerFixture(),[])),patch.object(prepare.letters,'source_module',return_value=module), \
             patch.object(prepare.letters,'transform_record',side_effect=lambda r,*args,**kw:r):
            built=prepare.build(a);self.assertEqual(built['origin']['kind'],'existing_body_manifest')
            a.body_manifest=self.path/'absent_body.json'
            built=prepare.build(a);self.assertEqual(built['origin']['kind'],'isolated_metadata_build')
        self.assertFalse(a.source_manifest.exists());self.assertFalse(a.body_manifest.exists())

    def test_MMAR_blank_slot_and_six_choices_full_vocab_probability_and_missing_gold(self):
        import torch
        inputs=inputs_fixture('mmar');r=inputs['records'][4]
        logits=torch.zeros(41);logits[r['candidate_token_ids'][2]]=100.
        value=scoring.from_logits(r,logits)
        self.assertIsNone(value['candidate_logprobs'][2]);self.assertIsNone(value['candidate_probabilities'][2])
        self.assertNotEqual(value['prediction_index'],2)
        r=inputs['records'][6];logits=torch.zeros(41);logits[r['candidate_token_ids'][5]]=4;logits[40]=6
        value=scoring.from_logits(r,logits);self.assertEqual(value['prediction_label'],'f')
        expected=float(logits.log_softmax(-1)[r['candidate_token_ids'][5]])
        self.assertEqual(value['target_logprob'],expected);self.assertEqual(value['target_probability'],math.exp(expected))
        self.assertLess(sum(p for p in value['candidate_probabilities'] if p is not None),1)
        r=inputs['records'][5];value=scoring.from_logits(r,logits);self.assertFalse(value['correct'])
        self.assertIsNone(value['target_probability']);self.assertIsNone(value['target_logprob'])
        self.assertEqual(scoring.reason(r,value,value),'unmapped_gold');scoring.validate(r,value)

    def test_letter_decisions_match_existing_ablation_protocol_for_all_slots(self):
        from mizar_letter_analysis.runtime import decision as original_decision
        for task in config.TASKS:
            for r in inputs_fixture(task)['records']:
                lps=[-1.-i if enabled else None for i,enabled in enumerate(r['choice_valid_mask'])]
                old=original_decision(r,lps);new=scoring.decision(r,lps)
                self.assertEqual((new['prediction_index'],new['correct'],new['tie']),
                    (old['prediction_index'],old['correct'],old['option_score_tie']))

    def test_exp_ULP_validation_only_and_all_rejection_reasons(self):
        inputs=inputs_fixture('mmau');r=inputs['records'][0];clean=score(r,True,.6);wrong=score(r,False,.1)
        self.assertEqual(scoring.reason(r,clean,wrong),'accepted')
        self.assertEqual(scoring.reason(r,wrong,wrong),'clean_wrong')
        self.assertEqual(scoring.reason(r,clean,clean),'silence_correct')
        self.assertEqual(scoring.reason(r,clean,score(r,False,.65)),'nonpositive_probability_gap')
        changed=copy.deepcopy(clean)
        for i,p in enumerate(changed['candidate_probabilities']):changed['candidate_probabilities'][i]=math.nextafter(p,math.inf)
        changed['target_probability']=changed['candidate_probabilities'][r['gold_index']]
        scoring.validate(r,changed);scoring.same(r,changed,clean)
        changed['prediction_index']=(changed['prediction_index']+1)%4
        with self.assertRaises(ValueError):scoring.validate(r,changed)
        changed=copy.deepcopy(clean);changed['candidate_probabilities'][0]+=1e-7
        with self.assertRaises(ValueError):scoring.validate(r,changed)
        saturated=scoring.decision(r,[0.0,-10.,-11.,-12.]);saturated['target_logprob']=False
        with self.assertRaises(ValueError):scoring.validate(r,saturated)


class ScreeningAndSchedule(TempCase):
    def test_full1000_both_baselines_dynamic_all_selected_and_group_empty(self):
        for task in config.TASKS:
            inputs=inputs_fixture(task);accepted={i for i,r in enumerate(inputs['records']) if r['analysis_group']==list(config.GROUPS[task])[1] and i%3==0}
            ordered=full_screen(inputs,accepted);merged=screening.merge(shard_screen(ordered),inputs,EXPECTED)
            directory=self.path/task
            with contextlib.redirect_stdout(io.StringIO()):frozen,report=screening.publish(directory,merged,inputs,EXPECTED)
            self.assertEqual(report['total'],1000);self.assertEqual(frozen['selected'],len(accepted))
            self.assertEqual(report['clean_correct'],len(accepted));self.assertEqual(report['corrupted_correct'],0)
            self.assertEqual(frozen['group_counts'][list(config.GROUPS[task])[0]],0)
            for rank,rows in enumerate(shard_screen(ordered)):storage.write_jsonl(screening.rank_path(directory,rank),rows)
            self.assertEqual(screening.reconstruct(directory,inputs,EXPECTED),(frozen,report))
            with screening.rank_path(directory,0).open('ab') as stream:stream.write(b'{partial')
            self.assertEqual(len(screening.read_rank(directory,0,inputs,EXPECTED)),125)
            bad=copy.deepcopy(ordered[0]);bad.pop('sha256');bad['clean']['correct']=not bad['clean']['correct'];bad=storage.seal(bad)
            with self.assertRaises(ValueError):screening.validate(bad,inputs['records'][bad['source_manifest_index']],EXPECTED,0)

    def test_dynamic_shards_zero_one_seven_uneven_and_full1000(self):
        for n in (0,1,7,8,9,137,500,999,1000):
            parts=[config.indices(rank,n) for rank in range(8)]
            self.assertEqual(sorted(i for p in parts for i in p),list(range(n)))
            self.assertLessEqual(max(map(len,parts))-min(map(len,parts)),1)

    def test_screen_one_always_runs_both_inputs_including_clean_wrong_and_null_GT(self):
        import torch
        from mizar_layer_pair_search import audio
        from mizar_reasonaqa_tracing import cache
        a=SimpleNamespace(output_dir=self.path);r=inputs_fixture('mmar')['records'][5]
        cp=torch.zeros(1,383,576,dtype=torch.bfloat16);states=torch.ones(30,256,576,dtype=torch.bfloat16)
        logits=torch.zeros(41)
        with patch.object(audio,'load_fixed20',return_value=(torch.zeros(640000),r['audio_audit'])), \
             patch.object(runtime,'prefix',return_value=cp),patch.object(runtime,'forward',side_effect=[(logits,states),(logits,None)]) as fwd, \
             patch.object(cache,'save') as save:
            row=worker.screen_one(a,{},r,EXPECTED,0,torch.zeros(1,126,576,dtype=torch.bfloat16))
        self.assertEqual(fwd.call_count,2);self.assertEqual(row['reason'],'unmapped_gold');save.assert_not_called()

    def test_real_accepted_screen_cache_lossless(self):
        import torch
        from mizar_layer_pair_search import audio
        from mizar_reasonaqa_tracing import cache
        inputs=inputs_fixture('mmau');r=inputs['records'][0];r['gold_index']=0;r['gold_label']='a'
        a=SimpleNamespace(output_dir=self.path)
        cp=torch.randn(1,383,576,dtype=torch.bfloat16);silence=torch.zeros(1,126,576,dtype=torch.bfloat16)
        states=torch.ones(30,256,576,dtype=torch.bfloat16);clean=torch.zeros(41);clean[3]=5;corrupt=torch.zeros(41);corrupt[4]=5
        loaded={'audit':{k:'fixture' for k in ('torch','transformers','cuda','gpu','weights_dtype','forward_dtype','audio_frontend_dtype',
            'prefix_dtype','attention_implementation','sdpa_backends')}}
        with patch.object(audio,'load_fixed20',return_value=(torch.zeros(640000),r['audio_audit'])),patch.object(runtime,'prefix',return_value=cp), \
             patch.object(runtime,'forward',side_effect=[(clean,states),(corrupt,None)]):row=worker.screen_one(a,loaded,r,EXPECTED,0,silence)
        self.assertTrue(row['accepted'])
        data,meta=cache.load(self.path,r,EXPECTED,digest=row['cache_meta_sha256'])
        self.assertTrue(torch.equal(data['clean_states'],states));self.assertTrue(torch.equal(data['clean_prefix'],cp))
        self.assertTrue(torch.equal(data['corrupted_prefix'][:,126:],cp[:,126:]))
        self.assertEqual(meta['precision'],cache.precision(loaded['audit']))

    def test_completed_screen_rank_resume_reads_cache_without_reencoding_or_LM_forward(self):
        from mizar_layer_pair_search import audio
        from mizar_reasonaqa_tracing import cache
        inputs=inputs_fixture('mmau');rank=7;accepted_index=inputs['shuffle_order'][rank]
        ordered=full_screen(inputs,{accepted_index});parts=shard_screen(ordered)
        for r,rows in enumerate(parts):storage.write_jsonl(screening.rank_path(self.path,r),rows)
        with contextlib.redirect_stdout(io.StringIO()):frozen,_=screening.publish(self.path,ordered,inputs,EXPECTED)
        loaded={'audit':{k:'fixture' for k in ('torch','transformers','cuda','gpu','weights_dtype','forward_dtype',
            'audio_frontend_dtype','prefix_dtype','attention_implementation','sdpa_backends')}}
        class Dist:
            def gather_object(self,*args,**kwargs):pass
            def barrier(self):pass
        with patch.object(audio,'prepare_selected_rows') as prepare_audio,patch.object(audio,'clear_selected_rows'), \
             patch.object(cache,'load',return_value=(None,{})) as reader,patch.object(worker,'screen_one') as infer:
            actual=worker.screen(SimpleNamespace(task='mmau',output_dir=self.path),loaded,inputs,EXPECTED,None,rank,Dist())
        self.assertEqual(actual,frozen);infer.assert_not_called();self.assertEqual(reader.call_count,1)
        self.assertEqual(prepare_audio.call_args.args[0],[])

    def test_same_layer_eight_rank_sync_dynamic_N_and_no_repeated_forward_on_resume(self):
        inputs=inputs_fixture('mmau');ordered=full_screen(inputs,set(range(13)))
        frozen=screening.selected_manifest(ordered,inputs,EXPECTED);a=SimpleNamespace(task='mmau',output_dir=self.path)
        collected={};barriers=[];layer=7
        class Dist:
            def __init__(self,rank):self.rank=rank
            def broadcast_object_list(self,x,src):x[0]=(layer,frozen['manifest_sha256'])
            def gather_object(self,rows,out,dst):
                collected[self.rank]=rows
                if self.rank==0:out[:]=[collected[r] for r in range(8)]
            def barrier(self):barriers.append(layer)
        tensors={i:{'corrupted_prefix':i,'clean_states':'fixture'} for i in range(13)}
        with patch.object(runtime,'forward',side_effect=lambda *a,**kw:(None,None)) as fwd, \
             patch.object(scoring,'from_logits',side_effect=lambda r,l:score(r,True,.35)),contextlib.redirect_stdout(io.StringIO()):
            for rank in [*range(1,8),0]:worker.evaluate_layer(a,{},tensors,frozen,EXPECTED,layer,rank,Dist(rank))
            self.assertEqual(fwd.call_count,13)
            for rank in [*range(1,8),0]:worker.evaluate_layer(a,{},tensors,frozen,EXPECTED,layer,rank,Dist(rank))
            self.assertEqual(fwd.call_count,13)
        self.assertEqual(len(barriers),16)

    def test_idle_rank_controls_use_shared_probe_and_zero_N_skips_safely(self):
        import torch
        from mizar_reasonaqa_tracing import cache
        inputs=inputs_fixture('mmau');frozen=screening.selected_manifest(full_screen(inputs,{0}),inputs,EXPECTED)
        a=SimpleNamespace(output_dir=self.path);precision={k:'fixture' for k in ('torch','transformers','cuda','gpu','weights_dtype','forward_dtype',
            'audio_frontend_dtype','prefix_dtype','attention_implementation','sdpa_backends')}
        loaded={'audit':precision,'device':torch.device('cpu')};data={'clean_prefix':torch.zeros(1,383,576,dtype=torch.bfloat16),
            'corrupted_prefix':torch.zeros(1,383,576,dtype=torch.bfloat16),'clean_states':torch.zeros(30,256,576,dtype=torch.bfloat16)}
        with patch.object(cache,'load',return_value=(data,{})) as reader,patch.object(runtime,'controls',return_value={'status':'PASS'}) as checks:
            result=worker.check_rank(a,loaded,inputs,frozen,EXPECTED,7,torch.zeros(1,126,576,dtype=torch.bfloat16),{})
        self.assertEqual(result['probe_selected_index'],0);self.assertEqual(reader.call_count,1);self.assertEqual(checks.call_count,1)
        empty=screening.selected_manifest(full_screen(inputs,set()),inputs,EXPECTED)
        with patch.object(cache,'load') as reader:
            result=worker.check_rank(a,loaded,inputs,empty,EXPECTED,7,torch.zeros(1,126,576,dtype=torch.bfloat16),{})
        self.assertEqual(result['status'],'SKIPPED_NO_QUALIFYING');reader.assert_not_called()


class RealNumerics(TempCase):
    def test_real30_layer_BF16_blank_slot_full_vocab_capture_patching_and_no_accumulation(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.set_num_threads(1);torch.manual_seed(1234)
        r=inputs_fixture('mmar')['records'][4]
        for backend in ('eager','sdpa'):
            cfg=LlamaConfig(vocab_size=41,hidden_size=16,intermediate_size=32,num_hidden_layers=30,num_attention_heads=2,
                num_key_value_heads=2,max_position_embeddings=1024,tie_word_embeddings=True,use_cache=False)
            cfg._attn_implementation=backend;lm=LlamaForCausalLM(cfg).eval().requires_grad_(False)
            loaded={'lm':lm,'device':torch.device('cpu')};cp=lm.get_input_embeddings()(torch.randint(1,41,(1,383))).to(torch.bfloat16)
            sp=cp.clone();sp[:,:126]=torch.randn_like(sp[:,:126]);weights={k:v.clone() for k,v in lm.state_dict().items()}
            with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
                native=lm(inputs_embeds=cp,attention_mask=None,use_cache=False).logits[0,-1].float().cpu()
                clean,states=runtime.forward(loaded,cp,capture=True);corrupt,_=runtime.forward(loaded,sp)
                self.assertTrue(torch.equal(native,clean));self.assertEqual(states.shape,(30,256,16))
                check=runtime.controls(loaded,cp,sp,states,r,scoring.from_logits(r,clean),scoring.from_logits(r,corrupt))
                self.assertEqual(check['status'],'PASS');self.assertTrue(check['layer30_clean_bitwise'])
                for layer in range(1,31):
                    logits,_=runtime.forward(loaded,sp,layer=layer,clean_states=states)
                    self.assertIsNone(scoring.from_logits(r,logits)['candidate_probabilities'][2])
                again,_=runtime.forward(loaded,sp);self.assertTrue(torch.equal(again,corrupt))
                for k,v in lm.state_dict().items():self.assertTrue(torch.equal(v,weights[k]))


class FinalReports(TempCase):
    def fixtures(self,task,accepted):
        inputs=inputs_fixture(task);ordered=full_screen(inputs,accepted);directory=self.path/task;directory.mkdir(exist_ok=True)
        for rank,rows in enumerate(shard_screen(ordered)):storage.write_jsonl(screening.rank_path(directory,rank),rows)
        with contextlib.redirect_stdout(io.StringIO()):frozen,selection=screening.publish(directory,ordered,inputs,EXPECTED)
        return SimpleNamespace(task=task,stage='all',output_dir=directory),inputs,frozen,selection

    def test_both_full1000_dynamic_finalizers_groups_tables_four_figures_and_heatmap_map(self):
        for task in config.TASKS:
            # Explicit synthetic model outputs, preserving actual1000 group sizes.
            a,inputs,frozen,selection=self.fixtures(task,set(range(0,1000,7)))
            for rank in range(8):storage.write_json(a.output_dir/'ranks'/f'rank{rank:02d}'/'rank_report.json',{'fixture':True})
            def read(directory,rank,frozen,expected,layer):
                return [storage.seal(results.make_row(frozen['records'][i],expected,frozen,layer,
                    frozen['records'][i]['screening']['clean'] if layer==30 else score(frozen['records'][i],True,.1+.5*layer/30)))
                    for i in config.indices(rank,frozen['selected'])]
            with patch.object(results,'read_rank',side_effect=read),patch.object(audit,'verify',return_value={'sha256':'SYNTHETIC_AUDIT'}):
                report=evaluate.finalize(a,inputs,EXPECTED)
            self.assertEqual(report['status'],'PASS');self.assertEqual(report['screened'],1000)
            self.assertEqual(report['patched_decisions'],frozen['selected']*30);self.assertEqual(len(report['plots']),8)
            self.assertEqual(len((a.output_dir/'layer_scores.csv').read_text().splitlines()),1+30*(1+len(config.GROUPS[task])))
            rows=[json.loads(s) for s in (a.output_dir/'heatmap_row_map.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),frozen['selected']);self.assertEqual(sorted(r['selected_index'] for r in rows),list(range(frozen['selected'])))
            self.assertEqual(storage.read_json(a.output_dir/'layer_scores.json')[-1]['mean_RR'],1.)
            for p in report['plots']:self.assertGreater(Path(p).stat().st_size,1000)
            if os.environ.get('RSMOL_BENCHMARK_TRACING_QA'):
                import shutil
                qa=Path(os.environ['RSMOL_BENCHMARK_TRACING_QA'])/task;qa.mkdir(parents=True,exist_ok=True)
                for p in report['plots']:shutil.copy2(p,qa/Path(p).name)

    def test_zero_selected_is_completed_not_quota_failure_and_produces_explained_empty_figures(self):
        a,inputs,frozen,selection=self.fixtures('mmau',set())
        self.assertEqual(selection['status'],'NO_QUALIFYING');self.assertEqual(frozen['selected'],0)
        for rank in range(8):storage.write_json(a.output_dir/'ranks'/f'rank{rank:02d}'/'rank_report.json',{'fixture':True})
        with patch.object(audit,'verify',return_value={'sha256':'SYNTHETIC_AUDIT'}),patch.object(results,'read_rank') as read:
            report=evaluate.finalize(a,inputs,EXPECTED)
        self.assertEqual(report['status'],'NO_QUALIFYING');self.assertEqual(report['patched_decisions'],0);read.assert_not_called()
        self.assertEqual(len(report['plots']),8);self.assertEqual(storage.read_json(a.output_dir/'layer_scores.json'),[])

    def test_rank_audit_checks_dynamic_cache_inventory_and_completion(self):
        from mizar_reasonaqa_tracing import cache
        a,inputs,frozen,selection=self.fixtures('mmar',{0,1,6})
        models=[];checks=[]
        for rank in range(8):
            model={k:'fixture' for k in ('torch','transformers','cuda','gpu','weights_dtype','forward_dtype','audio_frontend_dtype',
                'prefix_dtype','attention_implementation','sdpa_backends')};model['binding']={'rank':rank,'worker_cuda_visible_devices':str(rank)}
            own=config.indices(rank,3);check={'status':'PASS','rank':rank,'identity_sha256':EXPECTED['identity_sha256'],
                'probe_selected_index':own[0] if own else 0,'state_shape':[30,256,576],'exp_max_ULPs':8,'silence_sha256':'fixture',
                'capture_native_bitwise':True,'layer0_corrupted_bitwise':True,'layer30_clean_bitwise':True,
                'cache_recapture_bitwise':True,'saved_logprobs_exact':True}
            folder=a.output_dir/'ranks'/f'rank{rank:02d}'
            for name,val in [('model_audit.json',model),('runtime_check.json',check)]:storage.write_json(folder/name,storage.seal(val))
            storage.write_json(folder/'rank_report.json',storage.seal({'status':'PASS','task':'mmar','identity_sha256':EXPECTED['identity_sha256'],
                'manifest_sha256':frozen['manifest_sha256'],'indices':own,'layers':list(range(1,31))}))
            storage.write_json(folder/'cache_inventory.json',storage.seal({'identity_sha256':EXPECTED['identity_sha256'],
                'entries':[{'selected_index':i,'source_manifest_index':frozen['records'][i]['source_manifest_index'],'meta_sha256':'EXPLICIT_CACHE_DIGEST'} for i in own]}))
            models.append(model);checks.append(check)
        storage.write_json(a.output_dir/'runtime_checks.json',storage.seal({'status':'PASS','identity_sha256':EXPECTED['identity_sha256'],
            'rank_audits':models,'rank_checks':checks}))
        with patch.object(cache,'load',return_value=(None,{'sha256':'EXPLICIT_CACHE_DIGEST','audio_audit':{'fixture':True}})) as reader:
            self.assertEqual(audit.verify(a.output_dir,inputs,frozen,EXPECTED)['status'],'PASS');self.assertEqual(reader.call_count,3)
            path=a.output_dir/'ranks/rank00/rank_report.json';bad=storage.read_json(path);bad.pop('sha256');bad['indices']=[];storage.write_json(path,storage.seal(bad))
            with self.assertRaisesRegex(ValueError,'exact dynamic'):audit.verify(a.output_dir,inputs,frozen,EXPECTED)

    def test_coordinator_single_launch_and_both_submit_resource_and_argument_contracts(self):
        for task in config.TASKS:
            a=arguments.parse_args(['--task',task,'--output-dir',str(self.path/(task+"'s output"))]);plan=submit.command(a,'fixture')
            for flag,val in (('-p','pdgpu-5090'),('-g','8'),('-c','64'),('-m','256G'),('-n','1')):self.assertEqual(plan[plan.index(flag)+1],val)
            self.assertEqual(arguments.parse_args(arguments.forwarded(a)),a);self.assertIn(str(a.output_dir),shlex.split(plan[-1]))
            expected=storage.seal({'fixture':True},'identity_sha256')
            with patch.object(prepare,'ensure',return_value={'shuffle_order':list(range(1000)),'manifest_sha256':'fixture'}), \
                 patch.object(identity,'build',return_value=expected),patch.object(identity,'verify'),patch.object(evaluate,'launch') as launch, \
                 patch.object(evaluate,'finalize',side_effect=[None,{'status':'PASS'}]),contextlib.redirect_stdout(io.StringIO()):
                report=evaluate.run(a)
            self.assertEqual(report['status'],'PASS');self.assertEqual(launch.call_count,1)

    def test_existing138_files_unchanged_and_all_reused_source_guards_hold(self):
        snapshot=storage.read_json(ROOT/'tests/fixtures/mizar_benchmark_tracing_original_sources.json')
        for name,digest in snapshot.items():self.assertEqual(storage.text_sha(ROOT/name),digest,name)
        self.assertGreaterEqual(len(identity.sources()),170)


if __name__=='__main__':unittest.main()
