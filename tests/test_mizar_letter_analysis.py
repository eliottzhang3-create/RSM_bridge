"""Letter/EOT reruns: real numerical checks and explicit three-task fixtures."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/RSmol'))
from mizar_layer_pair_search import storage
from mizar_letter_analysis import arguments,audit,cache,config,cutoff_cache,evaluate,identity,manifest,results,runtime,submit,worker


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP'))
        self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)


class FixtureTokenizer:
    """Explicit tokenizer fixture; actual release tokenizer is audited remotely."""
    def encode(self,text,add_special_tokens=True):
        if len(text)==1 and 'a'<=text<='z':return [3+ord(text)-97]
        suffix=text.endswith(config.EOT_SUFFIX)
        body=text[:-len(config.EOT_SUFFIX)] if suffix else text
        return [30+ord(c)%10 for c in body]+([0] if suffix else [])

    def __call__(self,text,**kwargs):
        ids=self.encode(text)[:256]
        return {'input_ids':ids+[17]*(256-len(ids)),'attention_mask':[1]*len(ids)+[0]*(256-len(ids))}


def source_records(task):
    from mizar_layer_ablation import dataset as mmau
    from mizar_layer_ablation_mmar import dataset as mmar
    rows=[]
    mapping=storage.read_json(config.PACKAGE/'mmau_mapping.json')['groups']
    groups=[g for g,n in config.GROUP_COUNTS[task].items() for _ in range(n)]
    for i,group in enumerate(groups):
        if task=='mmau':
            raw={'id':str(i),'question':f'Fixture {i}?','choices':['one','two','three','four'],
                 'answer':['one','two','three','four'][i%4],
                 'task':'music' if i<334 else 'sound' if i<667 else 'speech',
                 'category':'fixture','sub-category':mapping[group]['subcategories'][i%len(mapping[group]['subcategories'])]}
            record=mmau.normalize(i,raw)
        elif task=='mmar':
            raw={'id':str(i),'question':f'Fixture {i}?','choices':['one','two','three','four'],
                 'answer':['one','two','three','four'][i%4],'category':group,
                 'sub-category':'fixture','modality':'sound','language':'en','audio_path':'./audio/fixture.wav'}
            if i==4:raw['choices']=['one','two','','four']
            if i==5:raw['answer']='one\ntwo'
            if i==6:raw.update(choices=['one','two','three','four','five','six'],answer='six')
            record=mmar.normalize(i,raw)
        else:
            record={'dataset':group,'global_index':i,'source_row_index':i,'id':str(i),'key':f'{group}:{i}',
                    'prompt':f'fixture {i}? a) one b) two c) three d) four',
                    'choice_bodies':['one','two','three','four'],'gold_index':i%4,'gold_original':['one','two','three','four'][i%4],
                    'category':'fixture','choices_original':['one','two','three','four']}
        ids=[FixtureTokenizer().encode(c) if c else [] for c in record['choice_bodies']]
        record.update(choice_token_ids=ids,prompt_token_ids=[17]*256,prompt_attention_mask_audit_only=[0]*256,
                      prompt_original_token_count=0,prompt_truncated=False,
                      audio_ref={'kind':'file','path':'explicit-source-fixture'},audio_audit={'fixture':True})
        rows.append(record)
    return rows


def records_fixture(task):
    return [manifest.transform_record(r,FixtureTokenizer(),task) for r in source_records(task)]


def execution(task,layer):
    if task=='cutoff':return {'cutoff_layer_1based':layer,'executed_layers_1based':list(range(1,31)),
        'masked_layers_1based':[] if layer is None else list(range(layer,31)),
        'blocked_audio_plus_sep_tokens':127,'checked_forwards':1}
    return {'drop_layer_1based':layer,'executed_original_layers_1based':[i for i in range(1,31) if i!=layer],
            'effective_layers':30 if layer is None else 29,'checked_forwards':1}


def score_fixture(record,task,layer,correct=True):
    target=record['gold_index'] if record['gold_index'] is not None else 0
    valid=[i for i,v in enumerate(record['choice_valid_mask']) if v]
    prediction=target if correct else next(i for i in valid if i!=target)
    scores=[-4. if v else None for v in record['choice_valid_mask']];scores[prediction]=-1.
    return {**runtime.decision(record,scores),'execution':execution(task,layer)}


def row_fixture(record,task,layer,correct=True):
    return storage.seal({'identity_sha256':'synthetic-identity','task':task,'condition':config.condition_name(task,layer),
        'condition_layer_1based':layer,'global_index':record['global_index'],'key':record['key'],
        'record_sha256':storage.object_sha(record),**score_fixture(record,task,layer,correct)})


class Contracts(TempCase):
    def test_three_separate_8x5090_commands_outputs_and_condition_grids(self):
        outputs=[];manifests=[]
        for task in config.TASKS:
            args=arguments.parse_args(['--task',task,'--output-dir',str(self.path/(task+"'s output"))])
            cmd=submit.command(args,'fixture123')
            for flag,value in (('-p','pdgpu-5090'),('-g','8'),('-c','64'),('-m','256G'),('-n','1')):
                self.assertEqual(cmd[cmd.index(flag)+1],value)
            self.assertEqual(cmd.count('submit'),1)
            parsed=shlex.split(cmd[cmd.index('--cmd')+1])
            self.assertEqual(parsed[:4],['bash','mizar_letter_analysis/run.sh',task,'evaluate'])
            self.assertEqual(arguments.parse_args(arguments.forwarded(args)).output_dir,args.output_dir)
            outputs.append(args.output_dir);manifests.append(args.manifest)
            self.assertEqual(config.conditions(task),(None,*range(3,29)) if task=='cutoff' else (None,*range(2,30)))
            with patch.object(evaluate.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run, \
                 contextlib.redirect_stdout(io.StringIO()):evaluate.launch(args)
            self.assertEqual(run.call_count,1);self.assertIn('--nproc-per-node=8',run.call_args.args[0])
        self.assertEqual(len(set(outputs)),3);self.assertEqual(len(set(manifests)),3)
        self.assertEqual([len(config.indices(r)) for r in range(8)],[125]*8)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            arguments.parse_args(['--task','mmau','--manifest',str(self.path/'old/m.json'),'--source-manifest',str(self.path/'old/m.json')])

    def test_original_sources_and_inference_protocol_preserved(self):
        protected=storage.read_json(ROOT/'tests/fixtures/mizar_letter_analysis_original_sources.json')
        for name,digest in protected['sources_lf_sha256'].items():self.assertEqual(storage.text_sha(ROOT/name),digest,name)
        self.assertIn('mizar_letter_analysis/runtime.py',identity.sources())
        for task in config.TASKS:
            protocol=config.protocol(task)
            self.assertEqual(protocol['forward'],'CUDA_BF16_autocast')
            self.assertIn('first20',protocol['audio']);self.assertFalse(protocol['TF32']);self.assertFalse(protocol['use_cache'])

    def test_check_args_all_tasks_imports_no_torch(self):
        for task in config.TASKS:
            code=f"import sys; from mizar_letter_analysis.evaluate import main; assert main(['--task','{task}','--check-args'])==0; assert 'torch' not in sys.modules"
            run=subprocess.run([sys.executable,'-B','-c',code],cwd=config.ROOT,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)


class ManifestConversion(TempCase):
    def test_same_all1000_IDs_audio_GT_options_only_letters_and_EOT_change(self):
        for task in config.TASKS:
            original=source_records(task);records=records_fixture(task)
            self.assertEqual([r['key'] for r in records],[r['key'] for r in original])
            for r,old in zip(records,original):
                self.assertEqual(r['prompt'],old['prompt']+config.EOT_SUFFIX)
                self.assertEqual(r['source_record_sha256'],storage.object_sha(old))
                self.assertTrue(r['prompt_eot_retained'])
                self.assertIn(0,r['prompt_token_ids'])
                self.assertEqual(r['choice_token_ids'],[[tid] if v else [] for tid,v in zip(r['letter_token_ids'],r['choice_valid_mask'])])
                for key in ('audio_ref','audio_audit','gold_index','gold_original','choice_bodies'):
                    self.assertEqual(r[key],old[key])
            self.assertEqual(dict(__import__('collections').Counter(r['analysis_group'] for r in records)),config.GROUP_COUNTS[task])

    def test_MMAR_six_choices_empty_slot_and_unmapped_GT_are_retained(self):
        records=records_fixture('mmar')
        self.assertEqual(records[6]['option_letters'],list('abcdef'))
        self.assertEqual(records[6]['gold_index'],5)
        self.assertEqual(records[4]['choice_token_ids'][2],[])
        self.assertFalse(records[4]['choice_valid_mask'][2])
        self.assertIsNone(records[5]['gold_index'])
        for i in (4,5,6):
            value=score_fixture(records[i],'mmar',None)
            self.assertIsNone(value['option_loglikelihood'][2]) if i==4 else None
        self.assertFalse(score_fixture(records[5],'mmar',None)['correct'])

    def test_letter_ties_invalid_scores_and_multi_token_tokenizer_rejected(self):
        record=records_fixture('mmau')[0]
        value=runtime.decision(record,[-2.,-2.,-3.,-4.])
        self.assertEqual(value['prediction_letter'],'a');self.assertTrue(value['option_score_tie'])
        with self.assertRaises(ValueError):runtime.decision(record,[float('nan'),-2.,-3.,-4.])
        broken=copy.deepcopy(record);broken['choice_valid_mask'][2]=False
        with self.assertRaises(ValueError):runtime.decision(broken,[-2.,-2.,0.,-4.])
        class Multi(FixtureTokenizer):
            def encode(self,text,add_special_tokens=True):return [1,2]
        with self.assertRaisesRegex(ValueError,'one ordinary'):manifest.transform_record(source_records('mmau')[0],Multi(),'mmau')
        long=source_records('mmau')[0];long['prompt']='x'*300
        transformed=manifest.transform_record(long,FixtureTokenizer(),'mmau')
        self.assertTrue(transformed['prompt_truncated']);self.assertFalse(transformed['prompt_eot_retained'])
        self.assertTrue(transformed['prompt'].endswith(config.EOT_SUFFIX))

    def build_fixture(self,task):
        source_path=self.path/task/'old/manifest.json';source_path.parent.mkdir(parents=True)
        data_file=self.path/task/'audio.fixture';data_file.write_bytes(b'explicit synthetic audio bytes')
        original=source_records(task)
        for r in original:r['audio_ref']['path']=str(data_file)
        old=storage.seal({'records':original,'files':[storage.file_identity(data_file)],
            'source_arguments':{'model_dir':str(config.MODEL_DIR.resolve())}},'manifest_sha256')
        storage.write_json(source_path,old)
        args=arguments.parse_args(['--task',task,'--source-manifest',str(source_path),
            '--manifest',str(self.path/task/'new/manifest.json'),'--output-dir',str(self.path/task/'out')])
        with patch.object(manifest,'load_source',return_value=old), \
             patch.object(manifest,'load_tokenizer',return_value=(FixtureTokenizer(),[])), \
             contextlib.redirect_stdout(io.StringIO()):
            value=manifest.ensure(args)
        return args,value,source_path,data_file

    def test_actual_converter_all_tasks_atomic_reuse_and_source_bytes_unchanged(self):
        for task in config.TASKS:
            args,value,source,_=self.build_fixture(task)
            before=source.read_bytes()
            self.assertEqual(len(value['records']),1000)
            with patch.object(manifest,'build',side_effect=AssertionError('must not rebuild')),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(manifest.ensure(args),value)
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual(manifest.load(args.manifest,full=True),value)
            self.assertEqual(value['group_counts'],config.GROUP_COUNTS[task])

    def test_conversion_tampering_source_audio_and_mixed_protocol_rejected(self):
        args,value,source,audio=self.build_fixture('mmar')
        edits=[lambda x:x['records'][0].update(prompt='missing EOT'),
               lambda x:x['records'][0].update(choice_token_ids=[[1,2]]*4),
               lambda x:x.update(protocol={**x['protocol'],'scoring':'body_sum'}),
               lambda x:x['records'][0].update(analysis_group='unknown')]
        for edit in edits:
            broken=copy.deepcopy(value);broken.pop('manifest_sha256');edit(broken)
            storage.write_json(args.manifest,storage.seal(broken,'manifest_sha256'))
            with self.assertRaises(ValueError):manifest.load(args.manifest)
        storage.write_json(args.manifest,value);audio.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'file identity changed'):manifest.load(args.manifest,full=True)

    def test_source_loader_does_not_regenerate_existing_inputs_and_bootstraps_absent_inputs(self):
        args=arguments.parse_args(['--task','mmau','--source-manifest',str(self.path/'old/manifest.json')])
        fake={'records':[{}]*1000,'source_arguments':{'model_dir':str(args.model_dir)}}
        def publish(original):
            original.manifest.parent.mkdir(parents=True,exist_ok=True);original.manifest.write_text('{}')
            self.assertEqual(original.manifest,args.source_manifest)
        module=SimpleNamespace(load=lambda *a,**kw:fake,ensure=publish)
        with patch.object(manifest,'source_module',return_value=module):
            self.assertEqual(manifest.load_source(args),fake)
            module.ensure=lambda *_:(_ for _ in ()).throw(AssertionError('existing source must stay frozen'))
            self.assertEqual(manifest.load_source(args),fake)

    def test_real_local_MMAU_PSR_and_MMAR_four_groups_and_exception_slots(self):
        from mizar_layer_ablation import dataset as mmau
        from mizar_layer_ablation_mmar import dataset as mmar
        files={'mmau':Path(r'C:/Xlance/assets/Mizar/Mizar_159M-main/Mizar_159M-main/data/evaluation/mmau_mini_official.json'),
               'mmar':Path(r'C:/Xlance/assets/MMAR/MMAR-meta.json')}
        for task,path in files.items():
            if not path.is_file():self.skipTest('optional local metadata absent: '+str(path))
            normalizer=mmau.normalize if task=='mmau' else mmar.normalize
            rows=[]
            for i,raw in enumerate(storage.read_json(path)):
                r=normalizer(i,raw);r['choice_token_ids']=[[30] if c else [] for c in r['choice_bodies']]
                rows.append(manifest.transform_record(r,FixtureTokenizer(),task))
            self.assertEqual(dict(__import__('collections').Counter(r['analysis_group'] for r in rows)),config.GROUP_COUNTS[task])
            if task=='mmau':self.assertEqual(sum(r['duplicate_choice_bodies'] for r in rows),27)
            if task=='mmar':
                self.assertEqual(sum(r['gold_index'] is None for r in rows),5)
                self.assertEqual(sum(not all(r['choice_valid_mask']) for r in rows),4)
                self.assertEqual(max(len(r['option_letters']) for r in rows),6)


class ResultsAndControl(TempCase):
    def test_sum_counts_by_group_all_tasks_trace_and_score_validation(self):
        for task in config.TASKS:
            records=records_fixture(task);expected={'identity_sha256':'synthetic-identity'}
            layer=config.conditions(task)[1]
            baseline=[row_fixture(r,task,None,r['global_index']<290) for r in records]
            rows=[row_fixture(r,task,layer,r['global_index']<390) for r in records]
            merged=results.merge([[rows[i] for i in config.indices(rank)] for rank in range(8)],records,task,layer,expected)
            summary=results.summarize(merged,baseline,records,task,layer)
            self.assertEqual(summary['total'],1000);self.assertAlmostEqual(summary['delta_vs_baseline'],.1)
            self.assertEqual({g:v['total'] for g,v in summary['groups'].items()},config.GROUP_COUNTS[task])
            self.assertEqual(sum(g['correct'] for g in summary['groups'].values()),summary['correct'])
            broken=copy.deepcopy(rows[0]);broken.pop('sha256');broken['correct']=not broken['correct']
            with self.assertRaises(ValueError):results.validate(storage.seal(broken),records[0],task,layer,expected)
            broken=copy.deepcopy(rows[0]);broken.pop('sha256');broken['execution']={}
            with self.assertRaises(ValueError):results.validate(storage.seal(broken),records[0],task,layer,expected)

    def test_partial_journal_resume_wrong_identity_and_order(self):
        records=records_fixture('mmau');expected={'identity_sha256':'synthetic-identity'}
        path=self.path/'ranks/rank00/ablate_02.jsonl'
        storage.write_jsonl(path,[row_fixture(records[i],'mmau',2) for i in (0,8)])
        with path.open('ab') as stream:stream.write(b'{partial')
        self.assertEqual(len(results.read_rank(self.path,0,records,'mmau',2,expected)),2)
        with self.assertRaises(ValueError):results.read_rank(self.path,0,records,'mmau',2,{'identity_sha256':'other'})
        storage.write_jsonl(path,[row_fixture(records[i],'mmau',2) for i in (8,0)])
        with self.assertRaises(ValueError):results.read_rank(self.path,0,records,'mmau',2,expected)

    def test_actual_eight_rank_condition_sync_and_resume_for_all_three_tasks(self):
        for task in config.TASKS:
            records=records_fixture(task);expected={'identity_sha256':'synthetic-identity'}
            args=SimpleNamespace(task=task,output_dir=self.path/task);gathered={};barriers=[]
            class Dist:
                def __init__(self,rank,layer):self.rank,self.layer=rank,layer
                def broadcast_object_list(self,control,src):control[0]=(task,self.layer,'synthetic-identity')
                def gather_object(self,rows,destination,dst):
                    gathered[self.rank]=rows
                    if self.rank==0:destination[:]=[gathered[r] for r in range(8)]
                def barrier(self):barriers.append(self.layer)
            def set_condition(loaded,layer):loaded['layer']=layer
            def scored(loaded,p,r):return score_fixture(r,task,loaded['layer'])
            with patch.object(runtime,'set_condition',side_effect=set_condition),patch.object(runtime,'score',side_effect=scored) as score, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                for layer in (None,config.conditions(task)[1]):
                    for rank in [*range(1,8),0]:
                        worker.evaluate_condition(args,{},dict.fromkeys(config.indices(rank),'prefix'),records,expected,layer,rank,Dist(rank,layer))
                for rank in [*range(1,8),0]:
                    worker.evaluate_condition(args,{}, {},records,expected,config.conditions(task)[1],rank,Dist(rank,config.conditions(task)[1]))
            self.assertEqual(score.call_count,2000);self.assertEqual(len(barriers),24)
            logs=[json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(logs[0]['event'],'BASELINE_DONE');self.assertEqual(len(logs),3)
            with patch.object(worker,'evaluate_condition') as condition:worker.work(args,None,None,records,expected,0,None)
            self.assertEqual([call.args[5] for call in condition.call_args_list],list(config.conditions(task)))

    def test_coordinator_one_launch_and_full_coverage_for_each_task(self):
        for task in config.TASKS:
            args=arguments.parse_args(['--task',task,'--output-dir',str(self.path/task)])
            expected=storage.seal({'fixture':True},'identity_sha256')
            with patch.object(manifest,'ensure',return_value={'manifest_sha256':'fixture'}) as prepare, \
                 patch.object(identity,'build',return_value=expected),patch.object(identity,'verify'), \
                 patch.object(evaluate,'launch') as launch,patch.object(evaluate,'finalize',side_effect=[None,{'status':'PASS'}]), \
                 contextlib.redirect_stdout(io.StringIO()):report=evaluate.run(args)
            self.assertEqual((prepare.call_count,launch.call_count,report['status']),(1,1,'PASS'))


def runtime_artifacts(directory,records,task):
    expected={'identity_sha256':'synthetic-identity','task':task};models=[];checks=[];caches=[]
    for rank in range(8):
        model={'binding':{'rank':rank,'worker_cuda_visible_devices':str(rank)},'EXPLICIT_SYNTHETIC_CPU':True}
        check={'status':'PASS','rank':rank,'identity_sha256':'synthetic-identity','EXPLICIT_SYNTHETIC_CPU':True}
        if task=='cutoff':check.update(baseline_native_bitwise_equal=True,cutoff_isolation=[{'cutoff_layer_1based':L,'status':'PASS','audio_sep_intervention_max_abs_error':0} for L in (3,28)])
        else:check['bypass_vs_identity_output']=[{'layer_1based':L,'status':'PASS','max_abs_error':0} for L in (2,29)]
        folder=directory/'ranks'/f'rank{rank:02d}'
        storage.write_json(folder/'model_audit.json',storage.seal(model));storage.write_json(folder/'runtime_check.json',storage.seal(check))
        storage.write_json(folder/'rank_report.json',storage.seal({'status':'PASS','task':task,'identity_sha256':'synthetic-identity',
            'indices':config.indices(rank),'conditions_completed':len(config.conditions(task))}))
        if task!='cutoff' or rank==0:
            cache_folder=directory/'cache' if task=='cutoff' else directory/'cache'/f'rank{rank:02d}'
            indices=list(range(1000)) if task=='cutoff' else config.indices(rank)
            width=3 if task=='cutoff' else 4;digests=[]
            for index in indices:
                cache_folder.mkdir(parents=True,exist_ok=True);stem=f'prefix_{index:0{width}d}'
                path=cache_folder/(stem+'.pt');path.write_bytes(b'EXPLICIT_SYNTHETIC_NOT_A_TENSOR')
                meta=storage.seal({'record_sha256':storage.object_sha(records[index]),'file_sha256':storage.file_sha(path)})
                storage.write_json(cache_folder/(stem+'.json'),meta);digests.append(meta['sha256'])
            meta=storage.seal({'cache_identity':{'run_identity_sha256':'synthetic-identity'},'indices':indices,'entry_meta_sha256':digests})
            storage.write_json(cache_folder/'cache_manifest.json',meta);caches.append(meta)
        models.append(model);checks.append(check)
    storage.write_json(directory/'runtime_checks.json',storage.seal({'status':'PASS','identity_sha256':'synthetic-identity',
        'rank_audits':models,'rank_checks':checks,'cache_manifests':caches}))
    return expected


class FullThreeExperiments(TempCase):
    def test_actual85000_decision_finalizers_group_tables_figures_and_tamper(self):
        total=0
        for task in config.TASKS:
            records=records_fixture(task);directory=self.path/task
            expected=runtime_artifacts(directory,records,task)
            args=SimpleNamespace(task=task,output_dir=directory)
            def reader(directory,rank,records,task,layer,expected):
                threshold=290 if layer is None else 390 if layer in (config.conditions(task)[1],config.conditions(task)[-1]) else 350
                return [row_fixture(records[i],task,layer,i<threshold) for i in config.indices(rank)]
            frozen={'task':task,'records':records,'input_audit':{'EXPLICIT_SYNTHETIC':True}}
            with patch.object(results,'read_rank',side_effect=reader) as read:report=evaluate.finalize(args,frozen,expected)
            self.assertEqual(read.call_count,len(config.conditions(task))*8)
            self.assertEqual(report['decisions'],len(config.conditions(task))*1000)
            self.assertEqual(report['top_tied_layers_1based'],[config.conditions(task)[1],config.conditions(task)[-1]])
            self.assertEqual(len(storage.read_json(directory/'ranking.json')),len(config.conditions(task))-1)
            self.assertEqual(len((directory/'accuracy_by_group.csv').read_text().splitlines()),1+len(config.conditions(task))*len(config.GROUP_COUNTS[task]))
            for filename in report['plots']['files']:self.assertGreater((directory/filename).stat().st_size,1000)
            if os.environ.get('RSMOL_LETTER_QA'):
                from mizar_letter_analysis.plot import make_plot
                target=Path(os.environ['RSMOL_LETTER_QA'])/task;target.mkdir(parents=True,exist_ok=True)
                storage.write_json(target/'condition_scores.json',storage.read_json(directory/'condition_scores.json'));make_plot(target)
            cache_path=directory/'cache/prefix_000.pt' if task=='cutoff' else directory/'cache/rank00/prefix_0000.pt'
            cache_path.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):audit.verify(directory,expected,records)
            total+=report['decisions']
        self.assertEqual(total,85000)


ML=all(importlib.util.find_spec(n) for n in ('torch','transformers','safetensors'))


@unittest.skipUnless(ML,'requires real CPU Torch/Transformers')
class RealLlama(TempCase):
    def model(self,backend='eager'):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.set_num_threads(1);torch.manual_seed(1234)
        cfg=LlamaConfig(vocab_size=41,hidden_size=16,intermediate_size=24,num_hidden_layers=30,num_attention_heads=2,
            num_key_value_heads=2,max_position_embeddings=1024,use_cache=False,tie_word_embeddings=True,attention_dropout=0.)
        cfg._attn_implementation=backend
        return torch,LlamaForCausalLM(cfg).float().eval().requires_grad_(False)

    def attach(self,lm,task,prefix_tokens=6,blocked_tokens=3):
        if task=='cutoff':
            from mizar_audio_cutoff1000.runtime import AudioCutoff
            ctl=AudioCutoff(lm,hidden_size=16,prefix_tokens=prefix_tokens,blocked_tokens=blocked_tokens)
            return {'lm':lm,'task':task,'cutoff_controller':ctl,'device':__import__('torch').device('cpu')}
        from mizar_layer_analysis_common.runtime import LayerRouter
        return {'lm':lm,'task':task,'controller':LayerRouter(lm,hidden_size=16,prefix_tokens=prefix_tokens),'device':__import__('torch').device('cpu')}

    def test_one_forward_exact_gather_first_answer383_BF16_not_body_probability(self):
        for backend in ('eager','sdpa'):
            torch,lm=self.model(backend)
            prefix=lm.get_input_embeddings()(torch.randint(1,41,(1,383))).to(torch.bfloat16)
            loaded=self.attach(lm,'mmar',383)
            record=records_fixture('mmar')[6]
            with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
                expected=lm(inputs_embeds=prefix,attention_mask=None,labels=None,use_cache=False).logits[0,382].float().log_softmax(-1)
                before=loaded['controller'].forwards
                actual=runtime.score(loaded,prefix,record)
                self.assertEqual(loaded['controller'].forwards-before,1)
                self.assertEqual(actual['option_loglikelihood'],[float(expected[i]) for i in record['letter_token_ids']])
                altered=copy.deepcopy(record);altered['source_body_token_ids_audit_only']=[[1]*100 for _ in range(6)]
                self.assertEqual(runtime.score(loaded,prefix,altered)['option_loglikelihood'],actual['option_loglikelihood'])
                self.assertEqual(actual['prediction_letter'],record['option_letters'][max(range(6),key=lambda i:float(expected[record['letter_token_ids'][i]]))])

    def test_all_routes_BF16_native_baseline_cutoff_isolation_and_identity_equivalence(self):
        for backend in ('eager','sdpa'):
            for task in ('cutoff','mmau'):
                torch,lm=self.model(backend)
                prefix=lm.get_input_embeddings()(torch.tensor([[1,2,3,4,5,6]])).to(torch.bfloat16)
                record=records_fixture(task)[0]
                weights={k:v.clone() for k,v in lm.state_dict().items()}
                with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
                    native=lm(inputs_embeds=prefix,attention_mask=None,use_cache=False).logits
                    loaded=self.attach(lm,task)
                    self.assertTrue(torch.equal(native,lm(inputs_embeds=prefix,attention_mask=None,use_cache=False).logits))
                    self.assertEqual(runtime.checks(loaded,prefix,record)['status'],'PASS')
                    for layer in config.conditions(task):
                        runtime.set_condition(loaded,layer);value=runtime.score(loaded,prefix,record)
                        trace=value['execution'];self.assertTrue(all(s is None or __import__('math').isfinite(s) for s in value['option_loglikelihood']))
                        if task=='cutoff':self.assertEqual(trace['masked_layers_1based'],[] if layer is None else list(range(layer,31)))
                        else:self.assertEqual(trace['executed_original_layers_1based'],[i for i in range(1,31) if i!=layer])
                runtime.set_condition(loaded,None)
                for key,v in lm.state_dict().items():self.assertTrue(torch.equal(v,weights[key]))

    def test_empty_letter_ignored_even_if_its_logit_is_highest(self):
        import torch
        record=records_fixture('mmar')[4]
        class Model:
            def __call__(self,**kwargs):
                out=torch.zeros(1,383,41);out[0,-1,record['letter_token_ids'][2]]=100.
                return SimpleNamespace(logits=out)
        class Controller:
            def evidence(self):return execution('mmar',None)
        loaded={'lm':Model(),'device':torch.device('cpu'),'task':'mmar','controller':Controller()}
        value=runtime.score(loaded,torch.zeros(1,383,16,dtype=torch.bfloat16),record)
        self.assertIsNone(value['option_loglikelihood'][2]);self.assertNotEqual(value['prediction_index'],2)

    def test_real_new_EOT_prefix_caches_reused_and_corruption_rejected_both_strategies(self):
        import torch
        records=records_fixture('mmau');expected={'identity_sha256':'synthetic-identity'}
        precision={k:'fixture' for k in ('torch','transformers','cuda','gpu','weights_dtype','forward_dtype','audio_frontend_dtype','prefix_dtype','attention_implementation','sdpa_backends')}
        loaded={'audit':precision,'device':torch.device('cpu')}
        with patch.object(runtime,'prefix',return_value=torch.zeros(1,383,576,dtype=torch.bfloat16)) as encode, \
             patch.object(cache.audio,'load_fixed20',return_value=(None,{'fixture':True})),contextlib.redirect_stdout(io.StringIO()):
            values,_=cache.prepare_local(loaded,records,self.path/'local',expected,0)
            self.assertEqual(encode.call_count,125)
            cache.prepare_local(loaded,records,self.path/'local',expected,0);self.assertEqual(encode.call_count,125)
            with self.assertRaises(ValueError):cache.prepare_local(loaded,records,self.path/'local',{'identity_sha256':'body-run'},0)
        # Small explicit fixture for the rank0 cache implementation; real run is 1000.
        with patch.object(config,'ROWS',2),patch.object(runtime,'prefix',return_value=torch.zeros(1,383,576,dtype=torch.bfloat16)) as encode, \
             patch.object(cutoff_cache.audio,'load_fixed20',return_value=(None,{'fixture':True})),contextlib.redirect_stdout(io.StringIO()):
            folder=self.path/'shared';cutoff_cache.prepare(loaded,records[:2],folder,expected)
            self.assertEqual(encode.call_count,2)
            cutoff_cache.prepare(loaded,records[:2],folder,expected);self.assertEqual(encode.call_count,2)
            values,_=cutoff_cache.load_all(loaded,records[:2],folder,expected);self.assertEqual(len(values),2)
            (folder/'prefix_001.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):cutoff_cache.load_all(loaded,records[:2],folder,expected)


if __name__=='__main__':unittest.main()
