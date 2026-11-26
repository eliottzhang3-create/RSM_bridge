"""500-question continuation: preserve original screening and reuse exact caches."""
import contextlib
import copy
import io
import json
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
from mizar_reasonaqa_tracing import cache, config as original_config, identity as original_identity
from mizar_reasonaqa_tracing500 import arguments,audit,config,evaluate,identity,manifest,results,runtime,source,submit,worker
from test_mizar_reasonaqa_tracing import EXPECTED,record,outcome,scores


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP'))
        self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)

    def fixture(self,accepted_count=730):
        directory=self.path/'original';directory.mkdir()
        audio=self.path/'fixture.wav';audio.write_bytes(b'EXPLICIT_SYNTHETIC_AUDIO')
        audio_sha=storage.file_sha(audio)
        rows=[record(i,yesno=i%5==0) for i in range(5000)]
        for r in rows:r['audio_ref']['path']=str(audio)
        order=list(range(5000));random.Random(1234).shuffle(order)
        accepted_positions=set(random.Random(18).sample(range(5000),accepted_count))
        expected=storage.seal({'fixture':'EXPLICIT_SYNTHETIC_SOURCE',
            'source_arguments':{'model_dir':str(self.path/'model'),'source_manifest':str(self.path/'fixed5000.json'),
                'test_json':str(self.path/'test.json'),'audiocaps_root':str(self.path/'audio'),
                'clotho_root':str(self.path/'clotho'),'clotho_aqa_root':str(self.path/'aqa')},
            'checkpoint':str(self.path/'final.ckpt'),'pool_manifest':str(self.path/'pool.json'),
            'official_bundle':{'EXPLICIT_SYNTHETIC':True}},'identity_sha256')
        history=[];accepted=0
        for pos,index in enumerate(order):
            r=rows[index];value=outcome(r,pos in accepted_positions);value.pop('sha256')
            value['identity_sha256']=expected['identity_sha256'];value['audio_audit']['audio_source_sha256']=audio_sha
            if value['accepted']:
                cache_dir=directory/'cache'/f'source_{index:04d}';cache_dir.mkdir(parents=True)
                path=cache_dir/'tensors.pt';path.write_bytes(b'EXPLICIT_SYNTHETIC_CACHE_NOT_TENSORS')
                meta=storage.seal({'identity_sha256':expected['identity_sha256'],'record_sha256':storage.object_sha(r),
                    'source_manifest_index':index,'audio_audit':value['audio_audit'],'precision':{'EXPLICIT_SYNTHETIC':True},
                    'file_sha256':storage.file_sha(path),
                    'tensor_shapes':{'clean_prefix':[1,383,576],'corrupted_prefix':[1,383,576],'clean_states':[30,256,576]},
                    'tensor_dtypes':{'clean_prefix':'torch.bfloat16','corrupted_prefix':'torch.bfloat16','clean_states':'torch.bfloat16'}})
                storage.write_json(cache_dir/'entry.json',meta);value['cache_meta_sha256']=meta['sha256']
            value.update(shuffle_position=pos,acceptance_order=accepted if value['accepted'] else None)
            accepted+=int(value['accepted']);history.append(storage.seal(value))
        pool={'records':rows,'shuffle_order':order,'shuffle_sha256':storage.object_sha(order),'manifest_sha256':'synthetic-pool'}
        report=storage.seal({'status':'INSUFFICIENT_QUALIFYING','identity_sha256':expected['identity_sha256'],
            'pool_size':5000,'scanned':5000,'accepted':accepted_count,'required':1000,
            'reasons':dict(__import__('collections').Counter(r['reason'] for r in history)),
            'clean_correct':accepted_count,'corrupted_forwards':accepted_count,
            'journal_sha256':storage.object_sha(history),'shuffle_sha256':pool['shuffle_sha256']})
        storage.write_json(directory/'run_config.json',expected);storage.write_json(directory/'selection_report.json',report)
        storage.write_jsonl(directory/'screening/journal.jsonl',history)
        args=arguments.parse_args(['--source-output-dir',str(directory),'--output-dir',str(self.path/'new500')])
        return args,pool,expected,history


class SourceImport(TempCase):
    def test_real_import_first500_of730_original5000_order_and_source_readonly(self):
        args,pool,expected,history=self.fixture()
        before={p.relative_to(args.source_output_dir).as_posix():storage.file_sha(p) for p in args.source_output_dir.rglob('*') if p.is_file()}
        with patch.object(original_identity,'verify',return_value=pool) as verify:
            context=source.load(args.source_output_dir,full=True,verify_caches=True)
            self.assertEqual(verify.call_args.args[1],expected);self.assertTrue(verify.call_args.kwargs['full'])
        frozen=source.manifest(context,args.source_output_dir)
        first500=[r for r in history if r['accepted']][:500]
        ordered=sorted(first500,key=lambda x:pool['records'][x['source_manifest_index']]['source_row_index'])
        self.assertEqual([r['source_manifest_index'] for r in frozen['records']],[r['source_manifest_index'] for r in ordered])
        self.assertEqual(sorted(r['acceptance_order'] for r in frozen['records']),list(range(500)))
        self.assertEqual(frozen['source_total_qualifying'],730);self.assertEqual(frozen['stop_shuffle_position'],first500[-1]['shuffle_position'])
        self.assertTrue(all(r['screening']['clean']['correct'] and not r['screening']['corrupted']['correct'] for r in frozen['records']))
        after={p.relative_to(args.source_output_dir).as_posix():storage.file_sha(p) for p in args.source_output_dir.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        # Exercise CPU publication and reuse with no Torch/capture/model loading.
        with patch.object(source,'load',return_value=context),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(manifest.main(arguments.forwarded(args)),0)
            self.assertEqual(manifest.main(arguments.forwarded(args)),0)
        saved=storage.read_json(args.output_dir/'selected/manifest.json');self.assertEqual(saved,frozen)
        self.assertFalse((args.output_dir/'cache').exists(),'must reference original tensors without copying')
        new_expected=storage.read_json(args.output_dir/'run_config.json')
        with patch.object(source,'load',return_value=context):
            self.assertEqual(identity.verify(args,new_expected)[1],frozen)
        bad=copy.deepcopy(context);bad['snapshots']['selection_report.json']='changed'
        with patch.object(source,'load',return_value=bad):
            with self.assertRaisesRegex(ValueError,'changed'):identity.verify(args,new_expected)
        entry=source.verify_cache(args.source_output_dir,pool['records'][first500[0]['source_manifest_index']],first500[0],expected)
        path=args.source_output_dir/'cache'/f'source_{first500[0]["source_manifest_index"]:04d}'/'tensors.pt'
        path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'content changed'):
            source.verify_cache(args.source_output_dir,pool['records'][first500[0]['source_manifest_index']],first500[0],expected)

    def test_insufficient500_and_corrupt_tail_refused_without_mutation(self):
        args,pool,expected,history=self.fixture(499)
        with patch.object(original_identity,'verify',return_value=pool):
            with self.assertRaisesRegex(ValueError,'only 499'):source.load(args.source_output_dir)
        path=args.source_output_dir/'screening/journal.jsonl'
        with path.open('ab') as stream:stream.write(b'{partial')
        before=path.read_bytes()
        with self.assertRaisesRegex(ValueError,'incomplete tail'):source.read_journal(path)
        self.assertEqual(path.read_bytes(),before)

    def test_source_guards_and_inference_semantics_unchanged(self):
        guards=storage.read_json(config.PACKAGE/'reference_guard.json')['sources_lf_sha256']
        for name,digest in guards.items():self.assertEqual(storage.text_sha(config.ROOT/name),digest,name)
        self.assertGreater(len(identity.sources()),70)
        for key,value in original_config.PROTOCOL.items():
            if key not in ('schema','target_rows','stop','selected_order'):
                self.assertEqual(config.PROTOCOL[key],value,key)
        self.assertEqual(config.TARGET,500);self.assertEqual(original_config.TARGET,1000)


def frozen_fixture():
    rows=[]
    for i in range(500):
        r=record(i,yesno=i%5==0);o=outcome(r)
        r.update(selected_index=i,acceptance_order=i,shuffle_position=i,
            screening={k:o[k] for k in ('clean','corrupted','probability_gap','cache_meta_sha256','audio_audit')})
        rows.append(r)
    return storage.seal({'records':rows,'source_total_qualifying':730,
        'category_counts':{'audiocaps_mcq':400,'clotho_aqa_yesno':100}},'manifest_sha256')


class Scheduling(TempCase):
    def test_exact500_sharding_all30_conditions_resume_and_no_screening(self):
        frozen=frozen_fixture();args=SimpleNamespace(output_dir=self.path)
        self.assertEqual([len(config.indices(r)) for r in range(8)],[63,63,63,63,62,62,62,62])
        self.assertEqual(sorted(i for r in range(8) for i in config.indices(r)),list(range(500)))
        gathered={};barriers=[];active={'layer':1}
        class Dist:
            def __init__(self,rank):self.rank=rank
            def broadcast_object_list(self,control,src):control[0]=(active['layer'],frozen['manifest_sha256'])
            def gather_object(self,rows,out,dst):
                gathered[self.rank]=rows
                if self.rank==0:out[:]=[gathered[r] for r in range(8)]
            def barrier(self):barriers.append(active['layer'])
        tensors={i:{'corrupted_prefix':i,'clean_states':'explicit-fixture'} for i in range(500)}
        def logits(loaded,p,**kw):return p,None
        def decision(r,values):return r['screening']['clean'] if active['layer']==30 else scores(r,True,.35)
        def stats(rows):return {'total':len(rows),'mean_RR':1. if active['layer']==30 else .5,'median_RR':.5,
            'patched_accuracy':1.,'mean_Ppatched':.35}
        with patch.object(runtime,'forward',side_effect=logits) as forward,patch.object(runtime,'from_logits',side_effect=decision), \
             patch.object(results,'stats',side_effect=stats),contextlib.redirect_stdout(io.StringIO()):
            for layer in range(1,31):
                active['layer']=layer
                for rank in [*range(1,8),0]:worker.evaluate_layer(args,{},tensors,frozen,EXPECTED,layer,rank,Dist(rank))
            self.assertEqual(forward.call_count,15000)
            for rank in [*range(1,8),0]:worker.evaluate_layer(args,{},tensors,frozen,EXPECTED,30,rank,Dist(rank))
            self.assertEqual(forward.call_count,15000)
        self.assertEqual(len(barriers),248)
        self.assertEqual(len(storage.read_json(self.path/'layers/patch_30/summary.json')['groups']),3)

    def test_actual_finalizer15000_tables_figures_and_failed_worker_details(self):
        args=SimpleNamespace(output_dir=self.path,source_output_dir=self.path/'original')
        frozen=frozen_fixture()
        for rank in range(8):storage.write_json(self.path/'ranks'/f'rank{rank:02d}'/'rank_report.json',{'EXPLICIT_SYNTHETIC':True})
        def reader(directory,rank,frozen,expected,layer):
            return [storage.seal(results.make_row(frozen['records'][i],EXPECTED,frozen,layer,
                frozen['records'][i]['screening']['clean'] if layer==30 else scores(frozen['records'][i],True,.1+.5*layer/30)))
                for i in config.indices(rank)]
        with patch.object(results,'read_rank',side_effect=reader),patch.object(audit,'verify',return_value={'sha256':'synthetic-audit'}):
            report=evaluate.finalize(args,frozen,EXPECTED,{})
        self.assertEqual(report['status'],'PASS');self.assertEqual(report['patched_decisions'],15000)
        self.assertEqual(report['new_screening_forwards'],0);self.assertEqual(len(report['plots']),8)
        self.assertEqual(len((self.path/'layer_scores.csv').read_text().splitlines()),121)
        self.assertTrue(all(Path(p).stat().st_size>1000 for p in report['plots']))
        if os.environ.get('RSMOL_TRACING500_QA'):
            import shutil
            qa=Path(os.environ['RSMOL_TRACING500_QA']);qa.mkdir(parents=True,exist_ok=True)
            for p in report['plots']:shutil.copy2(p,qa/Path(p).name)
        fail={'status':'FAILED','rank':2,'phase':'runtime_controls','error':"ValueError('explicit fixture')",'traceback':'EXPLICIT_FIXTURE'}
        storage.write_json(self.path/'ranks/rank02/failure.json',fail)
        with patch.object(evaluate.subprocess,'run',return_value=SimpleNamespace(returncode=1)),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'rank2 runtime_controls'):evaluate.launch(args)

    def test_one_job_resources_argument_roundtrip_and_disjoint_outputs(self):
        a=arguments.parse_args(['--output-dir',str(self.path/"a's result")])
        plan=submit.command(a,'fixture')
        for flag,val in (('-p','pdgpu-5090'),('-g','8'),('-c','64'),('-m','256G'),('-n','1')):
            self.assertEqual(plan[plan.index(flag)+1],val)
        self.assertIn(str(a.output_dir),shlex.split(plan[-1]));self.assertEqual(arguments.parse_args(arguments.forwarded(a)),a)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):arguments.parse_args(['--source-output-dir',str(self.path),'--output-dir',str(self.path/'child')])

    def test_actual_audit_references_source500_caches_and_rejects_wrong63_row_completion(self):
        args=SimpleNamespace(output_dir=self.path,source_output_dir=self.path/'source')
        frozen=frozen_fixture();context={'expected':{'identity_sha256':'original-source'},
            'pool':{'records':[record(i,yesno=i%5==0) for i in range(500)]}}
        models=[];checks=[]
        for rank in range(8):
            model={k:'fixture' for k in ('torch','transformers','cuda','gpu','weights_dtype','forward_dtype',
                'audio_frontend_dtype','prefix_dtype','attention_implementation','sdpa_backends')}
            model['binding']={'rank':rank,'worker_cuda_visible_devices':str(rank)}
            check={'status':'PASS','rank':rank,'identity_sha256':EXPECTED['identity_sha256'],'silence_sha256':'fixture',
                'capture_vs_native_logits_bitwise_equal':True,'layer0_equals_corrupted_logits_bitwise':True,
                'layer30_equals_clean_logits_bitwise':True,'cache_equals_repeated_clean_states_bitwise':True,
                'state_shape':[30,256,576],'text_start':127}
            folder=self.path/'ranks'/f'rank{rank:02d}'
            for name,value in [('model_audit.json',model),('runtime_check.json',check)]:storage.write_json(folder/name,storage.seal(value))
            storage.write_json(folder/'rank_report.json',storage.seal({'status':'PASS','identity_sha256':EXPECTED['identity_sha256'],
                'manifest_sha256':frozen['manifest_sha256'],'indices':config.indices(rank),'layers':list(range(1,31))}))
            storage.write_json(folder/'cache_inventory.json',storage.seal({'identity_sha256':EXPECTED['identity_sha256'],
                'entries':[{'selected_index':i,'source_manifest_index':i,'meta_sha256':'explicit-fixture-digest'} for i in config.indices(rank)]}))
            models.append(model);checks.append(check)
        storage.write_json(self.path/'runtime_checks.json',storage.seal({'status':'PASS','identity_sha256':EXPECTED['identity_sha256'],
            'rank_audits':models,'rank_checks':checks}))
        with patch.object(source,'manifest',return_value=frozen),patch.object(storage,'file_sha',return_value='explicit-synthetic-audio'), \
             patch.object(cache,'load',return_value=(None,{'sha256':'explicit-fixture-digest'})) as reader:
            self.assertEqual(audit.verify(args,frozen,EXPECTED,context)['status'],'PASS');self.assertEqual(reader.call_count,500)
            self.assertEqual(reader.call_args.args[0],args.source_output_dir)
            bad=storage.read_json(self.path/'ranks/rank00/rank_report.json');bad.pop('sha256');bad['indices']=config.indices(0)[:-1]
            storage.write_json(self.path/'ranks/rank00/rank_report.json',storage.seal(bad))
            with self.assertRaisesRegex(ValueError,'exact62/63'):audit.verify(args,frozen,EXPECTED,context)

    def test_old_model_capture_functions_and_real_BF16_cache_are_reused_exactly(self):
        import torch
        from mizar_reasonaqa_tracing import runtime as old
        self.assertIs(runtime.forward,old.forward);self.assertIs(runtime.controls,old.controls);self.assertIs(runtime.load,old.load)
        r=record();cp=torch.randn(1,383,576,dtype=torch.bfloat16);sp=cp.clone();sp[:,:126]=0
        states=torch.randn(30,256,576,dtype=torch.bfloat16)
        meta=cache.save(self.path,r,EXPECTED,cp,sp,states,{'audio_source_sha256':'fixture'},{'precision':'fixture'})
        before=storage.file_sha(cache.folder(self.path,0)/'tensors.pt')
        data,actual=cache.load(self.path,r,EXPECTED,digest=meta['sha256'],precision={'precision':'fixture'})
        self.assertTrue(torch.equal(data['clean_states'],states));self.assertEqual(actual,meta)
        self.assertEqual(storage.file_sha(cache.folder(self.path,0)/'tensors.pt'),before)


if __name__=='__main__':unittest.main()
