"""Two isolated tasks: real CKA math/tiny Llama plus explicit pipeline fixtures."""
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
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/RSmol')); sys.path.insert(0,str(ROOT/'tests/fixtures'))
from mizar_layer_pair_search import storage,data as old_data
from mizar_layer_analysis_common import arguments,audit,cache,config,evaluate,identity,runtime,submit
from mizar_layer_cka import capture,metrics,task as cka
from mizar_layer_ablation import dataset,results,task as ablation
from cka_official_reference import cka as reference_cka


def records_fixture(rows=1000):
    result=[]
    for i in range(rows):
        result.append({'global_index':i,'key':f'mmau:{i}','id':str(i),'prompt':f'explicit fixture question {i}',
            'choice_bodies':['short','long answer'],'choice_token_ids':[[3],[4,5]],'gold_index':i%2,
            'task':'music' if i<334 else 'sound' if i<667 else 'speech','difficulty':('easy','medium','hard')[i%3],
            'category':'fixture','sub_category':'fixture','prompt_token_ids':[17]*256,
            'prompt_attention_mask_audit_only':[1]*10+[0]*246,
            'audio_ref':{'kind':'file','path':'explicit-fixture-not-production'},'audio_audit':{'fixture':True}})
    return result


def scores_fixture(record,layer,correct=True):
    pred=record['gold_index'] if correct else 1-record['gold_index']
    scores=[-3.,-4.]; scores[pred]=-1.
    return {'prediction_index':pred,'correct':correct,'option_loglikelihood':scores,'option_token_counts':[1,2],
            'option_mean_loglikelihood':[scores[0],scores[1]/2],'option_score_tie':False,
            'execution':{'drop_layer_1based':layer,'executed_original_layers_1based':[i for i in range(1,31) if i!=layer],
                         'effective_layers':30 if layer is None else 29,'checked_forwards':1}}


def row_fixture(record,layer,correct=True):
    return storage.seal({'identity_sha256':'synthetic-identity','condition':results.name(layer),
        'global_index':record['global_index'],'key':record['key'],'record_sha256':storage.object_sha(record),
        **scores_fixture(record,layer,correct)})


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP')); self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)


class CKAMath(unittest.TestCase):
    def test_standard_and_debiased_match_official_reference(self):
        rng=np.random.default_rng(42); features=rng.normal(size=(30,100,40))
        for debiased in (False,True):
            value=metrics.matrix(features,debiased)
            for i,j in ((0,0),(0,1),(3,17),(12,29)):
                self.assertAlmostEqual(value[i,j],reference_cka(features[i],features[j],debiased),places=12)
            np.testing.assert_allclose(value,value.T,atol=1e-12)
            np.testing.assert_allclose(np.diag(value),1,atol=1e-12)

    def test_translation_rotation_and_scaling_invariance(self):
        rng=np.random.default_rng(12); x=rng.normal(size=(100,16)); q,_=np.linalg.qr(rng.normal(size=(16,16)))
        features=np.stack([x if i%2==0 else 3*x@q+7 for i in range(30)])
        for debiased in (False,True): np.testing.assert_allclose(metrics.matrix(features,debiased),1,atol=1e-12)

    def test_small_sample_high_dimensional_bias_and_negative_values_not_clipped(self):
        rng=np.random.default_rng(123); features=rng.normal(size=(30,100,576))
        standard,debiased=metrics.matrix(features),metrics.matrix(features,True)
        off=~np.eye(30,dtype=bool)
        self.assertGreater(float(standard[off].mean()),.7)
        self.assertLess(abs(float(debiased[off].mean())),.03)
        self.assertLess(float(debiased.min()),0)

    def test_degenerate_inputs_and_wrong_shapes_rejected(self):
        for debiased in (False,True):
            with self.assertRaises(ValueError): metrics.matrix(np.ones((30,100,16)),debiased)
        with self.assertRaises(ValueError): metrics.matrix(np.ones((29,100,16)))
        with self.assertRaises(ValueError): metrics.center_gram(np.eye(3),True)

    def test_permutation_null_is_deterministic_and_breaks_correspondence(self):
        x=np.random.default_rng(1).normal(size=(100,12)); features=np.stack([x]*30)
        for debiased in (False,True):
            mean,std=metrics.permutation_null(features,debiased,repeats=5)
            again,_=metrics.permutation_null(features,debiased,repeats=5)
            np.testing.assert_array_equal(mean,again)
            self.assertTrue(np.all(std>=0)); self.assertLess(float(mean.mean()),.25)


class Contracts(TempCase):
    def test_independent_commands_resources_outputs_and_long_paths(self):
        for task in ('cka','ablation'):
            args=arguments.parse_args(task,['--output-dir',str(self.path/(task+"'s output"))])
            cmd=submit.command(args,task,'fixture123')
            for flag,value in (('-p','pdgpu-3090'),('-g','8'),('-c','64'),('-m','256G'),('-n','1')):
                self.assertEqual(cmd[cmd.index(flag)+1],value)
            self.assertEqual(cmd.count('submit'),1); self.assertLess(len(cmd[cmd.index('-j')+1]),60)
            compute=shlex.split(cmd[cmd.index('--cmd')+1])
            self.assertEqual(compute[:4],['bash','mizar_layer_analysis_common/run.sh',task,'evaluate'])
            self.assertIn(str(args.output_dir),compute)
            with patch.object(evaluate.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as mocked,contextlib.redirect_stdout(io.StringIO()):
                evaluate.launch(args,task)
            self.assertIn('--nproc-per-node=8',mocked.call_args.args[0]); self.assertEqual(mocked.call_count,1)
        self.assertNotEqual(arguments.parse_args('cka',[]).output_dir,arguments.parse_args('ablation',[]).output_dir)

    def test_original_sources_unchanged_and_task_identity_is_separate(self):
        frozen=storage.read_json(ROOT/'tests/fixtures/mizar_layer_analysis_original_sources.json')
        for name,digest in frozen['sources_lf_sha256'].items(): self.assertEqual(storage.text_sha(ROOT/name),digest,name)
        first,second=identity.sources('cka'),identity.sources('ablation')
        self.assertNotIn('mizar_layer_ablation/task.py',first); self.assertNotIn('mizar_layer_cka/task.py',second)

    def test_exact28_layers_and_fixed_eight_rank_shards(self):
        self.assertEqual(results.CONDITIONS,(None,*range(2,30)))
        self.assertEqual([len(config.indices(r,1000)) for r in range(8)],[125]*8)
        self.assertEqual([len(config.indices(r,100)) for r in range(8)],[13]*4+[12]*4)
        for bad in (1,30,True):
            with self.assertRaises(ValueError): results.name(bad)

    def test_cli_preview_does_not_import_torch(self):
        for task in ('cka','ablation'):
            code=f"import sys; from mizar_layer_analysis_common.evaluate import main; assert main('{task}',['--check-args'])==0; assert 'torch' not in sys.modules"
            run=subprocess.run([sys.executable,'-B','-c',code],cwd=config.ROOT,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)


class DatasetContracts(TempCase):
    def test_duplicate_slots_preserved_and_original_label_maps_GT(self):
        row={'id':'one','question':'Which sound?','choices':['(A) one','(B) two','(C) two'],'answer':'(C) two','task':'speech'}
        record=dataset.normalize(0,row)
        self.assertEqual(record['choice_bodies'],['one','two','two']); self.assertEqual(record['gold_index'],2)
        self.assertEqual(record['gold_body_matching_indices'],[1,2]); self.assertTrue(record['duplicate_choice_bodies'])
        self.assertEqual(record['prompt'],'which sound? a) one b) two c) two')
        self.assertEqual(len(record['valid_original_slots']),3)

    def test_full1000_real_metadata_has_no_filtered_rows(self):
        path=Path(r'C:\Xlance\assets\Mizar\Mizar_159M-main\Mizar_159M-main\data\evaluation\mmau_mini_official.json')
        if not path.exists(): self.skipTest('optional local Mizar metadata absent')
        rows=storage.read_json(path); records=[dataset.normalize(i,row) for i,row in enumerate(rows)]
        self.assertEqual(len(records),1000); self.assertEqual(sum(r['duplicate_choice_bodies'] for r in records),27)
        self.assertEqual(sum(len(r['gold_body_matching_indices'])>1 for r in records),16)

    def test_invalid_GT_and_empty_options_fail_instead_of_skipping(self):
        row={'id':'x','question':'Question','choices':['a','b'],'answer':'c'}
        with self.assertRaises(ValueError): dataset.normalize(0,row)
        row.update(answer='a',choices=['a',''])
        with self.assertRaises(ValueError): dataset.normalize(0,row)

    def test_summaries_count_all1000_and_only_sum_scores(self):
        records=records_fixture(); expected={'identity_sha256':'synthetic-identity'}
        base=[row_fixture(r,None,r['global_index']<290) for r in records]
        rows=[row_fixture(r,2,r['global_index']<390) for r in records]
        merged=results.merge([[rows[i] for i in config.indices(r,1000)] for r in range(8)],records,2,expected)
        summary=results.summarize(merged,base,records,2)
        self.assertEqual(summary['accuracy'],.39); self.assertEqual(summary['delta_vs_baseline'],.1)
        self.assertEqual(summary['newly_correct'],100); self.assertEqual(summary['total'],1000)
        broken=copy.deepcopy(rows[0]); broken.pop('sha256'); broken['execution']['executed_original_layers_1based']=list(range(1,31))
        with self.assertRaises(ValueError): results.validate(storage.seal(broken),records[0],2,expected)

    def test_partial_journal_recovery_wrong_identity_and_wrong_order(self):
        records=records_fixture(); expected={'identity_sha256':'synthetic-identity'}
        path=self.path/'ranks/rank00/ablate_02.jsonl'
        storage.write_jsonl(path,[row_fixture(records[i],2) for i in (0,8)])
        with path.open('ab') as stream: stream.write(b'{partial')
        self.assertEqual(len(results.read_rank(self.path,0,records,2,expected)),2)
        with self.assertRaises(ValueError): results.read_rank(self.path,0,records,2,{'identity_sha256':'other'})
        storage.write_jsonl(path,[row_fixture(records[8],2),row_fixture(records[0],2)])
        with self.assertRaises(ValueError): results.read_rank(self.path,0,records,2,expected)

    def test_actual_synchronized_worker_condition_and_resume(self):
        records=records_fixture(); args=SimpleNamespace(output_dir=self.path); expected={'identity_sha256':'synthetic-identity'}
        gathered={}
        class Controller:
            def set_drop(self,layer): self.drop=layer
            def evidence(self): return scores_fixture(records[0],self.drop)['execution']
        class Dist:
            def __init__(self,rank,layer): self.rank,self.layer=rank,layer
            def broadcast_object_list(self,control,src): control[0]=(self.layer,'synthetic-identity')
            def gather_object(self,rows,destination,dst):
                gathered[self.rank]=rows
                if self.rank==0: destination[:]=[gathered[r] for r in range(8)]
            def barrier(self): pass
        with patch.object(runtime,'score',side_effect=lambda loaded,p,r:{k:v for k,v in scores_fixture(r,loaded['controller'].drop).items() if k!='execution'}) as score,contextlib.redirect_stdout(io.StringIO()) as output:
            for layer in (None,2,29):
                for rank in [1,2,3,4,5,6,7,0]:
                    ablation.evaluate_condition(args,{'controller':Controller()},{i:'prefix' for i in config.indices(rank,1000)},records,expected,layer,rank,Dist(rank,layer))
            for rank in [1,2,3,4,5,6,7,0]: ablation.evaluate_condition(args,{'controller':Controller()},{},records,expected,2,rank,Dist(rank,2))
        self.assertEqual(score.call_count,3000)
        self.assertEqual([json.loads(line)['event'] for line in output.getvalue().splitlines()],['BASELINE_DONE','ABLATION_DONE','ABLATION_DONE','ABLATION_DONE'])


def runtime_fixture(directory,records,task):
    expected={'identity_sha256':'synthetic-identity'}; audits=[]; checks=[]; caches=[]
    for rank in range(8):
        model={'binding':{'rank':rank,'worker_cuda_visible_devices':str(rank)},'EXPLICIT_SYNTHETIC_CPU':True}
        check={'status':'PASS','rank':rank,'identity_sha256':'synthetic-identity','EXPLICIT_SYNTHETIC_CPU':True}
        if task=='cka': check['capture_native_logits_bitwise_equal']=True
        else: check['bypass_vs_identity_output']=[{'status':'PASS','layer_1based':i,'max_abs_error':0} for i in (2,29)]
        folder=directory/'ranks'/f'rank{rank:02d}'
        storage.write_json(folder/'model_audit.json',storage.seal(model)); storage.write_json(folder/'runtime_check.json',storage.seal(check))
        storage.write_json(folder/'rank_report.json',storage.seal({'status':'PASS','identity_sha256':'synthetic-identity','indices':config.indices(rank,len(records))}))
        cache_folder=directory/'cache'/f'rank{rank:02d}'; hashes=[]
        for index in config.indices(rank,len(records)):
            cache_folder.mkdir(parents=True,exist_ok=True)
            path=cache_folder/f'prefix_{index:04d}.pt'; path.write_bytes(b'EXPLICIT_SYNTHETIC_NOT_A_TENSOR')
            meta=storage.seal({'record_sha256':storage.object_sha(records[index]),'file_sha256':storage.file_sha(path)})
            storage.write_json(cache_folder/f'prefix_{index:04d}.json',meta); hashes.append(meta['sha256'])
        cache_meta=storage.seal({'cache_identity':{'run_identity_sha256':'synthetic-identity'},'indices':config.indices(rank,len(records)),'entry_meta_sha256':hashes})
        storage.write_json(cache_folder/'cache_manifest.json',cache_meta)
        audits.append(model); checks.append(check); caches.append(cache_meta)
    storage.write_json(directory/'runtime_checks.json',storage.seal({'status':'PASS','identity_sha256':'synthetic-identity',
        'rank_audits':audits,'rank_checks':checks,'rank_caches':caches}))
    return expected


class FullAblation(TempCase):
    def test_full29x1000_actual_finalizer_outputs_and_cache_tampering(self):
        records=records_fixture(); expected=runtime_fixture(self.path,records,'ablation')
        args=SimpleNamespace(output_dir=self.path)
        def reader(directory,rank,records,layer,expected):
            threshold=290 if layer is None else 390 if layer in (2,29) else 350
            return [row_fixture(records[i],layer,i<threshold) for i in config.indices(rank,1000)]
        with patch.object(results,'read_rank',side_effect=reader) as read:
            report=ablation.finalize(args,{'records':records},expected)
        self.assertEqual(read.call_count,29*8); self.assertEqual(report['decisions'],29000)
        self.assertEqual(report['top_tied_layers_1based'],[2,29]); self.assertEqual(report['baseline_accuracy'],.29)
        self.assertEqual(len(storage.read_json(self.path/'ranking.json')),28)
        self.assertEqual(len((self.path/'accuracy_by_layer.csv').read_text().splitlines()),30)
        (self.path/'cache/rank00/prefix_0000.pt').write_bytes(b'corrupted')
        with self.assertRaises(ValueError): audit.verify(self.path,expected,records)


ML=all(importlib.util.find_spec(name) for name in ('torch','transformers','safetensors'))


@unittest.skipUnless(ML,'requires real CPU Torch/Transformers for tensor checks')
class RealLlama(TempCase):
    def make_model(self,backend='eager'):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.set_num_threads(1); torch.manual_seed(1234)
        cfg=LlamaConfig(vocab_size=41,hidden_size=16,intermediate_size=24,num_hidden_layers=30,num_attention_heads=2,
                        num_key_value_heads=2,max_position_embeddings=1024,attention_dropout=0,use_cache=False,tie_word_embeddings=True)
        cfg._attn_implementation=backend
        lm=LlamaForCausalLM(cfg).float().eval().requires_grad_(False)
        ctl=runtime.LayerRouter(lm,hidden_size=16,prefix_tokens=6)
        return torch,{'lm':lm,'device':torch.device('cpu'),'controller':ctl}

    def test_every_single_layer_skip_trace_and_identity_equivalence_BF16(self):
        for backend in ('eager','sdpa'):
            torch,loaded=self.make_model(backend); lm=loaded['lm']; ctl=loaded['controller']
            prefix=lm.get_input_embeddings()(torch.tensor([[1,2,3,4,5,6]])).to(torch.bfloat16)
            weights={k:v.clone() for k,v in lm.state_dict().items()}
            with torch.autocast('cpu',dtype=torch.bfloat16):
                check=runtime.check_skip_equivalence(loaded,prefix,{'choice_token_ids':[[9,10]]})
                self.assertEqual(check['status'],'PASS')
                for layer in results.CONDITIONS:
                    ctl.set_drop(layer); runtime.option_score(lm,prefix,[9,10],loaded['device'])
                    self.assertEqual(ctl.trace,[i for i in range(1,31) if i!=layer])
            ctl.set_drop(None)
            for k,v in lm.state_dict().items(): self.assertTrue(torch.equal(v,weights[k]))
            for invalid in (1,30):
                with self.assertRaises(ValueError): ctl.set_drop(invalid)

    def test_raw_block_capture_pools_valid_tokens_and_preserves_baseline(self):
        torch,loaded=self.make_model(); lm=loaded['lm']
        prefix=lm.get_input_embeddings()(torch.tensor([[1,2,3,4,5,17]]))
        raw=[]
        def hook(_m,_a,out): raw.append((out[0] if isinstance(out,tuple) else out).detach().clone())
        handles=[layer.register_forward_hook(hook) for layer in loaded['controller'].original]
        with torch.inference_mode(): native=lm(inputs_embeds=prefix,attention_mask=None,use_cache=False,output_hidden_states=True)
        for h in handles: h.remove()
        features,last=capture.extract_layout(loaded,prefix,[1,1,0],audio_tokens=2,prompt_start=3)
        self.assertTrue(torch.equal(last,native.logits[:,-1].float()))
        for i in range(30):
            torch.testing.assert_close(features[0,i],raw[i][0,:2].float().mean(0),rtol=0,atol=0)
            torch.testing.assert_close(features[1,i],raw[i][0,3:5].float().mean(0),rtol=0,atol=0)
            torch.testing.assert_close(features[2,i],raw[i][0,-1].float(),rtol=0,atol=0)
        self.assertFalse(torch.allclose(features[2,-1],native.hidden_states[-1][0,-1]))

    def test_FP32_prefix_cache_computes_each_assigned_encoder_once_and_rejects_changes(self):
        import torch
        records=records_fixture(100); expected={'identity_sha256':'synthetic-identity'}
        audit_fixture={key:'fixture' for key in ('torch','transformers','cuda','gpu','weights_dtype','forward_dtype','audio_frontend_dtype','prefix_dtype','attention_implementation','sdpa_backends')}
        loaded={'audit':audit_fixture,'device':torch.device('cpu')}
        with patch.object(runtime,'prefix',return_value=torch.zeros((1,383,576),dtype=torch.bfloat16)) as encode, \
             patch.object(cache.audio,'load_fixed20',return_value=(torch.zeros(1),{'fixture':True})),contextlib.redirect_stdout(io.StringIO()):
            values,_=cache.prepare_local(loaded,records,self.path,expected,0)
            self.assertEqual(encode.call_count,13); self.assertEqual(set(values),set(config.indices(0,100)))
            cache.prepare_local(loaded,records,self.path,expected,0); self.assertEqual(encode.call_count,13)
            with self.assertRaises(ValueError): cache.prepare_local(loaded,records,self.path,{'identity_sha256':'other'},0)

    def test_CKA_all100_real_tensor_artifacts_actual_global_finalizer(self):
        import torch
        records=records_fixture(100); expected=runtime_fixture(self.path,records,'cka')
        folder=self.path/'features'; folder.mkdir()
        rng=np.random.default_rng(1234)
        for record in records:
            index=record['global_index']; value=torch.from_numpy(rng.normal(size=(3,30,576)).astype('float32'))
            path=folder/f'features_{index:03d}.pt'; cache.save_tensor(path,value)
            from mizar_layer_pair_search.audio import tensor_sha
            meta=storage.seal({'index':index,'identity_sha256':'synthetic-identity','record_sha256':storage.object_sha(record),
                'file_sha256':storage.file_sha(path),'tensor_sha256':tensor_sha(value),'execution':{
                    'executed_original_layers_1based':list(range(1,31)),'drop_layer_1based':None}})
            storage.write_json(folder/f'features_{index:03d}.json',meta)
        args=SimpleNamespace(output_dir=self.path,permutation_repeats=2)
        with contextlib.redirect_stdout(io.StringIO()): report=cka.finalize(args,{'records':records},expected)
        self.assertEqual(report['matrices'],6); self.assertEqual(report['rows'],100)
        pooled=np.load(self.path/'pooled_features.npz'); self.assertEqual(pooled['audio_mean'].shape,(30,100,576))
        matrices=np.load(self.path/'cka_matrices.npz'); self.assertEqual(len(matrices.files),6)
        np.testing.assert_allclose(np.diag(matrices['audio_mean_debiased']),1,atol=1e-12)
        self.assertEqual(storage.verify_seal(storage.read_json(self.path/'artifacts.json'))['identity_sha256'],'synthetic-identity')


if __name__=='__main__': unittest.main()
