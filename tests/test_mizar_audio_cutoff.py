"""CPU contracts/control flow and real small Llama numerical regressions."""
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
from mizar_audio_cutoff import arguments, config, evaluate, frozen_manifest, identity, results, runtime, submit, worker


def records_fixture():
    return [{'global_index':i, 'key':f'{config.DATASETS[i//25]}:{i}', 'dataset':config.DATASETS[i//25],
             'prompt':f'explicit synthetic question {i}', 'choice_bodies':['short','long choice'],
             'choice_token_ids':[[3],[4,5]], 'gold_index':i % 2} for i in range(100)]


def score_fixture(record, cutoff, correct=True):
    prediction = record['gold_index'] if correct else 1-record['gold_index']
    sums = [-3., -4.]
    sums[prediction] = -1.
    return {'option_loglikelihood':sums, 'prediction_index':prediction, 'correct':correct,
        'option_token_counts':[1,2], 'option_mean_loglikelihood':[sums[0],sums[1]/2], 'option_score_tie':False,
        'execution':{'executed_layers_1based':list(range(1,31)), 'masked_layers_1based':config.masked_layers(cutoff),
                     'cutoff_layer_1based':cutoff, 'blocked_audio_plus_sep_tokens':127, 'checked_forwards':1}}


def row_fixture(record, cutoff, correct=True):
    return storage.seal({'identity_sha256':'synthetic-identity', 'condition':config.condition_name(cutoff),
        'cutoff_layer_1based':cutoff, 'global_index':record['global_index'], 'key':record['key'],
        'record_sha256':storage.object_sha(record), **score_fixture(record,cutoff,correct)})


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP'))
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)


class Contracts(TempCase):
    def test_exact19_conditions_layer_boundaries_and_rank_coverage(self):
        self.assertEqual(config.CONDITIONS, (None,*range(3,21)))
        self.assertEqual(config.masked_layers(3),list(range(3,31)))
        self.assertEqual(config.masked_layers(20),list(range(20,31)))
        for world in (1,8):
            self.assertEqual(sorted(i for r in range(world) for i in config.indices(r,world)),list(range(100)))
        self.assertEqual([len(config.indices(r,8)) for r in range(8)],[13]*4+[12]*4)
        for bad in (0,1,2,21,30,True):
            with self.assertRaises(ValueError): config.condition_name(bad)

    def test_manifest_is_shared_outputs_are_distinct_and_no_generator(self):
        one, eight = arguments.parse_args([]), arguments.parse_args(['--world-size','8'])
        self.assertEqual(one.manifest,eight.manifest)
        self.assertNotEqual(one.output_dir,eight.output_dir)
        self.assertNotIn('manifest.ensure',Path(evaluate.__file__).read_text())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments.parse_args(['--manifest',str(self.path/'out/m.json'),'--output-dir',str(self.path/'out')])

    def test_submit_one_job_correct_resources_quoting_and_worker_count(self):
        for world, partition, cpus, memory in ((1,'pdgpu-4090','8','32G'),(8,'pdgpu-3090','64','256G')):
            args = arguments.parse_args(['--world-size',str(world),'--output-dir',str(self.path/"output's space")])
            cmd = submit.command(args,'fixture123')
            for flag,value in (('-p',partition),('-c',cpus),('-m',memory),('-g',str(world)),('-n','1')):
                self.assertEqual(cmd[cmd.index(flag)+1],value)
            self.assertEqual(cmd.count('submit'),1)
            self.assertLess(len(cmd[cmd.index('-j')+1]),60)
            parsed = shlex.split(cmd[cmd.index('--cmd')+1])
            self.assertEqual(parsed[:3],['bash','mizar_audio_cutoff/run.sh','evaluate'])
            self.assertIn(str(args.output_dir),parsed)
            with patch.object(evaluate.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run, \
                 contextlib.redirect_stdout(io.StringIO()): evaluate.launch(args)
            self.assertEqual(run.call_count,1)
            self.assertIn(f'--nproc-per-node={world}',run.call_args.args[0])
            self.assertIn('--max-restarts=0',run.call_args.args[0])

    def test_existing_tracked_sources_and_reference_guard_unchanged(self):
        frozen = storage.read_json(ROOT/'tests/fixtures/mizar_audio_cutoff_original_sources.json')
        for name,digest in frozen['sources_lf_sha256'].items():
            self.assertEqual(storage.text_sha(ROOT/name),digest,name)
        self.assertIn('mizar_audio_cutoff/runtime.py',identity.sources())

    def test_cli_preview_imports_no_torch_or_production_assets(self):
        script = "import sys; from mizar_audio_cutoff.evaluate import main; assert main(['--check-args']) == 0; assert 'torch' not in sys.modules"
        run = subprocess.run([sys.executable,'-B','-c',script],cwd=config.ROOT,capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)

    def test_source_identity_rejects_cross_world_and_protocol(self):
        args = arguments.parse_args(['--world-size','1'])
        base = {'schema':config.SCHEMA,'protocol':config.PROTOCOL,'world_size':8,'sources':identity.sources(),
                'manifest_path':str(args.manifest),'model_dir':str(args.model_dir),'checkpoint':str(args.checkpoint)}
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            identity.verify(args,storage.seal(base,'identity_sha256'))


class ResultsAndResume(TempCase):
    def test_raw_sums_and_all_trace_fields_verified(self):
        record = records_fixture()[0]
        row = row_fixture(record,3)
        results.validate_row(row,record,3,'synthetic-identity')
        for edit in ({'prediction_index':1},{'correct':False},{'condition':'cutoff_04'},
                     {'option_loglikelihood':[float('nan'),-1]}, {'generation':{'fake':True}},
                     {'execution':{**row['execution'],'masked_layers_1based':list(range(4,31))}},
                     {'execution':{**row['execution'],'blocked_audio_plus_sep_tokens':126}}):
            broken = {k:v for k,v in row.items() if k != 'sha256'}
            broken.update(edit)
            with self.assertRaises(ValueError):
                results.validate_row(storage.seal(broken),record,3,'synthetic-identity')

    def test_exact_weighted_coverage_100_not104_and_pair_changes(self):
        records = records_fixture()
        base = [row_fixture(r,None,correct=r['global_index']<50) for r in records]
        rows = [row_fixture(r,3,correct=(r['global_index']<45 or r['global_index']>=95)) for r in records]
        parts = [[rows[i] for i in config.indices(rank,8)] for rank in range(8)]
        combined = results.merge(parts,records,3,'synthetic-identity',8)
        summary = results.summarize(combined,records,3,base)
        self.assertEqual((summary['accuracy'],summary['delta_vs_baseline']),(.5,0))
        self.assertEqual(summary['newly_wrong_indices'],list(range(45,50)))
        self.assertEqual(summary['newly_correct_indices'],list(range(95,100)))
        parts[0].pop()
        with self.assertRaises(ValueError): results.merge(parts,records,3,'synthetic-identity',8)

    def test_journal_partial_write_resume_and_complete_corruption(self):
        records = records_fixture()
        path = self.path/'ranks/rank00/cutoff_03.jsonl'
        storage.write_jsonl(path,[row_fixture(records[i],3) for i in range(3)])
        with path.open('ab') as stream: stream.write(b'{incomplete')
        self.assertEqual(len(results.read_rank(self.path,0,records,3,'synthetic-identity',1)),3)
        with self.assertRaises(ValueError): results.read_rank(self.path,0,records,3,'other',1)
        with path.open('ab') as stream: stream.write(b'{broken}\n')
        with self.assertRaises(ValueError): results.read_rank(self.path,0,records,3,'synthetic-identity',1)

    def test_actual_worker_baseline_cutoffs_and_resume_both_topologies(self):
        records = records_fixture()
        for world in (1,8):
            args = SimpleNamespace(world_size=world,output_dir=self.path/f'world{world}')
            expected = {'identity_sha256':'synthetic-identity'}
            gathered = {}
            class Controller:
                def set_cutoff(self,cutoff): self.cutoff=cutoff
            class Dist:
                def __init__(self,rank,cutoff): self.rank,self.cutoff=rank,cutoff
                def broadcast_object_list(self,values,src): values[0]=(config.condition_name(self.cutoff),self.cutoff,'synthetic-identity')
                def gather_object(self,rows,destination,dst):
                    gathered[self.rank]=rows
                    if self.rank==0: destination[:]=[gathered[r] for r in range(world)]
                def barrier(self): pass
            with patch.object(runtime,'score',side_effect=lambda loaded,p,r:score_fixture(r,loaded['cutoff_controller'].cutoff)) as score, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                for cutoff in (None,3,20):
                    for rank in [*range(1,world),0]:
                        worker.evaluate_condition(args,{'cutoff_controller':Controller()},{i:'prefix' for i in range(100)},
                                                  records,expected,cutoff,rank,Dist(rank,cutoff))
                for rank in [*range(1,world),0]:
                    worker.evaluate_condition(args,{'cutoff_controller':Controller()},{},records,expected,3,rank,Dist(rank,3))
            self.assertEqual(score.call_count,300)
            lines = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual([row['event'] for row in lines],['BASELINE_DONE','CUTOFF_DONE','CUTOFF_DONE','CUTOFF_DONE'])


class FrozenManifestCompatibility(TempCase):
    def manifest_fixture(self, historical=True):
        from mizar_layer_pair_search import config as old_config
        file=self.path/'real-source-bytes.wav'
        file.write_bytes(b'explicit file-hash fixture; audio not decoded by manifest loading')
        records=records_fixture()
        for row in records:
            row.update(prompt_token_ids=[17]*256, audio_audit={'waveform_sha256':'explicit fixture'})
        sources=frozen_manifest.reference.generator_sources()
        if historical:
            sources[frozen_manifest.MMSU_SOURCE]=frozen_manifest.HISTORICAL_MMSU
        value=storage.seal({'schema':old_config.SCHEMA,'protocol':old_config.PROTOCOL,'seed':1234,
            'records':records,'files':[storage.file_identity(file)],'generator_sources':sources},'manifest_sha256')
        path=self.path/'manifest.json'
        storage.write_json(path,value)
        return path,value,file

    def test_real_git_change_is_only_unused_prompt_helpers_and_constants(self):
        import ast,hashlib
        C=frozen_manifest.COMPATIBILITY
        path='code/RSmol/'+frozen_manifest.MMSU_SOURCE
        def read(revision):
            return subprocess.run(['git','show',revision+':'+path],cwd=ROOT,check=True,
                                  capture_output=True).stdout.decode('utf-8').replace('\r\n','\n')
        old,new=read(C['historical_git']),read(C['prompt_update_git'])
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),frozen_manifest.HISTORICAL_MMSU)
        self.assertEqual(hashlib.sha256(new.encode()).hexdigest(),frozen_manifest.CURRENT_MMSU)
        def used_tree(text):
            tree=ast.parse(text)
            tree.body=[node for node in tree.body if not (
                isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ('prompt','official_prompt'))
                and not (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
                    ('PROMPT_VERSION','PROMPT_POLICY','OUTPUT_NAMESPACE') for t in node.targets))]
            return ast.dump(tree,include_attributes=False)
        self.assertEqual(used_tree(old),used_tree(new))

    def test_historical_manifest_original_error_reproduced_then_load_unchanged(self):
        path,value,_=self.manifest_fixture()
        before=path.read_bytes()
        with self.assertRaisesRegex(ValueError,'different parser/tokenizer/audio'):
            frozen_manifest.reference.load(path)
        loaded=frozen_manifest.load(path)
        self.assertEqual(loaded,value)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(loaded['manifest_sha256'],value['manifest_sha256'])
        self.assertEqual(frozen_manifest.source_audit(loaded)['status'],'COMPATIBLE')
        # The old search remains strict after using the new cutoff-only adapter.
        with self.assertRaises(ValueError): frozen_manifest.reference.load(path)

    def test_current_manifest_keeps_exact_source_status(self):
        path,value,_=self.manifest_fixture(historical=False)
        self.assertEqual(frozen_manifest.load(path),value)
        self.assertEqual(frozen_manifest.source_audit(value)['status'],'EXACT')

    def test_other_audio_parser_missing_source_or_unknown_MMSU_change_rejected(self):
        _,value,_=self.manifest_fixture()
        for name in ('mizar_layer_pair_search/audio.py','mizar_layer_pair_search/data.py',frozen_manifest.MMSU_SOURCE):
            broken=copy.deepcopy(value); broken['generator_sources'][name]='unknown-sha'
            with self.assertRaisesRegex(ValueError,'unapproved frozen manifest source differences'):
                frozen_manifest.source_audit(broken)
        broken=copy.deepcopy(value); broken['generator_sources'].pop('mizar_layer_pair_search/audio.py')
        with self.assertRaises(ValueError): frozen_manifest.source_audit(broken)

    def test_manifest_record_tampering_and_source_file_changes_still_rejected(self):
        path,value,file=self.manifest_fixture()
        broken=copy.deepcopy(value); broken['records'][0]['prompt_token_ids'][0]=99
        storage.write_json(path,broken)
        with self.assertRaisesRegex(ValueError,'content hash mismatch'): frozen_manifest.load(path)
        storage.write_json(path,value); file.write_bytes(b'different audio bytes')
        with self.assertRaisesRegex(ValueError,'file identity changed'): frozen_manifest.load(path)

    def test_coordinator_and_worker_identity_both_use_compatible_loader(self):
        path,value,_=self.manifest_fixture()
        args=arguments.parse_args(['--manifest',str(path),'--output-dir',str(self.path/'out')])
        expected=storage.seal({'identity_sha256':'synthetic-identity','manifest_sha256':value['manifest_sha256']})
        with patch.object(identity,'build',return_value=expected),patch.object(identity,'verify') as verify, \
             patch.object(evaluate,'launch') as launch,patch.object(evaluate,'finalize',side_effect=[None,{'status':'PASS','decisions':1900}]), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            report=evaluate.run(args)
        self.assertEqual(report['status'],'PASS')
        self.assertEqual(launch.call_count,1)
        self.assertIn('COMPATIBLE',output.getvalue())
        self.assertIn('frozen_manifest.load',Path(identity.__file__).read_text())


def runtime_fixture(directory,world):
    expected={'identity_sha256':'synthetic-identity','world_size':world}
    audits,checks=[],[]
    for rank in range(world):
        audit={'binding':{'rank':rank,'worker_cuda_visible_devices':str(rank)},'SYNTHETIC_CPU_ONLY':True}
        check={'status':'PASS','rank':rank,'identity_sha256':'synthetic-identity','baseline_native_bitwise_equal':True,
               'cutoff_isolation':[{'status':'PASS','cutoff_layer_1based':L,'audio_sep_intervention_max_abs_error':0} for L in (3,20)]}
        audits.append(audit); checks.append(check)
        folder=directory/'ranks'/f'rank{rank:02d}'
        storage.write_json(folder/'model_audit.json',storage.seal(audit))
        storage.write_json(folder/'runtime_check.json',storage.seal(check))
        storage.write_json(folder/'rank_report.json',storage.seal({'status':'PASS','identity_sha256':'synthetic-identity',
            'binding':audit['binding'],'conditions_completed':19,'rows_per_condition':len(config.indices(rank,world))}))
    entries=[]
    for index in range(100):
        folder=directory/'cache'; folder.mkdir(exist_ok=True)
        path=folder/f'prefix_{index:03d}.pt'; path.write_bytes(b'SYNTHETIC_NOT_A_TENSOR')
        meta=storage.seal({'file_sha256':storage.file_sha(path)})
        storage.write_json(folder/f'prefix_{index:03d}.json',meta); entries.append(meta['sha256'])
    cache=storage.seal({'rows':100,'cache_identity':{'run_identity_sha256':'synthetic-identity'},'entry_meta_sha256':entries})
    storage.write_json(directory/'cache/cache_manifest.json',cache)
    storage.write_json(directory/'runtime_checks.json',storage.seal({'status':'PASS','world_size':world,
        'identity_sha256':'synthetic-identity','rank_audits':audits,'rank_checks':checks,'cache_manifest_sha256':cache['sha256']}))
    return expected


class FullSimulation(TempCase):
    def test_actual_finalizer19x100_both_topologies_csv_and_plot(self):
        records=records_fixture()
        for world in (1,8):
            directory=self.path/f'w{world}'; expected=runtime_fixture(directory,world)
            def synthetic_rows(directory,rank,records,cutoff,identity_sha,world):
                threshold=29 if cutoff is None else 39 if cutoff==3 else 35
                return [row_fixture(records[i],cutoff,i<threshold) for i in config.indices(rank,world)]
            with patch.object(results,'read_rank',side_effect=synthetic_rows) as read:
                report=evaluate.finalize(directory,{'records':records},expected)
            self.assertEqual(read.call_count,19*world)
            self.assertEqual((report['decisions'],report['generation_count']),(1900,0))
            self.assertEqual((report['baseline_accuracy'],report['best_cutoff_accuracy']),(.29,.39))
            self.assertEqual(report['top_tied_cutoffs'],[3])
            self.assertEqual(len((directory/'accuracy_by_cutoff.csv').read_text().splitlines()),20)
            self.assertEqual(len(storage.read_json(directory/'ranking.json')),18)
            self.assertEqual(len((directory/'conditions/cutoff_03/paired_changes.jsonl').read_text().splitlines()),10)
            (directory/'cache/prefix_099.pt').write_bytes(b'corrupted')
            with self.assertRaises(ValueError): evaluate.verify_artifacts(directory,expected)

    def test_failed_runtime_or_duplicate_GPU_audits_rejected(self):
        expected=runtime_fixture(self.path,8)
        original=storage.read_json(self.path/'runtime_checks.json')
        for edit in ({'status':'FAILED'},{'world_size':1},{'rank_audits':[original['rank_audits'][0]]*8}):
            storage.write_json(self.path/'runtime_checks.json',storage.seal({**{k:v for k,v in original.items() if k!='sha256'},**edit}))
            with self.assertRaises(ValueError): evaluate.verify_artifacts(self.path,expected)


ML=all(importlib.util.find_spec(name) for name in ('torch','transformers','safetensors'))


@unittest.skipUnless(ML,'requires real Torch and Transformers; no production weights are needed')
class RealLlama(unittest.TestCase):
    def make_model(self,backend='eager'):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.set_num_threads(1); torch.manual_seed(1234)
        cfg=LlamaConfig(vocab_size=41,hidden_size=16,intermediate_size=24,num_hidden_layers=30,
                        num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=1024,
                        attention_dropout=0.,use_cache=False,tie_word_embeddings=True)
        cfg._attn_implementation=backend
        lm=LlamaForCausalLM(cfg).float().eval().requires_grad_(False)
        x=lm.get_input_embeddings()(torch.tensor([[1,2,3,4,5,6,7,8]]))
        return torch,lm,x

    def attach(self,lm):
        return runtime.AudioCutoff(lm,hidden_size=16,prefix_tokens=6,blocked_tokens=3)

    def forward(self,lm,x,**kw):
        return lm(inputs_embeds=x,attention_mask=None,use_cache=False,**kw)

    def test_native_baseline_all19_and_weights_exactly_preserved(self):
        for backend in ('eager','sdpa'):
            torch,lm,x=self.make_model(backend)
            native=self.forward(lm,x).logits
            weights={k:v.clone() for k,v in lm.state_dict().items()}
            ctl=self.attach(lm)
            self.assertTrue(torch.equal(native,self.forward(lm,x).logits))
            for L in config.CONDITIONS:
                ctl.set_cutoff(L); logits=self.forward(lm,x).logits
                self.assertTrue(bool(torch.isfinite(logits).all()))
                self.assertEqual(ctl.trace,list(range(1,31)))
                self.assertEqual(ctl.masked,config.masked_layers(L))
            for k,v in lm.state_dict().items(): self.assertTrue(torch.equal(v,weights[k]))

    def test_BF16_autocast_real383prefix_audio126_SEP1_both_backends(self):
        for backend in ('eager','sdpa'):
            torch,lm,_=self.make_model(backend)
            prefix=lm.get_input_embeddings()(torch.randint(1,41,(1,383))).to(torch.bfloat16)
            with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
                native=self.forward(lm,prefix).logits
                ctl=runtime.AudioCutoff(lm,hidden_size=16,prefix_tokens=383,blocked_tokens=127)
                baseline=self.forward(lm,prefix).logits
                self.assertTrue(torch.equal(native,baseline))
                self.assertEqual(baseline.dtype,torch.bfloat16)
                ctl.set_cutoff(3)
                normal=self.forward(lm,prefix).logits[:,127:]
                def perturb(h):
                    changed=h.clone(); changed[:,:127]=7.; return changed
                ctl.intervention=perturb
                isolated=self.forward(lm,prefix).logits[:,127:]
                self.assertTrue(torch.equal(normal,isolated))
                ctl.intervention=None
                ids=[9,10,11]
                actual=runtime.reference.option_score(lm,prefix,ids,torch.device('cpu'))
                history,expected=prefix,0.
                for token in ids:
                    expected+=float(self.forward(lm,history).logits[0,-1].float().log_softmax(-1)[token])
                    history=torch.cat([history,lm.get_input_embeddings()(torch.tensor([[token]])).to(torch.bfloat16)],dim=1)
                self.assertAlmostEqual(actual,expected,places=5)

    def test_masks_audio_AND_separator_no_empty_rows_and_causality(self):
        torch,lm,x=self.make_model(); ctl=self.attach(lm); ctl.set_cutoff(3)
        out=self.forward(lm,x,output_attentions=True)
        for index,attention in enumerate(out.attentions,1):
            if index>=3:
                self.assertEqual(int(torch.count_nonzero(attention[:,:,3:,:3])),0)
            self.assertEqual(int(torch.count_nonzero(torch.triu(attention,diagonal=1))),0)
            self.assertTrue(bool(torch.isfinite(attention).all()))
        mask=runtime.cutoff_mask(8,x.device,x.dtype,3)[0,0]
        self.assertTrue(bool(torch.isfinite(mask).any(dim=-1).all()))
        self.assertTrue(bool(torch.isneginf(mask[3:,:3]).all()))
        changed=x.clone(); changed[:,7]=100
        torch.testing.assert_close(self.forward(lm,changed).logits[:,:7],out.logits[:,:7],rtol=0,atol=0)

    def test_boundary_intervention_isolated_but_early_audio_is_retained(self):
        for backend in ('eager','sdpa'):
            torch,lm,x=self.make_model(backend); ctl=self.attach(lm)
            for L in (3,20):
                ctl.set_cutoff(L); normal=self.forward(lm,x).logits[:,3:]
                def perturb(hidden):
                    h=hidden.clone(); h[:,:3]=17.; return h
                ctl.intervention=perturb
                altered=self.forward(lm,x).logits[:,3:]
                torch.testing.assert_close(altered,normal,rtol=0,atol=0)
                ctl.intervention=None
                early=x.clone(); early[:,:3]=17.
                self.assertGreater(float((self.forward(lm,early).logits[:,3:]-normal).abs().max()),0)

    def test_runtime_checks_execute_real_native_and_cutoff_comparisons(self):
        torch,lm,x=self.make_model(); ctl=self.attach(lm)
        loaded={'lm':lm,'device':torch.device('cpu'),'cutoff_controller':ctl}
        checks=runtime.runtime_checks(loaded,x[:,:6],{'choice_token_ids':[[9,10]]})
        self.assertEqual(checks['status'],'PASS')
        self.assertTrue(checks['baseline_native_bitwise_equal'])
        self.assertEqual([r['audio_sep_intervention_max_abs_error'] for r in checks['cutoff_isolation']],[0,0])

    def test_sum_likelihood_shift_matches_stepwise_manual_no_future_targets(self):
        torch,lm,x=self.make_model(); ctl=self.attach(lm); ctl.set_cutoff(3)
        prefix=x[:,:6]; ids=[9,10,11]; history=prefix; expected=0.
        for token in ids:
            expected+=float(self.forward(lm,history).logits[0,-1].float().log_softmax(-1)[token])
            history=torch.cat([history,lm.get_input_embeddings()(torch.tensor([[token]]))],dim=1)
        actual=runtime.reference.option_score(lm,prefix,ids,torch.device('cpu'))
        self.assertAlmostEqual(actual,expected,places=5)
        one=runtime.reference.option_score(lm,prefix,[9],torch.device('cpu'))
        self.assertAlmostEqual(one,float(self.forward(lm,prefix).logits[0,-1].log_softmax(-1)[9]),places=5)

    def test_reject_pruned_layers_cache_and_reordered_modules(self):
        torch,lm,x=self.make_model(); ctl=self.attach(lm)
        with self.assertRaises(ValueError): self.forward(lm,x,past_key_values='forbidden')
        lm.model.layers=torch.nn.ModuleList(list(reversed(lm.model.layers)))
        with self.assertRaises(ValueError): ctl.set_cutoff(3)


if __name__=='__main__': unittest.main()
