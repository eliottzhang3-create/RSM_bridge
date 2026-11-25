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
from mizar_audio_cutoff1000 import arguments, cache, config, data, evaluate, manifest, identity, results, runtime, submit, worker


def records_fixture():
    return [{'global_index':i, 'key':f'{config.DATASETS[i//250]}:{i}', 'dataset':config.DATASETS[i//250],
             'prompt':f'explicit synthetic question {i}', 'choice_bodies':['short','long choice'],
             'choice_token_ids':[[3],[4,5]], 'gold_index':i % 2} for i in range(config.ROWS)]


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


class FixtureTokenizer:
    """Explicit tokenizer fixture; production uses the release-verified tokenizer."""
    def encode(self, text, add_special_tokens=True):
        return [30 + ord(c) % 19 for c in text]

    def __call__(self, text, **kwargs):
        ids = self.encode(text)[:256]
        return {'input_ids': ids + [17]*(256-len(ids)), 'attention_mask': [1]*len(ids)+[0]*(256-len(ids))}


def pools_fixture():
    pools = {}
    for name in config.DATASETS:
        rows = []
        for i in range(300):
            if name == 'reasonaqa':
                row = {'subtype': 'AudioCaps-MCQ.json', 'filepath1': f'{i}.wav', 'filepath2': '',
                       'input': f'Question {i}? a) short b) long choice c) third d) fourth', 'answer': 'short'}
            elif name == 'mmsu':
                row = {'id': str(i), 'question': f'Question {i}?', 'choice_a': 'short',
                       'choice_b': None, 'choice_c': 'long choice', 'choice_d': 'fourth',
                       'answer_gt': 'short', 'task_name': 'synthetic'}
            else:
                row = {'id': str(i), 'question': f'Question {i}?', 'choices': ['short', 'long choice'],
                       'answer': 'short', 'category': 'synthetic'}
            rows.append((i, row))
        pools[name] = rows
    return pools


class Manifest1000(TempCase):
    def test_deterministic_250_each_no_replacement_and_same_eligibility_as_old(self):
        import random
        from mizar_layer_pair_search import data as old_data
        pools = pools_fixture()
        rows, audit = data.select_records(pools, FixtureTokenizer())
        again, audit2 = data.select_records(pools, FixtureTokenizer())
        self.assertEqual((rows, audit), (again, audit2))
        self.assertEqual(len(rows), 1000)
        self.assertEqual(len({r['key'] for r in rows}), 1000)
        expected = sorted(random.Random(1234).sample(range(300), 250))
        for name in config.DATASETS:
            self.assertEqual(audit[name]['selected_eligible_indices'], expected)
            chosen = [r for r in rows if r['dataset'] == name]
            self.assertEqual([r['source_row_index'] for r in chosen], expected)
            for index, raw in pools[name]:
                self.assertEqual(data.normalize_record(name, index, raw), old_data.normalize_record(name, index, raw))
        # Null MMSU choices drop their original slot and are relabeled contiguously.
        mmsu = next(r for r in rows if r['dataset'] == 'mmsu')
        self.assertEqual(mmsu['valid_original_slots'], [0, 2, 3])
        self.assertIn('b) long choice c) fourth', mmsu['prompt'])

    def test_exclusions_insufficient_pool_duplicate_ids_and_oversized_options(self):
        pools = pools_fixture()
        pools['mmau'].append((300, {'id': 'excluded', 'question': '?', 'choices': ['A) same', 'B) SAME'], 'answer': 'same'}))
        _, audit = data.select_records(pools, FixtureTokenizer())
        self.assertEqual(audit['mmau']['excluded'], {'empty_or_duplicate_normalized_options': 1})
        pools['mmau'] = pools['mmau'][:249]
        with self.assertRaisesRegex(ValueError, 'need 250 eligible'): data.select_records(pools, FixtureTokenizer())
        pools = pools_fixture(); pools['mmau'].append(pools['mmau'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate eligible ID'): data.select_records(pools, FixtureTokenizer())
        for edit in ({'subtype': 'AudioCaps-Caption.json'}, {'filepath2': 'other.wav'}):
            with self.assertRaises(data.Ineligible): data.normalize_record('reasonaqa', 0, {**pools['reasonaqa'][0][1], **edit})
        raw = {**pools['mmau'][0][1], 'answer': 'missing'}
        with self.assertRaises(data.Ineligible): data.normalize_record('mmau', 0, raw)
        record = data.normalize_record('mmau', 0, pools['mmau'][0][1]); record['choice_bodies'][0] = 'x'*642
        with self.assertRaisesRegex(data.Ineligible, 'exceeds_1024'): data.tokenize_record(record, FixtureTokenizer())

    def build_fixture(self, compare=False):
        from mizar_s1_ced_mmsu_eval import dataset as mmsu
        source = self.path / 'explicit-source.dat'; source.write_bytes(b'explicit metadata/audio fixture')
        argv = ['--manifest', str(self.path/'manifest/manifest.json'), '--output-dir', str(self.path/'output')]
        for name in ('mmau-metadata', 'mmar-metadata', 'reasonaqa-test'):
            argv.extend(['--'+name, str(source)])
        if compare:
            old = storage.seal({'records': [{'key': 'mmau:0'}]}, 'manifest_sha256')
            storage.write_json(self.path/'old.json', old)
            argv += ['--compare-manifest', str(self.path/'old.json')]
        args = arguments.parse_args(argv)
        def attach(records, *_):
            return [{**r, 'audio_ref': {'kind': 'file', 'path': str(source)}} for r in records]
        with patch.object(manifest, 'load_tokenizer', return_value=(FixtureTokenizer(), [])), \
             patch.object(mmsu, 'shard_paths', return_value=[source]), \
             patch.object(data, 'read_pools', return_value=(pools_fixture(), [], {'synthetic': True})), \
             patch.object(data, 'attach_audio', side_effect=attach), \
             patch.object(manifest.audio, 'load_fixed20', return_value=(None, {'waveform_sha256': 'explicit_fixture'})) as audio, \
             contextlib.redirect_stdout(io.StringIO()):
            frozen = manifest.ensure(args)
        self.assertEqual(audio.call_count, 1000)
        return args, frozen, source

    def test_actual_builder_1000_audio_audits_atomic_reuse_and_source_checks(self):
        args, value, source = self.build_fixture()
        before = args.manifest.read_bytes()
        self.assertEqual(manifest.load(args.manifest), value)
        with patch.object(manifest, 'build', side_effect=AssertionError('must reuse')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(manifest.ensure(args), value)
        self.assertEqual(args.manifest.read_bytes(), before)
        other = copy.copy(args); other.mmau_metadata = self.path/'other.json'
        with self.assertRaisesRegex(ValueError, 'source arguments differ'): manifest.ensure(other)
        broken = copy.deepcopy(value); broken['generator_sources']['mizar_layer_pair_search/audio.py'] = 'changed'
        broken = storage.seal({k:v for k,v in broken.items() if k!='manifest_sha256'}, 'manifest_sha256')
        storage.write_json(args.manifest, broken)
        with self.assertRaisesRegex(ValueError, 'source mismatch'): manifest.load(args.manifest)
        storage.write_json(args.manifest, value)
        source.write_bytes(b'mutated source')
        with self.assertRaisesRegex(ValueError, 'file identity changed'): manifest.load(args.manifest)

    def test_old_schema_wrong_count_seed_token_edits_and_overlap(self):
        args, value, _ = self.build_fixture(compare=True)
        common = [r for r in value['records'] if r['key'] == 'mmau:0']
        self.assertEqual(value['previous_manifest_overlap']['rows_in_common'], len(common))
        self.assertFalse(value['previous_manifest_overlap']['sampling_affected'])
        for edit in ({'schema': 'old100'}, {'seed': 42}, {'records': value['records'][:100]}):
            with self.assertRaises(ValueError):
                data.validate_manifest(storage.seal({**{k:v for k,v in value.items() if k!='manifest_sha256'}, **edit}, 'manifest_sha256'))
        broken = copy.deepcopy(value); broken['records'][0]['prompt_token_ids'][0] += 1
        storage.write_json(args.manifest, broken)
        with self.assertRaisesRegex(ValueError, 'content hash mismatch'): manifest.load(args.manifest)

    def test_missing_selected_audio_is_fatal_without_resampling(self):
        from mizar_s1_ced_mmsu_eval import dataset as mmsu
        args = arguments.parse_args(['--mmar-audio-root', str(self.path)])
        row = {'id': 'missing', 'question': '?', 'choices': ['yes', 'no'], 'answer': 'yes', 'audio_path': 'absent.wav'}
        record = data.normalize_record('mmar', 0, row)
        pools = {name: [] for name in config.DATASETS}; pools['mmar'] = [(0, row)]
        # Fixture has no production MMSU files; only exercise selected MMAR audio.
        with patch.object(mmsu, 'shard_paths', return_value=[]), self.assertRaises(FileNotFoundError):
            data.attach_audio([record], pools, args)

    def test_coordinator_ensures_manifest_before_one_launch(self):
        args = arguments.parse_args(['--manifest', str(self.path/'m/manifest.json'), '--output-dir', str(self.path/'o')])
        expected = storage.seal({'fixture': True}, 'identity_sha256')
        with patch.object(manifest, 'ensure', return_value={'manifest_sha256': 'fixture'}) as ensure, \
             patch.object(identity, 'build', return_value=expected), patch.object(identity, 'verify'), \
             patch.object(evaluate, 'launch') as launch, \
             patch.object(evaluate, 'finalize', side_effect=[None, {'status': 'PASS', 'decisions': 27000}]), \
             contextlib.redirect_stdout(io.StringIO()):
            report = evaluate.run(args)
        self.assertEqual((ensure.call_count, launch.call_count, report['decisions']), (1, 1, 27000))


class Contracts(TempCase):
    def test_exact27_conditions_layer_boundaries_and_rank_coverage(self):
        self.assertEqual(config.CONDITIONS, (None,*range(3,29)))
        self.assertEqual(config.masked_layers(3),list(range(3,31)))
        self.assertEqual(config.masked_layers(28),list(range(28,31)))
        for world in (1,8):
            self.assertEqual(sorted(i for r in range(world) for i in config.indices(r,world)),list(range(config.ROWS)))
        self.assertEqual([len(config.indices(r,8)) for r in range(8)],[125]*8)
        for bad in (0,1,2,29,30,True,3.0):
            with self.assertRaises(ValueError): config.condition_name(bad)

    def test_new_manifest_shared_outputs_distinct_and_auto_generator(self):
        one, eight = arguments.parse_args([]), arguments.parse_args(['--world-size','8'])
        self.assertEqual(one.manifest,eight.manifest)
        self.assertNotEqual(one.output_dir,eight.output_dir)
        self.assertIn('manifest.ensure',Path(evaluate.__file__).read_text())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments.parse_args(['--manifest',str(self.path/'out/m.json'),'--output-dir',str(self.path/'out')])

    def test_submit_one_job_correct_resources_quoting_and_worker_count(self):
        for world, partition, cpus, memory in ((1,'pdgpu-4090','8','32G'),(8,'pdgpu-3090','64','256G'),
                                               (8,'pdgpu-5090','64','256G')):
            args = submit.parse_args(['--world-size',str(world),'--partition',partition,
                                      '--output-dir',str(self.path/"output's space")])
            cmd = submit.command(args,'fixture123')
            for flag,value in (('-p',partition),('-c',cpus),('-m',memory),('-g',str(world)),('-n','1')):
                self.assertEqual(cmd[cmd.index(flag)+1],value)
            self.assertEqual(cmd.count('submit'),1)
            self.assertLess(len(cmd[cmd.index('-j')+1]),60)
            parsed = shlex.split(cmd[cmd.index('--cmd')+1])
            self.assertEqual(parsed[:3],['bash','mizar_audio_cutoff1000/run.sh','evaluate'])
            self.assertIn(str(args.output_dir),parsed)
            with patch.object(evaluate.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run, \
                 contextlib.redirect_stdout(io.StringIO()): evaluate.launch(args)
            self.assertEqual(run.call_count,1)
            self.assertIn(f'--nproc-per-node={world}',run.call_args.args[0])
            self.assertIn('--max-restarts=0',run.call_args.args[0])
            worker_args = arguments.parse_args(arguments.forwarded(args))
            self.assertEqual(worker_args.output_dir, args.output_dir)
        default5090 = submit.parse_args(['--world-size','8','--partition','pdgpu-5090'])
        self.assertTrue(default5090.output_dir.name.endswith('_8gpu5090'))
        self.assertEqual(default5090.manifest, arguments.parse_args([]).manifest)

    def test_existing_tracked_sources_and_reference_guard_unchanged(self):
        frozen = storage.read_json(ROOT/'tests/fixtures/mizar_audio_cutoff1000_original_sources.json')
        for name,digest in frozen['sources_lf_sha256'].items():
            self.assertEqual(storage.text_sha(ROOT/name),digest,name)
        self.assertIn('mizar_audio_cutoff1000/runtime.py',identity.sources())

    def test_cli_preview_imports_no_torch_or_production_assets(self):
        script = "import sys; from mizar_audio_cutoff1000.evaluate import main; assert main(['--check-args']) == 0; assert 'torch' not in sys.modules"
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

    def test_exact_weighted_coverage1000_and_subsets_and_pair_changes(self):
        records = records_fixture()
        base = [row_fixture(r,None,correct=r['global_index']<500) for r in records]
        rows = [row_fixture(r,3,correct=(r['global_index']<450 or r['global_index']>=950)) for r in records]
        parts = [[rows[i] for i in config.indices(rank,8)] for rank in range(8)]
        combined = results.merge(parts,records,3,'synthetic-identity',8)
        summary = results.summarize(combined,records,3,base)
        self.assertEqual((summary['accuracy'],summary['delta_vs_baseline']),(.5,0))
        self.assertEqual(summary['newly_wrong_indices'],list(range(450,500)))
        self.assertEqual(summary['newly_correct_indices'],list(range(950,1000)))
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
                for cutoff in (None,3,28):
                    for rank in [*range(1,world),0]:
                        worker.evaluate_condition(args,{'cutoff_controller':Controller()},{i:'prefix' for i in range(config.ROWS)},
                                                  records,expected,cutoff,rank,Dist(rank,cutoff))
                for rank in [*range(1,world),0]:
                    worker.evaluate_condition(args,{'cutoff_controller':Controller()},{},records,expected,3,rank,Dist(rank,3))
            self.assertEqual(score.call_count,3000)
            lines = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual([row['event'] for row in lines],['BASELINE_DONE','CUTOFF_DONE','CUTOFF_DONE','CUTOFF_DONE'])


def runtime_fixture(directory,world):
    expected={'identity_sha256':'synthetic-identity','world_size':world}
    audits,checks=[],[]
    for rank in range(world):
        audit={'binding':{'rank':rank,'worker_cuda_visible_devices':str(rank)},'SYNTHETIC_CPU_ONLY':True}
        check={'status':'PASS','rank':rank,'identity_sha256':'synthetic-identity','baseline_native_bitwise_equal':True,
               'cutoff_isolation':[{'status':'PASS','cutoff_layer_1based':L,'audio_sep_intervention_max_abs_error':0} for L in (3,28)]}
        audits.append(audit); checks.append(check)
        folder=directory/'ranks'/f'rank{rank:02d}'
        storage.write_json(folder/'model_audit.json',storage.seal(audit))
        storage.write_json(folder/'runtime_check.json',storage.seal(check))
        storage.write_json(folder/'rank_report.json',storage.seal({'status':'PASS','identity_sha256':'synthetic-identity',
            'binding':audit['binding'],'conditions_completed':len(config.CONDITIONS),'rows_per_condition':len(config.indices(rank,world))}))
    entries=[]
    for index in range(config.ROWS):
        folder=directory/'cache'; folder.mkdir(exist_ok=True)
        path=folder/f'prefix_{index:03d}.pt'; path.write_bytes(b'SYNTHETIC_NOT_A_TENSOR')
        meta=storage.seal({'file_sha256':storage.file_sha(path)})
        storage.write_json(folder/f'prefix_{index:03d}.json',meta); entries.append(meta['sha256'])
    cache=storage.seal({'rows':config.ROWS,'cache_identity':{'run_identity_sha256':'synthetic-identity'},'entry_meta_sha256':entries})
    storage.write_json(directory/'cache/cache_manifest.json',cache)
    storage.write_json(directory/'runtime_checks.json',storage.seal({'status':'PASS','world_size':world,
        'identity_sha256':'synthetic-identity','rank_audits':audits,'rank_checks':checks,'cache_manifest_sha256':cache['sha256']}))
    return expected


class FullSimulation(TempCase):
    def test_actual_finalizer27x1000_both_topologies_csv_and_plot(self):
        records=records_fixture()
        for world in (1,8):
            directory=self.path/f'w{world}'; expected=runtime_fixture(directory,world)
            def synthetic_rows(directory,rank,records,cutoff,identity_sha,world):
                threshold=290 if cutoff is None else 390 if cutoff in (3,28) else 350
                return [row_fixture(records[i],cutoff,i<threshold) for i in config.indices(rank,world)]
            with patch.object(results,'read_rank',side_effect=synthetic_rows) as read:
                report=evaluate.finalize(directory,{'records':records},expected)
            self.assertEqual(read.call_count,27*world)
            self.assertEqual((report['decisions'],report['generation_count']),(27000,0))
            self.assertEqual((report['baseline_accuracy'],report['best_cutoff_accuracy']),(.29,.39))
            self.assertEqual(report['top_tied_cutoffs'],[3,28])
            self.assertEqual(len((directory/'accuracy_by_cutoff.csv').read_text().splitlines()),28)
            self.assertEqual(len(storage.read_json(directory/'ranking.json')),26)
            self.assertEqual(len((directory/'conditions/cutoff_03/paired_changes.jsonl').read_text().splitlines()),100)
            (directory/'cache/prefix_999.pt').write_bytes(b'corrupted')
            with self.assertRaises(ValueError): evaluate.verify_artifacts(directory,expected)

    def test_failed_runtime_or_duplicate_GPU_audits_rejected(self):
        expected=runtime_fixture(self.path,8)
        original=storage.read_json(self.path/'runtime_checks.json')
        for edit in ({'status':'FAILED'},{'world_size':1},{'rank_audits':[original['rank_audits'][0]]*8}):
            storage.write_json(self.path/'runtime_checks.json',storage.seal({**{k:v for k,v in original.items() if k!='sha256'},**edit}))
            with self.assertRaises(ValueError): evaluate.verify_artifacts(self.path,expected)


ML=all(importlib.util.find_spec(name) for name in ('torch','transformers','safetensors'))


@unittest.skipUnless(ML, 'requires real Torch')
class PrefixCache(TempCase):
    def test_real_BF16_serialization_resume_no_reencoding_and_tamper_rejected(self):
        import torch
        embed = torch.nn.Embedding(32, 576).eval()
        rows = records_fixture()[:2]
        for r in rows:
            r.update(audio_ref={'kind': 'file', 'path': 'explicit-audio-fixture'},
                     audio_audit={'waveform_sha256': 'fixture'}, prompt_token_ids=[17]*256)
        with torch.inference_mode():
            prefix = torch.cat([torch.ones(1,126,576).to(torch.bfloat16),
                                embed(torch.tensor([[0]])).to(torch.bfloat16),
                                embed(torch.tensor([[17]*256])).to(torch.bfloat16)], dim=1)
        precision = {k: 'explicit_CPU_fixture' for k in ('torch', 'transformers', 'cuda', 'gpu', 'weights_dtype',
                     'forward_dtype', 'audio_frontend_dtype', 'prefix_dtype', 'attention_implementation', 'sdpa_backends')}
        loaded = {'audit': precision, 'device': torch.device('cpu'),
                  'lm': SimpleNamespace(get_input_embeddings=lambda: embed)}
        expected = {'identity_sha256': 'fixture'}
        folder = self.path/'cache'
        # A two-row cache unit fixture; production still requires exactly 1000.
        with patch.object(config, 'ROWS', 2), patch.object(runtime, 'prefix', return_value=prefix) as encode, \
             patch.object(cache.audio, 'load_fixed20', return_value=(None, rows[0]['audio_audit'])), \
             contextlib.redirect_stdout(io.StringIO()):
            cache.prepare(loaded, rows, folder, expected)
            self.assertEqual(encode.call_count, 2)
            encode.reset_mock()
            cache.prepare(loaded, rows, folder, expected)
            values, _ = cache.load_all(loaded, rows, folder, expected)
            self.assertEqual(encode.call_count, 0)
            self.assertTrue(all(torch.equal(v, prefix) for v in values.values()))
            (folder/'cache_manifest.json').unlink()
            recovered = cache.prepare(loaded, rows, folder, expected)
            self.assertTrue(recovered['online_serialized_check']['recovered_after_entry_commit'])
            self.assertEqual(encode.call_count, 0)
            (folder/'prefix_001.pt').write_bytes(b'explicit corruption')
            with self.assertRaisesRegex(ValueError, 'file mismatch'): cache.load_all(loaded, rows, folder, expected)


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

    def test_native_baseline_all27_and_weights_exactly_preserved(self):
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
            for L in (3,28):
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
