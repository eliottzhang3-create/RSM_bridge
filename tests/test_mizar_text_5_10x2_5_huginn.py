"""Route identity, original data/update parity, and independent epoch startup."""
from __future__ import annotations
import ast
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'code/RSmol'))
from mizar_text_5_10x2_5_huginn import contract as C
from mizar_text_5_10x2_5_huginn import conversion as V
from mizar_text_5_10x2_5_huginn import publication as P

TRAIN_PATH = ROOT / 'code/RSmol/scripts/train_stage4_mizar_text_5_10x2_5_huginn_ddp.py'
spec = importlib.util.spec_from_file_location('mizar_text_train_test', TRAIN_PATH)
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)


def model_fixture(path, step=None):
    path.mkdir(parents=True)
    config = C.build_target_config(types.SimpleNamespace(
        num_hidden_layers=30, hidden_size=576, vocab_size=49152, tie_word_embeddings=True))
    (path / 'config.json').write_text(json.dumps(vars(config)), encoding='utf-8')
    (path / 'model.safetensors').write_bytes(b'fake model; not read by JSON tests')
    (path / 'tokenizer.json').write_text('{}', encoding='utf-8')
    if step is None:
        V.write_json(path / 'conversion_complete.json', {'complete': True, 'text_route': C.ROUTE,
                     'huginn': C.HUGINN_FIELDS,
                     'source': C.source_identity(), 'physical_layers': 20, 'logical_layers': 30})
    else:
        (path / 'training_state.pt').write_bytes(b'POISON: weights-only must never deserialize this')
        V.write_json(path / 'checkpoint_complete.json', {'complete': True, 'mode': 'formal', 'text_route': C.ROUTE,
                     'architecture': C.ARCHITECTURE, 'world_size': 8, 'optimizer_step': step,
                     'rng_states_by_rank': [str(i) for i in range(8)],
                     'data_cursors_by_rank': [str(i) for i in range(8)]})
    return path


def state_fixture(max_lr='8e-4'):
    config = T._parse_args(['--max-lr', max_lr])
    return {'configuration': asdict(config), 'optimizer_step': 500,
            'checkpoint_contract': {'text_route': C.ROUTE, 'architecture': C.ARCHITECTURE,
                                    'huginn': C.HUGINN_FIELDS, 'rng_states_by_rank': True},
            'rng_states_by_rank': {str(i):{} for i in range(8)},
            'scheduler_config': T.scheduler_metadata(total_steps=9244, warmup_steps=463,
                                                      max_lr=config.max_lr, min_lr=config.min_lr)}


def ast_functions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


class ContractTests(unittest.TestCase):
    def test_original_sources_unchanged(self):
        guard = json.loads((C.HERE / 'source_guard.json').read_text(encoding='utf-8'))
        for name, digest in guard.items():
            with self.subTest(file=name):
                normalized = (ROOT / name).read_text(encoding='utf-8').encode()
                self.assertEqual(hashlib.sha256(normalized).hexdigest(), digest)

    def test_original_update_data_optimizer_functions_preserved(self):
        old = ast_functions(ROOT / 'code/RSmol/scripts/train_stage4_5_10_5_ddp.py')
        new = ast_functions(TRAIN_PATH)
        names = ['discover_parquet_files', 'assign_shards', '_iter_shard_texts', 'collate_dynamic_padding',
                 'DistributedParquetStream', '_bf16_autocast', '_causal_loss_sum',
                 'token_weighted_gradient_scale', 'optimizer_parameter_audit', '_training_window',
                 'cosine_warmup_factor', 'compute_warmup_steps', '_tokenize_ids', '_load_scheduler']
        for name in names:
            with self.subTest(function=name):
                self.assertIn(name, old)
                self.assertEqual(old[name], new[name])
        # Preserve the entire update loop, allowing only the recipe guard and
        # explicit per-rank RNG gathering at checkpoint boundaries.
        old_tree=ast.parse((ROOT/'code/RSmol/scripts/train_stage4_5_10_5_ddp.py').read_text(encoding='utf-8'))
        new_tree=ast.parse(TRAIN_PATH.read_text(encoding='utf-8'))
        old_run=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_training')
        new_run=next(n for n in new_tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_training')
        for n in ast.walk(new_run):
            if isinstance(n,ast.If):
                n.body=[child for child in n.body if not (isinstance(child,ast.Expr)
                    and isinstance(child.value,ast.Call) and isinstance(child.value.func,ast.Name)
                    and child.value.func.id=='validate_formal_recipe')]
        class RemoveRngGather(ast.NodeTransformer):
            def visit_Call(self, node):
                node.keywords=[k for k in node.keywords if k.arg!='rng_states_by_rank']
                return self.generic_visit(node)
            def visit_Assign(self, node):
                if any(isinstance(t,ast.Name) and t.id=='rng_states_by_rank' for t in node.targets):
                    return None
                return self.generic_visit(node)
        new_run=RemoveRngGather().visit(new_run)
        self.assertEqual(ast.dump(old_run,include_attributes=False),ast.dump(new_run,include_attributes=False))
        imports=[node for node in ast.walk(new_tree) if isinstance(node,ast.ImportFrom)
                 and node.module=='code.RSmol.recursive_model_mizar_text_5_10x2_5_huginn']
        actual=ast_functions(ROOT/'code/RSmol/recursive_model_mizar_text_5_10x2_5_huginn.py')
        for node in imports:
            for name in node.names:
                self.assertIn(name.name,actual)

    def test_source_is_final_not_base_asset(self):
        identity = C.source_identity()
        self.assertEqual(identity['checkpoint_relative_path'], 'checkpoints/final/stage3_q4_seed20260905.ckpt')
        self.assertEqual(identity['checkpoint_sha256'], '0c542f63e11ed55d360feb0d66e7be0e8c6a878f44cffd302a92388f915388a2')
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(V.Path, 'stat') as stat, patch.object(V, 'sha256_file') as digest:
                records = [C.release_manifest()['checkpoint']] + [r for r in C.release_manifest()['files']
                    if r['path'].startswith('assets/smollm2-135m/') and not r['path'].endswith('model.safetensors')]
                stat.side_effect = [types.SimpleNamespace(st_size=r['bytes']) for r in records]
                digest.side_effect = [r['sha256'] for r in records]
                checked = V.verify_source_files(Path(tmp))
            self.assertEqual(checked, records)
            self.assertNotIn('model.safetensors', str(checked))
            self.assertNotIn('ced-small', str(checked))

    def test_bad_source_hash_rejected(self):
        with patch.object(V.Path, 'stat', return_value=types.SimpleNamespace(st_size=637470191)), \
             patch.object(V, 'sha256_file', return_value='bad'):
            with self.assertRaises(ValueError):
                V.verify_source_files(Path('/fake'))

    def test_config_and_layer_mapping_preserve_original(self):
        source = types.SimpleNamespace(num_hidden_layers=30, hidden_size=576, vocab_size=49152,
                                       tie_word_embeddings=True, rope_scaling={'rope_type': 'default'})
        before = copy.deepcopy(vars(source))
        target = C.build_target_config(source)
        C.validate_model_config(vars(target))
        self.assertEqual(vars(source), before)
        self.assertEqual(C.SOURCE_LAYERS, (0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29))
        self.assertEqual(C.SCHEDULE, tuple(range(15)) + tuple(range(5,20)))
        for i, original in enumerate(C.SOURCE_LAYERS):
            self.assertEqual(V.source_key(f'model.layers.{i}.self_attn.q_proj.weight'),
                             f'model.layers.{original}.self_attn.q_proj.weight')
        self.assertEqual(V.source_key('lm_head.weight'), 'lm_head.weight')

    def test_huginn_distribution_and_wrong_architecture_rejected(self):
        raw=vars(C.build_target_config(types.SimpleNamespace(num_hidden_layers=30)))
        self.assertAlmostEqual(C.STATE_STD,math.sqrt(2/5))
        self.assertEqual(C.HUGINN_FIELDS['huginn_concat_order'],['state','prelude'])
        for key in C.HUGINN_FIELDS:
            broken={**raw,key:'wrong'}
            with self.subTest(key=key),self.assertRaises(ValueError):
                C.validate_huginn_config(broken)

    def test_attention_cache_rope_inherited_and_no_global_registration(self):
        old=ast_functions(ROOT/'code/RSmol/recursive_model_5_10_5.py')
        newpath=ROOT/'code/RSmol/recursive_model_mizar_text_5_10x2_5_huginn.py'
        new=ast_functions(newpath)
        for name in ['LogicalSlotCacheView','_causal_mask','_call_decoder_layer',
                     'make_dynamic_cache','build_5_10_5_schedule','build_5_10_5_source_mapping']:
            self.assertEqual(old[name],new[name],name)
        self.assertNotIn('register_auto_class',new)
        self.assertNotIn('.detach(',newpath.read_text())

    def test_actual_production_loop_with_numpy_matches_two_recurrence_equations(self):
        # Execute the actual model's loop AST with small NumPy activations.
        # This validates dispatch locally without pretending to run PyTorch.
        import numpy as np
        tree=ast.parse((ROOT/'code/RSmol/recursive_model_mizar_text_5_10x2_5_huginn.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MizarHuginnModel')
        forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        loop=next(n for n in forward.body if isinstance(n,ast.For))
        class Tensor(np.ndarray):
            @property
            def device(self): return 'cpu'
            def to(self,dtype): return self.astype(dtype).view(Tensor)
        def tensor(x): return np.asarray(x,dtype=np.float32).view(Tensor)
        weight=tensor(np.random.default_rng(41).normal(0,.2,(4,8)))
        embedding=tensor(np.random.default_rng(7).normal(0,.3,(2,3,4)))
        noise=tensor(np.random.default_rng(13).normal(0,C.STATE_STD,(2,3,4)))
        def layer(x,i): return tensor(np.tanh(x*(1+i*.02)+(i+1)*.01))
        def stack(x,indices):
            for i in indices: x=layer(x,i)
            return x
        e=stack(embedding,range(5))
        s1=stack(tensor(np.concatenate([noise,e],-1)@weight.T),range(5,15))
        s2=stack(tensor(np.concatenate([s1,e],-1)@weight.T),range(5,15))
        expected=stack(s2,range(15,20))
        for explicit in (False,True):
            inputs=[]; trace=[]; samples=[]
            def adapter(x):
                inputs.append(x.copy())
                return tensor(x@weight.T)
            def call(i,x,**kwargs):
                trace.append(i)
                return (layer(x,i),)
            def sample(x):
                samples.append(x.copy())
                return noise.copy()
            namespace=dict(self=types.SimpleNamespace(layers=list(range(20)),
                        logical_to_physical=C.SCHEDULE,recurrent_adapter=adapter),
                    hidden_states=embedding.copy(),initial_state=noise if explicit else None,
                    sample_initial_state=sample,torch=types.SimpleNamespace(
                        cat=lambda xs,dim:tensor(np.concatenate(xs,axis=dim))),
                    output_hidden_states=False,output_attentions=False,cache=None,
                    causal_mask=None,position_ids=None,use_cache=False,cache_position=None,
                    position_embeddings=None,_call_decoder_layer=call)
            exec(compile(ast.Module(body=[loop],type_ignores=[]),'<actual-model-loop>','exec'),namespace)
            np.testing.assert_array_equal(namespace['hidden_states'],expected)
            self.assertEqual(trace,list(C.SCHEDULE))
            self.assertEqual(len(samples),0 if explicit else 1)
            self.assertEqual(len(inputs),2)
            np.testing.assert_array_equal(inputs[0],np.concatenate([noise,e],-1))
            np.testing.assert_array_equal(inputs[1],np.concatenate([s1,e],-1))

    def test_formal_budget_and_final_saves(self):
        config = T._parse_args([])
        self.assertEqual((config.world_size, config.micro_batch_size, config.gradient_accumulation_steps), (8,8,16))
        self.assertEqual((config.max_optimizer_steps, config.scheduler_total_steps, config.warmup_steps), (9244,9244,463))
        self.assertEqual(T.DEFAULT_FORMAL_GLOBAL_SAMPLES, 9465856)
        self.assertEqual(T.formal_save_steps()[-3:], [8500,9000,9244])

    def test_both_epoch_lrs_and_default_min(self):
        for max_lr, min_lr in [('8e-4','8e-5'), ('2e-4','2e-5')]:
            for options in (['--max-lr',max_lr], ['--max-lr',max_lr,'--min-lr',min_lr]):
                config = T._parse_args(options)
                self.assertEqual(config.learning_rate, float(max_lr))
                self.assertAlmostEqual(config.min_lr, float(min_lr))
                self.assertEqual(config.scheduler_total_steps, 9244)

    def test_every_lr_point_matches_original_schedule(self):
        config = T._parse_args([])
        for max_lr in (8e-4,2e-4):
            values = [max_lr * T.cosine_warmup_factor(i,9244,463,max_lr=max_lr,min_lr=max_lr/10)
                      for i in range(9245)]
            self.assertAlmostEqual(values[0],max_lr/10)
            self.assertAlmostEqual(values[463],max_lr)
            self.assertAlmostEqual(values[-1],max_lr/10)
            self.assertTrue(all(a <= b for a,b in zip(values[:463],values[1:464])))
            self.assertTrue(all(a >= b for a,b in zip(values[463:],values[464:])))
            for i,value in enumerate(values):
                if i <= 463:
                    expected=max_lr/10 + (max_lr-max_lr/10)*i/463
                else:
                    expected=max_lr/10+(max_lr-max_lr/10)*0.5*(1+math.cos(math.pi*(i-463)/(9244-463)))
                self.assertAlmostEqual(value,expected,places=14)

    def test_bad_lr_and_conflicting_recipe(self):
        for args in (['--max-lr','nan'],['--min-lr','inf'],['--max-lr','-1'],
                     ['--max-lr','2e-4','--min-lr','8e-5'],['--micro-batch-size','4'],
                     ['--world-size','4'],['--scheduler-total-steps','18488'],['--seed','1']):
            with self.subTest(args=args), self.assertRaises(ValueError):
                T._parse_args(args)
        config=T._parse_args([])
        config.max_lr=float('nan')
        with self.assertRaises(ValueError):
            T.validate_formal_recipe(config)

    def test_resume_keeps_same_lr_and_rejects_route_change(self):
        for lr in ('8e-4','2e-4'):
            state=state_fixture(lr)
            self.assertEqual(T.validate_formal_resume_state(state,T._parse_args(['--max-lr',lr]))['optimizer_step'],500)
            other='2e-4' if lr=='8e-4' else '8e-4'
            with self.assertRaisesRegex(ValueError,'LR mismatch'):
                T.validate_formal_resume_state(state,T._parse_args(['--max-lr',other]))
            state['checkpoint_contract']['text_route']='mesh'
            with self.assertRaises(ValueError):
                T.validate_formal_resume_state(state)

    def test_resume_rejects_rank0_only_rng_or_wrong_noise_contract(self):
        for mutate in ('rng','noise'):
            state=state_fixture()
            if mutate=='rng': state['rng_states_by_rank']={'0':{}}
            else: state['checkpoint_contract']['huginn']={}
            with self.assertRaisesRegex(ValueError,'RNG contract'):
                T.validate_formal_resume_state(state)

    def test_weights_only_final_reads_no_training_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=model_fixture(Path(tmp)/'checkpoint-step-009244',9244)
            result=C.inspect_initialization(path)
            self.assertEqual(result['start_optimizer_step'],0)
            self.assertFalse(result['restore_optimizer_scheduler_rng_cursor'])
            self.assertEqual(result['fresh_seed'],0)
            self.assertEqual(C.inspect_initialization(path,resume=True)['start_optimizer_step'],9244)

    def test_partial_or_wrong_route_initialization_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=model_fixture(Path(tmp)/'checkpoint-step-000500',500)
            with self.assertRaisesRegex(ValueError,'final'):
                C.inspect_initialization(path)
            self.assertEqual(C.inspect_initialization(path,resume=True)['start_optimizer_step'],500)
            marker=json.loads((path/'checkpoint_complete.json').read_text())
            for key,value in [('text_route','other'),('world_size',4),('complete',False),
                              ('data_cursors_by_rank',['0']),('rng_states_by_rank',['0'])]:
                broken={**marker,key:value}
                V.write_json(path/'checkpoint_complete.json',broken)
                with self.assertRaises(ValueError):
                    C.inspect_initialization(path,resume=True)

    def test_conversion_and_training_sources_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=model_fixture(Path(tmp)/'converted')
            self.assertEqual(C.inspect_initialization(path)['mode'],'weights_only_new_epoch')
            with self.assertRaises(ValueError):
                C.inspect_initialization(path,resume=True)
            with self.assertRaises(FileExistsError):
                V.validate_output(Path(tmp)/'source',path)
            with self.assertRaises(ValueError):
                V.validate_output(Path(tmp),ROOT/'bad-model')

    def test_no_overwrite_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage=Path(tmp)/'stage'; stage.mkdir(); (stage/'ok').write_text('yes')
            out=Path(tmp)/'out'
            P.atomic_publish(stage,out)
            self.assertEqual((out/'ok').read_text(),'yes')
            stage.mkdir()
            with self.assertRaises(FileExistsError):
                P.atomic_publish(stage,out)
            self.assertTrue(stage.exists())

    def test_original_shard_order_and_direct_no_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'data'; data.mkdir()
            files=[data/name for name in T.EXPECTED_PARQUET_NAMES]
            for p in files: p.write_bytes(b'not parquet; footer audits must not run')
            config=T._parse_args(['--data-dir',str(Path(tmp))])
            with patch.object(T,'audit_parquet_shards',side_effect=AssertionError('no scan')):
                audit,manifest=T.build_training_data_manifest(config,world_size=8)
            expected=files.copy(); random.Random(0).shuffle(expected)
            self.assertEqual(manifest['shards_shuffled'],list(map(str,expected)))
            self.assertEqual(list(manifest['rank_shard_counts'].values()),[11]*5+[10]*3)
            self.assertEqual(audit['status'],'NOT_RUN')
            self.assertEqual(manifest['formal_global_samples'],9465856)

    def test_real_stream_starts_from_first_shard_and_drops_short_tails(self):
        reads=[]
        def texts(path,rng,buffer_size):
            self.assertEqual(buffer_size,4096)
            reads.append(path)
            return iter(['x']*(9 if path.name=='first' else 17))
        tokenizer=lambda text,**kwargs: {'input_ids':[3,4,5]}
        with patch.object(T,'_iter_shard_texts',side_effect=texts), \
             patch.object(T,'collate_dynamic_padding',side_effect=lambda rows,**kwargs:rows):
            stream=T.DistributedParquetStream([Path('first'),Path('second')],tokenizer,
                    rank=0,micro_batch_size=8,context_length=1024,pad_token_id=0,
                    seed=0,record_buffer_size=4096,device=None)
            self.assertEqual([len(batch) for batch in itertools.islice(iter(stream),3)],[8,8,8])
            self.assertEqual(reads,[Path('first'),Path('second')])
            self.assertEqual(stream.cursor()['samples_seen'],24)
            self.assertFalse(stream.cursor()['bounded_shuffle_bitwise_exact'])

    def test_actual_dry_run_requires_no_gpu_or_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=T._parse_args(['--dry-run','--output-dir',str(Path(tmp)/'dry'),
                                 '--model-path',str(Path(tmp)/'missing'),'--max-lr','2e-4','--min-lr','2e-5'])
            with patch.object(T,'_prepare_distributed',side_effect=AssertionError('No CUDA')):
                report=T.run(config)
            self.assertEqual(report['status'],'PASS')
            self.assertEqual(report['expected_global_effective_batch'],1024)
            self.assertEqual(report['scheduler']['total_steps_for_schedule'],9244)

    def test_actual_epoch2_start_does_not_restore_optimizer_rng_or_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            model=model_fixture(tmp/'checkpoint-step-009244',9244)
            data=tmp/'data'; data.mkdir()
            for name in T.EXPECTED_PARQUET_NAMES: (data/name).touch()
            config=T._parse_args(['--model-path',str(model),'--data-dir',str(data),
                                 '--output-dir',str(tmp/'epoch2'),'--max-lr','2e-4','--min-lr','2e-5'])
            dist=types.ModuleType('torch.distributed')
            dist.is_initialized=lambda: True
            dist.broadcast_object_list=Mock(); dist.barrier=Mock(); dist.destroy_process_group=Mock()
            torch=types.ModuleType('torch'); torch.distributed=dist
            torch.load=Mock(side_effect=AssertionError('weights-only must not read training state'))
            with patch.dict(sys.modules,{'torch':torch,'torch.distributed':dist}), \
                 patch.object(T,'_prepare_distributed',return_value=(0,8,'cuda:0',True)), \
                 patch.object(T,'_set_seed') as seed, \
                 patch.object(T,'_load_runtime_model',return_value=(object(),object())), \
                 patch.object(T,'_validate_recursive_architecture',return_value={}), \
                 patch.object(T,'recursive_forward_trace_audit',return_value={'forward_trace_ok':True}), \
                 patch.object(T,'audit_parquet_shards',side_effect=AssertionError('no scan')), \
                 patch.object(T,'_restore_training_state',side_effect=AssertionError('no restore')), \
                 patch.object(T,'_git_commit',return_value='local-test'), \
                 patch.object(T,'run_training',return_value={'status':'PASS','optimizer_steps':9244}) as train:
                result=T.run(config)
            train.assert_called_once()
            self.assertIsNone(train.call_args.kwargs['resume_state'])
            self.assertEqual(config.initialization['start_optimizer_step'],0)
            self.assertEqual(config.initialization['source_training_step'],9244)
            self.assertEqual(result['dataset_audit']['status'],'NOT_RUN')
            seed.assert_called()
            phases=[json.loads(line)['phase'] for line in (tmp/'epoch2/stage4_lifecycle_rank0.jsonl').read_text().splitlines()]
            self.assertLess(phases.index('forward_trace_done'),phases.index('training_start'))


if __name__=='__main__':
    unittest.main()
