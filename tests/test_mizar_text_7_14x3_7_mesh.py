"""Dependency-free contract, source rejection, data replay and recipe checks."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / 'code/RSmol'
sys.path.insert(0, str(RSMOL))
from mizar_text_7_14x3_7_mesh import contract as C, conversion as V, data as D, recipe as R
from mizar_text_7_14x3_7_mesh.publication import atomic_publish


def recipe(**kwargs):
    values = dict(gate='FORMAL', world_size=8, micro_batch_size=8,
                  gradient_accumulation_steps=16, max_optimizer_steps=9244,
                  scheduler_total_steps=9244, warmup_steps=463, save_every=500,
                  context_length=1024, checkpoint_retention=3, max_lr=8e-4,
                  min_lr=8e-5, steps_per_epoch=9244, epochs=1, max_microbatches=None)
    return SimpleNamespace(**{**values, **kwargs})


class ContractTests(unittest.TestCase):
    def test_historical_copy_snapshot_and_independent_helpers(self):
        guard = json.loads((RSMOL/'mizar_text_7_14x3_7_mesh/source_guard.json').read_text(encoding='utf-8'))
        self.assertEqual(len(guard), 16)
        for name, digest in guard.items():
            self.assertIn('7_14x2_7_mesh',name)
            self.assertRegex(digest,r'^[0-9a-f]{64}$')
        # The two-loop route can evolve independently after the copy (e.g. its LR).
        # Verify the three-loop shared helpers against their original snapshot.
        for helper in ('data.py','publication.py'):
            name = 'code/RSmol/mizar_text_7_14x2_7_mesh/'+helper
            # Git may convert checkout line endings on another host.
            content = (RSMOL/'mizar_text_7_14x3_7_mesh'/helper).read_bytes().replace(b'\r\n', b'\n')
            self.assertEqual(hashlib.sha256(content).hexdigest(), guard[name], helper)

    def test_confirmed_mapping_and_full_execution(self):
        self.assertEqual([i+1 for i in C.SOURCE_LAYERS[:7]], [1,2,4,5,6,7,8])
        self.assertEqual([i+1 for i in C.SOURCE_LAYERS[7:21]], list(range(9,18))+list(range(19,24)))
        self.assertEqual([i+1 for i in C.SOURCE_LAYERS[21:]], list(range(24,31)))
        self.assertEqual(len(set(C.SOURCE_LAYERS)), 28)
        self.assertEqual(len(C.SCHEDULE), 56)
        self.assertEqual(C.SCHEDULE, tuple(range(7))+tuple(range(7,21))*3+tuple(range(21,28)))
        self.assertEqual([C.SCHEDULE.count(i) for i in range(28)], [1]*7+[3]*14+[1]*7)
        self.assertEqual(len(C.ROUTER_KEYS), 16)
        self.assertEqual(C.ROUTER_PARAMETER_COUNT, 8*(576*6+6))

    def test_config_aliases_and_wrong_routes_fail_closed(self):
        config = SimpleNamespace(num_hidden_layers=30, hidden_size=576, vocab_size=49152,
                                 tie_word_embeddings=True, model_type='llama')
        target = C.build_target_config(config)
        C.validate_model_config(vars(target))
        self.assertEqual(config.num_hidden_layers, 30)
        for key in C.architecture_fields():
            bad = copy.deepcopy(vars(target))
            bad[key] = None
            with self.assertRaisesRegex(ValueError, key):
                C.validate_model_config(bad)

    def test_two_and_three_loop_configs_reject_each_other(self):
        from mizar_text_7_14x2_7_mesh import contract as two
        for owner, other in ((C, two), (two, C)):
            raw = {**other.architecture_fields(), 'hidden_size':576, 'vocab_size':49152,
                   'model_type':'llama', 'tie_word_embeddings':True}
            with self.assertRaisesRegex(ValueError, 'config mismatch'):
                owner.validate_model_config(raw)

    def test_official_final_source_and_layer_key_mapping(self):
        identity = C.source_identity()
        self.assertEqual(identity['checkpoint_relative_path'], 'checkpoints/final/stage3_q4_seed20260905.ckpt')
        self.assertEqual(identity['checkpoint_sha256'], '0c542f63e11ed55d360feb0d66e7be0e8c6a878f44cffd302a92388f915388a2')
        for index, source in enumerate(C.SOURCE_LAYERS):
            self.assertEqual(V.source_key(f'model.layers.{index}.mlp.gate_proj.weight'),
                             f'model.layers.{source}.mlp.gate_proj.weight')
        self.assertEqual(V.source_key('model.norm.weight'), 'model.norm.weight')

    def test_fresh_json_identity_rejects_wrong_initialization_and_training_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            raw = {**C.architecture_fields(), 'hidden_size':576, 'vocab_size':49152,
                   'model_type':'llama', 'tie_word_embeddings':True}
            marker = {'complete':True, 'text_route':C.ROUTE, 'source':C.source_identity(),
                      'physical_layers':28, 'logical_layers':56}
            metadata = {
                'status':'PASS', 'text_route':C.ROUTE, 'architecture_contract':C.ARCHITECTURE,
                'source':C.source_identity(), 'inherited_weights_exact':True,
                'final_lm_strict_load':True, 'reload_tensors_logits_tokenizer_equal':True,
                'router_initialization':C.ROUTER_INITIALIZATION, 'router_seed':0,
                'physical_layers':28, 'logical_layers':56, 'memory_slots':6, 'router_groups':4,
                'source_layer_indices_0based':list(C.SOURCE_LAYERS), 'fresh_router_tensors':sorted(C.ROUTER_KEYS),
            }
            for name,value in (('config.json',raw),('conversion_complete.json',marker),
                               ('recursive_conversion_metadata.json',metadata)):
                (folder/name).write_text(json.dumps(value),encoding='utf-8')
            # This helper reads identity only; actual tensor loading is a separate check.
            (folder/'model.safetensors').touch(); (folder/'tokenizer.json').write_text('{}')
            self.assertEqual(C.inspect_initialization(folder)['start_optimizer_step'],0)
            metadata['router_initialization'] = {'distribution':'normal_then_clamp'}
            (folder/'recursive_conversion_metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'initialization'):
                C.inspect_initialization(folder)
            (folder/'training_state.pt').touch()
            with self.assertRaisesRegex(ValueError,'resume-from'):
                C.inspect_initialization(folder)

    def test_source_assets_exclude_base_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [{'path': name, 'bytes': len(value), 'sha256': ''} for name, value in (
                ('checkpoints/final/final.ckpt', b'final'),
                ('assets/smollm2-135m/config.json', b'config'))]
            for item in records:
                target = root/item['path']; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'final' if target.suffix == '.ckpt' else b'config')
                item['sha256'] = C.sha256_file(target)
            release = {'checkpoint': records[0], 'files': [records[1], {'path':'assets/smollm2-135m/model.safetensors'}]}
            with patch.object(V, 'release_manifest', return_value=release):
                self.assertEqual(len(V.verify_source_files(root)), 2)
                (root/records[0]['path']).write_bytes(b'wrong')
                with self.assertRaises(ValueError):
                    V.verify_source_files(root)

    def test_no_overwrite_publication_and_code_output_rejection(self):
        with self.assertRaises(ValueError):
            V.validate_output(Path('/source'), ROOT/'new-model')
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); staging = parent/'staging'; staging.mkdir()
            (staging/'complete').write_text('yes')
            output = parent/'output'
            atomic_publish(staging, output)
            self.assertEqual((output/'complete').read_text(), 'yes')
            other = parent/'other'; other.mkdir()
            with self.assertRaises(FileExistsError):
                atomic_publish(other, output)

    def test_formal_batch_and_scheduler_contract(self):
        R.validate_recipe(recipe())
        R.validate_recipe(recipe(micro_batch_size=4, gradient_accumulation_steps=32))
        for overrides in ({'micro_batch_size':4}, {'max_optimizer_steps':3081},
                          {'world_size':1}, {'max_lr':2e-4}, {'context_length':2048}):
            with self.assertRaises(ValueError):
                R.validate_recipe(recipe(**overrides))
        cfg = recipe()
        rates = [R.learning_rate(step, cfg) for step in range(9245)]
        self.assertEqual(rates[0], 8e-5)
        self.assertAlmostEqual(rates[463], 8e-4)
        self.assertEqual(rates[9244], 8e-5)
        self.assertTrue(all(a <= b for a,b in zip(rates[:463], rates[1:464])))
        self.assertTrue(all(a >= b for a,b in zip(rates[463:], rates[464:])))
        # Analytic DDP-average gradient, for unequal token populations.
        counts = [1,3,8,2,5,9,4,6]
        grad_sums = [2,-1,12,3,7,5,-4,6]
        scaled = sum(g*R.gradient_scale(8,sum(counts)) for g in grad_sums)/8
        self.assertAlmostEqual(scaled, sum(grad_sums)/sum(counts))

    def test_all_new_python_parses_and_has_no_old_route_import(self):
        paths = list((RSMOL/'mizar_text_7_14x3_7_mesh').glob('*.py'))
        paths += [RSMOL/'recursive_model_mizar_text_7_14x3_7_mesh.py']
        paths += list((RSMOL/'scripts').glob('*mizar_text_7_14x3_7_mesh*.py'))
        for path in paths:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            compile(tree, str(path), 'exec')
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn('5_10', node.module or '', str(path))
                    self.assertNotIn('7_14x2_7', node.module or '', str(path))
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn('7_14x2_7', alias.name, str(path))
                if isinstance(node, ast.Call):
                    self.assertNotEqual(getattr(node.func, 'attr', None), 'register')


class DataTests(unittest.TestCase):
    def test_manifest_and_rank_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'data'; folder.mkdir()
            for i in range(85):
                (folder/f'{i:03d}.parquet').write_bytes(b'fixture')
            paths = D.discover_shards(Path(tmp))
            assigned = D.assign_shards(paths, 8, 0)
            self.assertEqual(list(map(len, assigned)), [11]*5+[10]*3)
            self.assertEqual(set(sum(assigned, [])), set(paths))
            self.assertEqual(len(D.manifest_records(paths)), 85)
            (folder/'084.parquet').unlink()
            with self.assertRaises(ValueError):
                D.discover_shards(Path(tmp))

    def test_resume_replays_real_shuffle_buffer_across_shards_and_rollover(self):
        # Stub parquet I/O only: the actual shuffle/tokenize/cursor loop runs.
        parquet = ModuleType('pyarrow.parquet')
        values = {f'{i}.parquet':[None,'']+[str(i*100+j) for j in range(31)] for i in range(4)}
        class ParquetFile:
            def __init__(self, path): self.path = path
            def iter_batches(self, **kwargs):
                rows = values[self.path.name]
                for pos in range(0, len(rows), 4):
                    yield SimpleNamespace(column=lambda _name, chunk=rows[pos:pos+4]: SimpleNamespace(to_pylist=lambda:chunk))
        parquet.ParquetFile = ParquetFile
        arrow = ModuleType('pyarrow'); arrow.parquet = parquet
        def tokenizer(text, **kwargs):
            self.assertEqual(kwargs, {'add_special_tokens':False, 'truncation':True, 'max_length':16})
            return {'input_ids':[int(text)+1, 1]}
        def new_stream():
            return D.DistributedParquetStream([Path(f'{i}.parquet') for i in range(4)], tokenizer,
                    rank=1, world_size=2, batch_size=3, context_length=16, pad_token_id=0, buffer_size=5)
        with patch.dict(sys.modules, {'pyarrow':arrow, 'pyarrow.parquet':parquet}), patch.object(D, 'collate', side_effect=lambda rows,pad:rows):
            stream = new_stream(); iterator = iter(stream)
            for _ in range(7): next(iterator)
            cursor = json.loads(json.dumps(stream.cursor()))
            expected = [next(iterator) for _ in range(47)]
            resumed = new_stream(); resumed.restore_cursor(cursor)
            restarted = iter(resumed)
            self.assertEqual([next(restarted) for _ in range(47)], expected)
            self.assertEqual(resumed.cursor(), stream.cursor())
            changed = new_stream(); changed.batch_size = 4
            with self.assertRaisesRegex(ValueError, 'batch_size'):
                changed.restore_cursor(cursor)


if __name__ == '__main__':
    unittest.main()
