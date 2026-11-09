"""Behavioral contracts, source isolation and data/resume regression checks."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from text_x3_test_support import BASE, ROOT, ROUTES, SCHEDULE, SOURCE_MAPPING, helpers, model_constants


class RouteContracts(unittest.TestCase):
    def test_attention_cache_and_mesh_math_preserved_from_mothers(self):
        def definitions(path):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            return {node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        pure=definitions(BASE/'recursive_model_5_10x3_5.py')
        old=definitions(BASE/'recursive_model_5_10_5.py')
        for name in ('LogicalSlotCacheView','_make_cache','_cache_seq_length','_causal_mask','_call_decoder_layer'):
            self.assertEqual(ast.dump(pure[name]),ast.dump(old[name]),name)
        mesh=definitions(BASE/'recursive_model_5_10x3_5_mesh_6slot.py')
        old=definitions(BASE/'recursive_model_5_10x3_5_mesh_7slot.py')
        for name in ('LogicalSlotCacheView','_init_router','_call_layer','_causal_mask'):
            self.assertEqual(ast.dump(mesh[name]),ast.dump(old[name]),name)
        actual={n.name:n for n in mesh['MeshLlamaModel'].body if isinstance(n,ast.FunctionDef)}
        expected={n.name:n for n in old['MeshLlamaModel'].body if isinstance(n,ast.FunctionDef)}
        for name in ('_route','_write','_read','_run_stack','forward'):
            self.assertEqual(ast.dump(actual[name]),ast.dump(expected[name]),name)

    def test_sources_unchanged(self):
        for route in ROUTES:
            guard = json.loads((BASE / f'text_{route}/source_guard.json').read_text())
            self.assertEqual(len(guard), 10)
            for path, expected in guard.items():
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), expected, path)

    def test_schedule_and_router_geometry(self):
        for route in ROUTES:
            c = model_constants(route)
            self.assertEqual(list(c['LOGICAL_TO_PHYSICAL']), SCHEDULE)
            self.assertEqual(c['SOURCE_LAYER_INDICES_0BASED'], SOURCE_MAPPING)
            self.assertEqual((c['LOGICAL_LAYER_COUNT'], c['PHYSICAL_LAYER_COUNT'], c['RECURSIVE_LOOPS']), (40, 20, 3))
            self.assertEqual([SCHEDULE.count(i) for i in range(20)], [1]*5+[3]*10+[1]*5)
            self.assertEqual(c['MEMORY_SLOT_COUNT'], 6 if 'mesh' in route else 0)
            self.assertEqual(c['ROUTER_PARAMETER_COUNT'], 8 if 'mesh' in route else 0)
        self.assertEqual(8 * (576*6 + 6), 27696)

    def test_same_actual_data_and_update_helpers(self):
        def definitions(route):
            tree = ast.parse((BASE / f'scripts/train_stage4_{route}_ddp.py').read_text())
            return {node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        pure, mesh = [definitions(route) for route in ROUTES]
        mother_tree=ast.parse((BASE/'scripts/train_stage4_5_10x2_5_mesh_7slot_v2_ddp.py').read_text())
        mother={node.name:ast.dump(node,include_attributes=False) for node in mother_tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        for name in ('DistributedParquetStream', '_cosine_lr', 'token_weighted_gradient_scale', '_configure_precision', '_autocast_context', '_capture_rng_state', '_restore_rng_state', '_manifest', '_seed'):
            self.assertEqual(pure[name], mesh[name], name)
            self.assertEqual(pure[name], mother[name], name)
        # Compare the real accumulation/update statements through optimizer.step.
        texts = [(BASE / f'scripts/train_stage4_{route}_ddp.py').read_text() for route in ROUTES]
        blocks = [text[text.index('            optimizer.zero_grad'):text.index('            routing_owner =') if 'routing_owner =' in text else text.index('            router_stats = {}')] for text in texts]
        blocks = [block.replace('RSMOL_'+route.upper(), 'ROUTE') for route, block in zip(ROUTES, blocks)]
        self.assertEqual(blocks[0], blocks[1])
        self.assertIn('(loss_sum / config.gradient_accumulation_steps).backward()', blocks[0])
        self.assertIn('parameter.grad.mul_(scale)', blocks[0])

    def test_formal_arguments_and_wrong_geometry_rejection(self):
        for route in ROUTES:
            m = helpers(route)
            config = m._parse_args([])
            self.assertEqual((config.gate, config.world_size, config.micro_batch_size, config.gradient_accumulation_steps), ('FORMAL', 8, 4, 32))
            self.assertEqual((config.max_optimizer_steps, config.scheduler_total_steps, config.warmup_steps), (9244, 9244, 463))
            for extra in (['--micro-batch-size','8'], ['--context-length','512'], ['--max-optimizer-steps','10'], ['--max-lr','0.001'], ['--save-every','100'], ['--max-microbatches','2'], ['--seed','1'], ['--tokenizer-path','/wrong_tokenizer']):
                with self.assertRaises(ValueError, msg=(route, extra)):
                    m._parse_args(extra)

    def test_full_lr_curve_and_token_scale(self):
        for route in ROUTES:
            m = helpers(route)
            c = m._parse_args([])
            values = [m._cosine_lr(i, c) for i in range(9245)]
            self.assertAlmostEqual(values[0], 8e-4/463)
            self.assertAlmostEqual(values[462], 8e-4)
            self.assertAlmostEqual(values[463], 8e-4)
            self.assertAlmostEqual(values[9244], 8e-5)
            self.assertGreater(values[9243], 8e-5)
            self.assertTrue(all(a <= b for a,b in zip(values[:463], values[1:463])))
            self.assertTrue(all(a >= b for a,b in zip(values[463:], values[464:])) )
            for tokens in (1, 77, 123456):
                self.assertEqual(m.token_weighted_gradient_scale(world_size=8, global_window_tokens=tokens, gradient_accumulation_steps=32), 256/tokens)
            with self.assertRaises(ValueError):
                m.token_weighted_gradient_scale(world_size=8, global_window_tokens=0, gradient_accumulation_steps=32)

    def test_source_detection_original_clean20_and_reject_trained_or_other_route(self):
        for route in ROUTES:
            m = helpers(route, converter=True)
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)
                def config(value):
                    (p/'config.json').write_text(json.dumps(value))
                config({'num_hidden_layers': 30})
                self.assertEqual(m.detect_source(p)['mapping'], SOURCE_MAPPING)
                config({'num_hidden_layers':30,'recursive_layer_count':20,'recursive_loops':2,'architectures':['RecursiveLlamaForCausalLM']})
                (p/'recursive_conversion_metadata.json').write_text(json.dumps({'conversion':'SmolLM2-5-10-5','status':'ok'}))
                self.assertEqual(m.detect_source(p)['mapping'], tuple(range(20)))
                (p/'training_state.pt').touch()
                with self.assertRaises(ValueError): m.detect_source(p)
                (p/'training_state.pt').unlink()
                for wrong in ({'num_hidden_layers':40,'recursive_layer_count':20,'recursive_loops':3}, {'num_hidden_layers':30,'mesh_memory_slots':6}, {'num_hidden_layers':30,'recursive_architecture_contract':'other'}):
                    config(wrong)
                    with self.assertRaises(ValueError): m.detect_source(p)

    def test_fresh_route_provenance_and_cross_route_rejection(self):
        for route in ROUTES:
            m = helpers(route)
            groups = 4 if 'mesh' in route else 0
            routers = sorted(f'model.{kind}_routers.{i}.{part}' for kind in ('write','read') for i in range(groups) for part in ('weight','bias'))
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)
                name = 'mesh_conversion_metadata.json' if groups else 'recursive_conversion_metadata.json'
                metadata = dict(status='ok', architecture_contract=m.MODEL_ARCHITECTURE_CONTRACT, memory_slots=m.MEMORY_SLOT_COUNT, router_groups=groups, inherited_weights_exact=True, source_training_markers=[], fresh_router_tensors=routers, save_reload_tensors_exact=True, save_reload_logits_exact=True, tokenizer_roundtrip_exact=True, forward_trace=SCHEDULE)
                (p/name).write_text(json.dumps(metadata))
                m._validate_fresh_model(p)
                for key, value in (('router_groups',3), ('architecture_contract','other'), ('save_reload_logits_exact',False), ('fresh_router_tensors',routers[1:] if routers else ['bad'])):
                    bad = dict(metadata, **{key:value})
                    (p/name).write_text(json.dumps(bad))
                    with self.assertRaises(ValueError): m._validate_fresh_model(p)

    def test_resume_geometry_precision_and_route_contract(self):
        for route in ROUTES:
            m = helpers(route)
            c = m._parse_args([])
            state = dict(training_contract=m.TRAINING_CONTRACT, precision_policy=m.PRECISION_POLICY, configuration=asdict(c), optimizer_step=500,
                         scheduler=dict(type='cosine_warmup',step=500,total_steps=9244,warmup_steps=463,max_lr=c.max_lr,min_lr=c.min_lr))
            m._validate_resume_contract(state, c)
            for key, value in (('gate','D'), ('micro_batch_size',8), ('seed',1), ('world_size',4), ('warmup_steps',1), ('max_optimizer_steps',10)):
                bad = copy.deepcopy(state)
                bad['configuration'][key] = value
                with self.assertRaises(ValueError): m._validate_resume_contract(bad,c)
            for key, value in (('training_contract','other_route'), ('precision_policy',{}), ('optimizer_step',9245)):
                bad = dict(state, **{key:value})
                with self.assertRaises(ValueError): m._validate_resume_contract(bad,c)

    def test_actual_stream_cursor_skips_consumed_rows(self):
        class Tensor:
            def long(self): return self
            def ne(self, value): return self
            def clone(self): return self
            def bool(self): return self
            def __eq__(self, value): return False
            def __setitem__(self, key, value): pass
        class Batch:
            def __init__(self, rows): self.rows, self.num_rows = rows, len(rows)
            def column(self, name): return types.SimpleNamespace(to_pylist=lambda:self.rows)
        class Parquet:
            def __init__(self, path): pass
            def iter_batches(self, batch_size, **kwargs):
                for i in range(0, 9, batch_size): yield Batch([f'row{j}' for j in range(i,min(i+batch_size,9))])
        fake = types.ModuleType('pyarrow.parquet'); fake.ParquetFile = Parquet
        parent = types.ModuleType('pyarrow'); parent.parquet = fake
        for route in ROUTES:
            m = helpers(route)
            seen = []
            def tokenizer(texts, **kwargs):
                seen.append(texts)
                return {'input_ids':Tensor(), 'attention_mask':Tensor()}
            def stream(): return m.DistributedParquetStream([Path(f'{i:02}.parquet') for i in range(85)],tokenizer,rank=3,world_size=8,batch_size=4,context_length=1024,pad_token_id=0)
            with patch.dict('sys.modules',{'pyarrow':parent,'pyarrow.parquet':fake}):
                original = stream(); it = iter(original)
                next(it); next(it)
                cursor = original.cursor()
                self.assertEqual(cursor['row_offset'],8)
                self.assertEqual(cursor['shard_order'], [f'{i:02}.parquet' for i in range(3,85,8)])
                resumed = stream(); resumed.restore_cursor(cursor)
                next(iter(resumed))
                self.assertEqual(seen[-1],['row8'])
                self.assertEqual(resumed.cursor()['microbatches_seen'],3)
                wrong = dict(cursor,row_offset=3)
                resumed.restore_cursor(wrong)
                with self.assertRaises(ValueError): next(iter(resumed))
                with self.assertRaises(ValueError): resumed.restore_cursor(dict(cursor,rank=0))

    def test_output_guard_is_direct_python_behavior(self):
        for route in ROUTES:
            m = helpers(route)
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                c=m.Stage4Config(output_dir=root/'new',model_path=root/'model',data_dir=root/'data')
                m._prepare_output(c,0,1)
                (c.output_dir/'keep').touch()
                with self.assertRaises(ValueError): m._prepare_output(c,0,1)
                self.assertTrue((c.output_dir/'keep').exists())
                c.output_dir=root/'model'
                with self.assertRaises(ValueError): m._prepare_output(c,0,1)

    def test_optimizer_all_trainable_groups_and_router_membership(self):
        class Param:
            requires_grad=True
            def __init__(self,ndim): self.ndim=ndim
        for route in ROUTES:
            m=helpers(route)
            pairs=[('model.layers.0.weight',Param(2)),('model.norm.weight',Param(1))]
            if 'mesh' in route:
                pairs += [(f'model.{kind}_routers.{i}.{part}',Param(2 if part=='weight' else 1)) for kind in ('write','read') for i in range(4) for part in ('weight','bias')]
            captured={}
            def adam(groups,**kwargs):
                captured.update(groups=groups,kwargs=kwargs)
                return object()
            m.torch.optim=types.SimpleNamespace(AdamW=adam)
            _,audit=m._optimizer(types.SimpleNamespace(named_parameters=lambda:iter(pairs)),m.Stage4Config())
            self.assertEqual([g['weight_decay'] for g in captured['groups']],[0.1,0.0])
            self.assertEqual({id(p) for g in captured['groups'] for p in g['params']},{id(p) for _,p in pairs})
            self.assertEqual(len(audit['router_parameters']),16 if 'mesh' in route else 0)
            self.assertEqual(captured['kwargs']['betas'],(0.9,0.95))


if __name__ == '__main__': unittest.main()
