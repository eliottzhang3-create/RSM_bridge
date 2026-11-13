"""Checkpoint identity and fresh audio state; synthetic fixtures, no production assets."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from mizar_text_x3_support import ROOT, ROUTE, ARCH, SCHEDULE, final_text_metadata

sys.path.insert(0, str(ROUTE))
import audit_reference as audit
import contract
from models import recursive_text

PARENT = ROUTE.with_name('mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8')


def config():
    return dict(model_type='llama', architectures=['RecursiveLlama5_10x3_5ForCausalLM'],
                hidden_size=576, num_hidden_layers=40, recursive_layer_count=20,
                recursive_loops=3, recursive_prefix_layer_count=5,
                recursive_middle_layer_count=10, recursive_suffix_layer_count=5,
                recursive_loops_scope='middle_only', vocab_size=49152, eos_token_id=0,
                recursive_architecture_contract=ARCH, logical_to_physical=SCHEDULE,
                recursive_source_layer_indices_0based=list(contract.TEXT_MODEL_SOURCE_MAPPING))


def checkpoint(root):
    path = root / 'checkpoint-009244'
    path.mkdir()
    contract.write_json(path / 'config.json', config())
    (path / 'tokenizer.json').write_text('{}')
    (path / 'model.safetensors').write_bytes(b'synthetic inventory; not tensor weights')
    (path / 'training_state.pt').write_bytes(b'POISON: audio must never deserialize text training state')
    final_text_metadata(path)
    return path


class IdentityTests(unittest.TestCase):
    def test_accepts_config_generated_by_actual_pure_x3_converter(self):
        from text_x3_test_support import helpers
        converter = helpers('5_10x3_5', converter=True)
        source = NS(num_hidden_layers=30, hidden_size=576, vocab_size=49152,
                    tie_word_embeddings=True, model_type='llama', eos_token_id=0, pad_token_id=0)
        target = converter._target_config(source, {}, Path('/synthetic/source'))
        self.assertEqual(contract.validate_recursive_config_values(vars(target))['recursive_loops'], 3)
        self.assertEqual(target.logical_to_physical, SCHEDULE)

    def test_actual_x2_to_x3_conversion_reproduces_and_audits_known_stale_alias(self):
        from text_x3_test_support import helpers
        original = NS(num_hidden_layers=30, hidden_size=576, vocab_size=49152,
                      tie_word_embeddings=True, model_type='llama', eos_token_id=0, pad_token_id=0)
        x2 = helpers('5_10_5', converter=True).build_target_config(original)
        self.assertEqual(x2.recursive_logical_to_physical_schedule, list(contract.LEGACY_X2_SCHEDULE))
        x3 = helpers('5_10x3_5', converter=True)._target_config(x2, {}, Path('/synthetic/clean20'))
        self.assertEqual(x3.logical_to_physical, SCHEDULE)
        self.assertEqual(x3.recursive_logical_to_physical_schedule, list(contract.LEGACY_X2_SCHEDULE))
        self.assertEqual(contract.validate_recursive_config_values(vars(x3))['logical_layers'], 40)
        record = contract.recursive_config_compatibility(vars(x3))
        self.assertEqual(record['overrides'][contract.LEGACY_SCHEDULE_ALIAS]['canonical'], SCHEDULE)
        self.assertFalse(record['source_checkpoint_modified'])
        # Validation must not mutate the source dict or silently substitute execution.
        self.assertEqual(x3.recursive_logical_to_physical_schedule, list(contract.LEGACY_X2_SCHEDULE))

    def test_only_known_unused_alias_is_accepted_effective_x3_fields_stay_strict(self):
        legacy = dict(config(), **{contract.LEGACY_SCHEDULE_ALIAS: list(contract.LEGACY_X2_SCHEDULE)})
        contract.validate_recursive_config_values(legacy)
        for field, wrong in (('logical_to_physical', list(contract.LEGACY_X2_SCHEDULE)),
                             ('logical_to_physical_schedule', list(contract.LEGACY_X2_SCHEDULE)),
                             ('recursive_logical_to_physical', list(contract.LEGACY_X2_SCHEDULE)),
                             (contract.LEGACY_SCHEDULE_ALIAS, list(range(30))),
                             (contract.LEGACY_SCHEDULE_ALIAS, None),
                             ('recursive_loops', 2), ('mesh_memory_slots', 6),
                             ('recursive_architecture_contract', 'other')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.validate_recursive_config_values(dict(legacy, **{field: wrong}))
        missing_primary = dict(legacy, logical_to_physical_schedule=SCHEDULE)
        missing_primary.pop('logical_to_physical')
        with self.assertRaises(ValueError): contract.validate_recursive_config_values(missing_primary)
        canonical = dict(config(), **{contract.LEGACY_SCHEDULE_ALIAS: SCHEDULE})
        self.assertEqual(contract.recursive_config_compatibility(canonical)['overrides'], {})

    def test_existing_checkpoint_contract_records_alias_without_rewriting_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory))
            contract.write_json(path/'config.json', dict(config(), **{
                contract.LEGACY_SCHEDULE_ALIAS: list(contract.LEGACY_X2_SCHEDULE)}))
            before = {p.name: p.read_bytes() for p in path.iterdir()}
            record = contract.recursive_checkpoint_contract(path)
            self.assertEqual(record['config']['logical_to_physical'], SCHEDULE)
            overrides = record['schedule_metadata_compatibility']['overrides']
            self.assertEqual(overrides[contract.LEGACY_SCHEDULE_ALIAS]['source'], list(contract.LEGACY_X2_SCHEDULE))
            self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})

    def test_loaded_model_normalizes_metadata_after_actual_schedule_and_module_audit(self):
        class Config(NS):
            def to_dict(self): return vars(self).copy()
        cfg = Config(**dict(config(), **{contract.LEGACY_SCHEDULE_ALIAS: list(contract.LEGACY_X2_SCHEDULE)}))
        model = NS(config=cfg, model=NS(layers=[object() for _ in range(20)], logical_to_physical=SCHEDULE),
                   named_parameters=lambda: [], named_buffers=lambda: [])
        loaded_audit = dict(logical_layer_count=40, physical_layer_count=20, recursive_layer_count=20,
                            recursive_loops=3, middle_recurrent_count=10, model_physical_module_count=20,
                            schedule=SCHEDULE)
        source_cfg = cfg.to_dict()
        model.model.logical_to_physical = list(contract.LEGACY_X2_SCHEDULE)
        with self.assertRaises(ValueError): recursive_text.validate_loaded_model(model, loaded_audit)
        self.assertEqual(cfg.to_dict(), source_cfg)
        model.model.logical_to_physical = SCHEDULE
        recursive_text.validate_loaded_model(model, loaded_audit)
        self.assertEqual(cfg.recursive_logical_to_physical_schedule, SCHEDULE)
        self.assertEqual(model.recursive_config_compatibility['overrides'][contract.LEGACY_SCHEDULE_ALIAS]['source'],
                         list(contract.LEGACY_X2_SCHEDULE))

    def test_exact_source_and_pure_x3_geometry_rejects_other_models(self):
        self.assertEqual(contract.DEFAULT_TEXT_MODEL_DIR,
            '/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x3_5/formal_1epoch_bf16_20261109_153922858684473/checkpoint-009244')
        self.assertEqual(contract.validate_recursive_config_values(config())['logical_layers'], 40)
        for key, wrong in (('num_hidden_layers', 30), ('recursive_loops', 2),
                           ('mesh_enabled', True), ('mesh_memory_slots', 6),
                           ('recursive_architecture_contract', 'mesh'),
                           ('logical_to_physical', SCHEDULE[:30]),
                           ('recursive_source_layer_indices_0based', list(range(20))),
                           ('architectures', ['RecursiveLlama5_10x3_5MeshForCausalLM'])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                contract.validate_recursive_config_values(dict(config(), **{key: wrong}))

    def test_text_metadata_only_does_not_deserialize_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory))
            record = contract.recursive_checkpoint_contract(path)
            self.assertEqual(record['weight_files'], ['model.safetensors'])
            self.assertEqual(record['provenance']['optimizer_step'], 9244)
            self.assertFalse(record['provenance']['load_text_training_state'])
            self.assertEqual(record['config']['logical_to_physical'], SCHEDULE)

    def test_incomplete_wrong_step_or_mesh_text_metadata_is_rejected(self):
        cases = [('recursive_checkpoint_metadata.json', 'optimizer_step', 9243),
                 ('recursive_checkpoint_metadata.json', 'training_contract', 'mesh'),
                 ('recursive_checkpoint_metadata.json', 'router_groups', 4),
                 ('recursive_checkpoint_metadata.json', 'memory_slots', 6),
                 ('checkpoint_complete.json', 'complete_marker', False),
                 ('checkpoint_manifest.json', 'status', 'writing'),
                 ('checkpoint_manifest.json', 'files', ['../escape']),
                 ('checkpoint_manifest.json', 'files', ['missing.safetensors'])]
        for name, key, value in cases:
            with self.subTest(name=name, key=key), tempfile.TemporaryDirectory() as directory:
                path = checkpoint(Path(directory))
                original = json.loads((path / name).read_text())
                contract.write_json(path / name, dict(original, **{key: value}))
                with self.assertRaises(ValueError): contract.recursive_checkpoint_contract(path)

    def test_text_strict_loader_rejects_partial_weights_before_registering_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory))
            cls = Mock()
            cls.from_pretrained.return_value = (object(), {'missing_keys': ['model.layers.5.mlp.up_proj.weight']})
            with patch.object(recursive_text, '_import_recursive_model', return_value=(cls, Mock())):
                with self.assertRaisesRegex(ValueError, 'weights do not match'):
                    recursive_text.load_recursive_text_model(path)
            cls.from_pretrained.assert_called_once_with(path.resolve(), local_files_only=True, output_loading_info=True)

    def test_every_lr_point_and_gradnorm_policy_matches_reference(self):
        spec = importlib.util.spec_from_file_location('x3_reference_contract', PARENT / 'contract.py')
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for step in range(17314):
            self.assertEqual(contract.lr_factor(step, 17313), old.lr_factor(step, 17313))
        scopes = []
        for route in (PARENT, ROUTE):
            scope = {}
            audit.execute([audit.find_function(audit.tree(route / 'optim.py'), '_clip_grad')], scope)
            scopes.append(scope)
        rng = random.Random(1234)
        tracker = NS(overdrive_factor=2.5, momentum=.995)
        for _ in range(1000):
            args = tracker, 10**rng.uniform(-8, 8), 10**rng.uniform(-8, 8), rng.random()
            self.assertEqual(scopes[0]['_clip_grad'](*args), scopes[1]['_clip_grad'](*args))

    def test_fresh_audio_initialization_starts_at_zero_and_creates_all_new_state(self):
        body = audit.find_function(audit.tree(ROUTE / 'train.py'), 'train').body
        start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                     and ast.unparse(n).startswith('optimizer, scheduler, groups = build_optimizer'))
        stop = next(i for i, n in enumerate(body[start:], start)
                    if isinstance(n, ast.If) and ast.unparse(n.test).startswith('staging.info'))
        load = Mock(side_effect=AssertionError('fresh run must not deserialize text training state'))
        opt, sched, scaler, tracker = Mock(), Mock(), Mock(), Mock()
        torch = NS(load=load, cuda=NS(amp=NS(GradScaler=Mock(return_value=scaler))))
        scope = dict(config={'resume_checkpoint': ''}, model=object(), geometry={'total_steps': 17313},
                     torch=torch, build_optimizer=Mock(return_value=(opt, sched, [])),
                     GradNormTracker=Mock(return_value=tracker), ROUTE=contract.ROUTE, EPOCHS=3)
        audit.execute(body[start:stop], scope)
        self.assertEqual((scope['step'], scope['start_epoch'], scope['start_batch']), (0, 0, 0))
        self.assertIsNone(scope['resume_rng'])
        scope['build_optimizer'].assert_called_once_with(scope['model'], 17313)
        torch.cuda.amp.GradScaler.assert_called_once_with(enabled=True)
        scope['GradNormTracker'].assert_called_once_with()
        load.assert_not_called()
        for obj in (opt, sched, scaler, tracker): obj.load_state_dict.assert_not_called()

    def test_full_audio_resume_rejects_other_route_or_source_before_loading_state(self):
        resume = next(n for n in audit.find_function(audit.tree(ROUTE / 'train.py'), 'train').body
                      if isinstance(n, ast.If) and ast.unparse(n.test) == "config['resume_checkpoint']")
        valid = dict(schema_version=1, mode='formal', route_variant=contract.ROUTE, contract_sha256='same')
        for change in (dict(route_variant='old_double_loop'), dict(contract_sha256='other_text_source'),
                       dict(mode='smoke'), dict(schema_version=2)):
            objects = [Mock() for _ in range(5)]
            scope = dict(config=dict(resume_checkpoint='synthetic.ckpt', contract_sha256='same'),
                         torch=NS(load=Mock(return_value=dict(valid, **change))), ROUTE=contract.ROUTE,
                         **dict(zip(('model', 'optimizer', 'scheduler', 'scaler', 'tracker'), objects)))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'identical route'):
                audit.execute([resume], scope)
            for obj in objects: obj.load_state_dict.assert_not_called()

    def test_same_route_resume_restores_every_state_cursor_and_rank_specific_rng(self):
        resume = next(n for n in audit.find_function(audit.tree(ROUTE / 'train.py'), 'train').body
                      if isinstance(n, ast.If) and ast.unparse(n.test) == "config['resume_checkpoint']")
        saved = dict(schema_version=1, mode='formal', route_variant=contract.ROUTE,
                     contract_sha256='same', total_step=5772, epoch=1, next_optimizer_step=1,
                     state_dict={'all_three_loop_weights': 1}, optimizer={'moments': 2},
                     scheduler={'last_epoch': 5772}, grad_scaler={'scale': 3},
                     grad_norm_tracker={'running_norm': 4}, random_state_by_rank=[{'rank': r} for r in range(8)])
        objects = [Mock() for _ in range(5)]
        objects[2].last_epoch = 5772
        scope = dict(config=dict(resume_checkpoint='synthetic.ckpt', contract_sha256='same'),
                     torch=NS(load=Mock(return_value=saved)), ROUTE=contract.ROUTE,
                     geometry={'steps_per_epoch': 5771}, EPOCHS=3, rank=3,
                     **dict(zip(('model', 'optimizer', 'scheduler', 'scaler', 'tracker'), objects)))
        audit.execute([resume], scope)
        objects[0].load_state_dict.assert_called_once_with(saved['state_dict'], strict=True)
        for obj, key in zip(objects[1:], ('optimizer', 'scheduler', 'grad_scaler', 'grad_norm_tracker')):
            obj.load_state_dict.assert_called_once_with(saved[key])
        self.assertEqual((scope['step'], scope['start_epoch'], scope['start_batch']), (5772, 1, 1))
        self.assertEqual(scope['resume_rng'], {'rank': 3})

    def test_audit_json_files_are_exposed_to_git(self):
        import subprocess
        for name in ('source_guard.json', 'reference_audit.json', 'parent_reference_audit.json'):
            path = ROUTE / name
            self.assertTrue(path.is_file())
            result = subprocess.run(['git', 'check-ignore', str(path)], cwd=ROOT, capture_output=True)
            self.assertEqual(result.returncode, 1, result.stdout.decode(errors='replace'))


if __name__ == '__main__':
    unittest.main()
