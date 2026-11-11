"""Exact-copy and initialization checks for the isolated text-epoch1 audio run."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / 'code/RSmol/mellow_official_training_c8204d8_adamw_cosine'
PARENT = FAMILY / 'mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8'
ROUTE = FAMILY / 'mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8'
LAUNCHER = ROOT / 'code/RSmol/run_mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_5090.sh'
EXPECTED_TEXT = ('/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/'
                 'mizar_text_5_10x2_5_gbs1024/epoch1/checkpoint-step-009244')
sys.path.insert(0, str(ROUTE))
import contract
from models import recursive_text
import audit_reference


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized(path):
    return path.read_bytes().replace(b'\r\n', b'\n')


def function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


class Epoch1AudioLoopLoRAIdentityTests(unittest.TestCase):
    def test_exact_runtime_and_launcher_diff_whitelist_and_entire_recipe(self):
        result = audit_reference.audit()
        self.assertTrue(result["common_recipe_equal"])
        self.assertTrue(result["training_parity"]["entire_train_file_byte_identical"])
        self.assertTrue(result["backbone_parity"]["entire_backbone_file_byte_identical"])
        self.assertIn("models/loop_lora.py", result["byte_identical_sources"])


    def test_all_original_sources_untouched(self):
        guard = json.loads((ROUTE / 'source_guard.json').read_text(encoding='utf-8'))
        self.assertEqual(len(guard), 35)
        for name, digest in guard.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256(normalized(ROOT / name)).hexdigest(), digest)

    def test_training_sources_except_identity_and_audit_are_byte_identical(self):
        changed = {'contract.py', 'run.sh', 'audit_reference.py'}
        matched = []
        for original in sorted(PARENT.rglob('*')):
            if not original.is_file() or '__pycache__' in original.parts:
                continue
            name = original.relative_to(PARENT).as_posix()
            if name in {'README.md', 'reference_audit.json', '.gitignore'} or name in changed:
                continue
            with self.subTest(file=name):
                self.assertEqual(original.read_bytes(), (ROUTE / name).read_bytes())
            matched.append(name)
        self.assertIn('train.py', matched)
        self.assertIn('optim.py', matched)
        self.assertIn('models/ced.py', matched)
        self.assertIn('models/recursive_text.py', matched)
        self.assertIn('data/audiotext_dataset.py', matched)




    def test_existing_mizar_text_configuration_is_accepted_without_reconversion(self):
        text = load_module('epoch1_text_contract_fixture', ROOT / 'code/RSmol/mizar_text_5_10x2_5/contract.py')
        original = SimpleNamespace(num_hidden_layers=30, hidden_size=576, vocab_size=49152,
                                   tie_word_embeddings=True, model_type='llama', eos_token_id=0,
                                   pad_token_id=0)
        cfg = vars(text.build_target_config(original))
        self.assertEqual(tuple(cfg['recursive_source_layer_indices_0based']), contract.TEXT_MODEL_SOURCE_MAPPING)
        self.assertEqual(tuple(cfg['logical_to_physical']), contract.TEXT_MODEL_SCHEDULE)
        self.assertEqual(contract.validate_recursive_config_values(cfg)['physical_layers'], 20)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'checkpoint-step-009244'
            root.mkdir()
            contract.write_json(root / 'config.json', cfg)
            (root / 'tokenizer.json').write_text('{}')
            (root / 'model.safetensors').write_bytes(b'synthetic inventory; no real weights')
            (root / 'training_state.pt').write_bytes(b'POISON: audio must never deserialize text training state')
            record = contract.recursive_checkpoint_contract(root)
            self.assertEqual(record['weight_files'], ['model.safetensors'])
            self.assertNotIn('training_state.pt', record['weight_files'])

    def test_entire_semantic_recipe_matches_parent_except_route_and_code_identity(self):
        old = load_module('epoch1_audio_parent_contract', PARENT / 'contract.py')
        prepared = dict(rows=1477492, rows_canonical_sha256='synthetic-rows',
                        mapping_sha256='synthetic-mapping', source_rows=contract.SOURCE_COUNTS)
        new = contract.semantic_contract(prepared, EXPECTED_TEXT, '/models/CED_small')
        original = old.semantic_contract(prepared, EXPECTED_TEXT, '/models/CED_small')
        self.assertNotEqual(new['route'], original['route'])
        self.assertNotEqual(contract.digest(new), old.digest(original))
        for key in set(new) | set(original):
            if key not in {'route', 'code_sha256'}:
                with self.subTest(field=key):
                    self.assertEqual(new[key], original[key])

    def test_every_audio_lr_point_and_gradnorm_policy_are_identical(self):
        old = load_module('epoch1_audio_parent_schedule', PARENT / 'contract.py')
        for step in range(17314):
            self.assertEqual(contract.lr_factor(step, 17313), old.lr_factor(step, 17313))
        self.assertEqual(contract.lr_factor(0, 17313), 0.1)
        self.assertAlmostEqual(contract.lr_factor(17313, 17313), 0.1)
        scopes = []
        for route in (PARENT, ROUTE):
            tree = ast.parse((route / 'optim.py').read_text(encoding='utf-8'))
            scope = {}
            node = copy.deepcopy(function(tree, '_clip_grad'))
            exec(compile(ast.Module([node], type_ignores=[]), 'actual_gradnorm', 'exec'), scope)
            scopes.append(scope)
        rng = random.Random(1234)
        tracker = SimpleNamespace(overdrive_factor=2.5, momentum=.995)
        for _ in range(1000):
            args = tracker, 10**rng.uniform(-8,8), 10**rng.uniform(-8,8), rng.random()
            self.assertEqual(scopes[0]['_clip_grad'](*args), scopes[1]['_clip_grad'](*args))

    def test_fresh_audio_initialization_executes_step_zero_branch_and_resets_all_state(self):
        # Execute the real optimizer/scaler/tracker + optional resume portion of train().
        tree = ast.parse((ROUTE / 'train.py').read_text(encoding='utf-8'))
        body = function(tree, 'train').body
        start = next(i for i,n in enumerate(body) if isinstance(n, ast.Assign)
                     and ast.unparse(n).startswith('optimizer, scheduler, groups = build_optimizer'))
        stop = next(i for i,n in enumerate(body[start:], start)
                    if isinstance(n, ast.If) and ast.unparse(n.test).startswith('staging.info'))
        selected = ast.Module(copy.deepcopy(body[start:stop]), type_ignores=[])
        load = Mock(side_effect=AssertionError('fresh audio run must not read text training_state.pt'))
        opt, sched, scaler, tracker = Mock(), Mock(), Mock(), Mock()
        torch = SimpleNamespace(load=load, cuda=SimpleNamespace(amp=SimpleNamespace(
            GradScaler=Mock(return_value=scaler))))
        scope = dict(config={'resume_checkpoint': ''}, model=object(), geometry={'total_steps':17313},
                     torch=torch, build_optimizer=Mock(return_value=(opt,sched,[])),
                     GradNormTracker=Mock(return_value=tracker), ROUTE=contract.ROUTE, EPOCHS=3)
        exec(compile(selected, 'actual_audio_initialization', 'exec'), scope)
        self.assertEqual((scope['step'],scope['start_epoch'],scope['start_batch']), (0,0,0))
        self.assertIsNone(scope['resume_rng'])
        scope['build_optimizer'].assert_called_once_with(scope['model'],17313)
        torch.cuda.amp.GradScaler.assert_called_once_with(enabled=True)
        scope['GradNormTracker'].assert_called_once_with()
        load.assert_not_called()
        for obj in (opt,sched,scaler,tracker): obj.load_state_dict.assert_not_called()

    def test_audio_resume_rejects_parent_checkpoint_before_loading_any_state(self):
        tree = ast.parse((ROUTE / 'train.py').read_text(encoding='utf-8'))
        resume = next(n for n in function(tree,'train').body if isinstance(n,ast.If)
                      and ast.unparse(n.test) == "config['resume_checkpoint']")
        load = Mock(return_value=dict(schema_version=1,mode='formal',
                    route_variant='ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8_v1',
                    contract_sha256='same-fixture-hash'))
        model,opt,sched,scaler,tracker=Mock(),Mock(),Mock(),Mock(),Mock()
        scope=dict(config=dict(resume_checkpoint='parent.ckpt',contract_sha256='same-fixture-hash'),
                   torch=SimpleNamespace(load=load),ROUTE=contract.ROUTE,
                   model=model,optimizer=opt,scheduler=sched,scaler=scaler,tracker=tracker)
        with self.assertRaisesRegex(ValueError,'identical route'):
            exec(compile(ast.Module([copy.deepcopy(resume)],type_ignores=[]),'actual_audio_resume','exec'),scope)
        for obj in (model,opt,sched,scaler,tracker): obj.load_state_dict.assert_not_called()

    def test_text_loader_never_reads_training_state_and_rejects_partial_lm(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            cfg=dict(model_type='llama',architectures=['RecursiveLlamaForCausalLM'],
                     hidden_size=576,num_hidden_layers=30,recursive_layer_count=20,
                     recursive_loops=2,recursive_prefix_layer_count=5,
                     recursive_middle_layer_count=10,recursive_suffix_layer_count=5,
                     recursive_loops_scope='middle_only',vocab_size=49152,eos_token_id=0,
                     logical_to_physical=list(contract.TEXT_MODEL_SCHEDULE),
                     recursive_source_layer_indices_0based=list(contract.TEXT_MODEL_SOURCE_MAPPING))
            contract.write_json(path/'config.json',cfg)
            (path/'training_state.pt').write_bytes(b'poison')
            cls=Mock()
            cls.from_pretrained.return_value=(object(),{'missing_keys':['model.layers.5.mlp.up_proj.weight']})
            with patch.object(recursive_text,'_import_recursive_model',return_value=(cls,Mock())):
                with self.assertRaisesRegex(ValueError,'weights do not match'):
                    recursive_text.load_recursive_text_model(path)
            cls.from_pretrained.assert_called_once_with(path.resolve(),local_files_only=True,output_loading_info=True)

    def test_bash_syntax_and_lf(self):
        bash = 'C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else 'bash'
        for path in (LAUNCHER, ROUTE/'run.sh'):
            self.assertNotIn(b'\r',path.read_bytes())
            result=subprocess.run([bash,'-n',path.as_posix()],capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__':
    unittest.main()
