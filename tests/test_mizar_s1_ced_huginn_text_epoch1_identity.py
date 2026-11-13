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
PARENT = FAMILY / 'mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small'
ROUTE = FAMILY / 'mizar_s1_ced_small_huginn_text_epoch1'
LAUNCHER = ROOT / 'code/RSmol/run_mizar_s1_ced_small_huginn_text_epoch1_5090.sh'
EXPECTED_TEXT = ('/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/'
                 'mizar_text_5_10x2_5_huginn_gbs1024/epoch1/checkpoint-step-009244')
sys.path.insert(0, str(ROUTE))
import contract
from models import recursive_text
import audit_reference
from mizar_text_5_10x2_5_huginn import contract as text_contract


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized(path):
    return path.read_bytes().replace(b'\r\n', b'\n')


def function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def text_config():
    source = SimpleNamespace(num_hidden_layers=30, hidden_size=576, vocab_size=49152,
                             tie_word_embeddings=True, model_type='llama', eos_token_id=0, pad_token_id=0)
    return vars(text_contract.build_target_config(source))


def text_checkpoint(path):
    path.mkdir(parents=True, exist_ok=True)
    contract.write_json(path/'config.json', text_config())
    (path/'tokenizer.json').write_text('{}')
    (path/'model.safetensors').write_bytes(b'synthetic weight inventory')
    (path/'training_state.pt').write_bytes(b'POISON: do not deserialize text training state')
    contract.write_json(path/'checkpoint_complete.json',dict(complete=True,mode='formal',
        optimizer_step=9244,world_size=8,text_route=text_contract.ROUTE,
        architecture=text_contract.ARCHITECTURE,data_cursors_by_rank=[str(i) for i in range(8)],
        rng_states_by_rank=[str(i) for i in range(8)]))
    return path


def fake_model():
    shared=object()
    cfg=text_config()
    model=SimpleNamespace(config=SimpleNamespace(to_dict=lambda:cfg))
    model.model=SimpleNamespace(layers=[object() for _ in range(20)],
        logical_to_physical=contract.TEXT_MODEL_SCHEDULE,
        recurrent_adapter=SimpleNamespace(bias=None,weight=SimpleNamespace(shape=(576,1152),requires_grad=True)))
    model.get_input_embeddings=lambda:SimpleNamespace(weight=shared)
    model.lm_head=SimpleNamespace(weight=shared)
    model.named_parameters=lambda:iter([('model.recurrent_adapter.weight',model.model.recurrent_adapter.weight)])
    model.named_buffers=lambda:iter([])
    return model


def model_audit():
    return dict(logical_layer_count=30,physical_layer_count=20,recursive_layer_count=20,
        recursive_loops=2,middle_recurrent_count=10,model_physical_module_count=20,
        schedule=list(contract.TEXT_MODEL_SCHEDULE),adapter_parameters=663552,adapter_calls_per_forward=2)


class Epoch1AudioIdentityTests(unittest.TestCase):
    def test_all_original_sources_untouched(self):
        guard = json.loads((ROUTE / 'source_guard.json').read_text(encoding='utf-8'))
        self.assertEqual(len(guard), 35)
        for name, digest in guard.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256(normalized(ROOT / name)).hexdigest(), digest)

    def test_all_copied_sources_except_allowed_files_are_byte_identical(self):
        changed = {'.gitignore', 'contract.py', 'run.sh', 'models/recursive_text.py', 'write_config.py', 'audit_reference.py', 'train.py', 'optim.py'}
        matched = []
        for original in sorted(PARENT.rglob('*')):
            if not original.is_file() or '__pycache__' in original.parts:
                continue
            name = original.relative_to(PARENT).as_posix()
            if name in {'README.md', 'reference_audit.json'} or name in changed:
                continue
            with self.subTest(file=name):
                self.assertEqual(original.read_bytes(), (ROUTE / name).read_bytes())
            matched.append(name)
        audit_reference.verify_training_delta()
        audit_reference.verify_optimizer_delta()
        self.assertIn('models/ced.py', matched)
        self.assertIn('models/decoder.py', matched)
        self.assertIn('data/audiotext_dataset.py', matched)

    def test_audit_manifests_are_included_by_scoped_git_add(self):
        expected = (PARENT / '.gitignore').read_text(encoding='utf-8').rstrip() + '\n'
        expected += '!source_guard.json\n!parent_reference_audit.json\n'
        self.assertEqual((ROUTE / '.gitignore').read_text(encoding='utf-8').strip(), expected.strip())
        for name in ('source_guard.json', 'parent_reference_audit.json'):
            result = subprocess.run(['git', 'check-ignore', '--no-index', '-q', str(ROUTE / name)],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)

    def test_contract_diff_preserves_audio_recipe_except_huginn_identity(self):
        result=audit_reference.audit_parent()
        self.assertTrue(result['training_ast_preserved_except_diagnostics'])
        self.assertEqual(contract.DEFAULT_TEXT_MODEL_DIR, EXPECTED_TEXT)
        self.assertEqual(contract.TEXT_MODEL_CONTRACT,text_contract.ARCHITECTURE)

    def test_runtime_diff_changes_only_help_and_tmpfs_namespace(self):
        expected = (PARENT / 'run.sh').read_text(encoding='utf-8').replace(
            'run_mizar_s1_gbs256_recursive_5_10_5_ced_small_5090.sh',
            'run_mizar_s1_ced_small_huginn_text_epoch1_5090.sh').replace(
            '/dev/shm/mizar_s1_gbs256_recursive_5_10_5_ced_small_',
            '/dev/shm/mizar_s1_gbs256_ced_small_huginn_text_epoch1_')
        self.assertEqual(expected, (ROUTE / 'run.sh').read_text(encoding='utf-8'))

    def test_launcher_diff_changes_only_init_and_isolation_names(self):
        expected = (ROOT / 'code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_5090.sh').read_text(encoding='utf-8')
        expected = expected.replace(PARENT.name, ROUTE.name).replace(
            '/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/'
            'formal-epoch2-continue-20260902_184936/checkpoint-step-009244', EXPECTED_TEXT).replace(
            'mizar_s1_ced_small_gbs256_recursive_5_10_5_5090',
            'mizar_s1_ced_small_huginn_text_epoch1_gbs256_5090').replace(
            'mizar-s1-ced-rec515-gbs256-', 'mizar-s1-ced-huginn-e1-').replace(
            'mizar_s1_gbs256_recursive_5_10_5_ced_small.', 'mizar_s1_ced_small_huginn_text_epoch1_5090.').replace(
            'Submitted Mizar S1 / CED-Small GBS256 / recursive 5-10-5 non-MeSH:',
            'Submitted Mizar S1 / CED-Small GBS256 / Huginn text epoch1 initialization:')
        self.assertEqual(expected, LAUNCHER.read_text(encoding='utf-8'))
        self.assertIn('TEXT_MODEL_DIR=' + EXPECTED_TEXT, expected)

    def test_existing_mizar_text_configuration_is_accepted_without_reconversion(self):
        text = text_contract
        original = SimpleNamespace(num_hidden_layers=30, hidden_size=576, vocab_size=49152,
                                   tie_word_embeddings=True, model_type='llama', eos_token_id=0,
                                   pad_token_id=0)
        cfg = vars(text.build_target_config(original))
        self.assertEqual(tuple(cfg['recursive_source_layer_indices_0based']), contract.TEXT_MODEL_SOURCE_MAPPING)
        self.assertEqual(tuple(cfg['logical_to_physical']), contract.TEXT_MODEL_SCHEDULE)
        self.assertEqual(contract.validate_recursive_config_values(cfg)['physical_layers'], 20)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'checkpoint-step-009244'
            text_checkpoint(root)
            record = contract.recursive_checkpoint_contract(root)
            self.assertEqual(record['weight_files'], ['model.safetensors'])
            self.assertNotIn('training_state.pt', record['weight_files'])
            self.assertEqual(record['initialization']['audio_seed'],1234)
            self.assertFalse(record['initialization']['restore_text_training_state'])

    def test_entire_semantic_recipe_matches_parent_except_route_and_code_identity(self):
        old = load_module('epoch1_audio_parent_contract', PARENT / 'contract.py')
        prepared = dict(rows=1477492, rows_canonical_sha256='synthetic-rows',
                        mapping_sha256='synthetic-mapping', source_rows=contract.SOURCE_COUNTS)
        new = contract.semantic_contract(prepared, EXPECTED_TEXT, '/models/CED_small')
        original = old.semantic_contract(prepared, EXPECTED_TEXT, '/models/CED_small')
        self.assertNotEqual(new['route'], original['route'])
        self.assertNotEqual(contract.digest(new), old.digest(original))
        for key in set(new) | set(original):
            if key not in {'route', 'code_sha256', 'text_model_contract', 'text_model_architecture',
                           'recursive_model_source_sha256', 'recursive_model_sources', 'huginn', 'audio_boundary_normalization'}:
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
                    route_variant='ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_v1',
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
            path=text_checkpoint(Path(directory)/'checkpoint-step-009244')
            cls=Mock(); model=fake_model()
            for failure in (None,'missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
                info={} if failure is None else {failure:['model.recurrent_adapter.weight']}
                cls.from_pretrained.return_value=(model,info)
                model.model.enable_recurrent_boundary_norms=Mock()
                with self.subTest(failure=failure),patch.object(recursive_text,'validate_loaded_model') as validate, patch.object(recursive_text,'_import_recursive_model',
                        return_value=(cls,lambda m:model_audit())):
                    if failure:
                        with self.assertRaisesRegex(ValueError,'weights do not match'):
                            recursive_text.load_recursive_text_model(path)
                    else:
                        self.assertIs(recursive_text.load_recursive_text_model(path),model)
                        model.model.enable_recurrent_boundary_norms.assert_called_once_with()
                        self.assertEqual(validate.call_count,2)
                        self.assertTrue(validate.call_args.kwargs['require_boundary_norms'])
                    if failure:
                        model.model.enable_recurrent_boundary_norms.assert_not_called()
                self.assertEqual(cls.from_pretrained.call_args.kwargs,dict(local_files_only=True,output_loading_info=True))

    def test_bash_syntax_and_lf(self):
        bash = 'C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else 'bash'
        for path in (LAUNCHER, ROUTE/'run.sh'):
            self.assertNotIn(b'\r',path.read_bytes())
            result=subprocess.run([bash,'-n',path.as_posix()],capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_huginn_checkpoint_rejects_wrong_architecture_noise_mapping_or_incomplete_marker(self):
        cfg=text_config()
        changes={'architectures':['RecursiveLlamaForCausalLM'],'text_route':'ordinary-text',
                 'huginn_state_std':1.0,'huginn_concat_order':['prelude','state'],
                 'huginn_adapter_bias':True,'huginn_noise_samples_per_forward':2,
                 'recursive_source_layer_indices_0based':list(range(20)),
                 'mesh_enabled':True,'vocab_size':32}
        for key,value in changes.items():
            with self.subTest(key=key),self.assertRaises(ValueError):
                contract.validate_recursive_config_values({**cfg,key:value})
        with tempfile.TemporaryDirectory() as directory:
            path=text_checkpoint(Path(directory)/'checkpoint-step-009244')
            marker=json.loads((path/'checkpoint_complete.json').read_text())
            for key,value in {'complete':False,'text_route':'other','optimizer_step':500,
                              'rng_states_by_rank':['0']}.items():
                contract.write_json(path/'checkpoint_complete.json',{**marker,key:value})
                with self.subTest(marker=key),self.assertRaises(ValueError):
                    contract.recursive_checkpoint_contract(path)

    def test_loaded_model_requires_shared_trainable_adapter_and_tied_head(self):
        recursive_text.validate_loaded_model(fake_model(),model_audit())
        for case in ('bias','shape','frozen','missing','untied','mesh','layers'):
            model=fake_model()
            adapter=model.model.recurrent_adapter
            if case=='bias': adapter.bias=object()
            if case=='shape': adapter.weight.shape=(576,576)
            if case=='frozen': adapter.weight.requires_grad=False
            if case=='missing': del model.model.recurrent_adapter
            if case=='untied': model.lm_head.weight=object()
            if case=='mesh': model.named_buffers=lambda:iter([('model.mesh_router.weight',object())])
            if case=='layers': model.model.layers=[object()]*20
            with self.subTest(case=case),self.assertRaises(ValueError):
                recursive_text.validate_loaded_model(model,model_audit())

    def test_source_identity_covers_huginn_model_dependency_and_release_contract(self):
        sources=contract.recursive_model_source_files()
        self.assertEqual(len(sources),3)
        self.assertIn('code/RSmol/mizar_text_5_10x2_5_huginn/contract.py',sources)
        self.assertIn('code/RSmol/mizar_official_eval/release_manifest.json',sources)
        for name,value in sources.items():
            self.assertEqual(hashlib.sha256(normalized(ROOT/name)).hexdigest(),value)
        self.assertEqual(contract.recursive_model_source_sha256(),contract.digest(sources))
        with patch.object(contract,'recursive_model_source_files',return_value={**sources,'model':'changed'}):
            self.assertNotEqual(contract.recursive_model_source_sha256(),contract.digest(sources))

    def test_route_contract_import_works_after_stdlib_code_has_loaded(self):
        # Torch/pdb may import Python's `code` before the audio contract.
        # This route must keep that module intact while finding Huginn.
        command=('import code,sys; original=code; '
                 'assert hasattr(code,"InteractiveConsole"); '
                 'sys.path.insert(0,sys.argv[1]); import contract; '
                 'assert contract.TEXT_MODEL_CONTRACT.endswith("huginn_loops_2"); '
                 'assert sys.modules["code"] is original')
        with tempfile.TemporaryDirectory() as directory:
            result=subprocess.run([sys.executable,'-B','-c',command,str(ROUTE)],cwd=directory,
                                  capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_real_audio_resume_selects_rank_rng_and_full_state(self):
        tree=ast.parse((ROUTE/'train.py').read_text(encoding='utf-8'))
        block=next(n for n in function(tree,'train').body if isinstance(n,ast.If)
                   and ast.unparse(n.test)=="config['resume_checkpoint']")
        for rank in range(8):
            states=[{'rank':r} for r in range(8)]
            saved=dict(schema_version=1,route_variant=contract.ROUTE,mode='formal',contract_sha256='same',
                state_dict={'trained_A':'included'},optimizer={'moments':'saved'},scheduler={'step':5771},
                grad_scaler={'scale':65536},grad_norm_tracker={'running':'saved'},
                total_step=5771,epoch=1,next_optimizer_step=0,random_state_by_rank=states)
            model,opt,sched,scaler,tracker=Mock(),Mock(),Mock(),Mock(),Mock()
            sched.last_epoch=5771
            scope=dict(config=dict(resume_checkpoint='audio.ckpt',contract_sha256='same'),
                       torch=SimpleNamespace(load=Mock(return_value=saved)),ROUTE=contract.ROUTE,
                       model=model,optimizer=opt,scheduler=sched,scaler=scaler,tracker=tracker,
                       geometry=dict(steps_per_epoch=5771),EPOCHS=3,rank=rank)
            exec(compile(ast.Module([copy.deepcopy(block)],type_ignores=[]),'actual-audio-resume','exec'),scope)
            self.assertEqual(scope['resume_rng'],states[rank])
            self.assertEqual((scope['step'],scope['start_epoch'],scope['start_batch']),(5771,1,0))
            for obj,key in ((model,'state_dict'),(opt,'optimizer'),(sched,'scheduler'),
                            (scaler,'grad_scaler'),(tracker,'grad_norm_tracker')):
                self.assertEqual(obj.load_state_dict.call_args.args[0],saved[key])

    def test_audio_source_probes_restore_rng_and_preserve_mode_on_failure(self):
        tree=ast.parse((ROUTE/'train.py').read_text(encoding='utf-8'))
        node=copy.deepcopy(function(tree,'probe_sources'))
        from contextlib import nullcontext
        model=Mock(); model.side_effect=RuntimeError('synthetic failed Huginn probe')
        saved={'noise-stream':'before probe'}
        restore=Mock()
        scope=dict(rng_state=Mock(return_value=saved),restore_rng=restore,
                   torch=SimpleNamespace(no_grad=nullcontext),collate_fn=lambda x:x,
                   move_batch=lambda batch,device:batch)
        exec(compile(ast.Module([node],type_ignores=[]),'actual-source-probe','exec'),scope)
        with self.assertRaisesRegex(RuntimeError,'Huginn probe'):
            scope['probe_sources'](model,Mock(),{'probe_rows':{'AudioCaps':[1]}},'cuda:0')
        model.train.assert_called_once_with()
        restore.assert_called_once_with(saved,'cuda:0')


if __name__ == '__main__':
    unittest.main()
