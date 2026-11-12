"""Pretrained LoRA identity/loading and complete audio-parent parity checks."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT/'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced48_text_epoch1_loop_lora_r8'
LAUNCHER = ROOT/'code/RSmol/run_mizar_s1_ced48_text_epoch1_loop_lora_r8_5090.sh'
sys.path.insert(0,str(ROUTE))
import audit_reference as A
import contract as C
from models import recursive_text as L


def text_config():
    return dict(model_type='llama',architectures=['MizarTextLoopLoRAR8ForCausalLM'],
                hidden_size=576,vocab_size=49152,eos_token_id=0,tie_word_embeddings=True,
                num_hidden_layers=30,recursive_layer_count=20,recursive_loops=2,
                recursive_prefix_layer_count=5,recursive_middle_layer_count=10,
                recursive_suffix_layer_count=5,recursive_loops_scope='middle_only',
                recursive_source_layer_indices_0based=list(C.TEXT_MODEL_SOURCE_MAPPING),
                logical_to_physical=list(C.TEXT_MODEL_SCHEDULE),text_route=C.TEXT_SOURCE_ROUTE,
                loop_lora_contract=copy.deepcopy(C.TEXT_LORA_CONTRACT))


def source_fixture(root):
    path=root/'checkpoint-step-009244';path.mkdir()
    C.write_json(path/'config.json',text_config())
    C.write_json(path/'checkpoint_complete.json',dict(complete=True,mode='formal',
        text_route=C.TEXT_SOURCE_ROUTE,architecture=C.TEXT_MODEL_ARCHITECTURE,
        optimizer_step=9244,world_size=8,data_cursors_by_rank=[str(i) for i in range(8)]))
    (path/'model.safetensors').write_bytes(b'synthetic fixture; never decode as weights')
    (path/'tokenizer.json').write_text('{}')
    (path/'training_state.pt').write_bytes(b'POISON: text state must never be deserialized')
    return path


class IdentityTests(unittest.TestCase):
    def test_full_parent_sources_recipe_backbone_and_update_loop(self):
        result=A.audit()
        self.assertTrue(result['protected_originals']['all_unchanged'])
        self.assertEqual(result['protected_originals']['files'],37)
        self.assertTrue(result['audio_semantic_recipe_equal'])
        self.assertTrue(result['training_parity']['entire_train_function_preserved_after_explicit_audits'])
        self.assertTrue(result['backbone_parity']['one_constructor_install_before_hf_weight_load'])
        self.assertIn('optim.py',result['byte_identical_sources'])

    def test_constructor_prepares_lora_before_load_and_loader_never_reinstalls(self):
        tree=A.tree(ROUTE/'models/recursive_layer_inject.py')
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RecursiveLlamaForCausalLM')
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        calls=[ast.unparse(n.value) for n in init.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)]
        self.assertEqual(calls[-2:],['self.post_init()','self.model.enable_loop_lora()'])
        loader=A.find_function(A.tree(ROUTE/'models/recursive_text.py'),'load_recursive_text_model')
        self.assertNotIn('enable_loop_lora',ast.unparse(loader))
        self.assertNotIn('register_auto_class',ast.unparse(loader))

    def test_source_config_is_actual_text_route_and_rejects_wrong_lora(self):
        C.validate_recursive_config_values(text_config())
        values=[('architectures',['RecursiveLlamaForCausalLM']),('text_route','old'),
                ('loop_lora_contract',{**C.TEXT_LORA_CONTRACT,'rank':4}),
                ('recursive_loops',3),('logical_to_physical',list(range(30))),
                ('mesh_enabled',True),('tie_word_embeddings',False),('vocab_size',32)]
        for key,value in values:
            with self.subTest(key=key),self.assertRaises(ValueError):
                C.validate_recursive_config_values({**text_config(),key:value})
        # Preserve the TEXT-stage LR metadata while audio uses discriminative LR.
        self.assertEqual(C.TEXT_LORA_CONTRACT['lr_policy'],'same_as_base')
        self.assertEqual(C.LOOP_LORA_CONTRACT['max_lr'],6e-4)
        self.assertEqual(C.LOOP_LORA_CONTRACT['fresh_parameters'],0)

    def test_marker_requires_complete_final_source_but_never_reads_training_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=source_fixture(Path(tmp))
            result=C.recursive_checkpoint_contract(path)
            self.assertFalse(result['initialization']['read_text_training_state'])
            self.assertNotIn('training_state.pt',result['weight_files'])
            marker=json.loads((path/'checkpoint_complete.json').read_text())
            for key,value in [('complete',False),('text_route','old'),('optimizer_step',9000),
                              ('architecture','old'),('world_size',4),('data_cursors_by_rank',['0'])]:
                C.write_json(path/'checkpoint_complete.json',{**marker,key:value})
                with self.subTest(key=key),self.assertRaises(ValueError): C.recursive_checkpoint_contract(path)

    def test_strict_loader_validates_every_failure_before_adding_audio(self):
        for failure in (None,'missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
            events=[]
            core=NS(enable_layer_injection=lambda:events.append('CED gates'),
                    enable_audio_reinjection=lambda:events.append('input branch'),
                    enable_loop_lora=Mock(side_effect=AssertionError('trained LoRA must not reset')))
            model=NS(model=core,generation_config=NS(use_cache=True))
            cls=Mock();cls.from_pretrained.return_value=(model,{} if failure is None else {failure:['bad']})
            fake_torch=NS(float32='float32')
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                path=source_fixture(Path(tmp))
                with patch.dict(sys.modules,torch=fake_torch), \
                     patch.object(L,'_import_recursive_model',return_value=(cls,Mock())), \
                     patch.object(L,'validate_loaded_model',side_effect=lambda *args:events.append('validate')), \
                     patch.object(L,'validate_loop_lora'),patch.object(L,'validate_serial_injection'), \
                     patch.object(L,'audit_inherited_text_weights',side_effect=lambda *args:events.append('compare') or {'lora_tensor_count':280}):
                    if failure:
                        with self.assertRaisesRegex(ValueError,'weights do not match'): L.load_recursive_text_model(path)
                        self.assertEqual(events,[])
                    else:
                        self.assertIs(L.load_recursive_text_model(path),model)
                        self.assertEqual(events,['validate','compare','CED gates','input branch','compare'])
                        self.assertFalse(model.generation_config.use_cache)
                        self.assertFalse(model.text_initialization_audit['source']['lora_reset_after_load'])
                    core.enable_loop_lora.assert_not_called()
                    self.assertEqual(cls.from_pretrained.call_args.kwargs,dict(
                        local_files_only=True,output_loading_info=True,torch_dtype='float32'))

    def test_actual_audio_state_initialization_is_fresh_and_never_reads_text_optimizer(self):
        body=A.find_function(A.tree(ROUTE/'train.py'),'train').body
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                   and ast.unparse(n).startswith('optimizer, scheduler, groups = build_optimizer'))
        stop=next(i for i,n in enumerate(body[start:],start) if isinstance(n,ast.If)
                  and ast.unparse(n.test).startswith('staging.info'))
        optimizer,scheduler,scaler,tracker=[Mock() for _ in range(4)]
        torch=NS(load=Mock(side_effect=AssertionError('must not deserialize text state')),
                 cuda=NS(amp=NS(GradScaler=Mock(return_value=scaler))))
        scope=dict(config={'resume_checkpoint':''},model=object(),geometry={'total_steps':17313},
                   torch=torch,build_optimizer=Mock(return_value=(optimizer,scheduler,[])),
                   GradNormTracker=Mock(return_value=tracker),ROUTE=C.ROUTE,EPOCHS=3)
        A.execute(body[start:stop],scope)
        self.assertEqual((scope['step'],scope['start_epoch'],scope['start_batch']),(0,0,0))
        self.assertIsNone(scope['resume_rng']);torch.load.assert_not_called()
        for obj in (optimizer,scheduler,scaler,tracker): obj.load_state_dict.assert_not_called()

    def test_resume_rejects_old_audio_route_before_loading_weights(self):
        guard=next(n for n in A.find_function(A.tree(ROUTE/'train.py'),'train').body
                   if isinstance(n,ast.If) and ast.unparse(n.test)=="config['resume_checkpoint']")
        for previous in ('ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8_v1',
                         'ced_small_mizar_s1_fixed20_gbs256_text_epoch1_5_10x2_5_v1'):
            model,optimizer,scheduler,scaler,tracker=[Mock() for _ in range(5)]
            saved=dict(schema_version=1,mode='formal',route_variant=previous,contract_sha256='hash')
            scope=dict(config=dict(resume_checkpoint='audio.ckpt',contract_sha256='hash'),
                       torch=NS(load=lambda *a,**kw:saved),ROUTE=C.ROUTE,model=model,
                       optimizer=optimizer,scheduler=scheduler,scaler=scaler,tracker=tracker)
            with self.subTest(route=previous),self.assertRaisesRegex(ValueError,'identical route'):
                A.execute([guard],scope)
            for obj in (model,optimizer,scheduler,scaler,tracker): obj.load_state_dict.assert_not_called()

    def test_same_semantics_every_lr_point_and_staging_independent_resume_identity(self):
        scope={'__file__':str(A.PARENT/'contract.py')};A.execute(A.tree(A.PARENT/'contract.py').body,scope)
        for i in range(17314): self.assertEqual(C.lr_factor(i,17313),scope['lr_factor'](i,17313))
        prepared=dict(rows=1477492,rows_canonical_sha256='rows',mapping_sha256='map',source_rows=C.SOURCE_COUNTS)
        recipe=C.semantic_contract(prepared,'text','ced')
        self.assertEqual(recipe,C.semantic_contract(dict(prepared,stage_root='other_job'),'text','ced'))
        self.assertNotEqual(C.digest(recipe),C.digest({**recipe,'text_model_dir':'other-checkpoint'}))

    def test_launcher_defaults_and_syntax(self):
        text=LAUNCHER.read_text()
        self.assertIn('TEXT_MODEL_DIR='+C.DEFAULT_TEXT_MODEL_DIR,text)
        self.assertIn('mizar_s1_ced48_text_epoch1_loop_lora_r8_5090/index_cache',text)
        self.assertIn('/dev/shm/mizar_s1_gbs256_ced48_text_epoch1_loop_lora_r8_', (ROUTE/'run.sh').read_text())
        bash='C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else 'bash'
        for path in (LAUNCHER,ROUTE/'run.sh'):
            self.assertNotIn(b'\r',path.read_bytes())
            result=subprocess.run([bash,'-n',path.as_posix()],capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)


class InheritedTensorAuditNumpyTests(unittest.TestCase):
    """Execute the production equality audit using CPU NumPy tensor fixtures."""
    def fixture(self):
        class Tensor(np.ndarray):
            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
            def numpy(self): return np.asarray(self)
            def numel(self): return self.size
        state={f'model.layers.{layer}.{target}.{kind}.{loop}':np.full((2,2),.1+loop,dtype=np.float32).view(Tensor)
               for layer in range(5,15) for target in C.LOOP_LORA_CONTRACT['targets']
               for kind in ('lora_A','lora_B') for loop in (0,1)}
        embed=np.ones((2,2),dtype=np.float32).view(Tensor)
        state.update({'model.embed_tokens.weight':embed,'lm_head.weight':embed})
        source={k:v.copy() for k,v in state.items() if k!='lm_head.weight'}
        model=NS(state_dict=lambda:state,get_input_embeddings=lambda:NS(weight=embed),
                 get_output_embeddings=lambda:NS(weight=embed))
        torch=NS(float32=np.dtype('float32'),isfinite=np.isfinite,equal=np.array_equal)
        return state,source,model,torch

    def test_all_nonzero_trained_a_b_are_compared_and_audio_keys_are_excluded(self):
        state,source,model,torch=self.fixture()
        with patch.dict(sys.modules,torch=torch),patch.object(L,'checkpoint_tensors',return_value=iter(source.items())):
            audit=L.audit_inherited_text_weights(Path('/fixture'),model)
        self.assertEqual(audit['lora_tensor_count'],280)
        self.assertFalse(audit['lora_reset_after_load'])
        state['model.audio_gate.weight']=next(iter(state.values())).copy()
        with patch.dict(sys.modules,torch=torch),patch.object(L,'checkpoint_tensors',return_value=iter(source.items())):
            self.assertEqual(L.audit_inherited_text_weights(Path('/fixture'),model),audit)

    def test_a_single_reset_a_b_or_missing_branch_is_rejected(self):
        for change in ('reset_A','reset_B','missing_B','unexpected'):
            state,source,model,torch=self.fixture()
            key=next(n for n in source if ('.lora_A.' if change=='reset_A' else '.lora_B.') in n)
            if change.startswith('reset'): state[key].fill(0)
            elif change=='missing_B': del source[key]
            else: source['extra.weight']=next(iter(source.values()))
            with self.subTest(change=change),patch.dict(sys.modules,torch=torch), \
                 patch.object(L,'checkpoint_tensors',return_value=iter(source.items())),self.assertRaises(ValueError):
                L.audit_inherited_text_weights(Path('/fixture'),model)


if __name__=='__main__': unittest.main()
