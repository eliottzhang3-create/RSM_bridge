"""Execute transfer/loader/identity functions with explicit CPU fixtures."""
from __future__ import annotations

import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from text_x3_test_support import ROOT, BASE, helpers
from text_x3_router_transfer_support import Tensor, donor_metadata, provenance, source_tensors, transfer

ROUTE='5_10x3_5_mesh_6slot'


class RouterTransferTests(unittest.TestCase):
    def test_python_defaults_and_explicit_router_source_are_separate_from_clean_lm(self):
        converter=helpers(ROUTE,converter=True)
        args=converter.parse_args([])
        self.assertEqual(args.router_checkpoint,transfer.DEFAULT_ROUTER_CHECKPOINT)
        self.assertTrue(args.source_checkpoint.as_posix().endswith('models/SmolLM2-5-10-5'))
        chosen=converter.parse_args(['--router-checkpoint','router donor with spaces','--source-checkpoint','clean LM'])
        self.assertEqual(chosen.router_checkpoint,Path('router donor with spaces'))
        self.assertEqual(chosen.source_checkpoint,Path('clean LM'))

    def test_exact_eight_router_mapping(self):
        mapping=transfer.router_tensor_mapping()
        self.assertEqual(len(mapping),16)
        for kind,groups in (('write',(0,1,1,2)),('read',(0,0,1,2))):
            for index,original in enumerate(groups):
                for part in ('weight','bias'):
                    self.assertEqual(mapping[f'model.{kind}_routers.{index}.{part}'],f'model.{kind}_routers.{original}.{part}')

    def test_donor_json_and_file_identity_never_reads_training_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=donor_metadata(Path(tmp)/'checkpoint-009244')
            original=Path.open
            def checked_open(path,*args,**kwargs):
                if path.name=='training_state.pt': raise AssertionError('text training state was read')
                return original(path,*args,**kwargs)
            with patch.object(Path,'open',checked_open):
                result=transfer.validate_router_checkpoint(path,16)
            self.assertEqual(result['optimizer_step'],9244)
            self.assertEqual(result['weight_files'][0]['name'],'model.safetensors')
            self.assertEqual(len(result['files_sha256']),4)

    def test_wrong_slots_architecture_step_or_incomplete_donor_rejected(self):
        cases=[('config.json','mesh_memory_slots',7),('config.json','hidden_size',32),
               ('config.json','recursive_loops',3),('mesh_checkpoint_metadata.json','optimizer_step',9000),
               ('mesh_checkpoint_metadata.json','training_contract','wrong'),
               ('checkpoint_complete.json', 'complete_marker', False), ('checkpoint_manifest.json', 'files', ["../escape"])]
        for name,key,value in cases:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                path=donor_metadata(Path(tmp)/'checkpoint-009244')
                record=json.loads((path/name).read_text()); record[key]=value
                (path/name).write_text(json.dumps(record))
                with self.assertRaises(ValueError): transfer.validate_router_checkpoint(path,16)

    def test_safetensors_reader_fetches_only_twelve_router_tensors(self):
        converter=helpers(ROUTE,converter=True); tensors=source_tensors(); fetched=[]
        class Handle:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def keys(self): return ['model.layers.0.mlp.up_proj.weight','lm_head.weight',*tensors]
            def get_tensor(self,name): fetched.append(name); return tensors[name]
        with patch.dict(sys.modules,{'safetensors':NS(safe_open=lambda *args,**kwargs:Handle())}):
            actual=converter._load_router_weights(Path('/synthetic'),{'weight_files':[{'name':'model.safetensors'}],'hidden_size':16})
        self.assertEqual(set(fetched),set(tensors)); self.assertEqual(len(fetched),12)
        self.assertEqual(set(actual),set(tensors))

    def test_missing_extra_duplicate_bad_shape_dtype_or_nonfinite_router_rejected(self):
        converter=helpers(ROUTE,converter=True)
        for fault in ('missing','extra','shape','dtype','nonfinite'):
            tensors=source_tensors(); name=next(iter(tensors))
            if fault=='missing': tensors.pop(name)
            if fault=='extra': tensors['model.read_routers.3.bias']=Tensor(np.zeros(6))
            if fault=='shape': tensors[name]=Tensor(np.zeros((1,)))
            if fault=='dtype': tensors[name].dtype='torch.bfloat16'
            if fault=='nonfinite': tensors[name].array.flat[0]=np.nan
            with self.subTest(fault=fault),self.assertRaises(ValueError): transfer.validate_router_tensors(tensors,16)
        tensors=source_tensors()
        class Handle:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def keys(self): return list(tensors)
            def get_tensor(self,name): return tensors[name]
        with patch.dict(sys.modules,{'safetensors':NS(safe_open=lambda *args,**kwargs:Handle())}):
            with self.assertRaisesRegex(ValueError,'duplicate'):
                converter._load_router_weights(Path('/synthetic'),dict(hidden_size=16,weight_files=[{'name':'a'},{'name':'b'}]))

    def test_real_copy_function_preserves_values_and_all_target_storage_independent(self):
        converter=helpers(ROUTE,converter=True); source=source_tensors()
        params={k:Tensor(np.zeros((6,16) if k.endswith('.weight') else (6,))) for k in transfer.router_tensor_mapping()}
        params['model.layers.0.weight']=Tensor(np.full((16,16),7.))
        model=NS(config=NS(hidden_size=16),get_parameter=lambda name:params[name],named_parameters=lambda:params.items())
        with patch.dict(sys.modules,{'torch':NS(no_grad=nullcontext)}):
            record=converter._transfer_routers(model,source,provenance()['source'])
        for target,original in transfer.router_tensor_mapping().items():
            np.testing.assert_array_equal(params[target].array,source[original].array)
        np.testing.assert_array_equal(params['model.layers.0.weight'].array,np.full((16,16),7.))
        self.assertEqual(len({params[k].untyped_storage().data_ptr() for k in transfer.router_tensor_mapping()}),16)
        params['model.write_routers.2.weight'].array+=1
        np.testing.assert_array_equal(params['model.write_routers.1.weight'].array,source['model.write_routers.1.weight'].array)
        with self.assertRaises(ValueError): transfer.validate_materialized_routers(model,record)

    def test_materialized_alias_or_frozen_router_rejected_even_when_values_match(self):
        converter=helpers(ROUTE,converter=True)
        for failure in ('alias','frozen'):
            params={k:Tensor(np.zeros((6,16) if k.endswith('.weight') else (6,))) for k in transfer.router_tensor_mapping()}
            model=NS(config=NS(hidden_size=16),get_parameter=lambda name:params[name],named_parameters=lambda:params.items())
            with patch.dict(sys.modules,{'torch':NS(no_grad=nullcontext)}):
                record=converter._transfer_routers(model,source_tensors(),provenance()['source'])
            if failure=='alias': params['model.write_routers.2.weight']=params['model.write_routers.1.weight']
            else: params['model.write_routers.1.weight'].requires_grad=False
            with self.subTest(failure=failure),self.assertRaises(ValueError):
                transfer.validate_materialized_routers(model,record)

    def test_tampered_transfer_mapping_hashes_or_random_fallback_rejected(self):
        good=provenance(); transfer.validate_transfer_provenance(good)
        for key,value in (('contract','random'),('tensor_mapping',{}),('source_tensor_sha256',{}),
                          ('independent_target_storage',False),('load_source_lm',True),('random_router_fallback',True)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                transfer.validate_transfer_provenance(dict(good,**{key:value}))
        bad=copy.deepcopy(good); bad['target_tensor_sha256']['model.read_routers.1.bias']='0'*64
        with self.assertRaises(ValueError): transfer.validate_transfer_provenance(bad)

    def test_existing_model_pure_route_and_donor_sources_unchanged(self):
        guard=json.loads((BASE/'text_5_10x3_5_mesh_6slot/router_transfer_source_guard.json').read_text())
        self.assertEqual(len(guard),28)
        for path,expected in guard.items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),expected,path)


if __name__=='__main__': unittest.main()
