"""Actual loop/loader/diagnostics execution with explicit NumPy CPU fixtures.

This supplements the optional real Torch autograd/HF tests. It does not claim
Torch, CUDA, production weights or eight-rank execution was tested locally.
"""
import ast
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_huginn_text_epoch1"
sys.path.insert(0, str(ROUTE))
import contract
import audit_reference
from models import recursive_text
from test_mizar_s1_ced_huginn_text_epoch1_identity import text_checkpoint, fake_model, model_audit


class Tensor(np.ndarray):
    @property
    def device(self): return "cpu"
    def detach(self): return self
    def to(self, dtype=None):
        return self if dtype == "cpu" else self.astype(dtype).view(Tensor)
    def abs(self): return np.abs(self).view(Tensor)
    def mul_(self, value): self *= value


def tensor(value): return np.asarray(value, dtype=np.float32).view(Tensor)


NUMPY_TORCH = SimpleNamespace(
    tensor=tensor, float64=np.float64, isfinite=np.isfinite,
    linalg=SimpleNamespace(norm=lambda x, p: np.linalg.norm(x.flatten(), ord=p),
        vector_norm=lambda x, ord, dtype: np.linalg.norm(x.astype(dtype).flatten(), ord=ord)))


def functions(path, names, scope):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [copy.deepcopy(audit_reference.find_function(tree, name)) for name in names]
    exec(compile(ast.Module(nodes, type_ignores=[]), "actual-production-functions", "exec"), scope)
    return scope


class HuginnNormalizationTests(unittest.TestCase):
    def test_actual_loop_matches_equations_and_norm_boundaries_for_all_683_positions(self):
        tree = ast.parse((ROUTE / "models/huginn.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MizarHuginnModel")
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        loop = next(n for n in forward.body if isinstance(n, ast.For))
        rng = np.random.default_rng(41)
        weight = tensor(rng.normal(0,.2,(4,8)))
        embedding = tensor(rng.normal(0,2,(2,683,4)))
        noise = tensor(rng.normal(0,contract.HUGINN_FIELDS['huginn_state_std'],(2,683,4)))
        gain_in, gain_out = tensor([.8,1.,1.1,1.3]), tensor([1.2,.9,.7,1.])
        def norm(x, gain): return tensor(x / np.sqrt(np.mean(x*x,-1,keepdims=True)+1e-5) * gain)
        def layer(x,i): return tensor(np.tanh(.85*x + tensor([.01,.02,-.03,.04])*(i+1)))
        def stack(x, indices):
            for i in indices: x=layer(x,i)
            return x
        e = stack(embedding,range(5))
        s1 = norm(stack(norm(tensor(np.concatenate([noise,e],-1)@weight.T),gain_in),range(5,15)),gain_out)
        s2 = norm(stack(norm(tensor(np.concatenate([s1,e],-1)@weight.T),gain_in),range(5,15)),gain_out)
        expected = stack(s2,range(15,20))
        for explicit in (False,True):
            trace, norm_trace, adapter_inputs = [], [], []
            sample=Mock(return_value=noise.copy())
            def adapter(x):
                adapter_inputs.append(x.copy())
                return tensor(x@weight.T)
            def norm_in(x):
                norm_trace.append(('in',len(trace)))
                return norm(x,gain_in)
            def norm_out(x):
                norm_trace.append(('out',len(trace)))
                return norm(x,gain_out)
            def call(i,x,**kwargs):
                trace.append(i)
                return (layer(x,i),)
            scope = dict(self=SimpleNamespace(layers=list(range(20)),logical_to_physical=contract.TEXT_MODEL_SCHEDULE,
                            recurrent_adapter=adapter,recurrent_input_norm=norm_in,recurrent_output_norm=norm_out),
                         hidden_states=embedding.copy(),initial_state=noise if explicit else None,
                         sample_initial_state=sample,torch=SimpleNamespace(cat=lambda xs,dim:tensor(np.concatenate(xs,dim))),
                         output_hidden_states=False,output_attentions=False,cache=None,causal_mask=None,
                         position_ids=None,use_cache=False,cache_position=None,position_embeddings=None,_call_decoder_layer=call)
            exec(compile(ast.Module([loop], type_ignores=[]),'actual-normalized-loop','exec'),scope)
            np.testing.assert_array_equal(scope['hidden_states'],expected)
            np.testing.assert_array_equal(adapter_inputs[0],np.concatenate([noise,e],-1))
            np.testing.assert_array_equal(adapter_inputs[1],np.concatenate([s1,e],-1))
            self.assertEqual(norm_trace,[('in',5),('out',15),('in',15),('out',25)])
            self.assertEqual(trace,list(contract.TEXT_MODEL_SCHEDULE))
            self.assertEqual(sample.call_count,0 if explicit else 1)

    def test_installer_adds_two_unit_norms_and_preserves_existing_modules_and_values(self):
        class LlamaRMSNorm:
            def __init__(self,h,eps):
                self.weight=SimpleNamespace(shape=(h,),requires_grad=True,value=np.ones(h))
                self.variance_epsilon=eps
            def to(self,**kwargs): return self
        scope = functions(ROUTE/'models/huginn.py',['enable_recurrent_boundary_norms'],
                          {'BOUNDARY_NORM_CONTRACT':contract.BOUNDARY_NORM_CONTRACT})
        # Resolve only the import in the actual method through a tiny explicit fixture.
        fake_import=SimpleNamespace(LlamaRMSNorm=LlamaRMSNorm)
        model=fake_model()
        model.config.hidden_size, model.config.rms_norm_eps = 576, 1e-5
        model.model.config=model.config
        weight=model.model.recurrent_adapter.weight
        weight.device,weight.dtype='cpu','float32'
        adapter, layers = model.model.recurrent_adapter, model.model.layers
        fn=scope['enable_recurrent_boundary_norms']
        with patch.dict(sys.modules,{'transformers.models.llama.modeling_llama':fake_import}):
            fn(model.model)
            norm=model.model.recurrent_input_norm
            fn(model.model)
        self.assertIs(model.model.recurrent_adapter,adapter)
        self.assertIs(model.model.layers,layers)
        self.assertIs(model.model.recurrent_input_norm,norm)
        self.assertIsNot(norm,model.model.recurrent_output_norm)
        for n in (norm,model.model.recurrent_output_norm): np.testing.assert_array_equal(n.weight.value,np.ones(576))
        recursive_text.validate_loaded_model(model,model_audit(),require_boundary_norms=True)
        for change in ('shape','frozen','epsilon','shared'):
            candidate=copy.deepcopy(model)
            candidate.get_input_embeddings=lambda:SimpleNamespace(weight=candidate.lm_head.weight)
            if change=='shape': candidate.model.recurrent_input_norm.weight.shape=(575,)
            if change=='frozen': candidate.model.recurrent_input_norm.weight.requires_grad=False
            if change=='epsilon': candidate.model.recurrent_input_norm.variance_epsilon=1.
            if change=='shared': candidate.model.recurrent_output_norm=candidate.model.recurrent_input_norm
            with self.subTest(change=change),self.assertRaises(ValueError):
                recursive_text.validate_loaded_model(candidate,model_audit(),require_boundary_norms=True)

    def test_text_source_rejects_audio_normalized_config_and_legacy_resume_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=text_checkpoint(Path(directory)/'checkpoint-step-009244')
            raw=json.loads((path/'config.json').read_text())
            contract.write_json(path/'config.json',{**raw,'huginn_audio_boundary_norms':contract.BOUNDARY_NORM_CONTRACT})
            with self.assertRaisesRegex(ValueError,'original Huginn text checkpoint'):
                contract.validate_text_initialization(path)
        prepared=dict(rows=1477492,source_rows=contract.SOURCE_COUNTS,rows_canonical_sha256='rows',mapping_sha256='map')
        current=contract.semantic_contract(prepared,'text','ced')
        legacy={k:v for k,v in current.items() if k!='audio_boundary_normalization'}
        self.assertNotEqual(contract.digest(current),contract.digest(legacy))
        tree=ast.parse((ROUTE/'train.py').read_text())
        block=next(n for n in audit_reference.find_function(tree,'train').body if isinstance(n,ast.If)
                   and ast.unparse(n.test)=="config['resume_checkpoint']")
        state=dict(schema_version=1,route_variant=contract.ROUTE,mode='formal',contract_sha256=contract.digest(legacy))
        model=Mock()
        scope=dict(config=dict(resume_checkpoint='legacy.ckpt',contract_sha256=contract.digest(current)),
                   torch=SimpleNamespace(load=Mock(return_value=state)),ROUTE=contract.ROUTE,model=model)
        with self.assertRaisesRegex(ValueError,'identical route'):
            exec(compile(ast.Module([block],type_ignores=[]),'actual-legacy-resume','exec'),scope)
        model.load_state_dict.assert_not_called()

    def test_actual_tracker_diagnostics_preserve_original_finite_updates_and_inspect_before_clip(self):
        scope=functions(ROUTE/'gradient_diagnostics.py',['json_number','inspect_gradients'],
                        {'math':math,'torch':NUMPY_TORCH})
        def tracker(path, **extra):
            tree=ast.parse(path.read_text())
            node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GradNormTracker')
            namespace={'torch':NUMPY_TORCH,'math':math,**extra}
            exec(compile(ast.Module([node],type_ignores=[]),'actual-tracker','exec'),namespace)
            return namespace['GradNormTracker']()
        healthy=tracker(ROUTE/'optim.py',inspect_gradients=Mock(side_effect=AssertionError('diagnostics on normal step')))
        parent=tracker(audit_reference.PARENT/'optim.py')
        for values in ([3.,4.],[100.,-2.],[.001,.002]):
            a=SimpleNamespace(grad=tensor(values),requires_grad=True)
            b=SimpleNamespace(grad=tensor(values),requires_grad=True)
            self.assertEqual(healthy.track_and_clip_([('p',a)]),parent.track_and_clip_([('p',b)]))
            np.testing.assert_array_equal(a.grad,b.grad)
            self.assertEqual(healthy.state_dict(),parent.state_dict())
            self.assertIsNone(healthy.last_anomaly)
        for value,reason in ((1e6,'finite_gradient_norm_reached_emergency_threshold'),
                             (1e20,'fp32_norm_overflow_with_finite_gradients'),
                             (float('inf'),'nonfinite_gradient_values')):
            candidate=tracker(ROUTE/'optim.py',inspect_gradients=scope['inspect_gradients'])
            parameter=SimpleNamespace(grad=tensor([value,value]),requires_grad=True)
            with np.errstate(over='ignore',invalid='ignore'):
                candidate.track_and_clip_([('test.weight',parameter)])
            report=candidate.last_anomaly
            self.assertEqual(report['reason'],reason)
            self.assertEqual(report['inspection_point'],'after_unscale_before_clipping')
            json.dumps(report,allow_nan=False)
            if math.isfinite(value):
                self.assertTrue(math.isclose(report['gradient_l2_fp64_before_clip'],math.sqrt(2)*float(np.float32(value)),rel_tol=1e-12))
                self.assertEqual(report['largest_gradients_fp64'][0]['max_abs'],float(np.float32(value)))

    def test_failure_block_writes_rank_and_step_with_original_diagnostic(self):
        tree=ast.parse((ROUTE/'train.py').read_text())
        block=next(n for n in ast.walk(audit_reference.find_function(tree,'train')) if isinstance(n,ast.If)
                   and ast.unparse(n.test)=='tracker.last_anomaly is not None')
        with tempfile.TemporaryDirectory() as directory:
            scope=dict(tracker=SimpleNamespace(last_anomaly={'reason':'finite_gradient_norm_reached_emergency_threshold'}),
                       rank=2,step=9,scaler=SimpleNamespace(get_scale=lambda:65536.),update_losses=[np.float32(2.)],
                       json_number=float,lr_used=[3e-4],update_rows=[17],update_sources=['Clotho'],
                       write_json=contract.write_json,run_dir=Path(directory),json=json,print=Mock())
            exec(compile(ast.Module([block],type_ignores=[]),'actual-failure-report','exec'),scope)
            report=json.loads((Path(directory)/'gradient_failure_rank2_step000010.json').read_text())
            self.assertEqual((report['rank'],report['optimizer_step'],report['completed_optimizer_steps']),(2,10,9))
            self.assertEqual(report['grad_scaler'],65536.)
            self.assertEqual(report['microbatch_losses'],[2.])


if __name__=='__main__': unittest.main()
