"""Portable real-CPU tensor and operational regression for the isolated HRM route."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/RSmol'))
from text_hrm_h2l3 import contract as C

ML = all(importlib.util.find_spec(n) is not None for n in ('torch', 'transformers', 'safetensors', 'tokenizers', 'pyarrow'))
if ML:
    import numpy as np
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from text_hrm_h2l3.model import SmolHRMForCausalLM, convert_state, strict_load, load_model
    from text_hrm_h2l3.conversion import convert, conversion_parity, forward_audit, inspect_artifact
    from text_hrm_h2l3.train import make_optimizer, causal_loss_sum, gradient_audit, build_binding
    from text_hrm_h2l3.checkpoint import (save_checkpoint, restore_checkpoint, capture_rng,
        restore_rng, optimizer_names, validate_optimizer, ema_state_dict)
    from text_hrm_h2l3.data import ParquetStream
    torch.set_num_threads(1)


def source_model():
    torch.manual_seed(123)
    config = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=30,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=97,
        tie_word_embeddings=True, pad_token_id=0, eos_token_id=1,
        rms_norm_eps=1e-5, rope_theta=100000.0, attention_dropout=0.0,
        attention_bias=False)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).float()
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            layer.input_layernorm.weight.copy_(torch.linspace(.7, 1.4, 16) + i * .001)
            layer.post_attention_layernorm.weight.copy_(torch.linspace(1.3, .8, 16) + i * .001)
    return model


def hrm_model():
    source = source_model()
    target = SmolHRMForCausalLM(C.target_config(source.config)).float()
    strict_load(target, convert_state(source.state_dict(), target))
    return source, target


def tokenizer():
    vocab = {'[PAD]': 0, '[EOS]': 1, '[UNK]': 2} | {f'w{i}': i + 3 for i in range(94)}
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                  eos_token='[EOS]', unk_token='[UNK]')


def tick(model, optimizer, step=1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer.param_groups[0]['lr'] = C.learning_rate(step)
    ids = torch.tensor([[3, 4, 5, 6, 7], [7, 6, 5, 4, 3]])
    loss = model(input_ids=ids, labels=ids, bp_steps=C.bp_steps(step)).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


class ContractTests(unittest.TestCase):
    def test_actual_85_shard_discovery_and_training_manifest(self):
        from text_hrm_h2l3.data import expected_parquet_names, discover_parquet_files
        from text_hrm_h2l3.train import data_manifest
        with tempfile.TemporaryDirectory(prefix='hrm_manifest_') as tmp:
            root = Path(tmp)
            data = root / 'data'
            data.mkdir()
            names = expected_parquet_names()
            self.assertEqual(len(names),85)
            self.assertEqual(names[0],'train-00000-of-00085.parquet')
            self.assertEqual(names[-1],'train-00084-of-00085.parquet')
            for name in names:
                (data / name).touch()
            self.assertEqual(discover_parquet_files(root),discover_parquet_files(data))
            manifest = data_manifest(root)
            self.assertEqual(len(manifest['shards']),85)
            self.assertEqual(list(manifest['assignment']['rank_shard_counts'].values()),[11]*5+[10]*3)
            assigned = [p for paths in manifest['assignment']['rank_shards'].values() for p in paths]
            self.assertEqual(len(set(assigned)),85)
            (data / names[0]).unlink()
            with self.assertRaises(ValueError): data_manifest(root)

    def test_mapping_and_geometry(self):
        self.assertEqual(C.SOURCE_LAYERS, (0,1,2,3,4,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29))
        self.assertEqual(len(C.SCHEDULE), 50)
        self.assertEqual([C.SCHEDULE.count(i) for i in range(20)], [1]*5 + [6]*5 + [2]*5 + [1]*5)
        self.assertEqual(C.RECIPE['world_size'] * C.RECIPE['micro_batch_size'] * C.RECIPE['accumulation_steps'], 1024)

    def test_full_lr_and_bp_domains(self):
        counts = {3: 0, 4: 0, 5: 0}
        for step in range(1, 9245):
            self.assertAlmostEqual(C.learning_rate(step), min(step, 3000) * .0003 / 3000, places=18)
            expected = 3 if step <= 924 else 4 if step <= 1848 else 5
            self.assertEqual(C.bp_steps(step), expected)
            counts[expected] += 1
        self.assertEqual(counts, {3: 924, 4: 924, 5: 7396})
        for bad in (0, -1, 9245, 1.5, True):
            with self.assertRaises(ValueError): C.learning_rate(bad)
            with self.assertRaises(ValueError): C.bp_steps(bad)

    def test_original_tracked_sources_unchanged(self):
        guards = json.loads((C.HERE / 'protected_sources.json').read_text(encoding='utf-8'))
        self.assertGreater(len(guards), 3000)
        for name, expected in guards.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest(), expected, name)

    def test_copied_data_and_llama_helpers_ast(self):
        pairs = [('scripts/train_stage4_mizar_text_5_10x2_5_huginn_ddp.py', 'data.py',
                  {'expected_parquet_names','discover_parquet_files','assign_shards','_iter_shard_texts','_tokenize_ids','collate_dynamic_padding'}),
                 ('recursive_model_5_20x2_5.py', 'llama_compat.py', {'_causal_mask','_call_decoder_layer'})]
        for original, copied, names in pairs:
            a = ast.parse((ROOT / 'code/RSmol' / original).read_text(encoding='utf-8'))
            b = ast.parse((C.HERE / copied).read_text(encoding='utf-8'))
            a = {n.name: ast.dump(n) for n in a.body if isinstance(n,ast.FunctionDef) and n.name in names}
            b = {n.name: ast.dump(n) for n in b.body if isinstance(n,ast.FunctionDef) and n.name in names}
            self.assertEqual(a,b)

    def test_python310_syntax_and_no_external_asset_import(self):
        paths = list(C.HERE.glob('*.py')) + [Path(__file__), ROOT/'tests/validate_text_hrm_h2l3.py']
        for path in paths:
            source = path.read_text(encoding='utf-8')
            ast.parse(source, feature_version=(3,10))
            if path.parent == C.HERE:
                self.assertNotIn('C:/Xlance/assets', source)
        train = (C.HERE/'train.py').read_text(encoding='utf-8')
        self.assertNotIn('clip_grad_norm_(',train)
        self.assertNotIn('GradScaler(',train)

    def test_dry_run_real_cli_no_assets(self):
        result = subprocess.run([sys.executable,'-B','-m','text_hrm_h2l3.train','--dry-run',
                                 '--output-dir','/no/assets/output with spaces'],
                                cwd=ROOT/'code/RSmol',text=True,capture_output=True,check=True)
        value = json.loads(result.stdout)
        self.assertFalse(value['assets_read'])
        self.assertEqual(value['recipe']['total_steps'],9244)
        self.assertFalse(value['gpu_initialized'])


@unittest.skipUnless(ML, 'requires real Torch/Transformers/Safetensors/PyArrow/Tokenizers')
class TensorTests(unittest.TestCase):
    def test_production_model_parameter_count(self):
        config = C.target_config(LlamaConfig(**C.SOURCE_CONFIG))
        config._attn_implementation = 'sdpa'
        C.validate_config(config.to_dict())
        with torch.device('meta'):
            model = SmolHRMForCausalLM(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()),102420288)
        self.assertIs(model.model.embed_tokens.weight,model.lm_head.weight)

    def test_real_two_rank_ddp_global_token_mean_and_shared_modules(self):
        import torch.multiprocessing as mp
        if not torch.distributed.is_gloo_available():
            self.skipTest('CPU Gloo unavailable')
        with tempfile.TemporaryDirectory(prefix='hrm_ddp_') as tmp:
            root=Path(tmp)
            mp.spawn(ddp_worker,args=((root/'rendezvous').as_uri(),str(root)),nprocs=2,join=True)
            for rank in range(2):
                report=json.loads((root/f'rank{rank}.json').read_text(encoding='utf-8'))
                self.assertTrue(report['passed'])
                self.assertEqual(report['bp_tested'],[3,4,5])

    def test_nontrivial_norm_folding_and_zero_gate_parity(self):
        source, model = hrm_model()
        self.assertLess(max(conversion_parity(source,model)['layer_max_abs_errors']),3e-6)
        self.assertIs(model.model.embed_tokens.weight,model.lm_head.weight)
        for i, layer in enumerate(model.model.layers):
            if 5 <= i < 15:
                self.assertFalse(hasattr(layer.input_layernorm,'weight'))
                self.assertEqual(int(torch.count_nonzero(layer.self_attn.gate_proj.weight)),0)
            else:
                for key, value in layer.state_dict().items():
                    self.assertTrue(torch.equal(value,source.model.layers[C.SOURCE_LAYERS[i]].state_dict()[key]))

    def test_production_width_cancellation_fp64_and_exact_tensor_guard(self):
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
        from text_hrm_h2l3.model import GatedLlamaAttention, ParameterlessRMSNorm
        # Real 576/1536, 9:3 GQA block with larger learned-gamma/output scales.
        # Aliasing identical fixture blocks limits memory; the production model
        # remains independently tested for 20 physical modules/102420288 params.
        torch.manual_seed(917)
        config = LlamaConfig(**(C.SOURCE_CONFIG | {'vocab_size':97}))
        config._attn_implementation = 'sdpa'
        block = LlamaDecoderLayer(config,0).float()
        with torch.no_grad():
            block.input_layernorm.weight.copy_(torch.linspace(.7,8,576))
            block.post_attention_layernorm.weight.copy_(torch.linspace(8,.7,576))
            block.self_attn.o_proj.weight.mul_(2)
            block.mlp.down_proj.weight.mul_(2)
        middle = copy.deepcopy(block)
        middle.self_attn = GatedLlamaAttention(C.target_config(config),5)
        torch.nn.init.zeros_(middle.self_attn.gate_proj.weight)
        middle.input_layernorm = ParameterlessRMSNorm(config.rms_norm_eps)
        middle.post_attention_layernorm = ParameterlessRMSNorm(config.rms_norm_eps)
        source, target = torch.nn.Module(), torch.nn.Module()
        source.config, target.config = config, C.target_config(config)
        for model, layers in ((source,[block]*30),(target,[block]*5+[middle]*10+[block]*5)):
            model.model = torch.nn.Module()
            model.model.layers = torch.nn.ModuleList(layers)
            model.model.rotary_emb = LlamaRotaryEmbedding(config=model.config)
        target.load_state_dict(convert_state(source.state_dict(),target),strict=True)
        audit = conversion_parity(source,target)
        fallback = [r for r in audit['layers'] if 'fp64' in r]
        self.assertTrue(fallback,'fixture must reproduce the old FP32 elementwise rejection')
        self.assertGreater(max(audit['layer_max_abs_errors']),3e-6)
        for report in fallback:
            self.assertTrue(report['fp32']['passed'])
            self.assertTrue(all(value['passed'] for value in report['fp64'].values()))
        # A tiny stored-weight corruption is rejected independently of output tolerances.
        with torch.no_grad(): middle.self_attn.q_proj.weight[0,0].add_(1e-7)
        with self.assertRaisesRegex(ValueError,'tensor check failed.*self_attn.q_proj.weight'):
            conversion_parity(source,target)

    def test_conversion_parity_rejects_gate_compensation_norm_and_nonfinite(self):
        from text_hrm_h2l3.conversion import _parity_metrics
        for mode in ('gate','oproj','gamma','mapping','eps'):
            source, target = hrm_model()
            layer = target.model.layers[5]
            with torch.no_grad():
                if mode == 'gate': layer.self_attn.gate_proj.weight[0,0] = 1e-7
                if mode == 'oproj': layer.self_attn.o_proj.weight.mul_(.5)
                if mode == 'gamma': layer.mlp.up_proj.weight[:,0].mul_(1.01)
                if mode == 'mapping': layer.self_attn.k_proj.weight.copy_(source.model.layers[7].self_attn.k_proj.weight)
                if mode == 'eps': layer.input_layernorm.eps *= 2
            with self.assertRaisesRegex(ValueError,'tensor check failed'):
                conversion_parity(source,target)
        with self.assertRaises(ValueError): _parity_metrics(torch.tensor([float('nan')]),torch.ones(1))
        good = torch.ones(2,4)
        bad = good.clone(); bad[0,0] += .01
        self.assertFalse(_parity_metrics(bad,good)['passed'])

    def test_conversion_parity_rejects_wrong_forward_with_correct_tensors(self):
        from functools import wraps
        source, target = hrm_model()
        layer = target.model.layers[5]
        original_forward = layer.forward
        @wraps(original_forward)
        def wrong_forward(*args,**kwargs):
            value = original_forward(*args,**kwargs)
            return (value[0] + .01,) if isinstance(value,tuple) else value + .01
        layer.forward = wrong_forward
        with self.assertRaisesRegex(ValueError,'fold/gate parity failed at target layer 5'):
            conversion_parity(source,target)

    def test_real_trace_norm_calls_and_causality(self):
        _, model = hrm_model()
        audit, _ = forward_audit(model)
        self.assertEqual(audit['magicnorm_calls'],{'entry':1,'L':6,'H':2})
        self.assertEqual(audit['trace'],list(C.SCHEDULE))

    def test_independent_recurrence_nonzero_gates(self):
        from text_hrm_h2l3.llama_compat import _causal_mask, _call_decoder_layer
        _, model = hrm_model()
        model.eval()
        with torch.no_grad():
            for layer in model.model.layers[5:15]: layer.self_attn.gate_proj.weight.normal_(0,.2)
            ids = torch.tensor([[3,4,5,6]])
            x = model.model.embed_tokens(ids)
            pos = torch.arange(4).unsqueeze(0)
            mask = _causal_mask(attention_mask=None,batch_size=1,query_length=4,past_length=0,dtype=x.dtype,device=x.device)
            cs = model.model.rotary_emb(x,pos)
            def stack(value, indices):
                for i in indices:
                    value = _call_decoder_layer(model.model.layers[i],value,attention_mask=mask,
                        position_ids=pos,cache=None,use_cache=False,cache_position=pos[0],position_embeddings=cs,output_attentions=False)[0]
                return value
            h = model.model.entry_norm(stack(x,range(5)))
            l = model.model.zL_init
            for _ in range(2):
                for _ in range(3): l=model.model.l_exit_norm(stack(l+h,range(5,10)))
                h=model.model.h_exit_norm(stack(h+l,range(10,15)))
            expected=model.lm_head(model.model.norm(stack(h,range(15,20))))
            actual=model(input_ids=ids).logits
            self.assertTrue(torch.equal(expected,actual))

    def test_bp3_4_5_prelude_gradients_and_call_selection(self):
        _, model = hrm_model()
        ids=torch.tensor([[3,4,5,6],[6,5,4,3]])
        for k in (3,4,5):
            model.zero_grad(set_to_none=True)
            model.model.audit_gradients=True
            model(input_ids=ids,labels=ids,bp_steps=k).loss.backward()
            self.assertTrue(gradient_audit(model,k)['passed'])
            self.assertEqual(len([1 for _,on in model.model.last_calls if on]),k)
            self.assertFalse(model.model.zL_init.requires_grad)
            model.model.audit_refs.clear()

    def test_fp32_norm_under_bfloat16_autocast(self):
        _, model=hrm_model()
        with torch.autocast('cpu',dtype=torch.bfloat16):
            out=model(input_ids=torch.tensor([[3,4,5,6]]),bp_steps=5).logits
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertEqual(model.model.zL_init.dtype,torch.float32)
        self.assertTrue(all(p.dtype==torch.float32 for p in model.parameters()))

    def test_pad_eos_and_shifted_token_weighted_accumulation(self):
        from text_hrm_h2l3.data import collate_dynamic_padding
        _, model=hrm_model()
        batch=collate_dynamic_padding([[3,0,4,5],[6,7]],pad_token_id=0)
        self.assertEqual(int(batch['labels'][0,1]),0)
        self.assertEqual(int(batch['labels'][1,2]),-100)
        logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).logits
        summed=causal_loss_sum(logits,batch['labels'])
        labels=batch['labels'][:,1:]
        expected=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,97),labels.reshape(-1),ignore_index=-100,reduction='sum')
        self.assertTrue(torch.equal(summed,expected))
        model.zero_grad(set_to_none=True)
        (summed/int((labels!=-100).sum())).backward()
        full=[p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        for row in range(2):
            value=model(input_ids=batch['input_ids'][row:row+1],attention_mask=batch['attention_mask'][row:row+1]).logits
            (causal_loss_sum(value,batch['labels'][row:row+1])/4).backward()
        for a,p in zip(full,model.parameters()): self.assertTrue(torch.allclose(a,p.grad,atol=3e-6,rtol=3e-4))

    def test_attention_padding_matches_unpadded(self):
        _,model=hrm_model(); model.eval()
        with torch.no_grad():
            a=model(input_ids=torch.tensor([[3,4,5,0,0]]),attention_mask=torch.tensor([[1,1,1,0,0]])).logits[:,:3]
            b=model(input_ids=torch.tensor([[3,4,5]])).logits
        self.assertTrue(torch.allclose(a,b,atol=3e-6,rtol=3e-5))

    def test_initial_state_independent_rng_and_persistent(self):
        _,a=hrm_model(); _,b=hrm_model()
        self.assertTrue(torch.equal(a.model.zL_init,b.model.zL_init))
        before=torch.get_rng_state().clone()
        a(input_ids=torch.tensor([[3,4,5]]))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_reject_cache_gradient_checkpointing_bad_bp(self):
        _,model=hrm_model(); ids=torch.tensor([[3,4]])
        for kwargs in ({'use_cache':True},{'bp_steps':2},{'attention_mask':torch.ones(1,1,2,2)}):
            with self.assertRaises(ValueError): model(input_ids=ids,**kwargs)
        with self.assertRaises(ValueError): model.gradient_checkpointing_enable()

    def test_strict_tensor_inventory_and_tied_alias_reject(self):
        _,model=hrm_model()
        for mode in ('missing','extra','dtype','nan','tied'):
            state={k:v.clone() for k,v in model.state_dict().items()}
            key='model.layers.5.self_attn.gate_proj.weight'
            if mode=='missing': del state[key]
            if mode=='extra': state['wrong']=torch.zeros(1)
            if mode=='dtype': state[key]=state[key].bfloat16()
            if mode=='nan': state[key][0,0]=float('nan')
            if mode=='tied': state['lm_head.weight'][0,0]+=1
            with self.assertRaises(ValueError): strict_load(model,state)

    def test_actual_official_optimizer_matches_independent_numpy(self):
        from text_hrm_h2l3.adam_atan2 import AdamATan2
        p=torch.nn.Parameter(torch.tensor([.2,-.4,.6]))
        opt=AdamATan2([p],lr=0.,betas=(.9,.95),weight_decay=.1,ema=.9999)
        value=p.detach().numpy().astype(np.float64); m=np.zeros(3); v=np.zeros(3); ema=value.copy()
        for t in range(1,21):
            lr=C.learning_rate(t); g=np.array([.1*t,-.3,.2/t])
            p.grad=torch.tensor(g,dtype=torch.float32); opt.param_groups[0]['lr']=lr; opt.step()
            value*=1-lr*.1; m=.9*m+.1*g; v=.95*v+.05*g*g
            value-=lr/(1-.9**t)*np.arctan2(m,np.sqrt(v/(1-.95**t)))
            ema=.9999*ema+.0001*value
        np.testing.assert_allclose(p.detach().numpy(),value,atol=2e-7,rtol=2e-6)
        np.testing.assert_allclose(opt.state[p]['param_ema'].numpy(),ema,atol=2e-7,rtol=2e-6)

    def test_real_conversion_publish_reload_and_source_guard(self):
        with tempfile.TemporaryDirectory(prefix='hrm_convert_') as tmp:
            source=Path(tmp)/'source'; out=Path(tmp)/'converted'
            original=source_model(); original.save_pretrained(source,safe_serialization=True)
            tokenizer().save_pretrained(source)
            before=C.file_inventory(source)
            result=convert(source,out,threads=1,production=False)
            self.assertEqual(result['source_layers'],list(C.SOURCE_LAYERS))
            self.assertEqual(result['physical_layers'],20)
            self.assertEqual(result['execution_layers'],50)
            inspect_artifact(out,production=False)
            self.assertEqual(before,C.file_inventory(source))
            with self.assertRaises(ValueError): convert(source,out,production=False)
            with self.assertRaises(ValueError): inspect_artifact(out,production=True)
            model=load_model(out,production=False)
            opt=make_optimizer(model); tick(model,opt)
            trained=Path(tmp)/'trained'; model.save_pretrained(trained,safe_serialization=True)
            reloaded=load_model(trained,production=False)
            self.assertTrue(all(torch.equal(v,reloaded.state_dict()[k]) for k,v in model.state_dict().items()))
            self.assertGreater(float(model.model.layers[5].self_attn.gate_proj.weight.detach().abs().sum()),0)

    def test_real_data_cursor_replay_midbuffer_boundary_and_epoch(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory(prefix='hrm_data_') as tmp:
            files=[]
            for i in range(2):
                file=Path(tmp)/f'shard{i}.parquet'; files.append(file)
                pq.write_table(pa.table({'text':[f'w{j} w{(j+1)%80} w2' for j in range(i*31,(i+1)*31)]}),file)
            for after in (1,13,15,16,29,30,31):
                stream=ParquetStream(files,tokenizer(),rank=3,buffer_size=4)
                it=iter(stream)
                for _ in range(after): next(it)
                cursor=copy.deepcopy(stream.cursor())
                expected=[next(it)['input_ids'] for _ in range(5)]
                restored=ParquetStream(files,tokenizer(),rank=3,buffer_size=4,cursor=cursor)
                actual=iter(restored)
                self.assertTrue(all(torch.equal(a,next(actual)['input_ids']) for a in expected))
                it.close()
                actual.close()

    def test_full_checkpoint_optimizer_ema_rank_rng_next_update(self):
        with tempfile.TemporaryDirectory(prefix='hrm_resume_') as tmp:
            _,model=hrm_model(); opt=make_optimizer(model); tick(model,opt)
            binding=build_binding(initialization={'synthetic':True},data={'fixture':True},environment={'fixture':True},production=False)
            ranks={}
            for rank in range(8):
                random.seed(rank); np.random.seed(rank); torch.manual_seed(rank)
                ranks[str(rank)]=dict(rng=capture_rng(),cursor=dict(version=1,rank=rank,microbatches=64,epoch=1,shard=0,in_shard=64,shard_start_rng=random.Random(rank).getstate()))
            ckpt=save_checkpoint(tmp,model,opt,tokenizer(),step=1,binding=binding,rank_states=ranks,production=False,retain=False)
            restored=load_model(ckpt,production=False); opt2=make_optimizer(restored)
            step,cursor=restore_checkpoint(ckpt,restored,opt2,binding=binding,rank=3,production=False)
            self.assertEqual(step,1); self.assertEqual(cursor['rank'],3)
            expected_rng=ranks['3']['rng']; self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng['cpu']))
            tick(model,opt,2); tick(restored,opt2,2)
            for name,value in model.state_dict().items(): self.assertTrue(torch.equal(value,restored.state_dict()[name]),name)
            for p,q in zip(model.parameters(),restored.parameters()):
                for key in opt.state[p]: self.assertTrue(torch.equal(opt.state[p][key],opt2.state[q][key]))
            ema=load_model(ckpt/'ema',production=False)
            raw_at_save=load_model(ckpt,production=False)
            self.assertTrue(torch.equal(ema.model.zL_init,raw_at_save.model.zL_init))
            self.assertFalse(torch.equal(ema.model.layers[5].self_attn.gate_proj.weight,raw_at_save.model.layers[5].self_attn.gate_proj.weight))
            with self.assertRaises(ValueError): restore_checkpoint(ckpt,restored,opt2,binding=binding|{'environment':{'changed':True}},rank=3,production=False)
            with self.assertRaises(ValueError): inspect_artifact(ckpt,production=False)

    def test_optimizer_restore_rejects_inventory_moment_lr_step(self):
        _,model=hrm_model(); opt=make_optimizer(model); tick(model,opt)
        names=optimizer_names(model,opt)
        for mode in ('names','missing','nan','lr','step','shape'):
            saved=copy.deepcopy(opt.state_dict()); changed=copy.deepcopy(names)
            first=next(iter(saved['state']))
            if mode=='names': changed[0][0],changed[0][1]=changed[0][1],changed[0][0]
            if mode=='missing': del saved['state'][first]['param_ema']
            if mode=='nan': saved['state'][first]['exp_avg'].flatten()[0]=float('nan')
            if mode=='lr': saved['param_groups'][0]['lr']=.2
            if mode=='step': saved['state'][first]['step'].fill_(2)
            if mode=='shape': saved['state'][first]['exp_avg']=torch.zeros(1)
            with self.assertRaises(ValueError): validate_optimizer(model,opt,saved,changed,1)

    def test_real_checkpoint_retention_and_corruption_rejection(self):
        from text_hrm_h2l3.checkpoint import retain_checkpoints
        with tempfile.TemporaryDirectory(prefix='hrm_retention_') as tmp:
            root = Path(tmp)
            _,model = hrm_model()
            opt = make_optimizer(model)
            binding = build_binding(initialization={'synthetic':True},data={'fixture':True},
                environment={'fixture':True},production=False)
            for step in range(1,5):
                tick(model,opt,step)
                ranks = {str(rank):dict(rng=capture_rng(),cursor=dict(rank=rank,
                    microbatches=64*step)) for rank in range(8)}
                latest = save_checkpoint(root,model,opt,tokenizer(),step=step,binding=binding,
                    rank_states=ranks,production=False)
            self.assertEqual(sorted(p.name for p in root.glob('checkpoint-step-*')),
                [f'checkpoint-step-{step:06d}' for step in (2,3,4)])
            self.assertEqual(json.loads((root/'latest_complete_checkpoint.json').read_text())['step'],4)
            # A failed/corrupt save must never be silently treated as a complete checkpoint.
            with (latest/'training_state.pt').open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaises(ValueError): inspect_artifact(latest,resume=True,production=False)
            with self.assertRaises(ValueError): retain_checkpoints(root)
            self.assertEqual(len(list(root.glob('checkpoint-step-*'))),3)


class ShellTests(unittest.TestCase):
    def test_bash_syntax(self):
        bash=shutil.which('bash')
        if os.name=='nt': bash='C:/Program Files/Git/bin/bash.exe'
        if not bash or not Path(bash).exists(): self.skipTest('Bash unavailable')
        for file in ('scripts/convert_text_hrm_h2l3.sh','scripts/train_text_hrm_h2l3_ddp.sh','run_text_hrm_h2l3_3090.sh'):
            subprocess.run([bash,'-n',str(ROOT/'code/RSmol'/file)],check=True,capture_output=True)

    def test_real_bash_vc_runtime_argument_chain(self):
        bash='C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else shutil.which('bash')
        if not bash or not Path(bash).exists(): self.skipTest('Bash unavailable')
        with tempfile.TemporaryDirectory(prefix='hrm shell ') as tmp:
            temp=Path(tmp); binpath=temp/'bin'; binpath.mkdir()
            conda=temp/'conda base'; (conda/'etc/profile.d').mkdir(parents=True)
            (conda/'etc/profile.d/conda.sh').write_text('conda() { return 0; }\n',encoding='utf-8')
            (binpath/'python').write_text('#!/bin/bash\nexec '+shlex.quote(Path(sys.executable).as_posix())+' "$@"\n',encoding='utf-8')
            (binpath/'vc').write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$HRM_TEST_VC"\n',encoding='utf-8')
            (binpath/'torchrun').write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$HRM_TEST_TORCHRUN"\nwhile [[ "$1" != --module ]]; do shift; done\nshift\nexec python -B -m "$@" --dry-run\n',encoding='utf-8')
            for path in binpath.iterdir(): path.chmod(0o755)
            env=os.environ.copy(); env['PATH']=str(binpath)+os.pathsep+env['PATH']
            env['MSYS_NO_PATHCONV']='1'
            env['MSYS2_ARG_CONV_EXCL']='*'
            env.update(HRM_TEST_VC=str(temp/'vc.txt'),HRM_TEST_TORCHRUN=str(temp/'torchrun.txt'),
                RSMOL_HRM_CONDA_BASE=conda.as_posix(),RSMOL_MIZAR_TEXT_5_10X2_5_HUGINN_RESUME_FROM='/wrong/old/resume')
            launcher=ROOT/'code/RSmol/run_text_hrm_h2l3_3090.sh'
            result=subprocess.run([bash,str(launcher),'--model-dir','/models with spaces/hrm','--data-dir','/data with spaces',
                                   '--output-dir','/outputs with spaces/hrm'],env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            args=(temp/'vc.txt').read_text(encoding='utf-8').splitlines()
            for option,value in [('-p','pdgpu-3090'),('-c','64'),('-m','256G'),('-g','8'),('-n','1')]: self.assertEqual(args[args.index(option)+1],value)
            self.assertLess(len(args[args.index('-j')+1]),60)
            command=args[args.index('--cmd')+1]
            actual=subprocess.run([bash,'-c',command],env=env,text=True,capture_output=True)
            self.assertEqual(actual.returncode,0,actual.stderr)
            value=json.loads(actual.stdout)
            self.assertEqual(Path(value['model_dir']),Path('/models with spaces/hrm'))
            self.assertIsNone(value['resume_from'])
            self.assertEqual(value['recipe']['global_batch_size'],1024)
            self.assertIn('--nproc_per_node=8',(temp/'torchrun.txt').read_text(encoding='utf-8'))


def ddp_worker(rank, init_method, output):
    """Real two-process CPU DDP; compare production accumulation to global CE."""
    import datetime
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from text_hrm_h2l3.train import accumulate_window
    from text_hrm_h2l3.data import collate_dynamic_padding
    dist.init_process_group('gloo',init_method=init_method,rank=rank,world_size=2,
                            timeout=datetime.timedelta(seconds=90))
    try:
        _,model=hrm_model(); reference=copy.deepcopy(model)
        ddp=DistributedDataParallel(model,broadcast_buffers=False,find_unused_parameters=False)
        windows=[
            [collate_dynamic_padding([[3,4,5,6],[7,8]],pad_token_id=0),
             collate_dynamic_padding([[9,10,11],[12,13,14,15]],pad_token_id=0)],
            [collate_dynamic_padding([[16,17],[18,19,20]],pad_token_id=0),
             collate_dynamic_padding([[21,22,23,24,25],[26,27,28]],pad_token_id=0)]]
        tokens=sum(int((b['labels'][:,1:]!=-100).sum()) for w in windows for b in w)
        differences=[]
        for k in (3,4,5):
            model.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
            loss,count=accumulate_window(ddp,model,windows[rank],device=torch.device('cpu'),world_size=2,bp=k)
            expected_loss=0.0
            for window in windows:
                for batch in window:
                    value=reference(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],bp_steps=k).logits
                    local=causal_loss_sum(value,batch['labels'])
                    expected_loss+=float(local.detach())
                    (local/tokens).backward()
            if count!=tokens or abs(float(loss)-expected_loss)>2e-4:
                raise AssertionError('DDP global denominator/loss mismatch')
            for p,q in zip(model.parameters(),reference.parameters()):
                if p.grad is None or not torch.allclose(p.grad,q.grad,atol=5e-6,rtol=5e-4):
                    raise AssertionError('DDP shared-module/token-weighted gradient mismatch')
                differences.append(float((p.grad-q.grad).abs().max()))
        (Path(output)/f'rank{rank}.json').write_text(json.dumps(dict(passed=True,bp_tested=[3,4,5],
            gradient_max_abs_error=max(differences),world_size=2,backend='gloo',cuda_used=False)),encoding='utf-8')
    finally:
        dist.destroy_process_group()


if __name__=='__main__': unittest.main(verbosity=2)
