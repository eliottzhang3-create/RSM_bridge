"""Optional real-tensor checks; run in remote rsmol, no production assets needed."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code/RSmol"
HAS_ML = all(importlib.util.find_spec(name) is not None for name in ("torch","transformers","safetensors"))


@unittest.skipUnless(HAS_ML, "requires torch, transformers and safetensors in remote rsmol")
class SixSlotTensorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(CODE))
        sys.path.insert(0,str(CODE / "scripts"))
        import torch
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        import convert_stepwise_5_10x2_5_mesh_6slot as converter
        import recursive_model_5_10x2_5_mesh_6slot as model_module
        import train_stage4_5_10x2_5_mesh_6slot_ddp as trainer
        cls.torch, cls.converter, cls.model_module, cls.trainer = torch,converter,model_module,trainer
        cls.temp = tempfile.TemporaryDirectory(prefix="mesh-six-slot-tensors-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.source = Path(cls.temp.name) / "source"
        cls.target = Path(cls.temp.name) / "converted"
        torch.manual_seed(17)
        config = LlamaConfig(num_hidden_layers=30,hidden_size=32,intermediate_size=64,num_attention_heads=4,num_key_value_heads=2,vocab_size=64,max_position_embeddings=128,tie_word_embeddings=True)
        config._attn_implementation = "eager"
        cls.original = LlamaForCausalLM(config)
        cls.original.save_pretrained(cls.source,safe_serialization=True)
        cls.report = converter.convert(types.SimpleNamespace(source_checkpoint=cls.source,output_dir=cls.target,seed=0,allow_training_state=False,allow_overwrite=False))

    def load_model(self):
        model,info = self.model_module.RecursiveLlamaForCausalLM.from_pretrained(self.target,local_files_only=True,torch_dtype=self.torch.float32,output_loading_info=True)
        self.trainer._validate_loading_info(info)
        return model

    def test_conversion_inherits_exact_weights_and_reloads_six_slot_routers(self):
        model = self.load_model()
        self.assertTrue(self.report["inherited_weights_exact"])
        for target_index,source_index in enumerate(self.model_module.SOURCE_LAYER_INDICES_0BASED):
            actual = model.model.layers[target_index].state_dict()
            expected = self.original.model.layers[source_index].state_dict()
            self.assertEqual(set(actual),set(expected))
            for key in actual:
                self.assertTrue(self.torch.equal(actual[key],expected[key]),(target_index,key))
        self.assertTrue(self.torch.equal(model.model.embed_tokens.weight,self.original.model.embed_tokens.weight))
        self.assertEqual(model.lm_head.weight.data_ptr(),model.model.embed_tokens.weight.data_ptr())
        routers = [*model.model.write_routers,*model.model.read_routers]
        self.assertEqual(len({id(router) for router in routers}),6)
        for router in routers:
            self.assertEqual(tuple(router.weight.shape),(6,32))
            self.assertTrue(self.torch.equal(router.bias,self.torch.zeros(6)))
        self.trainer._validate_model_audit(self.model_module.parameter_audit(model))

    def test_real_forward_memory_schedule_and_both_loop_gradients(self):
        torch = self.torch
        model = self.load_model().train()
        model.model.audit_mode = True
        model.model.gradient_audit_mode = True
        ids = torch.randint(0,64,(2,7))
        out = model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids,use_cache=False)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        core = model.model
        self.assertEqual(core.last_memory_shape,(2,6,7,32))
        self.assertTrue(torch.equal(core.last_initial_memory[:,0],core.embed_tokens(ids)))
        self.assertEqual(torch.count_nonzero(core.last_initial_memory[:,1:]).item(),0)
        self.assertEqual([entry["physical_index"] for entry in core.last_forward_trace],list(self.model_module.LOGICAL_TO_PHYSICAL))
        self.assertEqual(len(core.last_core_input_refs),2)
        self.assertEqual(len(core.last_core_output_refs),2)
        for value in [*core.last_core_input_refs,*core.last_core_output_refs]:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
        for name,parameter in model.named_parameters():
            if "routers." in name:
                self.assertIsNotNone(parameter.grad,name)
                self.assertTrue(torch.isfinite(parameter.grad).all(),name)

    def test_real_checkpoint_roundtrip_preserves_optimizer_and_rank_rng(self):
        torch = self.torch
        model = self.load_model()
        with tempfile.TemporaryDirectory() as folder:
            config = self.trainer.Stage4Config(output_dir=Path(folder),world_size=1,max_optimizer_steps=10,scheduler_total_steps=10,warmup_steps=1)
            optimizer,_ = self.trainer._optimizer(model,config)
            ids = torch.randint(0,64,(1,4))
            model(input_ids=ids,labels=ids,use_cache=False).loss.backward()
            optimizer.step()
            class Tokenizer:
                def save_pretrained(self,path):
                    (path / "tokenizer_config.json").write_text(json.dumps({"fixture":True}),encoding="utf-8")
            rng = self.trainer._capture_rng_state(torch.device("cpu"))
            checkpoint = self.trainer._checkpoint(model,Tokenizer(),optimizer,10,config,0,{"manifest":[],"data_cursors_by_rank":{"0":{"row_offset":40}},"rng_states_by_rank":{"0":rng}})
            state = self.trainer._load_checkpoint_state(checkpoint)
            self.assertTrue(state["optimizer"]["state"])
            self.assertEqual(state["data_cursors_by_rank"]["0"]["row_offset"],40)
            self.assertTrue(torch.equal(state["rng_states_by_rank"]["0"]["torch_cpu"],rng["torch_cpu"]))


if __name__ == "__main__":
    unittest.main()
