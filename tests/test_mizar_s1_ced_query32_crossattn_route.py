"""Stdlib route, strict-loading, optimizer, provenance and Bash submission checks."""
import ast
import copy
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
NAME = "mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_query32_crossattn"
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine" / NAME
sys.path.insert(0, str(ROUTE))
import audit_parent
import contract
from models import recursive_text


def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<actual-route-functions>", "exec"), scope)


def bash_path(path):
    value = path.as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


class Param:
    requires_grad = True
    def __init__(self, shape):
        self.shape, self.ndim = shape, len(shape)
    def numel(self):
        return math.prod(self.shape)


class RouteTests(unittest.TestCase):
    def test_parent_source_and_complete_backbone_parity(self):
        result = audit_parent.audit()
        self.assertGreaterEqual(len(result["identical_files"]), 13)
        self.assertTrue(result["backbone_classes_ast_equal_after_additions"])
        self.assertTrue(result["original_mapper_ast_equal"])
        self.assertTrue(result["complete_epoch_update_loop_ast_equal_after_diagnostics"])

    def test_unchanged_formal_geometry_and_precise_architecture(self):
        g = contract.geometry(sum(contract.SOURCE_COUNTS.values()))
        self.assertEqual((g["steps_per_epoch"],g["total_steps"],g["warmup_steps"],g["dropped_rows_per_epoch"]),
                         (5771,17313,866,116))
        self.assertEqual(contract.lr_factor(0, g["total_steps"]), .1)
        arch = contract.cross_attention_contract()
        self.assertEqual(arch["ced_blocks_0based"], [2,5,8,11])
        self.assertEqual(arch["logical_indices_0based"], [5,15])
        self.assertEqual(arch["new_parameters"], 6241733)
        self.assertEqual(arch["gate1"]["scope"], "sample")
        self.assertEqual(arch["gate2"]["scope"], "text_token")
        self.assertEqual(arch["round_bias"]["applies_to"], "Q_input_only")
        self.assertTrue(contract.ROUTE.endswith("_query32_crossattn_v1"))
        self.assertEqual(arch["normalization"]["eps"], 1e-5)
        self.assertFalse(arch["normalization"]["output"]["elementwise_affine"])
        self.assertEqual(arch["residual_scale"], .3)
        self.assertFalse(arch["residual_scale_trainable"])
        self.assertEqual(arch["residual_base"], "original_unnormalized_text")
        self.assertFalse(arch["cross_attention"]["out_proj_zero_init"])
        compression = arch['query_compression']
        self.assertEqual(compression['bank_shape'], [4,32,576])
        self.assertTrue(compression['shared_queries_across_depths'])
        self.assertTrue(compression['shared_attention_across_depths'])
        self.assertFalse(compression['query_residual'])
        self.assertFalse(compression['memory_norm']['elementwise_affine'])

    def test_parent_recipe_equal_except_explicit_architecture_lr_and_identity(self):
        prepared = dict(rows=1477492, source_rows=contract.SOURCE_COUNTS,
                        rows_canonical_sha256="rows", mapping_sha256="mapping", stage_root="node_a")
        candidate = contract.semantic_contract(prepared, "text", "ced")
        scope = {"__file__": str(audit_parent.PARENT/"contract.py")}
        execute(audit_parent.tree(audit_parent.PARENT/"contract.py").body, scope)
        parent = scope["semantic_contract"](prepared, "text", "ced")
        self.assertEqual({k for k in candidate if candidate[k] != parent.get(k)},
                         {"route","code_sha256","recursive_model_source_sha256","audio_cross_attention","adamw"})
        self.assertEqual(candidate["adamw"], dict(parent["adamw"], new_parameters_lr=6e-4))
        prepared["stage_root"] = "another_node"
        self.assertEqual(candidate, contract.semantic_contract(prepared, "text", "ced"))
        self.assertNotEqual(contract.digest(parent), contract.digest(candidate))
        self.assertEqual(contract.recursive_model_source_sha256(), hashlib.sha256(
            (ROUTE/"models/recursive_crossattn.py").read_bytes().replace(b"\r\n",b"\n")).hexdigest())

    def parameter_inventory(self):
        pairs = [("audio_encoder.base.mapper.weight",Param((576,384))),
                 ("audio_encoder.base.mapper.bias",Param((576,))),
                 ("caption_decoder.lm.model.layers.5.weight",Param((576,576))),
                 ("caption_decoder.lm.model.layers.5.bias",Param((576,)))]
        prefix = "caption_decoder.lm.model.audio_crossattn."
        for i in range(4):
            pairs.extend([(prefix+f"projections.{i}.weight",Param((576,1536))),
                          (prefix+f"projections.{i}.bias",Param((576,)))])
            for component,width in (("audio_input_norms",1536),("audio_output_norms",576)):
                pairs.extend([(prefix+f"{component}.{i}.weight",Param((width,))),
                              (prefix+f"{component}.{i}.bias",Param((width,)))])
        pairs.extend((prefix+name,Param(shape)) for name,shape in (
            ("gate1.weight",(4,576)),("gate1.bias",(4,)),("gate2.weight",(1,576)),("gate2.bias",(1,)),
            ("cross_attention.in_proj_weight",(1728,576)),("cross_attention.in_proj_bias",(1728,)),
            ("cross_attention.out_proj.weight",(576,576)),("cross_attention.out_proj.bias",(576,)),
            ("round_bias.0",(576,)),("round_bias.1",(576,)),
            ("text_norm.weight",(576,)),("text_norm.bias",(576,)),
            ("query_tokens",(32,576)),("query_norm.weight",(576,)),("query_norm.bias",(576,)),
            ("compression_attention.in_proj_weight",(1728,576)),("compression_attention.in_proj_bias",(1728,)),
            ("compression_attention.out_proj.weight",(576,576)),("compression_attention.out_proj.bias",(576,))))
        return pairs

    def test_actual_optimizer_all_new_parameters_and_mapper_6e4_original_lm_3e4(self):
        class Optimizer:
            def __init__(self, groups, **kwargs):
                self.param_groups,self.kwargs = groups,kwargs
        stub = SimpleNamespace(optim=SimpleNamespace(AdamW=Optimizer,
            lr_scheduler=SimpleNamespace(LambdaLR=lambda opt, fn: (opt,fn))))
        scope = dict(torch=stub,lr_factor=contract.lr_factor)
        nodes = [audit_parent.function(audit_parent.tree(ROUTE/"optim.py"),name)
                 for name in ("parameter_role","build_optimizer")]
        execute(nodes,scope)
        pairs = self.parameter_inventory()
        optimizer,_,audit = scope["build_optimizer"](SimpleNamespace(named_parameters=lambda: iter(pairs)),17313)
        self.assertEqual(optimizer.kwargs["betas"],(.9,.999))
        self.assertEqual(optimizer.kwargs["eps"],1e-8)
        self.assertEqual(sum(g["numel"] for g in audit if g["role"].startswith("crossattn_")),6241733)
        seen=[]
        for group in audit:
            self.assertEqual(group["base_lr"],3e-4 if group["role"] == "default" else 6e-4)
            for name in group["names"]:
                param=dict(pairs)[name]
                self.assertEqual(group["weight_decay"],0 if param.ndim<=1 or name.endswith((".bias","_bias")) else .01)
                seen.append(name)
        self.assertCountEqual(seen,[n for n,_ in pairs])
        norm_names = [n for n,_ in pairs if any(
            component in n for component in ("audio_input_norms", "audio_output_norms", "text_norm"))]
        self.assertEqual(len(norm_names), 18)
        self.assertEqual(sum(dict(pairs)[name].numel() for name in norm_names), 18048)
        self.assertTrue(all(g["base_lr"] == 6e-4 and g["weight_decay"] == 0.
                            for g in audit if any(n in norm_names for n in g["names"])))
        for bad in ("audio_encoder.base.ced.blocks.0.weight","caption_decoder.lm.model.audio_crossattn.typo.weight",
                    "caption_decoder.lm.model.audio_crossattn.residual_scale",
                    "caption_decoder.lm.model.audio_crossattn.memory_norm.weight",
                    "caption_decoder.lm.model.audio_crossattn.output_norm.weight"):
            with self.assertRaises(ValueError):
                scope["parameter_role"](bad)

    def test_strict_load_and_audit_precede_new_registration_and_failures_propagate(self):
        config=dict(model_type="llama",architectures=["RecursiveLlamaForCausalLM"],hidden_size=576,
            num_hidden_layers=30,recursive_layer_count=20,recursive_loops=2,recursive_prefix_layer_count=5,
            recursive_middle_layer_count=10,recursive_suffix_layer_count=5,recursive_loops_scope="middle_only",
            logical_to_physical=list(contract.TEXT_MODEL_SCHEDULE),vocab_size=32,eos_token_id=0,
            recursive_source_layer_indices_0based=list(contract.TEXT_MODEL_SOURCE_MAPPING))
        audit=dict(logical_layer_count=30,physical_layer_count=20,recursive_layer_count=20,recursive_loops=2,
                   middle_recurrent_count=10,model_physical_module_count=20,schedule=list(contract.TEXT_MODEL_SCHEDULE))
        for failure in (None,"missing_keys","unexpected_keys","mismatched_keys","error_msgs"):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                (Path(directory)/"config.json").write_text("{}")
                events=[]
                core=SimpleNamespace(layers=[object() for _ in range(20)],logical_to_physical=contract.TEXT_MODEL_SCHEDULE,
                                     enable_audio_cross_attention=lambda: events.append("register"))
                model=SimpleNamespace(model=core,config=SimpleNamespace(to_dict=lambda: config),
                    generation_config=SimpleNamespace(use_cache=True),named_parameters=lambda:iter([]),named_buffers=lambda:iter([]))
                cls=Mock()
                cls.from_pretrained.return_value=model,{} if failure is None else {failure:["bad"]}
                def parameter_audit(candidate):
                    events.append("strict_audit")
                    return audit
                with patch.object(recursive_text,"_import_recursive_model",return_value=(cls,parameter_audit)), \
                     patch.object(recursive_text,"validate_audio_cross_attention",return_value={}):
                    if failure:
                        with self.assertRaisesRegex(ValueError,"weights do not match"):
                            recursive_text.load_recursive_text_model(directory)
                        self.assertEqual(events,[])
                    else:
                        self.assertIs(recursive_text.load_recursive_text_model(directory),model)
                        self.assertEqual(events,["strict_audit","register"])
                        self.assertFalse(model.config.use_cache)
                        self.assertFalse(model.generation_config.use_cache)

    def test_actual_module_audit_checks_every_projection_bias_and_parameter_count(self):
        prefix="caption_decoder.lm.model.audio_crossattn."
        params={n[len(prefix):]:p for n,p in self.parameter_inventory() if n.startswith(prefix)}
        adapter=SimpleNamespace(width=576,text_start=127,audio_tokens=126,prompt_tokens=256,memory_tokens=32,
            residual_scale=.3, cross_attention=SimpleNamespace(num_heads=9,dropout=0.,batch_first=True),
            compression_attention=SimpleNamespace(embed_dim=576,num_heads=9,dropout=0.,batch_first=True),
            named_parameters=lambda:iter(params.items()))
        def norm(width, affine=True):
            return SimpleNamespace(normalized_shape=(width,),eps=1e-5,elementwise_affine=affine)
        adapter.audio_input_norms=[norm(1536) for _ in range(4)]
        adapter.audio_output_norms=[norm(576) for _ in range(4)]
        adapter.text_norm,adapter.output_norm=norm(576),norm(576,False)
        adapter.query_norm,adapter.memory_norm=norm(576),norm(576,False)
        model=SimpleNamespace(model=SimpleNamespace(audio_crossattn=adapter,layers=list(range(20)),logical_to_physical=contract.TEXT_MODEL_SCHEDULE))
        self.assertEqual(recursive_text.validate_audio_cross_attention(model)["actual_parameters"],6241733)
        for attribute,bad_value in (("residual_scale",1.),("residual_scale",.1),("residual_scale",Param(())),
                                    ("output_norm",norm(576)),("text_norm",norm(1536)),
                                    ("memory_tokens",126),("memory_norm",norm(576)),
                                    ("query_norm",norm(576,False)),
                                    ("compression_attention",adapter.cross_attention),
                                    ("audio_input_norms",[norm(1536)]*4),
                                    ("audio_output_norms",[norm(576)]*3)):
            original=getattr(adapter,attribute)
            setattr(adapter,attribute,bad_value)
            with self.subTest(attribute=attribute),self.assertRaises(ValueError):
                recursive_text.validate_audio_cross_attention(model)
            setattr(adapter,attribute,original)
        adapter.text_norm.eps=1e-6
        with self.assertRaisesRegex(ValueError,"epsilon"):
            recursive_text.validate_audio_cross_attention(model)
        adapter.text_norm.eps=1e-5
        params["projections.3.weight"]=Param((576,384))
        with self.assertRaisesRegex(ValueError,"inventory"):
            recursive_text.validate_audio_cross_attention(model)

    def test_formal_entry_and_tmpfs_ownership_prefix_are_consistent(self):
        launcher=(ROOT/"code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer36912_query32_crossattn_5090.sh").read_text()
        runner=(ROUTE/"run.sh").read_text()
        self.assertIn("mizar_s1_ced_small_gbs256_recursive_5_10_5_layer36912_query32_crossattn_5090/index_cache",launcher)
        self.assertIn("/dev/shm/mizar_s1_gbs256_ced_layer36912_query32_crossattn_",runner)
        self.assertIn("run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer36912_query32_crossattn_5090.sh",runner)
        self.assertNotIn("--mode smoke",launcher+runner)
        self.assertNotIn("smoke-report",launcher+runner)
        self.assertIn("--nproc_per_node=8",runner)
        self.assertIn('conda activate mellow_c8204d8',runner)

    def test_full_resume_identity_state_and_cursor_are_required(self):
        g = contract.geometry(1477492)
        cfg = {"mode": "formal", "contract": {"synthetic": True}}
        cfg["contract_sha256"] = contract.digest(cfg["contract"])
        payload = dict(schema_version=1, route_variant=contract.ROUTE, mode="formal",
            contract=cfg["contract"], contract_sha256=cfg["contract_sha256"],
            state_dict={}, optimizer={}, scheduler={"last_epoch": 3}, grad_scaler={},
            grad_norm_tracker={}, random_state_by_rank=[dict.fromkeys(("python","numpy","torch","cuda")) for _ in range(8)],
            total_step=3, epoch=0, next_optimizer_step=3, num_epochs=3,
            loss_reduction="mean_of_rank_local_microbatch_token_means_then_DDP_mean",
            batch_geometry=dict(world_size=8,microbatch_size=8,gradient_accumulation_steps=4,global_batch_size=256))
        contract.validate_resume_payload(payload,cfg,g)
        mutations = [("route_variant", "parent"), ("route_variant", contract.ROUTE[:-1]+"2"),
                     ("route_variant", "ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer36912_crossattn_v2"),
                     ("contract_sha256", "wrong"),
                     ("contract", {"synthetic": False}), ("random_state_by_rank", [{}]*8),
                     ("random_state_by_rank", payload["random_state_by_rank"][:7]),
                     ("scheduler", {"last_epoch": 2}), ("next_optimizer_step", 4),
                     ("total_step", -1), ("total_step", True), ("num_epochs", 2),
                     ("loss_reduction", "global_token_mean"), ("batch_geometry", {})]
        for field,value in mutations:
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.validate_resume_payload(dict(payload,**{field:value}),cfg,g)
        for field in payload:
            bad=dict(payload);bad.pop(field)
            with self.subTest(missing=field),self.assertRaises(ValueError):
                contract.validate_resume_payload(bad,cfg,g)
        completed=dict(payload,total_step=17313,epoch=3,next_optimizer_step=0,scheduler={"last_epoch":17313})
        with self.assertRaisesRegex(ValueError,"already-completed"):
            contract.validate_resume_payload(completed,cfg,g)

    def test_tensor_entry_exists_and_formal_does_not_automatically_execute_it(self):
        self.assertTrue((ROOT/"tests/test_mizar_s1_ced_query32_crossattn_tensor.py").is_file())
        self.assertTrue((ROOT/"tests/validate_mizar_s1_ced_query32_crossattn.py").is_file())
        runner=(ROUTE/"run.sh").read_text()
        self.assertNotIn("python runtime_checks.py",runner)
        trainer=audit_parent.tree(ROUTE/"train.py")
        body=audit_parent.function(trainer,"train")
        expression=ast.unparse(body)
        self.assertLess(expression.index("validate_resume_payload(saved, config, geometry)"),
                        expression.index("model.load_state_dict(saved['state_dict'], strict=True)"))
        self.assertIn("opened_gate_audit_deadline = min(step + 32, max_steps)",expression)


# Reuse the existing executable Bash fresh/resume/error-path fixture, changing
# only the route strings. The fixture invokes the real new submission script.
base=ast.parse((ROOT/"tests/test_mizar_s1_ced_audio_reinject_route.py").read_text(encoding="utf-8"))
shell_test=copy.deepcopy(audit_parent.function(audit_parent.function(base,"CEDRouteTests"),
    "test_launcher_resources_arguments_and_rejections_before_submit"))
class NewRouteStrings(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            node.value=node.value.replace("ced_small_audio_reinject","ced_small_layer36912_query32_crossattn")
        return node
shell_test=NewRouteStrings().visit(shell_test)
execute([shell_test],globals())
setattr(RouteTests,shell_test.name,globals()[shell_test.name])


if __name__ == "__main__":
    unittest.main()
