"""Source parity and original-file protection; no weights or GPU execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from contract import (SOURCE_COUNTS, LOOP_LORA_CONTRACT, SERIAL_INJECTION_CONTRACT,
                      TEXT_MEMORY_CONTRACT, code_digest, semantic_contract, write_json)
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
PARENT = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject_loop_lora_r8")
SERIAL = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16")
BASELINE = REPO/"audit_reports/mizar_s1_textmesh_ced_loop_lora_r16_parent_baseline.json"

def tree(path): return ast.parse(path.read_text(encoding="utf-8-sig"))
def find_function(module, name):
    found = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1: raise ValueError(f"expected one function {name}, got {len(found)}")
    return found[0]
def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),
                 "textmesh-source-audit", "exec"), scope)
def equal(actual, expected, label):
    if ast.dump(actual) != ast.dump(expected): raise ValueError("unexpected semantic difference: " + label)
def rename(s):
    return s.replace("recursive_mesh_ced_small_layer4_8_inject_loop_lora_r8", "recursive_textmesh_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16").replace("recursive_mesh_layer4_8_inject_loop_lora_r8", "recursive_textmesh_layer8_4_inject_audio_reinject_loop_lora_r16")
def protected_parents():
    files=json.loads(BASELINE.read_text(encoding="utf-8"))["files"]
    for name,expected in files.items():
        if hashlib.sha256((REPO/name).read_bytes().replace(bytes([13,10]),bytes([10]))).hexdigest()!=expected:
            raise ValueError("protected original changed: "+name)
    return {"files":len(files),"all_unchanged":True}

def audit_training():
    old,new=[find_function(tree(p/"train.py"),"train") for p in (PARENT,ROOT)]
    removed=[]
    class Strip(ast.NodeTransformer):
        def visit_If(self,node):
            if "contract.get('audio_reinjection')" in ast.unparse(node.test):
                removed.append("recipe_guard"); return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            target=ast.unparse(node.targets[0]) if len(node.targets)==1 else ""
            if target in {"model.caption_decoder.lm.model.collect_reinject_stats", "record['serial_injection_trace']",
                          "record['input_reinjection_rank0_last_microbatch']", "record['text_memory_shape']",
                          "rank_check['serial_injection']", "rank_check['text_memory']"}:
                removed.append(target); return None
            return self.generic_visit(node)
        def visit_Dict(self,node):
            pairs=[]
            for k,v in zip(node.keys,node.values):
                if isinstance(k,ast.Constant) and k.value in {"serial_injection","text_memory"}: removed.append(k.value)
                else: pairs.append((k,v))
            node.keys,node.values=[k for k,v in pairs],[v for k,v in pairs]
            return self.generic_visit(node)
        def visit_Subscript(self,node):
            if ast.unparse(node)=="SERIAL_INJECTION_CONTRACT['new_parameters']": return ast.Constant(value=2444930)
            return self.generic_visit(node)
        def visit_Constant(self,node):
            mapping={"all serial audio branches and both loop LoRA A/B pairs must receive gradients after the first update":"both CED MLPs and both loop LoRA A/B pairs must receive gradients after the first update",
                     "each serial audio branch and both loop LoRA A/B pairs must change during training":"each CED MLP/gate and both loop LoRA A/B pairs must change during training"}
            return ast.Constant(value=mapping[node.value]) if isinstance(node.value,str) and node.value in mapping else node
    Strip().visit(new)
    if len(removed)!=9: raise ValueError(f"unexpected trainer delta: {removed}")
    equal(new,old,"entire train function after explicit contract/diagnostic additions")
    return {"entire_train_function_ast_equal_after_explicit_audits":True,"explicit_additions":removed}

def audit():
    inventory=lambda p:{f.relative_to(p).as_posix() for f in p.rglob("*") if f.suffix in {".py",".sh"} and "__pycache__" not in f.parts}
    if inventory(ROOT)!=inventory(PARENT): raise ValueError("copied source inventory differs")
    changed={"audit_reference.py","contract.py","models/recursive_layer_inject.py","models/recursive_text.py","models/loop_lora.py","optim.py","runtime_checks.py","train.py","write_config.py"}
    identical=[]
    for name in sorted(inventory(PARENT)-changed):
        if (ROOT/name).read_text(encoding="utf-8-sig")!=rename((PARENT/name).read_text(encoding="utf-8-sig")):
            raise ValueError("unexpected copied source change: "+name)
        identical.append(name)
    for name in ("optim.py","models/loop_lora.py"): equal(tree(ROOT/name),tree(SERIAL/name),"exact serial rank16 "+name)
    preserved={"train.py":("answer_loss","move_batch","rng_state","restore_rng","save_checkpoint","reload_audit","inspect_tokenizer","audit_checkpoint_tokenizer","probe_sources","main","weights_changed"),
               "contract.py":("geometry","data_order","rank_seed","raw_references","lr_factor","recursive_checkpoint_contract","validate_ced_assets","validate_tokenizer_contract","validate_checkpoint_tokenizer","code_digest","recursive_model_source_sha256"),
               "models/recursive_text.py":("validate_recursive_config","validate_loaded_model")}
    for name,methods in preserved.items():
        for method in methods: equal(find_function(tree(ROOT/name),method),find_function(tree(PARENT/name),method),name+":"+method)
    old,new=[tree(p/"models/recursive_layer_inject.py") for p in (PARENT,ROOT)]
    class StripBackbone(ast.NodeTransformer):
        def visit_ClassDef(self,node):
            if node.name!="MeshLlamaModel": return node
            node.body=[n for n in node.body if not isinstance(n,ast.FunctionDef) or n.name not in {"forward","enable_audio_reinjection","inject_input_audio","reinjection_diagnostics"}]
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if ast.unparse(node.targets[0])=="self.last_router_query_shapes[name]": return None
            return self.generic_visit(node)
    StripBackbone().visit(old); StripBackbone().visit(new)
    equal(new,old,"whole backbone except explicit boundary forward/serial methods/shape audit")
    for method in ("enable_audio_reinjection","inject_input_audio","reinjection_diagnostics"):
        equal(find_function(tree(ROOT/"models/recursive_layer_inject.py"),method),find_function(tree(SERIAL/"models/recursive_layer_inject.py"),method),"serial "+method)
    scope={"__file__":str(PARENT/"contract.py")};execute(tree(PARENT/"contract.py").body,scope)
    prepared=dict(rows=sum(SOURCE_COUNTS.values()),source_rows=SOURCE_COUNTS,rows_canonical_sha256="rows",mapping_sha256="mapping")
    recipes=[scope["semantic_contract"](prepared,"text","ced"),semantic_contract(prepared,"text","ced")]
    for recipe in recipes:
        for key in ("route","code_sha256","recursive_model_source_sha256","layer_injection","loop_lora","audio_reinjection","serial_injection","text_memory"): recipe.pop(key,None)
    if recipes[0]!=recipes[1]: raise ValueError("inherited training recipe differs")
    return {"status":"PASS","scope":"local source audit only; no remote execution","route_code_sha256":code_digest(),
            "protected_originals":protected_parents(),"identical_source_files_except_route_names":identical,
            "preserved_functions":preserved,"training_parity":audit_training(),"common_recipe_equal":True,
            "loop_lora":LOOP_LORA_CONTRACT,"serial_injection":SERIAL_INJECTION_CONTRACT,"text_memory":TEXT_MEMORY_CONTRACT}
if __name__=="__main__":
    result=audit();write_json(ROOT/"reference_audit.json",result);print(json.dumps(result,ensure_ascii=False,indent=2))
