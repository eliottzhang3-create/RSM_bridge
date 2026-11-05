"""Audit isolated four-set modality LoRA against unchanged textmesh rank16 mother."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from contract import (SOURCE_COUNTS, LOOP_LORA_CONTRACT, SERIAL_INJECTION_CONTRACT,
                      TEXT_MEMORY_CONTRACT, code_digest, semantic_contract, write_json)
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
PARENT = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_textmesh_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16")
BASELINE = REPO/"audit_reports/mizar_s1_textmesh_modality_lora_r8_parent_baseline.json"

def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))
def find_function(module, name):
    found = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise ValueError(f"expected one function {name}, got {len(found)}")
    return found[0]
def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),
                 "modality-lora-source-audit", "exec"), scope)
def equal(actual, expected, label):
    if ast.dump(actual) != ast.dump(expected):
        raise ValueError("unexpected inherited difference: " + label)
def protected_parents():
    files=json.loads(BASELINE.read_text(encoding="utf-8"))["files"]
    for name,expected in files.items():
        if hashlib.sha256((REPO/name).read_bytes().replace(bytes([13,10]),bytes([10]))).hexdigest()!=expected:
            raise ValueError("protected original changed: "+name)
    return {"files":len(files),"all_unchanged":True}

def audit_backbone():
    old,new=[tree(p/"models/recursive_layer_inject.py") for p in (PARENT,ROOT)]
    changes=[]
    class Strip(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if node.module=="models.loop_lora":
                for alias in list(node.names):
                    if alias.name=="reset_loop_lora_trace":node.names.remove(alias);changes.append("reset_import")
            return node
        def visit_Expr(self,node):
            if ast.unparse(node)=="reset_loop_lora_trace(self)":changes.append("metadata_reset");return None
            return self.generic_visit(node)
    Strip().visit(new)
    if changes!=["reset_import","metadata_reset"]:raise ValueError(f"unexpected backbone deltas: {changes}")
    equal(new,old,"entire backbone including exact memory/query/write, attention and injection forward")
    return {"entire_backbone_ast_equal_after_metadata_reset":True,"explicit_additions":changes}

def audit_training():
    old,new=[find_function(tree(p/"train.py"),"train") for p in (PARENT,ROOT)]
    changes=[]
    class Strip(ast.NodeTransformer):
        def visit_Assign(self,node):
            if ast.unparse(node.targets[0])=="record['modality_lora_execution']":changes.append("execution_diagnostic");return None
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if isinstance(node.value,str) and "all four modality/loop LoRA A/B pairs" in node.value:
                changes.append("message");return ast.Constant(value=node.value.replace("all four modality/loop LoRA A/B pairs","both loop LoRA A/B pairs"))
            return node
    Strip().visit(new)
    if changes!=["message","execution_diagnostic","message"]:raise ValueError(f"unexpected trainer deltas: {changes}")
    equal(new,old,"entire train function after explicit modality diagnostics/messages")
    return {"entire_train_function_ast_equal_after_explicit_audits":True,"explicit_additions":changes}

def audit():
    inventory=lambda p:{f.relative_to(p).as_posix() for f in p.rglob("*") if f.suffix in {".py",".sh"} and "__pycache__" not in f.parts}
    if inventory(ROOT)!=inventory(PARENT):raise ValueError("copied source inventory differs")
    changed={"audit_reference.py","contract.py","models/recursive_layer_inject.py","models/loop_lora.py","optim.py","runtime_checks.py","train.py","write_config.py"}
    identical=[]
    for name in sorted(inventory(PARENT)-changed):
        expected=(PARENT/name).read_text(encoding="utf-8-sig").replace("loop_lora_r16","loop_modality_lora_r8")
        if (ROOT/name).read_text(encoding="utf-8-sig")!=expected:raise ValueError("unexpected copied source change: "+name)
        identical.append(name)
    preserved={"train.py":("collective_require","rng_state","restore_rng","tensor_digest","role_snapshots","validate_trainable_model","weights_changed","move_batch","answer_loss","inspect_tokenizer","audit_checkpoint_tokenizer","probe_sources","save_checkpoint","reload_audit","main"),
               "contract.py":("geometry","lr_factor","data_order","rank_seed","raw_references","recursive_checkpoint_contract","validate_ced_assets","validate_tokenizer_contract","validate_checkpoint_tokenizer","code_digest","recursive_model_source_sha256")}
    for name,methods in preserved.items():
        for method in methods:equal(find_function(tree(ROOT/name),method),find_function(tree(PARENT/name),method),name+":"+method)
    # Optimizer and GradNorm algorithms remain exactly inherited. Only role coverage changes.
    old,new=[tree(p/"optim.py") for p in (PARENT,ROOT)]
    for label in ("GradNormTracker",):
        equal(next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name==label),next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name==label),label)
    opts=[find_function(t,"build_optimizer") for t in (old,new)]
    for opt in opts:opt.body=[n for n in opt.body if not isinstance(n,ast.If)]
    equal(*opts,"optimizer/scheduler except explicit role coverage guard")
    # Loop scope management must retain the original exception-safe finally reset.
    equal(find_function(tree(ROOT/"models/loop_lora.py"),"call_with_loop_lora"),find_function(tree(PARENT/"models/loop_lora.py"),"call_with_loop_lora"),"scope selection and cleanup")
    scope={"__file__":str(PARENT/"contract.py")};execute(tree(PARENT/"contract.py").body,scope)
    prepared=dict(rows=sum(SOURCE_COUNTS.values()),source_rows=SOURCE_COUNTS,rows_canonical_sha256="rows",mapping_sha256="mapping")
    recipes=[scope["semantic_contract"](prepared,"text","ced"),semantic_contract(prepared,"text","ced")]
    for recipe in recipes:
        for key in ("route","code_sha256","recursive_model_source_sha256","loop_lora"):recipe.pop(key,None)
    if recipes[0]!=recipes[1]:raise ValueError("non-LoRA training recipe differs")
    return {"status":"PASS","scope":"local source audit; no real assets or GPU execution","route_code_sha256":code_digest(),
            "protected_originals":protected_parents(),"identical_source_files_except_route_names":identical,
            "preserved_functions":preserved,"training_parity":audit_training(),"backbone_parity":audit_backbone(),
            "common_recipe_equal":True,"loop_lora":LOOP_LORA_CONTRACT,"serial_injection":SERIAL_INJECTION_CONTRACT,"text_memory":TEXT_MEMORY_CONTRACT}
if __name__=="__main__":
    result=audit();write_json(ROOT/"reference_audit.json",result);print(json.dumps(result,ensure_ascii=False,indent=2))
