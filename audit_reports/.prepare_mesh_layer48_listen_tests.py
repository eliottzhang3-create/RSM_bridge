from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[1]
ROUTE=ROOT/'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_listen'
DIRECT=ROUTE.with_name('mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject')
def write(p,s): p.write_text(s,encoding='utf-8',newline='\n')
def read(p): return p.read_text(encoding='utf-8-sig')
for name in ('route','shared_start'):
    text=read(ROOT/f'tests/test_mizar_s1_mesh_ced_layer48_{name}.py')
    text=text.replace('layer4_8_inject','layer4_8_listen').replace('layer48_', 'layer48_listen_')
    write(ROOT/f'tests/test_mizar_s1_mesh_ced_layer48_listen_{name}.py',text)
text=read(ROOT/'tests/validate_mizar_s1_mesh_ced_layer48.py')
text=text.replace('layer4_8_inject','layer4_8_listen').replace('mesh_ced_layer48_', 'mesh_ced_layer48_listen_')
write(ROOT/'tests/validate_mizar_s1_mesh_ced_layer48_listen.py',text)

text=read(DIRECT/'audit_reference.py')
text=text.replace('models/recursive_layer_inject.py','models/recursive_mesh_listen.py')
text=text.replace('"enable_layer_injection", "inject_audio", "injection_diagnostics"','"enable_audio_listen", "listen_audio", "listen_diagnostics"')
text=text.replace('self.layer_inject_tokens','self.listen_audio_tokens').replace('self.inject_audio','self.listen_audio')
text=text.replace('self.last_injection_trace','self.last_listen_trace').replace('self.last_injection_stats','self.last_listen_stats')
text=text.replace('"loop == 0"}', '"loop == 0", "attention_mask is not None"}')
text=text.replace('"guard": 5','"guard": 6')
text=text.replace('"layer_injection"','"audio_listen"')
text=text.replace('"restore_rng", "save_checkpoint",\n                     "reload_audit",', '"restore_rng",')
start=text.index('    old_loop = ')
end=text.index('    scope = {"__file__"',start)
text=text[:start]+'''    old_train = find_function(tree(PARENT/"train.py"), "train")
    new_train = find_function(tree(ROOT/"train.py"), "train")

    class StripDiagnostics(ast.NodeTransformer):
        def visit_Assign(self, node):
            names = [ast.unparse(t) for t in node.targets]
            if any(n in {"listen_core", "listen_gradient_state", "listen_initial_sha",
                         "listen_core.collect_listen_stats", "model.caption_decoder.lm.model.listen_gradient_state"}
                   or n.startswith("rank_check['listen_") or n == "rank_check['audio_listen']"
                   or n == "record['audio_listen_rank0_last_microbatch']" for n in names):
                return None
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call):
                name = ast.unparse(node.value.func)
                if name in {"validate_resume_payload", "audit_listen_update"}:
                    return None
                if name == "collective_require" and any(isinstance(a, ast.Constant)
                        and a.value == "listen gradient audit never completed" for a in node.value.args):
                    return None
            return self.generic_visit(node)

        def visit_If(self, node):
            if ast.unparse(node.test) == "listen_core.listen_gradient_state is None":
                return None
            return self.generic_visit(node)

        def visit_Dict(self, node):
            pairs = [(k,v) for k,v in zip(node.keys,node.values()) if not isinstance(k,ast.Constant)
                     or k.value not in {"audio_listen", "new_parameter_count", "listen_initial_sha256"}]
            node.keys, node.values = [x[0] for x in pairs], [x[1] for x in pairs]
            return self.generic_visit(node)

    equal(StripDiagnostics().visit(new_train), old_train, "entire training function after explicit listen/audit additions")
    save = find_function(tree(ROOT/"train.py"), "save_checkpoint")
    for n in ast.walk(save):
        if isinstance(n,ast.Dict):
            pairs=[(k,v) for k,v in zip(n.keys,n.values()) if not isinstance(k,ast.Constant) or k.value!="listen_gradient_state"]
            n.keys,n.values=[p[0] for p in pairs],[p[1] for p in pairs]
    equal(save,find_function(tree(PARENT/"train.py"),"save_checkpoint"),"checkpoint writer except added unlock state")
    reload = find_function(tree(ROOT/"train.py"), "reload_audit")
    reload.body=[n for n in reload.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and ast.unparse(n.value.func)=="validate_resume_payload") and not (isinstance(n,ast.If)
        and ast.unparse(n.test)=="set(saved['state_dict']) != set(model.state_dict())")]
    equal(reload,find_function(tree(PARENT/"train.py"),"reload_audit"),"reload except extra full-state validation")
''' + text[end:]
write(ROUTE/'audit_reference.py',text)
print('Prepared audit, launcher/shared-start tests and validation entry')
