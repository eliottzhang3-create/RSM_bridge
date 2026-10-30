from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT/'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_listen'
DIRECT = ROUTE.with_name('mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject')

def read(path): return path.read_text(encoding='utf-8-sig')
def write(path, text): path.write_text(text, encoding='utf-8', newline='\n')
def change(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new)
def replace_function(text, name, replacement):
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, name
    n = nodes[0]
    lines = text.splitlines(keepends=True)
    return ''.join(lines[:n.lineno-1])+replacement.rstrip()+'\n\n'+''.join(lines[n.end_lineno:])

p=ROUTE/'optim.py'
text=read(p)
if 'NEW_ROLES =' not in text:
    text=change(text, 'from contract import lr_factor', '''from contract import lr_factor

NEW_ROLES = tuple(f"layer{layer}_{kind}" for layer in (4, 8) for kind in ("mlp", "gate", "listen"))
TRAINABLE_ROLES = ("map", "default", *NEW_ROLES)''')
    text=change(text, 'if {key[0] for key in groups} != {"map", "default", "layer4_mlp", "layer8_mlp", "layer4_gate", "layer8_gate", "layer4_listen", "layer8_listen"}:', 'if {key[0] for key in groups} != set(TRAINABLE_ROLES):')
write(p,text)

p=ROUTE/'train.py'
text=read(p)
text=change(text, 'validate_ced_assets, write_json)', 'validate_ced_assets, validate_resume_payload, write_json)')
text=change(text, 'from optim import GradNormTracker, build_optimizer, parameter_role', '''from optim import GradNormTracker, build_optimizer, parameter_role, TRAINABLE_ROLES, NEW_ROLES
from models.recursive_text import validate_audio_listen''')
for name in ('role_snapshots','validate_trainable_model','first_gradient_audit'):
    src=read(DIRECT/'train.py')
    n=next(n for n in ast.walk(ast.parse(src)) if isinstance(n,ast.FunctionDef) and n.name==name)
    replacement='\n'.join(src.splitlines()[n.lineno-1:n.end_lineno])
    replacement=replacement.replace('validate_layer_injection', 'validate_audio_listen')
    replacement=replacement.replace('last_injection_trace','last_listen_trace')
    if name=='first_gradient_audit':
        replacement=change(replacement, '            required.append(f"layer{layer}_mlp")', '            required.extend((f"layer{layer}_mlp", f"layer{layer}_listen"))')
        replacement=change(replacement, 'and all(not audit["nonzero_by_role"][f"layer{layer}_mlp"]', 'and all(not audit["nonzero_by_role"][f"layer{layer}_{kind}"]')
        replacement=change(replacement, 'for layer in (4, 8) if closed[str(layer)])', 'for layer in (4, 8) if closed[str(layer)] for kind in ("mlp", "listen"))')
        replacement=change(replacement, '    return audit', '''    audit["physical_trace"] = [item["physical_index"] for item in core.last_forward_trace]
    audit["passed"] = audit["passed"] and audit["physical_trace"] == list(TEXT_MODEL_SCHEDULE)
    return audit''')
    text=replace_function(text,name,replacement)

helpers='''def listen_gradient_audit(model):
    result = {role: {"present": True, "finite": True, "nonzero": False} for role in NEW_ROLES}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        role = parameter_role(name)
        if role not in result:
            continue
        gradient = parameter.grad
        result[role]["present"] &= gradient is not None
        result[role]["finite"] &= gradient is not None and bool(torch.isfinite(gradient).all().item())
        result[role]["nonzero"] |= gradient is not None and bool(gradient.ne(0).any().item())
    return result


def audit_listen_update(model, state, update, run_dir, rank, device):
    """All ranks persist the same cumulative unlock state; zero upstream start is expected."""
    if state["complete"]:
        return
    branches = listen_gradient_audit(model)
    collective_require(all(item["present"] and item["finite"] for item in branches.values()),
                       "missing/nonfinite CED layer4/8 listen gradients", device)
    flags = torch.tensor([int(branches[role]["nonzero"]) for role in NEW_ROLES], device=device)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    for role, nonzero in zip(NEW_ROLES, flags.cpu().tolist()):
        state["seen"][role] = state["seen"].get(role, False) or bool(nonzero)
    state["complete"] = all(state["seen"].values())
    if state["complete"]:
        state["opened_at_update"] = update
        write_json(run_dir / f"listen_gradient_open_rank{rank}.json", {**state, "branches": branches})
    collective_require(state["complete"] or update < state["deadline"],
                       "CED layer4/8 listen gradients did not unlock within 32 real updates", device)


'''
text=change(text,'def weights_changed(model, before):',helpers+'def weights_changed(model, before):')
text=change(text,'                   "num_epochs": EPOCHS,','''                   "num_epochs": EPOCHS,
                   "listen_gradient_state": model.caption_decoder.lm.model.listen_gradient_state,''')
text=change(text,'    required = {"state_dict",','''    validate_resume_payload(saved, config, config["contract"]["geometry"], allow_completed=True)
    if set(saved["state_dict"]) != set(model.state_dict()):
        raise ValueError("checkpoint state tensor inventory differs")
    required = {"state_dict",''')
text=change(text,'        model.load_state_dict(saved["state_dict"], strict=True)', '''        validate_resume_payload(saved, config, geometry)
        model.caption_decoder.lm.model.listen_gradient_state = saved["listen_gradient_state"]
        model.load_state_dict(saved["state_dict"], strict=True)''')
text=change(text,'    gradient_audit = None','''    gradient_audit = None
    listen_core = model.caption_decoder.lm.model
    if listen_core.listen_gradient_state is None:
        listen_core.listen_gradient_state = {"complete": False,
            "seen": {role: False for role in NEW_ROLES}, "deadline": 32}
    listen_gradient_state = listen_core.listen_gradient_state
    listen_initial_sha = tensor_digest((name, p) for name, p in model.named_parameters()
                                      if p.requires_grad and parameter_role(name) in NEW_ROLES)''')
text=change(text,'"world_size": dist.get_world_size(),','''"audio_listen": validate_audio_listen(model.caption_decoder.lm),
              "new_parameter_count": 2450700, "listen_initial_sha256": listen_initial_sha,
              "world_size": dist.get_world_size(),''')
text=change(text,'            for micro_index in range(ACCUMULATION_STEPS):','''            listen_core.collect_listen_stats = step < 2 or (step + 1) % 25 == 0
            for micro_index in range(ACCUMULATION_STEPS):''')
text=change(text,'            total_norm, min_scale = tracker.track_and_clip_(model.named_parameters())','''            audit_listen_update(model, listen_gradient_state, step + 1, run_dir, rank, device)
            total_norm, min_scale = tracker.track_and_clip_(model.named_parameters())''')
text=change(text,'                record["staging_wait_seconds"] = staging.wait_seconds','''                record["audio_listen_rank0_last_microbatch"] = listen_core.listen_diagnostics()
                record["staging_wait_seconds"] = staging.wait_seconds''')
text=change(text,'    rank_check["staging_wait_seconds"] = staging.wait_seconds','''    collective_require(listen_gradient_state["complete"], "listen gradient audit never completed", device)
    rank_check["audio_listen"] = validate_audio_listen(model.caption_decoder.lm)
    rank_check["listen_gradient_state"] = listen_gradient_state
    rank_check["listen_initial_sha256"] = listen_initial_sha
    rank_check["listen_final_sha256"] = tensor_digest((name, p) for name, p in model.named_parameters()
                                                     if p.requires_grad and parameter_role(name) in NEW_ROLES)
    rank_check["staging_wait_seconds"] = staging.wait_seconds''')
write(p,text)
print('Configured training, gradient audit and complete resume:',ROUTE)
