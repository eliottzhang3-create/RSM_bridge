"""Check real pinned signatures before GPU allocation; no model imports."""
import ast
import inspect
from .registry import variant_spec

def signature(node):
    args=node.args;parameters=[];positional=[*args.posonlyargs,*args.args]
    for i,arg in enumerate(positional):
        default=inspect.Parameter.empty if i<len(positional)-len(args.defaults) else None
        kind=inspect.Parameter.POSITIONAL_ONLY if i<len(args.posonlyargs) else inspect.Parameter.POSITIONAL_OR_KEYWORD
        parameters.append(inspect.Parameter(arg.arg,kind,default=default))
    if args.vararg:parameters.append(inspect.Parameter(args.vararg.arg,inspect.Parameter.VAR_POSITIONAL))
    for arg,default in zip(args.kwonlyargs,args.kw_defaults):
        parameters.append(inspect.Parameter(arg.arg,inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if default is None else None))
    if args.kwarg:parameters.append(inspect.Parameter(args.kwarg.arg,inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(parameters)

def audit_interfaces(target,root):
    calls=[('models/recursive_text.py','load_recursive_text_model',1)]
    if variant_spec(target)['family']=='mesh':
        calls += [('models/recursive_text.py',name,1) for name in ('validate_recursive_config','validate_audio_read3','validate_loop_lora')]
    else:calls += [('models/decoder.py','inference_logits',4),('models/decoder.py','prefix_validity',2)]
    found={}
    for file,name,count in calls:
        tree=ast.parse((root/file).read_text(encoding='utf-8'))
        node=next((n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name),None)
        if node is None:raise ValueError(f'missing native interface {root/file}:{name}')
        sig=signature(node);sig.bind(*[object() for _ in range(count)])
        found[file+':'+name]=str(sig)
    return {'status':'PASS','signatures':found,'policy':'pinned native interfaces; no fresh flag on audio8 read3'}
